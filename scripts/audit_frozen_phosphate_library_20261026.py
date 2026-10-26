"""Read-only ALL195306 audit with H-inclusive assembly and marked-P-first SMILES."""

import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors

from structure_pretrain.glycerol_core_collection.chemistry import canonical, sugar_and_inositol
from structure_pretrain.glycerol_core_collection.collection import dump_json, file_hash, write_csv
from structure_pretrain.headgroup_target_expansion.chemistry import (
    ACYL_P,
    PEROXIDE,
    extra_glycerol_lipid_anchor,
    fatty_N_acyl_PE_PS_head,
)

ROOT = Path("outputs/headgroup_training_targets_v02_20261026")
SOURCE = Path("data/structure_library_v0.3/polar_library.parquet")
SIDECAR = Path(
    "outputs/glycerol_core_collection_v01_20261023/frozen_195306_future_core_annotations.parquet"
)
SCAFFOLD = Chem.MolFromSmiles("CC(=O)OCC(OC(C)=O)CO[*:9]")


def atom_counts(mol):
    return Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms() if a.GetAtomicNum())


def assemble_test(head, marker, p):
    n = head.GetNumAtoms()
    edit = Chem.RWMol(Chem.CombineMols(head, SCAFFOLD))
    dummy = next(a for a in SCAFFOLD.GetAtoms() if a.GetAtomicNum() == 0)
    edit.AddBond(p, n + dummy.GetNeighbors()[0].GetIdx(), Chem.BondType.SINGLE)
    for i in sorted((marker, n + dummy.GetIdx()), reverse=True):
        edit.RemoveAtom(i)
    mol = edit.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def main():
    before = file_hash(SOURCE)
    old = json.loads(
        Path(
            "outputs/glycerol_core_collection_v01_20261023/frozen_library_annotation_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert before == old["original_parquet_sha256"]
    pf = pq.ParquetFile(SOURCE)
    sf = pq.ParquetFile(SIDECAR)
    assert pf.metadata.num_rows == sf.metadata.num_rows == 195306
    scaf_counts = atom_counts(SCAFFOLD)
    glycerol_mass = (
        3 * Chem.GetPeriodicTable().GetAtomicWeight("C")
        + 7 * Chem.GetPeriodicTable().GetAtomicWeight("H")
        + 3 * Chem.GetPeriodicTable().GetAtomicWeight("O")
    )
    checks = Counter()
    risk = Counter()
    examples = []
    writer = None
    done = 0
    cols = [
        "polar_id",
        "core_average_mass_Da",
        "sugar_ring_count",
        "inositol_ring_count",
        "filter_reasons",
    ]
    for batch, side in zip(
        pf.iter_batches(batch_size=512, columns=["polar_id", "polar_fragment_smiles"]),
        sf.iter_batches(batch_size=512, columns=cols),
        strict=True,
    ):
        rows = []
        for original, previous in zip(batch.to_pylist(), side.to_pylist(), strict=True):
            assert original["polar_id"] == previous["polar_id"]
            row = {
                "polar_id": original["polar_id"],
                "P_first_polar_smiles": "",
                "formal_structure_pass": False,
                "marker_and_attachment_P_pass": False,
                "P_first_graph_roundtrip_pass": False,
                "complete_neutral_assembly_and_H_formula_pass": False,
                "independent_core_mass_pass": False,
                "future_500_600_policy_pass": False,
                "audit_failure": "",
                "review_flags": "",
                "core_average_mass_Da": previous["core_average_mass_Da"],
                "mass_limit_Da": 500.0,
            }
            try:
                with rdBase.BlockLogs():
                    mol = Chem.MolFromSmiles(original["polar_fragment_smiles"])
                if mol is None or len(Chem.GetMolFrags(mol)) != 1:
                    raise ValueError("invalid_or_disconnected_SMILES")
                if any(a.GetIsotope() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
                    raise ValueError("isotope_or_radical")
                markers = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
                if (
                    len(markers) != 1
                    or markers[0].GetAtomMapNum() != 1
                    or markers[0].GetDegree() != 1
                ):
                    raise ValueError("unique_marker_required")
                p = markers[0].GetNeighbors()[0].GetIdx()
                if mol.GetAtomWithIdx(p).GetAtomicNum() != 15:
                    raise ValueError("marker_not_bound_to_P")
                first = Chem.MolToSmiles(mol, rootedAtAtom=p, canonical=True, isomericSmiles=False)
                back = Chem.MolFromSmiles(first)
                if canonical(mol, False) != canonical(back, False):
                    raise ValueError("P_first_graph_changed")
                assembled = assemble_test(mol, markers[0].GetIdx(), p)
                if Chem.GetFormalCharge(assembled) != 0 or any(
                    a.GetAtomicNum() == 0 for a in assembled.GetAtoms()
                ):
                    raise ValueError("assembly_not_complete_neutral")
                if atom_counts(assembled) != atom_counts(mol) + scaf_counts:
                    raise ValueError("assembly_formula_including_H_changed")
                mass = Descriptors.MolWt(mol) + glycerol_mass
                if abs(mass - previous["core_average_mass_Da"]) > 1e-5:
                    raise ValueError("independent_mass_vs_previous_sidecar_mismatch")
                sugar, ino = sugar_and_inositol(mol)
                assert (sugar, ino) == (
                    previous["sugar_ring_count"],
                    previous["inositol_ring_count"],
                )
                limit = 600.0 if sugar or ino else 500.0
                flags = [
                    x
                    for x in previous["filter_reasons"].split(";")
                    if x and x != "core_mass_above_limit"
                ]
                if mass > limit + 1e-8:
                    flags.append("core_mass_above_500_600_policy")
                if mol.HasSubstructMatch(ACYL_P):
                    flags.append("reactive_acyl_phosphate_mixed_anhydride")
                if mol.HasSubstructMatch(PEROXIDE):
                    flags.append("peroxide_requires_review")
                if extra_glycerol_lipid_anchor(mol):
                    flags.append("extra_glycerol_lipid_anchor_including_short_acyl")
                if fatty_N_acyl_PE_PS_head(mol):
                    flags.append("extra_fatty_N_acyl_PE_PS_head")
                if sum(a.GetAtomicNum() == 15 for a in mol.GetAtoms()) > 4:
                    flags.append("more_than_four_P")
                row.update(
                    P_first_polar_smiles=first,
                    formal_structure_pass=True,
                    marker_and_attachment_P_pass=True,
                    P_first_graph_roundtrip_pass=True,
                    complete_neutral_assembly_and_H_formula_pass=True,
                    independent_core_mass_pass=True,
                    future_500_600_policy_pass=not flags,
                    review_flags=";".join(sorted(set(flags))),
                    mass_limit_Da=limit,
                )
                checks["formal_structure_pass"] += 1
                checks["complete_assembly_H_formula_pass"] += 1
                checks["P_first_graph_and_marker_pass"] += 1
                checks["independent_core_mass_pass"] += 1
                checks["future_500_600_policy_pass"] += not flags
                risk.update(set(flags))
            except (ValueError, RuntimeError, AssertionError) as error:
                row["audit_failure"] = str(error)
                checks["formal_or_assembly_audit_failed"] += 1
                if len(examples) < 50:
                    examples.append({**row, "original_smiles": original["polar_fragment_smiles"]})
            rows.append(row)
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(
                ROOT / "background_195306_audit_P_first.parquet.part",
                table.schema,
                compression="zstd",
            )
        writer.write_table(table)
        done += len(rows)
        if done % 8192 == 0:
            print(json.dumps({"background_audit_processed": done}), flush=True)
    writer.close()
    assert done == 195306 and file_hash(SOURCE) == before
    (ROOT / "background_195306_audit_P_first.parquet.part").replace(
        ROOT / "background_195306_audit_P_first.parquet"
    )
    report = {
        "records": done,
        "original_SHA256": before,
        "original_file_unchanged": True,
        "check_counts": dict(checks),
        "review_risk_counts_overlap": dict(risk),
        "background_not_merged_into_target_proposal": True,
        "assembly_test": "explicit marked P plus fixed glycerol and two acetyl reference chains; exact elemental counts including H",  # noqa: E501
        "structure_validity_is_not_natural_occurrence_or_stability_proof": True,
        "mass_policy": "ordinary P/nonP cores500; sugar or inositol600; original training inputs unchanged",  # noqa: E501
    }
    dump_json(ROOT / "background_audit_summary.json", report)
    write_csv(ROOT / "background_formal_audit_failures.csv", examples)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
