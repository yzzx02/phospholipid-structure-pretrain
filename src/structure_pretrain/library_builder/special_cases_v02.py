"""Targeted offline motif audits and a small representative set, never a full library."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem, rdBase

from structure_pretrain.polar_plausibility.pipeline import dump_json, write_tsv

from .assembly import assemble
from .canonicalize import nonisomeric_smiles, parse_molecule
from .chain_combinations import make_combination
from .chain_library import import_chains
from .schemas import PolarRule
from .validation import validate_polar_fragment

MOTIF_SMARTS = {
    "cyanomethyl": "[P;X4](=[O])-[O;X2]-[CH2]-[C]#[N]",
    "2-cyanoethyl": "[P;X4](=[O])-[O;X2]-[CH2]-[CH2]-[C]#[N]",
}
MOTIFS = {name: Chem.MolFromSmarts(smarts) for name, smarts in MOTIF_SMARTS.items()}
ACYL_ARM = "[O]-[C](=[O])-" + "~".join(["[#6]"] * 7)
ETHER_ARM = "[O]-[CH2]~" + "~".join(["[#6]"] * 7)
PHOSPHATIDYL_QUERIES = [
    Chem.MolFromSmarts(f"[P]-[O]-[CH2]-[CH](-{left})-[CH2]-{right}")
    for left, right in ((ACYL_ARM, ACYL_ARM), (ACYL_ARM, ETHER_ARM), (ETHER_ARM, ACYL_ARM))
]
PI_FRAGMENT = "O=P(O)(OC1C(O)C(O)C(O)C(O)C1O)[*:1]"
REPRESENTATIVES = ("A-16:0_A-18:1", "A-18:0_A-20:4", "O-18:0_A-20:4")


def motifs_in(smiles):
    # Every nitrile contains a triple bond; this prefilter cannot remove an exact hit.
    if "#" not in smiles or "P" not in smiles or "N" not in smiles:
        return []
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Unparseable nitrile search record")
    return [name for name, pattern in MOTIFS.items() if mol.HasSubstructMatch(pattern)]


def atom_counts(molecule):
    return Counter(
        atom.GetSymbol() for atom in Chem.AddHs(molecule).GetAtoms() if atom.GetAtomicNum()
    )


def phosphatidyl_nitrile_contexts(smiles, motif):
    """Require nitrile and two long lipid arms to be attached to the SAME phosphorus."""
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid raw graph in context audit")
    nitrile_p = {m[0] for m in molecule.GetSubstructMatches(MOTIFS[motif])}
    lipid_p = {m[0] for q in PHOSPHATIDYL_QUERIES for m in molecule.GetSubstructMatches(q)}
    result = []
    for index in sorted(nitrile_p & lipid_p):
        phosphorus = molecule.GetAtomWithIdx(index)
        free_acid = any(
            o.GetAtomicNum() == 8
            and o.GetDegree() == 1
            and (o.GetTotalNumHs() == 1 or o.GetFormalCharge() == -1)
            for o in phosphorus.GetNeighbors()
        )
        single_o = sum(
            b.GetOtherAtom(phosphorus).GetAtomicNum() == 8
            and b.GetBondType() == Chem.BondType.SINGLE
            for b in phosphorus.GetBonds()
        )
        result.append(
            "free_acid_phosphatidyl_nitrile_diester"
            if free_acid
            else "phosphatidyl_phosphotriester_protection_unresolved"
            if single_o == 3
            else "phosphatidyl_other_P_substitution_review"
        )
    return result


def audit_nitrile_context(output):
    output = Path(output)
    contexts = []
    with (output / "nitrile_raw_hits.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            for kind, count in sorted(
                Counter(
                    phosphatidyl_nitrile_contexts(row["canonical_smiles"], row["motif"])
                ).items()
            ):
                contexts.append(
                    dict(
                        row,
                        context_class=kind,
                        matched_P_count=count,
                        evidence_source="PubChem",
                        source_id=f"CID:{row['cid']}",
                        notes=(
                            "Substructure candidate; triester does not by itself prove "
                            "a protecting intermediate or natural final lipid."
                        ),
                    )
                )
    write_tsv(
        output / "nitrile_raw_full_lipid_candidates.tsv",
        contexts,
        [
            "cid",
            "motif",
            "context_class",
            "matched_P_count",
            "canonical_smiles",
            "evidence_source",
            "source_id",
            "notes",
        ],
    )
    return {
        "nitrile_phosphatidyl_raw_candidate_cids": len({r["cid"] for r in contexts}),
        "nitrile_phosphatidyl_context_unique_cids": {
            kind: len({r["cid"] for r in contexts if r["context_class"] == kind})
            for kind in sorted({r["context_class"] for r in contexts})
        },
        "nitrile_raw_full_lipid_context_definition": (
            "Same P carries exact nitrile motif and AA/OA glycerol scaffold with >=8 C per arm. "
            "Phosphotriester candidates require review of protection status; "
            "candidate hits do not verify an exact generated representative graph."
        ),
    }


def mono_keto_heads(smiles=PI_FRAGMENT):
    parent = validate_polar_fragment(smiles, 1)
    rings = [
        ring
        for ring in parent.GetRingInfo().AtomRings()
        if len(ring) == 6 and all(parent.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in ring)
    ]
    if len(rings) != 1:
        raise ValueError("Require exactly one six-carbon inositol ring")
    ring = set(rings[0])
    linked = [
        i
        for i in ring
        for o in parent.GetAtomWithIdx(i).GetNeighbors()
        if o.GetAtomicNum() == 8 and any(n.GetAtomicNum() == 15 for n in o.GetNeighbors())
    ]
    if len(linked) != 1:
        raise ValueError("Require one phosphate-linked inositol ring carbon")
    sites = [
        (i, o.GetIdx())
        for i in sorted(ring)
        if parent.GetAtomWithIdx(i).GetTotalNumHs() == 1
        for o in parent.GetAtomWithIdx(i).GetNeighbors()
        if o.GetAtomicNum() == 8 and o.GetDegree() == 1 and o.GetTotalNumHs() == 1
    ]
    unique = {}
    for carbon, oxygen in sites:
        changed = Chem.RWMol(parent)
        changed.GetBondBetweenAtoms(carbon, oxygen).SetBondType(Chem.BondType.DOUBLE)
        for index in (carbon, oxygen):
            atom = changed.GetAtomWithIdx(index)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(True)
        molecule = changed.GetMol()
        Chem.SanitizeMol(molecule)
        key = nonisomeric_smiles(molecule)
        checked = validate_polar_fragment(key, 1)
        difference = atom_counts(checked) - atom_counts(parent)
        signed = {
            element: atom_counts(checked)[element] - count
            for element, count in atom_counts(parent).items()
        }
        if difference or {e: d for e, d in signed.items() if d} != {"H": -2}:
            raise ValueError("assembly_error: inositol head oxidation must be delta H=-2 only")
        # Only exocyclic C-O was edited; all ring and phosphate bonds retain their indices.
        for bond in parent.GetBonds():
            if {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} == {carbon, oxygen}:
                continue
            other = molecule.GetBondBetweenAtoms(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            if other is None or other.GetBondType() != bond.GetBondType():
                raise ValueError("assembly_error: ring/phosphate connectivity changed")
        distance = len(Chem.GetShortestPath(parent, linked[0], carbon)) - 1
        unique.setdefault(key, set()).add(distance)
    return {
        "oxidizable_oh_sites": len(sites),
        "generated_candidates": len(sites),
        "unique_mono_keto_heads": len(unique),
        "variants": [
            {"polar_fragment_smiles": key, "ring_graph_distances": sorted(distances)}
            for key, distances in sorted(unique.items())
        ],
    }


def special_rule(smiles, identifier, source_id, source="PubChem"):
    return PolarRule(
        identifier,
        source,
        source_id,
        "",
        smiles,
        1,
        "pubchem_expanded",
        True,
        True,
        False,
        (identifier,),
        (f"{source}:{source_id}",),
    )


def audit_special_cases(review_dir, chains_path, output, approved_path=None, log=print):
    review_dir, output = Path(review_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Special-case output must be empty")
    errors = []
    chains = import_chains(chains_path, errors)
    if errors or Counter(c.linkage_type for c in chains.values()) != {"A": 177, "O": 84, "P": 61}:
        raise ValueError("Special-case audit requires the 322-chain frozen table")
    approved_tiers = {}
    if approved_path:
        with Path(approved_path).open(encoding="utf-8-sig", newline="") as stream:
            approved_tiers = {
                r["polar_id"]: r["polar_plausibility_tier"]
                for r in csv.DictReader(stream, delimiter="\t")
            }
    keto = mono_keto_heads()
    pi_key = nonisomeric_smiles(parse_molecule(PI_FRAGMENT))
    wanted = {pi_key, *(r["polar_fragment_smiles"] for r in keto["variants"])}
    exact, hits = {}, []
    scanned = 0
    for batch in pq.ParquetFile(review_dir / "pubchem_polar_fragments.parquet").iter_batches(5000):
        for row in batch.to_pylist():
            key = row["polar_fragment_smiles"]
            if key in wanted:
                exact[key] = row
            for motif in motifs_in(key):
                hits.append(
                    dict(
                        row,
                        motif=motif,
                        current_status=row["status"],
                        plausibility_tier=approved_tiers.get(row["polar_id"], "NOT_APPROVED"),
                    )
                )
        scanned += batch.num_rows
        if scanned % 50000 == 0:
            log(f"special fragments {scanned}", flush=True)
    if pi_key not in exact:
        raise ValueError("Standard PI fragment absent from snapshot")
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "motif",
        "polar_id",
        "polar_fragment_smiles",
        "source_cids",
        "polar_family",
        "current_status",
        "plausibility_tier",
        "source_count",
    ]
    write_tsv(output / "nitrile_fragment_hits.tsv", hits, fields)
    review_hits = []
    with (review_dir / "pubchem_polar_review.tsv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            for motif in motifs_in(row["polar_fragment_smiles"]):
                review_hits.append((row["polar_id"], motif))
    if set(review_hits) != {(row["polar_id"], row["motif"]) for row in hits}:
        raise ValueError("TSV / Parquet nitrile search disagreement")
    raw_hits, raw_parse_failures = [], []
    raw_scanned = 0
    for batch in pq.ParquetFile(review_dir / "pubchem_polar_raw.parquet").iter_batches(
        batch_size=10000, columns=["cid", "canonical_smiles", "title"]
    ):
        for row in batch.to_pylist():
            try:
                for motif in motifs_in(row["canonical_smiles"]):
                    raw_hits.append(dict(row, motif=motif))
            except ValueError as exc:
                # Raw contains upstream chemistry rejections as well as usable records.
                # Invalid raw graphs are recorded and never promoted to evidence.
                raw_parse_failures.append(dict(row, reason=str(exc)))
        raw_scanned += batch.num_rows
        if raw_scanned % 100000 == 0:
            log(f"special raw {raw_scanned}", flush=True)
    write_tsv(
        output / "nitrile_raw_hits.tsv", raw_hits, ["motif", "cid", "canonical_smiles", "title"]
    )
    write_tsv(
        output / "nitrile_search_rejections.tsv",
        raw_parse_failures,
        ["cid", "canonical_smiles", "reason"],
    )
    cases, heads = [], {}

    def add_head(smiles, family, source_cids, note, identifier=None, distances=None):
        identifier = identifier or "special_" + hashlib.sha256(smiles.encode()).hexdigest()[:20]
        is_real_head = bool(source_cids)
        source = "PubChem" if is_real_head else "assembled_from_known_polar"
        source_id = (
            json.dumps(source_cids)
            if is_real_head
            else "PI_parent_CIDs:" + json.dumps(exact[pi_key]["source_cids"])
        )
        rule = special_rule(smiles, identifier, source_id, source)
        heads[identifier] = {
            **rule.row(),
            "source_cids": source_cids,
            "is_special_case": family != "PI_reference",
            "special_case_family": family,
            "evidence_source": source,
            "is_full_lipid_reported": False,
            "is_programmatically_assembled": True,
            "notes": note,
            "polar_plausibility_tier": approved_tiers.get(identifier, "SPECIAL_PENDING"),
        }
        return rule, source, source_id

    for motif in MOTIFS:
        eligible = [
            r
            for r in hits
            if r["motif"] == motif
            and r["status"] == "APPROVABLE"
            and (not approved_path or r["polar_id"] in approved_tiers)
        ]
        eligible.sort(key=lambda r: (r["carbon_count"], r["heavy_atoms"], r["polar_id"]))
        if not eligible:
            continue
        row = eligible[0]
        note = (
            "Real PubChem polar fragment; full representative graph assembled, "
            "no exact full-lipid evidence assigned."
        )
        rule, source, sid = add_head(
            row["polar_fragment_smiles"], motif, row["source_cids"], note, row["polar_id"]
        )
        for label in REPRESENTATIVES:
            assembled = assemble(chains, make_combination(tuple(label.split("_")), chains), rule)
            cases.append(
                {
                    "case_id": f"{motif}_{label}",
                    "motif": motif,
                    "polar_id": rule.polar_id,
                    "polar_fragment_smiles": rule.polar_fragment_smiles,
                    "full_lipid_smiles": assembled["canonical_smiles"],
                    "neutral_formula": assembled["neutral_formula"],
                    "chains": label,
                    "evidence_source": source,
                    "source_id": sid,
                    "is_full_lipid_reported": False,
                    "is_programmatically_assembled": True,
                    "is_special_case": True,
                    "special_case_family": motif,
                    "notes": note,
                }
            )
    pi_cases = []
    parent_rule, _, _ = add_head(
        pi_key,
        "PI_reference",
        exact[pi_key]["source_cids"],
        "Standard PI reference; representative placement is computational.",
        exact[pi_key]["polar_id"],
    )
    for label in REPRESENTATIVES[:2]:
        combo = make_combination(tuple(label.split("_")), chains)
        reference = assemble(chains, combo, parent_rule)
        for variant in [
            {"polar_fragment_smiles": pi_key, "ring_graph_distances": []},
            *keto["variants"],
        ]:
            smiles = variant["polar_fragment_smiles"]
            is_reference = smiles == pi_key
            if is_reference:
                rule = parent_rule
                source, sid = "PubChem", parent_rule.source_id
            else:
                real = exact.get(smiles)
                rule, source, sid = add_head(
                    smiles,
                    "keto_inositol",
                    real["source_cids"] if real else [],
                    "OH->C=O at one ring site; no exact full-lipid evidence assigned. "
                    "Ring distance is not biochemical/IUPAC numbering. "
                    "Historical phosphatidylinosose class evidence does not identify "
                    "this chain pair or regioisomer.",
                    real["polar_id"] if real else None,
                )
            assembled = assemble(chains, combo, rule)
            delta = {
                e: assembled[f"neutral_formula_{e}"] - reference[f"neutral_formula_{e}"]
                for e in ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I")
            }
            if {e: d for e, d in delta.items() if d} != ({} if is_reference else {"H": -2}):
                raise ValueError("assembly_error: full keto-PI formula check failed")
            pi_cases.append(
                {
                    "case_id": f"{rule.polar_id}_{label}",
                    "polar_id": rule.polar_id,
                    "oxidized_ring_position_if_known": "reference"
                    if is_reference
                    else "graph_distance_from_P_linked_ring_C:"
                    + json.dumps(variant["ring_graph_distances"]),
                    "polar_fragment_smiles": smiles,
                    "full_lipid_smiles": assembled["canonical_smiles"],
                    "neutral_formula": assembled["neutral_formula"],
                    "delta_formula_vs_PI": "0" if is_reference else "H-2",
                    "chains": label,
                    "evidence_source": source,
                    "source_id": sid,
                    "is_full_lipid_reported": False,
                    "is_programmatically_assembled": True,
                    "is_special_case": not is_reference,
                    "special_case_family": "PI_reference" if is_reference else "keto_inositol",
                    "assembly_error": False,
                    "notes": heads[rule.polar_id]["notes"],
                }
            )
    write_tsv(output / "nitrile_special_cases.tsv", cases, list(cases[0]) if cases else ["case_id"])
    write_tsv(output / "keto_pi_special_cases.tsv", pi_cases, list(pi_cases[0]))
    supplemental = list(heads.values())
    write_tsv(output / "special_case_polars.tsv", supplemental, list(supplemental[0]))
    must = [r for r in supplemental if r["is_special_case"]]
    write_tsv(
        output / "must_include_polar_ids.tsv",
        must,
        [
            "polar_id",
            "polar_fragment_smiles",
            "is_special_case",
            "special_case_family",
            "is_programmatically_assembled",
        ],
    )
    report = {
        "fragment_records_scanned": scanned,
        "raw_records_scanned": raw_scanned,
        "raw_nitrile_candidate_parse_failures": len(raw_parse_failures),
        "motif_smarts": MOTIF_SMARTS,
        "nitrile_fragment_hits_all_status": dict(Counter(r["motif"] for r in hits)),
        "nitrile_approvable_heads": dict(
            Counter(r["motif"] for r in hits if r["status"] == "APPROVABLE")
        ),
        "nitrile_approved_heads": dict(
            Counter(r["motif"] for r in hits if r["polar_id"] in approved_tiers)
        ),
        "nitrile_raw_cid_hits": dict(Counter(r["motif"] for r in raw_hits)),
        "keto_pi": keto,
        "exact_keto_head_snapshot_hits": sum(
            v["polar_fragment_smiles"] in exact for v in keto["variants"]
        ),
        "nitrile_representative_full_structures": len(cases),
        "keto_pi_representative_full_structures": len(pi_cases),
        "exact_full_lipid_evidence_assigned_to_generated_graphs": 0,
        "reported_boolean_definition": (
            "Exact generated graph independently matched to a cited full-lipid record; "
            "false does not assert nonexistence."
        ),
        "must_include_head_count": len(must),
        "all_cases_programmatically_assembled": True,
        "full_lipid_library_generated": False,
        "training_performed": False,
        "approved_polar_table_used": str(approved_path) if approved_path else None,
    }
    report.update(audit_nitrile_context(output))
    dump_json(output / "special_case_report.json", report)
    return report
