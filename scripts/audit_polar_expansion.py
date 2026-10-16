"""Cross-table audit of expanded polar provenance and chain-only budget simulations."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem, rdBase

from structure_pretrain.library_builder.chain_library import boolean
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_plausibility.pipeline import dump_json
from structure_pretrain.pubchem_polar.chemistry import normalize_fragment


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def audit(baseline_path, review_dir, output):
    output, review_dir = Path(output), Path(review_dir)
    report = json.loads((output / "polar_expansion_report.json").read_text(encoding="utf-8"))
    assert report["baseline_sha256"] == file_hash(Path(baseline_path))
    baseline = {r["polar_id"]: r for r in read_rows(baseline_path)}
    targets = pq.read_table(
        review_dir / "pubchem_polar_fragments.parquet",
        columns=["polar_id", "polar_fragment_smiles", "source_cids"],
        filters=[("status", "=", "MULTIPLE_ATTACHMENT_AMBIGUITY")],
    ).to_pylist()
    targets = {r["polar_id"]: r for r in targets}
    expanded_path = output / "polar_moieties_expanded_v0.2.tsv"
    assert file_hash(expanded_path) == report["approved_table_sha256"]
    heads, endpoints, sample_sites = {}, defaultdict(dict), defaultdict(list)
    families, origins = Counter(), Counter()
    for row in read_rows(expanded_path):
        identifier = row["polar_id"]
        assert identifier not in heads
        source_cids = json.loads(row["source_cids"])
        parent_cids = json.loads(row["parent_source_cids"])
        parent_ids = json.loads(row["analog_parent_polar_ids"])
        family, origin = row["chemistry_family"], row["structure_origin"]
        assert row["polar_plausibility_tier"] in {"CORE", "EXTENDED"}
        if identifier in baseline:
            for field, old_value in baseline[identifier].items():
                if field.startswith("allow_"):
                    assert boolean(row[field]) == boolean(old_value)
                elif field == "source_cids":
                    assert source_cids == json.loads(old_value)
                else:
                    assert row[field] == old_value
        elif origin == "pubchem_multi_attachment_split":
            old = targets[identifier]
            assert row["polar_fragment_smiles"] == old["polar_fragment_smiles"]
            assert set(source_cids).issubset(old["source_cids"])
            sites = json.loads(row["source_attachment_provenance"])
            assert {s["cid"] for s in sites} == set(source_cids)
            assert row["first_stage_status"] == "MULTIPLE_ATTACHMENT_AMBIGUITY"
            for site in sites:
                cid, index = site["cid"], site["source_attachment_atom_index"]
                value = (
                    site["selected_component_attachment_atom_index"],
                    row["polar_fragment_smiles"],
                )
                assert index not in endpoints[cid] or endpoints[cid][index] == value
                endpoints[cid][index] = value
                sample_sites[family].append((cid, index))
        else:
            assert origin == "generated_phosphate_analog" and family == "phosphate"
            assert source_cids == [] and int(row["source_count"]) == 0
            assert parent_ids and parent_cids
            assert row["source"] == "generated_phosphate_analog" and row["original_smiles"] == ""
        heads[identifier] = {
            "smiles": row["polar_fragment_smiles"],
            "family": family,
            "origin": origin,
            "source_cids": source_cids,
            "parent_cids": parent_cids,
            "parent_ids": parent_ids,
        }
        families[family] += 1
        origins[origin] += 1
    assert baseline.keys() <= heads.keys()
    assert len(heads) == report["approved_polar_count"]
    assert dict(families) == report["expanded_family_counts"]
    assert dict(origins) == report["expanded_origin_counts"]
    assert len({h["smiles"] for h in heads.values()}) == len(heads)
    for row in heads.values():
        if row["parent_ids"]:
            parent_cids = set()
            for identifier in row["parent_ids"]:
                assert heads[identifier]["family"] == "phosphonate"
                assert heads[identifier]["origin"] != "generated_phosphate_analog"
                parent_cids.update(heads[identifier]["source_cids"])
            assert set(row["parent_cids"]) == parent_cids
    # Validate EVERY reported source atom index; independently rederive 50 sites/family.
    samples = set()
    for family, sites in sample_sites.items():
        samples.update(
            sorted(set(sites), key=lambda pair: hashlib.sha256(str(pair).encode()).hexdigest())[:50]
        )
    checked, rederived, source_count = 0, 0, 0
    for batch in pq.ParquetFile(review_dir / "pubchem_polar_raw.parquet").iter_batches(
        batch_size=10000, columns=["cid", "original_smiles"]
    ):
        for raw in batch.to_pylist():
            cid = raw["cid"]
            if cid not in endpoints:
                continue
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(raw["original_smiles"])
            assert molecule is not None
            mappings = []
            components = Chem.GetMolFrags(molecule, asMols=True, fragsMolAtomMapping=mappings)
            indices = [
                i
                for i, mol in enumerate(components)
                if any(a.GetAtomicNum() == 15 for a in mol.GetAtoms())
            ]
            assert len(indices) == 1
            component_index = indices[0]
            for original_index, (local_index, key) in endpoints[cid].items():
                oxygen = molecule.GetAtomWithIdx(original_index)
                assert oxygen.GetAtomicNum() == 8 and oxygen.GetDegree() == 1
                assert oxygen.GetNeighbors()[0].GetAtomicNum() == 15
                assert oxygen.GetTotalNumHs() == 1 or oxygen.GetFormalCharge() == -1
                assert mappings[component_index][local_index] == original_index
                checked += 1
                if (cid, original_index) in samples:
                    edited = Chem.RWMol(components[component_index])
                    dummy = Chem.Atom(0)
                    dummy.SetAtomMapNum(1)
                    dummy.SetNoImplicit(True)
                    edited.ReplaceAtom(local_index, dummy)
                    fragment = edited.GetMol()
                    Chem.SanitizeMol(fragment)
                    actual_key, ambiguous, _ = normalize_fragment(fragment)
                    assert not ambiguous and actual_key == key
                    rederived += 1
            source_count += 1
    assert source_count == len(endpoints) and rederived == len(samples)
    analogs = pq.read_table(output / "phosphate_analog_audit.parquet").to_pylist()
    eligible = [r for r in analogs if r["eligible"]]
    accepted = [r for r in eligible if r["tier"] in {"CORE", "EXTENDED"}]
    assert len(eligible) == report["analog_counts"]["eligible_phosphonate"]
    assert len(accepted) == report["analog_counts"]["passed_generations"]
    assert all(json.loads(r["heavy_atom_delta"]) == {"O": 1} for r in eligible)
    assert sum(bool(r["is_new_unique"]) for r in accepted) == origins["generated_phosphate_analog"]
    for row in accepted:
        target = heads[row["merged_polar_id"]]
        assert target["smiles"] == row["polar_fragment_smiles"]
        assert row["parent_polar_id"] in target["parent_ids"]
    budgets = []
    for path in sorted((output / "family_dry_run").glob("dry_run_*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        maximum = summary["max_structures"]
        rows = pq.read_table(
            output / "family_dry_run" / f"per_head_budget_{maximum}.parquet"
        ).to_pylist()
        assert len(rows) == summary["polar_count"]
        assert baseline.keys() <= {r["polar_id"] for r in rows}
        assert heads.keys() <= {r["polar_id"] for r in rows}
        assert min(r["mandatory_count"] for r in rows) >= 3
        assert all(
            r["mandatory_count"] + r["extra_quota"] == r["planned_background_count"] for r in rows
        )
        assert (
            sum(r["planned_background_count"] for r in rows)
            == summary["estimated_full_structures"]
            <= maximum
            <= 2_000_000
        )
        assert (
            sum(r["mandatory_count"] for r in rows)
            == summary["mandatory_anchor_and_special_structures"]
        )
        assert sum(r["lyso_quota"] for r in rows) == summary["estimated_lyso"]
        by_family = Counter()
        for row in rows:
            by_family[row["chemistry_family"]] += row["planned_background_count"]
        assert dict(by_family) == summary["estimated_full_by_chemistry_family"]
        assert summary["estimated_AA"] + summary["estimated_OA"] + summary[
            "estimated_PE_PA"
        ] + summary["estimated_lyso"] == sum(by_family.values())
        special = [r for r in rows if r["is_special_case"]]
        assert len(special) == 5 and all(r["planned_background_count"] >= 3 for r in special)
        assert not summary["full_lipid_library_generated"] and not summary["training_performed"]
        budgets.append(
            {
                "maximum": maximum,
                "polar_count": len(rows),
                "planned_pairs": sum(by_family.values()),
                "must_include_heads": len(special),
            }
        )
    assert len(budgets) == 2
    manifest = json.loads(
        (output / "family_dry_run/inputs_manifest.json").read_text(encoding="utf-8")
    )
    frozen = json.loads((review_dir / "chain_freeze_report.json").read_text(encoding="utf-8"))
    assert file_hash(Path(manifest["chains"]["path"])) == frozen["sha256"]
    result = {
        "status": "passed",
        "baseline_preserved": len(baseline),
        "expanded_heads_verified": len(heads),
        "source_cid_endpoints_verified": source_count,
        "all_source_attachment_indices_checked": checked,
        "independent_canonical_rederivation_samples": rederived,
        "eligible_analog_O_plus_one_records_checked": len(eligible),
        "generated_CIDs_not_misattributed": True,
        "frozen_chain_table_unchanged": True,
        "budget_audits": budgets,
        "rdkit_version": rdBase.rdkitVersion,
        "training_performed": False,
        "full_lipid_library_generated": False,
    }
    dump_json(output / "verification_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="data/private/pubchem_polar_plausibility_v0.1/polar_moieties_pubchem_v0.1.tsv",
    )
    parser.add_argument("--review-dir", default="data/private/pubchem_polar_review")
    parser.add_argument("--output", default="data/private/polar_expansion_v0.2")
    args = parser.parse_args()
    print(json.dumps(audit(args.baseline, args.review_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
