"""Recheck source-level fast-gate rejects against target-only element rules."""

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from collect_observed_glycerol_heads_v03 import process_batch
from rdkit import rdBase

from structure_pretrain.glycerol_core_collection.chemistry import parse
from structure_pretrain.glycerol_core_collection.collection import dump_json, file_hash
from structure_pretrain.glycerol_substructure_collection.chemistry import verify_assembly

ROOT = Path("outputs/glycerol_core_collection_v03_20261026")
RESULT = ROOT / "source_target_element_recheck_results.json.gz"


def extract_part(path):
    result_path = path.with_name(path.name.replace("input_", "result_"))
    if result_path.exists():
        with gzip.open(result_path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    with rdBase.BlockLogs():
        accepted, rejected, counts, cids = process_batch(path)
    result = {"accepted": accepted, "rejected": rejected, "counts": counts, "CIDs": cids}
    with gzip.open(result_path.with_suffix(".gz.part"), "wt", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False)
    result_path.with_suffix(".gz.part").replace(result_path)
    return result


def extract():
    with gzip.open(
        ROOT / "raw/source_target_element_recheck.json.gz", "rt", encoding="utf-8"
    ) as stream:
        source = json.load(stream)
    folder = ROOT / "raw/element_recheck_parts"
    folder.mkdir(exist_ok=True)
    jobs = []
    for i in range(0, len(source), 2500):
        path = folder / f"input_{i:06d}.json.gz"
        if not path.exists():
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump(source[i : i + 2500], stream, ensure_ascii=False)
        jobs.append(path)
    accepted, rejected, cids, counts = [], [], [], Counter()
    with ProcessPoolExecutor(max_workers=2) as executor:
        for result in executor.map(extract_part, jobs):
            accepted.extend(result["accepted"])
            rejected.extend(result["rejected"])
            cids.extend(result["CIDs"])
            counts.update(result["counts"])
            print(json.dumps({"source_element_recheck_processed": len(cids)}), flush=True)
    with gzip.open(RESULT, "wt", encoding="utf-8") as stream:
        json.dump(
            {"accepted": accepted, "rejected": rejected, "counts": counts, "CIDs": cids},
            stream,
            ensure_ascii=False,
        )
    print(
        json.dumps(
            {
                "supplement_sources": len(cids),
                "accepted_source_CIDs": counts.get("accepted_source_CIDs", 0),
                "accepted_edges": len(accepted),
            }
        ),
        flush=True,
    )


def merge():
    summary_path = ROOT / "collection_summary_v03.json"
    s = json.loads(summary_path.read_text("utf-8"))
    if s.get("target_element_source_recheck_applied"):
        print("Target-element source recheck was already merged.")
        return
    with gzip.open(RESULT, "rt", encoding="utf-8") as stream:
        sub = json.load(stream)
    selected = set(sub["CIDs"])
    pending_path = ROOT / "pending_source_or_core.parquet"
    pending = pq.read_table(pending_path).to_pylist()
    source_rejected = {r["CID"] for r in pending if r["record_kind"] == "source"}
    assert selected <= source_rejected, "Never double count previously accepted source CIDs"
    path = ROOT / "combined_candidate_cores_v03.parquet"
    assert file_hash(path) == s["combined_parquet_SHA256"]
    rows = pq.read_table(path).to_pylist()
    by_key = {r["anchored_graph_key"]: r for r in rows}
    added = 0
    with gzip.open(ROOT / "observed_source_mapping.jsonl.gz", "at", encoding="utf-8") as stream:
        for item in sub["accepted"]:
            cid, name = item.pop("source_CID"), item.pop("source_name")
            key = item["anchored_graph_key"]
            if key not in by_key:
                proof = verify_assembly(parse(item["target_smiles"]))
                assert proof["core_formula"] == item["core_formula"]
                by_key[key] = {
                    **item,
                    "background_polar_ids": [],
                    "expansion_target_id": "",
                    "origin_category": "observed_PubChem_nonP_terminal_core",
                    "display_name": "Observed substructure from PubChem CID" + str(cid),
                    "new_split": "unassigned",
                    "full_assembly_H_formula_pass": True,
                    "PubChem_source_CIDs": [],
                    "observed_substructure_evidence": True,
                    "source_without_fatty_chains_observed": False,
                    "source_uncut_core_observed": False,
                    "source_without_nonfatty_cuts_observed": False,
                }
                added += 1
            row = by_key[key]
            row["observed_substructure_evidence"] = True
            row["source_roundtrip_H_formula_pass"] = True
            if cid not in row["PubChem_source_CIDs"] and len(row["PubChem_source_CIDs"]) < 12:
                row["PubChem_source_CIDs"].append(cid)
            row["source_without_fatty_chains_observed"] |= not item["observed_fatty_chain_count"]
            row["source_uncut_core_observed"] |= not item["source_cut_substituent_count"]
            row["source_without_nonfatty_cuts_observed"] |= not item[
                "source_nonfatty_substituent_cut_count"
            ]
            stream.write(
                json.dumps(
                    {
                        "source_CID": cid,
                        "target_id": item["target_id"],
                        "source_name": name,
                        "source_url": "https://pubchem.ncbi.nlm.nih.gov/compound/" + str(cid),
                        "observed_fatty_chain_count": item["observed_fatty_chain_count"],
                        "source_cut_substituent_count": item["source_cut_substituent_count"],
                        "source_nonfatty_substituent_cut_count": item[
                            "source_nonfatty_substituent_cut_count"
                        ],
                        "source_glycerol_atom_indices": item["source_glycerol_atom_indices"],
                        "source_head_O_atom_index": item["source_head_O_atom_index"],
                        "source_component_smiles": item.get("source_component_smiles", ""),
                        "source_roundtrip_scope": "connected_source_component",
                        "source_graph_and_H_formula_roundtrip": True,
                        "evidence_kind": "database_substructure_not_natural_occurrence_proof",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    final = sorted(by_key.values(), key=lambda r: r["target_id"])
    assert len({r["target_id"] for r in final}) == len(final)
    for r in final:
        f = dict(re.findall(r"([A-Z][a-z]?)(\d*)", r["core_formula"]))
        r["halogen_count"] = sum(int(f[e] or 1) for e in ("F", "Cl", "Br", "I") if e in f)
    pq.write_table(
        pa.Table.from_pylist(final), path.with_suffix(".parquet.part"), compression="zstd"
    )
    path.with_suffix(".parquet.part").replace(path)
    removed = [r for r in pending if r["CID"] in selected]
    pending = [r for r in pending if r["CID"] not in selected]
    for cid, kind, why, target, position, mass in sub["rejected"]:
        pending.append(
            {
                "CID": cid,
                "record_kind": kind,
                "reasons": why,
                "target_id": target,
                "head_position": position,
                "core_mass_Da": mass,
            }
        )
    pq.write_table(
        pa.Table.from_pylist(removed),
        ROOT / "superseded_source_gate_rejections.parquet",
        compression="zstd",
    )
    pq.write_table(pa.Table.from_pylist(pending), pending_path, compression="zstd")
    reasons = Counter(reason for r in pending for reason in r["reasons"].split(";"))
    examples = defaultdict(list)
    for r in pending:
        for reason in r["reasons"].split(";"):
            if len(examples[reason]) < 12:
                examples[reason].append(r)
    dump_json(ROOT / "rejection_examples_private.json", dict(examples))
    s["source_counts"]["accepted_source_CIDs"] += sub["counts"].get("accepted_source_CIDs", 0)
    s["source_counts"]["accepted_source_core_edges"] += len(sub["accepted"])
    s["new_unique_nonP_cores"] += added
    s["final_nonP_cores"] += added
    s["final_core_count"] = len(final)
    for field, flag in (
        ("nonP_graphs_with_observed_database_substructure", "observed_substructure_evidence"),
        ("nonP_graphs_observed_without_fatty_chains", "source_without_fatty_chains_observed"),
        ("nonP_graphs_observed_as_uncut_whole_cores", "source_uncut_core_observed"),
        (
            "nonP_graphs_observed_without_nonfatty_site_cuts",
            "source_without_nonfatty_cuts_observed",
        ),
    ):
        s[field] = sum(r["P_count"] == 0 and r[flag] for r in final)
    s.update(
        element_restriction_scope="retained_target_core",
        target_element_source_recheck_applied=True,
        target_element_source_recheck_count=len(selected),
        recheck_additional_core_count=added,
        full_assembly_proof_count=len(final),
        rejection_counts_are_events_and_can_overlap=dict(reasons),
        combined_parquet_SHA256=file_hash(path),
    )
    dump_json(summary_path, s)
    print(
        json.dumps(
            {
                "supplement_merged": True,
                "added_unique_cores": added,
                "final_nonP": s["final_nonP_cores"],
                "total": len(final),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if not args.merge_only:
        extract()
    if not args.extract_only:
        merge()
