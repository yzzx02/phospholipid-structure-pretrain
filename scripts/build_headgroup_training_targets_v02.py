"""One-edit, anchor-preserving expansion of ALL accepted lipid head families."""

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase

from structure_pretrain.glycerol_core_collection.collection import (
    dump_json,
    file_hash,
    legacy_core,
    read_csv,
    write_csv,
)
from structure_pretrain.headgroup_target_expansion.chemistry import (
    amino_acid_from_pg_polar,
    aminoacylate,
    free_head_alcohols,
    local_edits,
    policy,
    representations,
    standard_core,
)

CONFIG = Path("configs/headgroup_training_targets_v02_20261026.json")


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert (
        config["phosphorus_core_mass_limit_Da"] == config["nonphosphorus_core_mass_limit_Da"] == 500
    )
    assert config["sugar_or_inositol_core_mass_limit_Da"] == 600
    root = Path(config["root"])
    root.mkdir(exist_ok=True, parents=True)
    base = json.loads(Path(config["base_catalog"]).read_text(encoding="utf-8"))
    records, source_events, all_existing = {}, [], {}
    rejected = Counter()
    rejected_examples = defaultdict(list)
    inputs = {
        str(CONFIG): file_hash(CONFIG),
        config["background"]: file_hash(config["background"]),
        config["base_catalog"]: file_hash(config["base_catalog"]),
        str(root / config["amino_acid_atlas"]): file_hash(root / config["amino_acid_atlas"]),
    }
    parents = {}

    def add(
        mol,
        origin,
        name,
        *,
        parent=None,
        rule="",
        sites=(),
        donor="",
        source_id="",
        urls=(),
        natural=False,
        prior_ids=(),
    ):
        rep = representations(mol)
        desc = policy(mol, generated=origin == "rule_variant")
        event = {
            "target_id": rep["target_id"],
            "origin": origin,
            "name": name,
            "parent_target_id": parent,
            "rule": rule,
            "site_atom_indices": list(sites),
            "donor_amino_acid": donor,
            "source_id": source_id,
            "source_urls": list(urls),
            "specific_natural_structure_verified": bool(natural),
            "filter_reasons": desc["filter_reasons"],
        }
        source_events.append(event)
        if desc["filter_reasons"]:
            for reason in desc["filter_reasons"]:
                rejected[reason] += 1
                if len(rejected_examples[reason]) < 20:
                    rejected_examples[reason].append({**rep, "name": name, "origin": origin})
            return None
        ident = rep["target_id"]
        if ident not in records:
            records[ident] = {
                **rep,
                **desc,
                "display_name": name,
                "names": set(),
                "origins": set(),
                "source_urls": set(),
                "parent_target_ids": set(),
                "rules": set(),
                "amino_acids": set(),
                "existing_catalog_ids": set(),
                "natural_core_evidence": False,
                "human_decision": "未判断",
                "human_note": "",
                "provenance_count": 0,
                "is_proposed_training_target": True,
            }
        row = records[ident]
        row["names"].add(name)
        row["origins"].add(origin)
        row["rules"].update([rule] if rule else [])
        row["amino_acids"].update([donor] if donor else [])
        row["source_urls"].update(urls)
        row["existing_catalog_ids"].update(prior_ids)
        if parent:
            row["parent_target_ids"].add(parent)
        row["natural_core_evidence"] |= natural
        row["provenance_count"] += 1
        if origin == "existing_reference":
            if len(name) < len(row["display_name"]):
                row["display_name"] = name
            row["human_decision"] = "沿用参考"
        return ident

    # Re-evaluate the original reference under the NEW500/600 policy; do not edit it.
    for row in base:
        try:
            mol, notes = standard_core(row["core_smiles_chain_sites"])
            rep = representations(mol)
            all_existing[rep["anchored_graph_key"]] = row["core_id"]
            if "exclude" in row["manual_decisions"]:
                continue
            ident = add(
                mol,
                "existing_reference",
                row["display_name"],
                source_id=row["core_id"],
                urls=row["source_urls"],
                natural=row["natural_core_verified"],
                prior_ids=[row["core_id"]],
            )
            if ident:
                parents.setdefault(ident, mol)
        except (ValueError, RuntimeError, AssertionError) as error:
            rejected[type(error).__name__ + ":" + str(error)[:80]] += 1
    references = set(records)
    atlas = read_csv(root / config["amino_acid_atlas"])
    aa_audit = []
    acids = {}
    source_path = str((root / config["amino_acid_atlas"]).resolve())
    for row in atlas:
        acids[row["abbreviation"]] = amino_acid_from_pg_polar(row["pg_aa_polar_smiles"])
        for family, field, statusfield in (
            ("direct_P", "phosphate_aa_polar_smiles", "phosphate_aa_status"),
            ("PG_terminal_ester", "pg_aa_polar_smiles", "pg_aa_status"),
        ):
            mol, notes = standard_core(legacy_core(row[field])["core_smiles_chain_sites"])
            rep = representations(mol)
            desc = policy(mol)
            existed = rep["anchored_graph_key"] in all_existing
            natural_rule = row["abbreviation"] in ("Ala", "Lys") and family == "PG_terminal_ester"
            urls = ["https://pubmed.ncbi.nlm.nih.gov/26261323/"] if natural_rule else []
            ident = add(
                mol,
                "user_amino_acid_atlas",
                row["abbreviation"] + " / " + family,
                source_id=source_path + ":" + row["abbreviation"] + ":" + family,
                donor=row["abbreviation"],
                urls=urls,
                natural=natural_rule,
            )
            if ident:
                parents.setdefault(ident, mol)
            aa_audit.append(
                {
                    "amino_acid": row["amino_acid"],
                    "abbreviation": row["abbreviation"],
                    "series": family,
                    "target_id": rep["target_id"],
                    "already_in_existing_reference": existed,
                    "existing_catalog_id": all_existing.get(rep["anchored_graph_key"], ""),
                    "included_in_target_proposal": bool(ident),
                    "filter_reasons": desc["filter_reasons"],
                    "source_status_verbatim": row[statusfield],
                    "original_polar_smiles": row[field],
                    "P_first_polar_smiles": rep["polar_P_first_smiles"],
                    "target_smiles": rep["target_smiles"],
                    "core_formula": desc["core_formula"],
                    "core_average_mass_Da": desc["core_average_mass_Da"],
                    "mass_limit_Da": desc["mass_limit_Da"],
                    "normalization_notes": notes,
                    "source_kind": "user_supplied_proposed_structure; literature only where separately linked",  # noqa: E501
                    "source_url": "; ".join(urls),
                }
            )
    before_variants = set(records)
    # Apply each rule once to the accepted seed, never recursively to new products.
    # This is a deterministic bounded neighbourhood, not an unconstrained generator.
    aminoester = Chem.MolFromSmarts("[OX2D2]-C(=O)-[CX4]-[N]")
    attempts = Counter()
    valid_rule_edges = Counter()
    for number, (parent, mol) in enumerate(sorted(parents.items()), 1):
        label = records[parent]["display_name"]
        for rule, sites, apply in local_edits(mol):
            attempts[rule] += 1
            try:
                with rdBase.BlockLogs():
                    product = apply()
                ident = add(
                    product,
                    "rule_variant",
                    label + " / " + rule,
                    parent=parent,
                    rule=rule,
                    sites=sites,
                )
                if ident:
                    valid_rule_edges[rule] += 1
            except (ValueError, RuntimeError, AssertionError) as error:
                rejected["invalid_edit:" + rule + ":" + type(error).__name__] += 1
        if not mol.HasSubstructMatch(aminoester):
            for site in free_head_alcohols(mol):
                for aa, acid in acids.items():
                    rule = "alpha_aminoacyl_ester"
                    attempts[rule] += 1
                    try:
                        with rdBase.BlockLogs():
                            product = aminoacylate(mol, site, acid)
                        ident = add(
                            product,
                            "rule_variant",
                            label + " / " + aa + " ester",
                            parent=parent,
                            rule=rule,
                            sites=[site],
                            donor=aa,
                        )
                        if ident:
                            valid_rule_edges[rule] += 1
                    except (ValueError, RuntimeError, AssertionError) as error:
                        rejected["invalid_edit:" + rule + ":" + type(error).__name__] += 1
        if number % 20 == 0:
            print(
                json.dumps(
                    {
                        "seed_processed": number,
                        "seeds": len(parents),
                        "unique_targets": len(records),
                    }
                ),
                flush=True,
            )

    background = pq.read_table(config["background_sidecar"], columns=["future_core_id", "polar_id"])
    # Full anchored equality is checked in the audit; this is a preliminary capped-core cross-reference.  # noqa: E501
    from structure_pretrain.glycerol_core_collection.chemistry import stable_id

    frozen_keys = defaultdict(list)
    for row in background.to_pylist():
        frozen_keys[row["future_core_id"]].append(row["polar_id"])
    final = []
    for ident, row in records.items():
        for key, value in list(row.items()):
            if isinstance(value, set):
                row[key] = sorted(value)
        row["origin_category"] = (
            "既有头基"
            if ident in references
            else "图册新增氨基酸头基"
            if ident in before_variants
            else "规则扩展变体"
        )
        row["in_background_capped_core"] = stable_id(row["capped_core_smiles"], "GC") in frozen_keys
        row["source_name_count"] = len(row["names"])
        row["names"] = sorted(row["names"], key=len)[:12]
        # Human names remain descriptive; no invented official lipid abbreviation.
        row["human_note"] = (
            "来源位点与原始立体标注保留；模板不按立体异构增号"
            if row["existing_catalog_ids"]
            else ""
        )
        final.append(row)
    final.sort(
        key=lambda r: (
            r["origin_category"] != "既有头基",
            r["origin_category"] != "图册新增氨基酸头基",
            r["display_name"],
            r["target_id"],
        )
    )
    with gzip.open(
        root / "training_target_catalog_private.json.gz", "wt", encoding="utf-8"
    ) as stream:
        json.dump(final, stream, ensure_ascii=False)
    pq.write_table(
        pa.Table.from_pylist(final), root / "training_targets.parquet", compression="zstd"
    )
    write_csv(root / "training_targets.csv", final)
    write_csv(root / "amino_acid_atlas_40_audit.csv", aa_audit)
    with gzip.open(root / "derivation_sources.jsonl.gz", "wt", encoding="utf-8") as stream:
        for event in source_events:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    dump_json(root / "rejection_examples_private.json", dict(rejected_examples))
    summary = {
        "proposed_training_target_count": len(final),
        "scope": "anchored glycerol plus head cores; no fatty chain combinations; background is NOT included in total",  # noqa: E501
        "background_count_is_not_target_count": True,
        "template_bank_count_is_not_target_count": True,
        "category_counts_disjoint": dict(Counter(r["origin_category"] for r in final)),
        "reference_targets_after_new_mass_policy": len(references),
        "imported_atlas_records": len(aa_audit),
        "atlas_already_present_in_reference": sum(
            r["already_in_existing_reference"] for r in aa_audit
        ),
        "atlas_eligible_records": sum(r["included_in_target_proposal"] for r in aa_audit),
        "atlas_new_unique_eligible_graphs": len(before_variants - references),
        "nonP_targets": sum(r["P_count"] == 0 for r in final),
        "sugar_or_inositol_targets": sum(r["mass_limit_Da"] == 600 for r in final),
        "aminoacyl_rule_target_graphs": sum("alpha_aminoacyl_ester" in r["rules"] for r in final),
        "attempted_edits_by_rule": dict(attempts),
        "accepted_derivation_edges_by_rule": dict(valid_rule_edges),
        "rejection_reason_counts_overlap": dict(rejected),
        "parent_seed_count": len(parents),
        "mass_limits_Da": {"ordinary_P_or_nonP": 500, "sugar_or_inositol": 600},
        "new_targets_are_proposals_not_confirmed_natural_lipids": True,
        "current_V2_training_inputs_changed": False,
        "new_library_training_started": False,
        "input_SHA256": inputs,
        "configuration_SHA256": file_hash(CONFIG),
    }
    for name, expected in inputs.items():
        assert file_hash(name) == expected
    dump_json(root / "target_summary.json", summary)
    (root / "authoring").mkdir(exist_ok=True)
    with gzip.open(
        root / "authoring/workbook_payload_private.json.gz", "wt", encoding="utf-8"
    ) as stream:
        json.dump(
            {"summary": summary, "targets": final, "amino_acids": aa_audit},
            stream,
            ensure_ascii=False,
        )
    print(
        json.dumps(
            {
                "complete": True,
                "targets": len(final),
                "categories": summary["category_counts_disjoint"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
