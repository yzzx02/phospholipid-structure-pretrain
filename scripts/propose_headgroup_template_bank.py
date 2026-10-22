"""Integrate human review and propose CPU-only fixed 200/250/300/350 reference banks.

Per the user's 2026-10-22 correction, the complete frozen pool is available to
template selection. Historical splits are metadata, not eligibility criteria.
"""

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, rdBase

from structure_pretrain.reported_headgroups.chemistry import builder_trial
from structure_pretrain.reported_headgroups.size_audit import describe_polar
from structure_pretrain.reported_headgroups.template_selection import (
    FORMULA_ELEMENTS, FUNCTION_NAMES, FUNCTION_SMARTS, MORGAN, balanced_pick, coverage_balanced_pick,
    bank_diagnostics, coverage, digest, fp, functional_mask, functional_names,
    load_review, phosphate_comparison_key, read_csv, write_csv, nearest_without_self, current_compatibility,
)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def stratum(row):
    if row["average_mass_Da"] > 600:
        return f"P{row['P']}/mass_above_600_known_only"
    mass_bin = min(int(row["average_mass_Da"] // 150), 3)
    return f"P{row['P']}/mass_{mass_bin * 150}_{(mass_bin + 1) * 150}"


def quantiles(values):
    return {str(q): float(np.quantile(values, q)) for q in (0, .1, .25, .5, .75, .9, .95, 1)}


def refresh_compatibility(out):
    """Refresh metadata only; reuse unchanged selection and all exact distances."""
    vocabulary = json.loads(Path('data/private/pretraining_v03/polar_tokenizer.json').read_text())['tokens']
    approved = json.loads((out/'approved_catalog_private.json').read_text(encoding='utf-8'))
    by_id = {}
    for row in approved:
        row.update(current_compatibility(row['polar_smiles'], vocabulary))
        by_id[row['headgroup_id']] = row
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    summary['current_model_eligible_approved_graphs'] = sum(r['training_reference_eligible'] for r in approved)
    summary['current_decoder_target_eligible_approved_graphs'] = sum(r['target_label_eligible'] for r in approved)
    summary['approved_training_blockers'] = dict(Counter(reason for r in approved for reason in r['training_reference_blockers'].split(';') if reason))
    summary['approved_target_label_blockers'] = dict(Counter(reason for r in approved for reason in r['target_label_blockers'].split(';') if reason))
    dump(out/'approved_catalog_private.json', approved)
    write_csv(out/'approved_reference_catalog.csv', approved)
    write_csv(out/'approved_current_model_blocked.csv', [r for r in approved if not r['training_reference_eligible']])
    integrated = read_csv(out/'integrated_review_all_324.csv')
    for row in integrated:
        if row['headgroup_id'] in by_id:
            row.update(by_id[row['headgroup_id']])
    fields = list(dict.fromkeys(k for r in integrated for k in r))
    write_csv(out/'integrated_review_all_324.csv', integrated, fields)
    banks = json.loads((out/'proposed_banks_private.json').read_text(encoding='utf-8'))
    metadata = ('training_reference_eligible', 'training_reference_blockers', 'target_label_eligible', 'target_label_blockers')
    for size, rows in banks.items():
        for row in rows:
            if row['headgroup_id'] in by_id:
                row.update({k: by_id[row['headgroup_id']][k] for k in metadata})
        write_csv(out/f'proposed_bank_{size}.csv', rows, list(dict.fromkeys(k for r in rows for k in r)))
    dump(out/'proposed_banks_private.json', banks)
    dump(out/'summary.json', summary)
    payload_path = out/'authoring/workbook_payload_private.json'
    if payload_path.exists():
        payload = json.loads(payload_path.read_text(encoding='utf-8'))
        payload['summary'] = summary
        for row in payload['rows']:
            if row['headgroup_id'] in by_id:
                row.update({k: by_id[row['headgroup_id']][k] for k in metadata})
        dump(payload_path, payload)
    print(json.dumps({'template_feature_pass': summary['current_model_eligible_approved_graphs'],
                      'decoder_target_label_pass':summary['current_decoder_target_eligible_approved_graphs']},ensure_ascii=False))


def diagnostic_summary(metrics, indices, masks, formulas, bank_masks, bank_formula):
    indices = np.asarray(indices, dtype=int)
    similarity = metrics["nearest_tanimoto"][indices]
    combined = 0
    for mask in bank_masks:
        combined |= int(mask)
    category = {}
    for bit, name in enumerate(FUNCTION_NAMES):
        subset = [i for i in indices if int(masks[i]) & (1 << bit)]
        category[name] = {
            "target_count": len(subset),
            "template_count": sum(bool(int(m) & (1 << bit)) for m in bank_masks),
            "target_fraction_with_similarity_at_least_0.5": (
                float(np.mean(metrics["nearest_tanimoto"][subset] >= .5)) if subset else None),
        }
    elements = {}
    for i, element in enumerate(FORMULA_ELEMENTS):
        targets = int(np.sum(formulas[indices, i] > 0))
        templates = int(np.sum(bank_formula[:, i] > 0))
        elements[element] = {"target_count": targets, "template_count": templates,
                             "uncovered": targets > 0 and templates == 0}
    element_sets = defaultdict(list)
    for i in indices:
        signature = ";".join(e for e, count in zip(FORMULA_ELEMENTS, formulas[i]) if count)
        element_sets[signature].append(int(i))
    template_sets = Counter(";".join(e for e, c in zip(FORMULA_ELEMENTS, f) if c)
                            for f in bank_formula)
    return {
        **coverage(similarity),
        "very_far_threshold": "Morgan Tanimoto < 0.3 (operational diagnostic, not a chemical verdict)",
        "very_far_count": int(np.sum(similarity < .3)),
        "below_0.5_count": int(np.sum(similarity < .5)),
        "nearest_tanimoto_quantiles": quantiles(similarity),
        "formula_distance_quantiles": {k: quantiles(v[indices]) for k, v in metrics.items()
                                        if "formula" in k or "element_L1" in k},
        "same_formula_different_graph_fraction": float(np.mean(metrics["minimum_formula_L1_including_H"][indices] == 0)),
        "functional_categories": category,
        "elements": elements,
        "element_sets": {signature: {"target_count": len(idx), "template_count": template_sets[signature],
                                      "fraction_with_similarity_at_least_0.5": float(np.mean(metrics['nearest_tanimoto'][idx] >= .5))}
                         for signature, idx in sorted(element_sets.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/reported_headgroups_v02_20261021"))
    parser.add_argument("--review", type=Path, default=Path("../头基预测人工审核"))
    parser.add_argument("--output", type=Path, default=Path("outputs/headgroup_template_proposal_20261022"))
    parser.add_argument('--compatibility-only', action='store_true', help='Refresh template/target compatibility, preserving graph selection')
    args = parser.parse_args()
    start = time.perf_counter()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.compatibility_only:
        refresh_compatibility(out)
        return
    library_path = Path("data/structure_library_v0.3/polar_library.parquet")
    groups_path = Path("data/private/pretraining_v03/polar_groups.parquet")
    descriptor_path = Path("outputs/polar_size_scope_audit_20261022/per_graph_descriptors_private.parquet")
    source_csv = args.source / "reported_headgroups_v02.csv"
    review_json, review_csv = [args.review / ("headgroup_manual_review." + ext) for ext in ("json", "csv")]
    hashes = {str(p.resolve()): digest(p) for p in (source_csv, review_json, review_csv, library_path,
                                                  groups_path, descriptor_path)}
    source_rows, review = load_review(source_csv, review_json, review_csv)
    snapshot = out / "human_review_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in (review_json, review_csv):
        target = snapshot / path.name
        if target.exists() and digest(target) != digest(path):
            raise ValueError("Never overwrite a different human review snapshot")
        if not target.exists():
            shutil.copyfile(path, target)
    print("Human JSON/CSV/source integrity passed", flush=True)
    library = pq.read_table(library_path).to_pylist()
    descriptions = {r["polar_id"]: r for r in pq.read_table(descriptor_path).to_pylist()}
    groups = {r["canonical_polar_smiles"]: r for r in pq.read_table(groups_path).to_pylist()}
    assert len(library) == len(groups) == len(descriptions) == 195306
    splits = [groups[r["polar_fragment_smiles"]]["split"] for r in library]
    descriptors = [descriptions[r["polar_id"]] for r in library]
    assert all(r['polar_fragment_smiles'] == d['polar_fragment_smiles'] for r, d in zip(library, descriptors))
    cache = out / "selection_cache_private.npz"
    cache_meta = out / "selection_cache_manifest.json"
    feature_signature = {'schema': 'rooted-polar-fingerprint-functional-v1',
                         'functional_SMARTS': FUNCTION_SMARTS,
                         'Morgan_radius': 2, 'Morgan_bits': 2048, 'includeChirality': False}
    if cache.exists() and cache_meta.exists():
        manifest = json.loads(cache_meta.read_text())
        assert manifest['source_hashes'] == hashes
        assert manifest['rdkit'] == rdBase.rdkitVersion
        assert manifest['feature_signature'] == feature_signature
        cached = np.load(cache, allow_pickle=False)
        packed, keys, masks = [cached[k] for k in ("packed", "keys", "masks")]
    else:
        packed = np.zeros((len(library), 256), dtype=np.uint8)
        keys, masks = [], []
        for i, row in enumerate(library):
            mol = Chem.MolFromSmiles(row["polar_fragment_smiles"])
            packed[i] = np.frombuffer(DataStructs.BitVectToBinaryText(MORGAN.GetFingerprint(mol)), dtype=np.uint8)
            keys.append(phosphate_comparison_key(row["polar_fragment_smiles"]))
            masks.append(functional_mask(mol))
            if (i + 1) % 20000 == 0:
                print(f"CPU graph/fingerprint/features {i + 1}/{len(library)}", flush=True)
        keys, masks = np.asarray(keys), np.asarray(masks, dtype=np.uint32)
        np.savez(cache, packed=packed, keys=keys, masks=masks)
        dump(cache_meta, {"source_hashes": hashes, "graphs": len(library), "rdkit": rdBase.rdkitVersion,
                          'feature_signature':feature_signature})
    fingerprints = [DataStructs.CreateFromBinaryText(bytes(r)) for r in packed]
    tokens = set(json.loads(Path("data/private/pretraining_v03/polar_tokenizer.json").read_text())["tokens"])
    approved, integrated, seeds, normalization_map = [], [], [], []
    key_members = defaultdict(list)
    for row in source_rows:
        key_members[phosphate_comparison_key(row["polar_smiles"])].append(row)
    for row in source_rows:
        decision = review["decisions"].get(row["headgroup_id"], {})
        entry = {**row, "human_decision": decision.get("decision", ""),
                 "human_note": decision.get("note", "")}
        entry["phosphate_comparison_key"] = phosphate_comparison_key(row["polar_smiles"])
        entry["frozen_split"] = groups.get(row["polar_smiles"], {}).get("split", "external")
        if entry["human_decision"] == "keep":
            descriptor = describe_polar(row["polar_smiles"])
            entry.update({k: descriptor[k] for k in (*FORMULA_ELEMENTS, "open_polar_formula", "average_mass_Da", "smiles_tokens")})
            entry["token_length_including_BOS_EOS"] = descriptor["smiles_tokens"] + 2
            entry.update(current_compatibility(row['polar_smiles'], tokens))
            approved.append(entry)
        integrated.append(entry)
    seen = set()
    for row in approved:
        # The conceptual bank preserves every human-approved graph, including
        # those requiring future Builder/vocabulary/length work before training.
        if row["human_decision"] == "keep":
            if row["phosphate_comparison_key"] in seen:
                row["training_reference_eligible"] = False
                row["training_reference_blockers"] += ";approved_phosphate_protonation_duplicate"
            else:
                seen.add(row["phosphate_comparison_key"])
                seeds.append(row)
    for key, members in key_members.items():
        approved_members = [r for r in approved if r["phosphate_comparison_key"] == key]
        if len(members) > 1 and approved_members:
            representative = sorted(approved_members, key=lambda r: (not r["training_reference_eligible"], r["headgroup_id"]))[0]
            for member in members:
                normalization_map.append({"headgroup_id": member["headgroup_id"],
                                          "representative_id": representative["headgroup_id"],
                                          "original_polar_smiles": member["polar_smiles"],
                                          "comparison_only_key": key,
                                          "human_decision": review["decisions"].get(member["headgroup_id"], {}).get("decision", ""),
                                          "action": "preserve_source_and_decision; alias_for_lookup; no_new_template"})
    all_fields = list(dict.fromkeys(k for row in integrated for k in row))
    write_csv(out / "integrated_review_all_324.csv", integrated, all_fields)
    write_csv(out / "approved_reference_catalog.csv", approved)
    write_csv(out / "approved_current_model_blocked.csv", [r for r in approved if not r["training_reference_eligible"]])
    write_csv(out / "phosphate_protonation_aliases.csv", normalization_map,
              ["headgroup_id", "representative_id", "original_polar_smiles", "comparison_only_key", "human_decision", "action"])
    approved_ids = {r["headgroup_id"] for r in approved}
    alias_to_id = {r["headgroup_id"]: r["representative_id"] for r in normalization_map}
    for filename in ("headgroup_sources.csv", "headgroup_name_alias_mapping.csv"):
        original = read_csv(args.source / filename)
        merged = [{**r, "original_headgroup_id": r["headgroup_id"],
                   "headgroup_id": alias_to_id.get(r["headgroup_id"], r["headgroup_id"])}
                  for r in original if r["headgroup_id"] in approved_ids or r["headgroup_id"] in alias_to_id]
        write_csv(out / filename, merged)
    # Near neighbours are suggestions only; all original approved graphs survive.
    seed_fps = [fp(r["polar_smiles"]) for r in seeds]
    approved_fps = [fp(r["polar_smiles"]) for r in approved]
    close_pairs = []
    for i, fingerprint in enumerate(approved_fps):
        for j, similarity in enumerate(DataStructs.BulkTanimotoSimilarity(fingerprint, approved_fps[:i])):
            if similarity >= .85:
                close_pairs.append({"headgroup_id_1": approved[j]["headgroup_id"],
                                    "name_1": approved[j]["abbreviation"] or approved[j]["full_name"],
                                    "headgroup_id_2": approved[i]["headgroup_id"],
                                    "name_2": approved[i]["abbreviation"] or approved[i]["full_name"],
                                    "Morgan_Tanimoto": similarity, "automatic_deletion": False})
    write_csv(out / "approved_similar_pairs_for_optional_review.csv", close_pairs,
              ["headgroup_id_1", "name_1", "headgroup_id_2", "name_2", "Morgan_Tanimoto", "automatic_deletion"])
    known_keys = set(key_members)
    candidate_indices, reasons = [], Counter()
    seen_candidate = set()
    for i, (row, d, split, key) in enumerate(zip(library, descriptors, splits, keys)):
        reason = None
        if str(key) in known_keys:
            reason = "already_in_reviewed_source_catalog"
        elif d["extra_anchor"]:
            reason = "extra_lipid_anchor"
        elif d["average_mass_Da"] > 600:
            reason = "mass_above_600"
        elif d["smiles_tokens"] + 2 > 84:
            reason = "current_decoder_length"
        elif str(key) in seen_candidate:
            reason = "candidate_protonation_duplicate"
        if reason:
            reasons[reason] += 1
        else:
            seen_candidate.add(str(key))
            candidate_indices.append(i)
    candidate_indices.sort(key=lambda i: library[i]["polar_id"])
    candidate_fps = [fingerprints[i] for i in candidate_indices]
    formula = np.asarray([[d[e] for e in FORMULA_ELEMENTS] for d in descriptors], dtype=np.int16)
    seed_formula = np.asarray([[r[e] for e in FORMULA_ELEMENTS] for r in seeds], dtype=np.int16)
    seed_masks = [functional_mask(Chem.MolFromSmiles(r["polar_smiles"])) for r in seeds]
    candidate_formula = formula[candidate_indices]
    seed_strata = [stratum(r) for r in seeds]
    candidate_strata = [stratum(descriptors[i]) for i in candidate_indices]
    print(f"Approved {len(approved)}; conceptual seed {len(seeds)}; full-pool provisional candidates {len(candidate_indices)}", flush=True)
    diverse_picks, _ = balanced_pick(seed_fps, candidate_fps, seed_formula, candidate_formula,
                                  seed_masks, masks[candidate_indices], seed_strata, candidate_strata, 350)
    seed_keys = [r['phosphate_comparison_key'] for r in seeds]
    random = np.random.default_rng(20261022)
    probe_indices = random.choice(len(library), 4096, replace=False)
    picks, traces, coreset = coverage_balanced_pick(
        seed_fps, candidate_fps, seed_formula, candidate_formula, seed_masks,
        masks[candidate_indices], seed_strata, candidate_strata, 350,
        [fingerprints[i] for i in probe_indices], keys[probe_indices], seed_keys,
        keys[candidate_indices], diverse_picks,
    )
    coreset['probe_random_seed'] = 20261022
    baseline = {}
    for method, offsets in [('diversity_only', diverse_picks),
                            *[(f'uniform_random_{s}', np.random.default_rng(s).choice(len(candidate_indices), 350-len(seeds), replace=False).tolist()) for s in (17,42,101)]]:
        added = [candidate_indices[i] for i in offsets[:250-len(seeds)]]
        values = nearest_without_self(fingerprints, keys, seed_fps+[fingerprints[i] for i in added], seed_keys+[str(keys[i]) for i in added])
        baseline[method] = coverage(values)
    dump(out/'selection_baselines_250.json', baseline)
    selected_indices = [candidate_indices[i] for i in picks]
    new_rows = []
    for rank, (index, trace) in enumerate(zip(selected_indices, traces), 1):
        row, d = library[index], descriptors[index]
        trial = builder_trial(row["polar_fragment_smiles"])
        assert trial["independent_element_H_balance"]
        new_rows.append({"headgroup_id": row["polar_id"], "polar_id": row["polar_id"],
                         "polar_smiles": row["polar_fragment_smiles"], "selection_rank": rank,
                         "first_bank_size": len(seeds) + rank,
                         "display_name": f"候选 T{rank:03d}", "human_decision": "",
                         "human_note": "", "selection_status": "algorithm_proposal_not_approved",
                         "frozen_split": splits[index], "builder_compatible": True,
                         "builder_formula": trial["neutral_formula"],
                         "functional_categories": functional_names(masks[index]),
                         **{k: d[k] for k in (*FORMULA_ELEMENTS, "open_polar_formula", "average_mass_Da", "smiles_tokens")},
                         "chemistry_family": row["chemistry_family"],
                         "structure_origin": row["structure_origin"],
                         "polar_plausibility_tier": row["polar_plausibility_tier"],
                         "is_full_lipid_reported": row["is_full_lipid_reported"],
                         "evidence_source": row["evidence_source"],
                         "source_cids": row["source_cids"], "parent_source_cids": row["parent_source_cids"],
                         "source_attachment_provenance": row["source_attachment_provenance"],
                         "phosphate_comparison_key": str(keys[index]), **trace})
    write_csv(out / "new_template_proposals_ranked.csv", new_rows)
    selected_set = set(selected_indices)
    assert all(str(keys[i]) not in known_keys for i in selected_indices)
    diagnostics, bank_payload = {}, {}
    partition = {"all_frozen": list(range(len(library))),
                 **{split: [i for i, value in enumerate(splits) if value == split]
                    for split in ("train", "validation", "test")}}
    sizes = [len(seeds), 200, 250, 300, 350]
    for size in sizes:
        added = selected_indices[:size - len(seeds)]
        bank_fps = seed_fps + [fingerprints[i] for i in added]
        bank_formula = np.vstack([seed_formula, formula[added]]) if added else seed_formula
        bank_keys = [r["phosphate_comparison_key"] for r in seeds] + [str(keys[i]) for i in added]
        bank_masks = seed_masks + [int(masks[i]) for i in added]
        metrics = bank_diagnostics(fingerprints, formula, keys, bank_fps, bank_formula, bank_keys)
        diagnostics[str(size)] = {name: diagnostic_summary(metrics, indices, masks, formula, bank_masks, bank_formula)
                                   for name, indices in partition.items()}
        # A fixed target population disjoint from every proposed template, for honest size comparisons.
        seed_keys = {r["phosphate_comparison_key"] for r in seeds}
        fixed_unseen = [i for i in range(len(library)) if i not in selected_set and str(keys[i]) not in seed_keys]
        diagnostics[str(size)]["fixed_unseen_targets"] = diagnostic_summary(metrics, fixed_unseen, masks, formula, bank_masks, bank_formula)
        pq.write_table(pa.table({"polar_id": [r["polar_id"] for r in library], "split": splits,
                                 **{k: v for k, v in metrics.items()}}), out / f"target_coverage_bank_{size}_private.parquet")
        bank_rows = [{**r, "selection_status": "human_approved", "selection_rank": 0,
                      "display_name": r["abbreviation"] or r["full_name"],
                      "functional_categories": functional_names(mask)} for r, mask in zip(seeds, seed_masks)]
        bank_rows += new_rows[:size - len(seeds)]
        fields = list(dict.fromkeys(k for r in bank_rows for k in r))
        write_csv(out / f"proposed_bank_{size}.csv", bank_rows, fields)
        bank_payload[str(size)] = bank_rows
        print(f"Coverage full195306/bank{size}: sim>=.5 {diagnostics[str(size)]['all_frozen']['fraction_at_least_0.5']:.3%}", flush=True)
    summary = {
        "human_decision_counts": dict(Counter(d.get("decision", "") for d in review["decisions"].values())),
        "source_graphs": len(source_rows), "approved_catalog_graphs": len(approved),
        "approved_builder_pass": sum(r["builder_compatible"] == "True" for r in approved),
        "approved_original_split_counts": dict(Counter(r["frozen_split"] for r in approved)),
        "approved_training_blockers": dict(Counter(reason for r in approved for reason in r["training_reference_blockers"].split(';') if reason)),
        "conceptual_human_approved_seed_graphs": len(seeds),
        "current_model_eligible_approved_graphs": sum(r["training_reference_eligible"] for r in approved),
        "current_decoder_target_eligible_approved_graphs": sum(r["target_label_eligible"] for r in approved),
        "approved_target_label_blockers": dict(Counter(reason for r in approved for reason in r['target_label_blockers'].split(';') if reason)),
        "approved_similar_pairs_at_least_0.85": len(close_pairs),
        "protonation_alias_record_count": len(normalization_map),
        "candidate_pool": len(candidate_indices), "candidate_exclusion_first_reason_counts": dict(reasons),
        "candidate_pool_scope": "complete original <=600 Da pool regardless of old split; no extra lipid anchor, current tokenizer; no new C/N/S or halogen bans",
        "method": {"algorithm": "coverage-aware facility-location coreset plus diversity and mass/P balance",
                   "coreset": coreset,
                   "fingerprint": "Morgan radius2/2048/chirality false; attachment retained",
                   "formula_scales_C_H_N_O_P_S_F_Cl_Br_I": [6, 12, 2, 4, 1, 1, 1, 1, 1, 1],
                   "balance": "mass150Da bins x P_count, square-root candidate-population targets, maximum 1.25 bonus",
                   "new_template_max_similarity_to_previous": .85,
                   "functional_SMARTS": FUNCTION_SMARTS,
                   "historical_splits": "ignored for selection at user's direction; files preserved, future evaluation must be redesigned",
                   "normalization": "terminal phosphate-O protonation comparison only; no changed labels",
                   "self_match": "each target's own phosphate-protonation group excluded from all nearest distances"},
        "proposed_sizes": sizes, "default_review_size": 250,
        "new_proposals_for_250": 250 - len(seeds), "new_proposals_for_350": len(new_rows),
        "proposals_origin_counts_250": dict(Counter(r['structure_origin'] for r in new_rows[:250-len(seeds)])),
        "proposals_historical_split_counts_250": dict(Counter(r['frozen_split'] for r in new_rows[:250-len(seeds)])),
        "proposals_max_open_polar_mass_350": max(r['average_mass_Da'] for r in new_rows),
        "all_selected_builder_and_independent_formula_H_balance_pass": True,
        "no_new_proposal_human_approved": True, "training_started": False,
        "frozen_library_changed": False, "source_hashes": hashes,
        "rdkit": rdBase.rdkitVersion, "wall_seconds": time.perf_counter() - start,
    }
    for path, expected in hashes.items():
        assert digest(path) == expected, "An immutable input changed"
    dump(out / "summary.json", summary)
    dump(out / "coverage_summary.json", diagnostics)
    dump(out / "proposed_banks_private.json", bank_payload)
    dump(out / "approved_catalog_private.json", approved)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_hashes', 'method')}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
