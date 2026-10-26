"""Recheck every old mass and chain flag; retain the user's 43,100-row standby."""

import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from freeze_combined_core_pool_v02 import polar_to_marked_core
from rdkit.Chem import Descriptors

from structure_pretrain.glycerol_core_collection.chemistry import canonical, capped_core, parse
from structure_pretrain.glycerol_core_collection.collection import dump_json, file_hash
from structure_pretrain.glycerol_substructure_collection.chemistry import (
    acyclic_long_chain,
    policy,
    representations,
    verify_assembly,
)

ROOT = Path("outputs/glycerol_core_collection_v03_20261026")
SOURCE = Path("data/structure_library_v0.3/polar_library.parquet")
OLD = Path("outputs/headgroup_training_targets_v02_20261026")
SCOPE = {
    "core_mass_above_500_600_policy",
    "extra_head_long_alkyl_or_acyl_chain",
    "extra_glycerol_lipid_anchor_including_short_acyl",
    "extra_fatty_N_acyl_PE_PS_head",
}


def main():
    ROOT.mkdir(exist_ok=True, parents=True)
    source_hash = file_hash(SOURCE)
    old_summary = json.loads((OLD / "background_audit_summary.json").read_text("utf-8"))
    assert source_hash == old_summary["original_SHA256"]
    source = pq.ParquetFile(SOURCE)
    prior = pq.ParquetFile(OLD / "background_195306_audit_P_first.parquet")
    writers = {}
    counts, reasons, partition, recoveries = Counter(), Counter(), Counter(), Counter()
    recovered = []
    maximum_mass_error = 0.0

    def write(name, rows):
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        if name not in writers:
            writers[name] = pq.ParquetWriter(
                ROOT / (name + ".parquet.part"), table.schema, compression="zstd"
            )
        writers[name].write_table(table)

    for a, b in zip(
        source.iter_batches(batch_size=512, columns=["polar_id", "polar_fragment_smiles"]),
        prior.iter_batches(batch_size=512),
        strict=True,
    ):
        audit, standby = [], []
        for src, old in zip(a.to_pylist(), b.to_pylist(), strict=True):
            assert src["polar_id"] == old["polar_id"]
            head = parse(src["polar_fragment_smiles"])
            core = polar_to_marked_core(src["polar_fragment_smiles"])
            mass = Descriptors.MolWt(capped_core(core))
            difference = abs(mass - old["core_average_mass_Da"])
            maximum_mass_error = max(maximum_mass_error, difference)
            assert difference < 1e-5
            polar_mass = Descriptors.MolWt(head)
            old_flags = set(filter(None, old["review_flags"].split(";")))
            flags = old_flags - {"halogen_hold_for_natural_evidence"}
            ring_rescue = "extra_head_long_alkyl_or_acyl_chain" in flags and not acyclic_long_chain(
                head
            )
            if ring_rescue:
                flags.remove("extra_head_long_alkyl_or_acyl_chain")
                counts["legacy_chain_flag_was_ring_path_not_acyclic_chain"] += 1
            over = mass > old["mass_limit_Da"] + 1e-8
            assert over == ("core_mass_above_500_600_policy" in old_flags)
            shift = over and polar_mass <= old["mass_limit_Da"] + 1e-8
            counts["mass_within_Polar_but_over_after_glycerol"] += shift
            counts["independent_full_core_mass_matches_prior"] += 1
            counts["mass_exceeds"] += over
            passed = not flags
            category = (
                "pass"
                if passed
                else "outside_new_scope"
                if flags & SCOPE
                else "chemical_or_protecting_group_hold_only"
            )
            partition[category] += 1
            reasons.update(flags)
            core_text = canonical(core, False)
            if shift:
                standby.append(
                    {
                        "polar_id": src["polar_id"],
                        "marked_core_smiles": core_text,
                        "P_first_polar_smiles": old["P_first_polar_smiles"],
                        "Polar_mass_Da": polar_mass,
                        "glycerol_core_mass_Da": mass,
                        "mass_limit_Da": old["mass_limit_Da"],
                        "other_review_flags": ";".join(
                            sorted(flags - {"core_mass_above_500_600_policy"})
                        ),
                        "status": "user_requested_standby_not_in_main_pool",
                    }
                )
            if passed and not old["future_500_600_policy_pass"]:
                desc = policy(core)
                assert not desc["filter_reasons"], desc["filter_reasons"]
                qa = verify_assembly(core)
                rep = representations(core)
                recovered.append(
                    {
                        **rep,
                        "core_formula": qa["core_formula"],
                        "core_average_mass_Da": mass,
                        "P_count": desc["P_count"],
                        "background_polar_ids": [src["polar_id"]],
                        "expansion_target_id": "",
                        "origin_category": "old_background_readmitted",
                        "display_name": "",
                        "new_split": "unassigned",
                        "full_assembly_H_formula_pass": True,
                    }
                )
                why = (
                    "halogen_and_ring_rule"
                    if ring_rescue and "halogen_hold_for_natural_evidence" in old_flags
                    else "ring_rule"
                    if ring_rescue
                    else "halogen_rule"
                )
                recoveries[why] += 1
            audit.append(
                {
                    "polar_id": src["polar_id"],
                    "core_average_mass_Da_recomputed": mass,
                    "mass_limit_Da": old["mass_limit_Da"],
                    "mass_matches_prior": True,
                    "prior_flags": old["review_flags"],
                    "new_flags": ";".join(sorted(flags)),
                    "v03_pass": passed,
                    "legacy_chain_ring_false_positive": ring_rescue,
                    "mass_shift_standby": shift,
                }
            )
            counts["records"] += 1
        write("phosphate_scope_reaudit", audit)
        write("standby_43100_mass_shift", standby)
        if counts["records"] % 16384 == 0:
            print(json.dumps({"old_rechecked": counts["records"]}), flush=True)
    for name, writer in writers.items():
        writer.close()
        (ROOT / (name + ".parquet.part")).replace(ROOT / (name + ".parquet"))
    assert counts["records"] == 195306
    assert counts["mass_within_Polar_but_over_after_glycerol"] == 43100
    assert file_hash(SOURCE) == source_hash
    pq.write_table(
        pa.Table.from_pylist(recovered), ROOT / "old_recovered_cores.parquet", compression="zstd"
    )
    report = {
        "counts": dict(counts),
        "new_disjoint_partition": dict(partition),
        "new_overlapping_reasons": dict(reasons),
        "readmitted": dict(recoveries),
        "newly_readmitted_total": len(recovered),
        "maximum_independent_mass_error_Da": maximum_mass_error,
        "original_SHA256_unchanged": source_hash,
        "previous_scope_excluded": 77656,
        "mass_43100_saved_separately_not_added": True,
        "halogen_rule_uses_existing_ten_element_vocabulary": True,
        "chain_rule_correction": (
            "only acyclic carbon paths; carbocyclic paths were false positives"
        ),
    }
    dump_json(ROOT / "phosphate_scope_reaudit_summary.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
