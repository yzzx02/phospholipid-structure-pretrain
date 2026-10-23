"""Stream V2 BOS-only raw/constraint generation, truth-free acceptance and metrics."""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from structure_pretrain.formal_v2.common import (
    ROOT,
    contract,
    digest,
    emit,
    read,
    verify_data,
    write,
)
from structure_pretrain.formal_v2.generation import (
    StreamingAggregate,
    generate,
    inference_inputs,
    score_beams,
    target_metrics,
)
from structure_pretrain.formal_v2.retention import owned_bytes
from structure_pretrain.reported_headgroups.formal_template_data import V03, CoreData, make_model
from structure_pretrain.training.quality_v03 import load_builder_rules
from structure_pretrain.utils.attention import attention_backend
from train_template_guided_fp5000 import initialize


def cohort(root, name):
    split = np.load(root / "data/splits.npy")
    if name in ("validation_full", "test"):
        return np.flatnonzero(split == (1 if name == "validation_full" else 2))
    destination = root / "audit/validation_pilot300_private.npy"
    if destination.exists():
        chosen = np.load(destination)
        if len(chosen) != 300 or len(set(chosen)) != 300 or not np.all(split[chosen] == 1):
            raise ValueError("Frozen pilot cohort changed or invalid")
        return chosen
    rng = np.random.default_rng(20261023)
    groups = np.flatnonzero(split == 1)
    lengths = np.load(root / "data/token_lengths.npy")
    occupancy = np.load(root / "data/occupancy.npy")
    masks = np.load(root / "data/functional_masks.npy")
    formulas = np.load(root / "data/polar_formulas.npy")
    sims = np.load(root / "data/reference_similarities.npy")[groups, occupancy[groups], 0]
    partitions = {}
    for i, g in enumerate(groups):
        key = (
            int(occupancy[g]),
            int(np.digitize(sims[i], [0.4, 0.6, 0.8])),
            int(np.digitize(lengths[g], [24, 40, 60])),
            int(min(formulas[g, 4], 3)),
            int(np.digitize(formulas[g, 0], [6, 11, 16, 26])),
        )
        partitions.setdefault(key, []).append(int(g))
    selected = set()
    for bit in range(16):
        hits = groups[(masks[groups] & (1 << bit)) != 0]
        if len(hits):
            selected.add(int(rng.choice(hits)))
    for key in sorted(partitions):
        if len(selected) >= 300:
            break
        selected.add(int(rng.choice(partitions[key])))
    remaining = np.asarray(sorted(set(groups) - selected))
    selected.update(rng.choice(remaining, 300 - len(selected), replace=False).tolist())
    chosen = np.asarray(sorted(selected), dtype=np.int64)
    np.save(destination, chosen)
    write(
        root / "audit/validation_pilot_cohort.json",
        {
            "N": 300,
            "split": "validation",
            "seed": 20261023,
            "sha256": digest(destination),
            "strata": (
                "functional category, chain occupancy, similarity, output length, P "
                "count and carbon bins"
            ),
            "selected_before_generation": True,
        },
    )
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cohort", choices=("validation_pilot", "validation_full", "test"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = ROOT.resolve(), args.output.resolve()
    if (
        not os.environ.get("POLAR_SUPERVISOR_PID")
        or not torch.cuda.is_available()
        or (root / "GPU_STOP.json").exists()
    ):
        raise ValueError("Independent CUDA supervision and no stop latch required")
    if out.parent != root or out.exists():
        raise ValueError("Fresh private V2 evaluation directory required")
    out.mkdir()

    def event(**row):
        row["unix_time"] = time.time()
        with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        emit(**row)

    event(stage="verifying_immutable_data")
    verify_data(root)
    binding = contract(root)
    initialize()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_relative_to(root):
        raise ValueError("Formal V2 checkpoint required")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["contract"] != binding or state["schema"] != "formal_v2_full_restart":
        raise ValueError("Checkpoint/source/data mismatch")
    model = make_model(root).cuda().eval()
    model.load_state_dict(state["model"])
    del state
    data = CoreData(root, "cuda")
    groups = cohort(root, args.cohort)
    heads = (
        pq.read_table(V03 / "polar_groups.parquet", columns=["canonical_polar_smiles"])
        .column(0)
        .to_pylist()
    )
    train_keys = set(np.load(root / "data/comparison_keys.npy")[data.splits == 0])
    bank = read(root / "data/template_identity_private.json")
    vocab = read(V03 / "chain_vocab.json")["tokens"]
    chains, known_heads = load_builder_rules(
        V03, "data/structure_library_v0.3", progress=lambda row: event(**row)
    )
    reference_indices = np.load(root / "data/references.npy")
    occupancy = np.load(root / "data/occupancy.npy")
    outcomes = {}
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    subset = (
        set(np.load(root / "audit/cluster_unseen_test_groups_private.npy"))
        if args.cohort == "test"
        else set()
    )
    for mode in ("raw", "elements_grammar"):
        aggregate = StreamingAggregate()
        cluster_metrics = StreamingAggregate()
        buffer, writer = [], None
        last_heartbeat = time.monotonic()

        def heartbeat(token):
            nonlocal last_heartbeat
            if time.monotonic() - last_heartbeat > 15:
                event(stage="decoding", mode=mode, token_step=token)
                last_heartbeat = time.monotonic()

        try:
            with gzip.open(
                out / f"{mode}_raw_candidates_private.jsonl.gz",
                "wt",
                encoding="utf-8",
                compresslevel=6,
            ) as stream:
                for index, g in enumerate(groups):
                    tick = time.monotonic()
                    batch = data.batch([g])
                    with torch.inference_mode(), attention_backend("math"):
                        beams = generate(
                            model,
                            inference_inputs(batch),
                            data.tokenizer,
                            mode=mode,
                            beam_size=20,
                            progress=heartbeat,
                        )
                    templates = [
                        bank[int(i)]["polar_smiles"]
                        for i in reference_indices[g, occupancy[g]]
                        if i >= 0
                    ]
                    raw, accepted = score_beams(
                        beams,
                        data.tokenizer,
                        chain_tokens=[vocab[int(t)] for t in data.chains[int(g)].cpu().tolist()],
                        chains=chains,
                        known_heads=known_heads,
                        full_formula=data.full[int(g)].cpu().numpy(),
                        polar_formula=data.polar[int(g)].cpu().numpy(),
                    )
                    record = {
                        "target_group": int(g),
                        "mode": mode,
                        "raw_candidates": raw,
                        "accepted_unique": accepted,
                        "raw_metrics": target_metrics(
                            raw,
                            heads[int(g)],
                            templates,
                            train_keys,
                            chain_count=2 - int(occupancy[g]),
                        ),
                        "filtered_metrics": target_metrics(
                            accepted,
                            heads[int(g)],
                            templates,
                            train_keys,
                            chain_count=2 - int(occupancy[g]),
                        ),
                    }
                    aggregate.add(record)
                    if int(g) in subset:
                        cluster_metrics.add(record)
                    stream.write(
                        json.dumps(
                            {
                                "target_group": int(g),
                                "mode": mode,
                                "decode_seconds": time.monotonic() - tick,
                                "raw_candidates": raw,
                                "accepted_unique_source_ranks": [
                                    c["source_rank"] for c in accepted
                                ],
                            }
                        )
                        + "\n"
                    )
                    slim = {
                        "target_group": int(g),
                        "accepted_count": len(accepted),
                        "template_best_Tanimoto": record["raw_metrics"][
                            "template_best_Polar_Tanimoto"
                        ],
                    }
                    for scope in ("raw", "filtered"):
                        for k in (1, 5, 10):
                            for key, value in record[scope + "_metrics"][str(k)].items():
                                slim[f"{scope}_{key}_{k}"] = float(value)
                    buffer.append(slim)
                    if len(buffer) >= 256 or index + 1 == len(groups):
                        table = pa.Table.from_pylist(buffer)
                        if writer is None:
                            writer = pq.ParquetWriter(
                                out / f"{mode}_per_target_private.parquet",
                                table.schema,
                                compression="zstd",
                            )
                        writer.write_table(table)
                        buffer.clear()
                        if owned_bytes(root) > 12 * 2**30:
                            raise ValueError("New V2 storage budget exceeded; preserve and stop")
                    if index % 10 == 0 or index + 1 == len(groups):
                        event(
                            stage="evaluation_progress",
                            mode=mode,
                            targets=index + 1,
                            total=len(groups),
                            no_candidate=aggregate.no_candidate,
                            elapsed_seconds=time.monotonic() - began,
                        )
        finally:
            if writer:
                writer.close()
        outcomes[mode] = aggregate.result()
        if cluster_metrics.n:
            outcomes[mode]["cluster_unseen_test_diagnostic"] = cluster_metrics.result()
    report = {
        "status": "completed",
        "cohort": args.cohort,
        "targets": len(groups),
        "checkpoint_sha256": digest(checkpoint),
        "checkpoint": str(checkpoint),
        "contract": binding,
        "beam_size": 20,
        "modes": outcomes,
        "elapsed_seconds": time.monotonic() - began,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
        "target_prefix": False,
        "ranking": "decoder mean log probability; truth-free acceptance/dedup",
        "fingerprint": (
            "Morgan radius2/2048, no chirality; primary full core=chainless; "
            "Polar reported separately; every gain compares candidate and "
            "templates in the SAME representation"
        ),
        "head_definition": (
            "delete mapped anchor P, dummy and terminal O/S neighbours; retain "
            "all other components/P"
        ),
        "topk_recall_definition": (
            "best single candidate in decoder prefix; all target failures assigned0"
        ),
        "novelty_definition": "phosphate comparison key absent from TRAIN cores",
        "stereochemistry": "unchanged nonisomeric frozen labels; not stereoisomer prediction",
        "test_used_for_optimizer_selection": False,
    }
    write(out / "evaluation_report.json", report)
    event(
        stage="evaluation_completed", targets=len(groups), elapsed_seconds=report["elapsed_seconds"]
    )


if __name__ == "__main__":
    main()
