"""CPU-only V2 known-reference coverage, group diagnostics and cluster holdout."""

import gzip
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from usearch.index import Index

from structure_pretrain.formal_v2.common import (
    ROOT,
    configuration,
    digest,
    emit,
    verify_data,
    write,
)
from structure_pretrain.formal_v2.retrieval import exact_retrieve
from structure_pretrain.polar_only_v04.chemistry import chainless_molecule
from structure_pretrain.reported_headgroups.formal_template_data import FEATURES, V03
from structure_pretrain.reported_headgroups.template_selection import (
    FUNCTION_NAMES,
    MORGAN,
    phosphate_comparison_key,
    read_csv,
)


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"N": 0}
    return {
        "N": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "at_least_04": float(np.mean(values >= 0.4)),
        "at_least_06": float(np.mean(values >= 0.6)),
        "histogram_01_bins": np.histogram(values, np.linspace(0, 1, 11))[0].tolist(),
    }


def packed(mol):
    return np.frombuffer(DataStructs.BitVectToBinaryText(MORGAN.GetFingerprint(mol)), np.uint8)


def figure_save(fig, folder, name):
    for ext in ("png", "svg", "pdf"):
        path = folder / f"{name}.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        if ext == "svg":
            path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    began = time.monotonic()
    c = configuration()
    root = ROOT.resolve()
    verify_data(root)
    data, audit = root / "data", root / "audit"
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    if (audit / "B_passed.json").exists():
        raise FileExistsError("Coverage already complete; preserve report")
    split = np.load(data / "splits.npy")
    eligible, train, test = (
        np.flatnonzero(split != 255),
        np.flatnonzero(split == 0),
        np.flatnonzero(split == 2),
    )
    polar = np.load(data / "polar_fingerprints.npy", mmap_mode="r")
    keys = np.load(data / "comparison_keys.npy")
    fp = np.load(FEATURES / "chainless_fingerprints.npy", mmap_mode="r")
    lookup = np.empty((len(split), 2), dtype=np.int32)
    lookup[
        np.load(FEATURES / "variant_heads.npy"), 2 - np.load(FEATURES / "variant_occupancy.npy")
    ] = np.arange(len(fp))
    known = [
        r
        for r in read_csv(
            "outputs/headgroup_template_proposal_20261022/approved_reference_catalog.csv"
        )
        if r["builder_compatible"] == "True" and r["training_reference_eligible"] == "True"
    ]
    unique = {}
    for row in known:
        unique.setdefault(phosphate_comparison_key(row["polar_smiles"]), []).append(row)
    query_keys = np.asarray(list(unique))
    queries = [rs[0] for rs in unique.values()]
    known_stats, private_rows = {}, []
    for mode in ("polar", "chainless_double", "chainless_single"):
        occ = 0 if mode.endswith("double") else 1
        bank = polar[train] if mode == "polar" else fp[lookup[train, occ]]
        q = np.asarray(
            [
                packed(
                    Chem.MolFromSmiles(r["polar_smiles"])
                    if mode == "polar"
                    else chainless_molecule(r["polar_smiles"], 2 - occ)
                )
                for r in queries
            ]
        )
        idx, scores, proof = exact_retrieve(
            q, bank, query_keys, keys[train], train.astype(str), k=1, audit_count=len(q), event=emit
        )
        assert np.all(idx >= 0)
        np.save(audit / f"known_{mode}_scores.npy", scores[:, 0])
        known_stats[mode] = {**distribution(scores[:, 0]), "exact_audit": proof}
        for i, row in enumerate(queries):
            private_rows.append(
                {
                    "mode": mode,
                    "headgroup_id": row["headgroup_id"],
                    "names": [r["abbreviation"] or r["full_name"] for r in unique[query_keys[i]]],
                    "polar_smiles": row["polar_smiles"],
                    "source_url": row["source_url"],
                    "nearest_train_group": int(train[idx[i, 0]]),
                    "max_nonself_Tanimoto": float(scores[i, 0]),
                    "low_similarity_lt04": bool(scores[i, 0] < 0.4),
                }
            )
    with gzip.open(
        audit / "known_reference_neighbours_private.jsonl.gz", "wt", encoding="utf-8"
    ) as stream:
        for row in private_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write(
        audit / "B1_known_references.json",
        {
            "approved_compatible_records": len(known),
            "unique_phosphate_equivalent_groups": len(queries),
            "statistics": known_stats,
            "query_scope": "all approved feature-compatible known references; NOT MS/MS finetuning",
            "reference_scope": "171478 TRAIN cores; same/equivalent excluded",
        },
    )
    emit(stage="B1_completed", known_records=len(known), unique=len(queries))

    table = pq.read_table(audit / "dual_fingerprint_per_core_private.parquet").to_pydict()
    pscore = np.asarray(table["polar_top1_score"])
    actual_occ = np.load(data / "occupancy.npy")[eligible]
    double = np.asarray(table["chainless_0_top1_score"])
    single = np.asarray(table["chainless_1_top1_score"])
    actual = np.where(actual_occ == 0, double, single)
    heads = (
        pq.read_table(V03 / "polar_groups.parquet", columns=["chemistry_family"])
        .column(0)
        .to_numpy()
    )
    masks = np.load(data / "functional_masks.npy")[eligible]
    formulas = np.load(data / "polar_formulas.npy")[eligible]
    strata = {"all": np.ones(len(eligible), dtype=bool)}
    strata.update({"family:" + str(f): heads[eligible] == f for f in np.unique(heads[eligible])})
    strata.update(
        {"functional:" + name: (masks & (1 << i)) != 0 for i, name in enumerate(FUNCTION_NAMES)}
    )
    strata.update({"chain:" + name: actual_occ == o for o, name in enumerate(("double", "single"))})
    strata.update(
        {
            f"C:{lo}-{hi - 1}": (formulas[:, 0] >= lo) & (formulas[:, 0] < hi)
            for lo, hi in ((0, 6), (6, 11), (11, 16), (16, 26), (26, 1000))
        }
    )
    coverage = {}
    for name, code in (("train", 0), ("validation", 1), ("test", 2)):
        coverage[name] = {
            label: {
                "Polar": distribution(pscore[mask & (split[eligible] == code)]),
                "chainless_actual_occupancy": distribution(
                    actual[mask & (split[eligible] == code)]
                ),
            }
            for label, mask in strata.items()
        }
    table["chainless_actual_top1_score"] = actual
    table["chemistry_family"] = heads[eligible]
    table["functional_mask"] = masks
    pq.write_table(
        pa.Table.from_pydict(table), audit / "coverage_per_core_private.parquet", compression="zstd"
    )
    write(audit / "B2_coverage.json", coverage)

    # Sparse connected components are diagnostic approximations. Every qualifying
    # test core is ALSO checked exhaustively against ALL train fingerprints.
    threshold = c["audit"]["cluster_diagnostic_threshold"]
    cluster_path = audit / "cluster_ids_private.npy"
    if not cluster_path.exists():
        index = Index(
            ndim=2048,
            metric="tanimoto",
            dtype="b1",
            connectivity=32,
            expansion_add=128,
            expansion_search=128,
        )
        index.add(np.arange(len(eligible)), np.asarray(polar[eligible]), threads=1)
        parent = np.arange(len(eligible), dtype=np.int32)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for start in range(0, len(eligible), 2048):
            results = index.search(
                np.asarray(polar[eligible[start : start + 2048]]), count=32, threads=4
            )
            for i, (indices, distances) in enumerate(zip(results.keys, results.distances), start):
                for j in indices[np.asarray(distances) <= 1 - threshold + 1e-6]:
                    j = int(j)
                    if j == i:
                        continue
                    left = DataStructs.CreateFromBinaryText(bytes(polar[eligible[i]]))
                    right = DataStructs.CreateFromBinaryText(bytes(polar[eligible[j]]))
                    if DataStructs.TanimotoSimilarity(left, right) >= threshold:
                        a, b = find(i), find(j)
                        parent[max(a, b)] = min(a, b)
            if start % 16384 == 0:
                emit(
                    stage="B_cluster_sparse",
                    queries=min(start + 2048, len(eligible)),
                    total=len(eligible),
                )
        clusters = np.asarray([find(i) for i in range(len(eligible))], dtype=np.int32)
        np.save(cluster_path, clusters)
    clusters = np.load(cluster_path)
    nearest_path = audit / "test_to_train_exact_scores.npy"
    if not nearest_path.exists():
        idx, scores, proof = exact_retrieve(
            polar[test],
            polar[train],
            keys[test],
            keys[train],
            train.astype(str),
            k=1,
            batch_size=512,
            audit_count=512,
            event=emit,
        )
        np.save(nearest_path, scores[:, 0])
        np.save(audit / "test_to_train_exact_indices.npy", train[idx[:, 0]])
        write(audit / "test_to_train_exact_audit.json", proof)
    max_train = np.load(nearest_path)
    train_components = set(clusters[split[eligible] == 0])
    # Purge every entire diagnostic component containing a near-training test core.
    test_clusters = clusters[split[eligible] == 2]
    contaminated = set(test_clusters[max_train >= threshold]) | train_components
    unseen = np.asarray([x not in contaminated for x in test_clusters])
    np.save(audit / "cluster_unseen_test_groups_private.npy", test[unseen])
    write(
        audit / "cluster_diagnostic.json",
        {
            "status": "passed",
            "fingerprint": "Polar Morgan radius2/2048",
            "threshold": threshold,
            "components": len(set(clusters)),
            "sparse_graph": (
                "HNSW top32 edges, RDKit threshold verified; approximate components, "
                "no claim of exhaustive clustering"
            ),
            "exact_guard": (
                "every test core exhaustively queried against all171478 TRAIN; whole "
                "component purged if any test member has score>=0.8"
            ),
            "unseen_test_targets": int(unseen.sum()),
            "unseen_test_components": len(set(test_clusters[unseen])),
            "test_targets_near_train_ge08": int((max_train >= threshold).sum()),
            "test_max_train_distribution": distribution(max_train),
            "frozen_test_size": len(test),
            "original_split_changed": False,
            "subset_used_for_LR_selection": False,
        },
    )
    emit(stage="B_cluster_completed", unseen_targets=int(unseen.sum()))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "template-guided-v2",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode, color in zip(known_stats, ("#176b90", "#bc5d3b", "#379a70")):
        values = np.load(audit / f"known_{mode}_scores.npy")
        axes[0].hist(
            values,
            bins=np.linspace(0, 1, 21),
            histtype="step",
            color=color,
            label=mode,
            linewidth=1.8,
        )
        axes[1].plot(
            np.sort(values), np.arange(1, len(values) + 1) / len(values), label=mode, color=color
        )
    axes[0].set_ylabel("Known reference count")
    axes[1].set_ylabel("Cumulative fraction")
    for ax in axes:
        ax.set_xlabel("Maximum nonself Tanimoto to TRAIN cores")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle(f"Audited known references vs structural pretraining space (N={len(queries)})")
    fig.text(
        0.5,
        -0.01,
        (
            "Morgan radius2/2048; TRAIN bank171,478; same/phosphate-equivalent "
            "excluded; exhaustive exact + RDKit."
        ),
        ha="center",
        fontsize=8,
    )
    figure_save(fig, figures, "B1_known_reference_similarity")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, scores, name in zip(
        axes, (pscore, actual), ("Polar Morgan", "chainless Morgan (actual occupancy)")
    ):
        for label, code, color in (
            ("Train", 0, "#176b90"),
            ("Validation", 1, "#bc5d3b"),
            ("Test", 2, "#379a70"),
        ):
            values = np.sort(scores[split[eligible] == code])
            # Plot a deterministic thin curve; statistics always use every core.
            take = np.linspace(0, len(values) - 1, min(3000, len(values))).astype(int)
            ax.plot(
                values[take],
                (take + 1) / len(values),
                label=f"{label} N={len(values):,}",
                color=color,
            )
        ax.set_title(name)
        ax.set(
            xlabel="Nearest nonself template Tanimoto",
            ylabel="Cumulative fraction",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(fontsize=8)
    fig.suptitle("Fixed 5,000-template coverage; K3 retrieval, similarity threshold0.2")
    fig.text(
        0.5,
        -0.01,
        (
            "Same/equivalent excluded; exhaustive Tanimoto + RDKit,512/512 "
            "oracle matches per representation; original frozen splits."
        ),
        ha="center",
        fontsize=8,
    )
    figure_save(fig, figures, "B2_template_coverage")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for values, label in ((double, "double"), (single, "single")):
        counts, xe, ye = np.histogram2d(pscore, values, bins=35, range=((0, 1), (0, 1)))
        ax = axes[0 if label == "double" else 1]
        image = ax.imshow(
            np.log10(counts.T + 1),
            origin="lower",
            extent=(0, 1, 0, 1),
            aspect="equal",
            cmap="Blues",
        )
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.7)
        ax.set(
            title=f"{label}; same190,553 queries",
            xlabel="Polar nearest Tanimoto",
            ylabel="chainless nearest Tanimoto",
        )
        fig.colorbar(image, ax=ax, label="log10(count+1)")
    fig.suptitle("Representation boundary audit: Polar vs fixed-occupancy chainless")
    fig.text(
        0.5,
        -0.01,
        (
            "Morgan radius2/2048; fixed5,000 bank; self/equivalent excluded; "
            "exact retrieval; scaffold bits are shared context."
        ),
        ha="center",
        fontsize=8,
    )
    figure_save(fig, figures, "A2_dual_fingerprint")
    files = [
        *figures.glob("*"),
        audit / "B1_known_references.json",
        audit / "B2_coverage.json",
        audit / "cluster_diagnostic.json",
    ]
    write(
        audit / "B_passed.json",
        {
            "status": "passed",
            "elapsed_seconds": time.monotonic() - began,
            "source_sha256": digest(__file__),
            "assets_sha256": {str(p.relative_to(root)): digest(p) for p in files},
            "visual_QA": "pending PDF rendering before formal gate",
        },
    )
    emit(stage="B_completed", elapsed_seconds=time.monotonic() - began)


if __name__ == "__main__":
    main()
