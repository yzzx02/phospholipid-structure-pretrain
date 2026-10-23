"""Render the compact V2 audit figures without repeating chemical retrieval."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from structure_pretrain.formal_v2.common import ROOT, digest, read, write


def save(fig, name):
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.24, top=0.80, wspace=0.30)
    for text in fig.texts:
        if text.get_position()[1] < 0.1:
            text.set_position((0.5, 0.025))
    for ext in ("png", "svg", "pdf"):
        destination = ROOT / "figures" / (name + "." + ext)
        fig.savefig(destination, dpi=220, bbox_inches="tight")
        if ext == "svg":
            destination.write_text(
                "\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def main():
    audit = ROOT / "audit"
    known = read(audit / "B1_known_references.json")
    table = pq.read_table(audit / "coverage_per_core_private.parquet").to_pydict()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "template-guided-v2",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for mode, color in zip(known["statistics"], ("#176b90", "#bc5d3b", "#379a70")):
        values = np.load(audit / f"known_{mode}_scores.npy")
        axes[0].hist(
            values,
            bins=np.linspace(0, 1, 21),
            histtype="step",
            label=mode,
            color=color,
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
    fig.suptitle(
        "Audited known references vs structural pretraining space "
        f"(N={known['unique_phosphate_equivalent_groups']})",
        y=0.96,
    )
    fig.text(
        0.5,
        0.02,
        (
            "Morgan radius2/2048; TRAIN bank171,478; same/phosphate-equivalent "
            "excluded; exhaustive exact + RDKit."
        ),
        ha="center",
        fontsize=8,
    )
    save(fig, "B1_known_reference_similarity")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    splits = np.asarray(table["split"])
    for ax, key, name in zip(
        axes,
        ("polar_top1_score", "chainless_actual_top1_score"),
        ("Polar Morgan", "chainless Morgan (actual occupancy)"),
    ):
        scores = np.asarray(table[key])
        for label, code, color in (
            ("Train", 0, "#176b90"),
            ("Validation", 1, "#bc5d3b"),
            ("Test", 2, "#379a70"),
        ):
            values = np.sort(scores[splits == code])
            take = np.linspace(0, len(values) - 1, min(3000, len(values))).astype(int)
            ax.plot(
                values[take],
                (take + 1) / len(values),
                label=f"{label} N={len(values):,}",
                color=color,
            )
        ax.set(
            title=name,
            xlabel="Nearest nonself template Tanimoto",
            ylabel="Cumulative fraction",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(fontsize=8)
    fig.suptitle("Fixed 5,000-template coverage; K3 retrieval, threshold0.2", y=0.96)
    fig.text(
        0.5,
        0.02,
        (
            "Self/equivalent excluded; exhaustive Tanimoto + RDKit,512/512 "
            "oracle matches per representation; frozen splits."
        ),
        ha="center",
        fontsize=8,
    )
    save(fig, "B2_template_coverage")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, key, label in zip(
        axes, ("chainless_0_top1_score", "chainless_1_top1_score"), ("double", "single")
    ):
        counts, _, _ = np.histogram2d(
            table["polar_top1_score"], table[key], bins=35, range=((0, 1), (0, 1))
        )
        im = ax.imshow(
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
        fig.colorbar(im, ax=ax, label="log10(count+1)")
    fig.suptitle("Representation boundary audit: Polar vs fixed-occupancy chainless", y=0.96)
    fig.text(
        0.5,
        0.02,
        (
            "Morgan radius2/2048; fixed5,000 bank; self/equivalent excluded; "
            "exact retrieval; scaffold bits are shared context."
        ),
        ha="center",
        fontsize=8,
    )
    save(fig, "A2_dual_fingerprint")
    proof = read(audit / "B_passed.json")
    proof["figure_source_sha256"] = digest(__file__)
    proof["assets_sha256"] = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (ROOT / "figures").glob("*")
        if p.suffix in ("png", "svg", "pdf") and "QA" not in p.stem
    }
    proof["visual_QA"] = "pending final render"
    write(audit / "B_passed.json", proof)


if __name__ == "__main__":
    main()
