"""Publication-style aggregate figures; run in the separate G-drive plotting env."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "cache/matplotlib-similarity")
)
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("outputs/polar_similarity_eval_v04_20261020")
DEST = Path("docs/results/POLAR_PRETRAIN_SIMILARITY_EVAL_V04_FIGURES")
MODELS = ("20k_3000", "50k_3000", "50k_5000")
LABELS = ("20k / 3,000", "50k / 3,000", "50k / 5,000")
COLORS = ("#67788b", "#d18a28", "#167d74")


def save(fig, name, hashes):
    for extension in ("png", "svg", "pdf"):
        path = DEST / f"{name}.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
        if extension == "svg":
            text = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(text + "\n", encoding="utf-8", newline="\n")
        hashes[path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    private = json.loads((args.root / "target_metrics_private.json").read_text())
    aggregate = json.loads((args.root / "aggregate.json").read_text())
    assert aggregate["complete"] and len(private) == 6
    DEST.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    hashes, plot_data = {}, {}
    grid = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(
        2, 4, figsize=(14, 6.8), sharex=True, sharey=True, layout="constrained"
    )
    for row, beam in enumerate((20, 50)):
        for col, n in enumerate((1, 3, 5, 10)):
            ax = axes[row, col]
            for model, label, color in zip(MODELS, LABELS, COLORS):
                values = np.asarray([r["s"][str(n)] for r in private[f"{model}_b{beam}"]])
                counts = (values[:, None] >= grid[None] - 1e-12).sum(0)
                ax.step(grid, counts / 512 * 100, where="post", label=label, color=color, lw=2)
                plot_data[f"A_b{beam}_{model}_Top{n}"] = {
                    "thresholds": grid.tolist(),
                    "counts_at_or_above": counts.tolist(),
                    "denominator": 512,
                }
            ax.set_title(f"Beam {beam} | Top-{n}")
            ax.set(xlim=(0, 1), ylim=(0, 101), xticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.grid(alpha=0.18)
            if col == 0:
                ax.set_ylabel("Targets at or above threshold (%)")
            if row == 1:
                ax.set_xlabel("Polar Tanimoto threshold")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "A  Model-ranked Top-N maximum Polar similarity | all 512 diagnostic targets", fontsize=14
    )
    save(fig, "Figure_A_TopN_Cumulative", hashes)
    for beam in (20, 50):
        fig, axes = plt.subplots(
            3, 3, figsize=(10.6, 10.3), sharex=True, sharey=True, layout="constrained"
        )
        for row, (model, label) in enumerate(zip(MODELS, LABELS)):
            records = private[f"{model}_b{beam}"]
            x = np.asarray([r["template_best"] for r in records])
            for col, n in enumerate((1, 5, 10)):
                ax = axes[row, col]
                y = np.asarray([r["s"][str(n)] for r in records])
                h = ax.hexbin(
                    x,
                    y,
                    gridsize=24,
                    extent=(0, 1, 0, 1),
                    mincnt=1,
                    cmap="Blues",
                    bins="log",
                    vmin=1,
                    vmax=100,
                )
                ax.plot([0, 1], [0, 1], "--", color="#666666", lw=1)
                ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
                improved = int((y > x + 1e-12).sum())
                ax.text(
                    0.03,
                    0.96,
                    f"Gain > 0: {improved}/512",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )
                ax.set_title(f"{label} | Top-{n}")
                if col == 0:
                    ax.set_ylabel("Generated similarity to target")
                if row == 2:
                    ax.set_xlabel("Best input-template similarity to target")
        fig.colorbar(
            h, ax=axes.ravel().tolist(), shrink=0.65, label="Targets per hexagon (log scale)"
        )
        fig.suptitle(
            f"B  Paired template vs generated similarity | Beam {beam}, all 512 targets",
            fontsize=14,
        )
        save(fig, f"Figure_B_Template_Generated_Beam{beam}", hashes)
    fig, axes = plt.subplots(
        2, 3, figsize=(11.6, 7.5), sharex=True, sharey=True, layout="constrained"
    )
    for row, beam in enumerate((20, 50)):
        records = private[f"50k_5000_b{beam}"]
        x = np.asarray([r["template_best"] for r in records])
        for col, n in enumerate((1, 5, 10)):
            ax = axes[row, col]
            delta = np.asarray([r["s"][str(n)] for r in records]) - x
            h = ax.hexbin(
                x,
                delta,
                gridsize=25,
                extent=(0, 1, -1, 1),
                mincnt=1,
                cmap="YlGnBu",
                bins="log",
                vmin=1,
                vmax=100,
            )
            ax.axhline(0, color="#333333", lw=1)
            for cut in (0.4, 0.7):
                ax.axvline(cut, color="#777777", ls="--", lw=0.8)
            low = x < 0.4
            ax.text(
                0.03,
                0.96,
                f"Low-template gain > 0: {int((delta[low] > 1e-12).sum())}/{int(low.sum())}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
            )
            ax.set_title(f"Beam {beam} | Top-{n}")
            ax.set(xlim=(0, 1), ylim=(-1, 1))
            if col == 0:
                ax.set_ylabel("Generated minus best-template similarity")
            if row == 1:
                ax.set_xlabel("Best-template Tanimoto to target")
    fig.colorbar(h, ax=axes.ravel().tolist(), shrink=0.75, label="Targets per hexagon (log scale)")
    fig.suptitle(
        "C  Template proximity and generation gain | 50k / 5,000, all 512 targets", fontsize=14
    )
    save(fig, "Figure_C_Template_Gain", hashes)
    (args.root / "figure_A_aggregate_curve_data.json").write_text(json.dumps(plot_data) + "\n")
    (DEST / "FIGURE_SHA256.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"figures": len(hashes), "directory": str(DEST)}))


if __name__ == "__main__":
    main()
