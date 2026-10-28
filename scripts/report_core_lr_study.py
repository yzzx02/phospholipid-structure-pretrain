"""Public aggregate report and unsmoothed publication figures; no private rows exported."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structure_pretrain.core_lr_study.common import ROOT, configuration, digest, read, write

STEM = "CORE_LR_SCHEDULE_STUDY_20261028"
PUBLIC = Path("docs/results")
FIGURES = PUBLIC / "figures" / "core_lr_schedule_study_20261028"
LABELS = {
    "E1": "E1 Constant",
    "E2": "E2 Cosine",
    "E3": "E3 Warmup 4k + Cosine",
    "E4": "E4 Warmup 10k + Cosine",
}
COLORS = {"E1": "#0072B2", "E2": "#D55E00", "E3": "#009E73", "E4": "#CC79A7"}


def optional(path):
    return read(path) if path.exists() else None


def draw(histories):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )
    specs = [
        ("training_token_CE", "Training token cross-entropy", "training_loss", True),
        ("validation_token_CE", "Validation token cross-entropy", "validation_loss", True),
        ("LR", "Actual learning rate (last update of epoch)", "learning_rate", True),
        ("validation_token_CE", "Validation token cross-entropy", "validation_full_history", False),
    ]
    paths = []
    for key, ylabel, name, common in specs:
        fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
        for exp, history in histories.items():
            rows = [r for r in history if r["epoch"] <= 50 or not common]
            if rows:
                ax.plot(
                    [r["epoch"] for r in rows],
                    [r[key] for r in rows],
                    label=LABELS[exp],
                    color=COLORS[exp],
                    linewidth=1.6,
                )
                if key == "validation_token_CE":
                    best = min(rows, key=lambda r: r[key])
                    ax.scatter([best["epoch"]], [best[key]], color=COLORS[exp], s=24, zorder=3)
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.set_xlim(
            1, 50 if common else max(50, max((len(h) for h in histories.values()), default=0))
        )
        ax.set_xticks([1, 10, 20, 30, 40, 50] if common else [1, 10, 20, 30, 40, 50, 60, 70, 80])
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        if key == "LR":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if histories["E3"]:
            ax.axvline(48, color="#888888", linewidth=0.8, linestyle=":")
        ax.set_title(
            "Epochs 1–50 (measured values only)"
            if common
            else "Complete observed validation histories"
        )
        for suffix in ("png", "svg"):
            path = FIGURES / f"{name}.{suffix}"
            fig.savefig(path, dpi=300, metadata={"Date": None} if suffix == "svg" else None)
            if suffix == "svg":
                path.write_text(
                    "\n".join(x.rstrip() for x in path.read_text(encoding="utf-8").splitlines())
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
            paths.append(path)
        plt.close(fig)
    return paths


def main():
    c = configuration()
    histories, training, evaluations, weights = {}, {}, {}, {}
    for exp in LABELS:
        run = ROOT / f"{exp}_seed42"
        history = optional(run / "validation_history.json")
        if history is None:
            history = read(ROOT / "data/baseline_E3_history.json") if exp == "E3" else []
        assert [r["epoch"] for r in history] == list(range(1, len(history) + 1))
        assert all(r["validation_targets"] == 9493 for r in history)
        histories[exp] = history
        training[exp] = optional(run / "training_report.json")
        evaluations[exp] = optional(ROOT / f"{exp}_validation_beam32/evaluation_report.json")
        pointer = optional(run / "best.json")
        weights[exp] = (
            {k: pointer[k] for k in ("sha256", "epoch", "step", "best_validation_loss")}
            if pointer
            else None
        )
    paths = draw(histories)
    state = optional(ROOT / "state.json") or {"status": "prepared"}
    result = {
        "status": state["status"],
        "state": state,
        "configuration": c,
        "data": {"train": 170872, "validation": 9493, "test_untouched": 9499, "bank": 8000},
        "parameter_count": 16479940,
        "common_window_has50_real_values_each": all(len(h) >= 50 for h in histories.values()),
        "histories": histories,
        "training": training,
        "selected_weights": weights,
        "evaluations": evaluations,
        "original_E3": {
            "completed_epochs": 48,
            "best_epoch": 43,
            "best_validation_token_CE": 0.09594137142809196,
            "best_weight_sha256": c["original_E3_best_sha256"],
            "epoch48_weight_sha256": c["original_E3_epoch48_sha256"],
            "modified": False,
        },
        "initialization_evidence": read(ROOT / "data/initialization_audit.json"),
        "figures_sha256": {p.as_posix(): digest(p) for p in paths},
        "interpretation": "validation schedule comparison; target-structure-oracle "
        "templates; not real MS/MS evaluation",
        "test_used_for_tuning": False,
    }
    PUBLIC.mkdir(exist_ok=True, parents=True)
    write(PUBLIC / f"{STEM}.json", result)
    lines = [
        "# Whole glycerol-core learning-rate schedule comparison",
        "",
        f"Status: **{state['status']}**; stage: `{state.get('stage', 'prepared')}`.",
        "",
        "Shared TRAIN170,872 / validation9,493 / bank8,000; frozen test9,499 is unused. "
        "Same model (16,479,940 parameters), seed42, Batch32, AdamW, FP32/Math/TF32off.",
        "",
        "Minimum50 complete epochs for measured common plots; then patience5, maximum80. "
        "Patience accumulates throughout and is not reset at50. Best "
        "weight is the global minimum full-validation token CE, "
        "which may occur before or after50. All cosine cycles remain80 "
        "epochs (427,200 updates), minimum LR1e-5.",
        "",
        "E3 original48-epoch run and epoch43 best remain immutable. "
        "Independent continuation restores epoch48 model, "
        "AdamW moments, scheduler and RNG, without a new warmup. E3 "
        "figures join its original1–48 and measured49+; "
        "the dotted line marks48. Original best is retained unless "
        "continuation improves validation CE. "
        "E1/E2/E4 load one identical initial tensor/RNG snapshot; old E3 "
        "has recorded seed/source recipe evidence, "
        "but no saved initial tensor snapshot, so direct initial-tensor "
        "equality with historical E3 is not claimed.",
        "",
        "| Case | Schedule | Observed epochs | Minimum validation CE | Best epoch |",
        "|---|---|---:|---:|---:|",
    ]
    for exp, h in histories.items():
        best = min(h, key=lambda r: r["validation_token_CE"]) if h else None
        loss = f"{best['validation_token_CE']:.8f}" if best else "pending"
        lines.append(
            f"| {exp} | {LABELS[exp]} | {len(h)} | {loss} | "
            f"{best['epoch'] if best else 'pending'} |"
        )
    lines += [
        "",
        "![Training CE](figures/core_lr_schedule_study_20261028/training_loss.png)",
        "",
        "![Validation CE](figures/core_lr_schedule_study_20261028/validation_loss.png)",
        "",
        "![Actual LR](figures/core_lr_schedule_study_20261028/learning_rate.png)",
        "",
        "Curves contain measured, unsmoothed values only. During running, "
        "missing future epochs remain blank. "
        "Common plots show1–50; the separate full-history PNG/SVG retains later epochs and minima.",
        "",
        "## Unified validation generation",
        "",
        "All9,493 validation targets, Beam32, max96, same online "
        "elements+basic grammar; final RDKit/assembly/H-inclusive "
        "formula checks. No repair or probability renormalization. All "
        "targets and missing beam slots stay in denominators. "
        "Raw below already includes online constraints, before posterior "
        "formula filtering and graph deduplication. "
        "Template retrieval uses target structure; results are a "
        "structural pretraining comparison, not spectral accuracy.",
        "",
        "| Case | Validation n | Raw Exact@10 | Filtered Exact@10 | "
        "Filtered Similarity@10 | Formula target coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for exp, e in evaluations.items():
        if e:
            lines.append(
                f"| {exp} | {e['targets']} | {e['raw']['exact_at_10']:.6f} | "
                f"{e['posterior_filtered']['exact_at_10']:.6f} | "
                f"{e['posterior_filtered']['Similarity_at_10']:.6f} | "
                f"{e['budget']['targets_with_formula_correct_candidate']:.6f} |"
            )
        else:
            lines.append(f"| {exp} | pending | pending | pending | pending | pending |")
    complete = all(e is not None for e in evaluations.values())
    if complete:
        ce_case = min(training, key=lambda k: training[k]["best_validation_token_CE"])
        exact_case = max(
            evaluations, key=lambda k: evaluations[k]["posterior_filtered"]["exact_at_10"]
        )
        sim_case = max(
            evaluations, key=lambda k: evaluations[k]["posterior_filtered"]["Similarity_at_10"]
        )
        lines += [
            "",
            f"Lowest validation CE: {ce_case}; highest filtered validation Exact@10: {exact_case}; "
            f"highest filtered validation Similarity@10: {sim_case}. "
            "This is one paired seed42 comparison; differences do not "
            "establish across-seed superiority.",
        ]
    lines += [
        "",
        "P/nonP metrics, Exact@1/5, functional-group recovery and gain "
        "over best input template are in aggregate JSON. "
        "Private structures, raw predictions, weights and caches stay on "
        "G. Each new run retains best+latest2 checkpoints. "
        "Original runs, locks and checkpoints are preserved.",
        "",
    ]
    (PUBLIC / f"{STEM}.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(
        {
            "status": state["status"],
            "epochs": {k: len(v) for k, v in histories.items()},
            "generation_complete": complete,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
