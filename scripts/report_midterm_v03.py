"""Publish aggregate checkpoint comparisons; never copy private sample records."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def collect(root):
    quality = {}
    for path in sorted((root / "evaluation").glob("*/quality_report.json")):
        value = read(path)
        monitor = read(path.parent / "host_monitor.json")
        assert monitor["child_exit_code"] == 0 and not monitor["stop_reason"]
        assert not monitor["new_windows_gpu_dumps"]
        key = (value["encoder"], value["step"])
        assert key not in quality, f"Ambiguous duplicate evaluation: {key}"
        value["host_monitor"] = {
            k: monitor[k] for k in ("child_exit_code", "stop_reason", "elapsed_seconds")
        }
        value["host_monitor"]["new_windows_gpu_dump_count"] = 0
        events = [
            json.loads(line) for line in (path.parent / "events.jsonl").read_text().splitlines()
        ]
        reused = any(event.get("stage") == "reused_completed_raw" for event in events)
        value["raw_generation_reused_from_completed_attempt"] = reused
        value["reported_seconds_include_raw_generation"] = not reused
        comparison = path.parent / "previous_smoke_comparison.json"
        if comparison.exists():
            value["previous_128_smoke_agreement"] = read(comparison)
        quality[key] = value
    phases = []
    for path in sorted((root / "training").glob("*/training_metrics.json")):
        value = read(path)
        monitor = read(path.parent / "host_monitor.json")
        assert monitor["child_exit_code"] == 0 and not monitor["stop_reason"]
        assert not monitor["new_windows_gpu_dumps"]
        assert all(value["gradient_checks"].values())
        assert all(value["parameters_changed"].values())
        assert value["checkpoint_state_dict_identical"]
        phases.append(
            {
                "encoder": value["contract"]["model_config"]["encoder"],
                **{
                    k: value[k]
                    for k in (
                        "optimizer_steps_total",
                        "optimizer_steps_this_run",
                        "resumed_from_step",
                        "train_targets",
                        "train_examples_consumed",
                        "parameters",
                        "gradient_checks",
                        "parameters_changed",
                        "checkpoint_state_dict_identical",
                        "checkpoint_reload_logits_close",
                    )
                },
                "resources": read(path.parent / "resource_metrics.json"),
                "new_windows_gpu_dump_count": 0,
            }
        )
    return quality, phases


def paired_changes(root, encoder, previous_step, current_step, cohort_rows):
    records = {}
    for path in (root / "evaluation").glob("*/quality_report.json"):
        report = read(path)
        if report["encoder"] == encoder and report["step"] in (previous_step, current_step):
            records[report["step"]] = {
                row["row"]: row
                for row in (
                    json.loads(line)
                    for line in (path.parent / "raw_predictions_private.jsonl")
                    .read_text()
                    .splitlines()
                )
                if row["row"] in cohort_rows
            }
    before, after = records[previous_step], records[current_step]
    assert set(before) == set(after) == cohort_rows
    result = {}
    for metric, getter in {
        "eos": lambda r: r["raw"]["eos"],
        "valid_polar": lambda r: r["raw"]["polar_valid"],
        "polar_top5": lambda r: r["top5_polar_recalled"],
        "chain_set_exact": lambda r: r["chain_set_exact"],
        "formula_match": lambda r: r["raw"]["formula_match"],
        "full_exact": lambda r: r["raw"]["full_exact"],
    }.items():
        gains = sum(not getter(before[k]) and getter(after[k]) for k in before)
        losses = sum(getter(before[k]) and not getter(after[k]) for k in before)
        result[metric] = {
            "gained_targets": gains,
            "lost_targets": losses,
            "net_change": gains - losses,
            "denominator": len(before),
        }
    return result


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/midterm_v03"))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    quality, phases = collect(args.root)
    for (encoder, step), report in sorted(quality.items()):
        teacher = report["cohorts"]["validation_pool"]["teacher_forced"]
        generation = report["cohorts"]["diagnostic_512"]["raw_generation"]
        print(
            json.dumps(
                {
                    "encoder": encoder,
                    "step": step,
                    "val_loss": teacher["loss"],
                    **{
                        k: generation[k]
                        for k in (
                            "eos_rate",
                            "valid_polar_rate",
                            "top5_polar_recall",
                            "chain_set_exact_rate",
                            "full_formula_match_rate",
                            "full_structure_exact_rate",
                            "target_morgan_similarity",
                        )
                    },
                }
            )
        )
    if not args.publish:
        return
    decision = read(args.root / "decisions.json")
    pairs = {}
    for encoder in ("fingerprint", "graph"):
        steps = sorted(step for kind, step in quality if kind == encoder)
        for current in steps[1:]:
            for cohort in ("smoke_128", "diagnostic_512"):
                rows = set(np.load(args.root / f"{cohort}.npy").tolist())
                pairs[f"{encoder}/1001_to_{current}/{cohort}"] = paired_changes(
                    args.root, encoder, 1001, current, rows
                )
    result = {
        "schema": "midterm-comparison-v1",
        "decision": decision,
        "evaluations": list(quality.values()),
        "training_phases": phases,
        "paired_changes_from_1001": pairs,
        "test_targets_used": 0,
        "full_training_started": False,
        "private_samples_or_checkpoints_published": False,
        "checkpoint_integrity_audit": read(args.root / "checkpoint_integrity_audit.json"),
    }
    destination = Path("docs/results/MIDTERM_STRUCTURE_20261019.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# 结构预训练中程实验结果",
        "",
        decision["recommendation"],
        "",
        "所有生成指标均为未经修复、未经公式筛选的原始预测；完整指标、失败阶段、长度分布、"
        "后验筛选覆盖率和配对变化见 [JSON](../results/MIDTERM_STRUCTURE_20261019.json)。",
        "",
        "## 原 2,000 条 validation + 512 条分层生成诊断",
        "",
        "|编码器|Step|Val loss|Token acc|EOS|Polar valid|Top-5|Chain set|Formula|Full|Morgan|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (encoder, step), r in sorted(quality.items()):
        t = r["cohorts"]["validation_pool"]["teacher_forced"]
        g = r["cohorts"]["diagnostic_512"]["raw_generation"]
        cells = [encoder, str(step), f"{t['loss']:.4f}", f"{t['polar_token_accuracy']:.2%}"]
        cells += [
            f"{g[k]:.2%}"
            for k in (
                "eos_rate",
                "valid_polar_rate",
                "top5_polar_recall",
                "chain_set_exact_rate",
                "full_formula_match_rate",
                "full_structure_exact_rate",
            )
        ]
        cells.append(f"{g['target_morgan_similarity']:.4f}")
        lines.append("|" + "|".join(cells) + "|")
    lines += [
        "",
        "## 原 128 条 smoke 对照",
        "",
        "|编码器|Step|EOS|Polar valid|Top-5|Chain set|Formula|Full exact|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (encoder, step), r in sorted(quality.items()):
        g = r["cohorts"]["smoke_128"]["raw_generation"]
        cells = [encoder, str(step)] + [
            f"{g[k]:.2%}"
            for k in (
                "eos_rate",
                "valid_polar_rate",
                "top5_polar_recall",
                "chain_set_exact_rate",
                "full_formula_match_rate",
                "full_structure_exact_rate",
            )
        ]
        lines.append("|" + "|".join(cells) + "|")
    lines += [
        "",
        "## 可选 Top-5 分子式筛选（512 条；不修复）",
        "",
        "|编码器|Step|覆盖条数|覆盖率|Full exact 条数|总体 Full exact|筛选内 Full exact|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (encoder, step), report in sorted(quality.items()):
        filtered = report["cohorts"]["diagnostic_512"]["raw_generation"]["optional_formula_filter"]
        conditional = filtered["full_exact_among_selected"]
        conditional_text = "无候选" if conditional is None else f"{conditional:.2%}"
        lines.append(
            f"|{encoder}|{step}|{filtered['selected_targets']}|"
            f"{filtered['coverage']:.2%}|{filtered['full_exact_count']}|"
            f"{filtered['overall_full_exact']:.2%}|{conditional_text}|"
        )
    lines += [
        "",
        "## 续训资源和恢复",
        "",
        "|编码器|区间|参数量|Steps/s|Samples/s|峰值分配显存 GB|状态重载|",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for phase in phases:
        r = phase["resources"]
        lines.append(
            f"|{phase['encoder']}|{phase['resumed_from_step']}→"
            f"{phase['optimizer_steps_total']}|{phase['parameters']:,}|"
            f"{r['steps_per_second']:.2f}|{r['samples_per_second']:.1f}|"
            f"{r['peak_VRAM_allocated_bytes'] / 1e9:.3f}|一致|"
        )
    lines += ["", "## 决策依据和限制", ""]
    lines += ["- " + item for item in decision["reasons"]]
    lines += [
        "",
        "- 512 条为分层诊断样本，包含约三分之一单链样本，不能将其比例外推到整个库。",
        "- 冻结 validation 无 P/A 样本；不移动 split 来制造覆盖。",
        "- 20,000 条训练池不等于完整 201.5 万条预训练；尚无 MS/MS 输入或微调验证。",
        "- Math attention 下本轮未新增 GPU 转储，仅支持这些已执行负载的稳定性；"
        "不证明驱动问题彻底解决或硬件必然正常。",
        "- 首次 FP1001 生成已完成，但报告收尾发生 Path 类型错误；"
        "修正后以哈希绑定复用全部 640 条原始输出，只重算 teacher forcing。",
        "",
    ]
    Path("docs/experiments/MIDTERM_STRUCTURE_RESULTS_20261019.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
