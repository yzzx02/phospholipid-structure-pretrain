"""Publish only aggregate preparation/gate/start status, keeping all graphs private."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem

from structure_pretrain.reported_headgroups.formal_template_data import ROOT, digest, read, write
from structure_pretrain.reported_headgroups.template_selection import (
    FUNCTION_NAMES,
    functional_mask,
)


def main():
    folder = ROOT / "data"
    manifest = read(folder / "manifest.json")
    selected = read(folder / "nested_templates_20000_private.json")
    library = pq.read_table(
        "data/structure_library_v0.3/polar_library.parquet", columns=["polar_id"]
    )
    with np.load(
        "outputs/headgroup_template_proposal_20261022/selection_cache_private.npz"
    ) as cache:
        masks = cache["masks"]
    by_id = dict(zip(library.column(0).to_pylist(), masks.tolist()))
    chosen_masks = np.asarray(
        [
            by_id[r["template_id"]]
            if r["template_id"] in by_id
            else functional_mask(Chem.MolFromSmiles(r["polar_smiles"]))
            for r in selected
        ],
        dtype=np.uint32,
    )
    functions = {}
    present = {name: int(((masks & (1 << i)) != 0).sum()) for i, name in enumerate(FUNCTION_NAMES)}
    for size in manifest["nested_sizes"]:
        counts = {
            name: int(((chosen_masks[:size] & (1 << i)) != 0).sum())
            for i, name in enumerate(FUNCTION_NAMES)
        }
        mass = np.asarray([r["average_mass_Da"] for r in selected[:size]])
        functions[str(size)] = {
            "functional_counts": counts,
            "uncovered_classes_present_in_frozen_pool": [
                name for name, n in present.items() if n and not counts[name]
            ],
            "mass_Da_quantiles": {
                str(q): float(np.quantile(mass, q)) for q in (0, 0.1, 0.5, 0.9, 1)
            },
            "diagnostic_only_not_template_or_model_tuning_on_heldout": True,
        }
    coverage = read(folder / "template_scale_coverage.json")
    report = {
        "status": "prepared_not_yet_trained",
        "experiment": "fixed5000_K3_seed42",
        "data_manifest_sha256": digest(folder / "manifest.json"),
        "frozen_core_count": manifest["frozen_core_count"],
        "scope_core_count": manifest["scope_core_count"],
        "split_counts": manifest["split_counts"],
        "approved_seed_templates": manifest["seed_templates"],
        "main_bank_size": manifest["main_bank_size"],
        "builder_H_formula_pass_count": 5000,
        "no_self_or_equivalent_references": manifest["no_target_core_self_reference"],
        "no_validation_or_test_reference_graphs": manifest[
            "no_validation_or_test_reference_graphs"
        ],
        "retrieval_audits": manifest["retrieval_audits"],
        "coverage": coverage,
        "functional_coverage": functions,
        "frozen_pool_functional_counts": present,
        "paper": {
            "url": "https://doi.org/10.1038/s41467-026-72149-6",
            "main_batch_size": 32,
            "selection": "minimum_validation_token_cross_entropy",
            "patience_complete_epochs": 5,
        },
        "generation_metrics_select_checkpoint": False,
        "generation_evaluation_completed": False,
        "other_pool_size_and_K_training_completed": False,
    }
    for filename, key in [("cpu_gate.json", "cpu_gate"), ("gpu_training_gate.json", "gpu_gate")]:
        if (ROOT / filename).exists():
            gate = read(ROOT / filename)
            report[key] = {"passed": gate["passed"], "audits": gate["audits"]}
            report["environment"] = gate["contract"]["runtime_versions"]
    if (ROOT / "selected_runtime.json").exists():
        runtime = read(ROOT / "selected_runtime.json")
        report["selected_batch_size"] = runtime["selected_batch_size"]
        report["batch_benchmark"] = runtime["measurements"]
    run = ROOT / "train_main_k3_seed42"
    status = ROOT / "train_main_k3_seed42.supervisor.json"
    if status.exists():
        state = read(status)
        report["status"] = "training_" + state["status"]
        report["supervisor_pid"] = state["supervisor_pid"]
        report["worker_launcher_pid"] = state["child_pid"]
        report["monitor"] = {
            "gpu_stop_latched": (ROOT / "GPU_STOP.json").exists(),
            "auto_retry": False,
            "timeout_seconds": state["timeout_seconds"],
        }
        if (run / "events.jsonl").exists():
            rows = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
            training = [r for r in rows if r["stage"] == "train_progress"]
            report["latest_observed_progress"] = training[-1] if training else None
        if (run / "last.json").exists():
            report["last_checkpoint_step"] = read(run / "last.json")["step"]
        if (run / "training_report.json").exists():
            report["training_completion"] = read(run / "training_report.json")
    epoch = ROOT / "gpu_monitor_epoch_private.json"
    if epoch.exists():
        report["unreadable_dump_directories"] = read(epoch)["existing_dumps"][
            "unreadable_directories"
        ]
    recovery = ROOT / "GPU_RESTART_AUTHORIZATION.json"
    if recovery.exists():
        recovery_state = read(recovery)
        report["GPU_retry_authorized"] = recovery_state["authorized"]
        prior = Path(recovery_state["prior_failed_root"]) / "benchmark_batch16_32/host_monitor.json"
        if prior.exists():
            old = read(prior)
            report["initial_GPU_attempt"] = {
                name: old[name]
                for name in (
                    "child_exit_code",
                    "stop_reason",
                    "display_reset_count",
                    "new_gpu_dump_count",
                    "protected_checkpoints_unchanged",
                    "unreadable_dump_directories",
                )
            }
            report["initial_GPU_attempt"]["failure"] = (
                "Strict deterministic_algorithms conflicted with torch2.9 CUDA nll_loss2d; reverted to historical runtime"
            )
        if not recovery_state["authorized"]:
            report["status"] = "awaiting_explicit_GPU_retry_authorization_after_CPU_repair"
    destination = Path("docs/results/TEMPLATE_GUIDED_FP5000_START_20261022.json")
    write(destination, report)
    text = [
        "# 固定5,000模板训练：启动与准备记录",
        "",
        f"当前快照状态：`{report['status']}`。本文件是实际执行快照，不表示全部实验完成。",
        "",
        "选模按[MetGenX原文](https://doi.org/10.1038/s41467-026-72149-6)：完整验证集最小交叉熵；连续5个完整epoch不改善即停止。",
        "分子式/Builder有效性、Similarity@10、官能团恢复及相对最佳输入模板的收益用于随后评价，不参与选checkpoint。",
        "",
        f"冻结原库 {manifest['frozen_core_count']:,} 图，派生范围 {manifest['scope_core_count']:,} 图；训练/验证/测试分别为 "
        + "/".join(f"{n:,}" for n in manifest["split_counts"].values())
        + "。",
        f"主参考池5,000图包含{manifest['seed_templates']}个人工审核兼容种子，其余为训练核心补选；全部5,000通过Builder及独立含H元素平衡。",
        "原始库、旧split、恢复目录和旧权重保留；无验证/测试核心进入参考池，每个目标排除自体和等价结构。",
        "",
        "|模板池|最近模板Tanimoto中位数|≥0.4比例|≥0.6比例|未覆盖官能团类别|",
        "|---:|---:|---:|---:|---|",
    ]
    for size in manifest["nested_sizes"]:
        c = coverage[str(size)]["all_frozen_cores"]
        text.append(
            f"|{size:,}|{c['median_nearest_tanimoto']:.3f}|{c['fraction_at_least_0.4']:.1%}|{c['fraction_at_least_0.6']:.1%}|{', '.join(functions[str(size)]['uncovered_classes_present_in_frozen_pool']) or '无'}|"
        )
    text.extend(
        [
            "",
            "覆盖是全195,306图的事后诊断，排除自体参考；不是生成准确率，也不参与选模。完整结构/公式分布和官能团计数见同名JSON。",
            "覆盖采用rooted Polar Morgan；训练入口沿用chainless Morgan。检索为HNSW候选加精确重排，覆盖统计仍是近似诊断，精确抽样命中率见JSON。",
            "",
        ]
    )
    if "selected_batch_size" in report:
        text += [
            f"实际batch：**{report['selected_batch_size']}**。",
            "",
            "|batch|目标/秒|秒/步|峰值allocated MiB|",
            "|---:|---:|---:|---:|",
        ]
        text += [
            f"|{r['batch_size']}|{r['targets_per_second']:.1f}|{r['seconds_per_step']:.4f}|{r['peak_allocated_MiB']:.1f}|"
            for r in report["batch_benchmark"]
        ]
    if report.get("latest_observed_progress"):
        r = report["latest_observed_progress"]
        text += [
            "",
            f"启动核查时已到step {r['step']:,} / epoch {r['epoch']}。这是训练loss，不是最终生成效果。",
        ]
    if "last_checkpoint_step" in report:
        text += [f"已原子保存并生成SHA的最近checkpoint：step {report['last_checkpoint_step']:,}。"]
    if "initial_GPU_attempt" in report:
        text += [
            "",
            "首次GPU测速在第一步loss处因新增严格确定性开关不兼容CUDA nll_loss2d退出；未完成反向传播。",
            "修订撤回该额外开关，恢复历史运行选项；原失败记录和停止锁保留。监控无驱动重置/新GPU dump，12份旧权重SHA不变。",
            "根据项目AGENTS.md，失败GPU作业不得自动重试。人工明确授权前，r1目录只完成CPU检查。",
        ]
    text += [
        "",
        "所有数据、权重、逐图参考和预测保存在G盘私有输出；仅代码、配置及聚合报告公开。",
        "监督System4101、WER141/117、可访问dump和NVIDIA遥测；受保护WATCHDOG目录不可读，记录访问限制；出错锁定且不自动重试。",
        "后续待完成：按早停规则完成主训练、选定模型后的生成评价、其他池规模/K/种子实验。",
        "执行契约：[TEMPLATE_GUIDED_FP5000_RUN.md](../experiments/TEMPLATE_GUIDED_FP5000_RUN.md)。",
        "",
    ]
    destination.with_suffix(".md").write_text("\n".join(text), encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("status", "split_counts", "approved_seed_templates", "main_bank_size")
            }
        )
    )


if __name__ == "__main__":
    main()
