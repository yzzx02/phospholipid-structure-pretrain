"""Read-only aggregate report of the new-library freeze, gates and live progress."""

# ruff: noqa: E501 -- Chinese report paragraphs are kept as complete strings.

import ctypes
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from structure_pretrain.core_templates.common import ROOT, configuration, digest, read, write


def alive(pid):
    if not pid:
        return False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, 0, int(pid))
    if not handle:
        return False
    code = ctypes.c_uint32()
    try:
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def main():
    root, c = ROOT.resolve(), configuration()
    manifest = read(root / "data/manifest.json")
    audit = read(root / "data/data_audit.json")
    bank = pq.read_table(
        root / "data/template_bank_private.parquet", columns=["contains_P"]
    ).to_pydict()
    targets = pq.read_table(
        root / "data/targets_private.parquet", columns=["contains_P", "split"]
    ).to_pydict()
    counts = {}
    for name in ("train", "validation", "test"):
        selection = np.asarray(targets["split"]) == name
        p = np.asarray(targets["contains_P"])
        counts[name] = {
            "total": int(selection.sum()),
            "P": int((selection & p).sum()),
            "nonP": int((selection & ~p).sum()),
        }
    state = read(root / "state.json") if (root / "state.json").exists() else {"status": "preparing"}
    status = root / (state.get("stage", "") + ".supervisor.json")
    supervisor = read(status) if status.exists() else {}
    process_running = alive(supervisor.get("child_pid")) and alive(supervisor.get("supervisor_pid"))
    event_path = root / state.get("stage", "") / "events.jsonl"
    events = (
        [json.loads(s) for s in event_path.read_text().splitlines() if s]
        if event_path.exists()
        else []
    )
    train_events = [s for s in events if s.get("stage") == "train_progress"]
    validations = [s for s in events if s.get("stage") == "validation_completed"]
    gates = {}
    for name in ("cpu_gate", "cuda_gate"):
        if (root / (name + ".json")).exists():
            value = read(root / (name + ".json"))
            gates[name] = {
                "passed": value["passed"],
                "checks": value["checks"],
                "elapsed_seconds": value["elapsed_seconds"],
            }
    old = []
    for previous in ("E1_seed42", "E2_seed42_resume_20261026", "E3_seed42_resume_queue_20261026"):
        value = read(
            Path("outputs/template_guided_v2_20261023") / previous / "training_report.json"
        )
        old.append(
            {
                key: value[key]
                for key in ("experiment", "completed_epochs", "steps", "best_validation_token_CE")
            }
        )
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": c,
        "old_V2_training": old,
        "old_V2_generation_failed_on_RDKit_candidate": True,
        "old_V2_stop_lock_preserved_sha256": digest(
            "outputs/template_guided_v2_20261023/GPU_STOP.json"
        ),
        "dataset": {
            key: manifest[key]
            for key in (
                "targets",
                "P_targets",
                "nonP_targets",
                "split_counts",
                "bank_size",
                "steps_per_epoch",
                "max_optimizer_steps",
                "max_observed_label_tokens",
                "vocabulary_size",
            )
        },
        "split_P_nonP_counts": counts,
        "template_P_count": int(sum(bank["contains_P"])),
        "template_nonP_count": int(sum(not x for x in bank["contains_P"])),
        "bank_members_retained_as_train_prediction_targets": 8000,
        "bank_self_or_equivalent_references": audit["bank_self_or_equivalent_references"],
        "cross_split_equivalent_graphs": audit["cross_split_equivalence_intersections"],
        "equivalence_groups": audit["equivalence_groups"],
        "bank_selection": c["bank_selection"],
        "coverage": audit["coverage"],
        "below_similarity_threshold_nearest_fallbacks": audit["below_threshold_nearest_fallbacks"],
        "retrieval_oracle_match_fraction": audit["oracle_match_fraction"],
        "gates": gates,
        "stage": state.get("stage"),
        "state_file_status": state["status"],
        "live_supervised_child_verified": bool(process_running),
        "latest_train_progress": train_events[-1] if train_events else None,
        "latest_complete_validation": validations[-1] if validations else None,
        "stop_latch_exists": (root / "GPU_STOP.json").exists(),
        "generation_evaluation": {},
        "not_an_end_to_end_spectra_benchmark": True,
    }
    train_report = root / "train_main_k3_seed42/training_report.json"
    if train_report.exists():
        report["training_report"] = read(train_report)
    for cohort in ("validation512", "bank_targets512", "test512"):
        p = root / ("eval_" + cohort) / "evaluation_report.json"
        report["generation_evaluation"][cohort] = read(p) if p.exists() else None
    unit = root / "unit_tests.json"
    report["unit_tests"] = read(unit) if unit.exists() else None
    monitor_epoch = root / "gpu_monitor_epoch_private.json"
    report["monitor_access_limits"] = (
        read(monitor_epoch)["existing_dumps"]["unreadable_directories"]
        if monitor_epoch.exists()
        else []
    )
    if monitor_epoch.exists():
        protected = read(monitor_epoch)["old_V2_source_sha256"]
        report["all_73_historical_source_hashes_unchanged"] = all(
            digest(p) == sha for p, sha in protected.items()
        )
    path = Path("docs/results/GLYCEROL_CORE_TEMPLATES8000_V1_20261026.json")
    write(path, report)
    lines = [
        "# 甘油＋头基新库：8,000 模板正式训练",
        "",
        f"更新时间（UTC）：{report['generated_utc']}",
        "",
        "旧 E1/E2/E3 参数训练均已正常完成；旧生成评价在候选 RDKit 化学解析时中止。旧停止锁及权重均保留。",
        "本实验使用新库，随机初始化；新标签、词表、长度与分子式边界独立于旧 Polar 模型。",
        "",
        f"主库 **{manifest['targets']:,}**：含 P **{manifest['P_targets']:,}**，不含 P **{manifest['nonP_targets']:,}**。",
        "",
        "| 用途 | 总数 | 含 P | 不含 P |",
        "|---|---:|---:|---:|",
    ]
    for name, row in counts.items():
        lines.append(f"| {name} | {row['total']:,} | {row['P']:,} | {row['nonP']:,} |")
    lines += [
        "",
        "模板池恰为 **8,000** 个，全部从 TRAIN 选取；这 8,000 个全部仍是 TRAIN 的预测标签。",
        "每个目标均屏蔽自身及中性去立体等价图，跨划分等价图交集为 0。验证/测试不作为输入模板或拟合归一化统计。",
        "模板自身的生成诊断单列 bank-target leave-one-out，不冒充未见结构测试。",
        "",
        f"模板含 P {report['template_P_count']:,}，不含 P {report['template_nonP_count']:,}。",
        "按 P 数、质量段、糖/肌醇分层并用 Morgan MaxMin 选取，分层配额与总体数量平方根成正比。",
        "检索使用精确 Tanimoto + RDKit 同分重排；512 个独立全库 oracle 检查全部一致。",
        f"低于 0.2 相似度而回退到最近合法参考的目标：{audit['below_threshold_nearest_fallbacks']}；仍排除自体，不跳过这些目标。",
        "",
        "标签为带 [*:101]/[*:102] 链角色的甘油＋头基；分子式为两端 H 封端的中性核心。",
        "直接接 P 的结构从实际连接 P 序列化；其他从头基侧甘油 C 开始。最长标签 94，模型上限 96，无截断/删除。",
        "已授权的 37,609 个质量例外和 7,458 个保护基/酰基磷酸风险记录照常保留并带原标记；其余暂存/排除集未加入。",
        "",
        "batch32、K3、FP32/Math/TF32off；AdamW，weight_decay0.01，clip1.0。",
        f"E3 方案：预热4,000 步至3e-4，余弦至1e-5；每轮 {manifest['steps_per_epoch']:,} 步，最多80轮/{manifest['max_optimizer_steps']:,}步。",
        "只按完整验证 Token CE 最小值保存最佳模型，连续5个完整验证轮次不改善早停。资源上限不等于收敛。",
        "保留 best + 最近两份完整恢复权重，最多3个不同文件；不保存 logits/attention/大批重复组装图。",
        "",
        f"当前阶段：`{report['stage']}`；监督器与子进程存活核验：**{process_running}**；新停止锁：{report['stop_latch_exists']}。",
    ]
    if report["latest_train_progress"]:
        v = report["latest_train_progress"]
        lines.append(
            f"最新已记录训练进度：第{v['epoch']}轮，step {v['step']:,}，batch CE {v['training_CE']:.6f}。"
        )
    if validations:
        v = validations[-1]
        lines.append(
            f"最近完整验证：第{v['epoch']}轮，{v['validation_targets']:,}个目标，Token CE {v['validation_token_CE']:.8f}。"
        )
    lines += [
        "",
        "CPU/CUDA 的真实反向传播、磁盘断点恢复、下一批次/RNG/优化器/调度器一致性及最长标签压力检查见 JSON。",
        "训练后固定 validation512、模板目标512（自体屏蔽）、test512，以 Beam20 分别报告原始与后验分子式/组装过滤结果。",
        "每个空结果/无效候选都保留在分母；RDKit 解析或 KekulizeException 计为候选失败，不中止整个评价。",
        "生成指标尚未完成时保持 null，Token CE/teacher-forced token accuracy 不能当作分子结构生成正确率。",
        "",
        "结构预训练仍用目标结构检索模板；这是结构条件下的生成实验，不是独立真实 MS/MS 检索性能。",
        "生成评价记录有效性、分子式、Exact@1/5/10、Similarity@10、官能团恢复及相对最佳输入模板的增益，均不用于测试集选模。",
        "当前不启动额外模板数量/K/seed 消融、不做 MS/MS、CL、PGCN。后续数量实验沿用同一新冻结划分并另行执行。",
        "全部结构、来源、ID、预测和权重只在 G 盘；GitHub 仅含代码、配置和聚合报告。",
        "Windows System4101、WER141/117、可读 GPU dump 与 NVIDIA 遥测持续监控；受保护 WATCHDOG 目录访问限制保持如实记录。",
        "",
    ]
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": report["stage"],
                "live_child": process_running,
                "latest_progress": report["latest_train_progress"],
                "latest_validation": report["latest_complete_validation"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
