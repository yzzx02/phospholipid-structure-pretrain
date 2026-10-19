"""Publish aggregate evidence only; never read/publish per-target predictions or structures."""

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04.prepare import DEFAULT_ROOT, SAMPLES, V03, dump
from structure_pretrain.polar_only_v04.retrieval import chosen_mode, quantiles

RUNS = Path("outputs/pretraining_v04/runs")


def cohort_pairs():
    refs = {m: np.load(DEFAULT_ROOT / f"{m}_rows.npy", mmap_mode="r") for m in ("H", "HC")}
    scores = {m: np.load(DEFAULT_ROOT / f"{m}_scores.npy", mmap_mode="r") for m in refs}
    result = {}
    for name in ("train_pool", "validation_pool", "diagnostic_512", "overfit_32"):
        rows = np.load(SAMPLES / f"{name}.npy")
        count = {m: (a[rows] >= 0).sum(-1) for m, a in refs.items()}
        h, fallback = chosen_mode(rows, count["H"], count["HC"], force_h=name == "overfit_32")
        r = np.where(h[:, None], refs["H"][rows, :3], refs["HC"][rows, :3])
        s = np.where(h[:, None], scores["H"][rows, :3], scores["HC"][rows, :3])
        mask = r >= 0
        result[name] = {
            "targets": len(rows),
            "H_targets": int(h.sum()),
            "HC_targets": int((~h).sum()),
            "fallback_targets": int(fallback.sum()),
            "template_pairs": int(mask.sum()),
            "same_chain_pair_fraction": float(mask[h].sum() / mask.sum()),
            "different_head_pair_fraction": 1.0,
            "less_than_K3_targets": int((mask.sum(-1) < 3).sum()),
            "retrieval_top1_quantiles": quantiles(s[:, 0]),
            "retrieval_topK_quantiles": quantiles(s[mask]),
        }
    return result


def main():
    output = Path("docs/results/POLAR_ONLY_V04_RESULTS.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path("outputs/pretraining_v04")
    tests = read_json(root / "validation_checks.json")
    training, evaluations, monitors = {}, {}, {}
    for directory in sorted(RUNS.iterdir()):
        for name, target in (
            ("training_metrics", training),
            ("evaluation_metrics", evaluations),
            ("host_monitor", monitors),
        ):
            path = directory / f"{name}.json"
            if path.exists():
                target[directory.name] = read_json(path)
                if name == "host_monitor":
                    record = target[directory.name]
                    if "steps" in record:
                        record["supervisor_steps_argument"] = record.pop("steps")
                    evaluation = directory / "evaluation_metrics.json"
                    if evaluation.exists():
                        record["evaluated_checkpoint_step"] = read_json(evaluation)[
                            "checkpoint_step"
                        ]
                if name == "training_metrics":
                    contract = read_json(directory / "resolved_config_private.json")
                    target[directory.name]["checkpoint_contract"] = {
                        key: contract[key]
                        for key in (
                            "schema",
                            "version",
                            "kind",
                            "config",
                            "outputs",
                            "known_chains_required",
                            "target_rows_sha256",
                            "train_targets",
                            "source_labels_sha256",
                            "tokenizer_sha256",
                            "features_sha256",
                            "retrieval_sha256",
                            "formula_stats_sha256",
                            "attention_backend",
                            "precision",
                            "initialization",
                        )
                    }
    files = sorted(
        [
            *Path("src/structure_pretrain/polar_only_v04").glob("*.py"),
            *Path("configs").glob("polar_only_v04_*.yaml"),
            Path("train_polar_only_v04.py"),
            Path("tests/test_polar_only_v04.py"),
            Path("scripts/supervise_polar_only_v04.ps1"),
            Path("scripts/audit_polar_only_v04_preservation.py"),
            Path("scripts/audit_polar_only_v04_checkpoints.py"),
            Path(__file__).relative_to(Path.cwd()),
        ]
    )
    hashes = {str(path): file_hash(path) for path in files}
    implementation = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    baseline = read_json("docs/results/MIDTERM_STRUCTURE_20261019.json")
    baseline_5000 = [
        {"encoder": e["encoder"], "step": e["step"], "cohort": e["cohorts"]["diagnostic_512"]}
        for e in baseline["evaluations"]
        if e["step"] == 5000
    ]
    preservation_path = root / "preservation_audit.json"
    decision_path = root / "decision.json"
    result = {
        "schema": "polar-only-v04-aggregate-results-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "issue": "https://github.com/yzzx02/phospholipid-structure-pretrain/issues/1",
        "implemented_and_tested": {
            "status": "passed",
            "validation": tests,
            "CPU_real_data": read_json(root / "cpu_integration_audit.json"),
            "K5_CPU_real_data": read_json(root / "K5_cpu_audit.json"),
        },
        "awaiting_local_GPU": [
            encoder
            for encoder in ("fingerprint", "graph")
            if not any(
                e["encoder"] == encoder and e["kind"] == "smoke" and e["checkpoint_step"] >= 1000
                for e in evaluations.values()
            )
        ],
        "GPU_measured": {
            "training": training,
            "evaluations": evaluations,
            "host_monitors": monitors,
        },
        "new_checkpoint_integrity_audit": read_json(root / "checkpoints_audit.json"),
        "whole_run_GPU_safety_audit": read_json(root / "whole_run_gpu_safety_audit.json"),
        "data": {
            "source_binding": read_json(DEFAULT_ROOT / "source_binding.json"),
            "formula_audit": read_json(DEFAULT_ROOT / "formula_audit.json"),
            "fragment_integrity_audit": read_json(DEFAULT_ROOT / "fragment_integrity_audit.json"),
            "retrieval_audit": read_json(DEFAULT_ROOT / "retrieval_audit.json"),
            "formula_statistics": read_json(DEFAULT_ROOT / "formula_stats_train_only.json"),
            "samples": read_json(SAMPLES / "manifest.json"),
            "cohort_pair_statistics": cohort_pairs(),
        },
        "reproducibility": {
            "seed": 42,
            "attention": "math",
            "precision": "fp32",
            "effective_smoke_batch": 16,
            "max_Polar_length_including_BOS_EOS": 84,
            "source_files_sha256": hashes,
            "implementation_digest": implementation,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "torch-geometric", "rdkit", "pyarrow", "usearch")
            },
            "host_environment": read_json(root / "environment_host.json"),
            "split_sha256": file_hash(V03 / "row_splits.npy"),
        },
        "v03_preservation": read_json(preservation_path)
        if preservation_path.exists()
        else {"status": "pending"},
        "v03_5000_descriptive_reference": baseline_5000,
        "comparison_caveat": "v0.3 predicts chains; v0.4 uses GIVEN target chains and different "
        "formula/retrieval inputs. Full-molecule metrics are conditional and not a causal "
        "encoder-only ablation. Same512 target IDs, oracle retrieval; not MS/MS accuracy.",
        "decision": read_json(decision_path) if decision_path.exists() else {"status": "pending"},
        "bounded_extension_decisions": [
            read_json(root / name)
            for name in ("extension_to3000.json", "extension_to5000.json")
            if (root / name).exists()
        ],
        "test_used_for_training_or_model_selection": False,
        "full_pretraining_started": False,
        "MSMS_finetuning_started": False,
        "private_structures_predictions_indexes_checkpoints_published": False,
    }
    dump(output, result)
    write_markdown(result, output.with_suffix(".md"))
    print(
        json.dumps({"aggregate_report": str(output), "awaiting_GPU": result["awaiting_local_GPU"]})
    )


def write_markdown(result, path):
    def pct(value):
        return f"{100 * value:.2f}%"

    gpu = result["GPU_measured"]
    evals = list(gpu["evaluations"].values())
    smoke = sorted(
        (e for e in evals if e["kind"] == "smoke"),
        key=lambda e: (e["checkpoint_step"], e["encoder"]),
    )
    lines = [
        "# v0.4 Polar-only：本机实测结果",
        "",
        f"生成时间（UTC）：{result['generated_utc']}。需求：[Issue #1]({result['issue']})。",
        "[方法与完整复现](../experiments/POLAR_ONLY_V04.md) · "
        "[聚合 JSON／配置与数据 SHA256](POLAR_ONLY_V04_RESULTS.json)",
        "",
        "## 结论",
        "",
        result["decision"].get("summary", "实验按阶段进行中，尚未作最终判断。"),
        "",
        "当前是已知目标链、可信中性式及离线结构 oracle 检索下的 Polar 生成。"
        "不是 MS/MS 鉴定准确率，也不能把库中已有验证头基的命中称为新化合物发现。",
        "",
        "## 已实现且通过测试",
        "",
        f"- 独立 v0.4 模型、dataset、训练入口、配置及 checkpoint 契约；无 Chain Head／Chain Loss。"
        f" {result['implemented_and_tested']['validation']['pytest_passed']} 项 pytest、"
        "Ruff、pip check 通过。",
        "- FP/GINE 真实数据 CPU forward/backward、参数更新、"
        "无目标 Polar 标签的 greedy、worker加载通过。",
        "- 201.5万行 attached-fragment 公式恒等式全部通过；负数／不匹配0。"
        "671条原审计（全部71特殊结构）及额外10,000条 TRAIN 跨家族／连接类型重建零失败。",
        "- 全部390,612个无链 head/occupancy 变体的 canonical、dummy、"
        "glycerol–O–P、元素组成审计通过；"
        "独立重算512个图／指纹并核对所有 Arrow shard哈希。",
        "- TCLS＋全部模板 `[B,K+1,512]` 保留，K3/5/10 padding／置换测试通过；"
        "多链候选按 B×N 独立公式及 memory，未创建假谱图。"
        "K5另用真实数据拟合TRAIN-only统计并通过FP/GINE CPU前向、反向和greedy；K5 GPU未执行。",
        "",
        "## 待本地 GPU 运行",
        "",
        ("初始1000-step实验尚待：" + ", ".join(result["awaiting_local_GPU"]))
        if result["awaiting_local_GPU"]
        else "要求的两模型初始1000-step实验均已实测完成。",
        "本轮还完成了两模型3000步扩展；5000步与50k池未执行，不自动启动完整预训练或MS/MS微调。",
        "",
        "## 数据与检索审计",
        "",
        "冻结 split：TRAIN1,813,451／validation100,792／test100,757；canonical Polar graph交集0。"
        "所有存储参考逐条验证 TRAIN-only、异头基、H/HC链条件与新指纹 Tanimoto。",
        "阈值0.2，K3（缓存至10）；不足K保留真实数量。H／HC各64个穷举对照的最优top1分数命中率均100%。",
        "下表为阈值后的覆盖；阈值前的可用参考数与全部相似度分位数见JSON。",
        "",
        "| split | 模式 | ≥1 | ≥3 | ≥5 | zero |",
        "|---|---|---:|---:|---:|---:|",
    ]
    retrieval = result["data"]["retrieval_audit"]
    for split, record in retrieval["splits"].items():
        for mode, m in record["modes"].items():
            a = m["at_least"]
            lines.append(f"| {split} | {mode} | {a['1']} | {a['3']} | {a['5']} | {m['zero']} |")
    lines += [
        "",
        "TRAIN／validation／test的显式fallback目标数分别为85／4／7；"
        "fallback后zero均为0。TRAIN实际H目标80.0013%、同链模板对79.9977%，所有模板对异头基。",
        "初版CPU准备在常见lyso链上频繁穷举，已停止该CPU任务，改建排除该链的合法TRAIN候选索引；"
        "保留初版日志，未将该准备阶段算作GPU训练。",
        "",
        "## GPU实测结果：32头基过拟合",
        "",
        "32个不同TRAIN头基、相同已知链组合、全部H参考；随机初始化，dropout0、batch8、lr3e-4。",
        "| 编码器 | step | EOS | 合法Polar | greedy Polar exact | Top5 exact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for e in sorted((e for e in evals if e["kind"] == "overfit"), key=lambda e: e["encoder"]):
        m = e["metrics"]
        lines.append(
            f"| {e['encoder']} | {e['checkpoint_step']} | {pct(m['eos_rate'])} | "
            f"{pct(m['polar_valid_rate'])} | {pct(m['polar_exact_rate'])} | "
            f"{pct(m['top5_polar_graph_recall'])} |"
        )
    lines += [
        "",
        "两模型均真实从200步checkpoint恢复模型、优化器、RNG和消费样本游标，继续到201步。"
        "检查点reload logits一致（容差内），优化器step与目标step一致。"
        "这是TRAIN记忆诊断，不是未见头基准确率。",
        "",
        "## GPU实测结果：有界真实 smoke",
        "",
        "匹配20,000训练目标（18,112个头基）、2,000验证目标及固定512个不同验证头基。"
        "FP与GINE从新随机初始化开始，非v0.3权重；有效batch16，K3，lr1e-4、Math SDPA／FP32。",
        "greedy从BOS开始；不强制EOS、不修复化学结构、不预先按公式筛选。",
        "",
        "| 编码器 | step | EOS | 合法Polar | Polar exact | Polar式exact | "
        "完整式exact（给定链） | Top5 Polar | TF-EOS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for e in smoke:
        m, tf = e["metrics"], e["teacher_forcing"]
        values = [
            m[k]
            for k in (
                "eos_rate",
                "polar_valid_rate",
                "polar_exact_rate",
                "polar_formula_match_rate",
                "formula_match_rate",
                "top5_polar_graph_recall",
            )
        ]
        lines.append(
            f"| {e['encoder']} | {e['checkpoint_step']} | "
            + " | ".join(pct(x) for x in (*values, tf["EOS_accuracy"]))
            + " |"
        )
    lines += [
        "",
        "完整图exact、模板复制率、RDKit合法率、TF token／CE、生成长度、检索top1/topK、"
        "目标相似度分位数，以及family／length／retrieval similarity分层均在聚合JSON中保留。",
        "",
        "### 可选Top5完整公式过滤（给定链）",
        "",
        "| 编码器 | step | 选中/512 | 覆盖率 | 完整命中数 | 总体exact | 选中内exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for e in smoke:
        m = e["metrics"]["optional_top5_formula_filter"]
        conditional = (
            pct(m["full_exact_among_selected"])
            if m["full_exact_among_selected"] is not None
            else "—"
        )
        lines.append(
            f"| {e['encoder']} | {e['checkpoint_step']} | {m['selected_targets']} | "
            f"{pct(m['coverage'])} | {m['full_exact_count']} | "
            f"{pct(m['overall_full_exact'])} | {conditional} |"
        )
    lines += [
        "",
        "筛选只在同一组原始beam5中选首个完整式匹配者，未补充或修复候选。"
        "选中内准确率必须与覆盖率、总体准确率一起看。",
        "",
        "## 失败原因与解释",
        "",
    ]
    for encoder in ("fingerprint", "graph"):
        nodes = [e for e in smoke if e["encoder"] == encoder]
        if nodes:
            latest = nodes[-1]
            lines.append(
                f"- {encoder}最后实测节点{latest['checkpoint_step']}步，512目标的raw失败分类："
                + json.dumps(latest["metrics"]["failure_categories"], ensure_ascii=False)
                + "。"
            )
    lines += [
        "",
        "公式条件输入不等于严格计数约束；语法无效与合法但式不匹配需分别分析。"
        "32样本过拟合通过说明接线与优化能工作，不证明大规模泛化。"
        "所有结果只覆盖固定验证诊断集，不能外推至全库或实际谱图。",
        "",
        "## v0.3参照与不可混比的条件",
        "",
        "v0.3的5000步、同一512目标：FP/GINE合法Polar43.95%／48.44%，greedy Polar及完整图exact均0，"
        "Top5为4/512／0/512；完整式3/512／4/512。FP可选过滤后总体完整命中2/512。",
        "v0.3需要预测链，v0.4已给定真实链；同时改变模板表示、公式和检索。"
        "因此完整式／完整图指标是不同条件下的描述性参照，不能归因为单一编码器改进。"
        "v0.3原始报告与checkpoint原样保留。",
        "",
        "## 资源、故障监控与资产保留",
        "",
    ]
    monitors = list(gpu["host_monitors"].values())
    lines.append(
        f"本轮已记录{len(monitors)}个GPU子任务：新增GPU dump "
        f"{sum(m['new_windows_gpu_dump_count'] for m in monitors)}，显示驱动重置 "
        f"{sum(m['display_reset_count'] for m in monitors)}，watchdog停止 "
        f"{sum(bool(m['stop_reason']) for m in monitors)}。未改变TDR、驱动或核心依赖。"
    )
    lines.append("")
    for encoder in ("fingerprint", "graph"):
        tasks = [
            m for m in gpu["training"].values() if m["encoder"] == encoder and m["kind"] == "smoke"
        ]
        if tasks:
            lines.append(
                f"- {encoder}训练任务峰值VRAM allocated："
                f"{max(m['peak_VRAM_allocated_bytes'] for m in tasks) / 2**30:.3f}GiB；"
                f"峰值进程RAM {max(m['peak_RAM_bytes'] for m in tasks) / 2**30:.3f}GiB。"
            )
    p = result["v03_preservation"]
    lines += [
        "",
        f"v0.3保留审计：{p['status']}；589个既有代码/输出、38个旧checkpoints及320个冻结文件"
        "（4,334,776,708 bytes）无内容变化。所有新大数组、检索索引、"
        "逐目标预测与模型权重均在WSL Linux文件系统，未上传。",
        "Math路径的本次有界稳定运行不证明此前默认attention黑屏已经永久修复。"
        "触发GPU dump、显示重置、黑屏通知或stall后应停止，不自动重试。",
        "",
        "## 配置与复现",
        "",
        "[FP配置](../../configs/polar_only_v04_fp.yaml) · "
        "[GINE配置](../../configs/polar_only_v04_gine.yaml) · "
        "[训练入口](../../train_polar_only_v04.py) · "
        "[Windows watchdog](../../scripts/supervise_polar_only_v04.ps1)",
        "具体命令见方法；所有实际训练／评价节点、checkpoint SHA、"
        "数据/检索/标准化/代码SHA、软件版本在JSON中。"
        "Test仅作固定数据/覆盖审计，未用于训练、生成评价或选择。未执行CL四链、完整201.5万预训练或MS/MS微调。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
