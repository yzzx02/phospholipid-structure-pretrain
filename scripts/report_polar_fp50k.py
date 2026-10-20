"""Publish aggregate FP50k results and external report hashes; no private row export."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04.fp50k_data import ROOT, get_contract


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def markdown(report):
    lines = [
        "# v0.4-FP 50k 稳定版：实测结果",
        "",
        f"完成状态：{report['status']}；累计 {report['optimizer_steps']:,} optimizer steps，"
        f"{report['consumed_targets']:,} 次目标输入。",
        "本轮从随机权重训练，与旧 20k/3,000-step 模型独立；没有开展全库预训练或 MS/MS 微调。",
        "固定 FP 架构 16,468,671 参数，Math/FP32、TF32关闭、有效batch16；"
        "内部连接标记与 Builder 保持原样。",
        "",
        "## 数据和工程门槛",
        "",
        "50,000 个真实 TRAIN 行对应 50,000 个不同 Polar graph、31,206 种式；"
        "旧 20k 池为 18,112 个不同 graph、11,233 种式。",
        "单链5,374、双链44,626；A/A31,067、O/A13,558、稀有P/A1。"
        "phosphate21,305、phosphonate19,367、polyphosphate9,328。",
        "选择覆盖 family、长度和元素类别，并优先覆盖式；"
        "因此同时改变了 graph 覆盖及抽样分布，不能将差异只归因于池大小。",
        "K3、阈值0.2，实际149,967个模板配对，H占80.1996%，5个目标合法fallback，零无模板目标。",
        "复用相同检索契约下的冻结全TRAIN公式统计；"
        "所有实际模板对重新检查 TRAIN-only、异头基、链型和相似度。",
        "",
        f"正式运行前CPU测试 {report['cpu_tests_at_gate']} 项通过；最终回归见 JSON。"
        "真实CPU连续4步与2+2恢复权重、优化器、随机状态、消费位置和loss序列完全一致。",
        "CUDA实际20+5步探针通过，再从随机权重开始正式实验。"
        "子模块审计单独核对 FP、公式MLP、融合、TCLS、Set Encoder、Decoder "
        "的更新和非零优化器梯度矩。",
        "",
        "## 分阶段训练",
        "",
        "|阶段|恢复起点→累计步数|末100步平均训练CE|验证CE|验证Token准确率|"
        "训练及检查秒|目标/秒|峰值显存MiB|峰值RAM MiB|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in report["training_stages"]:
        val = stage["validation_history"][-1]
        lines.append(
            f"|{stage['name']}|{stage['step_start']}→{stage['step']}|"
            f"{stage['last100_train_CE_mean']:.4f}|{val['polar_CE']:.4f}|"
            f"{pct(val['token_accuracy'])}|{stage['elapsed_seconds']:.2f}|"
            f"{stage['targets_per_second']:.1f}|{stage['peak_VRAM_allocated_bytes'] / 2**20:.1f}|"
            f"{stage['peak_RAM_bytes'] / 2**20:.1f}|"
        )
    lines += [
        "",
        "训练耗时含本阶段验证、保存和重载检查，不含进程初始加载及生成评价。"
        "PyTorch显存不含Windows显示/驱动的全部开销；RAM为主进程峰值，未合计DataLoader workers。"
        "宿主遥测另存JSON。",
        "",
        *report["stage_decision_summary"],
        "",
        "## 生成结果",
        "",
        "表中 Greedy 为原始无约束搜索；Beam5/20/50为elements_grammar搜索。"
        "严格候选均经含H的Polar式、连接、Builder及完整中性式检查和canonical图去重。",
        "每种搜索的验证前输出 raw 与 strict_post_filter 分开保存；约束搜索的 raw 不是无约束Beam。",
        "",
        "|模型/步数|搜索|目标数|原始候选Polar合法率|严格式覆盖|无候选|合法异构体/目标|严格Top1|严格Top5|严格Top10|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, case in report["generation"].items():
        m = case["metrics"]["strict_post_filter"]
        top = m["polar_topk_exact_over_all_targets"]
        label = "Greedy" if "_raw_beam1_" in key else "Beam" + key.split("beam")[1].split("_")[0]
        model = key.split("_elements_")[0].split("_raw_")[0]
        lines.append(
            f"|{model}|{label}|{m['targets']}|{pct(m['raw_chemical_valid_polar_fraction'])}|"
            f"{pct(m['strict_formula_coverage'])}|{pct(m['no_qualified_candidate_rate'])}|"
            f"{m['accepted_unique_per_target']['mean']:.3f}|{pct(top['1'])}|"
            f"{pct(top['5'])}|{pct(top['10'])}|"
        )
    lines += [
        "",
        "Beam5 的64条前缀仅作阶段A快检，不与完整512条基线直接比较。"
        "Top-k分母包括无候选目标；Greedy的Top5/10与Beam5的Top10不适用。",
        "",
        "## Polar图相似度和错误",
        "",
        "Tanimoto使用Polar-only Morgan r2/2048、非手性，保留内部连接dummy。"
        "全目标Top1/best均数中无候选记0；合格候选条件分布另列，不能混用。",
        "",
        "|模型/搜索|严格Top1 Tanimoto 全目标|严格best Tanimoto 全目标|错误合格候选Tanimoto均数|"
        "式正确但图错误|模板复制候选|新于TRAIN候选|目标进入严格Beam|进入但排10名后|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, case in report["generation"].items():
        d, m = case["diagnostics"], case["metrics"]["strict_post_filter"]
        wrong = d["wrong_accepted_polar_tanimoto"]["mean"]
        lines.append(
            f"|{key}|{d['strict_top1_polar_tanimoto_all_targets']['mean']:.4f}|"
            f"{d['strict_best_polar_tanimoto_all_targets']['mean']:.4f}|"
            f"{wrong:.4f}|{pct(m['formula_correct_wrong_polar_structure_fraction_among_accepted'])}|"
            f"{d['exact_input_template_copy_candidates']}|{d['accepted_novel_to_entire_train']}|"
            f"{d['true_graph_in_accepted_beam']}|{d['true_graph_present_below_rank10']}|"
            if wrong is not None
            else f"|{key}|—|—|—|—|0|0|0|0|"
        )
    lines += [
        "",
        "JSON同时给出分位数、候选与Top1模板的相似关系、TRAIN/50k池已见情况、"
        "family/长度/模板相似度分层、多合法连接图目标数及真实目标未进入Beam的数量。",
        "错误标志为非互斥graph差异：连接P/全部P二跳邻域、官能团及到连接P的图距离、"
        "末端原子/键、非P分支度、环组成。没有使用SMILES编辑距离；不能可靠归因时为unclassified。",
        "",
        "|模型/搜索|错误候选数|P邻域差异|官能团/距离差异|末端差异|分支差异|环差异|未分类|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, case in report["generation"].items():
        d = case["diagnostics"]
        e = d["graph_difference_feature_flags_wrong_candidates"]
        lines.append(
            f"|{key}|{d['wrong_accepted_polar_tanimoto']['n']}|"
            f"{e.get('phosphate_neighborhood', 0)}|{e.get('functional_group_graph_distances', 0)}|"
            f"{e.get('terminal_atoms', 0)}|{e.get('branch_degree_patterns', 0)}|"
            f"{e.get('ring_patterns', 0)}|{e.get('unclassified', 0)}|"
        )
    lines += ["", "## 与旧20k/3,000比较", ""]
    for key, case in report["generation"].items():
        if "paired_vs_20k_3000" not in case:
            continue
        p = case["paired_vs_20k_3000"]
        lines.append(
            f"- {key}：覆盖差 {p['coverage']['mean_delta'] * 100:+.2f} 个百分点；"
            f"Top5差 {p['exact5']['mean_delta'] * 100:+.2f}、"
            f"Top10差 {p['exact10']['mean_delta'] * 100:+.2f} 个百分点；"
            f"全目标Top1 Polar Tanimoto差 {p['top1_tanimoto']['mean_delta']:+.4f}。"
        )
    lines += [
        "",
        "512条为反复使用的诊断集。JSON的配对bootstrap区间仅描述这些诊断目标的重采样差异，"
        "不代表独立泛化置信度。20k/3,000对50k/5,000还改变了步数，只能描述性比较。",
        "",
        "## GPU监控和结论",
        "",
        f"完成{len(report['host_monitor_jobs'])}个独立受监控CUDA作业；"
        f"观察到显示重置事件{report['monitor_totals']['display_resets']}，"
        f"可读目录新增/变更GPU dump {report['monitor_totals']['new_dumps']}。"
        "全部作业前后保护的checkpoint哈希一致，无自动重试。",
        "普通进程无法读取受保护WATCHDOG目录，不能把未观察到事件解释为绝无转储；"
        "System4101、WER141/117、其他可读目录及NVIDIA遥测仍持续监控。",
        "",
        *report["conclusions"],
        "",
        "生成正确式、生成正确Polar、真实MS/MS证据支持是三种不同结果。TEST未用于本轮训练或选模。",
        "",
        "## 哈希与复现",
        "",
        f"训练源码提交：`{report['execution_source_commit']}`；最终交付提交由提交回执/用户回复提供。",
        f"训练池row数组SHA256：`{report['pool']['target_rows_sha256']}`。",
        "",
        "|Step|Checkpoint SHA256|",
        "|---:|---|",
    ]
    for step, item in sorted(report["checkpoints"].items()):
        lines.append(f"|{step}|`{item['sha256']}`|")
    lines += [
        "",
        "源码、数据资产、检索、公式统计、训练/评价/监控文件哈希见聚合JSON；"
        "报告文件自身哈希见 `POLAR_FP_BASE_50K_SHA256.json`。权重、行ID和逐目标预测保留G盘。",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root", type=Path, default=ROOT)
    p.add_argument("--source-commit", required=True)
    args = p.parse_args()
    root = args.experiment_root
    config, _, contract = get_contract(root, verify_assets=True)
    decision = read_json(root / "final_decision.json")
    generation = read_json(root / "analysis_aggregate.json")
    assert not (root / "GPU_STOP.json").exists()
    hosts = []
    stages = []
    checkpoints = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if (folder / "host_monitor.json").exists():
            host = read_json(folder / "host_monitor.json")
            assert host["child_exit_code"] == 0 and host["stop_reason"] is None
            assert host["protected_checkpoints_unchanged"]
            hosts.append(
                {
                    "name": folder.name,
                    "host_monitor_sha256": file_hash(folder / "host_monitor.json"),
                    **host,
                }
            )
        if not folder.name.startswith("train_") or not (folder / "training_metrics.json").exists():
            continue
        m = read_json(folder / "training_metrics.json")
        assert m["contract"] == contract and not m["probe"]
        history = m.pop("history")
        m.pop("contract")
        m.update(
            name=folder.name,
            first100_train_CE_mean=float(np.mean([x["polar_CE"] for x in history[:100]])),
            last100_train_CE_mean=float(np.mean([x["polar_CE"] for x in history[-100:]])),
            training_metrics_sha256=file_hash(folder / "training_metrics.json"),
        )
        stages.append(m)
        for checkpoint in (folder / "checkpoint").glob("step_*.pt"):
            step = int(checkpoint.stem.split("_")[1])
            checkpoints[step] = {
                "sha256": file_hash(checkpoint),
                "bytes": checkpoint.stat().st_size,
            }
    stages.sort(key=lambda s: s["step"])
    for a, b in zip(stages, stages[1:]):
        assert (
            a["step"] == b["step_start"] and a["checkpoint_sha256"] == b["resume_checkpoint_sha256"]
        )
    final_step = stages[-1]["step"]
    assert final_step in (3000, 5000) and set(checkpoints) == set(range(1000, final_step + 1, 1000))
    for beam in (20, 50):
        assert f"50k_step{final_step}_elements_grammar_beam{beam}_n512" in generation
    assert f"50k_step{final_step}_raw_beam1_n512" in generation
    audit_files = sorted(root.glob("checkpoint_audit_*.json"))
    env = read_json(root / "environment.json")
    report = {
        "schema": "polar_fp_base_50k_results_v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": decision["status"],
        "optimizer_steps": final_step,
        "consumed_targets": final_step * 16,
        "execution_source_commit": args.source_commit,
        "config": contract["config"],
        "parameters": 16468671,
        "pool": read_json(root / "pool/manifest_private.json"),
        "contract": contract,
        "environment": env,
        "requirements_lock_sha256": file_hash(root / "requirements.lock.txt"),
        "cpu_tests_at_gate": 366,
        "cpu_gate": read_json(root / "cpu_gate.json"),
        "gpu_training_gate": read_json(root / "gpu_training_gate.json"),
        "final_validation": read_json(root / "final_validation.json"),
        "checkpoint_submodule_audits": [read_json(path) for path in audit_files],
        "training_stages": stages,
        "checkpoints": checkpoints,
        "generation": generation,
        "stage_decision_summary": decision["stage_summary"],
        "conclusions": decision["conclusions"],
        "host_monitor_jobs": hosts,
        "monitor_totals": {
            "display_resets": sum(h["display_reset_count"] for h in hosts),
            "new_dumps": sum(h["new_gpu_dump_count"] for h in hosts),
        },
        "private_target_rows_or_predictions_exported": False,
        "independent_test_used": False,
        "diagnostic_cohort_reused": True,
    }
    # Verify the committed training sources match the run contract.
    for name, expected in contract["source_sha256"].items():
        # Git normalizes CRLF to LF; compare working source through the index filter.
        actual = subprocess.check_output(
            ["git", "hash-object", "--path=" + name, "--stdin"], input=Path(name).read_bytes()
        ).strip()
        committed = subprocess.check_output(
            ["git", "rev-parse", f"{args.source_commit}:{name}"]
        ).strip()
        assert actual == committed and file_hash(Path(name)) == expected
    out = Path("docs/results")
    j, md = out / "POLAR_FP_BASE_50K_RESULTS.json", out / "POLAR_FP_BASE_50K_RESULTS.md"
    j.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md.write_text(markdown(report), encoding="utf-8")
    hashes = {path.as_posix(): file_hash(path) for path in (j, md)}
    (out / "POLAR_FP_BASE_50K_SHA256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "step": final_step, "reports_sha256": hashes}))


if __name__ == "__main__":
    main()
