"""Publish aggregate metrics only; private target IDs/structures stay in WSL."""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from structure_pretrain.data.prepare_v03 import atom_counts
from structure_pretrain.library_builder.canonicalize import parse_molecule
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04.chemistry import derive_polar_formula
from structure_pretrain.polar_only_v04.formula_beam_eval import (
    EXPECTED_COHORT,
    EXPERIMENT,
    summarize,
)
from structure_pretrain.polar_only_v04.prepare import SAMPLES, V03, row_hash
from structure_pretrain.utils.formula import ELEMENTS

BASE = "2914c9b6fdce273ae2677637b9618a98a1fd064b"
OUTPUT = Path("docs/results/POLAR_FORMULA_BEAM_V04_RESULTS.json")
REPORT = OUTPUT.with_suffix(".md")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_records(records, rows, start, count):
    """Reject missing, repeated, reordered or substituted validation targets."""
    if len(records) != count:
        raise ValueError("Incomplete private predictions")
    if [r["cohort_index"] for r in records] != list(range(start, start + count)):
        raise ValueError("Incorrect cohort positions")
    if [r["row"] for r in records] != rows[start : start + count].tolist():
        raise ValueError("Target rows differ from the frozen validation cohort")


def failure_stages(records):
    stages = Counter()
    for record in records:
        cs = record["raw_candidates"]
        if record["accepted_unique"]:
            continue
        if not cs:
            stages["search_has_no_surviving_beam"] += 1
        elif not any(c["eos"] for c in cs):
            stages["no_EOS"] += 1
        elif not any(c["polar_valid"] for c in cs):
            stages["no_chemically_valid_Polar"] += 1
        elif not any(c["polar_formula_match_including_H"] for c in cs):
            stages["no_exact_Polar_formula_including_H"] += 1
        elif not any(c["polar_formula_match_including_H"] and c["assembled"] for c in cs):
            stages["Builder_rejected_formula_correct_Polar"] += 1
        else:
            stages["no_exact_complete_neutral_formula"] += 1
    return dict(stages)


@lru_cache(maxsize=65536)
def graph_counts(text):
    return atom_counts(parse_molecule(text))


def formula_errors(records, mode):
    """Independent final RDKit audit distinguishes H-only from heavy errors."""
    formulas = np.load(V03 / "formulas.npy", mmap_mode="r")
    chains = np.load(V03 / "chain_labels.npy", mmap_mode="r")
    root = Path("data/private/pretraining_v04")
    chain_counts = np.load(root / "chain_counts.npy")
    scaffold = np.load(root / "scaffold_counts.npy")
    profile, elements = Counter(), Counter()
    heavy = [i for i, symbol in enumerate(ELEMENTS) if symbol != "H"]
    for record in records:
        row = record["row"]
        expected = derive_polar_formula(formulas[row], chains[row], chain_counts, scaffold)
        for candidate in record["raw_candidates"]:
            if not candidate["polar_valid"]:
                continue
            actual = graph_counts(candidate["canonical_polar"])
            differences = actual != expected
            heavy_match = not differences[heavy].any()
            if mode != "raw" and not heavy_match:
                raise ValueError("A chemically valid constrained output violates its heavy budget")
            profile["chemically_valid_Polar_candidates"] += 1
            profile["heavy_formula_exact"] += int(heavy_match)
            profile["H_only_mismatch"] += int(heavy_match and differences.any())
            profile["all_elements_exact_including_H"] += int(not differences.any())
            for i in np.flatnonzero(differences):
                elements[ELEMENTS[i]] += 1
    return {**dict(profile), "element_mismatch_counts": dict(elements)}


def aggregate_case(root, encoder, mode, beam_size, rows, allow_incomplete):
    records, jobs, failed_jobs, inputs = [], [], [], None
    for start in (0, 256):
        folder = root / f"{encoder}_{mode}_b{beam_size}_s{start}"
        host_path, metrics_path = folder / "host_monitor.json", folder / "metrics.json"
        host = read(host_path) if host_path.exists() else None
        unsafe = host and (
            host["child_exit_code"] != 0
            or host["stop_reason"]
            or host["new_windows_gpu_dump_count"]
            or host["display_reset_count"]
            or host["training_started"]
        )
        if unsafe:
            if not allow_incomplete:
                raise ValueError(f"Unsafe or failed GPU run: {folder.name}")
            partial_path = folder / "predictions_private.jsonl"
            partial = (
                [json.loads(line) for line in partial_path.read_text().splitlines()]
                if partial_path.exists()
                else []
            )
            check_records(partial, rows, start, len(partial))
            failure = {
                "name": folder.name,
                "host_monitor": host,
                "partial_completed_targets": len(partial),
                "planned_targets": 256,
                "included_in_main_comparison": False,
                "private_predictions_sha256": file_hash(partial_path) if partial else None,
            }
            if partial:
                failure["partial_diagnostics_not_a_512_target_result"] = {
                    "raw": summarize(partial, beam_size, filtered=False),
                    "strict_post_filter": summarize(partial, beam_size, filtered=True),
                    "target_failure_stages": failure_stages(partial),
                }
            failed_jobs.append(failure)
            continue
        if not metrics_path.exists() or host is None:
            if allow_incomplete:
                continue
            raise ValueError(f"Missing completed monitored job: {folder.name}")
        metrics = read(metrics_path)
        if (
            metrics["status"] != "completed"
            or metrics["cohort_rows_sha256"] != EXPECTED_COHORT
            or metrics["encoder"] != encoder
            or metrics["mode"] != mode
            or metrics["beam_size"] != beam_size
            or metrics["start"] != start
            or metrics["count"] != 256
            or metrics["device"] != "cuda"
            or metrics["precision"] != "fp32"
            or metrics["attention_backend"] != "math"
            or metrics["training_started"]
        ):
            raise ValueError(f"Run contract mismatch: {folder.name}")
        if inputs is not None and metrics["inputs"] != inputs:
            raise ValueError("Checkpoint/provenance changed between chunks")
        inputs = metrics["inputs"]
        part = [
            json.loads(line)
            for line in (folder / "predictions_private.jsonl").read_text().splitlines()
        ]
        check_records(part, rows, start, 256)
        records.extend(part)
        jobs.append(
            {
                "name": folder.name,
                "host_monitor": host,
                "inference_seconds": metrics["elapsed_seconds"],
                "peak_VRAM_allocated_bytes": metrics["peak_VRAM_allocated_bytes"],
                "peak_RAM_kib": metrics["peak_RAM_kib"],
                "torch": metrics["torch"],
                "hip": metrics["hip"],
                "private_predictions_sha256": file_hash(folder / "predictions_private.jsonl"),
                "metrics_sha256": file_hash(folder / "metrics.json"),
            }
        )
    result = {
        "encoder": encoder,
        "mode": mode,
        "beam_size": beam_size,
        "status": "complete"
        if len(records) == 512
        else ("stopped" if failed_jobs else "incomplete" if records else "not_run"),
        "evaluated_targets": len(records),
        "required_targets": 512,
        "inputs": inputs,
        "jobs": jobs,
        "failed_jobs": failed_jobs,
    }
    if records:
        result.update(
            raw=summarize(records, beam_size, filtered=False),
            strict_post_filter=summarize(records, beam_size, filtered=True),
            target_failure_stages=failure_stages(records),
            raw_formula_error_profile_among_valid_Polar=formula_errors(records, mode),
            resources={
                "host_wall_seconds": sum(j["host_monitor"]["elapsed_seconds"] for j in jobs),
                "inference_seconds": sum(j["inference_seconds"] for j in jobs),
                "peak_VRAM_allocated_bytes": max(j["peak_VRAM_allocated_bytes"] for j in jobs),
                "peak_RAM_kib": max(j["peak_RAM_kib"] for j in jobs),
            },
            retrieval_mode_strata={
                str(mode_id): {
                    "raw": summarize(subset, beam_size, filtered=False),
                    "strict_post_filter": summarize(subset, beam_size, filtered=True),
                }
                for mode_id in sorted({r["mode"] for r in records})
                if (subset := [r for r in records if r["mode"] == mode_id])
            },
        )
    return result


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def markdown(result):
    lines = [
        "# v0.4 Polar-only 分子式约束推断实测",
        "",
        f"状态：**{result['status']}**。本轮只复用 3,000 步权重做推断，没有训练。",
        "固定 512 个不同的 validation Polar 头基；Morgan/FP 与 GINE 共用同一批目标。",
        "本实验仍使用 structure-oracle 检索与可信中性式、给定链，不能称为 MS/MS 鉴定准确率。",
        "",
        f"正式作业完成 {result['execution_summary']['completed_gpu_jobs']}/36，"
        f"中断 {result['execution_summary']['failed_gpu_jobs']} 个；"
        f"完整 512 目标搜索组合 {result['execution_summary']['complete_cases']}/18。",
        "Beam 5 的两模型四方法对照完整；Morgan Beam 20 仅普通 Beam 及其后验过滤完整。",
        "Morgan Beam 20 元素约束的 241 条中断记录仅作单独诊断；"
        "其他 Beam 20 与正式 Beam 50 未运行。",
        "",
        "方法、计数规则、限制与复现命令见 [实验方法](../experiments/POLAR_FORMULA_BEAM_V04.md)，",
        "精确计数、失败类别、候选数量分布、分层指标和哈希见 "
        "[聚合 JSON](POLAR_FORMULA_BEAM_V04_RESULTS.json)。",
        "",
        "## 原始输出与最终输出",
        "",
        "每个方法都先保存未修复的原始 beams。下表“原始合法”按其原始候选数计算；",
        "“普通 Beam”行保留原始 Decoder 排序和输出，其他行仅返回通过完整检查、"
        "按 canonical Polar graph 去重的候选。",
        "“覆盖”始终表示至少一个候选同时满足 Polar 元素式（含 H）、Builder 及完整中性式的目标比例。",
        "完整组合的 Top-k 以全部 512 目标为分母；不完整组合以表中实际目标数为分母。"
        "Beam 5 的 Top-10 不适用；无合格候选不移出分母。",
        "",
        "|Encoder|Beam|方法|目标数|原始合法|严格式覆盖|无合格候选|"
        "合法异构体均数|Polar Top-1|Top-5|Top-10|",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "raw": "普通 Beam",
        "elements": "重原子计数＋最终检查",
        "elements_grammar": "重原子＋基础语法＋最终检查",
    }
    for case in result["cases"]:
        if not case["evaluated_targets"]:
            continue
        methods = [(labels[case["mode"]], case["strict_post_filter"])]
        if case["mode"] == "raw":
            methods = [
                ("普通 Beam", case["raw"]),
                ("普通 Beam＋后验式过滤", case["strict_post_filter"]),
            ]
        for label, metric in methods:
            top = metric["polar_topk_exact_over_all_targets"]
            lines.append(
                f"|{case['encoder']}|{case['beam_size']}|{label}|{case['evaluated_targets']}|"
                f"{pct(metric['raw_chemical_valid_polar_fraction'])}|{pct(metric['strict_formula_coverage'])}|"
                f"{pct(metric['no_qualified_candidate_rate'])}|{metric['accepted_unique_per_target']['mean']:.3f}|"
                f"{pct(top['1'])}|{pct(top['5'])}|{pct(top['10'])}|"
            )
    lines += [
        "",
        "后验式过滤直接使用同一次普通 Beam 的候选，没有额外搜索。约束方法的原始输出 Top-k、",
        "最终过滤后的 Top-k 分开保存在 JSON 的 `raw` / `strict_post_filter` 中；"
        "表中约束方法展示最终输出。",
        "合法异构体均数包含无候选目标的 0；同一 canonical 图的不同 SMILES 不重复计数。",
        "",
        "## 原始输出的分母与失败",
        "",
        "约束可能让所有前缀死亡，返回少于 beam 宽度的候选，甚至空列表。"
        "因此候选合法率的分母会改变，不能单凭该比例判断改善。",
        "下表同时给出原始候选数及按全部目标计算的原始 Top-1 合法率。这些原始指标不受最终过滤影响。",
        "",
        "|Encoder|Beam|搜索|原始候选数|原始 Top-1 合法率|无存活 beam 目标|仅有未完成 beam 目标|",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        if not case["evaluated_targets"]:
            continue
        raw, failures = case["raw"], case["target_failure_stages"]
        lines.append(
            f"|{case['encoder']}|{case['beam_size']}|{case['mode']}|"
            f"{raw['raw_candidate_count']}|{pct(raw['raw_top1_chemical_valid_polar_rate'])}|"
            f"{failures.get('search_has_no_surviving_beam', 0)}|{failures.get('no_EOS', 0)}|"
        )
    lines += [
        "",
        "## 分子式正确仍可能结构错误",
        "",
        "下表错误率的分母为通过全部检查的唯一候选，Top-1 错误率的分母为有合格输出的目标。",
        "分子式约束不决定连接方式；生成了合法异构体并不等于找回目标结构。",
        "",
        "|Encoder|Beam|搜索|合格唯一候选数|式正确但 Polar 图错误|"
        "合格 Top-1 图错误|训练未见合法候选数|",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        if not case["evaluated_targets"]:
            continue
        metric = case["strict_post_filter"]
        lines.append(
            f"|{case['encoder']}|{case['beam_size']}|{case['mode']}|{metric['accepted_unique_candidates']}|"
            f"{pct(metric['formula_correct_wrong_polar_structure_fraction_among_accepted'])}|"
            f"{pct(metric['formula_correct_wrong_top1_fraction_among_formula_correct_top1'])}|"
            f"{metric['accepted_novel_to_train_candidates']}|"
        )
    lines += [
        "",
        "“训练未见”仅表示 canonical Polar graph 不在 TRAIN 标签集合中，不是全球化学新颖性声明。",
        "候选验收未使用该集合过滤，未要求头基必须存在于冻结数据库。",
        "",
        "## 资源与故障监控",
        "",
        "每个搜索/编码器/beam 组合分成两个 256 目标的串行 GPU 作业。每作业上限 1,200 秒，",
        "20 秒无进展停止；Windows watchdog dump 每 500 ms、System 4101 每 5 秒检查。",
        "进程结束及下一作业开始还会复查自本实验 epoch 以来的故障，防止遗漏迟到转储。",
        "",
        "|Encoder|Beam|搜索|推断＋验证秒|含加载宿主秒|峰值 PyTorch 分配显存 MiB|峰值进程 RAM MiB|",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        if "resources" not in case:
            continue
        res = case["resources"]
        lines.append(
            f"|{case['encoder']}|{case['beam_size']}|{case['mode']}|{res['inference_seconds']:.2f}|"
            f"{res['host_wall_seconds']:.2f}|{res['peak_VRAM_allocated_bytes'] / 2**20:.1f}|"
            f"{res['peak_RAM_kib'] / 1024:.1f}|"
        )
    lines += [
        "",
        "显存为 PyTorch allocator 的最大 allocated 值，不包含 Windows 显示占用或全部驱动开销。",
        "推断时间含模板编码、解码、RDKit/Builder 验证和逐目标落盘，"
        "不含模型/库的初始加载；宿主时间包含加载。",
        "普通 Beam 的后验过滤复用其解码，耗时不重复计入；JSON 另列 decode / validation 时间。",
        "",
        "## GPU 故障与停止",
        "",
        "2026-10-20 10:44（北京时间），Windows 新增 `WATCHDOG-20261020-1044.dmp`。",
        "匹配该转储的两条 WER 事件均为 `LiveKernelEvent 141`，用户确认发生黑屏、随后恢复。",
        "该码表示显示引擎未及时响应，见 [Microsoft 定义]"
        "(https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/"
        "bug-check-0x141---video-engine-timeout-detected)。",
        "监督器因新 dump 写入停止锁，终止推断；外层作业循环以失败退出，没有重试或后续 GPU 作业。",
        "停止处理后 WSL 子进程退出码为 15；stderr 为空，未观察到 PyTorch 数值异常记录。",
        "Windows System 4101 为 0 仍发生了真实故障，不能只靠该事件判断 GPU 是否正常。",
        "精确根因未定位；本次结果说明 Math Attention＋FP32 仍未消除这台主机的黑屏风险。",
        "",
        "|中断作业|已保存目标/计划|新 GPU dump|退出码|停止原因|",
        "|---|---:|---:|---:|---|",
    ]
    for case in result["cases"]:
        for failed in case["failed_jobs"]:
            host = failed["host_monitor"]
            lines.append(
                f"|{failed['name']}|{failed['partial_completed_targets']}/256|"
                f"{host['new_windows_gpu_dump_count']}|{host['child_exit_code']}|"
                f"{host['stop_reason']}|"
            )
    lines += [
        "",
        "中断作业的分母不是 512，因此未纳入主对照；JSON 单独保留其实际前缀目标的诊断聚合。",
        "此外保留两个各 2 目标的 GPU 预检（含 GINE Beam 50），不把预检当作正式比较。",
        "",
        "## 本轮结论",
        "",
    ]
    for encoder in ("fingerprint", "graph"):
        base = next(
            c
            for c in result["cases"]
            if (c["encoder"], c["mode"], c["beam_size"]) == (encoder, "raw", 5)
        )
        grammar = next(
            c
            for c in result["cases"]
            if (c["encoder"], c["mode"], c["beam_size"]) == (encoder, "elements_grammar", 5)
        )
        if base["status"] == grammar["status"] == "complete":
            a, b = base["strict_post_filter"], grammar["strict_post_filter"]
            lines.append(
                f"- {encoder} Beam 5：严格分子式覆盖率由 {pct(a['strict_formula_coverage'])} "
                f"增至 {pct(b['strict_formula_coverage'])}；Polar Top-5 exact 由 "
                f"{round(a['polar_topk_exact_over_all_targets']['5'] * 512)}/512 增至 "
                f"{round(b['polar_topk_exact_over_all_targets']['5'] * 512)}/512。"
            )
    lines += [
        "- 元素计数能提高公式候选覆盖，但单独使用会降低原始化学合法率；"
        "基础语法和最终含 H 验证仍必要。",
        "- 同式合法候选大部分仍是错误目标图，约束不能替代结构识别与排序；"
        "不能据覆盖提升宣称鉴定准确率高。",
        "- 不能根据未完成的实验判断 Beam 20/50 是否优于 Beam 5，或判断哪种 encoder 普遍更优。",
        "- 本轮没有启动 5,000 步、50k 池、全量训练或 MS/MS 微调；"
        "故障停止锁保留，后续 GPU 工作停止。",
        "",
        "## 保护与限制",
        "",
        f"GPU 前 CPU 测试：{result['cpu_gate']['tests_passed']} 项通过；"
        "两个 checkpoint 各审计 512 个真实标签的元素式及语法前缀。",
        f"最终回归：{result['final_validation'].get('pytest_passed', '待核验')} 项测试，"
        "另有 Ruff、PowerShell 语法与 UTC 时间转换回归检查。",
        f"旧实验保护审计：{result['preservation_audit'].get('status')}；"
        f"保护文件 {result['preservation_audit'].get('protected_files', '待核验')} 个，"
        f"checkpoints {result['preservation_audit'].get('protected_checkpoints', '待核验')} 个。",
        "继续使用 ROCm 7.2.1、Math Attention、FP32；Windows 显卡驱动及 Python 依赖未调整。",
        f"停止锁：`{result['stop_latch']}`。历史宿主黑屏记录保留；"
        "即使本实验未观察到新故障，也不能证明此前故障已解决。",
        "Beam 128 未运行；Beam 50 仅有 2 目标预检，没有完成固定 512 目标正式比较。",
        "私有结构、逐目标预测及 checkpoints 留在 WSL Linux 文件系统；"
        "GitHub 仅包含代码、配置和聚合结果。",
        "",
        f"原始基线 commit：`{BASE}`。检查点、标签、tokenizer、检索及样本哈希见聚合 JSON。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=EXPERIMENT)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    rows = np.load(SAMPLES / "diagnostic_512.npy")
    if len(rows) != 512 or row_hash(rows) != EXPECTED_COHORT:
        raise ValueError("Validation cohort changed")
    cases = [
        aggregate_case(args.root, encoder, mode, beam, rows, args.allow_incomplete)
        for beam in (5, 20, 50)
        for encoder in ("fingerprint", "graph")
        for mode in ("raw", "elements", "elements_grammar")
    ]
    complete = all(c["status"] == "complete" for c in cases)
    stop_path = args.root / "GPU_STOP.json"
    protection_path = args.root / "preservation_after.json"
    protection = read(protection_path) if protection_path.exists() else {"status": "pending"}
    validation_path = args.root / "validation_final.json"
    validation = read(validation_path) if validation_path.exists() else {"passed": False}
    roundtrip_path = args.root / "target_roundtrip_audit.json"
    roundtrip = read(roundtrip_path) if roundtrip_path.exists() else {"passed": False}
    if complete and not args.allow_incomplete:
        if (
            protection.get("status") != "passed"
            or not validation.get("passed")
            or not roundtrip.get("passed")
        ):
            raise ValueError("Final CPU regression and old-experiment preservation audit required")
    epoch = read(args.root / "gpu_monitor_epoch_private.json")
    completed_jobs = sum(len(c["jobs"]) for c in cases)
    failed_jobs = sum(len(c["failed_jobs"]) for c in cases)
    safety_path = args.root / "gpu_safety_final.json"
    safety = read(safety_path) if safety_path.exists() else None
    result = {
        "experiment": "v0.4-formula-constrained-beam-inference",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "stopped_gpu_fault"
        if safety and safety["status"] == "stopped_gpu_fault"
        else "complete"
        if complete and not stop_path.exists()
        else "incomplete",
        "base_commit": BASE,
        "training_started": False,
        "validation_heads": 512,
        "cohort_rows_sha256": EXPECTED_COHORT,
        "beam_sizes": [5, 20, 50],
        "beam_sizes_with_complete_two_encoder_comparison": [5],
        "formal_beam_50_targets_evaluated": 0,
        "optional_beam_128_run": False,
        "methods": ["original_beam", "same_beams_posterior_filter", "elements", "elements_grammar"],
        "cpu_gate": read(args.root / "cpu_gate.json"),
        "baseline_Beam5_regression": read(args.root / "baseline_regression_audit.json")
        if (args.root / "baseline_regression_audit.json").exists()
        else None,
        "stop_latch": read(stop_path) if stop_path.exists() else None,
        "gpu_safety": safety,
        "execution_summary": {
            "planned_gpu_jobs": 36,
            "completed_gpu_jobs": completed_jobs,
            "failed_gpu_jobs": failed_jobs,
            "not_run_gpu_jobs": 36 - completed_jobs - failed_jobs,
            "complete_cases": sum(c["status"] == "complete" for c in cases),
            "required_cases": 18,
            "warmup_gpu_jobs_not_in_comparison": 2,
        },
        "warmups_not_in_comparison": [
            {
                "run": folder.name,
                "metrics": read(folder / "metrics.json"),
                "host_monitor": read(folder / "host_monitor.json"),
            }
            for folder in sorted(args.root.glob("warmup_*"))
            if (folder / "metrics.json").exists()
        ],
        "environment": {
            "gpu": "AMD Radeon RX 9070 XT (required by every GPU runner)",
            "windows_driver": [d for d in epoch["driver"] if "9070 XT" in d["Name"]],
            "monitor_start_utc": epoch["utc"],
            "attention_backend": "math",
            "precision": "fp32",
            "interpreter": "/home/administrator/miniconda3/envs/torch-rocm721/bin/python",
            "dependencies_changed": False,
            "driver_changed": False,
        },
        "inference_source_sha256": {
            str(path): file_hash(path)
            for path in (
                Path("src/structure_pretrain/polar_only_v04/constrained_decode.py"),
                Path("src/structure_pretrain/polar_only_v04/formula_beam_eval.py"),
                Path("scripts/supervise_formula_beam_v04.ps1"),
                Path("configs/polar_formula_beam_v04.yaml"),
            )
        },
        "executed_inference_source_sha256": read(
            args.root / "executed_inference_source_sha256.json"
        )
        if (args.root / "executed_inference_source_sha256.json").exists()
        else None,
        "post_stop_supervisor_change": (
            "System 4101 query now converts UTC epoch to local time and explicitly rechecks UTC; "
            "decoder/evaluation modules unchanged"
        ),
        "cases": cases,
        "preservation_audit": protection,
        "final_validation": validation,
        "target_roundtrip_audit": roundtrip,
        "candidate_acceptance_audit": read(args.root / "candidate_acceptance_audit.json")
        if (args.root / "candidate_acceptance_audit.json").exists()
        else None,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    REPORT.write_text(markdown(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "complete_cases": sum(c["status"] == "complete" for c in cases),
                "required_cases": 18,
            }
        )
    )


if __name__ == "__main__":
    main()
