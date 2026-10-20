"""Aggregate native CUDA jobs, keeping bounded pilots separate from full 512-head results."""

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04.formula_beam_eval import EXPECTED_COHORT, summarize
from structure_pretrain.polar_only_v04.prepare import SAMPLES, row_hash

spec = importlib.util.spec_from_file_location(
    "formula_report_shared", Path(__file__).with_name("report_formula_beam_v04.py")
)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
DEFAULT_ROOT = Path("outputs/formula_constrained_v04_rtx4070_20261020_r1")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def aggregate_jobs(root, rows, prior_root=None):
    cases, preflights = {}, []
    audited = {}
    folders = list(root.iterdir())
    if prior_root is not None:
        audit = read(root / "prior_native_validation_audit.json")
        if (
            not audit["passed"]
            or audit["classification_changes"]
            or audit["prior_gate_sha256"] != file_hash(prior_root / "cpu_gate.json")
            or audit["new_gate_sha256"] != file_hash(root / "cpu_gate.json")
        ):
            raise ValueError("Prior native results require a successful immutable revision audit")
        audited = {job["name"]: job for job in audit["jobs"]}
        folders += list(prior_root.iterdir())
    for folder in sorted(folders):
        if not folder.is_dir() or not (folder / "host_monitor.json").exists():
            continue
        prior = prior_root is not None and folder.parent.resolve() == prior_root.resolve()
        host = read(folder / "host_monitor.json")
        if host.get("gpu_backend") != "native_cuda":
            raise ValueError("Cannot mix AMD and native CUDA jobs")
        if prior and host["child_exit_code"] == 0 and not host["stop_reason"]:
            job = audited[folder.name]
            for filename, key in (
                ("metrics.json", "metrics_sha256"),
                ("host_monitor.json", "host_monitor_sha256"),
                ("predictions_private.jsonl", "predictions_sha256"),
            ):
                if file_hash(folder / filename) != job[key]:
                    raise ValueError("A prior job changed after CPU revalidation")
        if folder.name.startswith("preflight_"):
            preflights.append(
                {"name": folder.name, "origin_experiment": folder.parent.name, "host_monitor": host}
            )
            continue
        key = host["encoder"], host["mode"], host["beam_size"]
        case = cases.setdefault(key, {"records": [], "jobs": [], "failed_jobs": [], "inputs": None})
        safe = (
            host["child_exit_code"] == 0
            and not host["stop_reason"]
            and host["new_windows_gpu_dump_count"] == host["display_reset_count"] == 0
            and host["checkpoints_unchanged"]
            and not host["training_started"]
        )
        if not safe:
            partial = folder / "predictions_private.jsonl"
            case["failed_jobs"].append(
                {
                    "name": folder.name,
                    "origin_experiment": folder.parent.name,
                    "host_monitor": host,
                    "included_in_comparison": False,
                    "partial_targets_excluded": len(partial.read_text().splitlines())
                    if partial.exists()
                    else 0,
                }
            )
            continue
        metric = read(folder / "metrics.json")
        if any(metric[k] != host[k] for k in ("encoder", "mode", "beam_size", "start", "count")):
            raise ValueError("Host/decoder run contract mismatch")
        if (
            metric["status"] != "completed"
            or metric["cohort_rows_sha256"] != EXPECTED_COHORT
            or metric["gpu_backend"] != "native_cuda"
            or metric["hip"]
            or metric["precision"] != "fp32"
            or metric["attention_backend"] != "math"
            or metric["training_started"]
        ):
            raise ValueError("Native CUDA run contract mismatch")
        if case["inputs"] is not None and metric["inputs"] != case["inputs"]:
            raise ValueError("Checkpoint or frozen inputs differ between jobs")
        case["inputs"] = metric["inputs"]
        records = [
            json.loads(line)
            for line in (folder / "predictions_private.jsonl").read_text().splitlines()
        ]
        shared.check_records(records, rows, metric["start"], metric["count"])
        case["records"].extend(records)
        case["jobs"].append(
            {
                "name": folder.name,
                "origin_experiment": folder.parent.name,
                "canonical_reparse_guard_revalidated_on_CPU": prior,
                "host_monitor": host,
                "metrics_sha256": file_hash(folder / "metrics.json"),
                "private_predictions_sha256": file_hash(folder / "predictions_private.jsonl"),
                "inference_seconds": metric["elapsed_seconds"],
                "peak_RAM_kib": metric["peak_RAM_kib"],
                "peak_VRAM_allocated_bytes": metric["peak_VRAM_allocated_bytes"],
                "torch": metric["torch"],
                "cuda": metric["cuda"],
                "gpu_name": metric["gpu_name"],
            }
        )
    results = []
    for beam in (5, 20, 50):
        for encoder in ("fingerprint", "graph"):
            for mode in ("raw", "elements", "elements_grammar"):
                case = cases.get((encoder, mode, beam), {})
                records = sorted(case.get("records", []), key=lambda r: r["cohort_index"])
                indices = [r["cohort_index"] for r in records]
                if len(set(indices)) != len(indices):
                    raise ValueError("Overlapping job slices would double-count validation heads")
                complete = indices == list(range(512))
                result = {
                    "encoder": encoder,
                    "mode": mode,
                    "beam_size": beam,
                    "status": "complete_512"
                    if complete
                    else "stopped"
                    if case.get("failed_jobs")
                    else "bounded_subset"
                    if records
                    else "not_run",
                    "evaluated_targets": len(records),
                    "required_targets": 512,
                    "cohort_indices": indices,
                    "inputs": case.get("inputs"),
                    "jobs": case.get("jobs", []),
                    "failed_jobs": case.get("failed_jobs", []),
                }
                if records:
                    result.update(
                        raw=summarize(records, beam, filtered=False),
                        strict_post_filter=summarize(records, beam, filtered=True),
                        target_failure_stages=shared.failure_stages(records),
                        raw_formula_error_profile_among_valid_Polar=shared.formula_errors(
                            records, mode
                        ),
                    )
                results.append(result)
    return results, preflights


def markdown(result):
    pct = shared.pct
    lines = [
        "# v0.4 Polar-only：RTX 4070 SUPER 原生 Windows CUDA 接续验证",
        "",
        f"状态：**{result['status']}**。复用 Morgan/GINE 3,000 步权重，只做推断。",
        "所有目标来自冻结的 diagnostic_512；子集实验按实际目标数计算，不能作为完整 512 目标结论。",
        "旧 AMD 输出与停止锁保留，未拼接进 NVIDIA 结果。没有重新训练、修改标签或重建检索缓存。",
        "保留内部 [*:1]、Math Attention、FP32；最终使用 RDKit/Builder "
        "严格检查含 H 的 Polar 式和完整中性式。",
        "使用 structure-oracle 检索、可信式和给定链；这些指标不是 MS/MS 鉴定准确率。",
        "",
        f"CPU 回归 {result['cpu_gate']['tests_passed']} 项通过；"
        "两份真实 checkpoint 各审计 512 个标签；",
        "512 个目标的完整分子式（含 H）及完整图重建全部通过。",
        "",
        "|Encoder|Beam|方法|目标数|原始候选合法率|严格式覆盖|无合格候选|合法异构体均数|Top-1|Top-5|Top-10|",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        if not case["evaluated_targets"]:
            continue
        methods = [(case["mode"], case["strict_post_filter"])]
        if case["mode"] == "raw":
            methods = [
                ("原始 Beam", case["raw"]),
                ("同批候选＋后验式过滤", case["strict_post_filter"]),
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
        "原始搜索输出与严格检查后的输出在 JSON 的 raw / strict_post_filter 中分别保存。",
        "约束模式表格展示严格检查后的排序；普通 Beam 行展示未经后验过滤的原始排序。",
        "没有合格候选的目标保留在分母中；Beam 5 的 Top-10 不适用。",
        "",
        "|Encoder|Beam|搜索|合格候选数|式正确但图错误比例|推断与验证秒|峰值显存 MiB|峰值 RAM MiB|",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        if not case["evaluated_targets"]:
            continue
        metric, jobs = case["strict_post_filter"], case["jobs"]
        lines.append(
            f"|{case['encoder']}|{case['beam_size']}|{case['mode']}|{metric['accepted_unique_candidates']}|"
            f"{pct(metric['formula_correct_wrong_polar_structure_fraction_among_accepted'])}|"
            f"{sum(j['inference_seconds'] for j in jobs):.2f}|"
            f"{max(j['peak_VRAM_allocated_bytes'] for j in jobs) / 2**20:.1f}|"
            f"{max(j['peak_RAM_kib'] for j in jobs) / 1024:.1f}|"
        )
    lines += [
        "",
        "资源数值仅对应本机本次实际完成的目标，不与 AMD 中断前缀合并。",
        "显存为 PyTorch allocator 峰值，不含 Windows 显示及全部驱动开销。",
        "推断与验证耗时不含初始加载；各作业宿主总耗时、故障事件、遥测及权重保护记录见 JSON。",
        "",
        "监控持续查询 System 4101 和 Application WER LiveKernelEvent 141/117，并记录 nvidia-smi。",
        "当前非管理员进程不能读取受保护的 WATCHDOG 目录；该可见性限制逐作业明确记录，",
        "因此没有将“未观察到事件”解释为“绝无故障转储”。可读目录仍检查新增或变更的 dump。",
        "无进展、总时限、监控异常、CUDA 子进程失败或新 GPU 故障均写入独立停止锁，禁止自动重试。",
        "",
        "## 本次结论",
        "",
        *result["conclusions"],
        "",
        "私有结构、逐目标预测、冻结数据和权重留在 G 盘；GitHub 仅发布代码、配置及聚合指标。",
        "",
    ]
    comparisons = result["comparison_to_prior_AMD_Beam5"]
    if comparisons:
        same = sum(c["reported_quality_metrics_match"] for c in comparisons)
        lines += [
            "## Beam 5 回归参考",
            "",
            f"本机完成的 {len(comparisons)} 组完整 Beam 5 中，{same} 组的候选数量、"
            "合法率、严格式覆盖、合格唯一候选数和 Top-k exact 与旧 AMD 报告完全一致。",
            "逐项差异与输入哈希比较见 JSON 的 comparison_to_prior_AMD_Beam5。",
            "此处只比较聚合指标，没有合并旧样本、候选或计时。",
            "",
        ]
    repair = result.get("canonical_validation_repair_audit")
    if repair:
        lines += [
            "## 候选验证异常与修订",
            "",
            "首次运行在 Morgan Beam 50 元素＋语法分块遇到 RDKit canonical Polar "
            "重新解析异常，进程退出并触发停止锁。没有观察到 CUDA 错误或新的 GPU 故障事件。",
            "同一目标在禁用 GPU 的 CPU 路径重现；增加 canonical Polar 重新解析检查后，"
            "不合格候选被拒绝，其余候选继续处理，不修复结构或修改公式。",
            f"修订前已完成的 {len(repair['jobs'])} 个作业（含预检）共 "
            f"{repair['candidates_checked']:,} 个候选，经修订后的检查重审，判定及 canonical "
            "graph 均无变化；因此复用这些完整分块。",
            f"中断的 {repair['failed_partial_targets_excluded']} 条前缀不进入比较；"
            "失败分块在新版本、新目录及新 CPU gate 下完整重跑。",
            "原停止锁、原始日志和执行源码保留；新监控 epoch 延续原 UTC 起点，覆盖修复间隙。",
            "",
        ]
    return "\n".join(lines)


def conclusions(cases, complete):
    lines = [
        "- 原生 Windows CUDA 已实际加载两份原始 v0.4 权重，并完成有监控的推断；没有重新训练。",
    ]
    for encoder in ("fingerprint", "graph"):
        subset = [
            c
            for c in cases
            if c["encoder"] == encoder
            and c["mode"] == "elements_grammar"
            and c["status"] == "complete_512"
        ]
        if len(subset) == 3:
            metrics = [c["strict_post_filter"] for c in subset]
            coverage = "/".join(shared.pct(m["strict_formula_coverage"]) for m in metrics)
            top5 = "/".join(
                str(round(m["polar_topk_exact_over_all_targets"]["5"] * 512)) for m in metrics
            )
            lines.append(
                f"- {encoder}：重原子＋语法的 Beam 5/20/50 覆盖率分别为 {coverage}；"
                f"Top-5 exact 分别为 {top5} / 512。"
            )
    lines.append("- 严格式覆盖不等于目标结构正确率；应同时检查式正确但图错误的比例和 Top-k exact。")
    if not complete:
        lines.append("- 尚未完成的 512 目标组合不能用于判断 Beam 20/50 的总体优势。")
    lines.append("- 没有启动 5,000 步、50k 池、全量训练或 MS/MS 微调；Beam 128 未运行。")
    return lines


def compare_prior_beam5(cases):
    prior = read(Path("docs/results/POLAR_FORMULA_BEAM_V04_RESULTS.json"))
    fields = (
        "raw_candidate_count",
        "raw_chemical_valid_polar_fraction",
        "strict_formula_coverage",
        "accepted_unique_candidates",
        "polar_topk_exact_over_all_targets",
    )
    comparisons = []
    for case in cases:
        if case["beam_size"] != 5 or case["status"] != "complete_512":
            continue
        old = next(
            c
            for c in prior["cases"]
            if c["beam_size"] == 5 and c["encoder"] == case["encoder"] and c["mode"] == case["mode"]
        )
        differences = {
            f"{view}.{key}": {"AMD": old[view][key], "native_CUDA": case[view][key]}
            for view in ("raw", "strict_post_filter")
            for key in fields
            if old[view][key] != case[view][key]
        }
        comparisons.append(
            {
                "encoder": case["encoder"],
                "mode": case["mode"],
                "same_checkpoint_and_input_hashes": old["inputs"] == case["inputs"],
                "reported_quality_metrics_match": not differences,
                "differences": differences,
            }
        )
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prior-root", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/results/POLAR_FORMULA_BEAM_CUDA_V04_RESULTS.json")
    )
    args = parser.parse_args()
    rows = np.load(SAMPLES / "diagnostic_512.npy")
    if len(rows) != 512 or row_hash(rows) != EXPECTED_COHORT:
        raise ValueError("Frozen cohort changed")
    revision_path = args.root / "prior_native_validation_audit.json"
    revision = read(revision_path) if revision_path.exists() else None
    prior_root = args.prior_root or (Path(revision["prior_root"]) if revision else None)
    cases, preflights = aggregate_jobs(args.root, rows, prior_root)
    complete = all(c["status"] == "complete_512" for c in cases)
    stopped = (args.root / "GPU_STOP.json").exists()
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "stopped_native_cuda_validation"
        if stopped
        else "complete_512_matrix"
        if complete
        else "bounded_native_cuda_validation",
        "required_full_matrix": "2 encoders x 3 searches x 3 beams x 512 targets",
        "cohort_rows_sha256": EXPECTED_COHORT,
        "cases": cases,
        "execution_summary": {
            "completed_formal_jobs": sum(len(c["jobs"]) for c in cases),
            "failed_formal_jobs": sum(len(c["failed_jobs"]) for c in cases),
            "complete_512_cases": sum(c["status"] == "complete_512" for c in cases),
            "required_cases": 18,
            "preflight_jobs": len(preflights),
            "completed_target_evaluations": sum(c["evaluated_targets"] for c in cases),
        },
        "comparison_to_prior_AMD_Beam5": compare_prior_beam5(cases),
        "preflight_jobs": preflights,
        "cpu_gate": read(args.root / "cpu_gate.json"),
        "final_validation": read(args.root / "validation_final.json")
        if (args.root / "validation_final.json").exists()
        else None,
        "canonical_validation_repair_audit": revision,
        "environment": read(args.root / "environment.json")
        if (args.root / "environment.json").exists()
        else None,
        "training_started": False,
        "stop_latch": read(args.root / "GPU_STOP.json")
        if (args.root / "GPU_STOP.json").exists()
        else None,
        "conclusions": conclusions(cases, complete),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_cases": sum(c["status"] == "complete_512" for c in cases),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
