"""Aggregate-only status/final results and paired comparisons; never starts a GPU job."""

# ruff: noqa: E501 -- Complete Chinese report paragraphs/table strings.

import gzip
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import binomtest

from structure_pretrain.core_ablation.common import ROOT, TEMPLATE_ROOT, configuration, read

DEST = Path("docs/results/CORE_FORMULA_COMPARISON_20261027")


def paired_test(left, right):
    if not left.exists() or not right.exists():
        return None

    formulas = np.load(TEMPLATE_ROOT / "data/formulas.npy", mmap_mode="r")

    def load(path):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return {
                r["target_id"]: {
                    "P": r["contains_P"],
                    "raw": r["raw"],
                    "posterior_filtered": r["posterior_filtered"],
                    "formula_group": tuple(int(x) for x in formulas[r["row_index"]]),
                }
                for r in (json.loads(line) for line in stream)
            }

    a, b = load(left), load(right)
    assert a.keys() == b.keys() and len(a) == 9499
    result = {}
    for name, condition in (("all", None), ("P", True), ("nonP", False)):
        keys = [k for k in a if condition is None or a[k]["P"] == condition]
        for k in keys:
            assert a[k]["formula_group"] == b[k]["formula_group"]
        groups, inverse = np.unique(
            np.asarray([a[k]["formula_group"] for k in keys]), axis=0, return_inverse=True
        )
        sizes = np.bincount(inverse)
        part = {"targets": len(keys), "formula_condition_groups": len(groups)}
        for mode in ("raw", "posterior_filtered"):
            values = np.asarray(
                [
                    [
                        a[k][mode]["exact_at_10"],
                        b[k][mode]["exact_at_10"],
                        a[k][mode]["Similarity_at_10"],
                        b[k][mode]["Similarity_at_10"],
                    ]
                    for k in keys
                ],
                dtype=float,
            )
            win = int(np.sum((values[:, 0] == 1) & (values[:, 1] == 0)))
            loss = int(np.sum((values[:, 0] == 0) & (values[:, 1] == 1)))
            rng = np.random.default_rng(20261027)
            delta = values[:, [0, 2]] - values[:, [1, 3]]
            # The Formula-only output is identical for a repeated formula. Resample
            # formula clusters together rather than pretending these are independent
            # condition queries. Keep the target-weighted point estimate.
            sums = np.column_stack([np.bincount(inverse, weights=delta[:, j]) for j in (0, 1)])
            samples = []
            for _ in range(1024):
                chosen = rng.integers(0, len(groups), size=len(groups))
                samples.append(sums[chosen].sum(0) / sizes[chosen].sum())
            intervals = np.quantile(samples, [0.025, 0.975], axis=0)
            part[mode] = {
                "template_minus_formula_exact10": float(delta[:, 0].mean()),
                "template_minus_formula_Similarity10": float(delta[:, 1].mean()),
                "exact10_paired_95CI": intervals[:, 0].tolist(),
                "Similarity10_paired_95CI": intervals[:, 1].tolist(),
                "bootstrap_unit": "target-formula cluster; target-weighted paired difference",
                "template_only_exact10_hits": win,
                "formula_only_exact10_hits": loss,
                "McNemar_exact_two_sided_p": float(binomtest(win, win + loss).pvalue)
                if win + loss
                else 1.0,
            }
        result[name] = part
    return result


def live_progress(state):
    name = state.get("stage", "")
    path = ROOT / name / "events.jsonl"
    latest = None
    if path.exists():
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            for line in reversed(stream.read().splitlines()):
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("stage") in (
                    "full_test_batch_completed",
                    "train_progress",
                    "validation_completed",
                ):
                    latest = row
                    break
    status_path = ROOT / (name + ".supervisor.json")
    supervisor = read(status_path) if status_path.exists() else {}
    # Use the same ordinary Windows process-liveness check as the published
    # launch reporter; importing that module does not run its main().
    from report_core_templates8000 import alive

    active = (
        supervisor.get("status") == "running"
        and alive(supervisor.get("child_pid"))
        and alive(supervisor.get("supervisor_pid"))
    )
    return {
        "live_supervised_child_verified": bool(active),
        "latest_record": latest,
        "latest_telemetry": supervisor.get("latest_telemetry"),
    }


def atomic_text(path, text):
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def main():
    c = configuration()
    state = read(ROOT / "state.json") if (ROOT / "state.json").exists() else {"status": "preparing"}
    if state["status"] == "completed" and (ROOT / "PUBLICATION_DONE.json").exists():
        print(
            {
                "status": "completed_and_published",
                "publication": read(ROOT / "PUBLICATION_DONE.json"),
            }
        )
        return
    reports = {}
    for model in ("template", "formula_only"):
        reports[model] = {}
        for mode in c["decoding_modes"]:
            p = ROOT / f"{model}_full_{mode}/evaluation_report.json"
            reports[model][mode] = read(p) if p.exists() else None
    training_path = ROOT / "formula_only_train_seed42/training_report.json"
    training = read(training_path) if training_path.exists() else None
    history_path = ROOT / "formula_only_train_seed42/validation_history.json"
    history = read(history_path) if history_path.exists() else None
    best_path = ROOT / "formula_only_train_seed42/best.json"
    best = read(best_path) if best_path.exists() else None
    paired = {}
    for mode in c["decoding_modes"]:
        if reports["template"][mode] is not None and reports["formula_only"][mode] is not None:
            paired[mode] = paired_test(
                ROOT / f"template_full_{mode}/raw_candidates_private.jsonl.gz",
                ROOT / f"formula_only_full_{mode}/raw_candidates_private.jsonl.gz",
            )
    result = {
        "generated_Asia_Shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "state": state,
        "current_progress": live_progress(state),
        "configuration": c,
        "evaluations": reports,
        "formula_only_training": training,
        "formula_only_full_validation_history": history,
        "formula_only_best_checkpoint": {k: v for k, v in best.items() if k != "path"}
        if best
        else None,
        "paired_template_vs_formula": paired,
        "primary_comparison": "elements_grammar, posterior_filtered Exact@10 on all9499",
        "paired_CI_scope": "paired target-formula-cluster bootstrap, one fixed training seed; not training-seed variability",
        "template_inputs_are_structure_oracle": True,
        "no_independent_real_MSMS_claim": True,
        "raw_predictions_and_weights_remain_private_on_G": True,
        "template_epoch43_weight_sha256": c["template_checkpoint_sha256"],
    }
    control_run = ROOT / "formula_only_train_seed42/run_contract.json"
    baseline_run = TEMPLATE_ROOT / "train_main_k3_seed42/run_contract.json"
    result["parameter_counts"] = {
        "template": read(baseline_run)["parameter_count"],
        "formula_only": read(control_run)["parameter_count"] if control_run.exists() else None,
    }
    if result["parameter_counts"]["formula_only"] is not None:
        assert (
            result["parameter_counts"]["template"]
            == result["parameter_counts"]["formula_only"]
            == 16479940
        )
    conclusion = "Formula-only的完整约束评价尚未完成，暂不能判断模板增益。"
    if "elements_grammar" in paired:
        primary = paired["elements_grammar"]["all"]["posterior_filtered"]
        delta = primary["template_minus_formula_exact10"]
        lo, hi = primary["exact10_paired_95CI"]
        if delta > 0 and lo > 0:
            conclusion = (
                "在本次固定数据/seed及oracle结构检索条件下，模板提高了主要指标Exact@10，"
                "且公式分组配对区间下界大于0；不外推为真实MS/MS性能。"
            )
        elif delta < 0 and hi < 0:
            conclusion = "本次主要指标不支持模板提升：Formula-only的Exact@10更高，且配对区间不含0。"
        else:
            conclusion = "本次主要指标的公式分组配对区间包含0，未获得明确的模板Exact@10提升证据。"
    result["predeclared_primary_conclusion"] = conclusion
    # Include actual numerical/gradient/recovery/grammar proofs without structures.
    for gate_name in ("cpu_gate", "cuda_gate"):
        p = ROOT / (gate_name + ".json")
        if p.exists():
            value = read(p)
            result[gate_name] = {
                "passed": value["passed"],
                "checks": value["checks"],
                "inference_checks": value.get("inference_checks"),
            }
    lines = [
        "# 全部9,499测试目标：分子式约束解码与Formula-only对照",
        "",
        f"更新时间：{result['generated_Asia_Shanghai']}；当前阶段：`{state.get('stage', 'preparing')}`，状态：`{state['status']}`。",
        "",
        "此前512结果只有分子式条件输入与后验过滤，没有逐token约束。这里将原始、元素约束、元素＋语法约束严格分开。",
        "固定第43轮最佳模板模型（SHA见JSON）、全部9,499测试目标（含P8,225 / 非P1,274）、Beam20、最大长度96。",
        "约束在原log_softmax之后屏蔽不允许的token，不对可行集合重新归一化；不强行EOS、不修复/改写分子图或分子式。",
        "生成中逐项核对C/N/O/P/S/F/Cl/Br/I及唯一101/102端口，未消耗完禁止EOS；语法模式检查括号/环/键/O连接端口。",
        "H不按SMILES字符计数；最终严格核对H封端核心式、RDKit、固定甘油/端口角色及参考链组装。",
        "",
        "Formula-only独立随机初始化，使用完全相同TRAIN/validation/test、词表、参数张量形状与总数、优化器/学习率、batch32、最大80轮/427,200步和patience5。",
        "其条件只来自目标分子式，经固定维度展开进入原网络；不打开/传入参考指纹、参考式、差值、检索排名或掩码。",
        "参数形状相同不等于输入信息量相同；这里有意移除结构信息。三份公式token用于匹配memory规模，并非虚构参考结构。",
        "公式统计仅由TRAIN目标式计算；最佳权重只按全部9,493验证目标的Token CE选择。实际早停步数另报，未声称实际训练更新次数必然相同。",
        "",
        "| 模型 | 解码 | 评价目标数 | 原始Exact@1/5/10 | 原始Similarity@10 | 后验过滤Exact@10 | 核心有效/固定Beam预算 | 式正确/固定Beam预算 | 至少一合格式候选 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    progress = result["current_progress"]
    row = progress["latest_record"]
    lines.insert(4, f"监督器及子进程存活核验：{progress['live_supervised_child_verified']}。")
    if row:
        if row["stage"] == "full_test_batch_completed":
            lines.insert(
                5,
                f"当前队列已完成 {row['completed']:,}/{row['total']:,} 个目标；未完成队列不填最终指标。",
            )
        elif row["stage"] == "train_progress":
            lines.insert(
                5, f"Formula-only当前第{row['epoch']}轮、step {row['step']:,}；尚非最终结果。"
            )

    def pct(x):
        return f"{100 * x:.2f}%"

    for model in reports:
        for mode, r in reports[model].items():
            if r is None:
                lines.append(f"| {model} | {mode} | 未完成 | — | — | — | — | — | — |")
                continue
            a, p, budget = r["raw"], r["posterior_filtered"], r["budget"]
            lines.append(
                f"| {model} | {mode} | {r['targets']:,} | {pct(a['exact_at_1'])}/{pct(a['exact_at_5'])}/{pct(a['exact_at_10'])} | {a['Similarity_at_10']:.4f} | {pct(p['exact_at_10'])} | {pct(budget['core_valid_fraction_of_Beam20_budget'])} | {pct(budget['full_H_formula_correct_fraction_of_Beam20_budget'])} | {pct(budget['targets_with_formula_correct_candidate'])} |"
            )
    lines += [
        "",
        "所有失败目标均在分母；固定Beam预算的比例以9,499×20为分母，空槽记失败。同时JSON另报实际输出候选分母、RDKit解析率及式正确@1/5/10。",
        "所有模式均另外保留后验过滤结果；含P/非P的全部指标分别在JSON报告。",
        "",
    ]
    if training:
        lines += [
            f"Formula-only已完成{training['completed_epochs']}轮/{training['steps']:,}步，最佳第{training['best_epoch']}轮，验证CE={training['best_validation_token_CE']:.10f}。",
            "",
        ]
    else:
        lines += [
            "Formula-only最终训练与对照指标尚未完成，JSON对应位置为null，不能提前判断模板是否提升。",
            "",
        ]
    lines += [
        "| 模型/解码 | 组别 | 原始Exact@10 | 过滤Exact@10 | 过滤Similarity@10 | 过滤相对共同模板基线平均收益 | 正收益目标比例 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in reports:
        for mode, r in reports[model].items():
            if r:
                for group in ("P", "nonP"):
                    sub, a = r[group], r[group]["posterior_filtered"]
                    lines.append(
                        f"| {model}/{mode} | {group} (n={sub['targets']:,}) | {pct(sub['raw']['exact_at_10'])} | {pct(a['exact_at_10'])} | {a['Similarity_at_10']:.4f} | {a['gain_over_best_input_template']:.4f} | {pct(a['beats_best_input_template'])} |"
                    )
    if paired:
        lines += [
            "",
            "配对比较（模板模型−Formula-only；同一测试目标）：",
            "",
            "| 解码/输出 | 组别 | Exact@10差值 | 目标配对95%CI | Similarity@10差值 |",
            "|---|---|---:|---|---:|",
        ]
        for mode, pair in paired.items():
            for group, value in pair.items():
                for output in ("raw", "posterior_filtered"):
                    r = value[output]
                    lo, hi = r["exact10_paired_95CI"]
                    lines.append(
                        f"| {mode}/{output} | {group} | {pct(r['template_minus_formula_exact10'])} | [{pct(lo)}, {pct(hi)}] | {r['template_minus_formula_Similarity10']:.4f} |"
                    )
    lines += ["", conclusion, ""]
    lines += [
        "",
        "Formula-only的收益也使用同一固定最佳结构模板作为共同评估基线；该基线只在解码完成后用于评分，不输入Formula-only。",
        "结构模板模型使用目标图检索参考，因此本对照检验的是oracle结构条件带来的增益，不等于真实MS/MS检索提升。",
        "配对置信区间按目标分子式分组重采样，保留相同公式查询的相关性；仅反映本划分/训练seed，不代表独立训练seed间方差。原始/约束/后验过滤不混合，TRAIN模板目标不混入测试。",
        "GPU作业由独立Windows监督器监控System4101/WER141/117、可读dump和NVIDIA遥测；WATCHDOG权限限制继续保留。",
        "完整结构、全部逐目标候选和权重仅在G盘；GitHub只发布聚合报告、配置与代码。",
        "",
    ]
    DEST.parent.mkdir(exist_ok=True)
    atomic_text(DEST.with_suffix(".json"), json.dumps(result, ensure_ascii=False, indent=2))
    atomic_text(DEST.with_suffix(".md"), "\n".join(lines))
    print(
        json.dumps(
            {
                "report": str(DEST),
                "status": state["status"],
                "completed_evaluations": sum(
                    r is not None for x in reports.values() for r in x.values()
                ),
                "formula_only_training_completed": training is not None,
            }
        ),
        flush=True,
    )
    if state["status"] == "completed":
        # Publication was authorized for completed experiments. The separate CPU
        # publisher checks the full saved outputs and an empty Git index; it never
        # stages private outputs or starts/retries any GPU job.
        result = subprocess.run([sys.executable, "scripts/publish_formula_comparison_results.py"])
        if result.returncode:
            print(
                "Final numerical results are complete; GitHub publication remains pending",
                flush=True,
            )


if __name__ == "__main__":
    main()
