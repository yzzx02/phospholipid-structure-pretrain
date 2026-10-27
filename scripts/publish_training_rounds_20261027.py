"""CPU-only final reports: verify saved histories, weights and raw prediction aggregates.

Never starts/resumes a GPU job or edits an experiment root. Private structures,
weights and raw candidate streams stay on G. Only aggregate reports/plots are written.
"""

# ruff: noqa: E501 -- Preserve complete Chinese Markdown paragraphs and table rows.

import gzip
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from structure_pretrain.glycerol_core_collection.chemistry import capped_core

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OLD = Path("outputs/template_guided_v2_20261023")
NEW = Path("outputs/glycerol_core_templates8000_v1_20261026")
PUBLIC = Path("docs/results")
FIGURES = PUBLIC / "figures/training_rounds_20261027"
OLD_NAME = "TEMPLATE_GUIDED_V2_FINAL_20261027"
NEW_NAME = "GLYCEROL_CORE_TEMPLATES8000_FINAL_20261027"
INDEX_NAME = "TRAINING_ROUNDS_FINAL_20261027"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )


def close(actual, expected):
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"Aggregate mismatch: {actual} != {expected}")


def monitor_summary(path):
    data = read(path)
    keys = (
        "child_exit_code",
        "stop_reason",
        "elapsed_seconds",
        "display_reset_count",
        "new_GPU_dump_count",
        "protected_checkpoints_unchanged",
        "all_exploratory_weights_inventory_unchanged",
        "unreadable_dump_directories",
    )
    return {k: data[k] for k in keys}


def training_result(root, name, validation_count):
    run = root / name
    result = read(run / "training_report.json")
    history = read(run / "validation_history.json")
    best = read(run / "best.json")
    settings = read(run / "run_contract.json")
    assert result["status"] == "completed" and result["test_used_for_selection"] is False
    assert [r["epoch"] for r in history] == list(range(1, result["completed_epochs"] + 1))
    assert all(r["validation_targets"] == validation_count for r in history)
    minimum = min(history, key=lambda r: r["validation_token_CE"])
    close(minimum["validation_token_CE"], result["best_validation_token_CE"])
    close(minimum["validation_token_CE"], best["best_validation_loss"])
    assert minimum["epoch"] == result["best_epoch"] == best["epoch"]
    assert minimum["step"] == best["step"] and history[-1]["step"] == result["steps"]
    assert digest(best["path"]) == best["sha256"]
    assert result["stop_reason"] == "patience5_complete_validation_epochs"
    assert result["completed_epochs"] - result["best_epoch"] == result["bad_epochs"] == 5
    for row in history[-5:]:
        assert row["validation_token_CE"] >= minimum["validation_token_CE"]
    mon = monitor_summary(run / "host_monitor.json")
    assert mon["child_exit_code"] == 0 and mon["stop_reason"] is None
    retention = read(run / "checkpoint_retention.json")
    assert len(retention["retained"]) <= 3
    return {
        "run": name,
        "training": result,
        "full_validation_history": history,
        "best_checkpoint": {
            k: best[k]
            for k in (
                "filename",
                "sha256",
                "step",
                "epoch",
                "best_validation_loss",
                "CPU_reload_verified",
            )
        },
        "best_teacher_forced_token_accuracy": minimum["validation_token_accuracy"],
        "source_commit": settings["source_commit"],
        "fresh_random_initialization_this_run": settings["fresh_random_initialization"],
        "resumed_from_checkpoint": bool(settings["resume_from"]),
        "parameter_count": settings["parameter_count"],
        "precision": settings["precision"],
        "monitor": mon,
        "retained_full_checkpoints": len(retention["retained"]),
        "verification": {
            "minimum_full_validation_CE_matches_best": True,
            "actual_best_weight_SHA256_verified": True,
            "all_complete_validation_epochs_present": True,
        },
        "private_input_sha256": {
            p.relative_to(root).as_posix(): digest(p)
            for p in (
                run / "training_report.json",
                run / "validation_history.json",
                run / "best.json",
                run / "run_contract.json",
                run / "host_monitor.json",
            )
        },
    }


def target_metrics(raw, accepted, baseline):
    output = {}
    for name, choices in (("raw", raw), ("posterior_filtered", accepted)):
        part = {f"exact_at_{k}": int(any(r["exact"] for r in choices[:k])) for k in (1, 5, 10)}
        part["Similarity_at_10"] = max((r["similarity"] for r in choices[:10]), default=0)
        part["functional_group_F1_at_10"] = max(
            (r["functional_group_F1"] for r in choices[:10]), default=0
        )
        part["gain_over_best_input_template"] = part["Similarity_at_10"] - baseline
        part["beats_best_input_template"] = int(part["gain_over_best_input_template"] > 1e-8)
        output[name] = part
    output["raw"].update(
        candidate_count=len(raw),
        valid_candidates=sum(r["chemical_valid"] for r in raw),
        formula_correct_candidates=sum(r["formula_match"] for r in raw),
        formula_correct_structure_wrong_candidates=sum(
            r["formula_match"] and not r["exact"] for r in raw
        ),
    )
    output["posterior_filtered"].update(
        distinct_legal_formula_candidates=len(accepted), no_qualified_candidate=int(not accepted)
    )
    return output


def aggregate(rows):
    result = {"targets": len(rows), "all_targets_in_denominator": True}
    for name in ("raw", "posterior_filtered"):
        result[name] = {k: float(np.mean([r[name][k] for r in rows])) for k in rows[0][name]}
    n = sum(r["raw"]["candidate_count"] for r in rows)
    f = sum(r["raw"]["formula_correct_candidates"] for r in rows)
    result["raw"]["candidate_chemical_valid_fraction"] = sum(
        r["raw"]["valid_candidates"] for r in rows
    ) / max(n, 1)
    result["raw"]["formula_correct_but_wrong_structure_fraction"] = sum(
        r["raw"]["formula_correct_structure_wrong_candidates"] for r in rows
    ) / max(f, 1)
    return result


def compare_metrics(actual, expected):
    assert actual["targets"] == expected["targets"]
    assert expected["all_targets_in_denominator"]
    for name in ("raw", "posterior_filtered"):
        assert actual[name].keys() == expected[name].keys()
        for key in actual[name]:
            close(actual[name][key], expected[name][key])


def audit_cohort(name, gold_rows, best_sha):
    run = NEW / ("eval_" + name)
    report = read(run / "evaluation_report.json")
    indices = np.load(NEW / "data" / (name + ".npy"))
    expected = {gold_rows[int(i)]["target_id"]: gold_rows[int(i)] for i in indices}
    assert len(expected) == len(indices) == report["targets"] == 512
    assert report["status"] == "completed" and report["checkpoint_sha256"] == best_sha
    expected_split = {"validation512": "validation", "bank_targets512": "train", "test512": "test"}[
        name
    ]
    assert all(r["split"] == expected_split for r in expected.values())
    if name == "bank_targets512":
        assert all(r["is_template_bank_member"] for r in expected.values())
    rows, by_P, seen_ids = [], {True: [], False: []}, set()
    canonical_cache = {}
    raw_path = run / "raw_candidates_private.jsonl.gz"
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            target = expected[record["target_id"]]
            assert record["target_id"] not in seen_ids
            seen_ids.add(record["target_id"])
            raw, accepted, seen_graphs = record["candidates"], [], set()
            assert len(raw) == report["beam_size"] == 20
            for rank, candidate in enumerate(raw, 1):
                assert candidate["raw_rank"] == rank
                if not candidate["chemical_valid"]:
                    assert not candidate["formula_match"] and not candidate["exact"]
                    continue
                assert candidate["eos"]
                text = candidate["smiles"]
                if text not in canonical_cache:
                    mol = Chem.MolFromSmiles(text)
                    assert mol is not None
                    canonical_cache[text] = (
                        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False),
                        rdMolDescriptors.CalcMolFormula(capped_core(mol)),
                    )
                key, formula = canonical_cache[text]
                assert candidate["formula_match"] == (formula == target["core_formula"])
                assert candidate["exact"] == (key == target["anchored_graph_key"])
                if candidate["formula_match"] and key not in seen_graphs:
                    seen_graphs.add(key)
                    accepted.append(candidate)
            metrics = target_metrics(raw, accepted, record["best_input_template_similarity"])
            for mode in ("raw", "posterior_filtered"):
                for key, value in metrics[mode].items():
                    close(value, record[mode][key])
            rows.append(metrics)
            by_P[target["contains_P"]].append(metrics)
    assert seen_ids == expected.keys()
    compare_metrics(aggregate(rows), report)
    compare_metrics(aggregate(by_P[True]), report["P"])
    compare_metrics(aggregate(by_P[False]), report["nonP"])
    mon = monitor_summary(run / "host_monitor.json")
    assert mon["child_exit_code"] == 0 and mon["stop_reason"] is None
    replay = {
        "status": "passed",
        "targets": len(rows),
        "raw_candidates": sum(len_row["raw"]["candidate_count"] for len_row in rows),
        "unique_target_IDs_match_fixed_cohort": True,
        "all_invalid_and_empty_targets_retained": True,
        "valid_candidate_H_capped_formulas_rechecked": True,
        "valid_candidate_exact_graph_flags_rechecked": True,
        "dedup_and_ranks_replayed_from_saved_candidates": True,
        "all_overall_P_nonP_aggregates_match_saved_report": True,
        "raw_stream_sha256": digest(raw_path),
        "report_sha256": digest(run / "evaluation_report.json"),
    }
    print(json.dumps({"replay_cohort": name, "targets": len(rows), "status": "passed"}), flush=True)
    return {"metrics": report, "host_monitor": mon, "publication_replay_audit": replay}


def save_figure(figure, stem):
    figure.savefig(FIGURES / (stem + ".png"), dpi=180, bbox_inches="tight")
    svg = FIGURES / (stem + ".svg")
    figure.savefig(svg, bbox_inches="tight")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    plt.close(figure)


def figures(old, new):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, run in old["experiments"].items():
        history = run["full_validation_history"]
        axes[0].plot(
            [r["epoch"] for r in history], [r["validation_token_CE"] for r in history], label=name
        )
    axes[0].set(
        title="Old Polar: complete validation CE", ylabel="Token cross entropy", xlabel="Epoch"
    )
    axes[0].legend()
    for name, run in old["experiments"].items():
        history = run["full_validation_history"]
        axes[1].plot(
            [r["epoch"] for r in history], [r["training_token_CE"] for r in history], label=name
        )
    axes[1].set(title="Old Polar: training CE", ylabel="Token cross entropy", xlabel="Epoch")
    axes[1].legend()
    figure.tight_layout()
    save_figure(figure, "old_polar_training")
    history = new["experiment"]["full_validation_history"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["training_token_CE"] for r in history], label="Train")
    axes[0].plot(epochs, [r["validation_token_CE"] for r in history], label="Full validation")
    axes[0].axvline(
        new["experiment"]["training"]["best_epoch"], color="grey", ls="--", label="Best epoch43"
    )
    axes[0].set(title="New whole core: token CE", xlabel="Epoch", ylabel="Token cross entropy")
    axes[0].legend()
    axes[1].plot(epochs, [r["LR"] for r in history])
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    axes[1].set(title="New whole core: epoch-end LR", xlabel="Epoch", ylabel="Learning rate")
    figure.tight_layout()
    save_figure(figure, "new_core_training")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    names = ("validation512", "bank_targets512", "test512")
    titles = ("Validation (n=512)", "Bank-target TRAIN LOO (n=512)", "Test subset (n=512)")
    for axis, name, title in zip(axes, names, titles):
        report = new["generation"][name]["metrics"]
        for mode, shift, color, label in (
            ("raw", -0.18, "#6084ad", "Raw Beam20"),
            ("posterior_filtered", 0.18, "#249c90", "Posterior filtered"),
        ):
            values = [100 * report[mode][f"exact_at_{k}"] for k in (1, 5, 10)]
            bars = axis.bar(np.arange(3) + shift, values, width=0.36, label=label, color=color)
            axis.bar_label(bars, fmt="%.1f", fontsize=8)
        axis.set(
            title=title,
            xticks=[0, 1, 2],
            xticklabels=["Exact@1", "Exact@5", "Exact@10"],
            ylim=(0, 103),
        )
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Targets with an exact structure (%)")
    axes[1].legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    save_figure(figure, "new_core_generation")


def pct(value):
    return f"{100 * value:.2f}%"


def markdown_reports(old, new, overview):
    old_lines = [
        "# 旧 Polar 模板引导训练：E1/E2/E3 最终结果",
        "",
        "发布核验日期：2026-10-27（北京时间）。三组训练均已完成；旧生成评价异常中止，未补跑。",
        "",
        "标签为含 P Polar 片段，带旧连接标记；有效目标190,553，TRAIN171,478 / validation9,527 / test9,548。",
        "固定 TRAIN 模板5,000、K3、batch32；FP32 / Math Attention / TF32off。",
        "E1 固定1e-4；E2 预热4,000步至1e-4后余弦至1e-5；E3 预热至3e-4后余弦至1e-5。",
        "三组只按完整验证 Token CE 最小值选模，patience5，不用测试集选择。",
        "",
        "| 实验 | 完成轮数 | 总步数 | 最佳轮数 | 最佳验证 Token CE | 本次完成段用时 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, run in old["experiments"].items():
        t = run["training"]
        old_lines.append(
            f"| {name} | {t['completed_epochs']} | {t['steps']:,} | {t['best_epoch']} | "
            f"{t['best_validation_token_CE']:.10f} | {t['elapsed_seconds'] / 3600:.3f} h |"
        )
    old_lines += [
        "",
        "E3 在这三组相同数据/标签实验中验证 CE 最低。E2 从旧中断点恢复；表中用时是完成运行段，非整个跨日期总耗时。",
        "全部56/64/64轮训练与完整验证曲线已核对，最佳轮数、步数、CE与实际权重 SHA256 一致。",
        "",
        "![旧三组训练与验证曲线](figures/training_rounds_20261027/old_polar_training.png)",
        "",
        "生成评价于2026-10-26中止：RDKit `KekulizeException` 来自候选结构处理，监督器退出 child_failed。",
        "原 state.json 的 running 是历史残留；本报告依据训练报告、监督器及停止锁判定。",
        "保留原停止锁、部分原始输出及故障记录。未将不完整前缀当作完整验证/测试结果；生成指标在 JSON 中为 null。",
        "因此本轮不能给出可靠的最终分子生成 Top-k / 分子式覆盖 / Similarity@10。",
        "",
        "| 实验 | 最佳权重文件（仅保存在G盘） | SHA256 |",
        "|---|---|---|",
    ]
    for name, run in old["experiments"].items():
        b = run["best_checkpoint"]
        old_lines.append(f"| {name} | {b['filename']} | {b['sha256']} |")
    old_lines += [
        "",
        "这些完成运行段的监控未观察到显卡重置或新增 GPU 转储；WATCHDOG 目录仍无读取权限。",
        "旧73项冻结源代码哈希及停止锁保持不变。本次仅CPU核验与发布，不训练、不恢复故障评价。",
        "",
        f"[完整聚合JSON与逐轮指标]({OLD_NAME}.json) · [两轮结果总览]({INDEX_NAME}.md)",
        "",
    ]
    (PUBLIC / (OLD_NAME + ".md")).write_text("\n".join(old_lines), encoding="utf-8", newline="\n")
    t = new["experiment"]["training"]
    lines = [
        "# 甘油＋头基新库：8,000 模板最终训练与生成结果",
        "",
        "发布核验日期：2026-10-27（北京时间）。训练及预定三个生成队列均已正常完成。",
        f"训练在第 **{t['completed_epochs']}** 轮按patience5早停；最佳第 **{t['best_epoch']}** 轮，"
        f"完整验证 Token CE **{t['best_validation_token_CE']:.10f}**。",
        "",
        "主库189,864：含P164,370，不含P25,494；TRAIN170,872 / validation9,493 / test9,499。",
        "模板8,000来自TRAIN，含P6,244、不含P1,756；8,000个均保留为训练预测标签，每轮各一次。",
        "每个目标排除自身和等价输入，跨划分等价图交集为0。模板目标留一式重建属于已训练标签诊断。",
        "标签含固定甘油及两个脂肪酸角色端口101/102；分子式为H封端的中性甘油＋头基。",
        "随机初始化，沿用E3方案；batch32 / K3 / AdamW / FP32 / Math / TF32off，最长标签94、上限96，无截断。",
        "只按全部9,493个验证目标Token CE选最佳；测试不用于模型选择。",
        "",
        f"共 **{t['steps']:,}** 步，训练用时 **{t['elapsed_seconds'] / 3600:.3f} h**；"
        "48轮完整历史与最小值对应权重均核验通过。",
        "",
        "![新库训练与学习率曲线](figures/training_rounds_20261027/new_core_training.png)",
        "",
        "每个生成队列512个目标、Beam20。测试这里只评价固定512子集，不是完整9,499测试目标。",
        "原始Beam与后验分子式/端口/组装过滤分别报告；没有新增逐token元素/语法约束解码实验。",
        "无效候选、缺EOS、无合格结果均保留在分母。",
        "",
        "| 队列 | 输出 | Top-1 exact | Top-5 exact | Top-10 exact | Similarity@10 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    labels = {
        "validation512": "验证512",
        "bank_targets512": "TRAIN模板目标512（留一式）",
        "test512": "测试512",
    }
    for name, label in labels.items():
        r = new["generation"][name]["metrics"]
        for mode, title in (("raw", "原始"), ("posterior_filtered", "后验过滤")):
            a = r[mode]
            lines.append(
                f"| {label} | {title} | {pct(a['exact_at_1'])} | {pct(a['exact_at_5'])} | "
                f"{pct(a['exact_at_10'])} | {a['Similarity_at_10']:.4f} |"
            )
    lines += [
        "",
        "![原始与后验过滤exact](figures/training_rounds_20261027/new_core_generation.png)",
        "",
        "| 队列 | 原始候选合法率 | 至少1个严格式正确候选 | 平均合法去重候选数 | 无合格候选 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in labels.items():
        r = new["generation"][name]["metrics"]
        a = r["posterior_filtered"]
        lines.append(
            f"| {label} | {pct(r['raw']['candidate_chemical_valid_fraction'])} | "
            f"{pct(1 - a['no_qualified_candidate'])} | {a['distinct_legal_formula_candidates']:.3f} | "
            f"{pct(a['no_qualified_candidate'])} |"
        )
    test = new["generation"]["test512"]["metrics"]
    a, raw = test["posterior_filtered"], test["raw"]
    lines += [
        "",
        f"测试后验过滤的Similarity@10比最佳输入模板平均增加 **{a['gain_over_best_input_template']:.4f}**；"
        f"**{pct(a['beats_best_input_template'])}** 的目标获得正收益。",
        f"官能团F1@10：原始 **{raw['functional_group_F1_at_10']:.4f}**，过滤后 **{a['functional_group_F1_at_10']:.4f}**。",
        "官能团按整个中性核心统计，包含固定甘油；Similarity指纹不以固定甘油原子为中心，邻域仍可含连接环境。",
        f"原始式正确候选中结构仍错误的比例为 **{pct(raw['formula_correct_but_wrong_structure_fraction'])}**；"
        "这是所有Beam候选的比例，不是目标Top-1错误率，说明仅有分子式正确不足以识别连接异构体。",
        "",
        "| 测试类别 | 目标数 | 原始Top-10 exact | 后验过滤Top-10 exact | 后验过滤Similarity@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("P", "nonP"):
        r = test[key]
        lines.append(
            f"| {key} | {r['targets']} | {pct(r['raw']['exact_at_10'])} | "
            f"{pct(r['posterior_filtered']['exact_at_10'])} | {r['posterior_filtered']['Similarity_at_10']:.4f} |"
        )
    b = new["experiment"]["best_checkpoint"]
    lines += [
        "",
        "nonP测试只有68个，分组指标应按该样本量解释。",
        "结构预训练检索使用目标图，故这些是structure-oracle条件生成结果，不是独立真实MS/MS鉴定准确率。",
        "原始与过滤候选次序不同，过滤可以提高exact，同时降低平均最大相似度；两种结果不可混用。",
        "模板目标512是已训练标签的留一式参考重建，不能与未见测试512合并报成独立测试。",
        "",
        f"最佳权重：`{b['filename']}`；SHA256：`{b['sha256']}`。权重及全部逐条候选只在G盘。",
        "发布前从1,536个目标/30,720个原始候选流重放排名、去重及所有总体/P/nonP聚合；",
        "合法候选的H封端分子式和exact图标记重新核对，全部与保存报告一致。",
        "四个已完成GPU作业（训练＋三个评价）无观察到的显卡重置/新增GPU转储；WATCHDOG权限限制保留。",
        "旧数据、旧停止锁与冻结源文件保持不变；本次发布未重新训练/解码。",
        "",
        f"[聚合JSON、全部48轮指标及发布核验]({NEW_NAME}.json) · [两轮结果总览]({INDEX_NAME}.md)",
        "",
    ]
    (PUBLIC / (NEW_NAME + ".md")).write_text("\n".join(lines), encoding="utf-8", newline="\n")
    index = [
        "# 两轮训练最终结果（2026-10-27发布）",
        "",
        f"- [旧Polar固定5,000模板：E1/E2/E3训练]({OLD_NAME}.md) · [JSON]({OLD_NAME}.json)",
        f"- [新甘油＋头基固定8,000模板：训练及生成]({NEW_NAME}.md) · [JSON]({NEW_NAME}.json)",
        "",
        "| 轮次 | 完成状态 | 最佳完整验证CE | 生成评价 |",
        "|---|---|---:|---|",
        "| 旧Polar E1/E2/E3 | 56/64/64轮早停 | E3:0.1236898564 | 候选RDKit异常中止，完整指标未取得 |",
        "| 新甘油＋头基8,000模板 | 48轮早停，最佳43轮 | 0.0959413714 | validation512、TRAIN模板目标512、test512均完成 |",
        "",
        "两轮标签、预测边界、目标库及划分不同，不能仅凭CE更低宣称新模型优于旧模型。",
        "旧模型的生成错误没有伪装为完成；新模型测试512后验过滤Top-1/5/10 exact分别为38.87%/63.87%/69.92%。",
        "新模型仍属于目标结构检索模板的条件生成实验，不能当作真实MS/MS鉴定性能。",
        "",
        "公开内容：聚合Markdown/JSON、逐轮训练与验证指标、PNG/SVG曲线、最佳权重SHA256及发布核验代码。",
        "私有内容：完整结构/来源/模板ID/逐目标预测/权重/缓存，全部继续留G盘，不上传GitHub。",
        "本次是CPU核验与发布，没有重新训练、清理旧停止锁或重试故障评价。",
        "",
        f"[发布审计JSON]({INDEX_NAME}.json)",
        "",
    ]
    (PUBLIC / (INDEX_NAME + ".md")).write_text("\n".join(index), encoding="utf-8", newline="\n")


def main():
    PUBLIC.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    old_lock_sha = digest(OLD / "GPU_STOP.json")
    frozen = read(OLD / "queue_started.json")["contract"]["source_sha256"]
    assert len(frozen) == 73 and all(digest(p) == sha for p, sha in frozen.items())
    from structure_pretrain.core_templates.common import contract

    assert contract() == read(NEW / "queue_started.json")["contract"]
    print("Both frozen source contracts verified", flush=True)
    generated = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    old = {
        "schema": "old_Polar_final_aggregate",
        "published_Asia_Shanghai": generated,
        "data": {
            "eligible": 190553,
            "train": 171478,
            "validation": 9527,
            "test": 9548,
            "template_pool": 5000,
            "K": 3,
            "batch_size": 32,
            "label_scope": "P_Polar_fragment",
        },
        "experiments": {
            key: training_result(OLD, name, 9527)
            for key, name in (
                ("E1", "E1_seed42"),
                ("E2", "E2_seed42_resume_20261026"),
                ("E3", "E3_seed42_resume_queue_20261026"),
            )
        },
        "generation": {
            "status": "failed_incomplete",
            "final_metrics": None,
            "exception": "rdkit.Chem.rdchem.KekulizeException",
            "supervisor": read(OLD / "eval_validation_pilot_resume_20261026.supervisor.json"),
            "monitor": monitor_summary(
                OLD / "eval_validation_pilot_resume_20261026/host_monitor.json"
            ),
            "partial_raw_predictions_are_not_final_metrics": True,
        },
        "old_stop_lock_sha256": old_lock_sha,
        "old_state_json_running_is_stale": True,
        "no_GPU_restart_or_evaluation_retry": True,
    }
    old["selected_experiment_by_complete_validation_CE"] = min(
        old["experiments"],
        key=lambda k: old["experiments"][k]["training"]["best_validation_token_CE"],
    )
    state = read(NEW / "state.json")
    assert state["status"] == "completed"
    manifest = read(NEW / "data/manifest.json")
    new = {
        "schema": "glycerol_core8000_final_aggregate",
        "published_Asia_Shanghai": generated,
        "state": state,
        "data": {
            k: manifest[k]
            for k in (
                "targets",
                "P_targets",
                "nonP_targets",
                "split_counts",
                "steps_per_epoch",
                "max_optimizer_steps",
                "bank_size",
                "max_observed_label_tokens",
                "vocabulary_size",
            )
        },
        "configuration": read("configs/glycerol_core_templates8000_v1.json"),
        "experiment": training_result(NEW, "train_main_k3_seed42", 9493),
        "bank_targets": read(NEW / "epoch_target_inclusion_audit.json"),
        "generation": {},
        "retrieval_is_target_structure_oracle": True,
        "not_a_real_MSMS_identification_benchmark": True,
        "raw_and_posterior_filtered_are_distinct": True,
    }
    gold_rows = pq.read_table(
        NEW / "data/targets_private.parquet",
        columns=[
            "target_id",
            "anchored_graph_key",
            "core_formula",
            "contains_P",
            "split",
            "is_template_bank_member",
        ],
    ).to_pylist()
    for name in ("validation512", "bank_targets512", "test512"):
        new["generation"][name] = audit_cohort(
            name, gold_rows, new["experiment"]["best_checkpoint"]["sha256"]
        )
    figures(old, new)
    assert digest(OLD / "GPU_STOP.json") == old_lock_sha
    assert all(digest(p) == sha for p, sha in frozen.items())
    assert contract() == read(NEW / "queue_started.json")["contract"]
    overview = {
        "schema": "two_training_rounds_publication",
        "published_Asia_Shanghai": generated,
        "old_report": OLD_NAME,
        "new_report": NEW_NAME,
        "old_73_frozen_source_hashes_unchanged": True,
        "new_training_source_contract_unchanged": True,
        "old_GPU_stop_lock_preserved": True,
        "best_weight_hashes_verified": 4,
        "generation_raw_targets_replayed": 1536,
        "generation_raw_candidates_replayed": 30720,
        "private_structures_weights_and_predictions_uploaded": False,
        "GPU_jobs_started_by_publication": 0,
        "CPU_publication_script_sha256": digest(__file__),
        "not_a_controlled_cross_round_loss_comparison": True,
        "figure_sha256": {p.name: digest(p) for p in sorted(FIGURES.glob("*"))},
    }
    markdown_reports(old, new, overview)
    write_json(PUBLIC / (OLD_NAME + ".json"), old)
    write_json(PUBLIC / (NEW_NAME + ".json"), new)
    overview["result_sha256"] = {
        p.name: digest(p)
        for p in (
            PUBLIC / (OLD_NAME + ".json"),
            PUBLIC / (OLD_NAME + ".md"),
            PUBLIC / (NEW_NAME + ".json"),
            PUBLIC / (NEW_NAME + ".md"),
        )
    }
    write_json(PUBLIC / (INDEX_NAME + ".json"), overview)
    print(
        json.dumps(
            {
                "status": "passed",
                "report": str(PUBLIC / (INDEX_NAME + ".md")),
                "GPU_jobs_started": 0,
                "replayed_targets": 1536,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
