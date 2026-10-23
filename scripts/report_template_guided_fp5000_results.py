"""CPU-only detailed training report and aggregate scientific figures; no generation."""

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from structure_pretrain.reported_headgroups.formal_template_data import ROOT, digest, read, write

DEST = Path("docs/results/TEMPLATE_GUIDED_FP5000_RESULTS_20261023")
FIG = Path("docs/results/figures/template_guided_fp5000_20261023")


def quantiles(values):
    a = np.asarray(values, dtype=float)
    return {
        name: float(fn(a))
        for name, fn in [("min", np.min), ("median", np.median), ("mean", np.mean), ("max", np.max)]
    }


def figures(history, coverage, telemetry):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    FIG.mkdir(parents=True, exist_ok=True)
    epochs = np.asarray([h["epoch"] for h in history])
    train = np.asarray([h["training_token_CE"] for h in history])
    val = np.asarray([h["validation_token_cross_entropy"] for h in history])
    acc = np.asarray([h["validation_token_accuracy"] for h in history]) * 100
    best = int(np.argmin(val))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), layout="constrained")
    ax = axes[0, 0]
    ax.plot(epochs, train, label="Train (dropout on)", color="#2474a8", lw=2)
    ax.plot(epochs, val, label="Validation (eval mode)", color="#df7a29", lw=2)
    ax.set(
        title="A  Token cross-entropy",
        xlabel="Epoch",
        ylabel="Mean CE (PAD excluded, EOS included)",
    )
    ax.legend(frameon=False)
    ax = axes[0, 1]
    ax.plot(epochs, acc, color="#258577", lw=2)
    ax.scatter(epochs[-1], acc[-1], color="#258577", s=30)
    ax.annotate(
        f"{acc[-1]:.2f}%",
        (epochs[-1], acc[-1]),
        xytext=(-6, -22),
        textcoords="offset points",
        ha="right",
    )
    ax.set(
        title="B  Teacher-forced token accuracy",
        xlabel="Epoch",
        ylabel="Validation accuracy (%)",
        ylim=(84, 97),
    )
    ax = axes[1, 0]
    ax.plot(epochs[29:], val[29:], color="#df7a29", marker="o", ms=3, lw=1.5)
    ax.scatter(epochs[best], val[best], marker="*", s=130, color="#963d20", zorder=4)
    ax.text(
        0.97,
        0.93,
        f"Best: epoch {epochs[best]}\nCE {val[best]:.6f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#ddd", "alpha": 0.9, "pad": 4},
    )
    ax.set(
        title="C  Late validation loss; stopped at local epoch cap",
        xlabel="Epoch",
        ylabel="Validation token CE",
    )
    ax = axes[1, 1]
    sizes = np.asarray(sorted(int(s) for s in coverage))
    for threshold, color in [(0.4, "#2474a8"), (0.6, "#258577")]:
        values = [
            coverage[str(s)]["all_frozen_cores"][f"fraction_at_least_{threshold}"] * 100
            for s in sizes
        ]
        ax.plot(
            sizes, values, marker="o", label=f"Nearest Tanimoto >= {threshold}", color=color, lw=2
        )
    ax.axvline(5000, color="#999", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{s:,}" for s in sizes])
    ax.set(
        title="D  Template coverage (post-selection diagnostic)",
        xlabel="Reference pool size",
        ylabel="Frozen cores covered (%)",
        ylim=(0, 100),
    )
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.suptitle("Fixed 5,000-template Polar training | K=3 | batch=32 | seed=42", fontsize=14)
    fig.savefig(FIG / "training_summary.png", dpi=180)
    fig.savefig(FIG / "training_summary.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for ax, name, label, color in zip(
        axes,
        ("temperature_C", "utilization_percent", "power_W"),
        ("GPU temperature (C)", "GPU utilization (%)", "GPU power (W)"),
        ("#df7a29", "#2474a8", "#258577"),
    ):
        ax.plot(
            [x["seconds"] / 60 for x in telemetry],
            [x[name] for x in telemetry],
            color=color,
            lw=0.8,
        )
        ax.set(xlabel="Minutes since supervisor started", ylabel=label)
        ax.grid(alpha=0.18)
    fig.suptitle(
        "Sampled device-wide NVIDIA telemetry (shared with other applications)", fontsize=12
    )
    fig.savefig(FIG / "gpu_telemetry.png", dpi=180)
    fig.savefig(FIG / "gpu_telemetry.svg")
    plt.close(fig)
    # Matplotlib emits trailing path-data spaces; keep generated text diffs clean.
    for path in FIG.glob("*.svg"):
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
            + "\n",
            encoding="utf-8",
            newline="\n",
        )


def main():
    run = ROOT / "train_main_k3_seed42"
    result, history = read(run / "training_report.json"), read(run / "validation_history.json")
    host, run_config = read(run / "host_monitor.json"), read(run / "run_contract.json")
    data, best = read(ROOT / "data/manifest.json"), read(run / "best.json")
    aggregate = read(Path("docs/results/TEMPLATE_GUIDED_FP5000_START_20261022.json"))
    assert (
        result["status"] == "completed"
        and host["child_exit_code"] == 0
        and host["stop_reason"] is None
    )
    assert len(history) == result["completed_epochs"] == 50
    assert all(
        x["validation_targets"] == 9527 and x["validation_tokens"] == 429014 for x in history
    )
    assert best["best_validation_loss"] == min(x["validation_token_cross_entropy"] for x in history)
    assert digest(best["path"]) == best["sha256"]
    values, records = [], []
    for sample in host["telemetry"]:
        row = next(csv.reader([sample["gpu"]], skipinitialspace=True))
        record = dict(
            seconds=sample["seconds"],
            temperature_C=float(row[2]),
            memory_used_MiB=float(row[3]),
            memory_total_MiB=float(row[4]),
            utilization_percent=float(row[5]),
            power_W=float(row[6]),
        )
        records.append(record)
    for name in ("temperature_C", "memory_used_MiB", "utilization_percent", "power_W"):
        values.append((name, quantiles([r[name] for r in records])))
    rows = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    end = [x for x in rows if x["stage"] == "training_completed"][-1]["unix_time"]
    tz = timezone(timedelta(hours=8))
    checkpoints = sorted((run / "checkpoint").glob("*.pt"))
    first, last = history[0], history[-1]
    measurements = read(ROOT / "selected_runtime.json")["measurements"]
    speeds = {m["batch_size"]: m["targets_per_second"] for m in measurements}
    gain = speeds[32] / speeds[16] - 1
    validation_reduction = (
        1 - last["validation_token_cross_entropy"] / first["validation_token_cross_entropy"]
    )
    improvement40 = (
        1 - last["validation_token_cross_entropy"] / history[39]["validation_token_cross_entropy"]
    )
    best_so_far, bad, max_bad = float("inf"), 0, 0
    for row in history:
        if row["validation_token_cross_entropy"] < best_so_far:
            best_so_far, bad = row["validation_token_cross_entropy"], 0
        else:
            bad += 1
            max_bad = max(max_bad, bad)
    pending = [
        "Polar chemical validity",
        "full neutral formula accuracy including H",
        "Builder assembly success",
        "candidate coverage and no-candidate rate",
        "Polar exact Top1/5/10",
        "Similarity@10",
        "functional-group recovery",
        "gain over best input template",
        "formula-correct wrong-graph rate",
        "legal isomer candidates per target",
        "Restricted catalog ranking",
        "pool/K/seed generation comparisons",
    ]
    summary = {
        "schema": "fixed5000_training_results_v1",
        "scope": "completed training and template diagnostics; generation not yet evaluated",
        "experiment": "fixed5000_K3_batch32_seed42",
        "completed_Asia_Shanghai": datetime.fromtimestamp(end, tz).isoformat(),
        "results": result,
        "validation_history": history,
        "validation_token_accuracy_teacher_forced": last["validation_token_accuracy"],
        "relative_validation_CE_reduction_epoch1_to50": validation_reduction,
        "relative_validation_CE_reduction_epoch40_to50": improvement40,
        "maximum_consecutive_epochs_without_validation_improvement": max_bad,
        "training_validation_CE_gap_epoch50": last["validation_token_cross_entropy"]
        - last["training_token_CE"],
        "data": {
            k: data[k]
            for k in (
                "frozen_core_count",
                "scope_core_count",
                "split_counts",
                "seed_templates",
                "main_bank_size",
            )
        },
        "configuration": run_config["configuration"],
        "model_parameters": run_config["parameter_count"],
        "best_checkpoint": {
            "filename": Path(best["path"]).name,
            "sha256": best["sha256"],
            "SHA_reverified": True,
            "epoch": result["best_epoch"],
            "step": best["step"],
        },
        "runtime": {
            "training_hours": result["elapsed_seconds"] / 3600,
            "supervisor_hours": host["elapsed_seconds"] / 3600,
            "effective_train_core_exposures": data["split_counts"]["train"] * len(history),
            "effective_train_cores_per_second_including_validation_checkpoints": data[
                "split_counts"
            ]["train"]
            * len(history)
            / result["elapsed_seconds"],
            "checkpoint_count": len(checkpoints),
            "checkpoint_total_GiB": sum(p.stat().st_size for p in checkpoints) / 2**30,
            "peak_tensor_allocated_MiB": result["peak_allocated_MiB"],
            "batch_benchmark": measurements,
            "batch32_throughput_gain_over16": gain,
            "telemetry_samples": len(records),
            "device_wide_telemetry_statistics": dict(values),
        },
        "monitor": {
            k: host[k]
            for k in (
                "child_exit_code",
                "stop_reason",
                "display_reset_count",
                "new_gpu_dump_count",
                "protected_checkpoints_unchanged",
                "unreadable_dump_directories",
            )
        },
        "template_coverage": aggregate["coverage"],
        "template_functional_counts_5000": aggregate["functional_coverage"]["5000"][
            "functional_counts"
        ],
        "diagnostic_scope": read(ROOT / "data/diagnostic_scope.json"),
        "generation_metrics": {name: None for name in pending},
        "generation_evaluation_completed": False,
        "source_sha256": {
            str(p): digest(p)
            for p in [
                run / "training_report.json",
                run / "validation_history.json",
                run / "host_monitor.json",
                run / "run_contract.json",
                ROOT / "data/manifest.json",
            ]
        },
    }
    figures(history, aggregate["coverage"], records)
    write(DEST.with_suffix(".json"), summary)
    chosen = {1, 2, 5, 10, 20, 30, 40, 47, 48, 49, 50}
    milestone = "\n".join(
        f"|{r['epoch']}|{r['step']:,}|{r['training_token_CE']:.6f}|{r['validation_token_cross_entropy']:.6f}|{r['validation_token_accuracy']:.4%}|"
        for r in history
        if r["epoch"] in chosen
    )
    all_epochs = "\n".join(
        f"|{r['epoch']}|{r['step']:,}|{r['training_token_CE']:.6f}|{r['validation_token_cross_entropy']:.6f}|{r['validation_token_accuracy']:.4%}|"
        for r in history
    )
    coverage_table = "\n".join(
        f"|{int(size):,}|{c['all_frozen_cores']['mean_nearest_tanimoto']:.3f}|{c['all_frozen_cores']['median_nearest_tanimoto']:.3f}|{c['all_frozen_cores']['fraction_at_least_0.4']:.2%}|{c['all_frozen_cores']['fraction_at_least_0.6']:.2%}|{c['minimum_formula_L1_including_H_quantiles']['0.5']:.0f}|"
        for size, c in aggregate["coverage"].items()
    )
    function_table = "\n".join(
        f"|{name}|{count:,}|" for name, count in summary["template_functional_counts_5000"].items()
    )
    missing_table = "\n".join(f"|{name}|尚未评价|" for name in pending)
    temperature = summary["runtime"]["device_wide_telemetry_statistics"]["temperature_C"]
    text = f"""# 固定5,000模板引导 Polar 生成模型：训练结果报告

日期：2026-10-23。实验：fixed5000 / K3 / batch32 / seed42。本报告基于已完成的训练、完整验证与主机监控原始日志；本次整理仅使用CPU，不追加训练或启动生成实验。

## 结果与结论

主训练已于 **{summary["completed_Asia_Shanghai"]}（上海）** 正常结束，共 **50轮、267,950步**，训练耗时 **{result["elapsed_seconds"] / 3600:.2f}小时**。
停止原因是预设的50 epoch上限；连续5轮验证loss无改善的早停条件没有触发。
逐轮复核得到，最长连续无新低为 **{max_bad}轮**，未达到patience5。
按全部50轮完整验证集的最低token交叉熵选择，最佳为 **第50轮 / step267950**，验证CE **{last["validation_token_cross_entropy"]:.6f}**。

从第1到第50轮，验证CE下降 **{validation_reduction:.2%}**；teacher-forced token准确率从 **{first["validation_token_accuracy"]:.4%}** 提升到 **{last["validation_token_accuracy"]:.4%}**。
后者计算真实前缀条件下下一token预测的正确比例，PAD排除、EOS计入。整条SMILES的Exact、结构化学有效率和分子式正确率需要自由生成后另行计算。
目前有证据支持“当前监督任务得到明显学习进展、训练和恢复流程稳定”；生成头基的科学可用性仍需要下文列出的生成评价支持。

![训练曲线和模板覆盖](figures/template_guided_fp5000_20261023/training_summary.png)

## 数据和运行配置

|项目|实际设置|
|---|---|
|冻结原库|195,306个Polar graph，原文件与历史split保留|
|当前单连接派生范围|190,553个核心；共4,753个核心不进入本轮范围|
|训练/验证/测试|171,478 / 9,527 / 9,548个核心|
|每epoch样本|每个训练核心一次，每核心固定一个真实完整脂质代表行|
|完整验证|每轮9,527核心、429,014个有效token，共50轮|
|参考池|5,000图，其中118个人工审核兼容种子，其余为算法补选|
|参考入口|单/双链占位匹配的chainless Morgan FP；目标式由可信全式与已知链推导|
|自体与留出排除|排除目标自体/等价图；验证/测试图不进入训练参考池|
|输入参考数量|K3，检索缓存最多10，结构阈值0.2|
|模型|现有v0.4 FP结构，16,468,671参数；当前Tokenizer和带[*:1]标签保留|
|初始化|seed42随机初始化，未继承GPU测速/恢复probe权重|
|优化器|AdamW，lr1e-4，weight decay0.01，clip1，梯度累积1|
|精度/注意力|FP32 / Math SDPA / TF32关闭|
|执行设备|NVIDIA GeForce RTX 4070 SUPER，12,282 MiB|
|保存|每1,000步及每个完整epoch原子保存，best/last指针分开|

5,000输入模板全部通过Builder和独立含H元素平衡检查。此项描述输入模板检查；输出候选的Builder合格率仍待生成评价。
600Da和当前84-token标签上限用于本轮范围检查；审核库保留其原有来源和全部合法记录。
模板补选使用训练核心的近邻密度、MW/P/N/S/卤素分层和0.85 Tanimoto球排除，完整冻结库不被重写。
等价排除使用当前phosphate_comparison_key：去除立体标签、规范末端磷酸O的质子化，保留连接标记与重原子连接；这里没有进行全部互变异构体和所有质子化体系的全局归并。

选模依据沿用[MetGenX原文](https://doi.org/10.1038/s41467-026-72149-6)的验证loss原则和patience5规则；batch32对齐其主实验。
50轮是本地执行上限，网络、学习率、weight decay和检索阈值采用用户要求保留的现有项目配置。论文的结构数据、模型设置和任务范围与本实验不同，本文不进行跨论文loss或准确率排名。

## 训练过程与末期表现

|epoch|optimizer step|训练token CE|完整验证token CE|验证token准确率（真实前缀）|
|---:|---:|---:|---:|---:|
{milestone}

第40到50轮，验证CE从{history[39]["validation_token_cross_entropy"]:.6f}降至{last["validation_token_cross_entropy"]:.6f}，相对下降 **{improvement40:.2%}**。
第47轮已达到0.138465，48/49轮有波动，第50轮再创最低值。末期改进幅度减小，但停止时没有满足patience5；达到轮数上限不能作为完成早停收敛的证据。
最终训练CE为{last["training_token_CE"]:.6f}，验证CE为{last["validation_token_cross_entropy"]:.6f}，差值为{summary["training_validation_CE_gap_epoch50"]:.6f}。
训练CE是在dropout开启、epoch内权重不断更新时累计，验证CE是在epoch末eval模式计算，比较时应保留这一计算条件差异；该差值单独不足以判断过拟合。
目前只完成seed42一次运行，尚无多随机种子的方差或置信区间。
历史FP50k实验采用不同训练池、检索缓存和训练预算；本次尚未完成同目标、同候选预算的并列生成评价，因此不据两次loss直接判断新模型优于历史模型。

## 模板池覆盖诊断

下表在全195,306个冻结核心上计算，排除自身/等价参考，使用rooted Polar Morgan。它是选池后的离线结构诊断；训练特征入口使用chainless Morgan。
HNSW提供候选，随后精确位向量重排。主池单/双链输入检索各64个随机样本的精确top1分数命中为63/64；各池覆盖检索抽样记录见JSON。全量覆盖统计仍属于近似检索结果。

|模板数|最近模板Tanimoto均值|中位数|≥0.4覆盖|≥0.6覆盖|最近分子式L1距离中位数（含H）|
|---:|---:|---:|---:|---:|---:|
{coverage_table}

只有5,000池完成了本次生成模型训练。其他池在此仅有参考覆盖统计；不能据覆盖率推断其生成成绩，也不能据此判定5,000是最优池规模。
5,000池覆盖了全部16个预设SMARTS官能团/键类型标志。各标志允许重叠，数量分布并不均匀：

|结构标志|5,000模板中数量|
|---|---:|
{function_table}

稀少的腈基、P–S、P–N等类型需要在后续生成评价中分层检查。“参考池出现过某类”与“模型能正确恢复该类”分别评价。

## 时间、资源与稳定性

|项目|实测结果|
|---|---:|
|训练进程总耗时（含验证/保存）|{result["elapsed_seconds"]:.1f}秒 / {result["elapsed_seconds"] / 3600:.2f}小时|
|监督器总耗时|{host["elapsed_seconds"]:.1f}秒|
|累计训练核心呈现次数|{summary["runtime"]["effective_train_core_exposures"]:,}|
|端到端有效吞吐（含验证/保存）|{summary["runtime"]["effective_train_cores_per_second_including_validation_checkpoints"]:.1f}核心/秒|
|batch16测速|{speeds[16]:.1f}核心/秒|
|batch32测速|{speeds[32]:.1f}核心/秒，提升{gain:.2%}|
|训练峰值tensor allocated|{result["peak_allocated_MiB"]:.1f}MiB|
|主训练checkpoint|{len(checkpoints)}份，合计{summary["runtime"]["checkpoint_total_GiB"]:.2f}GiB|
|NVIDIA采样|{len(records)}次|
|GPU温度|{temperature["min"]:.0f}–{temperature["max"]:.0f}°C，中位数{temperature["median"]:.0f}°C|
|GPU利用率采样|均值{summary["runtime"]["device_wide_telemetry_statistics"]["utilization_percent"]["mean"]:.1f}%，中位数{summary["runtime"]["device_wide_telemetry_statistics"]["utilization_percent"]["median"]:.0f}%|
|显示驱动重置 / 新GPU dump|{host["display_reset_count"]} / {host["new_gpu_dump_count"]}|
|历史12份权重SHA|全部保持不变|
|监督器/训练退出|正常，退出码0|

![NVIDIA遥测](figures/template_guided_fp5000_20261023/gpu_telemetry.png)

NVIDIA遥测是设备总量，会包含其他桌面应用；tensor allocated来自PyTorch，二者统计口径不同。进程CPU/RAM峰值本轮未记录。
CPU与CUDA均完成真实反传和磁盘checkpoint保存/重载检查。CPU连续4步与2+2步结果精确一致；CUDA模型/优化器/logits/RNG通过预先声明的1e-5数值容差检查。
监控覆盖System4101、WER LiveKernelEvent141/117和可访问dump；受保护WATCHDOG目录不可读，该限制保留在主机报告中。
首次非r1测速的严格确定性开关错误及停止锁均保留；r1在用户明确授权后运行，本次完成后未追加训练。

## 最佳权重与复核文件

最佳/最近权重均为 `step_267950_epoch_050.pt`。重新计算SHA与best指针一致，并核对其loss为全部50轮完整验证的最低值。

```text
G:\\yzx\\linzhiyuce\\phospholipid-structure-pretrain\\outputs\\template_guided_fp5000_20261022_r1\\train_main_k3_seed42\\checkpoint\\step_267950_epoch_050.pt
SHA256: {best["sha256"]}
```

同一私有运行目录保留training_report.json、validation_history.json、best.json、last.json、host_monitor.json、run_contract.json和events.jsonl。
独立完整统计见[同名JSON](TEMPLATE_GUIDED_FP5000_RESULTS_20261023.json)；[早期准备/启动记录](TEMPLATE_GUIDED_FP5000_START_20261022.md)继续保留。
权重、完整结构和逐目标数据保留在G盘，公开内容为代码、配置、聚合统计和图。

## 尚未完成的生成评价

|指标/实验|状态|
|---|---|
{missing_table}

应固定本次按验证loss选出的第50轮权重，先以有界小批量验证生成、式约束和Builder通路，再运行固定验证诊断和独立结构测试。
原diagnostic512中有495个头基属于本轮单连接派生范围；后续须明确495/512的分母、17个范围外核心及代表链行差异，避免把范围变化当作成绩改进。
完整独立测试池为9,548核心。真实MS/MS检索和Restricted白名单排序仍属于后续应用阶段。
测试分数和化学指标用于评价已选模型；模型选择继续以完整验证loss为准。

## 附：全部50轮日志

|epoch|optimizer step|训练token CE|完整验证token CE|验证token准确率（真实前缀）|
|---:|---:|---:|---:|---:|
{all_epochs}
"""
    DEST.with_suffix(".md").write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(DEST.with_suffix(".md")),
                "best_epoch": result["best_epoch"],
                "best_loss": result["best_validation_loss"],
                "validation_token_accuracy": last["validation_token_accuracy"],
                "validation_CE_reduction": validation_reduction,
                "telemetry": dict(values),
                "checkpoint_count": len(checkpoints),
                "checkpoint_GiB": summary["runtime"]["checkpoint_total_GiB"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
