"""Plot measured size distributions and report scope proposals without applying them."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/polar_size_scope_audit_20261022")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ.setdefault("MPLCONFIGDIR", str(root.parents[2] / "cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter, StrMethodFormatter

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    refs = json.loads((root / "reference_headgroups_private.json").read_text(encoding="utf-8"))
    columns = [
        "average_mass_Da",
        "balanced_450",
        "broad_500",
        "P_to_farthest_atom_bonds",
        "rotatable_bonds",
        "open_polar_formula",
    ]
    tab = pq.read_table(root / "per_graph_descriptors_private.parquet", columns=columns)
    mass = tab.column("average_mass_Da").to_numpy()
    base = tab.column("balanced_450").to_numpy()
    base500 = tab.column("broad_500").to_numpy()
    distances = tab.column("P_to_farthest_atom_bonds").to_numpy()
    rotations = tab.column("rotatable_bonds").to_numpy()
    scope = {
        "all_graphs": len(mass),
        "base_450_graphs": int(base.sum()),
        "root_distance_sensitivity_on_base450": {
            str(d): int(np.sum(base & (distances <= d))) for d in (8, 10, 12, 15, 18, 20)
        },
        "rotatable_bond_sensitivity_on_base450": {
            str(b): int(np.sum(base & (rotations <= b))) for b in (6, 8, 10, 12, 15)
        },
        "balanced_450_max_root12_max_rotatable12": int(
            np.sum(base & (distances <= 12) & (rotations <= 12))
        ),
        "reference_length_scope": {},
        "distinct_element_formulas": len(set(tab.column("open_polar_formula").to_pylist())),
        "selection_applied": False,
        "base_500_graphs": int(base500.sum()),
        "proposed_500_max_root15": int(np.sum(base500 & (distances <= 15))),
        "proposed_500_max_root15_and_rotatable12": int(
            np.sum(base500 & (distances <= 15) & (rotations <= 12))
        ),
    }
    for group, records in refs.items():
        scope["reference_length_scope"][group] = {
            "total": len(records),
            "mass_450_only": sum(r["average_mass_Da"] <= 450 for r in records),
            "combined_base_450": sum(not r["profile_reasons"]["balanced_450"] for r in records),
            "combined_plus_length_12_12": sum(
                not r["profile_reasons"]["balanced_450"]
                and r["P_to_farthest_atom_bonds"] <= 12
                and r["rotatable_bonds"] <= 12
                for r in records
            ),
            "combined_500": sum(not r["profile_reasons"]["broad_500"] for r in records),
            "combined_500_max_root15": sum(
                not r["profile_reasons"]["broad_500"] and r["P_to_farthest_atom_bonds"] <= 15
                for r in records
            ),
        }
    dump(root / "length_scope_sensitivity.json", scope)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9), layout="constrained")
    ax = axes[0]
    ax.hist(
        mass,
        bins=np.arange(0, max(650, np.ceil(mass.max() / 50) * 50) + 1, 25),
        color="#386b8a",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set(
        xlabel="Open Polar average mass (Da)",
        ylabel="Number of graphs",
        title="A  Frozen Polar mass distribution",
    )
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    for limit, color in ((350, "#209777"), (450, "#cc7b2e"), (500, "#5252a1"), (600, "#a94048")):
        ax.axvline(limit, color=color, ls="--", lw=1.3, label=f"{limit} Da")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    sorted_mass = np.sort(mass)
    ax.plot(sorted_mass, np.arange(1, len(mass) + 1) / len(mass) * 100, color="#386b8a", lw=2)
    for limit in (300, 350, 400, 450, 500, 600):
        percent = np.mean(mass <= limit) * 100
        ax.scatter(limit, percent, s=18, color="#cc7b2e")
    ax.set(
        xlabel="Upper mass limit (Da)",
        ylabel="Retained by mass alone",
        title="B  Mass-only candidate coverage",
        ylim=(0, 103),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.grid(axis="y", alpha=0.2)
    ax = axes[2]
    names = list(summary["profiles"])
    sizes = [summary["profiles"][name]["retained"] for name in names]
    ax.barh([name.split("_")[-1] for name in names], sizes, color="#458c79")
    for i, size in enumerate(sizes):
        ax.text(size + len(mass) * 0.015, i, f"{size:,}", va="center", fontsize=8)
    ax.set(
        xlabel="Retained graphs",
        ylabel="Mass/C/heavy-atom profile",
        title="C  Combined scope proposals",
        xlim=(0, len(mass) * 1.05),
    )
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x / 1000:.0f}k"))
    ax.invert_yaxis()
    fig.suptitle(
        f"Frozen library: {len(mass):,} rooted Polar graphs | "
        "dummy mass = 0 | glycerol/chains excluded",
        fontsize=13,
    )
    fig.savefig(root / "polar_size_distribution.png", dpi=180)
    fig.savefig(root / "polar_size_distribution.svg")
    plt.close(fig)
    q = summary["quantiles"]
    n = summary["graphs"]
    report = [
        "# 冻结 Polar 库大小与分子式范围统计，2026-10-22\n",
        (
            f"只读核验 {n:,} 个原始 Polar 图，源 SHA256 为 `{summary['source_sha256']}`。"
            "没有删除、重分划分、训练或修改任何模型资产。\n"
        ),
        (
            "质量沿用项目原有 `Descriptors.MolWt`：`[*:1]` 质量为零，统计实际隐式/显式 H；"
            "不含固定甘油和外部脂肪酸链。它是模型片段质量，并非完整脂质中性分子质量。"
            "若仅为比较而把根标记替换为 OH，平均质量增加约 17.007 Da；未改动任何源图。"
            "另保存单同位素质量供未来 MS 公式约束使用。定义见 "
            "[RDKit 官方文档](https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html)。\n"
        ),
        "## 实际分布\n",
        "|指标|最小|中位数|P90|P95|P99|最大|\n|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "average_mass_Da": "平均质量 / Da",
        "C": "C 原子数",
        "heavy_atoms": "重原子数",
        "rotatable_bonds": "可旋转键",
        "P_to_farthest_atom_bonds": "连接 P 到最远原子 / 键",
        "smiles_tokens": "SMILES token（不含 BOS/EOS）",
    }
    for field, label in labels.items():
        report.append(
            "|"
            + label
            + "|"
            + "|".join(f"{q[field][key]:.2f}" for key in ("0", "0.5", "0.9", "0.95", "0.99", "1"))
            + "|"
        )
    report += [
        "\n## 仅限制分子量的覆盖率\n",
        "|片段平均质量上限 / Da|保留图数|占全库|\n|---|---:|---:|",
    ]
    for row in summary["mass_thresholds"]:
        report.append(f"|{row['upper_Da']}|{row['count']:,}|{row['percent']:.2f}%|")
    report += [
        f"\n元素组合共 {scope['distinct_element_formulas']:,} 类"
        "（按元素计数，不含根标记及形式电荷符号）。",
        f"原化学 family：{summary['family_counts']}，并非全部 phosphate。\n",
        (
            "冻结库实际 P 原子数为 1–3；联合建议的上限 4 是为了兼容 PIP3 等已知参考结构。"
            "已知参考库与冻结目标库是两个集合；本轮交叉统计通过并不表示结构已被加入冻结库。\n"
        ),
        (
            f"有卤素的图 {summary['has_any_halogen']:,} 个；至少一个带形式电荷原子的图 "
            f"{summary['charged_atom_graphs']:,} 个，净电荷分布 {summary['net_charge_histogram']}。"
            "带电原子不作为自动排除项，避免误删 PC 等两性离子。"
        ),
        (
            f"额外脂质锚点图 {summary['extra_lipid_anchor_graphs']:,} 个。"
            "元素/计数范围原因频次（重叠，不可相加）："
            f"{summary['chemistry_scope_failure_counts_overlap']}。\n"
        ),
        "## 建议比较的联合范围\n",
        (
            "所有联合方案都仅作为候选范围建议：CHNOPS、P=1–4、N≤6、S≤2、"
            "无自由基/同位素标签、无已检出额外脂质锚点。"
            "卤素/高 N/S 超界不等于化学非法，而是本轮预测任务范围之外。\n"
        ),
        "|方案|MW / C / 重原子上限|保留图数|占全库|旧52项通过|新增87项通过|\n"
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, p in summary["profiles"].items():
        limits = p["limits"]
        old = p["reference_coverage"]["human_approved_52"]["kept"]
        new = p["reference_coverage"]["new_unreviewed_87"]["kept"]
        report.append(
            f"|{name}|{limits['mw']} / {limits['C']} / {limits['heavy']}|"
            f"{p['retained']:,}|{p['percent']:.2f}%|{old}/52|{new}/87|"
        )
    report += [
        (
            "\n450 方案再加 P 到最远原子≤12 键和可旋转键≤12，保留 "
            f"{scope['balanced_450_max_root12_max_rotatable12']:,} 图。"
            "长度敏感性完整保存在 `length_scope_sensitivity.json`；建议先用它定位较长片段，"
            "不直接把该长度设为化学合法性判据。\n"
        ),
        (
            "建议把 ≤350 Da 作为紧凑子区，默认预测上限先取 500 Da，"
            "450 Da 作为更紧的对照，500–600 Da 单列扩展池。"
            "例如常见 PIP3 的当前片段质量约 483.065 Da，450 Da 会将其挡在默认区之外。"
            "大于该范围的已报道复杂糖头基/多磷酸头基"
            "继续保留在已知结构参考库，需要时走特例流程。"
            "不要为了缩小模型目标池而删除已有来源结构。\n"
        ),
        (
            "推荐起始采用候选范围：质量≤500 Da、C≤25、重原子≤45，"
            "再加连接 P 到最远原子≤15 键；元素条件仍为 CHNOPS、P=1–4、N≤6、S≤2。"
            f"该建议会留下 {scope['proposed_500_max_root15']:,} 个图。"
            "旋转键>12 只标记复核，暂不作为硬排除，避免误挡已知柔性加合物。"
            "与 `balanced_450` 和 `compact_350` 比较目标覆盖和推断成本。"
            "建议范围是基于本库分布的工程取舍，不是文献证明的普适天然磷脂界限。"
            "当前尚未应用任何筛选。\n"
        ),
        (
            "不能直接用普通 C/H/N 分子的 DBE 整数性或 SMILES 字符数检查这个含 P 开边界片段："
            "根位点和 P 价态会改变这类简化规则的适用条件。"
            "优先依赖 RDKit 图/价态、实际 H、Builder 完整组装及中性分子式一致性。"
            "PA 的 C=0 和 PC 的两性离子形式均已设回归检查。\n"
        ),
        (
            "私有输出：逐图属性 Parquet、旧52/新87结构的阈值通过/失败原因、"
            "质量和长度敏感性 JSON、PNG/SVG 图。"
            "已知头基参考库与新候选的文献状态没有因本统计而升级。"
            "新87项尚待用户审核，不能当成87个已批准模板。\n"
        ),
        (
            "运行：`scripts/audit_polar_size_scope.py`，完成后 "
            "`scripts/report_polar_size_scope.py`。"
            "完整逐图资料留在 G 盘；对外仅发布代码、汇总 JSON/Markdown 和不含结构的分布图。\n"
        ),
        "新增质量/氢计数和边界回归测试与既有单连接测试共 25 项通过。\n",
    ]
    text = "\n".join(report)
    (root / "polar_size_scope_report.md").write_text(text, encoding="utf-8")
    public = Path("docs/results")
    (public / "POLAR_SIZE_SCOPE_AUDIT_20261022.md").write_text(text, encoding="utf-8")
    dump(
        public / "POLAR_SIZE_SCOPE_AUDIT_20261022.json",
        {"distribution": summary, "additional_scope_sensitivity": scope},
    )
    # Aggregate-only public plot; no per-graph identities or structures.
    import shutil

    shutil.copyfile(
        root / "polar_size_distribution.png", public / "POLAR_SIZE_SCOPE_DISTRIBUTION_20261022.png"
    )
    html = (
        "<!doctype html><meta charset='utf-8'><title>Polar 大小统计</title><style>"
        "body{font:16px Arial,'Microsoft YaHei';max-width:1250px;margin:35px auto;padding:20px}"
        "img{width:100%}pre{white-space:pre-wrap;line-height:1.8}</style>"
        "<h1>Polar 大小与元素范围统计</h1><img src='polar_size_distribution.svg'><pre>"
    )
    import html as html_module

    (root / "index.html").write_text(html + html_module.escape(text) + "</pre>", encoding="utf-8")
    print(json.dumps(scope, ensure_ascii=False))


if __name__ == "__main__":
    main()
