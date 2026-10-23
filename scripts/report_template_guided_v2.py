"""Read-only V2 status report; export aggregate statistics and figures only."""

import shutil
from pathlib import Path

from structure_pretrain.formal_v2.common import ROOT, read, write
from structure_pretrain.formal_v2.retention import owned_bytes


def main():
    audit = ROOT / "audit"
    a1 = read(audit / "A1_integrity.json")
    a2 = read(audit / "A2_dual_fingerprint.json")
    a3 = read(audit / "A3_retrieval.json")
    b1 = read(audit / "B1_known_references.json")
    b2 = read(audit / "B2_coverage.json")
    cluster = read(audit / "cluster_diagnostic.json")
    chemistry = read(audit / "C_passed.json")
    cpu = read(ROOT / "cpu_gate.json") if (ROOT / "cpu_gate.json").exists() else None
    cuda = read(ROOT / "cuda_gate.json") if (ROOT / "cuda_gate.json").exists() else None
    phase = read(ROOT / "state.json")
    reports = {
        exp: read(ROOT / f"{exp}_seed42/training_report.json")
        if (ROOT / f"{exp}_seed42/training_report.json").exists()
        else None
        for exp in ("E1", "E2", "E3")
    }

    def gate_summary(g):
        if g is None:
            return None
        return {
            "passed": g["passed"],
            "device": g["device"],
            "checks": [
                {k: v for k, v in check.items() if k not in ("losses", "all_modules_updated")}
                for check in g["checks"]
            ],
        }

    result = {
        "schema": "formal_v2_aggregate_status",
        "original_core_count": a1["frozen_cores"],
        "eligible_core_count": a1["eligible"],
        "split_counts": a1["split_counts"],
        "identity": {
            k: a1[k]
            for k in (
                "cross_split_normalized_intersections",
                "all_full_lipid_rows_match_core_split",
                "formula_identities_checked",
                "bank_Builder_H_checks",
                "chainless_bank_fingerprint_checks",
                "traceable_human_seeds",
            )
        },
        "dual_fingerprint": a2,
        "reference_mask_counts": a3["mask_count_by_occupancy"],
        "retrieval_oracle": {
            key: {
                k: v
                for k, v in value.items()
                if k
                in (
                    "queries",
                    "bank",
                    "K",
                    "oracle_sample",
                    "oracle_match_fraction",
                    "boundary_tie_fallbacks",
                )
            }
            for key, value in a3["methods"].items()
        },
        "known_reference": b1,
        "coverage": b2,
        "cluster_diagnostic": cluster,
        "chemistry_roundtrip_cases": len(chemistry["chemistry_cases"]),
        "BOS_only_probe_calls": chemistry["BOS_first_token_calls"],
        "tests": read(ROOT / "unit_tests.json"),
        "cpu_gate": gate_summary(cpu),
        "cuda_gate": gate_summary(cuda),
        "phase": phase,
        "training": reports,
        "new_owned_GiB": owned_bytes(ROOT) / 2**30,
        "retention": "newV2 best+latest2, max3full/run; old exploratory assets ALL preserved",
        "generation": {
            name: read(ROOT / f"eval_{name}/evaluation_report.json")["modes"]
            if (ROOT / f"eval_{name}/evaluation_report.json").exists()
            else None
            for name in ("validation_pilot", "validation_full", "test")
        },
        "Formula_only": "conditional later stage; not started by initial queue",
        "historical_generation_metrics_not_invented": True,
    }
    target = Path("docs/results")
    target.mkdir(exist_ok=True)
    write(target / "TEMPLATE_GUIDED_V2_STATUS_20261023.json", result)
    images = target / "figures/template_guided_v2_20261023"
    images.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "figures").glob("*"):
        if source.suffix in (".png", ".svg", ".pdf") and "QA" not in source.stem:
            shutil.copyfile(source, images / source.name)
    rows = []
    for occupancy in ("0", "1"):
        v = a2[occupancy]
        rows.append(
            f"|{v['occupancy']}|{100 * v['top1_identity_agreement']:.2f}%|"
            f"{100 * v['mean_top3_intersection_over3']:.2f}%|{v['top1_score_Spearman']:.4f}|"
            f"{v['top1_changed_vs_exploratory_HNSW']:,}|"
        )
    cover = []
    for name in ("train", "validation", "test"):
        for rep in ("Polar", "chainless_actual_occupancy"):
            v = b2[name]["all"][rep]
            cover.append(
                f"|{name}|{rep}|{v['N']:,}|{v['mean']:.4f}|{v['median']:.4f}|"
                f"{100 * v['at_least_04']:.2f}%|{100 * v['at_least_06']:.2f}%|"
            )
    status = {exp: "已完成" if reports[exp] else "未完成" for exp in reports}
    body = f"""# 模板引导正式 V2.0：审计与执行状态（2026-10-23）

当前状态：`{phase.get("stage", phase.get("A", "preflight"))}`。本文件是实际状态快照；
未完成的训练和生成指标在JSON中为null，不把预备审计当作正式模型结果。

用户V2.0计划已落地，旧50轮实验归为探索性实验，其全部文件保留。
新正式实验独立随机初始化，不继承旧模型。全部数据和输出留G盘。

原冻结195,306核心保持不变；有效190,553，训练171,478、验证9,527、测试9,548。
跨划分身份/磷酸等价交集均为0；2,015,000条完整脂质记录与核心划分一致。
195,306个目标分子式恒等式（含H）、5,000个模板Builder/H、10,000个模板chainless
指纹逐一检查通过；118个人工审核种子来源可追溯。

Polar片段与chainless输入分别计算，后者包含甘油与101/102链位点，
共享骨架会提高相似度，不能混用其绝对分数。三种查询均对同一5,000模板池
执行精确检索并排除自身/等价项，独立512/512的Top3身份及分数全部吻合RDKit。

|chainless占据数|两表示Top1身份一致|Top3交集/3均值|最近分数Spearman|对旧HNSW的Top1身份变化数|
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

最后一列是身份差异数，含并列身份处理的影响，**不是HNSW严格错误率**。
新V2缓存统一采用精确结果；原缓存、标签和5,000模板身份保留。
所有190,553目标在0.2阈值下均有3个合法参考，不静默丢弃难样本。

127个已审核且兼容的参考头基对TRAIN的非自身最大Polar相似度均值
{b1["statistics"]["polar"]["mean"]:.4f}、中位数{b1["statistics"]["polar"]["median"]:.4f}，
四分位数{b1["statistics"]["polar"]["q25"]:.4f}/{b1["statistics"]["polar"]["q75"]:.4f}。
这不是MS/MS微调集。全部名称、SMILES、来源与邻居的逐条清单留私有目录。

|查询集|指纹|核心数|均值|中位数|≥0.4|≥0.6|
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(cover)}

逐化学家族、16项官能团、单双链和碳数区间的覆盖统计见聚合JSON。
三张图均提供PNG/SVG/PDF，PDF经Poppler渲染并逐张目视检查。

![已审核参考头基与结构预训练库](figures/template_guided_v2_20261023/B1_known_reference_similarity.png)

![固定5000模板覆盖](figures/template_guided_v2_20261023/B2_template_coverage.png)

![双指纹诊断](figures/template_guided_v2_20261023/A2_dual_fingerprint.png)

原测试9,548中，4,694个与TRAIN最近邻Polar相似度≥0.8。新增4,815个目标的
诊断子集，保留于4,778个稀疏图诊断簇。每个测试目标均另行对全部TRAIN精确排查。
HNSW top32稀疏图的簇属于近似分量，不能当作穷举完整簇划分；此子集仅作额外
近训练排除诊断，不替代原测试，不参与学习率选择。

PE/PG/PS、复杂N、S、多P的12项单双链Builder与含H往返通过；
多P连接依据标记邻接P，不取第一个P。随机模型实际BOS-only解码166次调用通过，
该探针不代表训练质量。{result["tests"]["count"]}项相关测试通过。
CPU真实恢复门槛：{bool(cpu and cpu["passed"])}；CUDA真实恢复门槛：{bool(cuda and cuda["passed"])}。

|实验|学习率|上限|状态|
|---|---|---|---|
|E1|固定1e-4，无scheduler|80epoch/428720更新|{status["E1"]}|
|E2|4000步warmup→1e-4，cosine→1e-5|相同|{status["E2"]}|
|E3|4000步warmup→3e-4，cosine→1e-5|相同|{status["E3"]}|

只用完整验证Token CE选择权重及学习率，patience5；测试不用于优化。
300验证试评、完整验证及完整测试按普通/元素语法约束Beam20分别报告原始和过滤结果。
完整核心Similarity使用chainless，Polar与可变头基另报；每项Template Gain
均在同一种指纹内比较。所有失败与空候选目标保留在分母，指标不提前填数。
Formula-only和额外随机种子须在模板阶段结果稳定性评估后实施，本初始队列未启动。

新输出采用**最佳＋最近两份**完整断点，去重后最多3份/组，约0.6GiB/组；
原子保存、哈希及CPU重载通过后才清理本组已登记过时文件。原探索性主训练
317份权重约58.39GiB全部保留。共享不可变数组；每目标原始候选gzip、指标Parquet、
流式汇总，无logits/attention/全配对矩阵。当前新增独占输出约{result["new_owned_GiB"]:.3f}GiB；
新V2软预算8GiB、硬预算12GiB，超限停止，不触碰历史目录。

私有根目录：`outputs/template_guided_v2_20261023`。所有GPU工作使用独立监督器；
故障锁禁止自动重试。WATCHDOG目录读取不可用，其他Windows事件、可读dump和NVIDIA遥测保留。

执行约定见[正式V2.0](../experiments/TEMPLATE_GUIDED_FORMAL_V2.md)。
更新本状态快照：`python scripts/report_template_guided_v2.py`，该命令不启动训练。
"""
    (target / "TEMPLATE_GUIDED_V2_STATUS_20261023.md").write_text(body, encoding="utf-8")
    print(
        {
            "report": str(target / "TEMPLATE_GUIDED_V2_STATUS_20261023.md"),
            "new_owned_GiB": result["new_owned_GiB"],
        }
    )


if __name__ == "__main__":
    main()
