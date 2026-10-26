# ruff: noqa: E501
"""Verify the frozen unified library and publish only aggregate reports."""

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from structure_pretrain.glycerol_core_collection.collection import dump_json, file_hash

ROOT = Path("outputs/glycerol_core_unified_v04_20261026")
V2 = Path("outputs/template_guided_v2_20261023")


def main():
    s = json.loads((ROOT / "library_manifest_v04.json").read_text("utf-8"))
    assert all(file_hash(p) == h for p, h in s["input_SHA256_unchanged"].items())
    contract = json.loads((V2 / "queue_started.json").read_text("utf-8"))["contract"]
    assert all(file_hash(p) == h for p, h in contract["source_sha256"].items())
    assert file_hash(V2 / "data/manifest.json") == contract["data_manifest_sha256"]
    seen, admitted_old, groups = set(), set(), Counter()
    shared_schema = pq.read_schema(s["files"]["main_core_library_v04"]["path"])
    for name, meta in s["files"].items():
        assert file_hash(meta["path"]) == meta["sha256"]
        assert pq.read_schema(meta["path"]).equals(shared_schema)
        rows = pq.read_table(meta["path"]).to_pylist()
        assert len(rows) == meta["rows"]
        for row in rows:
            assert row["target_id"] not in seen
            seen.add(row["target_id"])
            assert (
                row["core_scope"] == "glycerol_plus_head" and row["fixed_glycerol_backbone_present"]
            )
            assert (
                row["head_attachment_position"] == "terminal" and row["assembly_H_formula_verified"]
            )
            groups[row["admission_basis"]] += 1
            if row["selection_status"] == "main":
                admitted_old.update(row["background_polar_ids"])
    assert len(seen) == s["all_status_count"] == 226487
    assert len(admitted_old) == 158683
    cleanup = json.loads((ROOT / "cache_cleanup_receipt.json").read_text("utf-8"))
    assert cleanup["applied"]
    assert not any(
        Path(r["path"]).suffix.lower() in {".pt", ".pth", ".ckpt"} for r in cleanup["files"]
    )
    reports = {}
    for exp, run in (
        ("E1", "E1_seed42"),
        ("E2", "E2_seed42_resume_20261026"),
        ("E3", "E3_seed42_resume_queue_20261026"),
    ):
        path = V2 / run / "training_report.json"
        reports[exp] = json.loads(path.read_text("utf-8")) if path.exists() else None
    evaluation_path = V2 / "eval_validation_pilot_resume_20261026.supervisor.json"
    evaluation = json.loads(evaluation_path.read_text("utf-8"))
    s.update(
        final_identity_and_status_QA=True,
        all_statuses_disjoint=True,
        original_P_in_main=158683,
        original_P_remaining_standby=5491,
        original_P_remaining_excluded_or_review=31132,
        admission_group_counts=dict(groups),
        related_tests_passed=48,
        cache_cleanup={
            "deleted_files": cleanup["deleted_files"],
            "logical_bytes_removed": cleanup["logical_bytes_removed"],
            "disk_free_before": cleanup["disk_free_before"],
            "disk_free_after": cleanup["disk_free_after"],
            "installed_torch_version": cleanup["installed_torch_version"],
            "preserved": cleanup["preserved"],
        },
        current_model_training_reports=reports,
        generation_evaluation_status={
            "run": "eval_validation_pilot_resume_20261026",
            "status": evaluation["status"],
            "reason": "unhandled RDKit KekulizeException in generated-candidate metrics",
            "stop_lock_preserved": (V2 / "GPU_STOP.json").exists(),
            "GPU_retry_started_by_this_curation": False,
        },
    )
    md = f"""# 甘油＋头基统一候选库 v0.4 固定版

## 已执行的用户决定

固定本轮非P核心25,494个，不继续做新检索或规则扩展。
在v0.3主库144,797项上，恢复37,609个仅因加甘油后超重的原暂存图，
以及7,458个仅因疑似保护基／活泼酰基磷酸而暂缓的图。
后者含4,990个仅保护基、2,358个仅酰基磷酸、110个同时命中；每图只收一次。
风险与质量标签保留，没有去保护、改连接、修分子式或升级天然证据。

| 统一数据集 | 核心数 |
|---|---:|
| 含P主库 | 164,370 |
| 非P主库 | 25,494 |
| **主库合计** | **189,864** |
| 继续暂存：超重同时有其他标记 | 5,491 |
| 其余排除或待核 | 31,132 |
| 三种状态的无重复总数 | 226,487 |

原195,306条对应：主库158,683、暂存5,491、其他31,132；原文件未修改。
主库另含先前扩展与非P来源，不能把所有状态226,487项直接当作训练目标。
普通500 Da、糖／肌醇600 Da仍为名义质量范围；这次37,609项是明确的逐ID例外，
不自动放开其余超重结构，也不自动放回“超重＋其他标记”的5,491项。

## 统一格式

三个Parquet表采用相同字段和同一HT编号体系，每个无立体、带连接角色的图一个ID。
189,864项主库均包含固定甘油＋完整头基，两个外接脂肪链位点为`[*:101]`和`[*:102]`。
101为剩余末端甘油氧，102为中央甘油氧；头基连在另一末端氧，不增加sn2头基。
旧144,797项本来已含甘油，因此复用原图；恢复项从原Polar接入一次，不重复加甘油。
主标签`target_smiles`始终是完整核心；`glycerol_first_smiles`为同图的甘油侧起始表示。
与甘油直接相连的P用于P起始序列，多P不按SMILES出现顺序猜连接。
若P仅位于远端头基、或完全不含P，则从头基连接侧甘油碳开始。
`legacy_Polar_fragment_smiles`只作旧Polar模型的片段兼容参考，不作为新库的完整核心标签。

| 字段 | 含义 |
|---|---|
| `target_id` | 保留原HT稳定编号，不按立体异构另编号 |
| `core_scope` | 统一为glycerol_plus_head |
| `selection_status` | main / standby / excluded_or_review |
| `admission_basis` | 原主库保留、用户质量例外、用户风险标记放行等 |
| `target_smiles` | 包含甘油与两个链位点的默认序列 |
| `capped_core_smiles` | 链位点以H封端后的无虚拟原子核心 |
| `core_formula`、`core_average_mass_Da` | H封端中性核心的分子式／平均质量 |
| `nominal_mass_limit_Da`、`user_mass_exception` | 名义门槛和本次逐ID质量例外 |
| `screening_flags`、`risk_flags` | 保留原筛查标记，风险标记不自动变成删除指令 |
| `background_polar_ids`、`PubChem_source_CIDs_sample` | 原记录回查；完整PubChem映射仍见源文件 |
| `evidence_basis`、`source_origins` | 区分数据库子结构、参考记录及规则来源 |
| `assembly_H_formula_verified` | 参考链完整组装与含H分子式已核验 |

有名称时保留名称与旧别名，缺名称时使用稳定HT编号；源表中的名称、引用和立体信息继续保留。
数据库子结构及规则候选不因被放回主库而自动成为天然已报道磷脂。
后续读取主库时应使用`selection_status`和已批准的`admission_basis`，
不要再用旧v0.3硬过滤规则悄悄删除本次已批准例外。

## 校验

全部226,487图的起始原子、末端头基、两个链位点和图往返已检查，三个状态完全去重。
v0.3未改变的144,797图沿用哈希绑定的组装证明；新增恢复45,067图重新进行参考酰链完整组装，
含H分子式与电中性全部通过。其余图保留原195,306库的既有检查及本轮表示往返证明。
非P的HT集合与v0.3逐ID完全一致。48项相关测试通过。
原库、v0.3、8,351项参考／规则快照与73项受保护V2训练源文件保持原SHA。

## 缓存清理

根据哈希清单移除{cleanup["deleted_files"]:,}个可重建缓存文件，逻辑大小
{cleanup["logical_bytes_removed"] / 1e9:.3f} GB，含pip下载缓存、已安装PyTorch对应的离线安装包、
Python／pytest／ruff缓存和已被最终结果覆盖的CPU提取中间件。
PyTorch环境本体仍为{cleanup["installed_torch_version"]}，离线安装包的版本／SHA及恢复约束留在清理回执。
未遍历junction，逐文件检查绝对路径后删除；变化或占用中的缓存跳过。
完整源结构和映射、原恢复／迁移包、全部权重／预测、手工审核、监控与停止锁均保留。
详细清单为私有`cache_cleanup_plan.json`及`cache_cleanup_receipt.json`。

## 当前模型任务

E1／E2／E3训练阶段均已正常结束。随后validation pilot生成评价因一条候选的RDKit芳香性
KekulizeException中止；原state.json的running值是遗留状态，监督器报告failed。
该错误发生在生成评价，不代表本新库结构检查失败；未删除故障输入、停止锁或自动重试GPU。
本次数据整理未修改当前V2数据、词表、架构、权重或训练预算，新库也尚未训练。

## 文件

私有目录：`G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/glycerol_core_unified_v04_20261026/`。

- `main_core_library_v04.parquet`：189,864项固定主候选库。
- `standby_core_library_v04.parquet`：5,491项继续暂存。
- `excluded_core_library_v04.parquet`：31,132项其他排除／待核，同样含甘油标签。
- `library_manifest_v04.json`：输入／输出SHA、来源表指针与固定口径。

完整结构及逐图标记保留G盘，GitHub只发布处理代码和聚合报告。
"""
    (ROOT / "library_report_v04.md").write_text(md, encoding="utf-8")
    dump_json(ROOT / "delivery_summary_v04.json", s)
    dump_json(Path("docs/results/GLYCEROL_CORE_UNIFIED_V04_20261026.json"), s)
    Path("docs/results/GLYCEROL_CORE_UNIFIED_V04_20261026.md").write_text(md, encoding="utf-8")
    print(
        json.dumps(
            {
                "main": s["main_count"],
                "P": s["main_P_count"],
                "nonP": s["main_nonP_count"],
                "cache_GB_removed": cleanup["logical_bytes_removed"] / 1e9,
                "original_inputs_and_model_sources_unchanged": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
