"""Count and freeze the future combined pool without changing current training."""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from structure_pretrain.glycerol_core_collection.chemistry import (
    canonical,
    capped_core,
    parse,
    reassemble,
)
from structure_pretrain.glycerol_core_collection.collection import dump_json, file_hash
from structure_pretrain.headgroup_target_expansion.chemistry import (
    balanced_parent,
    representations,
)

EXPANSION = Path("outputs/headgroup_training_targets_v02_20261026")
ROOT = Path("outputs/glycerol_core_training_pool_v02_20261026")
BACKGROUND = Path("data/structure_library_v0.3/polar_library.parquet")
V2 = Path("outputs/template_guided_v2_20261023")
GLYCEROL = parse("[*:101]OCC(O[*:102])CO")
GLYCEROL_O = next(
    a.GetIdx() for a in GLYCEROL.GetAtoms() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
)
TAILS = [(101, "CC(=O)[*:101]"), (102, "CC(=O)[*:102]")]
OUTSIDE_SCOPE = {
    "core_mass_above_500_600_policy",
    "extra_head_long_alkyl_or_acyl_chain",
    "extra_glycerol_lipid_anchor_including_short_acyl",
    "extra_fatty_N_acyl_PE_PS_head",
}
RISK_LABELS = {
    "core_mass_above_500_600_policy": "超过500／600 Da核心质量范围",
    "reactive_acyl_phosphate_mixed_anhydride": "活泼酰基磷酸混合酸酐",
    "halogen_hold_for_natural_evidence": "含卤素，待天然来源证据",
    "extra_head_long_alkyl_or_acyl_chain": "头基存在额外长烷基／酰基链",
    "synthetic_protecting_group_suspected": "疑似合成保护基",
    "extra_fatty_N_acyl_PE_PS_head": "PE／PS额外脂肪酰化",
    "extra_glycerol_lipid_anchor_including_short_acyl": "额外甘油脂锚点（含短酰基）",
}


def polar_to_marked_core(text):
    """Preserve the explicit attachment P and the two fixed glycerol chain roles."""
    head = parse(text)
    markers = [a for a in head.GetAtoms() if a.GetAtomicNum() == 0]
    assert len(markers) == 1 and markers[0].GetAtomMapNum() == 1
    assert markers[0].GetDegree() == 1
    p = markers[0].GetNeighbors()[0]
    assert p.GetAtomicNum() == 15
    offset = GLYCEROL.GetNumAtoms()
    edit = Chem.RWMol(Chem.CombineMols(GLYCEROL, head))
    edit.AddBond(GLYCEROL_O, offset + p.GetIdx(), Chem.BondType.SINGLE)
    edit.RemoveAtom(offset + markers[0].GetIdx())
    result = edit.GetMol()
    Chem.SanitizeMol(result)
    result, _ = balanced_parent(result)
    Chem.RemoveStereochemistry(result)
    return result


def counts(mol):
    return Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms() if a.GetAtomicNum())


def write_reports(s):
    dump_json(Path("docs/results/TRAINING_POOL_COUNTS_V02_20261026.json"), s)
    pool, old, active = s["combined_future_pool"], s["old_background"], s["active_V2"]
    part = old["disjoint_partition"]
    risk_rows = "\n".join(
        f"| {RISK_LABELS.get(reason, reason)} | {count:,} |"
        for reason, count in old["overlapping_reasons"].items()
    )
    text = f"""# 训练数据总量、旧库取舍与SMILES核对（2026-10-26）

## 最终口径

当前8,351个扩展核心范围按用户指令固定；合并原磷酸库时先按同一带甘油连接角色的无立体图去重。
普通核心含P／不含P均≤500 Da，糖基或已确认肌醇例外≤600 Da；
质量按去脂肪酸并以H封端后的甘油＋头基计算。
不增加新规则变体，不将脂肪酸排列组合或5,000输入模板数量加到核心总数。

| 数据口径 | 数量 |
|---|---:|
| 当前正在运行的V2训练集，全部含P | {active["train"]:,} |
| 当前V2验证集 | {active["validation"]:,} |
| 当前V2测试集 | {active["test"]:,} |
| 当前V2可用核心总数 | {active["total_eligible"]:,} |
| 新扩展库，含P＋非P | 8,351 |
| 原195,306库按新条件通过的记录 | {old["pass_rows"]:,} |
| 上项去重后的核心 | {old["pass_unique_graphs"]:,} |
| 扩展库与通过旧库的重复核心 | {s["new_overlaps_kept_old_unique"]:,} |
| 扩展库在通过旧库之外的净新增 | {s["new_unique_beyond_kept_old"]:,} |
| **下一轮合并候选总库，已去重** | **{pool["total"]:,}** |
| 合并候选中的含P核心 | {pool["with_P"]:,} |
| 合并候选中的非P核心 | {pool["without_P"]:,} |

计算式：{old["pass_unique_graphs"]:,}＋8,351－{s["new_overlaps_kept_old_unique"]:,}＝{pool["total"]:,}。
这个合并总库尚未划分新的训练／验证／测试集，也未进入当前V2续训，不能称为已训练样本量。

未施加新范围时，原库与扩展库的无重复并集为{s["union_before_new_scope_filter"]:,}；
原库与扩展库重复{s["new_overlaps_all_old_unique"]:,}项。
原库经统一质子化约定、去立体并保留连接角色后，内部合并
{old["duplicate_rows_after_stereo_free_protonation_normalization"]:,}条重复记录。
原始文件、原始ID及来源全部保留，未按新规则删改。

## 原195,306条的取舍

| 互斥状态 | 记录数 |
|---|---:|
| 通过 | {part["pass"]:,} |
| 超出质量／额外链或锚点范围 | {part["outside_new_scope"]:,} |
| 未超出前项，但因证据或活泼结构暂缓 | {part["held_for_evidence_or_chemical_review_only"]:,} |
| 合计 | 195,306 |

共{old["exclude_or_hold_rows"]:,}条不进入新范围默认库。以下理由有重叠，不能相加；
卤素及疑似保护基等待核实项，不等同于SMILES错误。

| 原审计标记 | 数量（可重叠） |
|---|---:|
{risk_rows}

## SMILES实际检查及边界

原195,306条全部通过RDKit形式结构、唯一标记P、图往返、完整中性组装及含H分子式检查。
本次将通过旧库与8,351项新库统一成带101/102外接链角色的核心，并独立检查全部
**{pool["total"]:,}**个合并图：P／甘油起始表示往返、Polar片段往返、两条参考酰链组装、
净电荷和含H元素守恒均通过。不同连接角色不因H封端后对称而误合并；不另计立体异构体。
P直接连甘油时从该P开始，其他结构从头基连接侧甘油碳开始，多P不按出现顺序猜连接点。

通过这些检查说明图、价态、标记及计算分子式自洽；不等于每个候选均天然存在或已验证稳定，
也不能代替上游来源中尚未证实的连接位点证据。未来模型生成的SMILES仍须逐条通过相同检查，
不能预先保证所有预测均正确。新全核心标签的词表、长度和数据划分尚未冻结，原Polar标签不替换。

## 冻结与文件

本次仅固定候选数据范围，未启动新库训练。原库、新库等{len(s["frozen_input_SHA256"])}项输入摘要
及{s["protected_current_training_source_files_unchanged"]}项受保护训练代码／配置核对无变更。
原库、新8,351项Parquet／CSV、参数和审计表的SHA256在私有freeze_manifest.json保存。
完整合并表位于G盘`outputs/glycerol_core_training_pool_v02_20261026/combined_candidate_cores.parquet`，
包含原Polar ID、新目标ID、统一核心SMILES、分子式、来源类别和未分配的新split。
只保存压缩Parquet与摘要，不重复生成近十万个SVG／PNG。
"""
    Path("docs/results/TRAINING_POOL_COUNTS_V02_20261026.md").write_text(text, encoding="utf-8")
    (ROOT / "training_pool_counts.md").write_text(text, encoding="utf-8")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    inputs = [
        BACKGROUND,
        EXPANSION / "background_195306_audit_P_first.parquet",
        EXPANSION / "training_targets.parquet",
        EXPANSION / "training_targets.csv",
        EXPANSION / "target_summary.json",
        Path("configs/headgroup_training_targets_v02_20261026.json"),
        V2 / "data/manifest.json",
    ]
    frozen = {str(p): file_hash(p) for p in inputs}
    old_audit = json.loads((EXPANSION / "background_audit_summary.json").read_text("utf-8"))
    assert frozen[str(BACKGROUND)] == old_audit["original_SHA256"]
    contract = json.loads((V2 / "queue_started.json").read_text("utf-8"))["contract"]
    assert frozen[str(V2 / "data/manifest.json")] == contract["data_manifest_sha256"]
    assert all(file_hash(p) == h for p, h in contract["source_sha256"].items())
    new = pq.read_table(EXPANSION / "training_targets.parquet").to_pylist()
    assert len(new) == 8351
    new_by_key = {r["anchored_graph_key"]: r for r in new}
    assert len(new_by_key) == len(new)
    all_old, kept_old = {}, {}
    matched_all, matched_kept = set(), set()
    new_vs_old_held = []
    risk = Counter()
    source_partition = Counter()
    original = pq.ParquetFile(BACKGROUND)
    audits = pq.ParquetFile(EXPANSION / "background_195306_audit_P_first.parquet")
    checked = 0
    for source, audit in zip(
        original.iter_batches(batch_size=1024, columns=["polar_id", "polar_fragment_smiles"]),
        audits.iter_batches(batch_size=1024),
        strict=True,
    ):
        for src, row in zip(source.to_pylist(), audit.to_pylist(), strict=True):
            assert src["polar_id"] == row["polar_id"]
            assert row["formal_structure_pass"] and not row["audit_failure"]
            mol = polar_to_marked_core(src["polar_fragment_smiles"])
            key = canonical(mol, False)
            all_old.setdefault(key, []).append(src["polar_id"])
            flags = set(filter(None, row["review_flags"].split(";")))
            assert row["future_500_600_policy_pass"] == (not flags)
            risk.update(flags)
            if flags & OUTSIDE_SCOPE:
                source_partition["outside_new_scope"] += 1
            elif flags:
                source_partition["held_for_evidence_or_chemical_review_only"] += 1
            else:
                source_partition["pass"] += 1
                kept_old.setdefault(key, []).append(src["polar_id"])
            if key in new_by_key:
                matched_all.add(key)
                if flags:
                    new_vs_old_held.append(
                        {
                            "target_id": new_by_key[key]["target_id"],
                            "polar_id": src["polar_id"],
                            "flags": sorted(flags),
                        }
                    )
                else:
                    matched_kept.add(key)
            checked += 1
        if checked % 16384 == 0:
            print(json.dumps({"old_records_counted": checked}), flush=True)
    assert checked == 195306
    assert source_partition["pass"] == 91335
    # Any disagreement must be resolved explicitly, never silently reintroduced.
    dump_json(ROOT / "new_vs_old_held_private.json", new_vs_old_held)
    assert not new_vs_old_held, "new candidates collide with excluded old graphs; resolve first"
    pool_keys = set(kept_old) | set(new_by_key)
    rows, qa = [], Counter()
    for i, key in enumerate(sorted(pool_keys), 1):
        mol = parse(key)
        rep = representations(mol)
        assert rep["anchored_graph_key"] == key
        # All three serialized representations refer to this one attachment-aware graph.
        if rep["polar_P_first_smiles"]:
            rebuilt = polar_to_marked_core(rep["polar_P_first_smiles"])
            assert canonical(rebuilt, False) == key
        cap = capped_core(mol)
        assembled = reassemble(mol, TAILS)
        expected = counts(cap) + Counter({"C": 4, "H": 4, "O": 2})
        assert counts(assembled) == expected
        assert Chem.GetFormalCharge(assembled) == 0
        assert not any(a.GetAtomicNum() == 0 for a in assembled.GetAtoms())
        new_row = new_by_key.get(key)
        mass = Descriptors.MolWt(cap)
        if new_row:
            assert abs(new_row["core_average_mass_Da"] - mass) < 1e-5
            assert new_row["target_id"] == rep["target_id"]
        p_count = sum(a.GetAtomicNum() == 15 for a in cap.GetAtoms())
        qa["with_P" if p_count else "without_P"] += 1
        qa["full_assembly_H_formula_pass"] += 1
        qa["rooted_roundtrip_pass"] += 1
        rows.append(
            {
                **rep,
                "core_formula": rdMolDescriptors.CalcMolFormula(cap),
                "core_average_mass_Da": mass,
                "P_count": p_count,
                "background_polar_ids": kept_old.get(key, []),
                "expansion_target_id": new_row["target_id"] if new_row else "",
                "origin_category": new_row["origin_category"] if new_row else "old_background",
                "display_name": new_row["display_name"] if new_row else "",
                "new_split": "unassigned",
                "full_assembly_H_formula_pass": True,
            }
        )
        if i % 8192 == 0:
            print(json.dumps({"combined_graphs_verified": i}), flush=True)
    path = ROOT / "combined_candidate_cores.parquet"
    assert len({r["target_id"] for r in rows}) == len(rows)
    pq.write_table(
        pa.Table.from_pylist(rows), path.with_suffix(".parquet.part"), compression="zstd"
    )
    path.with_suffix(".parquet.part").replace(path)
    split_values = np.load(V2 / "data/splits.npy", mmap_mode="r")
    eligible_rows = np.flatnonzero(split_values != 255)
    v2_audit = json.loads((V2 / "audit/A1_integrity.json").read_text("utf-8"))
    active_counts = v2_audit["split_counts"]
    assert len(eligible_rows) == v2_audit["eligible"] == 190553
    split_check = Counter(int(x) for x in split_values[eligible_rows])
    assert sorted(split_check.values()) == sorted(active_counts.values())
    assert all(file_hash(p) == h for p, h in frozen.items())
    assert all(file_hash(p) == h for p, h in contract["source_sha256"].items())
    s = {
        "client_time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "active_V2": {
            "train": active_counts["train"],
            "validation": active_counts["validation"],
            "test": active_counts["test"],
            "total_eligible": 190553,
            "contains_nonP": False,
            "inputs_changed": False,
        },
        "frozen_expansion": {
            "total": len(new),
            "with_P": sum(r["P_count"] > 0 for r in new),
            "without_P": sum(r["P_count"] == 0 for r in new),
            "user_acceptance": (
                "current candidate scope; "
                "not individual natural-evidence verification"
            ),
        },
        "old_background": {
            "original_rows": checked,
            "unchanged": True,
            "unique_normalized_anchored_graphs": len(all_old),
            "duplicate_rows_after_stereo_free_protonation_normalization": checked - len(all_old),
            "pass_rows": source_partition["pass"],
            "pass_unique_graphs": len(kept_old),
            "exclude_or_hold_rows": checked - source_partition["pass"],
            "disjoint_partition": dict(source_partition),
            "overlapping_reasons": dict(risk),
        },
        "union_before_new_scope_filter": len(all_old) + len(new) - len(matched_all),
        "new_overlaps_all_old_unique": len(matched_all),
        "new_overlaps_kept_old_unique": len(matched_kept),
        "new_unique_beyond_kept_old": len(new) - len(matched_kept),
        "combined_future_pool": {
            "total": len(rows),
            "with_P": qa["with_P"],
            "without_P": qa["without_P"],
            "split": "unassigned",
            "training_started": False,
            "not_fatty_chain_combinations_or_template_pool_size": True,
        },
        "combined_SMILES_QA": dict(qa),
        "SMILES_limits": (
            "graph/valence/attachment/roundtrip/formula checks; not proof of natural occurrence, "
            "stability or every upstream source assignment"
        ),
        "old_formal_SMILES_checks": old_audit["check_counts"],
        "mass_policy_Da": {"ordinary_P_and_nonP": 500, "sugar_or_inositol": 600},
        "frozen_input_SHA256": frozen,
        "combined_parquet_SHA256": file_hash(path),
        "protected_current_training_source_files_unchanged": len(contract["source_sha256"]),
        "storage_bytes": path.stat().st_size,
    }
    dump_json(ROOT / "freeze_manifest.json", s)
    write_reports(s)
    print(json.dumps(s, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        write_reports(json.loads((ROOT / "freeze_manifest.json").read_text("utf-8")))
    else:
        main()
