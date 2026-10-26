# ruff: noqa: E501
"""Aggregate report and a small visual audit; keep all molecular tables on G."""

import html
import json
import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from structure_pretrain.glycerol_core_collection.collection import dump_json, file_hash

ROOT = Path("outputs/glycerol_core_collection_v03_20261026")
V2 = Path("outputs/template_guided_v2_20261023")


def report():
    s = json.loads((ROOT / "collection_summary_v03.json").read_text("utf-8"))
    assert s["element_restriction_scope"] == "retained_target_core"
    contract = json.loads((V2 / "queue_started.json").read_text("utf-8"))["contract"]
    assert all(file_hash(p) == h for p, h in contract["source_sha256"].items())
    assert file_hash(V2 / "data/manifest.json") == contract["data_manifest_sha256"]
    assert all(file_hash(p) == h for p, h in s["protected_input_SHA256_unchanged"].items())
    combined = ROOT / "combined_candidate_cores_v03.parquet"
    assert file_hash(combined) == s["combined_parquet_SHA256"]
    rows = pq.read_table(combined).to_pylist()
    assert len(rows) == len({r["target_id"] for r in rows}) == s["final_core_count"]
    assert all(r["head_attachment_position"] == "terminal" for r in rows)
    assert all(r["full_assembly_H_formula_pass"] for r in rows)
    standby = pq.read_table(ROOT / "standby_43100_mass_shift.parquet", columns=["marked_core_smiles"])
    standby_keys = set(standby["marked_core_smiles"].to_pylist())
    assert len(standby_keys) == standby.num_rows == 43100
    assert not standby_keys.intersection(r["anchored_graph_key"] for r in rows)
    observed = [r for r in rows if r["origin_category"] == "observed_PubChem_nonP_terminal_core"]
    assert len(observed) == s["new_unique_nonP_cores"]
    assert all(r["P_count"] == 0 and r["source_roundtrip_H_formula_pass"] for r in observed)
    s.update(
        standby_and_main_disjoint=True,
        distinct_total_including_standby=len(rows)+43100,
        client_time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        protected_training_source_files_unchanged=len(contract["source_sha256"]),
        role_position_P_count_and_identity_recheck=True,
        newly_collected_nonP_with_halogen=sum(r["halogen_count"] > 0 for r in observed),
        total_with_halogen=sum(r["halogen_count"] > 0 for r in rows),
    )
    qa_path = ROOT / "final_delivery_QA.json"
    if qa_path.exists():
        s["delivery_QA"] = json.loads(qa_path.read_text("utf-8"))
    old = s["old_reaudit"]
    part = old["new_disjoint_partition"]
    doc = f"""# 甘油子结构扩充、卤素放行与旧库复核 v0.3

统计时间：{s["client_time"]}。

## 最新用户约束

原195,306条库、旧8,351项扩展和当前V2训练输入全部保留。
新PubChem分支只收不含P、头基连在甘油末端氧的核心；sn2中间连接不入选。
去立体后只记录末端连接，原始sn1／sn3／立体名称继续保留在来源记录中。
不再要求来源已经连接脂肪酸；有明确甘油＋头基子结构即可作为候选。
普通去酰基H封端核心500 Da，糖基／肌醇600 Da；不是按完整来源脂质质量筛选。
允许现有10种元素C/H/N/O/P/S/F/Cl/Br/I，无单独的卤素原子数量上限。
不因含卤素或缺天然证据而拒绝数据库候选；仍保留天然证据状态。

## 旧77,656条排除是否正确

全部195,306条已用带明确甘油位点的核心重新H封端并独立复算质量，
与旧数值最大误差{old["maximum_independent_mass_error_Da"]:.2g} Da。
质量超限68,050条确实按甘油＋头基计算，没有把外部脂肪链算进去。
其中43,100条Polar本身未超限、加甘油后超限，按用户最新指令单独暂存，不进主库。

旧长链规则有一处过严：会沿碳环走出六碳路径，误记为长脂肪链。
本次6,298条此类标记已纠正，只数非环碳链；其中3,327条因此可以恢复入库，
其余仍可能超重或有其他风险。不能把旧77,656条全部解释为真正不合法的结构。

| 原库在v0.3的互斥状态 | 数量 |
|---|---:|
| 保留 | {part["pass"]:,} |
| 超出当前质量／链／锚点范围 | {part["outside_new_scope"]:,} |
| 其他化学或保护基标记暂缓 | {part["chemical_or_protecting_group_hold_only"]:,} |
| 合计 | 195,306 |

仅放开卤素恢复18,954条；另有3,036条只需修正环路径、291条同时需放开卤素和修正环路径。
合计恢复22,281条。原来的文件与停止锁均未改动。

## 新非P结构检索及数量

按[PubChem官方PUG REST子结构接口](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)查询
明确的非环三碳甘油三氧骨架，不按PC／PE等类别名称限制检索。
833,819条源CID完整取得，小于2,000,000查询上限；在同一结果集排除353,646条含P源CID，
480,173条非P源记录全部取得并逐条处理。这里的源分子数不等于最终核心数。

| 数据口径 | 去重核心数 |
|---|---:|
| 旧非P核心 | 2,170 |
| 本次PubChem净新增非P核心 | {s["new_unique_nonP_cores"]:,} |
| 非P合计 | {s["final_nonP_cores"]:,} |
| 含P合计（原库／旧扩展及恢复） | {s["final_P_cores"]:,} |
| **主候选总库** | **{s["final_core_count"]:,}** |
| 另存超重待定核心，不计入上项 | 43,100 |
| 主库＋暂存的无重复合计 | {s["distinct_total_including_standby"]:,} |

本次有{s["nonP_graphs_with_observed_database_substructure"]:,}个非P核心获得该查询中的数据库子结构对应，
其中{s["nonP_graphs_observed_as_uncut_whole_cores"]:,}个以无需切分共价取代基的完整含甘油组分出现
（采用统一质子化约定）；
{s["nonP_graphs_observed_without_nonfatty_site_cuts"]:,}个有无需切其他非脂肪酸取代基的来源，
其余可作为明确切分其他甘油位置取代基后得到的末端头基子结构候选。
新的数据库分支没有含P或sn2连接的保留项；旧扩展候选的规则来源仍保留，未升级为天然记录。
前一版2,170只是有限来源加局部变体，不是非P甘油头基的完整化学空间。

## 结构处理及可靠性

保留整个选定的末端头基图。在甘油另两个O的明确连接键上做子结构切分，
分别记录切下的是脂肪酸／醚链还是其他非脂肪酸取代基；已有自由羟基可直接标成未来外接位点。
不会把选定的氨基酸／多官能有机酸头基误当作普通脂肪酸删除。
若两个末端都有不同头基分支，可保留两个独立候选；交联而无法明确切分的情况单独留记录。
这些是来源分子确实包含的子结构，不代表该H封端核心本身已经作为完整独立化合物收录。
全部入选来源分解通过源图及含H分子式重组检查，切分原子位置和取代基类型均可追溯。
多组分来源按明确的含甘油连通组分处理，其他组分保留在原始记录中；重组证明针对该连通组分。
10种元素限制用于最终核心，不能因为切掉的取代基或对离子含其他元素而整条误排。
每个主库核心都具备连接角色、起始SMILES往返及两条参考酰链的中性组装含H公式证明；
旧99,192项证明在哈希完全一致的前提下复用，恢复项和新观察分别重新验证。
交付复核还逐条检查202,801条来源映射，涉及165,816个实际支持入选核心的源CID，
均指向存在的目标ID及本次非P查询记录；43项相关测试通过。

P头基从实际连接P起始，非P从头基连接侧甘油碳起始。源CID／名称／原始SMILES与
目标ID关系可追溯；不依赖任意canonical SMILES中的第一个P。
结构检查说明图自洽，不证明所有数据库候选天然存在、稳定或可合成。
本次完整性限定于明确的PubChem甘油查询及这些筛选规则，不能宣称穷尽所有网站或所有甘油脂化学空间。

## 数据与训练

当前V2仍为TRAIN171,478／validation9,527／test9,548、全部含P，
训练输入和{s["protected_training_source_files_unchanged"]}项受保护训练代码／配置保持原SHA。
本新库未开始训练，也尚未划分新训练／验证／测试集。
主要私有文件：

- `combined_candidate_cores_v03.parquet`：合并主库。
- `standby_43100_mass_shift.parquet`：用户指定暂存，含Polar／核心质量及其他标记。
- `phosphate_scope_reaudit.parquet`：旧195,306项新旧筛选对照。
- `pending_source_or_core.parquet`：未入选来源及原因，区分源记录和核心事件。
- `observed_source_mapping.jsonl.gz`：完整CID／名称／目标ID来源映射。
- `raw/nonP_properties/`：压缩的完整非P源结构属性，支持重做提取。
- `index.html`：小样本结构核对页，避免再生成数十万张图片。

完整数据均在G盘，GitHub仅发布代码和聚合结果。
"""
    out = Path("docs/results/GLYCEROL_SUBSTRUCTURE_V03_20261026")
    out.with_suffix(".md").write_text(doc, encoding="utf-8")
    dump_json(out.with_suffix(".json"), s)
    (ROOT / "collection_report.md").write_text(doc, encoding="utf-8")
    dump_json(ROOT / "delivery_summary_v03.json", s)
    return s, rows


def gallery(s, rows):
    rng = random.Random(42)
    observed = [r for r in rows if r["origin_category"] == "observed_PubChem_nonP_terminal_core"]
    selected = []
    for predicate in (
        lambda r: r["halogen_count"] > 0,
        lambda r: "S" in r["core_formula"],
        lambda r: "N" in r["core_formula"],
        lambda r: r["target_smiles"].count("O") >= 6,
    ):
        group = [r for r in observed if predicate(r)]
        selected.extend(rng.sample(group, min(10, len(group))))
    selected.extend(rng.sample(observed, min(50, len(observed))))
    cards, drawn = [], set()
    for r in selected:
        if r["target_id"] in drawn:
            continue
        drawn.add(r["target_id"])
        drawer = rdMolDraw2D.MolDraw2DSVG(500, 340)
        drawer.DrawMolecule(Chem.MolFromSmiles(r["target_smiles"]))
        drawer.FinishDrawing()
        cid = r["PubChem_source_CIDs"][0]
        label = html.escape(r["display_name"])
        text = html.escape(r["target_smiles"])
        cards.append(
            f"<article><b>{r['target_id']}</b><p>{label}</p>{drawer.GetDrawingText()}"
            f"<p>{r['core_formula']} · {r['core_average_mass_Da']:.2f} Da · 甘油末端连接</p>"
            f"<details><summary>SMILES与来源</summary><code>{text}</code>"
            f'<p><a href="https://pubchem.ncbi.nlm.nih.gov/compound/{cid}">PubChem CID {cid}</a></p>'
            "</details></article>"
        )
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>非P末端连接核心核对</title>
<style>body{{font:16px system-ui;background:#eef3f6;margin:25px;color:#183349}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}article{{background:white;padding:18px;border:1px solid #ccd6df;border-radius:12px;overflow:hidden}}svg{{max-width:100%;height:auto}}p{{overflow-wrap:anywhere}}code{{word-break:break-all}}</style>
<h1>非P甘油＋头基：新收录结构核对样本</h1>
<p>非P总计{s["final_nonP_cores"]:,}个；本页仅展示{len(cards)}个新收录样本，不是完整数量。
本轮要求：不含P、头基位于甘油末端氧、普通核心≤500 Da，糖／肌醇≤600 Da。
*101／*102为未来脂肪链位点。数据库记录不自动代表天然已报道。</p><main>{"".join(cards)}</main></html>"""
    (ROOT / "index.html").write_text(page, encoding="utf-8")
    dump_json(ROOT / "gallery_QA.json", {"graphs": len(cards), "no_remote_image_dependency": True})


if __name__ == "__main__":
    summary, data = report()
    gallery(summary, data)
    print(
        json.dumps(
            {
                "complete": True,
                "total": summary["final_core_count"],
                "nonP": summary["final_nonP_cores"],
            }
        ),
        flush=True,
    )
