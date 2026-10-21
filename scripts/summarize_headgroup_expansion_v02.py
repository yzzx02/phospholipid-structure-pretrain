"""Reconcile private expansion/review outputs and publish structure-free aggregates."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args()
    library, review = args.library.resolve(), args.review.resolve()
    master_path = library / "reported_headgroups_v02.csv"
    master = {r["headgroup_id"]: r for r in read_csv(master_path)}
    rows = read_csv(review / "headgroup_filter_review.csv")
    summary = json.loads((review / "filter_summary.json").read_text(encoding="utf-8"))
    collection = json.loads((library / "collection_summary.json").read_text(encoding="utf-8"))
    expansion = json.loads((library / "expansion_summary.json").read_text(encoding="utf-8"))
    qa = json.loads((review / "authoring/delivery_qa.json").read_text(encoding="utf-8"))
    assert summary["input_sha256"] == digest(master_path) == qa["source_sha256"]
    assert all(r["polar_smiles"] == master[r["headgroup_id"]]["polar_smiles"] for r in rows)
    new = [r for r in rows if r["is_new"] == "True"]
    candidates = [r for r in new if r["auto_status"] != "exclude"]
    approved = [r for r in rows if r["prior_approved"] == "True"]
    assert len(approved) == expansion["approval"]["approved_graphs"]
    assert not any(r["auto_status"] == "exclude" for r in approved)
    assert len(candidates) == summary["remaining_graphs"]
    assert len(new) == sum(summary["new_counts"].values())
    for group in (candidates, approved):
        assert all(r["source"] and r["source_url"] and r["polar_smiles"] for r in group)
    approved_ids = {r["headgroup_id"] for r in approved}
    enriched = [
        {
            **master[r["headgroup_id"]],
            "preferred_review_name": r["display_name"],
            "human_decision": r["human_decision"],
            "human_note": r["human_note"],
        }
        for r in approved
    ]
    write_csv(library / "approved_v01/approved_headgroups_v02.csv", enriched)
    for filename in ("headgroup_sources.csv", "headgroup_name_alias_mapping.csv"):
        write_csv(
            library / "approved_v01" / ("v02_" + filename),
            [r for r in read_csv(library / filename) if r["headgroup_id"] in approved_ids],
        )
    pending = json.loads(
        (library / "raw/targeted_followup_pending.json").read_text(encoding="utf-8")
    )
    write_csv(library / "targeted_followup_pending.csv", pending)
    queries = json.loads((library / "raw/pubchem_queries.json").read_text(encoding="utf-8"))
    assert all(
        int(q["reported_total"]) == q["downloaded"] for q in queries if q["kind"] == "Entrez"
    )
    aggregate = {
        "schema": "reported-headgroups-v02-20261021",
        "raw_reference_graphs": len(master),
        "previous_reference_graphs": summary["previous_graphs"],
        "new_graphs": len(new),
        "new_partition": summary["new_counts"],
        "new_default_review_graphs": len(candidates),
        "previous_human_approved_graphs": len(approved),
        "previous_human_rejected_graphs": expansion["approval"]["decision_counts"]["exclude"],
        "potential_reference_pool_pending_new_review": len(approved) + len(candidates),
        "new_review_literature_verified": sum(
            r["literature_verified"] == "True" for r in candidates
        ),
        "new_review_literature_unverified": sum(
            r["literature_verified"] != "True" for r in candidates
        ),
        "new_review_builder_pass": sum(r["builder_compatible"] == "True" for r in candidates),
        "previous_approved_builder_pass": sum(r["builder_compatible"] == "True" for r in approved),
        "new_review_unusual_P_bonding_flags": sum(
            "unusual_database_P_bonding" in r["reason_codes"] for r in candidates
        ),
        "new_graph_source_overlap": dict(Counter(s for r in new for s in r["source"].split("; "))),
        "new_graphs_in_frozen_195306": sum(
            master[r["headgroup_id"]]["in_frozen_195306"] == "True" for r in new
        ),
        "new_review_graphs_in_frozen_195306": sum(
            master[r["headgroup_id"]]["in_frozen_195306"] == "True" for r in candidates
        ),
        "raw_reference_collection": collection,
        "pubchem_property_records": len(
            json.loads((library / "raw/pubchem_records.json").read_text(encoding="utf-8"))
        ),
        "pubchem_search_count": len(queries),
        "pubchem_text_query_coverage_complete_within_requested_queries": True,
        "nist_webbook_name_queries": 3,
        "nist_new_graphs": 0,
        "targeted_followup_names_without_assigned_graph": len(pending),
        "tests_passed": args.tests_passed,
        "validation": {k: qa[k] for k in qa if k not in {"source_sha256", "partition"}},
        "identity": "rooted nonisomeric Polar graph; original stereochemistry retained",
        "training_performed": False,
        "formal_template_library_modified": False,
        "spectra_generated": False,
    }
    browser_qa = review / "authoring/browser_qa.json"
    if browser_qa.exists():
        aggregate["browser_validation"] = json.loads(browser_qa.read_text(encoding="utf-8"))
    report = f"""# Polar 头基来源补充与人工审核 v0.2，2026-10-21

上一批审核 JSON/CSV 已交叉校验：52 个保留、1 个排除，选择与备注完整保留。
三个 15-isoLG–PE 加合物使用具体加合物显示名称，不再只显示 PE。
原始 v0.1、原审核页、冻结库、Builder、模型和权重均未改动。

|本轮集合|不同 Polar 图数|
|---|---:|
|相对整个 v0.1 新增|{len(new)}|
|新增初筛保留|{summary["new_counts"]["retain"]}|
|新增待判断|{summary["new_counts"]["review"]}|
|新增明确额外脂质锚点，排除|{summary["new_counts"]["exclude"]}|
|默认供人工审核的新图|{len(candidates)}|
|上一批已确认保留|{len(approved)}|
|旧保留加新候选，尚非正式模板库|{len(approved) + len(candidates)}|

新候选中 {aggregate["new_review_literature_verified"]} 个有本批核实的文献结构来源，
{aggregate["new_review_literature_unverified"]} 个尚未核实具体文献，按数据库/规则来源候选保留。
{aggregate["new_review_builder_pass"]} 个通过原有 Builder；其余暂不视为可直接组装。
旧保留 52 个中 49 个通过 Builder，3 个氨基酰 PG 位置图继续作为人工认可的参考保留。
共有 {aggregate["new_review_unusual_P_bonding_flags"]} 个新候选含特殊 P 连接标记，置于待判断区，
不自动修正 P–O–N、P–O–O、P–N、P–S、P=S 等来源图。

本批读取 PubChem 官方 API 的 {aggregate["pubchem_property_records"]:,} 条去重 CID 结构记录。
36 个名称查询与 7 个文本查询的返回结构按实际甘油 O–P 子图检查；
查询结果覆盖完毕只针对所执行的查询，不宣称穷尽 PubChem、NIST 或全部磷脂。
NIST WebBook 的三个具体名称查询均为 Name Not Found，本批没有据此新增结构。
新图来源覆盖 PubChem 125、补充文献 20，交集 3；这些来源数不可直接相加。
LIPID MAPS、LSG、MS-DIAL 原有记录和全部来源关联继续保留。

Am-PE 与已有 Am-Hex-PE 合并同图，DMPE/MMPE 已存在；新增来源不重复计数。
Am-PS 的连接根据原始合成图及未标记产物记载核对，
见 [ACS Omega 原文](https://doi.org/10.1021/acsomega.8b01893)，不是仅凭名称猜测。
PtdGlc 的糖 1 位与 6 位连接分开；6 位来源见
[1983 年合成研究](https://doi.org/10.1021/jo00160a014)，
1 位定义见 [PtdGlc 合成方法](https://doi.org/10.1007/978-1-0716-2910-9_6)。
另收录明确连接的羟脯氨酸、Ser-Gly-Gly、serinol、
N-(2-hydroxyethyl)alanine 及数据库中的氨基酰 PG 等具体图。
部分氨基酰 PG 没有完成原始文献核实，不能与明确文献结构混称。
非经典醇头基按 [已合成产物结构图](https://doi.org/10.1016/j.jbc.2021.100691)
保留单磷脂酰连接图；没有通用缩写时使用英文名称，不另造脂质类别缩写。
Ornithyl-PG、BisTris 连接位置、phosphatidyltyrosine 实际合成例和
Lactose-Amadori-PE 的准确图仍待核实，本批不推测位置或枚举假想结构。

继续排除额外脂肪酸/醚长链、甾醇锚点、CL/BMP/LBPA 和第二个双酰化甘油臂。
第二脂质臂按实际骨架识别，短链也排除；支化异戊二烯锚点同时检查碳组分与主路径。
短乙酰基、氨基酸/小肽取代和多个 P 本身不自动排除，由用户看图判断。
只使用非芳香碳稠环识别甾醇，避免把芳香染料误当甾醇。

“约 20 万头基只含磷酸”不符合现有生成配置：
`configs/polar_expansion.yaml` 包含 phosphate、phosphonate 和 polyphosphate。
含甘油磷酸的真实非经典结构可以提供模板候选，但必须保留正确根连接并检查完整组装。
本批新图中 {aggregate["new_graphs_in_frozen_195306"]} 个已存在于冻结 195,306 图中；
新默认候选中 {aggregate["new_review_graphs_in_frozen_195306"]} 个已存在。
名称或来源补充并不等于向冻结库新增一个训练图。

原始扩展参考库共 {len(master)} 图，包含已排除及来源待核实项，不能称为全部人工通过，
也不能称为全部文献已报道。原始立体 SMILES、CID/DOI、证据定位和名称映射完整留存。
{qa["individual_svg_png_pairs"]} 对独立 SVG/PNG、{qa["remaining_atlas_pages"]} 页新增审核图册、
{qa["excluded_atlas_pages"]} 页新增排除图册与 9 页旧保留图册均已生成。
Excel 三个工作表共 {qa["embedded_workbook_images"]} 张嵌入结构图，
旧选择已预填，新选择未代填；Excel 和 HTML 不自动同步。
{args.tests_passed} 项相关测试通过；独立检查已核对全部图身份、SVG 原子覆盖、
Excel 实际保存的 ID/人工选择/缓存公式统计与来源 SHA256。

复现入口为 `fetch_headgroup_expansion_v02.py`（文本上限用 `--text-limit 10000`）、
`expand_reported_headgroups_v02.py`、`build_headgroup_structure_review_v01.py`
的 `--previous-csv`/`--human-review-json` 模式、Artifact Tool 工作簿导出脚本及独立审计脚本。
人工文献结构输入为私有 `raw/curated_additions.jsonl`，补充原文、图和 API 返回留在 G 盘。
GitHub 仅发布处理代码及不含逐图结构/选择的汇总。没有训练、扩池、预测谱图或修改正式模板库。
"""
    (library / "headgroup_expansion_summary.md").write_text(report, encoding="utf-8")
    (library / "expansion_aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    public = Path("docs/results")
    public.mkdir(parents=True, exist_ok=True)
    (public / "REPORTED_HEADGROUPS_V02_EXPANSION.md").write_text(report, encoding="utf-8")
    (public / "REPORTED_HEADGROUPS_V02_EXPANSION.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    files = [p for p in (library / "raw").rglob("*") if p.is_file()]
    files += [
        master_path,
        library / "headgroup_sources.csv",
        library / "headgroup_name_alias_mapping.csv",
    ]
    manifest = {str(p.relative_to(library)): digest(p) for p in sorted(files)}
    (library / "private_source_sha256_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: aggregate[k]
                for k in (
                    "new_partition",
                    "new_default_review_graphs",
                    "new_review_literature_verified",
                    "new_review_literature_unverified",
                    "new_review_builder_pass",
                    "new_review_graphs_in_frozen_195306",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
