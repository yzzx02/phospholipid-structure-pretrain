"""Complete cross-references and compact outputs without altering any target graph."""

import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem

from structure_pretrain.glycerol_core_collection.chemistry import canonical, capped_core, stable_id
from structure_pretrain.glycerol_core_collection.collection import dump_json, read_csv, write_csv

ROOT = Path("outputs/headgroup_training_targets_v02_20261026")


def main():
    rows = pq.read_table(ROOT / "training_targets.parquet").to_pylist()
    base = json.loads(
        Path("outputs/glycerol_core_collection_v01_20261023/core_catalog_private.json").read_text(
            encoding="utf-8"
        )
    )
    base_by_id = {r["core_id"]: r for r in base}
    by_id = {r["target_id"]: r for r in rows}
    for r in rows:
        if r["origin_category"] == "既有头基":
            labels = [base_by_id[k]["display_name"] for k in r["existing_catalog_ids"]]
            if labels:
                r["display_name"] = min(labels, key=len)
    for r in rows:
        urls = set(u for value in r["source_urls"] for u in re.findall(r"https?://[^\s;|]+", value))
        for ident in r["existing_catalog_ids"]:
            urls.update(e["source_url"] for e in base_by_id[ident]["natural_core_evidence"])
        r["source_urls"] = sorted(urls)
    for r in rows:
        if r["origin_category"] == "规则扩展变体":
            urls = {u for ident in r["parent_target_ids"] for u in by_id[ident]["source_urls"]}
            r["source_urls"] = sorted(urls)[:8]
        r["source_urls_scope"] = (
            "precursor evidence, not evidence for a newly generated product"
            if r["origin_category"] == "规则扩展变体"
            else "reference or supplied source"
        )
    keys = set(
        pq.read_table(
            "outputs/glycerol_core_collection_v01_20261023/frozen_195306_future_core_annotations.parquet",
            columns=["future_core_id"],
        )["future_core_id"].to_pylist()
    )
    aa = read_csv(ROOT / "amino_acid_atlas_40_audit.csv")
    for r in aa:
        m = Chem.MolFromSmiles(r["target_smiles"])
        r["in_frozen_background"] = stable_id(canonical(capped_core(m), False), "GC") in keys
    write_csv(ROOT / "amino_acid_atlas_40_audit.csv", aa)
    summary = json.loads((ROOT / "target_summary.json").read_text(encoding="utf-8"))
    summary["atlas_already_in_background"] = sum(r["in_frozen_background"] for r in aa)
    references = {r["target_id"] for r in rows if r["origin_category"] == "既有头基"}
    witness = Counter()
    preferred_events = {}
    with gzip.open(ROOT / "derivation_sources.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            x = json.loads(line)
            if (
                x["origin"] == "rule_variant"
                and not x["filter_reasons"]
                and x["target_id"] in references
            ):
                witness[x["rule"]] += 1
            if (
                x["origin"] == "rule_variant"
                and not x["filter_reasons"]
                and x["target_id"] in by_id
            ):
                parent_name = by_id[x["parent_target_id"]]["display_name"]
                key = (len(parent_name), parent_name, x["rule"])
                if (
                    x["target_id"] not in preferred_events
                    or key < preferred_events[x["target_id"]][0]
                ):
                    preferred_events[x["target_id"]] = (key, parent_name + " / " + x["rule"])
    for r in rows:
        if r["origin_category"] == "规则扩展变体":
            r["display_name"] = preferred_events[r["target_id"]][1]
            r["names"] = list(
                dict.fromkeys(
                    [r["display_name"]]
                    + [by_id[k]["display_name"] for k in r["parent_target_ids"]]
                    + r["names"]
                )
            )[:20]
    summary["rules_recover_existing_reference_graphs"] = dict(witness)
    structural_hash = hashlib.sha256(
        "\n".join(
            r["target_id"] + "\t" + r["target_smiles"]
            for r in sorted(rows, key=lambda r: r["target_id"])
        ).encode()
    ).hexdigest()
    summary["target_ID_and_label_SHA256"] = structural_hash
    background = json.loads((ROOT / "background_audit_summary.json").read_text(encoding="utf-8"))
    summary["background_audit"] = background
    # This describes provenance, without treating every reference as a new manual vote.
    for r in rows:
        if r["human_decision"] == "既有认可候选":
            r["human_decision"] = "沿用参考"
    pq.write_table(
        pa.Table.from_pylist(rows), ROOT / "training_targets.parquet", compression="zstd"
    )
    write_csv(ROOT / "training_targets.csv", rows)
    with gzip.open(
        ROOT / "training_target_catalog_private.json.gz", "wt", encoding="utf-8"
    ) as stream:
        json.dump(rows, stream, ensure_ascii=False)
    with gzip.open(
        ROOT / "authoring/workbook_payload_private.json.gz", "wt", encoding="utf-8"
    ) as stream:
        json.dump(
            {"summary": summary, "targets": rows, "amino_acids": aa}, stream, ensure_ascii=False
        )
    dump_json(ROOT / "target_summary.json", summary)
    # Patch the already-validated gallery view without regenerating/rechecking SVGs.
    page = ROOT / "index.html"
    if page.exists():
        s = page.read_text(encoding="utf-8")
        start = s.index("const DATA=") + len("const DATA=")
        end = s.index(";const KEY=", start)
        view = json.loads(s[start:end])
        for item in view:
            r = by_id[item["target_id"]]
            item["core_smiles"] = r["target_smiles"]
            item["source_urls"] = r["source_urls"]
            item["display_name"] = r["display_name"]
            item["names"] = r["names"]
        s = (
            s[:start]
            + json.dumps(view, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
            + s[end:]
        )
        s = s.replace("甘油＋头基（链位点 H 封端）", "拟生成核心（带外接链标记）")
        s = (
            s.replace("结构检查通过＋待判断", "本次新增候选")
            .replace("全部连接图", "全部拟训练目标")
            .replace("排除候选</option>", "已收录基础</option>")
        )
        s = s.replace(
            "scope==='passed'&&r.selection_status!=='排除候选'",
            "scope==='passed'&&r.origin_category!=='既有头基'",
        )
        s = s.replace(
            "scope==='excluded'&&r.selection_status==='排除候选'",
            "scope==='excluded'&&r.origin_category==='既有头基'",
        )
        if "不计入195306背景库" not in s:
            s = s.replace(
                "</header>",
                "<p>拟生成目标不计入195306背景库、5000模板池或外接脂肪酸排列组合。规则扩展不等于已报道天然脂质。</p></header>",
            )
        if "甘油起始核心<br>" not in s:
            s = s.replace(
                "<p>${esc(r.names.join",
                "<p>甘油起始核心<br><code>${esc(r.glycerol_first_smiles)}</code></p><p>连接P起始Polar片段<br><code>${esc(r.polar_P_first_smiles)}</code></p><p>${esc(r.names.join",
            )
        page.write_text(s, encoding="utf-8")
    # Remove only these owned, rebuildable duplicate JSONs; compact core tables stay.
    for relative in (
        "training_target_catalog_private.json",
        "authoring/workbook_payload_private.json",
    ):
        p = ROOT / relative
        assert p.resolve().is_relative_to(ROOT.resolve())
        if p.exists():
            p.unlink()
    print(
        json.dumps(
            {
                "targets": len(rows),
                "atlas_in_background": summary["atlas_already_in_background"],
                "witness_rules": dict(witness),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
