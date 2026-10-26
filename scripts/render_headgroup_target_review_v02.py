# ruff: noqa: E501
"""Validate every proposed graph and embed compressed SVGs in an offline gallery."""

import base64
import gzip
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from render_glycerol_core_review_v01 import TEMPLATE, drawing

from structure_pretrain.glycerol_core_collection.chemistry import canonical, capped_core, reassemble
from structure_pretrain.glycerol_core_collection.collection import dump_json, read_csv
from structure_pretrain.headgroup_target_expansion.chemistry import frame, policy

ROOT = Path("outputs/headgroup_training_targets_v02_20261026")


def counts(mol):
    return Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms() if a.GetAtomicNum())


def draw_checked(row):
    mol = Chem.MolFromSmiles(row["target_smiles"])
    assert mol is not None
    assert canonical(mol, False) == row["anchored_graph_key"]
    gly = Chem.MolFromSmiles(row["glycerol_first_smiles"])
    assert canonical(gly, False) == row["anchored_graph_key"]
    _, p = frame(mol)
    if row["rooted_at_attachment_P"]:
        assert p == 0 and mol.GetAtomWithIdx(0).GetAtomicNum() == 15
    else:
        assert mol.GetAtomWithIdx(0).GetAtomicNum() == 6
    assert gly.GetAtomWithIdx(0).GetAtomicNum() == 6
    full = reassemble(mol, [(101, "CC(=O)[*:101]"), (102, "CC(=O)[*:102]")])
    expected = counts(capped_core(mol))
    expected.update({"C": 4, "H": 4, "O": 2})
    assert counts(full) == expected and Chem.GetFormalCharge(full) == 0
    assert not policy(mol)["filter_reasons"]
    return row["target_id"], drawing({"display_smiles": row["target_smiles"]})


def main():
    rows = pq.read_table(ROOT / "training_targets.parquet").to_pylist()
    images = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for i, (ident, svg) in enumerate(executor.map(draw_checked, rows), 1):
            images[ident] = svg
            if i % 1000 == 0:
                print(json.dumps({"checked_and_drawn": i, "total": len(rows)}), flush=True)
    selected = [r for r in rows if r["origin_category"] == "图册新增氨基酸头基"]
    selected += [r for r in rows if r["origin_category"] == "既有头基"][:12]
    for rule in sorted({s for r in rows for s in r["rules"]}):
        selected += [r for r in rows if rule in r["rules"]][:3]
    selected = list({r["target_id"]: r for r in selected}.values())
    (ROOT / "atlas").mkdir(exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 12)
    for start in range(0, len(selected), 12):
        sheet = Image.new("RGB", (1680, 1640), "#eef3f5")
        paint = ImageDraw.Draw(sheet)
        for j, row in enumerate(selected[start : start + 12]):
            x, y = (j % 3) * 560, (j // 3) * 410
            png = drawing({"display_smiles": row["target_smiles"]}, True)
            sheet.paste(Image.open(io.BytesIO(png)).convert("RGB"), (x, y + 54))
            name = row["display_name"]
            paint.text(
                (x + 12, y + 3),
                name[:45] + ("…" if len(name) > 45 else ""),
                font=font,
                fill="#243c50",
            )
            paint.text(
                (x + 12, y + 29),
                f"{row['core_formula']}   {row['core_average_mass_Da']:.1f}/{row['mass_limit_Da']:.0f} Da",
                font=small,
                fill="#476477",
            )
        sheet.save(ROOT / "atlas" / f"core_atlas_{start // 12 + 1:02}.png", optimize=True)
    view = []
    for r in rows:
        view.append(
            {
                **{
                    k: r[k]
                    for k in [
                        "target_id",
                        "display_name",
                        "core_formula",
                        "core_average_mass_Da",
                        "mass_limit_Da",
                        "P_count",
                        "rules",
                        "amino_acids",
                        "origin_category",
                        "target_smiles",
                        "polar_P_first_smiles",
                        "glycerol_first_smiles",
                    ]
                },
                "core_id": r["target_id"],
                "names": r["names"],
                "abbreviations": [],
                "selection_status": "结构检查通过",
                "evidence_label": r["origin_category"],
                "natural_core_verified": r["natural_core_evidence"],
                "natural_linkage_rule_candidate": r["origin_category"] == "规则扩展变体",
                "manual_decisions": [],
                "manual_notes": [],
                "proposed_filter_reasons": r["rules"],
                "core_smiles": r["target_smiles"],
                "display_smiles": r["target_smiles"],
                "whole_lipid_source_count": 0,
                "actual_acyl_lipid_source_count": 0,
                "source_urls": r["source_urls"],
                "natural_core_evidence": [],
                "evidence_category": r["origin_category"],
            }
        )
    zipped = base64.b64encode(
        gzip.compress(json.dumps(images, separators=(",", ":")).encode(), compresslevel=6)
    ).decode()
    page = TEMPLATE.replace(
        "__DATA__",
        json.dumps(view, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"),
    ).replace("__COUNT__", str(len(rows)))
    page = page.replace(
        "glycerol-core-v01-connectivity-only-20261023", "headgroup-targets-v02-500-600-20261026"
    )
    page = page.replace("甘油＋头基核心结构审核", "拟生成训练核心审核").replace(
        "其余 ≤400 Da", "普通核心 ≤500 Da"
    )
    page = page.replace("甘油＋头基（链位点 H 封端）", "拟生成核心（带外接链标记）")
    page = page.replace("core_sources.jsonl.gz", "derivation_sources.jsonl.gz").replace(
        "glycerol_core_review", "headgroup_target_review"
    )
    page = (
        page.replace("结构检查通过＋待判断", "本次新增候选")
        .replace("全部连接图", "全部拟训练目标")
        .replace("排除候选</option>", "已收录基础</option>")
    )
    page = page.replace(
        "scope==='passed'&&r.selection_status!=='排除候选'",
        "scope==='passed'&&r.origin_category!=='既有头基'",
    )
    page = page.replace(
        "scope==='excluded'&&r.selection_status==='排除候选'",
        "scope==='excluded'&&r.origin_category==='既有头基'",
    )
    page = page.replace(
        "</header>",
        "<p>本页为拟生成目标，不计入195306背景库、5000模板池或外接脂肪酸排列组合。规则扩展不等于已报道天然脂质。</p></header>",
    )
    page = page.replace(
        "<p>${esc(r.names.join",
        "<p>甘油起始核心<br><code>${esc(r.glycerol_first_smiles)}</code></p><p>连接P起始Polar片段<br><code>${esc(r.polar_P_first_smiles)}</code></p><p>${esc(r.names.join",
    )
    page = page.replace("无立体连接图", "拟训练核心").replace(
        "完整脂质结构来源 ${r.whole_lipid_source_count} 条，其中实际有酰基链 ${r.actual_acyl_lipid_source_count} 条。",
        '${esc(r.origin_category)}；${esc(r.amino_acids.join("; "))}',
    )
    page = page.replace('SVG</a> · <a href="png/${r.core_id}.png">PNG</a>', "SVG</a>")
    page = page.replace(
        '<a href="svg/${r.core_id}.svg">SVG</a>',
        '<a download="${r.core_id}.svg" href="svg/${r.core_id}.svg">SVG</a>',
    )
    page = page.replace("svg/${r.core_id}.svg", "${imageURL(r.core_id)}")
    needle = "$('grid').innerHTML=visible.map"
    assert needle in page
    page = page.replace(
        needle, "$('grid').innerHTML=visible.slice(pageIndex*40,(pageIndex+1)*40).map"
    )
    page = page.replace(
        '<div id="grid"></div>',
        '<div class="toolbar"><button id="prev">上一页</button><span id="pageCounter"></span><button id="next">下一页</button></div><div id="grid"></div>',
    )
    page = page.replace(
        "<script>",
        '<script>let pageIndex=0,SVGS={};const imageURL=id=>"data:image/svg+xml;charset=utf-8,"+encodeURIComponent(SVGS[id]||"<svg xmlns=\'http://www.w3.org/2000/svg\'></svg>");',
        1,
    )
    old = "const pages=Math.ceil(DATA.filter(r=>r.selection_status!=='排除候选').length/12);"
    assert old in page
    page = page.replace(old, "const pages=" + str((len(selected) + 11) // 12) + ";")
    extra = """
const oldStatus=status;status=function(){oldStatus();$('pageCounter').textContent=`第 ${pageIndex+1}/${Math.max(1,Math.ceil(visible.length/40))} 页`;$('prev').disabled=pageIndex===0;$('next').disabled=(pageIndex+1)*40>=visible.length};
const oldRender=render;render=function(){pageIndex=Math.min(pageIndex,Math.max(0,Math.ceil(visible.length/40)-1));oldRender()};
$('prev').onclick=()=>{pageIndex--;render()};$('next').onclick=()=>{pageIndex++;render()};$('search').oninput=$('scope').onchange=()=>{pageIndex=0;render()};
const READY=(async()=>{const bytes=Uint8Array.from(atob('__ZIP__'),c=>c.charCodeAt(0));const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));SVGS=JSON.parse(await new Response(stream).text());if(Object.keys(SVGS).length!==DATA.length)throw Error('结构图数量不匹配');render()})();READY.catch(e=>{$('status').textContent='读取失败，请用新版Chrome或Edge打开：'+e.message});
""".replace("__ZIP__", zipped)
    page = page.replace("</script>", extra + "</script>")
    page = page.replace("PNG 图册：", "PNG 样例图册（全部结构可在本页下载SVG）：")
    (ROOT / "index.html").write_text(page, encoding="utf-8")
    qa = {
        "all_targets_checked": len(rows),
        "every_target_assembled_and_H_formula_checked": True,
        "P_first_is_actual_attachment_P": True,
        "glycerol_rooted_graph_checked": True,
        "all_target_SVGs_embedded": len(images),
        "PNG_sample_graphs": len(selected),
        "PNG_atlas_pages": (len(selected) + 11) // 12,
        "HTML_bytes": (ROOT / "index.html").stat().st_size,
    }
    dump_json(ROOT / "target_chemistry_and_visual_QA.json", qa)
    payload = {
        "summary": json.loads((ROOT / "target_summary.json").read_text(encoding="utf-8")),
        "targets": rows,
        "amino_acids": read_csv(ROOT / "amino_acid_atlas_40_audit.csv"),
    }
    with gzip.open(
        ROOT / "authoring/workbook_payload_private.json.gz", "wt", encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False)
    print(json.dumps(qa), flush=True)


if __name__ == "__main__":
    main()
