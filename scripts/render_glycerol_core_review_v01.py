# ruff: noqa: E501
# Embedded HTML / report prose and source URLs intentionally retain readable text.
"""Offline RDKit gallery. One card/ID per connection graph, no stereo numbering."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path("outputs/glycerol_core_collection_v01_20261023")
LABELS = {
    "natural_reported_core": "天然头基有文献依据",
    "database_whole_lipid_natural_unverified": "有完整脂质结构；天然来源未核实",
    "reported_synthetic_core": "有合成记录；天然来源未核实",
    "natural_linkage_rule_candidate": "天然连接规则扩展候选",
    "source_definition_or_reference_natural_unverified": "来源规则或旧参考；天然来源未核实",
}


def drawing(row, png=False):
    mol = Chem.MolFromSmiles(row["display_smiles"])
    Chem.RemoveStereochemistry(mol)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(560, 350) if png else rdMolDraw2D.MolDraw2DSVG(560, 350)
    drawer.drawOptions().addStereoAnnotation = False
    highlights = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in (0, 15)]
    colors = {
        i: (0.62, 0.86, 0.95) if mol.GetAtomWithIdx(i).GetAtomicNum() == 15 else (0.78, 0.89, 0.82)
        for i in highlights
    }
    drawer.DrawMolecule(mol, highlightAtoms=highlights, highlightAtomColors=colors)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def main():
    rows = json.loads((ROOT / "core_catalog_private.json").read_text(encoding="utf-8"))
    for name in ("svg", "png", "atlas"):
        (ROOT / name).mkdir(exist_ok=True)
    review = []
    for i, row in enumerate(rows, 1):
        (ROOT / "svg" / (row["core_id"] + ".svg")).write_text(drawing(row), encoding="utf-8")
        (ROOT / "png" / (row["core_id"] + ".png")).write_bytes(drawing(row, True))
        review.append({**row, "number": i, "evidence_label": LABELS[row["evidence_category"]]})
    # A compact contact sheet supports browsing without opening every file.
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 13)
    passed = [x for x in review if x["selection_status"] != "排除候选"]
    for start in range(0, len(passed), 12):
        sheet = Image.new("RGB", (1680, 1640), "#edf2f5")
        paint = ImageDraw.Draw(sheet)
        for j, row in enumerate(passed[start : start + 12]):
            x, y = (j % 3) * 560, (j // 3) * 410
            image = Image.open(ROOT / "png" / (row["core_id"] + ".png")).convert("RGB")
            sheet.paste(image, (x, y + 54))
            name = row["display_name"]
            if len(name) > 45:
                name = name[:42] + "..."
            paint.text((x + 12, y + 5), f"{row['number']:03}  {name}", fill="#233b50", font=font)
            paint.text(
                (x + 12, y + 30),
                f"{row['core_formula']}  {row['core_average_mass_Da']:.1f} / {row['mass_limit_Da']:.0f} Da",
                fill="#405468",
                font=small,
            )
        sheet.save(ROOT / "atlas" / f"core_atlas_{start // 12 + 1:02}.png", optimize=True)
    payload = json.dumps(review, ensure_ascii=False).replace("</", "<\\/")
    page = TEMPLATE.replace("__DATA__", payload).replace("__COUNT__", str(len(rows)))
    (ROOT / "index.html").write_text(page, encoding="utf-8")
    print(
        json.dumps(
            {
                "graphs": len(rows),
                "atlas_graphs": len(passed),
                "atlas_pages": (len(passed) + 11) // 12,
            }
        )
    )


TEMPLATE = r"""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>甘油＋头基核心结构审核</title>
<style>*{box-sizing:border-box}body{margin:0;background:#eef2f5;color:#263b4c;font:16px/1.6 "Microsoft YaHei",Arial,sans-serif}header{padding:24px 32px;background:#233d52;color:white}h1{margin:0 0 8px;font-size:28px}header p{margin:4px 0}main{padding:22px 28px}.toolbar{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:12px}input,select,button{font:inherit;padding:8px;border:1px solid #bccbd5;border-radius:5px;background:white}button{cursor:pointer}button.primary{background:#285c79;color:white}#search{min-width:340px;flex:1}#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:18px}.card{background:white;padding:18px;border:1px solid #d4dfe6;border-radius:10px;min-width:0}.card h2{font-size:19px;margin:5px 0;overflow-wrap:anywhere}.meta{color:#627c8d;font-size:13px}.badge{padding:3px 7px;border-radius:4px;background:#e9f1f7;display:inline-block}.natural{background:#e3f3e8}.excluded{background:#fae4df}.card img{width:100%;height:300px;object-fit:contain;cursor:zoom-in}.reason{color:#974522;font-size:14px}.decisions{display:flex;gap:7px;margin-top:12px}.decisions button.active{background:#285c79;color:white}.note{width:100%;margin-top:8px}summary{cursor:pointer}details p{font-size:14px;overflow-wrap:anywhere}code{font-family:Consolas,monospace;font-size:12px;overflow-wrap:anywhere}.sources a{display:block;overflow-wrap:anywhere}#status{margin:12px 0}dialog{border:0;border-radius:10px;width:min(1100px,96vw)}dialog::backdrop{background:#142635c9}dialog img{width:100%;height:70vh;object-fit:contain}a{color:#28678d}.warn{color:#805825}#atlas a{margin-right:12px}</style>
<header><h1>甘油＋头基核心结构审核</h1><p>__COUNT__ 个连接图。每图一个 ID，忽略立体异构；保留不同连接位置。糖基／肌醇核心 ≤600 Da，其余 ≤400 Da。</p><p>质量：去除脂肪酸后，甘油链位点用 H 封端。图中绿色 *:101/*:102 是外接脂肪链位置，蓝色为 P。文献天然证据只针对已核实的核心，不证明数据库的每一种链组合。</p></header>
<main><div class="toolbar"><input id="search" placeholder="搜索名称、简写、ID 或分子式"><select id="scope"><option value="passed">结构检查通过＋待判断</option value="all">全部连接图</option><option value="natural">天然核心文献依据</option><option value="nonP">非磷酸核心</option><option value="rule">规则扩展候选</option><option value="excluded">排除候选</option></select><button class="primary" id="export">导出审核 CSV</button><button id="backup">备份审核 JSON</button><label>导入备份 <input id="restore" type="file" accept=".json"></label></div>
<div class="meta">新增审核与旧人工决定分别保存。浏览器尽量自动保存；跨浏览器请导出备份。完整来源在 core_sources.jsonl.gz；本页与 Excel 均为拟议库，不会修改正在训练的数据。</div><div id="status"></div><div id="atlas"></div><div id="grid"></div></main><dialog id="zoom"><button onclick="document.getElementById('zoom').close()">关闭</button><h2 id="zoomtitle"></h2><img id="zoomimg"></dialog>
<script>const DATA=__DATA__;const KEY='glycerol-core-v01-connectivity-only-20261023';let decisions={};try{decisions=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}const $=id=>document.getElementById(id);const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let visible=[];function save(){try{localStorage.setItem(KEY,JSON.stringify(decisions))}catch(e){$('status').textContent='本地保存失败，请立即导出 JSON 备份。'}}
function status(){const vals=Object.values(decisions);$('status').textContent=`显示 ${visible.length} / ${DATA.length} 个连接图；本轮已审核 ${vals.filter(x=>x.decision).length}，保留 ${vals.filter(x=>x.decision==='保留').length}，排除 ${vals.filter(x=>x.decision==='排除').length}，待判断 ${vals.filter(x=>x.decision==='待判断').length}。`}
function render(){const q=$('search').value.trim().toLowerCase(),scope=$('scope').value;visible=DATA.filter(r=>(!q||[r.core_id,r.display_name,r.core_formula,...r.names,...r.abbreviations].join(' ').toLowerCase().includes(q))&&(scope==='all'||scope==='passed'&&r.selection_status!=='排除候选'||scope==='natural'&&r.natural_core_verified||scope==='nonP'&&r.P_count===0||scope==='rule'&&r.natural_linkage_rule_candidate||scope==='excluded'&&r.selection_status==='排除候选'));$('grid').innerHTML=visible.map(r=>{const d=decisions[r.core_id]||{};return `<article class="card"><div class="meta">${String(r.number).padStart(3,'0')} · ${esc(r.core_id)}</div><h2>${esc(r.display_name)}</h2><span class="badge ${r.natural_core_verified?'natural':''}">${esc(r.evidence_label)}</span> <span class="badge ${r.selection_status==='排除候选'?'excluded':''}">${esc(r.selection_status)}</span><img loading="lazy" src="svg/${r.core_id}.svg" data-zoom="${r.core_id}" alt="${esc(r.display_name)}"><p>${esc(r.core_formula)} · ${r.core_average_mass_Da.toFixed(1)} Da / ${r.mass_limit_Da} Da · ${r.P_count} P</p><div class="reason">${esc(r.proposed_filter_reasons.join('; '))}</div><div class="meta">原人工决定：${esc(r.manual_decisions.join('; ')||'无')} ${esc(r.manual_notes.join('; '))}<br>完整脂质结构来源 ${r.whole_lipid_source_count} 条，其中实际有酰基链 ${r.actual_acyl_lipid_source_count} 条。</div><div class="decisions">${['保留','排除','待判断'].map(v=>`<button data-id="${r.core_id}" data-v="${v}" class="${d.decision===v?'active':''}">${v}</button>`).join('')}<button data-id="${r.core_id}" data-v="">撤销本轮</button></div><input class="note" data-note="${r.core_id}" placeholder="本轮备注（可选）" value="${esc(d.note||'')}"><details><summary>SMILES、全部名称与来源</summary><p><b>甘油＋头基（链位点 H 封端）</b><br><code>${esc(r.core_smiles)}</code></p><p><b>带链连接位点的显示 SMILES</b><br><code>${esc(r.display_smiles)}</code></p><p>${esc(r.names.join('; '))}</p><div class="sources">${r.natural_core_evidence.map(x=>`<p><a target="_blank" rel="noopener" href="${esc(x.source_url)}">${esc(x.name)} · PMID ${esc(x.pmid)}</a>${esc(x.scope)}</p>`).join('')}${r.source_urls.slice(0,12).map(x=>`<a target="_blank" rel="noopener" href="${esc(x)}">${esc(x)}</a>`).join('')}</div><p>图文件：<a href="svg/${r.core_id}.svg">SVG</a> · <a href="png/${r.core_id}.png">PNG</a></p></details></article>`}).join('');status()}
$('grid').onclick=e=>{if(e.target.dataset.v!==undefined){const id=e.target.dataset.id;decisions[id]={...(decisions[id]||{}),decision:e.target.dataset.v,updated_at:new Date().toISOString()};save();render()}if(e.target.dataset.zoom){const r=DATA.find(x=>x.core_id===e.target.dataset.zoom);$('zoomtitle').textContent=r.display_name;$('zoomimg').src=`svg/${r.core_id}.svg`;$('zoom').showModal()}};$('grid').oninput=e=>{if(e.target.dataset.note){const id=e.target.dataset.note;decisions[id]={...(decisions[id]||{}),note:e.target.value,updated_at:new Date().toISOString()};save();status()}};
function download(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000)}$('backup').onclick=()=>download('glycerol_core_review.json',JSON.stringify({schema:KEY,decisions},null,2),'application/json');$('export').onclick=()=>{const quote=x=>'"'+String(x??'').replace(/"/g,'""')+'"';const rows=[['core_id','name','human_decision','human_note','selection_status','evidence_category','core_smiles'],...DATA.map(r=>[r.core_id,r.display_name,decisions[r.core_id]?.decision||'',decisions[r.core_id]?.note||'',r.selection_status,r.evidence_category,r.core_smiles])];download('glycerol_core_review.csv','\uFEFF'+rows.map(r=>r.map(quote).join(',')).join('\r\n'),'text/csv;charset=utf-8')};$('restore').onchange=async e=>{try{const x=JSON.parse(await e.target.files[0].text());if(x.schema!==KEY||!x.decisions)throw Error('不是本版备份');const ids=new Set(DATA.map(r=>r.core_id));for(const [id,d] of Object.entries(x.decisions)){if(ids.has(id))decisions[id]=d}save();render()}catch(err){alert('导入失败：'+err.message)}};$('search').oninput=render;$('scope').onchange=render;const pages=Math.ceil(DATA.filter(r=>r.selection_status!=='排除候选').length/12);$('atlas').innerHTML='PNG 图册：'+Array.from({length:pages},(_,i)=>`<a href="atlas/core_atlas_${String(i+1).padStart(2,'0')}.png" target="_blank">${i+1}</a>`).join('');render();</script></html>"""


if __name__ == "__main__":
    main()
