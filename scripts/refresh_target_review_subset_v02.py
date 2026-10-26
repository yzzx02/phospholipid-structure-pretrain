# ruff: noqa: E501
"""Reuse validated identical SVGs after stricter scope filtering; no graph changes."""

import base64
import gzip
import io
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from render_glycerol_core_review_v01 import drawing

from structure_pretrain.glycerol_core_collection.collection import dump_json, read_csv


def main():
    root = Path("outputs/headgroup_training_targets_v02_20261026")
    page = root / "index.html"
    text = page.read_text(encoding="utf-8")
    start = text.index("const DATA=") + len("const DATA=")
    end = text.index(";const KEY=", start)
    old = json.loads(text[start:end])
    old_by_id = {r["target_id"]: r for r in old}
    encoded = re.search(r"atob\('([A-Za-z0-9+/=]+)'\)", text)
    images = json.loads(gzip.decompress(base64.b64decode(encoded[1])))
    rows = pq.read_table(root / "training_targets.parquet").to_pylist()
    view = []
    for row in rows:
        assert row["target_id"] in old_by_id
        r = old_by_id[row["target_id"]]
        assert r["target_smiles"] == row["target_smiles"]
        r.update(
            display_name=row["display_name"],
            names=row["names"],
            source_urls=row["source_urls"],
            core_smiles=row["target_smiles"],
            origin_category=row["origin_category"],
            evidence_label=row["origin_category"],
        )
        view.append(r)
    subset = {r["target_id"]: images[r["target_id"]] for r in rows}
    newzip = base64.b64encode(
        gzip.compress(json.dumps(subset, separators=(",", ":")).encode(), compresslevel=6)
    ).decode()
    # Replace the later span first, then the earlier data span so offsets stay valid.
    text = text[: encoded.start(1)] + newzip + text[encoded.end(1) :]
    text = (
        text[:start]
        + json.dumps(view, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        + text[end:]
    )
    text = text.replace(str(len(old)) + " 个连接图", str(len(rows)) + " 个连接图")
    page.write_text(text, encoding="utf-8")
    selected = [r for r in rows if r["origin_category"] == "图册新增氨基酸头基"]
    selected += [r for r in rows if r["origin_category"] == "既有头基"][:12]
    for rule in sorted({s for r in rows for s in r["rules"]}):
        selected += [r for r in rows if rule in r["rules"]][:3]
    selected = list({r["target_id"]: r for r in selected}.values())
    pages = (len(selected) + 11) // 12
    text = re.sub(r"const pages=\d+;", f"const pages={pages};", text)
    page.write_text(text, encoding="utf-8")
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 12)
    for start in range(0, len(selected), 12):
        sheet = Image.new("RGB", (1680, 1640), "#eef3f5")
        draw = ImageDraw.Draw(sheet)
        for j, r in enumerate(selected[start : start + 12]):
            x, y = (j % 3) * 560, (j // 3) * 410
            sheet.paste(
                Image.open(
                    io.BytesIO(drawing({"display_smiles": r["target_smiles"]}, True))
                ).convert("RGB"),
                (x, y + 54),
            )
            draw.text((x + 12, y + 3), r["display_name"][:45], font=font, fill="#243c50")
            draw.text(
                (x + 12, y + 29),
                f"{r['core_formula']}  {r['core_average_mass_Da']:.1f}/{r['mass_limit_Da']:.0f} Da",
                font=small,
                fill="#476477",
            )
        sheet.save(root / "atlas" / f"core_atlas_{start // 12 + 1:02}.png", optimize=True)
    qa = json.loads((root / "target_chemistry_and_visual_QA.json").read_text(encoding="utf-8"))
    qa.update(
        all_targets_checked=len(rows),
        all_target_SVGs_embedded=len(subset),
        PNG_sample_graphs=len(selected),
        PNG_atlas_pages=(len(selected) + 11) // 12,
        HTML_bytes=page.stat().st_size,
        previously_validated_graphs_reused_without_change=True,
        extra_anchor_rule_refiltered=True,
    )
    dump_json(root / "target_chemistry_and_visual_QA.json", qa)
    summary = json.loads((root / "target_summary.json").read_text(encoding="utf-8"))
    with gzip.open(
        root / "authoring/workbook_payload_private.json.gz", "wt", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "summary": summary,
                "targets": rows,
                "amino_acids": read_csv(root / "amino_acid_atlas_40_audit.csv"),
            },
            stream,
            ensure_ascii=False,
        )
    print(
        json.dumps({"targets": len(rows), "old": len(old), "PNG_samples": len(selected)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
