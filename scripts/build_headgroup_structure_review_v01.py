"""Draw and review a non-destructive, single-anchor selection of the v0.1 library."""

import argparse
import csv
import hashlib
import html
import io
import json
import math
import re
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, rdBase
from rdkit.Chem import rdCoordGen, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from structure_pretrain.reported_headgroups.single_anchor_review import analyze_single_anchor
from structure_pretrain.reported_headgroups.source_review_v02 import phosphorus_review_flags

LABELS = {"retain": "初筛保留", "exclude": "初筛排除", "review": "待人工判断"}
COLORS = {"retain": "#245c49", "exclude": "#924438", "review": "#875c16"}
ROOT_BLUE = (0.20, 0.63, 0.82)
EXTRA_RED = (0.96, 0.64, 0.58)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(row[k], ensure_ascii=False)
                    if isinstance(row[k], (dict, list))
                    else row[k]
                    for k in fields
                }
            )


def depict(row, out):
    mol = Chem.MolFromSmiles(row["polar_smiles"])
    original = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    complex_polar = mol.GetNumAtoms() >= 45 and row["auto_status"] != "exclude"
    if complex_polar:
        rdCoordGen.AddCoords(mol)
    else:
        rdDepictor.Compute2DCoords(mol)
    width, height = (1600, 1000) if complex_polar or mol.GetNumAtoms() > 90 else (1100, 700)
    extra = set(row["highlight_extra_atoms"])
    roots = {row["attachment_atom"], row["attachment_P_atom"]}
    atom_colors = {i: EXTRA_RED for i in extra}
    atom_colors.update({i: ROOT_BLUE for i in roots})
    bond_colors = {
        b.GetIdx(): EXTRA_RED
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in extra and b.GetEndAtomIdx() in extra
    }
    bond_colors.update(
        {
            b.GetIdx(): ROOT_BLUE
            for b in mol.GetBonds()
            if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} == roots
        }
    )
    images = {}
    for kind, drawer_cls in (
        ("svg", rdMolDraw2D.MolDraw2DSVG),
        ("png", rdMolDraw2D.MolDraw2DCairo),
    ):
        drawer = drawer_cls(width, height)
        options = drawer.drawOptions()
        options.padding = 0.09
        options.bondLineWidth = 2.2
        options.fixedFontSize = 22
        options.addStereoAnnotation = False
        options.atomLabels[row["attachment_atom"]] = "*:1"
        options.fontFile = str(Path("C:/Windows/Fonts/arial.ttf"))
        drawer.DrawMolecule(
            mol,
            highlightAtoms=sorted(atom_colors),
            highlightAtomColors=atom_colors,
            highlightBonds=sorted(bond_colors),
            highlightBondColors=bond_colors,
        )
        drawer.FinishDrawing()
        data = drawer.GetDrawingText()
        dest = out / kind / (row["headgroup_id"] + "." + kind)
        if kind == "svg":
            dest.write_text(data, encoding="utf-8")
            # Prefix all element classes with a unique molecule ID only in the HTML.
            svg = re.sub(r"<\?xml.*?\?>|<!DOCTYPE.*?>", "", data, flags=re.S)
            svg = re.sub(
                r"width='[^']+' height='[^']+'", "width='100%' height='100%'", svg, count=1
            )
            images["svg"] = svg
            ET.fromstring(data)
        else:
            dest.write_bytes(data)
            with Image.open(io.BytesIO(data)) as image:
                assert image.size == (width, height)
                assert image.getbbox()
        images[kind + "_path"] = str(dest.relative_to(out)).replace("\\", "/")
    assert original == Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return images


def wrap_text(text, draw, font, width, max_lines=3):
    lines, line = [], ""
    for char in str(text):
        if draw.textlength(line + char, font=font) > width:
            lines.append(line)
            line = ""
        line += char
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-2] + "…"
    return lines


def make_atlas(rows, out, label, title):
    # Six structures per page leave enough room for sugars and polyphosphates.
    title_font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 34)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 23)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    page_paths = []
    for page_index, start in enumerate(range(0, len(rows), 6), 1):
        batch = rows[start : start + 6]
        w, h, tile_w, tile_h, gap = 2080, 2390, 1000, 720, 24
        image = Image.new("RGB", (w, h), "#f2f4f7")
        draw = ImageDraw.Draw(image)
        draw.text(
            (32, 24),
            f"{title}  {page_index:02d}/{math.ceil(len(rows) / 6):02d}",
            fill="#20384d",
            font=title_font,
        )
        draw.text(
            (32, 82),
            "蓝色为 P-[*:1] 连接位点；红色为额外长链/甾醇；图保持原始 Polar 连接",
            fill="#526273",
            font=font,
        )
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">',
            f'<rect width="{w}" height="{h}" fill="#f2f4f7"/>',
            '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif}</style>',
            '<text x="32" y="56" font-size="34" fill="#20384d">'
            f"{html.escape(title)} {page_index:02d}</text>",
            '<text x="32" y="106" font-size="23" fill="#526273">'
            "蓝色：P-[*:1] 连接位点；红色：额外脂质锚点</text>",
        ]
        for offset, row in enumerate(batch):
            x = 28 + (offset % 2) * (tile_w + gap)
            y = 150 + (offset // 2) * (tile_h + gap)
            draw.rounded_rectangle(
                (x, y, x + tile_w, y + tile_h), radius=12, fill="white", outline="#dce2e9", width=2
            )
            name = row["display_name"]
            text = f"{row['review_number']}   {name}"
            draw.text((x + 20, y + 16), text[:65], fill="#20384d", font=font)
            draw.text((x + 20, y + 52), row["headgroup_id"], fill="#617487", font=small)
            with Image.open(out / row["png_path"]) as mol:
                mol.thumbnail((tile_w - 28, 500), Image.Resampling.LANCZOS)
                image.paste(
                    mol.convert("RGB"),
                    (x + (tile_w - mol.width) // 2, y + 92 + (500 - mol.height) // 2),
                )
            status = LABELS[row["auto_status"]]
            draw.text((x + 20, y + 594), status, fill=COLORS[row["auto_status"]], font=font)
            reason = wrap_text(row["filter_reason"], draw, font, tile_w - 40, max_lines=2)
            for j, line in enumerate(reason):
                draw.text((x + 20, y + 632 + j * 31), line, fill="#526273", font=font)
            svg = ET.fromstring(row["svg"])
            svg.set("x", str(x + 14))
            svg.set("y", str(y + 92))
            svg.set("width", str(tile_w - 28))
            svg.set("height", "500")
            svg_parts += [
                f'<rect x="{x}" y="{y}" width="{tile_w}" height="{tile_h}" '
                'rx="12" fill="white" stroke="#dce2e9"/>',
                f'<text x="{x + 20}" y="{y + 42}" font-size="23" fill="#20384d">'
                f"{html.escape(text[:65])}</text>",
                f'<text x="{x + 20}" y="{y + 76}" font-size="20" fill="#617487">'
                f"{row['headgroup_id']}</text>",
                ET.tostring(svg, encoding="unicode"),
                f'<text x="{x + 20}" y="{y + 620}" font-size="23" '
                f'fill="{COLORS[row["auto_status"]]}">{status}</text>',
            ]
            for j, line in enumerate(reason):
                svg_parts.append(
                    f'<text x="{x + 20}" y="{y + 660 + j * 31}" font-size="23" '
                    f'fill="#526273">{html.escape(line)}</text>'
                )
        image.save(out / "atlas_png" / f"{label}_{page_index:02d}.png", dpi=(180, 180))
        svg_parts.append("</svg>")
        svg_path = out / "atlas_svg" / f"{label}_{page_index:02d}.svg"
        svg_path.write_text("\n".join(svg_parts), encoding="utf-8")
        ET.parse(svg_path)
        page_paths.append(f"atlas_png/{label}_{page_index:02d}.png")
    return page_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/reported_headgroups_v01_20261020/reported_headgroups_v01.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reported_headgroups_single_anchor_review_20261021"),
    )
    parser.add_argument("--refresh-rendered-artifacts", action="store_true")
    parser.add_argument("--previous-csv", type=Path)
    parser.add_argument("--human-review-json", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if (out / "index.html").exists() and not args.refresh_rendered_artifacts:
        raise ValueError("Review output exists; choose a new --output to preserve manual decisions")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("svg", "png", "atlas_png", "atlas_svg", "authoring"):
        (out / name).mkdir(exist_ok=True)
    before = sha256(args.input)
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    previous_ids, initial_decisions, previous_numbers = set(), {}, {}
    expansion = args.previous_csv is not None
    if expansion:
        with args.previous_csv.open(encoding="utf-8-sig", newline="") as stream:
            previous_ids = {r["headgroup_id"] for r in csv.DictReader(stream)}
        if args.human_review_json is None:
            raise ValueError("An expansion must retain its explicit previous human review")
        review = json.loads(args.human_review_json.read_text(encoding="utf-8-sig"))
        assert review["input_sha256"] == sha256(args.previous_csv)
        initial_decisions = review["decisions"]
        with args.human_review_json.with_suffix(".csv").open(encoding="utf-8-sig") as stream:
            previous_numbers = {
                r["headgroup_id"]: r["review_number"] for r in csv.DictReader(stream)
            }
    for row in rows:
        row["builder_compatible"] = row["builder_compatible"].lower() == "true"
        row["literature_verified"] = row["literature_verified"].lower() == "true"
        row.update(
            analyze_single_anchor(
                row["polar_smiles"],
                row["abbreviation"],
                row["builder_compatible"],
                row["review_reason"],
            )
        )
        mol = Chem.MolFromSmiles(row["polar_smiles"])
        row["carbon_count"] = sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
        row["P_count"] = sum(a.GetAtomicNum() == 15 for a in mol.GetAtoms())
        row["is_new"] = expansion and row["headgroup_id"] not in previous_ids
        row["prior_approved"] = (
            initial_decisions.get(row["headgroup_id"], {}).get("decision") == "keep"
        )
        # Keep the source class as an alias, but do not label isoLG adducts as PE.
        iso_adduct = re.search(r"15-isoLG (.+)$", row["full_name"])
        row["display_name"] = (
            f"15-isoLG {iso_adduct.group(1)} (PE adduct)"
            if row["abbreviation"] == "PE" and iso_adduct
            else row["abbreviation"]
            or (
                row["full_name"]
                if len(row["full_name"]) < 100
                else "Polar " + row["headgroup_id"][3:11]
            )
        )
        if row["is_new"] and row["auto_status"] != "exclude":
            flags = phosphorus_review_flags(row["polar_smiles"])
            if flags:
                row["auto_status"] = "review"
                row["reason_codes"].append("unusual_database_P_bonding")
                row["filter_reason"] += "；" + "；".join(flags)
        row["source_urls"] = list(
            dict.fromkeys(re.findall(r"https?://[^\s;\"\],]+", row["source_url"]))
        )[:8]
    priority = {
        "PC": 0,
        "PE": 1,
        "PA": 2,
        "PS": 3,
        "PG": 4,
        "PI": 5,
        "PIP": 6,
        "PIP2": 7,
        "PIP3": 8,
    }
    order = {"retain": 0, "review": 1, "exclude": 2}
    rows.sort(
        key=lambda r: (
            0 if r["is_new"] else 1 if r["prior_approved"] else 2,
            order[r["auto_status"]],
            priority.get(r["abbreviation"], 50),
            r["abbreviation"].casefold(),
            r["headgroup_id"],
        )
    )
    ids = {r["headgroup_id"] for r in rows}
    assert len(ids) == len(rows)
    new_index = 0
    for index, row in enumerate(rows, 1):
        if row["is_new"]:
            new_index += 1
            row["review_number"] = f"N{new_index:03d}"
        else:
            row["review_number"] = previous_numbers.get(row["headgroup_id"], f"R{index:03d}")
        initial = initial_decisions.get(row["headgroup_id"], {})
        row["human_decision"] = initial.get("decision", "")
        row["human_note"] = initial.get("note", "")
        row["search_text"] = " ".join(
            str(row[k]) for k in ("headgroup_id", "abbreviation", "full_name", "aliases")
        ).casefold()
        row.update(depict(row, out))
    counts = dict(Counter(r["auto_status"] for r in rows))
    active_rows = [r for r in rows if r["is_new"]] if expansion else rows
    remaining = [r for r in active_rows if r["auto_status"] != "exclude"]
    excluded = [r for r in active_rows if r["auto_status"] == "exclude"]
    summary = {
        "input_graphs": len(rows),
        "counts": counts,
        "remaining_graphs": len(remaining),
        "remaining_builder_compatible": sum(r["builder_compatible"] for r in remaining),
        "retained_builder_compatible": sum(
            r["builder_compatible"] for r in rows if r["auto_status"] == "retain"
        ),
        "status_by_reason": {
            status: dict(
                Counter(
                    code for r in rows if r["auto_status"] == status for code in r["reason_codes"]
                )
            )
            for status in order
        },
        "status_by_abbreviation": {
            status: dict(
                Counter(
                    r["abbreviation"] or "(unnamed)" for r in rows if r["auto_status"] == status
                )
            )
            for status in order
        },
        "input_sha256": before,
        "source_library_unchanged": True,
        "filter_version": "single-anchor-v02-20261021",
        "rdkit_version": rdBase.rdkitVersion,
        "long_chain_rule": (
            "linear path >=8 non-ring non-aromatic C bearing only C/H, or a branched "
            "component >=8 C with a path >=6; explicit CL/BMP/LBPA/NAPE/NAPS classes, "
            "second diacylated glycerol arms and sterol anchors additionally excluded"
        ),
        "review_rule": (
            "5-7 carbon hydrophobic segments, short/amino acyl esters, existing Builder "
            "failures or source naming review; multiple P and sugar rings alone are allowed"
        ),
        "stereochemistry": (
            "current model nonisomeric Polar; "
            "original stereochemistry remains in the versioned source records"
        ),
        "training_performed": False,
    }
    if expansion:
        summary.update(
            {
                "expansion": True,
                "previous_graphs": len(previous_ids),
                "new_graphs": len(active_rows),
                "new_counts": dict(Counter(r["auto_status"] for r in active_rows)),
                "initial_decisions": initial_decisions,
                "previous_approved_graphs": sum(r["prior_approved"] for r in rows),
                "new_literature_verified_graphs": sum(
                    r["literature_verified"] for r in active_rows
                ),
            }
        )
    csv_fields = [
        "review_number",
        "headgroup_id",
        "abbreviation",
        "display_name",
        "full_name",
        "auto_status",
        "human_decision",
        "human_note",
        "filter_reason",
        "polar_smiles",
        "builder_compatible",
        "incompatibility_reason",
        "literature_verified",
        "reason_codes",
        "longest_hydrocarbon_path",
        "short_or_amino_acyl_count",
        "sterol_anchor_detected",
        "carbon_count",
        "P_count",
        "svg_path",
        "png_path",
        "source",
        "source_id_or_doi",
        "source_url",
        "aliases",
        "original_smiles",
        "review_reason",
        "is_new",
        "prior_approved",
    ]
    write_csv(out / "headgroup_filter_review.csv", rows, csv_fields)
    write_csv(
        out / "headgroups_retained.csv",
        [r for r in active_rows if r["auto_status"] == "retain"],
        csv_fields,
    )
    write_csv(out / "headgroups_excluded.csv", excluded, csv_fields)
    write_csv(
        out / "headgroups_manual_review.csv",
        [r for r in active_rows if r["auto_status"] == "review"],
        csv_fields,
    )
    summary["remaining_atlas_pages"] = make_atlas(remaining, out, "remaining", "保留及待判断结构")
    summary["excluded_atlas_pages"] = make_atlas(excluded, out, "excluded", "初筛排除结构")
    if expansion:
        approved = [r for r in rows if r["prior_approved"]]
        summary["approved_atlas_pages"] = make_atlas(
            approved, out, "approved", "上批已人工确认结构"
        )
        write_csv(out / "previous_approved_headgroups.csv", approved, csv_fields)
        write_csv(out / "new_review_candidates.csv", remaining, csv_fields)
    template = (
        Path(__file__).with_name("headgroup_structure_review.html").read_text(encoding="utf-8")
    )
    page = template.replace(
        "__DATA__", json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    ).replace("__SUMMARY__", json.dumps(summary, ensure_ascii=False).replace("<", "\\u003c"))
    (out / "index.html").write_text(page, encoding="utf-8")
    # Workbook needs only the flat review fields and image paths, not the entire provenance payload.
    payload = {"summary": summary, "rows": [{k: r[k] for k in csv_fields} for r in rows]}
    (out / "authoring/review_payload_private.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    (out / "filter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    assert sha256(args.input) == before
    report_counts = summary.get("new_counts", counts)
    report = (
        Path(__file__)
        .with_name(
            "headgroup_expansion_review_notes.md"
            if expansion
            else "headgroup_structure_review_notes.md"
        )
        .read_text(encoding="utf-8")
        .format(
            input_graphs=len(active_rows),
            retain=report_counts.get("retain", 0),
            exclude=report_counts.get("exclude", 0),
            review=report_counts.get("review", 0),
            remaining=len(remaining),
        )
    )
    (out / "README_review.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
