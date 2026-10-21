"""Verify immutable graph identity, rendered coverage and saved workbook contents."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image
from rdkit import Chem


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = json.loads((root / "filter_summary.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == summary["input_sha256"]
    source = {r["headgroup_id"]: r for r in read_rows(args.source)}
    rows = read_rows(root / "headgroup_filter_review.csv")
    counts = Counter(r["auto_status"] for r in rows)
    assert dict(counts) == summary["counts"]
    assert {r["headgroup_id"] for r in rows} == source.keys()
    assert len(rows) == len(source) == summary["input_graphs"]
    expansion = summary.get("expansion", False)
    active_rows = [r for r in rows if r.get("is_new") == "True"] if expansion else rows
    active_counts = Counter(r["auto_status"] for r in active_rows)
    initial_decisions = summary.get("initial_decisions", {})
    for status, name in (
        ("retain", "headgroups_retained.csv"),
        ("exclude", "headgroups_excluded.csv"),
        ("review", "headgroups_manual_review.csv"),
    ):
        assert {r["headgroup_id"] for r in read_rows(root / name)} == {
            r["headgroup_id"] for r in active_rows if r["auto_status"] == status
        }
    html = (root / "index.html").read_text(encoding="utf-8")
    embedded = json.loads(html.split("const DATA=", 1)[1].split(";\nconst SUMMARY=", 1)[0])
    assert len(embedded) == len(rows)
    for row in rows:
        assert row["polar_smiles"] == source[row["headgroup_id"]]["polar_smiles"]
        expected = initial_decisions.get(row["headgroup_id"], {})
        assert row["human_decision"] == expected.get("decision", "")
        assert row["human_note"] == expected.get("note", "")
        mol = Chem.MolFromSmiles(row["polar_smiles"])
        svg = (root / row["svg_path"]).read_text(encoding="utf-8")
        ET.fromstring(svg)
        drawn = {int(i) for i in re.findall(r"atom-(\d+)", svg)}
        assert drawn == set(range(mol.GetNumAtoms())), row["headgroup_id"]
        with Image.open(root / row["png_path"]) as image:
            image.verify()
        if row["auto_status"] == "retain":
            assert row["builder_compatible"] == "True"
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(root / "headgroup_structure_review.xlsx") as book:
        strings = ET.fromstring(book.read("xl/sharedStrings.xml"))
        strings = ["".join(s.itertext()) for s in strings]
        media = [n for n in book.namelist() if n.startswith("xl/media/")]
        expected_groups = [
            [r for r in active_rows if r["auto_status"] != "exclude"],
            [r for r in active_rows if r["auto_status"] == "exclude"],
        ]
        if expansion:
            expected_groups.append([r for r in rows if r["prior_approved"] == "True"])
        assert len(media) == sum(map(len, expected_groups))
        for i, group in enumerate(expected_groups, 1):
            expected = len(group)
            xml = ET.fromstring(book.read(f"xl/worksheets/sheet{i}.xml"))
            cells = {c.get("r"): c for c in xml.findall(".//s:c", ns)}
            for row_index in range(7, expected + 7):
                cell = cells[f"D{row_index}"]
                value = cell.find("s:v", ns).text
                if cell.get("t") == "s":
                    value = strings[int(value)]
                choice = group[row_index - 7]["human_decision"]
                assert value == {"keep": "保留", "exclude": "排除", "review": "待判断"}.get(
                    choice, "未判断"
                )
                id_cell = cells[f"H{row_index}"]
                saved_id = id_cell.find("s:v", ns).text
                if id_cell.get("t") == "s":
                    saved_id = strings[int(saved_id)]
                assert saved_id == group[row_index - 7]["headgroup_id"]
            assert float(cells["I4"].find("s:v", ns).text) == sum(
                bool(r["human_decision"]) for r in group
            )
            if i == 1:
                assert [int(cells[a].find("s:v", ns).text) for a in ("B4", "D4", "F4")] == [
                    active_counts[k] for k in ("retain", "exclude", "review")
                ]
    atlas_paths = (
        summary["remaining_atlas_pages"]
        + summary["excluded_atlas_pages"]
        + summary.get("approved_atlas_pages", [])
    )
    for rel in atlas_paths:
        with Image.open(root / rel) as image:
            image.verify()
        ET.parse(root / rel.replace("atlas_png/", "atlas_svg/").replace(".png", ".svg"))
    qa = {
        "all_source_graphs_unchanged": True,
        "all_atom_indices_present_in_svg": True,
        "unique_graphs": len(rows),
        "partition": dict(counts),
        "embedded_workbook_images": len(media),
        "saved_workbook_human_decisions_preserved": True,
        "workbook_cached_counts_verified": True,
        "individual_svg_png_pairs": len(rows),
        "remaining_atlas_pages": len(summary["remaining_atlas_pages"]),
        "excluded_atlas_pages": len(summary["excluded_atlas_pages"]),
        "source_sha256": summary["input_sha256"],
        "new_graphs": len(active_rows) if expansion else 0,
        "new_partition": dict(active_counts) if expansion else {},
        "previous_human_kept": sum(d.get("decision") == "keep" for d in initial_decisions.values()),
    }
    (root / "authoring/delivery_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Enumerate only deliverable folders, never the authoring/node_modules junction.
    deliverables = [
        p for p in root.iterdir() if p.is_file() and p.suffix not in {".zip", ".ndjson"}
    ]
    for folder in ("svg", "png", "atlas_svg", "atlas_png"):
        deliverables.extend(p for p in (root / folder).iterdir() if p.is_file())
    archive = root / "headgroup_structure_review_bundle.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
        for file in deliverables:
            bundle.write(file, file.relative_to(root))
    qa["bundle_megabytes"] = round(archive.stat().st_size / 1024**2, 2)
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
