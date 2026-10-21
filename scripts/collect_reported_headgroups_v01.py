"""Collect actual source structures; no enumeration of hypothetical headgroups."""

import argparse
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem

from structure_pretrain.reported_headgroups.chemistry import (
    classify_polar,
    extract_full,
    lsg_fragment,
    nonisomeric_smiles,
    stable_id,
)

ROOT = Path("outputs/reported_headgroups_v01_20261020")


def read_csv(path, delimiter=","):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                }
            )


def abbrev(name):
    match = re.match(r"^([A-Za-z][A-Za-z0-9_+\-]*(?:\[[^\]]+\])?)\s*(?:\(|(?:[OP]-)?\d+[:/])", name)
    return match.group(1) if match else ""


def lsg_sources(raw):
    path = raw / "LSG/Lipids/Classes.py"
    text = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(text)
    commit = (raw / "LSG/.git/shallow").read_text().strip()
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        section = "\n".join(text.splitlines()[cls.lineno - 1 : cls.end_lineno])
        name = cls.name
        tooltip = next(
            (
                n.value.value
                for n in cls.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "tooltip" for t in n.targets)
                and isinstance(n.value, ast.Constant)
            ),
            name,
        )
        given_name = next(
            (
                n.value.value
                for n in cls.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "givenName" for t in n.targets)
                and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str)
            ),
            "",
        )
        for keyword in (
            n for n in ast.walk(cls) if isinstance(n, ast.keyword) and n.arg == "smiles"
        ):
            expr = keyword.value
            value = (
                expr.value if isinstance(expr, ast.Constant) and isinstance(expr.value, str) else ""
            )
            source_expr = ast.get_source_segment(text, expr)
            if "P" not in (value or source_expr):
                continue
            yield {
                "source": "LSG",
                "source_id_or_doi": name,
                "source_url": f"https://github.com/98104781/LSG/blob/{commit}/Lipids/Classes.py#L{keyword.lineno}",
                "full_name": tooltip,
                "abbreviation": name,
                "aliases": [given_name] if given_name else [],
                "original_smiles": value,
                "structure_input_kind": "LSG_open_fragment" if value else "dynamic_template",
                "source_expression": source_expr,
                "citation_candidates": sorted(
                    set(re.findall(r"10\.\d{4,9}/[A-Za-z0-9.()_;/\-]+", section))
                ),
                "source_kind": "source_code_structure_literature_unverified",
                "msms_status": "theoretical",
                "msms_reference": "LSG fragmentation rules; no spectra generated",
            }


def msdial_sources(raw):
    text = (raw / "SmilesInchikeyGenerator.cs").read_text(encoding="utf-8-sig")
    sha = json.loads((raw / "msdial_github_tree.json").read_text())["sha"]
    for number, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("//"):
            continue
        match = re.search(r'\{"([^\"]+)",\s*"([^\"]+)"\}', line)
        if not match or "P" not in match[2]:
            continue
        name, original = match.groups()
        # Native ring-label chain sockets become mapped dummies, not guessed tail chemistry.
        capped = re.sub(r"%([1-9][0-9])", lambda m: f"[*:{100 + int(m[1])}]", original).rstrip(".")
        yield {
            "source": "MS-DIAL",
            "source_id_or_doi": f"{sha}:{name}",
            "source_url": f"https://github.com/systemsomicslab/MsdialWorkbench/blob/{sha}/src/Common/CommonStandard/Lipidomics/SmilesInchikeyGenerator.cs#L{number}",
            "abbreviation": name,
            "full_name": name,
            "aliases": [],
            "original_smiles": original,
            "adapter_smiles": capped,
            "structure_input_kind": "MSDIAL_chain_socket_template",
            "source_kind": "source_code_structure_literature_unverified",
            "msms_status": "theoretical",
            "msms_reference": "MS-DIAL/LipidBlast structural rule; no spectrum inference",
        }


def lmsd_sources(raw):
    path = raw / "LM_download.tsv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        snapshot_date = next(f).strip()
        for row in csv.DictReader(f, delimiter="\t"):
            if "P" not in row.get("smiles", ""):
                continue
            yield {
                "source": "LIPID MAPS",
                "source_id_or_doi": row["lm_id"],
                "source_url": f"https://www.lipidmaps.org/databases/lmsd/{row['lm_id']}",
                "abbreviation": abbrev(row["abbrev"]) or abbrev(row["name"]),
                "full_name": row["name"] or row["sys_name"],
                "aliases": [x.strip() for x in row["synonyms"].split(";") if x.strip()],
                "original_smiles": row["smiles"],
                "structure_input_kind": "full_lipid",
                "source_kind": "database_only_literature_unverified",
                "pubchem_cid": row["pubchem_cid"],
                "systematic_name": row["sys_name"],
                "category": row["core"],
                "subclass": row["sub_class"],
                "source_formula": row["formula"],
                "snapshot_date": snapshot_date,
                "msms_status": "unknown",
                "msms_reference": "",
            }


def read_jsonl(path):
    if path.exists():
        return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]
    return []


def collect(root, extra=None, source_raw=None):
    # A new collection can read an immutable previous snapshot without copying
    # large downloads or writing new files into the previous raw directory.
    raw = Path(source_raw) if source_raw is not None else root / "raw"
    sources = [*lsg_sources(raw), *msdial_sources(raw), *lmsd_sources(raw)]
    sources += read_jsonl(raw / "literature_structures.jsonl")
    sources += read_jsonl(raw / "pubchem_structures.jsonl")
    sources += read_jsonl(raw / "msdial_lipidblast_structures.jsonl")
    sources += read_jsonl(raw / "user_imports.jsonl")
    sources += read_jsonl(raw / "literature_pending.jsonl")
    if extra:
        sources += read_jsonl(extra)
    groups, links, pending, mapping, seen_links = {}, [], [], {}, set()
    audit = Counter()
    for index, source in enumerate(sources):
        if index % 1000 == 0:
            print(
                json.dumps({"source_records": index, "unique_polar_graphs": len(groups)}),
                flush=True,
            )
        kind = source["structure_input_kind"]
        try:
            if kind == "LSG_open_fragment":
                results = [lsg_fragment(source["original_smiles"])]
            elif kind == "standard_polar":
                mol = Chem.MolFromSmiles(source["original_smiles"])
                if mol is None or not mol.GetNumAtoms():
                    raise ValueError("invalid_source_polar")
                if any(a.GetIsotope() for a in mol.GetAtoms()):
                    raise ValueError("isotope_label_requires_explicit_review")
                dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
                if (
                    len(dummies) != 1
                    or dummies[0].GetDegree() != 1
                    or dummies[0].GetNeighbors()[0].GetAtomicNum() != 15
                ):
                    raise ValueError("explicit_single_attachment_P_required")
                dummies[0].SetAtomMapNum(1)
                polar = nonisomeric_smiles(mol)
                results = [
                    {
                        "polar_smiles": polar,
                        "boundary_method": "explicit_source_polar_attachment",
                        "roundtrip_graph": False,
                        "roundtrip_formula": False,
                        **classify_polar(polar),
                    }
                ]
            elif kind in ("full_lipid", "MSDIAL_chain_socket_template"):
                results = extract_full(source.get("adapter_smiles", source["original_smiles"]))
            else:
                results = []
            if not results:
                raise ValueError(
                    "dynamic_template_needs_actual_structure"
                    if kind == "dynamic_template"
                    else "no_unambiguous_chain_bearing_glycerol_O_P_boundary"
                )
        except (ValueError, RuntimeError) as exc:
            audit["unresolved_source_records"] += 1
            pending.append(
                {
                    **source,
                    "review_id": stable_id(source["source"] + source["source_id_or_doi"], "REV"),
                    "review_reason": str(exc),
                    "headgroup_id": "",
                }
            )
            continue
        audit["resolved_source_records"] += 1
        for result in results:
            polar = result["polar_smiles"]
            identifier = stable_id(polar)
            entry = groups.setdefault(
                identifier,
                {
                    "headgroup_id": identifier,
                    "polar_smiles": polar,
                    "names": set(),
                    "abbreviations": set(),
                    "head_names": set(),
                    "sources": [],
                    "roundtrip_graph_source_count": 0,
                    "roundtrip_formula_source_count": 0,
                    **classify_polar(polar),
                },
            )
            names = [
                source["full_name"],
                source.get("systematic_name", ""),
                *source.get("aliases", []),
            ]
            ab = source.get("abbreviation", "")
            if ab:
                entry["abbreviations"].add(ab)
                names.append(ab)
            if source["source"] in ("LSG", "Literature"):
                entry["head_names"].add(source["full_name"])
            for name in names:
                if name:
                    entry["names"].add(name)
                    norm = " ".join(name.casefold().split())
                    mapping[(norm, identifier)] = {
                        "query_name": name,
                        "normalized_name": norm,
                        "headgroup_id": identifier,
                        "polar_smiles": polar,
                        "name_kind": "abbreviation" if name == ab else "source_name_or_alias",
                    }
            key = (identifier, source["source"], source["source_id_or_doi"])
            if key in seen_links:
                continue
            seen_links.add(key)
            link = {
                **source,
                **result,
                "headgroup_id": identifier,
                "source_record_id": stable_id("|".join(key), "SRC"),
            }
            links.append(link)
            entry["sources"].append(link)
            entry["roundtrip_graph_source_count"] += int(result.get("roundtrip_graph", False))
            entry["roundtrip_formula_source_count"] += int(result.get("roundtrip_formula", False))
    frozen = (
        pq.read_table(
            "data/structure_library_v0.3/polar_library.parquet", columns=["polar_fragment_smiles"]
        )
        .column(0)
        .to_pylist()
    )
    frozen = frozenset(frozen)
    heads = []
    for entry in groups.values():
        preferred = {
            "PC",
            "PE",
            "PA",
            "PG",
            "PI",
            "PS",
            "PT",
            "MMPE",
            "DMPE",
            "PPA",
            "CDPDG",
            "Am-PE",
            "Am-PS",
            "PtdGlc",
        }
        refs = sorted(
            entry.pop("sources"),
            key=lambda x: (
                x["source"] != "Literature",
                x["source"] != "LSG",
                x.get("abbreviation", "") not in preferred,
                len(x.get("abbreviation", "")),
                x["source_id_or_doi"],
            ),
        )
        names, abs_, headnames = (
            sorted(entry.pop("names")),
            sorted(entry.pop("abbreviations")),
            sorted(entry.pop("head_names")),
        )
        primary = refs[0]
        head_aliases = [
            alias
            for ref in refs
            if ref["source"] in ("LSG", "Literature", "PubChem")
            for alias in ref.get("aliases", [])
        ]
        main_ab = next(
            (s.get("abbreviation", "") for s in refs if s.get("abbreviation", "") in preferred),
            min(
                abs_,
                key=lambda a: (
                    a.startswith(("o", "p", "Ether", "L", "Ox", "Cer", "As")),
                    len(a),
                    a,
                ),
            )
            if abs_
            else "",
        )
        row = {
            **entry,
            "abbreviation": main_ab,
            "full_name": primary["full_name"],
            "aliases": "; ".join(sorted(set(abs_ + headnames + head_aliases) - {main_ab})),
            "original_smiles": primary["original_smiles"],
            "source": "; ".join(sorted({s["source"] for s in refs})),
            "source_id_or_doi": primary["source_id_or_doi"],
            "source_url": primary["source_url"],
            "source_kind": "; ".join(sorted({s["source_kind"] for s in refs})),
            "source_count": len(refs),
            "name_alias_count": len({" ".join(name.casefold().split()) for name in names}),
            "msms_status": "; ".join(sorted({s.get("msms_status", "unknown") for s in refs})),
            "in_frozen_195306": entry["polar_smiles"] in frozen,
            "literature_verified": any(s.get("literature_verified") is True for s in refs),
            "P_count": sum(
                a.GetAtomicNum() == 15 for a in Chem.MolFromSmiles(entry["polar_smiles"]).GetAtoms()
            ),
            "review_reason": "" if entry["builder_compatible"] else entry["incompatibility_reason"],
        }
        if "PECM" in abs_ and "COC(=O)N" in row["polar_smiles"]:
            row["review_reason"] = (
                "LSG_PECM_source_label_says_carboxymethyl_but_graph_is_methyl_carbamate"
            )
        heads.append(row)
        if row["review_reason"]:
            pending.append({**row, "review_id": stable_id(row["headgroup_id"], "REV")})
    heads.sort(key=lambda x: (x["abbreviation"] or "~", x["headgroup_id"]))
    summary = {
        "source_records": len(sources),
        "distinct_resolved_polar_graphs": len(heads),
        "accurate_polar_smiles": len(heads),
        "builder_compatible": sum(h["builder_compatible"] for h in heads),
        "literature_verified_graphs": sum(h["literature_verified"] for h in heads),
        "source_database_only_or_literature_unverified_graphs": sum(
            not h["literature_verified"] for h in heads
        ),
        "graphs_with_abbreviation": sum(bool(h["abbreviation"]) for h in heads),
        "english_or_systematic_name_only": sum(not h["abbreviation"] for h in heads),
        "in_frozen_195306": sum(h["in_frozen_195306"] for h in heads),
        "frozen_polar_rows": len(frozen),
        "pending_review_rows": len(pending),
        "source_mapping_rows": len(links),
        "name_mapping_rows": len(mapping),
        "source_overlap_graph_counts": {
            s: sum(s in h["source"].split("; ") for h in heads)
            for s in sorted({x["source"] for x in sources})
        },
        "all_source_record_outcomes": dict(audit),
        "training_performed": False,
        "spectra_generated": False,
        "theoretical_headgroups_enumerated": False,
        "identity": (
            "rooted nonisomeric canonical graph; "
            "original stereochemistry retained in source records"
        ),
    }
    write_csv(root / "reported_headgroups_v01.csv", heads)
    name_rows = sorted(mapping.values(), key=lambda x: (x["normalized_name"], x["headgroup_id"]))
    ordinals = Counter()
    for name in name_rows:
        ordinals[name["normalized_name"]] += 1
        name["query_match_index"] = ordinals[name["normalized_name"]]
        name["query_lookup_key"] = name["normalized_name"] + "|" + str(name["query_match_index"])
    summary["ambiguous_normalized_names"] = sum(n > 1 for n in ordinals.values())
    write_csv(root / "headgroup_name_alias_mapping.csv", name_rows)
    write_csv(root / "headgroup_sources.csv", links)
    write_csv(root / "headgroups_pending_review.csv", pending)
    (root / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--extra-jsonl", type=Path)
    args = p.parse_args()
    collect(args.root, args.extra_jsonl)
