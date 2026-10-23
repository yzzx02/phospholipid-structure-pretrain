"""Independent source-backed future-core collection. Never writes training assets."""

import ast
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem

from .chemistry import (
    canonical,
    core_definition,
    descriptors,
    extract_glycerol_core,
    glycosyl_rule_candidates,
    parse,
    root_phosphate,
    stable_id,
)


def read_csv(path):
    return list(csv.DictReader(Path(path).open(encoding="utf-8-sig", newline="")))


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0]) if rows else fields or ["source_id", "reason"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                    if k in fields
                }
            )


def gz_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def legacy_core(text):
    # Reuse the frozen scaffold convention, but do not apply its old Decoder
    # vocabulary/element/"intact lipid" guard to this independent reference archive.
    # Attachment remains exclusively the explicit marker-bound P atom.
    head = parse(text)
    markers = [a for a in head.GetAtoms() if a.GetAtomicNum() == 0]
    if len(markers) != 1 or markers[0].GetAtomMapNum() != 1 or markers[0].GetDegree() != 1:
        raise ValueError("unique_explicit_Polar_attachment_required")
    p = markers[0].GetNeighbors()[0]
    if p.GetAtomicNum() != 15:
        raise ValueError("Polar_attachment_is_not_P")
    scaffold = parse("[*:101]OCC(O[*:102])CO")
    offset = scaffold.GetNumAtoms()
    edit = Chem.RWMol(Chem.CombineMols(scaffold, head))
    edit.AddBond(offset + p.GetIdx(), offset - 1, Chem.BondType.SINGLE)
    edit.RemoveAtom(offset + markers[0].GetIdx())
    mol = edit.GetMol()
    Chem.SanitizeMol(mol)
    return core_definition(canonical(mol))


def collect(config):
    root = Path(config["root"])
    root.mkdir(parents=True, exist_ok=True)
    snapshot = Path(config["source_snapshot"])
    groups, source_rows, pending, source_core = {}, [], [], {}
    stages = Counter()

    def reject(source, identifier, name, reason, url=""):
        pending.append(
            {
                "source": source,
                "source_id": identifier,
                "name": name,
                "reason": reason,
                "source_url": url,
            }
        )

    def add(
        record,
        source,
        identifier,
        name,
        original,
        url,
        kind,
        abbreviation="",
        aliases=(),
        extra=None,
    ):
        cap = parse(record["capped_core_smiles"])
        key = canonical(cap, False)
        ident = stable_id(key, "GC")  # Deliberately ONE connectivity ID only.
        if key not in groups:
            mol = Chem.Mol(parse(record["core_smiles_chain_sites"]))
            Chem.RemoveStereochemistry(mol)
            no_stereo_record = core_definition(canonical(mol))
            groups[key] = {
                "core_id": ident,
                "core_smiles": key,
                "core_smiles_chain_sites": no_stereo_record["core_smiles_chain_sites"],
                "display_smiles": no_stereo_record["core_display_smiles"],
                "P_first_status": no_stereo_record["P_first_status"],
                **descriptors(key),
                "names": set(),
                "abbreviations": set(),
                "sources": set(),
                "source_ids": [],
                "source_urls": set(),
                "legacy_headgroup_ids": set(),
                "manual_decisions": set(),
                "manual_notes": set(),
                "whole_lipid_source_count": 0,
                "actual_acyl_lipid_source_count": 0,
                "natural_linkage_rule_candidate": False,
                "natural_core_evidence": [],
                "reported_synthesis_evidence": [],
                "has_defined_source_stereochemistry": False,
                "source_stereo_variants_collapsed": set(),
            }
        group = groups[key]
        for label in [name, *aliases]:
            if label:
                group["names"].add(label)
        if abbreviation:
            group["abbreviations"].add(abbreviation)
        group["sources"].add(source)
        group["source_urls"].add(url)
        group["source_ids"].append(source + ":" + str(identifier))
        group["source_stereo_variants_collapsed"].add(record["capped_core_smiles"])
        group["has_defined_source_stereochemistry"] |= "@" in record["capped_core_smiles"]
        observed = kind == "observed_chain_bearing_database_structure"
        group["whole_lipid_source_count"] += int(observed)
        group["actual_acyl_lipid_source_count"] += int(
            observed and record.get("acyl_chain_count", 0) > 0
        )
        group["natural_linkage_rule_candidate"] |= kind == "natural_linkage_rule_candidate"
        source_core[(source, str(identifier))] = key
        row = {
            "source_record_id": stable_id(source + ":" + str(identifier) + ":" + key, "SRC"),
            "core_id": ident,
            "source": source,
            "source_id": str(identifier),
            "name": name,
            "abbreviation": abbreviation,
            "aliases": list(aliases),
            "source_url": url,
            "source_kind": kind,
            "original_smiles": original,
            "source_stereo_preserved": True,
            "full_lipid_with_chains": observed,
            "actual_acyl_chains_present": observed and record.get("acyl_chain_count", 0) > 0,
            "natural_occurrence_of_this_full_species_verified": False,
            "extraction": record,
            **(extra or {}),
        }
        source_rows.append(row)
        stages[source + ":accepted_sources"] += 1
        return group

    # The full cached LMSD snapshot includes GL as well as GP. Do not redownload/copy it.
    lm_rows = []
    with (snapshot / "LM_download.tsv").open(encoding="utf-8", newline="") as stream:
        next(stream)
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["lm_id"].startswith(("LMGP", "LMGL")):
                lm_rows.append(row)
    for i, row in enumerate(lm_rows):
        ident, name = row["lm_id"], row["name"]
        url = "https://www.lipidmaps.org/databases/lmsd/" + ident
        try:
            record = extract_glycerol_core(row["smiles"])
            # Plain mono/diacylglycerols have no head; outside this head/core collection.
            if record["connectivity_smiles"] == "OCC(O)CO":
                raise ValueError("bare_glycerol_no_polar_head_out_of_scope")
            add(
                record,
                "LIPID_MAPS",
                ident,
                name,
                row["smiles"],
                url,
                "observed_chain_bearing_database_structure",
                row["abbrev"].split("(")[0].strip(),
                [row["sys_name"], row["synonyms"]],
                {
                    "pubchem_cid": row["pubchem_cid"],
                    "source_formula": row["formula"],
                    "sub_class": row["sub_class"],
                    "snapshot": "2026-10-20",
                },
            )
        except (ValueError, RuntimeError, AssertionError) as error:
            reject("LIPID_MAPS", ident, name, str(error), url)
        if i % 2000 == 0:
            print(
                json.dumps({"stage": "LMSD", "processed": i, "core_graphs": len(groups)}),
                flush=True,
            )

    records = gz_json(root / "raw/pubchem_records.json.gz")
    membership = gz_json(root / "raw/pubchem_membership.json.gz")
    for i, row in enumerate(records):
        cid = str(row["CID"])
        text = row.get("SMILES", row.get("ConnectivitySMILES", ""))
        url = "https://pubchem.ncbi.nlm.nih.gov/compound/" + cid
        names = membership.get(cid, [])
        donors = [x["name"] for x in names if x["kind"] == "sugar_parent"]
        if donors:
            candidates = glycosyl_rule_candidates(text)
            if not candidates:
                reject(
                    "PubChem_donor",
                    cid,
                    "; ".join(donors),
                    "no_explicit_cyclic_free_anomeric_OH_no_guessed_cyclization",
                    url,
                )
            for n, candidate in enumerate(candidates):
                add(
                    candidate,
                    "PubChem_linkage_rule",
                    cid + ":" + str(n),
                    "glycosylglycerol candidate from " + donors[0],
                    text,
                    url,
                    "natural_linkage_rule_candidate",
                    aliases=donors,
                    extra={
                        "donor_cid": cid,
                        "parent_is_sugar_not_observed_lipid": True,
                        "rule_evidence": "glycosyl O-glycerol linkage observed in MGDG/DGDG; extension to this donor is hypothetical",  # noqa: E501
                    },
                )
            continue
        try:
            record = extract_glycerol_core(text)
            add(
                record,
                "PubChem",
                cid,
                row.get("IUPACName", "CID " + cid),
                text,
                url,
                "observed_chain_bearing_database_structure",
                aliases=[x["name"] for x in names if x["kind"] == "full_lipid"],
                extra={
                    "query_membership": names,
                    "source_formula": row.get("MolecularFormula", ""),
                },
            )
        except (ValueError, RuntimeError, AssertionError) as error:
            reject("PubChem", cid, row.get("IUPACName", ""), str(error), url)
        if i % 3000 == 0:
            print(
                json.dumps({"stage": "PubChem", "processed": i, "core_graphs": len(groups)}),
                flush=True,
            )

    # Parse literal source rules. Dynamic head/tail construction is never guessed.
    path = snapshot / "LSG/Lipids/Classes.py"
    content = path.read_text(encoding="utf-8-sig")
    lines = content.splitlines()
    for node in ast.parse(content).body:
        if not isinstance(node, ast.ClassDef) or not any(
            isinstance(b, ast.Attribute) and b.attr == "Glycerolipid" for b in node.bases
        ):
            continue
        literals = []
        for child in ast.walk(node):
            if (
                isinstance(child, ast.keyword)
                and child.arg == "smiles"
                and isinstance(child.value, ast.Constant)
                and isinstance(child.value.value, str)
            ):
                literals.append(child.value.value)
        if not literals:
            continue
        if len(set(literals)) != 1:
            reject("LSG", node.name, node.name, "multiple_or_dynamic_head_definitions_pending")
            continue
        url = (
            "https://github.com/98104781/LSG/blob/519d004ac2c182090b845d22863faf592baa6b52/Lipids/Classes.py#L"
            + str(node.lineno)
        )
        original = literals[0]
        try:
            r = core_definition(original + "OCC(O[*:102])CO[*:101]")
            add(
                r,
                "LSG",
                node.name,
                node.name,
                original,
                url,
                "source_core_definition",
                node.name,
                extra={
                    "source_expression": "sn3.smiles + OCC(O-sn2)CO-sn1; source GenerateLipids.py line555",  # noqa: E501
                    "citation_candidates": re.findall(
                        r"10\.\d{4,9}/[^\s;]+", "\n".join(lines[node.lineno - 1 : node.end_lineno])
                    ),
                },
            )
        except (ValueError, RuntimeError, AssertionError) as error:
            reject("LSG", node.name, node.name, str(error), url)

    path = snapshot / "SmilesInchikeyGenerator.cs"
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip().startswith("//"):
            continue
        match = re.search(r'\{"([^"]+)",\s*"([^"]+)"\}', line)
        if not match or "%10" not in match[2]:
            continue
        name, original = match.groups()
        url = (
            "https://github.com/systemsomicslab/MsdialWorkbench/blob/349d7ef6edbd09e34c0eb38b849bd9fd8af79e7e/src/Common/CommonStandard/Lipidomics/SmilesInchikeyGenerator.cs#L"
            + str(line_no)
        )
        text = original.rstrip(".").replace("%10", "[*:101]").replace("%20", "[*:102]")
        try:
            r = core_definition(text)
            if r["connectivity_smiles"] == "OCC(O)CO":
                continue
            add(
                r,
                "MS-DIAL",
                name,
                name,
                original,
                url,
                "source_core_definition",
                name,
                extra={
                    "source_expression": "replace explicit source ring-closure chain sockets; no spectrum generated"  # noqa: E501
                },
            )
        except (ValueError, RuntimeError, AssertionError) as error:
            reject("MS-DIAL", name, name, str(error), url)

    # Preserve every manual decision, including contradictory votes across a now-merged graph.
    approved = {
        x["headgroup_id"]: x
        for x in read_csv(
            "outputs/headgroup_template_proposal_20261022/approved_reference_catalog.csv"
        )
    }
    decisions = read_csv(Path(config["known_library"]) / "reported_headgroups_v02.csv")
    manual_path = Path("../头基预测人工审核/headgroup_manual_review.csv")
    manual = {x["headgroup_id"]: x for x in read_csv(manual_path)} if manual_path.exists() else {}
    for row in decisions:
        try:
            r = legacy_core(row["polar_smiles"])
            group = add(
                r,
                "existing_known_v02",
                row["headgroup_id"],
                row["full_name"],
                row["polar_smiles"],
                row["source_url"],
                "existing_reference_polar",
                row["abbreviation"],
                [row["aliases"]],
                {
                    "original_is_marked_Polar_not_full_lipid": True,
                    "original_source_kind": row["source_kind"],
                    "old_builder_compatible": row["builder_compatible"],
                },
            )
            group["legacy_headgroup_ids"].add(row["headgroup_id"])
            decision = approved.get(row["headgroup_id"], manual.get(row["headgroup_id"], {}))
            if decision:
                vote = decision.get("human_decision", decision.get("decision", ""))
                if vote:
                    group["manual_decisions"].add(vote)
                note = decision.get("human_note", decision.get("note", ""))
                if note:
                    group["manual_notes"].add(note)
        except (ValueError, RuntimeError, AssertionError) as error:
            reject(
                "existing_known_v02",
                row["headgroup_id"],
                row["full_name"],
                str(error),
                row["source_url"],
            )

    # Already curated source evidence; only explicit biological records get a natural-core flag.
    natural_old = {
        "PC",
        "PE",
        "PS",
        "PI",
        "PG",
        "PA",
        "PT",
        "Ala-PG",
        "Lys-PG",
        "PIP2[3',5']",
        "PIP2[4',5']",
        "PIP2[3',4']",
        "PIP3",
        "PGP",
        "PnE",
    }
    evidence_rows = []
    for row in read_csv(snapshot / "literature_curation.csv"):
        key = source_core.get(("LIPID_MAPS", row["lm_id"])) or source_core.get(
            ("LSG", row["lsg_class"])
        )
        if row["polar_smiles"]:
            try:
                key = legacy_core(row["polar_smiles"])["connectivity_smiles"]
            except (ValueError, RuntimeError, AssertionError):
                pass
        if not key:
            continue
        group = groups.get(key)
        if not group:
            continue
        evidence = {
            "core_id": group["core_id"],
            "name": row["abbreviation"],
            "source_url": row["source_url"],
            "pmid": row["pmid"],
            "doi": row["doi"],
            "scope": row["evidence_locator"],
            "source_kind": row["source_kind"],
            "exact_acyl_species_in_database_verified": False,
            "natural_core_verified": row["abbreviation"] in natural_old,
        }
        evidence_rows.append(evidence)
        if evidence["natural_core_verified"]:
            group["natural_core_evidence"].append(evidence)
        elif "synthetic" in row["source_kind"]:
            group["reported_synthesis_evidence"].append(evidence)

    # Natural nonphosphorus connectivity evidence. Graph must match an independently sourced core.
    new_evidence = [
        (
            "DGTS",
            "LIPID_MAPS",
            "LMGL07010002",
            "10897481",
            "",
            "Natural DGTS in plant tissues; core/class evidence, not proof of every source acyl composition.",  # noqa: E501
        ),
        (
            "DGDG",
            "LIPID_MAPS",
            "LMGL05010087",
            "9363436",
            "10.1093/glycob/7.7.935",
            "Isolated B. japonicum diglycosyl diacylglycerol; independent LMSD core connectivity compared without stereo.",  # noqa: E501
        ),
        (
            "GlcADG",
            "LIPID_MAPS",
            "LMGL05010088",
            "24558041",
            "10.1074/jbc.M113.519298",
            "Agrobacterium glucuronosyl diacylglycerol identified by MS/NMR; structural core scope.",  # noqa: E501
        ),
        (
            "3-SGDG",
            "LSG",
            "SGDG",
            "36266352",
            "10.1038/s41589-022-01165-6",
            "Primary Fig2d/e: endogenous galactose and 3-position sulfate verified; synthetic SGlcDG comparator is not natural evidence. Stereo merged only for template identity.",  # noqa: E501
        ),
        (
            "MGDG",
            "LSG",
            "MGDG",
            "36266352",
            "10.1038/s41589-022-01165-6",
            "Endogenous MGDG and MGMG reported in Extended Data Fig9; natural class/core evidence only.",  # noqa: E501
        ),
        (
            "SQDG",
            "LIPID_MAPS",
            "LMGL05010009",
            "28264430",
            "10.3390/md15030057",
            "Isolated Oxyrrhis marina SQDG, NMR/MS and chemical reactions establish the 6-sulfoquinovosyl glycerol core; not every database acyl species.",  # noqa: E501
        ),
        (
            "DGTA",
            "LIPID_MAPS",
            "LMGL07010016",
            "9309879",
            "",
            "Original cell-fractionation/immunomicroscopy work reports natural DGTA in Pavlova/Ochromonas; core/class scope with independent LMSD connectivity, not each acyl species.",  # noqa: E501
        ),
        (
            "DGCC",
            "LIPID_MAPS",
            "LMGL07010010",
            "",
            "10.1016/0031-9422(94)85041-0",
            "Primary Phytochemistry 1994 abstract: DGCC isolated from Pavlova lutheri and its glyceryl-carboxyhydroxymethylcholine structure determined; independent LMSD graph.",  # noqa: E501
        ),
    ]
    for name, source, identifier, pmid, doi, scope in new_evidence:
        key = source_core.get((source, identifier))
        if key is None:
            reject("literature_mapping", pmid, name, "independent_core_source_not_extracted")
            continue
        evidence = {
            "core_id": groups[key]["core_id"],
            "name": name,
            "pmid": pmid,
            "doi": doi,
            "source_url": "https://pubmed.ncbi.nlm.nih.gov/" + pmid + "/"
            if pmid
            else "https://doi.org/" + doi,
            "scope": scope,
            "natural_core_verified": True,
            "exact_acyl_species_in_database_verified": False,
            "source_kind": "primary_natural_lipid_core_evidence",
        }
        groups[key]["natural_core_evidence"].append(evidence)
        groups[key]["abbreviations"].add(name)
        evidence_rows.append(evidence)

    flat, alias_index = [], []
    for key, group in groups.items():
        for label in group["names"] | group["abbreviations"]:
            alias_index.append({"core_id": group["core_id"], "name_or_alias": label})
        for k, v in list(group.items()):
            if isinstance(v, set):
                group[k] = sorted(v)
        group["source_count"] = len(group["source_ids"])
        group["name_or_alias_count"] = len(group["names"])
        group["names"] = sorted(group["names"], key=lambda x: (len(x), x))[:40]
        group["source_ids"] = group["source_ids"][:100]
        group["source_urls"] = group["source_urls"][:20]
        group["all_sources_and_names_in_companion_index"] = True
        group["source_stereo_variant_count_merged_not_templates"] = len(
            group.pop("source_stereo_variants_collapsed")
        )
        abbreviations = [x for x in group["abbreviations"] if len(x) <= 40]
        group["display_name"] = "; ".join(abbreviations[:8]) or min(
            group["names"], key=len, default=group["core_id"]
        )
        group["natural_core_verified"] = bool(group["natural_core_evidence"])
        group["evidence_category"] = (
            "natural_reported_core"
            if group["natural_core_verified"]
            else "database_whole_lipid_natural_unverified"
            if group["whole_lipid_source_count"]
            else "reported_synthetic_core"
            if group["reported_synthesis_evidence"]
            else "natural_linkage_rule_candidate"
            if group["natural_linkage_rule_candidate"]
            else "source_definition_or_reference_natural_unverified"
        )
        reasons = list(group["structural_filter_reasons"])
        if any(v in ("exclude", "排除") for v in group["manual_decisions"]):
            reasons.append("prior_manual_exclude_or_merged_decision_conflict")
        if any(v in ("review", "待判断", "待人工判断") for v in group["manual_decisions"]):
            reasons.append("prior_manual_review_pending")
        group["proposed_filter_reasons"] = reasons
        hard = set(reasons) - {
            "halogen_hold_for_natural_evidence",
            "prior_manual_review_pending",
            "non_neutral_core_charge_pending",
        }
        group["selection_status"] = "排除候选" if hard else "待判断" if reasons else "结构检查通过"
        group["human_decision"] = "未判断"
        group["human_note"] = ""
        flat.append(group)
    flat.sort(
        key=lambda r: (
            r["selection_status"] != "结构检查通过",
            not r["natural_core_verified"],
            r["P_count"] > 0,
            r["display_name"],
        )
    )
    dump_json(root / "core_catalog_private.json", flat)
    pq.write_table(
        pa.Table.from_pylist(alias_index),
        root / "core_name_alias_index.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist(flat), root / "glycerol_core_catalog.parquet", compression="zstd"
    )
    with gzip.open(root / "core_sources.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in source_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(root / "glycerol_core_catalog.csv", flat)
    write_csv(root / "sources_pending_review.csv", pending)
    write_csv(root / "natural_core_literature.csv", evidence_rows)
    # Excel stays bounded: one row per graph, evidence and representative whole-lipid source rows.
    sample_sources = []
    for group in flat:
        candidates = [x for x in source_rows if x["core_id"] == group["core_id"]]
        chosen = {}
        for row in candidates:
            chosen.setdefault((row["source"], row["source_kind"]), row)
        sample_sources.extend(chosen.values())
    summary = {
        "unique_connectivity_core_graphs": len(flat),
        "separate_stereochemical_template_ids": False,
        "source_records": len(source_rows),
        "source_records_by_database": dict(stages),
        "source_records_not_extracted_or_out_of_scope": len(pending),
        "source_failure_reason_counts": dict(Counter(r["reason"] for r in pending)),
        "selection_status_counts": dict(Counter(r["selection_status"] for r in flat)),
        "evidence_category_counts": dict(Counter(r["evidence_category"] for r in flat)),
        "natural_core_graphs": sum(r["natural_core_verified"] for r in flat),
        "cores_with_observed_database_acyl_lipids": sum(
            r["actual_acyl_lipid_source_count"] > 0 for r in flat
        ),
        "nonphosphorus_core_graphs": sum(r["P_count"] == 0 for r in flat),
        "nonphosphorus_structure_passed": sum(
            r["P_count"] == 0 and r["selection_status"] == "结构检查通过" for r in flat
        ),
        "rule_extension_graphs": sum(r["natural_linkage_rule_candidate"] for r in flat),
        "core_mass_limits_Da": {"sugar_or_inositol": 600, "other": 400},
        "search_coverage": [
            {k: v for k, v in q.items() if k != "cids"}
            for q in gz_json(root / "raw/pubchem_queries.json.gz")
        ],
        "LMSD_GP_GL_records_examined": len(lm_rows),
        "PubChem_CIDs_examined": len(records),
        "natural_evidence_scope": "core connectivity only; database source acyl composition is not thereby verified as naturally occurring",  # noqa: E501
        "whole_frozen_library_rewritten": False,
        "active_training_inputs_changed": False,
        "collection_complete_for_all_PubChem": False,
    }
    dump_json(root / "collection_summary.json", summary)
    authoring = root / "authoring"
    authoring.mkdir(exist_ok=True)
    dump_json(
        authoring / "workbook_payload_private.json",
        {
            "summary": summary,
            "cores": flat,
            "literature": evidence_rows,
            "representative_sources": sample_sources,
        },
    )
    return groups, summary


def annotate_frozen(config, groups):
    """Stream a derived sidecar; no original Parquet, label, split or bank mutations."""
    source = Path(config["original_library"])
    root = Path(config["root"])
    before = file_hash(source)
    pf = pq.ParquetFile(source)
    cols = [
        x
        for x in (
            "polar_id",
            "polar_fragment_smiles",
            "source",
            "structure_origin",
            "is_full_lipid_reported",
            "is_programmatically_assembled",
            "source_cids",
        )
        if x in pf.schema.names
    ]
    writer = None
    statuses, origins, reasons, root_status, mass = Counter(), Counter(), Counter(), Counter(), []
    count = 0
    natural, whole, passing = 0, 0, 0
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        output = []
        for row in batch.to_pylist():
            text = row["polar_fragment_smiles"]
            out = {
                "polar_id": row["polar_id"],
                "original_polar_smiles": text,
                "original_source": str(row.get("source", "")),
                "original_structure_origin": str(row.get("structure_origin", "")),
                "original_is_full_lipid_reported": bool(row.get("is_full_lipid_reported")),
                "original_is_programmatically_assembled": bool(
                    row.get("is_programmatically_assembled")
                ),
                "original_source_cids": str(row.get("source_cids", "")),
            }
            try:
                display, pstatus = root_phosphate(text)
                r = legacy_core(text)
                d = descriptors(r["connectivity_smiles"])
                matched = groups.get(r["connectivity_smiles"], {})
                has_natural = bool(matched.get("natural_core_verified", False))
                has_whole = bool(matched.get("whole_lipid_source_count", 0))
                ok = not d["structural_filter_reasons"]
                out.update(
                    {
                        "display_polar_P_first": display,
                        "P_first_status": pstatus,
                        "future_core_id": stable_id(r["connectivity_smiles"], "GC"),
                        "neutral_glycerol_head_smiles": r["connectivity_smiles"],
                        "core_formula": d["core_formula"],
                        "core_average_mass_Da": d["core_average_mass_Da"],
                        "mass_limit_Da": d["mass_limit_Da"],
                        "sugar_ring_count": d["sugar_ring_count"],
                        "inositol_ring_count": d["inositol_ring_count"],
                        "halogen_count": d["halogen_count"],
                        "strict_structural_policy_pass": ok,
                        "filter_reasons": ";".join(d["structural_filter_reasons"]),
                        "matched_natural_core_evidence": has_natural,
                        "matched_database_whole_lipid_core": has_whole,
                        "natural_full_species_verified": False,
                        "evidence_category": "natural_reported_core"
                        if has_natural
                        else "database_whole_lipid_core_natural_unverified"
                        if has_whole
                        else "source_rule_or_database_fragment_natural_unverified",
                    }
                )
                natural += has_natural
                whole += has_whole
                passing += ok
                mass.append(d["core_average_mass_Da"])
                statuses["evaluated"] += 1
                reasons.update(d["structural_filter_reasons"])
                root_status[pstatus] += 1
            except (ValueError, RuntimeError, AssertionError) as error:
                out.update(
                    {
                        "display_polar_P_first": "",
                        "P_first_status": "pending",
                        "future_core_id": "",
                        "neutral_glycerol_head_smiles": "",
                        "core_formula": "",
                        "core_average_mass_Da": None,
                        "mass_limit_Da": None,
                        "sugar_ring_count": 0,
                        "inositol_ring_count": 0,
                        "halogen_count": 0,
                        "strict_structural_policy_pass": False,
                        "filter_reasons": str(error),
                        "matched_natural_core_evidence": False,
                        "matched_database_whole_lipid_core": False,
                        "natural_full_species_verified": False,
                        "evidence_category": "pending",
                    }
                )
                statuses["pending"] += 1
                reasons[str(error)] += 1
            origins[out["original_structure_origin"]] += 1
            output.append(out)
        table = pa.Table.from_pylist(output)
        if writer is None:
            # Fixed floats even when an early batch happens to contain integer caps.
            for key in ("core_average_mass_Da", "mass_limit_Da"):
                idx = table.schema.get_field_index(key)
                table = table.set_column(idx, key, table[key].cast(pa.float64()))
            writer = pq.ParquetWriter(
                root / "frozen_195306_future_core_annotations.parquet.part",
                table.schema,
                compression="zstd",
            )
        writer.write_table(table.cast(writer.schema))
        count += len(output)
        if count % 4096 == 0:
            print(json.dumps({"stage": "frozen_sidecar", "processed": count}), flush=True)
    if writer:
        writer.close()
    after = file_hash(source)
    assert before == after and count == pf.metadata.num_rows
    (root / "frozen_195306_future_core_annotations.parquet.part").replace(
        root / "frozen_195306_future_core_annotations.parquet"
    )
    import numpy as np

    result = {
        "original_records": count,
        "original_parquet_sha256": before,
        "original_parquet_unchanged": True,
        "status_counts": dict(statuses),
        "original_structure_origin_counts": dict(origins),
        "filter_reason_counts_overlap": dict(reasons),
        "strict_structural_policy_pass": passing,
        "matched_natural_core_graphs_records": natural,
        "matched_database_whole_lipid_core_records": whole,
        "P_first_status_counts": dict(root_status),
        "core_mass_quantiles_Da": {
            str(q): float(np.quantile(mass, q)) for q in (0, 0.25, 0.5, 0.75, 0.95, 0.99, 1)
        },
        "no_filter_applied_to_training": True,
        "stereo_not_separately_numbered": True,
    }
    dump_json(root / "frozen_library_annotation_summary.json", result)
    return result
