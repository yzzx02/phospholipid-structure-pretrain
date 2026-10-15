"""Review candidates, never approved chain rules. All support counts are distinct parents."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .pipeline import connect, get_state, write_json

ANCHORS = ["A-16:0", "A-18:0", "A-18:1", "A-20:4", "A-22:6"]
VARIANT_SCHEMA = pa.schema(
    [
        ("chain_token", pa.string()),
        ("linkage_type", pa.string()),
        ("carbon_count", pa.int32()),
        ("double_bond_count", pa.int32()),
        ("db_positions", pa.list_(pa.int32())),
        ("canonical_fragment_smiles", pa.string()),
        ("record_count", pa.int64()),
        ("unique_parent_structure_count", pa.int64()),
        ("unique_parent_classes", pa.list_(pa.string())),
        ("example_parent_names", pa.list_(pa.string())),
    ]
)
CANDIDATE_COLUMNS = [
    "chain_token",
    "candidate_fragment_smiles",
    "candidate_db_positions",
    "total_unique_parent_structures",
    "candidate_unique_parent_structures",
    "candidate_fraction",
    "number_of_positional_variants",
    "second_candidate_db_positions",
    "second_candidate_fraction",
    "classes_observed",
    "status",
    "observations",
    "cooccurring_variant_parent_excess",
]
COMPARISON_COLUMNS = [
    "chain_token",
    "status_2025",
    "representative_2025",
    "observations_2025",
    "present_2022",
    "representative_2022",
    "observations_2022",
    "version_agreement",
    "review_required",
    "status_2022",
    "other_linkage_tokens_in_other_version",
    "variant_set_equal",
]
COMPARISON_SCHEMA = pa.schema(
    [
        ("chain_token", pa.string()),
        ("status_2025", pa.string()),
        ("representative_2025", pa.string()),
        ("observations_2025", pa.int64()),
        ("present_2022", pa.bool_()),
        ("representative_2022", pa.string()),
        ("observations_2022", pa.int64()),
        ("version_agreement", pa.string()),
        ("review_required", pa.bool_()),
        ("status_2022", pa.string()),
        ("other_linkage_tokens_in_other_version", pa.list_(pa.string())),
        ("variant_set_equal", pa.bool_()),
    ]
)


def write_tsv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                    if k in columns
                }
            )


def candidate_status(total, variants, threshold, minimum):
    if total < minimum:
        return "LOW_EVIDENCE"
    if len(variants) == 1:
        return "UNIQUE"
    top, second = variants[0], variants[1]
    # A parent may support both variants. High co-occurring fractions must not imply dominance.
    if top / total >= threshold and second / total <= 1 - threshold + 1e-12 and top > second:
        return "DOMINANT"
    return "CONFLICT"


def aggregate(db, config):
    variants = []
    by_token = defaultdict(list)
    for vid, token, frag, kind, carbon, ndb, positions, n in db.execute(
        "SELECT id,token,fragment,kind,carbon,db,positions,n FROM variants ORDER BY token,fragment"
    ):
        count = db.execute("SELECT count(*) FROM vp WHERE variant=?", (vid,)).fetchone()[0]
        classes = [
            r[0]
            for r in db.execute("SELECT class FROM classes WHERE variant=? ORDER BY class", (vid,))
        ]
        names = [
            r[0]
            for r in db.execute("SELECT name FROM examples WHERE variant=? ORDER BY name", (vid,))
        ]
        row = {
            "chain_token": token,
            "linkage_type": kind,
            "carbon_count": carbon,
            "double_bond_count": ndb,
            "db_positions": json.loads(positions),
            "canonical_fragment_smiles": frag,
            "record_count": n,
            "unique_parent_structure_count": count,
            "unique_parent_classes": classes,
            "example_parent_names": names,
        }
        variants.append(row)
        by_token[token].append(row)
    candidates = {}
    for token, rows in by_token.items():
        rows.sort(
            key=lambda r: (-r["unique_parent_structure_count"], r["canonical_fragment_smiles"])
        )
        total = db.execute("SELECT count(*) FROM tp WHERE token=?", (token,)).fetchone()[0]
        first = rows[0]
        supports = [r["unique_parent_structure_count"] for r in rows]
        candidates[token] = {
            "chain_token": token,
            "candidate_fragment_smiles": first["canonical_fragment_smiles"],
            "candidate_db_positions": first["db_positions"],
            "total_unique_parent_structures": total,
            "candidate_unique_parent_structures": supports[0],
            "candidate_fraction": supports[0] / total,
            "number_of_positional_variants": len(rows),
            "second_candidate_db_positions": rows[1]["db_positions"] if len(rows) > 1 else [],
            "second_candidate_fraction": supports[1] / total if len(rows) > 1 else 0,
            "classes_observed": sorted({c for r in rows for c in r["unique_parent_classes"]}),
            "status": candidate_status(
                total,
                supports,
                config["dominant_fraction"],
                config["low_evidence_min_unique_parents"],
            ),
            "observations": sum(r["record_count"] for r in rows),
            "cooccurring_variant_parent_excess": sum(supports) - total,
        }
    return variants, candidates, by_token


def composition(fragment):
    mol = Chem.MolFromSmiles(fragment)
    # '*' has zero mass. Formulas describe the attached chain residue, not a capped free molecule.
    return rdMolDescriptors.CalcMolFormula(mol).replace("*", ""), rdMolDescriptors.CalcExactMolWt(
        mol
    )


def compare_versions(c25, c22, v25, v22):
    rows = []
    for token in sorted(c25.keys() | c22.keys()):
        a, b = c25.get(token), c22.get(token)
        exact_variants = False
        alternatives = []
        if a and b:
            sa = {r["canonical_fragment_smiles"] for r in v25[token]}
            sb = {r["canonical_fragment_smiles"] for r in v22[token]}
            exact_variants = sa == sb
            agreement = (
                "MATCH"
                if a["candidate_fragment_smiles"] == b["candidate_fragment_smiles"]
                and exact_variants
                else "POSITION_CONFLICT"
            )
        else:
            agreement = "ONLY_2025" if a else "ONLY_2022"
            present, other = (a, c22) if a else (b, c25)
            if token[0] in "OP":
                formula, _ = composition(present["candidate_fragment_smiles"])
                alternatives = [
                    t
                    for t, c in other.items()
                    if t[0] in "OP"
                    and t[0] != token[0]
                    and composition(c["candidate_fragment_smiles"])[0] == formula
                ]
                if alternatives:
                    agreement = "LINKAGE_CONFLICT"
        rows.append(
            {
                "chain_token": token,
                "status_2025": a["status"] if a else "ABSENT",
                "representative_2025": a["candidate_fragment_smiles"] if a else None,
                "observations_2025": a["observations"] if a else 0,
                "present_2022": b is not None,
                "representative_2022": b["candidate_fragment_smiles"] if b else None,
                "observations_2022": b["observations"] if b else 0,
                "version_agreement": agreement,
                "review_required": agreement != "MATCH"
                or any(c and c["status"] in {"CONFLICT", "LOW_EVIDENCE"} for c in (a, b)),
                "status_2022": b["status"] if b else "ABSENT",
                "other_linkage_tokens_in_other_version": alternatives,
                "variant_set_equal": exact_variants,
            }
        )
    return rows


def metadata_report(db, checkpoint):
    counts = checkpoint["counts"]
    result = {
        k: counts.get(k, 0)
        for k in ("total_records", "records_with_smiles", "records_without_smiles")
    }
    for field in ("smiles", "inchikey"):
        result["unique_" + field] = db.execute(f"SELECT count(*) FROM {field}").fetchone()[0]
    for field, prefix in (("top_ontology", "ontology:"), ("top_compound_class", "class:")):
        result[field] = sorted(
            (
                {"label": k[len(prefix) :], "records": v}
                for k, v in counts.items()
                if k.startswith(prefix)
            ),
            key=lambda r: -r["records"],
        )[:50]
    result.update(
        parse_failures=0,
        candidate_invalid_smiles=counts.get("rejection:invalid_smiles", 0),
        parse_failures_definition="container/schema failures; fatal and never silently skipped",
        full_scan_complete=checkpoint.get("complete", False),
        expected_records=checkpoint["expected_records"],
        source=checkpoint["identity"],
        peak_rss_bytes=checkpoint["peak_rss_bytes"],
        elapsed_seconds=checkpoint["elapsed_seconds"],
    )
    return result


def report(output, config):
    output = Path(output)
    review = output / "msdial_chain_review"
    review.mkdir(parents=True, exist_ok=True)
    versions = {}
    for version in ("2025", "2022"):
        db_path = output / "_state" / f"{version}.sqlite"
        if not db_path.exists():
            continue
        db = connect(db_path)
        checkpoint = get_state(db)
        if not checkpoint:
            db.close()
            continue
        variants, candidates, by_token = aggregate(db, config)
        pq.write_table(
            pa.Table.from_pylist(variants, VARIANT_SCHEMA),
            review / f"msdial{version}_chain_variants.parquet",
            compression="zstd",
        )
        write_tsv(
            review / f"msdial{version}_chain_representative_candidates.tsv",
            list(candidates.values()),
            CANDIDATE_COLUMNS,
        )
        meta = metadata_report(db, checkpoint)
        write_json(output / f"msdial{version}_metadata_report.json", meta)
        write_json(review / f"msdial{version}_metadata_report.json", meta)
        pe = []
        for token in candidates:
            if not token.startswith("P-"):
                continue
            supporting = []
            for vid, fragment, positions in db.execute(
                "SELECT id,fragment,positions FROM variants WHERE token=?", (token,)
            ):
                n = db.execute("SELECT count(*) FROM pe WHERE variant=?", (vid,)).fetchone()[0]
                if n:
                    names = [
                        r[0]
                        for r in db.execute("SELECT name FROM pe_examples WHERE variant=?", (vid,))
                    ]
                    supporting.append(
                        {
                            "fragment": fragment,
                            "positions": json.loads(positions),
                            "unique_parent_structures": n,
                            "examples": names,
                        }
                    )
            if supporting:
                supporting.sort(key=lambda r: (-r["unique_parent_structures"], r["fragment"]))
                union = db.execute(
                    """SELECT count(DISTINCT pe.parent) FROM pe
                    JOIN variants v ON v.id=pe.variant WHERE v.token=?""",
                    (token,),
                ).fetchone()[0]
                pe.append(
                    {
                        "source_version": version,
                        "chain_token": token,
                        "representative_db_positions": supporting[0]["positions"],
                        "unique_parent_structure_count": union,
                        "positional_variants": supporting,
                        "example_names": supporting[0]["examples"],
                    }
                )
        versions[version] = {
            "variants": variants,
            "candidates": candidates,
            "by_token": by_token,
            "checkpoint": checkpoint,
            "metadata": meta,
            "pe": pe,
        }
        db.close()
    empty = {"candidates": {}, "by_token": {}}
    a, b = versions.get("2025", empty), versions.get("2022", empty)
    comparison = compare_versions(a["candidates"], b["candidates"], a["by_token"], b["by_token"])
    write_tsv(review / "chain_version_comparison.tsv", comparison, COMPARISON_COLUMNS)
    pq.write_table(
        pa.Table.from_pylist(comparison, COMPARISON_SCHEMA),
        review / "msdial2022_chain_crosscheck.parquet",
        compression="zstd",
    )
    comparison_map = {r["chain_token"]: r for r in comparison}
    anchors = {}
    for token in ANCHORS:
        anchors[token] = {
            v: {
                "candidate": info["candidates"].get(token),
                "variants": info["by_token"].get(token, []),
            }
            for v, info in versions.items()
        }
        anchors[token]["version_comparison"] = comparison_map.get(token)
    write_json(review / "anchor_chain_validation.json", anchors)
    all_candidates = {t: c for info in versions.values() for t, c in info["candidates"].items()}
    op = []
    for p, pc in all_candidates.items():
        if not p.startswith("P-"):
            continue
        pf, pm = composition(pc["candidate_fragment_smiles"])
        for o, oc in all_candidates.items():
            if not o.startswith("O-"):
                continue
            of, om = composition(oc["candidate_fragment_smiles"])
            if pf != of:
                continue
            op.append(
                {
                    "p_token": p,
                    "o_candidate_token": o,
                    "p_fragment_formula": pf,
                    "o_fragment_formula": of,
                    "mass_equivalent": abs(pm - om) < 1e-6,
                    "observed_2025_p": p in a["candidates"],
                    "observed_2025_o": o in a["candidates"],
                    "observed_2022_p": p in b["candidates"],
                    "observed_2022_o": o in b["candidates"],
                    "status": "REVIEW_ONLY_COMPOSITION_EQUIVALENT",
                }
            )
    op_columns = [
        "p_token",
        "o_candidate_token",
        "p_fragment_formula",
        "o_fragment_formula",
        "mass_equivalent",
        "observed_2025_p",
        "observed_2025_o",
        "observed_2022_p",
        "observed_2022_o",
        "status",
    ]
    write_tsv(review / "op_equivalence_candidates.tsv", op, op_columns)
    pe = []
    for info in versions.values():
        for row in info["pe"]:
            row["version_agreement"] = comparison_map[row["chain_token"]]["version_agreement"]
            pe.append(row)
    write_tsv(
        review / "pe_plasmalogen_chain_report.tsv",
        pe,
        [
            "source_version",
            "chain_token",
            "representative_db_positions",
            "unique_parent_structure_count",
            "example_names",
            "version_agreement",
            "positional_variants",
        ],
    )
    summary = {}
    for version, info in versions.items():
        candidates, variants = info["candidates"], info["variants"]
        distributions = {}
        for kind in "AOP":
            subset = [c for t, c in candidates.items() if t.startswith(kind + "-")]
            distributions[kind] = {
                "unique_tokens": len(subset),
                "carbon_counts_by_token": dict(
                    sorted(
                        Counter(
                            int(c["chain_token"].split("-")[1].split(":")[0]) for c in subset
                        ).items()
                    )
                ),
                "double_bond_counts_by_token": dict(
                    sorted(Counter(int(c["chain_token"].split(":")[1]) for c in subset).items())
                ),
                "carbon_counts_by_observation": dict(
                    sorted(
                        Counter(
                            {
                                n: sum(
                                    r["record_count"]
                                    for r in variants
                                    if r["linkage_type"] == kind and r["carbon_count"] == n
                                )
                                for n in {
                                    r["carbon_count"] for r in variants if r["linkage_type"] == kind
                                }
                            }
                        ).items()
                    )
                ),
                "double_bond_counts_by_observation": {
                    n: sum(
                        r["record_count"]
                        for r in variants
                        if r["linkage_type"] == kind and r["double_bond_count"] == n
                    )
                    for n in sorted(
                        {r["double_bond_count"] for r in variants if r["linkage_type"] == kind}
                    )
                },
            }
        summary[version] = {
            "metadata": info["metadata"],
            "counts": info["checkpoint"]["counts"],
            "candidate_status_counts": {
                s: sum(c["status"] == s for c in candidates.values())
                for s in ("UNIQUE", "DOMINANT", "CONFLICT", "LOW_EVIDENCE")
            },
            "positional_variant_count_distribution": dict(
                Counter(c["number_of_positional_variants"] for c in candidates.values())
            ),
            "chain_distributions": distributions,
            "pe_p_unique_tokens": len(info["pe"]),
        }
    write_json(
        review / "extraction_report.json",
        {
            "versions": summary,
            "version_agreement_counts": dict(Counter(r["version_agreement"] for r in comparison)),
            "crosscheck_complete": all(
                v in versions and versions[v]["metadata"]["full_scan_complete"]
                for v in ("2025", "2022")
            ),
            "anchors": anchors,
            "op_equivalence_candidate_pairs": len(op),
            "pe_p": pe,
            "config": config,
            "training_performed": False,
            "final_library_built": False,
            "approved_chain_rules_generated": False,
            "notes": [
                "All tokens count graph C=C, including the vinyl bond for P.",
                "Source DB conventions and O-labeled vinyl ethers have separate evidence fields.",
                "Parent identity: canonical non-isomeric full SMILES; stereo is collapsed.",
                "Rejected modified chains may coexist with accepted ordinary chains in a parent.",
                "Parquet paths can be partitioned datasets; read with pyarrow.dataset.",
                "LOW_EVIDENCE precedes UNIQUE; DOMINANT requires second fraction <= 0.20.",
                "Cross-version candidate comparison is provisional until both full scans complete.",
                "LINKAGE_CONFLICT: absent exact token; opposite O/P with same residue formula.",
                "Terminal/conjugated graph DB positions are preserved pending biological review.",
            ],
        },
    )
    return summary
