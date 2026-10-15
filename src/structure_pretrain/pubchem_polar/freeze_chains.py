import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from structure_pretrain.library_builder.chain_library import import_chains


def freeze_chains(review_dir: Path, output: Path) -> dict:
    """Export only single-variant 2025 tokens, without editing source representatives."""
    variants = defaultdict(list)
    variant_path = review_dir / "msdial2025_chain_variants.parquet"
    candidate_path = review_dir / "msdial2025_chain_representative_candidates.tsv"
    for row in pq.read_table(variant_path).to_pylist():
        variants[row["chain_token"]].append(row)
    with candidate_path.open(encoding="utf-8-sig", newline="") as stream:
        candidates = {r["chain_token"]: r for r in csv.DictReader(stream, delimiter="\t")}
    rows, excluded = [], []
    for token, group in sorted(variants.items()):
        candidate = candidates[token]
        if len(group) != 1 or int(candidate["number_of_positional_variants"]) != 1:
            excluded.append(token)
            continue
        row = group[0]
        if (
            candidate["candidate_fragment_smiles"] != row["canonical_fragment_smiles"]
            or json.loads(candidate["candidate_db_positions"]) != row["db_positions"]
        ):
            raise ValueError(f"Candidate/variant disagreement: {token}")
        rows.append(
            {
                "chain_token": token,
                "linkage_type": row["linkage_type"],
                "carbon_count": row["carbon_count"],
                "double_bond_count": row["double_bond_count"],
                "representative_fragment_smiles": row["canonical_fragment_smiles"],
                "representative_double_bond_positions": json.dumps(row["db_positions"]),
                "p_db_count_includes_vinyl": "true" if row["linkage_type"] == "P" else "",
                "source": "MS-DIAL-2025:single-positional-variant;review-v0.1",
            }
        )
    counts = dict(Counter(r["linkage_type"] for r in rows))
    expected_excluded = {f"A-{n}:1" for n in (24, 26, 28, 30, 32, 34)}
    if counts != {"A": 177, "O": 84, "P": 61} or set(excluded) != expected_excluded:
        raise ValueError(f"Unexpected freeze scope: {counts}, excluded={excluded}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    rejects = []
    validated = import_chains(temporary, rejects)
    if rejects or len(validated) != 322:
        raise ValueError(f"Frozen table validation failed: {rejects}")
    temporary.replace(output)
    return {
        "counts": counts,
        "total": len(rows),
        "excluded": excluded,
        "positions_edited": False,
        "builder_validation_rejections": 0,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "input_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (variant_path, candidate_path)
        },
    }
