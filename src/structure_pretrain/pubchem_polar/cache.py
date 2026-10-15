"""Versioned chemistry checkpoints, independent of PubChem query membership."""

import hashlib
import itertools
import json
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import rdBase


class ChemistryCache:
    def __init__(self, directory, limits):
        from . import chemistry

        self.policy_hash = hashlib.sha256(
            Path(chemistry.__file__).read_bytes()
            + json.dumps(limits, sort_keys=True).encode()
            + rdBase.rdkitVersion.encode()
            + b"bundle-v1"
        ).hexdigest()
        self.db = sqlite3.connect(Path(directory) / "chemistry.sqlite", timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS records ("
            "cid INTEGER, smiles_hash TEXT, policy_hash TEXT, data TEXT, "
            "PRIMARY KEY(cid,smiles_hash,policy_hash))"
        )
        self.pending = 0

    def get(self, cid, smiles):
        digest = hashlib.sha256(smiles.encode()).hexdigest()
        row = self.db.execute(
            "SELECT data FROM records WHERE cid=? AND smiles_hash=? AND policy_hash=?",
            (cid, digest, self.policy_hash),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, cid, smiles, bundle):
        self.db.execute(
            "INSERT OR REPLACE INTO records VALUES (?,?,?,?)",
            (
                cid,
                hashlib.sha256(smiles.encode()).hexdigest(),
                self.policy_hash,
                json.dumps(bundle, ensure_ascii=False),
            ),
        )
        self.pending += 1
        if self.pending >= 1000:
            self.flush()

    def flush(self):
        self.db.commit()
        self.pending = 0

    def close(self):
        self.flush()
        self.db.close()


def parquet_rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1000):
        yield from batch.to_pylist()


def seed_from_review(cache, directory):
    """Reuse audited, unchanged chemistry; query IDs are always recomputed for the new run."""
    directory = Path(directory)
    audit = json.loads((directory / "verification_audit.json").read_text(encoding="utf-8"))
    report = json.loads((directory / "pubchem_polar_report.json").read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise ValueError("Only audited chemistry outputs may seed checkpoints")
    if report.get("chemistry_policy_sha256") != cache.policy_hash:
        raise ValueError("Seed chemistry/RDKit/limits fingerprint does not match")
    streams = {}
    for key, filename in (
        ("component", "p_components"),
        ("rejection", "polar_rejections"),
        ("candidates", "attachment_candidates"),
    ):
        groups = itertools.groupby(
            parquet_rows(directory / f"pubchem_{filename}.parquet"), key=lambda row: row["cid"]
        )
        # Materialize just one CID group before advancing the underlying iterator.
        streams[key] = iter((cid, list(group)) for cid, group in groups)
    heads = {key: next(stream, None) for key, stream in streams.items()}
    count = 0
    for raw in parquet_rows(directory / "pubchem_polar_raw.parquet"):
        cid, bundle = raw["cid"], {"component": None, "rejection": None, "candidates": []}
        for key, stream in streams.items():
            head = heads[key]
            if head is not None and head[0] == cid:
                cleaned = [
                    {
                        k: v
                        for k, v in row.items()
                        if k not in {"cid", "original_smiles", "matched_query_ids"}
                    }
                    for row in head[1]
                ]
                bundle[key] = cleaned if key == "candidates" else cleaned[0]
                heads[key] = next(stream, None)
        if not bundle["candidates"] and bundle["rejection"] is None:
            raise ValueError(f"Audited CID missing chemistry result: {cid}")
        cache.put(cid, raw["original_smiles"], bundle)
        count += 1
    if any(head is not None for head in heads.values()):
        raise ValueError("Unmatched rows in seed tables")
    cache.flush()
    return count
