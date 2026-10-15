"""Disposable SSD aggregation; durable per-CID chemistry lives in ChemistryCache."""

import hashlib
import itertools
import json
import os
import sqlite3
import tempfile
from pathlib import Path

STATUSES = ("APPROVABLE", "MULTIPLE_ATTACHMENT_AMBIGUITY", "FILTER_REVIEW", "CHARGE_AMBIGUOUS")
FILTERS = tuple(
    sorted(
        (
            "molecular_weight",
            "carbon_count",
            "heavy_atoms",
            "phosphorus_count",
            "long_aliphatic_carbon_run",
        )
    )
)


class FragmentStore:
    def __init__(self, cache_dir, fields, query_ids):
        self.root = Path(cache_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="polar-fragments-", suffix=".sqlite", dir=self.root)
        os.close(fd)
        self.path = Path(name).resolve()
        if self.path.parent != self.root:
            raise ValueError("Scratch path escaped the cache directory")
        self.db = sqlite3.connect(self.path)
        # Rebuilt from durable CID checkpoints after interruption; no fsync is needed here.
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("PRAGMA cache_size=-65536")
        self.db.executescript("""
            CREATE TABLE fragments (
              polar_id TEXT PRIMARY KEY, smiles TEXT UNIQUE, metadata TEXT,
              site_count INTEGER, status_rank INTEGER, query_mask INTEGER,
              review_mask INTEGER, filter_mask INTEGER);
            CREATE TABLE sources (polar_id TEXT, cid INTEGER,
              PRIMARY KEY (polar_id,cid)) WITHOUT ROWID;
        """)
        self.fields, self.query_ids, self.count = fields, sorted(query_ids), 0

    def __len__(self):
        return self.count

    def __contains__(self, key):
        return (
            self.db.execute("SELECT 1 FROM fragments WHERE smiles=?", (key,)).fetchone() is not None
        )

    def merge(self, candidate, cid, query_ids, known_keys):
        key = candidate["polar_fragment_smiles"]
        polar_id = "PCP_" + hashlib.sha256(key.encode()).hexdigest()[:16]
        metadata = {name: candidate.get(name) for name in self.fields}
        metadata.update(
            {
                "polar_id": polar_id,
                "allow_aa": True,
                "allow_oa": True,
                "allow_pa": key in known_keys.get("PE", []),
                "known_control_ids": [name for name, keys in known_keys.items() if key in keys],
            }
        )
        inserted = self.db.execute(
            "INSERT OR IGNORE INTO fragments VALUES (?,?,?,0,0,0,0,0)",
            (polar_id, key, json.dumps(metadata, ensure_ascii=False)),
        ).rowcount
        if inserted:
            self.count += 1
        elif self.db.execute(
            "SELECT smiles FROM fragments WHERE polar_id=?", (polar_id,)
        ).fetchone() != (key,):
            raise ValueError("Polar ID hash collision")
        rank = STATUSES.index(candidate["status"])
        query_mask = sum(1 << self.query_ids.index(q) for q in set(query_ids))
        filter_mask = sum(1 << FILTERS.index(flag) for flag in set(candidate["filter_flags"]))
        self.db.execute(
            "UPDATE fragments SET site_count=site_count+1,status_rank=max(status_rank,?),"
            "query_mask=query_mask|?,review_mask=review_mask|?,filter_mask=filter_mask|?"
            " WHERE polar_id=?",
            (rank, query_mask, 1 << rank, filter_mask, polar_id),
        )
        self.db.execute("INSERT OR IGNORE INTO sources VALUES (?,?)", (polar_id, cid))

    def flush(self):
        self.db.commit()

    def iter_rows(self):
        self.flush()
        cursor = self.db.execute(
            "SELECT f.*,s.cid FROM fragments f JOIN sources s USING(polar_id)"
            " ORDER BY f.polar_id,s.cid"
        )
        for _, group in itertools.groupby(cursor, key=lambda record: record[0]):
            first = next(group)
            row = json.loads(first[2])
            cids = [first[-1]] + [record[-1] for record in group]
            row.update(
                {
                    "source_cids": cids,
                    "source_count": len(cids),
                    "attachment_site_count_before_dedup": first[3],
                    "status": STATUSES[first[4]],
                    "matched_query_ids": [
                        q for i, q in enumerate(self.query_ids) if first[5] & (1 << i)
                    ],
                    "review_flags": sorted(
                        s for i, s in enumerate(STATUSES) if first[6] & (1 << i)
                    ),
                    "filter_flags": [f for i, f in enumerate(FILTERS) if first[7] & (1 << i)],
                }
            )
            yield row

    def close(self):
        self.db.close()
        for suffix in ("", "-wal", "-shm", "-journal"):
            target = Path(str(self.path) + suffix).resolve()
            if target.parent != self.root or not target.name.startswith("polar-fragments-"):
                raise ValueError("Unsafe scratch cleanup path")
            target.unlink(missing_ok=True)
