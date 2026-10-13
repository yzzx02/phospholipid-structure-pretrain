import hashlib
import json
import sqlite3
from pathlib import Path


class StructureStore:
    """Disk-backed canonical deduplication; never keep 1.5M full records in RAM."""

    def __init__(self, path: str | Path, limit: int):
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE structures (smiles TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.limit, self.count, self.duplicates_removed = limit, 0, 0
        self.operations = 0

    def add(self, row: dict) -> bool:
        smiles = row["canonical_smiles"]
        previous = self.connection.execute(
            "SELECT payload FROM structures WHERE smiles=?", (smiles,)
        ).fetchone()
        self.operations += 1
        if previous is not None:
            stored = json.loads(previous[0])
            for field in (
                "source_chain_rule",
                "source_polar_ids",
                "polar_provenance",
                "mapped_from_p_tokens",
            ):
                stored[field] = sorted(set(stored[field]) | set(row[field]))
            stored["is_anchor_combination"] |= row["is_anchor_combination"]
            self.connection.execute(
                "UPDATE structures SET payload=? WHERE smiles=?",
                (json.dumps(stored, sort_keys=True), smiles),
            )
            self.duplicates_removed += 1
            added = False
        else:
            if self.count >= self.limit:
                raise ValueError("Hard structure budget exhausted")
            row = dict(row, structure_id="pl_" + hashlib.sha256(smiles.encode()).hexdigest())
            self.connection.execute(
                "INSERT INTO structures VALUES (?,?)", (smiles, json.dumps(row, sort_keys=True))
            )
            self.count += 1
            added = True
        if self.operations % 2000 == 0:
            self.connection.commit()
        return added

    def records(self):
        self.connection.commit()
        for (payload,) in self.connection.execute("SELECT payload FROM structures ORDER BY smiles"):
            yield json.loads(payload)

    def close(self):
        self.connection.commit()
        self.connection.close()
