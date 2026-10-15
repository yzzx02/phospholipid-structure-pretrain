"""Chunk checkpoints, disk-backed distinct counts, and structure/parent aggregation."""

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path

import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import rdBase

from structure_pretrain.library_builder.msdial_inventory import lz4_prefix

from . import chemistry
from .chemistry import extract_structure, is_candidate, source_naming
from .reader import blocks, iter_metadata, raw_block

META_SCHEMA = pa.schema(
    [
        ("source_scan_id", pa.int64()),
        ("source_name", pa.string()),
        ("source_ontology", pa.string()),
        ("source_compound_class", pa.string()),
        ("smiles", pa.string()),
        ("source_inchikey", pa.string()),
        ("source_record_index", pa.int64()),
        ("source_version", pa.string()),
    ]
)
CHAIN_SCHEMA = pa.schema(
    [
        ("chain_token", pa.string()),
        ("linkage_type", pa.string()),
        ("carbon_count", pa.int32()),
        ("double_bond_count", pa.int32()),
        ("db_positions", pa.list_(pa.int32())),
        ("fragment_smiles", pa.string()),
        ("canonical_fragment_smiles", pa.string()),
        ("parent_structure_id", pa.int64()),
        ("source_scan_id", pa.int64()),
        ("source_name", pa.string()),
        ("source_ontology", pa.string()),
        ("source_compound_class", pa.string()),
        ("source_inchikey", pa.string()),
        ("source_record_index", pa.int64()),
        ("observed_attachment_position", pa.string()),
        ("source_version", pa.string()),
        ("p_db_count_includes_vinyl", pa.bool_()),
        ("source_ether_label", pa.string()),
        ("source_named_db_count", pa.int32()),
        ("source_p_db_count_includes_vinyl", pa.bool_()),
        ("source_naming_status", pa.string()),
    ]
)
REJECTION_SCHEMA = pa.schema(
    [
        ("source_scan_id", pa.int64()),
        ("source_name", pa.string()),
        ("source_ontology", pa.string()),
        ("source_compound_class", pa.string()),
        ("source_inchikey", pa.string()),
        ("source_record_index", pa.int64()),
        ("source_version", pa.string()),
        ("reason", pa.string()),
        ("detail", pa.string()),
    ]
)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_part(path, rows, schema):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression="zstd")
    os.replace(tmp, path)


class MemoryMonitor:
    def __init__(self):
        self.peak = 0
        self.event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        process = psutil.Process()
        while not self.event.is_set():
            self.peak = max(self.peak, process.memory_info().rss)
            self.event.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.event.set()
        self.thread.join()


def connect(path):
    db = sqlite3.connect(path)
    db.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA cache_size=-32768;
        PRAGMA temp_store=FILE;
        CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS smiles (value TEXT PRIMARY KEY, n INTEGER) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS inchikey (value TEXT PRIMARY KEY, n INTEGER) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS cache (smiles TEXT PRIMARY KEY, result TEXT) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS parents (id INTEGER PRIMARY KEY, smiles TEXT UNIQUE);
        CREATE TABLE IF NOT EXISTS variants (
            id INTEGER PRIMARY KEY, token TEXT, fragment TEXT, kind TEXT,
            carbon INTEGER, db INTEGER, positions TEXT, n INTEGER DEFAULT 0,
            UNIQUE(token, fragment));
        CREATE TABLE IF NOT EXISTS vp (variant INTEGER, parent INTEGER,
            PRIMARY KEY(variant,parent)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS tp (token TEXT, parent INTEGER,
            PRIMARY KEY(token,parent)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS classes (variant INTEGER, class TEXT,
            PRIMARY KEY(variant,class)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS examples (variant INTEGER, name TEXT,
            PRIMARY KEY(variant,name)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS pe (variant INTEGER, parent INTEGER,
            PRIMARY KEY(variant,parent)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS pe_examples (variant INTEGER, name TEXT,
            PRIMARY KEY(variant,name)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS pe_counts (variant INTEGER PRIMARY KEY, n INTEGER);
    """)
    return db


def get_state(db):
    row = db.execute("SELECT value FROM state WHERE key='checkpoint'").fetchone()
    return json.loads(row[0]) if row else None


def chemistry_fingerprint():
    payload = Path(chemistry.__file__).read_bytes() + rdBase.rdkitVersion.encode()
    return hashlib.sha256(payload).hexdigest()


def _bump_unique(db, table, counts):
    db.executemany(
        f"INSERT INTO {table} VALUES(?,?) ON CONFLICT(value) DO UPDATE SET n=n+excluded.n",
        sorted(counts.items()),  # Improve B-tree page locality within a bounded chunk.
    )


def _aggregate(db, result, row, counts):
    parent = result.get("parent")
    if not result["chains"]:
        return []
    db.execute("INSERT OR IGNORE INTO parents(smiles) VALUES(?)", (parent,))
    parent_id = db.execute("SELECT id FROM parents WHERE smiles=?", (parent,)).fetchone()[0]
    cls = row.get("compound_class") or row.get("ontology") or "(missing)"
    is_pe = cls.lower() in {"pe", "etherpe", "lysope", "lpe", "etherlpe", "oxpe"}
    observations = []
    for chain in result["chains"]:
        token, fragment = chain["chain_token"], chain["canonical_fragment_smiles"]
        db.execute(
            """INSERT INTO variants(token,fragment,kind,carbon,db,positions,n)
            VALUES(?,?,?,?,?,?,1) ON CONFLICT(token,fragment) DO UPDATE SET n=n+1""",
            (
                token,
                fragment,
                chain["linkage_type"],
                chain["carbon_count"],
                chain["double_bond_count"],
                json.dumps(chain["db_positions"]),
            ),
        )
        vid = db.execute(
            "SELECT id FROM variants WHERE token=? AND fragment=?", (token, fragment)
        ).fetchone()[0]
        db.execute("INSERT OR IGNORE INTO vp VALUES(?,?)", (vid, parent_id))
        db.execute("INSERT OR IGNORE INTO tp VALUES(?,?)", (token, parent_id))
        db.execute("INSERT OR IGNORE INTO classes VALUES(?,?)", (vid, cls))
        if db.execute("SELECT count(*) FROM examples WHERE variant=?", (vid,)).fetchone()[0] < 3:
            db.execute("INSERT OR IGNORE INTO examples VALUES(?,?)", (vid, row["name"] or ""))
        if is_pe and chain["linkage_type"] == "P":
            db.execute("INSERT OR IGNORE INTO pe VALUES(?,?)", (vid, parent_id))
            db.execute(
                "INSERT INTO pe_counts VALUES(?,1) ON CONFLICT(variant) DO UPDATE SET n=n+1", (vid,)
            )
            if (
                db.execute("SELECT count(*) FROM pe_examples WHERE variant=?", (vid,)).fetchone()[0]
                < 3
            ):
                db.execute(
                    "INSERT OR IGNORE INTO pe_examples VALUES(?,?)", (vid, row["name"] or "")
                )
        naming = source_naming(chain, row["name"])
        counts["naming:" + chain["linkage_type"] + ":" + naming["source_naming_status"]] += 1
        if naming["source_p_db_count_includes_vinyl"] is not None:
            counts[
                "source_p_includes_vinyl:" + str(naming["source_p_db_count_includes_vinyl"])
            ] += 1
        observations.append({**chain, **naming, "parent_structure_id": parent_id})
    return observations


def process_chunk(db, rows, version, start_index, counts, shared_cache=None):
    metadata, evidence, rejections = [], [], []
    smiles_counts, ik_counts = Counter(), Counter()
    local_cache = {}  # bounded by this chunk, cleared after checkpoint
    for relative, row in enumerate(rows):
        source = {
            "source_" + k: row.get(k)
            for k in ("scan_id", "name", "ontology", "compound_class", "inchikey")
        }
        source.update(source_version=version, source_record_index=start_index + relative)
        smiles = row["smiles"] or ""
        metadata.append({**source, "smiles": row["smiles"]})
        counts["total_records"] += 1
        counts["records_with_smiles" if smiles else "records_without_smiles"] += 1
        counts["ontology:" + (row["ontology"] or "(missing)")] += 1
        counts["class:" + (row["compound_class"] or "(missing)")] += 1
        if smiles:
            smiles_counts[smiles] += 1
        if row["inchikey"]:
            ik_counts[row["inchikey"]] += 1
        if not is_candidate(row):
            continue
        counts["candidate_records"] += 1
        if smiles in local_cache:
            result = local_cache[smiles]
        else:
            cached = db.execute("SELECT result FROM cache WHERE smiles=?", (smiles,)).fetchone()
            if cached:
                result = json.loads(cached[0])
            else:
                prior = (
                    shared_cache.execute(
                        "SELECT result FROM cache WHERE smiles=?", (smiles,)
                    ).fetchone()
                    if shared_cache is not None
                    else None
                )
                if prior:
                    result = json.loads(prior[0])
                    counts["shared_structure_cache_hits"] += 1
                else:
                    result = extract_structure(smiles)
                    counts["unique_structures_computed"] += 1
                db.execute("INSERT INTO cache VALUES(?,?)", (smiles, json.dumps(result)))
            local_cache[smiles] = result
        counts["rdkit_success_records"] += bool(result["parsed"])
        counts["backbone_success_records"] += bool(result["backbone"])
        counts["records_with_accepted_chains"] += bool(result["chains"])
        for failure in result["rejections"]:
            counts["rejection:" + failure["reason"]] += 1
            counts["detail:" + failure["detail"]] += 1
            cls = row.get("compound_class") or row.get("ontology") or "(missing)"
            counts["rejection_class:" + cls + ":" + failure["reason"]] += 1
            rejections.append({**source, **failure})
        chains = _aggregate(db, result, row, counts)
        for chain in chains:
            evidence.append({**source, **chain})
        counts["chain_observations"] += len(chains)
    _bump_unique(db, "smiles", smiles_counts)
    _bump_unique(db, "inchikey", ik_counts)
    return metadata, evidence, rejections


def scan(path, version, output, config, resume=False, start_block=0, max_records=None):
    output = Path(output)
    review = output / "msdial_chain_review"
    state_dir = output / "_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)
    db_path = state_dir / f"{version}.sqlite"
    if db_path.exists() and not resume:
        raise ValueError(f"Existing scan: use --resume or another output directory: {db_path}")
    db = connect(db_path)
    checkpoint = get_state(db)
    fingerprint = chemistry_fingerprint()
    marker = state_dir / f"{version}.chemistry.json"
    if marker.exists() and json.loads(marker.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Chemistry code/RDKit changed; use a new output directory")
    if not checkpoint:
        write_json(marker, {"fingerprint": fingerprint})
    shared_cache = None
    prior_marker = state_dir / "2025.chemistry.json"
    prior_db = state_dir / "2025.sqlite"
    if (
        version == "2022"
        and prior_marker.exists()
        and prior_db.exists()
        and json.loads(prior_marker.read_text())["fingerprint"] == fingerprint
    ):
        candidate = sqlite3.connect(prior_db.resolve().as_uri() + "?mode=ro", uri=True)
        candidate.execute("PRAGMA cache_size=-32768")
        prior_state = get_state(candidate)
        if prior_state and prior_state.get("complete"):
            shared_cache = candidate
        else:
            candidate.close()
    path = Path(path).resolve()
    identity = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "config": config,
        "start_block": start_block,
    }
    if checkpoint and checkpoint["identity"] != identity:
        raise ValueError("Source/config/start-block changed: cannot resume incompatible scan")
    counts = Counter(checkpoint["counts"]) if checkpoint else Counter()
    previous_seconds = checkpoint.get("elapsed_seconds", 0) if checkpoint else 0
    previous_peak = checkpoint.get("peak_rss_bytes", 0) if checkpoint else 0
    resume_index = checkpoint["next_index"] if checkpoint else 0
    source_index, processed = 0, 0
    block_info = list(blocks(path))
    import msgpack

    # Only five decompressed bytes are needed for each block's top-level array count.
    with path.open("rb") as stream:
        for block in block_info:
            stream.seek(block["offset"])
            prefix = lz4_prefix(stream, block["compressed"], 5)
            u = msgpack.Unpacker(raw=False)
            u.feed(prefix)
            block["records"] = u.read_array_header()
    total_expected = sum(b["records"] for b in block_info)
    evidence_name = f"msdial{version}_chain_evidence.parquet"
    started = time.monotonic()
    with MemoryMonitor() as memory:
        for block in block_info:
            end = source_index + block["records"]
            if block["block"] < start_block or end <= resume_index:
                source_index = end
                continue
            if max_records is not None and processed >= max_records:
                break
            skip = max(0, resume_index - source_index)
            print(json.dumps({"stage": "decompress", "version": version, **block}), flush=True)
            with raw_block(path, block, state_dir / "scratch") as raw:
                chunk = []
                next_index = source_index + skip
                for row in iter_metadata(raw, version, skip):
                    chunk.append(row)
                    processed += 1
                    limit = max_records is not None and processed >= max_records
                    if len(chunk) >= config["chunk_records"] or limit:
                        n = len(chunk)
                        meta, evidence, rejected = process_chunk(
                            db, chunk, version, next_index, counts, shared_cache
                        )
                        part = f"part-{next_index:09d}.parquet"
                        write_part(
                            output / f"msdial{version}_metadata.parquet" / part, meta, META_SCHEMA
                        )
                        write_part(review / evidence_name / part, evidence, CHAIN_SCHEMA)
                        write_part(
                            review / "rejections.parquet" / version / part,
                            rejected,
                            REJECTION_SCHEMA,
                        )
                        next_index += n
                        checkpoint = {
                            "identity": identity,
                            "next_index": next_index,
                            "counts": dict(counts),
                            "expected_records": total_expected,
                            "elapsed_seconds": previous_seconds + time.monotonic() - started,
                            "peak_rss_bytes": max(previous_peak, memory.peak),
                        }
                        db.execute(
                            "INSERT OR REPLACE INTO state VALUES('checkpoint',?)",
                            (json.dumps(checkpoint),),
                        )
                        # Renames precede commit; uncommitted parts are overwritten on resume.
                        db.commit()
                        print(
                            json.dumps(
                                {
                                    "version": version,
                                    "next_index": next_index,
                                    "counts": {k: v for k, v in counts.items() if ":" not in k},
                                    "elapsed_seconds": checkpoint["elapsed_seconds"],
                                    "peak_rss_mib": round(memory.peak / 2**20, 1),
                                }
                            ),
                            flush=True,
                        )
                        chunk = []
                    if limit:
                        break
                if chunk:
                    # Route the final short chunk through the same atomic commit path.
                    n = len(chunk)
                    meta, evidence, rejected = process_chunk(
                        db, chunk, version, next_index, counts, shared_cache
                    )
                    part = f"part-{next_index:09d}.parquet"
                    write_part(
                        output / f"msdial{version}_metadata.parquet" / part, meta, META_SCHEMA
                    )
                    write_part(review / evidence_name / part, evidence, CHAIN_SCHEMA)
                    write_part(
                        review / "rejections.parquet" / version / part, rejected, REJECTION_SCHEMA
                    )
                    checkpoint = {
                        "identity": identity,
                        "next_index": next_index + n,
                        "counts": dict(counts),
                        "expected_records": total_expected,
                        "elapsed_seconds": previous_seconds + time.monotonic() - started,
                        "peak_rss_bytes": max(previous_peak, memory.peak),
                    }
                    db.execute(
                        "INSERT OR REPLACE INTO state VALUES('checkpoint',?)",
                        (json.dumps(checkpoint),),
                    )
                    db.commit()
            source_index = end
    if checkpoint:
        checkpoint["complete"] = checkpoint["next_index"] == total_expected and start_block == 0
        checkpoint["elapsed_seconds"] = previous_seconds + time.monotonic() - started
        checkpoint["peak_rss_bytes"] = max(previous_peak, memory.peak)
        db.execute("INSERT OR REPLACE INTO state VALUES('checkpoint',?)", (json.dumps(checkpoint),))
        db.commit()
    db.close()
    if shared_cache is not None:
        shared_cache.close()
    return checkpoint
