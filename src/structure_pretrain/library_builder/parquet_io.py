from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .schemas import ELEMENTS

STRING = pa.string()
STRINGS = pa.list_(STRING)
CHAIN_SCHEMA = pa.schema(
    [
        ("chain_token", STRING),
        ("linkage_type", STRING),
        ("carbon_count", pa.int64()),
        ("double_bond_count", pa.int64()),
        ("representative_fragment_smiles", STRING),
        ("representative_double_bond_positions", pa.list_(pa.int64())),
        ("source", STRING),
        ("p_db_count_includes_vinyl", pa.bool_()),
    ]
)
POLAR_SCHEMA = pa.schema(
    [
        ("polar_id", STRING),
        ("source", STRING),
        ("source_id", STRING),
        ("original_smiles", STRING),
        ("polar_fragment_smiles", STRING),
        ("attachment_atom_map", pa.int64()),
        ("polar_family", STRING),
        ("allow_aa", pa.bool_()),
        ("allow_oa", pa.bool_()),
        ("allow_pa", pa.bool_()),
        ("source_polar_ids", STRINGS),
        ("provenance", STRINGS),
    ]
)
COMBINATION_SCHEMA = pa.schema(
    [
        ("combination_id", STRING),
        ("chain_tokens", STRINGS),
        ("linkage_family", STRING),
        ("total_carbon", pa.int64()),
        ("total_double_bonds", pa.int64()),
        ("is_lyso", pa.bool_()),
        ("mapped_from_p_tokens", STRINGS),
    ]
)
REJECTION_SCHEMA = pa.schema(
    [(field, STRING) for field in ("stage", "record_id", "reason", "message")]
)
STRUCTURE_SCHEMA = pa.schema(
    [
        ("structure_id", STRING),
        ("architecture", STRING),
        ("polar_id", STRING),
        ("polar_family", STRING),
        ("chain_tokens", STRINGS),
        ("linkage_family", STRING),
        ("representative_smiles", STRING),
        ("canonical_smiles", STRING),
        ("neutral_formula", STRING),
        *[(f"neutral_formula_{element}", pa.int64()) for element in ELEMENTS],
        ("inchikey", STRING),
        ("is_anchor_combination", pa.bool_()),
        ("is_lyso", pa.bool_()),
        ("representative_sn_assignment", pa.bool_()),
        ("source_chain_rule", STRINGS),
        ("source_polar_ids", STRINGS),
        ("polar_provenance", STRINGS),
        ("mapped_from_p_tokens", STRINGS),
        ("combination_id", STRING),
        ("total_chain_carbon", pa.int64()),
        ("total_chain_double_bonds", pa.int64()),
    ]
)


def write_rows(path: str | Path, rows, schema: pa.Schema, batch_size: int = 2000):
    with pq.ParquetWriter(path, schema=schema, compression="zstd") as writer:
        batch = []
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
