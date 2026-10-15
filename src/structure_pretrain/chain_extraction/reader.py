"""Verified LBM2 field selection. Peak arrays are skipped by MessagePack's C parser."""

import mmap
import struct
from contextlib import contextmanager
from pathlib import Path

import cramjam
import msgpack

SCHEMA_REF = "8556036fb73d77e4d703eb4263defe025d604a5d"
FIELDS = {
    "2025": {
        0: "scan_id",
        5: "name",
        7: "ontology",
        8: "smiles",
        9: "inchikey",
        14: "compound_class",
    },
    "2022": {
        0: "scan_id",
        3: "name",
        17: "ontology",
        7: "smiles",
        8: "inchikey",
        2: "compound_class",
    },
}


def blocks(path):
    """Validate ext32/type99/int32 envelopes without materializing a compressed block."""
    path = Path(path)
    with path.open("rb") as stream:
        number = 0
        while header := stream.read(11):
            start = stream.tell() - 11
            if len(header) != 11 or (header[0], header[5], header[6]) != (201, 99, 210):
                raise ValueError(f"Unsupported LBM2 envelope at {start}")
            size = int.from_bytes(header[1:5], "big") - 5
            raw = struct.unpack(">i", header[7:11])[0]
            end = start + 11 + size
            if size < 1 or raw < 1 or end > path.stat().st_size:
                raise ValueError("Truncated/invalid LBM2 block")
            yield {"block": number, "offset": start + 11, "compressed": size, "raw": raw}
            number += 1
            stream.seek(end)


@contextmanager
def raw_block(path, block, scratch):
    """Native LZ4 into a disk mmap, avoiding a 1 GiB Python bytes allocation.

    Mapped pages are reclaimable by the OS; the raw file is removed after use.
    No Python list of block records is ever built.
    """
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    target = scratch / f"block-{block['block']}.raw"
    with Path(path).open("rb") as source, target.open("w+b") as output:
        output.truncate(block["raw"])
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as compressed:
            with mmap.mmap(output.fileno(), 0) as uncompressed:
                view = memoryview(compressed)[
                    block["offset"] : block["offset"] + block["compressed"]
                ]
                try:
                    written = cramjam.lz4.decompress_block_into(view, uncompressed)
                    if written != block["raw"]:
                        raise ValueError("LBM2 decoded length mismatch")
                    uncompressed.flush()
                finally:
                    view.release()
    try:
        with target.open("rb") as stream:
            yield stream
    finally:
        target.unlink(missing_ok=True)


def iter_metadata(stream, version, skip_records=0):
    unpacker = msgpack.Unpacker(
        stream,
        raw=False,
        strict_map_key=False,
        read_size=1024 * 1024,
        max_buffer_size=64 * 1024 * 1024,
    )
    count = unpacker.read_array_header()
    selected = FIELDS[str(version)]
    for index in range(count):
        if index < skip_records:
            unpacker.skip()
            continue
        length = unpacker.read_array_header()
        if not (26 <= length <= 29 if str(version) == "2025" else length == 26):
            raise ValueError(f"Unverified {version} record length {length}")
        row = {"block_record_index": index}
        for field in range(length):
            if field in selected:
                row[selected[field]] = unpacker.unpack()
            else:
                unpacker.skip()
        if not isinstance(row["scan_id"], int):
            raise ValueError("Scan ID is not an integer")
        for name in ("name", "ontology", "compound_class", "smiles", "inchikey"):
            if row[name] is not None and not isinstance(row[name], str):
                raise ValueError(f"Unexpected {version} field type: {name}")
        yield row


def block_record_count(stream):
    unpacker = msgpack.Unpacker(stream, raw=False, read_size=16)
    return unpacker.read_array_header()
