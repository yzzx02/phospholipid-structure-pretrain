"""Bounded-prefix inspection of MS-DIAL v5 LBM2 metadata, without loading spectra."""

import struct
from collections import Counter
from pathlib import Path

import msgpack

SCHEMA_SOURCE = "https://github.com/systemsomicslab/MsdialWorkbench/blob/8556036fb73d77e4d703eb4263defe025d604a5d/src/Common/CommonStandard/Components/MoleculeMsReference.cs"
CONTAINER_SOURCE = "https://github.com/systemsomicslab/MsdialWorkbench/blob/8556036fb73d77e4d703eb4263defe025d604a5d/src/Common/CommonStandard/MessagePack/LargeListMessagePack.cs"


def lz4_prefix(stream, compressed_length: int, output_limit: int) -> bytes:
    """Decode only a bounded prefix of a raw LZ4 block, including overlapping matches."""
    output = bytearray()
    remaining = compressed_length

    def read(size):
        nonlocal remaining
        if size > remaining:
            raise ValueError("Truncated raw LZ4 block")
        data = stream.read(size)
        if len(data) != size:
            raise ValueError("Truncated raw LZ4 block")
        remaining -= size
        return data

    def extended(length):
        if length == 15:
            while True:
                value = read(1)[0]
                length += value
                if value != 255:
                    break
        return length

    while remaining and len(output) < output_limit:
        token = read(1)[0]
        literal = extended(token >> 4)
        amount = min(literal, output_limit - len(output))
        output.extend(read(amount))
        if amount != literal or len(output) == output_limit or not remaining:
            break
        offset = int.from_bytes(read(2), "little")
        if offset < 1 or offset > len(output):
            raise ValueError("Invalid raw LZ4 match offset")
        match = min(extended(token & 15) + 4, output_limit - len(output))
        pattern = bytes(output[-offset:])
        output.extend((pattern * ((match + offset - 1) // offset))[:match])
    return bytes(output)


def inspect_lbm2(path: str | Path, v5_metadata: bool = False, sample_records: int = 32):
    path = Path(path).resolve()
    blocks, samples = [], []
    with path.open("rb") as stream:
        while True:
            start = stream.tell()
            header = stream.read(11)
            if not header:
                break
            if len(header) != 11 or header[0] != 0xC9 or header[5] != 99 or header[6] != 0xD2:
                raise ValueError("Unsupported LBM2 container header; no automatic schema guessing")
            extension_length = int.from_bytes(header[1:5], "big")
            uncompressed_length = struct.unpack(">i", header[7:11])[0]
            if extension_length < 5 or uncompressed_length < 5:
                raise ValueError("Invalid LBM2 block lengths")
            end = start + 6 + extension_length
            if end > path.stat().st_size:
                raise ValueError("LBM2 block exceeds file size")
            prefix = lz4_prefix(stream, extension_length - 5, 4 * 1024 * 1024 if v5_metadata else 5)
            unpacker = msgpack.Unpacker(
                raw=False, strict_map_key=False, max_buffer_size=len(prefix) + 1
            )
            unpacker.feed(prefix)
            count = unpacker.read_array_header()
            block = {
                "file_offset": start,
                "compressed_bytes": extension_length - 5,
                "uncompressed_bytes": uncompressed_length,
                "records": count,
            }
            if v5_metadata:
                selected = {
                    0: "scan_id",
                    5: "name",
                    7: "ontology",
                    8: "smiles",
                    9: "inchikey",
                    14: "compound_class",
                }
                for _ in range(min(sample_records, count)):
                    try:
                        fields = unpacker.read_array_header()
                        if not 26 <= fields <= 29:
                            raise ValueError(
                                "Not the explicitly supported MoleculeMsReference v5 schema"
                            )
                        row = {}
                        for index in range(fields):
                            if index in selected:
                                row[selected[index]] = unpacker.unpack()
                            else:
                                unpacker.skip()  # Spectrum is skipped, not interpreted.
                        if not isinstance(row["scan_id"], int) or not isinstance(row["name"], str):
                            raise ValueError(
                                "MoleculeMsReference field types differ from the verified schema"
                            )
                        samples.append(row)
                    except msgpack.OutOfData:
                        block["metadata_prefix_exhausted"] = True
                        break
            blocks.append(block)
            stream.seek(end)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "blocks": blocks,
        "total_records_from_array_headers": sum(block["records"] for block in blocks),
        "sample_records": samples,
        "sample_compound_classes": dict(
            Counter(row.get("compound_class") or row.get("ontology") for row in samples)
        ),
        "sample_is_exhaustive": False,
        "training_performed": False,
        "spectrum_processing_performed": False,
        "schema_source": SCHEMA_SOURCE,
        "container_source": CONTAINER_SOURCE,
        "chain_rules_extracted": False,
    }
