import io
import struct

import msgpack
import pytest

from structure_pretrain.library_builder.msdial_inventory import inspect_lbm2, lz4_prefix


def block(records):
    raw = msgpack.packb(records)
    length, extra = len(raw), bytearray()
    if length >= 15:
        remaining = length - 15
        while remaining >= 255:
            extra.append(255)
            remaining -= 255
        extra.append(remaining)
    compressed = bytes([min(15, length) << 4]) + bytes(extra) + raw
    return (
        b"\xc9"
        + struct.pack(">I", len(compressed) + 5)
        + b"\x63\xd2"
        + struct.pack(">i", len(raw))
        + compressed
    )


def test_lbm2_headers_and_verified_metadata_schema(tmp_path):
    record = [None] * 29
    record[0], record[5], record[7], record[8], record[14] = 1, "PC example", "PC", "CCO", "PC"
    # Spectrum payload is skipped, never interpreted or exported.
    record[4] = [[1.2, 3.4, "unread"]]
    path = tmp_path / "synthetic.lbm2"
    path.write_bytes(block([record, record]) + block([record]))
    result = inspect_lbm2(path, True, 1)
    assert result["total_records_from_array_headers"] == 3
    assert len(result["sample_records"]) == 2
    assert result["sample_records"][0]["smiles"] == "CCO"
    assert result["spectrum_processing_performed"] is False
    assert "spectrum" not in result["sample_records"][0]


def test_lz4_overlapping_match_and_prefix_limit():
    # Literal "abc", followed by an overlapping offset=3 match of length 9.
    encoded = b"\x35abc\x03\x00"
    assert lz4_prefix(io.BytesIO(encoded), len(encoded), 12) == b"abcabcabcabc"
    assert lz4_prefix(io.BytesIO(encoded), len(encoded), 7) == b"abcabca"


def test_reject_unknown_container_instead_of_guessing(tmp_path):
    path = tmp_path / "wrong.lbm2"
    path.write_bytes(b"not_a_container")
    with pytest.raises(ValueError, match="Unsupported"):
        inspect_lbm2(path)
