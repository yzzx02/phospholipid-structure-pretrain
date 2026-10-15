import io
import json
from collections import Counter

import cramjam
import msgpack
import pyarrow.dataset as ds
import pytest

from structure_pretrain.chain_extraction.chemistry import (
    extract_structure,
    is_candidate,
    source_naming,
)
from structure_pretrain.chain_extraction.pipeline import connect, process_chunk, scan
from structure_pretrain.chain_extraction.reader import iter_metadata
from structure_pretrain.chain_extraction.reporting import (
    aggregate,
    candidate_status,
    compare_versions,
    report,
)

CONFIG = {
    "dominant_fraction": 0.8,
    "low_evidence_min_unique_parents": 5,
    "chunk_records": 3,
    "graph_db_count_includes_vinyl": True,
}


def chain(n, positions=(), kind="A"):
    return ("C(=O)" if kind == "A" else "C") + "".join(
        ("=" if i in positions else "") + "C" for i in range(1, n)
    )


def lipid(a, b=None, head="OCCN"):
    return f"O({a})CC(COP(=O)(O){head})O" + (b or "")


def row(smiles, name="PE misleading 99:9/99:9", cls="PE"):
    return {
        "scan_id": 1,
        "name": name,
        "ontology": "",
        "compound_class": cls,
        "smiles": smiles,
        "inchikey": "fixture",
    }


def get(n, pos=(), kind="A"):
    return extract_structure(lipid(chain(n, pos, kind)))["chains"][0]


@pytest.mark.parametrize(
    "kind,n,pos", [("A", 16, ()), ("A", 20, (5, 8, 11, 14)), ("O", 18, (9,)), ("P", 18, (1, 9))]
)
def test_graph_cleavage(kind, n, pos):
    c = get(n, pos, kind)
    assert c["linkage_type"] == kind
    assert c["chain_token"] == f"{kind}-{n}:{len(pos)}"
    assert c["db_positions"] == list(pos)
    assert c["fragment_smiles"].startswith("[*:1]")


def test_sn_swap_does_not_change_tokens():
    a, b = chain(16), chain(18, [9])
    x = extract_structure(lipid(a, b))["chains"]
    y = extract_structure(lipid(b, a))["chains"]
    assert sorted(c["chain_token"] for c in x) == sorted(c["chain_token"] for c in y)


@pytest.mark.parametrize(
    "bad,detail",
    [
        ("C(=O)CC(C)C", "branched_chain"),
        ("C(=O)CCC(O)C", "heteroatom_or_oxidized_chain"),
        ("C(=O)CCC1CC1", "cyclic_epoxy_or_cyclopropyl_chain"),
        ("C(=O)CC#CC", "triple_or_nonstandard_bond"),
    ],
)
def test_modified_rejection(bad, detail):
    result = extract_structure(lipid(bad))
    assert not result["chains"]
    assert result["rejections"][0] == {"reason": "unsupported_modified_chain", "detail": detail}


@pytest.mark.parametrize(
    "smiles,reason",
    [
        (None, "missing_smiles"),
        ("not-smiles", "invalid_smiles"),
        ("CCCC(=O)OCC", "not_target_backbone"),
    ],
)
def test_rejection_reason(smiles, reason):
    assert extract_structure(smiles)["rejections"][0]["reason"] == reason


def test_name_does_not_determine_positions_and_p_conventions():
    c = get(18, [1, 9], "P")
    assert source_naming(c, "PE P-18:1/16:0")["source_p_db_count_includes_vinyl"] is False
    assert source_naming(c, "PE P-18:2/16:0")["source_p_db_count_includes_vinyl"] is True
    assert (
        source_naming(c, "PE O-18:2/16:0")["source_naming_status"]
        == "SOURCE_LABEL_LINKAGE_CONFLICT"
    )
    assert c["db_positions"] == [1, 9]


@pytest.mark.parametrize("name", ["Ac2PIM2", "EtherPC", "OxPE", "CL", "BMP", "CDP-DG"])
def test_broad_text_screen(name):
    assert is_candidate(row(None, name, name))


def test_unknown_class_phosphate_safety_net():
    assert is_candidate(row(lipid(chain(4)), "unknown", "unknown"))


def test_duplicate_parent_and_positional_isomers(tmp_path):
    db = connect(tmp_path / "test.sqlite")
    s9 = lipid(chain(18, [9]))
    s11 = lipid(chain(18, [11]))
    counts = Counter()
    meta, evidence, _ = process_chunk(db, [row(s9)] * 20 + [row(s11)], "2025", 0, counts)
    variants, candidates, _ = aggregate(db, CONFIG)
    assert len(meta) == 21 and len(evidence) == 21
    assert len(variants) == 2
    c = candidates["A-18:1"]
    assert c["total_unique_parent_structures"] == 2
    assert c["candidate_fraction"] == 0.5
    assert c["observations"] == 21
    assert c["status"] == "LOW_EVIDENCE"
    db.close()


def test_identical_chain_twice_one_parent(tmp_path):
    db = connect(tmp_path / "test.sqlite")
    process_chunk(db, [row(lipid(chain(16), chain(16)))], "2025", 0, Counter())
    variants, candidates, _ = aggregate(db, CONFIG)
    assert variants[0]["record_count"] == 2
    assert variants[0]["unique_parent_structure_count"] == 1
    assert candidates["A-16:0"]["total_unique_parent_structures"] == 1
    db.close()


@pytest.mark.parametrize(
    "total,support,status",
    [
        (10, [10], "UNIQUE"),
        (10, [8, 2], "DOMINANT"),
        (10, [8, 8], "CONFLICT"),
        (10, [10, 9], "CONFLICT"),
        (10, [7, 3], "CONFLICT"),
        (3, [3], "LOW_EVIDENCE"),
    ],
)
def test_candidate_status(total, support, status):
    assert candidate_status(total, support, 0.8, 5) == status


def pack_lbm(path, records, version="2025", multiple=False):
    arrays = []
    for r in records:
        a = [None] * (29 if version == "2025" else 26)
        if version == "2025":
            fields = {
                0: "scan_id",
                5: "name",
                7: "ontology",
                8: "smiles",
                9: "inchikey",
                14: "compound_class",
            }
            a[4] = msgpack.ExtType(42, b"not-a-spectrum-object")
        else:
            fields = {
                0: "scan_id",
                3: "name",
                17: "ontology",
                7: "smiles",
                8: "inchikey",
                2: "compound_class",
            }
            a[11] = msgpack.ExtType(42, b"not-a-spectrum-object")
        for i, key in fields.items():
            a[i] = r[key]
        arrays.append(a)
    groups = [arrays[:2], arrays[2:]] if multiple else [arrays]
    with path.open("wb") as f:
        for group in groups:
            raw = msgpack.packb(group, use_bin_type=True)
            compressed = bytes(cramjam.lz4.compress_block(raw, store_size=False))
            f.write(
                b"\xc9"
                + (len(compressed) + 5).to_bytes(4, "big")
                + b"\x63\xd2"
                + len(raw).to_bytes(4, "big", signed=True)
                + compressed
            )
    return arrays


@pytest.mark.parametrize("version", ["2025", "2022"])
def test_selected_fields_never_unpack_spectrum(tmp_path, version):
    r = row(lipid(chain(16)))
    arrays = pack_lbm(tmp_path / "a.lbm2", [r], version)
    data = io.BytesIO(msgpack.packb(arrays, use_bin_type=True))
    rows = list(iter_metadata(data, version))
    assert rows[0]["smiles"] == r["smiles"]
    assert rows[0]["ontology"] == ""
    assert "spectrum" not in rows[0]


def test_stream_resume_matches_one_shot(tmp_path):
    records = [row(lipid(chain(n))) for n in range(4, 12)]
    source = tmp_path / "a.lbm2"
    pack_lbm(source, records, multiple=True)
    one, resumed = tmp_path / "one", tmp_path / "resumed"
    scan(source, "2025", one, CONFIG)
    scan(source, "2025", resumed, CONFIG, max_records=4)
    scan(source, "2025", resumed, CONFIG, resume=True)
    for name in [
        "msdial2025_metadata.parquet",
        "msdial_chain_review/msdial2025_chain_evidence.parquet",
    ]:
        x = ds.dataset(one / name, format="parquet").to_table().to_pylist()
        y = ds.dataset(resumed / name, format="parquet").to_table().to_pylist()
        assert sorted(x, key=lambda r: r["source_record_index"]) == sorted(
            y, key=lambda r: r["source_record_index"]
        )
    report(resumed, CONFIG)
    r = json.loads(
        (resumed / "msdial_chain_review/extraction_report.json").read_text(encoding="utf-8")
    )
    assert r["versions"]["2025"]["metadata"]["full_scan_complete"]
    assert not r["training_performed"]


def test_version_states():
    def candidate(fragment):
        return {"candidate_fragment_smiles": fragment, "status": "UNIQUE", "observations": 10}

    a16, a18 = get(16), get(18, [9])
    o4, p4 = get(4, [2], "O"), get(4, [1], "P")
    c25 = {
        "A-16:0": candidate(a16["canonical_fragment_smiles"]),
        "A-18:1": candidate(a18["canonical_fragment_smiles"]),
        "P-4:1": candidate(p4["canonical_fragment_smiles"]),
        "A-8:0": candidate(get(8)["canonical_fragment_smiles"]),
    }
    c22 = {
        "A-16:0": candidate(a16["canonical_fragment_smiles"]),
        "A-18:1": candidate(get(18, [11])["canonical_fragment_smiles"]),
        "O-4:1": candidate(o4["canonical_fragment_smiles"]),
        "A-10:0": candidate(get(10)["canonical_fragment_smiles"]),
    }
    v25 = {
        t: [{"canonical_fragment_smiles": c["candidate_fragment_smiles"]}] for t, c in c25.items()
    }
    v22 = {
        t: [{"canonical_fragment_smiles": c["candidate_fragment_smiles"]}] for t, c in c22.items()
    }
    states = {
        r["chain_token"]: r["version_agreement"] for r in compare_versions(c25, c22, v25, v22)
    }
    assert states == {
        "A-16:0": "MATCH",
        "A-18:1": "POSITION_CONFLICT",
        "P-4:1": "LINKAGE_CONFLICT",
        "O-4:1": "LINKAGE_CONFLICT",
        "A-8:0": "ONLY_2025",
        "A-10:0": "ONLY_2022",
    }


def test_cardiolipin_four_chains():
    s = "CCCC(=O)OCC(COP(=O)(O)OCC(O)COP(=O)(O)OCC(OC(=O)CCC)COC(=O)CCC)OC(=O)CCC"
    result = extract_structure(s)
    assert len(result["chains"]) == 4
    assert not result["rejections"]


def test_cooccurring_variants_use_parent_union(tmp_path):
    db = connect(tmp_path / "cooccur.sqlite")
    process_chunk(db, [row(lipid(chain(18, [9]), chain(18, [11])))], "2025", 0, Counter())
    _, candidates, _ = aggregate(db, {**CONFIG, "low_evidence_min_unique_parents": 1})
    c = candidates["A-18:1"]
    assert c["total_unique_parent_structures"] == 1
    assert c["candidate_fraction"] == c["second_candidate_fraction"] == 1
    assert c["cooccurring_variant_parent_excess"] == 1
    assert c["status"] == "CONFLICT"
    db.close()


def test_reader_calls_skip_for_spectrum(monkeypatch, tmp_path):
    import structure_pretrain.chain_extraction.reader as reader

    arrays = pack_lbm(tmp_path / "spy.lbm2", [row(lipid(chain(8)))])
    real = msgpack.Unpacker
    calls = Counter()

    class Spy:
        def __init__(self, *args, **kwargs):
            self.impl = real(*args, **kwargs)

        def read_array_header(self):
            return self.impl.read_array_header()

        def unpack(self):
            calls["unpack"] += 1
            return self.impl.unpack()

        def skip(self):
            calls["skip"] += 1
            return self.impl.skip()

    monkeypatch.setattr(reader.msgpack, "Unpacker", Spy)
    result = list(reader.iter_metadata(io.BytesIO(msgpack.packb(arrays)), "2025"))
    assert len(result) == 1
    assert calls == {"unpack": 6, "skip": 23}


def test_start_block_marks_partial_and_preserves_global_index(tmp_path):
    source = tmp_path / "start.lbm2"
    pack_lbm(source, [row(lipid(chain(n))) for n in range(4, 9)], multiple=True)
    output = tmp_path / "out"
    checkpoint = scan(source, "2025", output, CONFIG, start_block=1)
    assert not checkpoint["complete"]
    assert checkpoint["counts"]["total_records"] == 3
    data = ds.dataset(output / "msdial2025_metadata.parquet", format="parquet").to_table()
    assert data["source_record_index"].to_pylist() == [2, 3, 4]


def test_partial_normal_and_modified_chains_are_reported_separately():
    result = extract_structure(lipid(chain(16), "C(=O)CCC(O)C"))
    assert [c["chain_token"] for c in result["chains"]] == ["A-16:0"]
    assert result["rejections"][0]["reason"] == "unsupported_modified_chain"


def test_cross_version_cache_preserves_independent_evidence(tmp_path):
    source25, source22 = tmp_path / "2025.lbm2", tmp_path / "2022.lbm2"
    r16, r18, r20 = [row(lipid(chain(n))) for n in (16, 18, 20)]
    pack_lbm(source25, [r16] * 10 + [r18])
    pack_lbm(source22, [r16, r18, r20], version="2022")
    output = tmp_path / "out"
    scan(source25, "2025", output, CONFIG)
    state22 = scan(source22, "2022", output, CONFIG)
    assert state22["counts"]["shared_structure_cache_hits"] == 2
    assert state22["counts"]["unique_structures_computed"] == 1
    db = connect(output / "_state/2022.sqlite")
    _, candidates, _ = aggregate(db, CONFIG)
    assert candidates["A-16:0"]["observations"] == 1
    assert candidates["A-16:0"]["total_unique_parent_structures"] == 1
    db.close()


def test_pe_report_excludes_lyso_pe(tmp_path):
    source = tmp_path / "pe.lbm2"
    pack_lbm(
        source,
        [
            row(lipid(chain(18, [1], "P"), chain(16)), "PE P-18:0/16:0", "EtherPE"),
            row(lipid(chain(18, [1], "P"), chain(16)), "PE O-18:1/16:0", "EtherPE"),
            row(lipid(chain(18, [9], "O"), chain(16)), "PE P-18:0/16:0", "EtherPE"),
            row(lipid(chain(18, [1], "P")), "LPE P-18:0", "EtherLPE"),
            row(lipid(chain(16, [1], "P")), "LPE P-16:0", "EtherLPE"),
        ],
    )
    output = tmp_path / "out"
    scan(source, "2025", output, CONFIG)
    summary = report(output, CONFIG)
    assert summary["2025"]["pe_p_unique_tokens"] == 1
    data = json.loads(
        (output / "msdial_chain_review/extraction_report.json").read_text(encoding="utf-8")
    )
    assert data["pe_p"][0]["chain_token"] == "P-18:1"
    assert data["pe_p"][0]["unique_parent_structure_count"] == 1
    assert data["pe_p"][0]["scope"] == "PE_ONLY_EXCLUDES_LPE"
    assert data["source_label_conflict_observations"]["2025"] == 2
    assert data["source_label_review_required_tokens"] == 2
