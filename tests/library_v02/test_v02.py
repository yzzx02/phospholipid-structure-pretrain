import csv
import json
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from rdkit import Chem

from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.chain_combinations import active_anchors, make_combination
from structure_pretrain.library_builder.chain_library import import_chains
from structure_pretrain.library_builder.full_library_v02 import (
    AnchorOptions,
    build_v02,
    load_approved_v02,
    load_config_v02,
    plan_anchors,
    preflight_v02,
)
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.library_builder.schemas import load_build_config
from structure_pretrain.library_builder.special_cases_v02 import (
    PI_FRAGMENT,
    atom_counts,
    mono_keto_heads,
    motifs_in,
    phosphatidyl_nitrile_contexts,
    special_rule,
)
from structure_pretrain.polar_plausibility.pipeline import write_tsv

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/library_builder"


@pytest.fixture
def fixture_inputs(tmp_path):
    errors = []
    chains = import_chains(EXAMPLES / "chains.tsv", errors)
    with (EXAMPLES / "polar_moieties.tsv").open(encoding="utf-8", newline="") as stream:
        rows = [
            dict(row, polar_plausibility_tier="CORE", chemistry_family="phosphate")
            for row in csv.DictReader(stream, delimiter="\t")
        ]
    approved = tmp_path / "approved.tsv"
    write_tsv(approved, rows, list(rows[0]))
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "status": "complete",
                "approved_polar_count": len(rows),
                "approved_table_sha256": file_hash(approved),
            }
        )
    )
    config = replace(
        load_build_config(EXAMPLES / "demo_build.yaml"), schema_version="structure-library-v0.2"
    )
    return chains, approved, report, config


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("N#CCOP(=O)(O)[*:1]", ["cyanomethyl"]),
        ("N#CCCOP(=O)(O)[*:1]", ["2-cyanoethyl"]),
        ("N#CCP(=O)(O)[*:1]", []),
        ("N#CCCCOP(=O)(O)[*:1]", []),
        ("N#CC(OP(=O)(O)[*:1])C", []),
        ("O=P(O)(OCCN)[*:1]", []),
    ],
)
def test_nitrile_motifs_are_not_conflated(smiles, expected):
    assert motifs_in(smiles) == expected


def test_pi_has_five_sites_three_connectivity_variants_with_delta_h2():
    result = mono_keto_heads()
    assert (
        result["oxidizable_oh_sites"],
        result["generated_candidates"],
        result["unique_mono_keto_heads"],
    ) == (5, 5, 3)
    parent = atom_counts(Chem.MolFromSmiles(PI_FRAGMENT))
    for row in result["variants"]:
        mol = Chem.MolFromSmiles(row["polar_fragment_smiles"])
        diff = {e: atom_counts(mol)[e] - n for e, n in parent.items()}
        assert {e: d for e, d in diff.items() if d} == {"H": -2}
        assert len(mol.GetRingInfo().AtomRings()) == 1
        assert row["ring_graph_distances"] in ([1], [2], [3])


def test_invalid_raw_nitrile_graph_is_explicitly_rejected():
    with pytest.raises(ValueError, match="Unparseable"):
        motifs_in("N#CCOP(=O)(O)(O)(O)O")


def test_full_pi_formula_differences_with_real_assembly(fixture_inputs):
    chains, _, _, _ = fixture_inputs
    combo = make_combination(("A-16:0", "A-18:1"), chains)
    reference = assemble(chains, combo, special_rule(PI_FRAGMENT, "reference", "fixture"))
    for candidate in mono_keto_heads()["variants"]:
        rule = special_rule(candidate["polar_fragment_smiles"], "candidate", "fixture")
        result = assemble(chains, combo, rule)
        assert result["neutral_formula_H"] == reference["neutral_formula_H"] - 2
        for e in ("C", "N", "O", "P", "S"):
            assert result[f"neutral_formula_{e}"] == reference[f"neutral_formula_{e}"]


def test_phosphatidyl_and_nitrile_must_share_the_same_P(fixture_inputs):
    chains, _, _, _ = fixture_inputs
    combo = make_combination(("A-16:0", "A-18:1"), chains)
    row = assemble(chains, combo, special_rule("O=P(O)(O)[*:1]", "PA", "fixture"))
    mixture = row["canonical_smiles"] + ".N#CCCOP(=O)(O)O"
    assert phosphatidyl_nitrile_contexts(mixture, "2-cyanoethyl") == []


def test_free_acid_full_nitrile_diester_is_distinguished(fixture_inputs):
    chains, _, _, _ = fixture_inputs
    combo = make_combination(("A-16:0", "A-18:1"), chains)
    row = assemble(chains, combo, special_rule("N#CCCOP(=O)(O)[*:1]", "CN", "fixture"))
    assert phosphatidyl_nitrile_contexts(row["canonical_smiles"], "2-cyanoethyl") == [
        "free_acid_phosphatidyl_nitrile_diester"
    ]


def test_triester_is_flagged_for_review_without_claiming_natural_lipid(fixture_inputs):
    chains, _, _, _ = fixture_inputs
    combo = make_combination(("A-16:0", "A-18:1"), chains)
    row = assemble(chains, combo, special_rule("N#CCCOP(=O)(O)[*:1]", "CN", "fixture"))
    molecule = Chem.RWMol(Chem.MolFromSmiles(row["canonical_smiles"]))
    acid_oxygen = next(
        a
        for a in molecule.GetAtoms()
        if a.GetAtomicNum() == 8
        and a.GetDegree() == 1
        and a.GetTotalNumHs() == 1
        and a.GetNeighbors()[0].GetAtomicNum() == 15
    )
    acid_oxygen.SetNumExplicitHs(0)
    acid_oxygen.SetNoImplicit(True)
    methyl = molecule.AddAtom(Chem.Atom(6))
    molecule.AddBond(acid_oxygen.GetIdx(), methyl, Chem.BondType.SINGLE)
    Chem.SanitizeMol(molecule)
    assert phosphatidyl_nitrile_contexts(Chem.MolToSmiles(molecule), "2-cyanoethyl") == [
        "phosphatidyl_phosphotriester_protection_unresolved"
    ]


def test_approved_hash_and_tier_gate(fixture_inputs):
    _, approved, report, _ = fixture_inputs
    polars, details, _ = load_approved_v02(approved, report)
    assert len(polars) == 4 and all(
        d["polar_plausibility_tier"] == "CORE" for d in details.values()
    )
    approved.write_text(approved.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        load_approved_v02(approved, report)


def test_missing_formal_polar_input_blocks_assembly(tmp_path):
    with pytest.raises(ValueError, match="blocked"):
        load_approved_v02(tmp_path / "absent.tsv", tmp_path / "absent.json")


def test_three_anchors_preserve_all_heads_when_seven_exceed_budget(fixture_inputs):
    chains, approved, report, config = fixture_inputs
    polars, _, _ = load_approved_v02(approved, report)
    anchors = active_anchors(chains, config, [])
    selected, summary = plan_anchors(
        polars, anchors, replace(config, max_structures=20), AnchorOptions()
    )
    assert summary["adaptive_anchors"]
    assert len(selected) == 4
    assert all(set(summary["common_anchor_ids"]).issubset(combos) for combos in selected.values())
    assert min(map(len, selected.values())) >= 3
    assert sum(map(len, selected.values())) <= 20
    assert summary["common_anchor_ids"] == [a.combination_id for a in anchors[:3]]
    assert (
        selected
        == plan_anchors(polars, anchors, replace(config, max_structures=20), AnchorOptions())[0]
    )


def test_even_three_anchor_budget_conflict_never_deletes_heads(fixture_inputs):
    chains, approved, report, config = fixture_inputs
    polars, _, _ = load_approved_v02(approved, report)
    with pytest.raises(ValueError, match="no heads deleted"):
        plan_anchors(
            polars,
            active_anchors(chains, config, []),
            replace(config, max_structures=11),
            AnchorOptions(),
        )


def test_special_requirements_are_reserved_before_random_sampling(fixture_inputs):
    chains, approved, report, config = fixture_inputs
    polars, _, _ = load_approved_v02(approved, report)
    requested = {polars[0].polar_id: {"O-16:0_A-18:1"}}
    selected, _ = plan_anchors(
        polars,
        active_anchors(chains, config, []),
        replace(config, max_structures=20),
        AnchorOptions(),
        requested,
    )
    assert "O-16:0_A-18:1" in selected[polars[0].polar_id]


def test_default_preflight_requires_frozen_chains(fixture_inputs):
    _, approved, report, config = fixture_inputs
    with pytest.raises(ValueError, match="frozen"):
        preflight_v02(config, AnchorOptions(), EXAMPLES / "chains.tsv", approved, report)


def test_v02_configuration_loads_without_changing_v01():
    config, options = load_config_v02(ROOT / "configs/library_build_v02.yaml")
    assert config.max_structures == 1_500_000 and options.mandatory_common_anchor_count == 3
    assert (
        load_build_config(ROOT / "configs/library_build.yaml").schema_version
        == "structure-library-v0.1"
    )


def test_small_fixture_build_outputs_reproducible_provenance_and_coverage(
    fixture_inputs, tmp_path, monkeypatch
):
    chains, approved, report_path, config = fixture_inputs
    polars, details, required = load_approved_v02(approved, report_path)
    selected, plan = plan_anchors(
        polars, active_anchors(chains, config, []), config, AnchorOptions(), required
    )
    details[polars[0].polar_id].update(is_special_case=True, special_case_family="fixture")
    from structure_pretrain.library_builder.chain_combinations import build_combination_bank

    bank = build_combination_bank(chains, config)
    # Only replace the real-data 322-chain gate for this small, explicitly labelled fixture.
    prepared = (
        {
            **plan,
            "polar_tier_counts": {"CORE": 4},
            "real_filtered_polar_count": 4,
            "generated_supplemental_head_count": 0,
        },
        (chains, polars, details, selected, bank),
    )
    monkeypatch.setattr(
        "structure_pretrain.library_builder.full_library_v02.preflight_v02", lambda *a: prepared
    )
    outputs = []
    for name in ("first", "second"):
        output = tmp_path / name
        report = build_v02(
            config, AnchorOptions(), EXAMPLES / "chains.tsv", approved, report_path, output
        )
        assert report["total_structures"] <= config.max_structures
        assert report["structures_per_polar_min"] >= 7
        assert (
            report["AA_count"]
            and report["OA_count"]
            and report["PE_PA_count"]
            and report["lyso_count"]
        )
        assert report["anchor_coverage_min"] == 7 and report["within_1500000"]
        assert not report["training_performed"]
        rows = pq.read_table(output / "structures.parquet").to_pylist()
        assert all(
            row["is_programmatically_assembled"] and not row["is_full_lipid_reported"]
            for row in rows
        )
        assert pq.read_table(output / "special_cases.parquet").num_rows > 0
        outputs.append(rows)
    assert outputs[0] == outputs[1]
