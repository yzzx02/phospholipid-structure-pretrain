import pytest
from rdkit import Chem

from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.chain_combinations import make_combination
from structure_pretrain.library_builder.schemas import LibraryValidationError
from structure_pretrain.library_builder.validation import validate_polar_fragment


@pytest.mark.parametrize(
    "tokens,head,formula",
    [
        (("A-16:0", "A-18:1"), "demo_PC", "C42H82NO8P"),
        (("A-16:0", "A-18:1"), "demo_PE", "C39H76NO8P"),
        (("O-16:0", "A-18:1"), "demo_PE", "C39H78NO7P"),
        (("P-16:0", "A-18:1"), "demo_PE", "C39H76NO7P"),
        (("A-16:0",), "demo_PC", "C24H50NO7P"),
    ],
)
def test_complete_neutral_formula(inputs, tokens, head, formula):
    chains, polars, _ = inputs
    row = assemble(chains, make_combination(tokens, chains), polars[head])
    assert row["neutral_formula"] == formula
    mol = Chem.MolFromSmiles(row["canonical_smiles"])
    assert sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) == 0
    assert row["neutral_formula_P"] == 1 and row["neutral_formula_N"] == 1
    if head == "demo_PC":
        assert {atom.GetFormalCharge() for atom in mol.GetAtoms()} >= {-1, 0, 1}


def test_unsupported_element_rejected():
    with pytest.raises(LibraryValidationError) as error:
        validate_polar_fragment("[*:1]P(=O)(O)O[Si](C)(C)C", 1)
    assert error.value.reason == "unsupported_element"


def test_report_counts_rejected_unsupported_polar(inputs, tmp_path):
    import csv

    from structure_pretrain.library_builder.pipeline import build_library

    from .conftest import EXAMPLES

    _, _, config = inputs
    with (EXAMPLES / "polar_moieties.tsv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows, fields = list(reader), reader.fieldnames
    rows.append(
        dict(rows[0], polar_id="bad_silicon", polar_fragment_smiles="[*:1]P(=O)(O)O[Si](C)(C)C")
    )
    path = tmp_path / "polars.tsv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    report = build_library(
        config, EXAMPLES / "chains.tsv", path, tmp_path / "output", EXAMPLES / "p_to_o_mapping.tsv"
    )
    assert report["unsupported_element_count"] == 1
    assert report["number_of_polar_moieties"] == 4 and report["total_structures"] == 60


@pytest.mark.parametrize(
    "smiles,reason",
    [
        ("invalid_smiles", "invalid_smiles"),
        ("[*:1]P(=O)(O)(O)(O)O", "invalid_valence"),
    ],
)
def test_parse_failure_distinct_from_actual_valence_error(smiles, reason):
    with pytest.raises(LibraryValidationError) as error:
        validate_polar_fragment(smiles, 1)
    assert error.value.reason == reason
