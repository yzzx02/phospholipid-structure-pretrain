"""Inference constraints checked against RDKit and the unchanged lipid Builder."""

import numpy as np
import pytest
import torch

from structure_pretrain.data.prepare_v03 import atom_counts, canonical_polar
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.canonicalize import parse_molecule
from structure_pretrain.library_builder.chain_combinations import make_combination
from structure_pretrain.library_builder.schemas import ELEMENTS, ChainRule, PolarRule
from structure_pretrain.library_builder.validation import attachment
from structure_pretrain.polar_only_v04.constrained_decode import (
    ElementBudget,
    SyntaxState,
    constrained_beam_search,
    rank_unique_accepted,
    syntax_step,
    token_element,
)
from structure_pretrain.polar_only_v04.formula_beam_eval import score_beams, summarize

HEAD = "[*:1]P(=O)(O)OCCN"


@pytest.mark.parametrize(
    "token,element",
    [
        ("Cl", "Cl"),
        ("Br", "Br"),
        ("c", "C"),
        ("[nH]", "N"),
        ("[NH3+]", "N"),
        ("[13CH2:99]", "C"),
        ("[IH]", "I"),
        ("[*:1]", "*"),
        ("[H]", "H"),
        ("%12", None),
        ("1", None),
        ("=", None),
    ],
)
def test_atomic_lexemes_not_characters_or_map_digits(token, element):
    assert token_element(token) == element


def walk(text):
    tokenizer = SmilesTokenizer(["%12"])
    state = SyntaxState()
    for token in tokenizer.encode(text)[1:-1]:
        state = syntax_step(state, tokenizer.tokens[token])
        if state is None:
            return None
    return state


@pytest.mark.parametrize(
    "text",
    [
        HEAD,
        "NCCOP(=O)(O)[*:1]",
        "P([*:1])(=O)(O)OCCN",
        "[*:1]P(=O)(O)OC1CCCCC1",
        "[*:1]P(=O)(O)OC1CC1C1CC1",
        "[*:1]P(=O)(O)OC12CCC1CC2",
        "[*:1]P(=O)(O)OC%12CCCCC%12",
        "[*:1]P(=O)([O-])OCC[NH3+]",
        "[*:1]P(=O)(O)Oc1cc[nH]c1",
        "OP(=O)(O)OP(=O)(O)[*:1]",
    ],
)
def test_grammar_accepts_rdkit_legal_branches_rings_charges_and_multiple_P(text):
    parse_molecule(text)
    assert walk(text).complete


@pytest.mark.parametrize(
    "text",
    [
        "[*:1]P()O",
        "[*:1]P(=O",
        "[*:1]P==O",
        "[*:1]P(O))O",
        "[*:1]P(=O)(O)OC1CC",
        "[*:1]P(=O)(O)OC1C1",
        "[*:1]OP(=O)(O)O",
        "*P(=O)(O)O",
        "[*:1]=P(O)O",
        "[*:1]P([*:2])(O)O",
        "[*:1]P(=O)(O)O.C",
        "[*:1]P(=O)(O)O+",
    ],
)
def test_obvious_prefix_errors_and_existing_boundary_contract(text):
    state = walk(text)
    assert state is None or not state.complete


def test_multi_P_attachment_is_the_mapped_neighbor_not_first_P():
    mol = parse_molecule("OP(=O)(O)OP(=O)(O)[*:1]")
    _, connected = attachment(mol, 1)
    first = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 15)
    assert connected != first
    assert walk("OP(=O)(O)OP(=O)(O)[*:1]").dummy_degree == 1


def test_heavy_budget_blocks_excess_and_early_eos_but_never_guesses_H():
    tokenizer = SmilesTokenizer()
    target = atom_counts(parse_molecule(HEAD))
    budget = ElementBudget(tokenizer, target)
    zero = np.zeros(9, dtype=np.int64)
    assert not budget.allowed(zero)[tokenizer.eos_id]
    assert not budget.allowed(zero)[tokenizer.index["Cl"]]
    assert not budget.allowed(zero)[tokenizer.index["B"]]
    assert budget.allowed(budget.target)[tokenizer.eos_id]
    assert not budget.allowed(budget.target)[tokenizer.index["C"]]
    altered = target.copy()
    altered[1] += 50
    np.testing.assert_array_equal(ElementBudget(tokenizer, altered).target, budget.target)


class EosAndExcessPreferred(torch.nn.Module):
    max_length = 84

    def __init__(self, tokenizer, text):
        super().__init__()
        self.tokenizer, self.answer = tokenizer, tokenizer.encode(text)

    def forward(self, ids, memory, padding):
        logits = torch.full((*ids.shape, len(self.tokenizer)), -30.0)
        logits[:, -1, self.tokenizer.eos_id] = 10.0
        logits[:, -1, self.tokenizer.index["Cl"]] = 9.0
        desired = self.answer[min(ids.shape[1], len(self.answer) - 1)]
        logits[:, -1, desired] = max(float(logits[0, -1, desired]), 8.0)
        return logits


@pytest.mark.parametrize("grammar", [False, True])
def test_search_blocks_early_eos_and_excess_without_renormalizing(grammar):
    tokenizer = SmilesTokenizer()
    decoder = EosAndExcessPreferred(tokenizer, HEAD)
    candidate = constrained_beam_search(
        decoder,
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, dtype=torch.bool),
        tokenizer,
        atom_counts(parse_molecule(HEAD)),
        beam_size=1,
        grammar=grammar,
    )[0]
    assert tokenizer.decode(candidate["ids"]) == HEAD
    assert candidate["eos"] and not candidate["length_exhausted"]
    # High original EOS/Cl probability still costs likelihood despite the masks.
    assert candidate["log_probability"] < -10


def test_length_limit_does_not_force_eos_or_repair_a_structure():
    tokenizer = SmilesTokenizer()
    candidates = constrained_beam_search(
        EosAndExcessPreferred(tokenizer, HEAD),
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, dtype=torch.bool),
        tokenizer,
        atom_counts(parse_molecule(HEAD)),
        beam_size=1,
        max_length=2,
        grammar=True,
    )
    assert not candidates[0]["eos"] and candidates[0]["length_exhausted"]


@pytest.fixture
def builder_context():
    tokenizer = SmilesTokenizer()
    chains = {
        "A-16:0": ChainRule("A-16:0", "A", 16, 0, "[*:1]C(=O)" + "C" * 15, (), "fixture"),
    }
    rule = PolarRule("test", "test", "test", HEAD, HEAD, 1, "generated", True, True, False)
    full = assemble(chains, make_combination(("A-16:0", "A-16:0"), chains), rule)
    return tokenizer, {
        "polar_formula": atom_counts(parse_molecule(HEAD)),
        "chain_tokens": ["A-16:0", "A-16:0"],
        "chains": chains,
        "heads": {},
        "full_formula": np.array([full[f"neutral_formula_{e}"] for e in ELEMENTS]),
        "target_polar": canonical_polar(HEAD),
        "target_full": full["canonical_smiles"],
        "template_polars": [],
    }


def beam(tokenizer, text, score=-1.0):
    return {
        "ids": tokenizer.encode(text),
        "log_probability": score,
        "eos": True,
        "length_exhausted": False,
    }


def test_final_H_is_strict_for_both_polar_and_full_formula(builder_context):
    tokenizer, context = builder_context
    candidate = beam(tokenizer, HEAD)
    assert score_beams([candidate], tokenizer, **context)[0]["accepted"]
    context["polar_formula"][1] += 2
    scored = score_beams([candidate], tokenizer, **context)[0]
    assert scored["formula_match"] and not scored["accepted"]
    context["polar_formula"][1] -= 2
    context["full_formula"][1] += 2
    scored = score_beams([candidate], tokenizer, **context)[0]
    assert scored["polar_formula_match_including_H"] and not scored["accepted"]


def test_unparseable_canonical_candidate_is_rejected_without_losing_other_beams(
    builder_context, monkeypatch
):
    from structure_pretrain.training import quality_v03

    tokenizer, context = builder_context
    alternate = "P([*:1])(=O)(O)OCCN"
    original = quality_v03.canonical_polar
    monkeypatch.setattr(
        quality_v03,
        "canonical_polar",
        lambda text: "[*:1]P(=O)(O)Oc1cc1" if text == alternate else original(text),
    )
    candidates = score_beams(
        [beam(tokenizer, alternate, -1), beam(tokenizer, HEAD, -4)],
        tokenizer,
        **context,
    )
    assert not candidates[0]["polar_valid"] and not candidates[0]["accepted"]
    assert candidates[0]["reason"] == "sanitize_failed"
    assert candidates[0]["polar_smiles_internal"] == alternate
    assert candidates[1]["accepted"]


def test_novel_isomers_survive_but_canonical_duplicates_are_removed(builder_context):
    tokenizer, context = builder_context
    isomer = "[*:1]P(=O)(O)OC(C)N"
    duplicate = "NCCOP(=O)(O)[*:1]"
    candidates = score_beams(
        [beam(tokenizer, isomer, -1), beam(tokenizer, HEAD, -4), beam(tokenizer, duplicate, -9)],
        tokenizer,
        **context,
    )
    assert all(c["accepted"] for c in candidates)  # empty database lookup is allowed
    unique = rank_unique_accepted(candidates)
    assert len(unique) == 2
    assert not unique[0]["polar_exact"] and unique[1]["polar_exact"]


def test_empty_generation_is_explicit_failure_not_zero_denominator():
    records = [
        {
            "raw_candidates": [],
            "accepted_unique": [],
            "decode_seconds": 0.1,
            "validation_seconds": 0.0,
        }
    ]
    metrics = summarize(records, 5, filtered=True)
    assert metrics["strict_formula_coverage"] == 0
    assert metrics["no_qualified_candidate_rate"] == 1
    assert metrics["polar_topk_exact_over_all_targets"] == {"1": 0, "5": 0, "10": None}
    assert metrics["formula_correct_wrong_polar_structure_fraction_among_accepted"] is None
