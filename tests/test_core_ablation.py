"""Online constraints, batched beam equivalence and truly formula-only capacity gates."""

import numpy as np
import pytest
import torch
from rdkit import Chem

from structure_pretrain.core_ablation.beam import generate
from structure_pretrain.core_ablation.common import ROOT, model_config, read, tokenizer
from structure_pretrain.core_ablation.data import FormulaData, make_model
from structure_pretrain.core_ablation.grammar import Budget, State, step
from structure_pretrain.core_templates.chemistry import validate_candidate
from structure_pretrain.core_templates.data import make_model as make_reference_model
from structure_pretrain.core_templates.model import loss
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.glycerol_core_collection.chemistry import capped_core
from structure_pretrain.glycerol_substructure_collection.chemistry import counts
from structure_pretrain.models.polar_decoder import PolarDecoder
from structure_pretrain.utils.formula import ELEMENTS

PA = "P(=O)(O)(O)OCC(CO[*:101])O[*:102]"
ETHER = "C(OCC)C(O[*:102])CO[*:101]"
TOK = SmilesTokenizer(["[*:101]", "[*:102]", "[Cl+3]"])


def formula(text):
    c = counts(capped_core(Chem.MolFromSmiles(text)))
    return np.asarray([c[e] for e in ELEMENTS])


def consume(text, expected):
    budget = Budget(TOK, np.asarray([expected]))
    state, used = State(), np.zeros((1, 11), dtype=np.int16)
    for token in TOK.encode(text)[1:-1]:
        assert budget.allowed(used, [0], [state])[0, token], TOK.tokens[token]
        used += budget.increments[token]
        state = step(state, TOK.tokens[token])
        assert state is not None
    return budget, state, used


@pytest.mark.parametrize("text", [PA, ETHER])
def test_legal_P_and_nonP_labels_not_pruned(text):
    budget, state, used = consume(text, formula(text))
    assert state.complete and budget.allowed(used, [0], [state])[0, TOK.eos_id]


def test_no_early_EOS_and_no_element_over_budget():
    budget = Budget(TOK, [formula(PA)])
    used = np.zeros((1, 11), dtype=np.int16)
    allowed = budget.allowed(used, [0], [State()])[0]
    assert not allowed[TOK.eos_id] and not allowed[TOK.index["N"]]
    used[0, 0] = formula(PA)[0]
    assert not budget.allowed(used, [0])[0, TOK.index["C"]]


def test_port_map_digits_and_charge_digits_are_not_atoms():
    budget = Budget(TOK, [formula(PA)])
    assert budget.increments[TOK.index["[*:101]"]].sum() == 1
    assert budget.increments[TOK.index["[*:101]"]][:9].sum() == 0
    assert budget.increments[TOK.index["1"]].sum() == 0
    assert budget.increments[TOK.index["[Cl+3]"]].sum() == 1
    assert budget.increments[TOK.index["[NH2+]"]].sum() == 1


def test_both_ports_required_once_and_not_bonded_to_P():
    assert step(step(State(), "P"), "[*:101]") is None
    state = step(step(State(), "O"), "[*:101]")
    assert state and step(state, "C") is None
    assert step(step(State(), "O"), "[*:1]") is None
    assert step(State(), "[*:101]") is None


def test_basic_branch_ring_and_component_syntax():
    c = step(State(), "C")
    assert step(c, ")") is None
    assert step(step(c, "("), ")") is None
    assert step(step(c, "="), "#") is None
    assert step(c, ".") is None
    ring = step(c, "1")
    assert step(ring, "1") is None
    ring = step(step(ring, "C"), "C")
    assert step(ring, "1") is not None


def test_fast_masks_match_pure_grammar_transitions():
    text = "P(=O)(O)(OCC1CC1)OCC(CO[*:101])O[*:102]"
    budget = Budget(TOK, [formula(text)])
    state, used = State(), np.zeros((1, 11), dtype=np.int16)
    for token in TOK.encode(text)[1:-1]:
        mask = budget.allowed(used, [0], [state])[0]
        for i, piece in enumerate(TOK.tokens):
            if i == TOK.eos_id:
                continue
            expected = (
                budget.countable[i]
                and np.all(used[0] + budget.increments[i] <= budget.target[0])
                and step(state, piece) is not None
            )
            assert mask[i] == expected, piece
        used += budget.increments[token]
        state = step(state, TOK.tokens[token])


def test_H_checked_finally_without_SMILES_character_counting():
    wrong = formula(PA).copy()
    wrong[1] += 1
    budget, state, used = consume(PA, wrong)
    assert budget.allowed(used, [0], [state])[0, TOK.eos_id]
    result = validate_candidate(PA, "C3H10O6P")
    assert result["valid"] and not result["formula_match"]


class FakeDecoder(torch.nn.Module):
    def __init__(self, tok):
        super().__init__()
        self.bos_id, self.eos_id = tok.bos_id, tok.eos_id
        self.pad_id, self.unk_id = tok.pad_id, tok.unk_id
        self.max_length, self.vocab = 20, len(tok)

    def forward(self, ids, memory, padding):
        x = torch.arange(self.vocab, device=ids.device, dtype=torch.float32)[None]
        key = ids.sum(1, keepdim=True).float() * 0.017 + memory[:, 0, :1] + len(ids[0]) * 0.13
        logits = torch.sin(x * 0.47 + key) * 3 + x * 0.001
        logits[:, self.eos_id] += 5 if ids.shape[1] >= 4 else -5
        return logits[:, None].expand(-1, ids.shape[1], -1)


@pytest.mark.parametrize("batch", [1, 3])
def test_batched_raw_beam_matches_existing_sequential_reference(batch):
    decoder = FakeDecoder(TOK)
    mem = torch.arange(batch, dtype=torch.float32).reshape(batch, 1, 1)
    padding = torch.zeros((batch, 1), dtype=torch.bool)
    expected = PolarDecoder.generate(decoder, mem, padding, beam_size=5, max_length=12)
    actual = generate(decoder, mem, padding, TOK, None, mode="raw", beam_size=5, max_length=12)
    for e, a in zip(expected, actual):
        assert [x["ids"] for x in e] == [x["ids"] for x in a]
        np.testing.assert_allclose(
            [x["log_probability"] for x in e], [x["log_probability"] for x in a], rtol=0, atol=1e-10
        )
        assert [x["eos"] for x in e] == [x["eos"] for x in a]


def test_dead_constrained_beam_returns_failure_without_forcing_EOS():
    class OnlyEOS(FakeDecoder):
        def forward(self, ids, memory, padding):
            logits = torch.full((len(ids), ids.shape[1], self.vocab), -torch.inf)
            logits[:, :, self.eos_id] = 0
            return logits

    result = generate(
        OnlyEOS(TOK),
        torch.zeros((1, 1, 1)),
        torch.zeros((1, 1), dtype=torch.bool),
        TOK,
        [formula(PA)],
        mode="elements_grammar",
        beam_size=5,
        max_length=12,
    )
    assert result == [[]]


def test_masking_preserves_original_decoder_log_probabilities():
    decoder = FakeDecoder(TOK)
    memory, padding = torch.zeros((1, 1, 1)), torch.zeros((1, 1), dtype=torch.bool)
    logits = decoder(torch.tensor([[TOK.bos_id]]), memory, padding)[:, -1].clone()
    logits[:, [TOK.pad_id, TOK.bos_id, TOK.unk_id]] = -torch.inf
    logp = logits.log_softmax(-1)
    beams = generate(
        decoder,
        memory,
        padding,
        TOK,
        [formula(PA)],
        mode="elements_grammar",
        beam_size=5,
        max_length=2,
    )[0]
    assert beams
    for candidate in beams:
        assert candidate["log_probability"] == float(logp[0, candidate["ids"][1]])


def test_formula_only_has_identical_parameter_shapes_and_initialization():
    torch.manual_seed(42)
    baseline = make_reference_model()
    torch.manual_seed(42)
    control = make_model()
    left, right = dict(baseline.named_parameters()), dict(control.named_parameters())
    assert left.keys() == right.keys()
    assert sum(x.numel() for x in left.values()) == sum(x.numel() for x in right.values())
    assert all(torch.equal(left[k], right[k]) for k in left)


def test_formula_data_never_loads_reference_arrays(monkeypatch):
    original = np.load

    def watched(path, *args, **kwargs):
        assert not any(x in str(path) for x in ("references", "bank_", "packed", "similarities"))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", watched)
    data = FormulaData()
    assert set(data.batch(data.indices(0)[:2])) == {"target_core_formula", "target_polar_token_ids"}


def test_formula_only_rejects_all_reference_input_and_updates_all_modules():
    torch.set_num_threads(4)
    data = FormulaData()
    model = make_model().train()
    batch = data.batch(data.indices(0)[:4])
    with pytest.raises(ValueError):
        model.encode_templates(
            {
                "target_core_formula": batch["target_core_formula"],
                "fingerprints": torch.zeros(1, 2048),
            }
        )
    value = loss(model(batch), batch, data.tokenizer.pad_id)
    value.backward()
    for module in model.children():
        gradients = [p.grad for p in module.parameters() if p.requires_grad]
        if gradients:
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert any(torch.count_nonzero(g) for g in gradients)


def test_formula_only_inference_repeated_formula_ignores_target_label():
    model = make_model().eval()
    f = torch.tensor(np.asarray([formula(PA)]), dtype=torch.float32)
    with torch.no_grad():
        a = model.encode_templates({"target_core_formula": f})
        b = model.encode_templates({"target_core_formula": f.clone()})
    assert torch.equal(a["template_memory"], b["template_memory"])
    assert read(ROOT / "data/formula_statistics.json")["reference_features_used"] is False
    assert model_config().max_polar_length == 96 and "[*:102]" in tokenizer().tokens
