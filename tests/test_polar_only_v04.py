"""Portable CPU contracts: no private data required."""

import numpy as np
import pytest
import torch

from structure_pretrain.config import ModelConfig
from structure_pretrain.data.prepare_v03 import atom_counts
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.library_builder.canonicalize import parse_molecule
from structure_pretrain.polar_only_v04.chemistry import chainless_smiles, derive_polar_formula
from structure_pretrain.polar_only_v04.dataset import collate_polar
from structure_pretrain.polar_only_v04.features import (
    fragment_molecule,
    fragment_to_graph,
    packed_fingerprint,
)
from structure_pretrain.polar_only_v04.model import (
    PolarOnlyModel,
    candidate_polar_formulas,
    independent_candidate_batch,
    polar_loss,
)
from structure_pretrain.polar_only_v04.retrieval import chosen_mode, exact_scores
from structure_pretrain.utils.attention import attention_backend

HEAD = "[*:1]P(=O)(O)OCCN(C)C"


@pytest.fixture
def counts():
    scaffold = atom_counts(parse_molecule("[*:1]OCC(O[*:2])CO[*:3]"))
    chain = np.array(
        [
            np.zeros(10),
            atom_counts(parse_molecule("[*:1]C(=O)CCC")),
            atom_counts(parse_molecule("[*:1]CCCC")),
        ],
        dtype=np.int32,
    )
    return chain, scaffold


@pytest.mark.parametrize("chains", [[1, 1], [1, 2], [1, 0]])
def test_attached_formula_and_real_lyso_H(counts, chains):
    chain, scaffold = counts
    polar = atom_counts(parse_molecule(HEAD))
    full = scaffold + chain[chains].sum(0) + polar
    full[1] += chains[1] == 0
    np.testing.assert_array_equal(derive_polar_formula(full, chains, chain, scaffold), polar)
    fragment = fragment_molecule(chainless_smiles(HEAD, 1 if chains[1] == 0 else 2))
    expected = scaffold + polar
    expected[1] += chains[1] == 0
    np.testing.assert_array_equal(atom_counts(fragment), expected)


@pytest.mark.parametrize("bad", [[0, 0], [0, 1], [4, 1], [1.5, 0]])
def test_unknown_chains_never_zero_fill(counts, bad):
    with pytest.raises(ValueError):
        derive_polar_formula(np.ones(10), bad, *counts)


def test_negative_polar_formula_stops(counts):
    with pytest.raises(ValueError):
        derive_polar_formula(np.zeros(10), [1, 1], *counts)


def test_chain_normalization_and_occupancy():
    # The representation receives occupancy, not an acyl/ether chain identity.
    aa = chainless_smiles(HEAD, 2)
    oa = chainless_smiles(HEAD, 2)
    lyso = chainless_smiles(HEAD, 1)
    assert aa == oa and aa != lyso
    assert "102" not in lyso and "102" in aa
    molecule = fragment_molecule(aa)
    assert sum(a.GetAtomicNum() == 0 for a in molecule.GetAtoms()) == 2
    np.testing.assert_array_equal(
        packed_fingerprint(molecule), packed_fingerprint(fragment_molecule(oa))
    )


@pytest.mark.parametrize(
    "text", ["CCC", "[*:1]OCC", "[*:101]CCC", "[*:101]OC.C", "[*:101]OCCO[*:101]", "[*:102]OCC"]
)
def test_fragment_dummy_constraints(text):
    with pytest.raises(ValueError):
        fragment_molecule(text)


def test_fragment_requires_real_glycerol_O_P_boundary():
    for text in ("[*:101]OCC", "[*:101]OCC(O)COC", "[*:101]OCC(OC)COP(=O)(O)O"):
        with pytest.raises(ValueError):
            fragment_molecule(text)


def make_example(encoder, counts, head=HEAD, templates=2):
    chain, scaffold = counts
    polar = atom_counts(parse_molecule(head))
    f = torch.tensor(np.tile(polar, (templates, 1))).float()
    text = chainless_smiles(head, 2)
    structure = (
        fragment_to_graph(text)
        if encoder == "graph"
        else torch.tensor(
            np.unpackbits(packed_fingerprint(fragment_molecule(text)), bitorder="little")
        ).float()
    )
    return {
        "target_full_formula": torch.tensor(scaffold + chain[1] * 2 + polar).float(),
        "given_chain_tokens": [1, 1],
        "template_formulas": f,
        "delta_formulas": torch.zeros_like(f),
        "target_polar_token_ids": SmilesTokenizer().encode(head),
        "structures": [structure] * templates,
        "encoder": encoder,
    }


@pytest.mark.parametrize("encoder", ["fingerprint", "graph"])
def test_forward_backward_full_memory_and_label_free_generation(counts, encoder):
    torch.set_num_threads(2)
    config = ModelConfig(encoder=encoder, max_polar_length=84, dropout=0, template_dropout=0)
    model = PolarOnlyModel(config, SmilesTokenizer(), {}, *counts)
    assert not any("chain_head" in key for key in model.state_dict())
    batch = collate_polar([make_example(encoder, counts)], k=3)
    with attention_backend("math"):
        model.train()
        output = model(batch)
        assert output["template_memory"].shape == (1, 4, 512)
        assert output["memory_padding_mask"].tolist() == [[False, False, False, True]]
        loss = polar_loss(output, batch, 0)
        assert torch.isfinite(loss)
        loss.backward()
        for name, module in model.named_children():
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
            ), name
        model.eval()
        # No target polar labels are supplied to generation.
        clean = {k: v for k, v in batch.items() if k != "target_polar_token_ids"}
        result = model.generate(clean, max_length=5)
        assert result["polar_candidates"][0][0]["ids"][0] == 1


@pytest.mark.parametrize("k", [3, 5, 10])
def test_padding_and_template_permutation(counts, k):
    torch.set_num_threads(2)
    model = PolarOnlyModel(
        ModelConfig(encoder="fingerprint", max_polar_length=84, dropout=0, template_dropout=0),
        SmilesTokenizer(),
        {},
        *counts,
    ).eval()
    a = make_example("fingerprint", counts)
    a["structures"][1] = a["structures"][1].roll(21)
    batch = collate_polar([a], k=k)
    permuted = {key: value.clone() for key, value in batch.items()}
    for key in ("template_formulas", "delta_formulas", "template_mask"):
        permuted[key][:, :2] = permuted[key][:, :2].flip(1)
    permuted["fingerprints"] = permuted["fingerprints"].flip(0)
    with attention_backend("math"), torch.no_grad():
        first, other = model(batch), model(permuted)
    torch.testing.assert_close(first["polar_logits"], other["polar_logits"], atol=2e-6, rtol=1e-5)
    assert first["template_memory"].shape[1] == k + 1


def test_candidate_axis_formula_is_independent(counts):
    model = PolarOnlyModel(
        ModelConfig(encoder="fingerprint", max_polar_length=84), SmilesTokenizer(), {}, *counts
    )
    full = make_example("fingerprint", counts)["target_full_formula"][None]
    candidates = torch.tensor([[[1, 1], [1, 2]]])
    polar = candidate_polar_formulas(model, full, candidates)
    assert polar.shape == (1, 2, 10)
    assert polar[0, 0, 3] != polar[0, 1, 3]  # acyl vs ether O accounting


def test_retrieval_fallback_and_exact_bit_scores():
    rows = np.arange(10000)
    h, fallback = chosen_mode(rows, np.ones(10000), np.ones(10000))
    assert abs(h.mean() - 0.8) < 0.002 and not fallback.any()
    h, fallback = chosen_mode([0], np.array([0]), np.array([3]), force_h=True)
    assert not h[0] and fallback[0]
    with pytest.raises(ValueError):
        chosen_mode([0], np.array([0]), np.array([0]))
    q = np.array([3, 0], np.uint8)
    np.testing.assert_allclose(exact_scores(q, np.array([[3, 0], [1, 0]], np.uint8)), [1, 0.5])


def test_candidate_templates_remain_independent_memory(counts):
    torch.set_num_threads(2)
    config = ModelConfig(encoder="fingerprint", max_polar_length=84, dropout=0, template_dropout=0)
    model = PolarOnlyModel(config, SmilesTokenizer(), {}, *counts).eval()
    example = make_example("fingerprint", counts)
    full = example["target_full_formula"][None]
    chains = torch.tensor([[[1, 1], [1, 2]]])
    formulas = example["template_formulas"][None, None].expand(1, 2, -1, -1)
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    structure = torch.stack(example["structures"] * 2)
    batch = independent_candidate_batch(
        model, full, chains, formulas, mask, {"fingerprints": structure}
    )
    with attention_backend("math"), torch.no_grad():
        output = model.encode_templates(batch)
    assert output["template_memory"].shape == (2, 3, 512)
    assert not torch.equal(output["template_memory"][0], output["template_memory"][1])
    assert "target_polar_token_ids" not in batch


def test_polar_CE_keeps_EOS_ignores_PAD():
    labels = torch.tensor([[1, 4, 2, 0]])
    logits = torch.zeros(1, 3, 7)
    batch = {"target_polar_token_ids": labels}
    base = polar_loss({"polar_logits": logits}, batch, 0)
    changed_pad = logits.clone()
    changed_pad[0, 2, 6] = 100
    assert polar_loss({"polar_logits": changed_pad}, batch, 0) == base
    changed_eos = logits.clone()
    changed_eos[0, 1, 2] = 100
    assert polar_loss({"polar_logits": changed_eos}, batch, 0) < base
