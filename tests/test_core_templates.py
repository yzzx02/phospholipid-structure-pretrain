"""New-label formula boundaries, equivalent-target leakage and invalid-candidate regression."""

import numpy as np
import pytest
import torch
from rdkit import Chem

from structure_pretrain.core_templates.chemistry import equivalence_key, validate_candidate
from structure_pretrain.core_templates.common import ROOT, configuration, read, tokenizer
from structure_pretrain.core_templates.data import CoreData, make_model
from structure_pretrain.core_templates.metrics import aggregate, score_target
from structure_pretrain.core_templates.runtime import compare, restore_rng, rng_state
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.formal_v2.schedule import WarmupCosine
from structure_pretrain.glycerol_core_collection.chemistry import capped_core
from structure_pretrain.glycerol_substructure_collection.chemistry import counts

PA = "P(=O)(O)(O)OCC(CO[*:101])O[*:102]"
ETHER = "C(OCC)C(O[*:102])CO[*:101]"


def test_new_ports_roundtrip_and_no_silent_old_length_limit():
    t = SmilesTokenizer(["[*:101]", "[*:102]"])
    assert t.decode(t.encode(PA)) == PA
    assert t.encode(PA).count(t.index["[*:101]"]) == 1
    with pytest.raises(ValueError):
        SmilesTokenizer().encode(PA)


def test_formula_uses_H_capped_core_not_raw_two_port_graph():
    mol = Chem.MolFromSmiles(PA)
    raw, cap = counts(mol), counts(capped_core(mol))
    assert cap["H"] == raw["H"] + 2
    result = validate_candidate(PA, "C3H9O6P")
    assert result["valid"] and result["formula_match"]
    assert result["assembly_H_formula_verified"]


def test_nonP_whole_glycerol_is_a_legal_target():
    result = validate_candidate(ETHER)
    assert result["valid"] and result["counts"][4] == 0


@pytest.mark.parametrize(
    "text", ["", "c1cccc1", "C(=C)(=C)=C", "CC.O", "OP(=O)(O)OCC(CO[*:101])O[*:102]"]
)
def test_generated_failures_do_not_escape(text):
    assert not validate_candidate(text)["valid"]


def test_duplicate_port_roles_are_invalid():
    assert not validate_candidate(PA.replace("102", "101"))["valid"]


def test_terminal_central_chain_roles_cannot_be_swapped():
    swapped = PA.replace("101", "temporary").replace("102", "101").replace("temporary", "102")
    assert not validate_candidate(swapped)["valid"]


def test_extra_head_fatty_chain_is_rejected_without_graph_repair():
    extra = "P(=O)(O)(OCCCCCCCCCCCC)OCC(CO[*:101])O[*:102]"
    result = validate_candidate(extra)
    assert not result["valid"] and "extra_head_fatty_chain" in result["reason"]


def test_RDKit_kekulization_exception_is_a_failed_candidate(monkeypatch):
    def broken(*args, **kwargs):
        raise Chem.rdchem.KekulizeException("regression: untrusted candidate cannot kekulize")

    monkeypatch.setattr("structure_pretrain.core_templates.chemistry.verify_assembly", broken)
    result = validate_candidate(PA)
    assert not result["valid"] and "KekulizeException" in result["reason"]


def test_equivalent_stereo_protonation_and_role_labels_grouped():
    assert equivalence_key("OCC(O)CO") == equivalence_key("OC[C@H](O)CO")
    assert equivalence_key("OP(=O)(O)OCC(O)CO") == equivalence_key("[O-]P(=O)(O)OCC(O)CO")
    assert equivalence_key("OCC(O)CO") != equivalence_key("COCC(O)CO")


def test_raw_filtered_empty_target_denominators():
    target = {"target_id": "PA", "target_smiles": PA, "core_formula": "C3H9O6P"}
    empty = score_target(target, [], 0.7)
    invalid = score_target(target, [{"smiles": "c1cccc1", "eos": True}], 0.7)
    good = score_target(target, [{"smiles": PA, "eos": True}], 0.7)
    report = aggregate([empty, invalid, good])
    assert report["targets"] == 3
    assert report["raw"]["exact_at_10"] == 1 / 3
    assert report["posterior_filtered"]["no_qualified_candidate"] == 2 / 3
    assert report["raw"]["candidate_chemical_valid_fraction"] == 0.5


def test_no_EOS_is_not_a_completed_candidate():
    target = {"target_id": "PA", "target_smiles": PA, "core_formula": "C3H9O6P"}
    r = score_target(target, [{"smiles": PA, "eos": False}], 0.7)
    assert r["raw"]["exact_at_10"] == 0


def test_formula_mismatch_is_separate_from_structural_validity():
    target = {"target_id": "PA", "target_smiles": PA, "core_formula": "C3H9O6P"}
    r = score_target(target, [{"smiles": ETHER, "eos": True}], 0.7)
    assert r["raw"]["valid_candidates"] == 1
    assert r["posterior_filtered"]["distinct_legal_formula_candidates"] == 0


def test_grouped_split_and_quota_behavior():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "prep_core", Path("scripts/prepare_core_templates8000.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    keys = np.asarray(["one", "two", "one", "three", "four", "five"])
    splits = module.grouped_split(keys, fractions=(0.5, 0.25, 0.25))
    assert splits[0] == splits[2]
    np.testing.assert_array_equal(splits, module.grouped_split(keys, fractions=(0.5, 0.25, 0.25)))
    quota = module.balanced_quotas([1, 20, 100], 30)
    assert quota.sum() == 30 and np.all(quota <= np.asarray([1, 20, 100]))


def test_rng_and_scheduler_restore_contract():
    schedule = WarmupCosine(3e-4, 1e-5, 4000, 400000)
    state = rng_state("cpu")
    sample = torch.rand(5)
    restore_rng(state, "cpu")
    compare(sample, torch.rand(5), 0)
    assert schedule.value(1) < schedule.value(4000)


def test_frozen_all_labels_and_8000_bank_targets_are_retained():
    folder = ROOT / "data"
    manifest = read(folder / "manifest.json")
    assert manifest["targets"] == 189864
    assert sum(manifest["split_counts"].values()) == 189864
    splits, bank = np.load(folder / "splits.npy"), np.load(folder / "bank_rows.npy")
    keys, refs = np.load(folder / "equivalence_keys.npy"), np.load(folder / "references.npy")
    assert len(bank) == 8000 and np.all(splits[bank] == 0)
    assert len(set(keys[bank])) == 8000
    assert not np.any((keys[bank[np.maximum(refs, 0)]] == keys[:, None]) & (refs >= 0))
    sets = [set(keys[splits == i]) for i in range(3)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert np.load(folder / "bank_target_mask.npy").sum() == 8000
    lengths = np.load(folder / "token_lengths.npy")
    assert lengths.max() == 94 and np.count_nonzero(lengths > 84) == 562


def test_new_network_accepts_nonP_without_legacy_chain_arithmetic():
    data = CoreData(ROOT, "cpu")
    indices = np.flatnonzero(data.formulas[:, 4].numpy() == 0)[:2]
    assert len(indices) == 2
    model = make_model(ROOT).eval()
    with torch.no_grad():
        result = model.encode_templates(data.batch(indices))
    assert result["target_core_formula"][:, 4].sum() == 0
    assert result["template_memory"].shape[0] == 2
    assert configuration()["max_label_tokens"] >= int(data.lengths.max())
    assert "[*:101]" in tokenizer(ROOT).tokens
