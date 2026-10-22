import csv
import hashlib
import json

import numpy as np
import pytest

from structure_pretrain.reported_headgroups.template_selection import (
    bank_diagnostics, balanced_pick, coverage_balanced_pick, current_compatibility, fp, load_review, phosphate_comparison_key, methylene_contractions,
)

PC = "C[N+](C)(C)CCOP(=O)([O-])[*:1]"
PROTONATED_PC = "C[N+](C)(C)CCOP(=O)(O)[*:1]"


def test_decoder_length_does_not_limit_template_fingerprint_input():
    from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
    long_polar = 'OP(=O)(O' + 'CCO'*30 + ')[*:1]'
    result = current_compatibility(long_polar, SmilesTokenizer().tokens)
    assert result['training_reference_eligible']
    assert not result['target_label_eligible']
    assert result['target_label_blockers'] == 'target_label_length_above_84'


def test_exact_methylene_homolog_does_not_contract_rings():
    shorter = 'NCCOP(=O)(O)[*:1]'
    assert phosphate_comparison_key(shorter) in methylene_contractions('NCCCOP(=O)(O)[*:1]')
    assert not methylene_contractions('OP(=O)(OC1CCCC1)[*:1]')


def test_phosphate_protonation_comparison_preserves_other_bonds_and_root():
    key = phosphate_comparison_key(PC)
    assert key == phosphate_comparison_key(PROTONATED_PC)
    for different in ("C[N+](C)(C)CCOP(=S)([O-])[*:1]",
                      "C=P([O-])(OCC[N+](C)(C)C)[*:1]",
                      "OP(=O)(O)CCOP(=O)(O)[*:1]",
                      "OP(=O)([*:1])CCOP(=O)(O)O"):
        assert phosphate_comparison_key(different) != key
    assert phosphate_comparison_key("OP(=O)(O)CCOP(=O)(O)[*:1]") != phosphate_comparison_key("OP(=O)([*:1])CCOP(=O)(O)O")


def test_review_rejects_changed_graph_or_decision(tmp_path):
    source, review_csv, review_json = [tmp_path / p for p in ("source.csv", "review.csv", "review.json")]
    with source.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["headgroup_id", "polar_smiles"])
        w.writeheader(); w.writerow({"headgroup_id": "A", "polar_smiles": PC})
    def write_review(decision):
        with review_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["headgroup_id", "polar_smiles", "human_decision", "human_note"])
            w.writeheader(); w.writerow({"headgroup_id": "A", "polar_smiles": PC, "human_decision": decision, "human_note": ""})
    review_json.write_text(json.dumps({"schema": "headgroup-review-v01", "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                       "decisions": {"A": {"decision": "keep"}}}))
    write_review("keep")
    assert len(load_review(source, review_json, review_csv)[0]) == 1
    write_review("exclude")
    with pytest.raises(ValueError, match="disagreement"):
        load_review(source, review_json, review_csv)


def test_diagnostics_exclude_self_and_protonation_alias():
    smiles = [PC, PROTONATED_PC, "NCCOP(=O)(O)[*:1]"]
    keys = [phosphate_comparison_key(s) for s in smiles]
    fingerprints = [fp(s) for s in smiles]
    formula = np.array([[5,13,1,3,1,0,0,0,0,0], [5,14,1,3,1,0,0,0,0,0], [2,8,1,3,1,0,0,0,0,0]])
    metrics = bank_diagnostics(fingerprints, formula, keys, [fingerprints[0], fingerprints[2]], formula[[0,2]], [keys[0], keys[2]])
    assert metrics['nearest_bank_index'].tolist() == [1, 1, 0]
    assert metrics['minimum_formula_L1_including_H'][0] > 0
    assert np.all(metrics['nearest_tanimoto'] < 1)


def test_balanced_pick_is_nested_and_avoids_close_duplicates():
    smiles = [PC, "NCCOP(=O)(O)[*:1]", "NCC(=O)OP(=O)(O)[*:1]", "N#CCOP(=O)(O)[*:1]"]
    fingerprints = [fp(s) for s in smiles]
    formula = np.array([[5,13,1,3,1,0,0,0,0,0], [2,8,1,3,1,0,0,0,0,0], [2,6,1,4,1,0,0,0,0,0], [2,5,1,3,1,0,0,0,0,0]])
    args = ([fingerprints[0]], fingerprints[1:], formula[:1], formula[1:], [1], [1,2,4], ['P1/low'], ['P1/low']*3)
    smaller, _ = balanced_pick(*args, 2)
    larger, traces = balanced_pick(*args, 4)
    assert larger[:1] == smaller
    assert len(set(larger)) == 3
    assert all(t['nearest_prior_template_tanimoto'] <= .85 for t in traces)
    keys = [phosphate_comparison_key(s) for s in smiles]
    diverse = larger
    extra = ([fingerprints[1], fingerprints[2]], [keys[1], keys[2]], [keys[0]], keys[1:], diverse)
    small, _, _ = coverage_balanced_pick(*args, 2, *extra)
    big, trace, coreset = coverage_balanced_pick(*args, 4, *extra)
    assert big[:1] == small
    assert len(set(big)) == 3
    assert coreset['probe_targets'] == 2
    assert all(t['nearest_prior_template_tanimoto'] <= .85 for t in trace)
