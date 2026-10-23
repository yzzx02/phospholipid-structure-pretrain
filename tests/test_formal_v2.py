"""Hard gates for exact retrieval, optimizer-clock restart and compact retention."""

import copy

import numpy as np
import pytest
import torch

from structure_pretrain.formal_v2.retention import CompactCheckpoints, owned_bytes
from structure_pretrain.formal_v2.retrieval import exact_retrieve
from structure_pretrain.formal_v2.schedule import WarmupCosine


def test_exact_retrieval_ties_self_equivalents_and_small_bank():
    bank = np.zeros((40, 256), dtype=np.uint8)
    bank[:, :4] = 255
    ids = np.asarray([f"T{39 - i:03}" for i in range(40)])
    keys = np.asarray(["equivalent"] * 2 + [f"key{i}" for i in range(38)])
    idx, score, audit = exact_retrieve(bank[:1], bank, ["equivalent"], keys, ids)
    assert ids[idx[0]].tolist() == ["T000", "T001", "T002"]
    assert np.all(score == 1) and audit["boundary_tie_fallbacks"] == 1
    idx, score, _ = exact_retrieve(bank[:1], bank[:2], ["equivalent"], keys[:2], ids[:2])
    assert idx.tolist() == [[-1, -1, -1]] and np.all(score == 0)


def test_exact_retrieval_deterministic_and_matches_oracle():
    rng = np.random.default_rng(212)
    bank = rng.integers(0, 256, (43, 256), dtype=np.uint8)
    queries = bank[:11].copy()
    keys = np.arange(43).astype(str)
    first = exact_retrieve(queries, bank, keys[:11], keys, keys, batch_size=4)
    second = exact_retrieve(queries, bank, keys[:11], keys, keys, batch_size=7)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.any(first[0] == np.arange(11)[:, None])
    assert first[2]["oracle_match_fraction"] == 1


def test_scheduler_boundaries_and_update_clock():
    schedule = WarmupCosine(3e-4, 1e-5, 4000, 428720)
    assert schedule.value(1) == 3e-4 / 4000
    assert schedule.value(4000) == 3e-4
    assert schedule.value(4001) < schedule.value(4000)
    assert schedule.value(428720) == 1e-5
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))])
    with pytest.raises(ValueError):
        schedule.after_optimizer_step()
    schedule.before_optimizer_step(optimizer, 0)
    with pytest.raises(ValueError):
        schedule.before_optimizer_step(optimizer, 0)
    with pytest.raises(ValueError):
        schedule.state_dict()
    schedule.after_optimizer_step()
    with pytest.raises(ValueError):
        schedule.before_optimizer_step(optimizer, 0)


def test_real_optimizer_schedule_disk_restart_across_warmup(tmp_path):
    results = []
    for restart in (False, True):
        torch.manual_seed(7)
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 5), torch.nn.Dropout(0.3), torch.nn.Linear(5, 1)
        )
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01)
        scheduler = WarmupCosine(3e-4, 1e-5, 4000, 4008)
        rates = []
        for step in range(4008):
            rates.append(scheduler.before_optimizer_step(optimizer, step))
            optimizer.zero_grad(set_to_none=True)
            model(torch.randn(2, 3)).square().sum().backward()
            optimizer.step()
            scheduler.after_optimizer_step()
            if restart and step == 3999:
                path = tmp_path / "resume.pt"
                torch.save(
                    dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                        rng=torch.get_rng_state(),
                    ),
                    path,
                )
                model = copy.deepcopy(model)
                optimizer = torch.optim.AdamW(model.parameters())
                scheduler = WarmupCosine(3e-4, 1e-5, 4000, 4008)
                state = torch.load(path, weights_only=False)
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                torch.set_rng_state(state["rng"])
        results.append((copy.deepcopy(model.state_dict()), rates, scheduler.state_dict()))
    for key in results[0][0]:
        assert torch.equal(results[0][0][key], results[1][0][key])
    assert results[0][1:] == results[1][1:]


def checkpoint_payload(step):
    return {
        "model": {"weight": torch.tensor([float(step)])},
        "optimizer": {"state": {}},
        "scheduler": None,
        "rng": {},
        "contract": {},
        "progress": {"step": step, "epoch": step, "best_validation_loss": 1.0},
    }


def test_compact_retention_best_plus_latest_two_and_no_old_mutation(tmp_path):
    old = tmp_path / "exploratory.pt"
    old.write_bytes(b"protected old experiment")
    root = tmp_path / "new_v2"
    store = CompactCheckpoints(root / "E1", experiment_root=root)
    for step in range(1, 6):
        store.save(f"step_{step:06}_epoch_{step:03}.pt", checkpoint_payload(step), best=step == 1)
    assert sorted(p.name for p in store.folder.glob("*.pt")) == [
        "step_000001_epoch_001.pt",
        "step_000004_epoch_004.pt",
        "step_000005_epoch_005.pt",
    ]
    assert old.read_bytes() == b"protected old experiment"
    assert "pruned_verified_obsolete" in store.history.read_text()
    assert owned_bytes(root) > 0
    with pytest.raises(ValueError):
        CompactCheckpoints(tmp_path / "outside", experiment_root=root)


def test_checkpoint_nonfinite_never_promotes_or_prunes(tmp_path):
    store = CompactCheckpoints(tmp_path / "new/E1", experiment_root=tmp_path / "new")
    original = store.save("step_000001_epoch_001.pt", checkpoint_payload(1), best=True)
    state = checkpoint_payload(2)
    state["model"]["weight"][:] = float("nan")
    with pytest.raises(ValueError):
        store.save("step_000002_epoch_002.pt", state)
    assert (store.folder / original["filename"]).exists()
    assert not (store.folder / "step_000002_epoch_002.pt").exists()


def test_shared_hardlink_not_charged_to_new_budget(tmp_path):
    import os

    source = tmp_path / "shared.npy"
    source.write_bytes(b"a" * 1024)
    root = tmp_path / "v2"
    root.mkdir()
    os.link(source, root / "linked.npy")
    assert owned_bytes(root) == 0


def test_generation_boundary_removes_truth_and_rejects_prefix():
    from structure_pretrain.formal_v2.generation import generate, inference_inputs

    marker = object()
    assert inference_inputs({"fingerprints": marker, "target_polar_token_ids": marker}) == {
        "fingerprints": marker
    }
    with pytest.raises(ValueError, match="Truth token"):
        generate(None, {"target_polar_token_ids": marker}, None)


def test_marked_P_head_representation_ignores_first_serialized_P():
    from rdkit import Chem

    from structure_pretrain.formal_v2.generation import fingerprints, head_molecule, similarity

    first = "OP(=O)(O)OCCOP(=O)(O)[*:1]"
    second = "O=P(O)(OCCOP(=O)(O)O)[*:1]"
    mol = head_molecule(first)
    assert sum(a.GetAtomicNum() == 15 for a in mol.GetAtoms()) == 1
    assert similarity(fingerprints(first)[1], fingerprints(second)[1]) == 1
    assert Chem.MolToSmiles(mol)


def test_failure_candidates_and_zero_candidate_targets_stay_in_denominator():
    from structure_pretrain.formal_v2.generation import (
        StreamingAggregate,
        aggregate,
        target_metrics,
    )

    target = "NCCOP(=O)(O)[*:1]"
    candidate = {
        "polar_valid": True,
        "canonical_polar": target,
        "accepted": True,
        "rdkit_valid": True,
        "assembled": True,
        "formula_match": True,
    }
    failure = {
        "polar_valid": False,
        "accepted": False,
        "rdkit_valid": False,
        "assembled": False,
        "formula_match": False,
    }
    records = []
    for raw, strict in (([failure, candidate], [candidate]), ([], [])):
        records.append(
            {
                "raw_candidates": raw,
                "accepted_unique": strict,
                "raw_metrics": target_metrics(raw, target, [target], set()),
                "filtered_metrics": target_metrics(strict, target, [target], set()),
            }
        )
    result = aggregate(records)
    assert result["targets"] == 2 and result["raw"]["candidate_denominator"] == 2
    assert result["raw"]["1"]["Similarity"] == 0
    assert result["raw"]["5"]["Similarity"] == 0.5
    assert result["filtered"]["1"]["Exact_Hit"] == 0.5
    assert result["no_qualified_candidate_rate"] == 0.5
    assert result["raw"]["5"]["Template_Gain"] == -0.5
    streaming = StreamingAggregate()
    for row in records:
        streaming.add(row)
    assert streaming.result()["raw"]["5"] == result["raw"]["5"]
    assert (
        streaming.result()["raw"]["accepted_candidate_rate"]
        == result["raw"]["accepted_candidate_rate"]
    )


def test_acceptance_and_dedup_use_formula_and_score_without_truth():
    from rdkit import Chem

    from structure_pretrain.data.prepare_v03 import atom_counts
    from structure_pretrain.formal_v2.generation import accept_candidate, target_metrics
    from structure_pretrain.library_builder.schemas import ChainRule
    from structure_pretrain.polar_only_v04.constrained_decode import rank_unique_accepted
    from structure_pretrain.reported_headgroups.chemistry import builder_trial

    text = "NCCOP(=O)(O)[*:1]"
    built = builder_trial(text)
    chains = {
        f"A-{n}:0": ChainRule(f"A-{n}:0", "A", n, 0, "[*:1]C(=O)" + "C" * (n - 1), (), "test")
        for n in (16, 18)
    }
    inputs = dict(
        chain_tokens=list(chains),
        chains=chains,
        known_heads={},
        full_formula=atom_counts(Chem.MolFromSmiles(built["canonical_smiles"])),
        polar_formula=atom_counts(Chem.MolFromSmiles(text)),
    )
    one = accept_candidate(text, True, -2.0, **inputs)
    two = accept_candidate(text, True, -1.0, **inputs)
    assert one["accepted"] and two["accepted"]
    ranked = rank_unique_accepted([one, two])
    assert ranked == [two]
    # Target metrics may differ, but ranking/acceptance cannot change.
    target_metrics(ranked, text, [text], set())
    target_metrics(ranked, "O=P(O)(OCCO)[*:1]", [text], set())
    assert rank_unique_accepted([one, two]) == [two]
    assert not accept_candidate(text, False, -1.0, **inputs)["accepted"]
    wrong_H = dict(inputs, full_formula=inputs["full_formula"].copy())
    wrong_H["full_formula"][1] += 2
    assert not accept_candidate(text, True, -1.0, **wrong_H)["accepted"]
