"""Chemistry/permission/resume regression checks with small real builder fixtures."""

import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest
import torch
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from structure_pretrain.config import ModelConfig
from structure_pretrain.data.batching import collate_structure_examples
from structure_pretrain.data.molecular_graph import fingerprint_tensor
from structure_pretrain.data.prepare_v03 import SCHEMA, canonical_polar, stratified_groups
from structure_pretrain.data.real_structure_dataset import RealStructureDataset
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.canonicalize import parse_molecule
from structure_pretrain.library_builder.chain_combinations import make_combination
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.library_builder.schemas import ELEMENTS, ChainRule, PolarRule
from structure_pretrain.models.polar_decoder import PolarDecoder
from structure_pretrain.polar_plausibility.pipeline import dump_json
from structure_pretrain.retrieval.indexed_morgan import prepare_cache
from structure_pretrain.retrieval.train_only import (
    TrainCache,
    exact_query,
    formula_stats,
    retrieve,
    tie_aware_recall,
)
from structure_pretrain.training.generation_metrics import evaluate_generation


def test_canonical_head_aliases_and_invalid_boundary():
    assert canonical_polar("[*:1]P(=O)(O)OCC") == canonical_polar("CCOP(O)(=O)[*:9]", 9)
    with pytest.raises(ValueError):
        canonical_polar("[*:1]OP(=O)(O)OCC")
    with pytest.raises(ValueError):
        canonical_polar("[*:1]P(=O)([*:2])OCC")


def test_frozen_group_splits_are_deterministic():
    rows = [
        {
            "chemistry_family": "phosphate",
            "polar_plausibility_tier": "CORE",
            "structure_origin": "pubchem_baseline",
            "token_length": 30,
        }
        for _ in range(100)
    ]
    a, b = stratified_groups(rows), stratified_groups(rows)
    np.testing.assert_array_equal(a, b)
    assert np.bincount(a).tolist() == [90, 5, 5]


def test_recall_counts_neighbors_without_penalizing_shifted_ranks_or_boundary_ties():
    truth = [(0, 1.0), (1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)]
    hits = [(0, 1.0), (2, 0.8), (3, 0.7), (4, 0.6), (99, 0.6)]
    assert tie_aware_recall(hits, truth) == 0.8
    assert tie_aware_recall([(0, 1.0), (98, 0.9), (99, 0.9)], [(0, 1.0), (1, 0.9), (2, 0.9)]) == 1


def test_autoregression_never_reads_target_prefix_and_handles_eos():
    tokenizer = SmilesTokenizer()
    decoder = PolarDecoder(ModelConfig(), tokenizer).eval()
    with torch.no_grad():
        decoder.output.weight.zero_()
        decoder.output.bias.fill_(-10)
        decoder.output.bias[tokenizer.eos_id] = 10
    memory = torch.randn(2, 3, 512)
    padding = torch.zeros(2, 3, dtype=torch.bool)
    hypotheses = decoder.generate(memory, padding, beam_size=1, max_length=8)
    assert all(row[0]["ids"] == [tokenizer.bos_id, tokenizer.eos_id] for row in hypotheses)
    assert all(row[0]["eos"] and not row[0]["length_exhausted"] for row in hypotheses)
    with torch.no_grad():
        decoder.output.bias[tokenizer.eos_id] = -20
        decoder.output.bias[tokenizer.index["C"]] = 20
    unfinished = decoder.generate(memory[:1], padding[:1], beam_size=1, max_length=4)[0][0]
    assert not unfinished["eos"] and unfinished["length_exhausted"]
    assert len(unfinished["ids"]) == 4


@pytest.fixture()
def tiny_prepared(tmp_path):
    prepared, frozen, bank = (tmp_path / name for name in ("prepared", "frozen", "bank"))
    prepared.mkdir()
    (prepared / "rows").mkdir()
    chains = {
        "A-16:0": ChainRule("A-16:0", "A", 16, 0, "[*:1]C(=O)" + "C" * 15, (), "fixture"),
        "A-18:0": ChainRule("A-18:0", "A", 18, 0, "[*:1]C(=O)" + "C" * 17, (), "fixture"),
    }
    polars = ("[*:1]P(=O)(O)OCCN", "[*:1]P(=O)(O)OCCO", "[*:1]P(=O)(O)OCC#N")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    rows, formulas, labels, groups, splits = [], [], [], [], []
    for group, text in enumerate(polars):
        polar = PolarRule(
            f"p{group}", "fixture", "fixture", text, text, 1, "other", True, True, False
        )
        for ts in (("A-16:0", "A-16:0"), ("A-16:0", "A-18:0")):
            row = assemble(chains, make_combination(ts, chains), polar)
            row["structure_id"] = (
                "pl_" + hashlib.sha256(row["canonical_smiles"].encode()).hexdigest()
            )
            row.update(
                {
                    "polar_fragment_smiles": text,
                    "chemistry_family": "phosphate",
                    "polar_plausibility_tier": "CORE",
                    "structure_origin": "pubchem_baseline",
                }
            )
            row["morgan_fp_2048"] = DataStructs.BitVectToBinaryText(
                generator.GetFingerprint(parse_molecule(row["canonical_smiles"]))
            )
            rows.append(row)
            formulas.append([row[f"neutral_formula_{e}"] for e in ELEMENTS])
            labels.append([1, 1 if ts[1] == "A-16:0" else 2])
            groups.append(group)
            splits.append(group)
    parquet = tmp_path / "structures.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    prepare_cache(parquet, frozen)
    digest = file_hash(parquet)
    tokenizer = SmilesTokenizer()
    max_length = max(len(tokenizer.encode(canonical_polar(s))) for s in polars)
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "structures_sha256": digest,
        "rows": len(rows),
        "shard_size": 2,
        "max_polar_length": max_length,
        "shards": [{"file": f"rows/{s}.arrow"} for s in range(3)],
    }
    dump_json(prepared / "labels_manifest.json", manifest)
    dump_json(prepared / "reconstruction_audit.json", {"status": "passed"})
    dump_json(
        prepared / "polar_tokenizer.json",
        {"tokens": list(tokenizer.tokens), "max_polar_length": max_length},
    )
    dump_json(prepared / "chain_vocab.json", {"tokens": ["NULL", "A-16:0", "A-18:0"]})
    for name, values in (
        ("formulas", formulas),
        ("chain_labels", labels),
        ("polar_groups", groups),
        ("row_splits", splits),
        ("group_row_counts", [2, 2, 2]),
    ):
        np.save(prepared / f"{name}.npy", np.array(values))
    for shard in range(3):
        table = pa.Table.from_pylist(
            [
                {**r, "canonical_full_smiles": r["canonical_smiles"]}
                for r in rows[shard * 2 : shard * 2 + 2]
            ]
        )
        with pa.OSFile(str(prepared / f"rows/{shard}.arrow"), "wb") as sink:
            with ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
    return prepared, frozen, bank, rows


def test_train_bank_is_enforced_before_search_and_stats_are_train_only(tiny_prepared):
    prepared, frozen, bank, _ = tiny_prepared
    report = retrieve(prepared, frozen, bank, threads=1, batch_size=2)
    assert report["held_out_reference_rows"] == 0
    for mode in ("normal", "polar_excluded"):
        refs = np.load(prepared / f"retrieval_{mode}/template_rows.npy")
        assert np.isin(refs[refs >= 0], [0, 1]).all()
        assert not (refs[0] == 0).any() and not (refs[1] == 1).any()
    # The sole train head leaves ZERO legal cross-head references, without dropping targets.
    assert report["splits"]["polar_excluded"]["train"]["zero_template_targets"] == 2
    stats = formula_stats(prepared)
    assert stats["training_pairs_only"] and stats["pair_count"] == 2
    assert stats["template_mean"][0] == np.load(prepared / "formulas.npy")[:2, 0].mean()


@pytest.mark.parametrize("encoder", ["graph", "fingerprint"])
def test_lazy_dataset_uses_real_templates_without_padding_graphs(tiny_prepared, encoder):
    prepared, frozen, bank, rows = tiny_prepared
    retrieve(prepared, frozen, bank, threads=1, batch_size=2)
    data = RealStructureDataset(
        prepared, frozen, "validation", encoder, feature_cache_size=1, shard_cache_size=1
    )
    item = data[0]
    assert len(data.feature_cache) <= 1 and len(data.store._shards) <= 1
    assert "formula_available" not in item and "structure_similarity" not in item
    torch.testing.assert_close(
        item["delta_formulas"], item["target_formula"] - item["template_formulas"]
    )
    batch = collate_structure_examples([item])
    assert batch["target_chain_tokens"].shape == (1, 2)
    assert batch["template_counts"].tolist() == [2]
    if encoder == "graph":
        assert batch["graphs"].num_graphs == 2
    else:
        torch.testing.assert_close(
            item["structures"][0], fingerprint_tensor(item["template_smiles"][0])
        )
    assert len(item["target_polar_token_ids"]) <= data.max_length
    assert len(rows) == 6
    with pytest.raises(ValueError, match="split boundary"):
        RealStructureDataset(prepared, frozen, "train", target_rows=[2])


def test_sampler_resume_matches_uninterrupted_order():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "real_training", Path(__file__).parents[1] / "train_structure_real.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sampler = module.ReproducibleBatchSampler
    whole = list(sampler(13, 3, 12, 42))
    resumed = list(sampler(13, 3, 7, 42, consumed=15))
    assert resumed == whole[5:]


def test_packed_exact_scan_matches_exhaustive_rdkit_across_chunks_and_exclusions():
    rng = np.random.default_rng(42)
    cache = object.__new__(TrainCache)
    count = 8301  # Crosses the 8192-group streaming boundary.
    vectors = np.zeros((count, 256), dtype=np.uint8)
    vectors[:, :32] = 255
    vectors[:, 32:40] = rng.integers(0, 256, size=(count, 8), dtype=np.uint8)
    cache.vectors = vectors
    cache.train_groups = np.arange(count, dtype=np.uint32)
    cache.group_of_row = np.repeat(np.arange(count), 2)
    cache.fp = vectors[cache.group_of_row]
    cache.ids = np.array([f"id_{i:08d}" for i in range(count * 2)], dtype="S67")
    cache.heads = cache.group_of_row // 12
    cache.max_head = int(np.bincount(cache.heads).max())
    cache.postings = np.arange(count * 2)
    cache.offsets = np.arange(count + 1) * 2
    refs = [DataStructs.CreateFromBinaryText(bytes(v)) for v in vectors]
    for row in (0, 35, 16386):
        scores = np.repeat(DataStructs.BulkTanimotoSimilarity(refs[row // 2], refs), 2)
        for cross in (False, True):
            legal = (cache.postings != row) & (scores >= 0.4)
            if cross:
                legal &= cache.heads != cache.heads[row]
            candidates = cache.postings[legal]
            order = np.lexsort((cache.ids[candidates], -scores[candidates]))[:10]
            gold = candidates[order].tolist()
            hits = exact_query(cache, row, cross)
            assert [r for r, _ in hits] == gold
            np.testing.assert_allclose([s for _, s in hits], scores[gold], rtol=0, atol=1e-12)


def test_generation_metrics_assemble_raw_predictions_and_count_failures(tiny_prepared):
    prepared, frozen, bank, rows = tiny_prepared
    retrieve(prepared, frozen, bank, threads=1, batch_size=2)
    data = RealStructureDataset(prepared, frozen, "validation", "fingerprint")
    batch = collate_structure_examples([data[0]])
    chains = {
        "A-16:0": ChainRule("A-16:0", "A", 16, 0, "[*:1]C(=O)" + "C" * 15, (), "fixture"),
        "A-18:0": ChainRule("A-18:0", "A", 18, 0, "[*:1]C(=O)" + "C" * 17, (), "fixture"),
    }
    ids = data.tokenizer.encode("[*:1]P(=O)(O)OCCO")

    class FixedPrediction:
        # Tests the evaluator only, not a trained model or performance estimate.
        eos = True

        def eval(self):
            return self

        def generate(self, batch, beam_size, max_length):
            logits = torch.full((1, 2, 3), -10.0)
            logits[:, :, 1] = 10.0
            candidate = {"ids": ids, "eos": self.eos, "length_exhausted": not self.eos}
            return {"chain_logits": logits, "polar_candidates": [[candidate] * beam_size]}

    fixed = FixedPrediction()
    valid, _ = evaluate_generation(
        fixed, [batch], data.tokenizer, data.chain_vocab, chains, {}, max_length=64
    )
    for key in (
        "valid_polar_rate",
        "top5_polar_recall",
        "assembly_success_rate",
        "full_formula_match_rate",
        "full_structure_exact_rate",
        "target_morgan_similarity",
    ):
        assert valid[key] == 1
    assert valid["template_copy_rate"] == 0
    fixed.eos = False
    invalid, _ = evaluate_generation(
        fixed, [batch], data.tokenizer, data.chain_vocab, chains, {}, max_length=64
    )
    assert invalid["assembly_success_rate"] == 0
    assert invalid["target_morgan_similarity"] == 0
    assert invalid["length_exhausted_rate"] == 1
