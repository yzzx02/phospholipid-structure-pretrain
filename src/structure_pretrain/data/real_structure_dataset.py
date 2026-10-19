"""Memory-mapped target shards and bounded on-demand template graph/FP caches."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from torch.utils.data import Dataset

from structure_pretrain.data.chain_vocab import ChainVocabulary
from structure_pretrain.data.molecular_graph import smiles_to_graph
from structure_pretrain.data.prepare_v03 import SCHEMA, SPLITS, read_json
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.library_builder.pipeline import file_hash


class ShardedRowStore:
    def __init__(self, root, cache_shards=4):
        self.root = Path(root)
        self.manifest = read_json(self.root / "labels_manifest.json")
        self.cache_shards = cache_shards
        self._shards = OrderedDict()

    def row(self, index):
        index = int(index)
        if not 0 <= index < self.manifest["rows"]:
            raise IndexError(index)
        shard, local = divmod(index, self.manifest["shard_size"])
        if shard not in self._shards:
            handle = pa.memory_map(str(self.root / self.manifest["shards"][shard]["file"]), "r")
            table = ipc.open_file(handle).read_all()
            self._shards[shard] = (handle, table)
            if len(self._shards) > self.cache_shards:
                _, (old, _) = self._shards.popitem(last=False)
                old.close()
        self._shards.move_to_end(shard)
        table = self._shards[shard][1]
        return {c: table[c][local].as_py() for c in table.column_names}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_shards"] = OrderedDict()
        return state


class RealStructureDataset(Dataset):
    def __init__(
        self,
        prepared,
        frozen_index,
        split,
        encoder="fingerprint",
        mode="mixed",
        target_rows=None,
        feature_cache_size=256,
        shard_cache_size=4,
        seed=42,
        cross_fraction=0.5,
        max_top_k=10,
    ):
        self.root, self.frozen = Path(prepared), Path(frozen_index)
        self.manifest = read_json(self.root / "labels_manifest.json")
        self.retrieval = read_json(self.root / "retrieval_coverage_by_split.json")
        if self.manifest["schema"] != SCHEMA or self.manifest["status"] != "complete":
            raise ValueError("Real Dataset requires v0.3 audited labels")
        for key in ("structures_sha256", "schema"):
            if self.manifest[key] != self.retrieval[key]:
                raise ValueError("Retrieval/labels version binding mismatch")
        if self.retrieval["status"] != "complete":
            raise ValueError("Full legal retrieval audit must finish before training")
        if self.retrieval["labels_manifest_sha256"] != file_hash(
            self.root / "labels_manifest.json"
        ):
            raise ValueError("Retrieval belongs to different audited labels")
        if encoder not in ("graph", "fingerprint") or mode not in (
            "normal",
            "polar_excluded",
            "mixed",
        ):
            raise ValueError("Invalid encoder/retrieval mode")
        if split not in SPLITS or not 0 <= cross_fraction <= 1 or not 1 <= max_top_k <= 10:
            raise ValueError("Invalid split or cross-head mixture")
        self.encoder, self.mode, self.split = encoder, mode, split
        self.max_top_k = max_top_k
        self.seed, self.epoch, self.cross_fraction = seed, 0, cross_fraction
        self.feature_cache_size, self.feature_cache = feature_cache_size, OrderedDict()
        self.store = ShardedRowStore(prepared, shard_cache_size)
        self.formulas = np.load(self.root / "formulas.npy", mmap_mode="r")
        self.groups = np.load(self.root / "polar_groups.npy", mmap_mode="r")
        self.chains = np.load(self.root / "chain_labels.npy", mmap_mode="r")
        self.splits = np.load(self.root / "row_splits.npy", mmap_mode="r")
        self.fp = np.load(self.frozen / "fingerprints.npy", mmap_mode="r")
        self.ids = np.load(self.frozen / "row_ids.npy", mmap_mode="r")
        frozen = read_json(self.frozen / "fingerprint_manifest.json")
        if frozen["structures_sha256"] != self.manifest["structures_sha256"]:
            raise ValueError("Frozen FP cache belongs to a different library")
        self.targets = (
            np.flatnonzero(self.splits == SPLITS.index(split))
            if target_rows is None
            else np.asarray(target_rows, dtype=np.int64)
        )
        if np.any(self.splits[self.targets] != SPLITS.index(split)):
            raise ValueError("Target sample crosses frozen split boundary")
        self.refs = {
            m: np.load(self.root / f"retrieval_{m}/template_rows.npy", mmap_mode="r")
            for m in ("normal", "polar_excluded")
        }
        for mode, array in self.refs.items():
            if array.shape != (self.manifest["rows"], 10):
                raise ValueError("Wrong library row count in retrieval cache")
            manifest = read_json(self.root / f"retrieval_{mode}/manifest.json")
            if (
                manifest["index_sha256"] != self.retrieval["index_sha256"]
                or file_hash(self.root / f"retrieval_{mode}/template_rows.npy")
                != manifest["rows_sha256"]
            ):
                raise ValueError("Retrieval cache hash/bank mismatch")
        if self.fp.shape != (self.manifest["rows"], 256) or self.chains.shape != (
            self.manifest["rows"],
            2,
        ):
            raise ValueError("Frozen v0.3 fingerprints/chain labels row shape mismatch")
        tv = read_json(self.root / "polar_tokenizer.json")
        self.tokenizer = SmilesTokenizer(tv["tokens"])
        if list(self.tokenizer.tokens) != tv["tokens"]:
            raise ValueError("Tokenizer identity/order changed")
        self.max_length = tv["max_polar_length"]
        self.chain_vocab = ChainVocabulary(read_json(self.root / "chain_vocab.json")["tokens"])

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getstate__(self):
        # Spawn workers reopen mmap files instead of pickling gigabyte arrays.
        state = dict(self.__dict__)
        for key in ("formulas", "groups", "chains", "splits", "fp", "ids", "refs"):
            state.pop(key, None)
        state["feature_cache"] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for attr, name in (
            ("formulas", "formulas"),
            ("groups", "polar_groups"),
            ("chains", "chain_labels"),
            ("splits", "row_splits"),
        ):
            setattr(self, attr, np.load(self.root / f"{name}.npy", mmap_mode="r"))
        self.fp = np.load(self.frozen / "fingerprints.npy", mmap_mode="r")
        self.ids = np.load(self.frozen / "row_ids.npy", mmap_mode="r")
        self.refs = {
            m: np.load(self.root / f"retrieval_{m}/template_rows.npy", mmap_mode="r")
            for m in ("normal", "polar_excluded")
        }

    def __len__(self):
        return len(self.targets)

    def mode_for(self, row):
        if self.split != "train":
            return "polar_excluded"
        if self.mode != "mixed":
            return self.mode
        # Stable counter-derived draw, independent of worker timing/Python hash randomization.
        key = (
            int(row) * 2654435761 + self.seed * 2246822519 + self.epoch * 3266489917
        ) & 0xFFFFFFFF
        return "polar_excluded" if key / 2**32 < self.cross_fraction else "normal"

    def feature(self, row, smiles):
        row = int(row)
        if row not in self.feature_cache:
            self.feature_cache[row] = (
                smiles_to_graph(smiles)
                if self.encoder == "graph"
                else torch.from_numpy(
                    np.unpackbits(self.fp[row], bitorder="little").astype(np.float32)
                )
            )
            if len(self.feature_cache) > self.feature_cache_size:
                self.feature_cache.popitem(last=False)
        self.feature_cache.move_to_end(row)
        return self.feature_cache[row]

    def __getitem__(self, index):
        row = int(self.targets[index])
        refs = self.refs[self.mode_for(row)][row]
        refs = refs[refs >= 0][: self.max_top_k]
        if not len(refs):
            raise ValueError(
                f"Zero-template target {bytes(self.ids[row]).decode()}; coverage must report it"
            )
        if np.any(self.splits[refs] != 0) or np.any(refs == row):
            raise ValueError("Illegal held-out/self reference")
        if self.mode_for(row) == "polar_excluded" and np.any(self.groups[refs] == self.groups[row]):
            raise ValueError("Same canonical polar graph in cross-head references")
        target = self.store.row(row)
        templates = [self.store.row(r) for r in refs]
        if target["structure_id"].encode() != self.ids[row] or any(
            t["structure_id"].encode() != self.ids[r] for r, t in zip(refs, templates)
        ):
            raise ValueError("Target/template shard row order differs from frozen fingerprint IDs")
        polar = target["polar_fragment_smiles"]
        # Use canonical labels frozen in head audit, not heuristically cut full SMILES.
        from structure_pretrain.data.prepare_v03 import canonical_polar

        polar = canonical_polar(polar)
        polar_ids = self.tokenizer.encode(polar)
        if len(polar_ids) > self.max_length:
            raise ValueError("Audited polar length exceeded; no truncation allowed")
        formula = torch.tensor(np.array(self.formulas[row]), dtype=torch.float32)
        template_formula = torch.tensor(np.array(self.formulas[refs]), dtype=torch.float32)
        if not torch.isfinite(formula).all() or formula.sum() <= 0:
            raise ValueError("Missing mandatory target formula")
        texts = [t["canonical_full_smiles"] for t in templates]
        return {
            "target_id": target["structure_id"],
            "row_index": row,
            "canonical_full_smiles": target["canonical_full_smiles"],
            "polar_fragment_smiles": polar,
            "target_chain_tokens": self.chains[row].astype(np.int64).tolist(),
            "target_polar_token_ids": polar_ids,
            "target_formula": formula,
            "template_ids": [t["structure_id"] for t in templates],
            "template_smiles": texts,
            "template_polar_smiles": [t["polar_fragment_smiles"] for t in templates],
            "template_formulas": template_formula,
            "delta_formulas": formula[None] - template_formula,
            "template_mask": torch.ones(len(refs), dtype=torch.bool),
            "structures": [self.feature(r, text) for r, text in zip(refs, texts)],
            "encoder": self.encoder,
        }


def sample_targets(
    prepared, train_count=512, validation_count=128, seed=42, priority_rows=(), sample_output=None
):
    """Frozen metadata-only stratified samples; no test structures read for training."""
    import pyarrow.parquet as pq

    prepared = Path(prepared)
    sample_output = prepared if sample_output is None else Path(sample_output)
    sample_output.mkdir(parents=True, exist_ok=True)
    split = np.load(prepared / "row_splits.npy", mmap_mode="r")
    groups = np.load(prepared / "polar_groups.npy", mmap_mode="r")
    chains = np.load(prepared / "chain_labels.npy", mmap_mode="r")
    vocab = read_json(prepared / "chain_vocab.json")["tokens"]
    p_ids = [i for i, t in enumerate(vocab) if t.startswith("P-")]
    rare_p_rows = np.flatnonzero(np.isin(chains[:, 0], p_ids))
    heads = pq.read_table(prepared / "polar_groups.parquet").to_pylist()
    strata = np.array(
        [
            "/".join(
                (
                    r["chemistry_family"],
                    r["polar_plausibility_tier"],
                    r["structure_origin"],
                    str(r["token_length"] // 32),
                )
            )
            for r in heads
        ]
    )
    rng = np.random.default_rng(seed)
    samples = {}
    for i, count in ((0, train_count), (1, validation_count)):
        eligible = np.flatnonzero(split == i)
        shuffled = rng.permutation(eligible)
        selected = list(
            dict.fromkeys(int(r) for r in np.r_[priority_rows, rare_p_rows] if split[int(r)] == i)
        )[:count]
        selected_set, covered = set(selected), set()
        lengths = np.array([h["token_length"] for h in heads])
        for length in (int(lengths.min()), int(lengths.max())):
            matches = shuffled[lengths[groups[shuffled]] == length]
            if len(matches) and int(matches[0]) not in selected_set and len(selected) < count:
                selected.append(int(matches[0]))
                selected_set.add(int(matches[0]))
        # Also represent uncommon chain labels, including rare PE-P/A if train/val assigned.
        for row in shuffled:
            if len(selected) >= count:
                break
            key = (strata[groups[row]], int(chains[row, 0]), bool(chains[row, 1] == 0))
            if key not in covered and int(row) not in selected_set:
                selected.append(int(row))
                selected_set.add(int(row))
                covered.add(key)
            if len(selected) >= count:
                break
        if len(selected) < count:
            for row in shuffled:
                if int(row) not in selected_set:
                    selected.append(int(row))
                    selected_set.add(int(row))
                if len(selected) >= count:
                    break
        samples[SPLITS[i]] = np.array(selected, dtype=np.int64)
        np.save(sample_output / f"local_smoke_{SPLITS[i]}_rows.npy", samples[SPLITS[i]])
    return samples
