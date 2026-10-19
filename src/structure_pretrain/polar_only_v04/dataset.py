"""Bounded fragment caches, mandatory known-chain inputs, independent v0.4 batching."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.polar_only_v04.features import graph_from_record
from structure_pretrain.polar_only_v04.prepare import V03
from structure_pretrain.polar_only_v04.retrieval import chosen_mode


class FragmentStore:
    def __init__(self, root, max_shards=4, max_features=512):
        self.root = Path(root)
        self.manifest = read_json(self.root / "features_manifest.json")
        self.max_shards, self.max_features = max_shards, max_features
        self.shards, self.features = OrderedDict(), OrderedDict()

    def graph(self, variant):
        variant = int(variant)
        if variant not in self.features:
            shard, local = divmod(variant, self.manifest["shard_size"])
            if shard not in self.shards:
                path = self.root / self.manifest["shards"][shard]["file"]
                handle = pa.memory_map(str(path), "r")
                self.shards[shard] = (handle, ipc.open_file(handle).read_all())
                if len(self.shards) > self.max_shards:
                    _, (old, _) = self.shards.popitem(last=False)
                    old.close()
            self.shards.move_to_end(shard)
            table = self.shards[shard][1]
            record = {c: table[c][local].as_py() for c in table.column_names}
            if record["variant"] != variant:
                raise ValueError("Fragment feature shard order mismatch")
            self.features[variant] = graph_from_record(record)
            if len(self.features) > self.max_features:
                self.features.popitem(last=False)
        self.features.move_to_end(variant)
        return self.features[variant]

    def __getstate__(self):
        return {**self.__dict__, "shards": OrderedDict(), "features": OrderedDict()}


class PolarDataset(Dataset):
    def __init__(
        self,
        root,
        rows,
        encoder="fingerprint",
        k=3,
        force_h=False,
        split_id=0,
        h_fraction=0.8,
        seed=42,
    ):
        self.root, self.targets = Path(root), np.asarray(rows, dtype=np.int64)
        self.encoder, self.k, self.force_h = encoder, k, force_h
        if encoder not in ("fingerprint", "graph") or k not in (3, 5, 10):
            raise ValueError("Invalid independent v0.4 encoder/K")
        audit = read_json(self.root / "retrieval_audit.json")
        if audit["status"] != "passed" or audit["cache_K"] < k:
            raise ValueError("Complete independent legal retrieval audit required")
        if not (self.root / "formula_stats_train_only.json").exists():
            raise ValueError("TRAIN-only normalizer must be frozen first")
        self.formulas = np.load(V03 / "formulas.npy", mmap_mode="r")
        self.groups = np.load(V03 / "polar_groups.npy", mmap_mode="r")
        self.chains = np.load(V03 / "chain_labels.npy", mmap_mode="r")
        self.splits = np.load(V03 / "row_splits.npy", mmap_mode="r")
        if np.any(self.splits[self.targets] != split_id):
            raise ValueError("Frozen target split violation")
        self.row_variants = np.load(self.root / "row_variants.npy", mmap_mode="r")
        self.fp = np.load(self.root / "chainless_fingerprints.npy", mmap_mode="r")
        self.polar_formulas = np.load(self.root / "head_polar_formulas.npy", mmap_mode="r")
        self.refs = {m: np.load(self.root / f"{m}_rows.npy", mmap_mode="r") for m in ("H", "HC")}
        self.scores = {
            m: np.load(self.root / f"{m}_scores.npy", mmap_mode="r") for m in ("H", "HC")
        }
        counts = {m: (self.refs[m][self.targets] >= 0).sum(-1) for m in self.refs}
        self.h_mode, self.fallback = chosen_mode(
            self.targets, counts["H"], counts["HC"], fraction=h_fraction, seed=seed, force_h=force_h
        )
        if force_h and not self.h_mode.all():
            raise ValueError("Overfit requires actual same-chain different-head references")
        self.heads = pq.read_table(V03 / "polar_groups.parquet")[
            "canonical_polar_smiles"
        ].to_pylist()
        self.tokenizer = SmilesTokenizer(read_json(V03 / "polar_tokenizer.json")["tokens"])
        self.features = FragmentStore(root)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        row = int(self.targets[index])
        mode = "H" if self.h_mode[index] else "HC"
        refs = self.refs[mode][row, : self.k]
        refs = refs[refs >= 0]
        if not len(refs) or np.any(self.splits[refs] != 0):
            raise ValueError("Missing or non-TRAIN reference")
        if np.any(self.groups[refs] == self.groups[row]) or np.any(refs == row):
            raise ValueError("Reference leaks exact structure / canonical polar")
        same_chain = np.all(self.chains[refs] == self.chains[row], axis=-1)
        if np.any(same_chain != (mode == "H")):
            raise ValueError("Reference H/HC chain contract violation")
        v = self.row_variants[refs]
        f = torch.tensor(np.array(self.polar_formulas[self.groups[refs]]), dtype=torch.float32)
        target = self.heads[self.groups[row]]
        ids = self.tokenizer.encode(target)
        if len(ids) > 84:
            raise ValueError("No label truncation allowed")
        structures = (
            [self.features.graph(x) for x in v]
            if self.encoder == "graph"
            else [
                torch.tensor(np.unpackbits(self.fp[x], bitorder="little").copy()).float() for x in v
            ]
        )
        return {
            "row_index": row,
            "polar_fragment_smiles": target,
            "given_chain_tokens": np.array(self.chains[row], dtype=np.int64),
            "target_full_formula": torch.tensor(np.array(self.formulas[row])).float(),
            "target_polar_token_ids": ids,
            "template_rows": refs.tolist(),
            "template_polar_smiles": [self.heads[self.groups[r]] for r in refs],
            "template_formulas": f,
            "delta_formulas": torch.tensor(np.array(self.polar_formulas[self.groups[row]]))[None]
            - f,
            "structures": structures,
            "encoder": self.encoder,
            "mode": mode,
            "fallback": bool(self.fallback[index]),
            "retrieval_scores": self.scores[mode][row, : len(refs)].tolist(),
        }


def collate_polar(examples, pad_id=0, k=3):
    b = len(examples)
    length = max(len(x["target_polar_token_ids"]) for x in examples)
    result = {
        "target_full_formula": torch.stack([x["target_full_formula"] for x in examples]),
        "given_chain_tokens": torch.tensor(
            np.stack([x["given_chain_tokens"] for x in examples]), dtype=torch.long
        ),
        "template_formulas": torch.zeros(b, k, 10),
        "delta_formulas": torch.zeros(b, k, 10),
        "template_mask": torch.zeros(b, k, dtype=torch.bool),
        "target_polar_token_ids": torch.full((b, length), pad_id, dtype=torch.long),
    }
    structures = []
    for i, item in enumerate(examples):
        count = len(item["structures"])
        if not 0 < count <= k:
            raise ValueError("Explicit real-template count must be 1..K")
        result["template_mask"][i, :count] = True
        for name in ("template_formulas", "delta_formulas"):
            result[name][i, :count] = item[name]
        ids = item["target_polar_token_ids"]
        result["target_polar_token_ids"][i, : len(ids)] = torch.tensor(ids)
        structures.extend(item["structures"])
    if examples[0]["encoder"] == "graph":
        result["graphs"] = Batch.from_data_list(structures)
    else:
        result["fingerprints"] = torch.stack(structures)
    return result
