"""CPU chemistry round-trips and real BOS-only generation interface checks."""

import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow.parquet as pq
import torch
from rdkit import Chem

from structure_pretrain.data.prepare_v03 import atom_counts, canonical_polar, rule_from_row
from structure_pretrain.formal_v2.common import ROOT, digest, emit, read, verify_data, write
from structure_pretrain.formal_v2.generation import (
    accept_candidate,
    fingerprints,
    generate,
    inference_inputs,
)
from structure_pretrain.library_builder.schemas import ChainRule
from structure_pretrain.library_builder.validation import attachment
from structure_pretrain.polar_only_v04.chemistry import (
    attach_known_chains,
    chainless_molecule,
    derive_polar_formula,
)
from structure_pretrain.polar_only_v04.features import fragment_molecule, packed_fingerprint
from structure_pretrain.reported_headgroups.chemistry import builder_trial
from structure_pretrain.reported_headgroups.formal_template_data import (
    FEATURES,
    V03,
    CoreData,
    make_model,
)
from structure_pretrain.reported_headgroups.template_selection import read_csv
from structure_pretrain.utils.attention import attention_backend
from train_template_guided_fp5000 import initialize


def main():
    root = ROOT.resolve()
    out = root / "audit"
    if (out / "C_passed.json").exists():
        raise FileExistsError("Preserve completed C audit")
    verify_data(root)
    initialize()
    catalog = [
        r
        for r in read_csv(
            "outputs/headgroup_template_proposal_20261022/approved_reference_catalog.csv"
        )
        if r["builder_compatible"] == "True" and r["training_reference_eligible"] == "True"
    ]
    cases = {}
    for name in ("PE", "PG", "PS"):
        cases[name] = next(r["polar_smiles"] for r in catalog if r["abbreviation"] == name)
    for label, predicate in (
        ("complex_N", lambda m: sum(a.GetAtomicNum() == 7 for a in m.GetAtoms()) >= 2),
        ("S", lambda m: any(a.GetAtomicNum() == 16 for a in m.GetAtoms())),
        ("multi_P", lambda m: sum(a.GetAtomicNum() == 15 for a in m.GetAtoms()) >= 2),
    ):
        cases[label] = next(
            r["polar_smiles"] for r in catalog if predicate(Chem.MolFromSmiles(r["polar_smiles"]))
        )
    chains = {
        r["chain_token"]: rule_from_row(ChainRule, r)
        for r in pq.read_table("data/structure_library_v0.3/chain_library.parquet").to_pylist()
    }
    vocab = read(V03 / "chain_vocab.json")["tokens"]
    checks = []
    for name, text in cases.items():
        for count in (1, 2):
            specs = [
                {
                    "chain_token": t,
                    "linkage_type": chains[t].linkage_type,
                    "carbon_count": chains[t].carbon_count,
                    "double_bond_count": chains[t].double_bond_count,
                    "fragment_smiles": chains[t].representative_fragment_smiles,
                    "db_positions": chains[t].representative_double_bond_positions,
                }
                for t in ("A-16:0", "A-18:0")[:count]
            ]
            built = builder_trial(text, specs)
            core = chainless_molecule(text, count)
            assert packed_fingerprint(fragment_molecule(Chem.MolToSmiles(core))).shape == (256,)
            tokens = [c["chain_token"] for c in specs]
            roundtrip = attach_known_chains(core, tokens, chains)
            assert roundtrip[0] == built["canonical_smiles"]
            labels = [vocab.index(tokens[0]), vocab.index(tokens[1]) if count == 2 else 0]
            formula = atom_counts(Chem.MolFromSmiles(built["canonical_smiles"]))
            polar = atom_counts(Chem.MolFromSmiles(canonical_polar(text)))
            derived = derive_polar_formula(
                formula[None],
                np.asarray([labels]),
                np.load(FEATURES / "chain_counts.npy"),
                np.load(FEATURES / "scaffold_counts.npy"),
            )[0]
            np.testing.assert_array_equal(derived, polar)
            result = accept_candidate(
                text,
                True,
                -1.0,
                chain_tokens=tokens,
                chains=chains,
                known_heads={},
                full_formula=formula,
                polar_formula=polar,
            )
            assert result["accepted"] and built["independent_element_H_balance"]
            assert not accept_candidate(
                text,
                False,
                -1.0,
                chain_tokens=tokens,
                chains=chains,
                known_heads={},
                full_formula=formula,
                polar_formula=polar,
            )["accepted"]
            fingerprints(canonical_polar(text))
            checks.append({"case": name, "chains": count, "H_balance": True, "accepted": True})
    # The anchor is the P adjacent to [*:1], even if a different P serializes first.
    original = Chem.MolFromSmiles(cases["multi_P"])
    marker, root_p = attachment(original, 1)
    other_p = next(
        a.GetIdx() for a in original.GetAtoms() if a.GetAtomicNum() == 15 and a.GetIdx() != root_p
    )
    rooted_other = Chem.MolToSmiles(original, rootedAtAtom=other_p, canonical=False)
    assert (
        rooted_other.startswith("P")
        or rooted_other.startswith("OP")
        or rooted_other.startswith("[P")
    )
    assert canonical_polar(rooted_other) == canonical_polar(cases["multi_P"])
    assembled_other = builder_trial(rooted_other)
    assert (
        assembled_other["canonical_smiles"] == builder_trial(cases["multi_P"])["canonical_smiles"]
    )

    data = CoreData(root, "cpu")
    model = make_model(root).eval()
    query = int(data.indices(1)[0])
    inputs = inference_inputs(data.batch([query]))
    # Trap any attempted teacher-forced forward call; actual decoder must start BOS.
    starts = []
    original_forward = model.polar_decoder.forward

    def traced(ids, *args, **kwargs):
        starts.append(ids[:, 0].detach().cpu().tolist())
        assert all(i == data.tokenizer.bos_id for i in starts[-1])
        return original_forward(ids, *args, **kwargs)

    model.polar_decoder.forward = traced
    began = time.monotonic()
    records = []
    with torch.inference_mode(), attention_backend("math"):
        for mode in ("raw", "elements_grammar"):
            beams = generate(
                model,
                inputs,
                data.tokenizer,
                mode=mode,
                beam_size=2,
                progress=lambda step: (
                    emit(stage="C_BOS_only_probe", mode=mode, token_step=step)
                    if step % 20 == 0
                    else None
                ),
            )
            row = {
                "mode": mode,
                "target_group": query,
                "beam_size": 2,
                "raw_candidates": [
                    {
                        "text": data.tokenizer.decode(b["ids"]),
                        "eos": b["eos"],
                        "log_probability": b["log_probability"],
                    }
                    for b in beams
                ],
            }
            records.append(row)
    assert starts
    with gzip.open(
        out / "generation_interface_probe_private.jsonl.gz", "wt", encoding="utf-8"
    ) as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    write(
        out / "C_passed.json",
        {
            "status": "passed",
            "source_sha256": digest(__file__),
            "chemistry_cases": checks,
            "all_H_included": True,
            "multiP_first_atom_not_used": True,
            "generation_probe": (
                "random model, actual raw and constrained Beam2; NOT a trained quality result"
            ),
            "target_SMILES_prefix_supplied": False,
            "BOS_first_token_calls": len(starts),
            "decoder_probe_seconds": time.monotonic() - began,
            "truth_free_acceptance_and_ranking": True,
            "unsupported_legal_categories": "out_of_scope; original structures preserved",
            "failure_denominator_tests": "required in cpu_gate pytest suite before training",
        },
    )
    emit(stage="C_completed", chemistry_roundtrips=len(checks), decoder_calls=len(starts))


if __name__ == "__main__":
    main()
