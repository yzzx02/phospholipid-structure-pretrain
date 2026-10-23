"""Truth-free generation/acceptance; target graphs enter only AFTER ranking."""

import itertools
import math
from collections import Counter

import numpy as np
from rdkit import Chem, DataStructs, rdBase

from structure_pretrain.data.prepare_v03 import atom_counts, canonical_polar
from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.chain_combinations import make_combination, permitted
from structure_pretrain.library_builder.schemas import ELEMENTS, BuildConfig, PolarRule
from structure_pretrain.library_builder.validation import attachment
from structure_pretrain.polar_only_v04.chemistry import chainless_molecule
from structure_pretrain.polar_only_v04.constrained_decode import (
    constrained_beam_search,
    rank_unique_accepted,
)
from structure_pretrain.reported_headgroups.template_selection import (
    MORGAN,
    functional_mask,
    phosphate_comparison_key,
)
from structure_pretrain.training.quality_v03 import polar_checks


def inference_inputs(batch):
    # The generation API has no target token or target graph argument.
    return {key: value for key, value in batch.items() if key != "target_polar_token_ids"}


def generate(model, batch, tokenizer, *, mode="raw", beam_size=20, progress=None):
    if "target_polar_token_ids" in batch:
        raise ValueError("Truth token sequence forbidden at inference boundary")
    encoded = model.encode_templates(batch)
    memory, padding = encoded["template_memory"], encoded["memory_padding_mask"]
    if memory.shape[0] != 1:
        raise ValueError("One independently referenced target per decode")
    if mode == "raw":
        return model.polar_decoder.generate(memory, padding, beam_size, 84, progress=progress)[0]
    if mode != "elements_grammar":
        raise ValueError("Unknown frozen decoding mode")
    return constrained_beam_search(
        model.polar_decoder,
        memory,
        padding,
        tokenizer,
        encoded["target_polar_formula"][0].cpu().numpy(),
        beam_size=beam_size,
        max_length=84,
        grammar=True,
        progress=progress,
    )


def accept_candidate(
    text, eos, score, *, chain_tokens, chains, known_heads, full_formula, polar_formula
):
    """No true target graph is accepted by this function or by rank/dedup."""
    if not math.isfinite(score):
        raise ValueError("Nonfinite decoder score")
    with rdBase.BlockLogs():
        parsed = Chem.MolFromSmiles(text)
    result = {
        **polar_checks(text, eos),
        "text": text,
        "decoder_score": float(score),
        "accepted": False,
        "assembled": False,
        "formula_match": False,
        "polar_formula_match_including_H": False,
        "rdkit_valid": parsed is not None and parsed.GetNumAtoms() > 0,
    }
    if not result["polar_valid"]:
        if result.get("reason") in (
            "intact_lipid_as_polar",
            "unsupported_element",
            "unsupported_isotope",
        ):
            result["category"] = "out_of_scope"
        return result
    graph = result["canonical_polar"]
    result["polar_formula_match_including_H"] = bool(
        np.array_equal(atom_counts(Chem.MolFromSmiles(graph)), polar_formula)
    )
    rule = known_heads.get(
        graph,
        PolarRule(
            "generated", "model", "free_generation", text, text, 1, "generated", True, True, False
        ),
    )
    try:
        combo = make_combination(tuple(t for t in chain_tokens if t != "NULL"), chains)
        if not permitted(combo, rule, BuildConfig()):
            return {
                **result,
                "category": "out_of_scope",
                "reason": "unsupported_chain_polar_combination",
            }
        with rdBase.BlockLogs():
            full = assemble(chains, combo, rule)
    except (ValueError, KeyError) as exc:
        reason = getattr(exc, "reason", str(exc))
        category = (
            "out_of_scope"
            if reason in ("intact_lipid_as_polar", "unsupported_element")
            else "builder_failure"
        )
        return {**result, "category": category, "reason": reason}
    result["assembled"] = True
    result["formula_match"] = bool(
        np.array_equal([full[f"neutral_formula_{e}"] for e in ELEMENTS], full_formula)
    )
    result["accepted"] = bool(
        eos and result["formula_match"] and result["polar_formula_match_including_H"]
    )
    result["category"] = "accepted" if result["accepted"] else "formula_mismatch"
    return result


def score_beams(beams, tokenizer, **chemical_inputs):
    rows = []
    for rank, beam in enumerate(beams, 1):
        if beam["ids"][0] != tokenizer.bos_id:
            raise ValueError("Generated beam must begin with BOS")
        actual_eos = beam["ids"][-1] == tokenizer.eos_id
        if actual_eos != beam["eos"]:
            raise ValueError("EOS metadata differs from actual last token")
        row = accept_candidate(
            tokenizer.decode(beam["ids"]),
            actual_eos,
            beam["log_probability"] / (len(beam["ids"]) - 1),
            **chemical_inputs,
        )
        row.update(
            source_rank=rank,
            log_probability=beam["log_probability"],
            generated_tokens=len(beam["ids"]) - 1,
            length_exhausted=beam["length_exhausted"],
        )
        rows.append(row)
    # Raw order remains the decoder's order, including invalid hypotheses.
    return rows, rank_unique_accepted(rows)


def head_molecule(text):
    """Delete ONLY marked anchor P, marker and its terminal O/S ligands.

    All remaining components, including further P atoms, are retained. This is a
    declared variable-head similarity representation, never a Builder repair.
    """
    mol = Chem.MolFromSmiles(text)
    marker, root = attachment(mol, 1)
    if mol.GetAtomWithIdx(root).GetAtomicNum() != 15:
        raise ValueError("Marked attachment neighbour must be P")
    remove = {marker, root}
    remove.update(
        a.GetIdx()
        for a in mol.GetAtomWithIdx(root).GetNeighbors()
        if a.GetAtomicNum() in (8, 16) and a.GetDegree() == 1
    )
    edit = Chem.RWMol(mol)
    for i in sorted(remove, reverse=True):
        edit.RemoveAtom(i)
    result = edit.GetMol()
    if result.GetNumAtoms():
        Chem.SanitizeMol(result)
    return result


def fingerprints(text):
    mol = Chem.MolFromSmiles(text)
    head = head_molecule(text)
    return MORGAN.GetFingerprint(mol), MORGAN.GetFingerprint(head) if head.GetNumAtoms() else None


def similarity(left, right):
    if left is None or right is None:
        return float(left is None and right is None)
    return float(DataStructs.TanimotoSimilarity(left, right))


def target_metrics(
    candidates, target, templates, train_keys, *, top_values=(1, 5, 10), chain_count=2
):
    # Called after acceptance/ranking; no mutation or reordering of candidates.
    gold = canonical_polar(target)
    gold_fp, gold_head = fingerprints(gold)
    gold_core = MORGAN.GetFingerprint(chainless_molecule(gold, chain_count))
    gold_mask = functional_mask(Chem.MolFromSmiles(gold))
    template_similarities = [similarity(fingerprints(t)[0], gold_fp) for t in templates]
    template_best = max(template_similarities, default=0.0)
    template_best_core = max(
        (
            similarity(MORGAN.GetFingerprint(chainless_molecule(t, chain_count)), gold_core)
            for t in templates
        ),
        default=0.0,
    )
    values = []
    for candidate in candidates:
        value = {
            "similarity": 0.0,
            "head_similarity": 0.0,
            "polar_similarity": 0.0,
            "functional_recall": 0.0,
            "exact": False,
            "novel": False,
            "fingerprint": None,
        }
        if candidate["polar_valid"]:
            text = candidate["canonical_polar"]
            fp, head = fingerprints(text)
            mask = functional_mask(Chem.MolFromSmiles(text))
            value.update(
                similarity=similarity(
                    MORGAN.GetFingerprint(chainless_molecule(text, chain_count)), gold_core
                ),
                polar_similarity=similarity(fp, gold_fp),
                head_similarity=similarity(head, gold_head),
                functional_recall=(
                    int(mask & gold_mask).bit_count() / int(gold_mask).bit_count()
                    if gold_mask
                    else 1.0
                ),
                exact=text == gold,
                novel=phosphate_comparison_key(text) not in train_keys,
                fingerprint=fp,
            )
        values.append(value)
    output = {
        "candidate_count": len(candidates),
        "template_best_Polar_Tanimoto": template_best,
        "template_best_chainless_Tanimoto": template_best_core,
    }
    for k in top_values:
        prefix = values[:k]
        best = max((v["similarity"] for v in prefix), default=0.0)
        output[str(k)] = {
            "Similarity": best,
            "Polar_Similarity": max((v["polar_similarity"] for v in prefix), default=0.0),
            "Headgroup_Similarity": max((v["head_similarity"] for v in prefix), default=0.0),
            "Functional_group_Recall": max((v["functional_recall"] for v in prefix), default=0.0),
            "Exact_Hit": any(v["exact"] for v in prefix),
            "High_Similarity_ge08": best >= 0.8,
            "Template_Gain": best - template_best_core,
            "Polar_Template_Gain": max((v["polar_similarity"] for v in prefix), default=0.0)
            - template_best,
            "positive_Template_Gain": best > template_best_core + 1e-12,
            "Novelty_among_valid": sum(v["novel"] for v in prefix)
            / max(sum(v["fingerprint"] is not None for v in prefix), 1),
        }
    valid = [v["fingerprint"] for v in values[:10] if v["fingerprint"] is not None]
    diversity = [1 - similarity(a, b) for a, b in itertools.combinations(valid, 2)]
    output["diversity_top10"] = float(np.mean(diversity)) if diversity else 0.0
    return output


def aggregate(records):
    if not records:
        raise ValueError("Empty cohort forbidden")
    n = len(records)
    out = {
        "targets": n,
        "raw_candidates": sum(len(r["raw_candidates"]) for r in records),
        "no_qualified_candidate_rate": sum(not r["accepted_unique"] for r in records) / n,
        "all_targets_in_denominator": True,
    }
    for scope in ("raw", "filtered"):
        out[scope] = {
            str(k): {
                name: float(np.mean([r[scope + "_metrics"][str(k)][name] for r in records]))
                for name in records[0][scope + "_metrics"][str(k)]
            }
            for k in (1, 5, 10)
        }
        candidates = [
            c for r in records for c in r["raw_candidates" if scope == "raw" else "accepted_unique"]
        ]
        out[scope]["candidate_denominator"] = len(candidates)
        out[scope]["per_target_candidate_count_mean"] = len(candidates) / n
        for key in ("rdkit_valid", "polar_valid", "assembled", "formula_match", "accepted"):
            out[scope][key + "_candidate_rate"] = sum(c[key] for c in candidates) / max(
                len(candidates), 1
            )
            out[scope][key + "_top1_over_all_targets"] = (
                sum(
                    bool(
                        r["raw_candidates" if scope == "raw" else "accepted_unique"]
                        and r["raw_candidates" if scope == "raw" else "accepted_unique"][0][key]
                    )
                    for r in records
                )
                / n
            )
    return out


class StreamingAggregate:
    """Fixed-size numeric totals; never retain the cohort's candidate graphs."""

    def __init__(self):
        self.n = 0
        self.no_candidate = 0
        self.sums = {}
        self.counts = Counter()
        self.failures = Counter()
        self.legal_counts = Counter()

    def add(self, record):
        one = aggregate([record])
        self.n += 1
        self.no_candidate += not record["accepted_unique"]
        self.legal_counts[len(record["accepted_unique"])] += 1
        self.failures.update(
            c.get("reason", c.get("category", "unknown"))
            for c in record["raw_candidates"]
            if not c["accepted"]
        )
        for scope in ("raw", "filtered"):
            self.counts[scope] += one[scope]["candidate_denominator"]
            for k in ("1", "5", "10"):
                for name, value in one[scope][k].items():
                    key = (scope, k, name)
                    self.sums[key] = self.sums.get(key, 0.0) + value
            for name, value in one[scope].items():
                if name.endswith("_candidate_rate") or name.endswith("_top1_over_all_targets"):
                    key = (scope, name)
                    weight = (
                        one[scope]["candidate_denominator"]
                        if name.endswith("_candidate_rate")
                        else 1
                    )
                    self.sums[key] = self.sums.get(key, 0.0) + value * weight
            key = (scope, "diversity_top10")
            self.sums[key] = self.sums.get(key, 0.0) + record[scope + "_metrics"]["diversity_top10"]

    def result(self):
        if not self.n:
            raise ValueError("Empty cohort forbidden")
        out = {
            "targets": self.n,
            "raw_candidates": self.counts["raw"],
            "no_qualified_candidate_rate": self.no_candidate / self.n,
            "all_targets_in_denominator": True,
            "accepted_isomer_count_histogram": dict(self.legal_counts),
            "raw_failure_reasons": dict(self.failures),
        }
        for scope in ("raw", "filtered"):
            values = {str(k): {} for k in (1, 5, 10)}
            for key, total in self.sums.items():
                if key[0] != scope:
                    continue
                if len(key) == 3:
                    values[key[1]][key[2]] = total / self.n
                else:
                    denominator = (
                        self.counts[scope] if key[1].endswith("_candidate_rate") else self.n
                    )
                    values[key[1]] = total / max(denominator, 1)
            values.update(
                candidate_denominator=self.counts[scope],
                per_target_candidate_count_mean=self.counts[scope] / self.n,
            )
            out[scope] = values
        return out
