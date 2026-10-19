"""Free generation evaluation through the exact frozen mono-phosphatidyl builder."""

from collections import Counter

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from structure_pretrain.data.prepare_v03 import canonical_polar
from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.canonicalize import parse_molecule
from structure_pretrain.library_builder.chain_combinations import make_combination, permitted
from structure_pretrain.library_builder.schemas import ELEMENTS, BuildConfig, PolarRule


def prediction_polar_rule(smiles, known_heads):
    canonical = canonical_polar(smiles)
    if canonical in known_heads:
        return known_heads[canonical]
    # Novel heads cannot claim PE status to unlock P/A chemistry.
    return PolarRule(
        "generated", "model", "free_generation", smiles, smiles, 1, "generated", True, True, False
    )


def evaluate_generation(
    model, batches, tokenizer, chain_vocab, chains, known_heads, beam_size=5, max_length=64
):
    totals, failures, rows = Counter(), Counter(), []
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    model.eval()
    for batch in batches:
        # Model receives only template features/formula deltas. No target token prefix.
        generated = model.generate(batch, beam_size=1, max_length=max_length)
        beam = model.generate(batch, beam_size=beam_size, max_length=max_length)
        predicted = generated["chain_logits"].argmax(-1).cpu().numpy()
        true_chains = batch["target_chain_tokens"].cpu().numpy()
        for i, candidates in enumerate(generated["polar_candidates"]):
            totals["targets"] += 1
            target_graph = canonical_polar(batch["polar_fragment_smiles"][i])
            raw = candidates[0]
            text = tokenizer.decode(raw["ids"])
            totals["eos"] += int(raw["eos"])
            totals["length_exhausted"] += int(raw["length_exhausted"])
            nonnull = true_chains[i] != 0
            totals["nonnull_correct"] += int(np.sum((predicted[i] == true_chains[i]) & nonnull))
            totals["nonnull_count"] += int(nonnull.sum())
            totals["chain_count_correct"] += int(np.count_nonzero(predicted[i]) == nonnull.sum())
            totals["chain_set_exact"] += int(
                sorted(predicted[i].tolist()) == sorted(true_chains[i].tolist())
            )
            result = {
                "target_id": batch["target_id"][i],
                "generated_polar": text,
                "eos": raw["eos"],
                "length_exhausted": raw["length_exhausted"],
                "predicted_chains": [chain_vocab.tokens[j] for j in predicted[i]],
                "polar_valid": False,
                "full_assembled": False,
            }
            graphs = []
            for candidate in beam["polar_candidates"][i]:
                try:
                    graph = canonical_polar(tokenizer.decode(candidate["ids"]))
                    if candidate["eos"]:
                        graphs.append(graph)
                except ValueError:
                    pass
            totals["top5_recalled"] += int(target_graph in graphs)
            try:
                if not raw["eos"]:
                    raise ValueError("no_eos_length_exhausted")
                # Parsing validity and attachment validity are distinct metrics.
                parse_molecule(text)
                totals["valid_smiles"] += 1
                graph = canonical_polar(text)
                totals["valid_attachment"] += 1
                totals["valid_polar"] += 1
                totals["polar_exact"] += int(graph == target_graph)
                result["polar_valid"] = True
                copies = {canonical_polar(s) for s in batch["template_polar_smiles"][i]}
                totals["template_copy"] += int(graph in copies)
                rule = prediction_polar_rule(text, known_heads)
                tokens = tuple(chain_vocab.tokens[j] for j in predicted[i] if j != 0)
                combo = make_combination(tokens, chains)
                if not permitted(combo, rule, BuildConfig()):
                    raise ValueError("disallowed_chain_polar_combination")
                full = assemble(chains, combo, rule)
                totals["assembled"] += 1
                result["full_assembled"] = True
                result["generated_full_smiles"] = full["canonical_smiles"]
                counts = np.array([full[f"neutral_formula_{e}"] for e in ELEMENTS])
                totals["formula_match"] += int(
                    np.array_equal(counts, batch["target_formula"][i].cpu().numpy())
                )
                totals["full_exact"] += int(
                    full["canonical_smiles"] == batch["canonical_full_smiles"][i]
                )
                similarity = DataStructs.TanimotoSimilarity(
                    generator.GetFingerprint(parse_molecule(full["canonical_smiles"])),
                    generator.GetFingerprint(parse_molecule(batch["canonical_full_smiles"][i])),
                )
                totals["morgan_similarity_sum"] += similarity
                result["target_morgan_similarity"] = similarity
            except (ValueError, KeyError) as exc:
                failures[str(exc)] += 1
                result["failure"] = str(exc)
            rows.append(result)
    n = totals["targets"]
    metrics = {
        "evaluated_targets": n,
        "free_generation": True,
        "beam_size": beam_size,
        "generation_max_length": max_length,
        "generation_cap_is_not_target_label_truncation": True,
        "formula_post_filtering": "not applied; raw errors retained",
        "failure_counts": dict(failures),
        "non_null_chain_accuracy": totals["nonnull_correct"] / max(totals["nonnull_count"], 1),
        "target_morgan_similarity": totals["morgan_similarity_sum"] / max(n, 1),
        "target_morgan_similarity_among_assembled": (
            totals["morgan_similarity_sum"] / totals["assembled"] if totals["assembled"] else None
        ),
    }
    for name, key in (
        ("valid_polar_rate", "valid_polar"),
        ("rdkit_parse_rate", "valid_smiles"),
        ("exact_polar_graph_rate", "polar_exact"),
        ("top5_polar_recall", "top5_recalled"),
        ("attachment_valid_rate", "valid_attachment"),
        ("template_copy_rate", "template_copy"),
        ("chain_count_accuracy", "chain_count_correct"),
        ("chain_set_exact_rate", "chain_set_exact"),
        ("assembly_success_rate", "assembled"),
        ("full_formula_match_rate", "formula_match"),
        ("full_structure_exact_rate", "full_exact"),
        ("eos_rate", "eos"),
        ("length_exhausted_rate", "length_exhausted"),
    ):
        metrics[name] = totals[key] / max(n, 1)
    return metrics, rows
