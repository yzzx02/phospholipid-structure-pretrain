"""Read-only integration and provisional reference-bank selection helpers."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, includeChirality=False
)
FUNCTION_SMARTS = {
    "amine": "[NX3;!$(N-C=O);!$(N=P)]",
    "quaternary_ammonium": "[N+X4]",
    "alcohol_or_phenol": "[O;H1]-[#6;!$(C=O)]",
    "carboxyl_or_carboxylate": "[CX3](=O)[O;H1,-1]",
    "amide": "[NX3]-[CX3](=O)",
    "ester": "[CX3](=O)-[OX2]-[#6]",
    "ether": "[#6]-[OX2]-[#6]",
    "nitrile": "[CX2]#[NX1]",
    "heterocycle": "[!#6;!#1;!#0;R]",
    "aromatic_heterocycle": "[n,o,s]",
    "carbonyl_other": "[#6;!$(C(=O)[O,N])]=[OX1]",
    "sulfur": "[#16]",
    "halogen": "[F,Cl,Br,I]",
    "P_C_bond": "[#15]~[#6]",
    "P_S_bond": "[#15]~[#16]",
    "P_N_bond": "[#15]~[#7]",
}
FUNCTION_NAMES = tuple(FUNCTION_SMARTS)
FUNCTION_PATTERNS = tuple(Chem.MolFromSmarts(s) for s in FUNCTION_SMARTS.values())
FORMULA_ELEMENTS = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I")
FORMULA_SCALES = np.array([6, 12, 2, 4, 1, 1, 1, 1, 1, 1], dtype=np.float64)


def functional_mask(mol):
    return sum(1 << i for i, pattern in enumerate(FUNCTION_PATTERNS)
               if mol.HasSubstructMatch(pattern))


def functional_names(mask):
    return [name for i, name in enumerate(FUNCTION_NAMES) if int(mask) & (1 << i)]


def current_compatibility(smiles, vocabulary):
    """Separate template feature entry gates from Decoder target-label gates."""
    from structure_pretrain.data.smiles_tokenizer import TOKEN_PATTERN
    from structure_pretrain.polar_only_v04.chemistry import chainless_smiles
    from structure_pretrain.polar_only_v04.features import fragment_molecule, packed_fingerprint
    feature_errors = []
    try:
        for chain_count in (1, 2):
            mol = fragment_molecule(chainless_smiles(smiles, chain_count))
            assert packed_fingerprint(mol).shape == (256,)
    except (ValueError, RuntimeError) as exc:
        feature_errors.append(getattr(exc, 'reason', str(exc)))
    pieces = TOKEN_PATTERN.findall(smiles)
    label_errors = []
    if len(pieces) + 2 > 84:
        label_errors.append('target_label_length_above_84')
    unknown = set(pieces) - set(vocabulary)
    if unknown:
        label_errors.append('target_label_vocabulary:' + ','.join(sorted(unknown)))
    return {'training_reference_eligible': not feature_errors,
            'training_reference_blockers': ';'.join(feature_errors),
            'target_label_eligible': not feature_errors and not label_errors,
            'target_label_blockers': ';'.join(feature_errors + label_errors)}


def coverage_balanced_pick(seed_fps, candidate_fps, seed_formula, candidate_formula,
                          seed_masks, candidate_masks, seed_strata, candidate_strata,
                          count, probe_fps, probe_keys, seed_keys, candidate_keys,
                          diverse_picks, random_seed=42):
    """Facility-location coverage on a reproducible coreset plus diversity terms.

    Use 4,000 uniform candidate representatives plus the diverse MaxMin proposal;
    score coverage on 4,096 uniform full-pool probes. Full-pool exact diagnostics
    are computed afterwards, independently of this sampled optimization.
    """
    from collections import Counter
    rng = np.random.default_rng(random_seed)
    representatives = sorted(set(rng.choice(len(candidate_fps), min(4000, len(candidate_fps)), replace=False).tolist()) | set(diverse_picks))
    fps = [candidate_fps[i] for i in representatives]
    compositions = np.asarray(candidate_formula)[representatives]
    masks = np.asarray(candidate_masks)[representatives]
    strata = [candidate_strata[i] for i in representatives]
    keys = [candidate_keys[i] for i in representatives]
    nearest = nearest_similarities(fps, seed_fps)
    distances = np.full(len(fps), np.inf)
    probe_best = np.zeros(len(probe_fps), dtype=np.float32)
    for reference, composition, key in zip(seed_fps, seed_formula, seed_keys):
        np.minimum(distances, np.mean(np.abs(compositions-composition)/FORMULA_SCALES, axis=1), out=distances)
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(reference, probe_fps), dtype=np.float32)
        sims[np.asarray(probe_keys) == key] = 0
        np.maximum(probe_best, sims, out=probe_best)
    matrix = np.asarray([DataStructs.BulkTanimotoSimilarity(fingerprint, probe_fps) for fingerprint in fps], dtype=np.float32)
    probe_keys = np.asarray(probe_keys)
    for i, key in enumerate(keys):
        matrix[i, probe_keys == key] = 0
    selected_counts = Counter(seed_strata)
    population = Counter(candidate_strata)
    weights = {k: np.sqrt(v) for k, v in population.items()}
    weight_total = sum(weights.values())
    covered = 0
    for mask in seed_masks:
        covered |= int(mask)
    chosen, traces = [], []
    for step in range(count-len(seed_fps)):
        gains = np.mean(np.maximum(matrix-probe_best, 0), axis=1)
        expected = {k: (len(seed_fps)+step+1)*w/weight_total for k,w in weights.items()}
        balance = np.asarray([1+.25*max(0,(expected[k]-selected_counts[k])/expected[k]) for k in strata])
        unseen = np.asarray([(int(m)&~covered).bit_count() for m in masks])
        utility = balance * (.6*gains/max(float(gains.max()), 1e-12)
                             +.25*(1-nearest)+.1*np.minimum(distances,1)+.05*np.minimum(unseen,1))
        utility[nearest > .85] = -np.inf
        utility[chosen] = -np.inf
        index = int(np.argmax(utility))
        if not np.isfinite(utility[index]):
            raise ValueError("Coverage coreset has too few distinct candidates")
        traces.append({"nearest_prior_template_tanimoto": float(nearest[index]),
                       "minimum_scaled_formula_distance_to_prior": float(distances[index]),
                       "new_functional_categories": functional_names(int(masks[index])&~covered),
                       "stratum": strata[index], "selection_utility": float(utility[index]),
                       "probe_mean_similarity_gain": float(gains[index])})
        chosen.append(index)
        selected_counts[strata[index]] += 1
        covered |= int(masks[index])
        np.maximum(probe_best, matrix[index], out=probe_best)
        np.maximum(nearest, DataStructs.BulkTanimotoSimilarity(fps[index], fps), out=nearest)
        np.minimum(distances, np.mean(np.abs(compositions-compositions[index])/FORMULA_SCALES, axis=1), out=distances)
    return [representatives[i] for i in chosen], traces, {
        "candidate_coreset": len(representatives), "probe_targets": len(probe_fps),
        "random_seed": random_seed, "sampling": "uniform without replacement, plus diversity proposal",
        "objective_weights": {"probe_facility_gain": .6, "Morgan_novelty": .25,
                              "formula_novelty": .1, "new_functional_category": .05},
        "mass_P_stratum_max_bonus": 1.25,
    }


def nearest_without_self(queries, query_keys, references, reference_keys):
    best = np.zeros(len(queries), dtype=float)
    query_keys = np.asarray(query_keys)
    for reference, key in zip(references, reference_keys):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(reference, queries))
        sims[query_keys == key] = 0
        np.maximum(best, sims, out=best)
    return best


def balanced_pick(seed_fps, candidate_fps, seed_formula, candidate_formula,
                  seed_masks, candidate_masks, seed_strata, candidate_strata, count):
    """Greedy diversity selection, balanced across mass/P strata.

    Morgan novelty 70%, scaled elemental novelty 20%, previously uncovered
    functional categories 10%. A bounded stratum bonus follows square-root
    population quotas, so small and large formula ranges both receive coverage.
    New picks must have Morgan similarity <= .85 to every previous template.
    """
    from collections import Counter

    needed = count - len(seed_fps)
    if needed < 0:
        raise ValueError("Bank smaller than the approved compatible seed bank")
    nearest = nearest_similarities(candidate_fps, seed_fps)
    formula = np.asarray(candidate_formula, dtype=np.float64)
    formula_distance = np.full(len(formula), np.inf)
    for reference in seed_formula:
        np.minimum(formula_distance,
                   np.mean(np.abs(formula - reference) / FORMULA_SCALES, axis=1),
                   out=formula_distance)
    strata_counts = Counter(candidate_strata)
    weights = {k: np.sqrt(v) for k, v in strata_counts.items()}
    total_weight = sum(weights.values())
    selected_counts = Counter(seed_strata)
    covered = 0
    for mask in seed_masks:
        covered |= int(mask)
    chosen, traces = [], []
    for step in range(needed):
        expected = {k: (len(seed_fps) + step + 1) * w / total_weight for k, w in weights.items()}
        balance = np.array([1 + .5 * max(0, (expected[k] - selected_counts[k]) / expected[k])
                            for k in candidate_strata])
        unseen = np.array([(int(m) & ~covered).bit_count() for m in candidate_masks])
        utility = balance * (.7 * (1 - nearest) + .2 * np.minimum(formula_distance, 1)
                             + .1 * np.minimum(unseen, 1))
        utility[nearest > .85] = -np.inf
        utility[chosen] = -np.inf
        index = int(np.argmax(utility))
        if not np.isfinite(utility[index]):
            raise ValueError("Not enough distinct candidates under .85 similarity ceiling")
        traces.append({"nearest_prior_template_tanimoto": float(nearest[index]),
                       "minimum_scaled_formula_distance_to_prior": float(formula_distance[index]),
                       "new_functional_categories": functional_names(int(candidate_masks[index]) & ~covered),
                       "stratum": candidate_strata[index], "selection_utility": float(utility[index])})
        chosen.append(index)
        selected_counts[candidate_strata[index]] += 1
        covered |= int(candidate_masks[index])
        np.maximum(nearest, DataStructs.BulkTanimotoSimilarity(candidate_fps[index], candidate_fps), out=nearest)
        np.minimum(formula_distance,
                   np.mean(np.abs(formula - formula[index]) / FORMULA_SCALES, axis=1),
                   out=formula_distance)
    return chosen, traces


def bank_diagnostics(query_fps, query_formula, query_keys, bank_fps, bank_formula, bank_keys):
    """Full-pool exact distances excluding each query's own protonation group."""
    n = len(query_fps)
    formula = np.asarray(query_formula, dtype=np.int16)
    nearest = np.full(n, -1., dtype=np.float64)
    nearest_index = np.full(n, -1, dtype=np.int32)
    formula_l1 = np.full(n, np.inf)
    heavy_l1 = np.full(n, np.inf)
    scaled_l1 = np.full(n, np.inf)
    structural_formula_l1 = np.full(n, np.inf)
    query_keys = np.asarray(query_keys, dtype=object)
    for i, (reference, composition, key) in enumerate(zip(bank_fps, bank_formula, bank_keys)):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(reference, query_fps))
        same = query_keys == key
        sims[same] = -1
        better = sims > nearest
        delta = np.abs(formula - composition)
        full = delta.sum(axis=1).astype(float)
        structural_formula_l1[better] = full[better]
        nearest[better] = sims[better]
        nearest_index[better] = i
        heavy = full - delta[:, 1]
        scaled = np.mean(delta / FORMULA_SCALES, axis=1)
        full[same] = heavy[same] = scaled[same] = np.inf
        np.minimum(formula_l1, full, out=formula_l1)
        np.minimum(heavy_l1, heavy, out=heavy_l1)
        np.minimum(scaled_l1, scaled, out=scaled_l1)
    return {"nearest_tanimoto": nearest, "nearest_bank_index": nearest_index,
            "minimum_formula_L1_including_H": formula_l1,
            "minimum_heavy_element_L1": heavy_l1,
            "minimum_scaled_formula_L1": scaled_l1,
            "structural_nearest_formula_L1_including_H": structural_formula_l1}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False)
                             if isinstance(row.get(k), (dict, list)) else row.get(k, "")
                             for k in fields})


def load_review(source_csv, review_json, review_csv):
    """Require matching JSON/CSV decisions, graphs and immutable source binding."""
    rows = read_csv(source_csv)
    by_id = {r["headgroup_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate source headgroup ID")
    document = json.loads(Path(review_json).read_text(encoding="utf-8-sig"))
    if document.get("schema") != "headgroup-review-v01":
        raise ValueError("Unsupported review schema")
    if document.get("input_sha256") != digest(source_csv):
        raise ValueError("Review is bound to a different source library")
    reviewed = read_csv(review_csv)
    ids = [r["headgroup_id"] for r in reviewed]
    if len(set(ids)) != len(ids) or set(ids) != set(by_id):
        raise ValueError("Review CSV must contain each source ID exactly once")
    decisions = document["decisions"]
    if not set(decisions) <= set(by_id):
        raise ValueError("Unknown reviewed ID")
    for row in reviewed:
        identifier = row["headgroup_id"]
        decision = decisions.get(identifier, {})
        if decision.get("decision", "") not in {"", "keep", "exclude", "review"}:
            raise ValueError("Unknown decision")
        for key, expected in (
            ("polar_smiles", by_id[identifier]["polar_smiles"]),
            ("human_decision", decision.get("decision", "")),
            ("human_note", decision.get("note", "")),
        ):
            if row[key] != expected:
                raise ValueError(f"Review JSON/CSV/source disagreement: {identifier}/{key}")
    return rows, document


def phosphate_comparison_key(smiles):
    """Ignore only terminal phosphate-O protonation for grouping, never as a label.

    Preserve the dummy/root, every heavy atom/bond, P=S/P=C, ring and positional
    connectivity. Actual approved neutral labels remain untouched.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    for atom in mol.GetAtoms():
        if (atom.GetAtomicNum() == 8 and atom.GetFormalCharge() == 0
                and atom.GetDegree() == 1 and atom.GetTotalNumHs() == 1):
            neighbor = atom.GetNeighbors()[0]
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            if neighbor.GetAtomicNum() == 15 and bond.GetBondType() == Chem.BondType.SINGLE:
                atom.SetFormalCharge(-1)
                atom.SetNumExplicitHs(0)
                atom.SetNoImplicit(True)
    Chem.RemoveStereochemistry(mol)
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def methylene_contractions(smiles):
    """Exact graph checks for a single non-ring, degree-two CH2 insertion."""
    mol = Chem.MolFromSmiles(smiles)
    keys = set()
    for atom in mol.GetAtoms():
        if (atom.GetAtomicNum() != 6 or atom.GetTotalNumHs() != 2 or atom.GetDegree() != 2
                or atom.IsInRing() or atom.GetFormalCharge() or atom.GetIsAromatic()
                or any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds())):
            continue
        left, right = [n.GetIdx() for n in atom.GetNeighbors()]
        if mol.GetBondBetweenAtoms(left, right) is not None:
            continue
        editable = Chem.RWMol(mol)
        editable.AddBond(left, right, Chem.BondType.SINGLE)
        editable.RemoveAtom(atom.GetIdx())
        shorter = editable.GetMol()
        try:
            Chem.SanitizeMol(shorter)
            keys.add(phosphate_comparison_key(Chem.MolToSmiles(shorter, isomericSmiles=False)))
        except ValueError:
            pass
    return keys


def fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    return MORGAN.GetFingerprint(mol)


def seeded_maxmin(seed_fps, candidate_fps, count, seed=42):
    """Exact on-demand Morgan distance; no quadratic distance matrix."""
    if count < len(seed_fps) or count > len(seed_fps) + len(candidate_fps):
        raise ValueError("Requested bank size outside the available pool")
    all_fps = seed_fps + candidate_fps
    picks = list(MaxMinPicker().LazyBitVectorPick(
        all_fps, len(all_fps), count, firstPicks=list(range(len(seed_fps))), seed=seed
    ))
    return [i - len(seed_fps) for i in picks[len(seed_fps):]]


def nearest_similarities(queries, references):
    if not references:
        return np.zeros(len(queries), dtype=np.float64)
    # Updating one column at a time keeps memory linear in the query pool.
    best = np.zeros(len(queries), dtype=np.float64)
    for reference in references:
        np.maximum(best, DataStructs.BulkTanimotoSimilarity(reference, queries), out=best)
    return best


def coverage(values):
    values = np.asarray(values)
    return {
        "targets": len(values),
        "mean_nearest_tanimoto": float(np.mean(values)),
        "median_nearest_tanimoto": float(np.median(values)),
        "p10_nearest_tanimoto": float(np.quantile(values, .1)),
        **{f"fraction_at_least_{t}": float(np.mean(values >= t)) for t in (.3, .4, .5, .6, .7)},
    }
