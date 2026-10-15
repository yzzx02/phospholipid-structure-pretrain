import csv
import itertools
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

ALLOWED = {"C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
SIMPLE_COUNTERIONS = {
    "[Li+]",
    "[Na+]",
    "[K+]",
    "[Mg+2]",
    "[Ca+2]",
    "[NH4+]",
    "[F-]",
    "[Cl-]",
    "[Br-]",
    "[I-]",
}
DEFAULT_LIMITS = {
    "max_molecular_weight": 600,
    "max_carbon_count": 30,
    "max_heavy_atoms": 60,
    "max_phosphorus_count": 3,
    "max_long_aliphatic_carbon_run": 10,
}
UNCHARGER = rdMolStandardize.Uncharger(canonicalOrder=True)


class Reject(ValueError):
    def __init__(self, reason, detail=""):
        super().__init__(detail or reason)
        self.reason = reason


def canonical(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def read_controls(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def select_component(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise Reject("parse_or_sanitize_failure")
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        raise Reject("isotope_labels")
    components = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    p_parts = [m for m in components if any(a.GetAtomicNum() == 15 for a in m.GetAtoms())]
    if len(p_parts) > 1:
        raise Reject("multiple_p_components")
    if not p_parts:
        raise Reject("no_phosphorus")
    selected = p_parts[0]
    removed = []
    for component in components:
        if component is selected:
            continue
        smi = canonical(component)
        if smi not in SIMPLE_COUNTERIONS:
            if any(a.GetSymbol() not in ALLOWED for a in component.GetAtoms()):
                raise Reject("unsupported_elements", smi)
            raise Reject("unsupported_cocomponent", smi)
        removed.append(smi)
    if any(a.GetSymbol() not in ALLOWED for a in selected.GetAtoms()):
        raise Reject("unsupported_elements")
    if any(a.GetNumRadicalElectrons() for a in selected.GetAtoms()):
        raise Reject("radicals")
    return selected, removed


def phosphorus_family(mol):
    """Reject the entire component if ANY P has unsupported local chemistry."""
    has_pc, has_pop = False, False
    for p in (a for a in mol.GetAtoms() if a.GetAtomicNum() == 15):
        oxo, single_o, single_c = 0, 0, 0
        for bond in p.GetBonds():
            other = bond.GetOtherAtom(p)
            if other.GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.DOUBLE:
                oxo += 1
            elif bond.GetBondType() == Chem.BondType.SINGLE and other.GetAtomicNum() == 8:
                single_o += 1
                has_pop |= any(
                    n.GetAtomicNum() == 15 and n.GetIdx() != p.GetIdx()
                    for n in other.GetNeighbors()
                )
            elif bond.GetBondType() == Chem.BondType.SINGLE and other.GetAtomicNum() == 6:
                single_c += 1
            else:
                raise Reject("unsupported_phosphorus_chemistry")
        if oxo != 1 or (single_o, single_c) not in {(3, 0), (2, 1)}:
            raise Reject("unsupported_phosphorus_chemistry")
        has_pc |= single_c == 1
    return "polyphosphate" if has_pop else "phosphonate" if has_pc else "phosphate"


def acidic_sites(mol):
    result = []
    for o in mol.GetAtoms():
        if (
            o.GetAtomicNum() == 8
            and o.GetDegree() == 1
            and (o.GetFormalCharge() == -1 or o.GetTotalNumHs() == 1)
        ):
            bond = o.GetBonds()[0]
            p = bond.GetOtherAtom(o)
            if p.GetAtomicNum() == 15 and bond.GetBondType() == Chem.BondType.SINGLE:
                result.append(o.GetIdx())
    return result


def normalize_fragment(mol):
    # Uncharger preserves permanent N+ and neutralizes reversible NH3+/acid pairs.
    mol = UNCHARGER.uncharge(rdMolStandardize.Cleanup(mol))
    Chem.SanitizeMol(mol)
    Chem.RemoveStereochemistry(mol)
    # Erase input-dependent proton placement on phosphoric/phosphonic,
    # carboxylic and sulfonic acids, then enumerate balancing proton removals.
    # Keeping an arbitrary source O- would conceal inequivalent zwitterions.
    sites = []
    for o in mol.GetAtoms():
        if (
            o.GetAtomicNum() != 8
            or o.GetDegree() != 1
            or not (o.GetFormalCharge() == -1 or o.GetTotalNumHs() == 1)
        ):
            continue
        bond = o.GetBonds()[0]
        center = bond.GetOtherAtom(o)
        oxo = sum(
            b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(center).GetAtomicNum() == 8
            for b in center.GetBonds()
        )
        if bond.GetBondType() == Chem.BondType.SINGLE and (
            (center.GetAtomicNum() in (6, 15) and oxo == 1)
            or (center.GetAtomicNum() == 16 and oxo >= 2)
        ):
            sites.append(o.GetIdx())
            o.SetFormalCharge(0)
            o.SetNumExplicitHs(1)
            o.SetNoImplicit(True)
    Chem.SanitizeMol(mol)
    charge = Chem.GetFormalCharge(mol)
    variants = {canonical(mol)}
    if charge > 0:
        # Never change permanent N+/S+. Multiple inequivalent assignments -> review.
        if charge <= len(sites) and len(sites) <= 9:
            variants = set()
            for combination in itertools.combinations(sites, charge):
                candidate = Chem.Mol(mol)
                for i in combination:
                    atom = candidate.GetAtomWithIdx(i)
                    atom.SetFormalCharge(-1)
                    atom.SetNumExplicitHs(0)
                    atom.SetNoImplicit(True)
                Chem.SanitizeMol(candidate)
                variants.add(canonical(candidate))
    key = min(variants)
    standardized = Chem.MolFromSmiles(key)
    if standardized is None:
        raise Reject("fragment_sanitize_failure", "Normalized fragment SMILES cannot round-trip")
    ambiguous = len(variants) != 1 or Chem.GetFormalCharge(standardized) != 0
    return key, ambiguous, sorted(variants)


def max_aliphatic_run(mol):
    nodes = {
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and not a.GetIsAromatic() and not a.IsInRing()
    }
    adjacency = {
        i: [a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetIdx() in nodes]
        for i in nodes
    }

    # Removing all ring atoms leaves a forest; its longest simple path is a diameter.
    def farthest(start):
        distances, stack = {start: 1}, [start]
        for i in stack:
            for j in adjacency[i]:
                if j not in distances:
                    distances[j] = distances[i] + 1
                    stack.append(j)
        end = max(distances, key=distances.get)
        return end, distances[end], set(distances)

    longest = 0
    remaining = set(nodes)
    while remaining:
        end, _, visited = farthest(next(iter(remaining)))
        _, distance, _ = farthest(end)
        longest = max(longest, distance)
        remaining -= visited
    return longest


def fragment_descriptors(mol):
    counts = Counter(a.GetSymbol() for a in mol.GetAtoms())
    return {
        # Dummy mass is zero. Formula describes the open-valence fragment, excluding *.
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol).replace("*", ""),
        "molecular_weight": Descriptors.MolWt(mol),
        "carbon_count": counts["C"],
        "phosphorus_count": counts["P"],
        "nitrogen_count": counts["N"],
        "sulfur_count": counts["S"],
        "heavy_atoms": sum(a.GetAtomicNum() > 1 for a in mol.GetAtoms()),
        "formal_charge": Chem.GetFormalCharge(mol),
        "max_aliphatic_run": max_aliphatic_run(mol),
        "ring_containing": mol.GetRingInfo().NumRings() > 0,
    }


def process_smiles(smiles, limits=None, selected=None):
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    mol, counterions = selected if selected is not None else select_component(smiles)
    source_component = Chem.MolToSmiles(mol, isomericSmiles=True)
    family = phosphorus_family(mol)
    sites = acidic_sites(mol)
    if not sites:
        raise Reject("no_acidic_attachment")
    try:
        parent = UNCHARGER.uncharge(rdMolStandardize.Cleanup(mol))
        Chem.SanitizeMol(parent)
        parent_desc = fragment_descriptors(parent)
    except (ValueError, RuntimeError) as exc:
        raise Reject("standardization_sanitize_failure", str(exc)) from exc
    # Each source OH/O- atom is replaced in place. No P-O-C or P-C bond is cut.
    candidates = []
    for site in sites:
        candidate = Chem.RWMol(mol)
        dummy = Chem.Atom(0)
        dummy.SetAtomMapNum(1)
        dummy.SetNoImplicit(True)
        candidate.ReplaceAtom(site, dummy)
        raw = candidate.GetMol()
        try:
            Chem.SanitizeMol(raw)
            key, charge_ambiguous, alternatives = normalize_fragment(raw)
            desc = fragment_descriptors(Chem.MolFromSmiles(key))
        except (ValueError, RuntimeError) as exc:
            raise Reject("fragment_sanitize_failure", str(exc)) from exc
        flags = [
            name
            for name, value in (
                ("molecular_weight", parent_desc["molecular_weight"]),
                ("carbon_count", parent_desc["carbon_count"]),
                ("heavy_atoms", parent_desc["heavy_atoms"]),
                ("phosphorus_count", parent_desc["phosphorus_count"]),
                ("long_aliphatic_carbon_run", parent_desc["max_aliphatic_run"]),
            )
            if value > limits["max_" + name]
        ]
        candidates.append(
            {
                "polar_family": family,
                "polar_fragment_smiles": key,
                "standardized_fragment_smiles": key,
                "source_charge_form": Chem.MolToSmiles(raw, isomericSmiles=True),
                "source_attachment_atom_index": site,
                "charge_ambiguous": charge_ambiguous,
                "charge_alternatives": alternatives,
                "filter_flags": flags,
                **desc,
            }
        )
    distinct = len({c["polar_fragment_smiles"] for c in candidates})
    for c in candidates:
        c["attachment_site_count_before_dedup"] = len(sites)
        c["distinct_attachment_fragments"] = distinct
        c["status"] = (
            "CHARGE_AMBIGUOUS"
            if c["charge_ambiguous"]
            else "FILTER_REVIEW"
            if c["filter_flags"]
            else "MULTIPLE_ATTACHMENT_AMBIGUITY"
            if distinct > 1
            else "APPROVABLE"
        )
    return {
        "selected_p_component": source_component,
        "removed_counterions": counterions,
        "source_component_charge": Chem.GetFormalCharge(mol),
        "attachment_site_count": len(sites),
        "distinct_attachment_fragments": distinct,
        "candidates": candidates,
    }


def validate_controls(config, controls):
    lookup = {c["control_id"]: c for c in controls}
    results, known_keys = {}, {}
    for control_id, row in lookup.items():
        try:
            processed = process_smiles(row["smiles"], config.get("limits"))
            keys = sorted({c["polar_fragment_smiles"] for c in processed["candidates"]})
            passed = row["kind"] == "positive" and all(
                not c["charge_ambiguous"] for c in processed["candidates"]
            )
            known_keys[control_id] = keys
            results[control_id] = {"passed": passed, "keys": keys}
        except Reject as exc:
            results[control_id] = {"passed": row["kind"] == "negative", "reason": exc.reason}
    for query in config["queries"]:
        pattern = Chem.MolFromSmarts(query["smarts"])
        if pattern is None:
            raise ValueError(f"Invalid query: {query['query_id']}")
        for control_id in query["positive_controls"]:
            if not Chem.MolFromSmiles(lookup[control_id]["smiles"]).HasSubstructMatch(pattern):
                raise ValueError(f"Query misses control {query['query_id']}/{control_id}")
        # METAL is a whole-record chemical rejection: its phosphate can match the query.
        for control_id in set(query["negative_controls"]) - {"METAL"}:
            if Chem.MolFromSmiles(lookup[control_id]["smiles"]).HasSubstructMatch(pattern):
                raise ValueError(f"Query matches negative {query['query_id']}/{control_id}")
    if not all(r["passed"] for r in results.values()):
        raise ValueError(f"Controls failed: {results}")
    return results, known_keys
