"""Read-only descriptors for the model's rooted, open-boundary Polar graphs."""

from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from structure_pretrain.data.smiles_tokenizer import TOKEN_PATTERN
from structure_pretrain.reported_headgroups.single_anchor_review import analyze_single_anchor

ELEMENTS = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I")
NORMAL_ELEMENTS = {"C", "H", "N", "O", "P", "S"}
PROFILES = {
    "compact_300": {"mw": 300, "C": 12, "heavy": 25},
    "compact_350": {"mw": 350, "C": 15, "heavy": 30},
    "middle_400": {"mw": 400, "C": 18, "heavy": 35},
    "balanced_450": {"mw": 450, "C": 20, "heavy": 40},
    "broad_500": {"mw": 500, "C": 25, "heavy": 45},
    "extended_600": {"mw": 600, "C": 30, "heavy": 50},
}


def describe_polar(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Unparseable Polar")
    markers = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if (
        len(markers) != 1
        or markers[0].GetAtomMapNum() != 1
        or markers[0].GetDegree() != 1
        or markers[0].GetNeighbors()[0].GetAtomicNum() != 15
        or len(Chem.GetMolFrags(mol)) != 1
    ):
        raise ValueError("Expected one connected P-[*:1] Polar")
    atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum()]
    counts = Counter(a.GetSymbol() for a in atoms)
    # GetTotalNumHs accounts for implicit and bracket H, while explicit H atoms
    # are counted above. No SMILES character-based hydrogen arithmetic.
    counts["H"] += sum(a.GetTotalNumHs() for a in atoms)
    mw = Descriptors.MolWt(mol)
    isotope_count = sum(a.GetIsotope() != 0 for a in atoms)
    table = Chem.GetPeriodicTable()
    formula_mw = sum(table.GetAtomicWeight(e) * n for e, n in counts.items())
    if not isotope_count and abs(mw - formula_mw) > 1e-6:
        raise ValueError("Independent formula/mass balance failed")
    root = markers[0].GetNeighbors()[0].GetIdx()
    distances, queue = {root: 0}, [root]
    for i in queue:
        for atom in mol.GetAtomWithIdx(i).GetNeighbors():
            j = atom.GetIdx()
            if atom.GetAtomicNum() > 1 and j not in distances:
                distances[j] = distances[i] + 1
                queue.append(j)
    anchor = analyze_single_anchor(smiles)
    order = [e for e in ("C", "H") if counts[e]] if counts["C"] else []
    order += sorted(e for e in counts if counts[e] and e not in order)
    formula = "".join(e + (str(counts[e]) if counts[e] != 1 else "") for e in order)
    return {
        "open_polar_formula": formula,
        "average_mass_Da": mw,
        "monoisotopic_mass_Da": rdMolDescriptors.CalcExactMolWt(mol),
        "OH_capped_average_mass_Da": mw + table.GetAtomicWeight("O") + table.GetAtomicWeight("H"),
        **{e: counts[e] for e in ELEMENTS},
        "other_elements": ";".join(sorted(set(counts) - set(ELEMENTS))),
        "element_set": ";".join(sorted(e for e in counts if counts[e])),
        "heavy_atoms": sum(a.GetAtomicNum() > 1 for a in atoms),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "P_to_farthest_atom_bonds": max(distances.values()),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(mol),
        "smiles_tokens": len(TOKEN_PATTERN.findall(smiles)),
        "formal_charge": Chem.GetFormalCharge(mol),
        "charged_atom_count": sum(a.GetFormalCharge() != 0 for a in atoms),
        "radical_electrons": sum(a.GetNumRadicalElectrons() for a in atoms),
        "isotope_atoms": isotope_count,
        "H_C_ratio": counts["H"] / counts["C"] if counts["C"] else None,
        "O_C_ratio": counts["O"] / counts["C"] if counts["C"] else None,
        "extra_anchor": anchor["auto_status"] == "exclude",
        "anchor_reason_codes": ";".join(anchor["reason_codes"]),
        "hydrocarbon_path": anchor["longest_hydrocarbon_path"],
    }


def chemistry_reasons(row, nitrogen_max=6):
    """Scope proposals, not proof that excluded molecules are chemically invalid."""
    reasons = []
    if set(row["element_set"].split(";")) - NORMAL_ELEMENTS:
        reasons.append("outside_CHNOPS")
    if not 1 <= row["P"] <= 4:
        reasons.append("P_outside_1_to_4")
    if row["N"] > nitrogen_max:
        reasons.append(f"N_above_{nitrogen_max}")
    if row["S"] > 2:
        reasons.append("S_above_2")
    if row["radical_electrons"]:
        reasons.append("radical")
    if row["isotope_atoms"]:
        reasons.append("isotope_label")
    if row["extra_anchor"]:
        reasons.append("extra_lipid_anchor")
    return reasons


def profile_reasons(row, limits, nitrogen_max=6):
    reasons = chemistry_reasons(row, nitrogen_max)
    if row["average_mass_Da"] > limits["mw"]:
        reasons.append("mass_above_limit")
    if row["C"] > limits["C"]:
        reasons.append("C_above_limit")
    if row["heavy_atoms"] > limits["heavy"]:
        reasons.append("heavy_atoms_above_limit")
    return reasons
