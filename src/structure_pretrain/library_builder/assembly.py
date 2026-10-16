from rdkit import Chem

from .canonicalize import neutral_structure, parse_molecule
from .chain_library import chain_sort_key
from .schemas import ChainCombination, ChainRule, PolarRule
from .validation import attachment

SCAFFOLD = "[*:1]OCC(O[*:2])CO[*:3]"


def assemble(
    chains: dict[str, ChainRule],
    combo: ChainCombination,
    polar: PolarRule,
    *,
    fragment_parser=parse_molecule,
):
    tokens = tuple(sorted(combo.chain_tokens, key=chain_sort_key))
    if len(tokens) not in (1, 2):
        raise ValueError("v0.1 assembles mono-phosphatidyl with one or two chains")
    molecule = fragment_parser(SCAFFOLD)
    scaffold_dummies = {
        atom.GetAtomMapNum(): (atom.GetIdx(), atom.GetNeighbors()[0].GetIdx())
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 0
    }
    links, remove = [], [dummy for dummy, _ in scaffold_dummies.values()]
    fragments = [
        (index + 1, fragment_parser(chains[token].representative_fragment_smiles))
        for index, token in enumerate(tokens)
    ]
    fragments.append((3, fragment_parser(polar.polar_fragment_smiles)))
    for slot, fragment in fragments:
        offset = molecule.GetNumAtoms()
        dummy, neighbor = attachment(fragment)
        links.append((scaffold_dummies[slot][1], offset + neighbor))
        remove.append(offset + dummy)
        molecule = Chem.CombineMols(molecule, fragment)
    editable = Chem.RWMol(molecule)
    for left, right in links:
        editable.AddBond(left, right, Chem.BondType.SINGLE)
    for index in sorted(remove, reverse=True):
        editable.RemoveAtom(index)
    complete = editable.GetMol()
    smiles, formula, key, counts = neutral_structure(complete)
    return {
        "architecture": "mono_phosphatidyl",
        "polar_id": polar.polar_id,
        "polar_family": polar.polar_family,
        "chain_tokens": list(tokens),
        "linkage_family": combo.linkage_family,
        "representative_smiles": smiles,
        "canonical_smiles": smiles,
        "neutral_formula": formula,
        **{f"neutral_formula_{element}": value for element, value in counts.items()},
        "inchikey": key,
        "is_anchor_combination": False,
        "is_lyso": combo.is_lyso,
        "representative_sn_assignment": True,
        "source_chain_rule": sorted({chains[token].source for token in tokens}),
        "source_polar_ids": list(polar.source_polar_ids),
        "polar_provenance": list(polar.provenance),
        "mapped_from_p_tokens": list(combo.mapped_from_p_tokens),
        "combination_id": combo.combination_id,
        "total_chain_carbon": combo.total_carbon,
        "total_chain_double_bonds": combo.total_double_bonds,
    }
