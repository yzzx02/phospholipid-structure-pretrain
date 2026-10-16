from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from structure_pretrain.pubchem_polar.chemistry import max_aliphatic_run

PATTERNS = {
    "contains_carboxyl": "[C](=[O])-[O;H1,-1]",
    "contains_amino": "[N;X3;!$(N-C=O);!$(N-S(=O)=O)]",
    "contains_quaternary_ammonium": "[N+;X4;H0]",
    "contains_sulfonate_or_sulfate": "[S](=[O])(=[O])-[O;H1,-1]",
    "contains_polyol_like": "[C;!a](-[O;H1])-[C;!a](-[O;H1])",
}
QUERIES = {name: Chem.MolFromSmarts(smarts) for name, smarts in PATTERNS.items()}


def describe(molecule):
    counts = Counter(atom.GetSymbol() for atom in molecule.GetAtoms())
    heavy = sum(atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms())
    hetero = sum(counts[element] for element in ("O", "N", "S", "P"))
    return {
        # Open P-boundary fragment; dummy contributes neither mass nor heavy atoms.
        "molecular_weight": Descriptors.MolWt(molecule),
        "carbon_count": counts["C"],
        "heavy_atom_count": heavy,
        "phosphorus_count": counts["P"],
        "oxygen_count": counts["O"],
        "nitrogen_count": counts["N"],
        "sulfur_count": counts["S"],
        "halogen_count": sum(counts[e] for e in ("F", "Cl", "Br", "I")),
        "hetero_count": hetero,
        "hetero_fraction": hetero / heavy if heavy else 0,
        "max_aliphatic_carbon_run": max_aliphatic_run(molecule),
        "ring_count": rdMolDescriptors.CalcNumRings(molecule),
        "aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(molecule),
        "heterocycle_count": rdMolDescriptors.CalcNumHeterocycles(molecule),
        "formal_charge": Chem.GetFormalCharge(molecule),
        "rotatable_bond_count": rdMolDescriptors.CalcNumRotatableBonds(molecule),
        **{name: molecule.HasSubstructMatch(query) for name, query in QUERIES.items()},
    }
