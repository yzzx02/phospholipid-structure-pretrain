"""Seven atom fields, three bond fields, eight random-walk return probabilities."""

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import Data

ATOM_CARDINALITIES = (119, 9, 9, 9, 2, 2, 9)
BOND_CARDINALITIES = (4, 2, 2)
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)


def full_molecule(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not molecule.GetNumAtoms():
        raise ValueError("Invalid or empty full-structure SMILES")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("Supply one connected neutral molecule without adduct fragments")
    if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
        raise ValueError("Full template structures cannot contain attachment wildcards")
    if sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()) != 0:
        raise ValueError("Supply the neutral structure, not a charged precursor ion")
    return molecule


def structure_key(smiles: str) -> str:
    molecule = full_molecule(smiles)
    Chem.RemoveStereochemistry(molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def fingerprint(smiles: str):
    return MORGAN.GetFingerprint(full_molecule(smiles))


def fingerprint_tensor(smiles: str) -> torch.Tensor:
    array = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint(smiles), array)
    return torch.from_numpy(array)


def smiles_to_graph(smiles: str, rwse_steps: int = 8) -> Data:
    molecule = full_molecule(smiles)
    if not 1 <= rwse_steps or molecule.GetNumAtoms() > 512:
        raise ValueError("This small implementation requires rwse_steps>=1 and <=512 atoms")
    atoms = []
    for atom in molecule.GetAtoms():
        if not -4 <= atom.GetFormalCharge() <= 4:
            raise ValueError("Atom charge exceeds supported categorical range")
        atoms.append(
            [
                atom.GetAtomicNum(),
                min(atom.GetDegree(), 8),
                atom.GetFormalCharge() + 4,
                min(int(atom.GetHybridization()), 8),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
                min(atom.GetTotalNumHs(), 8),
            ]
        )
    types = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }
    edges, features = [], []
    adjacency = np.zeros((len(atoms), len(atoms)), dtype=np.float32)
    for bond in molecule.GetBonds():
        if bond.GetBondType() not in types:
            raise ValueError("Only single/double/triple/aromatic bonds are supported")
        row = [types[bond.GetBondType()], int(bond.GetIsConjugated()), int(bond.IsInRing())]
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for source, destination in ((left, right), (right, left)):
            edges.append((source, destination))
            features.append(row)
            adjacency[source, destination] = 1
    transition = adjacency / np.maximum(adjacency.sum(1, keepdims=True), 1)
    power = np.eye(len(atoms), dtype=np.float32)
    returns = []
    for _ in range(rwse_steps):
        power = power @ transition
        returns.append(power.diagonal().copy())
    return Data(
        x=torch.tensor(atoms, dtype=torch.long),
        edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous(),
        edge_attr=torch.tensor(features, dtype=torch.long).reshape(-1, 3),
        rwse=torch.from_numpy(np.stack(returns, axis=-1)),
        num_nodes=len(atoms),
    )
