"""Fragment-aware features, with validated dummy sites and explicit GINE marker roles."""

import numpy as np
import torch
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import Data

from structure_pretrain.library_builder.canonicalize import supported_atoms

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
SCAFFOLD_QUERY = Chem.MolFromSmarts("[#0]-[O]-[CH2]-[CH](-[O])-[CH2]-[O]-[P]")


def fragment_molecule(smiles):
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not molecule.GetNumAtoms() or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("One connected chainless fragment required")
    supported_atoms(molecule, allow_dummy=True)
    if sum(a.GetFormalCharge() for a in molecule.GetAtoms()) != 0:
        raise ValueError("Neutral chainless fragment required")
    dummies = [a for a in molecule.GetAtoms() if not a.GetAtomicNum()]
    if {a.GetAtomMapNum() for a in dummies} not in ({101}, {101, 102}):
        raise ValueError("Only occupied scaffold markers101/102 are allowed")
    if len(dummies) not in (1, 2) or len({a.GetAtomMapNum() for a in dummies}) != len(dummies):
        raise ValueError("Duplicate or invalid scaffold markers")
    for atom in dummies:
        if (
            atom.GetDegree() != 1
            or atom.GetNeighbors()[0].GetAtomicNum() != 8
            or atom.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE
        ):
            raise ValueError("A chain marker must have one single bond to scaffold O")
    by_map = {a.GetAtomMapNum(): a.GetIdx() for a in dummies}
    valid_scaffold = False
    for match in molecule.GetSubstructMatches(SCAFFOLD_QUERY):
        if match[0] != by_map[101]:
            continue
        second_oxygen = molecule.GetAtomWithIdx(match[4])
        if 102 in by_map:
            valid_scaffold |= by_map[102] in [a.GetIdx() for a in second_oxygen.GetNeighbors()]
        else:
            valid_scaffold |= second_oxygen.GetTotalNumHs() == 1
    if not valid_scaffold:
        raise ValueError("Markers must occupy frozen glycerol sites with glycerol-O-P boundary")
    return molecule


def packed_fingerprint(molecule):
    bits = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(MORGAN.GetFingerprint(molecule), bits)
    return np.packbits(bits, bitorder="little")


def fragment_features(molecule):
    atoms, roles = [], []
    for atom in molecule.GetAtoms():
        if not -4 <= atom.GetFormalCharge() <= 4:
            raise ValueError("Unsupported atom charge")
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
        roles.append(atom.GetAtomMapNum() - 100 if not atom.GetAtomicNum() else 0)
    kinds = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }
    edges, bonds = [], []
    for bond in molecule.GetBonds():
        if bond.GetBondType() not in kinds:
            raise ValueError("Unsupported bond")
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row = [kinds[bond.GetBondType()], int(bond.GetIsConjugated()), int(bond.IsInRing())]
        edges.extend([[left, right], [right, left]])
        bonds.extend([row, row])
    return {"atoms": atoms, "edges": edges, "bonds": bonds, "site_roles": roles}


def graph_from_record(record):
    return Data(
        x=torch.tensor(record["atoms"], dtype=torch.long),
        edge_index=torch.tensor(record["edges"], dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(record["bonds"], dtype=torch.long),
        site_role=torch.tensor(record["site_roles"], dtype=torch.long),
        num_nodes=len(record["atoms"]),
    )


def fragment_to_graph(smiles):
    return graph_from_record(fragment_features(fragment_molecule(smiles)))
