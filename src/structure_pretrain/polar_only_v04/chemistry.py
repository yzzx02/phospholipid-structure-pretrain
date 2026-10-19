"""Frozen-builder attached-fragment arithmetic; no cutting of complete lipids."""

import numpy as np
from rdkit import Chem

from structure_pretrain.data.prepare_v03 import atom_counts, canonical_polar
from structure_pretrain.library_builder.assembly import SCAFFOLD
from structure_pretrain.library_builder.canonicalize import neutral_structure, parse_molecule
from structure_pretrain.library_builder.chain_library import chain_sort_key
from structure_pretrain.library_builder.validation import attachment


def derive_polar_formula(full, chain_labels, chain_counts, scaffold_counts):
    full, labels = np.asarray(full), np.asarray(chain_labels)
    if full.ndim < 1 or labels.ndim < 1 or full.shape[-1] != 10 or labels.shape[-1] != 2:
        raise ValueError("Known full formula [...,10] and chain pair [...,2] required")
    if not np.isfinite(full).all() or np.any(full < 0) or np.any(full != np.floor(full)):
        raise ValueError("Trusted integral nonnegative full formula required")
    if (
        not np.issubdtype(labels.dtype, np.integer)
        or labels.shape[:-1] != full.shape[:-1]
        or np.any(labels < 0)
        or np.any(labels >= len(chain_counts))
    ):
        raise ValueError("Unknown chain token")
    count = (labels != 0).sum(-1)
    if np.any((count != 1) & (count != 2)) or np.any(labels[..., 0] == 0):
        raise ValueError("Supply one/two canonical known chains, NULL only in slot two")
    result = full.astype(np.int32) - np.asarray(scaffold_counts)
    result = result - np.asarray(chain_counts)[labels].sum(-2)
    result[..., 1] -= count == 1  # vacant glycerol oxygen gains exactly one H
    if np.any(result < 0) or np.any(result[..., 4] < 1):
        raise ValueError("Known chain candidate implies impossible P-containing polar formula")
    return result


def chainless_molecule(polar_smiles, chain_count):
    """Attach the original polar to scaffold slot3; occupied chain sites retain markers.

    AA/OA/PA deliberately share identical standardized O--* sites. Lyso slot2
    becomes a real OH, not a dummy. Maps101/102 identify occupied scaffold sites.
    """
    if chain_count not in (1, 2):
        raise ValueError("Only mono-phosphatidyl one/two-chain scaffolds supported")
    head = parse_molecule(canonical_polar(polar_smiles))
    scaffold = parse_molecule(SCAFFOLD)
    sites = {a.GetAtomMapNum(): a.GetIdx() for a in scaffold.GetAtoms() if a.GetAtomicNum() == 0}
    offset = scaffold.GetNumAtoms()
    dummy, phosphorus = attachment(head, 1)
    neighbor = scaffold.GetAtomWithIdx(sites[3]).GetNeighbors()[0].GetIdx()
    combined = Chem.RWMol(Chem.CombineMols(scaffold, head))
    combined.AddBond(neighbor, offset + phosphorus, Chem.BondType.SINGLE)
    remove = [sites[3], offset + dummy]
    for slot in (1, 2):
        if slot == 2 and chain_count == 1:
            remove.append(sites[slot])
        else:
            combined.GetAtomWithIdx(sites[slot]).SetAtomMapNum(100 + slot)
    for index in sorted(remove, reverse=True):
        combined.RemoveAtom(index)
    molecule = combined.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule


def chainless_smiles(polar_smiles, chain_count):
    molecule = chainless_molecule(polar_smiles, chain_count)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def attach_known_chains(molecule, chain_tokens, chains):
    """Independent round-trip from the new chainless representation."""
    tokens = sorted(chain_tokens, key=chain_sort_key)
    sites = {
        a.GetAtomMapNum(): (a.GetIdx(), a.GetNeighbors()[0].GetIdx())
        for a in molecule.GetAtoms()
        if a.GetAtomicNum() == 0
    }
    if set(sites) != set(range(101, 101 + len(tokens))):
        raise ValueError("Chain occupancy differs from actual chainless scaffold vacancies")
    complete, remove, links = molecule, [x[0] for x in sites.values()], []
    for slot, token in enumerate(tokens, 101):
        fragment = parse_molecule(chains[token].representative_fragment_smiles)
        dummy, neighbor = attachment(fragment, 1)
        offset = complete.GetNumAtoms()
        links.append((sites[slot][1], offset + neighbor))
        remove.append(offset + dummy)
        complete = Chem.CombineMols(complete, fragment)
    editable = Chem.RWMol(complete)
    for left, right in links:
        editable.AddBond(left, right, Chem.BondType.SINGLE)
    for index in sorted(remove, reverse=True):
        editable.RemoveAtom(index)
    return neutral_structure(editable.GetMol())


def fragment_count_inputs(chains, vocab):
    counts = np.zeros((len(vocab), 10), dtype=np.int32)
    for i, token in enumerate(vocab[1:], 1):
        counts[i] = atom_counts(parse_molecule(chains[token].representative_fragment_smiles))
    return counts, atom_counts(parse_molecule(SCAFFOLD))
