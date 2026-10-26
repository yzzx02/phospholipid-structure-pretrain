"""Fail-closed generated-core parsing; every invalid candidate remains a failure."""

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from structure_pretrain.glycerol_core_collection.chemistry import capped_core
from structure_pretrain.glycerol_substructure_collection.chemistry import (
    acyclic_long_chain,
    counts,
    representations,
    role_frame,
    verify_assembly,
)
from structure_pretrain.headgroup_target_expansion.chemistry import (
    extra_glycerol_lipid_anchor,
    fatty_N_acyl_PE_PS_head,
)
from structure_pretrain.reported_headgroups.template_selection import MORGAN, functional_mask
from structure_pretrain.utils.formula import ELEMENTS


def equivalence_key(cap):
    """Group role/protonation/stereo equivalents for splitting, not label rewriting."""
    mol = Chem.MolFromSmiles(cap) if isinstance(cap, str) else Chem.Mol(cap)
    if mol is None:
        raise ValueError("Invalid capped core")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def head_fingerprint(mol):
    """Exclude invariant glycerol/chain sockets from similarity and bank selection."""
    match, _, _, outside = role_frame(mol)
    head_atoms = set(range(mol.GetNumAtoms())) - set(match)
    head_atoms -= {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0}
    if outside not in head_atoms:
        raise ValueError("Missing head atoms")
    return MORGAN.GetFingerprint(mol, fromAtoms=sorted(head_atoms))


def validate_candidate(smiles, expected_formula=None, *, require_root=True):
    """Catch RDKit parse, kekulization, cap, frame and assembly exceptions alike."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"valid": False, "reason": "RDKit_parse_failed"}
        Chem.SanitizeMol(mol)
        if len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError("disconnected_core")
        if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()):
            Chem.RemoveStereochemistry(mol)
        if any(a.GetAtomicNum() and a.GetSymbol() not in ELEMENTS for a in mol.GetAtoms()):
            raise ValueError("element_outside_ten_element_vocabulary")
        match, _, head_c, outside = role_frame(mol)
        if head_c == match[2]:
            raise ValueError("sn2_head_attachment")
        ports = {
            a.GetAtomMapNum(): a.GetNeighbors()[0].GetIdx()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 0
        }
        if ports[102] != match[3] or ports[101] not in (match[1], match[5]):
            raise ValueError("chain_port_terminal_central_roles_swapped")
        root = outside if mol.GetAtomWithIdx(outside).GetAtomicNum() == 15 else head_c
        if require_root and root != 0:
            raise ValueError("wrong_attachment_serialization_root")
        cap = capped_core(mol)
        if Chem.GetFormalCharge(cap) != 0:
            raise ValueError("nonneutral_core")
        if (
            acyclic_long_chain(cap)
            or extra_glycerol_lipid_anchor(mol, match[2])
            or fatty_N_acyl_PE_PS_head(mol)
        ):
            raise ValueError("extra_head_fatty_chain_or_lipid_anchor")
        proof = verify_assembly(mol)
        formula = rdMolDescriptors.CalcMolFormula(cap)
        rep = representations(mol)
        return {
            "valid": True,
            "formula_match": expected_formula is None or formula == expected_formula,
            "core_formula": formula,
            "anchored_graph_key": rep["anchored_graph_key"],
            "equivalence_key": equivalence_key(cap),
            "assembly_H_formula_verified": proof["full_assembly_H_formula_pass"],
            "fingerprint": head_fingerprint(mol),
            "functional_mask": functional_mask(cap),
            "counts": [counts(cap)[e] for e in ELEMENTS],
        }
    except Exception as exc:
        # Generated strings are untrusted chemical data. Invalid candidates must
        # never abort a cohort or disappear from the raw/target denominator.
        return {"valid": False, "reason": type(exc).__name__ + ":" + str(exc)[:240]}


def similarity(a, b):
    return float(DataStructs.TanimotoSimilarity(a, b))
