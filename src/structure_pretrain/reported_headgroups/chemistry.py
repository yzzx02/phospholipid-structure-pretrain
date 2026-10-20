"""Conservative O-P boundary extraction with existing RDKit/Builder validation."""

import hashlib
from collections import Counter
from functools import lru_cache

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from structure_pretrain.chain_extraction.chemistry import _chain
from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.canonicalize import nonisomeric_smiles
from structure_pretrain.library_builder.schemas import ChainCombination, ChainRule, PolarRule
from structure_pretrain.library_builder.validation import validate_polar_fragment

BACKBONE = Chem.MolFromSmarts("[CH2X4]([OX2])-[CHX4]([OX2])-[CH2X4]-[OX2]-[P]")
SPHINGOID = Chem.MolFromSmarts("[P]-[OX2]-[CH2]-[CHX4](-[N])-[CHX4](-[OX2])-[#6]")
UNCHARGER = rdMolStandardize.Uncharger(canonicalOrder=True)


def stable_id(smiles, prefix="HG"):
    return prefix + "-" + hashlib.sha256(smiles.encode()).hexdigest()[:20].upper()


def normalized_molecule(smiles):
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid_source_SMILES")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("multiple_components_require_review")
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        raise ValueError("isotope_label_requires_explicit_review")
    # Protonation normalization only. Never change heavy-atom bonds or discard fragments.
    before = sorted((a.GetAtomicNum(), a.GetIsotope()) for a in mol.GetAtoms())
    mol = UNCHARGER.uncharge(mol)
    Chem.RemoveStereochemistry(mol)
    Chem.SanitizeMol(mol)
    assert before == sorted((a.GetAtomicNum(), a.GetIsotope()) for a in mol.GetAtoms())
    return mol


def add_attachment(mol, root, remove=()):
    rw = Chem.RWMol(mol)
    dummy = Chem.Atom(0)
    dummy.SetAtomMapNum(1)
    d = rw.AddAtom(dummy)
    rw.AddBond(root, d, Chem.BondType.SINGLE)
    for i in sorted(remove, reverse=True):
        rw.RemoveAtom(i)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return nonisomeric_smiles(out)


def builder_trial(polar, chains=None):
    """Test unchanged Builder; sourced chains additionally require exact graph round-trip."""
    validate_polar_fragment(polar, 1)
    if chains is None:
        chains = [
            {
                "chain_token": "A-16:0",
                "linkage_type": "A",
                "carbon_count": 16,
                "double_bond_count": 0,
                "fragment_smiles": "[*:1]C(=O)" + "C" * 15,
                "db_positions": [],
            },
            {
                "chain_token": "A-18:0",
                "linkage_type": "A",
                "carbon_count": 18,
                "double_bond_count": 0,
                "fragment_smiles": "[*:1]C(=O)" + "C" * 17,
                "db_positions": [],
            },
        ]
    rules = {}
    for c in chains:
        token = c["chain_token"]
        if token in rules and rules[token].representative_fragment_smiles != c["fragment_smiles"]:
            raise ValueError("same_chain_token_different_positional_isomers")
        rules[token] = ChainRule(
            token,
            c["linkage_type"],
            c["carbon_count"],
            c["double_bond_count"],
            c["fragment_smiles"],
            tuple(c["db_positions"]),
            "source_graph",
        )
    tokens = tuple(c["chain_token"] for c in chains)
    combo = ChainCombination("reported_roundtrip", tokens, "source", 0, 0, len(tokens) == 1)
    rule = PolarRule("reported", "source", "", "", polar, 1, "known_other", True, True, False)
    with rdBase.BlockLogs():
        result = assemble(rules, combo, rule)
    expected = Counter()
    for text in ["OCC(O)CO", polar, *(c["fragment_smiles"] for c in chains)]:
        cap = Chem.RWMol(Chem.MolFromSmiles(text))
        for atom in list(cap.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                cap.ReplaceAtom(atom.GetIdx(), Chem.Atom(8))
        capped = cap.GetMol()
        Chem.SanitizeMol(capped)
        expected.update(a.GetSymbol() for a in Chem.AddHs(capped).GetAtoms())
    expected.subtract({"H": 2 * (len(chains) + 1), "O": len(chains) + 1})
    actual = Counter(
        a.GetSymbol() for a in Chem.AddHs(Chem.MolFromSmiles(result["canonical_smiles"])).GetAtoms()
    )
    assert +expected == actual, "independent_assembly_element_H_balance_failed"
    result["independent_element_H_balance"] = True
    return result


@lru_cache(maxsize=32768)
def classify_polar(polar):
    mol = Chem.MolFromSmiles(polar)
    if mol is None:
        return {"builder_compatible": False, "incompatibility_reason": "invalid_polar"}
    try:
        built = builder_trial(polar)
        return {
            "builder_compatible": True,
            "incompatibility_reason": "",
            "builder_test_formula": built["neutral_formula"],
            "builder_test_smiles": built["canonical_smiles"],
        }
    except (ValueError, RuntimeError) as exc:
        return {
            "builder_compatible": False,
            "incompatibility_reason": getattr(exc, "reason", str(exc)),
        }


@lru_cache(maxsize=32768)
def extract_full(smiles):
    """Cut only a graph-observed glycerol-O-P bond, retaining its particular P."""
    mol = normalized_molecule(smiles)
    parent = nonisomeric_smiles(mol)
    parent_formula = rdMolDescriptors.CalcMolFormula(mol)
    matches = {}
    for match in mol.GetSubstructMatches(BACKBONE):
        c1, o1, c2, o2, c3, o3, p = match
        if any(
            sum(n.GetAtomicNum() == 6 for n in mol.GetAtomWithIdx(c).GetNeighbors()) != n
            for c, n in ((c1, 1), (c2, 2), (c3, 1))
        ):
            continue
        matches[(c1, c2, c3, p)] = match
    results = {}
    for match in matches.values():
        c1, o1, c2, o2, c3, o3, p = match
        # A chain-bearing scaffold is required; otherwise free glycerol on PG/CL is not an anchor.
        roots = [
            [n.GetIdx() for n in mol.GetAtomWithIdx(o).GetNeighbors() if n.GetIdx() != c]
            for c, o in ((c1, o1), (c2, o2))
        ]
        if not any(roots) or any(len(r) > 1 for r in roots):
            continue
        if any(mol.GetAtomWithIdx(r[0]).GetAtomicNum() not in (0, 6) for r in roots if r):
            continue
        cut = Chem.RWMol(mol)
        cut.RemoveBond(o3, p)
        cut = cut.GetMol()
        parts = Chem.GetMolFrags(cut)
        kept = next(indices for indices in parts if p in indices)
        if any(c in kept for c in (c1, c2, c3)):
            continue  # cyclic/cross-linked phospholipid: no single cut boundary
        polar = add_attachment(cut, p, set(range(mol.GetNumAtoms())) - set(kept))
        if sum(a.GetAtomicNum() == 0 for a in Chem.MolFromSmiles(polar).GetAtoms()) != 1:
            continue  # Remaining native template sockets do not specify an actual graph.
        info = {
            "polar_smiles": polar,
            "original_p_atom_index": p,
            "original_glycerol_O_atom_index": o3,
            "boundary_method": "observed_glycerol_O_P_cut",
            "normalized_parent_smiles": parent,
            "normalized_parent_formula": parent_formula,
            "roundtrip_graph": False,
            "roundtrip_formula": False,
            **classify_polar(polar),
        }
        chains, reasons = [], []
        for o, rs in zip((o1, o2), roots):
            if rs:
                chain, reason, _ = _chain(mol, o, rs[0], set(match))
                if chain:
                    chains.append(chain)
                else:
                    reasons.append(reason)
        if not reasons and len(chains) in (1, 2) and info["builder_compatible"]:
            try:
                result = builder_trial(polar, chains)
                info["roundtrip_graph"] = result["canonical_smiles"] == parent
                info["roundtrip_formula"] = result["neutral_formula"] == parent_formula
                if not info["roundtrip_graph"] or not info["roundtrip_formula"]:
                    info["roundtrip_reason"] = (
                        "source_positional_assignment_or_backbone_not_reproduced"
                    )
            except ValueError as exc:
                info["roundtrip_reason"] = str(exc)
        elif reasons:
            info["roundtrip_reason"] = ";".join(sorted(set(reasons)))
        else:
            info["roundtrip_reason"] = info["incompatibility_reason"]
        # Two valid equivalent anchor sites are merged only when their rooted graph is equal.
        if polar not in results or info["roundtrip_graph"]:
            results[polar] = info
    for match in mol.GetSubstructMatches(SPHINGOID):
        p, o, carbon = match[:3]
        cut = Chem.RWMol(mol)
        cut.RemoveBond(o, p)
        cut = cut.GetMol()
        kept = next(indices for indices in Chem.GetMolFrags(cut) if p in indices)
        if carbon in kept:
            continue
        backbone_part = next(indices for indices in Chem.GetMolFrags(cut) if carbon in indices)
        if sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in backbone_part) < 12:
            continue  # A short amino-acid head is not a sphingoid lipid backbone.
        polar = add_attachment(cut, p, set(range(mol.GetNumAtoms())) - set(kept))
        if sum(a.GetAtomicNum() == 0 for a in Chem.MolFromSmiles(polar).GetAtoms()) != 1:
            continue
        results.setdefault(
            polar,
            {
                "polar_smiles": polar,
                "original_p_atom_index": p,
                "original_glycerol_O_atom_index": o,
                "boundary_method": "observed_sphingoid_O_P_cut",
                "normalized_parent_smiles": parent,
                "normalized_parent_formula": parent_formula,
                "roundtrip_graph": False,
                "roundtrip_formula": False,
                "roundtrip_reason": "non_glycerol_source_backbone",
                **classify_polar(polar),
            },
        )
    return list(results.values())


def lsg_fragment(smiles):
    # Classes.py passes sn3.smiles directly before "OCC(...)CO..." in Glycerolipid.
    # Appending a dummy uses precisely that open end, even when several P atoms occur.
    mol = normalized_molecule(smiles + "[*:1]")
    polar = nonisomeric_smiles(mol)
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1 or dummies[0].GetNeighbors()[0].GetAtomicNum() != 15:
        raise ValueError("LSG_open_end_is_not_P")
    return {
        "polar_smiles": polar,
        "boundary_method": "LSG_documented_sn3_open_end",
        "roundtrip_graph": False,
        "roundtrip_formula": False,
        **classify_polar(polar),
    }
