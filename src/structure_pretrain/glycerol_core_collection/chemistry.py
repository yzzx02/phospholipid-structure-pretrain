"""Observed glycerol/chain cuts, stereo round-trips and future-core policy."""

import hashlib

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

BACKBONE = Chem.MolFromSmarts("[CH2X4](-[OX2])-[CHX4](-[OX2])-[CH2X4]-[OX2]")
UNCHARGER = rdMolStandardize.Uncharger(canonicalOrder=True)
PROTECTING = {
    "Boc": Chem.MolFromSmarts("[NX3]-C(=O)-O-C(C)(C)C"),
    "Cbz": Chem.MolFromSmarts("[NX3]-C(=O)-O-C-c1ccccc1"),
    "benzyl_ether": Chem.MolFromSmarts("[OX2]-[CH2]-c1ccccc1"),
}
ELEMENTS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}


def stable_id(smiles, prefix):
    return prefix + "-" + hashlib.sha256(smiles.encode()).hexdigest()[:20].upper()


def parse(text):
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(text)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("invalid_SMILES")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("multiple_components_pending")
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        raise ValueError("isotope_label_excluded")
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError("radical_excluded")
    return mol


def canonical(mol, stereo=True):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=stereo)


def connectivity_key(text):
    """One template per connectivity graph; source stereo remains in source rows."""
    return canonical(neutral(parse(text)), False)


def core_definition(text):
    """Validate explicit O-chain sockets on a single glycerol, without inventing tails."""
    mol = parse(text)
    sockets = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if not 1 <= len(sockets) <= 2:
        raise ValueError("definition_requires_one_or_two_chain_sockets")
    if any(a.GetAtomMapNum() not in (101, 102) or a.GetDegree() != 1 for a in sockets):
        raise ValueError("unsupported_chain_socket")
    oxygen_indices = {a.GetNeighbors()[0].GetIdx() for a in sockets}
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 8 for i in oxygen_indices):
        raise ValueError("chain_socket_not_glycerol_O")
    matches = [
        m
        for m in mol.GetSubstructMatches(BACKBONE, uniquify=False)
        if oxygen_indices <= {m[1], m[3]}
    ]
    if not matches:
        raise ValueError("sockets_not_on_same_terminal_and_central_glycerol_sites")
    cap = capped_core(mol)
    heads = set()
    for match in matches:
        others = [a for a in mol.GetAtomWithIdx(match[5]).GetNeighbors() if a.GetIdx() != match[4]]
        if len(others) == 1 and others[0].GetAtomicNum() == 15:
            heads.add(others[0].GetIdx())
    display, status = (
        root_oxo_P(mol, next(iter(heads)))
        if len(heads) == 1
        else (canonical(mol), "no_unambiguous_direct_glycerol_O_P")
    )
    return {
        "core_smiles_chain_sites": canonical(mol),
        "core_display_smiles": display,
        "capped_core_smiles": canonical(cap),
        "connectivity_smiles": canonical(cap, False),
        "P_first_status": status,
        "chain_count": len(sockets),
        "acyl_chain_count": 0,
        "parent_roundtrip_stereo": False,
        "parent_roundtrip_formula_including_H": False,
        "full_lipid_observed": False,
    }


def neutral(mol):
    # Protonation only. Heavy atom connectivity and all defined stereo survive.
    result = UNCHARGER.uncharge(Chem.Mol(mol))
    Chem.SanitizeMol(result)
    return result


def root_phosphate(text, attachment_map=1):
    """Display O=P from the P adjacent to the explicit marker, never first P."""
    mol = parse(text)
    marked = [
        a for a in mol.GetAtoms() if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == attachment_map
    ]
    if len(marked) != 1 or marked[0].GetDegree() != 1:
        raise ValueError("unique_explicit_attachment_required")
    root = marked[0].GetNeighbors()[0]
    if root.GetAtomicNum() != 15:
        raise ValueError("attachment_is_not_P")
    return root_oxo_P(mol, root.GetIdx())


def root_oxo_P(mol, p):
    oxygens = [
        b.GetOtherAtomIdx(p)
        for b in mol.GetAtomWithIdx(p).GetBonds()
        if b.GetBondType() == Chem.BondType.DOUBLE
        and b.GetOtherAtom(mol.GetAtomWithIdx(p)).GetAtomicNum() == 8
        and b.GetOtherAtom(mol.GetAtomWithIdx(p)).GetDegree() == 1
    ]
    if len(oxygens) != 1:
        return canonical(mol), "no_unique_source_P_equals_O_graph_preserved"
    text = Chem.MolToSmiles(mol, rootedAtAtom=oxygens[0], canonical=False, isomericSmiles=True)
    assert text.startswith(("O=P", "O=[P")) and canonical(parse(text)) == canonical(mol)
    return text, "O_equals_marked_P_first"


def capped_core(mol):
    result = Chem.RWMol(mol)
    markers = [a.GetIdx() for a in result.GetAtoms() if a.GetAtomicNum() == 0]
    for i in markers:
        atom = result.GetAtomWithIdx(i)
        if atom.GetDegree() != 1 or atom.GetNeighbors()[0].GetAtomicNum() != 8:
            raise ValueError("chain_site_not_single_O_dummy")
        atom.GetNeighbors()[0].SetNoImplicit(False)
    for i in sorted(markers, reverse=True):
        result.RemoveAtom(i)
    mol = result.GetMol()
    Chem.SanitizeMol(mol)
    return neutral(mol)


def chain_like(mol, oxygen, backbone_carbon):
    others = [a for a in mol.GetAtomWithIdx(oxygen).GetNeighbors() if a.GetIdx() != backbone_carbon]
    if not others:
        return None
    if len(others) != 1 or others[0].GetAtomicNum() != 6:
        return False
    atom = others[0]
    # Ester carbonyl establishes an actual source acyl connection, including short acyls.
    if any(
        b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(atom).GetAtomicNum() == 8
        for b in atom.GetBonds()
    ):
        # An alpha-aminoacyl ester (Ala/Lys-PG) is a head substituent, not a
        # second fatty-acyl glycerol anchor. Its amine is explicit in the graph.
        alpha = [a for a in atom.GetNeighbors() if a.GetAtomicNum() == 6]
        if any(any(n.GetAtomicNum() == 7 for n in a.GetNeighbors()) for a in alpha):
            return False
        return atom.GetIdx(), "acyl"
    seen = set()
    stack = [atom.GetIdx()]
    carbons = 0
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        current = mol.GetAtomWithIdx(i)
        if current.GetAtomicNum() != 6 or current.GetIsAromatic():
            continue
        if any(
            a.GetAtomicNum() not in (1, 6) and a.GetIdx() != oxygen for a in current.GetNeighbors()
        ):
            continue
        carbons += 1
        stack.extend(a.GetIdx() for a in current.GetNeighbors() if a.GetIdx() != oxygen)
    return (atom.GetIdx(), "ether_or_vinyl") if carbons >= 6 else False


def component(mol, start, blocked):
    seen = set()
    stack = [start]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        for atom in mol.GetAtomWithIdx(i).GetNeighbors():
            j = atom.GetIdx()
            if tuple(sorted((i, j))) not in blocked:
                stack.append(j)
    return seen


def fragment_with_socket(mol, keep, replaced, slot):
    edit = Chem.RWMol(mol)
    atom = edit.GetAtomWithIdx(replaced)
    atom.SetAtomicNum(0)
    atom.SetAtomMapNum(slot)
    atom.SetFormalCharge(0)
    atom.SetNumExplicitHs(0)
    atom.SetNoImplicit(True)
    atom.SetNumRadicalElectrons(0)
    atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for i in reversed(range(mol.GetNumAtoms())):
        if i not in keep and i != replaced:
            edit.RemoveAtom(i)
    result = edit.GetMol()
    Chem.SanitizeMol(result)
    return result


def reassemble(core, tails):
    mol = Chem.Mol(core)
    for slot, text in tails:
        tail = parse(text)
        left = next(
            a for a in mol.GetAtoms() if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == slot
        )
        right = next(
            a for a in tail.GetAtoms() if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == slot
        )
        offset = mol.GetNumAtoms()
        edit = Chem.RWMol(Chem.CombineMols(mol, tail))
        edit.AddBond(
            left.GetNeighbors()[0].GetIdx(),
            offset + right.GetNeighbors()[0].GetIdx(),
            Chem.BondType.SINGLE,
        )
        for i in sorted((left.GetIdx(), offset + right.GetIdx()), reverse=True):
            edit.RemoveAtom(i)
        mol = edit.GetMol()
        Chem.SanitizeMol(mol)
    return mol


def extract_glycerol_core(text):
    mol = parse(text)
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError("source_is_fragment_or_generic_R_not_full_lipid")
    candidates = {}
    for match in mol.GetSubstructMatches(BACKBONE, uniquify=False):
        c1, o1, c2, o2, c3, o3 = match
        a, b = chain_like(mol, o1, c1), chain_like(mol, o2, c2)
        h = chain_like(mol, o3, c3)
        if a is False or b is False or (a is None and b is None) or h not in (False, None):
            continue
        cuts = [
            (oxygen, chain[0], slot, chain[1])
            for oxygen, chain, slot in ((o1, a, 101), (o2, b, 102))
            if chain
        ]
        blocked = {tuple(sorted((o, carbon))) for o, carbon, _, _ in cuts}
        keep = component(mol, c2, blocked)
        if any(carbon in keep for _, carbon, _, _ in cuts):
            continue
        signature = (c2, tuple(sorted(tuple(sorted((o, carbon))) for o, carbon, _, _ in cuts)))
        candidates.setdefault(signature, (match, cuts, keep))
    if not candidates:
        raise ValueError("no_supported_chain_bearing_glycerol_boundary")
    centers = {value[0][2] for value in candidates.values()}
    if len(centers) > 1:
        raise ValueError("multiple_lipid_glycerol_anchors")
    alternatives = []
    for match, cuts, keep in candidates.values():
        kept = Chem.RWMol(mol)
        dummies = {carbon: slot for _, carbon, slot, _ in cuts}
        for carbon, slot in dummies.items():
            atom = kept.GetAtomWithIdx(carbon)
            atom.SetAtomicNum(0)
            atom.SetAtomMapNum(slot)
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(True)
            atom.SetNumRadicalElectrons(0)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for i in reversed(range(mol.GetNumAtoms())):
            if i not in keep and i not in dummies:
                kept.RemoveAtom(i)
        core = kept.GetMol()
        Chem.SanitizeMol(core)
        tails = []
        for oxygen, carbon, slot, _ in cuts:
            tail_set = component(mol, carbon, {tuple(sorted((o, c))) for o, c, _, _ in cuts})
            tail = fragment_with_socket(mol, tail_set, oxygen, slot)
            tails.append((slot, canonical(tail)))
        back = reassemble(core, tails)
        if canonical(back, False) != canonical(mol, False) or rdMolDescriptors.CalcMolFormula(
            back
        ) != rdMolDescriptors.CalcMolFormula(mol):
            raise ValueError("source_connectivity_or_formula_roundtrip_failed")
        cap = capped_core(core)
        # Find the observed head-adjacent P in the original graph, using the marked
        # glycerol-O-P bond; atom-count offset is recomputed after the cut.
        o3 = match[5]
        neighbours = [a for a in mol.GetAtomWithIdx(o3).GetNeighbors() if a.GetIdx() != match[4]]
        root_p = (
            neighbours[0].GetIdx()
            if len(neighbours) == 1 and neighbours[0].GetAtomicNum() == 15
            else None
        )
        retained = sorted(keep | set(dummies))
        display, status = (
            root_oxo_P(core, retained.index(root_p))
            if root_p is not None
            else (canonical(core), "no_direct_glycerol_O_P")
        )
        alternatives.append(
            {
                "core_smiles_chain_sites": canonical(core),
                "core_display_smiles": display,
                "P_first_status": status,
                "capped_core_smiles": canonical(cap),
                "connectivity_smiles": canonical(cap, False),
                "chain_count": len(cuts),
                "acyl_chain_count": sum(kind == "acyl" for _, _, _, kind in cuts),
                "original_chain_fragments": [
                    {"slot": slot, "smiles": smiles} for slot, smiles in tails
                ],
                "parent_roundtrip_connectivity": True,
                "parent_roundtrip_stereo": canonical(back) == canonical(mol),
                "parent_roundtrip_formula_including_H": True,
                "observed_glycerol_atoms": list(match),
                "observed_attachment_P_index": root_p,
                "source_parent_formula": rdMolDescriptors.CalcMolFormula(mol),
            }
        )
    # Terminally symmetric bare glycerol can produce equivalent orientations.
    unique = {r["core_smiles_chain_sites"]: r for r in alternatives}
    if len(unique) > 1:
        raise ValueError("ambiguous_glycerol_orientation_pending")
    return next(iter(unique.values()))


def sugar_and_inositol(mol):
    sugars = []
    inositols = []
    for ring in mol.GetRingInfo().AtomRings():
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if len(ring) not in (5, 6) or any(a.GetIsAromatic() for a in atoms):
            continue
        oxy = sum(a.GetAtomicNum() == 8 for a in atoms)
        carb = sum(a.GetAtomicNum() == 6 for a in atoms)
        external_O = sum(
            any(n.GetAtomicNum() == 8 and n.GetIdx() not in ring for n in a.GetNeighbors())
            for a in atoms
            if a.GetAtomicNum() == 6
        )
        if oxy == 1 and carb == len(ring) - 1 and external_O >= 2:
            sugars.append(ring)
        if len(ring) == 6 and carb == 6 and external_O == 6:
            inositols.append(ring)
    return len(sugars), len(inositols)


def extra_head_chain(mol, minimum=6):
    # Long carbon-only stretches: polyhydroxylated sugar/inositol carbons and
    # amino-acyl residues do not meet this definition.
    eligible = set()
    for atom in mol.GetAtoms():
        if (
            atom.GetAtomicNum() == 6
            and not atom.GetIsAromatic()
            and all(n.GetAtomicNum() in (1, 6) for n in atom.GetNeighbors())
        ):
            eligible.add(atom.GetIdx())

    def walk(i, seen):
        if len(seen) >= minimum:
            return True
        return any(
            walk(n.GetIdx(), seen | {n.GetIdx()})
            for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetIdx() in eligible and n.GetIdx() not in seen
        )

    return any(walk(i, {i}) for i in eligible)


def descriptors(text):
    mol = neutral(parse(text))
    sugar, inositol = sugar_and_inositol(mol)
    allowed = 600.0 if sugar or inositol else 400.0
    halogens = sum(a.GetAtomicNum() in (9, 17, 35, 53) for a in mol.GetAtoms())
    protecting = [name for name, pattern in PROTECTING.items() if mol.HasSubstructMatch(pattern)]
    reasons = []
    charge = Chem.GetFormalCharge(mol)
    if charge:
        reasons.append("non_neutral_core_charge_pending")
    if any(a.GetAtomicNum() not in ELEMENTS for a in mol.GetAtoms()):
        reasons.append("unsupported_element")
    if Descriptors.MolWt(mol) > allowed + 1e-8:
        reasons.append("core_mass_above_limit")
    if extra_head_chain(mol):
        reasons.append("extra_head_long_alkyl_or_acyl_chain")
    if halogens:
        reasons.append("halogen_hold_for_natural_evidence")
    if protecting:
        reasons.append("synthetic_protecting_group_suspected")
    return {
        "core_formula": rdMolDescriptors.CalcMolFormula(mol),
        "core_formal_charge": charge,
        "core_average_mass_Da": Descriptors.MolWt(mol),
        "core_exact_mass_Da": rdMolDescriptors.CalcExactMolWt(mol),
        "mass_limit_Da": allowed,
        "sugar_ring_count": sugar,
        "inositol_ring_count": inositol,
        "halogen_count": halogens,
        "P_count": sum(a.GetAtomicNum() == 15 for a in mol.GetAtoms()),
        "protecting_group_flags": protecting,
        "structural_filter_reasons": reasons,
        "stereocenter_count": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "unassigned_stereocenters": sum(
            tag == "?" for _, tag in Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        ),
    }


def glycosyl_rule_candidates(text):
    donor = parse(text)
    rings = [
        set(r)
        for r in donor.GetRingInfo().AtomRings()
        if len(r) in (5, 6) and sum(donor.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in r) == 1
    ]
    output = []
    for ring in rings:
        for carbon in ring:
            atom = donor.GetAtomWithIdx(carbon)
            if atom.GetAtomicNum() != 6 or not any(
                a.GetAtomicNum() == 8 and a.GetIdx() in ring for a in atom.GetNeighbors()
            ):
                continue
            for oxygen in atom.GetNeighbors():
                if (
                    oxygen.GetAtomicNum() != 8
                    or oxygen.GetIdx() in ring
                    or oxygen.GetDegree() != 1
                    or oxygen.GetTotalNumHs() != 1
                ):
                    continue
                glycerol = Chem.MolFromSmiles("OCC(O)CO")
                offset = donor.GetNumAtoms()
                edit = Chem.RWMol(Chem.CombineMols(donor, glycerol))
                edit.AddBond(oxygen.GetIdx(), offset + 1, Chem.BondType.SINGLE)
                edit.RemoveAtom(offset)  # source OH oxygen retained as the glycosidic oxygen
                cap = edit.GetMol()
                Chem.SanitizeMol(cap)
                core = Chem.RWMol(cap)
                # Original glycerol atoms1..5 now at offset..offset+4.
                for oxygen_idx, slot in ((offset + 2, 102), (offset + 4, 101)):
                    dummy = Chem.Atom(0)
                    dummy.SetAtomMapNum(slot)
                    i = core.AddAtom(dummy)
                    core.AddBond(oxygen_idx, i, Chem.BondType.SINGLE)
                core = core.GetMol()
                Chem.SanitizeMol(core)
                assert canonical(capped_core(core)) == canonical(neutral(cap))
                output.append(
                    {
                        "core_smiles_chain_sites": canonical(core),
                        "core_display_smiles": canonical(core),
                        "capped_core_smiles": canonical(neutral(cap)),
                        "connectivity_smiles": canonical(neutral(cap), False),
                        "P_first_status": "no_P_in_glycoside",
                        "chain_count": 2,
                        "acyl_chain_count": 0,
                        "rule": "observed_glycosyl_glycerol_linkage; source_anomeric_OH_only; backbone_stereo_unspecified",  # noqa: E501
                        "donor_anomeric_atom_index": carbon,
                        "donor_OH_atom_index": oxygen.GetIdx(),
                        "full_lipid_observed": False,
                        "parent_roundtrip_stereo": False,
                        "parent_roundtrip_formula_including_H": False,
                    }
                )
    return list({r["core_smiles_chain_sites"]: r for r in output}.values())
