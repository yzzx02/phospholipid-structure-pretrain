"""Anchored graph edits and chemical checks for a bounded candidate neighbourhood."""

from collections import Counter

from rdkit import Chem, rdBase

from structure_pretrain.glycerol_core_collection.chemistry import (
    BACKBONE,
    canonical,
    capped_core,
    chain_like,
    component,
    descriptors,
    neutral,
    parse,
    stable_id,
)

ACYL_P = Chem.MolFromSmarts("[P]-[OX2]-[CX3](=[OX1])")
PEROXIDE = Chem.MolFromSmarts("[O]-[O]")
YNOL = Chem.MolFromSmarts("[N,O,S]-[C]#[C]")
HEMIAMINAL = Chem.MolFromSmarts("[N]-[CX4](-[OX2H])")
N_ACYL_HEAD = Chem.MolFromSmarts("[P]-[O]-[#6]-[#6]-[N]-[C](=[O])")


def balanced_parent(mol):
    """Protonation convention only; source graph and original H are retained elsewhere."""
    result = neutral(mol)
    notes = []
    charge = Chem.GetFormalCharge(result)
    fixed_cations = sum(
        a.GetFormalCharge()
        for a in result.GetAtoms()
        if a.GetAtomicNum() == 7 and a.GetDegree() == 4
    )
    if charge > 0 and fixed_cations >= charge:
        acids = []
        for atom in result.GetAtoms():
            if atom.GetAtomicNum() != 8 or atom.GetDegree() != 1 or atom.GetTotalNumHs() != 1:
                continue
            n = atom.GetNeighbors()[0]
            if n.GetAtomicNum() == 15:
                acids.insert(0, atom.GetIdx())
            elif n.GetAtomicNum() == 6 and any(
                b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(n).GetAtomicNum() == 8
                for b in n.GetBonds()
            ):
                acids.append(atom.GetIdx())
        if len(acids) >= charge:
            result = Chem.Mol(result)
            for i in acids[:charge]:
                atom = result.GetAtomWithIdx(i)
                atom.SetFormalCharge(-1)
                atom.SetNumExplicitHs(0)
                atom.SetNoImplicit(True)
            Chem.SanitizeMol(result)
            notes.append("fixed_quaternary_N_balanced_by_acid_deprotonation_only")
    return result, notes


def frame(mol):
    sockets = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if (
        not sockets
        or len(sockets) > 2
        or any(a.GetAtomMapNum() not in (101, 102) or a.GetDegree() != 1 for a in sockets)
    ):
        raise ValueError("explicit_glycerol_chain_sockets_required")
    o_set = {a.GetNeighbors()[0].GetIdx() for a in sockets}
    matches = [
        m
        for m in mol.GetSubstructMatches(BACKBONE, uniquify=False)
        if o_set <= {m[1], m[3]} and mol.GetAtomWithIdx(m[5]).GetDegree() == 2
    ]
    if not matches:
        raise ValueError("no_unique_head_bearing_glycerol_frame")
    match = min(matches)
    c1, o1, c2, o2, c3, o3 = match
    others = [a.GetIdx() for a in mol.GetAtomWithIdx(o3).GetNeighbors() if a.GetIdx() != c3]
    if len(others) != 1:
        raise ValueError("unique_head_attachment_required")
    p = others[0] if mol.GetAtomWithIdx(others[0]).GetAtomicNum() == 15 else None
    return match, p


def standard_core(text):
    mol, notes = balanced_parent(parse(text))
    Chem.RemoveStereochemistry(mol)
    match, p = frame(mol)
    edit = Chem.RWMol(mol)
    for oxygen, marker in ((match[1], 101), (match[3], 102)):
        dummies = [a for a in edit.GetAtomWithIdx(oxygen).GetNeighbors() if a.GetAtomicNum() == 0]
        if dummies:
            dummies[0].SetAtomMapNum(marker)
        else:
            atom = Chem.Atom(0)
            atom.SetAtomMapNum(marker)
            j = edit.AddAtom(atom)
            edit.AddBond(oxygen, j, Chem.BondType.SINGLE)
    mol = edit.GetMol()
    Chem.SanitizeMol(mol)
    for a in mol.GetAtoms():
        if a.GetAtomicNum():
            a.SetAtomMapNum(0)
    return mol, notes


def representations(mol):
    match, p = frame(mol)
    key = canonical(mol, False)
    gly = Chem.MolToSmiles(mol, rootedAtAtom=match[4], canonical=True, isomericSmiles=False)
    first = (
        Chem.MolToSmiles(mol, rootedAtAtom=p, canonical=True, isomericSmiles=False)
        if p is not None
        else gly
    )
    assert canonical(parse(first), False) == key == canonical(parse(gly), False)
    polar = ""
    if p is not None:
        o3 = match[5]
        keep = component(mol, p, {tuple(sorted((o3, p)))}) | {o3}
        edit = Chem.RWMol(mol)
        atom = edit.GetAtomWithIdx(o3)
        atom.SetAtomicNum(0)
        atom.SetAtomMapNum(1)
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        for i in reversed(range(mol.GetNumAtoms())):
            if i not in keep:
                edit.RemoveAtom(i)
        head = edit.GetMol()
        Chem.SanitizeMol(head)
        marker = next(a for a in head.GetAtoms() if a.GetAtomicNum() == 0)
        pp = marker.GetNeighbors()[0].GetIdx()
        polar = Chem.MolToSmiles(head, rootedAtAtom=pp, canonical=True, isomericSmiles=False)
    return {
        "target_id": stable_id(key, "HT"),
        "anchored_graph_key": key,
        "target_smiles": first,
        "glycerol_first_smiles": gly,
        "polar_P_first_smiles": polar,
        "capped_core_smiles": canonical(capped_core(mol), False),
        "rooted_at_attachment_P": p is not None,
    }


def policy(mol, *, generated=False):
    capped = capped_core(mol)
    d = descriptors(canonical(capped, False))
    limit = 600.0 if d["sugar_ring_count"] or d["inositol_ring_count"] else 500.0
    reasons = [x for x in d["structural_filter_reasons"] if x != "core_mass_above_limit"]
    if d["core_average_mass_Da"] > limit + 1e-8:
        reasons.append("core_mass_above_500_600_policy")
    if capped.HasSubstructMatch(ACYL_P):
        reasons.append("reactive_acyl_phosphate_mixed_anhydride")
    if capped.HasSubstructMatch(PEROXIDE):
        reasons.append("peroxide_requires_review")
    current_frame, _ = frame(mol)
    if extra_glycerol_lipid_anchor(mol, current_frame[2]):
        reasons.append("extra_glycerol_lipid_anchor_including_short_acyl")
    if fatty_N_acyl_PE_PS_head(mol):
        reasons.append("extra_fatty_N_acyl_PE_PS_head")
    if generated and capped.HasSubstructMatch(YNOL):
        reasons.append("ynol_or_ynamine_requires_review")
    if generated and capped.HasSubstructMatch(HEMIAMINAL):
        reasons.append("hemiaminal_requires_review")
    if d["P_count"] > 4:
        reasons.append("more_than_four_P_outside_bounded_scope")
    d.update(
        mass_limit_Da=limit,
        filter_reasons=sorted(set(reasons)),
        structural_filter_reasons=sorted(set(reasons)),
    )
    return d


def extra_glycerol_lipid_anchor(mol, fixed_center=None):
    """Detect BMP/CL/Acyl-PG even for formyl/acetyl chains; alpha-aminoacyl is exempt."""
    for match in mol.GetSubstructMatches(BACKBONE, uniquify=False):
        if match[2] == fixed_center:
            continue
        if any(
            chain_like(mol, oxygen, carbon)
            for carbon, oxygen in ((match[0], match[1]), (match[2], match[3]), (match[4], match[5]))
        ):
            return True
    return False


def fatty_N_acyl_PE_PS_head(mol):
    """Reject an added carbon acyl residue on PE/PS, exempt alpha-aminoacyl peptides."""
    for match in mol.GetSubstructMatches(N_ACYL_HEAD):
        carbonyl = mol.GetAtomWithIdx(match[5])
        alpha = [n for n in carbonyl.GetNeighbors() if n.GetAtomicNum() == 6]
        if alpha and not any(any(a.GetAtomicNum() == 7 for a in n.GetNeighbors()) for n in alpha):
            return True
    return False


def protected(mol):
    match, p = frame(mol)
    fixed = set(match) | {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0}
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 15:
            fixed.add(a.GetIdx())
            fixed.update(
                b.GetOtherAtomIdx(a.GetIdx())
                for b in a.GetBonds()
                if b.GetBondType() == Chem.BondType.DOUBLE
            )
    return fixed


def finish(edit):
    mol = edit.GetMol()
    with rdBase.BlockLogs():
        Chem.SanitizeMol(mol)
    mol, _ = balanced_parent(mol)
    Chem.RemoveStereochemistry(mol)
    return mol


def add_substituent(mol, site, fragment):
    frag = parse(fragment)
    offset = mol.GetNumAtoms()
    edit = Chem.RWMol(Chem.CombineMols(mol, frag))
    atom = edit.GetAtomWithIdx(site)
    if atom.GetNumExplicitHs():
        atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
    atom.SetNoImplicit(False)
    edit.AddBond(site, offset, Chem.BondType.SINGLE)
    return finish(edit)


def local_edits(mol):
    """One typed edit; all source glycerol atoms and the connecting P are protected."""
    fixed = protected(mol)
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if i in fixed or not a.GetTotalNumHs():
            continue
        z = a.GetAtomicNum()
        if z == 8 and a.GetDegree() == 1 and a.GetFormalCharge() == 0:
            for label, frag in (
                ("O_methyl", "C"),
                ("O_ethyl", "CC"),
                ("O_allyl", "CC=C"),
                ("O_propargyl", "CC#C"),
            ):
                yield label, [i], lambda i=i, f=frag: add_substituent(mol, i, f)
            n = a.GetNeighbors()[0]
            if n.GetAtomicNum() == 6 and n.GetHybridization() == Chem.HybridizationType.SP3:
                for label, frag in (
                    ("O_acetyl_short", "C(=O)C"),
                    ("O_propionyl_short", "C(=O)CC"),
                    ("O_phosphorylation", "P(=O)(O)O"),
                    ("O_sulfation", "S(=O)(=O)O"),
                ):
                    yield label, [i], lambda i=i, f=frag: add_substituent(mol, i, f)
                if (
                    n.GetTotalNumHs()
                    and sum(x.GetAtomicNum() in (7, 8, 16) for x in n.GetNeighbors()) == 1
                ):

                    def oxidize(i=i, j=n.GetIdx()):
                        e = Chem.RWMol(mol)
                        e.GetBondBetweenAtoms(i, j).SetBondType(Chem.BondType.DOUBLE)
                        for k in (i, j):
                            e.GetAtomWithIdx(k).SetNumExplicitHs(0)
                            e.GetAtomWithIdx(k).SetNoImplicit(False)
                        return finish(e)

                    yield "alcohol_to_carbonyl", [i, n.GetIdx()], oxidize
        elif (
            z == 7
            and not a.GetIsAromatic()
            and a.GetFormalCharge() == 0
            and not any(
                n.GetAtomicNum() == 6
                and any(
                    b.GetBondType() == Chem.BondType.DOUBLE
                    and b.GetOtherAtom(n).GetAtomicNum() in (8, 16)
                    for b in n.GetBonds()
                )
                for n in a.GetNeighbors()
            )
        ):
            for label, frag in (
                ("N_methyl", "C"),
                ("N_ethyl", "CC"),
                ("N_allyl", "CC=C"),
                ("N_propargyl", "CC#C"),
            ):
                yield label, [i], lambda i=i, f=frag: add_substituent(mol, i, f)
        elif (
            z == 6 and not a.GetIsAromatic() and a.GetHybridization() == Chem.HybridizationType.SP3
        ):
            for label, frag in (
                ("C_methyl_branch", "C"),
                ("C_hydroxylation", "O"),
                ("C_carboxylation", "C(=O)O"),
            ):
                yield label, [i], lambda i=i, f=frag: add_substituent(mol, i, f)
    for a in mol.GetAtoms():
        if (
            a.GetIdx() in fixed
            or a.GetAtomicNum() != 7
            or a.GetIsAromatic()
            or a.GetFormalCharge()
            or a.GetDegree() != 3
        ):
            continue

        def quaternize(i=a.GetIdx()):
            e = Chem.RWMol(mol)
            e.GetAtomWithIdx(i).SetFormalCharge(1)
            return add_substituent(e.GetMol(), i, "C")

        yield "N_quaternization_methyl", [a.GetIdx()], quaternize
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.IsInRing() or a.GetIdx() in fixed or b.GetIdx() in fixed:
            continue
        if a.GetAtomicNum() != 6 or b.GetAtomicNum() != 6 or a.GetIsAromatic() or b.GetIsAromatic():
            continue
        if bond.GetBondType() == Chem.BondType.SINGLE:
            for order, need, label in (
                (Chem.BondType.DOUBLE, 1, "C_C_to_double"),
                (Chem.BondType.TRIPLE, 2, "C_C_to_triple"),
            ):
                if a.GetTotalNumHs() < need or b.GetTotalNumHs() < need:
                    continue

                def unsaturate(i=a.GetIdx(), j=b.GetIdx(), order=order):
                    e = Chem.RWMol(mol)
                    e.GetBondBetweenAtoms(i, j).SetBondType(order)
                    for k in (i, j):
                        e.GetAtomWithIdx(k).SetNumExplicitHs(0)
                        e.GetAtomWithIdx(k).SetNoImplicit(False)
                    return finish(e)

                yield label, [a.GetIdx(), b.GetIdx()], unsaturate


def amino_acid_from_pg_polar(text):
    mol = parse(text)
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() != 6:
            continue
        if not any(
            b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(carbon).GetAtomicNum() == 8
            for b in carbon.GetBonds()
        ):
            continue
        alpha = [
            n
            for n in carbon.GetNeighbors()
            if n.GetAtomicNum() == 6 and any(x.GetAtomicNum() == 7 for x in n.GetNeighbors())
        ]
        esters = [
            n
            for n in carbon.GetNeighbors()
            if n.GetAtomicNum() == 8
            and n.GetDegree() == 2
            and mol.GetBondBetweenAtoms(carbon.GetIdx(), n.GetIdx()).GetBondType()
            == Chem.BondType.SINGLE
        ]
        if not alpha or not esters:
            continue
        oxygen = esters[0]
        keep = component(mol, carbon.GetIdx(), {tuple(sorted((carbon.GetIdx(), oxygen.GetIdx())))})
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() in (0, 15) for i in keep):
            continue
        edit = Chem.RWMol(mol)
        # Retain the source ester O and H-cap it; no side-chain carboxyl guess.
        keep.add(oxygen.GetIdx())
        for i in reversed(range(mol.GetNumAtoms())):
            if i not in keep:
                edit.RemoveAtom(i)
        acid = finish(edit)
        return acid
    raise ValueError("explicit_alpha_aminoacyl_ester_not_found_in_source")


def aminoacylate(mol, site, acid):
    for carbon in acid.GetAtoms():
        if carbon.GetAtomicNum() != 6:
            continue
        if not any(
            b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(carbon).GetAtomicNum() == 8
            for b in carbon.GetBonds()
        ):
            continue
        if not any(
            n.GetAtomicNum() == 6 and any(x.GetAtomicNum() == 7 for x in n.GetNeighbors())
            for n in carbon.GetNeighbors()
        ):
            continue
        oh = [
            n
            for n in carbon.GetNeighbors()
            if n.GetAtomicNum() == 8 and n.GetDegree() == 1 and n.GetTotalNumHs() == 1
        ]
        if not oh:
            continue
        offset = mol.GetNumAtoms()
        e = Chem.RWMol(Chem.CombineMols(mol, acid))
        e.AddBond(site, offset + carbon.GetIdx(), Chem.BondType.SINGLE)
        target = e.GetAtomWithIdx(site)
        target.SetNumExplicitHs(0)
        target.SetNoImplicit(False)
        e.RemoveAtom(offset + oh[0].GetIdx())
        product = finish(e)

        # Independently verify complete elemental condensation, including H.
        def counts(m):
            return Counter(a.GetSymbol() for a in Chem.AddHs(m).GetAtoms() if a.GetAtomicNum())

        total = counts(capped_core(mol)) + counts(neutral(acid))
        total.subtract({"H": 2, "O": 1})
        assert +total == counts(capped_core(product))
        return product
    raise ValueError("source_alpha_carboxyl_OH_required")


def free_head_alcohols(mol):
    fixed = protected(mol)
    for a in mol.GetAtoms():
        if (
            a.GetIdx() in fixed
            or a.GetAtomicNum() != 8
            or a.GetDegree() != 1
            or a.GetTotalNumHs() != 1
        ):
            continue
        n = a.GetNeighbors()[0]
        if n.GetAtomicNum() == 6 and n.GetHybridization() == Chem.HybridizationType.SP3:
            yield a.GetIdx()
