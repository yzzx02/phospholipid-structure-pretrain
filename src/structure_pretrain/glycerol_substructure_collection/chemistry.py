"""Extract single-head glycerol cores without requiring existing fatty chains."""

from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from structure_pretrain.glycerol_core_collection.chemistry import (
    BACKBONE,
    canonical,
    capped_core,
    component,
    descriptors,
    parse,
    reassemble,
    stable_id,
)
from structure_pretrain.headgroup_target_expansion.chemistry import (
    ACYL_P,
    PEROXIDE,
    balanced_parent,
    extra_glycerol_lipid_anchor,
    fatty_N_acyl_PE_PS_head,
)

TAILS = [(101, "CC(=O)[*:101]"), (102, "CC(=O)[*:102]")]


def counts(mol):
    return Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms() if a.GetAtomicNum())


def normalize(mol):
    result, notes = balanced_parent(mol)
    Chem.RemoveStereochemistry(result)
    return result, notes


def acyclic_long_chain(mol, minimum=6, allowed_atoms=None):
    """Six carbon-only acyclic atoms, without walking through carbocyclic rings."""
    eligible = {
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6
        and not a.IsInRing()
        and not a.GetIsAromatic()
        and (allowed_atoms is None or a.GetIdx() in allowed_atoms)
        and all(n.GetAtomicNum() in (1, 6) for n in a.GetNeighbors())
    }

    def walk(i, path):
        if len(path) >= minimum:
            return True
        return any(
            walk(n.GetIdx(), path | {n.GetIdx()})
            for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetIdx() in eligible and n.GetIdx() not in path
        )

    return any(walk(i, {i}) for i in eligible)


def role_frame(mol):
    sockets = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(sockets) != 2 or {a.GetAtomMapNum() for a in sockets} != {101, 102}:
        raise ValueError("two_distinct_chain_roles_required")
    if any(a.GetDegree() != 1 or a.GetNeighbors()[0].GetAtomicNum() != 8 for a in sockets):
        raise ValueError("chain_role_not_single_glycerol_oxygen")
    oxygen_set = {a.GetNeighbors()[0].GetIdx() for a in sockets}
    frames = []
    for match in mol.GetSubstructMatches(BACKBONE, uniquify=False):
        if not oxygen_set <= {match[1], match[3], match[5]}:
            continue
        if any(mol.GetAtomWithIdx(i).IsInRing() for i in (match[0], match[2], match[4])):
            continue
        head_o = next(i for i in (match[1], match[3], match[5]) if i not in oxygen_set)
        head_c = next(
            n.GetIdx()
            for n in mol.GetAtomWithIdx(head_o).GetNeighbors()
            if n.GetIdx() in (match[0], match[2], match[4])
        )
        external = [
            n.GetIdx() for n in mol.GetAtomWithIdx(head_o).GetNeighbors() if n.GetIdx() != head_c
        ]
        if len(external) == 1:
            frames.append((match, head_o, head_c, external[0]))
    if not frames:
        raise ValueError("single_head_acyclic_glycerol_frame_required")
    return min(frames)


def representations(mol):
    match, head_o, head_c, external = role_frame(mol)
    key = canonical(mol, False)
    gly = Chem.MolToSmiles(mol, rootedAtAtom=head_c, canonical=True, isomericSmiles=False)
    p = external if mol.GetAtomWithIdx(external).GetAtomicNum() == 15 else None
    target = (
        Chem.MolToSmiles(mol, rootedAtAtom=p, canonical=True, isomericSmiles=False)
        if p is not None
        else gly
    )
    assert canonical(parse(target), False) == canonical(parse(gly), False) == key
    polar = ""
    if p is not None:
        keep = component(mol, p, {tuple(sorted((head_o, p)))}) | {head_o}
        if any(i in keep for i in (match[0], match[2], match[4])):
            raise ValueError("head_crosslinks_glycerol")
        edit = Chem.RWMol(mol)
        atom = edit.GetAtomWithIdx(head_o)
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
        polar = Chem.MolToSmiles(
            head,
            rootedAtAtom=marker.GetNeighbors()[0].GetIdx(),
            canonical=True,
            isomericSmiles=False,
        )
        assert canonical(parse(polar), False) == canonical(head, False)
    return {
        "target_id": stable_id(key, "HT"),
        "anchored_graph_key": key,
        "target_smiles": target,
        "glycerol_first_smiles": gly,
        "polar_P_first_smiles": polar,
        "capped_core_smiles": canonical(capped_core(mol), False),
        "rooted_at_attachment_P": p is not None,
        "head_attachment_position": "central" if head_c == match[2] else "terminal",
        "legacy_terminal_frame_compatible": head_c != match[2],
    }


def policy(mol, *, new_pubchem_branch=False):
    match, _, head_c, _ = role_frame(mol)
    cap = capped_core(mol)
    desc = descriptors(canonical(cap, False))
    discard = {
        "core_mass_above_limit",
        "halogen_hold_for_natural_evidence",
        "extra_head_long_alkyl_or_acyl_chain",
    }
    reasons = set(desc["structural_filter_reasons"]) - discard
    limit = 600.0 if desc["sugar_ring_count"] or desc["inositol_ring_count"] else 500.0
    if desc["core_average_mass_Da"] > limit + 1e-8:
        reasons.add("core_mass_above_500_600_policy")
    if acyclic_long_chain(cap):
        reasons.add("extra_head_long_alkyl_or_acyl_chain")
    if extra_glycerol_lipid_anchor(mol, match[2]):
        reasons.add("extra_glycerol_lipid_anchor_including_short_acyl")
    if fatty_N_acyl_PE_PS_head(mol):
        reasons.add("extra_fatty_N_acyl_PE_PS_head")
    if cap.HasSubstructMatch(ACYL_P):
        reasons.add("reactive_acyl_phosphate_mixed_anhydride")
    if cap.HasSubstructMatch(PEROXIDE):
        reasons.add("peroxide_requires_review")
    if desc["P_count"] > 4:
        reasons.add("more_than_four_P")
    if new_pubchem_branch and desc["P_count"]:
        reasons.add("P_containing_source_outside_new_nonP_branch")
    if new_pubchem_branch and head_c == match[2]:
        reasons.add("sn2_head_attachment_excluded")
    desc.update(
        mass_limit_Da=limit,
        filter_reasons=sorted(reasons),
        structural_filter_reasons=sorted(reasons),
    )
    return desc


def verify_assembly(mol):
    role_frame(mol)
    cap = capped_core(mol)
    product = reassemble(mol, TAILS)
    assert counts(product) == counts(cap) + Counter({"C": 4, "H": 4, "O": 2})
    assert Chem.GetFormalCharge(product) == 0
    assert not any(a.GetAtomicNum() == 0 for a in product.GetAtoms())
    return {
        "full_assembly_H_formula_pass": True,
        "core_formula": rdMolDescriptors.CalcMolFormula(cap),
        "core_average_mass_Da": Descriptors.MolWt(cap),
    }


def outside_neighbor(mol, carbon, oxygen):
    others = [n.GetIdx() for n in mol.GetAtomWithIdx(oxygen).GetNeighbors() if n.GetIdx() != carbon]
    if len(others) > 1:
        raise ValueError("branching_glycerol_oxygen")
    return others[0] if others else None


def simple_external_tail(mol, oxygen, external, glycerol_atoms):
    """Simple hydrocarbon acyl/ether only; amino/citrate heads are not erased."""
    if external is None:
        return False
    atom = mol.GetAtomWithIdx(external)
    if atom.GetAtomicNum() != 6:
        return False
    part = component(mol, external, {tuple(sorted((oxygen, external)))})
    if part & glycerol_atoms:
        return False
    atoms = [mol.GetAtomWithIdx(i) for i in part]
    if any(a.GetAtomicNum() not in (1, 6, 8) or a.IsInRing() for a in atoms):
        return False
    carbonyl = [
        b
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in part
        and b.GetEndAtomIdx() in part
        and b.GetBondType() == Chem.BondType.DOUBLE
        and {b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum()} == {6, 8}
    ]
    if len(carbonyl) == 1 and any(
        b.GetOtherAtom(atom).GetAtomicNum() == 8 and b.GetBondType() == Chem.BondType.DOUBLE
        for b in atom.GetBonds()
    ):
        # One carboxyl carbonyl, at most two chain oxygen atoms; other oxyacids stay heads.
        oxygen_count = sum(a.GetAtomicNum() == 8 for a in atoms)
        return oxygen_count == 1 or (
            oxygen_count <= 3 and acyclic_long_chain(mol, allowed_atoms=part)
        )
    return (
        not carbonyl
        and all(a.GetAtomicNum() in (1, 6) for a in atoms)
        and sum(a.GetAtomicNum() == 6 for a in atoms) >= 6
    )


def _fragment_with_marker(mol, keep, attach, slot):
    edit = Chem.RWMol(mol)
    atom = Chem.Atom(0)
    atom.SetAtomMapNum(slot)
    j = edit.AddAtom(atom)
    edit.AddBond(attach, j, Chem.BondType.SINGLE)
    for i in reversed(range(mol.GetNumAtoms())):
        if i not in keep:
            edit.RemoveAtom(i)
    result = edit.GetMol()
    Chem.SanitizeMol(result)
    return result


def _remove_unused_markers(mol, used):
    edit = Chem.RWMol(mol)
    remove = []
    for a in edit.GetAtoms():
        if a.GetAtomicNum() == 0 and a.GetAtomMapNum() not in used:
            a.GetNeighbors()[0].SetNumExplicitHs(0)
            a.GetNeighbors()[0].SetNoImplicit(False)
            remove.append(a.GetIdx())
    for i in sorted(remove, reverse=True):
        edit.RemoveAtom(i)
    result = edit.GetMol()
    Chem.SanitizeMol(result)
    return result


def extract_observed_cores(text, *, allow_other_site_substructure_cuts=False):
    """Each accepted head branch exists in the source; free OH can be future sockets."""
    if "." in text:
        parent = Chem.MolFromSmiles(text)
        if parent is None:
            raise ValueError("invalid_SMILES")
        output, statuses, seen = [], Counter(), set()
        for fragment in Chem.GetMolFrags(parent, asMols=True, sanitizeFrags=True):
            if not fragment.HasSubstructMatch(BACKBONE):
                continue
            component_text = canonical(fragment)
            items, state = extract_observed_cores(
                component_text,
                allow_other_site_substructure_cuts=allow_other_site_substructure_cuts,
            )
            statuses.update(state)
            for item in items:
                key = canonical(item["mol"], False)
                if key not in seen:
                    seen.add(key)
                    item.update(
                        source_parent_had_multiple_components=True,
                        source_component_smiles=component_text,
                    )
                    output.append(item)
        if not output:
            statuses["no_usable_glycerol_source_component"] += 1
        return output, statuses
    mol = parse(text)
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError("generic_source_structure")
    output, seen, statuses = [], set(), Counter()
    for match in mol.GetSubstructMatches(BACKBONE, uniquify=False):
        carbons = (match[0], match[2], match[4])
        if any(mol.GetAtomWithIdx(i).IsInRing() for i in carbons):
            continue
        oxygens = (match[1], match[3], match[5])
        frame_set = set(match)
        external = [outside_neighbor(mol, c, o) for c, o in zip(carbons, oxygens, strict=True)]
        if any(
            n is not None and mol.GetBondBetweenAtoms(o, n).GetBondType() != Chem.BondType.SINGLE
            for o, n in zip(oxygens, external, strict=True)
        ):
            statuses["non_single_glycerol_O_link_pending"] += 1
            continue
        tails = [
            simple_external_tail(mol, o, n, frame_set)
            for o, n in zip(oxygens, external, strict=True)
        ]
        for head_index, neighbor in enumerate(external):
            if neighbor is None or tails[head_index]:
                continue
            other = [i for i in range(3) if i != head_index]
            if not allow_other_site_substructure_cuts and any(
                external[i] is not None and not tails[i] for i in other
            ):
                statuses["multiple_nonchain_glycerol_substituents_pending"] += 1
                continue
            head_o = oxygens[head_index]
            head = component(mol, neighbor, {tuple(sorted((head_o, neighbor)))})
            if head & frame_set:
                statuses["crosslinked_or_cyclic_head_pending"] += 1
                continue
            keep = frame_set | head
            if sum(mol.GetAtomWithIdx(i).GetMass() for i in keep) > 600.0 + 1e-8:
                statuses["head_heavy_atom_mass_alone_above_600"] += 1
                continue
            # On terminal-head glycerol, retain the historical terminal101/central102 roles.
            assignments = (
                {other[0]: 101, other[1]: 102}
                if head_index == 1
                else {i: (102 if i == 1 else 101) for i in other}
            )
            edit = Chem.RWMol(mol)
            source_tails = []
            for i in other:
                slot, oxygen, tail_start = assignments[i], oxygens[i], external[i]
                if tail_start is not None:
                    part = component(mol, tail_start, {tuple(sorted((oxygen, tail_start)))})
                    if part & keep:
                        raise ValueError("tail_crosslinks_source_head")
                    tail = _fragment_with_marker(mol, part, tail_start, slot)
                    source_tails.append((slot, canonical(tail)))
                marker = Chem.Atom(0)
                marker.SetAtomMapNum(slot)
                j = edit.AddAtom(marker)
                edit.GetAtomWithIdx(oxygen).SetNumExplicitHs(0)
                edit.GetAtomWithIdx(oxygen).SetNoImplicit(False)
                edit.AddBond(oxygen, j, Chem.BondType.SINGLE)
            for i in reversed(range(mol.GetNumAtoms())):
                if i not in keep:
                    edit.RemoveAtom(i)
            raw_core = edit.GetMol()
            Chem.SanitizeMol(raw_core)
            observed_back = reassemble(
                _remove_unused_markers(raw_core, {s for s, _ in source_tails}), source_tails
            )
            assert canonical(observed_back, False) == canonical(mol, False)
            assert counts(observed_back) == counts(mol)
            core, notes = normalize(raw_core)
            key = canonical(core, False)
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "mol": core,
                    "observed_fatty_chain_count": sum(tails[i] for i in other),
                    "source_cut_substituent_count": len(source_tails),
                    "source_nonfatty_substituent_cut_count": sum(
                        external[i] is not None and not tails[i] for i in other
                    ),
                    "source_glycerol_atom_indices": list(match),
                    "source_head_O_atom_index": head_o,
                    "source_head_neighbor_atom_index": neighbor,
                    "source_roundtrip_H_formula_pass": True,
                    "source_head_branch_preserved": True,
                    "normalization_notes": notes,
                }
            )
    if not output:
        statuses["no_single_nonfatty_head_with_two_usable_glycerol_sites"] += 1
    return output, statuses
