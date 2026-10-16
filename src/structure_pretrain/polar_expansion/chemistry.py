from collections import Counter

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from structure_pretrain.library_builder.canonicalize import nonisomeric_smiles
from structure_pretrain.library_builder.validation import validate_fragment, validate_polar_fragment
from structure_pretrain.polar_plausibility.tiering import evaluate
from structure_pretrain.pubchem_polar.chemistry import phosphorus_family, process_smiles


class AnalogIneligible(ValueError):
    pass


def evaluate_graph(smiles, config, rules):
    # Reuse the same graph rules without relabelling original first-stage provenance.
    result = evaluate({"status": "APPROVABLE", "polar_fragment_smiles": smiles}, config, rules)
    result.pop("status")
    return result


def acid_family(fragment):
    molecule = Chem.RWMol(fragment)
    dummy = next(a for a in molecule.GetAtoms() if a.GetAtomicNum() == 0)
    oxygen = Chem.Atom(8)
    oxygen.SetNumExplicitHs(1)
    oxygen.SetNoImplicit(True)
    molecule.ReplaceAtom(dummy.GetIdx(), oxygen)
    parent = molecule.GetMol()
    Chem.SanitizeMol(parent)
    return phosphorus_family(parent)


def element_counts(molecule):
    return Counter(a.GetSymbol() for a in Chem.AddHs(molecule).GetAtoms() if a.GetAtomicNum())


def phosphate_analog(smiles):
    molecule = validate_polar_fragment(smiles, 1)
    phosphorus = [a for a in molecule.GetAtoms() if a.GetAtomicNum() == 15]
    if len(phosphorus) != 1:
        raise AnalogIneligible("ANALOG_MULTI_P")
    if acid_family(molecule) != "phosphonate":
        raise AnalogIneligible("ANALOG_NOT_PHOSPHONATE")
    p = phosphorus[0]
    pc = [
        b
        for b in p.GetBonds()
        if b.GetOtherAtom(p).GetAtomicNum() == 6 and b.GetBondType() == Chem.BondType.SINGLE
    ]
    if len(pc) != 1:
        raise AnalogIneligible("ANALOG_PC_BOND_COUNT")
    bond = pc[0]
    if bond.IsInRing():
        raise AnalogIneligible("ANALOG_RING_PC_BOND")
    carbon = bond.GetOtherAtomIdx(p.GetIdx())
    editable = Chem.RWMol(molecule)
    editable.RemoveBond(p.GetIdx(), carbon)
    oxygen = editable.AddAtom(Chem.Atom(8))
    editable.AddBond(p.GetIdx(), oxygen, Chem.BondType.SINGLE)
    editable.AddBond(oxygen, carbon, Chem.BondType.SINGLE)
    changed = editable.GetMol()
    Chem.SanitizeMol(changed)
    # Verify every unedited bond and every original atom; only the P-C edge is subdivided.
    for old in molecule.GetBonds():
        if old.GetIdx() == bond.GetIdx():
            continue
        new = changed.GetBondBetweenAtoms(old.GetBeginAtomIdx(), old.GetEndAtomIdx())
        if new is None or new.GetBondType() != old.GetBondType():
            raise ValueError("ANALOG_TRANSFORM_ERROR: original connectivity changed")
    before, after = element_counts(molecule), element_counts(changed)
    heavy_delta = {
        e: after[e] - before[e]
        for e in before.keys() | after.keys()
        if e != "H" and after[e] != before[e]
    }
    if heavy_delta != {"O": 1} or acid_family(changed) != "phosphate":
        raise ValueError("ANALOG_TRANSFORM_ERROR: expected only heavy-atom O+1")
    key = nonisomeric_smiles(changed)
    # A new P-O-C path can newly match the builder's intact-lipid boundary.
    # Let the shared plausibility evaluation place such analogs in HOLD.
    validate_fragment(key, 1)
    return {
        "polar_fragment_smiles": key,
        "fragment_formula": rdMolDescriptors.CalcMolFormula(changed).replace("*", ""),
        "parent_fragment_formula": rdMolDescriptors.CalcMolFormula(molecule).replace("*", ""),
        "heavy_atom_delta": heavy_delta,
        "hydrogen_delta": after["H"] - before["H"],
        "parent_P_atom_index": p.GetIdx(),
        "parent_C_atom_index": carbon,
    }


def enumerate_source(cid, smiles, target_keys):
    with rdBase.BlockLogs():
        processed = process_smiles(smiles)
        source_mol = Chem.MolFromSmiles(smiles)
    mappings = []
    components = Chem.GetMolFrags(source_mol, asMols=True, fragsMolAtomMapping=mappings)
    p_indices = [
        i for i, mol in enumerate(components) if any(a.GetAtomicNum() == 15 for a in mol.GetAtoms())
    ]
    if len(p_indices) != 1:
        raise ValueError("Rescue source requires exactly one phosphorus-containing component")
    index_map = mappings[p_indices[0]]
    grouped, skipped = {}, Counter()
    for candidate in processed["candidates"]:
        key = candidate["polar_fragment_smiles"]
        if key not in target_keys:
            skipped["NOT_TARGET_MULTI_FRAGMENT"] += 1
            continue
        if candidate["status"] not in {"MULTIPLE_ATTACHMENT_AMBIGUITY", "APPROVABLE"}:
            skipped[candidate["status"]] += 1
            continue
        entry = grouped.setdefault(key, {"polar_family": candidate["polar_family"], "sites": []})
        local_index = candidate["source_attachment_atom_index"]
        entry["sites"].append(
            {
                "cid": cid,
                "source_attachment_atom_index": index_map[local_index],
                "selected_component_attachment_atom_index": local_index,
            }
        )
    return grouped, {
        "cid": cid,
        "enumerated_sites": len(processed["candidates"]),
        "distinct_source_fragments": processed["distinct_attachment_fragments"],
        "target_distinct_fragments": len(grouped),
        "skipped_site_reasons": dict(skipped),
    }
