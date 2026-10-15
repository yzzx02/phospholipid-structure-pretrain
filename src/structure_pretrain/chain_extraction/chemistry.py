"""Conservative graph-derived linear chains on glycerol phosphate backbones."""

import re
from functools import lru_cache

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")
# Broad vocabulary, including modified and less common glycerophospholipids.
PRESCREEN = re.compile(
    r"phosph|plasmal|glyceroph|cardiolip|lyso|(?:^|[^a-z])(?:ether|ox)?"
    r"(?:pc|pe|pg|pi|ps|pa|pt|pmeoh|petoh|pbtoh|pme|pdme|cl|mlcl|dlcl|bmp|lbpa|"
    r"lpc|lpe|lpg|lpi|lps|lpa|nape|lnape|cdpdag|cdp-dg|pip|pip2|pip3)(?:$|[^a-z])"
    r"|pim\d",
    re.IGNORECASE,
)
BACKBONE = Chem.MolFromSmarts("[C;X4]([O])-[C;X4]([O])-[C;X4]-[O]-[P](=[O])([O])[O]")
ETHER_NAME = re.compile(r"(?<![A-Za-z])([OP])-?(\d+):(\d+)")


def is_candidate(row):
    # Elemental safety net catches unknown/abbreviated phosphate-containing classes.
    # It only widens the prescreen; acceptance still requires a graph backbone.
    return "P" in (row.get("smiles") or "") or bool(
        PRESCREEN.search(" ".join(row.get(k) or "" for k in ("name", "ontology", "compound_class")))
    )


def rejection(reason, detail, **kwargs):
    return {"chains": [], "rejections": [{"reason": reason, "detail": detail}], **kwargs}


@lru_cache(maxsize=8192)
def _fragment(kind, carbon_count, positions):
    """Serialize a validated graph path; cache only its exact graph-derived signature."""
    fragment = "[*:1]C" + ("(=O)" if kind == "A" else "")
    fragment += "".join(("=" if i in positions else "") + "C" for i in range(1, carbon_count))
    mol = Chem.MolFromSmiles(fragment)
    return fragment, Chem.MolToSmiles(mol, isomericSmiles=False)


def _chain(mol, oxygen, root, scaffold):
    """Cut O-C; the connected component must be a simple unmodified C path."""
    bond = mol.GetBondBetweenAtoms(oxygen, root)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None, "cannot_identify_chain_bond", "attachment_not_single"
    seen, todo = set(), [root]
    while todo:
        atom = todo.pop()
        if atom in seen:
            continue
        seen.add(atom)
        todo.extend(
            n.GetIdx()
            for n in mol.GetAtomWithIdx(atom).GetNeighbors()
            if n.GetIdx() != oxygen and n.GetIdx() not in seen
        )
    if seen & scaffold:
        return None, "cannot_identify_chain_bond", "chain_reconnects_to_scaffold"
    first = mol.GetAtomWithIdx(root)
    carbonyl = [
        n.GetIdx()
        for n in first.GetNeighbors()
        if n.GetAtomicNum() == 8
        and mol.GetBondBetweenAtoms(root, n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    ]
    kind = "A" if carbonyl else "O"
    carbon = {i for i in seen if mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
    other = seen - carbon - set(carbonyl)
    if other:
        return None, "unsupported_modified_chain", "heteroatom_or_oxidized_chain"
    if root not in carbon or len(carbonyl) > 1:
        return None, "cannot_identify_chain_bond", "attachment_not_chain_carbon"
    for i in seen:
        a = mol.GetAtomWithIdx(i)
        if a.IsInRing():
            return None, "unsupported_modified_chain", "cyclic_epoxy_or_cyclopropyl_chain"
        if a.GetIsAromatic() or a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons():
            return None, "unsupported_modified_chain", "aromatic_charged_isotopic_or_radical_chain"
    ordered, prev, current = [], None, root
    while True:
        ordered.append(current)
        nxt = [
            n.GetIdx()
            for n in mol.GetAtomWithIdx(current).GetNeighbors()
            if n.GetIdx() in carbon and n.GetIdx() != prev
        ]
        if len(nxt) > 1:
            return None, "unsupported_modified_chain", "branched_chain"
        if not nxt:
            break
        prev, current = current, nxt[0]
    if len(ordered) != len(carbon):
        return None, "unsupported_modified_chain", "nonlinear_chain"
    positions = []
    for pos, (a, b) in enumerate(zip(ordered, ordered[1:]), 1):
        bt = mol.GetBondBetweenAtoms(a, b).GetBondType()
        if bt == Chem.BondType.DOUBLE:
            positions.append(pos)
        elif bt != Chem.BondType.SINGLE:
            return None, "unsupported_modified_chain", "triple_or_nonstandard_bond"
    if kind == "O" and positions and positions[0] == 1:
        kind = "P"
    fragment, canonical = _fragment(kind, len(carbon), tuple(positions))
    return (
        {
            "chain_token": f"{kind}-{len(carbon)}:{len(positions)}",
            "linkage_type": kind,
            "carbon_count": len(carbon),
            "double_bond_count": len(positions),
            "db_positions": positions,
            "fragment_smiles": fragment,
            "canonical_fragment_smiles": canonical,
            "p_db_count_includes_vinyl": True if kind == "P" else None,
        },
        None,
        None,
    )


def extract_structure(smiles):
    if not smiles:
        return rejection("missing_smiles", "empty_smiles", parsed=False, backbone=False)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return rejection("invalid_smiles", "rdkit_parse_or_sanitize", parsed=False, backbone=False)
    parent = Chem.MolToSmiles(mol, isomericSmiles=False)
    matches = mol.GetSubstructMatches(BACKBONE)
    # SMARTS includes phosphate oxygens in symmetric permutations: deduplicate glycerol triples.
    backbones = {}
    for m in matches:
        c1, o1, c2, o2, c3, o3, phosphorus = m[:7]
        carbons = (c1, c2, c3)
        if any(
            sum(n.GetAtomicNum() == 6 for n in mol.GetAtomWithIdx(c).GetNeighbors()) != expected
            for c, expected in zip(carbons, (1, 2, 1))
        ):
            continue
        if len({c1, c2, c3, o1, o2, o3, phosphorus}) != 7:
            continue
        # A symmetric central cardiolipin glycerol can match in both directions.
        if not any(set(existing) == set(carbons) for existing in backbones):
            backbones[carbons] = (o1, o2, o3, phosphorus)
    if not backbones:
        return rejection(
            "not_target_backbone",
            "no_linear_glycerol_phosphate",
            parsed=True,
            backbone=False,
            parent=parent,
        )
    # CL can have two chain-bearing glycerols; their atom sets must be disjoint.
    used = set()
    for cs, os in backbones.items():
        atoms = set(cs) | set(os[:3])
        if used & atoms:
            return rejection(
                "ambiguous_backbone",
                "overlapping_glycerol_matches",
                parsed=True,
                backbone=False,
                parent=parent,
            )
        used |= atoms
    chains, rejected = [], []
    for cs, os in backbones.items():
        for position, (c, oxygen) in enumerate(zip(cs[:2], os[:2]), 1):
            oa = mol.GetAtomWithIdx(oxygen)
            attachments = [n.GetIdx() for n in oa.GetNeighbors() if n.GetIdx() != c]
            if not attachments:  # lyso / central cardiolipin hydroxyl
                continue
            if len(attachments) != 1:
                rejected.append({"reason": "cannot_identify_chain_bond", "detail": "oxygen_degree"})
                continue
            root = attachments[0]
            if mol.GetAtomWithIdx(root).GetAtomicNum() == 15:  # central CL glycerol
                continue
            chain, reason, detail = _chain(mol, oxygen, root, used)
            if chain:
                chain["observed_attachment_position"] = (
                    f"glycerol{list(backbones).index(cs) + 1}:C{position}"
                )
                chains.append(chain)
            else:
                rejected.append({"reason": reason, "detail": detail})
    if not chains and not rejected:
        rejected.append({"reason": "cannot_identify_chain_bond", "detail": "no_attached_chain"})
    return {
        "parsed": True,
        "backbone": True,
        "parent": parent,
        "chains": chains,
        "rejections": rejected,
    }


def source_naming(chain, name):
    """Compare a graph-derived chain to explicit O-/P- nomenclature; never infer positions."""
    candidates = {
        (label, int(db))
        for label, c, db in ETHER_NAME.findall(name or "")
        if int(c) == chain["carbon_count"]
    }
    if chain["linkage_type"] not in ("O", "P") or len(candidates) != 1:
        return {
            "source_ether_label": None,
            "source_named_db_count": None,
            "source_p_db_count_includes_vinyl": None,
            "source_naming_status": "UNRESOLVED",
        }
    label, db = candidates.pop()
    actual = chain["double_bond_count"]
    includes = (
        (db == actual) if chain["linkage_type"] == "P" and db in (actual, actual - 1) else None
    )
    status = "MATCHED" if db == actual else "DB_COUNT_DIFFERS"
    if chain["linkage_type"] != label:
        status = "SOURCE_LABEL_LINKAGE_CONFLICT"
    return {
        "source_ether_label": label,
        "source_named_db_count": db,
        "source_p_db_count_includes_vinyl": includes,
        "source_naming_status": status,
    }
