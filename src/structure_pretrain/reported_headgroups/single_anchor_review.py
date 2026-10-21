"""Conservative review of extra lipid anchors in already rooted Polar graphs.

This is a derived selection; no atom is removed and the source library is unchanged.
Multiple phosphorus atoms and large polar glycans are not lipid-anchor evidence.
"""

from rdkit import Chem

MULTI_ANCHOR_CLASSES = {"CL", "BPG", "BMP", "LBPA", "SLBPA", "DEACL"}
N_ACYL_CLASSES = {"NAPE", "NAPS", "PS-NAC", "PE-NAC"}


def carbon_paths(mol, *, include_components=False):
    """Return paths in non-ring aliphatic carbon components (linear/branched trees).

    Carbonyl carbons and carbons bearing heteroatoms are excluded. This prevents
    lysine, sugars and polyols from being treated as hydrocarbon lipid chains.
    Ring atoms are excluded, so the induced graph is a forest and exact tree
    diameters can be used without exponential path enumeration.
    """
    eligible = {
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6
        and not a.IsInRing()
        and not a.GetIsAromatic()
        and all(n.GetAtomicNum() in (1, 6) for n in a.GetNeighbors())
    }
    adj = {
        i: {n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()} & eligible for i in eligible
    }
    remaining = set(eligible)
    paths, components = [], []
    while remaining:
        seed = min(remaining)
        parent = {seed: None}
        queue = [seed]
        for node in queue:
            for nb in sorted(adj[node] - parent.keys()):
                parent[nb] = node
                queue.append(nb)
        component = set(queue)
        components.append(component)
        remaining -= component
        start = queue[-1]
        parent = {start: None}
        queue = [start]
        for node in queue:
            for nb in sorted(adj[node] - parent.keys()):
                parent[nb] = node
                queue.append(nb)
        path = []
        node = queue[-1]
        while node is not None:
            path.append(node)
            node = parent[node]
        paths.append(path)
    paths = sorted(paths, key=lambda p: (-len(p), p))
    return (paths, components) if include_components else paths


def analyze_single_anchor(smiles, abbreviation="", builder_compatible=True, review_reason=""):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Cannot depict invalid Polar SMILES")
    dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if (
        len(dummies) != 1
        or mol.GetAtomWithIdx(dummies[0]).GetAtomMapNum() != 1
        or mol.GetAtomWithIdx(dummies[0]).GetDegree() != 1
        or mol.GetAtomWithIdx(dummies[0]).GetNeighbors()[0].GetAtomicNum() != 15
        or len(Chem.GetMolFrags(mol)) != 1
    ):
        raise ValueError("Expected one connected Polar graph with a single P-[*:1] root")
    paths, components = carbon_paths(mol, include_components=True)
    longest = max((len(p) for p in paths), default=0)
    long_atoms = sorted({i for p in paths if len(p) >= 8 for i in p})
    branched_anchor_atoms = {
        i
        for component in components
        if len(component) >= 8 and any(len(p) >= 6 and p[0] in component for p in paths)
        for i in component
    }
    # A second diacylated glycerol arm is an extra lipid anchor even for short
    # chains. One amino-acyl ester on PG does not match this two-ester pattern.
    extra_arm = Chem.MolFromSmarts(
        "[CH2]([OX2][CX3](=[OX1]))-[CH]([OX2][CX3](=[OX1]))-[CH2][OX2][P]"
    )
    extra_arm_atoms = {i for match in mol.GetSubstructMatches(extra_arm) for i in match}
    rings = mol.GetRingInfo().AtomRings()
    carbon_rings = [
        set(r)
        for r in rings
        if len(r) in (5, 6)
        and all(
            mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and not mol.GetAtomWithIdx(i).GetIsAromatic()
            for i in r
        )
    ]
    fused = set()
    for i, ring in enumerate(carbon_rings):
        if sum(len(ring & other) >= 2 for j, other in enumerate(carbon_rings) if i != j):
            fused.update(ring)
    sterol = len(carbon_rings) >= 4 and len(fused) >= 15
    carbonyl = Chem.MolFromSmarts("[#6]-[CX3](=[OX1])-[$([OX2H0][#6]),N,S]")
    acyl_matches = mol.GetSubstructMatches(carbonyl)
    abbr = abbreviation.strip().upper()
    reasons, codes = [], []
    status = "retain"
    if abbr in MULTI_ANCHOR_CLASSES:
        status = "exclude"
        codes.append("explicit_multi_lipid_anchor_class")
        reasons.append("CL/BMP/LBPA 等额外磷脂酰臂或脂质锚点，按本轮范围排除")
    if abbr in N_ACYL_CLASSES and acyl_matches:
        status = "exclude"
        codes.append("N_acyl_PE_PS")
        reasons.append("NAPE/NAPS 型头基含额外 N-酰基连接")
    if longest >= 8:
        status = "exclude"
        codes.append("extra_hydrocarbon_chain_8C_or_more")
        reasons.append(f"Polar 内仍有连续 {longest} 个无杂原子取代的非环脂肪碳，存在额外长链")
    elif branched_anchor_atoms:
        status = "exclude"
        codes.append("extra_branched_hydrocarbon_anchor")
        reasons.append("头基含至少 8 碳的支化疏水片段及至少 6 碳主路径，构成额外脂质锚点")
    if extra_arm_atoms:
        status = "exclude"
        codes.append("extra_diacylated_glycerol_arm")
        reasons.append("头基内另有双酰化甘油磷脂臂；即使酰链较短也属于额外脂质锚点")
    if sterol:
        status = "exclude"
        codes.append("extra_sterol_lipid_anchor")
        reasons.append("头基含甾醇四稠环及疏水侧链，构成额外脂质锚点")
    if status != "exclude":
        if longest >= 5:
            codes.append("borderline_hydrophobic_segment")
            reasons.append(f"有 {longest} 碳疏水片段，是否属于额外脂质链需看图判断")
        if acyl_matches:
            codes.append("short_or_amino_acyl_substituent")
            reasons.append("含酯或酰胺取代，未检出明确的酰基长链，单列待判断")
        if not builder_compatible:
            codes.append("existing_builder_incompatible")
            reasons.append("无明确额外长链，但当前 Builder 不兼容，暂不列入可直接使用集合")
        if review_reason and review_reason not in {"intact_lipid_as_polar", "unsupported_element"}:
            codes.append("source_record_requires_review")
            reasons.append("来源库存在名称与结构待核实事项")
        if codes:
            status = "review"
        else:
            codes.append("single_root_no_extra_lipid_anchor_detected")
            reasons.append("单一 P-[*:1] 连接；未检出额外脂肪酸长链或脂质锚点")
    root = mol.GetAtomWithIdx(dummies[0]).GetNeighbors()[0].GetIdx()
    return {
        "auto_status": status,
        "reason_codes": codes,
        "filter_reason": "；".join(reasons),
        "longest_hydrocarbon_path": longest,
        "short_or_amino_acyl_count": len(acyl_matches),
        "sterol_anchor_detected": sterol,
        "highlight_extra_atoms": sorted(
            set(long_atoms) | branched_anchor_atoms | extra_arm_atoms | (fused if sterol else set())
        ),
        "attachment_atom": dummies[0],
        "attachment_P_atom": root,
    }
