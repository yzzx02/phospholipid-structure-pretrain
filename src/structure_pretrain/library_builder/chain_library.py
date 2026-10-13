import csv
import json
import re
from dataclasses import replace
from pathlib import Path

from rdkit import Chem

from .canonicalize import nonisomeric_smiles
from .schemas import ChainRule, LibraryValidationError, Rejection
from .validation import validate_fragment

TOKEN = re.compile(r"([AOP])-(\d+):(\d+)")


def boolean(text: str, default: bool | None = None):
    if text is None or not str(text).strip():
        return default
    text = str(text).strip().lower()
    if text in ("1", "true", "yes"):
        return True
    if text in ("0", "false", "no"):
        return False
    raise ValueError(f"Invalid boolean {text}")


def chain_sort_key(token: str):
    match = TOKEN.fullmatch(token)
    if not match:
        raise LibraryValidationError("chain_token_invalid", f"Invalid chain token {token}")
    linkage, carbon, db = match.groups()
    return {"O": 0, "P": 1, "A": 2}[linkage], int(carbon), int(db)


def validate_chain(rule: ChainRule) -> ChainRule:
    match = TOKEN.fullmatch(rule.chain_token)
    if not match or match.groups() != (
        rule.linkage_type,
        str(rule.carbon_count),
        str(rule.double_bond_count),
    ):
        raise LibraryValidationError(
            "chain_token_mismatch", "Token and supplied chain counts/type differ"
        )
    if not rule.source:
        raise LibraryValidationError(
            "source_missing", "Chain representative provenance is required"
        )
    molecule, dummy, start = validate_fragment(rule.representative_fragment_smiles, 1)
    if molecule.GetAtomWithIdx(start).GetSymbol() != "C":
        raise LibraryValidationError("chain_boundary_invalid", "Chain dummy must attach to carbon")
    carbons = {atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetSymbol() == "C"}
    oxygens = [atom for atom in molecule.GetAtoms() if atom.GetSymbol() == "O"]
    if len(carbons) != rule.carbon_count:
        raise LibraryValidationError(
            "carbon_count_mismatch", "Fragment carbon count differs from table"
        )
    if len(carbons) + len(oxygens) + 1 != molecule.GetNumAtoms():
        raise LibraryValidationError(
            "unsupported_chain_chemistry", "v0.1 requires unmodified A/O/P chains"
        )
    carbonyl = [
        bond
        for bond in molecule.GetAtomWithIdx(start).GetBonds()
        if bond.GetOtherAtomIdx(start) != dummy
        and bond.GetOtherAtom(molecule.GetAtomWithIdx(start)).GetSymbol() == "O"
        and bond.GetBondType() == Chem.BondType.DOUBLE
    ]
    if rule.linkage_type == "A" and (len(oxygens) != 1 or len(carbonyl) != 1):
        raise LibraryValidationError("linkage_mismatch", "A attachment must be an acyl carbonyl")
    if rule.linkage_type in ("O", "P") and oxygens:
        raise LibraryValidationError(
            "linkage_mismatch", "O/P fragment excludes the scaffold oxygen"
        )
    visited, positions, previous, current = set(), [], None, start
    while True:
        if current in visited:
            raise LibraryValidationError(
                "unsupported_chain_chemistry", "Chain must be linear and acyclic"
            )
        visited.add(current)
        next_atoms = [
            neighbor.GetIdx()
            for neighbor in molecule.GetAtomWithIdx(current).GetNeighbors()
            if neighbor.GetIdx() in carbons and neighbor.GetIdx() != previous
        ]
        if len(next_atoms) > 1:
            raise LibraryValidationError(
                "unsupported_chain_chemistry", "Branched chains need a future rule"
            )
        if not next_atoms:
            break
        following = next_atoms[0]
        bond_type = molecule.GetBondBetweenAtoms(current, following).GetBondType()
        if bond_type == Chem.BondType.DOUBLE:
            positions.append(len(visited))
        elif bond_type != Chem.BondType.SINGLE:
            raise LibraryValidationError(
                "unsupported_chain_chemistry", "Require C-C single/double bonds"
            )
        previous, current = current, following
    if visited != carbons:
        raise LibraryValidationError(
            "unsupported_chain_chemistry", "Chain carbon path is incomplete"
        )
    if tuple(positions) != rule.representative_double_bond_positions:
        raise LibraryValidationError(
            "double_bond_positions_mismatch", "Supplied positions differ from source SMILES"
        )
    observed = len(positions)
    if rule.linkage_type == "P":
        if not positions or positions[0] != 1 or rule.p_db_count_includes_vinyl is None:
            raise LibraryValidationError(
                "p_convention_missing",
                "P requires vinyl position 1 and an explicit external DB-count convention",
            )
        observed -= int(not rule.p_db_count_includes_vinyl)
    elif rule.linkage_type == "O" and positions and positions[0] == 1:
        raise LibraryValidationError(
            "linkage_mismatch", "An O-linked alpha vinyl fragment is chemically P"
        )
    if observed != rule.double_bond_count:
        raise LibraryValidationError(
            "double_bond_count_mismatch", "Fragment unsaturation differs from supplied convention"
        )
    if sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()):
        raise LibraryValidationError("nonneutral_structure", "Chain must have total charge zero")
    return replace(rule, representative_fragment_smiles=nonisomeric_smiles(molecule))


def import_chains(path: str | Path, rejections: list[Rejection]) -> dict[str, ChainRule]:
    grouped = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = set(ChainRule.__dataclass_fields__) - {"p_db_count_includes_vinyl"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Missing chain columns: {sorted(required - set(reader.fieldnames or []))}"
            )
        for number, row in enumerate(reader, 2):
            token = row.get("chain_token", f"line-{number}")
            try:
                positions = tuple(json.loads(row["representative_double_bond_positions"] or "[]"))
                if any(type(value) is not int for value in positions):
                    raise ValueError("DB positions must be an integer JSON list")
                rule = validate_chain(
                    ChainRule(
                        token,
                        row["linkage_type"],
                        int(row["carbon_count"]),
                        int(row["double_bond_count"]),
                        row["representative_fragment_smiles"],
                        positions,
                        row["source"],
                        boolean(row.get("p_db_count_includes_vinyl")),
                    )
                )
                grouped.setdefault(token, []).append(rule)
            except (ValueError, TypeError) as exc:
                rejections.append(
                    Rejection(
                        "chain_import", token, getattr(exc, "reason", "chain_row_invalid"), str(exc)
                    )
                )
    result = {}
    for token, rules in sorted(grouped.items(), key=lambda item: chain_sort_key(item[0])):
        first = rules[0]
        if any(replace(rule, source=first.source) != first for rule in rules):
            rejections.append(
                Rejection(
                    "chain_import",
                    token,
                    "conflicting_chain_representatives",
                    "One token has conflicting structures; all variants excluded",
                )
            )
            continue
        result[token] = replace(first, source=";".join(sorted({rule.source for rule in rules})))
    conventions = {
        rule.p_db_count_includes_vinyl for rule in result.values() if rule.linkage_type == "P"
    }
    if len(conventions) > 1:
        for token, rule in list(result.items()):
            if rule.linkage_type == "P":
                rejections.append(
                    Rejection(
                        "chain_import",
                        token,
                        "conflicting_p_db_conventions",
                        "One table must use one declared P DB-count convention",
                    )
                )
                del result[token]
    return result


def import_p_to_o(
    path: str | Path | None, chains: dict[str, ChainRule], rejections: list[Rejection]
):
    if path is None:
        return {}
    result, conflicts = {}, set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"p_token", "o_token"}.issubset(reader.fieldnames or []):
            raise ValueError("Mapping columns must be p_token, o_token")
        for row in reader:
            source, target = row["p_token"], row["o_token"]
            try:
                if source not in chains or target not in chains:
                    raise ValueError("Mapping references an unavailable validated chain")
                left, right = chains[source], chains[target]
                if (
                    left.linkage_type != "P"
                    or right.linkage_type != "O"
                    or left.carbon_count != right.carbon_count
                ):
                    raise ValueError("Mapping must explicitly connect same-carbon P to O")

                # Verify chemistry, never derive a mapping by adding one to the DB annotation.
                def composition(rule):
                    mol, _, _ = validate_fragment(rule.representative_fragment_smiles)
                    return (
                        sum(a.GetSymbol() == "C" for a in mol.GetAtoms()),
                        sum(a.GetTotalNumHs() for a in mol.GetAtoms()),
                    )

                if composition(left) != composition(right):
                    raise ValueError("P/O fragment elemental compositions differ")
                if source in result and result[source] != target:
                    conflicts.add(source)
                    raise ValueError("Conflicting external P-to-O mappings")
                result[source] = target
            except ValueError as exc:
                rejections.append(
                    Rejection("mapping_import", source, "p_to_o_mapping_invalid", str(exc))
                )
    return {source: target for source, target in result.items() if source not in conflicts}
