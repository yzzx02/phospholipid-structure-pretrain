from rdkit import Chem

from .canonicalize import parse_molecule, supported_atoms
from .schemas import LibraryValidationError


def attachment(molecule, expected_map: int | None = None):
    dummy = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy) != 1 or dummy[0].GetDegree() != 1:
        raise LibraryValidationError(
            "attachment_invalid", "Require exactly one degree-1 attachment dummy"
        )
    atom = dummy[0]
    if atom.GetAtomMapNum() < 1 or (
        expected_map is not None and atom.GetAtomMapNum() != expected_map
    ):
        raise LibraryValidationError(
            "attachment_invalid", "Mapped dummy must match attachment_atom_map"
        )
    bond = atom.GetBonds()[0]
    if bond.GetBondType() != Chem.BondType.SINGLE:
        raise LibraryValidationError("attachment_invalid", "Attachment bond must be single")
    return atom.GetIdx(), atom.GetNeighbors()[0].GetIdx()


def validate_fragment(smiles: str, expected_map: int | None = None):
    molecule = parse_molecule(smiles)
    supported_atoms(molecule, allow_dummy=True)
    dummy, neighbor = attachment(molecule, expected_map)
    return molecule, dummy, neighbor


def validate_polar_fragment(smiles: str, atom_map: int):
    molecule, _, neighbor = validate_fragment(smiles, atom_map)
    if molecule.GetAtomWithIdx(neighbor).GetSymbol() != "P":
        raise LibraryValidationError(
            "polar_boundary_invalid", "Glycerol supplies O; polar dummy must attach to P"
        )
    if sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()) != 0:
        raise LibraryValidationError(
            "nonneutral_structure", "Polar fragment must permit neutral complete assembly"
        )
    # Reject an intact acyl-glycerophospholipid supplied as a polar fragment.
    # This is a validation guard, never a cutting rule or a PubChem search query.
    patterns = (
        "[P]-O-[CH2]-[CH](-O-C(=O))-[CH2]-O",
        "[P]-O-[CH2]-[CH](-O)-[CH2]-O-C(=O)",
    )
    if any(molecule.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in patterns):
        raise LibraryValidationError(
            "intact_lipid_as_polar", "Supply the polar boundary fragment, not a complete lipid"
        )
    return molecule
