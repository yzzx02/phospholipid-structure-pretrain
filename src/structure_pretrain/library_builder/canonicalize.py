from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from .schemas import ELEMENTS, LibraryValidationError


def parse_molecule(smiles: str):
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not molecule.GetNumAtoms():
        raise LibraryValidationError("invalid_valence", "Invalid or unsanitizable SMILES")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise LibraryValidationError("disconnected_structure", "Require one connected structure")
    return molecule


def supported_atoms(molecule, allow_dummy: bool = False):
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 0 and allow_dummy:
            continue
        if atom.GetSymbol() not in ELEMENTS:
            raise LibraryValidationError(
                "unsupported_element", f"Unsupported element {atom.GetSymbol()}"
            )
        if atom.GetIsotope():
            raise LibraryValidationError(
                "unsupported_isotope", "Isotope labels are outside this schema"
            )
        if atom.GetNumRadicalElectrons():
            raise LibraryValidationError("unsupported_radical", "Require closed-shell structures")


def nonisomeric_smiles(molecule) -> str:
    molecule = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(molecule)
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 0:
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def neutral_structure(molecule) -> tuple[str, str, str, dict[str, int]]:
    with rdBase.BlockLogs():
        try:
            Chem.SanitizeMol(molecule)
        except Exception as exc:
            raise LibraryValidationError("invalid_valence", str(exc)) from exc
    supported_atoms(molecule)
    if sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()) != 0:
        raise LibraryValidationError(
            "nonneutral_structure", "Complete molecule must have total charge 0"
        )
    smiles = nonisomeric_smiles(molecule)
    normalized = parse_molecule(smiles)
    with rdBase.BlockLogs():
        key = Chem.MolToInchiKey(normalized)
    if not key:
        raise LibraryValidationError("inchikey_failed", "InChIKey generation failed")
    formula = rdMolDescriptors.CalcMolFormula(normalized)
    counts = {element: 0 for element in ELEMENTS}
    for atom in Chem.AddHs(normalized).GetAtoms():
        counts[atom.GetSymbol()] += 1
    return smiles, formula, key, counts
