import hashlib

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_id(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms():
        prefix = "murcko_"
        key = Chem.MolToSmiles(scaffold, isomericSmiles=False)
    else:
        # Acyclic graphs have no Murcko core; hash their whole connectivity instead.
        prefix = "acyclic_"
        key = Chem.MolToSmiles(molecule, isomericSmiles=False)
    return prefix + hashlib.sha256(key.encode()).hexdigest()
