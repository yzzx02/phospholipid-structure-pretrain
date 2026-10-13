from rdkit import Chem

from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.chain_combinations import make_combination


def test_oa_ether_at_terminal_representative_site(inputs):
    chains, polars, _ = inputs
    combo = make_combination(("A-18:1", "O-16:0"), chains)
    assert combo.chain_tokens == ("O-16:0", "A-18:1")
    row = assemble(chains, combo, polars["demo_PC"])
    mol = Chem.MolFromSmiles(row["canonical_smiles"])
    # From phosphate: CH2-CH(ester)-CH2-ether, independently of SMILES atom order.
    assert mol.HasSubstructMatch(Chem.MolFromSmarts("[P]-O-[CH2]-[CH](-O-C(=O))-[CH2]-O-[CH2]"))
    assert row["representative_sn_assignment"] is True
