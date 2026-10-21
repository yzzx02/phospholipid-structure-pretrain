"""Expose unusual database phosphorus bonding without repairing the source graph."""

from rdkit import Chem


def phosphorus_review_flags(smiles):
    mol = Chem.MolFromSmiles(smiles)
    patterns = [
        ("[P]-[O]-[N]", "源图含 P-O-N 共价连接；需核实是否为铵盐表示错误或真实特殊结构"),
        ("[P]-[O]-[O]", "源图含 P-O-O 过氧连接，需核实原始结构记录"),
        ("[P]=[C]", "含 P=C 连接，属于非常规磷脂类似物，单列审核"),
        ("[P]=[S]", "含 P=S 硫代磷酸结构，单列审核其模板适用性"),
        ("[P]-[N]", "含 P-N 磷酰胺连接，单列审核其模板适用性"),
        ("[P]-[S]", "含 P-S 连接，单列审核其模板适用性"),
    ]
    return [
        text for pattern, text in patterns if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))
    ]
