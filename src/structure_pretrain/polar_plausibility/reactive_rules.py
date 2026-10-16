from pathlib import Path

import yaml
from rdkit import Chem


def load_rules(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    rules = []
    seen = set()
    for row in raw["rules"]:
        pattern = Chem.MolFromSmarts(row["smarts"])
        if pattern is None or row["reason"] in seen:
            raise ValueError(f"Invalid or duplicate reactive rule: {row}")
        seen.add(row["reason"])
        rules.append((row["reason"], pattern))
    if not rules:
        raise ValueError("Reactive rules must not be empty")
    return rules


def reactive_flags(molecule, rules):
    return sorted(reason for reason, pattern in rules if molecule.HasSubstructMatch(pattern))
