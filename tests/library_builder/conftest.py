from pathlib import Path

import pytest

from structure_pretrain.library_builder.chain_library import import_chains
from structure_pretrain.library_builder.polar_library import import_polars
from structure_pretrain.library_builder.schemas import load_build_config

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "library_builder"


@pytest.fixture
def inputs():
    errors = []
    chains = import_chains(EXAMPLES / "chains.tsv", errors)
    polars = import_polars(EXAMPLES / "polar_moieties.tsv", errors)
    assert not errors
    config = load_build_config(EXAMPLES / "demo_build.yaml")
    return chains, {polar.polar_id: polar for polar in polars}, config
