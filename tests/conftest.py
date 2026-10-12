import pytest
import torch

from structure_pretrain.config import ExperimentConfig
from structure_pretrain.data.batching import collate_structure_examples
from structure_pretrain.data.dummy_structure_dataset import DummyStructureDataset
from structure_pretrain.utils.formula import FormulaStandardizer


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture(params=["graph", "fingerprint"])
def model_inputs(request):
    config = ExperimentConfig()
    config.model.encoder = request.param
    dataset = DummyStructureDataset(request.param)
    statistics = FormulaStandardizer.fit(dataset.examples[:6])
    batch = collate_structure_examples(dataset.examples[:4])
    return config, dataset, statistics, batch
