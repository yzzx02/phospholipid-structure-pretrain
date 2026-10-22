import numpy as np
import pytest
import torch

from structure_pretrain.reported_headgroups.formal_template_data import (
    CoreData,
    epoch_batches,
    update_selection,
)


@pytest.mark.parametrize("size", [16, 32])
def test_epoch_partition_and_resume_cursor(size):
    indices = np.arange(107)
    lengths = (indices * 17) % 83 + 1
    batches = epoch_batches(indices, lengths, size, 42, 3, 2)
    assert sorted(np.concatenate(batches).tolist()) == indices.tolist()
    assert len({int(i) for b in batches for i in b}) == len(indices)
    assert all(0 < len(b) <= size for b in batches)
    restored = epoch_batches(indices, lengths, size, 42, 3, 2)
    assert [b.tolist() for b in batches[2:]] == [b.tolist() for b in restored[2:]]
    assert [b.tolist() for b in batches] != [
        b.tolist() for b in epoch_batches(indices, lengths, size, 42, 4, 2)
    ]


def test_paper_selection_strict_validation_loss_and_five_bad_epochs():
    best, bad, improved = update_selection(float("inf"), 0, 2.0)
    assert improved and best == 2.0 and bad == 0
    for i in range(5):
        best, bad, improved = update_selection(best, bad, 2.0 + i * 0.01)
        assert not improved and bad == i + 1 and best == 2.0
    assert update_selection(best, bad, 1.99) == (1.99, 0, True)
    with pytest.raises(ValueError, match="Nonfinite"):
        update_selection(best, bad, float("nan"))


def test_chainless_occupancy_selects_matching_references_and_packing():
    data = CoreData.__new__(CoreData)
    data.device = torch.device("cpu")
    data.lengths = np.array([4, 5])
    data.refs = torch.tensor([[[0, -1], [1, -1]], [[0, -1], [2, -1]]])
    data.occupancy = torch.tensor([0, 1])
    data.bank_formula = torch.arange(30).reshape(3, 10).float()
    data.polar = torch.ones(2, 10) * 50
    data.bank_fp = torch.arange(2 * 3 * 8).reshape(2, 3, 8).float()
    data.tokens = torch.zeros(2, 5, dtype=torch.long)
    data.full = torch.ones(2, 10)
    data.chains = torch.ones(2, 2, dtype=torch.long)
    batch = data.batch([0, 1], k=2)
    assert batch["template_mask"].tolist() == [[True, False], [True, False]]
    torch.testing.assert_close(
        batch["fingerprints"], torch.stack([data.bank_fp[0, 0], data.bank_fp[1, 2]])
    )
    torch.testing.assert_close(batch["template_formulas"][:, 0], data.bank_formula[[0, 2]])
    assert torch.count_nonzero(batch["template_formulas"][:, 1]) == 0
    assert torch.count_nonzero(batch["delta_formulas"][:, 1]) == 0
    torch.testing.assert_close(
        (batch["template_formulas"] + batch["delta_formulas"])[:, 0], data.polar
    )
