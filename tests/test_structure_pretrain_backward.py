import torch

from structure_pretrain.models.structure_pretrain_model import StructurePretrainModel
from structure_pretrain.training.losses import pretraining_loss


def test_backward_and_optimizer_updates_each_module(model_inputs):
    config, dataset, stats, batch = model_inputs
    model = StructurePretrainModel(config.model, len(dataset.chain_vocab), dataset.tokenizer, stats)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    watched = {name: next(module.parameters()) for name, module in model.named_children()}
    previous = {name: value.detach().clone() for name, value in watched.items()}
    output = model(batch)
    loss = pretraining_loss(output, batch, 0, config.train)["loss"]
    loss.backward()
    for module in model.children():
        grads = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert grads and all(torch.isfinite(gradient).all() for gradient in grads)
        assert any(gradient.abs().sum() > 0 for gradient in grads)
    optimizer.step()
    assert all(not torch.equal(value, previous[name]) for name, value in watched.items())
