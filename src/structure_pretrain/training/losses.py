from torch.nn import functional as F

from structure_pretrain.config import TrainConfig


def pretraining_loss(output: dict, batch: dict, pad_id: int, config: TrainConfig):
    chain = F.cross_entropy(
        output["chain_logits"].flatten(0, 1), batch["target_chain_tokens"].flatten()
    )
    # BOS is decoder input, EOS is a supervised label; ignore PAD only.
    polar = F.cross_entropy(
        output["polar_logits"].flatten(0, 1),
        batch["target_polar_token_ids"][:, 1:].reshape(-1),
        ignore_index=pad_id,
    )
    return {
        "loss": config.lambda_chain * chain + config.lambda_polar * polar,
        "chain_loss": chain,
        "polar_loss": polar,
    }


def annotation_metrics(output: dict, batch: dict, pad_id: int):
    chains = output["chain_logits"].argmax(-1).eq(batch["target_chain_tokens"])
    labels = batch["target_polar_token_ids"][:, 1:]
    valid = labels.ne(pad_id)
    correct = output["polar_logits"].argmax(-1).eq(labels)
    return {
        "chain_slot_accuracy": chains.float().mean(),
        "chain_exact_accuracy": chains.all(-1).float().mean(),
        "polar_token_accuracy": correct[valid].float().mean(),
    }
