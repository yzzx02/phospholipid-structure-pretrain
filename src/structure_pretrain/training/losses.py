from torch.nn import functional as F

from structure_pretrain.config import TrainConfig


def pretraining_loss(output: dict, batch: dict, pad_id: int, config: TrainConfig):
    arch = F.cross_entropy(output["architecture_logits"], batch["target_architecture"])
    chain = F.cross_entropy(
        output["chain_logits"].flatten(0, 1), batch["target_chain_tokens"].flatten()
    )
    # BOS is decoder input, EOS is a supervised label; ignore PAD only.
    head = F.cross_entropy(
        output["headgroup_logits"].flatten(0, 1),
        batch["target_headgroup_token_ids"][:, 1:].reshape(-1),
        ignore_index=pad_id,
    )
    return {
        "loss": config.lambda_arch * arch + config.lambda_chain * chain + config.lambda_head * head,
        "architecture_loss": arch,
        "chain_loss": chain,
        "headgroup_loss": head,
    }


def annotation_metrics(output: dict, batch: dict, pad_id: int):
    chains = output["chain_logits"].argmax(-1).eq(batch["target_chain_tokens"])
    labels = batch["target_headgroup_token_ids"][:, 1:]
    valid = labels.ne(pad_id)
    correct = output["headgroup_logits"].argmax(-1).eq(labels)
    return {
        "architecture_accuracy": output["architecture_logits"]
        .argmax(-1)
        .eq(batch["target_architecture"])
        .float()
        .mean(),
        "chain_slot_accuracy": chains.float().mean(),
        "chain_exact_accuracy": chains.all(-1).float().mean(),
        "headgroup_token_accuracy": correct[valid].float().mean(),
    }
