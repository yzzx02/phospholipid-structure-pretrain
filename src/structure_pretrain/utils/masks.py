import torch


def drop_templates(mask: torch.Tensor, probability: float, training: bool) -> torch.Tensor:
    """True means valid. Never turn padding into data; retain at least one per target."""
    if mask.dtype != torch.bool or not mask.any(-1).all():
        raise ValueError("Every sample must have at least one valid template")
    if not 0 <= probability <= 1:
        raise ValueError("Template dropout probability must be in [0,1]")
    if not training or probability == 0:
        return mask.clone()
    kept = mask & (torch.rand(mask.shape, device=mask.device) >= probability)
    empty_rows = ~kept.any(-1)
    first_valid = mask.long().argmax(-1)
    rows = torch.arange(mask.shape[0], device=mask.device)[empty_rows]
    kept[rows, first_valid[empty_rows]] = True
    return kept
