"""Fail clearly if the environment cannot execute HIP attention/scatter forward and backward."""

import json

import torch
from torch.nn import functional as F


def main():
    info = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu_available": torch.cuda.is_available(),
    }
    print(json.dumps(info, indent=2))
    if not torch.version.hip or not torch.cuda.is_available():
        raise SystemExit("Install AMD's matching WSL ROCm PyTorch wheels before training")
    device = torch.device("cuda:0")  # ROCm intentionally exposes PyTorch's cuda API.
    x = torch.randn(2, 4, 10, 32, device=device, requires_grad=True)
    attention = F.scaled_dot_product_attention(x, x, x)
    scatter = torch.zeros(2, 4, 32, device=device).index_add(
        0, torch.tensor([0, 1], device=device), x.sum(2)
    )
    loss = attention.square().mean() + scatter.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
        raise SystemExit("Nonfinite ROCm attention/scatter result")
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "forward_backward": "passed"}))


if __name__ == "__main__":
    main()
