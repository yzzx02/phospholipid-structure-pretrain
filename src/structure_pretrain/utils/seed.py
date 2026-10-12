import lightning as L
import torch


def seed_everything(seed: int, cpu_threads: int = 2) -> None:
    torch.set_num_threads(cpu_threads)
    L.seed_everything(seed, workers=True)
