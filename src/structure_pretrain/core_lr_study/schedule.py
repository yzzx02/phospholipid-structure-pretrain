"""Restart-safe constant, no-warmup cosine and warmup-cosine update clocks."""

import math


class Schedule:
    def __init__(self, kind, peak_lr, minimum_lr, warmup_steps, total_steps):
        if kind not in ("constant", "cosine", "warmup_cosine"):
            raise ValueError("Unknown schedule")
        assert 0 < minimum_lr <= peak_lr and 0 <= warmup_steps < total_steps
        assert (kind == "warmup_cosine") == (warmup_steps > 0)
        self.kind = kind
        self.specification = dict(
            peak_lr=float(peak_lr),
            minimum_lr=float(minimum_lr),
            warmup_steps=int(warmup_steps),
            total_steps=int(total_steps),
        )
        self.updates_completed, self.prepared = 0, False

    def value(self, update):
        s = self.specification
        if not 1 <= update <= s["total_steps"]:
            raise ValueError("Schedule update outside planned80-epoch cycle")
        if self.kind == "constant":
            return s["peak_lr"]
        warm = s["warmup_steps"]
        if warm and update <= warm:
            return s["peak_lr"] * update / warm
        # Existing E3 warmup/cosine algebra is preserved exactly. Without warmup,
        # first actual update is peak and final planned update is the minimum.
        phase = (
            (update - warm) / (s["total_steps"] - warm)
            if warm
            else (update - 1) / (s["total_steps"] - 1)
        )
        return s["minimum_lr"] + 0.5 * (s["peak_lr"] - s["minimum_lr"]) * (
            1 + math.cos(math.pi * phase)
        )

    def before_optimizer_step(self, optimizer, completed_steps):
        if self.prepared or completed_steps != self.updates_completed:
            raise ValueError("Optimizer/scheduler cursor mismatch")
        value = self.value(completed_steps + 1)
        for group in optimizer.param_groups:
            group["lr"] = value
        self.prepared = True
        return value

    def after_optimizer_step(self):
        if not self.prepared:
            raise ValueError("No prepared optimizer update")
        self.updates_completed += 1
        self.prepared = False

    def state_dict(self):
        if self.prepared:
            raise ValueError("Save only between complete updates")
        return {
            "kind": self.kind,
            "specification": self.specification,
            "updates_completed": self.updates_completed,
        }

    def load_state_dict(self, value):
        kind = value.get("kind", "warmup_cosine")
        if kind != self.kind or value["specification"] != self.specification:
            raise ValueError("Frozen schedule changed")
        assert 0 <= value["updates_completed"] <= self.specification["total_steps"]
        self.updates_completed, self.prepared = int(value["updates_completed"]), False


def stop_for_patience(epoch, bad_epochs, minimum_epochs=50, patience=5):
    return epoch >= minimum_epochs and bad_epochs >= patience
