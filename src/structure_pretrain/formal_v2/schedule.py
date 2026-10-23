"""An optimizer-update clock; restart never repeats warmup."""

import math


class WarmupCosine:
    def __init__(self, peak_lr, minimum_lr, warmup_steps, total_steps):
        assert 0 < minimum_lr <= peak_lr and 0 < warmup_steps < total_steps
        self.specification = dict(
            peak_lr=float(peak_lr),
            minimum_lr=float(minimum_lr),
            warmup_steps=int(warmup_steps),
            total_steps=int(total_steps),
        )
        self.updates_completed = 0
        self.prepared = False

    def value(self, update):
        s = self.specification
        if not 1 <= update <= s["total_steps"]:
            raise ValueError("Schedule optimizer step outside frozen budget")
        if update <= s["warmup_steps"]:
            return s["peak_lr"] * update / s["warmup_steps"]
        position = (update - s["warmup_steps"]) / (s["total_steps"] - s["warmup_steps"])
        return s["minimum_lr"] + 0.5 * (s["peak_lr"] - s["minimum_lr"]) * (
            1 + math.cos(math.pi * position)
        )

    def before_optimizer_step(self, optimizer, completed_steps):
        if self.prepared or completed_steps != self.updates_completed:
            raise ValueError("Scheduler and optimizer clocks differ")
        lr = self.value(self.updates_completed + 1)
        for group in optimizer.param_groups:
            group["lr"] = lr
        self.prepared = True
        return lr

    def after_optimizer_step(self):
        if not self.prepared:
            raise ValueError("Cannot schedule without one optimizer update")
        self.updates_completed += 1
        self.prepared = False

    def state_dict(self):
        if self.prepared:
            raise ValueError("Checkpoint must be between complete optimizer updates")
        return {"specification": self.specification, "updates_completed": self.updates_completed}

    def load_state_dict(self, state):
        if state["specification"] != self.specification:
            raise ValueError("Frozen scheduler specification changed")
        if not 0 <= state["updates_completed"] <= self.specification["total_steps"]:
            raise ValueError("Invalid scheduler cursor")
        self.updates_completed = state["updates_completed"]
        self.prepared = False
