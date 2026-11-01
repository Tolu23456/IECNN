"""Schedules (spec 06, section 3)."""
from __future__ import annotations

import math
import random


def lr_at(step: int, total_steps: int, warmup: int, peak_lr: float) -> float:
    """Linear warmup then cosine decay to ~0."""
    if step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def temperature_at(step: int, total_steps: int, warmup: int,
                   tau_max: float, tau_min: float) -> float:
    """Cosine anneal of tau_0 from tau_max to tau_min (spec 04, section 6)."""
    if step < warmup:
        return tau_max
    progress = (step - warmup) / max(1, total_steps - warmup)
    return tau_min + 0.5 * (tau_max - tau_min) * (1.0 + math.cos(math.pi * min(1.0, progress)))


def weight_floor_at(step: int, total_steps: int, base_floor: float) -> float:
    """Weight floor is active until the final 10% of training, then set to 0."""
    if step >= 0.9 * total_steps:
        return 0.0
    return base_floor


def correlation_lambda(step: int, ramp_steps: int, final_value: float) -> float:
    """Ramp from 0 to final_value over ramp_steps."""
    if step >= ramp_steps:
        return final_value
    return final_value * (step / max(1, ramp_steps))


def weight_entropy_lambda(step: int, total_steps: int, start_value: float) -> float:
    """Weight-entropy bonus decays to zero by 20% of training (spec 06, 2.6)."""
    cut = 0.2 * total_steps
    if step >= cut:
        return 0.0
    return start_value * (1.0 - step / cut)


def use_hard_batch(step: int, total_steps: int, start_fraction: float,
                   batch_fraction: float, rng: random.Random) -> bool:
    """True for a fraction of batches after start_fraction of training."""
    if step < start_fraction * total_steps or batch_fraction <= 0:
        return False
    return rng.random() < batch_fraction
