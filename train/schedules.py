"""Schedules (spec 06, section 3)."""
from __future__ import annotations


def lr_at(step, total_steps, warmup, peak_lr):
    """Linear warmup then cosine decay."""
    raise NotImplementedError


def correlation_lambda(step, ramp_steps, final_value):
    """Ramp from 0 to final_value over ramp_steps."""
    raise NotImplementedError


def use_hard_batch(step, total_steps, start_fraction, batch_fraction, rng) -> bool:
    """True for a fraction of batches after start_fraction of training."""
    raise NotImplementedError
