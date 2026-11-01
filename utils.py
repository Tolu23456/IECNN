"""Small shared helpers."""
from __future__ import annotations

import math
import random


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (if installed) for reproducibility."""
    import os

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(int(os.environ.get("IECNN_TORCH_THREADS", "0")) or torch.get_num_threads())
    except ImportError:
        pass


def set_torch_threads(n: int) -> None:
    """Set the number of PyTorch intra-op threads; n <= 0 leaves it alone.

    Measured on CPU runners (EPYC 7763): 2 threads beats 4 because only 2
    physical cores exist (DECISIONS.md).
    """
    if n and n > 0:
        import torch
        torch.set_num_threads(n)


def get_device() -> str:
    """Return "cuda" if a GPU is available, else "cpu"."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def count_parameters(model) -> int:
    """Number of trainable parameters of a torch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def bits_per_char(total_nll_nats: float, num_chars: int) -> float:
    """bpc = total negative log-likelihood (nats) / (ln 2 * characters).

    Count characters of the ORIGINAL text, never tokens (spec 07, section 7).
    """
    if num_chars <= 0:
        raise ValueError("num_chars must be positive")
    return total_nll_nats / (math.log(2) * num_chars)
