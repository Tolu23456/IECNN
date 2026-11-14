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


def load_run_meta(ckpt_dir, ck: dict, config_fallback: str | None = None):
    """Load (summary, config) for a checkpoint dir, with honest fallbacks.

    A chunk killed by the 320-min job timeout never reaches the chunk-end
    summary/config writes, so re-score/generate jobs on such artifacts would
    previously die on a missing file. The caller passes its already-loaded
    checkpoint dict as `ck`. Fallback order:
      summary.json  -> partial summary rebuilt from ck (best_step / best_val /
                       step are checkpoint fields; steps_target from config)
      config_used.json -> deterministic load_config() of the repo config file
                       passed as config_fallback (loudly announced)
    Raises FileNotFoundError with a clear message if neither source works.
    """
    import json
    from pathlib import Path

    d = Path(ckpt_dir)
    s_path, c_path = d / "summary.json", d / "config_used.json"
    if s_path.is_file():
        summary = json.loads(s_path.read_text())
    else:
        summary = {}
    if c_path.is_file():
        config = json.loads(c_path.read_text())
    elif config_fallback:
        from config import load_config
        config = load_config(config_fallback)
        print(f"NOTE: config_used.json missing in {d}; rebuilt from "
              f"{config_fallback} (training-time merge)", flush=True)
    else:
        raise FileNotFoundError(
            f"{c_path} missing and no --config-fallback given; the training "
            "chunk was likely killed before any checkpoint metadata write")
    if not s_path.is_file():
        steps_target = int((config.get("training") or {}).get("steps", 1))
        summary = {
            "best_step": ck.get("best_step", ck.get("step", 0)),
            "best_val": ck.get("best_val"),
            "step": ck.get("step"),
            "steps_target": steps_target,
            "stop_kind": "killed_mid_chunk (metadata fallback)",
            "parameters": None,
        }
        print(f"NOTE: summary.json missing in {d}; rebuilt partial summary "
              f"from checkpoint fields (best_step {summary['best_step']})",
              flush=True)
    return summary, config

