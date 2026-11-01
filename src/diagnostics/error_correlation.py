"""Diagnostic: Weighted pairwise error correlation between dots, over training."""
from __future__ import annotations


def run(checkpoint_path: str, config_path: str, out_dir: str) -> dict:
    """Load a checkpoint, compute the diagnostic, write a table to out_dir, return a summary."""
    raise NotImplementedError
