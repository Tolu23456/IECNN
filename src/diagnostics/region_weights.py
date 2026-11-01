"""Diagnostic: Mean region weight by token type; loss of each region alone; loss by pass."""
from __future__ import annotations


def run(checkpoint_path: str, config_path: str, out_dir: str) -> dict:
    """Load a checkpoint, compute the diagnostic, write a table to out_dir, return a summary."""
    raise NotImplementedError
