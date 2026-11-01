"""Training loop: configs, seeds, CSV logging, checkpoints, resume."""
from __future__ import annotations


def train(config: dict, seed: int, out_dir: str, resume: bool = False) -> dict:
    """Train one model for one seed. Returns final metrics."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("TODO(agent): add argparse: --config, --overlay, --seed, --resume")
