import math
from pathlib import Path

from config import load_config
from utils import bits_per_char

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_bits_per_char_known_value():
    # nll of ln(2) nats per character is exactly 1 bit per character
    assert abs(bits_per_char(math.log(2) * 100, 100) - 1.0) < 1e-9


def test_rung_inheritance_chain():
    cfg = load_config(CONFIGS / "e4_decorrelation.yaml")
    assert cfg["model"]["n_regions"] == 4
    assert cfg["model"]["use_messages"] is True
    assert cfg["training"]["lambda_c"] == 0.01
    assert cfg["training"]["lambda_r"] == 0.3


def test_e0_is_plain_ensemble():
    cfg = load_config(CONFIGS / "e0_plain_ensemble.yaml")
    m = cfg["model"]
    assert m["n_passes"] == 1 and m["n_regions"] == 1
    assert m["use_weights"] is False and m["use_messages"] is False
    assert cfg["training"]["lambda_c"] == 0.0


def test_tiny_overlay_keeps_rung_flags():
    cfg = load_config(CONFIGS / "e0_plain_ensemble.yaml", "tiny.yaml")
    assert cfg["model"]["n_regions"] == 1
    assert cfg["model"]["d"] == 64
