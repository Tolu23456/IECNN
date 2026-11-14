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


# ---- run metadata fallback (timeout-killed chunks, 2026-11-14) -------------
def test_load_run_meta_reads_real_files(tmp_path):
    """Both files present: the on-disk values win, no fallback involved."""
    from utils import load_run_meta
    d = tmp_path / "run_seed0"
    d.mkdir()
    (d / "summary.json").write_text('{"best_step": 1680, "stop_kind": "wall_cap"}')
    (d / "config_used.json").write_text('{"name": "run", "model": {"kind": "iecnn"}}')
    summary, config = load_run_meta(d, {"best_step": 1})
    assert summary["best_step"] == 1680 and config["name"] == "run"


def test_load_run_meta_falls_back_to_checkpoint_fields(tmp_path):
    """summary.json + config_used.json both missing (timeout-killed chunk):
    summary is rebuilt from ck fields, config from the repo config file, and
    the stop_kind says honestly that it is a fallback."""
    from utils import load_run_meta
    d = tmp_path / "run_seed0"
    d.mkdir()
    ck = {"best_step": 1680, "best_val": 2.27685, "step": 1700}
    summary, config = load_run_meta(d, ck, str(CONFIGS / "base.yaml"))
    assert summary["best_step"] == 1680
    assert summary["best_val"] == 2.27685
    assert "fallback" in summary["stop_kind"]
    assert config["model"]["kind"] == "iecnn"   # loaded via load_config
    assert summary["steps_target"] == config["training"]["steps"]


def test_load_run_meta_summary_missing_but_config_present(tmp_path):
    """config_used.json exists, summary.json doesn't: summary rebuilt from
    ck + config (the pre-2026-11-14 killed-chunk artifact shape)."""
    from utils import load_run_meta
    d = tmp_path / "run_seed0"
    d.mkdir()
    (d / "config_used.json").write_text(
        '{"name": "run", "model": {"kind": "gru"}, "training": {"steps": 500}}')
    summary, config = load_run_meta(d, {"best_step": 300, "step": 320}, None)
    assert summary["best_step"] == 300 and summary["steps_target"] == 500


def test_load_run_meta_loud_failure_without_any_source(tmp_path):
    """No files and no fallback config = loud FileNotFoundError, not a
    silent wrong-model evaluation."""
    from utils import load_run_meta
    import pytest
    with pytest.raises(FileNotFoundError):
        load_run_meta(tmp_path, {"best_step": 0}, None)
