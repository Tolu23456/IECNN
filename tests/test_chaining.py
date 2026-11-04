"""Tolu's CI request (2026-11-04): two tiny chained chunks must CONTINUE the
step counter past chunk 1, and the workflow's artifact restore must extract at
the repo root. Guards against the 2026-11-03 chain-resume bug (the restore
unpacked into checkpoints/, producing checkpoints/checkpoints/..., so every
chained chunk silently re-trained from step 0)."""
from __future__ import annotations

import json
from pathlib import Path

from utils import set_seed


def _tiny_setup(tmp_path: Path):
    """Synthetic 64-base UBM + small corpus files + a tiny GRU config."""
    from UBM import UBM, BaseRecord

    set_seed(0)
    ubm = UBM()
    recs = [BaseRecord(id=len(ubm.bases) + i, bytes=ch.encode())
            for i, ch in enumerate(
                "abcdefghijklmnopqrstuvwxyz_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-")]
    ubm.add_bases(recs)
    vocab_p = tmp_path / "vocab.json"
    ubm.save(vocab_p)

    train_p = tmp_path / "train.txt"
    train_p.write_text("abcdefgh abcdef abcdef " * 1500, encoding="utf-8")
    val_p = tmp_path / "val.txt"
    val_p.write_text("abcdefgh " * 40, encoding="utf-8")

    cfg = {
        "name": "chain_test",
        "model": {"kind": "gru", "d": 16, "gru_hidden": 32},
        "data": {"dir": str(tmp_path), "seq_len": 32, "train_file": str(train_p),
                 "val_file": str(val_p)},
        "ubm": {"vocab_file": str(vocab_p)},
        "training": {"batch_size": 4, "lr": 1e-3, "warmup_steps": 2,
                     "eval_every": 4, "checkpoint_every": 4, "eval_chars": 400,
                     "steps": 8, "csv_dir": str(tmp_path / "results"),
                     "early_stop": {"patience_evals": 0, "wall_hours": 0.0}},
    }
    return cfg


def test_two_chained_chunks_continue_step_counter(tmp_path):
    """Chunk 1 trains steps 0-3; chunk 2 MUST resume and train 4-7. The 11-03
    bug made chunk 2 restart at 0 (steps_trained_this_run would be 8)."""
    from train.train import train_one

    cfg = _tiny_setup(tmp_path)
    out_dir = tmp_path / "ckpt" / "chain_test_seed0"   # nested on purpose

    s1 = train_one(cfg, 0, out_dir, max_steps=4)
    assert s1["steps_final"] == 4 and s1["steps_trained_this_run"] == 4

    s2 = train_one(cfg, 0, out_dir, max_steps=8, resume=True)
    assert s2["steps_final"] == 8
    assert s2["steps_trained_this_run"] == 4, (
        f"chunk 2 re-trained from step 0! trained {s2['steps_trained_this_run']}")

    # the CSV is one continuous log: single header, chunk-1 eval rows (3,3)
    # then chunk-2 rows (4 and 7,7). A from-step-0 restart would append a
    # second header and log step 3 again (4x) — this catches that.
    csv_p = tmp_path / "results" / "train_chain_test_seed0.csv"
    lines = [l for l in csv_p.read_text().splitlines() if l.strip()]
    assert sum(l.startswith("step,") for l in lines) == 1
    rows = [l.split(",")[0] for l in lines[1:]]
    assert rows == ["3", "3", "4", "7", "7"], rows

    # checkpoint continues: last saved step is 7
    import torch
    ck = torch.load(out_dir / "checkpoint.pt", weights_only=False)
    assert int(ck["step"]) == 7


def test_train_workflow_restores_artifact_at_repo_root():
    """Lint the chain's restore step: the artifact zip already carries the
    checkpoints/ prefix, so it must be extracted at '.', never into
    checkpoints/ (that created the checkpoints/checkpoints bug)."""
    import yaml

    wf_path = Path(".github/workflows/train.yml")
    text = wf_path.read_text(encoding="utf-8")
    assert "extractall('checkpoints')" not in text, \
        "restore step extracts INTO checkpoints/ — the 11-03 bug is back"
    assert "extractall('.')" in text, \
        "restore step must extract the artifact zip at the repo root"
    assert "-o checkpoints/resume.zip" not in text, \
        "resume zip must not be written inside checkpoints/"

    wf = yaml.safe_load(text)      # also proves the workflow stays valid YAML
    names = [s.get("name", "") for s in wf["jobs"]["train"]["steps"]]
    assert any("Restore previous chunk" in n for n in names), names
    assert json.dumps(wf["permissions"]) != "{}"
