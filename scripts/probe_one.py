"""One throughput config per process (thread pools do not leak between runs).

  ARCH=iecnn ARM_B=8 ARM_THREADS=1 python3 scripts/probe_one.py

Prints one machine-readable line:
  PROBE arch=<a> B=<b> threads=<t> s_step=<..> tok_s=<..>
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/content/IECNN")

import torch  # noqa: E402

from scripts.throughput import iecnn_cfg, tf_cfg, timed  # noqa: E402
from models.baselines import build_transformer_baseline  # noqa: E402
from models.iecnn import build_iecnn  # noqa: E402

arch = os.environ.get("ARCH", "iecnn")
b = int(os.environ.get("ARM_B", 8))
nthr = int(os.environ.get("ARM_THREADS", 2))
V = 256

torch.set_num_threads(nthr)
torch.manual_seed(0)
model = (build_iecnn(iecnn_cfg(), V) if arch == "iecnn"
         else build_transformer_baseline(tf_cfg(), V))
model.train()
s_step, tok_s = timed(model, b)
print(f"PROBE arch={arch} B={b} threads={nthr} s_step={s_step:.4f} "
      f"tok_s={tok_s:.0f}", flush=True)
