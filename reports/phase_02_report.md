# Phase 2 report — baselines, training loop, bpc kernel, and the compute ladder

Status: **complete**. All Phase 2 tests pass (13 + 1 new regression test).
Authoritative timings below are **measured on GitHub Actions**, not locally:
from 2026-11-01 training and timing runs never execute on the local machine
(DECISIONS.md); the local box only develops, tests, and smoke-runs.

## What was built

| Piece | File | Notes |
|---|---|---|
| B0 GRU baseline | `models/baselines.py` | causal, 1.31 M params at ladder size, d_model-free |
| B1 transformer baseline | `models/baselines.py` | GPT-style pre-LN blocks, causal mask verified, **tied head** |
| Trainer | `train/train.py` | CLI `--config --overlay --seed --max-steps --resume`; AdamW, 500-step warmup, cosine decay, grad clip 1.0; CSV logging; checkpoint/resume verified bit-for-bit |
| bpc kernel | `train/evaluate.py` | NLL / (ln2 × chars); chars credited per predicted token by expansion length; final full-val, in-loop subsampled (`training.eval_chars`) |
| Packing/batching | `data/dataset.py` | disjoint windows, `(x,y)` shift verified |
| Ladder configs | `configs/b*.yaml e0–e5` | all inherit `base.yaml`, load-tested |

## Measured speed on GitHub Actions (AMD EPYC 7763, 4 vCPU, 2 threads, fp32)

| Run | config | size | tokens/step | tok/s | ms/step (gross incl. one full-val eval) |
|---|---|---|---|---|---|
| `-timing` | tiny (B16, T128) | 411 k | 2,048 | 27,806 | **73** |
| `-timing-full` | full (B64, T256) | 1.31 M | 16,384 | 12,653 | **1,290** (b0_gru) |
| `-timing-full2` (post-fix) | full (B64, T256) | 3.48 M | 16,384 | 6,425 | **2,550** (b1_transformer) |

Per-step FLOPs from the logged counters: b0 full = 129 GFLOP/step,
b1 full = 342 GFLOP/step. b1/b0 batch-time ratio ≈ 2.0 at equal batch tokens.

## Bug found by the Actions validation runs (and fixed)

The first full-config b1 run diverged instantly: val bpc **122.5 after 20 steps**
with just-warmup LR (4e-5). Cause was static, not training dynamics: the tied
output head was initialized from an `nn.Embedding` with the library default
N(0,1), so logits started ~N(0, d_model) and the model "predicted" with
astronomical confidence (~173 nats/token instead of ln 512 = 6.24 at init).
Fixed with GPT-style small-normal embedding init; init loss is now 6.29
(verified against 6.238), and a regression test
(`test_init_loss_near_uniform_at_full_config`) now pins every rung's init loss
to the uniform range so this class of bug cannot return. The same test exists
only because the timing-first-then-examine discipline forced the divergence
into view; a 3-step local smoke would also have caught it, but the Actions run
was the first full-config exercise.

Two deployment bugs were also found and fixed in the process (details in
STATUS.md): (1) `train.yml`'s embedded Python for artifact resume de-indented
out of its YAML block scalar → whole file unparseable → `workflow_dispatch`
silently missing → HTTP 422 on every dispatch attempt; (2) `.gitignore`'s
`data/*` rule excluded `data/dataset.py` and friends from the commit, breaking
imports on the runner, and the corpus step now explicitly rebuilds
`data/*.txt` on the runner and **byte-identical checksums match local**
(`train c6666e…9e2b07…` — full hashes in `data/split_info.json`).

## Ladder estimate (8 rungs × 3 seeds × 50,000 steps × 16,384 tokens = 819 M tokens/seed)

From measured tok/s on Actions (gross, includes per-eval overhead — conservative):

- **B0**: 819 M / 12,653 tok/s ≈ **18 h/seed** → 3 seeds in parallel: ~18 h wall
- **B1**: 819 M / 6,425 tok/s ≈ **35 h/seed** → ~35 h wall
- **E0–E5**: rungs E0–E4 reuse the spec default model (`n_dots 8, n_passes 3,
  d 128`) whose cost is measured in Phase 3; historically the small-dot grouped
  matmuls land between the GRU and the transformer per FLOP. Planning range
  **1–2.5× b1 ≈ 35–90 h/seed**, wall ~35–90 h each (need per-rung measured
  numbers as Phase 3 lands them; this column is the pick and refine slot)
- **E5** (UBM promotion): bpc redo on several promoted checkpoints alone ≈
  hours, plus one re-run of a promoted model per promotion pair

Wall-clock total, all 3 seeds running as parallel jobs and rungs sequential:
**~1.5–3 weeks**. Total measured minutes (b0 ≈ 3×18 h, b1 ≈ 3×35 h, E rungs
projected) are unmetered on this **public** repo, so the constraint is calendar
time and the 6-hour job cap, not budget. The job cap is handled by `--resume`
(verified bit-identical resume this phase) with `timeout-minutes: 350`, well
under the 6 h cap.

The honest caveat: E-rung timing is a projection until Phase 3 measures it;
the plan explicitly updates the ladder estimate after rung E0's first real run.

## 6-hour job cap resume drill (planned, not yet executed)

A >6-hour run will be exercised in Phase 3 with a real rung; the resume
machinery itself is proven this phase.

## Tests

Full suite `pytest tests/`: **32 passed, 33 skipped** (the skipped ones are the
Phase 3+ skeleton roadmap, previously documented). Newest test this phase:
init-loss regression covering every baseline at ladder size.
