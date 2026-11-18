# IECNN status — 2026-11-19 (autonomous overnight block)

One-page state of the campaign. The evidence ledger is `DECISIONS.md` (every row
is a measured batch, with the in-session transformer anchor); this file is the
map, not the evidence.

## The objective

Beat the best small transformer on **both** currencies — bpc at equal wall
clock **and** seconds per step — while every mechanism pays for its own compute.
Runs are 180 s/arm so an idea can be killed in a few minutes; nothing scales up
until it survives that.

## Protocol (never vary these inside a comparison)

- Kaggle T4, eager, **180 s/arm**, B=32, T=256 (8192 tok/step), enwik8 60 MB
  train / 5 MB val, `data_fp a1c291f627/6d6c2819c2`, val slice 160 windows
  (stride 122).
- Every session carries its **own** `tf-swa` anchor; cross-session absolute bpc
  is not comparable.
- Report `bpc_final`, `bpc@eqtok` (interpolated at the common token count),
  `bpc_train`, the gap, tokens and `s/step`.
- A claim needs **two seeds** or a margin above the measured noise. The
  transformer's own seed spread is **0.0238**, which is larger than several
  single-session "wins" in the early ledger.
- 2-vCPU toy runs are valid for *speed* and for mechanisms that subtract
  capacity; they are invalid for added fitting power.

## Current frontier (d128 family)

| arm | params | s/step | bpc (in-session tf) | verdict |
|---|---|---|---|---|
| tf-swa | 408k | 0.0245–0.0263 | 2.061–2.092 | anchor |
| d128 ×4 | 559k | **0.0212** | 2.059–2.068 | 16% faster, ≈tf on bpc |
| **d128 ×5** | 658k | 0.0235–0.0255 | **2.047–2.057** | **replicated win** |
| d128 ×6 | 757k | 0.0281 | 2.052 | bpc yes, speed no |

Paired by seed, `tf − x5` = −0.013 / −0.045 (mean −0.029, clears the noise);
`tf − x4` = −0.009 / −0.019 (mean −0.014, inside the noise). So **x5 is the
replicated claim; x4 is a speed claim, not a bpc claim.**

Accuracy is not where the win is: tf top-1 0.593 / top-5 0.853 / bpc 2.018 vs
d128-x5 0.592 / 0.852 / bpc 1.999. Same guess, better tail — the gain is
distributional, which is exactly what a compressor cashes in.

## What the dots turned out to be

The committee was **not** redundant. Measured through the *trained* path
(one-hot pool, not the untrained per-dot logits that the first audit wrongly
used):

| dot | reads back (d192) | reads back (d128) | bpc alone | gate weight |
|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 3.28 | 0.41 |
| 1 | 2.96 | 0.77 | 6.99 | 0.35 |
| 2 | 5.87 | 2.81 | 7.21 | 0.24 |

- **The dots self-organise into time horizons** (0 / ~3 / ~6 tokens back at
  d192), reading with JS 0.20–0.27 nats and agreeing on only 6–10% of
  predictions. A multi-scale receptive field nobody designed.
- **The ladder is rationed by downstream capacity**: at d128 it compresses to
  {0, 0.8, 2.8} and two dots end up nearly identical reads (JS 0.015).
- **The gate tracks quality at d192** (0.385 weight on the best dot vs 0.333
  uniform, argmax = best 58.6% vs 33.3% chance) and is **anti-informative at
  d128** (0.334, 26.6%) — which is why `gate_agree` exists.
- Drop-one: the short-range dot is load-bearing (+4.9 bpc without it); the far
  dot adds 0.28. Horizons are seed-stable, their *labelling* permutes.

## Kept

`global_decay`, `read_ln` + `head_ln`, block depth, d128 stream, three dots,
`global_read`, LUG cell, B=32.

## Closed (measured, not argued)

- **Block sharing** (+0.108 / +0.053), `global_decay_k`, `read_heads=2`, N=2,
  N=4 on a contiguous span-16 window.
- **Geometric/quad bands** (+0.049 at reach 32, +0.079 at 64): dots retreat to
  the nearest slots, so contiguity is load-bearing and the horizons are a
  preference, not a window ceiling.
- **Batch size as a throughput lever**: doubling B doubles the step (1.90×) for
  +5–8% tokens; `s/1k tokens` is flat at 0.0030 in every arm. The step is
  FLOP-bound. B=32 fixed.
- **`pool_first`** (+0.006 bpc for 6% speed): the per-dot output projection does
  real work. The old ledger label "exact prune" was wrong and is corrected.
- **`w/d = 3` as a sufficient statistic**: at ratio 3.0, d96-w288 and d112-w336
  still lose to d128-w384. Ratio ~3 is a **lower bound**; absolute width carries
  quality.
- GRU cell (better per token, 16% slower per step → LUG wins on wall clock).
- `share_cell`, untied head, cell LayerNorm, d_s 64, `query_from_input`, `gla`,
  `ngram_bias`, `head_rank48`, `id_lags`, `head_skip`.

## Open at the time of writing

- `read3` — span 12/16, N=4 on span 16, and `gate_agree` at d128-x5 and d192-x4.
  Tests whether the compressed ladder is a window limit or a capacity limit, and
  whether a gate that can see the committee recovers the d192 behaviour.
- `ratio1` — expansion ratio **above** 3 at fixed d (w320/448/512). Every prior
  point was on the low side of 3.
- `cmp1` — eager vs `torch.compile` in one session (the first attempt died on a
  harness bug: `compile` was forwarded into `run()` and raised after the first
  arm had trained; fixed, and a failing arm no longer kills the session).
- `long1` — the same three arms at **900 s/arm**, to check the ranking survives
  5× the budget before anything is scaled up.
- `db1` — `dist_bias` (24 parameters, zero compute) where the failure actually
  is, plus the cheap end of the frontier (x3 at d128).

## Next if the numbers land as expected

1. If w > 3d pays, take the constant-compute line downwards: x4/w448 and
   x3/w512 are ~7% and ~20% cheaper per step at equal nominal block capacity.
   That is the only route to "massively faster" rather than "16% faster".
2. If span 16 spreads the ladder and helps, the read is under-provisioned and
   the window becomes an arm knob; if not, 8 contiguous slots are right.
3. Then scale: the whole point of the small-scale protocol is that only
   mechanisms that paid here get to consume real compute.
