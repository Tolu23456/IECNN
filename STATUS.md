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

- Kaggle T4, eager, **180 s/arm** (900 s for the long check), B=32, T=256
  (8192 tok/step), enwik8 60 MB train / 5 MB val,
  `data_fp a1c291f627/6d6c2819c2`, val slice 160 windows (stride 122).
- Every session carries its **own** `tf-swa` anchor; cross-session absolute bpc
  is not comparable.
- Report `bpc_final`, `bpc@eqtok` (interpolated at the common token count),
  `bpc_train`, the gap, tokens and `s/step`.
- A claim needs **two seeds** or a margin above the measured noise. The
  transformer's own seed spread is **0.0238**, which is larger than several
  single-session "wins" in the early ledger.
- 2-vCPU toy runs are valid for *speed* and for mechanisms that subtract
  capacity; they are invalid for added fitting power.

## Current best arms

In-session anchors are given for each row because absolute bpc drifts ±0.02
between sessions.

| arm | params | s/step | bpc | in-session tf | note |
|---|---|---|---|---|---|
| tf-swa | 408k | 0.0245–0.0263 | 2.061–2.092 | — | anchor |
| d128 ×5, w384 | 658k | 0.0235–0.0255 | 2.047–2.057 | 2.061–2.075 | **replicated win** |
| d128 ×4, w384 | 559k | **0.0212** | 2.059–2.068 | 2.068 | 16% faster, bpc ≈ tf |
| **d128 ×5, w512** | 823k | 0.0276 | **2.0445** | 2.0677 | best bpc so far |
| d128 ×5, span12 | 658k | 0.0243 | 2.0392 | 2.0654 | candidate (1 seed) |
| d128 ×4, span16 | 700k | 0.0267 | 2.0469 | 2.0654 | wash once you pay the step |

Paired by seed, `tf − x5` = −0.013 / −0.045 (mean −0.029, clears the noise);
`tf − x4` = −0.009 / −0.019 (mean −0.014, inside the noise). So **x5 is the
replicated claim; x4 is a speed claim, not a bpc claim.**

Accuracy is not where the win is: tf top-1 0.593 / top-5 0.853 / bpc 2.018 vs
d128-x5 0.592 / 0.852 / bpc 1.999. Same guess, better tail — the gain is
distributional, which is what a compressor cashes in.

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
  d128** (0.334, 26.6%) — but fixing that does not pay: see `gate_agree` below.
- Drop-one: the short-range dot is load-bearing (+4.9 bpc without it); the far
  dot adds 0.28. Horizons are seed-stable, their *labelling* permutes.

## Kept

`global_decay`, `read_ln` + `head_ln`, block depth, d128 stream, three dots,
`global_read`, LUG cell, B=32, and — new tonight — **wide blocks**: at fixed
d128, w/d 2.5 → 3.0 → 3.5 → 4.0 gives 2.0638 → 2.0530 → 2.0488 → **2.0445** at
equal wall clock, and it wins at equal tokens too. The block was
under-parameterised at w = 3d all along; every earlier ratio point (1.33, 2.0,
2.4) was on the too-narrow side, which is why "3.0 is optimal" looked true.

## Closed (measured, not argued)

- **`gate_agree`** (+0.002 bpc, **12% slower step**): the right diagnosis — the
  gate *is* blind at d128 — but giving it committee agreement does not let it
  rank dots better. Gate work is dropped.
- **Block sharing** (+0.108 / +0.053), `global_decay_k`, `read_heads=2`, N=2,
  N=4 on a contiguous span-8 window (viable only once the window has range, and
  then it is a wash).
- **Geometric/quad bands** (+0.049 at reach 32, +0.079 at 64): dots retreat to
  the nearest slots, so contiguity is load-bearing.
- **Batch size as a throughput lever**: doubling B doubles the step (1.90×) for
  +5–8% tokens; `s/1k tokens` is flat at 0.0030 in every arm. FLOP-bound.
- **`pool_first`** (+0.006 bpc for 6% speed): the per-dot output projection does
  real work.
- **span 16** (+0.001 vs span 8, and worse at equal tokens): past 12 slots the
  far dot turns nearly uniform. span 12 is the candidate, not span 16.
- **`w/d = 3` as a sufficient statistic** — it is a *lower* bound.
- GRU cell (better per token, 16% slower per step → LUG wins on wall clock).
- `share_cell`, untied head, cell LayerNorm, d_s 64, `query_from_input`, `gla`,
  `ngram_bias`, `head_rank48`, `id_lags`, `head_skip`.

## Open at the time of writing

- `shape2` — **the depth-vs-width trade at constant MAC budget**: x4/w512
  (459k MAC), x3/w512 and x2/w768 (393k each) against x5/w384 (491k). If x3/w512
  holds quality it is ~20% cheaper per step at equal bpc, i.e. the first genuine
  speed multiple.
- `long1` — tf, x4 and x5 at **900 s/arm**, to check the ranking survives 5× the
  budget before anything is scaled up.
- `cmp1` — eager vs `torch.compile` in one session (first attempt died on a
  harness bug: `compile` was forwarded into `run()` and raised after the first
  arm had trained; fixed, and a failing arm no longer kills the session).
- `db1` — `dist_bias` (24 parameters, zero compute) where the ladder actually
  collapses, plus the cheap end (x3 at w384).
- `read4` — seed-1 twins of span12 and N4-span16; neither is a result yet.

## Next if the numbers land as expected

1. If x3/w512 holds, the frontier pivots to wide-and-shallow and every remaining
   arm is measured on MAC/token, not on depth.
2. If span 12 replicates, it becomes an arm knob on top of the wide block, and
   the two effects (more slots, wider MLP) should compose — the read is 0.4% of
   the step.
3. Then scale: the whole point of the small-scale protocol is that only
   mechanisms that paid here get to consume real compute.
