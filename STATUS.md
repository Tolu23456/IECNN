# IECNN status — 2026-11-19 (autonomous overnight block)

One-page state of the campaign. The evidence ledger is `DECISIONS.md` (every row
is a measured batch with its in-session transformer anchor); this is the map, not
the evidence.

## The headline, corrected

**The dot model is a fast-start model with a ceiling at ~2.0 bpc, and the
transformer does not have that ceiling.** Same session, 900 s/arm, matched
tokens (val bpc, tf / d128-x4 / d128-x5):

| tokens | tf | x4 | x5 | tf − x4 |
|---|---|---|---|---|
| 20M | 2.4114 | 2.2893 | 2.2897 | **+0.122** |
| 60M | 2.1387 | 2.1452 | 2.1278 | −0.006 |
| 100M | 2.0599 | 2.0977 | 2.0761 | −0.038 |
| 150M | 1.9715 | 2.0593 | 2.0490 | −0.088 |
| 280M | 1.8948 | 2.0320 | 2.0059 | **−0.137** |

At the 180 s budget (~60M tokens) the two are level; at 900 s tf is 0.14 bpc
ahead and still descending, while the dot model has flattened. It is *not*
overfitting — `bpc_train` flattens too (1.892 for x5 against 1.803 for tf).
So every "beats tf" row earlier in the ledger is a **screening** result measured
inside the fast-start window, and none of them may be quoted as beating the
transformer without this table attached.

## What that changes about the plan

The question is no longer "what makes the step cheaper at 60M tokens" but
"**what lifts the plateau**". Two candidates, both cheap to test because the
read is 0.4% of the step:

1. **The far read cannot be content-addressed.** The query is static, so the
   global read is linear attention: it can learn "look 3 tokens back" but not
   "look at the token that matches this pattern". Attention is exactly the thing
   tf has and the dot model does not. Two ways to give it content addressing:
   `global_exact` + `query_from_input` (real T×T softmax, ~6% more MACs at
   T=256) and multi-pass (pass 2 re-reads the window with the pass-1 message as
   query — cheap, 2× cell).
2. **The cell is a diagonal recurrence with a d_s=96 state per dot** — a much
   smaller memory than attention over 256 tokens. GRU was better *per token* and
   worse *per step* at 180 s; at a long budget per-token quality dominates, so
   it deserves a re-ask.

Screening budget for this question is **360 s/arm (~110M tokens, past the
crossover)** — 5 arms in 30 min instead of 5×15, with the winner confirmed at
900 s.

## Frontier at the 180 s screening budget (still the best shapes)

| arm | params | s/step | bpc (in-session tf) |
|---|---|---|---|
| tf-swa | 408k | 0.0245–0.0263 | 2.061–2.092 |
| d128 ×5, w384 | 658k | 0.0235–0.0255 | 2.047–2.057 (2.061–2.075) |
| d128 ×4, w384 | 559k | **0.0212** | 2.059–2.068 |
| **d128 ×3, w512** | 559k | **0.0206** | **2.0483** (2.0627) |
| d128 ×5, w512 | 823k | 0.0276 | 2.0428–2.0445 (best bpc) |
| d128 ×5, span12 | 658k | 0.0243 | 2.0392 (candidate, 1 seed) |

Two structural results survive every budget check so far:

- **Wide-and-shallow beats deep-and-narrow.** At a constant MAC budget x4/w512
  ties x5/w384, and x3/w512 *beats* both while costing 14% less per step; x2/w768
  loses, so there is a hard depth floor at three blocks and the width axis has
  not turned over by ratio 4.0.
- **The expansion ratio's optimum is above 3.** At fixed d128: 2.5 → 2.0638,
  3.0 → 2.0530, 3.5 → 2.0488, 4.0 → 2.0445. Every earlier ratio point was on the
  too-narrow side, which is why "3.0" looked optimal.

## What the dots are (unchanged, and still the novel part)

Measured through the trained path (one-hot pool), not the untrained per-dot
logits the first audit wrongly used: the three dots **self-organise into time
horizons** — 0 / ~3 / ~6 tokens back at d192, 0 / 0.8 / 2.8 at d128 — agreeing on
only 6–10% of predictions with 0.4-nat-separated distributions. The gate tracks
the best dot at d192 (0.385 weight, argmax = best 58.6%) and is anti-informative
at d128 (0.334, 26.6%), but fixing that (`gate_agree`) bought nothing and cost
12% of the step, so gate work is closed. Horizons are seed-stable; their
labelling permutes, so per-dot claims must be permutation-invariant.

## Kept

`global_decay`, `read_ln` + `head_ln`, three dots, `global_read`, LUG cell,
B=32, wide blocks (ratio 4.0), and — pending replication — span 12.

## Closed (measured, not argued)

`gate_agree`; block sharing (+0.108/+0.053); `global_decay_k`; `read_heads=2`;
N=2; N=4 on a contiguous span-8 window; geometric/quad bands (+0.049/+0.079);
batch size as a throughput lever (FLOP-bound, `s/1k tok` flat at 0.0030);
`pool_first` (+0.006 for 6%); span 16; `w/d = 3` as sufficient statistic;
GRU at 180 s; `share_cell`; untied head; cell LayerNorm; d_s 64;
`query_from_input` and `gla` **at 180 s** (re-opened only for the long-budget
ceiling question); `ngram_bias`; `head_rank48`; `id_lags`; `head_skip`.

## Compile — an honest asymmetry worth knowing

`torch.compile` buys the transformer **25%** (0.0240 → 0.0192 s/step, 77M tokens
vs 61M, bpc 2.026 vs 2.061) and buys the dot model **nothing** (0.0237 → 0.0239,
bpc identical). The bpc difference is purely tokens: at equal tokens they are
2.1397 vs 2.1362. So against `tf-swa-c` the campaign is behind, and the gap is
engineering, not architecture — the region keeps an eager boundary around the
prefix scan and carries python-side state. `comp2` (CUDA graphs, and compile
with the boundary removed) and `comp3` (masked-matmul far read, which deletes the
only forced graph break) test whether it is fixable. This matters more now that
tf, not the dot model, is the arm to catch.

## Running / queued

- `long2` (running) — 900 s: does wide-shallow raise the ceiling?
- `mid1` (queued) — 360 s screen of the three ceiling hypotheses: content
  addressing (`gx`), multi-pass, GRU.
- `long4` — 900 s confirmation of whichever hypothesis wins.
- `diag`, `comp2`, `comp3` — compile fixability.
- `shape3` — width turnover (screening only).
- `db1`, `read4` — `dist_bias` at the d128 ladder; seed twins of span12 and
  N=4-span16.

## The thing to be careful about on wake

Two claims are now provisional rather than settled: **"we beat the transformer"**
(only true at ≤60M tokens) and **"span 12 is better"** (one seed, inside noise).
Everything else in the ledger has either two seeds or a margin that clears 0.024.
