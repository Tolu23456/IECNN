# IECNN status — 2026-11-19 (autonomous overnight block, updated)

One-page state of the campaign. The evidence ledger is `DECISIONS.md` (every row
is a measured batch with its in-session transformer anchor); this is the map, not
the evidence.

## The headline, corrected twice

**The dot model is a fast-start model with a ceiling at ~1.93 bpc, and it is a
CEILING IN THE TRAINING FIT, not in the data.** Same protocol, 900 s/arm:

| arm | tokens | **epochs** (of 60 MB) | val bpc | train bpc |
|---|---|---|---|---|
| tf-swa | 290.0M | 4.83 | **1.8406** | 1.8020 |
| x3/w512 | 357.1M | 5.95 | 1.9683 | 1.9026 |
| x5/w512 | 265.7M | 4.43 | 1.9317 | 1.8572 |

The dot model sees **1.2–1.4 more epochs than the transformer and still fits the
training set worse** (1.86–1.90 against 1.80). Val−train gap stays 0.04–0.07 on
both, so nothing is memorising: the model is *underfitting*, and capacity per
step — not tokens per second, not data volume — is what is binding. That is the
single most useful sentence in this file.

The gap is also budget-dependent, monotonically: dot −0.014 at 180 s, +0.043 at
360 s, +0.14 at 900 s. The crossover is ~200–260 s of wall clock, so **every
180 s "beats tf" row in the ledger was measured on the dot model's side of it**.
The 180 s budget is retired for architecture claims; 360 s is the floor (and two
360 s sessions reproduced the frontier to 0.0006 bpc, so the within-session claim
threshold there is ~0.01, not 0.024).

## What lifts the plateau — three routes, in order

1. **Cheaper steps (compile).** Root cause found: `gather_windows` branched on
   tensor data (the plain-window fast path), which is a hard dynamo guard
   failure, so every compiled dot run silently fell back to eager. With the
   decision moved to config time, `torch.compile(fullgraph=True)` now compiles
   the whole region (CPU-verified; GPU A/B running). If the 25% the transformer
   gets is available here too, the step gap becomes the whole story.
2. **Cheap extra mixing (`reread`).** The head is pool→MLP×3: a transformer block
   with the mixing sublayer deleted. `interleave` restored it at full cell price
   and lost (+0.10 bpc, +68% step); `reread` does it with one causal span-8,
   single-head read of the block's own output between blocks — ~6% of a block,
   zero-init so step 0 is bit-identical to the frontier arm.
3. **Capacity per step at long budgets.** x5/w512 beat x3/w512 by 0.037 at 900 s
   (the opposite of 180 s), so the long-budget shape favours depth; the untested
   version of that is **more dots** (N=4–5), which the ledger only ever rejected
   at 180 s.

## Frontier

At the 360 s screening budget (in-session tf 1.9639–1.9729):

| arm | params | s/step | bpc | bpc@eqtok |
|---|---|---|---|---|
| tf-swa | 408k | 0.0255–0.0265 | 1.9639–1.9729 | 2.044–2.052 |
| **d128 ×3, w512** | 559k | **0.0206–0.0215** | 2.0061–2.0067 | 2.079–2.082 |
| d128 ×3, p2 | 559k | 0.0282 | 2.0062 | 2.0609 |

At 900 s the same two arms are tf 1.8406 vs x3 1.9683, and x5/w512 is 1.9317.
The dot model is **19% faster per step and 23% ahead at 60M tokens**, and behind
by 0.04–0.14 at every budget past ~250 s. Both shapes are still moving.

Structural results that survive every budget check:

- **Wide-and-shallow wins in the fast-start regime** (x3/w512 beats x5/w384 and
  is 14% cheaper per step; hard depth floor at 3), **and inverts at 900 s**,
  where x5/w512 wins. The shape optimum is budget-dependent.
- **The expansion ratio's optimum is above 3**, still monotone at w/d 6.0.
- **p2 is the best model per token in the campaign** (2.0102 vs 2.0265
  bpc@eqtok) and a wash per second — the cleanest evidence that the ceiling is
  *mixing frequency*, not capacity.

## What the dots are (unchanged, and still the novel part)

Measured through the trained path (one-hot pool): the three dots **self-organise
into time horizons** — 0 / ~3 / ~6 tokens back at d192, 0 / 0.8 / 2.8 at d128 —
agreeing on only 6–10% of predictions with ~0.4-nat-separated distributions. The
gate tracks the best dot at d192 (0.385 weight, argmax = best 58.6%) and is
anti-informative at d128; `gate_agree` bought nothing for 12% of the step, so
gate work is closed. Horizons are seed-stable and their labelling permutes, so
per-dot claims must be permutation-invariant.

## Kept

`global_decay`, `read_ln` + `head_ln`, three dots, `global_read`, LUG cell,
B=32, wide blocks (ratio 4.0), depth 3 as the floor, 360 s as the screening
budget.

## Closed (measured, not argued)

`gate_agree`; block sharing; `global_decay_k`; `read_heads=2`; N=2; N=4 on a
contiguous span-8 window; geometric/quad bands; batch size as a throughput lever;
`pool_first`; **span 16 and span 12 (does not replicate at either budget)**;
GRU at both budgets; `share_cell`; untied head; cell LayerNorm; d_s 64;
`query_from_input`/`gla`; `ngram_bias`; `head_rank48`; `id_lags`; `head_skip`;
`gate_agree`; **`dist_bias`** (0.0045 < noise); `global_exact` (`gx`).

## Compile — the asymmetry was engineering

Earlier: compile bought tf **25%** and the dot model **0%**, which looked like an
architecture property. It was not. The region died in dynamo at
`models/memory.py:78` on a data-dependent guard (the plain-window fast-path test);
on T4 that surfaced as an `UNPACK_SEQUENCE` size mismatch 60 lines away, which is
why two diagnostics blamed the prefix scan. Fixed by passing the fast-path
decision as a Python bool from the config. Now `fullgraph=True` compiles, and
`cmp2` (tf, tf-c, x3, x3-c, x3-gx-c, x3-cro at 360 s) measures whether the dot
model gets the same 25%. If it does, `p2` becomes affordable too.

## Running / queued

- `long3` (running, 900 s) — lr 1e-3 vs 2e-3: is the plateau an optimiser artefact?
- `mid2b` (running, 360 s) — `reread` rr1/rr2 vs the frontier arm.
- `cmp2` (queued, 360 s) — the compile A/B above.
- then: `cmp3` (900 s: tf-c, x3-c, p2-c), and N=4–5 at 900 s if the ceiling route
  3 is to be tested properly.

## The thing to be careful about

Provisional, not settled: **"we beat the transformer"** (true only below ~250 s
of wall clock), the **reread** mechanism (built and unit-tested, no bpc yet), and
the compile win for the dot model (fullgraph works on CPU; the GPU number is what
counts). Everything else in the ledger has either two seeds or a margin that
clears the within-session noise.
