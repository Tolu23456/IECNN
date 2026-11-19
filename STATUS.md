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
2. **Cheap extra mixing (`reread`) — MEASURED, +0.032 bpc.** The head is
   pool→MLP×3: a transformer block with the mixing sublayer deleted.
   `interleave` restored it at full cell price and lost (+0.10 bpc, +68% step);
   `reread` does it with one causal span-8, single-head read of the block's own
   output between blocks — zero-init so step 0 is bit-identical to the frontier
   arm. At 360 s: rr1 1.9904 (+6% step), rr2 1.9735 (+16%), against 2.0051 for
   the matched frontier. **Confirmed at 900 s (long5): rr2 1.9333 vs x3 1.9662,
   and rr2 did it with 19% fewer tokens** — at equal compute the gap is ~0.045,
   so this is the rare mechanism that pays for its own step cost.
3. **Capacity per step at long budgets.** x5/w512 beat x3/w512 by 0.037 at 900 s
   (the opposite of 180 s), so the long-budget shape favours depth; the untested
   version of that is **more dots** (N=4–5), which the ledger only ever rejected
   at 180 s.

## The read cannot bind — and the fix is one shift

A synthetic associative-recall probe (`scripts/binding_probe.py`, chance
5.0 bpc) says the frontier cannot **instance-bind** at all, and that one shift
fixes it. Final binding bpc after 2000 CPU steps, same data/seed/budget:

| arm | binding bpc | what it proves |
|---|---|---|
| **frontier** (static q, 1 pass) | 3.933 | barely above chance 5.000 |
| `qi1` (query = previous row) | 4.045 | the *wrong* shift is worse than nothing |
| `p2` + `dynamic_query` | 4.192 | depth / state-query do not help |
| **`kl1` (key = previous row)** | **0.000** | the right shift solves it exactly |
| `kl1p2` | 0.000 | and depth adds nothing on top |
| tf d96/4h/3l | 3.159 | the transformer does not solve it in this budget |

The circuit that is missing is induction, and the code makes it impossible by
construction: the score is `q_n · k_t` with `q_n` a constant, so the read can
only ask "does the token at this fixed offset match my pattern" — a fixed
horizon (the measured 0/2/5-back dots). Induction needs **key[s] = x[s−1]**
with **value[s] = x[s]**, so that a query built from the current token
retrieves what *followed* an earlier occurrence of it. That is `k_lag=1`: one
shift of the key stream, zero parameters. `kl1` hits 0.011 bpc by step 250 and
0.000 by step 500; the leak check (targets that are random by construction)
stays at 5.01 for every arm, so nothing is contaminated.

Lagging the *query* instead (`q_shift=1`) matches `x[s]` and returns `x[s]`
itself — 4.045 bpc, i.e. the wrong-shift control that turns the probe from a
tuning comparison into a mechanism proof.

The dot population was never the problem: the same three dots with the same
head solve the task the moment the key stream is delayed.

**But on enwik8 the bundle does not pay at 360 s — and the bundle is not the
lag.** Same session: x3/w512 2.0067, `kl1-nx` 2.0263 (same step cost), `kl1`
2.0492 (+15% step). Those arms change *three* things against the frontier:
(1) `k_lag=1`, (2) `query_from_input=True`, (3) `global_exact=True` for `kl1`.
(2) was already measured at **+0.063 bpc** in batch 24, and it is not a
per-dot mechanism: the arm adds **one** 2048-param `W_qi` whose `q_add` is
broadcast over all three dots, so the content term is common-mode and can only
make the committee *less* distinct. The audit's collapse
(`distinct_per_pos` 2.99 → 2.05/1.91, `all_same` 0.004 → 0.106/0.229, per-dot
distances [5.02, 2.07, 0.00] → [0.55, 0.00, 1.15]) is also measured with the
static query only: `pointer_audit` drops `q_add`, so it cannot say what the
trained read did. And `key = W_k x[s−1] + rel(d_s)` shifts the content but not
the distance tag. So the honest closure is **"shared content query + uniform
lag"**, not `k_lag`. The missing control is `-qi0` (frontier +
`query_from_input`, no lag, no exact): if it lands at ~2.026 the lag is
innocent; if it lands at ~2.006 the lag carries the cost.

The fixed horizons *are* the population's information (drop-one costs
+4.9 bpc), so a region-wide lag is still the wrong shape: it moves the
distance-0 dot off its only possible anchor. Per-dot is the fair test.

So the probe's lesson is narrower than it looked: the model **cannot** bind,
but enwik8 at this scale does not pay for binding more than it pays for
horizons. The open question is a per-dot lag (keep the horizons, add one
induction read), not the region-wide one.

## What the models actually say (same leg, same prompt)

`sample1` trained tf and the frontier arm on the same 360 s leg (tf 1.9532,
dot 2.0075) and decoded the same unseen 120-byte val prompt. Greedy decoding
collapses for both (a loop is the highest-probability continuation at ~2 bpc),
but sampled decoding shows the difference the bpc gap is made of:

- **tf** produces *document structure*: section headers (`== External links ==`),
  templates (`{{main|...}}`), link markup, dates — i.e. a register and a plan
  that run for hundreds of bytes.
- **the dot model** produces locally fluent wiki-ish phrases with the *surface
  tokens* of that structure (`[[Massachusetts]]`, `[[Charles English|Central
  Society]]`) but no plan: links appear as n-gram furniture, not as sections.

That is the same story the probe tells numerically: the dot model has the
statistics of the text but not the variable binding that lets a plan persist.

## Frontier

At the 360 s screening budget (in-session tf 1.9639–1.9729):

| arm | params | s/step | bpc | bpc@eqtok |
|---|---|---|---|---|
| tf-swa | 408k | 0.0240–0.0265 | 1.9539–1.9729 | 2.021–2.052 |
| **d128 ×3, w512** | 559k | **0.0206–0.0215** | 2.0051–2.0067 | 2.072–2.082 |
| **d128 ×3, rr1** | 571k | 0.0219 | **1.9904** | 2.0528 |
| **d128 ×3, rr2** | 571k | 0.0240 | **1.9735** | 2.0321 |
| d128 ×3, p2 | 559k | 0.0282 | 2.0062 | 2.0609 |

`reread` (a causal span-8, single-head read of the head block's own output
between blocks, zero-init) is the **first mechanism to beat the matched
frontier at 360 s**: rr1 −0.0147 at +6% step, rr2 −0.0316 at +16%. rr2 costs
exactly what tf costs per step and is still 0.020 behind it, so the honest
claim is "0.032 bpc for 16% more step time", not "faster than tf".

At 900 s (long5, same session, common 277.70M tokens): **tf 1.8400 ·
x3 1.9662 · rr1 1.9470 · rr2 1.9333.** rr2 beats x3 by 0.0333 while spending
23% more per step and therefore doing **19% fewer tokens** — at equal compute
the gap is ~0.045. That is the "pay for your compute" test and reread passes
it (most of the ledger does not). It is still 0.093 behind tf, and tf is 19%
faster per step.

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

## Compile — engineering, with a second blocker

Earlier: compile bought tf **25%** and the dot model **0%**, which looked like an
architecture property. It was not the guard. The region died in dynamo at
`models/memory.py:78` on a data-dependent guard (the plain-window fast-path test);
on T4 that surfaced as an `UNPACK_SEQUENCE` size mismatch 60 lines away, which is
why two diagnostics blamed the prefix scan. Fixed by passing the fast-path
decision as a Python bool from the config.

`cmp2` (360 s) then showed the second, independent blocker: **Inductor's CUDA
`SplitScan` lowering dies on the `cumsum` in the global read** (`TypeError: list
indices must be integers or slices, not NoneType`, buffer `[32,256,3]`), so
`x3-c` and `x3-cro` failed at step 0 while `tf-c` ran 29% more steps and gained
0.075 bpc. `gx-c` compiles because the exact read has no scan, but the exact
read itself costs 0.026 bpc, so it is not a substitute. This is a *second*
defect, not the guard: CPU `fullgraph` compiles the scan fine; CUDA does not.

The fix is already in the tree: `ARM_BREAK_SCAN=1` -> `break_scan`, a
`@torch.compiler.disable` boundary around the scan only. `scn1` (tf, tf-c, x3,
x3-c with the boundary) measures whether the graph break keeps the compile win
on GPU. If it eats it, the fallback is a log-depth Hillis-Steele scan (pad+add,
all pointwise) instead of `cumsum`.

## Running / queued

- `dis1` (running, 360 s) — tf / x3 / **`qi0`** / `kl1-nx` in ONE session: the
  control that separates the lag from the shared content query inside bind2.
  `qi0` ~ 2.026 -> the lag is innocent; ~ 2.006 -> the lag carries the cost.
- `scn1` (running, 360 s) — tf / tf-c / x3 / **x3-c with `ARM_BREAK_SCAN=1`**:
  does the eager boundary around the prefix scan recover the compile win for
  the dot model? This is the biggest remaining throughput lever.
- then: `cmp3` at 900 s once the compile number is real, a flavour rerun on the
  saved 900 s weights, and the per-dot variant (one content query + one value
  lag on the far dot only) only if `qi0` clears the query.

## The thing to be careful about

Provisional, not settled: **"we beat the transformer"** (true only below ~250 s
of wall clock), the **reread** mechanism (a clear 360 s win at one seed; needs
long5's 900 s and a second seed before it is a claim), the compile win for the
dot model (fullgraph works on CPU; the GPU number is what counts), and the
**`k_lag` induction read** (bind2's loss is a *bundle* — uniform lag + a shared
content query — so it needs the `-qi0` control before the lag itself can be
blamed). Everything else in the ledger has either two seeds or a margin that
clears the within-session noise.
