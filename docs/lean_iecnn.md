# IECNN-Lite: where the compute went, and what replaced it

Status: implementation complete, toy A/B measured (numbers below), full-scale
runs unaffected (every new knob defaults to the old behaviour).

## 1. The accounting (scripts/flops_audit.py)

MACs per token, toy shape (d=128, d_s=64-128, N=8, P=3, w=16, V=256):

| arm | MACs/token | vs tf-d96-l3 | vs old IECNN |
|---|---:|---:|---:|
| tf-d96-l3 (389k params) | 374,784 | 1.00x | 9.20x cheaper |
| iecnn-gru-ds64 (383k params) | 1,317,376 | 3.52x | 2.62x cheaper |
| iecnn-lite-ds64 | **411,648** | **1.10x** | 8.38x cheaper |
| iecnn-lite-p2n4-ds64 | **171,008** | **0.46x** | 20.2x cheaper |

The audit predicts the measured slowdown before any run: the champion lineage
did 3.5x the transformer's MACs and was measured at 3.0x the step time
(0.110 vs 0.037 s/step). So step time was never a mystery — it was arithmetic.

Breakdown of the *old* IECNN: cell 78%, e_state 11%, per-dot logits 8%, head 3%,
read 0.2%. **The read — where all the seclusion design effort went — is 0.2% of
the compute.** No read mechanism can make IECNN faster.

## 2. The four prunes (each behind a flag, default = old behaviour)

### (a) `model.cell: lug` — Lean Update Gate
The GRU pays 3 input gates + 3 recurrent gates per dot per pass. Replaced by
an SRU-style cell with a **diagonal** recurrence and one input GEMM of width
2*d_s, no hidden GEMM at all:

```
z = sigmoid(x W_z + u * s + b_z)      # keep gate, recurrent term is elementwise
h = tanh(x W_h + b_h)                 # candidate
s' = (1 - z) h + z s
```

Same integration over passes, ~2.5x fewer cell MACs. Cell is 78% of the old
budget, so this alone is the single biggest lever.

### (b) `model.head_last_only` — the head runs once, not P times
In `head_mode: hidden` the region pooled the dots and ran the tied `d x V` head
on **every pass**, and each dot ran its `p_out` projection on every pass. Only
the last pass's distribution is ever used. Passes 1..P-1 still update states and
weights (that is the point of a pass), they just stop emitting a distribution.

### (c) `model.per_dot_logits: false` — stop building `[B,T,N,V]`
The per-dot logits exist only for the optional auxiliary losses. Every one of
them (oracle, boosted-dot, decorrelation, confidence) reads **exactly one
number per dot**: the log-probability of the true token. When those lambdas are
zero — and in the toy/CE-only setting they are — the tensor is pure waste. It
was 18% of the lean budget at d_s=128 and 39% at d_s=64, plus a multi-MB
`[B,T,N,V]` allocation per step.

### (d) committee size — `n_dots`, `n_passes`, `d_s`
The remaining cost is `N * P * d_in * 2 * d_s`. Halving N and P and using
d_s=64 takes the cell from 1.0M to 101k MACs/token. These are quality knobs,
so they are only cut where the A/B says the bpc holds.

## 3. A mechanism that costs zero: `model.window_mode: quad`

The pointer audit (scripts/ab_small.py) on the trained toy read:

* mean pick distance **0.17 tokens back**
* **86%** of picks are the window's last slot (the previous token)
* **61%** of positions: all 8 dots pick the *same* token

The soft read had collapsed into a bigram — the "memory" was reading `t-1`.
The transformer sees 64 tokens. That, not the cell, is why bpc was 0.42 behind.

`window_mode: quad` keeps the same number of slots (same gathers, same scores,
same mix) but spaces them quadratically from distance 0 to `window_reach`:
with 16 slots you read ~0,1,2,3,5,7,10,13,16,19,23,27,31,39,48,64 — the whole
context, at zero extra compute. The relative-distance embedding is indexed by
the true distance, so it stays meaningful.

## 4. Seclusion v2 (`model.seclusion_topk`)

Tolu's vision: every dot commits to exactly ONE token, in parallel, cheaply.
That is hard top-1 attention, and it is already parallel (batched scores +
argmax + gather). Two things were wrong:

1. the backward path still computed a **full soft attention read** over the
   window (so seclusion cost *more* than soft, not less);
2. argmax has no gradient, which is what the soft term exists to donate.

`seclusion_topk: k` restricts the straight-through gradient to the top-k
near-miss keys. That is both cheaper (w/k fewer terms, no `[f,N,w,d_v]` mix)
and truer to the mechanism: the tokens worth "shouting louder" are the
runners-up, not the ones 20 positions away. Default 0 keeps the old path.

Honest caveat, from the audit: the read is 0.2-2.6% of compute, so seclusion
can never be a *speed* win at a 16-32 token window. It only becomes one when
the memory is large (score cost `T*w*d_q` vs a pointer lookup), which is the
long-context regime the mechanism is really aimed at.

## 5. The learning rate was worth more than any architecture change

Equal 150 s toy, N4/P2/d_s=64 (132.5k params), everything else identical:

| lr | steps | s/step | bpc |
|---|---:|---:|---:|
| 3e-4 | 4013 | 0.0374 | 4.7156 |
| **1e-3** (all earlier work) | 4395 | 0.0342 | 4.2834 |
| **2e-3** | 4395 | 0.0342 | **4.1179** |
| 3e-3 | 3965 | 0.0379 | 4.1458 |

**-0.17 bpc for one number.** The model was simply under-trained: at a 150 s
budget with a constant lr it never left the "more lr is better" regime (which
is also why the cosine-over-budget schedule *lost* 0.28 bpc — it halves the
average lr).

The pointer audit changed completely with the lr, which retroactively explains
two failed mechanisms:

| | lr 1e-3 | lr 2e-3 |
|---|---:|---:|
| all dots pick the same token | 70.6% | **6.2%** |
| mean pick distance back | 0.11 | 0.93 |

The "committee is one bigram detector" collapse was an **under-training
artefact**, not an architectural flaw. Both `dot_dist_bias` (a per-dot
distance prior, built to fix the collapse) and `dynamic_query` (built to give
each pass a different read) were solving a problem that a correct lr had
already solved — both lost bpc and cost steps. Pruned.

## 6. Global causal read at O(T) (the novel mechanism)

A dot's query is **static** (no dynamic_query), so `score(s) = q_n·k_s` does
not depend on *where* we are reading. Full causal attention is then exactly a
normalised prefix sum:

```
e_s   = exp(q_n · k_s)
ctx_t = Σ_{s≤t} e_s v_s / Σ_{s≤t} e_s        (two cumsums)
```

No T×T tensor, no per-position GEMM, no softmax over a growing row. At
T=64/N=4/d_v=32 that is ~16k MAC-equivalents — **less than the LUG cell
(59k)**. `g_glob` mixes the windowed read and the global read per dot, so dots
can specialise (local precision vs whole-prefix context).

Verified (`tests/test_lite.py`): the cumsum read equals a naive per-position
softmax over the prefix to 1e-5, and perturbing a future token cannot change
any earlier log-prob.

Measured, equal 150 s, same session:

| arm | bpc | s/step |
|---|---:|---:|
| plain P2 | 4.1038 | 0.0349 |
| P2 + global | **4.0841** | 0.0396 |
| P1, no global | 4.2874 | 0.0255 |
| P1 + global | 4.1433 | 0.0294 |
| P1 + global, window 8 | 4.1286 | 0.0273 |

Two things this shows: (a) the second pass is worth ~0.18 bpc, so passes are
not free depth but they do work; (b) the global read buys ~0.02 bpc at P2 and
~0.14 bpc at P1, i.e. it substitutes for a pass at 60% of the cost.

## 7. Pruned, with numbers

| mechanism | bpc@eqtok price | step cost |
|---|---:|---:|
| `window_mode: bands` (per-pass bands) | +0.38 | +10% |
| `dynamic_query` (per-pass reads) | +0.18 | +18% |
| `dot_dist_bias` (per-dot distance prior) | +0.10 | +5% |
| `sched: cos` (cosine over the budget) | +0.28 | ~0 |
| `window_mode: quad` at N4 | +0.16 | ~0 |
| seclusion | +0.31 | +6% |
| slot table (per-dot carry) | +0.11 | +30% |

None of them survived the 30% rule; all are still reachable behind flags and
re-testable in `scripts/ab_small.py`.
