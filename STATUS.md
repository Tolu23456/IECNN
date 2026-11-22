# IECNN status — 2026-11-22

One-page state of the campaign. The evidence ledger is `DECISIONS.md` (every row
is a measured batch with its in-session transformer anchor); this is the map, not
the evidence.

## The position in one paragraph

**The objective is met in the regime it was set for.** At 900 s on the canonical
90/5 split, one healthy session, seven arms, common 255.55M tokens: `tf-swa-c`
**1.7960**, `x5-sc3e` **1.7740 / 1.7728** (seeds 0/1), `x5-sc3` 1.7825 / 1.7883,
`x4-sc3e` 1.7904 / 1.7894. So the dot model is **ahead of the transformer at
equal wall clock by 0.0226** (mean of two seeds, both negative) on **29% fewer
tokens**, ahead at equal tokens by ~0.05, and ahead at **matched total MACs at
every measured budget** (25T −0.094/−0.100, 50T −0.065/−0.068, 100T −0.022/−0.026,
150T −0.018/−0.019). To reach **2.00 bpc** it needs **27.1/29.1T MACs against the
transformer's 48.3T** — 1.8x less compute; `x4-sc3e` does it in **24.7/26.3T**
(2.0x) and is the best arm per MAC below ~100T, but at the 900 s endpoint it is
+0.016 behind `x5-sc3e`, so it is the compute-capped arm, not the headline.

**What paid for its own compute**, each confirmed on two seeds in-session:
`reread` (−0.033), head capacity `x5` (−0.046), the exact-match copy channel
`suffix_copy` (−0.0506), and its match-extension slope `sc3e` (−0.0120). The
copy channel is a **real induction mechanism**, not a mixture trick: replacing
its pointer with a random earlier position costs −0.1448 and is worse than
shutting the gate. **Plain voting over duplicate matches is closed** (they are
the same coincidence, not independent evidence); extension-based *selection* is
the priced successor and is screening now.

**Honest framing:** this is a small-scale, ≤2.0 bpc, T4 result. The per-MAC
advantage decays with budget (0.10 at 25T to 0.018 at 150T) and the curves are
not shown to cross beyond the measured range, so the claim is
compute-efficiency at ~2.0 bpc, not eventual superiority.

## What the decoder actually uses: a specialist channel, a 2+spare committee

`iecnn-genablate` (CPU, same weights, one inference change at a time, same
prompt and seed). Faithfulness is exact: recomputed val bpc **1.7740 / 1.7728**
for `x5-sc3e` s0/s1 vs `bpc_final` 1.774/1.7728, and 1.7972 for `tf-swa-c` vs
1.796 — the saved weights are the SWA average, so `bpc_final` is the reference,
not the last raw marker.

| Q1 repeated random pattern, greedy 16B | match | run | tf bpc |
|---|---|---|---|
| `full` | **1.000** | 16.0 | **0.654** |
| `off` (gate shut) | 0.008 | 0.0 | 9.844 |
| `slope0` (b=0, level kept) | 0.078 | 1.2 | 2.701 |
| `scram` (random pointer) | 0.008 | 0.0 | 10.537 |
| `tf-swa-c` control | 0.021 | 0.0 | 8.106 |
| drop dot 0 / 1 / 2 | 1.000 each | 16.0 | 0.655/0.652/0.653 |

- **The channel is worth 9.19 bpc on its niche and 0.006 on ordinary text**
  (Q2: 1.7354 full vs 1.7410 off, 192 positions). The 0.012 aggregate is small
  because the niche is small, not because the mechanism is weak.
- **The slope is 22% of the niche bpc but 92% of its behaviour**: flattening it
  costs 2.05 bpc yet drops exact continuation from 1.000 to 0.06–0.08 — the
  flat 14.6% copy weight leaves the right byte in the mixture but no longer
  winning the argmax. The level term is the other 7.14 bpc.
- **Induction is orthogonal to the dots**: dropping any single dot leaves
  match 1.000 / run 16.0 / bpc 0.652–0.655.
- **The falsifier holds**: flip one byte of the second copy (no k-gram matches)
  and the behaviour collapses to 0.016 while the teacher-forced gain falls from
  −9.19 to −6.41 bpc. It is exact-match continuation, not pattern completion,
  and `tf` cannot do it at all (8.11 bpc) — a real capability difference.

**The dots are a two-dot committee plus a spare**, and the shape is the same on
the model's own generated text as on real text (drop-one +0.168/+1.265/+3.932
self vs +0.138/+1.300/+3.762 real on s0; +1.059/+0.096/+4.316 vs
+0.962/+0.150/+4.171 on s1; mixture beats the best single dot by 1.44/1.23 bpc).
The distance-0 dot is essential in both seeds (drop +3.9/+4.3) and the best
solo dot (2.66/3.02 vs 5.57–5.69); one long-range dot carries real independent
value; the third is nearly redundant — **but which index is the redundant one
swaps between seeds** (spare = dot 0 at +0.17 on s0, dot 1 at +0.10 on s1). So:
division of labour yes, stable per-index meaning no. **Do not build index-based
routing.**

### And the gate parameters are already at their optimum

`iecnn-slope-ablation` fixes every weight of the trained `sc3e` and changes only
the slope at eval (faithfulness exact: analytic == model to 4 dp, 1.7740 /
1.7728):

| | s0 | s1 |
|---|---|---|
| learned a / b | −1.7684 / +0.2884 | −1.7679 / +0.2864 |
| coverage / copy accuracy | 23.3% / 63.0% | 23.3% / 63.0% |
| whole channel (gate shut) | **+0.0653** | **+0.0623** |
| slope only (b=0, a kept) | **+0.0132** | **+0.0129** |
| sweep optimum | b=0.25 (+0.0005) | b=0.25 (+0.0004) |

The eval-only control reproduces the trained two-seed difference (−0.0120) to
0.001, so the batch result is the mechanism. And **nothing is left in the
parameters**: b is within 0.0005 of its sweep optimum on both seeds, a is
optimal given b (a = −2/−1/0 costs 1.7746/1.7811/1.8184), and no cell of the
2-D grid beats the learned point by more than 0.0005.

Copy accuracy by extension is what the slope exists to exploit: **49.5% at
extension 0 → 87.5% at 8+**, with the gate rising 0.146 → 0.632. The 8+ bucket is
10.8% of all firing positions. The scalar arm is not `sc3e` with b=0 — it runs a
0.241 flat gate (a=−1.1447), which recovers 0.0047 of the difference, and that
is why the trained gap (0.0085/0.0155) is smaller than the b-ablation (0.0132).

Consequence for the next mechanism: gate tuning is **closed**. Raising the
average needs more **coverage** — which positions fire, i.e. selection among
multiple occurrences — which is exactly the in-flight `sc3ev`.

### Long continuations: a diagnostic, and one trap

`iecnn-longtext` decodes 350 bytes per prompt (5 prompts, T 0.7, 256-byte
sliding context) and measures the structure instead of eyeballing it. It is a
diagnostic, not a quality argument — the only powered quality comparison in the
ledger is 11-9, p=0.82.

**Pairwise longest common substring across the 5 samples** (mean / max bytes) is
the one statistic that sees a failure the within-sample ones cannot:

| | full | gate shut |
|---|---|---|
| `x5-sc3e` s0 | 11.4 / 33 | 11.0 / 27 |
| `x5-sc3e` s1 | **33.8 / 262** | 31.7 / 158 |
| `tf-swa-c` | 10.0 / 13 | — |

- On **s1**, two prompts with different prefixes become byte-identical from
  their 100th generated byte on (self-bpc agrees to 3 dp). Within-sample d2 is
  blind to it, and shutting the gate leaves 158 B shared, so **neither the
  channel nor the architecture is the cause** — it is seed-level, n=2, one seed
  each way.
- **Do not read a falling self-bpc as improvement.** Every arm and condition has
  a lower self-bpc on its last 100 bytes than its first 100 — *including tf*
  (2.82→1.32, 2.00→1.57). At T 0.7 the sampler reinforces its own most probable
  continuation, so a low late self-bpc marks drift *into* a groove and would
  flatter whichever arm loops hardest.
- Within-sample d2, rep4 and link balance are comparable across arms (dot
  max rep4 3–10 vs tf 4–9; no systematic `[[` imbalance), so the v1 note about
  nested unclosed links does not replicate.

## 2026-11-21 — capacity in the HEAD pays; the POPULATION is a fusion

| 900 s, common 272.23M tok, one session | params | s/step | bpc | bpc@eqtok |
|---|---|---|---|---|
| tf-swa-c | 408k | 0.0185 | **1.7923** | 1.8666 |
| rr2-c | 571k | 0.0206 | 1.8847 | 1.9451 |
| rr3-c | 571k | 0.0224 | 1.8769 | 1.9369 |
| **x5+rr2-c** | 835k | 0.0271 | **1.8383** | **1.8894** |

- **Head capacity (x5) moves TRAIN fit**, not just val: train 1.8677 → 1.8183,
  and the gap to tf halves on both clocks. `x5` pays for its 31% step cost.
  `rr3` is a null (−0.008).
- **Width, not depth.** At 9.1M tokens, `x5`→`x7` at w512 buys 0.0075 for 131k
  params and +5.6% step; `w512`→`w256` costs **+0.074**. Screening only.
- **The collapsed dot is the INPUT PATH.** With `n_passes=1` the cell sees only
  `[ctx, 0, 0, round]` and `ctx` is a softmax average of window *values*, so
  the distance-0 dot is the only route for the current token.
- **`input_res` moves the collapse rather than fixing it.** Injecting `v(t)`
  zero-param removes the distance-0 dot, and at 9.23M tokens all three dots
  then read the same **distance-1** slot (`distinct_per_pos` 2.97 → **1.0**):
  the freed dot does not re-specialise in that budget.
- **But the population is NOT redundant — that claim is withdrawn.** On the
  3 h checkpoint (base 1.8276 reproduced exactly) drop-one costs **+4.60 /
  +1.62 / +0.34** bpc, any dot alone is 4.0–7.9 bpc, and the per-position best
  single dot is 3.05 — 1.22 bpc *worse* than the mixture. Controls: uniform
  weights cost only +0.26 (the head is smooth, so the ablation is not
  fragility), a soft drop of dot 0 to the training floor still costs +4.29
  (not the exact zero), permuting dot identity costs +1.27 with a fixed
  permutation (the first pass said +0.89 because it re-drew the permutation
  every forward, which partly cancels), and time-shuffling the weights +0.36. So dot 0 is the **input path**, dots 1–2 read ~2 and ~2.7
  back, and the three are complementary. The target is to make the input path
  free and force the freed dots apart — **not** to delete dots.
- **Generation, n=8.** Both models peak at T=0.70 for a 0.875-nat target. At
  matched entropy the dot model is **more** varied, not less (d2 0.5900 vs
  0.5204, ~2.2σ), so "its text only looks different because it is sharper" is
  dead — and the blind judge still splits 11-9 for tf. Variety ≠ quality.
- **Where the gap is: INDUCTION AT DISTANCE 9–64.** Two corpus-only buckets on
  the 3 h checkpoints (no retraining). By **copy distance**: the dot model loses
  0.090 / 0.083 / 0.087 bpc at d = 1 / 2-3 / 4-7, **0.151 / 0.162** at
  d = 8-15 / 16-63, only 0.048 at 64-255, and is **0.039 BETTER** beyond 256 —
  72% of the gap outside its 8-slot window. By **induction support** (is the
  answer the byte that followed the last occurrence of this exact 3-byte
  context?): the decisive bucket is *right, d 9–64* — **10.4% of positions,
  carrying 55% of the whole 0.115 gap**, where the dot model is at **1.118 bpc
  against tf's 0.509**. Beyond 64 the two nearly agree (0.991 vs 0.900) on 40%
  of positions. Meanwhile 46% of positions (wrong guess, d>8) carry 9%.
  So the deficit is not uniform capacity: it is one band, and it is exactly the
  band a 8-slot window cannot address. **Do not re-stage the obvious fix:** the
  longer *contiguous* window was already measured at 180 s (span8 2.0564,
  span12 2.0392, span16 2.0577, seed noise 0.024) and its far dot's read
  entropy saturates at span16 (2.53 of 2.77 nats), i.e. extra slots stop buying
  resolution. That is a real, named weakness we currently have no cheap arm for
  — it needs a materially different read, not a wider one. `tf-w3/w8/w32`
  remain the causal control for whether *exact* access is the whole story.
- **The machine is per-host, not per-hour.** The same T4/torch did 0.0170 s/step
  at 10:38 and 0.3551 s/step at 09:00, both alone. A slow run is a bad host:
  absolute bpc from it is void, and an in-session comparison needs the per-arm
  `s/step` to agree within ~10%. The heartbeat prints `s/step` from the first
  minute with a >0.05 s/step abandon threshold for tf — but on this CLI the log
  stream appears only after the kernel ends, so it is a **post-hoc** check, not
  a live abort. A slow host is recognisable in the first line of output once
  the run is over (or via `kaggle kernels output`), and the run is then void.

**Now:** `tf-swa-w3/w8/w32-c` vs `tf-swa-c` price the exact-access horizon on
the healthy pool. The window is **per layer**, so at 3 layers the horizons are
**7 / 22 / 94** — `w3` is the arm matched to the dot model's 8-slot read, and
the other two map the curve. Pre-registered reading (written before the run
finished, so the result cannot be retro-fitted):

| tf-c → tf-w3 change (horizon 7 vs 256) | reading |
|---|---|
| −0.00 to −0.02 | horizon is NOT the gap; the dot model's deficit is in the interaction rule / state, and read-horizon work is closed |
| −0.02 to −0.10 | the gap is partly horizon; compare its size with the dot gap (0.046 at 900 s) |
| worse than −0.10 | a 7-slot transformer is much worse than the 8-slot dot read, so the dots' read is not the problem either and the deficit is elsewhere |

`w8` (22) and `w32` (94) say how fast the horizon saturates: if w22 ≈ tf-c the
transformer only needs a couple of dozen positions, and the dot model's 8-slot
read is the handicap; if w94 is still behind tf-c, exact access is worth much
more than 8 and no window arm will match.

**Status of that batch: still unmeasured, but the blocker is now known and
fixed.** Two pushes were aborted at the gate (`tf-swa-c` at **0.3562** then
**0.3663** s/step), and the cause was not the host: those kernels were pushed
with `mark_s=0`, which made the arm run a full validation inside every step (see
the withdrawn machine caveat above). With the guard in place the ladder is
re-queued with the `suffix_copy` screen, same batch, same anchor, unchanged.

## CONFIRMED AT 900 s, TWO SEEDS: suffix_copy −0.0506, dot model ahead of tf

`iecnn-sc9-gpu`, one healthy session, 900 s/arm, common 254.20M tokens:

| arm | seed | tokens | s/step | bpc | bpc@eqtok |
|---|---|---|---|---|---|
| `tf-swa-c` | 0 | 378.19M | 0.0195 | **1.7956** | 1.8726 |
| `x5+rr2-c` | 0 | 265.85M | 0.0278 | 1.8419 | 1.8909 |
| `x5+rr2-c-sc3` | 0 | 254.20M | 0.0290 | **1.7868** | 1.8284 |
| `x5+rr2-c` | 1 | 265.53M | 0.0278 | 1.8372 | 1.8897 |
| `x5+rr2-c-sc3` | 1 | 254.29M | 0.0290 | **1.7831** | 1.8295 |

Channel effect per seed **−0.0551 / −0.0461** (mean −0.0506) for **+4.3%**
compute, ~10x the within-pair seed spread; equal-token **−0.0625 / −0.0602**.
Versus the transformer at equal wall clock: **−0.0088 / −0.0125** (mean
−0.0107), both seeds negative, with 33% fewer tokens.

**Read it precisely — and the per-MAC question is now measured, not assumed.**
The earlier line here ("per-MAC the family is still 0.03–0.05 behind") described
the *plain* arm and was wrongly applied to the accepted one. Interpolating the
logged raw eval markers (the SWA endpoint is 0.045–0.061 better than the last
raw marker on *every* arm, so it is excluded from the comparison) gives bpc at
matched **total MACs**:

| total MACs | `tf-swa-c` | `x5` plain | `x5-sc3` s0 | `x5-sc3` s1 |
|---|---|---|---|---|
| 10T | 2.3561 | 2.2980 (−0.058) | **2.2701 (−0.086)** | 2.2868 (−0.069) |
| 25T | 2.1070 | 2.1070 (±0.000) | **2.0319 (−0.075)** | 2.0340 (−0.073) |
| 50T | 1.9840 | 2.0145 (+0.031) | **1.9396 (−0.044)** | 1.9478 (−0.036) |
| 100T | 1.8816 | 1.9394 (+0.058) | **1.8772 (−0.004)** | 1.8798 (−0.002) |
| 160T | 1.8436 | 1.9055 (+0.062) | **1.8424 (−0.001)** | 1.8448 (+0.001) |

MACs needed to reach a raw bpc: **2.10** tf 26.1T vs 18.1T/18.8T (1.4x less);
**2.00** 45.9T vs 31.2T/30.6T (1.47x); **1.95** 63.9T vs 43.5T/49.3T (1.3–1.5x);
**1.90** 87.6T vs 73.1T/81.9T (1.1–1.2x).

So with the channel the dot model is **ahead at matched compute from 10T to
~125T**, by 0.086 down to 0.004, and tied within seed noise by 150–160T. Without
the channel it is tied at 25T and behind 0.03–0.06 above 50T. The mechanism, not
the architecture family, is what buys the per-MAC win — and the win is
compute-limited-regime, closing at the top of the measured range.

All three pre-registered conditions are satisfied: 900 s confirm, scrambled
control (−0.1448), second seed. The mechanism is accepted.

## CONFIRMED AT 900 s, TWO SEEDS: extension slope −0.0120, tf gap −0.0226

`iecnn-c900-gpu`, one session, 900 s/arm, seven arms, common 255.55M tokens:

| arm | seed | tokens | s/step | bpc |
|---|---|---|---|---|
| `tf-swa-c` | 0 | 362.04M | 0.0204 | **1.7960** |
| `x5-sc3` | 0 | 258.66M | 0.0285 | 1.7825 |
| `x5-sc3` | 1 | 259.11M | 0.0285 | 1.7883 |
| `x5-sc3e` | 0 | 256.07M | 0.0288 | **1.7740** |
| `x5-sc3e` | 1 | 255.55M | 0.0289 | **1.7728** |
| `x4-sc3e` | 0 | 287.53M | 0.0257 | 1.7904 |
| `x4-sc3e` | 1 | 287.44M | 0.0257 | 1.7894 |

Slope effect −0.0085 / −0.0155 (mean **−0.0120**) for +1.1% step; seed spread
0.0058 (sc3) vs 0.0012 (sc3e). Versus tf at equal wall clock: **−0.0220 /
−0.0232**, about double the previous margin.

At matched total MACs `x5-sc3e` is ahead of tf at **every** budget (25T −0.094/
−0.100 … 150T −0.018/−0.019). `x4-sc3e` is the best arm at **low compute**
(25T: 1.9987/2.0047 vs 2.0188/2.0126 vs tf 2.1124), crossing sc3e near 100T.
MACs to 2.00 bpc: tf **48.3T**, x5-sc3e **29.1/27.1T**, x4-sc3e **24.7/26.3T**
— **1.8–2.0x less compute**. But at the 900 s endpoint x4 is +0.0164/+0.0166
behind x5-sc3e, so depth is a fixed-budget lever, not the headline.

## Voting is closed; extension-based SELECTION has headroom

`iecnn-copy-vote` on the unadapted checkpoint (40,960 positions, k=3, up to 3
earlier occurrences):

| channel | coverage | accuracy | gate gain (OOF) |
|---|---|---|---|
| single pointer | 23.1% | 62.9% | +0.0661 |
| last two must agree | 5.7% | 87.6% | +0.0325 |
| majority of 3 | 9.0% | 65.5% | +0.0368 |
| pick longest extension | 23.1% | 63.7% | +0.0743 |
| vote among max-extension | 23.1% | 64.3% | **+0.0793** |
| oracle (any occurrence right) | 15.6% | 100% | +0.1841 |

Voting fails for a concrete reason: where >=2 occurrences exist (n=3,689) the
single pointer is *already* 63.6% right, so duplicate 3-grams are the same
coincidence rather than independent evidence — majority voting fixes 2.5% of the
single pointer's errors and halves coverage. Using extension to **choose** the
occurrence fixes 5.6–7.5% at full coverage.

The oracle gap is the strategic number: 12.5% of single-wrong positions have the
right byte among their candidates and extvote finds 7.5%, so pointer selection
has ~1.4x its delivered gain unclaimed and a 2.8x ceiling. Priced post-hoc on an
unadapted model, which has over-predicted 4.5x before, so it is a screen target.

## 360 s screen: the extension gate and depth both pay (one seed, superseded)

`iecnn-ext360-gpu`, 360 s/arm, one seed, `data_fp a23bcd55c7/afc1109404`,
common 104.08M tokens: tf **1.8880**, `x5-sc3` **1.8354**, `x5-sc3e`
**1.8208**, `x4-sc3` **1.8451**, `x4-sc3e` **1.8322**.

- **Extension slope** (`sc3e`): −0.0146 at depth 5, −0.0129 at depth 4, for
  +1.1–1.2% step. Above seed noise, consistent across depths.
- **Depth 5→4** (`x4`): +0.0097 / +0.0114 bpc at each arm's own endpoint for
  −11% step and −16% params, i.e. ~13% more tokens in the same clock. Per MAC
  it is not a loss: at 10T `x4-sc3e` 2.1788 vs `x5-sc3e` 2.2210, at 20T 2.0312
  vs 2.0486, at 40T 1.9432 vs 1.9531, crossing only at ~50T (1.9320 vs 1.9287).
- Per-MAC deltas vs tf at 10/20/30/40/50/60T: `x5-sc3e` −0.116/−0.100/−0.090/
  −0.079/−0.059/−0.048; `x4-sc3e` −0.159/−0.118/−0.099/−0.089/−0.056/−0.043.

Confirm queued: 900 s, two seeds each for `x5-sc3`, `x5-sc3e`, `x4-sc3e`,
against an in-session `tf-swa-c`.

## suffix_copy pays at 360 s, and the dot family now beats the transformer

One healthy session, 360 s/arm, one seed, `data_fp a23bcd55c7/afc1109404`:

| arm | params | tokens | s/step | bpc | bpc@eqtok |
|---|---|---|---|---|---|
| `tf-swa-c` | 408,288 | 154.06M | 0.0192 | **1.8810** | 2.0209 |
| `tf-swa-w3-c` (horizon 7) | 408,288 | 157.89M | 0.0187 | 2.0787 | 2.1554 |
| `tf-swa-w8-c` (horizon 22) | 408,288 | 157.28M | 0.0188 | 2.0041 | 2.0859 |
| `x5+rr2-c` | 835,376 | 107.01M | 0.0276 | 1.8994 | 1.9635 |
| **`x5+rr2-c-sc3`** | 835,377 | 101.90M | 0.0290 | **1.8339** | **1.8907** |

`suffix_copy` buys **−0.0655** at equal wall clock and **−0.0728** at equal
tokens for **+5.1%** per-step compute, and puts the dot family **0.047 ahead of
the transformer** at equal wall clock. The pointer audit of the sc3 arm is
identical to its control (dots still at `dist_back` [4.0, 1.99, 0.0]), so the
gain is a new path, not a rescue of the collapsed dot.

The **scrambled-pointer audit has passed** (`iecnn-scramble-check`): with every
parameter, the mixture and the learned gate (0.2616) held fixed, replacing the
pointer with a random earlier position costs **−0.1448 bpc** (1.8339 → 1.9787)
and is **worse than forcing the gate shut** (1.8930). The gain is in the band:
right d 9-64 goes 1.2011 → 0.6053, while wrong guesses get *worse* (+0.0760 on
wrong d>8) — a scalar gate cannot see correctness and pays for its false
positives. So the margin is the **match**, not a peaked prior.

It also demonstrates the post-hoc caveat: on the model trained with the channel
the best constant gate is **0.00** (mixing now hurts) even though shutting the
gate costs 0.059 — the value is in the weights, so offline ceilings
under-estimate a trained channel (ceiling 0.052, delivered 0.0655).

One pre-registered gate remains: the **900 s confirm with seed-1 twins** in one
session (`iecnn-sc9-gpu`, running).

The horizon ladder also landed: truncating the transformer to a 7- or 22-token
horizon costs **+0.198 / +0.123**, far more than the dot model's whole gap, so
exact long-range access is where the value is — the premise the copy channel
exploits.

**The channel's ceiling, measured before the screen finished** (offline, no
training, `scripts/probe_copy_ceiling.py`): the 3-gram copy is valid at **98.1%**
of positions and correct at **52.1%** of those, and mixing it into the trained
model's own distribution at the best constant gate is worth **+0.0874 bpc** on
x5+rr2-c inside the model's own 256-token window (1.8383 → 1.7499 at g=0.18;
unwindowed it would be +0.0984, and only **36.7%** of positions have their most
recent 3-gram match inside 256), **+0.0949** on rr2-c, and **+0.0424** even on
the transformer. The **implemented** channel (the model's own `_suffix_copy` on
the same windows) is lower: 23.3% valid but 63.0% right where valid, ceiling
**+0.0515** on x5 / **+0.0575** on rr2-c — the corpus row credits matches that
sit before the window's start, which the arm cannot see. With a **random earlier pointer** the best gate is 0.00 and
the gain is **0.0000** on all three, so the margin is induction, not a peaked
prior. A per-position gate would be worth **+0.519** on x5, i.e. a constant gate
leaves most of the channel unused — that is the follow-up, not the first test.
A gate fitted only on **observable** features (match distance × model entropy,
2-fold out-of-fold) is worth **+0.1235**, already above the constant gate's
0.098, so the follow-up has a concrete form — **but only on corpus pointers.** Against
the channel the arm actually runs (in-window matches only), the same 2-fold
feature gate adds **+0.007** over the scalar (+0.0582 vs +0.0515 on x5), which is
inside the screen's noise: the scalar captures the channel, and the feature gate
is **deprioritized by measurement**. The rest of the oracle's 0.52 is only
reachable by knowing "is this guess right", which is the target itself.
Caveat: p does not adapt here, so this is an upper bound for a trained arm.

**Third pre-registration, for the `suffix_copy` screen** (`tf-swa-c`,
`tf-swa-w3-c`, `tf-swa-w8-c`, `x5+rr2-c`, `x5+rr2-c-sc3`, 360 s, one session,
one seed). The mechanism is a single learned gate on an exact 3-gram pointer, so
the screen can only do three things:

| outcome (x5+rr2-c-sc3 vs x5+rr2-c, same s/step) | reading |
|---|---|
| worse or flat (≥ −0.01) | the channel does not pay at this budget: reject it, log it, and do not tune the gate — the tests already rule out a silent no-op, so a flat result is a real verdict |
| better by 0.01–0.04 | a peaked-prior effect is not excluded; run `scripts/probe_suffix_copy.py` on the checkpoint (true vs scrambled pointers) before any further spend |
| better by > 0.04 | candidate, not result: 900 s confirm plus the scrambled-pointer control, and only then a second seed |

The point of the screen is that **`scrambled` must be worse than `true`** for
this to be induction rather than a mixture trick. If true ≈ scrambled on the
trained checkpoint, the mechanism is closed regardless of the bpc margin.

**The pre-registration below was written before the numbers, and the numbers
came back on the first branch -- so the reading is fixed, not chosen: head
capacity does not reach the band.** tf − x5 is 0.0460 total and the 9–64 band
contributes **−0.0531 (115% of the gap)**, while only **18%** of the rr2 → x5
gain lands in the band (44% is "wrong guess d>8", where x5 is now **better than
tf** by 0.0200, and 33% is "right d>64"). So the extra two head layers buy
mostly positions where induction *fails*, and the band still sits at **1.105
bpc against tf's 0.593** on 10.4% of positions. Consequences, both recorded:
**do not buy more width/depth for the band** (that includes the staged
`x5w768l4`/`x5w1024l3`), and the band is now a *read* problem whose defining
property is addressing the previous occurrence of the *context* — which no arm
in the family does directly.

Original pre-registration (`iecnn-gap-s900`, dataset `iecnn-ckpt-s900`): The 3 h result was
measured with a 3-layer head; the 900 s frontier arm has 5, and its gap is four
times smaller (0.046 vs 0.115). So the induction bucket table can separate
"capacity buys induction" from "the band is the read":

| what the pairwise table shows | reading |
|---|---|
| the 9–64 band still carries most of the **tf − x5** gap, and most of the **rr2 → x5** gain is outside it | head capacity does NOT reach the band. Stop buying width/depth for this; the band needs a different read (and that read is the next mechanism to design, not another arm in this family) |
| most of **rr2 → x5**'s gain falls in the 9–64 band | capacity does buy induction; further capacity is justified, and the width-by-depth arms are the right purchase |
| the band's share shrinks with the head for both reasons (band empties AND the rest shrinks) | mixed; report the two shares separately, no single-number claim |

Either way the result is a **causal check on the copy-distance story**, because
x5 and rr2 differ only in head depth, not in what the dots read. That is the
cleanest in-family control available without retraining.

The compute map (`scripts/flops_audit.py`) says why this is the right question and
not another capacity purchase: `x5+rr2-c` is **1.93×** the tf anchor's MACs/token
and **79% of it is the post-MLP head**, while the window read is **0.14%** and the
shared key/value projections 0.7%. So the read is cheap to *extend* and the head
is expensive: doubling the head costs ~50% and +655k params (`w1024` =
**3.45×** the anchor). But cheap is not sufficient — the longer contiguous
window is measured dead (see the induction entry above), so the compute map's
actionable consequence is the opposite direction: **width is the expensive
thing, so if capacity is bought it should be bought by trading depth away**
(`x5w768l4` / `x5w1024l3`, both 2.23× the anchor against x5's 1.93×).

One thing this does **not** measure: the dot model's horizon is not really 8.
Its window read is 8 exact slots, but the global decayed read is a prefix
attention with `g_decay` initialised at sigmoid(4.6) ≈ 0.99 and learned, i.e. a
fuzzy memory of ~100+ positions. So these arms price *exact* access beyond 7
positions; a large tf-w3 penalty would mean exact long-range access matters,
not that the dot model can only see 8 tokens.

The same session runs `x5+rr2-c`, so all three tf points and the dot point share
one machine. The population audit (`scripts/probe_population.py`) then answers
the other half — whether three dots are one dot's worth of function — on the 3 h
checkpoint, with no retraining.

## History — 2026-11-20: the constraint was the HEAD, and the gap halved

Three results landed, all on the canonical 90/5 split with an in-session
transformer anchor. They change what the campaign is doing.

**1. Head capacity pays — the first arm to move TRAIN fit.** 900 s/arm,
compiled, common 272.23M tokens:

| arm | params | s/step | tokens | bpc | bpc@eqtok | train |
|---|---|---|---|---|---|---|
| tf-swa-c | 408k | 0.0185 | 398.41M | **1.7923** | 1.8666 | 1.7920 |
| rr2-c | 571k | 0.0206 | 357.83M | 1.8847 | 1.9451 | 1.8677 |
| rr3-c | 571k | 0.0224 | 329.65M | 1.8769 | 1.9369 | 1.8682 |
| **x5+rr2-c** | 835k | 0.0271 | 272.23M | **1.8383** | **1.8894** | **1.8183** |

x5 adds 264k params, costs 31% more per step and therefore sees 24% fewer
tokens — and still wins by **0.046 wall / 0.056 equal-token**, with train bpc
down 0.049. The in-session gap to tf goes **0.092 → 0.046** (wall) and
**0.079 → 0.023** (tokens). So "the head is not the constraint" is dead: the
dot model was underfitting in its head stack, not only in its dots. rr3 is a
null against rr2 (-0.0078, one seed, inside the dot noise floor).

**2. The channels are asymmetric: value is full, key is 4× too wide.** Exact
subspace truncation at the weights on 3 h rr2-c (base and restored both
1.8276): `v` 32 dims cost +0.633 at 24/32 and +1.78 at 16/32 — every dim is
load-bearing, so there is no slack to reclaim. `k` 16 dims hold **80% of their
variance in ONE dimension** and PCA-4/16 costs +0.008 — it is over-provisioned,
not starved. `s` 96 is distributed (PR 41). Random/prefix bases are always
worse than PCA at equal k, so the channel is organised. Do not read "in use"
as "insufficient": the `d_v=64` arm is the direct test of whether more width
helps, and it is running.

**3. Generation does not support the dot model.** 20 unseen prompts, matched
seeds. rr2's text is more varied (distinct-2 0.583 vs 0.491) and a judge liked
one pair of it — but the cross-scoring kills the claim: tf finds rr2's samples
*less* likely (1.587) than its own (1.188), and prefers its own sample on 19/20
prompts. A blind judge over 20 balanced pairs splits **11-9 for tf (p=0.82)**.
The extra variety is what higher entropy looks like at fixed temperature
(rr2 H 1.243 nats vs tf 1.174, both calibrated at ECE ≈ 0.007). bpc stays the
metric; generation is not an architecture argument.

**Width, not depth.** A 9.1M-token screen separates the two: halving the head
block to w256 costs **+0.074 bpc** (x5 2.2436 → 2.3177), while adding two more
blocks at w512 buys **0.0075** for 131k params and 5.6% step time. `head_post=512`
stays; more depth is not a lane.

**The collapse is an input path, not a dead specialist.** With `n_passes=1` the
cell's input is `[ctx, 0, 0, round]`, and `ctx` is a 32-dim softmax average of
window *values* — so the only route from the current token into the state is a
dot that attends to distance 0. That is exactly the dot every audit finds
collapsed (`frac_end` 0.97–1.00), and it costs 1/3 of the population. `input_res`
adds `v(t)` to every dot's ctx: zero parameters, one add, flag-off
bit-identical. Running at 900 s against the x5 control — the audit on that arm
answers "does the third dot come back?" independently of machine speed.

**Machine caveat — WITHDRAWN, the "throttled hosts" were a marker bug.** The
same `tf-swa-c` arm that does 0.0185 s/step did 0.336 s/step in sessions pushed
with `mark_s=0`, and `MARK_S=0` made `el >= next_mark` true at **every** step:
each session ran a 160-window validation inside every training step. Every slow
log has `len(curve) == steps`; the one healthy log has `len(curve) == 1`, and
the arithmetic matches exactly (tf 0.3363 − 0.0176 = 0.319 s of validation per
step; x5 0.7623 − 0.027 = 0.735, ratio 2.3 ≈ 5 head layers vs 1). No host
throttling is evidenced. Fixed with a `MARK_S > 0` guard; `ARM_SLOW_GATE` is
kept as a **rate** test whose message points at per-step validation first.

## History — 2026-11-19 snapshot (numbers superseded by the table above)

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

**Resolved at 360 s: the lag is innocent and the content query is the cost.**
Same session (`dis1`, common 104.01M tokens): x3/w512 **2.0050**; the missing
control `qi0` (frontier + `query_from_input`, **no lag**) **2.0429**; `kl1-nx`
(the same query **plus** the lag) **2.0312**. So the shared content query alone
costs **+0.0379** (and +0.0491 at equal tokens), and adding the lag *recovers*
**0.0117** of it (0.0168 at equal tokens). bind2's 0.0196 was the query with
the induction gain only partly offsetting it — not a price paid for the lag.

The audit also kills the "the committee collapsed" explanation: `qi0` keeps
distinct_per_pos 2.72, `kl1-nx` drops it to 2.00 with `all_same` 0.126, yet
`kl1-nx` is the *better* arm. Dot distinctness does not track bpc here, the
same dissociation seen with `query_from_input` (batch 24: redundancy fell while
bpc rose) and N=4.

What remains open is narrow and bounded by measurement: the lag's whole value
here is the **0.0117** it recovers, and the region-wide shape applies the
query to all three dots (the +0.0379 cost). A per-dot version (one content
query + one lag on the far dot only, near dots bit-identical) can at best keep
that 0.0117 while avoiding most of the cost — worth one arm only if its step
overhead is ~1-2%, not a lane.

So the probe's lesson is narrower than it looked: the model **cannot** bind,
but at this scale an induction read is worth ~0.012 bpc, and the only
implementation tried so far pays +0.038 for it by broadcasting one content
query to every dot. Binding is not where the ceiling is.

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

## Long-corpus protocol (2026-11-19)

From here on, runs use the **canonical enwik8 split**: train = bytes [0, 90M),
val = bytes [90M, 95M), last 5 MB never touched. That is +50% training data over
the 60/5 MB window the first 173 rows used, and it moves the val slice, so
absolute bpc is **not comparable** with those rows: every session re-runs its
own anchor and claims stay in-session.

At 3 h on this corpus, a 900 s-era step time (0.0207 s) gives ~521k steps and
**4.27B tokens, i.e. ~47 passes** over the 90 MB train split. So "more epochs"
is satisfied by the budget, not by a loop change: the harness streams random
windows and `epochs` was never a knob.

Every long run also reports **generation**: 240 bytes decoded greedily and with
top-k 40 / T 0.7 from the first 120 bytes of the val split (seen by neither
arm), printed beside the true continuation. One decode path
(`models/generate.py`) is shared by the harness and the probes, so a sample
from a run and a sample from a probe cannot diverge.

Failure-proofing: every kernel runs `ARM_DRY=1` first -- build each arm, one
fwd/bwd, one val pass, the pointer audit, a generation -- and refuses to start
the timed run unless all arms pass. The dry pass now places the model on `DEV`
(it used to build a CPU model, feed it a CUDA batch and "fail" on exactly the
machine the check exists for).

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

At 900 s **compiled** (fr9, seed 0, common 355.75M tokens):

| arm | params | s/step | steps | tokens | bpc | bpc@eqtok | train |
|---|---|---|---|---|---|---|---|
| tf-swa-c | 408k | 0.0202 | 44,569 | 365.11M | **1.8293** | 1.8790 | 1.7885 |
| **d128 ×3, rr2-c** | 571k | 0.0207 | 43,427 | 355.75M | **1.9282** | 1.9764 | 1.8579 |
| d128 ×3, x3-c | 559k | 0.0192 | 46,825 | 383.59M | 1.9688 | 2.0200 | 1.8990 |

**Two seeds at 900 s (s2r9, seed 1, common 285.76M): tf 1.8733 · x3 1.9681 ·
rr2 1.9342.** rr2 − x3 = **+0.0339** here and +0.0329 in long5, with the dot
arms moving only 0.001–0.002 between seeds while tf moves **0.0333**. So reread
is now a two-seed claim (+0.033 ± 0.001), and the uncertainty in "the gap to
tf" (0.061–0.093) lives in the transformer arm, not in the dot arms. The
mechanism is kept.

One structural observation replicates too: in every session exactly one dot
collapses onto the window end (dist_back ~0.0, frac_end ~0.97), but *which*
dot dies is seed-dependent (dot 3 in long5, dot 2 in s2r9, in both the x3 and
rr2 arms). The population therefore runs at **~2 effective specialists out of
3**, which is a symmetry/initialisation fact, not a diversity score.

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
**N=1** (the cheapest dot is already worth 0.34 bpc on drop, so one dot starts
far behind -- the population is a fusion, see the 2026-11-21 row);
`query_from_input`/`gla`; `ngram_bias`; `head_rank48`; `id_lags`; `head_skip`;
`gate_agree`; **`dist_bias`** (0.0045 < noise); `global_exact` (`gx`);
**`query_from_input` (re-confirmed: `qi0` +0.0379 vs x3)**; the
**content-query + uniform-lag bundle** (`kl1-nx` 2.0312 vs 2.0050; inside it the
lag *recovers* 0.0117 rather than costing).

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

**`scn1` shows the boundary works but keeps only ~40% of the win.** 360 s, same
session: tf 1.9639 -> tf-c 1.8960 (step −22.4%, +29% steps); x3 2.0065 ->
x3-c 2.0003 (step −10.7%, +12% steps). So the dot model now compiles and gains,
but the graph break costs it about half of what compile gives the transformer.

The reframing that matters: **x3-c is now 7.6% faster per step than tf-c and
sees 7% more tokens, yet is 0.104 bpc behind.** The speed claim survives against
a compiled transformer, but only by ~8% — the gap is now entirely quality per
token, not throughput. (And tf-c is 0.028 better at *equal tokens*, replicated
across two sessions, which compile should not do; flagged, unexplained.)

**`hs1` kills that inference.** The Hillis-Steele scan removes the boundary
entirely, and it buys **1.4% of step, 0 bpc**: x3 0.0207 -> x3-c 0.0176
(−15.0%) -> x3-hs-c 0.0173 (−16.4%), against tf's −28.7%. So the graph break was
**never** the missing half; the dot model simply compiles less well (window
gathers and scatter-adds dominate). The lane is closed: keep `cumsum` +
`break_scan` (simpler, and `scan_mode` stays as a tested flag).

**`fr9` (900 s, compiled) sets the real head-to-head.** tf-c 1.8293 vs
rr2-c **1.9282** — a **0.0989** gap, and rr2-c is now **2.5% SLOWER per step**
than tf-c (0.0207 vs 0.0202) because reread's mixing is not free. The 360 s
"dot is faster per step than compiled tf" result was measured on x3 *without*
reread; adding the one mechanism that pays consumes it. Compile still helps
reread (−0.0051 bpc, −22% step), so the frontier is `rr2-c` at 1.9282.

## Running / queued

- `dis1` (done) — tf 1.9833 / x3 2.0050 / **`qi0` 2.0429** / `kl1-nx` 2.0312.
  The shared content query is the cost (+0.0379); the lag recovers 0.0117.
- `fr9` (done, 900 s) — **tf-c 1.8293 / x3-c 1.9688 / rr2-c 1.9282**: the
  compiled frontier. rr2-c is the best dot arm so far; the gap to tf-c is
  0.0989 at 2.5% *slower* steps, so the campaign no longer has a throughput
  claim once reread is in.
- `s2r9` (done, 900 s, seed 1) — tf 1.8733 / x3 1.9681 / rr2 1.9342: reread is
  a two-seed claim (+0.0339 here, +0.0329 in long5).
- `hs1` (done, 360 s) — the Hillis scan is a null: −1.4% step, +0.003 bpc. The
  eager boundary was never the cost; the scan lane is closed.
- `t3h` (done, 10,800 s/arm) — **tf-c 1.7123 vs rr2-c 1.8276**: gap 0.1153 at 4.25B
  tokens (~47 passes). Train bpc tf 1.7119 vs rr2 1.8073 — the dot model is
  behind on FITTING, not on generalising. Generation at 3 h is wiki-shaped for
  both, contentless for both. One dot still collapsed (dot 0 this time).
- `lad2` (done, 360 s, 90/5) — **the `reread_spans` ladder is a NULL**: rr3
  (8/8/8) 1.9345, rr3-lad (2/8/32) 1.9402, rr3-u32 (32/32/32) 1.9543, tf 1.9377.
  A longer horizon buys +0.0038 at equal tokens and costs 0.0057 at equal wall
  clock; uniform 32 costs 0.0198. So reread's gain is another mixing op at the
  same scale, not a missing horizon. Scale hypothesis closed; count (rr2 vs
  rr3 on the same corpus) is the open question.
- **implemented, NOT launched (needs authorization)** — `reread_spans`: one
  exact horizon per reread site. The stack currently reads span 8 at every
  site, which is the dot window's own scale, so the model has window-8 and the
  global decayed bag and nothing exact in between. Arms: `rr3` (8/8/8),
  `rr3-u32` (32/32/32) and `rr3-lad` (2/8/32) — same site count, same
  parameters (rr_* are shared), only the horizons differ, so the session
  separates "more mixing" from "a new scale". Tests pass locally, Kaggle
  suite pending. One 360 s session, 4 arms with the anchor.
- then: the Hillis-Steele scan if the ~10% step is worth chasing, a flavour
  rerun on the saved 900 s weights, and the per-dot lag only if bounded work
  turns up (it is capped at ~0.012).

## The thing to be careful about

Provisional, not settled: **"we beat the transformer"** (true only below ~250 s
of wall clock), the **reread** mechanism (a 360 s *and* 900 s win at one seed;
needs a second seed before it is a claim), and the compile win for the dot
model (compiles with `break_scan`, but keeps only ~40% of tf's win).
Everything else in the ledger has either two seeds or a margin that clears the
within-session noise. The `k_lag` question is settled, in the opposite direction
to the first reading.
