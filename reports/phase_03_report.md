# Phase 3 report — the core IECNN (dots, memory, regions, consensus, messages)

Commit range `0f1d5d2..d31f22a` (+ later timing fixes). Date: 2026-11-01/02.

## 1. What was built

Five modules, all spec-first (files 01–05 of the spec):

| File | Contents | Key spec points honored |
|---|---|---|
| `models/dot.py` | `DotPopulation` — N dots stored **stacked** on a dot axis `[P, N, in, out]`, computed with batched ops; hand-written GRU (built-ins cannot hold per-dot weights) | per-dot q_k / s0_k / GRU / output proj P_k / confidence head / trust-gate head; shared relative-position key embedding; round-signal embedding (16-d); error-feature input (3-d); `dynamic_query` (U_k s_prev); `share_pass_params` (D6); states reset every position (C1); output distribution through the **tied** embedding E [V,d] |
| `models/memory.py` | fixed-window memory: banded causal mask, sliding-window gather, region chaining `LN(Proj(h)+prev)` , relative embeddings, region-1 rows = plain embeddings (position NOT added to rows — it lives only in keys) | spec 02 §1–3 exactly; masked rows get −inf attention, not zero-filled |
| `models/consensus.py` | pure label-free functions: leave-one-out crowd mean, JS divergence, error features, scores, trust accumulation (T^p = ρT^{p−1}+u), softmax weights with floor+renormalize, log-linear and mixture pooling, `hard_top_m`, temperature schedule | spec 04 §2–6; scores clamp ±20 before softmax (§9); C2 (no labels anywhere) |
| `models/region.py` | N dots × P passes loop: pooled opinion → pass state → weights → per-pass consensus → region summary; hard mode; messages hook | spec 03; temperature τ_p = τ0·κ^(p−1); weight floor ε = 0.1/N in training, 0 late; hard vote = inference-only (C5) |
| `models/messages.py` | ring-neighbour message channel: write tanh(W_m s), read attention over ring; cyclic ring distance; one-pass delay structure | spec 05; no self-loops; `use_messages: false` builds no channel at all (D9) |
| `models/iecnn.py` | tied embedding; R chained regions (each reads previous region's summary-residual memory); cross-region competition ω_r ∝ α·(−entropy b_r)+gate_r, log-linear pooling across regions, re-normalized | spec 03 §competition (D5) |

## 2. Trainer wiring (changes to Phase-2 code)

- `build_model` now routes `kind: iecnn`.
- IECNN training kwargs per step: `tau0` (cosine anneal from `tau_max` after `tau_warmup` steps down to `tau_min`), `floor` (0.1/N dots, removed over the last 10% of training per spec 04 §3 — actually: floor fraction times `weight_floor_at(step,...)`), `hard_mode` (fraction `hard_batch_fraction` of batches after 70% of training, chosen by a dedicated RNG stream so resume stays deterministic).
- Aux losses (file 06 §2) wired **only when the rung config supplies lambdas**; no silent defaults. E0 config: none. E1 config: λ_d 0.1 (boosted per-dot with γ_b=2, crowd-wrongness weight detached), λ_k 0.05 (confidence calibration), λ_w 0.05 (weight-entropy bonus decaying to 0 by 20% of training). λ_r/λ_c remain 0 until E2/E4.
- Final evaluation additionally computes **hard-mode bpc** and logs `final_hard_bpc` / `mode_gap` into `summary.json` (spec 08 headline metric).
- `evaluate_bpc` now forwards `tau0`/`floor` and toggles hard mode for IECNN models.

## 3. Tests (28 new, all passing)

Spec test lists realized as: dot tests (shapes/sums, causality = 0.0 bit-exact change when future tokens change, every dot receives gradient, query cosine < 0.5 at init by QR construction, same-seed determinism, label-free C2 replay); memory tests (future leakage, zero attention on masked rows, parallel == token-by-token inference within 1e-3, chaining Proj ablation changes outputs); region tests (weights sum to 1, **E0 is bit-exactly the plain uniform ensemble** — arithmetic mean of dot distributions, round-signal matters, unsharing scales parameter count by exactly P, valid final distribution); consensus tests (floor renormalization closed form — after clamping to floor f and renormalizing, each weight ≥ f/(1+N·f), documented in the test; tau→0 one-hot, tau→∞ uniform; loo excludes only the dot's own pi and its own row uses only the others; identical dots make log-linear == mixture; top-m with m=N equals soft; temperature endpoints); messages tests (empty diagonal — no dot reads its own write, verified numerically as uniform-mean-of-neighbours with zeroed queries; one-pass delay — pass-1 output is bit-identical with the channel live vs zeroed, pass 2 differs; exact ring wrap for N=1/N=6 radius 2).

Full suite: **60 passed, 6 skipped** (remaining skips are Phase 4+ roadmap stubs).

Subtleties recorded in test docstrings (honesty notes):
- floors: after clamp+renormalize the minimum weight is f/(1+N·f) < f; the post-normalization weights can sit **below** the requested floor — this is the literal spec formula (clamp, then renormalize), not a bug.
- leave-one-out: when dot k's pi changes, k's own crowd row is unchanged (it excludes pi_k), but the *other* dots' crowd rows do change (they contain pi_k). Both facts are asserted.

## 4. Bugs found and fixed during the build

1. `DotPopulation.w_ih/w_hh/p_out` were accidentally stacked without the N axis (shape `[P,in,out]`); every dot shared GRU/output weights. Caught by a param-shape probe; fixed to `[P,N,...]` with the same generator stream.
2. `uniform_weights` built an `[..., N, V]`-shaped tensor (mis-counted trailing dims) → pool shapes exploded.
3. `gather_windows` used `Tensor.gather` with a wrongly-shaped index; replaced by fancy indexing `rows[:, idx]` which is correct and simpler.
4. `MessageChannel.ring_neighbors` used linear distance — the ring's distance is **cyclic** `min(|i−j|, N−|i−j|)`; fixed (recorded in DECISIONS.md).
5. **Cross-region competition einsum had a double bug** (caught by the memory test suite, not visible anywhere else): the stacked per-region log-probs were assembled `[B,R,T,V]` but pooled with `btr,brv->btv` — wrong axes AND wrong pairing; correct equation `btr,brtv->btv`. This would have silently corrupted every multi-region (E2+) run's final distribution; region 1-only rungs never touched the code path, which is why it took a test to find it.
6. `train.yml` bash arithmetic: `${CHUNKMIN:-320}` fills the default only on first use, so the ternary read empty → `expression expected: unbound variable` → **all three B0 ladder jobs crashed in their first minute** (caught on the runner, not locally).

## 5. Full-size runs on Actions (all real training on GitHub, per the standing rule)

| Run | config | result |
|---|---|---|
| B0 ladder `-r1` (seeds 0,1,2, 50k target) | b0_gru | **failed in minute 1** — workflow bash bug above; fixed; re-dispatched as `-r2` (in progress at time of writing) |
| E0 timing `-timing-E` | e0 full size | 5 steps OK; 1,257,892 params; **init loss 6.238 = ln 512 exactly** (tied-head init health confirmed at full size); 3.9 s/step incl. warmup/eval amortization |
| E1 timing `-timing-E` | e1 full size (P=3, weights on) | 5 steps OK; **~42 s/step** wall — flagged below; after the consensus-dedup fix re-measured at **22.1 s/step** (742 tok/s) |
| E0 timing `-timing2/3-E` | e0 full size | re-measured after the dedup: 3586 tok/s (an accidental extra JS computation in single-pass rungs had regressed it to 2775; fixed) |
| E1 component profile `-prof5-E1` | e1 full size | breakdown in §6 |

## 6. The E1 slowness: measured breakdown and what was done within Phase 3

Component timing on the runner (2 threads, full size, 4 timed steps):

| component | ms/step |
|---|---|
| dots_forward (3 passes) | 3,732 |
| consensus (3 passes) | 3,283 |
| forward, real graph (incl. competition, log_softmax, gathers) | 11,099 |
| aux losses | 73 |
| backward | 19,756 |
| **total** | **30,511** |

Two real inefficiencies identified and one fixed in-place:

- **Consensus computed twice per pass** — `error_features` recomputed the leave-one-out mean and JS that `compute_scores` had just built. Fixed by having `compute_scores` return its parts and the next pass's error feature reuse them (`error_features_from_parts`). Same numbers algebraically; all 28 core tests re-pass. Dispatched re-timing (`-timing2-E`) to measure the gain.
- **Remaining suspicion: grouped einsum paths** in `dots_forward` (GRU in-gates/head logits over the N axis) — likely 2–3× slower than plain bmm equivalents. NOT yet changed; changing them alters float summation order slightly, so it needs its own before/after equivalence check (planned for the top of Phase 4, with a test).

Also learned: two profile runs died with "runner shutdown signal" ~100 s in. Root cause was memory exhaustion of the 7 GB hosted runner by the profile script itself (warmup graph retained while building a second full graph). The **real trainer is the tightest constraint**: it survived full-size E1 5-step runs, but the margin is thin — memory-lean variants (e.g. not retaining per-pass `log_pi` in the exportable dict) are listed for Phase 4 before long E-run chains.

### Ladder implication (decision pushed to Checkpoint B)

- E0 measured ≈ 3.9 s/step (3586 tok/s incl. amortized eval/startup) → 50k steps ≈ **54 h/seed** (projection was 35–90 h: inside).
- E1 measured 42 → **22 s/step** after the dedup fix (742 tok/s; pure compute ≈ 300 s / 12 steps... measured 5-step wall 110 s = 5×22.1) → 50k steps ≈ **~300 h/seed** — outside the 35–90 h projection, ~10× a B1 seed, and ~2 weeks for the 3-seed ladder even fully parallel.
- Options presented at Checkpoint B: (a) run all rungs at a fixed 50k steps (honest, slow, ~2.5 weeks for E1); (b) equal-compute stepping (E-rungs get fewer steps to match B0/B1 measured FLOPs — spec-faithful "equal compute" but trains on less text); (c) more optimization first (bmm rewrites, maybe torch.compile) with equivalence tests, aiming E1 ≤ E0×(passes) ≈ 12 s/step ≈ 170 h/seed. Agent recommendation: (c) then re-measure, and only then pick (a) vs (b) with real numbers.

## 6b. Tolu chose (c): optimization round, re-measured (2026-11-02)

Three changes, each pinned by tests, dispatched and measured on Actions:

1. **Fast torch-dot path** (`models/dot.py`, `fast_dot_math` flag, default ON): the three heavy per-dot contractions (GRU in-gates over inputs, GRU hidden gates, tied head logits) rewritten from grouped einsums to plain batched GEMMs over a merged `[B*T, N]` layout; einsum version kept as `_forward_reference`. Equivalence test `test_pyi_fast_math_matches_reference` pins fast vs reference (incl. the dynamic-query variant and mask handling); suite went 28 → 61 tests after this session's additions.
2. **torch.profiler op table** added to `train/profile_step.py` (one real fwd+bwd).
3. **JS log-reuse** (`models/consensus.py`): `js_divergence` accepts the caller's already-computed `log_pi`, saving one `log()` over the `[., N, V]` tensor per pass.

Op-table reading (E1, contended runner): bmm 5.8 s, then mul/log/div/copy_/add/sub/neg/where ≈ elementwise chains over the `[f, N, V]` consensus tensors — that + backward is now the floor; further gains would need fusing (torch.compile), which re-compiles on float scalars and is kept out for now.

**Re-measured (timing4 → timing5, same contended conditions): 17.4 → 16.6 s/step (946 → 989 tok/s).** Below our own "worth another pass if ≥ ~1200 tok/s" bar — optimization stops here.

### Ladder cost with real numbers (Checkpoint B decision data)

| rung | s/step | 50k steps/seed |
|---|---|---|
| B0 | 0.96 (measured, `-r2` chunk 1) | ~13 h |
| E0 | 3.9 | ~54 h |
| E1 | 16.6 | ~230 h (~10 days, ~27 chained chunks/seed) |

### Second real training-signal finding (from the B0 `-r2` chunk-1 CSVs)

B0 trains 20k steps in chunk 1: train loss → 0.25 while **in-loop val bpc rises monotonically 1.95 (step 1k) → 5.43 (step 20k)** — genuine overfitting on the 540k-token corpus, not eval breakage (smooth curve, train loss falling). At ~14k tokens/step, 50k steps ≈ 3,900 epochs — far past memorization. Consequence implemented in `train/train.py`: **best-checkpoint tracking** — the trainer keeps the best in-loop point's weights and at the end re-scores that point on the FULL validation text (`best_full_val_bpc`, `best_step`, also hard-mode gap at the best point); the headline rung number is the best point, not the most-overfit final state. Resume carries `best_state` through chained chunks.

Also fixed (found breaking seed 0's chunk-1 commit step): results commit now `git pull --rebase --autostash`; the chain itself was unaffected (chain-control step succeeded and chunk 2 fired on schedule at 04:28 for all 3 seeds).

## 7. Workflow (train.yml) upgrades for long runs

- Seeds became a **matrix** (one job per seed, parallel).
- Self-chaining chunks: each job trains up to 320 min; only exit code 124 (timeout) triggers an automatic re-dispatch with `resume_artifact={config}-seeds-{seed}{suffix}`; **real crashes exit non-zero and fail loudly — no chaining on failure** (avoids infinite retry loops; decided and recorded).
- Artifacts have stable names per (rung, seed) and are overwritten each chunk (`overwrite: true`, retention 30 days).
- `timeout-minutes: 348` (hosted cap 360).
- The resume step restores `checkpoints/` before `train.train` auto-`--resume`s from the checkpoint in the out dir (resume itself was verified bit-identical in Phase 2; the chain handoff is being exercised right now by the B0 `-r2` ladder run).
- E0/E1 timing, the profile harness, and the B0 ladder all dispatched through it live.

## 8. Honest state

- The IECNN model is spec-complete for structure but has **not yet been trained long enough to move bpc**; everything "validation" so far is 5-step sanity (init-loss health, decreasing-ish small numbers, mode_gap ≈ 0 at init as expected since hard top-4-of-4 with m=N equals soft at tiny weight differences... in practice hard_top_m=4 of N=4 dots ⇒ identity, so mode gap ≈ 0 exactly at init).
- The workflow-chain handoff (6 h cap → artifact → resume) is dispatched and watched but not yet confirmed through a full 50k rung.
- E1 wall-clock harbor: implementation-bound until the bmm rewrite lands; recorded so nobody mistakes 42 s/step for intrinsic model cost.
