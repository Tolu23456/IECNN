# Why the IECNN is behind a plain transformer — full mechanism comparison

Written 2026-11-18. Every number below is measured in this repo (results/*.csv,
run logs); every mechanism claim is backed by the file/line cited.

---

## 0. The numbers first (honest baseline)

| | b1 transformer | e11b (current IECNN champion lane) | best IECNN ever |
|---|---|---|---|
| params | 1.67M (untied head) | 1.717M | 1.717M |
| batch × ctx | 8 × 256 = 2,048 tok/step | 64 × 256 = 16,384 tok/step | same |
| best bpc | **1.7658** @ step 6,240 | **2.12792** @ step 3,000 (still falling) | **1.82994** (E6) / 1.85999 (E8) |
| tokens to get there | 12.8M | 49.2M (and not there yet) | ~140M (E8 @ 8,520 steps) |
| wall clock | **19 min** (1,133 s) | 5.3 h so far | days (chunked Actions) |
| s/step | **0.182** | 6.4–11.4 | ~13.5 |
| ms per token | **0.089** | 0.39–0.70 | ~0.82 |

Two different gaps, and they must not be conflated:

1. **bpc at convergence: small.** The best IECNN ever is 1.82994 vs the
   transformer's 1.7658 = **+0.064 bpc (3.6% relative)**. e11b's 2.128 is a
   3k-step run still descending ~0.1 bpc/1000 steps; it is not a converged
   number.
2. **Compute: large.** 4.4–7.8× more CPU time per token, and ~50× more
   wall-clock to reach a comparable point (19 min vs days). The transformer
   also got further with **3.9× fewer tokens**.

So "far behind in both" is really: *slightly behind on bpc, far behind on
compute and sample efficiency.* The mechanisms below explain both.

---

## 1. How attention works (the whole mechanism is 8 lines)

```
q = h W_q ,  k = h W_k ,  v = h W_v            # per head: d -> d/H
a[t,j] = q_t · k_j / sqrt(d/H)                  # score of past position j for t
a[t,j] = -inf for j > t                          # causality
w = softmax_j(a[t,j])                            # weights over the past
ctx_t = Σ_j w[t,j] v_j                           # weighted sum of values
h_t = h_t + (ctx_t concat …) W_o                 # residual + output projection
```

What makes it work, in plain terms:

- **Content-based routing.** The weights are computed *from the data*: q_t
  asks "which earlier token matches the thing I'm representing", and every
  earlier token answers via its key. No positional prior, no fixed window —
  the model chooses its neighbours per token, per head, per layer.
- **One hop, zero distance.** Position t reads position j directly, even 255
  tokens back. Gradient from the loss at t reaches j in one step.
- **Soft, so differentiable.** All weights are continuous; every value gets a
  gradient proportional to its weight. Nothing is committed or discarded.
- **Parallel over the sequence.** The score matrix is one big GEMM; the whole
  batch×sequence is processed at once (0.18 s/step on 2 CPU threads here).
- **Cost** O(T²·d) for the scores + O(T²·d) for the value mix, per head, per
  layer; at T=256, d=128, 4 heads, 3 layers this is tiny and GEMM-friendly.

### The transformer stack — every mechanism, what it buys

| mechanism | where | what it buys | cost |
|---|---|---|---|
| token embedding | `baselines.py` | input identity | V×d |
| positional embedding (learned absolute here; RoPE/ALiBi in modern LMs) | `baselines.py` | order awareness | T×d |
| **multi-head self-attention** (4 heads × 3 layers) | `_CausalBlock` | 12 independent routing patterns per position — syntax, coreference, local n-grams, long-range | O(T²d) |
| causal mask | `torch.tril` | no future leakage (LM validity) | free |
| residual stream | `h = h + …` | information highway; gradients bypass blocks | free |
| pre-LayerNorm | `ln1`, `ln2` | stable deep training, no warmup tricks needed | O(BTd) |
| FFN / MLP 4d + GELU | `mlp_in/out` | per-token nonlinear computation — ~2/3 of all params | O(BT·4d²) |
| untied output head | `out` | d→V logits | V×d |
| log-softmax + CE | `train/train.py` | next-token objective | O(BTV) |
| AdamW + warmup + constant LR + clip + wd | `train/` | optimization | — |
| (inference) KV cache, flash attention, GQA/MQA, MoE | not here — modern variants | speed/capacity, not expressivity | — |

Depth: 3 distinct blocks → 3 rounds of *routing → mix → compute → route again*.
Effective receptive field of layer 3 = all 256 tokens, twice over.

---

## 2. The IECNN stack — every mechanism, what it buys

### Per region (e11b: one region, 3 passes, 8 dots)

| mechanism | code | what it does | what it buys | cost |
|---|---|---|---|---|
| memory rows | `memory.region1_rows` | region 1 rows = token embeddings | shared substrate every dot reads | O(BTd) |
| region chaining | `memory.next_region_rows` | rows_r = LN(Proj(summary_{r-1}) + rows_{r-1}) | deeper abstraction per region | R× |
| **windowed memory** | `region._window_memory` | `gather_windows(rows, span)` → [B,T,w,d] with banded causal validity; W_K, W_V; learned relative-distance key embedding added to keys | each dot reads the last `span` tokens (span=32 at R=1) | O(BTwd) |
| **dots** (N=8) | `dot.DotPopulation` | 8 readers with **separate** weights: query q_k, GRU cell (W_ih, W_hh, b), γ/trust projections, e_state projection | 8 different "personalities" over the same memory | N× the GEMMs |
| read (soft) | `dot._forward_fast` | scores = K·q_k/√d_q → mask → softmax → ctx | attention-like read, but **keys/values are shared**; dots differ only by their query | O(BTNwd) |
| read (secluded, e12) | `_secluded_ctx` | argmax over the window → copy ONE value, straight-through backward | hard commitment (an id), no dilution | same score cost |
| GRU update | `dot._forward_fast` | u = [ctx; h_pool; err; round_emb; msg; slot_read]; z,r,gates; s' = (1−z)tanh(...) + z·s | per-dot state, per position | O(BTNd·d_s) |
| **GRU state resets every position** (clarification C1) | `initial_state` per forward | no temporal carry at all | keeps causality & parallel training | — |
| consensus scores | `consensus.compute_scores_fast` | u_k = α·(−JS(p_k, ‖ loo_k)) + β·γ_k + gate_k | rewards informative disagreement | O(BTNV) in prob mode |
| trust accumulation | `consensus.update_trust` | T^p = ρT^{p−1} + u^p (ρ=0 in e11b) | smooths weights across passes | free |
| weights | `consensus.compute_weights` | w = softmax(T/τ_p), floor then renormalise | **8 scalar weights per position** | free |
| pooling | `pool_log_linear` | log b_t = Σ_k w_k log p_k | committee product-of-experts | O(BTNV) |
| passes (P=3) | `region._forward_hidden` | τ_p = τ0·κ^{p−1} (0.7^p), h_pool = Σ w_k s_k, error features | 3 rounds of opinion exchange | ×3 |
| error features | `consensus.error_features_from_parts` | e_k = [JS(own, crowd), crowd mass on own top, own peak] | tells a dot how redundant it is | O(BTNV) |
| shared head (e11) | `region._forward_hidden` | h_mix = Σ w_k e_state_k → ONE tied-vocab GEMM | no per-dot vocab GEMM on passes 1..P−1 | 1 GEMM instead of P·N |
| no-trust (e11b) | config ρ=0 | weights read γ directly | one less recurrence | — |
| slot table (e13) | `dot.update_slots/slot_context` | S private slots per dot, gated **detached** writes scanned over T, read into the GRU input | the only cross-position memory | O(BTNSd_w) |
| cross-region competition | `iecnn.forward` | ω = softmax(α·conf_r + gate_r) → log-linear mix of region log_b | combine regions (OFF: R=1) | O(BRTV) |
| messages | `models/messages.py` | inter-region state broadcast (OFF) | — | — |
| verifier head | `models/verifier.py` | MLP over 9 label-free features → scores (OFF) | learned dot selection | — |
| E6 losses | `train/losses.py` | oracle CE (λ_o=0.2, ON), decorrelation (λ_c=0.1, ON), confidence/entropy/verifier (OFF), outvote oversampling ×3 (ON via oracle) | direct per-dot teaching | O(BTNV) extra |

---

## 3. Side-by-side: the same job, two machines

| capability | transformer (b1) | IECNN (e11b) |
|---|---|---|
| token mixing | full causal attention, **all 256 positions**, every layer | windowed read, **span 32**, 3 passes |
| does the mixing evolve per pass? | yes — new q/k/v each layer | **no** — with `dynamic_query: false` the read is computed once and cached (`attn_cache`), so all 3 passes see the *identical* context |
| routing capacity | 4 heads × 256 keys × 3 layers = 3,072 continuous weights/position | 32 keys × 8 dots (softmax per dot), then 8 scalar weights × 3 passes |
| per-position state | residual stream d=128 through 3 blocks | GRU d_s=128, **reset every position** |
| temporal credit assignment | backprop through all 256 positions | **none** (by design C1); e13 slots are the only channel, and they are detached |
| depth | 3 distinct blocks + FFN | 1 block of weights applied 3× (`share_pass_params: true`) |
| nonlinearity | GELU FFN 4d per block (~2/3 params) | GRU sigmoid/tanh gates |
| normalisation | LayerNorm ×6 | none inside the dots (LN only in region chaining) |
| combination | residual **vector** add | hidden: weighted **vector** sum; prob: weighted **geometric mean of distributions** |
| output | untied head | tied embedding head (shared in e11) |
| training signal | CE | CE + oracle CE + decorrelation (+ outvote weighting) |
| kernels | 4 GEMMs/block, batched over B×T, GPU-friendly | ~20 small einsums with an N=8 leading dim, window gathers, python-level pass loop, CPU fp32 |

---

## 4. The five root causes of the gap (each with evidence)

**C1. Receptive field and mixing depth.** b1 reads 256 tokens directly in every
layer. e11b's dots read **32** and the read never changes across passes. The
256-token window exists in the spec (spans 32/64/128/256 across 4 regions) but
every vocab3-scale run has `n_regions: 1`, so only span 32 is live. This is the
largest single structural deficit and it is not an architecture verdict — it is
an *untested configuration* (E2, the 4-region depth rung, lost its checkpoint
and needs a fresh run).

**C2. The scalar bottleneck.** The committee's entire arbitration is **8 numbers
per position** (w_k). Attention composes 128-dim value vectors per head. The
measurement: on a strong checkpoint, pooled top-1 = 21.9% while the best dot
reaches 32.9% — **the pool throws away ~1/3 of the committee's knowledge**, and
the outvoted set is 9.4% of positions. Every attempt to fix the arbitration
better (verifier, JS weighting, trust) was bounded at **~0.047 bpc** by the
oracle selection ceiling. The bottleneck is the *shape* of the mix, not its
weights.

**C3. No temporal credit assignment.** The dot's GRU state resets every position
(C1 in the spec): by construction the model cannot learn "this token matters
because of what happened 100 tokens ago". b1 learns exactly that, in every layer.
e13's slot table is the first mechanism that touches this, and it is deliberately
detached (no gradient through time) — it can carry information forward but cannot
*learn* what to carry.

**C4. Fake depth.** `share_pass_params: true` means the 3 passes use the *same*
weights; combined with the cached read, a pass adds only "re-pool the opinions
and re-normalise". b1's 3 blocks have 3 different weight matrices plus a 4d FFN
each. Params match (1.72M) but the transformer spends them on **distinct**
transformations; IECNN spends them on 8 copies of one reader.

**C5. Engineering.** Many small ops with an N=8 leading dim, window gathers,
a Python loop over passes, no fused kernels, fp32 CPU with 2 threads. Measured:
0.089 ms/token (b1) vs 0.39–0.70 (e11b) → **4.4–7.8×**. This is real but it is
the *smallest* of the five: even at perfect kernel efficiency the bpc gap and
sample-efficiency gap would remain.

---

## 5. What would actually close it (ranked by expected bpc per engineering hour)

1. **Turn the window on** — run span 256 directly, or activate 4 regions
   (spans 32/64/128/256). Zero new mechanisms; E2 restart already planned.
   Expected: the biggest single move, because C1 is the biggest deficit.
2. **Dense routing instead of 8 scalars** — attention over dot *states*
   (query from the pool, keys = dots) rather than weighted averaging of 8
   scalars. Attacks C2, the measured 1/3 knowledge loss.
3. **Distinct per-pass weights** (`share_pass_params: false`) — real depth
   instead of re-application. Costs 3× params in the reader; can be offset
   by shrinking d_s.
4. **Dynamic queries** (`dynamic_query: true`) — makes each pass read
   differently; currently the cached read is frozen (C1).
5. **GPU + fused kernels** — 10× wall clock, no bpc change (blocked: 0 compute
   units).
6. **Keep e13 (slots)** — the only mechanism aimed at C3; verdict pending.

Predictions to be honest about: with 1–4 the family could plausibly match the
transformer's 1.766 at equal params. **Beating** it requires the committee to pay
for itself — and every measurement so far (0.047 bpc arbitration ceiling, 1/3
pooled-vs-oracle loss, 3–5× per-token cost) says the committee currently does
not. That is the decision Tolu has to make explicitly: IECNN as an
attention-beater (needs 1–4 to work *and* the committee to add value), or IECNN
as a different claim (interpretable, commitment-native, slot-memory) where
attention parity is not the target.

---

## 6. sage/

`sage/` contains no mechanisms: `architecture.sage` is one line ("sage runs on
SAGE (Synthesize) Architecture") and `Sage Versions.txt` is a 5-tier ambition
ladder (Alpha → Beta → Core → Prime → Nexus → Ascend → Infinite). Nothing in it
constrains the model design. The only connection worth stating: if the end goal
is a Sage-class self-improving system, the *foundation model* choice behind it
is exactly this comparison — and on current evidence a transformer buys the same
bpc for ~5× less compute, while the IECNN's differentiators (inspectable dots,
discrete commitment for slot memory) are capabilities a transformer lacks but
has not yet been shown to need.
