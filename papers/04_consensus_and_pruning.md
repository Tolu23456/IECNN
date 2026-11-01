# 04. Consensus and Pruning

This file defines how dot outputs become one decision: the weights, the pooling, the temperature, and the soft-to-hard transition.

Status: draft v0.1. Constraints C2 (label-free weights), C4 (no dot is deleted), C5 (hard vote is inference-only).

---

## 1. Overview

Within one region, at each pass p:

1. every dot k produces a distribution `pi_k` and a confidence logit `gamma_k` (file 01),
2. the region computes a **score** per dot from label-free signals,
3. scores are accumulated into **trust** and sharpened by a temperature into **weights** that sum to 1,
4. the region forms a **consensus** by weighted log-linear pooling,
5. the weighted average of dot states becomes the **pooled opinion** for the next pass.

"Pruning" means weights of weak dots are driven toward zero for this token. No parameters are removed, and all dots return for the next token (decision D3, clarification C4).

## 2. Label-free signals

At inference there is no true next token, so the weighting rule may only use quantities computed from dot outputs (constraint C2).

For dot k at pass p:

```
m_-k      = (1 / (N-1)) * sum_{j != k} pi_j            # leave-one-out mean of the others
agree_k   = - JS( pi_k , m_-k )                        # higher = closer to the crowd, bounded in [-ln2, 0]
conf_k    = gamma_k                                    # the dot's own confidence logit
gate_k    = g_k . s_k + b                              # learned trust term from the dot's state
```

- Leave-one-out removes a dot's own influence on the crowd it is compared with.
- JS divergence is bounded and symmetric, so one wild dot cannot produce an enormous score.

## 3. Scores, trust, and weights

```
u_k^p  = alpha * agree_k^p  +  beta * conf_k^p  +  gate_k^p           # alpha, beta learned scalars per region
T_k^p  = rho * T_k^{p-1}  +  u_k^p                                    # trust accumulates across passes; T^0 = 0
w_k^p  = softmax_k( T_k^p / tau_p )                                   # weights over the N dots of the region
tau_p  = tau_0 * kappa^(p-1)                                          # sharpening within the token
```

- Agreement alone would push all dots toward copying each other. The learned `gate_k` term lets a dot earn trust some other way (for example by encoding "I am the one that handles this pattern"). Because the gate is trained end to end through the final loss, a lone correct dot can learn to signal itself.
- `rho` (default 0.5) lets earlier passes matter without being permanent.
- `tau_0` is annealed over training (section 6); `kappa` (default 0.7) sharpens within every token.

### Weight floor

To avoid dead dots early in training, in soft mode apply a floor:

```
w_k <- max(w_k, eps)  then renormalize        # eps = 0.1 / N, set to 0 in the final phase of training
```

## 4. Consensus

Log-linear (weighted geometric) pooling of the dot distributions:

```
log b^p(v)  ∝  sum_k  w_k^p * log pi_k^p(v)
```

Properties:

- A dot with weight near 0 contributes essentially nothing, which is what "pruned" means.
- It is differentiable in both the weights and the dot distributions.
- Compute it in log space with a log-softmax at the end for numerical stability.
- Mixture pooling (`sum_k w_k pi_k`) is the alternative. It is more forgiving of one wrong dot but gives flatter distributions. Keep it as a config switch for an ablation.

Pooled opinion for the next pass:

```
h_pool^p = sum_k w_k^p * s_k^p         # [d_s], weighted average of dot states
```

## 5. Soft training vs hard inference

| | Training | Inference |
|---|---|---|
| Weights | Soft, from softmax with temperature | Hard: keep top-m dots per region by final weight, renormalize |
| Output | Log-linear pool with all dots (floored) | Log-linear pool restricted to survivors |
| Loss | Cross-entropy on the pooled output | None |
| Selection of next base | Not used | Argmax (or sampling) over the pooled distribution |

- Hard top-m default: m = 4 of 8 per region.
- **Mode gap**: the loss when switching from soft to hard weights at test time. This must be measured every evaluation (file 08). If the gap is large, use the mitigations in section 7.

### Why not a literal majority vote

A vote over token ids discards probability information, is non-differentiable, and breaks ties poorly over large vocabularies. It is only logged as a diagnostic:

```
plurality_agreement = fraction of dots whose argmax equals the consensus argmax
```

## 6. Temperature schedule

```
tau_0(step) = tau_max                                            for step < warmup
            = tau_min + 0.5 (tau_max - tau_min)(1 + cos(pi * (step - warmup) / (total - warmup)))    after
```

Defaults: `tau_max = 1.0`, `tau_min = 0.2`, warmup 2k steps. Lower `tau` means closer to all-or-nothing.

## 7. Mitigations for the soft-to-hard gap

1. Anneal the temperature far enough that training weights are already nearly hard at the end.
2. In a fraction of late training batches (for example 20%), run the hard top-m mask with straight-through gradients.
3. Optionally add a small penalty on the KL between the soft and the hard-masked output.
4. Report the mode gap with the model in both modes.

## 8. The bonus for being right when others are wrong

An earlier idea was to reward dots that are right when most others are wrong. Because the true label exists only in training, this **cannot** be part of the weighting rule at inference (constraint C2). It is implemented only as a training pressure (file 06, "boosted per-dot loss"): a dot's own cross-entropy is weighted by how wrong the consensus was, so dots learn to fix the consensus's mistakes.

## 9. Edge cases

| Case | Handling |
|---|---|
| All scores equal | Weights are uniform (plain average); fine. |
| One weight near 1 | Allowed late in training; flagged in early training (weight entropy diagnostic). |
| Non-finite scores | Clamp scores to [-20, 20] before softmax; compute JS with a small epsilon. |
| N = 1 region | Leave-one-out is undefined; use uniform weight (the model degenerates cleanly). |
| Ties in hard mode | Break by higher confidence, then by dot index. |

## 10. Cost for large vocabularies (open question)

Computing a full distribution per dot per pass costs `d x V` per dot. For byte-level V this is trivial. For large V:

- compute full distributions only at the last pass and use cheaper statistics earlier,
- restrict to a candidate set of the top-K bases proposed by a cheap region-level head,
- or use the hierarchical vocabulary idea (not in v0.1).

## 11. Tests

1. Weights sum to 1 per region per pass (within 1e-6).
2. As `tau` goes to 0, weights approach a one-hot on the highest trust.
3. As `tau` goes to infinity, weights approach uniform.
4. Leave-one-out: the crowd mean for dot k is unchanged if dot k's output is altered.
5. Label independence: weights computed with and without labels in the batch are identical.
6. Pooled output is a valid distribution; log-linear and mixture pooling agree when all `pi_k` are identical.
7. Hard top-m with m = N reproduces the soft output at tau close to 0.
