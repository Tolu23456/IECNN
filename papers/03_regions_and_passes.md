# 03. Regions and Passes

Regions are the stages of IECNN. Passes are the rounds of rethinking inside a region. This file specifies both, and the competition between regions.

Status: draft v0.1. Decisions D4 (regions as stages), D5 (cross-stage competition), D6 (shared parameters across passes), D7 (dots see their own previous guess).

---

## 1. Definitions

- **Region**: a group of N dots that share a memory view (file 02) and compete with one another.
- **Pass** (also "layer"): one round in which every dot in a region reads, updates, and outputs.
- **Stage ordering**: region 1 reads raw bases, region r reads the output of region r-1.

## 2. Structure

```
for region r in 1..R:
    memory M^r  <- built from region r-1 output (or embeddings for r = 1)
    for pass p in 1..P:
        every dot k in region r:
            read memory with its query
            update its state (GRU)
            output distribution pi, confidence gamma, trust term tr
        region computes weights w^p, consensus b^p, pooled opinion h_pool^p
        temperature tau_p = tau_0 * kappa^(p-1)
    region emits: final consensus b^r, region summary h^r (per position)

region weights omega over r  ->  final distribution y
```

## 3. Shared parameters across passes (D6)

The dots at pass p and pass p+1 are the **same parameters**. Differences between passes come from:

1. the state carried over (so the dot "remembers" its previous thinking),
2. the pooled opinion and error feature (the dot sees how it did),
3. the **round signal** `r_p`, a learned vector per pass index, added to the GRU input so the dot can behave differently early and late.

Consequences:

- Parameter count does not grow with P, so extra budget can go into more dots instead of more layers.
- The model is a small recurrent system unrolled P times. Training backpropagates through all passes (unrolled, P is small).
- Ablation to run: shared vs unshared per-pass parameters, to check whether sharing costs accuracy.

## 4. What a region is expected to specialize in

This is a **hope**, not a guarantee. Nothing forces region 1 to learn spelling and region 4 to learn meaning except the structure below.

Structural pressures toward the intended specialization:

- **Increasing spans** (file 02): short window early, long window late.
- **Stacked memory**: later regions only see what earlier regions produced (plus a residual).
- **Per-region auxiliary loss** (file 06): each region is trained to predict the next base on its own, so early regions must become useful on their own.

How to check: probe which region wins (largest region weight) by token type, section 8.

## 5. Competition inside a region

Handled by weights (file 04). Within a region the weights sum to 1; dots compete for share of the region's consensus.

## 6. Competition between regions (D5)

Different stages are answering different questions (what letter next vs what word fits), so raw per-dot scores are not comparable across regions. Therefore two levels:

1. **Inside a region**: dots compete via dot weights.
2. **Across regions**: each region as a whole gets a weight `omega_r`.

```
score_r  = alpha_R * ( - H(b^r) )  +  gate_r( h^r )          # H = entropy of the region consensus
omega    = softmax( score / tau_region )                       # sum over regions = 1
log y   ∝  sum_r omega_r * log b^r                             # log-linear pool of regional consensus
```

- `-H(b^r)` rewards a region that is confident (peaked consensus).
- `gate_r` is a small learned linear map on the region summary at the current position, so the model can learn "which region to trust here" from context.
- `tau_region` is annealed like the dot temperature, but with a floor higher than the dot floor so one region does not take over too early.

### What the region weights give you

Which region has the largest weight is a free diagnostic about the token: spelling-heavy tokens (names, rare words) should lean on early regions; common phrases on later ones. This is useful in the evaluation file (08).

## 7. Passes and halting

- **v0.1**: fixed P = 3 for every token.
- **Later, optional**: dispersion-based halting. After a pass, compute the dispersion among the region's dot distributions; if it is below a threshold, stop early for that token. Easy tokens finish in one pass; contested tokens use all passes.

Dispersion measure (label-free):

```
disp_p = mean_k  JS( pi_k^p , mean_j pi_j^p )
halt if disp_p < delta
```

The cheap first diagnostic is not to halt at all, but to check whether `disp_1` predicts the token's loss (AUROC), as suggested in the design discussion. If it does not, halting would not work either.

## 8. Diagnostics specific to regions and passes

| Diagnostic | What it tells you |
|---|---|
| Region weight vs token type (capitalized, rare word, within-word position, punctuation) | Whether stages really specialize |
| Loss of each region's own consensus (`b^r`) | Whether each region is useful alone |
| Loss by pass: `b^1`, `b^2`, `b^3` | Whether extra passes actually improve the output |
| Weight entropy by pass | Whether pruning is happening, and how fast |
| Gap between P=1 and P=3 | The headline test for rounds (rung E1) |

## 9. Failure modes

| Failure | Mitigation |
|---|---|
| Passes do nothing (P=3 equals P=1) | Report honestly; that kills the "iterative" claim. Try more passes, different round signal, stronger error feature. |
| One region dominates always | Higher region-level temperature floor; entropy bonus on omega early in training. |
| Later regions ignore earlier ones | Residual path in memory; monitor attention mass. |
| Unrolled training unstable | Gradient clipping, LayerNorm on states, smaller P at first. |

## 10. Tests

1. With P=1 and R=1 and uniform weights the model reduces to a plain ensemble (this is rung E0 and must be exactly reproducible by config).
2. Region weights sum to 1 at every position.
3. Zeroing the round signal changes outputs (otherwise it is ignored).
4. The final distribution is a valid distribution at every position.
5. Shared vs unshared pass parameters: confirm parameter counts differ by a factor of P.
