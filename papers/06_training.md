# 06. Training

How IECNN is trained: the objectives, the schedules, the configuration, and the signals to watch.

Status: draft v0.1. Constraints C2 (weights are label-free), C5 (hard voting is inference-only).

---

## 1. Training setup

- **Objective**: next-base prediction with teacher forcing. All positions in a sequence are processed in parallel.
- **Data**: text tokenized by UBM (file 07), packed into sequences of length `T` (default 256). Start with a small character-level corpus.
- **Gradients**: one backward pass through all dots, queries, the weighting rule, the memory projections, the unrolled passes, and the tied embedding.
- **Soft everything**: no hard selection during most of training (file 04).

## 2. Loss terms

Let `y_t` be the true next base, `b_final` the final pooled distribution, `b^r` region r's consensus, `pi_k` dot k's distribution at the last pass.

### 2.1 Main loss

```
L_main = CE( b_final , y )
```

### 2.2 Region auxiliary loss

Each region must be useful on its own, so early regions cannot be lazy.

```
L_region = sum_r CE( b^r , y )
```

### 2.3 Boosted per-dot loss

Each dot is also trained directly, but weighted by how wrong the consensus was. This is the training-only way of rewarding dots that are right when the crowd is wrong (file 04, section 8).

```
nu_t   = ( 1 - b_final(y_t) )^gamma_b         # detached; gamma_b default 2
L_dot  = (1 / N_total) * sum_k  mean_t [ nu_t * CE( pi_k , y_t ) ]
```

Risk: all dots chase the same hard tokens. The correlation penalty below counters this.

### 2.4 Decorrelation penalty (D2)

Punish dots for being wrong in the same way at the same time, **only on tokens where the consensus was wrong** (we do not punish agreement when everything is right).

```
e_k(t)  = - log pi_k( y_t )                              # dot k's error on token t
nu_t    = ( 1 - b_final(y_t) ) , detached                # weight: how wrong the crowd was
mu_k    = sum_t nu_t e_k(t) / sum_t nu_t
cov_ij  = sum_t nu_t ( e_i(t) - mu_i )( e_j(t) - mu_j ) / sum_t nu_t
rho_ij  = cov_ij / ( sqrt(cov_ii cov_jj) + eps )
L_corr  = mean_{ i < j , same region } relu( rho_ij )
```

- Computed within a region, over the tokens of the batch.
- Only positive correlation is penalized.
- Schedule: `lambda_c` ramps from 0 over the first 5k steps up to 0.01.
- Safety rule: if `lambda_c * L_corr` exceeds 10% of `L_main` for a long period, reduce `lambda_c`. Too strong a penalty makes dots deliberately wrong.

### 2.5 Confidence calibration (small)

So that the confidence logit `gamma_k` carries meaning:

```
L_conf = mean_k,t  BCE( sigmoid(gamma_k) , 1[ argmax pi_k == y_t ] )
```

### 2.6 Weight entropy bonus (early training only)

Prevents early collapse onto one dot:

```
L_w = - lambda_w(step) * mean_{regions, passes} H( w )        # lambda_w decays to 0 by 20% of training
```

### 2.7 Total

```
L = L_main + lambda_r L_region + lambda_d L_dot + lambda_c L_corr + lambda_k L_conf + L_w
```

Defaults:

| Weight | Default |
|---|---|
| lambda_r | 0.3 |
| lambda_d | 0.1 |
| lambda_c | 0.01 (ramped) |
| lambda_k | 0.05 |
| lambda_w | 0.05 decaying to 0 |
| gamma_b | 2 |

## 3. Schedules

| Quantity | Schedule |
|---|---|
| Learning rate | AdamW, peak 1e-3, linear warmup 500 steps, cosine decay |
| Temperature tau_0 | Cosine from 1.0 to 0.2 (file 04, section 6) |
| Weight floor eps | 0.1/N, set to 0 for the final 10% of steps |
| lambda_c | Ramp 0 to 0.01 over 5k steps |
| Hard-mode batches | 0% until 70% of training, then 20% with straight-through |
| Dot dropout | 5% of steps, one random dot masked per region |

## 4. Optimization and stability

- AdamW, betas (0.9, 0.95), weight decay 0.01.
- Gradient clipping at 1.0 (global norm).
- LayerNorm on the dot state and memory rows.
- bf16 for matmuls, fp32 for softmax, JS divergence, and the pooling in log space.
- Batch: 64 sequences of 256 bases to start.
- Seeds: 3 per configuration, always.

## 5. Dot revival (optional)

If a dot's ablation damage is below a threshold for K consecutive evaluations (default K = 5):

1. re-initialize its parameters (new random query, fresh weights),
2. train for a short burst with a boosted learning rate for only that dot,
3. log the event.

Use sparingly. If many dots die, the collapse problem is not solved and revival hides it.

## 6. Configuration

One config file controls every switch so each ladder rung is just a different config.

```yaml
model:
  d: 128
  d_q: 32
  d_v: 64
  d_s: 128
  n_dots: 8
  n_regions: 4
  n_passes: 3
  spans: [32, 64, 128, 256]
  share_pass_params: true
  dynamic_query: false
  pooling: log_linear        # or mixture
  use_weights: true          # false = uniform weights (rung E0)
  use_regions: true          # false = single region
  use_cross_region: true
  use_messages: false
  message_dim: 16
  message_topology: ring
  ring_radius: 2
  halting: none              # or dispersion
training:
  steps: 50000
  batch: 64
  seq_len: 256
  lr: 1.0e-3
  warmup: 500
  tau_max: 1.0
  tau_min: 0.2
  kappa: 0.7
  rho: 0.5
  lambda_r: 0.3
  lambda_d: 0.1
  lambda_c: 0.01
  lambda_k: 0.05
  hard_batch_fraction: 0.2
  hard_batch_start: 0.7
  seeds: [0, 1, 2]
```

Rung E0 is `n_passes: 1, use_weights: false, n_regions: 1, use_messages: false, lambda_c: 0`.

## 7. Monitoring (log every N steps)

| Metric | Why |
|---|---|
| Train and validation bits-per-character | Headline |
| Mode gap (hard minus soft loss) | Soft-to-hard risk |
| Weight entropy per region, per pass | Pruning health |
| Per-dot ablation damage and its spread | Specialization |
| Pairwise error correlation | Collapse |
| Plurality agreement | Crowd consensus level |
| Region weight distribution | Whether stages are used |
| Dot confidence calibration | Whether confidence means anything |
| Gradient norm per component | Instability |

## 8. Early warning signs

| Sign | Likely cause | Action |
|---|---|---|
| Ablation damage equal for all dots | Collapse or no specialization | Increase query diversity, check `lambda_c`. |
| Ablation damage near zero for all | Redundant ensemble | Reduce N, or the interaction is not helping. |
| Weight entropy drops to 0 early | Temperature too low or score scale too large | Raise `tau_max`, entropy bonus. |
| Loss spikes | Instability in unrolled passes | Lower LR, clip, reduce P. |
| Mode gap large | Soft-trained, hard-tested mismatch | Section 3 hard-mode batches, more annealing. |
| Correlation penalty dominates | `lambda_c` too high | Reduce. |

## 9. Compute budget

Per position, roughly:

```
cost ≈ sum_r  N * P * ( w_r (d_q + d_v)  +  GRU(d_in, d_s)  +  d_s d  +  d V )
```

Every comparison must be at **equal training FLOPs or equal wall-clock** (file 08). Report parameters and FLOPs for every run.

## 10. Reproducibility

- Fixed seeds and logged config for every run.
- Save the UBM vocabulary version with every checkpoint (file 07).
- Deterministic data order per seed.
- Checkpoint every 1k steps; keep the best by validation bits-per-character.

## 11. Tests

1. Rung E0 config reproduces a plain uniform ensemble.
2. A single optimization step reduces the training loss on a tiny fixed batch (overfit test: the model can memorize 10 sequences).
3. Loss terms are finite and have the expected scale at step 0.
4. The label-free signals are identical with labels hidden.
5. Training resumes from a checkpoint with identical subsequent loss values.
