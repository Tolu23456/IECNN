# 01. The Dot

A **dot** is the basic unit of IECNN: a very small learner with its own query, its own recurrent state, and its own output projection. Many dots read the same memory and produce their own predictions.

Status: draft v0.1. Related: `02_memory.md`, `04_consensus_and_pruning.md`.

---

## 1. Purpose

A dot should be **cheap, distinct, and informative**:

- Cheap: a few hundred thousand parameters at most, so dozens of them fit on one GPU.
- Distinct: its learned query makes it look at the memory differently from its neighbours.
- Informative: its distribution and its confidence both carry signal that the consensus can use.

A dot is **not** an independent model trained in isolation. All dots are trained jointly through the consensus loss, plus small per-dot auxiliary losses (file 06). "Each dot learns on its own" is meant in the sense that each has its own parameters and its own specialty, not separate training runs.

## 2. Parameters owned by each dot k

| Parameter | Shape | Purpose |
|---|---|---|
| `q_k` | `[d_q]` | Base query used to read the memory. |
| `U_k` (optional) | `[d_s, d_q]` | Lets the query depend on the dot's current state (dynamic query). |
| `s0_k` | `[d_s]` | Initial state at pass 1 (learned). |
| GRU weights | `[3, d_in + d_s, d_s]` | The recurrent update. |
| `P_k` | `[d_s, d]` | Projects state into the shared embedding space for output. |
| `c_k` | `[d_s]` + bias | Confidence head (a scalar logit). |
| `g_k` | `[d_s]` + bias | Trust gate head used by the weighting rule (file 04). |

Shared within a region (not per dot): the key projection `W_K`, the value projection `W_V`, the relative-distance embedding, the round-signal embeddings.
Shared across the whole model: the tied embedding matrix `E` of shape `[V, d]`, used to turn `P_k s` into logits.

With the defaults in file 00 (d_in about 227, d_s = 128) a dot has roughly 150k parameters.

## 3. Inputs to a dot at pass p, position t

| Input | Shape | Source |
|---|---|---|
| Context vector `c` | `[d_v]` | Read from memory through the dot's query (section 4). |
| Pooled opinion `h_pool` | `[d_s]` | Weighted average of dot states from pass p-1 in the same region (zero at pass 1). |
| Error feature `e` | `[3]` | How the dot's previous guess compares to the leave-one-out crowd (section 5). Zero at pass 1. |
| Round signal `r_p` | `[16]` | Learned embedding of the pass index. |
| Message `m_in` (optional) | `[d_m]` | From nearby dots in the same region at pass p-1 (file 05). |

The GRU input is the concatenation: `u = [c ; h_pool ; e ; r_p ; m_in]`.

## 4. Computation of one pass

Notation: memory rows for the visible window are `M` with shape `[w, d]`. Keys and values are computed once per region per position and shared by all dots in that region.

```
K = M W_K + rel_emb            # [w, d_q]   (shared in region)
V = M W_V                      # [w, d_v]   (shared in region)

q   = q_k                      # static query
q   = q_k + U_k s_prev         # optional dynamic query
a   = softmax( K q / sqrt(d_q) )          # [w], causal window mask applied
c   = a V                                  # [d_v], this dot's read

u     = concat(c, h_pool, e, r_p, m_in)
s_new = GRU_k(u, s_prev)                   # [d_s]

z     = E (P_k s_new)                      # logits, [V]   (E tied embedding)
pi    = softmax(z)                         # the dot's distribution
gamma = c_k . s_new + bias                 # confidence logit (scalar)
tr    = g_k . s_new + bias                 # trust-gate contribution (scalar)
```

Returned to the region: `pi`, `gamma`, `tr`, and `s_new` (for pooling and messages).

### Why queries make dots different

Every dot reads the **same** keys and values but with its own query, so two dots can pull different blends from the same memory. This is similar in form to multi-head attention with many tiny heads, so the extra value of IECNN must be demonstrated by the pass, weighting, and region mechanisms (the experiment ladder), not by this read step.

## 5. The error feature (self-correction)

Decision D7: a dot sees how its previous guess fared. Constraint C3: the signal must be label-free and must not reveal the crowd's full answer.

Let `pi_k` be the dot's distribution at the previous pass and `m_-k` the **leave-one-out mean** of the other dots' distributions in the region at that pass.

```
e = [ JS(pi_k, m_-k),                 # how far the dot is from the crowd
      m_-k[argmax pi_k],              # probability the crowd gives the dot's own top choice
      max pi_k ]                      # the dot's own peak confidence
```

- It is three scalars, so a dot cannot copy the crowd's distribution from it.
- It is measured against the crowd, not the label, because no label exists at inference.
- If the dot always moves toward the crowd after seeing a large JS value, it will collapse into the crowd. This is guarded against by the decorrelation penalty and the diagnostics in file 08.

## 6. Initialization

- `q_k`: random orthogonal-ish vectors, one seed stream per dot, so no two dots start identical.
- `s0_k`: small random normal (std 0.02).
- GRU weights: standard orthogonal recurrent, Xavier input.
- `P_k`: Xavier; `c_k`, `g_k`: zero bias, small weights.
- Dot dropout (optional, small): zero a random dot's contribution in a fraction of training steps to stop dependence on any single dot.

## 7. Dot-level variants (not in v0.1)

- **Private slice**: restrict a dot to a sub-range of the memory. Rejected for v0.1 (decision D1).
- **Larger-state dots** for later regions.
- **Heterogeneous dot types**: some dots with no recurrence, some with a different cell type.

## 8. Failure modes and mitigations

| Failure | Sign | Mitigation |
|---|---|---|
| Collapse: dots become near copies | Pairwise error correlation near 1; per-dot ablation damage near zero for all | Per-dot queries, correlation penalty, dot dropout. |
| Dead dot: weight stays near zero forever | Ablation damage about zero for thousands of steps | Weight floor, entropy bonus on weights early, optional revival (re-initialize and brief retrain). |
| Overconfident dot | Confidence high regardless of correctness | Calibration loss on `gamma` (file 06), cap on `gamma`. |
| Exploding state | NaN or huge loss spikes | LayerNorm on state, gradient clipping, bf16 with fp32 for softmax. |
| One-dot dominance | One weight near 1 for most tokens | Weight entropy floor, higher starting temperature. |

## 9. Implementation notes

- Stack all N dots of a region into tensors with a leading dot dimension, so the whole region runs as batched (grouped) matmuls: weights shaped `[N, in, out]`, activations `[B, T, N, ...]`.
- Keys and values are computed once per region, not per dot.
- Output logits cost `d x V` per dot per pass. This is cheap for byte-level vocabularies; for large vocabularies, compute logits only over a candidate subset or only at the last pass (open question, file 04).
- Causal masking must be exact. A unit test must confirm that changing token t+1 cannot change anything computed at position t.

## 10. Tests

1. Shapes: a forward pass returns `pi` of shape `[B, T, N, V]` that sums to 1 on the last axis.
2. Causality: perturbing a future token leaves earlier outputs bit-identical.
3. Gradients: every dot's parameters receive nonzero gradient after one training step.
4. Distinctness at init: pairwise cosine similarity between dot queries below 0.5.
5. Determinism: the same seed gives identical outputs.
6. Label leakage: the label-free signals (error feature, weight scores) are identical whether or not labels are provided in the batch.
