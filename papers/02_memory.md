# 02. Shared Memory

The shared memory is the table of recent positions that dots read from. It is the only way IECNN sees context across time (dot states reset every position; see clarification C1 in file 00).

Status: draft v0.1. Decision D1: shared memory. Decision D8: fixed window first.

---

## 1. Shape and meaning

For a position t, a region sees a table `M` of up to `w_r` rows, one row per recent position:

```
M = [ m_{t-w+1}, ..., m_{t-1}, m_t ]      shape [w_r, d]
```

Only positions at or before t are visible (causal). Positions before the start of the text are masked, not zero-filled.

## 2. Where the rows come from

### Region 1 (spelling-level)

```
m_j = Embed(x_j) + 0       # tied embedding E, shape [V, d]
```

Position information is not added to the row. It is added in the keys as a **relative distance embedding** (section 3), so the same memory row can mean different things depending on how far back it is.

### Region r > 1 (later stages)

Row j is built from region r-1's output at position j:

```
h_j^{r-1} = sum_k  w_k * s_k^{final}        # weighted average of the region's final dot states, [d_s]
m_j^{r}   = LayerNorm( Proj_r(h_j^{r-1}) + m_j^{r-1} )     # [d]
```

- `Proj_r` is a linear map `[d_s -> d]`.
- The residual term `m_j^{r-1}` carries the previous region's memory forward. This is a proposal (not yet decided) for stability, and it keeps the original embedding signal reachable at all depths.
- The weights `w_k` are the region's final weights (file 04), so pruned dots contribute little to the next region's memory.

Region r's computation at position t uses `h^{r-1}_j` for `j <= t`. In training, region r-1 is computed for all positions first (in parallel, causally masked), then region r.

## 3. Keys, values, and position

Computed once per region per position, shared by every dot in that region:

```
K_j = m_j W_K + rel[t - j]          # [d_q]   rel is a learned table over distances 0..w_r-1
V_j = m_j W_V                       # [d_v]
```

- `rel[distance]` is a learned embedding of the distance between the current position and the memory row, so no absolute positions are needed.
- A region with a short span (for example 32) naturally focuses on spelling; a region with a long span (for example 256) can reach phrase and sentence patterns.

## 4. Window sizes per region (proposal)

| Region | Role | Span w_r |
|---|---|---|
| 1 | spelling | 32 |
| 2 | sub-word / word | 64 |
| 3 | phrase | 128 |
| 4 | sentence-level meaning | 256 |

The global cap W = 256 is decided (D8). The per-region split is only a proposal; an ablation should compare it with every region using 256.

## 5. Training versus inference

- **Training**: all T positions in a sequence are processed in parallel. Position t's read uses a banded causal mask: row j visible iff `t - w_r < j <= t`.
- **Inference**: keep a sliding cache of the last `w_r` rows of keys and values per region. Each new token appends one row per region and drops the oldest.
- Region r at inference needs region r-1's output for the new position before it can run, so the regions run in sequence per token (this is inherent to stacking).

## 6. Complexity

Per position per region per pass, the read cost per dot is about `w_r * (d_q + d_v)` multiply-adds. Total read cost per position is roughly:

```
sum over regions r:  N * P * w_r * (d_q + d_v)
```

With the defaults (N=8, P=3, d_q=32, d_v=64, spans 32/64/128/256) this is about 1.0M multiply-adds per position for reads, which is small next to the output logits for large vocabularies.

## 7. Long-range memory (deferred)

Not in v0.1. Candidate designs for later:

1. **Compressive rows**: older positions are summarized into a few rows by a small learned pooling.
2. **Strided rows**: keep every k-th older position.
3. **State-space summary row**: one extra row carrying a running state.
4. **Retrieval rows**: pull in rows from a cache of earlier text.

Each of these changes only how rows are produced; dots and queries stay the same, which is why the window is a clean place to defer this.

## 8. Failure modes

| Failure | Mitigation |
|---|---|
| Future leakage | Banded causal mask plus the perturbation unit test. |
| Memory rows too similar (low information) | LayerNorm, residual, monitor row variance. |
| Short spans starve later regions | Ablation against uniform 256 spans. |
| Region r > 1 ignores region r-1 | Check attention mass on rows vs the residual path; monitor ablation of the Proj path. |

## 9. Tests

1. Causality: changing x_{t+1} changes no row visible at t.
2. Window edge: at t < w, masked rows have exactly zero attention weight.
3. Equivalence: parallel training mode and token-by-token cached inference give the same logits (within tolerance) on the same text.
4. Region chaining: ablating `Proj_r` (zeroing it) leaves only the residual path and changes loss measurably.
