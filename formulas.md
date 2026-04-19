# IECNN Formulas

Fifteen custom formulas purpose-built for the IECNN architecture.
F1–F9 form the core operational loop; F10–F15 extend it with metrics,
adaptive control, and hierarchical scoring.

All formulas are implemented in both **C** (compiled to `.so` for performance)
and **Python** (pure fallback). The C path is selected automatically at runtime.

---

## Formula 1 — Similarity Score

Measures how similar two predictions are in the unified BaseMapping space.
This is the foundation of all clustering and selection in IECNN.

$$S(p_i, p_j) = \alpha \cdot \cos(p_i, p_j) + (1 - \alpha) \cdot A(p_i, p_j)$$

**Components:**

$$\cos(p_i, p_j) = \frac{p_i \cdot p_j}{\|p_i\| \|p_j\|}$$

$$A(p_i, p_j) = \tanh\!\left(\frac{\|p_i\| + \|p_j\|}{2\sqrt{d}}\right) \cdot \frac{\cos(p_i,p_j) + 1}{2}$$

**Parameters:**
- $p_i, p_j \in \mathbb{R}^d$ — predictions in BaseMapping space ($d = 128$)
- $\alpha \in [0,1]$ — balance between structural match (cos) and magnitude-aware agreement ($A$)
- Default: $\alpha = 0.70$

**Why both?** Cosine similarity ignores magnitude, missing the case where two predictions
both point in the right direction *and* are strongly activated. $A$ captures this reinforcement.

---

## Formula 2 — Convergence Score

Scores a cluster $k$ by its internal cohesion and collective confidence.

$$C(k) = \frac{1}{|k|^2} \sum_{i \in k} \sum_{j \in k} S(p_i, p_j) \cdot \bar{c}_k$$

**Where:**
- $|k|$ — number of predictions in cluster $k$
- $S(p_i, p_j)$ — Formula 1 similarity
- $\bar{c}_k = \dfrac{1}{|k|} \sum_{i \in k} c_i$ — mean confidence of cluster members
- $c_i = \tanh\!\left(\dfrac{\|p_i\|}{\sqrt{d}}\right)$ — confidence of prediction $p_i$

**Interpretation:** High $C(k)$ means predictions in cluster $k$ all point in the same direction
and all have high magnitude — they are not only similar but confident.

---

## Formula 3 — Attention

Standard scaled dot-product attention, applied within the AIM layer to refine
inverted predictions using the BaseMap context.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

**In IECNN:**
- $Q$ — the inverted prediction (the "question")
- $K, V$ — the BaseMap representation (the "context")
- $d_k = 128$ — feature dimension (scaling factor)

This ensures that even inverted predictions are grounded in the actual input structure.

---

## Formula 4 — AIM Transform

Transforms a prediction through one of nine inversions, then refines with attention.

$$\hat{p}_i = \text{Attention}\!\left(Q,\ K,\ \mathcal{I}^{(\tau)}(p_i)\right)$$

**Where:**
- $p_i$ — original dot prediction
- $\mathcal{I}^{(\tau)}$ — one of 9 inversion functions (selected by type $\tau$)
- $\hat{p}_i$ — AIM-refined candidate prediction

**Nine inversion types $\mathcal{I}^{(\tau)}$:**

| $\tau$ | Name | Description |
|---|---|---|
| 1 | Feature | Negate dims above mean absolute value |
| 2 | Context | Swap high/low activation groups |
| 3 | Spatial | Reverse dimension group ordering |
| 4 | Scale | Rescale quarters by [4, 2, 0.5, 0.25] |
| 5 | Abstraction | Flip between patch and object level |
| 6 | Noise | Suppress dominant features |
| 7 | Relational | Invert dominant block correlations |
| 8 | Temporal | Reverse sequential dim ordering |
| 9 | Compositional | SVD decompose, permute, recompose |

---

## Formula 5 — Pruning Threshold

A prediction or cluster is kept only if its score exceeds the dynamic threshold $\tau$.

$$\text{Keep cluster } k \iff C(k) > \tau(t)$$

**Soft threshold** (individual predictions, Stage 1):
$$\tau_{\text{soft}}(t) = \tau_0 \cdot (1 + 0.5 \cdot \text{Dominance}(t-1))$$

**Hard threshold** (clusters, Stage 3):
$$\tau_{\text{hard}}(t) = \max\!\left(\tau_{\min} \cdot (1 + 2 \cdot \text{dom}),\; C_{\max} \cdot f\right)$$

Where $f = 0.08$ (floor fraction) ensures we never discard clusters too close to the leader.

---

## Formula 6 — Prediction Confidence

Normalized L2 norm mapped to $(0, 1)$ via tanh.

$$c(p) = \tanh\!\left(\frac{\|p\|}{\sqrt{d}}\right)$$

**Why tanh?** The norm can be large (predictions are scaled to $\sqrt{d}$), so
tanh provides a smooth, bounded confidence score without needing explicit normalization.

---

## Formula 7 — Sampling Temperature

Controls output diversity of dot predictions. Higher temperature → more varied candidates.

$$P(\text{dot generates variant } v) \propto \exp\!\left(\frac{\text{score}(v)}{T}\right)$$

**Adaptive temperature** (per dot):
$$T_\text{eff}(d, t) = T_d \cdot (1 + 0.3 \cdot H_C(t-1))$$

Where $H_C$ is the cluster entropy from the previous round (Formula 11). When entropy is
high (confused), temperature rises automatically to encourage exploration.

---

## Formula 8 — Bias Vector Update

After each learning cycle, the global bias vector $\mathbf{b}$ is updated
based on winning prediction characteristics.

$$\mathbf{b}_{t+1} = \mathbf{b}_t + \eta(t) \cdot (\mathbf{w}_t - \mathbf{b}_t)$$

**Where:**
- $\mathbf{b}_t \in \mathbb{R}^5$ — bias vector (attention, granularity, abstraction, inversion, temperature)
- $\mathbf{w}_t$ — mean bias profile of predictions that entered the winning cluster at round $t$
- $\eta(t)$ — adaptive learning rate (Formula 14)

---

## Formula 9 — Dominance Score

The primary convergence metric. One cluster "wins" when it holds a dominant fraction.

$$\text{Dominance}(k^*) = \frac{C(k^*)}{\displaystyle\sum_{k} C(k)} > \delta$$

**Stopping condition:** halt when $\text{Dominance}(k^*) > \delta$, where $\delta = 0.70$.

---

## Formula 10 — Dot Specialization Score

Measures how consistently a single dot produces similar predictions over time.

$$S_\text{spec}(d) = \frac{1}{|P_d|(|P_d|-1)/2} \sum_{i < j} S(p^d_i, p^d_j)$$

**Where:** $P_d = \{p^d_1, \dots, p^d_m\}$ is the set of recent predictions from dot $d$.

**Interpretation:**
- $S_\text{spec} \approx 1$ → specialized (always predicts in the same direction)
- $S_\text{spec} \approx 0$ → generalist (explores widely)

Used by **evolution** to identify dots whose niche is stable vs. random.

---

## Formula 11 — Cluster Entropy

Measures uncertainty in the cluster score distribution.

$$H_C = -\sum_{k} \frac{C(k)}{Z} \log \frac{C(k)}{Z}, \quad Z = \sum_k C(k)$$

**Normalized to $[0, 1]$:**

$$H_C^* = \frac{H_C}{\log |K|}$$

**Interpretation:**
- $H_C^* \approx 0$ → clear single winner (low entropy)
- $H_C^* \approx 1$ → all clusters equally scored (maximum confusion)

Used by the adaptive temperature system to increase exploration when the system is confused.

---

## Formula 12 — Temporal Stability

How much does the leading cluster centroid move between consecutive rounds?

$$\text{TS}(t) = S\!\left(\text{centroid}_t,\ \text{centroid}_{t-1}\right)$$

Using Formula 1 similarity. Range: $[-1, 1]$ (practically $[0, 1]$ for converging systems).

**Stopping condition:** halt when $\text{TS}(t) \geq 0.99$ (centroid barely moved).

Tracked by **ClusterMemory** across rounds.

---

## Formula 13 — Cross-Type Agreement

Measures consensus between dots of different specializations.

$$\text{CDA} = \frac{1}{\binom{|\mathcal{T}|}{2}} \sum_{\substack{a, b \in \mathcal{T} \\ a \neq b}} S\!\left(\hat{p}_a, \hat{p}_b\right)$$

**Where:**
- $\mathcal{T}$ — set of dot types (SEMANTIC, STRUCTURAL, CONTEXTUAL, …)
- $\hat{p}_t$ — centroid of predictions from type $t$ within a cluster

**Applied as a bonus** to the cluster's convergence score:
$$C'(k) = C(k) \cdot (1 + 0.15 \cdot \text{CDA}(k))$$

High CDA signals that the cluster's answer is robust across multiple perspectives.

---

## Formula 14 — Adaptive Learning Rate

The bias vector learning rate is reduced as convergence approaches.

$$\eta(t) = \eta_0 \cdot \left(1 - 0.8 \cdot \text{Dominance}(t-1)^2\right)$$

**Behaviour:**
- Early rounds ($\text{dom} \approx 0$): $\eta \approx \eta_0$ — fast adaptation
- Near convergence ($\text{dom} \approx 1$): $\eta \approx 0.2\,\eta_0$ — slow, stable

This prevents overshooting when the system is about to converge.

---

## Formula 15 — Hierarchical Convergence Score

Scores a **macro-cluster** (a group of micro-clusters) using both the micro-level
convergence scores and the cross-micro similarity.

$$\text{HC}(K) = \bar{C}_K \cdot \left(1 + \gamma \cdot \bar{S}^\text{cross}_K\right)$$

**Where:**
- $\bar{C}_K = \dfrac{1}{|M_K|} \sum_{m \in M_K} C(m)$ — mean micro-cluster score
- $\bar{S}^\text{cross}_K$ — mean pairwise similarity between micro-cluster centroids
- $\gamma$ — bonus weight (default 0.30)
- $M_K$ — set of micro-clusters in macro-cluster $K$

**Interpretation:** A macro-cluster scores highly if its micro-clusters are
individually strong *and* agree with each other.

---

## Stopping Conditions (Summary)

The iteration loop halts when **any** of these conditions is met:

| # | Condition | Formula | Threshold |
|---|---|---|---|
| 1 | Iteration budget | — | $t \geq T_{\max} = 12$ |
| 2 | Convergence dominance | F9 | $\text{dom}(k^*) \geq 0.70$ |
| 3 | Low novelty gain | F6 (NG) | $\text{NG}(t) < 0.05$ |
| 4 | Temporal stability | F12 | $\text{TS}(t) \geq 0.99$ |
| 5 | Score decline | — | Top score fell for 3 consecutive rounds |

---

## BaseMapping Feature Vector Layout

$$\mathbf{f}_i \in \mathbb{R}^{128}$$

| Dims | Size | Content |
|---|---|---|
| $[0:96]$ | 96 | Base embedding (hash-stable or composed from characters) |
| $[96:104]$ | 8 | Sinusoidal position encoding ($\sin/\cos$ pairs) |
| $[104:108]$ | 4 | Frequency features (relative, log, tanh-smoothed, sigmoid) |
| $[108:124]$ | 16 | Modifier flags (type, position, length, role, structural) |
| $[124:128]$ | 4 | Local context window summary |
