# IECNN Formulas

All formulas are custom-built for the IECNN architecture.

---

## 1. Similarity Score

Measures how similar two predictions are within the unified BaseMapping representation space.

$$S(p_i, p_j) = \alpha \cdot \cos(p_i, p_j) + (1 - \alpha) \cdot A(p_i, p_j)$$

Where:
- $p_i, p_j$ — two predictions in BaseMapping space
- $\cos(p_i, p_j) = \dfrac{p_i \cdot p_j}{\|p_i\| \|p_j\|}$ — cosine similarity (structural match)
- $A(p_i, p_j)$ — agreement strength (how strongly the two predictions reinforce each other)
- $\alpha \in [0, 1]$ — balance weight between similarity and agreement

---

## 2. Convergence Score

Scores a cluster $k$ of predictions by its internal cohesion and confidence.

$$C(k) = \frac{1}{|k|^2} \sum_{i \in k} \sum_{j \in k} S(p_i, p_j) \cdot \bar{c}_k$$

Where:
- $|k|$ — number of predictions in cluster $k$
- $S(p_i, p_j)$ — similarity score between predictions $i$ and $j$
- $\bar{c}_k = \dfrac{1}{|k|} \sum_{i \in k} c_i$ — mean confidence of predictions in the cluster
- $c_i$ — confidence score of individual prediction $p_i$

---

## 3. Attention Formula

Applied within the AIM layer to focus on the most relevant parts of a prediction or context.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Where:
- $Q$ — query matrix (what the dot is looking for)
- $K$ — key matrix (what is available in context)
- $V$ — value matrix (the information to retrieve)
- $d_k$ — dimensionality of the key vectors (scaling factor to stabilise gradients)

> In IECNN, $Q$, $K$, and $V$ are derived from BaseMapping representations, not raw token embeddings.

---

## 4. AIM Transformation

Transforms a prediction into an inverted candidate, then applies attention to refine it.

$$\hat{p}_i = \text{Attention}\!\left(Q, K,\, \mathcal{I}(p_i)\right)$$

Where:
- $p_i$ — original dot prediction
- $\mathcal{I}(p_i)$ — inversion of $p_i$ (feature, context, spatial, scale, abstraction, or noise inversion)
- $\hat{p}_i$ — AIM-refined candidate prediction

---

## 5. Pruning Threshold

A prediction or cluster is kept only if its convergence score exceeds the pruning threshold $\tau$.

$$\text{Keep cluster } k \iff C(k) > \tau$$

For early soft filtering, a lower threshold $\tau_{\text{soft}} < \tau$ is applied to individual predictions before clustering:

$$\text{Keep prediction } p_i \iff c_i > \tau_{\text{soft}}$$

---

## 6. Novelty Gain

Measures how much new information each iteration produces. Used by the Iteration Controller to detect when exploration is exhausted.

$$\text{NG}(t) = \frac{|\text{NewClusters}(t)|}{|\text{TotalClusters}(t)|}$$

Where:
- $\text{NewClusters}(t)$ — clusters at round $t$ that do not map into any cluster from round $t-1$
- $\text{TotalClusters}(t)$ — all clusters at round $t$

**Stop iterating when:** $\text{NG}(t) < \epsilon$, where $\epsilon$ is a small threshold close to 0.

---

## 7. Sampling Temperature

Controls diversity of dot outputs. Higher temperature = more diverse predictions; lower = more focused.

$$P(\text{dot generates variant } v) \propto \exp\!\left(\frac{\text{score}(v)}{T}\right)$$

Where:
- $\text{score}(v)$ — quality or confidence score of variant $v$
- $T$ — sampling temperature from the bias vector

---

## 8. Bias Vector Update

After each learning cycle, the dot generation bias vector $\mathbf{b}$ is updated based on which strategies produced winning predictions.

$$\mathbf{b}_{t+1} = \mathbf{b}_t + \eta \cdot (\mathbf{w}_t - \mathbf{b}_t)$$

Where:
- $\mathbf{b}_t$ — current bias vector (attention, granularity, abstraction, inversion, temperature)
- $\mathbf{w}_t$ — bias profile of the winning predictions in round $t$
- $\eta \in [0, 1]$ — learning rate controlling how fast the bias shifts

---

## 9. Stability Condition (Convergence Dominance)

The system halts when one cluster dominates the total weight pool.

$$\text{Dominance}(k^*) = \frac{C(k^*)}{\sum_{k} C(k)} > \delta$$

Where:
- $k^*$ — the leading cluster
- $\delta$ — dominance threshold (e.g., 0.75 means the top cluster holds 75% of total weight)

**Stop iterating when:** $\text{Dominance}(k^*) > \delta$ **or** $\text{NG}(t) < \epsilon$ **or** $t \geq T_{\max}$
