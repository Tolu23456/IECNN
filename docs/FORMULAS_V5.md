# IECNN V5 SOTA Formulas

The mathematical foundation of the IECNN architecture.

---

## Core Operations

### F1 — Similarity Score
$$S(p_i, p_j) = \alpha \cdot \cos(p_i, p_j) + (1 - \alpha) \cdot A(p_i, p_j)$$
Balances structural direction and magnitude agreement.

### F2 — Convergence Score
$$C(k) = \frac{1}{|k|^2} \sum_{i,j \in k} S(p_i, p_j) \cdot \bar{c}_k$$
Measures cluster cohesion and confidence.

---

## Selection & Reinforcement

### F16 — Emergent Utility Gradient (EUG)
$$U(t) = E[C_{t+1}(p)] - C_t(p)$$
The expected gain in structural clarity in the next round.

### F17 — Dot Reinforcement Pressure (DRP)
$$R_d = \lambda_1 C_d + \lambda_2 S_d + \lambda_3 J - \lambda_4 N_d$$
Pressure score used for tournament selection and inline mutation.

### F24 — Dot Fitness Function
$$F_d = R_d + \alpha C_d + \beta S_d + \gamma U_d - \delta N_d + \sigma Surprise + \zeta Grounding$$
Comprehensive score used by the evolution engine to rank dots.

---

## Cognitive Control (AGI Layer)

### F21 — Global Energy
$$E(t) = Entropy + Instability - Dominance$$
A measure of the system's "mental tension."

### F22 — Master System Objective
$$J(t) = Convergence + Utility - Energy$$
The global goal the architecture seeks to maximize.

### F23 — Adaptive Memory Plasticity
$$\rho(t) = \rho_{base} \cdot \tanh(1.0 - Energy)$$
Determines the rate at which knowledge is consolidated vs. preserved.

---

## V5 SOTA Reasoning

### F31 — Hierarchical Concept Formation (CFO)
$$Concept(k) = CFO(Centroid_k, Metadata_k)$$
A two-stage transform that names and registers winning clusters as permanent composite bases.

### F35 — Recursive Thinking Depth
$$Steps = \lfloor ReasoningDepth_{dot} \cdot 5 \rfloor$$
The number of internal simulations a dot performs during its "Inner Monologue."
