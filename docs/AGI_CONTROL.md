# IECNN AGI Control: Meta-Cognition & Dynamic Policy

IECNN includes an outer loop of cognitive management that allows the system to monitor and modulate its own "thinking" process.

---

## 1. Cognitive State Vector (CSV)

The CSV is the "internal mirror" of the system. It summarizes the mental state across 7 dimensions:
-   **Entropy**: Current level of confusion (uncertainty).
-   **Dominance**: The strength of the current leading consensus.
-   **Stability**: How much the representation is changing between rounds.
-   **Energy**: Global system tension (entropy + instability - dominance).
-   **EUG**: Emergent Utility Gradient (expected future improvement).
-   **Experience**: Cumulative call count.
-   **Reasoning Depth**: Current depth of internal dot simulations.

---

## 2. The Self-Model (SM)

The Self-Model is the "Ego" of the architecture. It maps the CSV to a set of **Internal Cognitive Actions**.

### Dynamic Thinking Policy (Light vs. Deep)
Based on the input's complexity, the Self-Model decides:
-   **Light Intuition**: For simple, familiar inputs, use fewer iterations and low reasoning depth to save computation.
-   **Deep Simulation**: For novel or contradictory inputs (high surprise), increase the iteration budget and deepen the "Neural Dot" reasoning loops.

### Stagnation Response (Aggressive Breakthrough)
If the system detects it is stuck in a local minima (low EUG), the SM triggers breakthrough actions:
-   Injection of high-variance noise.
-   Sudden temperature spikes in dots.
-   Random "Dimension Flips" to force the system into a new representational basin.

---

## 3. Learning the Policy

The SM learns via a discrete reinforcement signal:
$$Reward = \Delta Utility - \Delta Energy$$
Actions that lead to faster convergence (Utility) with less system tension (Energy) are reinforced in the SM's internal policy matrix.
