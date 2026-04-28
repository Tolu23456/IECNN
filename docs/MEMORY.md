# IECNN Memory: The Hierarchical Knowledge System

IECNN implements a multi-layered memory architecture that balances short-term context with long-term factual reliability.

---

## 1. Dot Memory (Short-Term & Adaptive)

Tracks the performance of individual Neural Dots within and across calls.

### Key Metrics
-   **Effectiveness**: The fraction of a dot's predictions that entered the winning cluster.
-   **Causal Accuracy**: How well the dot predicted the *actual* next token in a training sequence.
-   **Specialization**: A measure of a dot's internal consistency (using Welford's algorithm for $O(1)$ variance).

### Active Episodic Priming
Dots store **Episodic Exemplars** (successful mapping history). During the forward pass, the system retrieves and blends the top-3 most contextually relevant exemplars to "prime" the dot's weights, allowing for fast context-switching.

---

## 2. Cluster Memory (Volatile Patterns)

Maintains a rolling timeline of winning centroids within a single session. This ensures that the system maintains temporal stability between consecutive rounds of the same input.

---

## 3. World Knowledge Graph (Long-Term/M_long)

The permanent factual repository of the model.

### Surprise-Driven Consolidation
Knowledge is not added blindly. The system uses a **Surprise Filter**:
1.  New winning patterns are compared against existing graph nodes.
2.  If the pattern is **novel** (high surprise), a new conceptual node is created.
3.  If the pattern is **redundant** (low surprise), it is used to refine existing nodes.

This mechanism solves the "Explosive Memory" problem, keeping the model's footprint manageable while maximizing information density.

### Relational Linking
Concepts that frequently co-occur in winning clusters are automatically linked with edges in the graph, building a structured map of relational facts.

---

## 4. Adaptive Plasticity (F23)

Memory updates follow the principle of **Representational Stability**:
-   **Stable States**: When the system is confident and stable, the plasticity rate ($\rho$) increases, allowing it to quickly absorb new information.
-   **Unstable States**: When the system is confused (high energy), memory is "frozen" to prevent the corruption of existing knowledge by noise.
