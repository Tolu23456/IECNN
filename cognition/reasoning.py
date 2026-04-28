"""
Deep Reasoning Layer — recursive causal discovery and counterfactual analysis.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import similarity_score

class DeepReasoningLayer:
    """
    Implements 'What-If' reasoning by simulating interventions.

    1. Selects key structural tokens.
    2. Swaps them with counterfactual alternatives.
    3. Measures the 'Causal Impact' on the system consensus.
    """
    def __init__(self, model, alpha: float = 0.7):
        self.model = model
        self.alpha = alpha
        self._rng = np.random.RandomState(42)

    def simulate_intervention(self, text: str, token_idx: int, replacement: str) -> float:
        """
        Simulate a counterfactual intervention: What if token X was replacement Y?
        Returns the cosine shift (Causal Impact) in the final output latent.
        """
        # 1. Baseline Run
        baseline_res = self.model.run(text)
        baseline_vec = baseline_res.output

        # 2. Intervention
        # Use model's internal tokenizer for structural consistency
        tokens = self.model.base_mapper._tokenize(text)
        if token_idx >= len(tokens): return 0.0

        original_tok = tokens[token_idx]
        tokens[token_idx] = replacement
        # Re-join tokens: since _tokenize lowercases, we use a simple join
        intervention_text = " ".join(tokens)

        # 3. Intervention Run
        intervention_res = self.model.run(intervention_text)
        intervention_vec = intervention_res.output

        # 4. Calculate Causal Impact
        impact = 1.0 - similarity_score(baseline_vec, intervention_vec, self.alpha)
        return float(impact)

    def discover_dependencies(self, text: str) -> Dict[int, float]:
        """
        Heuristic Causal Discovery: identify which tokens have the
        most structural influence on the sentence's meaning.
        """
        tokens = self.model.base_mapper._tokenize(text)
        impacts = {}

        # Sample up to 5 tokens for intervention
        indices = self._rng.choice(len(tokens), size=min(5, len(tokens)), replace=False)

        for idx in indices:
            # Simple intervention: What if this token was a period (empty/null)?
            impact = self.simulate_intervention(text, idx, ".")
            impacts[int(idx)] = impact

        return impacts
