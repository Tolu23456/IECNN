"""
AGI Router — high-level intent detection and sub-pipeline routing.
"""

import numpy as np
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Intent(Enum):
    CHAT      = "chat"
    CODE      = "code"
    IMAGE     = "image"
    AUDIO     = "audio"
    REASONING = "reasoning"
    SEARCH    = "search"

class AGIRouter:
    """
    Analyzes user input to route it to the most appropriate IECNN sub-pipeline.
    Uses a lightweight IECNN pass (Layer 3-7) for intent classification.
    """
    def __init__(self, model):
        self.model = model
        # Intent centroids in 256-dim space (learned or pre-seeded)
        self.intent_anchors = {
            Intent.CHAT:      self._make_anchor("chat conversation dialogue talk"),
            Intent.CODE:      self._make_anchor("python code program function run"),
            Intent.IMAGE:     self._make_anchor("image picture draw generate visual"),
            Intent.AUDIO:     self._make_anchor("audio sound speak voice music"),
            Intent.REASONING: self._make_anchor("why because logic prove analyze"),
            Intent.SEARCH:    self._make_anchor("search find web lookup info"),
        }

    def _make_anchor(self, text: str) -> np.ndarray:
        """Create a stable semantic anchor for an intent."""
        # Use simple mean-pool of tokens for the anchor
        bm = self.model.base_mapper.transform(text)
        return bm.pool("mean")

    def detect_intent(self, text: str) -> Intent:
        """
        Route input to intent based on latent similarity.
        In SOTA, this uses a specialized 'Executive Pass'.
        """
        # 1. Get latent representation of input
        # Use a fast encode (no full pipeline for routing)
        input_vec = self.model.base_mapper.transform(text).pool("mean")

        # 2. Compare against intent anchors
        from formulas.formulas import similarity_score
        best_intent = Intent.CHAT
        best_sim = -1.0

        for intent, anchor in self.intent_anchors.items():
            sim = similarity_score(input_vec, anchor, alpha=self.model.alpha)
            if sim > best_sim:
                best_sim = sim
                best_intent = intent

        return best_intent

    def route(self, text: str, history: List = None) -> Tuple[Intent, str]:
        """Unified entry point for the AGI agent (v6 SOTA)."""
        intent = self.detect_intent(text)

        # History compression: incorporate previous context into current latent state
        context_prefix = ""
        if history:
            # Take last turn for immediate context
            last_u, last_b = history[-1]
            context_prefix = f"context [ {last_u} -> {last_b} ] "

        # Enhanced Routing Logic: Context-aware prompt construction
        if intent == Intent.CODE:
            prompt = f"{context_prefix}system [MODE: CODING] user {text} action WRITE_CODE bot"
        elif intent == Intent.IMAGE:
            prompt = f"{context_prefix}system [MODE: VISUAL] user {text} action GENERATE_IMAGE bot"
        elif intent == Intent.AUDIO:
            prompt = f"{context_prefix}system [MODE: AUDIO] user {text} action SPEAK bot"
        elif intent == Intent.REASONING:
            prompt = f"{context_prefix}system [MODE: LOGIC] user {text} action REASON bot"
        elif intent == Intent.SEARCH:
            prompt = f"{context_prefix}system [MODE: SEARCH] user {text} action SEARCH bot"
        else:
            prompt = f"{context_prefix}user {text} bot"

        return intent, prompt
