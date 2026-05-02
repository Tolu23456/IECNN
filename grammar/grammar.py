"""
Grammar module for IECNN.

Provides rule-based POS tagging for single vocab tokens and a grammar
transition weight table that biases generation toward grammatically
probable next tokens.

POS tags
--------
ARTICLE, NOUN, VERB, ADJECTIVE, ADVERB, PREPOSITION, CONJUNCTION, PRONOUN, OTHER

The bias is intentionally soft (scaled by bias_strength, default 0.30) so
that the semantic direction from the Peep Mechanism remains the dominant
signal — grammar shapes but does not override meaning.
"""

import numpy as np
from enum import IntEnum
from typing import List, Dict


class POS(IntEnum):
    ARTICLE     = 0
    NOUN        = 1
    VERB        = 2
    ADJECTIVE   = 3
    ADVERB      = 4
    PREPOSITION = 5
    CONJUNCTION = 6
    PRONOUN     = 7
    OTHER       = 8


_N_POS = len(POS)

_ARTICLES = {"the", "a", "an"}

_PRONOUNS = {
    "he", "she", "it", "they", "we", "you", "i", "me", "him", "her",
    "us", "them", "who", "what", "which", "that", "this", "these",
    "those", "my", "your", "his", "its", "our", "their", "myself",
    "yourself", "himself", "herself", "itself", "themselves",
}

_PREPOSITIONS = {
    "in", "on", "at", "to", "for", "with", "by", "from", "up", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "among", "under", "over", "against", "along", "following",
    "across", "behind", "beyond", "plus", "except", "without", "near",
    "since", "upon", "within", "throughout", "toward", "towards", "onto",
    "off", "down", "out", "around", "past", "despite", "inside", "outside",
}

_CONJUNCTIONS = {
    "and", "but", "or", "nor", "for", "yet", "so", "although", "because",
    "since", "unless", "until", "while", "if", "though", "whereas",
    "whether", "however", "therefore", "thus", "hence", "then", "when",
    "where", "as", "than", "once", "after", "before", "whenever",
}

_COMMON_VERBS = {
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can", "must",
    "said", "says", "say", "go", "goes", "went", "come", "came",
    "take", "took", "make", "made", "know", "knew", "think", "thought",
    "see", "saw", "look", "find", "found", "give", "gave",
    "tell", "told", "use", "used", "call", "called",
    "keep", "kept", "let", "feel", "felt", "become", "became",
    "leave", "left", "put", "mean", "meant", "set", "seem", "seemed",
    "need", "try", "tried", "ask", "asked", "show", "showed", "turn",
    "start", "move", "live", "play", "run", "ran", "stand", "hear", "heard",
    "hold", "bring", "happen", "write", "wrote", "provide", "sit", "sat",
    "appear", "change", "fall", "fell", "open", "opened", "learn", "learned",
    "walk", "win", "won", "offer", "remember", "love", "consider", "appear",
    "buy", "wait", "serve", "die", "send", "sent", "build", "built",
    "stay", "fall", "cut", "reach", "kill", "remain", "suggest", "raise",
    "pass", "sell", "require", "report", "decide", "pull",
}

_TRANSITIONS: Dict[int, Dict[int, float]] = {
    POS.ARTICLE: {
        POS.ADJECTIVE:   0.40,
        POS.NOUN:        0.35,
        POS.ADVERB:      0.15,
    },
    POS.ADJECTIVE: {
        POS.NOUN:        0.55,
        POS.ADJECTIVE:   0.20,
        POS.ADVERB:      0.05,
    },
    POS.NOUN: {
        POS.VERB:        0.40,
        POS.PREPOSITION: 0.25,
        POS.CONJUNCTION: 0.10,
        POS.PRONOUN:     0.05,
        POS.NOUN:        0.10,
    },
    POS.VERB: {
        POS.NOUN:        0.30,
        POS.ARTICLE:     0.25,
        POS.ADVERB:      0.20,
        POS.ADJECTIVE:   0.10,
        POS.PREPOSITION: 0.10,
        POS.PRONOUN:     0.10,
    },
    POS.ADVERB: {
        POS.VERB:        0.40,
        POS.ADJECTIVE:   0.30,
        POS.ADVERB:      0.10,
        POS.NOUN:        0.10,
    },
    POS.PREPOSITION: {
        POS.ARTICLE:     0.30,
        POS.NOUN:        0.25,
        POS.ADJECTIVE:   0.20,
        POS.PRONOUN:     0.20,
    },
    POS.CONJUNCTION: {
        POS.ARTICLE:     0.20,
        POS.NOUN:        0.20,
        POS.PRONOUN:     0.20,
        POS.VERB:        0.15,
        POS.ADJECTIVE:   0.10,
        POS.ADVERB:      0.10,
    },
    POS.PRONOUN: {
        POS.VERB:        0.55,
        POS.ADVERB:      0.20,
        POS.NOUN:        0.10,
    },
    POS.OTHER: {
        POS.VERB:        0.20,
        POS.NOUN:        0.20,
        POS.ARTICLE:     0.15,
        POS.PREPOSITION: 0.10,
    },
}

_START_WEIGHTS: Dict[int, float] = {
    POS.ARTICLE:     0.25,
    POS.NOUN:        0.20,
    POS.PRONOUN:     0.20,
    POS.ADJECTIVE:   0.15,
    POS.ADVERB:      0.10,
    POS.VERB:        0.05,
}


def tag_word(word: str) -> POS:
    """Rule-based POS tag for a single lowercase word."""
    w = word.lower()
    if w in _ARTICLES:
        return POS.ARTICLE
    if w in _PRONOUNS:
        return POS.PRONOUN
    if w in _PREPOSITIONS:
        return POS.PREPOSITION
    if w in _CONJUNCTIONS:
        return POS.CONJUNCTION
    if w in _COMMON_VERBS:
        return POS.VERB
    if w.endswith("ly") and len(w) > 4:
        return POS.ADVERB
    if any(w.endswith(s) for s in (
        "ful", "ous", "ive", "ical", "able", "ible", "less", "ary",
        "ory", "ent", "ant", "ular",
    )):
        return POS.ADJECTIVE
    if any(w.endswith(s) for s in (
        "tion", "sion", "ness", "ment", "ity", "ance", "ence",
        "dom", "hood", "ship", "age", "ure",
    )):
        return POS.NOUN
    if any(w.endswith(s) for s in ("ing", "ised", "ized", "ified")):
        return POS.VERB
    if w.endswith("ed") and len(w) > 4:
        return POS.VERB
    return POS.NOUN


class GrammarGuide:
    """
    Pre-tags a fixed word list and provides fast per-step weight arrays
    for generation, biasing toward grammatically likely next tokens.

    Parameters
    ----------
    words : list of str
        The generation vocab (same list used in causal_generate).
    bias_strength : float
        Scale factor on grammar bonuses.  Default 0.30 — soft nudge,
        semantic direction from Peep still dominates.
    """

    def __init__(self, words: List[str], bias_strength: float = 0.30):
        self.words         = words
        self.bias_strength = bias_strength
        n = len(words)

        self._tags = np.array(
            [int(tag_word(w)) for w in words], dtype=np.int32
        )

        n_rows = _N_POS + 1
        self._weight_matrix = np.zeros((n_rows, n), dtype=np.float32)

        for last_pos in range(_N_POS):
            table = _TRANSITIONS.get(last_pos, {})
            for next_pos, bonus in table.items():
                mask = (self._tags == int(next_pos))
                self._weight_matrix[last_pos][mask] = float(bonus) * bias_strength

        for next_pos, bonus in _START_WEIGHTS.items():
            mask = (self._tags == int(next_pos))
            self._weight_matrix[_N_POS][mask] = float(bonus) * bias_strength

    def weights(self, last_token: str = "") -> np.ndarray:
        """
        Return (V,) additive weight array for the next token.
        Pass last_token="" or omit for sentence-start position.
        """
        if not last_token:
            return self._weight_matrix[_N_POS]
        pos = int(tag_word(last_token.lower()))
        return self._weight_matrix[pos]

    def tag(self, word: str) -> POS:
        return tag_word(word)
