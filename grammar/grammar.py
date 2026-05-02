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
    # Copula / auxiliaries
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can", "must",
    # High-frequency main verbs
    "said", "says", "say", "go", "goes", "went", "come", "came",
    "take", "took", "taken", "make", "made", "know", "knew", "known",
    "think", "thought", "see", "saw", "seen", "look", "looked", "looking",
    "find", "found", "give", "gave", "given", "tell", "told",
    "use", "used", "call", "called", "keep", "kept", "let",
    "feel", "felt", "become", "became", "leave", "left", "put",
    "mean", "meant", "set", "seem", "seemed", "need", "try", "tried",
    "ask", "asked", "show", "showed", "shown", "turn", "turned",
    "start", "started", "move", "moved", "live", "lived",
    "play", "played", "run", "ran", "stand", "stood", "hear", "heard",
    "hold", "held", "bring", "brought", "happen", "happened",
    "write", "wrote", "written", "provide", "provided",
    "sit", "sat", "appear", "appeared", "change", "changed",
    "fall", "fell", "fallen", "open", "opened", "learn", "learned", "learnt",
    "walk", "walked", "win", "won", "offer", "offered",
    "remember", "remembered", "love", "loved", "consider", "considered",
    "buy", "bought", "wait", "waited", "serve", "served",
    "die", "died", "send", "sent", "build", "built",
    "stay", "stayed", "cut", "reach", "reached", "kill", "killed",
    "remain", "remained", "suggest", "suggested", "raise", "raised",
    "pass", "passed", "sell", "sold", "require", "required",
    "report", "reported", "decide", "decided", "pull", "pulled",
    "help", "helped", "follow", "followed", "stop", "stopped",
    "create", "created", "speak", "spoke", "spoken",
    "read", "return", "returned", "carry", "carried",
    "lose", "lost", "allow", "allowed", "add", "added",
    "spend", "spent", "grow", "grew", "grown",
    "meet", "met", "lead", "led", "understand", "understood",
    "watch", "watched", "continue", "continued",
    "claim", "claimed", "believe", "believed",
    "include", "included", "involve", "involved",
    "describe", "described", "produce", "produced",
    "enter", "entered", "accept", "accepted",
    "receive", "received", "hit", "got", "gotten",
    "began", "begin", "begun", "bring", "brought",
    "choose", "chose", "chosen", "drive", "drove",
    "fight", "fought", "fly", "flew", "flown",
    "grow", "knew", "rise", "rose", "risen",
    "run", "shake", "shook", "shaken", "steal", "stole",
    "swim", "swam", "swum", "throw", "threw", "thrown",
    "wake", "woke", "woken", "wear", "wore", "worn",
}

_COMMON_ADJECTIVES = {
    # Common descriptive adjectives
    "good", "bad", "great", "small", "large", "big", "little", "long",
    "high", "low", "old", "new", "young", "early", "late", "right", "wrong",
    "real", "true", "false", "free", "full", "open", "close", "clear",
    "hard", "soft", "strong", "weak", "fast", "slow", "short", "tall",
    "hot", "cold", "warm", "cool", "dark", "light", "bright", "deep",
    "wide", "narrow", "thick", "thin", "heavy", "light", "rich", "poor",
    "happy", "sad", "angry", "afraid", "glad", "sorry", "sure", "ready",
    "next", "last", "first", "second", "third", "main", "other", "own",
    "same", "different", "important", "possible", "impossible",
    "local", "national", "international", "public", "private", "general",
    "special", "common", "single", "whole", "entire", "complete",
    "recent", "current", "former", "future", "present", "past",
    "human", "natural", "social", "political", "economic", "cultural",
    "military", "religious", "scientific", "medical", "legal",
    "white", "black", "red", "green", "blue", "yellow", "brown", "grey",
    "gray", "orange", "purple", "pink", "golden", "silver",
    "beautiful", "ugly", "pretty", "lovely", "wonderful", "terrible",
    "excellent", "perfect", "simple", "complex", "difficult", "easy",
    "safe", "dangerous", "serious", "funny", "interesting", "boring",
    "famous", "unknown", "popular", "rare", "unique", "typical",
    "major", "minor", "central", "basic", "key", "prime", "chief",
    "senior", "junior", "ancient", "modern", "traditional", "classic",
    "foreign", "domestic", "urban", "rural", "wild", "tame",
    "physical", "mental", "moral", "emotional", "spiritual",
    "official", "unofficial", "formal", "informal", "ordinary",
}

_COMMON_ADVERBS = {
    # Frequency
    "always", "usually", "often", "sometimes", "rarely", "never",
    "frequently", "occasionally", "generally", "normally", "typically",
    # Manner
    "quickly", "slowly", "carefully", "easily", "suddenly", "quietly",
    "clearly", "closely", "directly", "firmly", "freely", "fully",
    "highly", "largely", "mainly", "nearly", "openly", "partly",
    "rapidly", "rather", "really", "simply", "slightly", "strongly",
    "truly", "widely", "briefly", "equally", "exactly", "fairly",
    # Degree
    "very", "quite", "just", "even", "still", "also", "too", "only",
    "much", "more", "most", "less", "least", "enough", "almost",
    "about", "already", "yet", "again", "soon", "now", "then",
    # Discourse
    "however", "therefore", "thus", "hence", "instead", "otherwise",
    "meanwhile", "nevertheless", "nonetheless", "indeed", "actually",
    "certainly", "apparently", "presumably", "perhaps", "probably",
    "possibly", "likely", "finally", "eventually", "initially",
    "previously", "recently", "currently", "immediately", "later",
    "together", "separately", "especially", "particularly", "specifically",
    "roughly", "approximately", "relatively", "absolutely", "completely",
    "entirely", "mostly", "especially", "merely", "barely", "hardly",
    "unfortunately", "fortunately", "surprisingly", "increasingly",
    "subsequently", "consequently", "accordingly", "simultaneously",
}

_BIGRAM_TRANSITIONS: Dict[tuple, Dict[int, float]] = {
    (POS.ARTICLE,     POS.ADJECTIVE):  {POS.NOUN: 0.82, POS.ADJECTIVE: 0.12},
    (POS.ARTICLE,     POS.ADVERB):     {POS.ADJECTIVE: 0.65, POS.NOUN: 0.25},
    (POS.ARTICLE,     POS.NOUN):       {POS.VERB: 0.55, POS.PREPOSITION: 0.25, POS.CONJUNCTION: 0.10},
    (POS.PRONOUN,     POS.VERB):       {POS.NOUN: 0.30, POS.ARTICLE: 0.28, POS.ADVERB: 0.22, POS.ADJECTIVE: 0.12},
    (POS.PRONOUN,     POS.ADVERB):     {POS.VERB: 0.55, POS.ADJECTIVE: 0.25, POS.ADVERB: 0.12},
    (POS.VERB,        POS.ARTICLE):    {POS.NOUN: 0.55, POS.ADJECTIVE: 0.30, POS.ADVERB: 0.10},
    (POS.VERB,        POS.ADVERB):     {POS.VERB: 0.42, POS.ADJECTIVE: 0.28, POS.NOUN: 0.18},
    (POS.VERB,        POS.ADJECTIVE):  {POS.NOUN: 0.60, POS.ADJECTIVE: 0.20, POS.CONJUNCTION: 0.12},
    (POS.PREPOSITION, POS.ARTICLE):    {POS.NOUN: 0.62, POS.ADJECTIVE: 0.28, POS.ADVERB: 0.08},
    (POS.PREPOSITION, POS.ADJECTIVE):  {POS.NOUN: 0.75, POS.ADJECTIVE: 0.15},
    (POS.PREPOSITION, POS.PRONOUN):    {POS.VERB: 0.60, POS.ADVERB: 0.20, POS.NOUN: 0.15},
    (POS.NOUN,        POS.PREPOSITION):{POS.ARTICLE: 0.38, POS.NOUN: 0.28, POS.ADJECTIVE: 0.20, POS.PRONOUN: 0.12},
    (POS.NOUN,        POS.CONJUNCTION):{POS.ARTICLE: 0.25, POS.NOUN: 0.22, POS.PRONOUN: 0.22, POS.VERB: 0.18},
    (POS.NOUN,        POS.VERB):       {POS.NOUN: 0.30, POS.ARTICLE: 0.28, POS.ADVERB: 0.20, POS.ADJECTIVE: 0.12},
    (POS.ADJECTIVE,   POS.NOUN):       {POS.VERB: 0.50, POS.PREPOSITION: 0.28, POS.CONJUNCTION: 0.14},
    (POS.ADJECTIVE,   POS.ADJECTIVE):  {POS.NOUN: 0.70, POS.ADJECTIVE: 0.18, POS.CONJUNCTION: 0.08},
    (POS.ADVERB,      POS.ADJECTIVE):  {POS.NOUN: 0.72, POS.ADJECTIVE: 0.18, POS.ADVERB: 0.06},
    (POS.ADVERB,      POS.VERB):       {POS.NOUN: 0.32, POS.ARTICLE: 0.28, POS.ADVERB: 0.22, POS.ADJECTIVE: 0.12},
    (POS.CONJUNCTION, POS.PRONOUN):    {POS.VERB: 0.72, POS.ADVERB: 0.18, POS.NOUN: 0.08},
    (POS.CONJUNCTION, POS.ARTICLE):    {POS.NOUN: 0.55, POS.ADJECTIVE: 0.30, POS.ADVERB: 0.10},
    (POS.CONJUNCTION, POS.VERB):       {POS.NOUN: 0.35, POS.ARTICLE: 0.30, POS.ADVERB: 0.22},
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
    """Rule-based POS tag for a single lowercase word.

    Priority order:
      closed-class sets first (articles, pronouns, prepositions, conjunctions)
      then open-class by explicit list (verbs, adjectives, adverbs)
      then morphological suffix rules as fallback
    """
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
    if w in _COMMON_ADJECTIVES:
        return POS.ADJECTIVE
    if w in _COMMON_ADVERBS:
        return POS.ADVERB
    # Suffix-based fallbacks (ordered most-specific first)
    if w.endswith("ly") and len(w) > 4:
        return POS.ADVERB
    if any(w.endswith(s) for s in (
        "ful", "ous", "ive", "ical", "able", "ible", "less", "ary",
        "ory", "ent", "ant", "ular", "ic",
    )):
        return POS.ADJECTIVE
    if any(w.endswith(s) for s in (
        "tion", "sion", "ness", "ment", "ity", "ance", "ence",
        "dom", "hood", "ship", "age", "ure",
    )):
        return POS.NOUN
    if any(w.endswith(s) for s in ("ing", "ised", "ized", "ified", "ifies")):
        return POS.VERB
    if w.endswith("ed") and len(w) > 4:
        return POS.VERB
    return POS.NOUN


class GrammarGuide:
    """
    Pre-tags a fixed word list and provides fast per-step weight arrays
    for generation, biasing toward grammatically likely next tokens.

    Supports both unigram and bigram POS context:
      • weights(last_token)            — standard single-token context
      • weights(last_token, prev_token) — bigram context (stronger signal)

    The bigram table has specific high-confidence transitions for common
    two-token patterns (e.g. "the big ___" → ARTICLE+ADJ strongly implies
    NOUN next, stronger than ADJECTIVE alone would suggest).

    Parameters
    ----------
    words : list of str
        The generation vocab (same list used in causal_generate).
    bias_strength : float
        Scale factor on grammar bonuses.  Default 0.50 — moderate nudge;
        semantic direction from Peep still dominates but grammar is clearly
        felt.
    """

    def __init__(self, words: List[str], bias_strength: float = 0.50):
        self.words         = words
        self.bias_strength = bias_strength
        n = len(words)

        self._tags = np.array(
            [int(tag_word(w)) for w in words], dtype=np.int32
        )

        # ── Unigram weight matrix: (n_pos+1, V) ─────────────────────
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

        # ── Bigram weight matrix: (n_pos, n_pos, V) ─────────────────
        # Built for (prev_pos, last_pos) pairs defined in _BIGRAM_TRANSITIONS.
        # For pairs not in the dict, falls back to the unigram row.
        self._bigram_matrix = np.zeros((_N_POS, _N_POS, n), dtype=np.float32)

        for (prev_pos, last_pos), table in _BIGRAM_TRANSITIONS.items():
            row = np.zeros(n, dtype=np.float32)
            for next_pos, bonus in table.items():
                mask = (self._tags == int(next_pos))
                row[mask] = float(bonus) * bias_strength
            self._bigram_matrix[prev_pos, last_pos] = row

        # Fill any (prev, last) pair not explicitly in bigram dict with
        # the unigram row for last_pos (so fallback is always correct).
        for pp in range(_N_POS):
            for lp in range(_N_POS):
                if (pp, lp) not in _BIGRAM_TRANSITIONS:
                    self._bigram_matrix[pp, lp] = self._weight_matrix[lp]

    def weights(self, last_token: str = "",
                prev_token: str = "") -> np.ndarray:
        """
        Return (V,) additive weight array for the next token.

        Parameters
        ----------
        last_token  : the most recently emitted token (or "")
        prev_token  : the token before last_token (or "")
        """
        if not last_token:
            return self._weight_matrix[_N_POS]
        last_pos = int(tag_word(last_token.lower()))
        if prev_token:
            prev_pos = int(tag_word(prev_token.lower()))
            return self._bigram_matrix[prev_pos, last_pos]
        return self._weight_matrix[last_pos]

    def anti_rep_mask(self, tag_history: list, penalty: float = 0.30) -> np.ndarray:
        """Return a (V,) penalty array that suppresses recently repeated POS.

        If the last 3 tags are the same POS, penalise all tokens of that POS
        by `penalty` (negative additive bias).  Helps break monotone runs of
        the same part-of-speech.
        """
        mask = np.zeros(len(self.words), dtype=np.float32)
        if len(tag_history) >= 3:
            last3 = [int(tag_word(t)) for t in tag_history[-3:]]
            if last3[0] == last3[1] == last3[2]:
                rep_pos = last3[0]
                mask[self._tags == rep_pos] = -penalty
        return mask

    def tag(self, word: str) -> POS:
        return tag_word(word)
