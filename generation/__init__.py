"""
IECNN Generation Engine — pluggable score-processor pipeline for
high-quality autoregressive text generation.

Analogous to HuggingFace's LogitsProcessorList + GenerationMixin but
implemented entirely in IECNN's convergence-based framework:
  • No gradient computation
  • No token probability distributions (we use cosine similarity scores)
  • No softmax logits (we work in unit-sphere embedding space)

Public API
----------
from generation import (
    # Score processors (apply in this recommended order)
    SemanticFieldBias,
    RepetitionPenalty,
    NoRepeatNGram,
    DegenerationPenalty,
    MinLengthGuard,
    TypicalFilter,
    NucleusFilter,
    MinPFilter,
    DynamicTemperature,
    ScoreProcessorList,
    # Context enrichment
    ContextHistory,
    ContextAnchor,
    # Voting
    MultiHeadConvergence,
)
"""

from generation.processors import (
    SemanticFieldBias,
    VocabFrequencyPrior,
    RepetitionPenalty,
    NoRepeatNGram,
    DegenerationPenalty,
    MinLengthGuard,
    ExponentialDecayLength,
    TypicalFilter,
    NucleusFilter,
    MinPFilter,
    EtaFilter,
    TailFreeFilter,
    TopKFilter,
    PromptDriftPenalty,
    LocalSemanticFilter,
    DotVariancePenalty,
    SurpriseBonus,
    DynamicTemperature,
    MirostatScheduler,
    BigramContinuationBonus,
    SemanticProximityPenalty,
    ScoreProcessorList,
    softmax_sample,
)
from generation.context_hist import ContextHistory, ContextAnchor
from generation.multihead    import MultiHeadConvergence

__all__ = [
    "SemanticFieldBias", "VocabFrequencyPrior",
    "RepetitionPenalty", "NoRepeatNGram",
    "DegenerationPenalty", "MinLengthGuard", "ExponentialDecayLength",
    "TypicalFilter", "NucleusFilter", "MinPFilter", "EtaFilter",
    "TailFreeFilter", "TopKFilter",
    "PromptDriftPenalty", "LocalSemanticFilter", "DotVariancePenalty",
    "SurpriseBonus",
    "DynamicTemperature", "MirostatScheduler",
    "BigramContinuationBonus", "SemanticProximityPenalty",
    "ScoreProcessorList", "softmax_sample",
    "ContextHistory", "ContextAnchor",
    "MultiHeadConvergence",
]
