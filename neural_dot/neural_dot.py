"""
Neural Dot — the fundamental prediction unit of IECNN.

A neural dot is fundamentally different from a neuron:
  Neuron   → tiny signal-passer in a fixed layered graph, no agency
  NeuralDot → complete stateless mini-predictor with its own view of the world

Each dot is characterized by:
  - A DotType: determines what aspect of the input it specialises in
  - A BiasVector: 5-dimensional control vector shaping its attention and output
  - Weight matrices W, P (per-head projection), Q (attention query basis)
  - No shared weights with any other dot

Dot types:
  SEMANTIC   — attends to semantic embedding dims [0:64], fine-grain
  STRUCTURAL — attends to structural/positional features [64:128]
  CONTEXTUAL — global view across all tokens, high abstraction
  RELATIONAL — detects cross-token relational patterns
  TEMPORAL   — sequential, position-weighted pooling
  GLOBAL     — uniform broad pooling, the "overview" specialist

Multi-head prediction:
  Each dot generates N_HEADS predictions using different head-specific
  projection matrices. This increases diversity from the same dot.

Memory-guided attention:
  If a dot's recent centroid from DotMemory is available, it is blended
  into the attention query to bias the dot toward historically good regions.
"""

import numpy as np
from enum import IntEnum
from typing import List, Optional, Tuple, Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import prediction_confidence, bias_vector_update


# ── Dot Types ─────────────────────────────────────────────────────────

class DotType(IntEnum):
    SEMANTIC   = 0  # semantic content
    STRUCTURAL = 1  # structure, position, flags
    CONTEXTUAL = 2  # full context, high abstraction
    RELATIONAL = 3  # cross-token relations
    TEMPORAL   = 4  # sequential, position-weighted
    GLOBAL     = 5  # uniform broad overview
    LOGIC      = 6  # structural/logical patterns (if-then, because)
    MORPH      = 7  # word structure and morphological flags


_TYPE_NAMES = {
    DotType.SEMANTIC:   "semantic",
    DotType.STRUCTURAL: "structural",
    DotType.CONTEXTUAL: "contextual",
    DotType.RELATIONAL: "relational",
    DotType.TEMPORAL:   "temporal",
    DotType.GLOBAL:     "global",
    DotType.LOGIC:      "logic",
    DotType.MORPH:      "morph",
}

_TYPE_DIM_RANGES = {
    DotType.SEMANTIC:   (0,   128),
    DotType.STRUCTURAL: (128, 256),
    DotType.CONTEXTUAL: (0,   256),
    DotType.RELATIONAL: (0,   256),
    DotType.TEMPORAL:   (0,   256),
    DotType.GLOBAL:     (0,   256),
    DotType.LOGIC:      (128, 256),
    DotType.MORPH:      (236, 252),  # focus strictly on morphological flag dims
}

# Default bias presets per type (attention, granularity, abstraction, inversion, temperature)
_TYPE_BIAS_PRESETS: Dict[DotType, Tuple[float, ...]] = {
    DotType.SEMANTIC:   (0.7, 0.3, 0.5, 0.3, 0.8),
    DotType.STRUCTURAL: (0.6, 0.5, 0.3, 0.2, 0.6),
    DotType.CONTEXTUAL: (0.4, 0.8, 0.8, 0.4, 1.2),
    DotType.RELATIONAL: (0.8, 0.4, 0.6, 0.5, 1.0),
    DotType.TEMPORAL:   (0.5, 0.6, 0.4, 0.3, 0.9),
    DotType.GLOBAL:     (0.3, 0.9, 0.7, 0.2, 0.7),
    DotType.LOGIC:      (0.9, 0.6, 0.7, 0.1, 0.5),
    DotType.MORPH:      (0.8, 0.2, 0.4, 0.1, 0.4), # precise focus
}

N_HEADS = 4  # number of prediction heads per dot

_NEXT_DOT_ID = 1000

def set_next_dot_id(val: int):
    global _NEXT_DOT_ID
    _NEXT_DOT_ID = val

def get_next_dot_id() -> int:
    global _NEXT_DOT_ID
    res = _NEXT_DOT_ID
    _NEXT_DOT_ID += 1
    return res


# ── Bias Vector ───────────────────────────────────────────────────────

class BiasVector:
    """
    5-dimensional control vector that shapes how a dot sees and processes input.

    Dimensions:
      0  attention_bias      — sharpness of focus (0=uniform, 1=peaked)
      1  granularity_bias    — patch size as fraction of sequence (0=tiny, 1=full)
      2  abstraction_bias    — raw features vs transformed (0=raw, 1=abstract)
      3  inversion_bias      — rate of AIM inversion (0=never, 1=always)
      4  sampling_temperature— output diversity (low=focused, high=diverse)
    """
    DIM = 5

    def __init__(self, attention_bias=0.5, granularity_bias=0.5,
                 abstraction_bias=0.5, inversion_bias=0.3, sampling_temperature=1.0):
        self.attention_bias       = float(np.clip(attention_bias,       0.0, 1.0))
        self.granularity_bias     = float(np.clip(granularity_bias,     0.0, 1.0))
        self.abstraction_bias     = float(np.clip(abstraction_bias,     0.0, 1.0))
        self.inversion_bias       = float(np.clip(inversion_bias,       0.0, 1.0))
        self.sampling_temperature = float(max(sampling_temperature, 1e-6))

    def to_array(self) -> np.ndarray:
        return np.array([
            self.attention_bias, self.granularity_bias,
            self.abstraction_bias, self.inversion_bias,
            self.sampling_temperature,
        ], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "BiasVector":
        return cls(*arr.tolist())

    @classmethod
    def from_dot_type(cls, dot_type: DotType,
                      rng: Optional[np.random.RandomState] = None) -> "BiasVector":
        """Initialise from the preset for a given DotType, with small noise."""
        preset = _TYPE_BIAS_PRESETS[dot_type]
        if rng is not None:
            noise = rng.randn(len(preset)).astype(np.float32) * 0.05
            arr   = np.clip(np.array(preset, np.float32) + noise, 0.01, 1.99)
            arr[-1] = max(arr[-1], 0.05)
            return cls.from_array(arr)
        return cls(*preset)

    @classmethod
    def random(cls, rng: Optional[np.random.RandomState] = None) -> "BiasVector":
        rng = rng or np.random.RandomState()
        return cls(rng.uniform(0.1, 0.9), rng.uniform(0.0, 1.0),
                   rng.uniform(0.0, 1.0), rng.uniform(0.0, 0.6),
                   rng.uniform(0.3, 2.0))

    def update(self, winning: "BiasVector", lr: float = 0.1) -> "BiasVector":
        """Formula 8: shift toward winning bias pattern."""
        return BiasVector.from_array(
            bias_vector_update(self.to_array(), winning.to_array(), lr)
        )

    def effective_temperature(self, base_entropy: float) -> float:
        """Adaptive temperature: rises when exploration needed, falls when converging."""
        return self.sampling_temperature * (1.0 + 0.3 * base_entropy)

    def __repr__(self):
        return (f"BiasVector(attn={self.attention_bias:.2f}, gran={self.granularity_bias:.2f}, "
                f"abst={self.abstraction_bias:.2f}, inv={self.inversion_bias:.2f}, "
                f"temp={self.sampling_temperature:.2f})")


# ── Neural Dot ────────────────────────────────────────────────────────

class NeuralDot:
    """
    Stateless independent prediction unit.

    Given a BaseMap slice, this dot:
      1. Selects a view based on dot type and granularity bias
      2. Applies type-specific pooling/attention
      3. Generates N_HEADS diverse predictions via head-specific projections
      4. Optionally blends in memory-guided attention
      5. Returns a list of (prediction, confidence, info) tuples
    """

    def __init__(self, dot_id: Optional[int] = None, feature_dim: int = 256,
                 bias: Optional[BiasVector] = None,
                 dot_type: DotType = DotType.SEMANTIC,
                 n_heads: int = N_HEADS,
                 seed: Optional[int] = None):
        self.dot_id      = dot_id if dot_id is not None else get_next_dot_id()
        self.feature_dim = feature_dim
        self.bias        = bias or BiasVector.from_dot_type(dot_type)
        self.dot_type    = dot_type
        self.n_heads     = n_heads
        self.max_heads   = 8 # Potential for growth

        seed = seed if seed is not None else dot_id * 31 + 7
        rng  = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(feature_dim)

        # Main transformation matrix
        self.W = rng.randn(feature_dim, feature_dim).astype(np.float32) * scale

        # Per-head projection matrices (feature_dim → feature_dim)
        # We pre-allocate more heads than default to allow for dynamic growth
        self.head_projs = [
            rng.randn(feature_dim, feature_dim).astype(np.float32) * scale
            for _ in range(self.max_heads)
        ]

        # Attention query basis (for computing the query from pooled input)
        self.Q_basis = rng.randn(feature_dim, feature_dim).astype(np.float32) * scale

        # Offset (learnable bias in linear transforms)
        self.b_offset = rng.randn(feature_dim).astype(np.float32) * scale * 0.1

        self._rng = np.random.RandomState(seed + 1)

    # ── Slice selection ───────────────────────────────────────────────

    def _select_slice(self, matrix: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Select a contiguous slice of the token sequence."""
        n, d = matrix.shape
        gran = self.bias.granularity_bias

        if self.dot_type == DotType.GLOBAL:
            return matrix, 0, n  # full sequence always

        if self.dot_type == DotType.CONTEXTUAL:
            patch_size = max(2, int(n * (0.5 + gran * 0.5)))
        elif self.dot_type == DotType.TEMPORAL:
            patch_size = max(1, int(n * (0.3 + gran * 0.5)))
        else:
            patch_size = max(1, int(n * (0.1 + gran * 0.7)))

        patch_size = min(patch_size, n)
        offset = int(self._rng.uniform(0, max(1, n - patch_size + 1)))
        start  = min(offset, n - 1)
        end    = min(start + patch_size, n)
        return matrix[start:end], start, end

    # ── Dim masking per type ──────────────────────────────────────────

    def _apply_dim_focus(self, v: np.ndarray) -> np.ndarray:
        """Zero out irrelevant dims based on dot type focus range."""
        lo, hi = _TYPE_DIM_RANGES[self.dot_type]
        if lo == 0 and hi == self.feature_dim:
            return v  # full range — no masking
        out = np.zeros_like(v)
        out[lo:hi] = v[lo:hi]
        return out

    # ── Pooling strategies ────────────────────────────────────────────

    def _pool(self, mat: np.ndarray, memory_hint: Optional[np.ndarray] = None) -> np.ndarray:
        """Pool the slice into a single vector; strategy depends on dot type."""
        if mat.shape[0] == 1:
            return mat[0]

        if self.dot_type == DotType.TEMPORAL:
            return self._temporal_pool(mat)
        if self.dot_type == DotType.RELATIONAL:
            return self._relational_pool(mat)
        if self.dot_type == DotType.LOGIC:
            return self._logic_pool(mat)
        if self.dot_type == DotType.MORPH:
            return self._morph_pool(mat)
        if self.dot_type == DotType.GLOBAL:
            return np.mean(mat, axis=0)

        # Standard attention pooling
        # Build query: Q_basis @ mean(mat), blended with memory hint
        base_query = np.tanh(self.Q_basis @ np.mean(mat, axis=0) + self.b_offset)
        if memory_hint is not None:
            blend = self.bias.attention_bias
            query = (1.0 - blend) * base_query + blend * memory_hint
        else:
            query = base_query

        scores = mat @ query * (self.bias.attention_bias * 8.0 + 1.0)
        scores -= np.max(scores)
        w = np.exp(scores); w /= w.sum() + 1e-10
        return (w[:, None] * mat).sum(axis=0)

    def _temporal_pool(self, mat: np.ndarray) -> np.ndarray:
        """Position-weighted pooling: recent tokens weighted higher."""
        n = mat.shape[0]
        weights = np.exp(np.linspace(-1.0, 0.0, n)).astype(np.float32)
        weights /= weights.sum()
        return (weights[:, None] * mat).sum(axis=0)

    def _relational_pool(self, mat: np.ndarray) -> np.ndarray:
        """
        Relational pooling: compute the difference between all token pairs,
        then average. Captures interaction patterns between tokens.
        """
        n = mat.shape[0]
        if n == 1: return mat[0]
        diffs = []
        for i in range(n):
            for j in range(i+1, n):
                diff = mat[i] - mat[j]
                diffs.append(diff)
        return np.mean(np.stack(diffs), axis=0)

    def _logic_pool(self, mat: np.ndarray) -> np.ndarray:
        """
        Logic pooling: focus on structural transitions and conditional patterns.
        Computes second-order differences (gradients of differences) to
        detect shifts in logical flow.
        """
        n = mat.shape[0]
        if n < 3: return np.mean(mat, axis=0)

        diffs = np.diff(mat, axis=0)
        accel = np.diff(diffs, axis=0)

        # Combine mean state with structural acceleration
        return 0.7 * np.mean(mat, axis=0) + 0.3 * np.mean(accel, axis=0)

    def _morph_pool(self, mat: np.ndarray) -> np.ndarray:
        """
        Morphological pooling: focuses on the variety and distribution of
        structural flags. Uses a variance-weighted pooling to highlight
        morphologically distinct tokens.
        """
        n = mat.shape[0]
        if n == 1: return mat[0]

        # Focus on flag dims [236:252] (indices 236 to 251)
        flags = mat[:, 236:252]
        # Weight tokens by how distinct their flags are (deviation from mean flags)
        mean_flags = np.mean(flags, axis=0)
        weights = np.linalg.norm(flags - mean_flags, axis=1)
        weights /= weights.sum() + 1e-10

        return (weights[:, None] * mat).sum(axis=0)

    # ── Abstraction transform ─────────────────────────────────────────

    def _abstract(self, v: np.ndarray) -> np.ndarray:
        """Linear + nonlinear mix controlled by abstraction_bias."""
        a = self.bias.abstraction_bias
        linear    = v
        nonlinear = np.tanh(self.W @ v + self.b_offset)
        return (1.0 - a) * linear + a * nonlinear

    # ── Per-head diversity ────────────────────────────────────────────

    def _project_head(self, v: np.ndarray, head: int, temperature: float) -> np.ndarray:
        """Project through head-specific matrix and add temperature noise."""
        H  = self.head_projs[head]
        p  = np.tanh(H @ v + self.b_offset * (head + 1) * 0.1)
        noise = self._rng.randn(len(p)).astype(np.float32) * temperature * 0.05
        return p + noise

    def _reason(self, v: np.ndarray, temperature: float) -> np.ndarray:
        """
        Internal Reasoning (Inner Monologue):
        Perform a micro-iteration internally to refine the vector.
        """
        refined = v
        # Perform 2 micro-steps of iterative refinement
        for _ in range(2):
            # Transform and gate
            delta = np.tanh(self.W @ refined + self.b_offset)
            # Add small noise per step based on temperature
            noise = self._rng.randn(len(v)).astype(np.float32) * temperature * 0.02
            refined = 0.8 * refined + 0.2 * delta + noise
        return refined

    # ── Main predict ──────────────────────────────────────────────────

    def predict(self, basemap, memory_hint: Optional[np.ndarray] = None,
                context_entropy: float = 0.5) -> List[Tuple[np.ndarray, float, Dict]]:
        """
        Generate N_HEADS candidate predictions from one basemap.

        Args:
          basemap       — BaseMap object from basemapping
          memory_hint   — optional recent centroid from DotMemory for attention guidance
          context_entropy — entropy from previous round (modulates temperature)

        Returns:
          list of (prediction, confidence, info) — one per head
        """
        sl, start, end = self._select_slice(basemap.matrix)
        pooled   = self._pool(sl, memory_hint)
        focused  = self._apply_dim_focus(pooled)
        abstract = self._abstract(focused)

        T = self.bias.effective_temperature(context_entropy)

        # Apply internal reasoning pass if abstraction bias is high enough
        if self.bias.abstraction_bias > 0.4:
            abstract = self._reason(abstract, T)

        results = []
        for h in range(self.n_heads):
            raw  = self._project_head(abstract, h, T)
            norm = np.linalg.norm(raw)
            pred = raw / norm * np.sqrt(self.feature_dim) if norm > 1e-10 else raw
            conf = prediction_confidence(pred)
            info = {
                "dot_id":    self.dot_id,
                "dot_type":  _TYPE_NAMES[self.dot_type],
                "head":      h,
                "slice":     (start, end),
                "bias":      self.bias,
                "source":    "original",
                "inversion_type": None,
            }
            results.append((pred, conf, info))
        return results

    def __repr__(self):
        return (f"NeuralDot(id={self.dot_id}, type={_TYPE_NAMES[self.dot_type]}, "
                f"heads={self.n_heads}, {self.bias})")


# ── Dot Generator ─────────────────────────────────────────────────────

class DotGenerator:
    """
    Creates a diverse pool of neural dots with varied types and biases.

    Distribution of types is controlled by `type_weights`. Default
    distribution favours semantic and contextual dots.
    """

    DEFAULT_TYPE_WEIGHTS = {
        DotType.SEMANTIC:   0.20,
        DotType.STRUCTURAL: 0.10,
        DotType.CONTEXTUAL: 0.10,
        DotType.RELATIONAL: 0.10,
        DotType.TEMPORAL:   0.10,
        DotType.GLOBAL:     0.10,
        DotType.LOGIC:      0.15,
        DotType.MORPH:      0.15,
    }

    def __init__(self, num_dots: int = 128, feature_dim: int = 256,
                 base_bias: Optional[BiasVector] = None,
                 type_weights: Optional[Dict[DotType, float]] = None,
                 n_heads: int = N_HEADS,
                 seed: int = 42):
        self.num_dots    = num_dots
        self.feature_dim = feature_dim
        self.base_bias   = base_bias or BiasVector()
        self.type_weights = type_weights or self.DEFAULT_TYPE_WEIGHTS
        self.n_heads     = n_heads
        self.seed        = seed
        self._rng        = np.random.RandomState(seed)

    def generate(self) -> List[NeuralDot]:
        """Create a fresh diverse dot pool."""
        types   = list(self.type_weights.keys())
        weights = np.array([self.type_weights[t] for t in types], dtype=np.float64)
        weights /= weights.sum()

        dots = []
        for i in range(self.num_dots):
            dot_type = DotType(self._rng.choice(types, p=weights))
            bias     = BiasVector.from_dot_type(dot_type, self._rng)
            dots.append(NeuralDot(
                dot_id=None, # will get unique ID
                feature_dim=self.feature_dim,
                bias=bias, dot_type=dot_type,
                n_heads=self.n_heads,
                seed=self.seed + i * 31 + int(dot_type) * 7,
            ))
        return dots

    def run_all(self, basemap, dots: List[NeuralDot],
                memory_hints: Optional[Dict[int, np.ndarray]] = None,
                context_entropy: float = 0.5) -> List[Tuple]:
        """Run all dots on the basemap; returns flat list of (pred, conf, info)."""
        all_results = []
        hints = memory_hints or {}
        for dot in dots:
            hint = hints.get(dot.dot_id)
            preds = dot.predict(basemap, memory_hint=hint, context_entropy=context_entropy)
            all_results.extend(preds)
        return all_results

    def type_distribution(self, dots: List[NeuralDot]) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for dot in dots:
            name = _TYPE_NAMES[dot.dot_type]
            dist[name] = dist.get(name, 0) + 1
        return dist
