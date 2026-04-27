"""
Neural Dot — the fundamental prediction unit of IECNN.
"""

import numpy as np
import ctypes
from enum import IntEnum
from typing import List, Optional, Tuple, Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import prediction_confidence, bias_vector_update, _fp

# ── Load C shared library ────────────────────────────────────────────
_lib = None

def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, "neural_dot_c.so")
    if os.path.exists(so_path):
        try:
            _lib = ctypes.CDLL(so_path)
            _lib.temporal_pool.restype = None
            _lib.relational_pool.restype = None
            _lib.logic_pool.restype = None
            _lib.project_head.restype = None
            _lib.apply_synergy_fast.restype = None
        except Exception:
            _lib = None
    return _lib

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

_TYPE_BIAS_PRESETS = {
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

def get_next_dot_id():
    global _NEXT_DOT_ID
    res = _NEXT_DOT_ID
    _NEXT_DOT_ID += 1
    return res

class BiasVector:
    def __init__(self, attention_bias=0.5, granularity_bias=0.5,
                 abstraction_bias=0.5, inversion_bias=0.3, sampling_temperature=1.0):
        self.attention_bias = float(np.clip(attention_bias, 0.0, 1.0))
        self.granularity_bias = float(np.clip(granularity_bias, 0.0, 1.0))
        self.abstraction_bias = float(np.clip(abstraction_bias, 0.0, 1.0))
        self.inversion_bias = float(np.clip(inversion_bias, 0.0, 1.0))
        self.sampling_temperature = float(max(sampling_temperature, 1e-6))

    def to_array(self):
        return np.array([self.attention_bias, self.granularity_bias, self.abstraction_bias,
                         self.inversion_bias, self.sampling_temperature], dtype=np.float32)

    @classmethod
    def from_array(cls, arr):
        return cls(*arr.tolist())

    @classmethod
    def from_dot_type(cls, dot_type, rng=None):
        preset = _TYPE_BIAS_PRESETS[dot_type]
        if rng:
            noise = rng.randn(len(preset)).astype(np.float32) * 0.05
            arr = np.clip(np.array(preset, np.float32) + noise, 0.01, 1.99)
            return cls.from_array(arr)
        return cls(*preset)

    @classmethod
    def random(cls, rng=None):
        rng = rng or np.random.RandomState()
        return cls(rng.uniform(0.1, 0.9), rng.uniform(0.0, 1.0),
                   rng.uniform(0.0, 1.0), rng.uniform(0.0, 0.6),
                   rng.uniform(0.3, 2.0))

    def effective_temperature(self, base_entropy):
        return self.sampling_temperature * (1.0 + 0.3 * base_entropy)

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
                 seed: Optional[int] = None,
                 birth_generation: int = 0,
                 empty: bool = False):
        self.dot_id      = dot_id if dot_id is not None else get_next_dot_id()
        self.feature_dim = feature_dim
        self.bias        = bias or BiasVector.from_dot_type(dot_type)
        self.dot_type    = dot_type
        self.n_heads     = n_heads

        # Phase-Coded state
        self.current_phase: float = 0.0
        self.max_heads   = 8 # Potential for growth
        self.birth_generation = int(birth_generation)

        if empty:
            self.W = None
            self.head_projs = []
            self.W_inv = None
            self.Q_basis = None
            self.b_offset = None
            self._rng = None
            return

        seed = seed if seed is not None else self.dot_id * 31 + 7
        rng  = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(feature_dim)

        # Optimize: Allocate one large buffer and slice it to reduce allocations
        total_mats = 1 + self.max_heads + 1 + 1 # W, head_projs, W_inv, Q_basis
        buffer = (rng.randn(total_mats * feature_dim, feature_dim) * scale).astype(np.float32)

        self.W = buffer[0:feature_dim].copy()
        self.head_projs = [buffer[(i+1)*feature_dim:(i+2)*feature_dim].copy() for i in range(self.max_heads)]
        self.W_inv = buffer[(self.max_heads+1)*feature_dim:(self.max_heads+2)*feature_dim].copy()
        self.Q_basis = buffer[(self.max_heads+2)*feature_dim:(self.max_heads+3)*feature_dim].copy()

        self.b_offset = (rng.randn(feature_dim) * scale * 0.1).astype(np.float32)
        self._rng = np.random.RandomState(seed + 1)

    def clone(self, new_id: bool = True):
        """Create a deep copy of this dot."""
        child = NeuralDot(
            dot_id=None if new_id else self.dot_id,
            feature_dim=self.feature_dim,
            bias=BiasVector.from_array(self.bias.to_array()),
            dot_type=self.dot_type,
            n_heads=self.n_heads,
            birth_generation=self.birth_generation,
            empty=True
        )
        child.W = self.W.copy()
        child.head_projs = [h.copy() for h in self.head_projs]
        child.W_inv = self.W_inv.copy()
        child.Q_basis = self.Q_basis.copy()
        child.b_offset = self.b_offset.copy()
        child.max_heads = self.max_heads
        child.current_phase = self.current_phase
        # Use a new RNG state for the child based on its new ID
        seed = child.dot_id * 31 + 8
        child._rng = np.random.RandomState(seed)
        return child

    # ── Pickle: store large weight matrices as float16 to halve disk size.
    # Runtime keeps float32 (no math change); only the on-disk representation
    # is compressed. Backward-compatible with pre-compression pickles.
    _F16_KEYS = ("W", "Q_basis", "b_offset", "W_inv")

    @staticmethod
    def _to_f16(arr):
        if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float16)
        return arr

    @staticmethod
    def _to_f32(arr):
        if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating) and arr.dtype != np.float32:
            return arr.astype(np.float32)
        return arr

    def __getstate__(self):
        state = self.__dict__.copy()
        for k in self._F16_KEYS:
            if k in state:
                state[k] = self._to_f16(state[k])
        if "head_projs" in state and isinstance(state["head_projs"], list):
            state["head_projs"] = [self._to_f16(h) for h in state["head_projs"]]
        state["_f16_compressed"] = True
        return state

    def __setstate__(self, state):
        state.pop("_f16_compressed", None)
        # Backfill defaults for fields added after some pickles were written.
        state.setdefault("birth_generation", 0)
        state.setdefault("max_heads", 8)
        state.setdefault("current_phase", 0.0)
        self.__dict__.update(state)
        # Always normalize to float32 at runtime regardless of how it was saved
        # (legacy pickles stored float64; new pickles store float16).
        for k in self._F16_KEYS:
            if k in self.__dict__:
                self.__dict__[k] = self._to_f32(self.__dict__[k])
        hp = self.__dict__.get("head_projs")
        if isinstance(hp, list):
            self.__dict__["head_projs"] = [self._to_f32(h) for h in hp]
        if "_rng" not in self.__dict__:
            self.__dict__["_rng"] = np.random.RandomState(int(self.dot_id) * 31 + 8)

    def predict(self, basemap, memory_hint=None, context_entropy=0.5, consensus=None):
        lib = _load_lib()
        sl, start, end = self._select_slice(basemap.matrix)

        if self.dot_type == DotType.TEMPORAL: pooled = self._temporal_pool(sl)
        elif self.dot_type == DotType.RELATIONAL: pooled = self._relational_pool(sl)
        elif self.dot_type == DotType.LOGIC: pooled = self._logic_pool(sl)
        else: pooled = np.mean(sl, axis=0)

        focused = self._apply_dim_focus(pooled)
        abstract = np.tanh(self.W @ focused + self.b_offset)
        T = self.bias.effective_temperature(context_entropy)

        results = []
        for h in range(self.n_heads):
            noise = self._rng.randn(self.feature_dim).astype(np.float32)
            if lib:
                out = np.zeros(self.feature_dim, dtype=np.float32)
                lib.project_head(_fp(abstract)[0], _fp(self.head_projs[h])[0], _fp(self.b_offset)[0],
                                 ctypes.c_int(self.feature_dim), ctypes.c_float(T), _fp(noise)[0], _fp(out)[0])
                pred = out
            else:
                pred = np.tanh(self.head_projs[h] @ abstract + self.b_offset) + noise * T * 0.05

            norm = np.linalg.norm(pred)
            if norm > 1e-10: pred = pred / norm * np.sqrt(self.feature_dim)
            results.append((pred, prediction_confidence(pred), {"dot_id": self.dot_id, "dot_type": _TYPE_NAMES[self.dot_type], "source": "original"}))
        return results

    def _select_slice(self, matrix: np.ndarray):
        """
        Select a contiguous slice of the basemap matrix.

        The slice length is controlled by granularity_bias:
          - high granularity → short focused slice (fine detail)
          - low  granularity → long slice (broad context)

        Returns (slice, start_idx, end_idx).
        """
        n = matrix.shape[0]
        patch_size = max(1, int(self.bias.granularity_bias * n))
        patch_size = min(patch_size, n)
        if n <= 1:
            return matrix, 0, n
        offset = int(self._rng.uniform(0, n - patch_size + 1))
        start  = max(0, min(offset, n - 1))
        end    = max(start + 1, min(start + patch_size, n))
        return matrix[start:end], start, end

    def _apply_dim_focus(self, v):
        lo, hi = _TYPE_DIM_RANGES[self.dot_type]
        out = np.zeros_like(v)
        out[lo:hi] = v[lo:hi]
        return out

    # ── Pooling strategies ────────────────────────────────────────────

    def _pool(self, mat: np.ndarray, memory_hint: Optional[np.ndarray] = None) -> np.ndarray:
        """Pool the slice into a single vector; strategy depends on dot type."""
        if mat.shape[0] == 1:
            return mat[0]

        # Multi-modal context awareness:
        # Check if slice contains mixed modalities via flags [248:252] (236+12:16)
        modalities = mat[:, 248:252]
        is_mixed = np.any(np.sum(modalities, axis=0) > 0)

        if is_mixed and self.dot_type in (DotType.RELATIONAL, DotType.LOGIC):
            # For mixed modalities, Relational/Logic dots focus on cross-modal gaps
            return self._cross_modal_pool(mat)

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
        Optimized to O(n) from O(n^2).
        """
        n = mat.shape[0]
        if n <= 1: return mat[0] if n == 1 else np.zeros(self.feature_dim, dtype=np.float32)

        # O(n) implementation: sum_{i < j} (x_i - x_j) = sum_{k=0}^{n-1} (n - 1 - 2k) * x_k
        k = np.arange(n)
        coeffs = (n - 1 - 2 * k).astype(np.float32)
        pooled = (coeffs[:, None] * mat).sum(axis=0)

        count = n * (n - 1) / 2
        return (pooled / (count + 1e-10)).astype(np.float32)

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

    def _cross_modal_pool(self, mat: np.ndarray) -> np.ndarray:
        """
        Cross-modal pooling: Focuses on interaction between different modalities.
        Identifies the boundary between modality groups and highlights transitions.
        """
        n = mat.shape[0]
        mod_flags = mat[:, 248:252]

        # Compute pairwise distance between tokens of different modalities
        # We simplify by using a modality-based weight
        weights = np.zeros(n, dtype=np.float32)
        for i in range(n):
            # High weight if neighbor has different modality
            if i > 0 and not np.array_equal(mod_flags[i], mod_flags[i-1]):
                weights[i] += 1.0
            if i < n-1 and not np.array_equal(mod_flags[i], mod_flags[i+1]):
                weights[i] += 1.0

        if weights.sum() < 1e-10:
            # If no transitions, we boost everything to encourage discovery
            weights = np.ones(n, dtype=np.float32)

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

    def local_update(self, winning_centroid: np.ndarray, winning_head: int, lr: float = 0.01):
        """
        Hebbian Local Plasticity (v4 SOTA):
        Nudge dot weights toward the winning consensus immediately after a win.
        This allows dots to 'absorb' patterns they correctly identified.
        """
        # Ensure we work with real part for weight updates
        target = np.real(winning_centroid).astype(np.float32)
        n = np.linalg.norm(target)
        if n > 1e-10: target /= n

        # 1. Update main transform W
        # We nudge a random row toward the target to maintain diversity
        row_idx = self._rng.randint(0, self.feature_dim)
        self.W[row_idx] = (1.0 - lr) * self.W[row_idx] + lr * target

        # 2. Update winning head projection
        H = self.head_projs[winning_head]
        # Soft nudge of the entire projection matrix
        # (Rank-1 update approximation)
        H_target = np.outer(target, target)
        self.head_projs[winning_head] = (1.0 - lr * 0.5) * H + (lr * 0.5) * H_target

        # 3. Normalize weights to prevent explosion
        norm_w = np.linalg.norm(self.W, axis=1, keepdims=True)
        self.W /= (norm_w + 1e-10)

    def _reason(self, v: np.ndarray, temperature: float,
                consensus: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Internal Reasoning (Inner Monologue):
        Perform a micro-iteration internally to refine the vector.
        Incorporates 'Cognitive Peer Pressure' if a consensus hint is available.
        """
        refined = v
        # Perform 2 micro-steps of iterative refinement
        for _ in range(2):
            # Transform and gate
            delta = np.tanh(self.W @ refined + self.b_offset)

            # Cognitive Peer Pressure: nudge toward hazy global summary
            if consensus is not None:
                # The dot self-corrects based on its attention bias
                # (high attention dots resist peer pressure more)
                pressure = 0.15 * (1.0 - self.bias.attention_bias)
                refined = (1.0 - pressure) * refined + pressure * consensus

            # Add small noise per step based on temperature
            noise = self._rng.randn(len(v)).astype(np.float32) * temperature * 0.02
            refined = 0.8 * refined + 0.2 * delta + noise
        return refined

    # ── Main predict ──────────────────────────────────────────────────

    def predict(self, basemap, memory_hint: Optional[np.ndarray] = None,
                context_entropy: float = 0.5,
                consensus: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, float, Dict]]:
        # Fast path: use already-pooled result if available (for batch prediction)
        if hasattr(self, "_current_pooled"):
            pooled, start, end = self._current_pooled
            del self._current_pooled
        else:
            pooled, start, end = self._get_pooled(basemap, memory_hint)

        n_tokens = max(int(basemap.matrix.shape[0]), 1)
        slice_phase = float(2.0 * np.pi * ((start + end) * 0.5) / n_tokens)

        # Determine modality
        modality = "text" # default
        if (end - start) > 0:
            sl = basemap.matrix[start:end]
            mod_flags = sl[:, 248:252]
            mod_counts = np.sum(mod_flags, axis=0)
            dom_mod_idx = np.argmax(mod_counts) if np.max(mod_counts) > 0 else 0
            modality = ["text", "image", "audio", "video"][dom_mod_idx]

        focused  = self._apply_dim_focus(pooled)
        abstract = self._abstract(focused)

        T = self.bias.effective_temperature(context_entropy)

        if self.bias.abstraction_bias > 0.4:
            abstract = self._reason(abstract, T, consensus=consensus)

        requested_inversion = None
        if self.bias.abstraction_bias > 0.6 and T > 1.2:
            requested_inversion = "relational"
        elif modality == "image" and self.bias.granularity_bias < 0.3:
            requested_inversion = "scale"
        if self.bias.attention_bias > 0.7:
            requested_inversion = "evolved"

        results = []
        p_phase = self.current_phase if self.current_phase != 0.0 else slice_phase
        phase_factor = np.exp(1j * p_phase)

        for h in range(self.n_heads):
            raw  = self._project_head(abstract, h, T)
            norm = np.linalg.norm(raw)
            pred = raw / norm * np.sqrt(self.feature_dim) if norm > 1e-10 else raw

            complex_pred = pred.astype(np.complex64) * phase_factor
            conf = prediction_confidence(complex_pred)

            results.append((complex_pred, conf, {
                "dot_id":    self.dot_id,
                "dot_type":  _TYPE_NAMES[self.dot_type],
                "head":      h,
                "slice":     (start, end),
                "phase":     p_phase,
                "bias":      self.bias,
                "source":    "original",
                "inversion_type": None,
                "modality":  modality,
                "requested_inversion": requested_inversion,
                "W_inv":     self.W_inv,
            }))
        return results

    def _get_pooled(self, basemap, memory_hint=None):
        sl, start, end = self._select_slice(basemap.matrix)
        pooled = self._pool(sl, memory_hint)
        return pooled, start, end

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
                context_entropy: float = 0.5,
                consensus: Optional[np.ndarray] = None) -> List[Tuple]:
        """
        Run all dots on the basemap.
        Optimized via semi-vectorization.
        """
        hints = memory_hints or {}

        # 1. Sequential pooling (strategy-dependent)
        for dot in dots:
            hint = hints.get(dot.dot_id)
            dot._current_pooled = dot._get_pooled(basemap, hint)

        # 2. Sequential prediction (currently)
        # TODO: Vectorize the transform/reasoning stage across dots
        all_results = []
        for dot in dots:
            preds = dot.predict(basemap, context_entropy=context_entropy, consensus=consensus)
            all_results.extend(preds)
        return all_results

    def type_distribution(self, dots: List[NeuralDot]) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for dot in dots:
            name = _TYPE_NAMES[dot.dot_type]
            dist[name] = dist.get(name, 0) + 1
        return dist
