"""
Neural Dot — the fundamental prediction unit of IECNN.
"""

import numpy as np
import ctypes
from enum import IntEnum
from typing import List, Optional, Tuple, Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import (
    prediction_confidence, bias_vector_update, _fp, similarity_score
)

# ── Load C shared library ────────────────────────────────────────────
_lib = None
_lib_check_done = False   # sentinel: True once we have attempted loading

def _set_restype(lib, name, restype):
    """Safely assign restype — silently skips symbols absent from a partial build."""
    try:
        getattr(lib, name).restype = restype
    except AttributeError:
        pass

def _load_lib():
    global _lib, _lib_check_done
    if _lib_check_done:          # return cached result (None or lib object)
        return _lib
    _lib_check_done = True
    here = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, "neural_dot_c.so")
    if os.path.exists(so_path):
        try:
            _lib = ctypes.CDLL(so_path)
            # Set restypes only for symbols present in this build.
            # reason_fast and predict_batch_c may be absent in partial builds;
            # _set_restype silently skips missing symbols rather than crashing.
            for sym in ("temporal_pool", "relational_pool", "logic_pool",
                        "project_head", "apply_synergy_fast",
                        "reason_fast", "predict_batch_c"):
                _set_restype(_lib, sym, None)
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
    ACTION     = 8  # task-oriented intent and tool usage
    LAZY       = 9  # simple, low-compute dot for broad patterns


_TYPE_NAMES = {
    DotType.SEMANTIC:   "semantic",
    DotType.STRUCTURAL: "structural",
    DotType.CONTEXTUAL: "contextual",
    DotType.RELATIONAL: "relational",
    DotType.TEMPORAL:   "temporal",
    DotType.GLOBAL:     "global",
    DotType.LOGIC:      "logic",
    DotType.MORPH:      "morph",
    DotType.ACTION:     "action",
    DotType.LAZY:       "lazy",
}

_TYPE_DIM_RANGES = {
    DotType.SEMANTIC:   (0,   128),
    DotType.STRUCTURAL: (128, 256),
    DotType.CONTEXTUAL: (0,   256),
    DotType.RELATIONAL: (0,   256),
    DotType.TEMPORAL:   (0,   256),
    DotType.GLOBAL:     (0,   256),
    DotType.LOGIC:      (128, 256),
    DotType.MORPH:      (236, 252),
    DotType.ACTION:     (128, 252), # focus on structure + flags for intent
    DotType.LAZY:       (0,   256),
}

_TYPE_BIAS_PRESETS = {
    DotType.SEMANTIC:   (0.7, 0.3, 0.5, 0.3, 0.8, 0.2),
    DotType.STRUCTURAL: (0.6, 0.5, 0.3, 0.2, 0.6, 0.2),
    DotType.CONTEXTUAL: (0.4, 0.8, 0.8, 0.4, 1.2, 0.3),
    DotType.RELATIONAL: (0.8, 0.4, 0.6, 0.5, 1.0, 0.3),
    DotType.TEMPORAL:   (0.5, 0.6, 0.4, 0.3, 0.9, 0.2),
    DotType.GLOBAL:     (0.3, 0.9, 0.7, 0.2, 0.7, 0.1),
    DotType.LOGIC:      (0.9, 0.6, 0.7, 0.1, 0.5, 0.5),
    DotType.MORPH:      (0.8, 0.2, 0.4, 0.1, 0.4, 0.2),
    DotType.ACTION:     (0.9, 0.7, 0.6, 0.2, 0.4, 0.6), # high reasoning for actions
    DotType.LAZY:       (0.1, 1.0, 0.1, 0.05, 1.5, 0.05), # high granularity, low reasoning
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

class BiasVector:
    def __init__(self, attention_bias=0.5, granularity_bias=0.5,
                 abstraction_bias=0.5, inversion_bias=0.3, sampling_temperature=1.0,
                 reasoning_depth=0.2):
        self.attention_bias = float(np.clip(attention_bias, 0.0, 1.0))
        self.granularity_bias = float(np.clip(granularity_bias, 0.0, 1.0))
        self.abstraction_bias = float(np.clip(abstraction_bias, 0.0, 1.0))
        self.inversion_bias = float(np.clip(inversion_bias, 0.0, 1.0))
        self.sampling_temperature = float(max(sampling_temperature, 1e-6))
        self.reasoning_depth = float(np.clip(reasoning_depth, 0.0, 1.0))

    def __getattr__(self, name):
        if name == "reasoning_depth":
            self.reasoning_depth = 0.2
            return 0.2
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def to_array(self):
        rd = getattr(self, 'reasoning_depth', 0.2)
        return np.array([self.attention_bias, self.granularity_bias, self.abstraction_bias,
                         self.inversion_bias, self.sampling_temperature, rd], dtype=np.float32)

    @classmethod
    def from_array(cls, arr):
        return cls(*arr.tolist())

    @classmethod
    def from_dot_type(cls, dot_type, rng=None):
        preset = list(_TYPE_BIAS_PRESETS[dot_type])
        if len(preset) < 6:
            preset.append(0.2)
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
        self.current_phase: float = 0.0
        self.max_heads   = 8
        self.birth_generation = int(birth_generation)

        if empty:
            self.W = None; self.head_projs = []; self.W_inv = None
            self.Q_basis = None; self.b_offset = None; self._rng = None
            return

        seed = seed if seed is not None else self.dot_id * 31 + 7
        rng  = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(feature_dim)
        total_mats = 1 + self.max_heads + 1 + 1
        buffer = (rng.randn(total_mats * feature_dim, feature_dim) * scale).astype(np.float32)
        self.W = buffer[0:feature_dim].copy()
        self.head_projs = [buffer[(i+1)*feature_dim:(i+2)*feature_dim].copy() for i in range(self.max_heads)]
        self.W_inv = buffer[(self.max_heads+1)*feature_dim:(self.max_heads+2)*feature_dim].copy()
        self.Q_basis = buffer[(self.max_heads+2)*feature_dim:(self.max_heads+3)*feature_dim].copy()
        self.b_offset = (rng.randn(feature_dim) * scale * 0.1).astype(np.float32)
        self._rng = np.random.RandomState(seed + 1)

    def clone(self, new_id: bool = True):
        child = NeuralDot(None if new_id else self.dot_id, self.feature_dim, BiasVector.from_array(self.bias.to_array()), self.dot_type, self.n_heads, None, self.birth_generation, True)
        child.W = self.W.copy(); child.head_projs = [h.copy() for h in self.head_projs]
        child.W_inv = self.W_inv.copy(); child.Q_basis = self.Q_basis.copy()
        child.b_offset = self.b_offset.copy(); child.max_heads = self.max_heads
        child.current_phase = self.current_phase; child._rng = np.random.RandomState(child.dot_id * 31 + 8)
        return child

    _F16_KEYS = ("W", "Q_basis", "b_offset", "W_inv")
    @staticmethod
    def _to_f16(arr): return arr.astype(np.float16) if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating) else arr
    @staticmethod
    def _to_f32(arr): return arr.astype(np.float32) if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating) and arr.dtype != np.float32 else arr

    def __getstate__(self):
        state = self.__dict__.copy()
        for k in self._F16_KEYS:
            if k in state: state[k] = self._to_f16(state[k])
        if "head_projs" in state: state["head_projs"] = [self._to_f16(h) for h in state["head_projs"]]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for k in self._F16_KEYS:
            if k in self.__dict__: self.__dict__[k] = self._to_f32(self.__dict__[k])
        if "head_projs" in self.__dict__: self.__dict__["head_projs"] = [self._to_f32(h) for h in self.__dict__["head_projs"]]
        if "_rng" not in self.__dict__: self.__dict__["_rng"] = np.random.RandomState(int(self.dot_id) * 31 + 8)

    def predict(self, basemap, memory_hint: Optional[np.ndarray] = None, context_entropy: float = 0.5, consensus: Optional[np.ndarray] = None, causal: bool = False, abstract: Optional[np.ndarray] = None) -> List[Tuple]:
        lib = _load_lib()
        if lib and hasattr(lib, "project_head") and abstract is not None:
            n_tokens = max(int(basemap.matrix.shape[0]), 1)
            start, end = 0, n_tokens
            p_phase = self.current_phase if self.current_phase != 0.0 else float(2.0 * np.pi * ((start + end) * 0.5) / n_tokens)
            phase_factor = np.exp(1j * p_phase); results = []
            T = self.bias.effective_temperature(context_entropy); seed = int(self._rng.randint(0, 2**31)); c_seed = ctypes.c_uint(seed)
            for h in range(self.n_heads):
                out = np.zeros(self.feature_dim, dtype=np.float32); offset = self.b_offset * (h + 1) * 0.1
                lib.project_head(_fp(abstract)[0], _fp(self.head_projs[h])[0], _fp(offset)[0], ctypes.c_int(self.feature_dim), ctypes.c_float(T), ctypes.byref(c_seed), _fp(out)[0])
                norm = np.linalg.norm(out)
                if norm > 1e-10: out *= np.sqrt(self.feature_dim) / norm
                complex_pred = out.astype(np.complex64) * phase_factor
                results.append((complex_pred, prediction_confidence(complex_pred), {"dot_id": self.dot_id, "dot_type": _TYPE_NAMES[self.dot_type], "head": h, "slice": (start, end), "phase": p_phase, "bias": self.bias, "source": "original"}))
            return results
        if hasattr(self, "_current_pooled"):
            pooled, start, end = self._current_pooled; del self._current_pooled
        else:
            pooled, start, end = self._get_pooled(basemap, memory_hint, causal=causal)
        n_tokens = max(int(basemap.matrix.shape[0]), 1); slice_phase = float(2.0 * np.pi * ((start + end) * 0.5) / n_tokens)
        modality = "text"
        if (end - start) > 0:
            sl = basemap.matrix[start:end]; mod_flags = sl[:, 248:252]; mod_counts = np.sum(mod_flags, axis=0)
            dom_mod_idx = np.argmax(mod_counts) if np.max(mod_counts) > 0 else 0; modality = ["text", "image", "audio", "video"][dom_mod_idx]
        focused = self._apply_dim_focus(pooled)
        if abstract is None: abstract = self._abstract(focused)
        T = self.bias.effective_temperature(context_entropy)
        if self.bias.abstraction_bias > 0.4: abstract = self._reason(abstract, T, consensus=consensus)
        results = []; p_phase = self.current_phase if self.current_phase != 0.0 else slice_phase
        phase_factor = np.exp(1j * p_phase); target_idx = end if end < n_tokens else None
        for h in range(self.n_heads):
            raw = self._project_head(abstract, h, T)
            raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)
            # Use float64 for norm: large float32 values (~1e38) overflow
            # x.dot(x) in float32, even after nan_to_num.
            norm = float(np.linalg.norm(raw.astype(np.float64)))
            pred = raw / norm * np.sqrt(self.feature_dim) if norm > 1e-10 else raw
            complex_pred = pred.astype(np.complex64) * phase_factor
            results.append((complex_pred, prediction_confidence(complex_pred), {"dot_id": self.dot_id, "dot_type": _TYPE_NAMES[self.dot_type], "head": h, "slice": (start, end), "target_idx": target_idx, "phase": p_phase, "bias": self.bias, "source": "original", "inversion_type": None, "modality": modality, "W_inv": self.W_inv}))
        return results

    def _select_slice(self, matrix: np.ndarray, causal: bool = False):
        n = matrix.shape[0]; patch_size = min(max(1, int(self.bias.granularity_bias * n)), n)
        if n <= 1: return matrix, 0, n
        offset = int(self._rng.beta(2.0, 1.0) * (n - patch_size)) if causal else int(self._rng.uniform(0, n - patch_size + 1))
        start = max(0, min(offset, n - 1)); end = max(start + 1, min(start + patch_size, n))
        return matrix[start:end], start, end

    def _apply_dim_focus(self, v):
        lo, hi = _TYPE_DIM_RANGES[self.dot_type]; out = np.zeros_like(v); out[lo:hi] = v[lo:hi]; return out

    def _pool(self, mat: np.ndarray, memory_hint: Optional[np.ndarray] = None) -> np.ndarray:
        if mat.shape[0] == 1: return mat[0]
        if self.dot_type == DotType.TEMPORAL: return self._temporal_pool(mat)
        if self.dot_type == DotType.RELATIONAL: return self._relational_pool(mat)
        if self.dot_type == DotType.LOGIC: return self._logic_pool(mat)
        if self.dot_type == DotType.MORPH: return self._morph_pool(mat)
        if self.dot_type == DotType.GLOBAL: return np.mean(mat, axis=0)
        base_query = np.tanh(self.Q_basis @ np.mean(mat, axis=0) + self.b_offset)
        query = (1.0 - self.bias.attention_bias) * base_query + self.bias.attention_bias * (memory_hint if memory_hint is not None else base_query)
        scores = mat @ query * (self.bias.attention_bias * 8.0 + 1.0); scores -= np.max(scores)
        w = np.exp(scores); w /= w.sum() + 1e-10; return (w[:, None] * mat).sum(axis=0)

    def _temporal_pool(self, mat):
        lib = _load_lib(); n = mat.shape[0]
        if lib:
            out = np.zeros(self.feature_dim, dtype=np.float32)
            lib.temporal_pool(_fp(mat)[0], ctypes.c_int(n), ctypes.c_int(self.feature_dim), _fp(out)[0])
            return out
        w = np.exp(np.linspace(-1.0, 0.0, n)).astype(np.float32); w /= w.sum(); return (w[:, None] * mat).sum(axis=0)

    def _relational_pool(self, mat):
        lib = _load_lib(); n = mat.shape[0]
        if lib:
            out = np.zeros(self.feature_dim, dtype=np.float32)
            lib.relational_pool(_fp(mat)[0], ctypes.c_int(n), ctypes.c_int(self.feature_dim), _fp(out)[0])
            return out
        if n <= 1: return mat[0] if n == 1 else np.zeros(self.feature_dim, dtype=np.float32)
        k = np.arange(n); coeffs = (n - 1 - 2 * k).astype(np.float32); return ((coeffs[:, None] * mat).sum(axis=0) / (n*(n-1)/2 + 1e-10)).astype(np.float32)

    def _logic_pool(self, mat):
        lib = _load_lib(); n = mat.shape[0]
        if lib:
            out = np.zeros(self.feature_dim, dtype=np.float32)
            lib.logic_pool(_fp(mat)[0], ctypes.c_int(n), ctypes.c_int(self.feature_dim), _fp(out)[0])
            return out
        if n < 3: return np.mean(mat, axis=0)
        accel = np.diff(np.diff(mat, axis=0), axis=0); return 0.7 * np.mean(mat, axis=0) + 0.3 * np.mean(accel, axis=0)

    def _morph_pool(self, mat):
        if mat.shape[0] == 1: return mat[0]
        f = mat[:, 236:252]; w = np.linalg.norm(f - np.mean(f, axis=0), axis=1); w /= w.sum() + 1e-10; return (w[:, None] * mat).sum(axis=0)

    def _cross_modal_pool(self, mat):
        n = mat.shape[0]; mf = mat[:, 248:252]; w = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if i > 0 and not np.array_equal(mf[i], mf[i-1]): w[i] += 1.0
            if i < n-1 and not np.array_equal(mf[i], mf[i+1]): w[i] += 1.0
        if w.sum() < 1e-10: w = np.ones(n, dtype=np.float32)
        w /= w.sum() + 1e-10; return (w[:, None] * mat).sum(axis=0)

    def _abstract(self, v): return (1.0 - self.bias.abstraction_bias) * v + self.bias.abstraction_bias * np.tanh(self.W @ v + self.b_offset)

    def _project_head(self, v, head, temperature):
        lib = _load_lib(); H = self.head_projs[head]; seed = int(self._rng.randint(0, 2**31)); offset = self.b_offset * (head + 1) * 0.1
        if lib:
            out = np.zeros(self.feature_dim, dtype=np.float32); c_seed = ctypes.c_uint(seed)
            lib.project_head(_fp(v)[0], _fp(H)[0], _fp(offset)[0], ctypes.c_int(self.feature_dim), ctypes.c_float(temperature), ctypes.byref(c_seed), _fp(out)[0])
            return out
        p = np.tanh(H @ v + offset); return p + self._rng.randn(self.feature_dim).astype(np.float32) * temperature * 0.05

    def local_update(self, winning_centroid, winning_head, lr=0.01):
        target = np.real(winning_centroid).astype(np.float32); n = np.linalg.norm(target)
        if n > 1e-10: target /= n
        row = self._rng.randint(0, self.feature_dim)
        self.W[row] = (1.0 - lr) * self.W[row] + lr * target
        self.head_projs[winning_head] = (1.0 - lr * 0.5) * self.head_projs[winning_head] + (lr * 0.5) * np.outer(target, target)
        self.W /= (np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-10)

    def _reason(self, v, temperature, consensus=None):
        lib = _load_lib(); steps = max(1, int(self.bias.reasoning_depth * 5))
        if lib and hasattr(lib, 'reason_fast'):
            seed = int(self._rng.randint(0, 2**31)); fc, _c = _fp(consensus) if consensus is not None else (None, None); v_work = v.copy().astype(np.float32)
            c_seed = ctypes.c_uint(seed); lib.reason_fast(_fp(v_work)[0], _fp(self.W)[0], _fp(self.b_offset)[0], ctypes.c_int(self.feature_dim), ctypes.c_int(steps), ctypes.c_float(temperature), ctypes.c_float(self.bias.attention_bias), fc, ctypes.byref(c_seed), ctypes.c_float(self.bias.abstraction_bias))
            return v_work
        refined = v
        for _ in range(steps):
            delta = np.tanh(self.W @ refined + self.b_offset)
            if consensus is not None: refined = (1.0 - 0.15*(1-self.bias.attention_bias)) * refined + 0.15*(1-self.bias.attention_bias) * consensus
            gate = 0.2 + 0.3 * self.bias.abstraction_bias; refined = (1.0 - gate) * refined + gate * delta + self._rng.randn(len(v)).astype(np.float32) * temperature * 0.02
            n_val = np.linalg.norm(refined); refined = refined / n_val * np.sqrt(self.feature_dim) if n_val > 1e-10 else refined
        return refined

    def _get_pooled(self, basemap, memory_hint=None, causal=False):
        sl, start, end = self._select_slice(basemap.matrix, causal=causal); return self._pool(sl, memory_hint), start, end

    def __repr__(self): return f"NeuralDot(id={self.dot_id}, type={_TYPE_NAMES[self.dot_type]}, heads={self.n_heads}, {self.bias})"

class DotGenerator:
    DEFAULT_TYPE_WEIGHTS = {
        DotType.SEMANTIC:   0.15,
        DotType.STRUCTURAL: 0.10,
        DotType.CONTEXTUAL: 0.10,
        DotType.RELATIONAL: 0.10,
        DotType.TEMPORAL:   0.10,
        DotType.GLOBAL:     0.05,
        DotType.LOGIC:      0.15,
        DotType.MORPH:      0.10,
        DotType.ACTION:     0.10,
        DotType.LAZY:       0.05
    }
    def __init__(self, num_dots=128, feature_dim=256, base_bias=None, type_weights=None, n_heads=N_HEADS, seed=42):
        self.num_dots = num_dots; self.feature_dim = feature_dim; self.base_bias = base_bias or BiasVector(); self.type_weights = type_weights or self.DEFAULT_TYPE_WEIGHTS
        self.n_heads = n_heads; self.seed = seed; self._rng = np.random.RandomState(seed)
        self._cached_W_stack = self._cached_HP_stack = self._cached_B_stack = self._cached_TYPES = self._cached_N_HEADS = self._cached_SEEDS = self._cached_BIAS = self._cached_dots_id = None
        self._dot_version: int = 0
        self._cached_version: int = -1

    def bust_cache(self):
        """
        Increment the mutation version so _ensure_caches() rebuilds the C
        weight stacks on the next forward pass.  Call after any in-place
        weight mutation (e.g. local_update) that does NOT replace the list.
        """
        self._dot_version += 1

    def batch_local_update(self, dots: List['NeuralDot'], target: np.ndarray,
                           winning_head: int = 0, lr: float = 0.005) -> None:
        """
        Vectorised W-only weight update applied to all dots simultaneously.

        This is the fast training path used by causal_train_pass().  It is
        ~256x faster than looping over dots and calling local_update() on
        each one because:

          1. The expensive O(dim²) outer-product head_proj update is skipped
             during training (the projection matrices stay stable; only W
             needs to track the target space).
          2. All row updates are a single NumPy broadcast over the dot axis.
          3. Row-wise renormalisation costs O(n_dots × dim) not O(n_dots × dim²).
          4. The method leaves the cache in a *fresh* state (version counters
             are kept in sync) so no extra bust_cache() call is required.
        """
        target_f32 = np.real(target).astype(np.float32)
        tn = np.linalg.norm(target_f32)
        if tn > 1e-10:
            target_f32 = target_f32 / tn          # normalised copy

        self._ensure_caches(dots)
        n   = len(dots)
        idx = np.arange(n, dtype=np.int64)

        # One random row per dot (uses each dot's own RNG for reproducibility)
        rows = np.array([d._rng.randint(0, self.feature_dim) for d in dots],
                        dtype=np.int64)

        # ── Vectorised row update ──────────────────────────────────────
        # _cached_W_stack shape: (n_dots, feature_dim, feature_dim)
        self._cached_W_stack[idx, rows] = (
            (1.0 - lr) * self._cached_W_stack[idx, rows]
            + lr * target_f32[None, :]          # broadcast across dot axis
        )

        # ── Row-only renormalisation (O(n×dim), not O(n×dim²)) ─────────
        row_vecs = self._cached_W_stack[idx, rows]               # (n, dim)
        norms    = np.linalg.norm(row_vecs, axis=1, keepdims=True)  # (n, 1)
        self._cached_W_stack[idx, rows] = row_vecs / (norms + 1e-10)

        # ── Sync back to individual dot objects ────────────────────────
        for i, dot in enumerate(dots):
            dot.W = self._cached_W_stack[i].copy()

        # Mark cache as fresh (cache and dot.W are now in sync)
        self._dot_version  += 1
        self._cached_version = self._dot_version

    def generate(self) -> List[NeuralDot]:
        types = list(self.type_weights.keys()); weights = np.array([self.type_weights[t] for t in types]); weights /= weights.sum(); dots = []
        for i in range(self.num_dots):
            dot_type = DotType(self._rng.choice(types, p=weights)); bias = BiasVector.from_dot_type(dot_type, self._rng)
            dots.append(NeuralDot(None, self.feature_dim, bias, dot_type, self.n_heads, self.seed + i * 31))
        return dots

    def _ensure_caches(self, dots: List[NeuralDot]):
        stale = (
            self._cached_W_stack is None
            or self._cached_dots_id != id(dots)
            or self._cached_version != self._dot_version
        )
        if stale:
            n = len(dots); dim = self.feature_dim
            self._cached_W_stack = np.ascontiguousarray(np.stack([d.W for d in dots]), dtype=np.float32)
            self._cached_HP_stack = np.ascontiguousarray(np.stack([np.pad(np.stack(d.head_projs), ((0, 8-len(d.head_projs)),(0,0),(0,0))) for d in dots]), dtype=np.float32)
            self._cached_B_stack = np.ascontiguousarray(np.stack([d.b_offset for d in dots]), dtype=np.float32)
            self._cached_TYPES = np.ascontiguousarray([int(d.dot_type) for d in dots], dtype=np.int32)
            self._cached_N_HEADS = np.ascontiguousarray([int(d.n_heads) for d in dots], dtype=np.int32)
            self._cached_SEEDS = np.ascontiguousarray([int(d.dot_id * 31 + 7) for d in dots], dtype=np.uint32)
            self._cached_BIAS = np.ascontiguousarray(np.stack([d.bias.to_array() for d in dots]), dtype=np.float32)
            self._cached_dots_id = id(dots)
            self._cached_version = self._dot_version

    def run_all(self, basemap, dots, memory_hints=None, context_entropy=0.5, consensus=None, causal=False) -> List[Tuple]:
        lib = _load_lib()
        if lib and hasattr(lib, "predict_batch_c"):
            self._ensure_caches(dots)
            n = len(dots); dim = self.feature_dim
            BM_mat = np.ascontiguousarray(np.real(basemap.matrix), dtype=np.float32); CONS = np.ascontiguousarray(np.real(consensus), dtype=np.float32) if consensus is not None else np.zeros(dim, dtype=np.float32)
            OUT_PREDS = np.zeros((n * 8, dim), dtype=np.float32); OUT_CONFS = np.zeros(n * 8, dtype=np.float32); OUT_STARTS = np.zeros(n, dtype=np.int32); OUT_ENDS = np.zeros(n, dtype=np.int32)
            lib.predict_batch_c(ctypes.c_int(n), ctypes.c_int(dim), ctypes.c_int(BM_mat.shape[0]), _fp(BM_mat)[0], _fp(self._cached_W_stack)[0], _fp(self._cached_HP_stack)[0], _fp(self._cached_B_stack)[0], _fp(self._cached_BIAS)[0], self._cached_TYPES.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), self._cached_N_HEADS.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), self._cached_SEEDS.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)), _fp(CONS)[0], ctypes.c_float(context_entropy), ctypes.c_int(1 if causal else 0), _fp(OUT_PREDS)[0], _fp(OUT_CONFS)[0], OUT_STARTS.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), OUT_ENDS.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
            res = []
            for i, dot in enumerate(dots):
                p = dot.current_phase if dot.current_phase != 0.0 else float(2.0*np.pi*((OUT_STARTS[i]+OUT_ENDS[i])*0.5)/max(BM_mat.shape[0],1)); pf = np.exp(1j * p)
                for h in range(dot.n_heads): res.append((OUT_PREDS[i*8+h].astype(np.complex64)*pf, float(OUT_CONFS[i*8+h]), {"dot_id": dot.dot_id, "dot_type": _TYPE_NAMES[dot.dot_type], "head": h, "slice": (int(OUT_STARTS[i]), int(OUT_ENDS[i])), "phase": p, "bias": dot.bias, "source": "original"}))
            return res
        res = []
        for d in dots: res.extend(d.predict(basemap, context_entropy=context_entropy, consensus=consensus, causal=causal))
        return res

    def type_distribution(self, dots: List[NeuralDot]) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for dot in dots:
            name = _TYPE_NAMES[dot.dot_type]
            dist[name] = dist.get(name, 0) + 1
        return dist
