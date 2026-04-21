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
    SEMANTIC   = 0
    STRUCTURAL = 1
    CONTEXTUAL = 2
    RELATIONAL = 3
    TEMPORAL   = 4
    GLOBAL     = 5
    LOGIC      = 6
    MORPH      = 7

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
    DotType.MORPH:      (236, 252),
}

_TYPE_BIAS_PRESETS = {
    DotType.SEMANTIC:   (0.7, 0.3, 0.5, 0.3, 0.8),
    DotType.STRUCTURAL: (0.6, 0.5, 0.3, 0.2, 0.6),
    DotType.CONTEXTUAL: (0.4, 0.8, 0.8, 0.4, 1.2),
    DotType.RELATIONAL: (0.8, 0.4, 0.6, 0.5, 1.0),
    DotType.TEMPORAL:   (0.5, 0.6, 0.4, 0.3, 0.9),
    DotType.GLOBAL:     (0.3, 0.9, 0.7, 0.2, 0.7),
    DotType.LOGIC:      (0.9, 0.6, 0.7, 0.1, 0.5),
    DotType.MORPH:      (0.8, 0.2, 0.4, 0.1, 0.4),
}

N_HEADS = 4
_NEXT_DOT_ID = 1000

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
    def __init__(self, dot_id=None, feature_dim=256, bias=None, dot_type=DotType.SEMANTIC, n_heads=N_HEADS, seed=None):
        self.dot_id = dot_id if dot_id is not None else get_next_dot_id()
        self.feature_dim = feature_dim
        self.bias = bias or BiasVector.from_dot_type(dot_type)
        self.dot_type = dot_type
        self.n_heads = n_heads
        self.max_heads = 8
        seed = seed or self.dot_id * 31 + 7
        rng = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(feature_dim)
        self.W = rng.randn(feature_dim, feature_dim).astype(np.float32) * scale
        self.head_projs = [rng.randn(feature_dim, feature_dim).astype(np.float32) * scale for _ in range(self.max_heads)]
        self.Q_basis = rng.randn(feature_dim, feature_dim).astype(np.float32) * scale
        self.b_offset = rng.randn(feature_dim).astype(np.float32) * scale * 0.1
        self._rng = np.random.RandomState(seed + 1)

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

    def _select_slice(self, matrix):
        n = matrix.shape[0]
        if n <= 1: return matrix, 0, n
        start = self._rng.randint(0, n-1)
        end = min(start + 10, n)
        return matrix[start:end], start, end

    def _apply_dim_focus(self, v):
        lo, hi = _TYPE_DIM_RANGES[self.dot_type]
        out = np.zeros_like(v)
        out[lo:hi] = v[lo:hi]
        return out

    def _temporal_pool(self, mat):
        lib = _load_lib()
        n, d = mat.shape
        out = np.zeros(d, dtype=np.float32)
        if lib:
            lib.temporal_pool(_fp(mat)[0], ctypes.c_int(n), ctypes.c_int(d), _fp(out)[0])
            return out
        return np.mean(mat, axis=0)

    def _relational_pool(self, mat):
        lib = _load_lib()
        n, d = mat.shape
        out = np.zeros(d, dtype=np.float32)
        if lib:
            lib.relational_pool(_fp(mat)[0], ctypes.c_int(n), ctypes.c_int(d), _fp(out)[0])
            return out
        return np.mean(mat, axis=0)

    def _logic_pool(self, mat):
        lib = _load_lib()
        n, d = mat.shape
        out = np.zeros(d, dtype=np.float32)
        if lib:
            lib.logic_pool(_fp(mat)[0], ctypes.c_int(n), ctypes.c_int(d), _fp(out)[0])
            return out
        return np.mean(mat, axis=0)

class DotGenerator:
    def __init__(self, num_dots=128, feature_dim=256, base_bias=None, n_heads=N_HEADS, seed=42):
        self.num_dots = num_dots
        self.feature_dim = feature_dim
        self.seed = seed
        self.n_heads = n_heads
        self._rng = np.random.RandomState(seed)

    def generate(self):
        return [NeuralDot(feature_dim=self.feature_dim, dot_type=DotType(self._rng.randint(0, 8)), seed=self.seed + i, n_heads=self.n_heads) for i in range(self.num_dots)]

    def run_all(self, basemap, dots, memory_hints=None, context_entropy=0.5, consensus=None):
        all_res = []
        for d in dots:
            all_res.extend(d.predict(basemap, context_entropy=context_entropy))
        return all_res

    def apply_synergy(self, v, peer_v, synergy_weight=0.15):
        lib = _load_lib()
        if lib:
            out = v.copy()
            lib.apply_synergy_fast(_fp(out)[0], _fp(peer_v)[0], ctypes.c_int(len(v)), ctypes.c_float(synergy_weight))
            return out
        return (1.0 - synergy_weight) * v + synergy_weight * peer_v

    def type_distribution(self, dots):
        dist = {}
        for d in dots:
            name = _TYPE_NAMES[d.dot_type]
            dist[name] = dist.get(name, 0) + 1
        return dist
