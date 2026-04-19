"""
Neural Dots — independent prediction units.

A neural dot is fundamentally different from a neuron:
  Neuron: tiny signal-passer in a fixed layered system
  Neural Dot: complete mini-predictor on its own

Each dot receives a slice of the BaseMap matrix, pools it using
attention shaped by its bias vector, applies its own weight matrix,
and returns a candidate prediction. Dots are stateless — memory
emerges from convergence, not from individual dot state.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import prediction_confidence, bias_vector_update
from basemapping.basemapping import BaseMap


class BiasVector:
    """
    5-dimensional control vector for dot generation.

    Dimensions:
      attention_bias     — sharpness of focus on input slice (0=uniform, 1=sharp)
      granularity_bias   — patch vs word vs global pooling (0=fine, 1=coarse)
      abstraction_bias   — raw features vs transformed (0=raw, 1=abstract)
      inversion_bias     — frequency of AIM inversion usage (0=never, 1=always)
      sampling_temperature — diversity of outputs (low=focused, high=diverse)
    """
    DIM = 5

    def __init__(self, attention_bias=0.5, granularity_bias=0.5,
                 abstraction_bias=0.5, inversion_bias=0.3, sampling_temperature=1.0):
        self.attention_bias      = float(np.clip(attention_bias, 0.0, 1.0))
        self.granularity_bias    = float(np.clip(granularity_bias, 0.0, 1.0))
        self.abstraction_bias    = float(np.clip(abstraction_bias, 0.0, 1.0))
        self.inversion_bias      = float(np.clip(inversion_bias, 0.0, 1.0))
        self.sampling_temperature = float(max(sampling_temperature, 1e-6))

    def to_array(self) -> np.ndarray:
        return np.array([self.attention_bias, self.granularity_bias,
                         self.abstraction_bias, self.inversion_bias,
                         self.sampling_temperature], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "BiasVector":
        return cls(*arr.tolist())

    @classmethod
    def random(cls, rng: Optional[np.random.RandomState] = None) -> "BiasVector":
        rng = rng or np.random.RandomState()
        return cls(rng.uniform(0.1, 0.9), rng.uniform(0.0, 1.0),
                   rng.uniform(0.0, 1.0), rng.uniform(0.0, 0.6),
                   rng.uniform(0.3, 2.0))

    def update(self, winning: "BiasVector", lr: float = 0.1) -> "BiasVector":
        """Formula 8 — shift toward winning bias pattern."""
        return BiasVector.from_array(bias_vector_update(self.to_array(), winning.to_array(), lr))

    def __repr__(self):
        return (f"BiasVector(attn={self.attention_bias:.2f}, gran={self.granularity_bias:.2f}, "
                f"abst={self.abstraction_bias:.2f}, inv={self.inversion_bias:.2f}, "
                f"temp={self.sampling_temperature:.2f})")


class NeuralDot:
    """
    A stateless, independent prediction unit.

    Given a slice of a BaseMap matrix, a dot:
      1. Selects its view based on granularity_bias
      2. Pools with attention shaped by attention_bias
      3. Applies abstraction transform (raw → semantic)
      4. Adds temperature noise for diversity
      5. Returns a (prediction_vector, confidence, info) tuple
    """

    def __init__(self, dot_id: int, feature_dim: int = 128,
                 bias: Optional[BiasVector] = None, seed: Optional[int] = None):
        self.dot_id      = dot_id
        self.feature_dim = feature_dim
        self.bias        = bias or BiasVector()
        seed             = seed if seed is not None else dot_id
        rng              = np.random.RandomState(seed)
        scale            = 1.0 / np.sqrt(feature_dim)
        self.W           = rng.randn(feature_dim, feature_dim).astype(np.float32) * scale
        self.b           = rng.randn(feature_dim).astype(np.float32) * scale * 0.1
        self._rng        = rng

    def _select_slice(self, basemap: BaseMap) -> Tuple[np.ndarray, int, int]:
        n = len(basemap)
        gran = self.bias.granularity_bias
        patch_size = max(1, int(n * (0.1 + gran * 0.9)))
        offset = int(self._rng.uniform(0, max(1, n - patch_size + 1)))
        start  = min(offset, n - 1)
        end    = min(start + patch_size, n)
        return basemap.matrix[start:end], start, end

    def _pool(self, mat: np.ndarray) -> np.ndarray:
        if mat.shape[0] == 1: return mat[0]
        query  = np.mean(mat, axis=0, keepdims=True)
        scores = (mat @ query.T).flatten() * (self.bias.attention_bias * 5.0)
        scores -= np.max(scores)
        w = np.exp(scores); w /= w.sum() + 1e-10
        return (w[:, None] * mat).sum(axis=0)

    def _abstract(self, v: np.ndarray) -> np.ndarray:
        a = self.bias.abstraction_bias
        return (1.0 - a) * v + a * np.tanh(self.W @ v + self.b)

    def _noise(self, v: np.ndarray) -> np.ndarray:
        T = self.bias.sampling_temperature
        return v + self._rng.randn(len(v)).astype(np.float32) * T * 0.05

    def predict(self, basemap: BaseMap) -> Tuple[np.ndarray, float, Dict]:
        sl, start, end = self._select_slice(basemap)
        pooled   = self._pool(sl)
        abstract = self._abstract(pooled)
        noisy    = self._noise(abstract)
        norm     = np.linalg.norm(noisy)
        pred     = noisy / norm * np.sqrt(self.feature_dim) if norm > 1e-10 else noisy
        conf     = prediction_confidence(pred)
        info     = {"dot_id": self.dot_id, "slice": (start, end), "bias": self.bias}
        return pred, conf, info

    def __repr__(self):
        return f"NeuralDot(id={self.dot_id}, {self.bias})"


class DotGenerator:
    """Creates a diverse pool of neural dots with varied bias vectors."""

    def __init__(self, num_dots: int = 64, feature_dim: int = 128,
                 base_bias: Optional[BiasVector] = None, seed: int = 42):
        self.num_dots    = num_dots
        self.feature_dim = feature_dim
        self.base_bias   = base_bias or BiasVector()
        self.seed        = seed
        self._rng        = np.random.RandomState(seed)

    def generate(self) -> List[NeuralDot]:
        dots = []
        base_arr = self.base_bias.to_array()
        for i in range(self.num_dots):
            noise = self._rng.randn(BiasVector.DIM).astype(np.float32) * 0.3
            arr   = np.clip(base_arr + noise, 0.01, 1.99)
            arr[-1] = max(arr[-1], 0.1)
            dots.append(NeuralDot(i, self.feature_dim, BiasVector.from_array(arr),
                                  seed=self.seed + i * 31))
        return dots

    def run_all(self, basemap: BaseMap, dots: List[NeuralDot]) -> List[Tuple]:
        return [dot.predict(basemap) for dot in dots]
