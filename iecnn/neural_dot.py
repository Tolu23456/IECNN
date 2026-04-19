import numpy as np
from typing import List, Optional, Tuple, Dict
from .basemapping import BaseMap
from .formulas import prediction_confidence, sampling_temperature_sample, bias_vector_update


class BiasVector:
    """
    Controls how a neural dot generates predictions.

    Each dimension targets a distinct aspect of how the dot
    perceives and processes its assigned input slice.

    Dimensions:
      attention_bias     — what parts of the input to focus on (0=uniform, 1=sharp)
      granularity_bias   — patch vs word vs global pooling level (0=fine, 1=coarse)
      abstraction_bias   — raw features vs semantic concepts (0=raw, 1=abstract)
      inversion_bias     — how often AIM-style inversions are applied (0=never, 1=always)
      sampling_temperature — diversity of dot outputs (low=focused, high=diverse)
    """

    DIM = 5

    def __init__(
        self,
        attention_bias: float = 0.5,
        granularity_bias: float = 0.5,
        abstraction_bias: float = 0.5,
        inversion_bias: float = 0.3,
        sampling_temperature: float = 1.0,
    ):
        self.attention_bias = float(np.clip(attention_bias, 0.0, 1.0))
        self.granularity_bias = float(np.clip(granularity_bias, 0.0, 1.0))
        self.abstraction_bias = float(np.clip(abstraction_bias, 0.0, 1.0))
        self.inversion_bias = float(np.clip(inversion_bias, 0.0, 1.0))
        self.sampling_temperature = float(max(sampling_temperature, 1e-6))

    def to_array(self) -> np.ndarray:
        return np.array([
            self.attention_bias,
            self.granularity_bias,
            self.abstraction_bias,
            self.inversion_bias,
            self.sampling_temperature,
        ], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "BiasVector":
        return cls(
            attention_bias=float(arr[0]),
            granularity_bias=float(arr[1]),
            abstraction_bias=float(arr[2]),
            inversion_bias=float(arr[3]),
            sampling_temperature=float(arr[4]),
        )

    @classmethod
    def random(cls, rng: Optional[np.random.RandomState] = None) -> "BiasVector":
        """Generate a random bias vector (for diverse dot initialization)."""
        if rng is None:
            rng = np.random.RandomState()
        return cls(
            attention_bias=rng.uniform(0.1, 0.9),
            granularity_bias=rng.uniform(0.0, 1.0),
            abstraction_bias=rng.uniform(0.0, 1.0),
            inversion_bias=rng.uniform(0.0, 0.6),
            sampling_temperature=rng.uniform(0.3, 2.0),
        )

    def update(self, winning: "BiasVector", learning_rate: float = 0.1) -> "BiasVector":
        """
        Formula 8 — update this bias vector toward the winning pattern.
        b_(t+1) = b_t + eta * (w_t - b_t)
        """
        b_new = bias_vector_update(self.to_array(), winning.to_array(), learning_rate)
        return BiasVector.from_array(b_new)

    def __repr__(self):
        return (
            f"BiasVector(attn={self.attention_bias:.2f}, "
            f"gran={self.granularity_bias:.2f}, "
            f"abst={self.abstraction_bias:.2f}, "
            f"inv={self.inversion_bias:.2f}, "
            f"temp={self.sampling_temperature:.2f})"
        )


class NeuralDot:
    """
    A small, independent prediction unit.

    Unlike a neuron (which passes signals in a fixed layer),
    a neural dot is a complete mini-predictor. It:
      1. Receives a slice/view of the BaseMap matrix
      2. Pools that slice using attention (shaped by its bias vector)
      3. Applies its weight matrix to produce a candidate prediction
      4. Adds controlled noise based on sampling temperature

    Dots are stateless — input → prediction, no internal state carried
    across iterations. Memory emerges from convergence, not individual dots.
    """

    def __init__(
        self,
        dot_id: int,
        feature_dim: int = 128,
        bias: Optional[BiasVector] = None,
        seed: Optional[int] = None,
    ):
        self.dot_id = dot_id
        self.feature_dim = feature_dim
        self.bias = bias or BiasVector()

        rng = np.random.RandomState(seed if seed is not None else dot_id)
        scale = 1.0 / np.sqrt(feature_dim)
        self.W = rng.randn(feature_dim, feature_dim).astype(np.float32) * scale
        self.b = rng.randn(feature_dim).astype(np.float32) * scale * 0.1
        self._rng = rng

    def _select_slice(self, basemap: BaseMap) -> Tuple[np.ndarray, int, int]:
        """
        Select which rows of the BaseMap matrix this dot operates on.
        Determined by granularity_bias:
          0.0 = fine (single patch / small group of bases)
          0.5 = word-level group
          1.0 = global (all bases)
        """
        n = len(basemap)
        gran = self.bias.granularity_bias

        if gran < 0.33:
            patch_size = max(1, n // 8)
        elif gran < 0.66:
            patch_size = max(1, n // 3)
        else:
            patch_size = n

        offset = int(self._rng.uniform(0, max(1, n - patch_size + 1)))
        start = min(offset, n - 1)
        end = min(start + patch_size, n)
        return basemap.matrix[start:end], start, end

    def _pool_with_attention(self, slice_matrix: np.ndarray) -> np.ndarray:
        """
        Pool the slice matrix into a single vector using attention.
        attention_bias controls sharpness:
          0.0 = uniform mean pooling
          1.0 = sharp attention (winner-takes-most)
        """
        if slice_matrix.shape[0] == 1:
            return slice_matrix[0]

        query = np.mean(slice_matrix, axis=0, keepdims=True)
        scores = slice_matrix @ query.T
        scores = scores.flatten()

        sharpness = self.bias.attention_bias * 5.0
        scores = scores * sharpness
        scores -= np.max(scores)
        weights = np.exp(scores)
        weights /= weights.sum() + 1e-10

        return (weights[:, None] * slice_matrix).sum(axis=0)

    def _apply_abstraction(self, vec: np.ndarray) -> np.ndarray:
        """
        abstraction_bias controls how much the dot transforms
        toward semantic (more processed) vs raw representations.
          0.0 = identity (raw features)
          1.0 = full non-linear transformation
        """
        alpha = self.bias.abstraction_bias
        raw = vec.copy()
        transformed = np.tanh(self.W @ vec + self.b)
        return (1.0 - alpha) * raw + alpha * transformed

    def _add_temperature_noise(self, vec: np.ndarray) -> np.ndarray:
        """
        Add noise scaled by sampling temperature.
        High temperature → more diverse predictions across dots.
        """
        T = self.bias.sampling_temperature
        noise_scale = T * 0.05
        noise = self._rng.randn(vec.shape[0]).astype(np.float32) * noise_scale
        return vec + noise

    def predict(self, basemap: BaseMap) -> Tuple[np.ndarray, float, Dict]:
        """
        Generate a single candidate prediction from this dot.

        Returns:
          prediction: np.ndarray of shape (feature_dim,)
          confidence: float in [0, 1]
          info: dict with slice info and dot metadata
        """
        slice_matrix, start, end = self._select_slice(basemap)
        pooled = self._pool_with_attention(slice_matrix)
        abstracted = self._apply_abstraction(pooled)
        with_noise = self._add_temperature_noise(abstracted)

        norm = np.linalg.norm(with_noise)
        if norm > 1e-10:
            prediction = with_noise / norm * np.sqrt(self.feature_dim)
        else:
            prediction = with_noise

        confidence = prediction_confidence(prediction)

        info = {
            "dot_id": self.dot_id,
            "slice": (start, end),
            "slice_size": end - start,
            "bias": self.bias,
        }

        return prediction, confidence, info

    def __repr__(self):
        return f"NeuralDot(id={self.dot_id}, {self.bias})"


class DotGenerator:
    """
    Generates a pool of neural dots, each with a different bias vector.

    The pool is designed for diverse coverage:
      - Some dots focus on fine-grained details (low granularity)
      - Some focus on global structure (high granularity)
      - Some use raw features (low abstraction)
      - Some use highly transformed features (high abstraction)
      - Inversion bias varies to control AIM usage
    """

    def __init__(
        self,
        num_dots: int = 64,
        feature_dim: int = 128,
        base_bias: Optional[BiasVector] = None,
        seed: int = 42,
    ):
        self.num_dots = num_dots
        self.feature_dim = feature_dim
        self.base_bias = base_bias or BiasVector()
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    def generate(self) -> List[NeuralDot]:
        """
        Generate the full pool of neural dots with diverse biases.
        Each dot gets a unique bias vector sampled around the base bias.
        """
        dots = []
        base_arr = self.base_bias.to_array()

        for i in range(self.num_dots):
            noise = self._rng.randn(BiasVector.DIM).astype(np.float32) * 0.3
            biased_arr = np.clip(base_arr + noise, 0.01, 1.99)
            biased_arr[-1] = max(biased_arr[-1], 0.1)

            bias = BiasVector.from_array(biased_arr)
            dot = NeuralDot(
                dot_id=i,
                feature_dim=self.feature_dim,
                bias=bias,
                seed=self.seed + i * 31,
            )
            dots.append(dot)

        return dots

    def run_all(self, basemap: BaseMap, dots: List[NeuralDot]) -> List[Tuple[np.ndarray, float, Dict]]:
        """
        Run all dots on the basemap and collect predictions.
        Each dot produces: (prediction_vector, confidence, info)
        """
        results = []
        for dot in dots:
            pred, conf, info = dot.predict(basemap)
            results.append((pred, conf, info))
        return results
