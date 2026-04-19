import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from .formulas import attention, aim_transform, prediction_confidence
from .basemapping import BaseMap


class InversionType:
    FEATURE = "feature"
    CONTEXT = "context"
    SPATIAL = "spatial"
    SCALE = "scale"
    ABSTRACTION = "abstraction"
    NOISE = "noise"

    ALL = [FEATURE, CONTEXT, SPATIAL, SCALE, ABSTRACTION, NOISE]


def invert_feature(p: np.ndarray) -> np.ndarray:
    """
    Feature Inversion — flip attribute signs.
    Instead of "bright edge," generate "dark edge."
    Instead of "convex shape," try "concave shape."
    Negates dominant dimensions to challenge assumptions about key attributes.
    """
    threshold = float(np.mean(np.abs(p)))
    inv = p.copy()
    dominant_mask = np.abs(p) > threshold
    inv[dominant_mask] = -p[dominant_mask]
    return inv


def invert_context(p: np.ndarray) -> np.ndarray:
    """
    Context Inversion — role reversal.
    A patch predicted as "foreground" is reinterpreted as "background."
    Swaps the high-activation and low-activation dimension groups,
    challenging the dot's assumption about importance.
    """
    n = len(p)
    half = n // 2
    sorted_idx = np.argsort(np.abs(p))
    low_half = sorted_idx[:half]
    high_half = sorted_idx[half:]
    inv = p.copy()
    temp = inv[low_half].copy()
    inv[low_half] = inv[high_half]
    inv[high_half] = temp
    return inv


def invert_spatial(p: np.ndarray) -> np.ndarray:
    """
    Spatial Inversion — mirror/rotate semantic spatial structure.
    Same content, different spatial meaning.
    Reverses the ordering of dimension groups (like mirroring or rotating).
    """
    n = len(p)
    group_size = max(1, n // 8)
    inv = p.copy()
    num_groups = n // group_size
    for g in range(num_groups // 2):
        start1 = g * group_size
        start2 = (num_groups - 1 - g) * group_size
        end1 = start1 + group_size
        end2 = start2 + group_size
        temp = inv[start1:end1].copy()
        inv[start1:end1] = inv[start2:end2]
        inv[start2:end2] = temp
    return inv


def invert_scale(p: np.ndarray) -> np.ndarray:
    """
    Scale Inversion — reinterpret local vs global structure.
    A pattern assumed to be "small detail" becomes part of a larger structure.
    Rescales dimension groups by inverse factors to shift zoom level.
    """
    n = len(p)
    inv = p.copy()
    quarter = n // 4
    if quarter > 0:
        inv[:quarter] = p[:quarter] * 4.0
        inv[quarter:2 * quarter] = p[quarter:2 * quarter] * 2.0
        inv[2 * quarter:3 * quarter] = p[2 * quarter:3 * quarter] * 0.5
        inv[3 * quarter:] = p[3 * quarter:] * 0.25
    norm = np.linalg.norm(inv)
    if norm > 1e-10:
        inv = inv / norm * np.linalg.norm(p)
    return inv


def invert_abstraction(p: np.ndarray, context_pool: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Abstraction Inversion — flip between levels of understanding.
    From patch → object hypothesis, or object → decomposed patches.
    If a context pool is available, mixes the prediction with its complement
    in the context space.
    """
    if context_pool is not None and context_pool.ndim >= 1:
        ctx = context_pool.flatten()[:len(p)]
        if len(ctx) < len(p):
            ctx = np.pad(ctx, (0, len(p) - len(ctx)))
        ctx_norm = np.linalg.norm(ctx)
        p_norm = np.linalg.norm(p)
        if ctx_norm > 1e-10:
            ctx_unit = ctx / ctx_norm
        else:
            ctx_unit = ctx
        projection = np.dot(p, ctx_unit) * ctx_unit
        complement = p - projection
        return complement + projection * 0.1
    else:
        n = len(p)
        half = n // 2
        inv = p.copy()
        inv[:half] = p[half:half + half]
        inv[half:half + half] = p[:half]
        return inv


def invert_noise(p: np.ndarray, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Noise/Absence Inversion — "What if this feature isn't actually there?"
    Suppresses dominant features to generate alternative predictions
    and explore what the system would predict without key cues.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    threshold = float(np.percentile(np.abs(p), 75))
    inv = p.copy()
    dominant = np.abs(p) > threshold
    inv[dominant] = inv[dominant] * rng.uniform(0.0, 0.2, size=dominant.sum())
    noise = rng.randn(len(p)).astype(np.float32) * float(np.std(p)) * 0.3
    inv = inv + noise
    return inv


INVERSION_FNS: Dict[str, Callable] = {
    InversionType.FEATURE: invert_feature,
    InversionType.CONTEXT: invert_context,
    InversionType.SPATIAL: invert_spatial,
    InversionType.SCALE: invert_scale,
    InversionType.ABSTRACTION: invert_abstraction,
    InversionType.NOISE: invert_noise,
}


class AIMLayer:
    """
    Attention Inverse Mechanism Layer.

    AIM operates in PARALLEL with original dot predictions — both
    originals and AIM variants enter the Convergence Layer together.
    They compete and reinforce each other, preserving emergent agreement.

    For each prediction:
      1. Select an inversion type based on inversion_bias
      2. Apply the inversion function
      3. Refine with attention over context (the BaseMap matrix)
      4. Return the refined candidate alongside the original

    During learning:
      If an inversion consistently wins over the original,
      the system adjusts dot generation bias toward that inversion's pattern.
    """

    def __init__(
        self,
        max_variants_per_dot: int = 3,
        seed: int = 0,
    ):
        self.max_variants_per_dot = max_variants_per_dot
        self._rng = np.random.RandomState(seed)

    def _select_inversion_types(self, inversion_bias: float) -> List[str]:
        """
        Select inversion types for this dot based on its inversion_bias.
        Higher inversion_bias → more inversion types attempted.
        """
        num_to_apply = max(1, int(inversion_bias * self.max_variants_per_dot + 0.5))
        num_to_apply = min(num_to_apply, len(InversionType.ALL))

        weights = np.ones(len(InversionType.ALL), dtype=np.float32)
        weights[0] *= 1.5
        weights[1] *= 1.2
        weights[-1] *= 0.8

        weights /= weights.sum()
        chosen_indices = self._rng.choice(
            len(InversionType.ALL),
            size=num_to_apply,
            replace=False,
            p=weights,
        )
        return [InversionType.ALL[i] for i in chosen_indices]

    def _apply_single_inversion(
        self,
        prediction: np.ndarray,
        inv_type: str,
        context: np.ndarray,
    ) -> np.ndarray:
        """Apply a single inversion type and refine with attention."""
        fn = INVERSION_FNS[inv_type]

        if inv_type == InversionType.ABSTRACTION:
            inverted = fn(prediction, context)
        elif inv_type == InversionType.NOISE:
            inverted = fn(prediction, self._rng)
        else:
            inverted = fn(prediction)

        if context.ndim == 1:
            ctx_2d = context.reshape(1, -1)
        else:
            ctx_2d = context

        refined = aim_transform(prediction, ctx_2d, lambda p: inverted)
        return refined

    def transform(
        self,
        predictions: List[Tuple[np.ndarray, float, Dict]],
        basemap: BaseMap,
    ) -> List[Tuple[np.ndarray, float, Dict]]:
        """
        Apply AIM to all predictions and return original + inverted variants.

        Returns a combined list of (prediction, confidence, info) tuples
        where info includes 'source': 'original' or 'aim:<inversion_type>'.
        """
        context = basemap.pool("mean")
        all_candidates = []

        for pred, conf, info in predictions:
            original_info = {**info, "source": "original", "inversion_type": None}
            all_candidates.append((pred, conf, original_info))

            inversion_bias = info.get("bias").inversion_bias if info.get("bias") else 0.3
            inv_types = self._select_inversion_types(inversion_bias)

            for inv_type in inv_types:
                try:
                    inverted = self._apply_single_inversion(pred, inv_type, context)
                    inv_conf = prediction_confidence(inverted)
                    inv_info = {
                        **info,
                        "source": f"aim:{inv_type}",
                        "inversion_type": inv_type,
                        "original_dot_id": info.get("dot_id"),
                    }
                    all_candidates.append((inverted, inv_conf, inv_info))
                except Exception:
                    pass

        return all_candidates

    def get_winning_inversions(
        self,
        candidates: List[Tuple[np.ndarray, float, Dict]],
        surviving_cluster_ids: List[int],
        cluster_assignments: List[int],
    ) -> Dict[str, int]:
        """
        Identify which inversion types are present in surviving clusters.
        Used to compute AIM-assisted learning signal.
        """
        winning_inversions: Dict[str, int] = {}
        surviving_set = set(surviving_cluster_ids)

        for i, (_, _, info) in enumerate(candidates):
            if i < len(cluster_assignments) and cluster_assignments[i] in surviving_set:
                inv_type = info.get("inversion_type")
                if inv_type is not None:
                    winning_inversions[inv_type] = winning_inversions.get(inv_type, 0) + 1

        return winning_inversions
