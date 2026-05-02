"""
AIM — Attention Inverse Mechanism

AIM runs in PARALLEL with original dot predictions. Both originals
and AIM variants enter the Convergence Layer together — they compete
and reinforce each other, preserving the emergent agreement principle.

Nine inversion types (F1–F6 original, F7–F9 new):
  FEATURE      — flip dominant attribute dimensions (bright→dark)
  CONTEXT      — swap high/low activation groups (foreground→background)
  SPATIAL      — reverse dimension group ordering (mirror/rotate)
  SCALE        — rescale groups by inverse factors (local↔global)
  ABSTRACTION  — flip between levels of understanding (patch↔object)
  NOISE        — suppress dominant features ("what if absent?")
  RELATIONAL   — invert pairwise correlations between dimension blocks
  TEMPORAL     — reverse temporal order in position-encoded dims
  COMPOSITIONAL— SVD decompose, permute singular vectors, recompose

Each inversion challenges a different assumption the dot made.
Strong inversions that still land in the winning cluster signal
that the prediction is robust across multiple perspectives.
"""

import numpy as np
import ctypes
import os
from typing import List, Tuple, Dict, Optional
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import aim_transform, prediction_confidence
from basemapping.basemapping import BaseMap


# ── Load C shared library ─────────────────────────────────────────────
_lib = None

def _load_lib():
    global _lib
    if _lib is not None: return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    so   = os.path.join(here, "aim_c.so")
    if os.path.exists(so):
        try:
            _lib = ctypes.CDLL(so)
            for fn in ("invert_feature", "invert_context", "invert_spatial",
                       "invert_scale", "invert_abstraction", "invert_noise",
                       "invert_relational", "invert_temporal", "invert_cross_modal"):
                getattr(_lib, fn).restype = None
        except Exception:
            _lib = None
    return _lib


def _fp(arr):
    a = np.ascontiguousarray(arr, np.float32)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), a


class InversionType:
    # Original six
    FEATURE       = "feature"
    CONTEXT       = "context"
    SPATIAL       = "spatial"
    SCALE         = "scale"
    ABSTRACTION   = "abstraction"
    NOISE         = "noise"
    # New three
    RELATIONAL    = "relational"
    TEMPORAL      = "temporal"
    COMPOSITIONAL = "compositional"
    # Cross-Modal
    CROSS_MODAL   = "cross_modal"
    # Evolved
    EVOLVED       = "evolved"

    ALL = [FEATURE, CONTEXT, SPATIAL, SCALE, ABSTRACTION, NOISE,
           RELATIONAL, TEMPORAL, COMPOSITIONAL, CROSS_MODAL, EVOLVED]

    # Weights for random selection (research-informed priors)
    WEIGHTS = {
        FEATURE:       1.5,
        CONTEXT:       1.2,
        SPATIAL:       1.0,
        SCALE:         1.0,
        ABSTRACTION:   0.9,
        NOISE:         0.8,
        RELATIONAL:    1.1,
        TEMPORAL:      1.0,
        COMPOSITIONAL: 0.7,
        CROSS_MODAL:   1.3,
        EVOLVED:       1.5,
    }


# ── Original Six Inversions (C-accelerated) ───────────────────────────

def _invert_feature(p: np.ndarray) -> np.ndarray:
    """Negate dimensions whose |value| exceeds the mean absolute value."""
    lib = _load_lib()
    p32 = np.ascontiguousarray(np.real(p), np.float32)
    out = np.zeros_like(p32)
    if lib:
        fp_, _p = _fp(p32); fo, _o = _fp(out)
        lib.invert_feature(fp_, ctypes.c_int(len(p32)), fo)
        return _o
    mean_abs = float(np.mean(np.abs(p32)))
    out = p32.copy(); out[np.abs(p32) > mean_abs] *= -1
    return out


def _invert_context(p: np.ndarray) -> np.ndarray:
    """Swap high-activation and low-activation halves."""
    lib = _load_lib()
    p32 = np.ascontiguousarray(np.real(p), np.float32)
    out = np.zeros_like(p32)
    if lib:
        fp_, _p = _fp(p32); fo, _o = _fp(out)
        lib.invert_context(fp_, ctypes.c_int(len(p32)), fo)
        return _o
    idx = np.argsort(np.abs(p32)); half = len(p32)//2
    out = p32.copy()
    out[idx[:half]], out[idx[half:]] = p32[idx[half:]].copy(), p32[idx[:half]].copy()
    return out


def _invert_spatial(p: np.ndarray) -> np.ndarray:
    """Reverse the ordering of equal-size dimension groups."""
    lib = _load_lib()
    p32 = np.ascontiguousarray(np.real(p), np.float32)
    out = np.zeros_like(p32)
    if lib:
        fp_, _p = _fp(p32); fo, _o = _fp(out)
        lib.invert_spatial(fp_, ctypes.c_int(len(p32)), fo)
        return _o
    out = p32.copy(); gs = max(1, len(p32)//8); ng = len(p32)//gs
    for g in range(ng//2):
        s1, s2 = g*gs, (ng-1-g)*gs
        out[s1:s1+gs], out[s2:s2+gs] = p32[s2:s2+gs].copy(), p32[s1:s1+gs].copy()
    return out


def _invert_scale(p: np.ndarray) -> np.ndarray:
    """Rescale dimension quarters by inverse factors, then renormalize."""
    lib = _load_lib()
    p32 = np.ascontiguousarray(np.real(p), np.float32)
    out = np.zeros_like(p32)
    if lib:
        fp_, _p = _fp(p32); fo, _o = _fp(out)
        lib.invert_scale(fp_, ctypes.c_int(len(p32)), fo)
        return _o
    out = p32.copy(); q = len(p32)//4; orig_n = np.linalg.norm(p32)
    if q > 0:
        for s, sc in enumerate([4.0, 2.0, 0.5, 0.25]):
            out[s*q:(s+1)*q] *= sc
        n = np.linalg.norm(out)
        if n > 1e-10 and orig_n > 1e-10: out *= orig_n / n
    return out


def _invert_abstraction(p: np.ndarray, ctx: Optional[np.ndarray] = None) -> np.ndarray:
    """Flip between levels of understanding using context projection."""
    lib = _load_lib()
    p32 = np.ascontiguousarray(np.real(p), np.float32)
    out = np.zeros_like(p32)
    if lib:
        fp_, _p = _fp(p32); fo, _o = _fp(out)
        if ctx is not None:
            ctx32 = np.ascontiguousarray(np.real(ctx.flatten()[:len(p32)]), np.float32)
            if len(ctx32) < len(p32):
                ctx32 = np.pad(ctx32, (0, len(p32)-len(ctx32)))
            fc, _c = _fp(ctx32)
            lib.invert_abstraction(fp_, fc, ctypes.c_int(len(p32)), fo)
        else:
            lib.invert_abstraction(fp_, None, ctypes.c_int(len(p32)), fo)
        return _o
    if ctx is not None:
        ctx_flat = ctx.flatten()[:len(p32)]
        ctx_n = np.linalg.norm(ctx_flat)
        if ctx_n > 1e-10:
            ctx_u = ctx_flat / ctx_n
            proj  = np.dot(p32, ctx_u) * ctx_u
            return (p32 - proj) + 0.1 * proj
    half = len(p32)//2
    out = p32.copy(); out[:half], out[half:] = p32[half:half+half].copy(), p32[:half].copy()
    return out


def _invert_noise(p: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Suppress dominant features; add small structured noise."""
    lib = _load_lib()
    p32 = np.ascontiguousarray(np.real(p), np.float32)
    out = np.zeros_like(p32)
    seed = int(rng.randint(0, 2**31))
    if lib:
        fp_, _p = _fp(p32); fo, _o = _fp(out)
        lib.invert_noise(fp_, ctypes.c_int(len(p32)), ctypes.c_uint(seed), fo)
        return _o
    thr = float(np.percentile(np.abs(p32), 75))
    out = p32.copy()
    dominant = np.abs(p32) > thr
    out[dominant] *= rng.uniform(0.0, 0.2, size=dominant.sum())
    out += rng.randn(len(p32)).astype(np.float32) * float(np.std(p32)) * 0.3
    return out


# ── Three New Inversions (Pure Python) ────────────────────────────────

def _invert_relational(p: np.ndarray) -> np.ndarray:
    """Relational Inversion (C-accelerated)."""
    lib = _load_lib()
    p32 = np.ascontiguousarray(np.real(p), np.float32)
    out = np.zeros_like(p32)
    if lib:
        fp_, _p = _fp(p32); fo, _o = _fp(out)
        lib.invert_relational(fp_, ctypes.c_int(len(p32)), fo)
        return _o
    return p32.copy()


def _invert_temporal(p: np.ndarray) -> np.ndarray:
    """Temporal Inversion (C-accelerated)."""
    lib = _load_lib()
    p32 = np.ascontiguousarray(np.real(p), np.float32)
    out = np.zeros_like(p32)
    if lib:
        fp_, _p = _fp(p32); fo, _o = _fp(out)
        lib.invert_temporal(fp_, ctypes.c_int(len(p32)), fo)
        return _o
    return p32.copy()


def _invert_compositional(p: np.ndarray) -> np.ndarray:
    """
    Compositional Inversion: reshape the vector into a pseudo 2D matrix,
    compute thin SVD, permute the singular vectors, and recompose.
    Produces a semantically restructured vector of the same norm.
    """
    n    = len(p)
    orig_norm = np.linalg.norm(p)
    # Reshape to near-square matrix
    rows = max(2, int(np.sqrt(n)))
    while n % rows != 0 and rows > 1: rows -= 1
    cols = n // rows

    try:
        mat = np.real(p).reshape(rows, cols).astype(np.float64)
        U, s, Vt = np.linalg.svd(mat, full_matrices=False)
        k = min(len(s), max(2, len(s) // 2))
        # Permute: rotate singular vectors by k//2 positions
        shift = k // 2
        U_perm  = np.roll(U[:, :k], shift, axis=1)
        Vt_perm = np.roll(Vt[:k, :], shift, axis=0)
        mat_inv = U_perm @ np.diag(s[:k]) @ Vt_perm
        out = mat_inv.flatten()[:n].astype(np.float32)
    except Exception:
        return p.copy()

    out_norm = np.linalg.norm(out)
    if out_norm > 1e-10 and orig_norm > 1e-10:
        out = out * (orig_norm / out_norm)
    return out


# ── Inversion dispatch ────────────────────────────────────────────────

def _invert_evolved(p: np.ndarray, dot_info: Dict) -> np.ndarray:
    """Evolved Inversion: uses the dot's learned W_inv matrix."""
    W_inv = dot_info.get("W_inv")
    if W_inv is None:
        # Fallback to feature inversion if W_inv not provided
        return _invert_feature(p)

    out = np.tanh(W_inv @ np.real(p).astype(np.float32))
    orig_norm = np.linalg.norm(p)
    out_norm = np.linalg.norm(out)
    if out_norm > 1e-10 and orig_norm > 1e-10:
        out = out * (orig_norm / out_norm)
    return out

def _invert_cross_modal(p: np.ndarray) -> np.ndarray:
    """Cross-Modal Inversion (C-accelerated)."""
    lib = _load_lib()
    p32 = np.ascontiguousarray(np.real(p), np.float32)
    out = np.zeros_like(p32)
    if lib:
        fp_, _p = _fp(p32); fo, _o = _fp(out)
        lib.invert_cross_modal(fp_, ctypes.c_int(len(p32)), fo)
        return _o
    return p32.copy()


def _apply_inversion(inv_type: str, pred: np.ndarray, ctx: np.ndarray,
                     rng: np.random.RandomState, dot_info: Optional[Dict] = None) -> np.ndarray:
    """Dispatch to the correct inversion function."""
    if inv_type == InversionType.FEATURE:
        return _invert_feature(pred)
    if inv_type == InversionType.CONTEXT:
        return _invert_context(pred)
    if inv_type == InversionType.SPATIAL:
        return _invert_spatial(pred)
    if inv_type == InversionType.SCALE:
        return _invert_scale(pred)
    if inv_type == InversionType.ABSTRACTION:
        return _invert_abstraction(pred, ctx)
    if inv_type == InversionType.NOISE:
        return _invert_noise(pred, rng)
    if inv_type == InversionType.RELATIONAL:
        return _invert_relational(pred)
    if inv_type == InversionType.TEMPORAL:
        return _invert_temporal(pred)
    if inv_type == InversionType.COMPOSITIONAL:
        return _invert_compositional(pred)
    if inv_type == InversionType.CROSS_MODAL:
        return _invert_cross_modal(pred)
    if inv_type == InversionType.EVOLVED and dot_info:
        return _invert_evolved(pred, dot_info)
    return pred.copy()


class AIMLayer:
    """
    Attention Inverse Mechanism Layer — 9 inversion types.

    Produces inverted variants of each dot prediction, then refines
    them with attention over the BaseMap context. Both originals and
    AIM variants enter Convergence together.

    Inversion type selection is weighted (see InversionType.WEIGHTS)
    and controlled by each dot's inversion_bias.
    """

    ALL_WEIGHTS = np.array(
        [InversionType.WEIGHTS[t] for t in InversionType.ALL], dtype=np.float64
    )

    def __init__(self, max_variants_per_dot: int = 4, seed: int = 0):
        self.max_variants = max_variants_per_dot
        self._rng = np.random.RandomState(seed)
        # Normalize weights
        self._weights = self.ALL_WEIGHTS / self.ALL_WEIGHTS.sum()

    def _pick_inversions(self, inv_bias: float, requested: Optional[str] = None) -> List[str]:
        """Pick inversions, prioritizing requested hypotheses."""
        if self.max_variants <= 0:
            return []

        n = max(1, round(inv_bias * self.max_variants))
        n = min(n, len(InversionType.ALL))

        chosen = []
        if requested and requested in InversionType.ALL:
            chosen.append(requested)

        remaining_n = n - len(chosen)
        if remaining_n > 0:
            # Mask out already chosen
            weights = self._weights.copy()
            if requested:
                try:
                    req_idx = InversionType.ALL.index(requested)
                    weights[req_idx] = 0.0
                    weights /= weights.sum()
                except ValueError:
                    pass

            idx = self._rng.choice(len(InversionType.ALL), size=remaining_n, replace=False, p=weights)
            chosen.extend([InversionType.ALL[i] for i in idx])

        return chosen

    def _apply(self, pred: np.ndarray, inv_type: str, ctx: np.ndarray, dot_info: Optional[Dict] = None) -> np.ndarray:
        inv = _apply_inversion(inv_type, pred, ctx, self._rng, dot_info=dot_info)
        ctx2d = ctx.reshape(1, -1) if ctx.ndim == 1 else ctx
        return aim_transform(pred, ctx2d, lambda _: inv)

    def transform(self, predictions: List[Tuple], basemap: BaseMap) -> List[Tuple]:
        """
        Apply AIM to all predictions.
        Returns originals + inverted variants (all combined for Convergence).
        """
        ctx = basemap.pool("mean")
        out = []
        for pred, conf, info in predictions:
            out.append((pred, conf, {**info, "source": "original", "inversion_type": None}))
            inv_bias = info.get("bias").inversion_bias if info.get("bias") else 0.3
            requested = info.get("requested_inversion")

            for inv_type in self._pick_inversions(inv_bias, requested=requested):
                try:
                    inv = self._apply(pred, inv_type, ctx, dot_info=info)
                    inv_conf = prediction_confidence(inv)
                    out.append((inv, inv_conf, {
                        **info,
                        "source":           f"aim:{inv_type}",
                        "inversion_type":   inv_type,
                        "original_dot_id":  info.get("dot_id"),
                    }))
                except Exception:
                    pass
        return out

    def inversion_summary(self, candidates: List[Tuple]) -> Dict[str, int]:
        """Count how many candidates exist per inversion type."""
        counts: Dict[str, int] = {}
        for _, _, info in candidates:
            t = info.get("inversion_type") or "original"
            counts[t] = counts.get(t, 0) + 1
        return counts
