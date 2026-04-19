"""
AIM — Attention Inverse Mechanism

AIM runs in PARALLEL with original dot predictions. Both originals
and AIM variants enter the Convergence Layer together — they compete
and reinforce each other, preserving the emergent agreement principle.

Six inversion types challenge different kinds of prediction assumptions:
  feature    — flip attribute signs (bright→dark, convex→concave)
  context    — swap role of high/low activations (foreground→background)
  spatial    — reverse dimension group ordering (mirror/rotate)
  scale      — rescale groups to reinterpret local vs global structure
  abstraction— flip between levels of understanding (patch→object or reverse)
  noise      — suppress dominant features ("what if this isn't there?")
"""

import numpy as np
import ctypes
import os
from typing import List, Tuple, Dict, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import aim_transform, prediction_confidence
from basemapping.basemapping import BaseMap


# ── Load C shared library ────────────────────────────────────────────
_lib = None

def _load_lib():
    global _lib
    if _lib is not None: return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    so   = os.path.join(here, "aim_c.so")
    if os.path.exists(so):
        try:
            _lib = ctypes.CDLL(so)
            for fn in ("invert_feature","invert_context","invert_spatial",
                       "invert_scale","invert_abstraction","invert_noise"):
                getattr(_lib, fn).restype = None
        except Exception:
            _lib = None
    return _lib


def _fp(arr):
    a = np.ascontiguousarray(arr, np.float32)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), a


class InversionType:
    FEATURE     = "feature"
    CONTEXT     = "context"
    SPATIAL     = "spatial"
    SCALE       = "scale"
    ABSTRACTION = "abstraction"
    NOISE       = "noise"
    ALL = [FEATURE, CONTEXT, SPATIAL, SCALE, ABSTRACTION, NOISE]


def _invert_feature(p: np.ndarray) -> np.ndarray:
    lib = _load_lib()
    p32 = np.ascontiguousarray(p, np.float32)
    out = np.zeros_like(p32)
    if lib:
        fp, _p = _fp(p32); fo, _o = _fp(out)
        lib.invert_feature(fp, ctypes.c_int(len(p32)), fo)
        return _o
    mean_abs = float(np.mean(np.abs(p32)))
    out = p32.copy(); out[np.abs(p32) > mean_abs] *= -1
    return out


def _invert_context(p: np.ndarray) -> np.ndarray:
    lib = _load_lib()
    p32 = np.ascontiguousarray(p, np.float32)
    out = np.zeros_like(p32)
    if lib:
        fp, _p = _fp(p32); fo, _o = _fp(out)
        lib.invert_context(fp, ctypes.c_int(len(p32)), fo)
        return _o
    idx = np.argsort(np.abs(p32)); half = len(p32)//2
    out = p32.copy()
    out[idx[:half]], out[idx[half:]] = p32[idx[half:]].copy(), p32[idx[:half]].copy()
    return out


def _invert_spatial(p: np.ndarray) -> np.ndarray:
    lib = _load_lib()
    p32 = np.ascontiguousarray(p, np.float32)
    out = np.zeros_like(p32)
    if lib:
        fp, _p = _fp(p32); fo, _o = _fp(out)
        lib.invert_spatial(fp, ctypes.c_int(len(p32)), fo)
        return _o
    out = p32.copy(); gs = max(1, len(p32)//8); ng = len(p32)//gs
    for g in range(ng//2):
        s1, s2 = g*gs, (ng-1-g)*gs
        out[s1:s1+gs], out[s2:s2+gs] = p32[s2:s2+gs].copy(), p32[s1:s1+gs].copy()
    return out


def _invert_scale(p: np.ndarray) -> np.ndarray:
    lib = _load_lib()
    p32 = np.ascontiguousarray(p, np.float32)
    out = np.zeros_like(p32)
    if lib:
        fp, _p = _fp(p32); fo, _o = _fp(out)
        lib.invert_scale(fp, ctypes.c_int(len(p32)), fo)
        return _o
    out = p32.copy(); q = len(p32)//4; orig_n = np.linalg.norm(p32)
    if q > 0:
        for s, sc in enumerate([4.0,2.0,0.5,0.25]):
            out[s*q:(s+1)*q] *= sc
        n = np.linalg.norm(out)
        if n > 1e-10 and orig_n > 1e-10: out *= orig_n/n
    return out


def _invert_abstraction(p: np.ndarray, ctx: Optional[np.ndarray] = None) -> np.ndarray:
    lib = _load_lib()
    p32  = np.ascontiguousarray(p, np.float32)
    out  = np.zeros_like(p32)
    if lib:
        fp, _p = _fp(p32)
        fo, _o = _fp(out)
        if ctx is not None:
            ctx32 = np.ascontiguousarray(ctx.flatten()[:len(p32)], np.float32)
            if len(ctx32) < len(p32):
                ctx32 = np.pad(ctx32, (0, len(p32)-len(ctx32)))
            fc, _c = _fp(ctx32)
            lib.invert_abstraction(fp, fc, ctypes.c_int(len(p32)), fo)
        else:
            lib.invert_abstraction(fp, None, ctypes.c_int(len(p32)), fo)
        return _o
    if ctx is not None:
        ctx_flat = ctx.flatten()[:len(p32)]
        ctx_n = np.linalg.norm(ctx_flat)
        if ctx_n > 1e-10:
            ctx_u = ctx_flat / ctx_n
            proj = np.dot(p32, ctx_u) * ctx_u
            return (p32 - proj) + 0.1 * proj
    half = len(p32)//2
    out = p32.copy(); out[:half], out[half:] = p32[half:half+half].copy(), p32[:half].copy()
    return out


def _invert_noise(p: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    lib = _load_lib()
    p32 = np.ascontiguousarray(p, np.float32)
    out = np.zeros_like(p32)
    seed = int(rng.randint(0, 2**31))
    if lib:
        fp, _p = _fp(p32); fo, _o = _fp(out)
        lib.invert_noise(fp, ctypes.c_int(len(p32)), ctypes.c_uint(seed), fo)
        return _o
    thr = float(np.percentile(np.abs(p32), 75))
    out = p32.copy()
    dominant = np.abs(p32) > thr
    out[dominant] *= rng.uniform(0.0, 0.2, size=dominant.sum())
    out += rng.randn(len(p32)).astype(np.float32) * float(np.std(p32)) * 0.3
    return out


_INVERSION_MAP = {
    InversionType.FEATURE:     _invert_feature,
    InversionType.CONTEXT:     _invert_context,
    InversionType.SPATIAL:     _invert_spatial,
    InversionType.SCALE:       _invert_scale,
    InversionType.ABSTRACTION: _invert_abstraction,
    InversionType.NOISE:       _invert_noise,
}


class AIMLayer:
    """
    Attention Inverse Mechanism Layer.

    Produces inverted variants of each dot prediction, then refines
    them with attention over the BaseMap context. Both originals and
    AIM variants enter Convergence together — not sequentially.
    """

    def __init__(self, max_variants_per_dot: int = 3, seed: int = 0):
        self.max_variants = max_variants_per_dot
        self._rng = np.random.RandomState(seed)

    def _pick_inversions(self, inv_bias: float) -> List[str]:
        n = max(1, int(inv_bias * self.max_variants + 0.5))
        n = min(n, len(InversionType.ALL))
        w = np.array([1.5, 1.2, 1.0, 1.0, 1.0, 0.8], np.float32)
        w /= w.sum()
        idx = self._rng.choice(len(InversionType.ALL), size=n, replace=False, p=w)
        return [InversionType.ALL[i] for i in idx]

    def _apply(self, pred: np.ndarray, inv_type: str, ctx: np.ndarray) -> np.ndarray:
        fn = _INVERSION_MAP[inv_type]
        if inv_type == InversionType.ABSTRACTION:
            inv = fn(pred, ctx)
        elif inv_type == InversionType.NOISE:
            inv = fn(pred, self._rng)
        else:
            inv = fn(pred)
        ctx2d = ctx.reshape(1, -1) if ctx.ndim == 1 else ctx
        return aim_transform(pred, ctx2d, lambda _: inv)

    def transform(self, predictions: List[Tuple], basemap: BaseMap) -> List[Tuple]:
        """
        Apply AIM to all predictions.
        Returns originals + inverted variants combined.
        """
        ctx = basemap.pool("mean")
        out = []
        for pred, conf, info in predictions:
            out.append((pred, conf, {**info, "source": "original", "inversion_type": None}))
            inv_bias = info.get("bias").inversion_bias if info.get("bias") else 0.3
            for inv_type in self._pick_inversions(inv_bias):
                try:
                    inv = self._apply(pred, inv_type, ctx)
                    out.append((inv, prediction_confidence(inv), {
                        **info,
                        "source": f"aim:{inv_type}",
                        "inversion_type": inv_type,
                        "original_dot_id": info.get("dot_id"),
                    }))
                except Exception:
                    pass
        return out
