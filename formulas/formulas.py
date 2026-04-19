"""
IECNN Formulas — Python layer with C acceleration via ctypes.

All 9 custom IECNN formulas. The C shared library is used for
performance-critical batch operations. Pure Python fallbacks are
always available if the library hasn't been compiled yet.
"""

import numpy as np
import ctypes
import os
from typing import List

# ── Load C shared library ────────────────────────────────────────────
_lib = None

def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, "formulas_c.so")
    if os.path.exists(so_path):
        try:
            _lib = ctypes.CDLL(so_path)
            _lib.cosine_sim.restype          = ctypes.c_float
            _lib.agreement_str.restype       = ctypes.c_float
            _lib.similarity_score.restype    = ctypes.c_float
            _lib.convergence_score.restype   = ctypes.c_float
            _lib.prediction_confidence.restype = ctypes.c_float
            _lib.pairwise_similarity_matrix.restype = None
            _lib.similarity_vs_all.restype   = None
            _lib.bias_vector_update.restype  = None
            _lib.attention_single.restype    = None
        except Exception:
            _lib = None
    return _lib


def _fp(arr: np.ndarray):
    """Return ctypes float pointer to a contiguous float32 array."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), a


# ── Formula 1 helpers ────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    lib = _load_lib()
    a32 = np.ascontiguousarray(a, np.float32)
    b32 = np.ascontiguousarray(b, np.float32)
    if lib:
        return float(lib.cosine_sim(
            a32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(len(a32))
        ))
    na = np.linalg.norm(a32); nb = np.linalg.norm(b32)
    return 0.0 if na < 1e-10 or nb < 1e-10 else float(np.dot(a32, b32) / (na * nb))


def agreement_strength(a: np.ndarray, b: np.ndarray) -> float:
    """Agreement strength between two predictions."""
    lib = _load_lib()
    a32 = np.ascontiguousarray(a, np.float32)
    b32 = np.ascontiguousarray(b, np.float32)
    if lib:
        return float(lib.agreement_str(
            a32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(len(a32))
        ))
    dot = float(np.dot(a32, b32))
    mag = np.linalg.norm(a32) * np.linalg.norm(b32)
    if mag < 1e-10: return 0.0
    norm_dot = dot / mag
    strength = (np.linalg.norm(a32) + np.linalg.norm(b32)) / 2.0
    ref = np.sqrt(len(a32))
    return float(np.tanh(strength / ref) * ((norm_dot + 1.0) / 2.0))


def similarity_score(a: np.ndarray, b: np.ndarray, alpha: float = 0.7) -> float:
    """Formula 1: S(p_i, p_j) = alpha*cos(p_i,p_j) + (1-alpha)*A(p_i,p_j)"""
    lib = _load_lib()
    a32 = np.ascontiguousarray(a, np.float32)
    b32 = np.ascontiguousarray(b, np.float32)
    if lib:
        return float(lib.similarity_score(
            a32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(len(a32)), ctypes.c_float(alpha)
        ))
    return alpha * cosine_similarity(a32, b32) + (1.0 - alpha) * agreement_strength(a32, b32)


def convergence_score(predictions: List[np.ndarray], confidences: List[float],
                      alpha: float = 0.7) -> float:
    """Formula 2: C(k) = (1/|k|²) * Σ S(p_i,p_j) * mean_confidence"""
    n = len(predictions)
    if n == 0: return 0.0
    if n == 1: return float(confidences[0])
    lib = _load_lib()
    stack = np.ascontiguousarray(np.stack(predictions), dtype=np.float32)
    confs = np.ascontiguousarray(confidences, dtype=np.float32)
    dim = stack.shape[1]
    if lib:
        return float(lib.convergence_score(
            stack.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            confs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n), ctypes.c_int(dim), ctypes.c_float(alpha)
        ))
    total = sum(similarity_score(predictions[i], predictions[j], alpha)
                for i in range(n) for j in range(n))
    return (total / (n * n)) * float(np.mean(confs))


def prediction_confidence(p: np.ndarray) -> float:
    """Normalized L2 norm as confidence measure."""
    lib = _load_lib()
    p32 = np.ascontiguousarray(p, np.float32)
    if lib:
        return float(lib.prediction_confidence(
            p32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(len(p32))
        ))
    return float(np.tanh(np.linalg.norm(p32) / np.sqrt(len(p32))))


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Formula 3: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k)) V"""
    lib = _load_lib()
    if Q.ndim == 1: Q = Q.reshape(1, -1)
    if K.ndim == 1: K = K.reshape(1, -1)
    if V.ndim == 1: V = V.reshape(1, -1)
    seq_len, dim = K.shape

    if Q.shape[0] == 1 and lib:
        Q32 = np.ascontiguousarray(Q[0], np.float32)
        K32 = np.ascontiguousarray(K, np.float32)
        V32 = np.ascontiguousarray(V, np.float32)
        out = np.zeros(dim, dtype=np.float32)
        lib.attention_single(
            Q32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            K32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            V32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(seq_len), ctypes.c_int(dim),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return out

    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    scores -= np.max(scores, axis=-1, keepdims=True)
    w = np.exp(scores); w /= (w.sum(axis=-1, keepdims=True) + 1e-10)
    return (w @ V).flatten()


def aim_transform(prediction: np.ndarray, context: np.ndarray, inversion_fn) -> np.ndarray:
    """Formula 4: p_hat = Attention(Q, K, Invert(p))"""
    inverted = inversion_fn(prediction)
    ctx = context.reshape(-1, len(prediction)) if context.ndim == 1 else context
    inv2d = inverted.reshape(1, -1)
    return attention(inv2d, ctx, ctx).flatten()


def novelty_gain(new_cluster_count: int, total_cluster_count: int) -> float:
    """Formula 6: NG(t) = |NewClusters(t)| / |TotalClusters(t)|"""
    if total_cluster_count == 0: return 0.0
    return new_cluster_count / total_cluster_count


def sampling_temperature_sample(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Formula 7: P(v) ∝ exp(score(v)/T)"""
    T = max(temperature, 1e-6)
    lp = scores / T; lp -= np.max(lp)
    p = np.exp(lp); p /= p.sum() + 1e-10
    return p


def bias_vector_update(b: np.ndarray, w: np.ndarray, lr: float = 0.1) -> np.ndarray:
    """Formula 8: b_(t+1) = b_t + eta*(w_t - b_t)"""
    lib = _load_lib()
    b32 = np.ascontiguousarray(b, np.float32)
    w32 = np.ascontiguousarray(w, np.float32)
    out = np.zeros_like(b32)
    if lib:
        lib.bias_vector_update(
            b32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(len(b32)), ctypes.c_float(lr),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return out
    return b32 + lr * (w32 - b32)


def dominance_score(leading: float, all_scores: List[float]) -> float:
    """Formula 9: Dominance(k*) = C(k*) / Σ C(k)"""
    total = sum(all_scores)
    if total < 1e-10: return 0.0
    return leading / total


def pairwise_similarity_matrix(predictions: List[np.ndarray], alpha: float = 0.7) -> np.ndarray:
    """Compute full n×n similarity matrix — C-accelerated when available."""
    n = len(predictions)
    if n == 0: return np.array([])
    lib = _load_lib()
    stack = np.ascontiguousarray(np.stack(predictions), dtype=np.float32)
    dim = stack.shape[1]
    out = np.zeros((n, n), dtype=np.float32)
    if lib:
        lib.pairwise_similarity_matrix(
            stack.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n), ctypes.c_int(dim), ctypes.c_float(alpha),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return out
    for i in range(n):
        for j in range(i, n):
            s = similarity_score(predictions[i], predictions[j], alpha)
            out[i, j] = s; out[j, i] = s
    return out
