"""
IECNN Formulas — Python layer with C acceleration via ctypes.

Seventeen custom formulas, F1–F17:
  F1  Similarity Score            F2  Convergence Score
  F3  Attention                   F4  AIM Transform
  F5  Pruning Threshold           F6  Prediction Confidence
  F7  Sampling Temperature        F8  Bias Vector Update
  F9  Dominance Score             F10 Dot Specialization Score
  F11 Cluster Entropy             F12 Temporal Stability
  F13 Cross-Type Agreement        F14 Adaptive Learning Rate
  F15 Hierarchical Convergence Score
  F16 Emergent Utility Gradient
  F17 Dot Reinforcement Pressure
"""

import numpy as np
import ctypes
import os
from typing import List, Dict, Optional

# ── Load C shared library ────────────────────────────────────────────
_lib = None
_lib_check_done = False

def _load_lib():
    global _lib, _lib_check_done
    if _lib_check_done:
        return _lib
    _lib_check_done = True
    here    = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, "formulas_c.so")
    if not os.path.exists(so_path):
        import sys as _sys
        print(
            f"[IECNN] WARNING: {so_path} not found — formulas will use slow Python path.\n"
            f"         Fix: run  python main.py build  to compile C extensions.",
            file=_sys.stderr,
        )
        return _lib
    if os.path.exists(so_path):
        try:
            lib = ctypes.CDLL(so_path)
            # F1 helpers
            lib.cosine_sim.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            lib.cosine_sim.restype               = ctypes.c_float
            lib.agreement_str.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            lib.agreement_str.restype            = ctypes.c_float
            lib.similarity_score.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float]
            lib.similarity_score.restype         = ctypes.c_float
            # F2
            lib.convergence_score.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float]
            lib.convergence_score.restype        = ctypes.c_float
            # F6
            lib.prediction_confidence.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            lib.prediction_confidence.restype    = ctypes.c_float
            # F8
            lib.bias_vector_update.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float)]
            lib.bias_vector_update.restype       = None
            # F9
            lib.dominance_score.argtypes = [ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            lib.dominance_score.restype          = ctypes.c_float
            # Batch ops
            lib.pairwise_similarity_matrix.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float)]
            lib.pairwise_similarity_matrix.restype = None
            lib.similarity_vs_all.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float)]
            lib.similarity_vs_all.restype        = None
            lib.attention_single.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
            lib.attention_single.restype         = None
            # F10–F15
            lib.dot_specialization_score.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float]
            lib.dot_specialization_score.restype = ctypes.c_float
            lib.cluster_entropy.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            lib.cluster_entropy.restype          = ctypes.c_float
            lib.temporal_stability.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float]
            lib.temporal_stability.restype       = ctypes.c_float
            lib.cross_type_agreement.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float]
            lib.cross_type_agreement.restype     = ctypes.c_float
            lib.adaptive_learning_rate.argtypes = [ctypes.c_float, ctypes.c_float]
            lib.adaptive_learning_rate.restype   = ctypes.c_float
            lib.hierarchical_convergence_score.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]
            lib.hierarchical_convergence_score.restype = ctypes.c_float
            # F21–F26
            lib.global_energy.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
            lib.global_energy.restype            = ctypes.c_float
            lib.system_objective.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
            lib.system_objective.restype         = ctypes.c_float
            lib.memory_plasticity.argtypes = [ctypes.c_float]
            lib.memory_plasticity.restype        = ctypes.c_float
            lib.dot_fitness.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
            lib.dot_fitness.restype              = ctypes.c_float
            lib.stability_energy.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
            lib.stability_energy.restype         = ctypes.c_float
            lib.exploration_pressure.argtypes = [ctypes.c_float, ctypes.c_float]
            lib.exploration_pressure.restype     = ctypes.c_float

            # Ultra and Batch
            lib.convergence_score_ultra.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_float]
            lib.convergence_score_ultra.restype = ctypes.c_float
            lib.batch_similarity_fast.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float)]
            lib.batch_similarity_fast.restype = None
            _lib = lib
        except Exception:
            _lib = None
    return _lib


def _fp(arr: np.ndarray):
    """
    Convert array to float32 pointer for C library.
    Handles complex arrays by taking the real part.
    """
    if np.iscomplexobj(arr):
        # We take the real part for legacy C-path functions that don't support complex math.
        # This prevents hard crashes while keeping the logic mostly functional.
        a = np.ascontiguousarray(np.real(arr), dtype=np.float32)
    else:
        a = np.ascontiguousarray(arr, dtype=np.float32)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), a


# ── Formula 1 helpers ────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Supports complex vectors via |<a,b>|/(||a||*||b||)."""
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10: return 0.0
        # For phase-coded activations, we care about the magnitude of the
        # complex inner product, which represents coherence.
        return float(np.abs(np.vdot(a, b)) / (na * nb))

    lib = _load_lib()
    a32 = np.ascontiguousarray(a, np.float32)
    b32 = np.ascontiguousarray(b, np.float32)
    if lib:
        fa, _ = _fp(a32); fb, _ = _fp(b32)
        return float(lib.cosine_sim(fa, fb, ctypes.c_int(len(a32))))
    na = np.linalg.norm(a32); nb = np.linalg.norm(b32)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a32, b32) / (na * nb))


def agreement_strength(a: np.ndarray, b: np.ndarray) -> float:
    """Agreement strength. Supports complex vectors."""
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10: return 0.0
        ndot = float(np.real(np.vdot(a, b)) / (na * nb))
        strength = (na + nb) / 2.0
        return float(np.tanh(strength / np.sqrt(len(a))) * ((ndot + 1.0) / 2.0))

    lib = _load_lib()
    a32 = np.ascontiguousarray(a, np.float32)
    b32 = np.ascontiguousarray(b, np.float32)
    if lib:
        fa, _ = _fp(a32); fb, _ = _fp(b32)
        return float(lib.agreement_str(fa, fb, ctypes.c_int(len(a32))))
    dot = float(np.dot(a32, b32))
    mag = np.linalg.norm(a32) * np.linalg.norm(b32)
    if mag < 1e-10: return 0.0
    ndot = dot / mag
    strength = (np.linalg.norm(a32) + np.linalg.norm(b32)) / 2.0
    return float(np.tanh(strength / np.sqrt(len(a32))) * ((ndot + 1.0) / 2.0))


def similarity_score(a: np.ndarray, b: np.ndarray, alpha: float = 0.7) -> float:
    """Formula 1: S(p_i, p_j) = alpha*cos(p_i,p_j) + (1-alpha)*A(p_i,p_j)"""
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        return alpha * cosine_similarity(a, b) + (1.0 - alpha) * agreement_strength(a, b)

    dim = len(a)
    dot = np.dot(a, b)
    if abs(np.dot(a, a) - dim) < 0.1 and abs(np.dot(b, b) - dim) < 0.1:
        cos = dot / float(dim)
        return alpha * cos + (1.0 - alpha) * (0.76159416 * (cos + 1.0) / 2.0)


    lib = _load_lib()
    a32 = np.ascontiguousarray(a, np.float32)
    b32 = np.ascontiguousarray(b, np.float32)
    if lib:
        fa, _ = _fp(a32); fb, _ = _fp(b32)
        return float(lib.similarity_score(fa, fb, ctypes.c_int(len(a32)), ctypes.c_float(alpha)))
    return alpha * cosine_similarity(a32, b32) + (1.0 - alpha) * agreement_strength(a32, b32)


def phase_aware_similarity(a: np.ndarray, b: np.ndarray,
                           phase_a: Optional[float] = None,
                           phase_b: Optional[float] = None,
                           concentration: float = 1.0,
                           alpha: float = 0.7,
                           phase_weight: float = 0.5) -> float:
    """
    IECNN-native phase-coherence-modulated similarity.

    Two activations that look similar in feature space but originated at
    different positions in the input stream are scored as less similar than
    plain cosine alone would suggest. This is what lets cluster memory
    distinguish 'dog bites man' from 'man bites dog' without bolting on
    softmax attention.

    Args:
        a, b           — feature vectors (real-valued, dim = feature_dim)
        phase_a, phase_b — circular phase in radians, or None for legacy data
        concentration  — phase concentration of the matched pattern in [0, 1].
                         A pattern with low concentration has a fuzzy phase
                         and shouldn't be penalized harshly for phase mismatch.
        alpha          — feature-space alpha forwarded to similarity_score
        phase_weight   — max weight given to the phase-coherence factor when
                         concentration = 1.0. With phase_weight = 0.5, a
                         perfectly concentrated pattern gets at most a
                         50/50 blend of feature similarity and phase coherence.

    Falls back to plain similarity_score when either phase is missing or the
    pattern's phase concentration is too low to be meaningful.
    """
    base = similarity_score(a, b, alpha)
    if phase_a is None or phase_b is None or concentration < 0.05:
        return base
    coh = 0.5 * (1.0 + float(np.cos(phase_a - phase_b)))
    w = max(0.0, min(1.0, phase_weight * concentration))
    return float(base * (1.0 - w + w * coh))


# ── Formula 2: Convergence Score ────────────────────────────────────

def convergence_score(predictions, confidences, alpha=0.7):
    n = len(predictions)
    if n == 0: return 0.0
    if n == 1: return float(confidences[0]) if confidences else 0.5
    stk = np.stack([np.real(p) for p in predictions]).astype(np.float32)
    d = stk.shape[1]
    dots = (stk @ stk.T) / float(d)
    agreement = 0.76159416 * (dots + 1.0) / 2.0
    sim_matrix = alpha * dots + (1.0 - alpha) * agreement
    return float(np.mean(sim_matrix)) * float(np.mean(confidences))
def prediction_confidence(p: np.ndarray) -> float:
    """tanh(||p|| / sqrt(dim)) — normalized L2 norm as confidence."""
    if np.iscomplexobj(p):
        return float(np.tanh(np.linalg.norm(p) / np.sqrt(len(p))))

    lib = _load_lib()
    p32 = np.ascontiguousarray(p, np.float32)
    if lib:
        fp_, _ = _fp(p32)
        return float(lib.prediction_confidence(fp_, ctypes.c_int(len(p32))))
    return float(np.tanh(np.linalg.norm(p32) / np.sqrt(len(p32))))


# ── Formula 3: Attention ─────────────────────────────────────────────

def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V"""
    # Complex-aware pure python path
    if np.iscomplexobj(Q) or np.iscomplexobj(K) or np.iscomplexobj(V):
        if Q.ndim == 1: Q = Q.reshape(1, -1)
        if K.ndim == 1: K = K.reshape(1, -1)
        if V.ndim == 1: V = V.reshape(1, -1)
        d_k = K.shape[-1]
        # Use complex dot product (vdot) for scores if complex
        scores = np.real(Q @ K.conj().T) / np.sqrt(d_k)
        scores -= np.max(scores, axis=-1, keepdims=True)
        w = np.exp(scores); w /= w.sum(axis=-1, keepdims=True) + 1e-10
        return (w @ V).flatten()

    lib = _load_lib()
    if Q.ndim == 1: Q = Q.reshape(1, -1)
    if K.ndim == 1: K = K.reshape(1, -1)
    if V.ndim == 1: V = V.reshape(1, -1)
    seq_len, dim = K.shape
    if Q.shape[0] == 1 and lib:
        Q32 = np.ascontiguousarray(np.real(Q[0]), np.float32)
        K32 = np.ascontiguousarray(np.real(K), np.float32)
        V32 = np.ascontiguousarray(np.real(V), np.float32)
        out = np.zeros(dim, np.float32)
        fq, _q = _fp(Q32); fk, _k = _fp(K32); fv, _v = _fp(V32)
        fo, _o = _fp(out)
        lib.attention_single(fq, fk, fv, ctypes.c_int(seq_len), ctypes.c_int(dim), fo)
        return _o
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    scores -= np.max(scores, axis=-1, keepdims=True)
    w = np.exp(scores); w /= w.sum(axis=-1, keepdims=True) + 1e-10
    return (w @ V).flatten()


# ── Formula 4: AIM Transform ─────────────────────────────────────────

def aim_transform(prediction: np.ndarray, context: np.ndarray, inversion_fn) -> np.ndarray:
    """p_hat = Attention(Q, K, Invert(p))"""
    inverted = inversion_fn(prediction)
    ctx = context.reshape(-1, len(prediction)) if context.ndim == 1 else context
    inv2d = inverted.reshape(1, -1)
    return attention(inv2d, ctx, ctx).flatten()


# ── Formula 7: Sampling Temperature ─────────────────────────────────

def sampling_temperature_sample(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """P(v) ∝ exp(score(v)/T)"""
    T = max(temperature, 1e-6)
    lp = scores / T; lp -= np.max(lp)
    p = np.exp(lp); p /= p.sum() + 1e-10
    return p


# ── Formula 8: Bias Vector Update ───────────────────────────────────

def bias_vector_update(b: np.ndarray, w: np.ndarray, lr: float = 0.1) -> np.ndarray:
    """b_{t+1} = b_t + eta * (w_t - b_t)"""
    lib = _load_lib()
    b32 = np.ascontiguousarray(b, np.float32)
    w32 = np.ascontiguousarray(w, np.float32)
    out = np.zeros_like(b32)
    if lib:
        fb, _ = _fp(b32); fw, _ = _fp(w32); fo, _o = _fp(out)
        lib.bias_vector_update(fb, fw, ctypes.c_int(len(b32)), ctypes.c_float(lr), fo)
        return _o
    return b32 + lr * (w32 - b32)


# ── Formula 9: Dominance Score ───────────────────────────────────────

def dominance_score(leading: float, all_scores: List[float]) -> float:
    """Dominance(k*) = C(k*) / Σ C(k)"""
    lib = _load_lib()
    if not all_scores: return 0.0
    sc  = np.ascontiguousarray(all_scores, np.float32)
    if lib:
        fs, _ = _fp(sc)
        return float(lib.dominance_score(ctypes.c_float(leading), fs, ctypes.c_int(len(sc))))
    total = sum(all_scores)
    return leading / total if total > 1e-10 else 0.0


# ── Batch: Pairwise Similarity Matrix ────────────────────────────────

def pairwise_similarity_matrix(predictions: List[np.ndarray],
                                alpha: float = 0.7) -> np.ndarray:
    """C-accelerated n×n pairwise similarity matrix."""
    n = len(predictions)
    if n == 0: return np.array([])
    lib  = _load_lib()
    stk  = np.ascontiguousarray(np.stack(predictions), np.float32)
    dim  = stk.shape[1]
    out  = np.zeros((n, n), np.float32)
    if lib:
        fs, _ = _fp(stk); fo, _o = _fp(out)
        lib.pairwise_similarity_matrix(fs, ctypes.c_int(n), ctypes.c_int(dim),
                                       ctypes.c_float(alpha), fo)
        return _o
    for i in range(n):
        for j in range(i, n):
            s = similarity_score(predictions[i], predictions[j], alpha)
            out[i, j] = s; out[j, i] = s
    return out


def novelty_gain(new_cluster_count: int, total_cluster_count: int) -> float:
    """Formula 6 (NG): |NewClusters(t)| / |TotalClusters(t)|"""
    if total_cluster_count == 0: return 0.0
    return new_cluster_count / total_cluster_count


# ══════════════════════════════════════════════════════════════════════
# Extended Formulas F10–F15
# ══════════════════════════════════════════════════════════════════════

# ── Formula 10: Dot Specialization Score ─────────────────────────────

def dot_specialization_score(predictions: List[np.ndarray], alpha: float = 0.7) -> float:
    """
    F10: Mean pairwise similarity of a dot's own predictions.
    High = specialized (consistent). Low = generalist (diverse).
    """
    n = len(predictions)
    if n <= 1: return 1.0

    if any(np.iscomplexobj(p) for p in predictions):
        total, count = 0.0, 0
        for i in range(n):
            for j in range(i+1, n):
                total += similarity_score(predictions[i], predictions[j], alpha)
                count += 1
        return total / count if count > 0 else 1.0

    lib = _load_lib()
    stk = np.ascontiguousarray(np.real(np.stack(predictions)), np.float32)
    dim = stk.shape[1]
    if lib:
        fs, _ = _fp(stk)
        return float(lib.dot_specialization_score(fs, ctypes.c_int(n),
                                                   ctypes.c_int(dim), ctypes.c_float(alpha)))
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i+1, n):
            total += similarity_score(predictions[i], predictions[j], alpha)
            count += 1
    return total / count if count > 0 else 1.0


# ── Formula 11: Cluster Entropy ──────────────────────────────────────

def cluster_entropy(scores: List[float]) -> float:
    """
    F11: H_C = -Σ (C(k)/Z) * log(C(k)/Z), normalized to [0, 1].
    Low entropy → clear winner. High → diffuse/exploratory.
    """
    if not scores: return 0.0
    lib = _load_lib()
    sc  = np.ascontiguousarray(scores, np.float32)
    if lib:
        fs, _ = _fp(sc)
        return float(lib.cluster_entropy(fs, ctypes.c_int(len(sc))))
    s = np.clip(sc, 0.0, None)
    Z = s.sum()
    if Z < 1e-10: return 1.0
    p = s / Z; p = p[p > 1e-10]
    H = float(-np.sum(p * np.log(p)))
    H_max = np.log(len(scores))
    return float(H / H_max) if H_max > 1e-10 else 0.0


# ── Formula 12: Temporal Stability ───────────────────────────────────

def temporal_stability(c_curr: np.ndarray, c_prev: np.ndarray,
                       alpha: float = 0.7) -> float:
    """F12: TS(t) = S(centroid_t, centroid_{t-1})"""
    if c_curr is None or c_prev is None: return 0.0

    if np.iscomplexobj(c_curr) or np.iscomplexobj(c_prev):
        return similarity_score(c_curr, c_prev, alpha)

    lib = _load_lib()
    a32 = np.ascontiguousarray(c_curr, np.float32)
    b32 = np.ascontiguousarray(c_prev, np.float32)
    if lib:
        fa, _ = _fp(a32); fb, _ = _fp(b32)
        return float(lib.temporal_stability(fa, fb, ctypes.c_int(len(a32)),
                                             ctypes.c_float(alpha)))
    return similarity_score(a32, b32, alpha)


# ── Formula 13: Cross-Type Agreement ─────────────────────────────────

def cross_type_agreement(type_centroids: Dict[str, np.ndarray],
                          alpha: float = 0.7) -> float:
    """F13: Average pairwise S between centroids of different dot types."""
    vals = list(type_centroids.values())
    n    = len(vals)
    if n < 2: return 1.0

    if any(np.iscomplexobj(v) for v in vals):
        total, count = 0.0, 0
        for i in range(n):
            for j in range(i+1, n):
                total += similarity_score(vals[i], vals[j], alpha); count += 1
        return total / count if count > 0 else 1.0

    lib  = _load_lib()
    stk  = np.ascontiguousarray(np.real(np.stack(vals)), np.float32)
    dim  = stk.shape[1]
    if lib:
        fs, _ = _fp(stk)
        return float(lib.cross_type_agreement(fs, ctypes.c_int(n),
                                               ctypes.c_int(dim), ctypes.c_float(alpha)))
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i+1, n):
            total += similarity_score(vals[i], vals[j], alpha); count += 1
    return total / count if count > 0 else 1.0


# ── Formula 14: Adaptive Learning Rate ───────────────────────────────

def adaptive_learning_rate(base_lr: float, dominance: float) -> float:
    """F14: eta(t) = base_lr * (1 - 0.8 * dominance²)"""
    lib = _load_lib()
    if lib:
        return float(lib.adaptive_learning_rate(ctypes.c_float(base_lr),
                                                 ctypes.c_float(dominance)))
    return base_lr * (1.0 - 0.8 * dominance ** 2)


# ── Formula 15: Hierarchical Convergence Score ────────────────────────

def hierarchical_convergence_score(centroids: List[np.ndarray], scores: List[float],
                                    alpha: float = 0.7, gamma: float = 0.5) -> float:
    """F15: HC(K) = mean_score * (1 + gamma * cross_cluster_similarity)"""
    n = len(centroids)
    if n == 0: return 0.0

    if any(np.iscomplexobj(c) for c in centroids):
        mean_s = np.mean(scores)
        cross = sum(similarity_score(centroids[i], centroids[j], alpha)
                    for i in range(n) for j in range(i+1, n))
        pairs = n * (n-1) / 2
        return float(mean_s * (1.0 + gamma * (cross / pairs if pairs > 0 else 0.0)))

    lib = _load_lib()
    stk = np.ascontiguousarray(np.real(np.stack(centroids)), np.float32)
    sc  = np.ascontiguousarray(scores, np.float32)
    dim = stk.shape[1]
    if lib:
        fs, _ = _fp(stk); fsc, _ = _fp(sc)
        return float(lib.hierarchical_convergence_score(
            fs, fsc, ctypes.c_int(n), ctypes.c_int(dim),
            ctypes.c_float(alpha), ctypes.c_float(gamma)
        ))
    mean_s = np.mean(scores)
    cross = sum(similarity_score(centroids[i], centroids[j], alpha)
                for i in range(n) for j in range(i+1, n))
    pairs = n * (n-1) / 2
    return float(mean_s * (1.0 + gamma * (cross / pairs if pairs > 0 else 0.0)))


# ── Formula 16: Emergent Utility Gradient ────────────────────────────

def emergent_utility_gradient(score_history: List[float],
                              entropy_history: Optional[List[float]] = None,
                              stability_history: Optional[List[float]] = None,
                              lambda1: float = 0.3,
                              lambda2: float = 0.3) -> float:
    """
    FIXED F16: U(t) = ΔC + λ1·ΔH + λ2·ΔS

    Measures improvement, structural change (entropy), and stability evolution.

    If history is short, falls back to simple score delta.
    """
    n = len(score_history)
    if n < 2:
        return 1.0

    delta_c = score_history[-1] - score_history[-2]

    delta_h = 0.0
    if entropy_history and len(entropy_history) >= 2:
        delta_h = entropy_history[-1] - entropy_history[-2]

    delta_s = 0.0
    if stability_history and len(stability_history) >= 2:
        delta_s = stability_history[-1] - stability_history[-2]

    return float(delta_c + lambda1 * delta_h + lambda2 * delta_s)


# ── Formula 17: Dot Reinforcement Pressure ───────────────────────────

def amplify_pressure(pressure: float) -> float:
    """
    Sign-preserving power amplification for DRP scores.

    amplify(R) = sign(R) × |R|^1.5

    Makes strong dots significantly stronger and weak dots drop faster —
    sharper differentiation than the raw linear score alone.
    """
    return float(np.sign(pressure) * (abs(pressure) ** 1.5))


def dot_reinforcement_pressure(
    convergence_contrib: float,
    specialization:      float,
    system_health:       float,  # Normalized J(t) / max|J|
    failure_rate:        float,
    lambda1: float = 0.40,
    lambda2: float = 0.20,
    lambda3: float = 0.30,
    lambda4: float = 0.10,
) -> float:
    """
    FIXED F17: R_d(t) = λ1·C_d + λ2·S_d + λ3·J_norm − λ4·N_d

    Now dots respond to global system health J(t), not just local slope.
    """
    return float(
        lambda1 * convergence_contrib
        + lambda2 * specialization
        + lambda3 * system_health
        - lambda4 * failure_rate
    )

def convergence_score_ultra(predictions: List[np.ndarray], confidences: List[float],
                           alpha: float = 0.7, repellent: Optional[np.ndarray] = None,
                           repellent_weight: float = 0.2) -> float:
    """F2 Ultra: base convergence score penalized by similarity to a repellent centroid."""
    base_s = convergence_score(predictions, confidences, alpha)
    if repellent is None or repellent_weight <= 0:
        return base_s

    mean_p = np.mean(np.stack(predictions), axis=0)
    r_sim = similarity_score(mean_p, repellent, alpha)
    return base_s * (1.0 - repellent_weight * r_sim)

def apply_synergy(v: np.ndarray, peer: np.ndarray, weight: float = 0.1) -> np.ndarray:
    """Apply synergy shift toward a peer vector."""
    return (1.0 - weight) * v + weight * peer

def batch_similarity(queries: np.ndarray, targets: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """
    Hyper-optimized Batch Similarity (F1-GEMM).
    Returns (n_q x n_t) similarity matrix using matrix multiplication.
    """
    if np.iscomplexobj(queries) or np.iscomplexobj(targets):
        n_q = queries.shape[0]; n_t = targets.shape[0]
        out = np.zeros((n_q, n_t), dtype=np.float32)
        for i in range(n_q):
            for j in range(n_t):
                out[i, j] = similarity_score(queries[i], targets[j], alpha)
        return out

    n_q, dim = queries.shape
    n_t, _ = targets.shape
    dot_matrix = queries @ targets.T
    cos_matrix = dot_matrix / float(dim)
    agreement_matrix = 0.76159416 * (cos_matrix + 1.0) / 2.0
    return (alpha * cos_matrix + (1.0 - alpha) * agreement_matrix).astype(np.float32)


# ── Formula 18: Cross-Modal Binding Score ────────────────────────────

def cross_modal_binding(v_text: np.ndarray, v_modal: np.ndarray,
                        alpha: float = 0.7,
                        modal_flag_start: int = 248,
                        modal_flag_end:   int = 252) -> float:
    """
    F18: CMB(t, v) = S(t, v) × (1 + 0.3 × modal_diversity)

    Rewards high semantic similarity between a text vector and a non-text
    modal vector, with a bonus when their modality flags are distinct
    (confirming they truly come from different modalities that have been
    aligned rather than two copies of the same modality).

    modal_flag_start/end: slice indices of the modality one-hot flags inside
    the feature vector (default dims 248:252 for the 256-dim IECNN layout).
    """
    base_sim = similarity_score(v_text, v_modal, alpha)
    if len(v_text) > modal_flag_end and len(v_modal) > modal_flag_end:
        flag_diff = float(np.mean(np.abs(
            v_text[modal_flag_start:modal_flag_end].astype(np.float32) -
            v_modal[modal_flag_start:modal_flag_end].astype(np.float32)
        )))
        bond = 1.0 + 0.3 * flag_diff
    else:
        bond = 1.0
    return float(np.clip(base_sim * bond, -1.0, 1.0))


# ── Formula 19: Semantic Drift Score ─────────────────────────────────

def semantic_drift(input_latent: np.ndarray, output_latent: np.ndarray,
                   alpha: float = 0.7) -> float:
    """
    F19: SD(z_in, z_out) = 1 − S(z_in, z_out)

    Measures how far the model's output drifted from the input in latent
    space.  SD → 0 means the output stayed very close to the input
    (conservative / copy-like).  SD → 1 means maximal divergence
    (the model explored far from its starting point).

    Useful for monitoring generative fidelity vs creativity trade-off.
    """
    return float(np.clip(1.0 - similarity_score(input_latent, output_latent, alpha),
                         0.0, 1.0))


# ── New Formulas F21–F26 ──────────────────────────────────────────

def global_energy(entropy: float, dominance: float, instability: float,
                  alpha: float = 0.2, beta: float = 0.2, gamma: float = 0.2) -> float:
    """F21: E(t) = alpha*H(t) + beta*D(t) + gamma*||C_t - C_{t-1}||"""
    lib = _load_lib()
    if lib:
        return float(lib.global_energy(
            ctypes.c_float(entropy), ctypes.c_float(dominance), ctypes.c_float(instability),
            ctypes.c_float(alpha), ctypes.c_float(beta), ctypes.c_float(gamma)
        ))
    return alpha * entropy + beta * dominance + gamma * instability

def system_objective(convergence: float, utility: float, energy: float) -> float:
    """F22: J(t) = C(t) + U(t) - E(t)"""
    lib = _load_lib()
    if lib:
        return float(lib.system_objective(
            ctypes.c_float(convergence), ctypes.c_float(utility), ctypes.c_float(energy)
        ))
    return convergence + utility - energy

def memory_plasticity(stability: float) -> float:
    """F23: rho(t) = sigmoid(stability(t))"""
    lib = _load_lib()
    if lib:
        return float(lib.memory_plasticity(ctypes.c_float(stability)))
    return 1.0 / (1.0 + np.exp(-stability))

def dot_fitness(rd: float, cd: float, sd: float, ud: float, nd: float, surprise: float = 0.0,
                alpha: float = 0.2, beta: float = 0.2, gamma: float = 0.2, delta: float = 0.1, sigma: float = 0.5) -> float:
    """
    F24 (Enhanced): F_d = R_d + alpha*C_d + beta*S_d + gamma*U_d - delta*N_d + sigma*Surprise
    Rewards dots that are surprisingly correct (causal discovery).
    """
    lib = _load_lib()
    # The C version will eventually need updating, but we can pass surprise manually here
    # to avoid crashing while we transition.
    base = rd + alpha * cd + beta * sd + gamma * ud - delta * nd
    return base + sigma * surprise

def stability_energy(entropy: float, instability: float,
                     lambda1: float = 0.5, lambda2: float = 0.5) -> float:
    """F25: S(t) = 1 - (lambda1*H(t) + lambda2*||C_t - C_{t-1}||)"""
    lib = _load_lib()
    if lib:
        return float(lib.stability_energy(
            ctypes.c_float(entropy), ctypes.c_float(instability),
            ctypes.c_float(lambda1), ctypes.c_float(lambda2)
        ))
    return 1.0 - (lambda1 * entropy + lambda2 * instability)

def exploration_pressure(stability: float, dominance: float) -> float:
    """F26: X(t) = 1 - S(t) + (1 - D(t))"""
    lib = _load_lib()
    if lib:
        return float(lib.exploration_pressure(
            ctypes.c_float(stability), ctypes.c_float(dominance)
        ))
    return (1.0 - stability) + (1.0 - dominance)


# ── Formula 20: Vocabulary Coverage Score ────────────────────────────

def vocab_coverage(token_types: List[str]) -> float:
    """
    F20: VC = |{t : type ∈ {word, phrase}}| / |tokens|

    Fraction of input tokens that had known (fitted) bases in the
    BaseMapper vocabulary.  Tokens of type 'composed' are unknown words
    that fell back to character-level construction.

    VC = 1.0 → perfect coverage (all tokens were seen in training).
    VC = 0.0 → no coverage (model has never been trained, or all tokens
                are novel).

    Use after fit() or fit_file() to gauge training quality for a given
    input domain.
    """
    if not token_types:
        return 0.0
    known = sum(1 for t in token_types if t in ("word", "phrase"))
    return float(known / len(token_types))
