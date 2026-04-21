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

def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    here    = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, "formulas_c.so")
    if os.path.exists(so_path):
        try:
            lib = ctypes.CDLL(so_path)
            # F1 helpers
            lib.cosine_sim.restype               = ctypes.c_float
            lib.agreement_str.restype            = ctypes.c_float
            lib.similarity_score.restype         = ctypes.c_float
            # F2
            lib.convergence_score.restype        = ctypes.c_float
            # F6
            lib.prediction_confidence.restype    = ctypes.c_float
            # F8
            lib.bias_vector_update.restype       = None
            # F9
            lib.dominance_score.restype          = ctypes.c_float
            # Batch ops
            lib.pairwise_similarity_matrix.restype = None
            lib.similarity_vs_all.restype        = None
            lib.attention_single.restype         = None
            # F10–F15
            lib.dot_specialization_score.restype = ctypes.c_float
            lib.cluster_entropy.restype          = ctypes.c_float
            lib.temporal_stability.restype       = ctypes.c_float
            lib.cross_type_agreement.restype     = ctypes.c_float
            lib.adaptive_learning_rate.restype   = ctypes.c_float
            lib.hierarchical_convergence_score.restype = ctypes.c_float
            _lib = lib
        except Exception:
            _lib = None
    return _lib


def _fp(arr: np.ndarray):
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), a


# ── Formula 1 helpers ────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
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
    lib = _load_lib()
    a32 = np.ascontiguousarray(a, np.float32)
    b32 = np.ascontiguousarray(b, np.float32)
    if lib:
        fa, _ = _fp(a32); fb, _ = _fp(b32)
        return float(lib.similarity_score(fa, fb, ctypes.c_int(len(a32)), ctypes.c_float(alpha)))
    return alpha * cosine_similarity(a32, b32) + (1.0 - alpha) * agreement_strength(a32, b32)


# ── Formula 2: Convergence Score ────────────────────────────────────

def convergence_score(predictions: List[np.ndarray], confidences: List[float],
                      alpha: float = 0.7) -> float:
    """C(k) = (1/|k|²) Σ S(p_i, p_j) · mean_confidence"""
    n = len(predictions)
    if n == 0: return 0.0
    if n == 1: return float(confidences[0]) if confidences else 0.5
    lib  = _load_lib()
    stk  = np.ascontiguousarray(np.stack(predictions), np.float32)
    cfx  = np.ascontiguousarray(confidences, np.float32)
    dim  = stk.shape[1]
    if lib:
        fs, _s = _fp(stk); fc, _c = _fp(cfx)
        return float(lib.convergence_score(fs, fc, ctypes.c_int(n),
                                           ctypes.c_int(dim), ctypes.c_float(alpha)))
    total = sum(similarity_score(predictions[i], predictions[j], alpha)
                for i in range(n) for j in range(n))
    return (total / (n * n)) * float(np.mean(confidences))


# ── Formula 6: Prediction Confidence ────────────────────────────────

def prediction_confidence(p: np.ndarray) -> float:
    """tanh(||p|| / sqrt(dim)) — normalized L2 norm as confidence."""
    lib = _load_lib()
    p32 = np.ascontiguousarray(p, np.float32)
    if lib:
        fp_, _ = _fp(p32)
        return float(lib.prediction_confidence(fp_, ctypes.c_int(len(p32))))
    return float(np.tanh(np.linalg.norm(p32) / np.sqrt(len(p32))))


# ── Formula 3: Attention ─────────────────────────────────────────────

def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V"""
    lib = _load_lib()
    if Q.ndim == 1: Q = Q.reshape(1, -1)
    if K.ndim == 1: K = K.reshape(1, -1)
    if V.ndim == 1: V = V.reshape(1, -1)
    seq_len, dim = K.shape
    if Q.shape[0] == 1 and lib:
        Q32 = np.ascontiguousarray(Q[0], np.float32)
        K32 = np.ascontiguousarray(K, np.float32)
        V32 = np.ascontiguousarray(V, np.float32)
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
    lib = _load_lib()
    stk = np.ascontiguousarray(np.stack(predictions), np.float32)
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
    lib  = _load_lib()
    stk  = np.ascontiguousarray(np.stack(vals), np.float32)
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
    lib = _load_lib()
    stk = np.ascontiguousarray(np.stack(centroids), np.float32)
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

def emergent_utility_gradient(score_history: List[float]) -> float:
    """
    F16: U(t) = E[C_{t+1}(p)] - C_t(p)

    Estimates the expected convergence gain in the next round by
    extrapolating the recent trend in top-cluster scores.

    Positive  → structure still improving; keep iterating.
    Near-zero / negative → no future utility gain expected; can stop.

    With 2 rounds: U = C_t - C_{t-1}  (simple delta)
    With 3+ rounds: U = 0.7*(C_t-C_{t-1}) + 0.3*(C_{t-1}-C_{t-2})
                    (recency-weighted to dampen transient spikes)
    """
    n = len(score_history)
    if n < 2:
        return 1.0  # insufficient history — assume gain is still possible
    delta = score_history[-1] - score_history[-2]
    if n >= 3:
        delta2 = score_history[-2] - score_history[-3]
        delta = 0.7 * delta + 0.3 * delta2
    return float(delta)


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
    eug:                 float,
    delta_u:             float,
    failure_rate:        float,
    lambda1: float = 0.40,
    lambda2: float = 0.20,
    lambda3: float = 0.30,
    beta:    float = 0.50,
    lambda4: float = 0.10,
) -> float:
    """
    F17: R_d(t) = λ1·C_d + λ2·S_d + λ3·U_norm·(1 + β·ΔU_norm) − λ4·N_d

    C_d       = convergence contribution (effectiveness score of dot d)
    S_d       = specialization score (consistency of predictions)
    U_norm    = tanh(eug × 5)  — normalized Emergent Utility Gradient (F16)
    ΔU_norm   = tanh(delta_u × 5) — normalized utility acceleration
    N_d       = failure_rate = 1 − effectiveness (penalty for low-quality dots)

    Positive pressure → dot is rewarded (kept, reproduced)
    Low pressure → dot decays toward removal or mutation
    """
    u_norm  = float(np.tanh(eug     * 5.0))
    du_norm = float(np.tanh(delta_u * 5.0))
    utility_term = u_norm * (1.0 + beta * du_norm)
    return float(
        lambda1 * convergence_contrib
        + lambda2 * specialization
        + lambda3 * utility_term
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
    """Batch similarity score: returns (n_q x n_t) similarity matrix."""
    lib = _load_lib()
    n_q, dim = queries.shape
    n_t, _ = targets.shape
    out = np.zeros((n_q, n_t), dtype=np.float32)
    if lib:
        import ctypes
        lib.batch_similarity_fast(_fp(queries)[0], ctypes.c_int(n_q), _fp(targets)[0],
                                 ctypes.c_int(n_t), ctypes.c_int(dim),
                                 ctypes.c_float(alpha), _fp(out)[0])
        return out
    # Fallback
    for i in range(n_q):
        for j in range(n_t):
            out[i, j] = similarity_score(queries[i], targets[j], alpha)
    return out
