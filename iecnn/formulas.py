import numpy as np
from typing import List


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def agreement_strength(a: np.ndarray, b: np.ndarray) -> float:
    """
    Agreement strength between two predictions.
    Measures how much two predictions reinforce each other
    beyond simple directional similarity.
    High when both vectors are strong and point the same direction.
    """
    dot = float(np.dot(a, b))
    mag = np.linalg.norm(a) * np.linalg.norm(b)
    if mag < 1e-10:
        return 0.0
    norm_dot = dot / mag
    strength = (np.linalg.norm(a) + np.linalg.norm(b)) / 2.0
    ref = np.sqrt(a.shape[0])
    return float(np.tanh(strength / ref) * ((norm_dot + 1.0) / 2.0))


def similarity_score(p_i: np.ndarray, p_j: np.ndarray, alpha: float = 0.7) -> float:
    """
    Formula 1 — Similarity Score.
    S(p_i, p_j) = alpha * cos(p_i, p_j) + (1 - alpha) * A(p_i, p_j)

    Combines structural similarity (cosine) with
    agreement strength to measure prediction closeness.
    """
    cos = cosine_similarity(p_i, p_j)
    ag = agreement_strength(p_i, p_j)
    return alpha * cos + (1.0 - alpha) * ag


def convergence_score(predictions: List[np.ndarray], confidences: List[float], alpha: float = 0.7) -> float:
    """
    Formula 2 — Convergence Score.
    C(k) = (1/|k|^2) * sum_i sum_j S(p_i, p_j) * mean_confidence

    Scores a cluster by its internal cohesion weighted by confidence.
    """
    n = len(predictions)
    if n == 0:
        return 0.0
    if n == 1:
        return float(confidences[0])

    total_sim = 0.0
    for i in range(n):
        for j in range(n):
            total_sim += similarity_score(predictions[i], predictions[j], alpha)

    mean_conf = float(np.mean(confidences))
    return (total_sim / (n * n)) * mean_conf


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Formula 3 — Attention Formula.
    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V

    Applied within AIM to focus on the most relevant parts
    of a prediction or context. Q, K, V are derived from
    BaseMapping representations.
    """
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-10)
    return weights @ V


def aim_transform(prediction: np.ndarray, context: np.ndarray, inversion_fn) -> np.ndarray:
    """
    Formula 4 — AIM Transformation.
    p_hat = Attention(Q, K, Invert(p))

    Applies an inversion to the prediction, then uses
    attention over context to refine the inverted candidate.
    """
    inverted = inversion_fn(prediction)
    if context.ndim == 1:
        context = context.reshape(1, -1)
    if inverted.ndim == 1:
        inverted = inverted.reshape(1, -1)
    Q = inverted
    K = context
    V = context
    refined = attention(Q, K, V)
    return refined.flatten()


def prediction_confidence(p: np.ndarray) -> float:
    """
    Confidence of a prediction: normalized L2 norm.
    Measures how strong/decisive a dot's prediction is.
    """
    dim = p.shape[0]
    return float(np.tanh(np.linalg.norm(p) / np.sqrt(dim)))


def novelty_gain(new_cluster_count: int, total_cluster_count: int) -> float:
    """
    Formula 6 — Novelty Gain.
    NG(t) = |NewClusters(t)| / |TotalClusters(t)|

    Measures how much new information each iteration adds.
    System stops when NG < epsilon (exploration exhausted).
    """
    if total_cluster_count == 0:
        return 0.0
    return new_cluster_count / total_cluster_count


def sampling_temperature_sample(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Formula 7 — Sampling Temperature.
    P(variant v) ∝ exp(score(v) / T)

    Controls diversity of dot outputs. High T = more diverse.
    """
    if temperature < 1e-6:
        temperature = 1e-6
    log_probs = scores / temperature
    log_probs -= np.max(log_probs)
    probs = np.exp(log_probs)
    probs /= probs.sum() + 1e-10
    return probs


def bias_vector_update(b: np.ndarray, w: np.ndarray, learning_rate: float = 0.1) -> np.ndarray:
    """
    Formula 8 — Bias Vector Update.
    b_(t+1) = b_t + eta * (w_t - b_t)

    Shifts dot generation bias toward winning strategies.
    """
    return b + learning_rate * (w - b)


def dominance_score(leading_cluster_score: float, all_cluster_scores: List[float]) -> float:
    """
    Formula 9 — Stability Condition (Convergence Dominance).
    Dominance(k*) = C(k*) / sum_k C(k)

    System halts when the top cluster holds delta fraction
    of total weight (default delta = 0.75).
    """
    total = sum(all_cluster_scores)
    if total < 1e-10:
        return 0.0
    return leading_cluster_score / total
