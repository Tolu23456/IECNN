#include "formulas.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ══════════════════════════════════════════════════════════════════════
   IECNN Formulas — C Implementation
   All performance-critical math for the IECNN pipeline.
   ══════════════════════════════════════════════════════════════════════ */

/* ── Formula 1 helpers ──────────────────────────────────────────────── */

/* Complex-aware helpers (assuming complex64 is two float32s) */
float complex_dot_real(const float *a, const float *b, int n, int is_complex) {
    float dot_r = 0.0f;
    if (is_complex) {
        /* complex dot product real part: Re(a * conj(b)) = a_r*b_r + a_i*b_i */
        for (int i = 0; i < n; i++) {
            dot_r += a[2*i] * b[2*i] + a[2*i+1] * b[2*i+1];
        }
    } else {
        for (int i = 0; i < n; i++) dot_r += a[i] * b[i];
    }
    return dot_r;
}

float complex_norm(const float *a, int n, int is_complex) {
    float norm_sq = 0.0f;
    if (is_complex) {
        for (int i = 0; i < 2*n; i++) norm_sq += a[i] * a[i];
    } else {
        for (int i = 0; i < n; i++) norm_sq += a[i] * a[i];
    }
    return sqrtf(norm_sq);
}

float cosine_sim(const float *a, const float *b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    /* Unrolled loop for speed */
    int i = 0;
    for (; i <= n - 4; i += 4) {
        dot += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3];
        na  += a[i] * a[i] + a[i+1] * a[i+1] + a[i+2] * a[i+2] + a[i+3] * a[i+3];
        nb  += b[i] * b[i] + b[i+1] * b[i+1] + b[i+2] * b[i+2] + b[i+3] * b[i+3];
    }
    for (; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    return denom < 1e-10f ? 0.0f : dot / denom;
}

float agreement_str(const float *a, const float *b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        dot += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3];
        na  += a[i] * a[i] + a[i+1] * a[i+1] + a[i+2] * a[i+2] + a[i+3] * a[i+3];
        nb  += b[i] * b[i] + b[i+1] * b[i+1] + b[i+2] * b[i+2] + b[i+3] * b[i+3];
    }
    for (; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float mag_a = sqrtf(na), mag_b = sqrtf(nb);
    float denom  = mag_a * mag_b;
    float ndot   = denom < 1e-10f ? 0.0f : dot / denom;
    float strength = (mag_a + mag_b) * 0.5f;
    float ref = sqrtf((float)n);
    return tanhf(strength / ref) * ((ndot + 1.0f) * 0.5f);
}

/* Formula 1: S(p_i, p_j) = alpha*cos + (1-alpha)*agreement */
float similarity_score(const float *a, const float *b, int n, float alpha) {
    return alpha * cosine_sim(a, b, n) + (1.0f - alpha) * agreement_str(a, b, n);
}

/* ── Formula 2: Convergence Score ──────────────────────────────────── */
float convergence_score(const float *preds, const float *confs,
                        int n_preds, int dim, float alpha) {
    if (n_preds == 0) return 0.0f;
    if (n_preds == 1) return confs ? confs[0] : 0.5f;

    float mean_conf = 0.0f;
    if (confs) {
        for (int i = 0; i < n_preds; i++) mean_conf += confs[i];
        mean_conf /= (float)n_preds;
    } else {
        mean_conf = 0.5f;
    }

    float total_sim = 0.0f;
    for (int i = 0; i < n_preds; i++) {
        for (int j = 0; j < n_preds; j++) {
            total_sim += similarity_score(preds + i*dim, preds + j*dim, dim, alpha);
        }
    }
    return (total_sim / ((float)n_preds * (float)n_preds)) * mean_conf;
}

/* ── Formula 6: Prediction Confidence ──────────────────────────────── */
float prediction_confidence(const float *p, int n) {
    float norm = 0.0f;
    for (int i = 0; i < n; i++) norm += p[i] * p[i];
    norm = sqrtf(norm);
    float ref = sqrtf((float)n);
    return tanhf(norm / ref);
}

/* ── Formula 8: Bias Vector Update ─────────────────────────────────── */
void bias_vector_update(const float *b, const float *w, int n, float lr, float *out) {
    for (int i = 0; i < n; i++) out[i] = b[i] + lr * (w[i] - b[i]);
}

/* ── Formula 9: Dominance Score ─────────────────────────────────────── */
float dominance_score(float leading, const float *all_scores, int n) {
    float total = 0.0f;
    for (int i = 0; i < n; i++) total += all_scores[i];
    return total < 1e-10f ? 0.0f : leading / total;
}

/* ── Pairwise Similarity Matrix ──────────────────────────────────────── */
void pairwise_similarity_matrix(const float *preds, int n, int dim,
                                float alpha, float *out) {
    for (int i = 0; i < n; i++) {
        out[i*n + i] = 1.0f;
        for (int j = i+1; j < n; j++) {
            float s = similarity_score(preds + i*dim, preds + j*dim, dim, alpha);
            out[i*n + j] = s;
            out[j*n + i] = s;
        }
    }
}

/* ── Similarity vs All ───────────────────────────────────────────────── */
void similarity_vs_all(const float *query, const float *targets, int n_targets,
                       int dim, float alpha, float *out) {
    for (int i = 0; i < n_targets; i++)
        out[i] = similarity_score(query, targets + i*dim, dim, alpha);
}

/* ── Formula 3: Attention (single query row) ─────────────────────────── */
void attention_single(const float *Q, const float *K, const float *V,
                      int seq_len, int dim, float *out) {
    float scale = 1.0f / sqrtf((float)dim);
    float *scores = (float *)malloc(seq_len * sizeof(float));

    float max_s = -1e30f;
    for (int i = 0; i < seq_len; i++) {
        float s = 0.0f;
        for (int d = 0; d < dim; d++) s += Q[d] * K[i*dim + d];
        scores[i] = s * scale;
        if (scores[i] > max_s) max_s = scores[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) { scores[i] = expf(scores[i]-max_s); sum += scores[i]; }
    for (int i = 0; i < seq_len; i++) scores[i] /= (sum + 1e-10f);

    for (int d = 0; d < dim; d++) {
        float v = 0.0f;
        for (int i = 0; i < seq_len; i++) v += scores[i] * V[i*dim + d];
        out[d] = v;
    }
    free(scores);
}

/* ══════════════════════════════════════════════════════════════════════
   Extended Formulas: F10–F15
   ══════════════════════════════════════════════════════════════════════ */

/* Formula 10: Dot Specialization Score
 * Mean pairwise similarity of a dot's recent predictions.
 * High = consistent (specialized). Low = diverse (generalist).
 */
float dot_specialization_score(const float *preds, int n_preds, int dim, float alpha) {
    if (n_preds <= 1) return 1.0f;
    int pairs = 0;
    float sum = 0.0f;
    for (int i = 0; i < n_preds; i++) {
        for (int j = i+1; j < n_preds; j++) {
            sum += similarity_score(preds + i*dim, preds + j*dim, dim, alpha);
            pairs++;
        }
    }
    return pairs > 0 ? sum / (float)pairs : 1.0f;
}

/* Formula 11: Cluster Entropy (normalized to [0, 1])
 * H_C = -Σ p_k * log(p_k),  p_k = C(k) / Z
 * Normalized: divide by log(n) so result is in [0, 1].
 */
float cluster_entropy(const float *scores, int n) {
    if (n <= 0) return 0.0f;
    if (n == 1) return 0.0f;

    float Z = 0.0f;
    for (int i = 0; i < n; i++) Z += scores[i];
    if (Z < 1e-10f) return 1.0f;  /* maximum uncertainty */

    float H = 0.0f;
    for (int i = 0; i < n; i++) {
        float p = scores[i] / Z;
        if (p > 1e-10f) H -= p * logf(p);
    }
    float H_max = logf((float)n);
    return H_max > 1e-10f ? H / H_max : 0.0f;
}

/* Formula 12: Temporal Stability
 * TS(t) = S(centroid_t, centroid_{t-1})
 */
float temporal_stability(const float *c_curr, const float *c_prev, int dim, float alpha) {
    return similarity_score(c_curr, c_prev, dim, alpha);
}

/* Formula 13: Cross-Type Agreement
 * Average pairwise S between type-specific centroids.
 * centroids: (n_types x dim) row-major
 */
float cross_type_agreement(const float *centroids, int n_types, int dim, float alpha) {
    if (n_types < 2) return 1.0f;
    float sum = 0.0f;
    int pairs = 0;
    for (int i = 0; i < n_types; i++) {
        for (int j = i+1; j < n_types; j++) {
            sum += similarity_score(centroids + i*dim, centroids + j*dim, dim, alpha);
            pairs++;
        }
    }
    return pairs > 0 ? sum / (float)pairs : 1.0f;
}

/* Formula 14: Adaptive Learning Rate
 * eta(t) = base_lr * (1 - 0.8 * dominance^2)
 * Slows down when one cluster dominates (near convergence).
 */
float adaptive_learning_rate(float base_lr, float dominance) {
    float d2 = dominance * dominance;
    return base_lr * (1.0f - 0.8f * d2);
}

/* Formula 15: Hierarchical Convergence Score
 * HC(K) = mean_score * (1 + gamma * cross_cluster_similarity)
 */
float hierarchical_convergence_score(const float *centroids, const float *scores,
                                     int n, int dim, float alpha, float gamma) {
    if (n <= 0) return 0.0f;

    float mean_score = 0.0f;
    for (int i = 0; i < n; i++) mean_score += scores[i];
    mean_score /= (float)n;

    float cross = 0.0f;
    int pairs = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            cross += similarity_score(centroids + i*dim, centroids + j*dim, dim, alpha);
            pairs++;
        }
    }
    float cross_sim = pairs > 0 ? cross / (float)pairs : 0.0f;
    return mean_score * (1.0f + gamma * cross_sim);
}

/* Formula 2 Ultra: Convergence with Dynamic Repellent
 * This version penalizes clusters that are too similar to a 'repellent' centroid,
 * which helps in increasing the discriminative gap.
 */
float convergence_score_ultra(const float *preds, const float *confs,
                             int n_preds, int dim, float alpha,
                             const float *repellent, float repellent_weight) {
    float base_c = convergence_score(preds, confs, n_preds, dim, alpha);
    if (!repellent || repellent_weight <= 0.0f) return base_c;

    float *mean_p = (float *)calloc(dim, sizeof(float));
    for (int i = 0; i < n_preds; i++) {
        for (int d = 0; d < dim; d++) mean_p[d] += preds[i*dim + d] / (float)n_preds;
    }

    float r_sim = similarity_score(mean_p, repellent, dim, alpha);
    free(mean_p);

    // Repellent penalty: score drops as it gets closer to repellent
    return base_c * (1.0f - repellent_weight * r_sim);
}

/* ── New Formulas F21–F26 ────────────────────────────────────────── */

float global_energy(float entropy, float dominance, float instability,
                    float alpha, float beta, float gamma) {
    return alpha * entropy + beta * dominance + gamma * instability;
}

float system_objective(float convergence, float utility, float energy) {
    return convergence + utility - energy;
}

float memory_plasticity(float stability) {
    return 1.0f / (1.0f + expf(-stability));
}

float dot_fitness(float rd, float cd, float sd, float ud, float nd,
                  float alpha, float beta, float gamma, float delta) {
    return rd + alpha * cd + beta * sd + gamma * ud - delta * nd;
}

float stability_energy(float entropy, float instability, float lambda1, float lambda2) {
    return 1.0f - (lambda1 * entropy + lambda2 * instability);
}

float exploration_pressure(float stability, float dominance) {
    return (1.0f - stability) + (1.0f - dominance);
}

/* Fast Batch Similarity Score
 * Efficiently computes similarity between a batch of queries and a batch of targets.
 */
void batch_similarity_fast(const float *queries, int n_q, const float *targets, int n_t,
                          int dim, float alpha, float *out_matrix) {
    for (int i = 0; i < n_q; i++) {
        for (int j = 0; j < n_t; j++) {
            out_matrix[i * n_t + j] = similarity_score(queries + i * dim, targets + j * dim, dim, alpha);
        }
    }
}
