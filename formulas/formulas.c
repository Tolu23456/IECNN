#include "formulas.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*
 * IECNN Formulas — C Implementation
 */

/* ── Formula 1 helpers ──────────────────────────────────────────── */

float cosine_sim(const float *a, const float *b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    return denom < 1e-10f ? 0.0f : dot / denom;
}

float agreement_str(const float *a, const float *b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float mag_a = sqrtf(na);
    float mag_b = sqrtf(nb);
    float denom = mag_a * mag_b;
    float norm_dot = denom < 1e-10f ? 0.0f : dot / denom;
    float strength = (mag_a + mag_b) * 0.5f;
    float ref = sqrtf((float)n);
    float tanh_s = tanhf(strength / ref);
    return tanh_s * ((norm_dot + 1.0f) * 0.5f);
}

/* Formula 1: S(p_i, p_j) = alpha * cos(p_i,p_j) + (1-alpha) * A(p_i,p_j) */
float similarity_score(const float *a, const float *b, int n, float alpha) {
    return alpha * cosine_sim(a, b, n) + (1.0f - alpha) * agreement_str(a, b, n);
}

/* ── Formula 6: Prediction Confidence ──────────────────────────── */

float prediction_confidence(const float *p, int n) {
    float norm = 0.0f;
    for (int i = 0; i < n; i++) norm += p[i] * p[i];
    norm = sqrtf(norm);
    float ref = sqrtf((float)n);
    return tanhf(norm / ref);
}

/* ── Formula 8: Bias Vector Update ─────────────────────────────── */

void bias_vector_update(const float *b, const float *w, int n, float lr, float *out) {
    for (int i = 0; i < n; i++) {
        out[i] = b[i] + lr * (w[i] - b[i]);
    }
}

/* ── Formula 2: Convergence Score ──────────────────────────────── */

float convergence_score(const float *preds, const float *confs,
                        int n_preds, int dim, float alpha) {
    if (n_preds == 0) return 0.0f;
    if (n_preds == 1) return confs[0];

    float total_sim = 0.0f;
    float mean_conf = 0.0f;

    for (int i = 0; i < n_preds; i++) mean_conf += confs[i];
    mean_conf /= (float)n_preds;

    for (int i = 0; i < n_preds; i++) {
        for (int j = 0; j < n_preds; j++) {
            const float *pi = preds + i * dim;
            const float *pj = preds + j * dim;
            total_sim += similarity_score(pi, pj, dim, alpha);
        }
    }

    return (total_sim / ((float)n_preds * (float)n_preds)) * mean_conf;
}

/* ── Batch: Pairwise Similarity Matrix ──────────────────────────── */

void pairwise_similarity_matrix(const float *preds, int n, int dim,
                                float alpha, float *out) {
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            const float *pi = preds + i * dim;
            const float *pj = preds + j * dim;
            float s = similarity_score(pi, pj, dim, alpha);
            out[i * n + j] = s;
            out[j * n + i] = s;
        }
    }
}

/* ── Similarity of one vector vs all targets ────────────────────── */

void similarity_vs_all(const float *query, const float *targets, int n_targets,
                       int dim, float alpha, float *out) {
    for (int i = 0; i < n_targets; i++) {
        out[i] = similarity_score(query, targets + i * dim, dim, alpha);
    }
}

/* ── Attention (single query row) ───────────────────────────────── */

void attention_single(const float *Q, const float *K, const float *V,
                      int seq_len, int dim, float *out) {
    float scale = 1.0f / sqrtf((float)dim);
    float *scores = (float *)malloc(seq_len * sizeof(float));

    /* scores = Q @ K^T * scale */
    float max_s = -1e30f;
    for (int i = 0; i < seq_len; i++) {
        float s = 0.0f;
        for (int d = 0; d < dim; d++) s += Q[d] * K[i * dim + d];
        scores[i] = s * scale;
        if (scores[i] > max_s) max_s = scores[i];
    }

    /* softmax */
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        scores[i] = expf(scores[i] - max_s);
        sum += scores[i];
    }
    for (int i = 0; i < seq_len; i++) scores[i] /= (sum + 1e-10f);

    /* weighted sum of V */
    for (int d = 0; d < dim; d++) {
        float v = 0.0f;
        for (int i = 0; i < seq_len; i++) v += scores[i] * V[i * dim + d];
        out[d] = v;
    }

    free(scores);
}
