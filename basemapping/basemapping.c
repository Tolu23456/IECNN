#include "basemapping.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*
 * BaseMapping — C Implementation
 * Embedding composition and pooling operations.
 */

/* ── Normalize vector in-place ──────────────────────────────────── */
void normalize_vector(float *v, int n) {
    float norm = 0.0f;
    for (int i = 0; i < n; i++) norm += v[i] * v[i];
    norm = sqrtf(norm);
    if (norm > 1e-10f) {
        for (int i = 0; i < n; i++) v[i] /= norm;
    }
}

/* ── Compose word embedding from character embeddings ───────────── */
/* Weighted sum of character embeddings, then normalize.
 * This is how unknown words are represented in BaseMapping:
 * one row per word, composed from pre-seeded a-z embeddings. */
void compose_from_chars(const float *char_embeds, const float *weights,
                        int n_chars, int embed_dim, float *out) {
    memset(out, 0, embed_dim * sizeof(float));
    float total_weight = 0.0f;
    for (int i = 0; i < n_chars; i++) total_weight += weights[i];
    if (total_weight < 1e-10f) total_weight = 1.0f;

    for (int i = 0; i < n_chars; i++) {
        float w = weights[i] / total_weight;
        for (int d = 0; d < embed_dim; d++) {
            out[d] += w * char_embeds[i * embed_dim + d];
        }
    }
    normalize_vector(out, embed_dim);
}

/* ── Sinusoidal position encoding ───────────────────────────────── */
void sinusoidal_position_enc(int pos, int total, int dim, float *out) {
    float rel = (total > 1) ? (float)pos / (float)(total - 1) : 0.0f;
    for (int k = 0; k < dim / 2; k++) {
        float freq = (float)(k + 1);
        out[2 * k]     = sinf(rel * 3.14159265f * freq);
        out[2 * k + 1] = cosf(rel * 3.14159265f * freq);
    }
}

/* ── Mean pooling ────────────────────────────────────────────────── */
void mean_pool(const float *matrix, int n_rows, int dim, float *out) {
    memset(out, 0, dim * sizeof(float));
    for (int i = 0; i < n_rows; i++) {
        for (int d = 0; d < dim; d++) {
            out[d] += matrix[i * dim + d];
        }
    }
    float inv_n = 1.0f / (float)n_rows;
    for (int d = 0; d < dim; d++) out[d] *= inv_n;
}

/* ── Attention-weighted pooling ──────────────────────────────────── */
void attention_pool(const float *matrix, const float *query,
                    int n_rows, int dim, float sharpness, float *out) {
    float *scores = (float *)malloc(n_rows * sizeof(float));

    /* dot product of each row with query */
    float max_s = -1e30f;
    for (int i = 0; i < n_rows; i++) {
        float s = 0.0f;
        for (int d = 0; d < dim; d++) s += matrix[i * dim + d] * query[d];
        scores[i] = s * sharpness;
        if (scores[i] > max_s) max_s = scores[i];
    }

    /* softmax */
    float sum = 0.0f;
    for (int i = 0; i < n_rows; i++) {
        scores[i] = expf(scores[i] - max_s);
        sum += scores[i];
    }
    for (int i = 0; i < n_rows; i++) scores[i] /= (sum + 1e-10f);

    /* weighted sum */
    memset(out, 0, dim * sizeof(float));
    for (int i = 0; i < n_rows; i++) {
        for (int d = 0; d < dim; d++) {
            out[d] += scores[i] * matrix[i * dim + d];
        }
    }

    free(scores);
}
