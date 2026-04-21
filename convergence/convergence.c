#include "convergence.h"
#include "../formulas/formulas.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*
 * Convergence Layer — C Implementation
 * Fast similarity matrix and cluster scoring.
 */

/* ── Pairwise similarity matrix ──────────────────────────────────── */
void compute_similarity_matrix(const float *preds, int n, int dim,
                                float alpha, float *out) {
    pairwise_similarity_matrix(preds, n, dim, alpha, out);
}

/* ── Score a cluster (Formula 2) ─────────────────────────────────── */
float score_cluster(const float *preds, const float *confs,
                    int n, int dim, float alpha) {
    return convergence_score(preds, confs, n, dim, alpha);
}

/* ── Compute centroid ────────────────────────────────────────────── */
void compute_centroid(const float *preds, int n, int dim, float *out) {
    memset(out, 0, dim * sizeof(float));
    float inv_n = 1.0f / (float)n;
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < dim; d++) {
            out[d] += preds[i * dim + d] * inv_n;
        }
    }
}

/* Ultra scoring interface */
float score_cluster_ultra(const float *preds, const float *confs,
                         int n, int dim, float alpha,
                         const float *repellent, float repellent_weight) {
    return convergence_score_ultra(preds, confs, n, dim, alpha, repellent, repellent_weight);
}
