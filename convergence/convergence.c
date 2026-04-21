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

int greedy_cluster(const float *preds, int n_preds, int dim,
                   float threshold, float alpha, int *out_cluster_ids) {
    if (n_preds <= 0) return 0;

    float *centroids = (float *)calloc(n_preds * dim, sizeof(float));
    int *counts = (int *)calloc(n_preds, sizeof(int));
    int n_clusters = 0;

    for (int i = 0; i < n_preds; i++) {
        const float *p = preds + i * dim;
        int best_cid = -1;
        float best_sim = threshold;

        for (int c = 0; c < n_clusters; c++) {
            float s = similarity_score(p, centroids + c * dim, dim, alpha);
            if (s > best_sim) {
                best_sim = s;
                best_cid = c;
            }
        }

        if (best_cid == -1) {
            // New cluster
            memcpy(centroids + n_clusters * dim, p, dim * sizeof(float));
            counts[n_clusters] = 1;
            out_cluster_ids[i] = n_clusters;
            n_clusters++;
        } else {
            // Join existing
            float *cent = centroids + best_cid * dim;
            int n_old = counts[best_cid];
            for (int d = 0; d < dim; d++) {
                cent[d] = (cent[d] * n_old + p[d]) / (float)(n_old + 1);
            }
            normalize_vector(cent, dim);
            counts[best_cid]++;
            out_cluster_ids[i] = best_cid;
        }
    }

    free(centroids);
    free(counts);
    return n_clusters;
}
