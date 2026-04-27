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

/* Optimized Greedy Clustering (O(N*C) where C is num clusters, vs O(N^2) naive) */
int greedy_cluster(const float *preds, int n, int dim, float alpha, float threshold, int *assign) {
    if (n <= 0) return 0;

    float *centroids = (float *)malloc(n * dim * sizeof(float));
    int *cluster_sizes = (int *)calloc(n, sizeof(int));
    int num_clusters = 0;

    for (int i = 0; i < n; i++) {
        int best_cid = -1;
        float best_sim = threshold;

        for (int c = 0; c < num_clusters; c++) {
            float s = similarity_score(preds + i * dim, centroids + c * dim, dim, alpha);
            if (s > best_sim) {
                best_sim = s;
                best_cid = c;
            }
        }

        if (best_cid == -1) {
            /* Create new cluster */
            memcpy(centroids + num_clusters * dim, preds + i * dim, dim * sizeof(float));
            cluster_sizes[num_clusters] = 1;
            assign[i] = num_clusters;
            num_clusters++;
        } else {
            /* Join existing cluster and update centroid (EMA-style or mean) */
            int size = cluster_sizes[best_cid];
            for (int d = 0; d < dim; d++) {
                centroids[best_cid * dim + d] = (centroids[best_cid * dim + d] * size + preds[i * dim + d]) / (size + 1);
            }
            cluster_sizes[best_cid]++;
            assign[i] = best_cid;
        }
    }

    free(centroids);
    free(cluster_sizes);
    return num_clusters;
}
