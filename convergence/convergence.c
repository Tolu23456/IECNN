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

/* Helper: compare function for sorting predictions by confidence/norm */
typedef struct {
    int index;
    float norm;
} PredInfo;

int compare_pred_info(const void *a, const void *b) {
    float na = ((PredInfo*)a)->norm;
    float nb = ((PredInfo*)b)->norm;
    return (nb > na) - (nb < na); /* Descending */
}

int greedy_cluster(const float *preds, int n_preds, int dim,
                   float threshold, float alpha, int *out_cluster_ids) {
    if (n_preds <= 0) return 0;

    /* ── Pass 0: Pre-sorting for Order Stability ── */
    /* Use a stable feature for sorting to break order dependency.
       We use the sum of elements as a stable (but fast) key. */
    PredInfo *sorted = (PredInfo *)malloc(n_preds * sizeof(PredInfo));
    for (int i = 0; i < n_preds; i++) {
        sorted[i].index = i;
        float s = 0.0f;
        for (int d = 0; d < dim; d++) s += preds[i * dim + d];
        sorted[i].norm = s;
    }
    qsort(sorted, n_preds, sizeof(PredInfo), compare_pred_info);

    float *centroids = (float *)calloc(n_preds * dim, sizeof(float));
    int n_clusters = 0;

    /* ── Pass 1: Global Sequential Discovery ── */
    for (int k = 0; k < n_preds; k++) {
        int i = sorted[k].index;
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
            memcpy(centroids + n_clusters * dim, p, dim * sizeof(float));
            out_cluster_ids[i] = n_clusters;
            n_clusters++;
        } else {
            out_cluster_ids[i] = best_cid;
        }
    }

    /* ── Pass 2, 3, 4: Refinement (Iterative batch re-assignment) ── */
    /* This converts the greedy result into a stable K-means-like result. */
    for (int pass = 0; pass < 3; pass++) {
        float *new_centroids = (float *)calloc(n_clusters * dim, sizeof(float));
        int *counts = (int *)calloc(n_clusters, sizeof(int));

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
            out_cluster_ids[i] = best_cid;
            if (best_cid >= 0) {
                counts[best_cid]++;
                for (int d = 0; d < dim; d++) new_centroids[best_cid * dim + d] += p[d];
            }
        }

        /* Update centroids from batch assignments */
        for (int c = 0; c < n_clusters; c++) {
            if (counts[c] > 0) {
                normalize_vector(new_centroids + c * dim, dim);
                memcpy(centroids + c * dim, new_centroids + c * dim, dim * sizeof(float));
            }
        }
        free(new_centroids);
        free(counts);
    }

    free(centroids);
    free(sorted);
    return n_clusters;
}
