#include "convergence.h"
#include "../formulas/formulas.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Optimized Greedy Clustering with Linear Similarity Trick */
int greedy_cluster(const float *preds, int n, int dim, float alpha, float threshold, int *assign) {
    if (n <= 0) return 0;

    float *centroids = (float *)malloc(n * dim * sizeof(float));
    int *cluster_sizes = (int *)calloc(n, sizeof(int));
    int num_clusters = 0;

    float inv_dim = 1.0f / (float)dim;
    float tanh1 = 0.76159416f;

    for (int i = 0; i < n; i++) {
        int best_cid = -1;
        float best_sim = threshold;
        const float *p = preds + i * dim;

        for (int c = 0; c < num_clusters; c++) {
            const float *cent = centroids + c * dim;
            float dot = 0.0f;
            for (int d = 0; d < dim; d++) dot += p[d] * cent[d];

            float cos = dot * inv_dim;
            float agreement = tanh1 * (cos + 1.0f) * 0.5f;
            float s = alpha * cos + (1.0f - alpha) * agreement;

            if (s > best_sim) {
                best_sim = s;
                best_cid = c;
            }
        }

        if (best_cid == -1) {
            memcpy(centroids + num_clusters * dim, p, dim * sizeof(float));
            cluster_sizes[num_clusters] = 1;
            assign[i] = num_clusters;
            num_clusters++;
        } else {
            int size = cluster_sizes[best_cid];
            float *cent = centroids + best_cid * dim;
            float f1 = (float)size / (float)(size + 1);
            float f2 = 1.0f / (float)(size + 1);
            for (int d = 0; d < dim; d++) {
                cent[d] = cent[d] * f1 + p[d] * f2;
            }
            cluster_sizes[best_cid]++;
            assign[i] = best_cid;
        }
    }

    free(centroids);
    free(cluster_sizes);
    return num_clusters;
}

void compute_similarity_matrix(const float *preds, int n, int dim, float alpha, float *out) {
    pairwise_similarity_matrix(preds, n, dim, alpha, out);
}

float score_cluster(const float *preds, const float *confs, int n, int dim, float alpha) {
    return convergence_score(preds, confs, n, dim, alpha);
}

void compute_centroid(const float *preds, int n, int dim, float *out) {
    memset(out, 0, dim * sizeof(float));
    float inv_n = 1.0f / (float)n;
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < dim; d++) out[d] += preds[i * dim + d] * inv_n;
    }
}

float score_cluster_ultra(const float *preds, const float *confs, int n, int dim, float alpha, const float *repellent, float repellent_weight) {
    return convergence_score_ultra(preds, confs, n, dim, alpha, repellent, repellent_weight);
}
