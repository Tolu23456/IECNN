#ifndef CONVERGENCE_H
#define CONVERGENCE_H

/*
 * Convergence Layer — C Implementation
 * Fast pairwise similarity computation and cluster scoring.
 */

/* Compute the full n×n pairwise similarity matrix using Formula 1.
 * preds:  (n x dim) row-major float array of predictions
 * n:      number of predictions
 * dim:    feature dimension
 * alpha:  balance between cosine and agreement strength
 * out:    (n x n) output similarity matrix (row-major)
 */
void compute_similarity_matrix(const float *preds, int n, int dim,
                                float alpha, float *out);

/* Score a single cluster using Formula 2 (Convergence Score).
 * preds:  (n_members x dim) predictions in the cluster
 * confs:  (n_members,) confidence per prediction
 * n:      number of members
 * dim:    feature dimension
 * alpha:  similarity weight
 * returns convergence score
 */
float score_cluster(const float *preds, const float *confs,
                    int n, int dim, float alpha);

/* Compute centroid of n predictions of given dim */
void compute_centroid(const float *preds, int n, int dim, float *out);

/* Sequential greedy clustering (Micro-clustering)
 * preds: (n_preds x dim)
 * n_preds: number of input predictions
 * dim: feature dimension
 * threshold: similarity threshold
 * alpha: formula 1 weight
 * out_cluster_ids: (n_preds,) array to store cluster ID for each prediction
 * returns number of clusters created
 */
int greedy_cluster(const float *preds, int n_preds, int dim,
                   float threshold, float alpha, int *out_cluster_ids);

#endif
