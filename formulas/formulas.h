#ifndef FORMULAS_H
#define FORMULAS_H

/* ── Formula 1 components ─────────────────────────────────────────── */
float cosine_sim(const float *a, const float *b, int n);
float agreement_str(const float *a, const float *b, int n);
float similarity_score(const float *a, const float *b, int n, float alpha);

/* ── Formula 2: Convergence Score ────────────────────────────────── */
float convergence_score(const float *preds, const float *confs,
                        int n_preds, int dim, float alpha);

/* ── Formula 6: Prediction Confidence ───────────────────────────── */
float prediction_confidence(const float *p, int n);

/* ── Formula 8: Bias Vector Update ──────────────────────────────── */
void bias_vector_update(const float *b, const float *w, int n, float lr, float *out);

/* ── Batch: Pairwise Similarity Matrix ──────────────────────────── */
void pairwise_similarity_matrix(const float *preds, int n, int dim,
                                float alpha, float *out);

/* ── Batch: One vs All similarity ───────────────────────────────── */
void similarity_vs_all(const float *query, const float *targets, int n_targets,
                       int dim, float alpha, float *out);

/* ── Formula 3: Attention (single query row) ─────────────────────── */
void attention_single(const float *Q, const float *K, const float *V,
                      int seq_len, int dim, float *out);

/* ── Formula 9: Dominance Score ─────────────────────────────────── */
float dominance_score(float leading, const float *all_scores, int n);

/* ────────────────────────────────────────────────────────────────── */
/* Extended Formulas (F10–F15)                                        */
/* ────────────────────────────────────────────────────────────────── */

/* Formula 10: Dot Specialization Score
 * Measures how consistently a dot produces similar predictions.
 * High score = specialized/focused dot. Low = generalist.
 * preds: (n_preds x dim) matrix of this dot's recent predictions
 */
float dot_specialization_score(const float *preds, int n_preds, int dim, float alpha);

/* Formula 11: Cluster Entropy
 * H_C = -Σ (C(k)/Z) * log(C(k)/Z),  normalized to [0,1]
 * Low entropy → clear dominant cluster.
 */
float cluster_entropy(const float *scores, int n);

/* Formula 12: Temporal Stability
 * TS(t) = S(centroid_t, centroid_{t-1})
 * How much the top cluster centroid moved between rounds.
 */
float temporal_stability(const float *c_curr, const float *c_prev, int dim, float alpha);

/* Formula 13: Cross-Type Agreement
 * Average pairwise similarity between type-specific centroids.
 * centroids: (n_types x dim), n_types centroids stacked row-major
 */
float cross_type_agreement(const float *centroids, int n_types, int dim, float alpha);

/* Formula 14: Adaptive Learning Rate
 * eta(t) = base_lr * (1 - 0.8 * dominance^2)
 * Slows learning when convergence is near.
 */
float adaptive_learning_rate(float base_lr, float dominance);

/* Formula 15: Hierarchical Convergence Score
 * HC(K) = mean_score * (1 + gamma * cross_cluster_similarity)
 * Rewards macro-clusters whose micro-clusters also agree internally.
 * centroids: (n_micro x dim), scores: (n_micro,)
 */
float hierarchical_convergence_score(const float *centroids, const float *scores,
                                     int n, int dim, float alpha, float gamma);

#endif
