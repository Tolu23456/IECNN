#ifndef FORMULAS_H
#define FORMULAS_H

/*
 * IECNN Custom Formulas — C Implementation
 * Performance-critical math operations for the IECNN pipeline.
 */

/* Formula 1 helpers */
float cosine_sim(const float *a, const float *b, int n);
float agreement_str(const float *a, const float *b, int n);
float similarity_score(const float *a, const float *b, int n, float alpha);

/* Formula 2: Convergence Score for a cluster */
float convergence_score(const float *preds, const float *confs, int n_preds, int dim, float alpha);

/* Formula 6: Confidence of a single prediction */
float prediction_confidence(const float *p, int n);

/* Formula 8: Bias vector update */
void bias_vector_update(const float *b, const float *w, int n, float lr, float *out);

/* Batch: compute full n×n pairwise similarity matrix */
void pairwise_similarity_matrix(const float *preds, int n, int dim, float alpha, float *out);

/* Batch: compute similarity of one vector against all others */
void similarity_vs_all(const float *query, const float *targets, int n_targets, int dim, float alpha, float *out);

/* Attention: softmax(QK^T / sqrt(d_k)) * V for single query row */
void attention_single(const float *Q, const float *K, const float *V,
                      int seq_len, int dim, float *out);

#endif
