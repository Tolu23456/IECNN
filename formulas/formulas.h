#ifndef FORMULAS_H
#define FORMULAS_H

/* ── Formula 1 components ─────────────────────────────────────────── */
float cosine_sim(const float *a, const float *b, int n);
float agreement_str(const float *a, const float *b, int n);
float similarity_score(const float *a, const float *b, int n, float alpha);

/* Normalize a vector in-place to unit length */
void normalize_vector(float *v, int n);

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

/* Formula 21: Global Energy Function
 * E(t) = alpha*H(t) + beta*D(t) + gamma*||C_t - C_{t-1}||
 */
float global_energy(float entropy, float dominance, float instability,
                    float alpha, float beta, float gamma);

/* Formula 22: System Objective (Master Function)
 * J(t) = C(t) + U(t) - E(t)
 */
float system_objective(float convergence, float utility, float energy);

/* Formula 23: Memory Plasticity (rho)
 * rho(t) = sigmoid(stability(t))
 * rho = 1 / (1 + exp(-stability))
 */
float memory_plasticity(float stability);

/* Formula 24: Dot Fitness Function
 * F_d = R_d + alpha*C_d + beta*S_d + gamma*U_d - delta*N_d
 */
float dot_fitness(float rd, float cd, float sd, float ud, float nd,
                  float alpha, float beta, float gamma, float delta);

/* Formula 25: Stability Energy
 * S(t) = 1 - (lambda1*H(t) + lambda2*||C_t - C_{t-1}||)
 */
float stability_energy(float entropy, float instability, float lambda1, float lambda2);

/* Formula 26: Exploration Pressure
 * X(t) = 1 - S(t) + (1 - D(t))
 */
float exploration_pressure(float stability, float dominance);

/* ── Cognition Layer Formulas F27–F35 ─────────────────────────── */

/* Formula 29: Reasoning Depth
 * R = log(1 + ||CS||) * S
 */
float reasoning_depth(float cs_norm, float stability);

/* Formula 30: Abstraction Gradient
 * AG = H - C
 */
float abstraction_gradient(float entropy, float convergence);

/* Formula 32: Planning Horizon
 * P = (S / (1 + H)) * R
 */
float planning_horizon(float stability, float entropy, float reasoning);

/* Formula 33: Goal Stability
 * G = C / (1 + |D|)
 */
float goal_stability(float convergence, float dominance);

/* Formula 35: Self-Model Update
 * SM_new = (1 - rho) * SM_old + rho * CS
 */
void self_model_update(float *sm, const float *cs, float rho, int n);

/* ── World & Planning Formulas F36–F45 ────────────────────────── */

/* Formula 37: World Update Function
 * W_new = lambda * W_old + (1 - lambda) * dO
 */
void world_update(float *w, const float *do_obs, float lambda, int n);

/* Formula 41: Plan Evaluation Function
 * V = Σ (gamma^k * J_k)
 */
float plan_evaluation(const float *j_scores, float gamma, int k_steps);

/* Formula 44: Memory Retrieval Attention
 * R_m = softmax(CS * M_long)
 */
void memory_retrieval_attention(const float *cs, const float *m_long,
                                int n_mem, int dim, float *out_weights);

/* Formula 45: Experience Consolidation
 * M_new = M_old + eta * (W - W_pred)
 */
void experience_consolidation(float *m, const float *w, const float *w_pred,
                              float eta, int n);

/* Formula 17 extension: Convergence Score Ultra
 * Adds a repellent term that penalises proximity to a previously-seen centroid,
 * preventing the cluster from collapsing back to an earlier attractor.
 * repellent: (dim,) vector of the previous best centroid (or NULL to skip)
 * repellent_weight: how strongly to penalise proximity (0 = off)
 */
float convergence_score_ultra(const float *preds, const float *confs,
                               int n_preds, int dim, float alpha,
                               const float *repellent, float repellent_weight);

#endif
