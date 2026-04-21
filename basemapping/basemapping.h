#ifndef BASEMAPPING_H
#define BASEMAPPING_H

/*
 * BaseMapping — C Implementation
 * Matrix composition and character-level embedding operations.
 */

/* Compose a word embedding from its constituent character embeddings.
 * char_embeds: (num_chars x embed_dim) matrix of character embeddings
 * weights:     (num_chars,) importance weights per character
 * n_chars:     number of characters
 * embed_dim:   embedding dimension
 * out:         (embed_dim,) output composed embedding (normalized)
 */
void compose_from_chars(const float *char_embeds, const float *weights,
                        int n_chars, int embed_dim, float *out);

/* Build sinusoidal position encoding for position pos out of total */
void sinusoidal_position_enc(int pos, int total, int dim, float *out);

/* Normalize a vector in-place to unit length */
void normalize_vector(float *v, int n);

/* Compute mean pooling of a matrix (n_rows x dim) into a single vector (dim) */
void mean_pool(const float *matrix, int n_rows, int dim, float *out);

/* Compute attention-weighted pooling:
 * matrix: (n_rows x dim)
 * query:  (dim,)         -- attention query
 * sharpness: controls attention peakedness
 * out:    (dim,)
 */
void attention_pool(const float *matrix, const float *query,
                    int n_rows, int dim, float sharpness, float *out);

/* Apply cooccurrence smoothing to vocabulary embeddings.
 * vocab: (n_vocab x embed_dim)
 * neighbor_indices: (n_vocab x k) - indices of top k neighbors
 * neighbor_weights: (n_vocab x k) - weights of top k neighbors
 * n_vocab: number of words
 * embed_dim: embedding dimension
 * k: number of neighbors per word
 * alpha: smoothing rate
 * out: (n_vocab x embed_dim) smoothed embeddings
 */
void cooccurrence_smooth(const float *vocab, const int *neighbor_indices,
                        const float *neighbor_weights, int n_vocab,
                        int embed_dim, int k, float alpha, float *out);

#endif
