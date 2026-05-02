#ifndef NEURAL_DOT_H
#define NEURAL_DOT_H

void temporal_pool(const float *mat, int n, int dim, int rtl, float *out);
void relational_pool(const float *mat, int n, int dim, float *out);
void logic_pool(const float *mat, int n, int dim, float *out);
void project_head(const float *v, const float *W, const float *b, int dim, float temperature, unsigned int *seed, float *out);
void reason_fast(float *v, const float *W, const float *b, int dim, int steps, float temperature, float attention_bias, const float *consensus, unsigned int *seed, float abstraction_bias);

void predict_batch_c(
    int num_dots, int dim, int seq_len,
    const float *basemap,
    const float *dots_W,
    const float *dots_head_projs,
    const float *dots_b_offset,
    const float *dots_bias,
    const int *dots_types,
    const int *dots_n_heads,
    const unsigned int *seeds,
    const float *consensus,
    float context_entropy,
    int causal,
    float *out_preds,
    float *out_confs,
    int *out_starts,
    int *out_ends,
    int training_mode
);

#endif
