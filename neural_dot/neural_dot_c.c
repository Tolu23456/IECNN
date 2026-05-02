#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "../shared_defs.h"
#include "../formulas/formulas.h"

/* ── Pre-generated Gaussian noise table ──────────────────────────────────────
 * Initialized once when the .so loads via __attribute__((constructor)).
 * Eliminates 1.57M logf/cosf/sqrtf calls per sentence in training mode.
 * Uses Box-Muller on a 16-bit LCG — good enough for stochastic noise.
 */
#define NOISE_TABLE_SIZE (1 << 16)  /* 65536 */
static float _g_noise_table[NOISE_TABLE_SIZE];

static void __attribute__((constructor)) _init_noise_table(void) {
    unsigned int seed = 0xDEADBEEFu;
    for (int i = 0; i < NOISE_TABLE_SIZE; i += 2) {
        seed = seed * 1103515245u + 12345u;
        float u1 = ((seed >> 8) & 0xFFFFFFu) * (1.0f / 16777216.0f) + 1e-7f;
        seed = seed * 1103515245u + 12345u;
        float u2 = ((seed >> 8) & 0xFFFFFFu) * (1.0f / 16777216.0f);
        float r   = sqrtf(-2.0f * logf(u1));
        float phi = 6.283185307f * u2;
        _g_noise_table[i]     = r * cosf(phi);
        _g_noise_table[i + 1] = r * sinf(phi);
    }
}

#define FAST_NOISE(idx) _g_noise_table[(unsigned int)(idx) & (NOISE_TABLE_SIZE - 1u)]

/* Optimized Slice and Pool selection */

void temporal_pool(const float *mat, int n, int dim, int rtl, float *out) {
    float sum_w = 0.0f;
    memset(out, 0, dim * sizeof(float));
    for (int i = 0; i < n; i++) {
        int idx = rtl ? (n - 1 - i) : i;
        float w = expf((float)idx / (float)n - 1.0f);
        sum_w += w;
        for (int d = 0; d < dim; d++) out[d] += mat[i * dim + d] * w;
    }
    for (int d = 0; d < dim; d++) out[d] /= (sum_w + 1e-10f);
}

void relational_pool(const float *mat, int n, int dim, float *out) {
    memset(out, 0, dim * sizeof(float));
    if (n <= 1) {
        if (n == 1) memcpy(out, mat, dim * sizeof(float));
        return;
    }
    float count = (float)n * (n - 1) / 2.0f;
    for (int k = 0; k < n; k++) {
        float coeff = (float)(n - 1 - 2 * k);
        for (int d = 0; d < dim; d++) {
            out[d] += coeff * mat[k * dim + d];
        }
    }
    for (int d = 0; d < dim; d++) out[d] /= (count + 1e-10f);
}

void logic_pool(const float *mat, int n, int dim, float *out) {
    if (n < 3) {
        memset(out, 0, dim * sizeof(float));
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dim; d++) out[d] += mat[i * dim + d] / (float)n;
        }
        return;
    }
    float accel[256];
    memset(accel, 0, sizeof(accel));
    for (int i = 0; i < n - 2; i++) {
        for (int d = 0; d < dim; d++) {
            float v1 = mat[i * dim + d];
            float v2 = mat[(i + 1) * dim + d];
            float v3 = mat[(i + 2) * dim + d];
            accel[d] += (v3 - 2.0f * v2 + v1) / (float)(n - 2);
        }
    }
    for (int d = 0; d < dim; d++) {
        float mean = 0.0f;
        for (int i = 0; i < n; i++) mean += mat[i * dim + d] / (float)n;
        out[d] = 0.7f * mean + 0.3f * accel[d];
    }
}

/* Linear Congruential Generator for fast internal noise */
unsigned int fast_rand(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (*seed / 65536) % 32768;
}

float fast_randn(unsigned int *seed) {
    float u1 = (float)fast_rand(seed) / 32768.0f;
    float u2 = (float)fast_rand(seed) / 32768.0f;
    return sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(6.283185f * u2);
}

void project_head(const float *v, const float *W, const float *b, int dim, float temperature, unsigned int *seed, float *out) {
    for (int i = 0; i < dim; i++) {
        float sum = b[i];
        for (int j = 0; j < dim; j++) sum += W[i * dim + j] * v[j];
        float noise = fast_randn(seed) * temperature * 0.05f;
        out[i] = tanhf(sum) + noise;
    }
}

void reason_fast(float *v, const float *W, const float *b, int dim, int steps, float temperature, float attention_bias, const float *consensus, unsigned int *seed, float abstraction_bias) {
    float delta[256];
    float gate = 0.2f + 0.3f * abstraction_bias;
    float pressure = 0.15f * (1.0f - attention_bias);
    float sqrt_dim = sqrtf((float)dim);

    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < dim; i++) {
            float sum = b[i];
            for (int j = 0; j < dim; j++) sum += W[i * dim + j] * v[j];
            delta[i] = tanhf(sum);
        }
        for (int i = 0; i < dim; i++) {
            if (consensus) v[i] = (1.0f - pressure) * v[i] + pressure * consensus[i];
            v[i] = (1.0f - gate) * v[i] + gate * delta[i] + fast_randn(seed) * temperature * 0.02f;
        }
        float norm_sq = 0.0f;
        for (int i = 0; i < dim; i++) norm_sq += v[i] * v[i];
        float norm = sqrtf(norm_sq);
        if (norm > 1e-10f) {
            float scale = sqrt_dim / norm;
            for (int i = 0; i < dim; i++) v[i] *= scale;
        }
    }
}

/* ── BATCH PIPELINE ──────────────────────────────────────────────────────────
 *
 * training_mode = 1: skip fast_randn (1.57M transcendental calls/sentence)
 *   → use zero noise for head projections
 *   → pass temperature=0 to reason_fast (zero noise in reasoning too)
 *   → safe because: causal_train_pass only needs the embedding direction,
 *     not stochastic diversity; noise is a regulariser for inference only.
 *
 * OMP: the #pragma omp parallel for that was here has been REMOVED.
 *   Sentence-level OMP in pipeline_batch_run_c handles all parallelism.
 *   Nesting OMP here caused the INNER loop to fall back to 1 thread while
 *   the OUTER loop also ran 1 thread → fully sequential.  Now the outer
 *   sentence loop is the only OMP region.
 */
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
) {
    float T_base = 1.0f + 0.3f * context_entropy;

    for (int d = 0; d < num_dots; d++) {
        unsigned int seed = seeds[d];
        const float *bias = dots_bias + d * 6;
        float gran_bias  = bias[1];
        float abs_bias   = bias[2];
        float temp_bias  = bias[4];
        float depth_bias = bias[5];

        /* 1. Slice selection */
        int patch_size = (int)(gran_bias * (float)seq_len);
        if (patch_size < 1) patch_size = 1;
        if (patch_size > seq_len) patch_size = seq_len;

        int offset = 0;
        if (seq_len > patch_size) {
            if (causal) {
                float u1 = (float)fast_rand(&seed) / 32768.0f;
                float u2 = (float)fast_rand(&seed) / 32768.0f;
                float bf = (u1 > u2) ? u1 : u2;
                offset = (int)(bf * (float)(seq_len - patch_size));
            } else {
                offset = fast_rand(&seed) % (seq_len - patch_size + 1);
            }
        }
        int start = offset;
        int end   = start + patch_size;
        out_starts[d] = start;
        out_ends[d]   = end;

        /* 2. Pooling */
        float pooled[256];
        const float *sl  = basemap + start * dim;
        int   sl_len     = end - start;
        int   type       = dots_types[d];

        int is_rtl = 0;
        if (sl_len > 0) {
            if (sl[EMBED_DIM + POS_DIM + FREQ_DIM + 11] > 0.5f) is_rtl = 1;
        }

        if      (type == 4) temporal_pool(sl, sl_len, dim, is_rtl, pooled);
        else if (type == 3) relational_pool(sl, sl_len, dim, pooled);
        else if (type == 6) logic_pool(sl, sl_len, dim, pooled);
        else if (type == 9) {
            for (int j = 0; j < dim; j++)
                pooled[j] = 0.5f * (sl[j] + sl[(sl_len - 1) * dim + j]);
        } else {
            memset(pooled, 0, sizeof(pooled));
            for (int i = 0; i < sl_len; i++)
                for (int j = 0; j < dim; j++) pooled[j] += sl[i * dim + j] / (float)sl_len;
        }

        /* 3. Dim Focus */
        if      (type == 0) { for (int i = 128; i < 256; i++) pooled[i] = 0; }
        else if (type == 1 || type == 6) { for (int i = 0; i < 128; i++) pooled[i] = 0; }
        else if (type == 7) {
            for (int i = 0; i < 236; i++) pooled[i] = 0;
            for (int i = 252; i < 256; i++) pooled[i] = 0;
        }

        /* 4. Abstract Transform */
        float abstract[256];
        const float *W = dots_W + d * dim * dim;
        const float *b = dots_b_offset + d * dim;
        for (int i = 0; i < dim; i++) {
            float sum = b[i];
            for (int j = 0; j < dim; j++) sum += W[i * dim + j] * pooled[j];
            float nonlinear = tanhf(sum);
            abstract[i] = (1.0f - abs_bias) * pooled[i] + abs_bias * nonlinear;
        }

        /* 5. Reasoning (skip noise in training mode by passing T=0) */
        float T = temp_bias * T_base;
        if (abs_bias > 0.4f) {
            int steps = (int)(depth_bias * 5.0f);
            if (steps < 1) steps = 1;
            float reason_T = training_mode ? 0.0f : T;
            reason_fast(abstract, W, b, dim, steps, reason_T, bias[0], consensus, &seed, abs_bias);
        }

        /* 6. Head projections
         *    training_mode=1: skip fast_randn (no transcendentals in hot path)
         *    training_mode=0: full noise for inference diversity
         */
        int n_heads = dots_n_heads[d];
        for (int h = 0; d < num_dots && h < n_heads; h++) {
            const float *HW  = dots_head_projs + (d * 8 + h) * dim * dim;
            float       *p_out = out_preds + (d * 8 + h) * dim;

            if (training_mode) {
                for (int i = 0; i < dim; i++) {
                    float sum = b[i] * (float)(h + 1) * 0.1f;
                    for (int j = 0; j < dim; j++) sum += HW[i * dim + j] * abstract[j];
                    p_out[i] = tanhf(sum);
                }
            } else {
                for (int i = 0; i < dim; i++) {
                    float sum = b[i] * (float)(h + 1) * 0.1f;
                    for (int j = 0; j < dim; j++) sum += HW[i * dim + j] * abstract[j];
                    p_out[i] = tanhf(sum) + fast_randn(&seed) * T * 0.05f;
                }
            }

            /* Norm & Confidence */
            float nsq = 0.0f;
            for (int i = 0; i < dim; i++) nsq += p_out[i] * p_out[i];
            float norm = sqrtf(nsq);
            if (norm > 1e-10f) {
                float sc = sqrtf((float)dim) / norm;
                for (int i = 0; i < dim; i++) p_out[i] *= sc;
            }
            out_confs[d * 8 + h] = tanhf(sqrtf(nsq) / sqrtf((float)dim));
        }
    }
}
