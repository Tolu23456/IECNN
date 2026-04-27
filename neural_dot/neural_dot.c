#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "../formulas/formulas.h"

/* Fast dot pooling and prediction */

void temporal_pool(const float *mat, int n, int dim, float *out) {
    float sum_w = 0.0f;
    memset(out, 0, dim * sizeof(float));
    for (int i = 0; i < n; i++) {
        float w = expf((float)i / (float)n - 1.0f);
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
    /* Optimized O(n) relational pooling: sum_{i < j} (x_i - x_j) = sum_{k=0}^{n-1} (n - 1 - 2k) * x_k */
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
    float *accel = (float *)calloc(dim, sizeof(float));
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
    free(accel);
}

void project_head(const float *v, const float *W, const float *b, int dim, float temperature, float *noise, float *out) {
    for (int i = 0; i < dim; i++) {
        float sum = b[i];
        for (int j = 0; j < dim; j++) sum += W[i * dim + j] * v[j];
        out[i] = tanhf(sum) + noise[i] * temperature * 0.05f;
    }
}

/* Dot Synergy: Multi-dot cross-query mechanism
 * Dots can adjust their own representation based on another 'peer' representation.
 */
void apply_synergy_fast(float *v, const float *peer, int dim, float synergy_weight) {
    for (int i = 0; i < dim; i++) {
        v[i] = (1.0f - synergy_weight) * v[i] + synergy_weight * peer[i];
    }
}
