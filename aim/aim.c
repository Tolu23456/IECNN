#include "aim.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*
 * AIM Inversions — C Implementation
 * Six ways to challenge a dot's prediction assumptions.
 */

/* ── Feature Inversion ──────────────────────────────────────────── */
/* Flip signs of dominant (above-mean-magnitude) dimensions.
 * "bright edge" → "dark edge", "convex" → "concave"          */
void invert_feature(const float *p, int n, float *out) {
    float mean_abs = 0.0f;
    for (int i = 0; i < n; i++) mean_abs += fabsf(p[i]);
    mean_abs /= (float)n;

    for (int i = 0; i < n; i++) {
        out[i] = (fabsf(p[i]) > mean_abs) ? -p[i] : p[i];
    }
}

/* ── Context Inversion ───────────────────────────────────────────── */
/* Swap high-activation and low-activation halves (role reversal).
 * "foreground" → "background"                                  */
void invert_context(const float *p, int n, float *out) {
    /* Build index array sorted by |p[i]| */
    int *idx = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;

    /* Bubble sort by absolute value (n is small, ~128) */
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (fabsf(p[idx[j]]) < fabsf(p[idx[i]])) {
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }
        }
    }

    memcpy(out, p, n * sizeof(float));
    int half = n / 2;
    for (int k = 0; k < half; k++) {
        float tmp = out[idx[k]];
        out[idx[k]] = out[idx[half + k]];
        out[idx[half + k]] = tmp;
    }

    free(idx);
}

/* ── Spatial Inversion ───────────────────────────────────────────── */
/* Reverse the ordering of equal-size dimension groups.
 * Same content, different spatial meaning.                     */
void invert_spatial(const float *p, int n, float *out) {
    memcpy(out, p, n * sizeof(float));
    int group_size = n / 8;
    if (group_size < 1) group_size = 1;
    int num_groups = n / group_size;

    float *tmp = (float *)malloc(group_size * sizeof(float));
    for (int g = 0; g < num_groups / 2; g++) {
        int s1 = g * group_size;
        int s2 = (num_groups - 1 - g) * group_size;
        memcpy(tmp,       out + s1, group_size * sizeof(float));
        memcpy(out + s1,  out + s2, group_size * sizeof(float));
        memcpy(out + s2,  tmp,      group_size * sizeof(float));
    }
    free(tmp);
}

/* ── Scale Inversion ─────────────────────────────────────────────── */
/* Rescale dimension quarters by inverse factors.
 * "small detail" ↔ "large structure"                           */
void invert_scale(const float *p, int n, float *out) {
    memcpy(out, p, n * sizeof(float));
    int q = n / 4;
    if (q < 1) return;

    /* Compute original norm for renormalization */
    float orig_norm = 0.0f;
    for (int i = 0; i < n; i++) orig_norm += p[i] * p[i];
    orig_norm = sqrtf(orig_norm);

    float scales[4] = {4.0f, 2.0f, 0.5f, 0.25f};
    for (int seg = 0; seg < 4; seg++) {
        int start = seg * q;
        int end   = (seg == 3) ? n : start + q;
        for (int i = start; i < end; i++) out[i] *= scales[seg];
    }

    /* Renormalize to match original norm */
    float new_norm = 0.0f;
    for (int i = 0; i < n; i++) new_norm += out[i] * out[i];
    new_norm = sqrtf(new_norm);
    if (new_norm > 1e-10f && orig_norm > 1e-10f) {
        float ratio = orig_norm / new_norm;
        for (int i = 0; i < n; i++) out[i] *= ratio;
    }
}

/* ── Abstraction Inversion ───────────────────────────────────────── */
/* Mix prediction with its complement in context space.
 * Flip between patch-level and object-level understanding.     */
void invert_abstraction(const float *p, const float *ctx, int n, float *out) {
    if (ctx == NULL) {
        /* Fallback: swap halves */
        int half = n / 2;
        for (int i = 0; i < half; i++) {
            out[i]        = p[half + i];
            out[half + i] = p[i];
        }
        return;
    }

    /* Compute projection of p onto ctx unit vector */
    float ctx_norm = 0.0f;
    for (int i = 0; i < n; i++) ctx_norm += ctx[i] * ctx[i];
    ctx_norm = sqrtf(ctx_norm);

    if (ctx_norm < 1e-10f) {
        memcpy(out, p, n * sizeof(float));
        return;
    }

    float proj_scalar = 0.0f;
    for (int i = 0; i < n; i++) proj_scalar += p[i] * (ctx[i] / ctx_norm);

    /* complement = p - projection, keep 10% of projection */
    for (int i = 0; i < n; i++) {
        float proj_component = proj_scalar * (ctx[i] / ctx_norm);
        out[i] = (p[i] - proj_component) + 0.1f * proj_component;
    }
}

/* ── Noise/Absence Inversion ─────────────────────────────────────── */
/* Suppress dominant features. "What if this isn't actually there?"  */
void invert_noise(const float *p, int n, unsigned int seed, float *out) {
    /* Simple LCG random number generator for portability */
    unsigned int state = seed;

    /* Find 75th percentile of |p| */
    float *abs_vals = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) abs_vals[i] = fabsf(p[i]);

    /* Partial sort to find 75th percentile (simple approach for small n) */
    float sorted[128];
    int nn = n < 128 ? n : 128;
    memcpy(sorted, abs_vals, nn * sizeof(float));
    /* Insertion sort */
    for (int i = 1; i < nn; i++) {
        float key = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j] > key) { sorted[j+1] = sorted[j]; j--; }
        sorted[j+1] = key;
    }
    float threshold = sorted[(int)(nn * 0.75f)];

    float std_p = 0.0f;
    for (int i = 0; i < n; i++) std_p += p[i] * p[i];
    std_p = sqrtf(std_p / (float)n) * 0.3f;

    for (int i = 0; i < n; i++) {
        /* LCG step */
        state = state * 1664525u + 1013904223u;
        float r = ((float)(state & 0xFFFFFF)) / (float)0xFFFFFF;

        if (abs_vals[i] > threshold) {
            out[i] = p[i] * (r * 0.2f);
        } else {
            out[i] = p[i];
        }

        /* Add small Gaussian-like noise via Box-Muller approximation */
        state = state * 1664525u + 1013904223u;
        float r2 = ((float)(state & 0xFFFFFF)) / (float)0xFFFFFF - 0.5f;
        out[i] += r2 * std_p;
    }

    free(abs_vals);
}
