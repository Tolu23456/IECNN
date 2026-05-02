#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "../shared_defs.h"
#include "../formulas/formulas.h"
#include "../neural_dot/neural_dot.h"
#include "../convergence/convergence.h"
#include "../aim/aim.h"

/* ── Thread-local pre-allocated buffers ──────────────────────────────────────
 *
 * Root cause eliminated: pipeline_run_c previously did malloc(~3 MB) on
 * every sentence.  On the first call per OMP thread, the OS maps ~768 new
 * 4-KB pages → ~1–2 ms of page-fault overhead PER THREAD PER SENTENCE.
 * With 8 parallel threads all faulting simultaneously, that serialised
 * through the kernel's page-fault handler → the measured ~1 710 ms "fixed
 * overhead" per run_batch call.
 *
 * Thread-local statics are allocated once per thread (lazily on first
 * access) and never freed, so subsequent calls go straight to L1/L2
 * after the first warm-up call.
 *
 * Sizing:
 *   TL_MAX_SEQ   = 512  tokens max sequence
 *   TL_MAX_CANDS = 3072 (128 dots × 8 heads × 3 for inversions)
 *   All dimensions hard-capped at 256 (FEATURE_DIM)
 */
#define TL_MAX_SEQ   512
#define TL_MAX_DOTS  256
#define TL_MAX_CANDS 3072

static _Thread_local float _tl_cur_basemap[TL_MAX_SEQ * 256];
static _Thread_local float _tl_all_preds[TL_MAX_CANDS * 256];
static _Thread_local float _tl_all_confs[TL_MAX_CANDS];
static _Thread_local int   _tl_all_starts[TL_MAX_DOTS];
static _Thread_local int   _tl_all_ends[TL_MAX_DOTS];
static _Thread_local int   _tl_assign[TL_MAX_CANDS];
static _Thread_local int   _tl_counts[TL_MAX_CANDS];

void pipeline_run_c(
    int num_dots, int dim, int seq_len,
    const float *basemap_matrix,
    const float *dots_W,
    const float *dots_head_projs,
    const float *dots_b_offset,
    const float *dots_bias,
    const int *dots_types,
    const int *dots_n_heads,
    const unsigned int *seeds,
    float alpha,
    float gamma,
    int max_iterations,
    float *out_vector,
    int *out_win_assign,
    int training_mode
) {
    /* Guard against oversized inputs rather than silently corrupting memory */
    if (seq_len > TL_MAX_SEQ || num_dots > TL_MAX_DOTS) {
        memset(out_vector, 0, dim * sizeof(float));
        return;
    }

    float *cur_basemap = _tl_cur_basemap;
    memcpy(cur_basemap, basemap_matrix, seq_len * dim * sizeof(float));

    int max_heads_per_dot = 8;
    int total_slots   = num_dots * max_heads_per_dot;
    int max_candidates = total_slots * 3;

    float *all_preds  = _tl_all_preds;
    float *all_confs  = _tl_all_confs;
    int   *all_starts = _tl_all_starts;
    int   *all_ends   = _tl_all_ends;
    int   *assign     = _tl_assign;
    int   *counts     = _tl_counts;

    float current_centroid[256];
    memset(current_centroid, 0, sizeof(current_centroid));

    for (int rnd = 0; rnd < max_iterations; rnd++) {
        predict_batch_c(num_dots, dim, seq_len, cur_basemap, dots_W, dots_head_projs,
                        dots_b_offset, dots_bias, dots_types, dots_n_heads, seeds,
                        rnd > 0 ? current_centroid : NULL, 0.5f, 0,
                        all_preds, all_confs, all_starts, all_ends,
                        training_mode);

        int active_idx = 0;
        for (int d = 0; d < num_dots; d++) {
            for (int h = 0; h < dots_n_heads[d]; h++) {
                int src_idx = d * max_heads_per_dot + h;
                if (active_idx != src_idx) {
                    memcpy(all_preds + active_idx * dim, all_preds + src_idx * dim, dim * sizeof(float));
                    all_confs[active_idx] = all_confs[src_idx];
                }
                active_idx++;
            }
        }
        int num_orig_active = active_idx;

        for (int i = 0; i < num_orig_active; i++) {
            float *orig = all_preds + i * dim;
            invert_feature(orig, dim, all_preds + active_idx * dim);
            all_confs[active_idx] = all_confs[i] * 0.9f; active_idx++;
            invert_relational(orig, dim, all_preds + active_idx * dim);
            all_confs[active_idx] = all_confs[i] * 0.85f; active_idx++;
        }

        int num_clusters = greedy_cluster(all_preds, active_idx, dim, alpha, 0.25f, assign);
        if (num_clusters <= 0) break;

        memset(counts, 0, num_clusters * sizeof(int));
        for (int i = 0; i < active_idx; i++) if (assign[i] >= 0) counts[assign[i]]++;

        int best_c = 0;
        for (int c = 1; c < num_clusters; c++) if (counts[c] > counts[best_c]) best_c = c;

        if (best_c != 0) {
            for (int i = 0; i < active_idx; i++) {
                if      (assign[i] == 0)      assign[i] = best_c;
                else if (assign[i] == best_c) assign[i] = 0;
            }
            best_c = 0;
        }

        float new_centroid[256];
        memset(new_centroid, 0, sizeof(new_centroid));
        int members = 0;
        for (int i = 0; i < active_idx; i++) {
            if (assign[i] == 0) {
                for (int d = 0; d < dim; d++) new_centroid[d] += all_preds[i * dim + d];
                members++;
            }
        }
        if (members > 0) {
            for (int d = 0; d < dim; d++) new_centroid[d] /= (float)members;
        } else break;

        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < dim; d++) {
                cur_basemap[i * dim + d] = 0.85f * cur_basemap[i * dim + d]
                                         + 0.15f * new_centroid[d];
            }
        }

        float diff = 0.0f;
        if (rnd > 0) {
            for (int d = 0; d < dim; d++) {
                float dv = new_centroid[d] - current_centroid[d];
                diff += dv * dv;
            }
            if (sqrtf(diff) < 0.001f) {
                memcpy(current_centroid, new_centroid, dim * sizeof(float));
                break;
            }
        }
        memcpy(current_centroid, new_centroid, dim * sizeof(float));
    }

    memcpy(out_vector, current_centroid, dim * sizeof(float));

    if (out_win_assign) {
        memset(out_win_assign, -1, total_slots * sizeof(int));
        int a_idx = 0;
        for (int d = 0; d < num_dots; d++) {
            for (int h = 0; h < dots_n_heads[d]; h++) {
                out_win_assign[d * max_heads_per_dot + h] = assign[a_idx++];
            }
        }
    }
}

void pipeline_batch_run_c(
    int num_sents, int num_dots, int dim, const int *seq_lens,
    const float *basemap_matrices, const int *basemap_offsets,
    const float *dots_W, const float *dots_head_projs, const float *dots_b_offset,
    const float *dots_bias, const int *dots_types, const int *dots_n_heads,
    const unsigned int *seeds, float alpha, float gamma, int max_iterations,
    float *out_vectors, int *out_win_assigns,
    int training_mode
) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int s = 0; s < num_sents; s++) {
        int total_slots = num_dots * 8;
        pipeline_run_c(num_dots, dim, seq_lens[s],
                       basemap_matrices + basemap_offsets[s] * dim,
                       dots_W, dots_head_projs, dots_b_offset, dots_bias,
                       dots_types, dots_n_heads, seeds,
                       alpha, gamma, max_iterations,
                       out_vectors + s * dim,
                       out_win_assigns ? (out_win_assigns + s * total_slots) : NULL,
                       training_mode);
    }
}
