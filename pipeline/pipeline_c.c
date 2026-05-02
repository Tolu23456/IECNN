#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "../shared_defs.h"
#include "../formulas/formulas.h"
#include "../neural_dot/neural_dot.h"
#include "../convergence/convergence.h"
#include "../aim/aim.h"

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
    int *out_win_assign
) {
    float *cur_basemap = (float *)malloc(seq_len * dim * sizeof(float));
    memcpy(cur_basemap, basemap_matrix, seq_len * dim * sizeof(float));

    int max_heads_per_dot = 8;
    int total_slots = num_dots * max_heads_per_dot;
    int max_candidates = total_slots * 3;

    float *all_preds = (float *)malloc(max_candidates * dim * sizeof(float));
    float *all_confs = (float *)malloc(max_candidates * sizeof(float));
    int *all_starts = (int *)malloc(num_dots * sizeof(int));
    int *all_ends = (int *)malloc(num_dots * sizeof(int));
    int *assign = (int *)malloc(max_candidates * sizeof(int));
    int *counts = (int *)malloc(max_candidates * sizeof(int));

    float current_centroid[256];
    memset(current_centroid, 0, sizeof(current_centroid));

    for (int rnd = 0; rnd < max_iterations; rnd++) {
        predict_batch_c(num_dots, dim, seq_len, cur_basemap, dots_W, dots_head_projs,
                        dots_b_offset, dots_bias, dots_types, dots_n_heads, seeds,
                        rnd > 0 ? current_centroid : NULL, 0.5f, 0,
                        all_preds, all_confs, all_starts, all_ends);

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

        // Ensure winner is ID 0
        if (best_c != 0) {
            for (int i = 0; i < active_idx; i++) {
                if (assign[i] == 0) assign[i] = best_c;
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
                cur_basemap[i * dim + d] = 0.85f * cur_basemap[i * dim + d] + 0.15f * new_centroid[d];
            }
        }

        float diff = 0.0f;
        if (rnd > 0) {
            for (int d = 0; d < dim; d++) {
                float d_v = new_centroid[d] - current_centroid[d];
                diff += d_v * d_v;
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

    free(cur_basemap); free(all_preds); free(all_confs); free(all_starts);
    free(all_ends); free(assign); free(counts);
}

void pipeline_batch_run_c(
    int num_sents, int num_dots, int dim, const int *seq_lens,
    const float *basemap_matrices, const int *basemap_offsets,
    const float *dots_W, const float *dots_head_projs, const float *dots_b_offset,
    const float *dots_bias, const int *dots_types, const int *dots_n_heads,
    const unsigned int *seeds, float alpha, float gamma, int max_iterations,
    float *out_vectors, int *out_win_assigns
) {
    #pragma omp parallel for
    for (int s = 0; s < num_sents; s++) {
        int total_slots = num_dots * 8;
        pipeline_run_c(num_dots, dim, seq_lens[s], basemap_matrices + basemap_offsets[s] * dim,
                       dots_W, dots_head_projs, dots_b_offset, dots_bias, dots_types,
                       dots_n_heads, seeds, alpha, gamma, max_iterations,
                       out_vectors + s * dim,
                       out_win_assigns ? (out_win_assigns + s * total_slots) : NULL);
    }
}
