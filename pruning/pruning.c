#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "../formulas/formulas.h"

/* Stage 1 Fast Deduplication
 * preds: (n x dim)
 * out_kept_indices: array of size n to store indices of kept items
 * returns number of kept items
 */
int deduplicate_fast(const float *preds, int n, int dim, float threshold, float alpha, int *out_kept_indices) {
    if (n <= 0) return 0;

    int num_kept = 0;
    for (int i = 0; i < n; i++) {
        const float *p = preds + i * dim;
        int is_dup = 0;
        for (int j = 0; j < num_kept; j++) {
            const float *q = preds + out_kept_indices[j] * dim;
            if (similarity_score(p, q, dim, alpha) > threshold) {
                is_dup = 1;
                break;
            }
        }
        if (!is_dup) {
            out_kept_indices[num_kept++] = i;
        }
    }
    return num_kept;
}
