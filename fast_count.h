#ifndef FAST_COUNT_H
#define FAST_COUNT_H

#include <stdint.h>

/*
 * fast_count.h — C-level word and ngram counting for IECNN fast training.
 *
 * Uses open-addressing hash tables with 64-bit keys so the counting
 * inner loop is pure integer arithmetic — no Python string creation,
 * no GIL, no Counter hash overhead.
 *
 * Separate key / count arrays match numpy dtype layout directly:
 *   bi_keys   → np.uint64[bi_slots]  (init to UINT64_MAX = empty marker)
 *   bi_counts → np.int32 [bi_slots]  (init to 0)
 *
 * Bigram  key encoding : id1 * vocab_size + id2
 * Trigram key encoding : (id1 * vocab_size + id2) * vocab_size + id3
 * Both are invertible and give unique keys for vocab_size ≤ 1 << 21.
 */

/*
 * fast_count_c — count unigrams + bigrams (+ trigrams) from pre-encoded
 *               int32 token sequences.
 *
 * Parameters
 * ----------
 * flat_ids    int32[total_tokens]  all token IDs concatenated
 * sent_lens   int32[n_sents]       per-sentence token counts
 * n_sents     number of sentences
 * vocab_size  distinct token types in this chunk
 * ngram_lo    minimum n for ngrams (usually 1 or 2)
 * ngram_hi    maximum n for ngrams (2 = bigrams only, 3 = + trigrams)
 * word_counts int32[vocab_size]    output — zeroed by caller
 * bi_keys     uint64[bi_slots]     bigram hash table keys (UINT64_MAX = empty)
 * bi_counts   int32 [bi_slots]     bigram hash table counts (zeroed)
 * bi_slots    power of two
 * tri_keys    uint64[tri_slots]    trigram hash table keys (NULL if ngram_hi<3)
 * tri_counts  int32 [tri_slots]    trigram hash table counts (NULL if ngram_hi<3)
 * tri_slots   power of two (0 if tri_keys == NULL)
 */
void fast_count_c(
    const int32_t*  flat_ids,
    const int32_t*  sent_lens,
    int32_t         n_sents,
    int32_t         vocab_size,
    int32_t         ngram_lo,
    int32_t         ngram_hi,
    int32_t*        word_counts,
    uint64_t*       bi_keys,
    int32_t*        bi_counts,
    uint32_t        bi_slots,
    uint64_t*       tri_keys,
    int32_t*        tri_counts,
    uint32_t        tri_slots
);

#endif /* FAST_COUNT_H */
