/*
 * fast_count.c — C-level word and ngram counting for IECNN fast training.
 *
 * Compiled as a shared library (fast_count_c.so) and driven from Python
 * via ctypes.  The Python caller pre-encodes each token as an int32 ID,
 * flattens all sentences into one array, and allocates the hash tables as
 * plain numpy uint64 / int32 arrays — zero Python objects in the hot path.
 *
 * Hash tables use open-addressing with linear probing and a 64-bit
 * MurmurHash3-inspired finaliser.  Separate key/count arrays keep stride
 * predictable for the CPU prefetcher.
 *
 * Throughput targets (4-core, 100k-sentence batch, small-medium vocab):
 *   unigram + bigram    ≥ 500 k sent/s
 *   + trigram           ≥ 300 k sent/s
 */

#include "fast_count.h"
#include <stdint.h>
#include <string.h>

/* ── 64-bit hash finaliser (MurmurHash3-inspired) ───────────────── */
static inline uint32_t _mix(uint64_t k) {
    k ^= k >> 33;
    k *= UINT64_C(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= UINT64_C(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return (uint32_t)k;
}

/* ── Open-addressing insert — linear probing ────────────────────── */
static inline void _ht_add(uint64_t* keys, int32_t* counts,
                            uint32_t mask, uint64_t key) {
    uint32_t h = _mix(key) & mask;
    while (keys[h] != UINT64_MAX && keys[h] != key) {
        h = (h + 1) & mask;
    }
    keys[h]  = key;
    counts[h]++;
}

/* ── Public API ─────────────────────────────────────────────────── */

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
) {
    const uint32_t bi_mask  = bi_slots  - 1;
    const uint32_t tri_mask = (tri_keys  != ((void*)0)) ? tri_slots  - 1 : 0;
    const uint64_t vs       = (uint64_t)vocab_size;
    const int32_t* p        = flat_ids;
    const int do_bi  = (ngram_hi >= 2) && (bi_keys  != ((void*)0));
    const int do_tri = (ngram_hi >= 3) && (tri_keys != ((void*)0));

    for (int32_t s = 0; s < n_sents; ++s) {
        const int32_t len = sent_lens[s];

        /* ── Unigrams ──────────────────────────────────────────── */
        for (int32_t i = 0; i < len; ++i) {
            const int32_t id = p[i];
            if ((uint32_t)id < (uint32_t)vocab_size) {
                word_counts[id]++;
            }
        }

        /* ── Bigrams ───────────────────────────────────────────── */
        if (do_bi && len >= 2) {
            for (int32_t i = 0; i < len - 1; ++i) {
                const uint64_t key = (uint64_t)p[i] * vs + (uint64_t)p[i + 1];
                _ht_add(bi_keys, bi_counts, bi_mask, key);
            }
        }

        /* ── Trigrams ──────────────────────────────────────────── */
        if (do_tri && len >= 3) {
            for (int32_t i = 0; i < len - 2; ++i) {
                const uint64_t key =
                    ((uint64_t)p[i] * vs + (uint64_t)p[i + 1]) * vs
                    + (uint64_t)p[i + 2];
                _ht_add(tri_keys, tri_counts, tri_mask, key);
            }
        }

        p += len;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * fast_tok_batch_ascii — Batch ASCII tokeniser + int-ID encoder.
 *
 * Processes n_texts sentences in one C call.  Each sentence is a
 * null-terminated ASCII string (caller lowercases before passing).
 *
 * Tokens written into out_ids (int32) using a caller-supplied word2id
 * hash table (separate key/value arrays, open-addressing, power-of-2 size).
 *
 * wt_keys[h]  : uint64  — packed ASCII word key (UINT64_MAX = empty)
 * wt_vals[h]  : int32   — word ID assigned at first insertion
 * *n_words    : current number of distinct words (updated in-place)
 * out_ids     : int32[total_output_tokens] — all token IDs concatenated
 * out_lens    : int32[n_texts]             — tokens per sentence
 *
 * Returns total token count, or -1 if any non-ASCII byte is encountered.
 * ═════════════════════════════════════════════════════════════════════ */

/* Pack up to 8 lowercase ASCII chars into a uint64 key.
 * Words longer than 8 chars are hashed via FNV-1a to guarantee uniqueness.*/
static inline uint64_t _word_key(const char* s, int32_t len) {
    if (len <= 8) {
        uint64_t k = 0;
        for (int32_t i = 0; i < len; i++)
            k = (k << 8) | (uint8_t)s[i];
        return k | ((uint64_t)(uint8_t)len << 56);   /* encode length */
    }
    /* FNV-1a for longer words */
    uint64_t h = UINT64_C(14695981039346656037);
    for (int32_t i = 0; i < len; i++)
        h = (h ^ (uint8_t)s[i]) * UINT64_C(1099511628211);
    return h | 1;   /* ensure != UINT64_MAX */
}

static inline int32_t _wt_get_or_insert(
        uint64_t* wt_keys, int32_t* wt_vals, uint32_t wt_mask,
        uint64_t key, int32_t* n_words) {
    uint32_t h = _mix(key) & wt_mask;
    while (wt_keys[h] != UINT64_MAX && wt_keys[h] != key)
        h = (h + 1) & wt_mask;
    if (wt_keys[h] == UINT64_MAX) {
        wt_keys[h] = key;
        wt_vals[h] = (*n_words)++;
    }
    return wt_vals[h];
}

int32_t fast_tok_batch_ascii(
    const char* const* texts,   /* array of n_texts pointers to C strings  */
    const int32_t*     text_lens,  /* byte length of each text               */
    int32_t            n_texts,
    uint64_t*          wt_keys,    /* word-table keys  (UINT64_MAX = empty)  */
    int32_t*           wt_vals,    /* word-table values (word ID)            */
    uint32_t           wt_slots,   /* power of two                           */
    int32_t*           n_words,    /* in/out: current vocab size             */
    int32_t*           out_ids,    /* output token IDs (flat)                */
    int32_t*           out_lens    /* output: tokens per sentence            */
) {
    const uint32_t wt_mask = wt_slots - 1;
    int32_t total = 0;

    for (int32_t t = 0; t < n_texts; t++) {
        const char* s   = texts[t];
        const int32_t n = text_lens[t];
        int32_t i = 0, sent_toks = 0;

        while (i < n) {
            unsigned char c = (unsigned char)s[i];

            if (c > 127) return -1;   /* non-ASCII: abort */

            if ((c | 32) >= 'a' && (c | 32) <= 'z') {
                /* Word token — collect and lowercase */
                int32_t start = i;
                while (i < n) {
                    unsigned char cc = (unsigned char)s[i];
                    if (!((cc | 32) >= 'a' && (cc | 32) <= 'z')) break;
                    /* lowercase in-place safe (read-only? no — make copy) */
                    i++;
                }
                /* Build lowercased key from original chars */
                char tmp[128]; int32_t wlen = i - start;
                if (wlen > 127) wlen = 127;
                for (int32_t k = 0; k < wlen; k++)
                    tmp[k] = (char)((unsigned char)s[start + k] | 32);
                uint64_t key = _word_key(tmp, wlen);
                out_ids[total++] = _wt_get_or_insert(
                        wt_keys, wt_vals, wt_mask, key, n_words);
                sent_toks++;
            } else if (c >= '0' && c <= '9') {
                /* Digit run */
                int32_t start = i;
                while (i < n && (unsigned char)s[i] >= '0' && (unsigned char)s[i] <= '9') i++;
                uint64_t key = _word_key(s + start, i - start);
                out_ids[total++] = _wt_get_or_insert(
                        wt_keys, wt_vals, wt_mask, key, n_words);
                sent_toks++;
            } else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                i++;   /* whitespace: skip */
            } else if (c >= 33 && c <= 126) {
                /* Single punctuation / symbol token */
                char tmp[2] = {(char)c, 0};
                uint64_t key = _word_key(tmp, 1);
                out_ids[total++] = _wt_get_or_insert(
                        wt_keys, wt_vals, wt_mask, key, n_words);
                i++; sent_toks++;
            } else {
                i++;   /* control char: skip */
            }
        }
        out_lens[t] = sent_toks;
    }
    return total;
}

/* ═══════════════════════════════════════════════════════════════════════
 * fast_tok_flat_ascii — Batch ASCII tokenise + encode in one C call.
 *
 * Input: all texts concatenated in text_buf, their byte offsets and lengths.
 * Each character is lowercased inline; non-ASCII → returns -1 immediately.
 *
 * Output:
 *   out_ids   flat int32 token IDs (allocated by caller, size = total tokens)
 *   out_lens  int32 per-text token counts
 *   wt_keys / wt_vals: open-addressing word-table (uint64 key → int32 ID)
 *   *n_words: in/out (current distinct word count)
 *
 * Returns total number of tokens written, or -1 on non-ASCII input.
 * ═════════════════════════════════════════════════════════════════════ */
int32_t fast_tok_flat_ascii(
    const char*    text_buf,    /* flat byte buffer: all texts concatenated  */
    const int32_t* text_offs,   /* byte offset of each text in text_buf      */
    const int32_t* text_lens,   /* byte length of each text (may be 0)       */
    int32_t        n_texts,
    uint64_t*      wt_keys,     /* word-table keys  (init to UINT64_MAX)     */
    int32_t*       wt_vals,     /* word-table values (word ID)               */
    uint32_t       wt_slots,    /* power of two                              */
    int32_t*       n_words,     /* in/out: current vocab size                */
    int32_t*       out_ids,     /* output token IDs (flat across all texts)  */
    int32_t*       out_lens     /* output: tokens per text                   */
) {
    const uint32_t wt_mask = wt_slots - 1;
    int32_t total = 0;

    for (int32_t t = 0; t < n_texts; t++) {
        const char*   s   = text_buf + text_offs[t];
        const int32_t len = text_lens[t];
        int32_t i = 0, sent_toks = 0;

        while (i < len) {
            unsigned char c = (unsigned char)s[i];
            if (c > 127) return -1;   /* non-ASCII: abort */

            if ((c | 32) >= 'a' && (c | 32) <= 'z') {
                /* Word token — lowercase and pack key */
                char tmp[128]; int32_t wlen = 0;
                while (i < len && wlen < 127) {
                    unsigned char cc = (unsigned char)s[i];
                    if (!((cc | 32) >= 'a' && (cc | 32) <= 'z')) break;
                    tmp[wlen++] = (char)((cc >= 'A' && cc <= 'Z') ? cc + 32 : cc);
                    i++;
                }
                uint64_t key = _word_key(tmp, wlen);
                out_ids[total++] = _wt_get_or_insert(
                        wt_keys, wt_vals, wt_mask, key, n_words);
                sent_toks++;
            } else if (c >= '0' && c <= '9') {
                const char* start = s + i;
                int32_t wlen = 0;
                while (i < len && (unsigned char)s[i] >= '0' && (unsigned char)s[i] <= '9')
                    { wlen++; i++; }
                uint64_t key = _word_key(start, wlen);
                out_ids[total++] = _wt_get_or_insert(
                        wt_keys, wt_vals, wt_mask, key, n_words);
                sent_toks++;
            } else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                i++;
            } else if (c >= 33 && c <= 126) {
                char tmp[1] = {(char)c};
                uint64_t key = _word_key(tmp, 1);
                out_ids[total++] = _wt_get_or_insert(
                        wt_keys, wt_vals, wt_mask, key, n_words);
                i++; sent_toks++;
            } else {
                i++;
            }
        }
        out_lens[t] = sent_toks;
    }
    return total;
}

/* ═══════════════════════════════════════════════════════════════════════
 * fast_tok_bytes_ascii — Tokenise + lowercase ASCII texts into a flat
 * byte buffer (tokens separated by '\x01', sentences laid out sequentially
 * with their byte lengths in out_byte_lens).
 *
 * This replaces Python regex.findall for ASCII-only text. Python then does:
 *   bytes(out_buf[pos:pos+slen]).split(b'\x01') → token list per sentence.
 *
 * Returns total bytes written into out_buf, or -1 if non-ASCII found.
 * ═════════════════════════════════════════════════════════════════════ */
int32_t fast_tok_bytes_ascii(
    const char*    text_buf,        /* all texts concatenated               */
    const int32_t* text_offs,       /* byte offset of each text             */
    const int32_t* text_lens,       /* byte length of each text             */
    int32_t        n_texts,
    char*          out_buf,         /* output bytes: tokens + '\x01'        */
    int32_t        out_cap,
    int32_t*       out_byte_lens    /* output: bytes written per sentence   */
) {
    int32_t out_total = 0;

    for (int32_t t = 0; t < n_texts; t++) {
        const char*   s   = text_buf + text_offs[t];
        const int32_t len = text_lens[t];
        int32_t i = 0;
        int32_t sent_start = out_total;
        int first_tok = 1;

        while (i < len) {
            unsigned char c = (unsigned char)s[i];
            if (c > 127) return -1;   /* non-ASCII */

            if ((c | 32) >= 'a' && (c | 32) <= 'z') {
                if (!first_tok && out_total < out_cap) out_buf[out_total++] = '\x01';
                first_tok = 0;
                while (i < len) {
                    unsigned char cc = (unsigned char)s[i];
                    if (!((cc | 32) >= 'a' && (cc | 32) <= 'z')) break;
                    if (out_total < out_cap)
                        out_buf[out_total++] = (char)((cc >= 'A' && cc <= 'Z') ? cc+32 : cc);
                    i++;
                }
            } else if (c >= '0' && c <= '9') {
                if (!first_tok && out_total < out_cap) out_buf[out_total++] = '\x01';
                first_tok = 0;
                while (i < len && (unsigned char)s[i] >= '0' && (unsigned char)s[i] <= '9') {
                    if (out_total < out_cap) out_buf[out_total++] = s[i];
                    i++;
                }
            } else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                i++;
            } else if (c >= 33 && c <= 126) {
                if (!first_tok && out_total < out_cap) out_buf[out_total++] = '\x01';
                first_tok = 0;
                if (out_total < out_cap) out_buf[out_total++] = (char)c;
                i++;
            } else {
                i++;
            }
        }
        out_byte_lens[t] = out_total - sent_start;
    }
    return out_total;
}
