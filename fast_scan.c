/*
 * fast_scan.c — OpenMP parallel corpus vocabulary scanner for IECNN.
 *
 * Reads a plain-text corpus file (one sentence per line) entirely in C,
 * with two OpenMP-parallel passes: unigram counting then bigram counting.
 * No Python in the hot path — mmap + thread-local hash tables + serial merge.
 *
 * Exported API:
 *   int32_t scan_corpus_omp(
 *       filepath, n_threads, max_words, max_bigrams,
 *       out_word_buf, out_word_offs, out_word_lens, out_word_cnts, out_n_words,
 *       out_bi_w1, out_bi_w2, out_bi_cnts, out_n_bigrams)
 *     Returns 0 on success, -1 on error.
 *
 *   int64_t count_lines_c(filepath)
 *     Returns number of non-empty lines, or -1 on error.
 *
 * Memory layout per thread (defaults, WT_CAP=128k, BT_CAP=256k):
 *   WordTab : ~10 MB (2.25 MB arrays + 8 MB arena)
 *   BiTab   :  ~3 MB
 *   6 threads: ~78 MB + GlobalWordTab ~15 MB ≈ 93 MB total.
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#ifdef _OPENMP
#  include <omp.h>
#endif

/* ── Tunable constants ───────────────────────────────────────────────────── */
#define WT_CAP      (1 << 17)    /* 128k slots per thread WordTab            */
#define WT_MASK     (WT_CAP - 1)
#define ARENA_CAP   (8 << 20)    /* 8 MB string arena per thread             */
#define GT_ARENA    (32 << 20)   /* 32 MB arena for global word table        */
#define BT_CAP      (1 << 18)    /* 256k slots per thread BiTab              */
#define BT_MASK     (BT_CAP - 1)
#define MAX_WORD    64           /* tokens longer than this are ignored       */

/* ── Hash functions ──────────────────────────────────────────────────────── */

static inline uint64_t _fnv1a(const char *s, int n)
{
    uint64_t h = UINT64_C(14695981039346656037);
    for (int i = 0; i < n; i++)
        h = (h ^ (uint8_t)s[i]) * UINT64_C(1099511628211);
    return h ? h : 1;   /* 0 reserved for "empty" */
}

static inline uint64_t _mix(uint64_t k)
{
    k ^= k >> 30; k *= UINT64_C(0xbf58476d1ce4e5b9);
    k ^= k >> 27; k *= UINT64_C(0x94d049bb133111eb);
    k ^= k >> 31;
    return k;
}

/* ── Per-thread word hash table ──────────────────────────────────────────── */

typedef struct {
    uint64_t keys[WT_CAP];   /* fnv1a hash (0 = empty)   */
    int32_t  offs[WT_CAP];   /* byte offset in arena (-1 = empty) */
    int16_t  lens[WT_CAP];   /* word length              */
    int32_t  cnts[WT_CAP];   /* occurrence count         */
    int32_t  size;
    char    *arena;           /* heap-allocated string storage    */
    int32_t  arena_used;
} WordTab;

static WordTab *wt_alloc(void)
{
    WordTab *t = (WordTab *)calloc(1, sizeof(WordTab));
    if (!t) return NULL;
    t->arena = (char *)malloc(ARENA_CAP);
    if (!t->arena) { free(t); return NULL; }
    memset(t->offs, -1, sizeof(t->offs));
    return t;
}

static void wt_free(WordTab *t)
{
    if (t) { free(t->arena); free(t); }
}

static inline void wt_add(WordTab *t, const char *w, int16_t wlen)
{
    uint64_t key = _fnv1a(w, wlen);
    uint32_t h = (uint32_t)(_mix(key) & WT_MASK);
    while (t->keys[h]) {
        if (t->keys[h] == key && t->lens[h] == wlen &&
            memcmp(t->arena + t->offs[h], w, wlen) == 0) {
            t->cnts[h]++;
            return;
        }
        h = (h + 1) & WT_MASK;
    }
    /* New entry — guard arena and load capacity */
    if (t->arena_used + wlen + 1 > ARENA_CAP) return;
    if (t->size >= WT_CAP / 2) return;
    t->keys[h] = key;
    t->offs[h] = t->arena_used;
    t->lens[h] = wlen;
    t->cnts[h] = 1;
    memcpy(t->arena + t->arena_used, w, wlen);
    t->arena[t->arena_used + wlen] = '\0';
    t->arena_used += wlen + 1;
    t->size++;
}

/* ── Global word table (serial after merge) ──────────────────────────────── */

typedef struct {
    uint64_t keys[WT_CAP];
    int32_t  offs[WT_CAP];
    int16_t  lens[WT_CAP];
    int32_t  cnts[WT_CAP];
    int32_t  ids [WT_CAP];   /* sequential word ID, -1 = empty */
    int32_t  size;
    int32_t  n_ids;
    char    *arena;
    int32_t  arena_used;
} GlobalTab;

static GlobalTab *gt_alloc(void)
{
    GlobalTab *g = (GlobalTab *)calloc(1, sizeof(GlobalTab));
    if (!g) return NULL;
    g->arena = (char *)malloc(GT_ARENA);
    if (!g->arena) { free(g); return NULL; }
    memset(g->offs, -1, sizeof(g->offs));
    memset(g->ids,  -1, sizeof(g->ids));
    return g;
}

static void gt_free(GlobalTab *g)
{
    if (g) { free(g->arena); free(g); }
}

/* Insert (or accumulate count) into global table. Returns word ID. */
static int32_t gt_add(GlobalTab *g, const char *w, int16_t wlen, int32_t cnt)
{
    uint64_t key = _fnv1a(w, wlen);
    uint32_t h = (uint32_t)(_mix(key) & WT_MASK);
    while (g->keys[h]) {
        if (g->keys[h] == key && g->lens[h] == wlen &&
            memcmp(g->arena + g->offs[h], w, wlen) == 0) {
            g->cnts[h] += cnt;
            return g->ids[h];
        }
        h = (h + 1) & WT_MASK;
    }
    if (g->arena_used + wlen + 1 > GT_ARENA) return -1;
    if (g->size >= WT_CAP / 2) return -1;
    g->keys[h] = key;
    g->offs[h]  = g->arena_used;
    g->lens[h]  = wlen;
    g->cnts[h]  = cnt;
    g->ids[h]   = g->n_ids++;
    memcpy(g->arena + g->arena_used, w, wlen);
    g->arena[g->arena_used + wlen] = '\0';
    g->arena_used += wlen + 1;
    g->size++;
    return g->ids[h];
}

/* Read-only lookup — safe for concurrent reads from multiple threads. */
static int32_t gt_lookup(const GlobalTab *g, const char *w, int16_t wlen)
{
    uint64_t key = _fnv1a(w, wlen);
    uint32_t h = (uint32_t)(_mix(key) & WT_MASK);
    while (g->keys[h]) {
        if (g->keys[h] == key && g->lens[h] == wlen &&
            memcmp(g->arena + g->offs[h], w, wlen) == 0)
            return g->ids[h];
        h = (h + 1) & WT_MASK;
    }
    return -1;
}

/* ── Per-thread bigram table ─────────────────────────────────────────────── */

typedef struct {
    uint64_t keys[BT_CAP];   /* (w1_id * n_words + w2_id), UINT64_MAX=empty */
    int32_t  cnts[BT_CAP];
    int32_t  size;
} BiTab;

static BiTab *bt_alloc(void)
{
    BiTab *b = (BiTab *)calloc(1, sizeof(BiTab));
    if (!b) return NULL;
    memset(b->keys, 0xff, sizeof(b->keys));   /* UINT64_MAX = empty */
    return b;
}

static void bt_free(BiTab *b) { free(b); }

static inline void bt_add(BiTab *b, uint64_t key)
{
    uint32_t h = (uint32_t)(_mix(key) & BT_MASK);
    while (b->keys[h] != UINT64_MAX && b->keys[h] != key)
        h = (h + 1) & BT_MASK;
    if (b->keys[h] == UINT64_MAX) {
        b->keys[h] = key;
        b->size++;
    }
    b->cnts[h]++;
}

/* ── ASCII tokeniser ─────────────────────────────────────────────────────── */
/*
 * Scans [p, lim) for the next word token.  Lowercases alpha, keeps digits.
 * Writes into buf[0..MAX_WORD-1].  Returns token length, 0 if no token found.
 * Advances *pp past the token (and any trailing non-word chars).
 */
static inline int16_t next_tok(const char **pp, const char *lim,
                                char buf[MAX_WORD])
{
    const char *p = *pp;
    /* Skip non-alphanumeric */
    while (p < lim) {
        unsigned char c = (unsigned char)*p;
        if ((c | 32) >= 'a' && (c | 32) <= 'z') break;
        if (c >= '0' && c <= '9') break;
        p++;
    }
    if (p >= lim) { *pp = p; return 0; }

    int16_t n = 0;
    while (p < lim && n < MAX_WORD) {
        unsigned char c = (unsigned char)*p;
        if ((c | 32) >= 'a' && (c | 32) <= 'z') {
            buf[n++] = (char)(c | 32);   /* lowercase */
            p++;
        } else if (c >= '0' && c <= '9') {
            buf[n++] = (char)c;
            p++;
        } else break;
    }
    /* Skip remainder of an overlong token */
    while (p < lim) {
        unsigned char c = (unsigned char)*p;
        if (!((c | 32) >= 'a' && (c | 32) <= 'z') && !(c >= '0' && c <= '9')) break;
        p++;
    }
    *pp = p;
    return n;
}

/* Find start of next line after position p. */
static inline const char *next_line(const char *p, const char *end)
{
    while (p < end && *p != '\n') p++;
    return (p < end) ? p + 1 : end;
}

/* ── Round up to next power of two ─────────────────────────────────────── */
static inline uint32_t next_pow2(uint32_t v)
{
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Public API
 * ══════════════════════════════════════════════════════════════════════════ */

int32_t scan_corpus_omp(
    const char *filepath,
    int32_t     n_threads,
    int32_t     max_words,
    int32_t     max_bigrams,
    char       *out_word_buf,    /* flat NUL-terminated words, concatenated   */
    int32_t    *out_word_offs,   /* byte offset of each word in out_word_buf  */
    int16_t    *out_word_lens,   /* byte length of each word                  */
    int32_t    *out_word_cnts,   /* occurrence count of each word             */
    int32_t    *out_n_words,     /* number of unique words found              */
    int32_t    *out_bi_w1,       /* word index 1 for bigram i                 */
    int32_t    *out_bi_w2,       /* word index 2 for bigram i                 */
    int32_t    *out_bi_cnts,     /* bigram count                              */
    int32_t    *out_n_bigrams    /* number of unique bigrams found            */
)
{
    *out_n_words   = 0;
    *out_n_bigrams = 0;

    /* ── Open and mmap the corpus file ─────────────────────────────── */
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return -1; }
    size_t fsize = (size_t)st.st_size;
    if (fsize == 0) { close(fd); return 0; }

    const char *buf = (const char *)mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (buf == MAP_FAILED) return -1;
#ifdef MADV_SEQUENTIAL
    madvise((void *)buf, fsize, MADV_SEQUENTIAL);
#endif
    const char *end = buf + fsize;

    /* ── Determine thread count ─────────────────────────────────────── */
#ifdef _OPENMP
    if (n_threads <= 0) n_threads = omp_get_max_threads();
    if (n_threads > 64)  n_threads = 64;
#else
    n_threads = 1;
#endif

    /* ── Allocate per-thread WordTabs ───────────────────────────────── */
    WordTab **wtabs = (WordTab **)calloc((size_t)n_threads, sizeof(WordTab *));
    if (!wtabs) { munmap((void *)buf, fsize); return -1; }
    for (int i = 0; i < n_threads; i++) {
        wtabs[i] = wt_alloc();
        if (!wtabs[i]) {
            for (int j = 0; j < i; j++) wt_free(wtabs[j]);
            free(wtabs); munmap((void *)buf, fsize); return -1;
        }
    }

    /* ── Phase 1: Parallel unigram counting ─────────────────────────── */
    size_t chunk = fsize / (size_t)n_threads;

#pragma omp parallel num_threads(n_threads)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        size_t s_off = (size_t)tid * chunk;
        size_t e_off = (tid == n_threads - 1) ? fsize : s_off + chunk;

        const char *p   = (tid == 0) ? buf : next_line(buf + s_off, end);
        const char *lim = (tid == n_threads - 1) ? end : next_line(buf + e_off, end);

        WordTab *wt = wtabs[tid];
        char tok[MAX_WORD];

        while (p < lim) {
            /* Find end of this line */
            const char *le = p;
            while (le < lim && *le != '\n') le++;

            const char *q = p;
            while (q < le) {
                int16_t tlen = next_tok(&q, le, tok);
                if (tlen > 0) wt_add(wt, tok, tlen);
            }
            p = (le < lim) ? le + 1 : lim;
        }
    } /* end parallel phase 1 */

    /* ── Phase 2: Serial merge → global vocab ───────────────────────── */
    GlobalTab *gt = gt_alloc();
    if (!gt) {
        for (int i = 0; i < n_threads; i++) wt_free(wtabs[i]);
        free(wtabs); munmap((void *)buf, fsize); return -1;
    }

    for (int t = 0; t < n_threads; t++) {
        WordTab *wt = wtabs[t];
        for (uint32_t h = 0; h < WT_CAP; h++) {
            if (!wt->keys[h]) continue;
            gt_add(gt, wt->arena + wt->offs[h], wt->lens[h], wt->cnts[h]);
        }
        wt_free(wtabs[t]);
        wtabs[t] = NULL;
    }
    free(wtabs);

    int32_t n_words = gt->n_ids;

    /* ── Phase 3: Parallel bigram counting (second pass) ───────────── */
    BiTab **btabs = (BiTab **)calloc((size_t)n_threads, sizeof(BiTab *));
    if (!btabs) { gt_free(gt); munmap((void *)buf, fsize); return -1; }
    for (int i = 0; i < n_threads; i++) {
        btabs[i] = bt_alloc();
        if (!btabs[i]) {
            for (int j = 0; j < i; j++) bt_free(btabs[j]);
            free(btabs); gt_free(gt); munmap((void *)buf, fsize); return -1;
        }
    }

    chunk = fsize / (size_t)n_threads;

#pragma omp parallel num_threads(n_threads)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        size_t s_off = (size_t)tid * chunk;
        size_t e_off = (tid == n_threads - 1) ? fsize : s_off + chunk;

        const char *p   = (tid == 0) ? buf : next_line(buf + s_off, end);
        const char *lim = (tid == n_threads - 1) ? end : next_line(buf + e_off, end);

        BiTab *bt = btabs[tid];
        char tok[MAX_WORD];

        while (p < lim) {
            const char *le = p;
            while (le < lim && *le != '\n') le++;

            const char *q = p;
            int32_t prev_id = -1;

            while (q < le) {
                int16_t tlen = next_tok(&q, le, tok);
                if (tlen > 0) {
                    int32_t id = gt_lookup(gt, tok, tlen);
                    if (id >= 0) {
                        if (prev_id >= 0) {
                            uint64_t key = (uint64_t)prev_id * (uint64_t)n_words
                                         + (uint64_t)id;
                            bt_add(bt, key);
                        }
                        prev_id = id;
                    }
                }
            }
            p = (le < lim) ? le + 1 : lim;
        }
    } /* end parallel phase 3 */

    munmap((void *)buf, fsize);

    /* ── Phase 4: Write word output (ordered by ID) ─────────────────── */
    {
        /* Build id → hash-slot reverse mapping */
        int32_t *id2slot = (int32_t *)malloc((size_t)n_words * sizeof(int32_t));
        if (id2slot) {
            for (uint32_t h = 0; h < WT_CAP; h++) {
                if (!gt->keys[h]) continue;
                int32_t wid = gt->ids[h];
                if (wid >= 0 && wid < n_words) id2slot[wid] = (int32_t)h;
            }
            int32_t buf_off = 0, wi = 0;
            for (int32_t wid = 0; wid < n_words && wi < max_words; wid++) {
                uint32_t h  = (uint32_t)id2slot[wid];
                int16_t  wl = gt->lens[h];
                const char *ws = gt->arena + gt->offs[h];
                if (buf_off + wl + 1 > max_words * (MAX_WORD + 1)) break;
                out_word_offs[wi] = buf_off;
                out_word_lens[wi] = wl;
                out_word_cnts[wi] = gt->cnts[h];
                memcpy(out_word_buf + buf_off, ws, wl);
                out_word_buf[buf_off + wl] = '\0';
                buf_off += wl + 1;
                wi++;
            }
            *out_n_words = wi;
            free(id2slot);
        }
    }

    /* ── Phase 5: Merge bigrams and write output ─────────────────────── */
    {
        /* Use a flat global bigram hash table (serial). */
        uint32_t gbi_cap  = next_pow2((uint32_t)max_bigrams * 2);
        uint32_t gbi_mask = gbi_cap - 1;
        uint64_t *gbi_keys = (uint64_t *)malloc((size_t)gbi_cap * sizeof(uint64_t));
        int32_t  *gbi_cnts = (int32_t  *)malloc((size_t)gbi_cap * sizeof(int32_t));

        if (gbi_keys && gbi_cnts) {
            memset(gbi_keys, 0xff, (size_t)gbi_cap * sizeof(uint64_t));
            memset(gbi_cnts, 0,    (size_t)gbi_cap * sizeof(int32_t));

            for (int t = 0; t < n_threads; t++) {
                BiTab *bt = btabs[t];
                for (uint32_t h = 0; h < BT_CAP; h++) {
                    if (bt->keys[h] == UINT64_MAX) continue;
                    uint64_t key = bt->keys[h];
                    int32_t  cnt = bt->cnts[h];
                    uint32_t gh = (uint32_t)(_mix(key) & gbi_mask);
                    while (gbi_keys[gh] != UINT64_MAX && gbi_keys[gh] != key)
                        gh = (gh + 1) & gbi_mask;
                    if (gbi_keys[gh] == UINT64_MAX) gbi_keys[gh] = key;
                    gbi_cnts[gh] += cnt;
                }
                bt_free(btabs[t]);
                btabs[t] = NULL;
            }

            int32_t bi_out = 0;
            int32_t nw = *out_n_words;
            for (uint32_t h = 0; h < gbi_cap && bi_out < max_bigrams; h++) {
                if (gbi_keys[h] == UINT64_MAX) continue;
                uint64_t key = gbi_keys[h];
                int32_t  w1  = (int32_t)(key / (uint64_t)n_words);
                int32_t  w2  = (int32_t)(key % (uint64_t)n_words);
                if (w1 < nw && w2 < nw) {
                    out_bi_w1[bi_out]  = w1;
                    out_bi_w2[bi_out]  = w2;
                    out_bi_cnts[bi_out] = gbi_cnts[h];
                    bi_out++;
                }
            }
            *out_n_bigrams = bi_out;
        }
        free(gbi_keys);
        free(gbi_cnts);
    }

    for (int i = 0; i < n_threads; i++) bt_free(btabs[i]);
    free(btabs);
    gt_free(gt);
    return 0;
}

/* ── count_lines_c: fast newline count ──────────────────────────────────── */
int64_t count_lines_c(const char *filepath)
{
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return -1; }
    size_t fsize = (size_t)st.st_size;
    if (fsize == 0) { close(fd); return 0; }

    const char *buf = (const char *)mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (buf == MAP_FAILED) return -1;

    int64_t n = 0;
    const char *p = buf, *e = buf + fsize;
    while (p < e) {
        if (*p == '\n') n++;
        p++;
    }
    /* Count last line if it doesn't end in newline */
    if (fsize > 0 && buf[fsize - 1] != '\n') n++;
    munmap((void *)buf, fsize);
    return n;
}
