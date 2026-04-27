"""
Honest head-to-head: IECNN vs all-MiniLM-L6-v2 on STS-Benchmark.

Both models produce sentence embeddings. We compute cosine similarity for
each pair, then correlate against human-rated ground truth (0-5 scale).
Higher Pearson/Spearman = better at capturing how similar humans think two
sentences are.

The Transformer here is a SMALL open-source model (22M params) trained on
~1B sentence pairs. It is NOT GPT-4 / Claude. It IS the standard baseline
in semantic similarity research.

The IECNN brain has seen perhaps a few hundred sentences in toy training.
Comparison is intentionally hard for IECNN; reporting the truth either way.
"""
import csv, time, sys, os, random
import numpy as np

random.seed(0)

# ── Load STS-B and stratify by score band ─────────────────────────────
pairs = []
with open("/tmp/stsb_test.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) != 3: continue
        try:
            s1, s2, score = row[0], row[1], float(row[2])
            pairs.append((s1, s2, score))
        except ValueError:
            continue

# Stratified sample: 5 pairs from each of 5 score bands [0-1, 1-2, 2-3, 3-4, 4-5]
bands = [[] for _ in range(5)]
for p in pairs:
    b = min(4, int(p[2]))
    bands[b].append(p)
sample = []
for b in bands:
    random.shuffle(b)
    sample.extend(b[:5])

print(f"Loaded {len(pairs)} STS-B test pairs; using stratified sample of {len(sample)}")
print()

# ── Compute IECNN similarities ────────────────────────────────────────
print("=" * 70)
print("  PHASE 1: IECNN")
print("=" * 70)
t0 = time.time()
from pipeline.pipeline import IECNN
from formulas.formulas import similarity_score

iecnn = IECNN(
    feature_dim=256, num_dots=128, n_heads=4,
    max_iterations=12, evolve=True, seed=42,
    persistence_path="global_brain.pkl",
)
print(f"  IECNN loaded in {time.time() - t0:.1f}s")

iecnn_sims = []
t_iecnn_total = 0.0
for i, (s1, s2, gold) in enumerate(sample):
    t1 = time.time()
    v1 = iecnn.encode(s1)
    v2 = iecnn.encode(s2)
    # Use plain cosine on real parts (vectors may be complex post-merge)
    if np.iscomplexobj(v1):
        v1 = v1.real
    if np.iscomplexobj(v2):
        v2 = v2.real
    sim = similarity_score(v1, v2)
    dt = time.time() - t1
    t_iecnn_total += dt
    iecnn_sims.append(sim)
    print(f"  [{i+1:2d}/{len(sample)}] gold={gold:.2f}  iecnn={sim:+.4f}  ({dt:.1f}s)")

print(f"\n  IECNN total: {t_iecnn_total:.1f}s  ({t_iecnn_total/len(sample):.1f}s/pair)")

# ── Compute Transformer similarities ──────────────────────────────────
print()
print("=" * 70)
print("  PHASE 2: all-MiniLM-L6-v2 (Microsoft, 22M params)")
print("=" * 70)
t0 = time.time()
from sentence_transformers import SentenceTransformer
tr = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print(f"  Transformer loaded in {time.time() - t0:.1f}s")

# Encode all unique sentences once
unique = list({s for p in sample for s in (p[0], p[1])})
t1 = time.time()
embs = {s: e for s, e in zip(unique, tr.encode(unique, normalize_embeddings=True))}
t_tr_total = time.time() - t1

tr_sims = []
for s1, s2, gold in sample:
    sim = float(np.dot(embs[s1], embs[s2]))  # already normalized
    tr_sims.append(sim)

print(f"  Transformer total: {t_tr_total:.2f}s  ({t_tr_total/len(sample)*1000:.1f}ms/pair)")

# ── Score ─────────────────────────────────────────────────────────────
def pearson(x, y):
    x, y = np.array(x), np.array(y)
    return float(np.corrcoef(x, y)[0, 1])

def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return pearson(rx, ry)

gold = [p[2] for p in sample]
p_iecnn, s_iecnn = pearson(iecnn_sims, gold), spearman(iecnn_sims, gold)
p_tr, s_tr = pearson(tr_sims, gold), spearman(tr_sims, gold)

print()
print("=" * 70)
print("  RESULTS")
print("=" * 70)
print(f"  {'Model':<35}  {'Pearson':>8}  {'Spearman':>8}  {'Time/pair':>10}")
print(f"  {'-'*35}  {'-'*8}  {'-'*8}  {'-'*10}")
print(f"  {'IECNN':<35}  {p_iecnn:+.4f}   {s_iecnn:+.4f}    {t_iecnn_total/len(sample):>7.2f}s")
print(f"  {'all-MiniLM-L6-v2 (Transformer)':<35}  {p_tr:+.4f}   {s_tr:+.4f}    {t_tr_total/len(sample)*1000:>7.1f}ms")
print()
print(f"  Higher Pearson/Spearman = better. Range -1 to +1, where +1 is perfect.")
print(f"  Random guessing scores ~0; STS-B SOTA is ~0.92 Pearson.")
print()

# Show worst IECNN cases vs best transformer cases for honesty
print("=" * 70)
print("  Per-pair detail (sorted by gold score)")
print("=" * 70)
order = sorted(range(len(sample)), key=lambda i: sample[i][2])
print(f"  {'gold':>5}  {'iecnn':>7}  {'  tr':>7}  {'sentence pair'}")
for i in order:
    s1, s2, g = sample[i]
    snip = f"'{s1[:30]}' ↔ '{s2[:30]}'"
    print(f"  {g:5.2f}  {iecnn_sims[i]:+7.4f}  {tr_sims[i]:+7.4f}  {snip}")
