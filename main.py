#!/usr/bin/env python3
"""
IECNN CLI — Iterative Emergent Convergent Neural Network

Usage:
  python main.py                       # full interactive demo
  python main.py encode "text"         # encode text → 128-dim vector
  python main.py sim "text A" "text B" # similarity between two texts
  python main.py compare "a" "b" "c"  # n×n similarity table
  python main.py memory                # show dot memory and evolution state
  python main.py build                 # compile C extensions

Component folders:
  aim/   basemapping/  convergence/  evaluation/
  evolution/  formulas/  iteration/  memory/
  neural_dot/  pipeline/  pruning/
"""

import sys
import os
import numpy as np
from typing import List

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

BAR = "─" * 66


# ── Build C extensions ────────────────────────────────────────────────
def _build_c(force: bool = False):
    import subprocess
    need = force or not all(os.path.exists(p) for p in [
        "formulas/formulas_c.so", "basemapping/basemapping_c.so",
        "aim/aim_c.so", "convergence/convergence_c.so",
    ])
    if not need:
        return
    print("[IECNN] Compiling C extensions...")
    result = subprocess.run(["bash", "build.sh"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[IECNN] C build warning:\n{result.stderr.strip()}")
        print("[IECNN] Continuing with pure Python fallbacks.")
    else:
        print("[IECNN] C extensions compiled successfully.")


# ── Corpus ────────────────────────────────────────────────────────────
CORPUS = [
    "neural networks learn patterns from data",
    "iterative convergence finds stable representations",
    "attention mechanisms focus on relevant features",
    "base mapping converts text to structured matrices",
    "neural dots predict independently without weight sharing",
    "pruning removes weak candidate predictions",
    "convergence groups similar predictions into clusters",
    "the winning cluster emerges through agreement",
    "learning updates the bias vector gradually",
    "novelty gain measures when exploration is exhausted",
    "dominant clusters satisfy the stopping condition",
    "aim inversion challenges prediction assumptions",
    "feature inversion flips dominant attribute signs",
    "spatial inversion reverses dimension group ordering",
    "context inversion swaps foreground and background roles",
    "scale inversion rescales groups by inverse factors",
    "abstraction inversion flips levels of understanding",
    "relational inversion inverts block correlation structure",
    "temporal inversion reverses sequential dimension order",
    "compositional inversion decomposes and recombines vectors",
    "the system pre-seeds letters a through z as primitives",
    "unknown words compose their embedding from character bases",
    "each token maps to exactly one matrix row",
    "phrases are detected as single base units",
    "the base vocabulary grows from corpus frequency",
    "sinusoidal position encoding captures token order",
    "frequency features encode how common a token is",
    "modifier flags capture structural properties of tokens",
    "semantic dots attend to meaning and content",
    "structural dots focus on form and position",
    "contextual dots view the full sequence globally",
    "relational dots detect cross-token interaction patterns",
    "temporal dots use position-weighted sequential pooling",
    "global dots pool uniformly across all tokens",
    "dot evolution selects effective dots for reproduction",
    "dot memory tracks which predictions won past rounds",
    "cluster memory records temporal stability across rounds",
    "hierarchical convergence scores nested cluster structure",
    "cross-type agreement rewards consensus among dot types",
    "adaptive learning rate slows when convergence is near",
]


def _make_model(verbose: bool = True):
    from pipeline.pipeline import IECNN
    if verbose:
        print(f"[IECNN] Initializing model...")
    model = IECNN(
        feature_dim=256, num_dots=128, n_heads=4,
        max_iterations=12, evolve=True, seed=42,
        persistence_path="global_brain.pkl"
    )
    if verbose:
        print(f"[IECNN] Fitting base vocabulary on {len(CORPUS)} corpus sentences...")
    model.fit(CORPUS)
    if model.base_mapper.persistence_path:
        model.base_mapper.save(model.base_mapper.persistence_path)
    if verbose:
        bm = model.base_mapper
        n_words  = sum(1 for t in bm._base_types.values() if t == "word")
        n_phrase = sum(1 for t in bm._base_types.values() if t == "phrase")
        print(f"[IECNN] Vocab: {n_words} word bases, {n_phrase} phrase bases, "
              f"{len(bm._primitive_embeddings)} primitives (a-z + 0-9 + punct)")
        print()
    return model


# ── Commands ──────────────────────────────────────────────────────────

def cmd_build():
    _build_c(force=True)
    print("[IECNN] Build complete.")


def cmd_encode(text: str):
    _build_c()
    model = _make_model()
    res   = model.run(text, verbose=True)
    vec   = res.output
    print(f"\n{BAR}")
    print(f"  Output vector ({len(vec)} dims)")
    print(f"  First 24 dims: {np.round(vec[:24], 3)}")
    print(f"  Norm         : {np.linalg.norm(vec):.4f}")
    print(f"  Stop reason  : {res.stop_reason}")
    print(f"  Rounds       : {res.summary['rounds']}")
    if res.metrics:
        m = res.metrics
        print(f"  Quality      : {m.convergence_quality:.4f}")
        print(f"  Entropy      : {m.cluster_entropy:.4f}")
        print(f"  Stability    : {m.temporal_stability:.4f}")


def cmd_similarity(texts: List[str]):
    _build_c()
    if len(texts) < 2:
        print("Need at least two texts."); return
    model = _make_model(verbose=False)
    from formulas.formulas import similarity_score
    vecs = [(t, model.encode(t)) for t in texts]
    print(f"\n{BAR}")
    print("  Pairwise Similarity:")
    print(BAR)
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            ta, va = vecs[i]; tb, vb = vecs[j]
            s = similarity_score(va, vb)
            print(f"  {s:+.4f}  │  '{ta[:35]}' ↔ '{tb[:35]}'")


def cmd_compare(texts: List[str]):
    _build_c()
    model = _make_model(verbose=False)
    print(f"\n  Computing {len(texts)}×{len(texts)} similarity matrix...")
    mat = model.compare(texts)
    labels = [t[:20] for t in texts]
    col_w  = max(7, max(len(l) for l in labels) + 1)
    header = " " * (col_w + 2) + "  ".join(f"{l:>{col_w}}" for l in labels)
    print(f"\n{BAR}\n  Similarity Matrix\n{BAR}")
    print(header)
    for i, row in enumerate(mat):
        row_str = "  ".join(f"{v:>{col_w}.4f}" for v in row)
        print(f"  {labels[i]:<{col_w}}  {row_str}")


def cmd_memory(model=None):
    _build_c()
    if model is None: model = _make_model(verbose=False)
    status = model.memory_status()
    dm = status["dot_memory"]
    cm = status["cluster_memory"]
    ev = status["evolution"]
    print(f"\n{BAR}")
    print("  IECNN Memory & Evolution State")
    print(BAR)
    print(f"  Calls completed  : {status['call_count']}")
    print(f"  Active dots      : {dm['active_dots']} / {dm['num_dots']}")
    print(f"  Mean effectiveness: {dm['mean_eff']:.4f}")
    print(f"  Max effectiveness : {dm['max_eff']:.4f}")
    print(f"  Top 5 dots       : {dm['top5']}")
    print(f"  Cluster patterns : {cm['patterns_stored']}")
    print(f"  Temporal stability: {cm['temporal_stability']:.4f}")
    print(f"  Evolution gen    : {ev['generation']}")
    print(f"  Evo top_eff      : {ev['top_dot_eff']:.4f}")
    print(f"  Evo mean_eff     : {ev['mean_eff']:.4f}")


def cmd_demo():
    _build_c()

    print()
    print("╔" + "═"*64 + "╗")
    print("║     IECNN — Iterative Emergent Convergent Neural Network     ║")
    print("╠" + "═"*64 + "╣")
    print("║  Novel architecture: neural dots + convergence learning       ║")
    print("║  No backpropagation. No fixed layers. Emergent agreement.     ║")
    print("╠" + "═"*64 + "╣")
    print("║  Layers: Input → BaseMap → Dots (6 types, 4 heads) → AIM    ║")
    print("║          → Prune → Converge (2-level) → Iterate → Output    ║")
    print("╚" + "═"*64 + "╝")
    print()

    model = _make_model(verbose=True)

    EXAMPLES = [
        "neural networks learn from data iteratively",
        "the base mapping converts words to structured matrices",
        "attention mechanisms focus on what matters most",
        "convergence finds agreement among independent predictions",
        "dot evolution selects the most effective prediction units",
        "relational inversion discovers cross-token structure",
    ]

    print(BAR)
    print("  Full Pipeline Run — 6 example inputs")
    print(BAR)

    results = []
    for text in EXAMPLES:
        res = model.run(text, verbose=True)
        results.append((text, res.output, res))
        sm = res.summary
        m  = res.metrics
        q  = f"{m.convergence_quality:.3f}" if m else "n/a"
        print(f"  → rounds={sm['rounds']:>2}  stop='{res.stop_reason:<22}'  "
              f"norm={np.linalg.norm(res.output):.3f}  quality={q}")

    print(f"\n{BAR}")
    print("  Pairwise Similarity Matrix (6 examples)")
    print(BAR)
    from formulas.formulas import similarity_score
    labels = [t[:28] for t, _, _ in results]
    vecs   = [v for _, v, _ in results]
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            s = similarity_score(vecs[i], vecs[j])
            print(f"  [{i}]×[{j}] {s:+.4f}  │  {labels[i]!r} ↔ {labels[j]!r}")

    print(f"\n{BAR}")
    print("  BaseMapping — token type demo")
    print(BAR)
    bm = model.base_mapper
    for tok in ["convergence", "iecnn", "xyz123", "a", "neural networks", "42"]:
        bmap = bm.transform(tok)
        typ  = bmap.token_types[0] if bmap.token_types else "?"
        embed_n = np.linalg.norm(bmap.matrix[0, :224])
        print(f"  '{tok:<20}' → type='{typ:<10}'  embed_norm={embed_n:.4f}")

    print(f"\n{BAR}")
    print("  Memory & Evolution State (after 6 calls)")
    print(BAR)
    cmd_memory(model)

    print(f"\n{BAR}")
    print("  Dot Type Distribution")
    print(BAR)
    dots = model._dots or []
    dist = model.dot_gen.type_distribution(dots)
    for dt, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        bar = "█" * cnt
        print(f"  {dt:<12} {cnt:>3}  {bar}")

    print(f"\n{BAR}")
    print("  BaseMapping Design")
    print(BAR)
    print("  ‣ a-z, 0-9, punctuation → pre-seeded primitive bases (always available)")
    print("  ‣ Corpus words/phrases  → 'word'/'phrase' bases (stable embeddings)")
    print("  ‣ Unknown words         → 'composed' (ONE row, built from char bases)")
    print("  ‣ Each token = ONE row  — words are NEVER split into character rows")
    print("  ‣ Feature vector: [224 embed | 8 pos | 4 freq | 16 flags | 4 ctx]")

    print(f"\n{BAR}")
    print("  Formula Summary (F1–F17)")
    print(BAR)
    formulas = [
        ("F1",  "Similarity Score",            "S(p,q) = α·cos(p,q) + (1-α)·A(p,q)"),
        ("F2",  "Convergence Score",           "C(k) = mean_sim(k) · mean_conf(k)"),
        ("F3",  "Attention",                   "softmax(QK^T/√d_k)·V"),
        ("F4",  "AIM Transform",               "p̂ = Attention(Q, K, I(p))"),
        ("F5",  "Pruning Threshold",           "keep k ⟺ C(k) > τ"),
        ("F6",  "Prediction Confidence",       "c(p) = tanh(||p||/√d)"),
        ("F7",  "Sampling Temperature",        "P(v) ∝ exp(score(v)/T)"),
        ("F8",  "Bias Vector Update",          "b_{t+1} = b_t + η(w_t - b_t)"),
        ("F9",  "Dominance Score",             "Dominance = C(k*)/ΣC(k)"),
        ("F10", "Dot Specialization",          "mean pairwise S of a dot's predictions"),
        ("F11", "Cluster Entropy",             "H_C = -Σ p_k log p_k  [normalized]"),
        ("F12", "Temporal Stability",          "TS(t) = S(centroid_t, centroid_{t-1})"),
        ("F13", "Cross-Type Agreement",        "CDA = mean S(centroid_a, centroid_b)"),
        ("F14", "Adaptive Learning Rate",      "η(t) = η_0 · (1 - 0.8·dom²)"),
        ("F15", "Hierarchical Conv. Score",    "HC(K) = mean_score · (1+γ·cross_sim)"),
        ("F16", "Emergent Utility Gradient",   "U(t) = E[C_{t+1}(p)] - C_t(p)"),
        ("F17", "Dot Reinforcement Pressure",  "R_d = λ1·C_d + λ2·S_d + λ3·U_n·(1+β·ΔU) - λ4·N_d"),
    ]
    for fid, name, formula in formulas:
        print(f"  {fid:<4} {name:<28} {formula}")

    print(f"\n{BAR}")
    print("  Interactive — enter text to encode, 'sim A | B', 'memory', or 'quit'")
    print(BAR)
    while True:
        try:
            user = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye."); break
        if not user: continue
        if user.lower() in ("q", "quit", "exit"):
            print("  Goodbye."); break
        if user.lower() == "memory":
            cmd_memory(model); continue
        if user.lower().startswith("sim ") and "|" in user:
            parts = user[4:].split("|", 1)
            from formulas.formulas import similarity_score
            va = model.encode(parts[0].strip())
            vb = model.encode(parts[1].strip())
            print(f"  Similarity: {similarity_score(va, vb):+.4f}")
            continue
        res = model.run(user, verbose=True)
        m = res.metrics
        print(f"  → norm={np.linalg.norm(res.output):.3f}  "
              f"rounds={res.summary['rounds']}  stop='{res.stop_reason}'  "
              f"quality={m.convergence_quality:.3f if m else 'n/a'}")


# ── Entry ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        cmd_demo()
    elif args[0] == "build":
        cmd_build()
    elif args[0] == "encode" and len(args) >= 2:
        cmd_encode(" ".join(args[1:]))
    elif args[0] == "sim" and len(args) >= 2:
        # Supports: python main.py sim "a" "b"  or  python main.py sim "a" | "b"
        all_text = " ".join(args[1:])
        if "|" in all_text:
            parts = all_text.split("|", 1)
            cmd_similarity([p.strip() for p in parts])
        elif len(args) >= 3:
            cmd_similarity([args[1], " ".join(args[2:])])
        else:
            print("Usage: python main.py sim 'text A' 'text B'")
    elif args[0] == "compare" and len(args) >= 3:
        cmd_compare(args[1:])
    elif args[0] == "memory":
        _build_c(); cmd_memory()
    else:
        print(__doc__)
