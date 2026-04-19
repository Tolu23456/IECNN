#!/usr/bin/env python3
"""
IECNN CLI — Iterative Emergent Convergent Neural Network

Usage:
  python main.py                   # interactive demo
  python main.py encode "text"     # encode a text to a vector
  python main.py sim "a" "b"       # similarity between two texts
  python main.py build             # compile C extensions only

Component folders at project root:
  aim/         basemapping/   convergence/   formulas/
  iteration/   neural_dot/    pipeline/      pruning/
"""

import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ── Build C extensions ────────────────────────────────────────────────
def _build_c():
    import subprocess
    print("[IECNN] Compiling C extensions...")
    result = subprocess.run(["bash", "build.sh"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[IECNN] C build warning:\n{result.stderr.strip()}")
        print("[IECNN] Continuing with pure Python fallbacks.")
    else:
        print("[IECNN] C extensions ready (C-accelerated paths active).")


def _c_missing() -> bool:
    return not all(os.path.exists(p) for p in [
        "formulas/formulas_c.so", "basemapping/basemapping_c.so",
        "aim/aim_c.so", "convergence/convergence_c.so",
    ])


# ── Corpus for base vocabulary discovery ──────────────────────────────
DEMO_CORPUS = [
    "neural networks learn from data",
    "iterative convergence finds stable representations",
    "attention mechanisms focus on relevant features",
    "base mapping converts text to structured matrices",
    "neural dots predict independently without weight sharing",
    "pruning removes weak candidate predictions",
    "convergence groups similar predictions into clusters",
    "the winning cluster emerges through agreement",
    "learning updates the bias vector gradually",
    "novelty gain measures exploration exhaustion",
    "dominant clusters satisfy the stopping condition",
    "aim inversion challenges prediction assumptions",
    "feature inversion flips dominant attribute signs",
    "spatial inversion reverses dimension group ordering",
    "context inversion swaps foreground and background roles",
    "scale inversion rescales groups by inverse factors",
    "abstraction inversion flips levels of understanding",
    "the system pre-seeds letters a through z as primitives",
    "unknown words compose their embedding from character bases",
    "each token maps to exactly one matrix row",
    "phrases are detected as single base units",
    "the base vocabulary grows from corpus frequency",
    "sinusoidal position encoding captures token order",
    "frequency features encode how common a token is",
    "modifier flags capture structural properties of tokens",
]


def _make_model(verbose=True):
    from pipeline.pipeline import IECNN
    if verbose:
        print("[IECNN] Initializing model (64 dots, 128-dim feature vectors)...")
    model = IECNN(feature_dim=128, num_dots=64, max_iterations=10)
    if verbose:
        print(f"[IECNN] Fitting base vocabulary on {len(DEMO_CORPUS)} corpus sentences...")
    model.fit(DEMO_CORPUS)
    if verbose:
        bm = model.base_mapper
        n_words  = sum(1 for t in bm._base_types.values() if t == "word")
        n_phrase = sum(1 for t in bm._base_types.values() if t == "phrase")
        print(f"[IECNN] Base vocab: {n_words} word bases, {n_phrase} phrase bases, "
              f"{len(bm._primitive_embeddings)} primitives (a-z + digits + punct)")
        print()
    return model


# ── Encode command ────────────────────────────────────────────────────
def cmd_encode(text: str):
    if _c_missing(): _build_c()
    model = _make_model()
    result = model.run(text, verbose=True)
    vec = result.output
    print(f"\n  Output vector (first 16 of {len(vec)} dims):")
    print("  ", np.round(vec[:16], 3))
    print(f"  Norm : {np.linalg.norm(vec):.4f}")
    print(f"  Stop : {result.stop_reason}  |  Rounds: {result.summary['rounds']}")


# ── Similarity command ────────────────────────────────────────────────
def cmd_similarity(a: str, b: str):
    if _c_missing(): _build_c()
    model = _make_model(verbose=False)
    from formulas.formulas import similarity_score
    va, vb = model.encode(a), model.encode(b)
    print(f"\n  Similarity:")
    print(f"    A: '{a}'")
    print(f"    B: '{b}'")
    print(f"  Score: {similarity_score(va, vb):+.4f}  (range: -1 to 1)")


# ── Build command ────────────────────────────────────────────────────
def cmd_build():
    _build_c()
    print("[IECNN] Build complete.")


# ── Interactive demo ──────────────────────────────────────────────────
def cmd_demo():
    if _c_missing(): _build_c()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     IECNN — Iterative Emergent Convergent Neural Network     ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Novel architecture: neural dots + convergence-based         ║")
    print("║  learning. No backpropagation. No fixed neuron layers.       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    model = _make_model(verbose=True)

    EXAMPLES = [
        "neural networks learn from data iteratively",
        "the base mapping converts words to structured matrices",
        "attention mechanisms focus on what matters most",
        "convergence finds agreement among independent predictions",
    ]

    print("─" * 64)
    print("  Running pipeline on example inputs:")
    print("─" * 64)

    results = []
    for text in EXAMPLES:
        res = model.run(text, verbose=True)
        results.append((text, res.output))
        sm = res.summary
        print(f"  → rounds={sm['rounds']}  stop='{res.stop_reason}'  "
              f"norm={np.linalg.norm(res.output):.3f}")

    print()
    print("─" * 64)
    print("  Pairwise similarity between example inputs:")
    print("─" * 64)
    from formulas.formulas import similarity_score
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            ta, va = results[i]; tb, vb = results[j]
            s = similarity_score(va, vb)
            print(f"  [{i}]×[{j}] = {s:+.4f}  |  {ta[:28]!r} ↔ {tb[:28]!r}")

    print()
    print("─" * 64)
    print("  BaseMapping demo — word types and character composition:")
    print("─" * 64)
    for test_tok in ["convergence", "iecnn", "xyz123", "a"]:
        bmap = model.base_mapper.transform(test_tok)
        typ  = bmap.token_types[0] if bmap.token_types else "?"
        norm = np.linalg.norm(bmap.matrix[0, :96])
        print(f"  '{test_tok}' → type='{typ}'  embed_norm={norm:.4f}")
    print()
    print("  BaseMapping design:")
    print("  ‣ a-z, 0-9, punctuation are PRE-SEEDED as primitive bases")
    print("  ‣ Known words from corpus → 'word' base (one row, stable embedding)")
    print("  ‣ Unknown words → 'composed' base (one row built from character bases)")
    print("  ‣ Frequent bigrams/trigrams → 'phrase' base (one row per phrase)")
    print("  ‣ Each token is ALWAYS one row — words are NEVER split into char rows")

    print()
    print("─" * 64)
    print("  Interactive — type any text to encode, 'sim A | B' for similarity,")
    print("  or 'quit' to exit.")
    print("─" * 64)
    while True:
        try:
            user = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break
        if not user: continue
        if user.lower() in ("quit", "exit", "q"):
            print("  Goodbye.")
            break
        if user.lower().startswith("sim ") and "|" in user:
            parts = user[4:].split("|", 1)
            va = model.encode(parts[0].strip())
            vb = model.encode(parts[1].strip())
            print(f"  Similarity: {similarity_score(va, vb):+.4f}")
            continue
        res = model.run(user, verbose=True)
        print(f"  → norm={np.linalg.norm(res.output):.3f}  "
              f"rounds={res.summary['rounds']}  stop='{res.stop_reason}'")


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        cmd_demo()
    elif args[0] == "build":
        cmd_build()
    elif args[0] == "encode" and len(args) >= 2:
        cmd_encode(" ".join(args[1:]))
    elif args[0] == "sim" and len(args) >= 3:
        if "|" in args:
            idx = args.index("|")
            cmd_similarity(" ".join(args[1:idx]), " ".join(args[idx+1:]))
        else:
            cmd_similarity(args[1], " ".join(args[2:]))
    else:
        print(__doc__)
