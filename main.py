#!/usr/bin/env python3
"""
IECNN CLI — Iterative Emergent Convergent Neural Network

Usage:
  python main.py                        # silent interactive prompt (no output until you ask)
  python main.py train <file>           # train on a text file (vocab + dot learning), persists
  python main.py generate "prompt"      # encode prompt then decode output text
  python main.py encode "text"          # encode text → 256-dim latent vector
  python main.py sim "text A" "text B"  # similarity between two texts
  python main.py compare "a" "b" "c"    # n×n similarity table
  python main.py memory                 # show dot memory and evolution state
  python main.py demo                   # run the original 6-example showcase
  python main.py build                  # compile C extensions

  Interactive commands (available in interactive mode):
    generate <prompt>   encode prompt then decode output
    train <filepath>    train on a text file
    sim A | B           pairwise similarity
    encode <text>       encode text and show vector summary
    memory              show dot memory + evolution state
    quit / q            exit
"""

import sys
import os
import numpy as np
from typing import List

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

BAR             = "─" * 66
PERSISTENCE     = "global_brain.pkl"


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


# Minimal seed corpus used only when no global_brain.pkl exists yet, so that
# the BaseMapper has a few primitives to fall back on. This is NOT training.
_SEED_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "neural networks learn patterns from data",
    "a sentence is a sequence of words",
    "letters and numbers compose all written text",
]


def _make_model(verbose: bool = False):
    from pipeline.pipeline import IECNN
    if verbose:
        print("[IECNN] Loading model...")
    model = IECNN(
        feature_dim=256, num_dots=128, n_heads=4,
        max_iterations=12, evolve=True, seed=42,
        persistence_path=PERSISTENCE,
    )
    # Only seed the vocab if absolutely empty (first ever run, no brain on disk)
    if not model.base_mapper.is_fitted:
        model.fit(_SEED_CORPUS)
        model.save_brain()
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
    model.save_brain()


def cmd_similarity(texts: List[str]):
    _build_c()
    if len(texts) < 2:
        print("Need at least two texts."); return
    model = _make_model()
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
    model = _make_model()
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
    if model is None: model = _make_model()
    status = model.memory_status()
    dm = status["dot_memory"]
    cm = status["cluster_memory"]
    ev = status["evolution"]
    print(f"\n{BAR}")
    print("  IECNN Memory & Evolution State")
    print(BAR)
    print(f"  Calls completed   : {status['call_count']}")
    print(f"  Active dots       : {dm['active_dots']} / {dm['num_dots']}")
    print(f"  Mean effectiveness: {dm['mean_eff']:.4f}")
    print(f"  Max effectiveness : {dm['max_eff']:.4f}")
    print(f"  Top 5 dots        : {dm['top5']}")
    print(f"  Cluster patterns  : {cm['patterns_stored']}")
    print(f"  Temporal stability: {cm['temporal_stability']:.4f}")
    print(f"  Evolution gen     : {ev['generation']}")
    print(f"  Evo top_eff       : {ev['top_dot_eff']:.4f}")
    print(f"  Evo mean_eff      : {ev['mean_eff']:.4f}")


def cmd_train(filepath: str, limit: int = 0):
    _build_c()
    if not os.path.exists(filepath):
        print(f"[IECNN] Corpus not found: {filepath}")
        return
    model = _make_model()
    if limit > 0:
        # Read up to `limit` non-empty lines and write to a temp file
        tmp = filepath + f".limit{limit}.tmp"
        kept = 0
        with open(filepath, "r", encoding="utf-8", errors="replace") as src, \
             open(tmp, "w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line or line.startswith("#"): continue
                dst.write(line + "\n")
                kept += 1
                if kept >= limit:
                    break
        print(f"[IECNN] Training on first {kept} lines of {filepath}")
        model.fit_file(tmp, verbose=True)
        try: os.remove(tmp)
        except OSError: pass
    else:
        model.fit_file(filepath, verbose=True)
    print(f"[IECNN] Brain saved to {PERSISTENCE} (+ companion files).")
    cmd_memory(model)


def cmd_generate_oneshot(prompt: str):
    _build_c()
    model = _make_model()
    print(f"[IECNN] Generating from: '{prompt}'")
    out = model.generate(prompt)
    print(f"  Output: {out}")
    model.save_brain()


def _interactive_loop(model):
    """Quiet REPL: nothing happens until the user types something."""
    from formulas.formulas import similarity_score
    print(f"  IECNN ready. Type a prompt, or 'help' for commands. Ctrl-D to exit.")
    while True:
        try:
            user = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye."); break
        if not user:
            continue
        low = user.lower()
        if low in ("q", "quit", "exit"):
            print("  Goodbye."); break
        if low in ("help", "?"):
            print("  Commands:")
            print("    generate <prompt>   encode prompt then decode output")
            print("    encode <text>       encode and show vector summary")
            print("    sim A | B           pairwise similarity")
            print("    train <filepath>    train on a text file")
            print("    memory              show dot memory + evolution state")
            print("    quit / q            exit")
            continue
        if low == "memory":
            cmd_memory(model); continue
        if low.startswith("sim ") and "|" in user:
            parts = user[4:].split("|", 1)
            va = model.encode(parts[0].strip())
            vb = model.encode(parts[1].strip())
            print(f"  Similarity: {similarity_score(va, vb):+.4f}")
            model.save_brain(); continue
        if low.startswith("generate "):
            prompt = user[9:].strip()
            if not prompt: print("  Usage: generate <prompt>"); continue
            print("  [generating...]", end="", flush=True)
            out = model.generate(prompt)
            print(f"\r  Output: {out}            ")
            model.save_brain(); continue
        if low.startswith("encode "):
            text = user[7:].strip()
            if not text: print("  Usage: encode <text>"); continue
            res = model.run(text, verbose=False)
            v = res.output
            print(f"  norm={np.linalg.norm(v):.3f}  rounds={res.summary['rounds']}  "
                  f"stop='{res.stop_reason}'")
            print(f"  first8: {np.round(v[:8], 3)}")
            model.save_brain(); continue
        if low.startswith("train "):
            fpath = user[6:].strip()
            if not fpath: print("  Usage: train <filepath>"); continue
            try:
                model.fit_file(fpath, verbose=True)
            except FileNotFoundError as e:
                print(f"  Error: {e}")
            continue
        # Default: treat raw input as a prompt to generate text from
        print("  [generating...]", end="", flush=True)
        out = model.generate(user)
        print(f"\r  Output: {out}            ")
        model.save_brain()


def cmd_interactive():
    _build_c()
    model = _make_model()
    _interactive_loop(model)


def cmd_demo():
    """Original showcase: 6 examples + similarity matrix + formula dump."""
    _build_c()
    print()
    print("╔" + "═"*64 + "╗")
    print("║     IECNN — Iterative Emergent Convergent Neural Network     ║")
    print("╠" + "═"*64 + "╣")
    print("║  Novel architecture: neural dots + convergence learning      ║")
    print("║  No backpropagation. No fixed layers. Emergent agreement.    ║")
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
    print("  Memory & Evolution State (after 6 calls)")
    print(BAR)
    cmd_memory(model)
    model.save_brain()


# ── Entry ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        cmd_interactive()
    elif args[0] == "build":
        cmd_build()
    elif args[0] == "demo":
        cmd_demo()
    elif args[0] == "memory":
        cmd_memory()
    elif args[0] == "encode" and len(args) >= 2:
        cmd_encode(" ".join(args[1:]))
    elif args[0] == "generate" and len(args) >= 2:
        cmd_generate_oneshot(" ".join(args[1:]))
    elif args[0] == "sim" and len(args) >= 2:
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
    elif args[0] == "train" and len(args) >= 2:
        # python main.py train <file> [--limit N]
        filepath = args[1]
        limit = 0
        if "--limit" in args:
            i = args.index("--limit")
            if i + 1 < len(args):
                try: limit = int(args[i+1])
                except ValueError: limit = 0
        cmd_train(filepath, limit=limit)
    else:
        print(__doc__)
