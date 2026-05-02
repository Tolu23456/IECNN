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
  python main.py chat                   # interactive chatbot mode
  python main.py memory                 # show dot memory and evolution state
  python main.py prune [--dry-run]      # compact the brain (drop dead dots + orphans)
                       [--min-outcomes N] [--min-age N]
  python main.py train <file> [--limit N] [--evolve] [--causal] [--prune-every N]
                       [--fast] [--workers N] [--shared-memory]
                                        # vocab-only by default; --evolve runs
                                        # full pipeline; --causal runs
                                        # next-token prediction training;
                                        # --prune-every keeps disk size bounded;
                                        # --shared-memory (with --fast) eliminates
                                        # text-pickling IPC via shared memory,
                                        # targeting >1M sent/s throughput
  python main.py demo                   # run the original 6-example showcase
  python main.py build                  # compile C extensions
  python main.py rebuild                # re-embed vocab with char n-gram embeddings
                                        # (run before calibrate for best quality)

  Global flag (any subcommand or interactive mode):
  --phase-coding      enable IECNN-native phase-coherence binding so cluster
                      memory can distinguish patterns with the same content
                      but different positions ("dog bites man" vs
                      "man bites dog"). Persists in the brain meta file.

  Interactive commands (available in interactive mode):
    generate <prompt>   encode prompt then decode output
    chat                enter conversational chatbot mode
    train <filepath>    train on a text file
    sim A | B           pairwise similarity
    encode <text>       encode text and show vector summary
    memory              show dot memory + evolution state
    prune               compact the brain (drop dead dots + orphans)
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


# ── All C extension shared libraries ─────────────────────────────────
_ALL_SO_FILES = [
    "formulas/formulas_c.so",
    "basemapping/basemapping_c.so",
    "aim/aim_c.so",
    "convergence/convergence_c.so",
    "pruning/pruning_c.so",
    "neural_dot/neural_dot_c.so",
    "decoding/decoder_c.so",
    "fast_count_c.so",
    "fast_scan_c.so",
    "pipeline/pipeline_c.so",
]

# ── Build C extensions ────────────────────────────────────────────────
def _build_c(force: bool = False):
    import subprocess
    missing = [p for p in _ALL_SO_FILES if not os.path.exists(p)]
    if not force and not missing:
        return

    if missing:
        print("[IECNN] Missing C extensions:")
        for p in missing:
            print(f"  ✗  {p}")

    print("[IECNN] Compiling C extensions (build.sh)...")
    result = subprocess.run(["bash", "build.sh"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[IECNN] Build FAILED:\n{result.stderr.strip()}")
        still_missing = [p for p in _ALL_SO_FILES if not os.path.exists(p)]
        if still_missing:
            print("[IECNN] The following C extensions are still unavailable:")
            for p in still_missing:
                print(f"  ✗  {p}")
            print("[IECNN] Performance will be significantly degraded without C acceleration.")
            print("[IECNN] Ensure gcc/openmp are installed then re-run: python main.py build")
    else:
        built = [p for p in _ALL_SO_FILES if os.path.exists(p)]
        print(f"[IECNN] C extensions ready: {len(built)}/{len(_ALL_SO_FILES)}")


# Minimal seed corpus used only when no global_brain.pkl exists yet, so that
# the BaseMapper has a few primitives to fall back on. This is NOT training.
_SEED_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "neural networks learn patterns from data",
    "a sentence is a sequence of words",
    "letters and numbers compose all written text",
]


_PHASE_CODING = False


def _make_model(verbose: bool = False):
    from pipeline.pipeline import IECNN
    if verbose:
        print("[IECNN] Loading model...")
    model = IECNN(
        feature_dim=256, num_dots=128, n_heads=4,
        max_iterations=12, evolve=True, seed=42,
        persistence_path=PERSISTENCE,
        phase_coding=_PHASE_CODING,
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
    print(f"  Rounds       : {res.summary.get('rounds', res.rounds)}")
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


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0:
            return f"{n:6.1f} {unit}"
        n /= 1024.0
    return f"{n:6.1f} TB"


def _brain_files() -> list:
    """Return [(path, size_bytes)] for all brain companion files that exist."""
    suffixes = ["", ".dots.pkl", ".dotmem.pkl", ".clustmem.pkl",
                ".evo.pkl", ".meta.pkl"]
    out = []
    for s in suffixes:
        p = PERSISTENCE + s
        if os.path.exists(p):
            out.append((p, os.path.getsize(p)))
    return out


def cmd_memory(model=None):
    _build_c()
    if model is None: model = _make_model()
    status = model.memory_status()
    dm = status["dot_memory"]
    cm = status["cluster_memory"]
    ev = status["evolution"]
    pool_size = len(model._dots) if model._dots else 0
    print(f"\n{BAR}")
    print("  IECNN Memory & Evolution State")
    print(BAR)
    print(f"  Calls completed   : {status['call_count']}")
    print(f"  Live dot pool     : {pool_size}")
    print(f"  Active dots       : {dm['active_dots']} / {dm['num_dots']}")
    print(f"  Mean effectiveness: {dm['mean_eff']:.4f}")
    print(f"  Max effectiveness : {dm['max_eff']:.4f}")
    print(f"  Top 5 dots        : {dm['top5']}")
    print(f"  Cluster patterns  : {cm['patterns_stored']}")
    print(f"  Temporal stability: {cm['temporal_stability']:.4f}")
    print(f"  Evolution gen     : {ev['generation']}")
    print(f"  Evo top_eff       : {ev['top_dot_eff']:.4f}")
    print(f"  Evo mean_eff      : {ev['mean_eff']:.4f}")
    files = _brain_files()
    if files:
        total = sum(sz for _, sz in files)
        print(f"\n  Brain on disk     : {_human_bytes(total).strip()} total")
        for p, sz in files:
            print(f"    {os.path.basename(p):<32} {_human_bytes(sz)}")
    # Dry-run prune preview so the user can see how much would be reclaimed
    preview = model.prune_dots(dry_run=True)
    if preview["removed_dots"] or preview["removed_history"]:
        print(f"\n  Prune preview     : would drop {preview['removed_dots']} dots "
              f"and {preview['removed_history']} history records")
        print(f"                      (run `prune` to apply)")


def cmd_prune(dry_run: bool = False, min_outcomes: int = 2, min_age_gens: int = 2):
    _build_c()
    model = _make_model()
    before_files = _brain_files()
    before_total = sum(sz for _, sz in before_files)

    stats = model.prune_dots(min_outcomes=min_outcomes,
                              min_age_gens=min_age_gens,
                              dry_run=dry_run)

    print(f"\n{BAR}")
    print(f"  IECNN Brain Prune  {'(DRY RUN)' if dry_run else ''}")
    print(BAR)
    print(f"  Generation        : {stats['generation']}")
    print(f"  min_outcomes      : {min_outcomes}")
    print(f"  min_age_gens      : {min_age_gens}")
    print(f"  Dots removed      : {stats['removed_dots']}  (kept {stats['kept_dots']})")
    print(f"  History removed   : {stats['removed_history']}  (kept {stats['kept_history']})")
    print(f"  Brain on disk now : {_human_bytes(before_total).strip()}")

    if dry_run:
        print(f"\n  Dry run — no files modified. Re-run without --dry-run to apply.")
        return

    if stats['removed_dots'] == 0 and stats['removed_history'] == 0:
        print(f"  Nothing to prune; brain is already compact.")
        return

    model.save_brain()
    after_files = _brain_files()
    after_total = sum(sz for _, sz in after_files)
    saved = before_total - after_total
    print(f"  Brain on disk new : {_human_bytes(after_total).strip()} "
          f"(saved {_human_bytes(saved).strip()})")


def cmd_train(filepath: str, limit: int = 0, evolve: bool = False,
              causal: bool = False, prune_every: int = 0,
              fast: bool = False, workers: int = None,
              shared_memory: bool = False, ultra: bool = False):
    _build_c()
    if not os.path.exists(filepath):
        print(f"[IECNN] Corpus not found: {filepath}")
        return
    model = _make_model()

    def _read_lines(path, cap=0):
        out = []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"): continue
                out.append(line)
                if cap and len(out) >= cap: break
        return out

    if ultra:
        import fast_train as _ft
        if causal:
            # ── OMP vocab scan + per-dot causal training → MaxEff > 0.5 ──
            print(f"[IECNN] Effective training (OMP vocab + causal): {filepath}")
            _ft.fast_effective_train(
                corpus_path=filepath,
                brain_path=PERSISTENCE,
                n_threads=0,
                n_sentences=limit if limit > 0 else 20_000,
                verbose=True,
            )
        else:
            # ── OMP single-call corpus scan — maximum counting throughput ─
            print(f"[IECNN] OMP ultra-fast vocab scan: {filepath}")
            _ft.fast_vocab_train_omp(
                corpus_path=filepath,
                brain_path=PERSISTENCE,
                n_threads=0,
                verbose=True,
            )
        return

    if fast:
        # ── Ultra-fast parallel training path ──────────────────────────
        import fast_train as _ft
        import multiprocessing as mp
        n_workers = workers or min(mp.cpu_count(), 8)

        if evolve:
            print(f"[IECNN] Fast full-pipeline training: {filepath}  "
                  f"(workers={n_workers})")
            _ft.fast_full_train(
                corpus_path=filepath,
                brain_path=PERSISTENCE,
                n_workers=n_workers,
                verbose=True,
            )
        else:
            print(f"[IECNN] Fast vocab training: {filepath}  "
                  f"(workers={n_workers}"
                  f"{', shmem' if shared_memory else ''})")
            _ft.fast_vocab_train(
                corpus_path=filepath,
                brain_path=PERSISTENCE,
                n_workers=n_workers,
                verbose=True,
                use_shmem=shared_memory,
            )
        return

    if causal:
        sentences = _read_lines(filepath, limit)
        print(f"[IECNN] Causal next-token training on {len(sentences)} lines"
              f"{f' (prune every {prune_every})' if prune_every else ''}")
        model.causal_train_pass(sentences, verbose=True, prune_every=prune_every)
    elif evolve:
        sentences = _read_lines(filepath, limit)
        print(f"[IECNN] Full-pipeline training on {len(sentences)} lines"
              f"{f' (prune every {prune_every})' if prune_every else ''}")
        model.train_pass(sentences, verbose=True, prune_every=prune_every)
    elif limit > 0:
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


def _print_cgen_result(model, result, tag):
    """Shared pretty-printer for cgen / nbest results."""
    import numpy as _np
    tokens = result["tokens"]
    confs  = result["confidences"]
    reason = result["stop_reason"]

    if not tokens:
        print(f"\r  Output: (empty — {reason})                      ")
        return

    print(f"\r  Output: {result['text']}                      ")
    print()

    import numpy as _np2
    avg_conf   = sum(confs) / len(confs) if confs else 0.0
    min_conf   = min(confs) if confs else 0.0
    max_conf   = max(confs) if confs else 0.0
    # Recency-weighted confidence: later tokens weight more (linear ramp)
    if confs:
        _rw = _np2.arange(1, len(confs) + 1, dtype=float)
        rw_conf = float(_np2.average(confs, weights=_rw))
    else:
        rw_conf = 0.0
    coherence    = result.get("coherence",     0.0)
    diversity    = result.get("diversity",     0.0)
    shannon_div  = result.get("shannon_div",   0.0)
    fluency      = result.get("fluency",       0.0)
    avg_entropy  = result.get("avg_entropy",   0.0)
    coh3         = result.get("coh3",          0.0)
    coh6         = result.get("coh6",          0.0)
    vocab_prec   = result.get("vocab_prec",    0.0)
    conf_var     = result.get("conf_variance",  0.0)
    conf_trend   = result.get("conf_trend",    0.0)
    pseudo_ppl   = result.get("pseudo_ppl",    0.0)
    score_gap    = result.get("avg_score_gap", 0.0)
    ctx_vel_mag  = result.get("ctx_vel_mag",   0.0)
    peep_hit     = result.get("peep_hit_rate", 0.0)
    tok_spread   = result.get("token_embed_spread", 0.0)
    rhythm       = result.get("rhythm_score",   0.0)
    anc_str      = result.get("anchor_strength", 0.0)
    coh_dir      = result.get("coh_direction",  0.0)
    prompt_type  = result.get("prompt_type",    "?")
    vocab_size   = result.get("vocab_size",    "?")
    coh_trend    = result.get("coh_trend",      0.0)
    sg_slope     = result.get("sg_slope",       0.0)
    vprec_ema    = result.get("vocab_prec_ema", 0.0)
    conf_decl    = result.get("conf_declining", False)

    # Per-token confidence bar visualization (5-block ASCII bar)
    # Each confidence is mapped to 0–5 filled blocks: ░ = 0, █████ = 1.0
    _BAR_FULL  = "█"
    _BAR_EMPTY = "░"
    _BAR_LEN   = 5
    def _conf_bar(c: float) -> str:
        filled = int(round(c * _BAR_LEN))
        filled = max(0, min(_BAR_LEN, filled))
        return _BAR_FULL * filled + _BAR_EMPTY * (_BAR_LEN - filled)

    # Low-confidence tokens get a ~ prefix so they stand out visually.
    # Threshold: 0.25 — below this the model is quite uncertain about the token.
    token_line = " ".join(
        f"{'~' if conf < 0.25 else ''}{tok}[{_conf_bar(conf)}]"
        for tok, conf in zip(tokens, confs)
    )

    # N-best scores (only present for nbest command)
    nbest_scores = result.get("n_best_scores")

    ptype = result.get("_ptype", "statement")
    print(f"  [{tag}]  Pipeline: TFS·z0.97  Nucleus·adaptive  "
          f"Eta·ε3e-4  MinP·adaptive  Mirostat·τ0.38+warmup  TwoPass·D32"
          f"  Prompt:{ptype}")

    peep_ready = model.peep is not None and model.peep.calibrated
    if peep_ready:
        ctx_eff = model._get_ctx_cached(result.get("_prompt", ""))
        if ctx_eff is not None:
            top3    = model.peep.top_k(ctx_eff, _np.zeros((128, 256)), k=3)
            ctx_n   = ctx_eff / (float(_np.linalg.norm(ctx_eff)) + 1e-9)
            top3_cs = model.peep.specialisations[top3] @ ctx_n
            dot_str = "  ".join(f"d{d}({cs:.2f})" for d, cs in zip(top3, top3_cs))
            print(f"  Peep top-3: {dot_str}")

    print(f"  Confidence: {token_line}")
    # Quality composite score (higher is better)
    _qscore = avg_conf * (0.80 + 0.20 * max(coherence, 0.0)) * (0.70 + 0.30 * diversity)
    if _qscore > 0.50:       quality = "Excellent"
    elif _qscore > 0.35:     quality = "Good"
    elif _qscore > 0.20:     quality = "Fair"
    else:                    quality = "Poor  ← run rebuild + calibrate"

    # Confidence histogram: 5 bins [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1.]
    _HIST_CHARS = " ▁▂▃▄▅▆▇█"
    _bins = [0] * 5
    for c in confs:
        _b = min(4, int(c * 5))
        _bins[_b] += 1
    _peak = max(_bins) if _bins else 1
    _hist_str = "".join(_HIST_CHARS[max(1, int(8 * b / (_peak + 1e-6)))]
                        for b in _bins)

    print(f"  Avg:{avg_conf:.3f}  RW:{rw_conf:.3f}  Min:{min_conf:.3f}  Max:{max_conf:.3f}  "
          f"Coh:{coherence:+.3f}  C3:{coh3:+.3f}  C6:{coh6:+.3f}  "
          f"Div:{diversity:.2f}  Sha:{shannon_div:.2f}  "
          f"Flu:{fluency:.2f}  H̄:{avg_entropy:.2f}  VPrec:{vocab_prec:.2f}  "
          f"Var:{conf_var:.3f}  Trend:{conf_trend:+.4f}  "
          f"Gap:{score_gap:.3f}  PPL:{pseudo_ppl:.1f}  "
          f"Vel:{ctx_vel_mag:.3f}  PHit:{peep_hit:.2f}  Spread:{tok_spread:.3f}  Rhythm:{rhythm:.2f}  "
          f"Anc:{anc_str:.3f}  CohDir:{coh_dir:+.3f}  PType:{prompt_type}  "
          f"CohTrend:{coh_trend:+.4f}  SGSlope:{sg_slope:+.4f}  "
          f"VPrecEMA:{vprec_ema:.3f}  {'[!DECLINING]' if conf_decl else ''}  "
          f"Hist:[{_hist_str}]  "
          f"Toks:{len(tokens)}  Vocab:{vocab_size}  Stop:{reason}")
    print(f"  Quality: {quality}  (score={_qscore:.3f})")

    if nbest_scores:
        scores_str = "  ".join(f"{s:.4f}" for s in nbest_scores)
        print(f"  N-best scores (ranked): {scores_str}")

    if not peep_ready:
        print(f"  (run 'calibrate' to activate Peep+MHC·70/30 guided generation)")


def _run_cgen(model, prompt: str):
    """Run causal_generate and print verbose per-token output."""
    peep_ready = model.peep is not None and model.peep.calibrated
    tag = "Peep+MHC·70/30" if peep_ready else "Contrastive·MHC"
    print(f"  [{tag}] thinking...", end="", flush=True)
    result = model.causal_generate(prompt)
    result["_prompt"] = prompt
    model._last_gen_result = result   # stash for topconfs / scorehist
    _print_cgen_result(model, result, tag)
    model.save_brain()


def _run_nbest(model, prompt: str, n: int = 3):
    """Run causal_generate_nbest and print the best result with all scores."""
    peep_ready = model.peep is not None and model.peep.calibrated
    tag = f"N-best·{n}  Peep+MHC·70/30" if peep_ready else f"N-best·{n}  Contrastive·MHC"
    print(f"  [{tag}] thinking ({n} candidates)...", end="", flush=True)
    result = model.causal_generate_nbest(prompt, n=n)
    result["_prompt"] = prompt
    _print_cgen_result(model, result, tag)
    model.save_brain()


def cmd_rebuild():
    """Re-embed all vocabulary words with FastText-style char n-gram embeddings.

    Replaces the old hash-based vectors with n-gram vectors so morphologically
    related words become cosine-similar (run/running, good/goodness, etc.).
    Vocabulary, co-occurrence data, and dot W-matrices are left unchanged.

    After rebuild, run 'calibrate' to realign the dot W-matrices with the
    new embedding space — this is required for good generation quality.
    """
    _build_c()
    model = _make_model()

    print(f"\n{BAR}")
    print("  IECNN — Vocab Embedding Rebuild (char n-gram / FastText-style)")
    print(BAR)
    print(f"  Vocab size  : {len(model.base_mapper._base_vocab):,} tokens")

    import time
    t0 = time.time()
    n = model.base_mapper.rebuild_vocab_embeddings(verbose=True)
    elapsed = time.time() - t0
    print(f"  Rebuilt     : {n:,} embeddings in {elapsed:.1f}s")

    # Verify morphological similarity improvement
    import numpy as np
    vocab = model.base_mapper._base_vocab
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    print()
    print("  Morphological similarity check (should be ≥ 0.20):")
    checks = [
        ("run", "running"),  ("walk", "walked"),
        ("nation", "national"), ("good", "goodness"),
    ]
    for a, b in checks:
        if a in vocab and b in vocab:
            c = cos(vocab[a], vocab[b])
            mark = "OK" if c >= 0.20 else "low"
            print(f"    {a:10} / {b:10} = {c:+.3f}  {mark}")
        else:
            print(f"    {a}/{b}: not in vocab (needs training first)")

    model._ctx_cache.clear()
    model.save_brain()
    print()
    print("  Saved. Run 'calibrate' next to realign dot W-matrices.")
    print(BAR)


def cmd_calibrate(corpus_path: str = "/tmp/corpus_300k.txt",
                  limit: int = 0):
    """
    Run a Peep calibration pass over a corpus.

    Reads the corpus line by line, runs causal_train_pass in calibration
    mode (ctx cache cleared first so recency-weighted vectors are used),
    then saves the trained Peep to disk alongside the brain.

    After this command, cgen will use Peep+Grammar generation instead of
    majority-vote.
    """
    _build_c()
    model = _make_model()

    if not os.path.exists(corpus_path):
        print(f"  Corpus not found: {corpus_path}")
        print(f"  Usage: calibrate [corpus_path]")
        return

    print(f"\n{BAR}")
    print("  IECNN — Peep Mechanism Calibration")
    print(f"  Corpus : {corpus_path}")
    print(BAR)

    model._ctx_cache.clear()

    sentences = []
    with open(corpus_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                sentences.append(line)
                if limit and len(sentences) >= limit:
                    break

    print(f"  Lines loaded : {len(sentences):,}")
    print(f"  Running causal pass to build Peep specialisations...")
    model.causal_train_pass(sentences, verbose=True)

    if model.peep is not None:
        stats = model.peep.stats()
        div   = stats.get("diversity", 0.0)
        div_ok = "healthy" if div >= 0.35 else "needs re-calibration"
        print(f"\n  Peep calibration complete:")
        print(f"    Active dots : {stats['active_dots']} / 128")
        print(f"    Total hits  : {stats['total_hits']:,}")
        print(f"    Max hits    : {stats['max_hits']:,}")
        print(f"    Diversity   : {div:.4f}  ({div_ok})")
        print(f"    Top 5 dots  : {stats['top5_dots']}")
    else:
        print("  Warning: Peep was not created (no training positions?)")

    model.save_brain()
    print(f"  Saved. Run 'cgen <prompt>' to use Peep+Grammar generation.")
    print(BAR)


def cmd_stats(model=None):
    """Print a comprehensive brain statistics summary."""
    _build_c()
    if model is None:
        model = _make_model()
    import numpy as _np

    print(f"\n{BAR}")
    print("  IECNN — Brain Statistics")
    print(BAR)

    # Vocabulary
    vocab      = model.base_mapper._base_vocab
    word_freq  = getattr(model.base_mapper, "_word_freq", {})
    base_types = getattr(model.base_mapper, "_base_types", {})
    n_words    = sum(1 for t in base_types.values() if t == "word")
    n_phrases  = sum(1 for t in base_types.values() if t == "phrase")
    n_subwords = len(vocab) - n_words - n_phrases
    freq_vals  = list(word_freq.values())
    print(f"  Vocabulary  : {len(vocab):>8,} total tokens")
    print(f"               {n_words:>8,} words  |  {n_phrases:>6,} phrases  |  {n_subwords:>6,} subwords")
    if freq_vals:
        print(f"  Freq stats  : min={min(freq_vals)}  "
              f"max={max(freq_vals):,}  "
              f"median={sorted(freq_vals)[len(freq_vals)//2]:,}")

    # Dots
    dots = model._ensure_dots()
    print(f"\n  Neural Dots : {len(dots):>3} / 128")
    if dots:
        w_norms = [float(_np.linalg.norm(d.W)) for d in dots]
        print(f"  W-norm      : min={min(w_norms):.3f}  "
              f"max={max(w_norms):.3f}  "
              f"mean={_np.mean(w_norms):.3f}")

    # Peep
    if model.peep is not None:
        pstats = model.peep.stats()
        calib  = "calibrated" if model.peep.calibrated else "NOT calibrated"
        print(f"\n  Peep        : {calib}")
        print(f"  Active dots : {pstats['active_dots']} / 128")
        print(f"  Total hits  : {pstats['total_hits']:,}")
        print(f"  Diversity   : {pstats.get('diversity',0.0):.4f}")
        print(f"  Top-5 dots  : {pstats['top5_dots']}")
    else:
        print(f"\n  Peep        : not built  (run calibrate)")

    # Ctx cache
    cache_sz = len(model._ctx_cache)
    print(f"\n  Ctx cache   : {cache_sz} entries")

    # Memory / evolution
    mem = model.memory_status()
    dm  = mem.get("dot_memory", {})
    ev  = mem.get("evolution",  {})
    print(f"  Call count  : {mem.get('call_count',0):,}")
    print(f"  Dot memory  : {dm.get('total_records',0):,} records  "
          f"|  {dm.get('active_dots',0)} active dots")
    print(f"  Generations : {ev.get('generations',0)}")

    # Filter chain
    print(f"\n  Filter chain: TopK(k=40,adaptive) → Typical(p=0.95) →")
    print(f"                TFS(z=0.97) → Nucleus(p=0.92→0.87) →")
    print(f"                Eta(ε=3e-4) → MinP(0.05)")
    print(f"  Mirostat    : τ=0.38 (warmup 0.85→0.65 over 6 steps, lr=0.08)")
    print(f"  Path A (Peep calibrated) : Peep·70 + MHC·30, head z-score")
    print(f"  Path B (uncalibrated)    : Contrastive MHC + dropout 10%")
    print(BAR)


def cmd_generate_oneshot(prompt: str):
    _build_c()
    model = _make_model()
    print(f"[IECNN] Generating from: '{prompt}'")
    out = model.generate(prompt)
    print(f"  Output: {out}")
    model.save_brain()


def cmd_cgen_oneshot(prompt: str):
    _build_c()
    model = _make_model()
    print(f"[IECNN] IECNN-native generation from: '{prompt}'")
    _run_cgen(model, prompt)


def cmd_chat(model=None):
    _build_c()
    if model is None: model = _make_model()
    print(f"\n{BAR}")
    print("  IECNN Chatbot Mode (Autoregressive Token-by-Token)")
    print("  Type 'exit' to return to main menu.")
    print(BAR)
    history = []
    while True:
        try:
            msg = input("\n  User: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in ("exit", "quit", "q"):
            break
        if not msg:
            continue

        print("  Bot: ", end="", flush=True)
        response = model.chat(msg, history=history, verbose=False)
        print(response)
        history.append((msg, response))
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
            print("    cgen <prompt>        generate (Contrastive·MHC or Peep+MHC·70/30 if calibrated)")
            print("    nbest <prompt>       best-of-3 reranked generation (picks highest coherence·conf)")
            print("    nbest5 <prompt>      same but 5 candidates")
            print("    nbest10 <prompt>     same but 10 candidates")
            print("    rebuild              re-embed vocab with char n-gram embeddings (run before calibrate)")
            print("    calibrate [path]     build Peep specialisations from corpus (enables Peep+MHC)")
            print("    generate <prompt>    encode prompt then beam-search decode")
            print("    encode <text>        encode and show vector summary")
            print("    sim A | B            pairwise similarity")
            print("    train <filepath>     train on a text file")
            print("    stats                comprehensive brain statistics")
            print("    memory               show dot memory + evolution state")
            print("    prune [dry]          compact the brain (drop dead dots + orphans)")
            print("    ctxvec <prompt>      show top-10 vocab words nearest context direction")
            print("    topwords [N]         show top-N vocab words by dot-outcome frequency")
            print("    peepstats            show Peep dot specialisation diagnostics")
            print("    compare A | B | C    pairwise similarity matrix")
            print("    histgen <prompt>     generate 5 variations, show side-by-side")
            print("    seedgen [N] <prompt> N seeded runs ranked by quality score")
            print("    topconfs             top-5 highest-confidence steps from last gen")
            print("    scorehist            ASCII bar chart of per-step score gaps")
            print("    confgraph [N]        sparkline of per-step confidence values")
            print("    dotscores            dot top-1 agreement vote table (last gen step)")
            print("    rhythmgraph          per-step confidence delta sparkline (▲▼─)")
            print("    gensummary           one-line quality summary of last generation")
            print("    analyze              brain diagnostics: dots, vocab, W-stack norms")
            print("    diffgen A | B        run two prompts, compare stats side-by-side")
            print("    trendgen [N] <prmt>  run same prompt N times, compare conf_ema trend")
            print("    confema              conf EMA vs raw avg, variance, sparkline")
            print("    qualplot             bar chart of N-best quality history (last 8)")
            print("    cohplot [N]          per-step coh3 trajectory sparkline")
            print("    velplot [N]          per-step context velocity EMA sparkline")
            print("    sfbplot [N]          per-step SFB strength sparkline")
            print("    speedrun <prompt>    run prompt at 5/10/15/20/25 max_tokens")
            print("    fulldiag             all trajectory sparklines for last generation")
            print("    benchprompts         run 5 standard prompts, compare metrics")
            print("    pctplot [N]          per-step score-percentile of chosen token")
            print("    repdiag              repetition / uniqueness diagnostic for last gen")
            print("    tempplot [N]         per-step temperature sparkline")
            print("    tokenmap             ASCII token-length profile of last generation")
            print("    gencompare <prompt>  causal_generate vs nbest side-by-side")
            print("    topconfgen [N] <pr>  run N times, auto-pick highest conf_ema")
            print("    pivotgen <w> | <pr>  generate with forced-first-token pivot")
            print("    coh6plot [N]         per-step 6-token coherence window sparkline")
            print("    stresstest           run 10 built-in prompts, report avg metrics")
            print("    varplot [N]          per-step score variance sparkline")
            print("    genheatmap           score histogram heat map for last generation")
            print("    topkplot [N]         per-step adaptive TopK k value sparkline")
            print("    confmatrix           2-D confidence × position heat map")
            print("    confoverlay [N]      raw confidence + EMA overlay sparklines")
            print("    entropyplot [N]      per-step H_norm entropy trajectory sparkline")
            print("    marginlog [N]        per-step top-1 score margin sparkline")
            print("    top3log [N]          per-step top-3 candidate tokens from last gen")
            print("    quit / q             exit")
            continue
        if low in ("stats",):
            cmd_stats(model); continue
        if low == "memory":
            cmd_memory(model); continue
        if low in ("prune", "prune --dry-run", "prune dry-run", "prune dry"):
            dry = "dry" in low
            stats = model.prune_dots(dry_run=dry)
            tag = " (dry run)" if dry else ""
            print(f"  Prune{tag}: dropped {stats['removed_dots']} dots, "
                  f"{stats['removed_history']} history records "
                  f"(kept {stats['kept_dots']} dots).")
            if not dry and (stats['removed_dots'] or stats['removed_history']):
                model.save_brain()
            continue
        if low.startswith("sim ") and "|" in user:
            parts = user[4:].split("|", 1)
            va = model.encode(parts[0].strip())
            vb = model.encode(parts[1].strip())
            print(f"  Similarity: {similarity_score(va, vb):+.4f}")
            model.save_brain(); continue
        if low.startswith("seedgen "):
            # seedgen [N] <prompt> — N seeded runs, show all + best by composite
            _sg_parts = user[8:].split(None, 1)
            if len(_sg_parts) == 2 and _sg_parts[0].isdigit():
                _sg_n, _sg_prompt = int(_sg_parts[0]), _sg_parts[1].strip()
            else:
                _sg_n, _sg_prompt = 5, user[8:].strip()
            if not _sg_prompt: print("  Usage: seedgen [N] <prompt>"); continue
            import numpy as _np_sg, random as _rnd_sg
            print(f"\n  seedgen: {_sg_n} seeded runs of '{_sg_prompt}'")
            print(f"  {'#':<3}  {'Text':<52}  Avg    Coh    Flu    PPL    Q")
            print(f"  {'-'*3}  {'-'*52}  -----  -----  -----  -----  -----")
            _sg_best_score = -1e9
            _sg_best_text  = ""
            for _si in range(_sg_n):
                _rnd_sg.seed(_si * 71 + 13)
                _np_sg.random.seed(_si * 71 + 13)
                _sr = model.causal_generate(_sg_prompt)
                _np_sg.random.seed(None); _rnd_sg.seed(None)
                _scs = _sr.get("confidences", [])
                _savg = float(_np_sg.mean(_scs)) if _scs else 0.0
                _scoh = _sr.get("coherence", 0.0)
                _sflu = _sr.get("fluency",   0.0)
                _sppl = _sr.get("pseudo_ppl", 0.0)
                _sq   = _savg * (0.80 + 0.20 * max(_scoh, 0.0)) * (0.50 + 0.50 * _sflu)
                _stxt = _sr.get("text", "")[:52]
                _best_mark = " *" if _sq > _sg_best_score else ""
                if _sq > _sg_best_score:
                    _sg_best_score = _sq
                    _sg_best_text  = _sr.get("text", "")
                print(f"  {_si+1:<3}  {_stxt:<52}  {_savg:.3f}  {_scoh:+.3f}  {_sflu:.3f}  {_sppl:6.1f}  {_sq:.3f}{_best_mark}")
            print(f"\n  Best: {_sg_best_text}")
            model.save_brain(); continue

        if low.startswith("histgen "):
            # histgen <prompt>  — generate 5 variations and show them side-by-side
            _hg_prompt = user[8:].strip()
            if not _hg_prompt: print("  Usage: histgen <prompt>"); continue
            import numpy as _np_hg, random as _rnd_hg
            _HG_N = 5
            print(f"\n  histgen: {_HG_N} variations of '{_hg_prompt}'")
            print(f"  {'#':<3}  {'Text':<55}  Avg   Coh   Flu   PPL")
            print(f"  {'-'*3}  {'-'*55}  -----  ----  ----  -----")
            for _hi in range(_HG_N):
                _rnd_hg.seed(_hi * 53 + 7)
                _np_hg.random.seed(_hi * 53 + 7)
                _hr = model.causal_generate(_hg_prompt)
                _np_hg.random.seed(None); _rnd_hg.seed(None)
                _hcs = _hr.get("confidences", [])
                _havg = float(_np_hg.mean(_hcs)) if _hcs else 0.0
                _hcoh = _hr.get("coherence", 0.0)
                _hflu = _hr.get("fluency", 0.0)
                _hppl = _hr.get("pseudo_ppl", 0.0)
                _htxt = _hr.get("text", "")[:55]
                print(f"  {_hi+1:<3}  {_htxt:<55}  {_havg:.3f}  {_hcoh:+.3f}  {_hflu:.3f}  {_hppl:6.1f}")
            model.save_brain(); continue

        if low == "peepstats":
            # peepstats  — show Peep specialisation diagnostics
            peep = model.peep
            if peep is None or not peep.calibrated:
                print("  Peep not calibrated. Run 'calibrate' first.")
                continue
            import numpy as _np_ps
            specs = peep.specialisations   # (D, embed_dim) unit-norm
            n_dots = specs.shape[0]
            # Pairwise cosine similarity between adjacent specialisations
            _ps_sims = []
            for _pi in range(min(n_dots - 1, 15)):
                _ps_sims.append(float(_np_ps.dot(specs[_pi], specs[_pi + 1])))
            _ps_mean = float(_np_ps.mean(_ps_sims)) if _ps_sims else 0.0
            _ps_min  = float(_np_ps.min(_ps_sims))  if _ps_sims else 0.0
            _ps_max  = float(_np_ps.max(_ps_sims))  if _ps_sims else 0.0
            # Variance of each dot's specialisation vector (spread / confidence)
            _ps_norms = _np_ps.linalg.norm(specs, axis=1)  # should all be ~1
            print(f"\n  Peep Specialisation Stats ({n_dots} dots calibrated)")
            print(f"  Adj-cosine mean:{_ps_mean:+.4f}  min:{_ps_min:+.4f}  max:{_ps_max:+.4f}")
            print(f"  Spec norm  mean:{_ps_norms.mean():.4f}  std:{_ps_norms.std():.4f}")
            print(f"  (lower adj-cos = more diverse dot specialisations)")
            continue

        if low == "analyze":
            # analyze — brain diagnostics: dot sim distribution, vocab coverage
            import numpy as _np_an
            _an_W  = model.W_stack          # (D, EMBED_DIM, EMBED_DIM) or (D, 256, 256)
            _an_wv = model.base_mapper
            _an_vocab = list(getattr(_an_wv, "_base_vocab", {}).keys())
            _an_V = len(_an_vocab)
            _an_D = model.dots.shape[0] if hasattr(model, "dots") else 0
            print(f"\n  Brain analysis:")
            print(f"  Dots: {_an_D}  Vocab: {_an_V}")
            # Dot norms
            if hasattr(model, "dots"):
                _an_norms = _np_an.linalg.norm(model.dots, axis=1)
                print(f"  Dot norms  mean:{_an_norms.mean():.4f}  "
                      f"std:{_an_norms.std():.4f}  "
                      f"min:{_an_norms.min():.4f}  max:{_an_norms.max():.4f}")
            # W-stack norms
            if hasattr(model, "W_stack"):
                _an_Wn = _np_an.linalg.norm(
                    _an_W.reshape(_an_W.shape[0], -1), axis=1)
                print(f"  W-stack norms  mean:{_an_Wn.mean():.4f}  "
                      f"std:{_an_Wn.std():.4f}  "
                      f"min:{_an_Wn.min():.4f}  max:{_an_Wn.max():.4f}")
            # Word-vec coverage
            if hasattr(model, "word_vecs"):
                _an_wvn = _np_an.linalg.norm(model.word_vecs, axis=1)
                _an_zero = int((_an_wvn < 1e-6).sum())
                print(f"  Word-vec norms  mean:{_an_wvn.mean():.4f}  "
                      f"zero-vecs:{_an_zero}/{_an_V}")
                # Vocab coverage: tokens with non-trivial embedding
                _an_cov = int((_an_wvn >= 0.01).sum())
                print(f"  Vocab coverage: {_an_cov}/{_an_V} "
                      f"({100*_an_cov/max(_an_V,1):.1f}%)")
            # Last gen peep hit rate
            _an_res = getattr(model, "_last_gen_result", None)
            if _an_res:
                print(f"  Last gen  Coh3:{_an_res.get('coherence',0):.3f}  "
                      f"Rhythm:{_an_res.get('rhythm_score',0):.2f}  "
                      f"CohDir:{_an_res.get('coh_direction',0):+.3f}  "
                      f"Anc:{_an_res.get('anchor_strength',0):.3f}  "
                      f"Stop:{_an_res.get('stop_reason','?')}")
            continue

        if low == "qualplot":
            # qualplot — bar chart of N-best quality scores from recent generations
            _qp_hist = getattr(model, "_quality_history", None)
            if not _qp_hist:
                print("  No quality history yet — run a few nbest or seedgen prompts first.")
                continue
            import numpy as _np_qp
            _qp_scores = [e["score"] for e in _qp_hist]
            _qp_texts  = [e["text"]  for e in _qp_hist]
            _qp_max    = max(_qp_scores) if _qp_scores else 1.0
            print(f"\n  N-best quality history ({len(_qp_scores)} entries):")
            print(f"  {'#':<4}  {'Score':<8}  {'Bar':<24}  Text")
            for _qi, (_qs, _qt) in enumerate(zip(_qp_scores, _qp_texts)):
                _qbar_len = int(20 * _qs / max(_qp_max, 1e-9))
                _qbar = "█" * _qbar_len + "░" * (20 - _qbar_len)
                _qmark = " ←best" if _qs == _qp_max else ""
                print(f"  {_qi+1:<4}  {_qs:<8.4f}  {_qbar}  {_qt}{_qmark}")
            print(f"  mean:{float(_np_qp.mean(_qp_scores)):.4f}  "
                  f"std:{float(_np_qp.std(_qp_scores)):.4f}  "
                  f"best:{_qp_max:.4f}")
            continue

        if low == "gensummary":
            # gensummary — one-liner summary of the last generation's key metrics
            _gs_res = getattr(model, "_last_gen_result", None)
            if _gs_res is None:
                print("  No generation yet. Run a prompt first."); continue
            import numpy as _np_gs
            _gs_c  = _gs_res.get("confidences", [])
            _gs_av = float(_np_gs.mean(_gs_c)) if _gs_c else 0.0
            _gs_coh= _gs_res.get("coherence",   0.0)
            _gs_flu= _gs_res.get("fluency",     0.0)
            _gs_ppl= _gs_res.get("pseudo_ppl",  0.0)
            _gs_rhy= _gs_res.get("rhythm_score",0.0)
            _gs_spr= _gs_res.get("token_embed_spread", 0.0)
            _gs_gap= _gs_res.get("avg_score_gap",0.0)
            _gs_tks= len(_gs_res.get("tokens", []))
            _gs_stp= _gs_res.get("stop_reason", "?")
            _gs_txt= _gs_res.get("text", "")[:60]
            _gs_q  = _gs_av * (0.80 + 0.20 * max(_gs_coh, 0.0)) * (0.70 + 0.30 * _gs_flu)
            _gs_ql = ("Excellent" if _gs_q > 0.50 else "Good" if _gs_q > 0.35
                      else "Fair" if _gs_q > 0.20 else "Poor")
            print(f"\n  Generation summary:")
            print(f"  Text  : {_gs_txt}")
            print(f"  Qual  : {_gs_ql} ({_gs_q:.3f})  "
                  f"Avg:{_gs_av:.3f}  Coh:{_gs_coh:+.3f}  Flu:{_gs_flu:.3f}")
            print(f"  PPL:{_gs_ppl:.1f}  Rhythm:{_gs_rhy:.2f}  "
                  f"Spread:{_gs_spr:.3f}  Gap:{_gs_gap:.3f}  "
                  f"Toks:{_gs_tks}  Stop:{_gs_stp}")
            continue

        if low == "confema":
            # confema — compare conf_ema_final vs raw avg confidence from last gen
            _ce_res = getattr(model, "_last_gen_result", None)
            if _ce_res is None:
                print("  No generation yet. Run a prompt first."); continue
            import numpy as _np_ce
            _ce_confs  = _ce_res.get("confidences", [])
            _ce_avg    = float(_np_ce.mean(_ce_confs)) if _ce_confs else 0.0
            _ce_ema    = _ce_res.get("conf_ema_final", 0.0)
            _ce_var    = _ce_res.get("conf_variance",  0.0)
            _ce_trend  = _ce_res.get("conf_trend",     0.0)
            _ce_rhythm = _ce_res.get("rhythm_score",   0.0)
            _ce_vprec  = _ce_res.get("vocab_prec_ema", 0.0)
            print(f"\n  Confidence EMA summary:")
            print(f"  Raw avg       : {_ce_avg:.4f}")
            print(f"  EMA (α=0.35)  : {_ce_ema:.4f}  "
                  f"(diff={_ce_ema-_ce_avg:+.4f})")
            print(f"  Variance      : {_ce_var:.4f}  "
                  f"Trend: {_ce_trend:+.5f}")
            print(f"  Rhythm        : {_ce_rhythm:.3f}  "
                  f"VPrecEMA: {_ce_vprec:.3f}")
            # Mini bar
            _SPARKS2 = " ▁▂▃▄▅▆▇█"
            _spark2 = "".join(_SPARKS2[min(8, int(c * 8))] for c in _ce_confs)
            print(f"  Sparkline     : {_spark2}")
            continue

        if low == "rhythmgraph":
            # rhythmgraph — show per-step confidence delta signs as sparkline
            _rg_res = getattr(model, "_last_gen_result", None)
            if _rg_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _rg_confs = _rg_res.get("confidences", [])
            _rg_toks  = _rg_res.get("tokens", [])
            if len(_rg_confs) < 2:
                print("  Need at least 2 steps for rhythm graph."); continue
            import numpy as _np_rg
            _rg_deltas = [_rg_confs[i] - _rg_confs[i-1] for i in range(1, len(_rg_confs))]
            _rg_rhythm = float(_rg_res.get("rhythm_score", 0.0))
            _UP   = "▲"
            _DOWN = "▼"
            _FLAT = "─"
            _rg_sparks = "".join(
                _UP if d > 0.005 else _DOWN if d < -0.005 else _FLAT
                for d in _rg_deltas
            )
            _alts = sum(1 for i in range(1, len(_rg_deltas))
                        if _rg_deltas[i] * _rg_deltas[i-1] < 0)
            print(f"\n  Confidence rhythm ({len(_rg_confs)} steps, rhythm={_rg_rhythm:.3f}):")
            print(f"  {_rg_sparks}")
            print(f"  {'Step':<5}  {'Token':<18}  Conf    Delta")
            for _ri, (conf, tok) in enumerate(zip(_rg_confs, _rg_toks)):
                _delta = _rg_deltas[_ri - 1] if _ri > 0 else 0.0
                _mark  = _UP if _delta > 0.005 else _DOWN if _delta < -0.005 else _FLAT
                print(f"  {_ri:<5}  {tok[:18]:<18}  {conf:.3f}   {_delta:+.3f} {_mark}")
            print(f"  Alternations: {_alts}/{len(_rg_deltas)-1}  rhythm_score={_rg_rhythm:.3f}")
            continue

        if low.startswith("marginlog"):
            # marginlog [N] — top-1 margin sparkline from last generation
            _ml_res = getattr(model, "_last_gen_result", None)
            if _ml_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ml_margins = _ml_res.get("top1_margins", [])
            _ml_toks    = _ml_res.get("tokens", [])
            if not _ml_margins:
                print("  No margin log in last result."); continue
            _ml_parts = low.split()
            _ml_n = (int(_ml_parts[1]) if len(_ml_parts) > 1
                     and _ml_parts[1].isdigit() else len(_ml_margins))
            import numpy as _np_ml
            _ml_arr  = _np_ml.array(_ml_margins[:_ml_n], dtype=float)
            _ml_max  = float(_ml_arr.max()) if _ml_arr.max() > 0 else 1.0
            _SPARKS3 = " ▁▂▃▄▅▆▇█"
            _ml_spark = "".join(
                _SPARKS3[min(8, int(m / _ml_max * 8))] for m in _ml_arr
            )
            print(f"\n  Top-1 score margin (step decisiveness, {len(_ml_arr)} steps):")
            print(f"  {_ml_spark}")
            print(f"  {'Step':<5}  {'Token':<18}  Margin  Bar")
            for _mi, (_mg, _mt) in enumerate(
                    zip(_ml_margins[:_ml_n],
                        _ml_toks + [""] * _ml_n)):
                _mbar_len = int(20 * _mg / _ml_max)
                _mbar = "█" * _mbar_len + "░" * (20 - _mbar_len)
                _hi   = "  ←high" if _mg == _ml_max else ""
                print(f"  {_mi:<5}  {_mt[:18]:<18}  {_mg:.4f}  {_mbar}{_hi}")
            print(f"  avg={_ml_arr.mean():.4f}  max={_ml_max:.4f}  "
                  f"min={_ml_arr.min():.4f}")
            continue

        if low.startswith("cohplot"):
            # cohplot [N] — sparkline + bar chart of per-step coh3 values
            _cp_res = getattr(model, "_last_gen_result", None)
            if _cp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cp_vals = _cp_res.get("coh3_steps", [])
            _cp_toks = _cp_res.get("tokens", [])
            if not _cp_vals:
                print("  No per-step coh3 data (need ≥3 tokens)."); continue
            _cp_parts = low.split()
            _cp_n = (int(_cp_parts[1]) if len(_cp_parts) > 1
                     and _cp_parts[1].isdigit() else len(_cp_vals))
            import numpy as _np_cp
            _cp_arr = _np_cp.array(_cp_vals[:_cp_n], dtype=float)
            _SPARKS4 = " ▁▂▃▄▅▆▇█"
            _cp_spark = "".join(_SPARKS4[min(8, int(c * 8))] for c in _cp_arr)
            print(f"\n  Per-step coh3 ({len(_cp_arr)} steps):")
            print(f"  {_cp_spark}")
            print(f"  {'Step':<5}  {'Token':<18}  Coh3   Bar")
            _cp_max = float(_cp_arr.max()) if len(_cp_arr) else 1.0
            for _cpi, (_cv, _ct) in enumerate(
                    zip(_cp_vals[:_cp_n], _cp_toks + [""] * _cp_n)):
                _cblen = int(20 * max(_cv, 0.0))
                _cbar = "█" * _cblen + "░" * (20 - _cblen)
                print(f"  {_cpi:<5}  {_ct[:18]:<18}  {_cv:.3f}  {_cbar}")
            print(f"  avg={_cp_arr.mean():.3f}  "
                  f"min={_cp_arr.min():.3f}  max={_cp_max:.3f}")
            continue

        if low.startswith("entropyplot"):
            # entropyplot [N] — sparkline + bar chart of per-step H_norm values
            _ep_res = getattr(model, "_last_gen_result", None)
            if _ep_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ep_vals = _ep_res.get("entropy_steps", [])
            _ep_toks = _ep_res.get("tokens", [])
            if not _ep_vals:
                print("  No per-step entropy data."); continue
            _ep_parts = low.split()
            _ep_n = (int(_ep_parts[1]) if len(_ep_parts) > 1
                     and _ep_parts[1].isdigit() else len(_ep_vals))
            import numpy as _np_ep
            _ep_arr = _np_ep.array(_ep_vals[:_ep_n], dtype=float)
            _SPARKS5 = " ▁▂▃▄▅▆▇█"
            _ep_spark = "".join(_SPARKS5[min(8, int(e * 8))] for e in _ep_arr)
            print(f"\n  Per-step entropy H_norm ({len(_ep_arr)} steps):")
            print(f"  {_ep_spark}")
            print(f"  {'Step':<5}  {'Token':<18}  H_norm Bar")
            for _epi, (_ev, _et) in enumerate(
                    zip(_ep_vals[:_ep_n], _ep_toks + [""] * _ep_n)):
                _eblen = int(20 * max(_ev, 0.0))
                _ebar = "█" * _eblen + "░" * (20 - _eblen)
                print(f"  {_epi:<5}  {_et[:18]:<18}  {_ev:.3f}  {_ebar}")
            print(f"  avg={_ep_arr.mean():.3f}  "
                  f"min={_ep_arr.min():.3f}  max={_ep_arr.max():.3f}")
            continue

        if low.startswith("top3log"):
            # top3log [N] — show per-step top-3 candidates from last generation
            _t3_res = getattr(model, "_last_gen_result", None)
            if _t3_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _t3_log  = _t3_res.get("_top3_log", [])
            _t3_toks = _t3_res.get("tokens", [])
            if not _t3_log:
                print("  No top-3 log in last result."); continue
            _t3_parts = low.split()
            _t3_n = int(_t3_parts[1]) if len(_t3_parts) > 1 and _t3_parts[1].isdigit() else len(_t3_log)
            print(f"\n  Per-step top-3 candidates (last generation, first {_t3_n} steps):")
            print(f"  {'Step':<5}  {'Chosen':<18}  "
                  f"{'#1 token':<18}  #1sc  {'#2 token':<18}  #2sc  "
                  f"{'#3 token':<18}  #3sc")
            for _t3i, (_step_top3, _chosen) in enumerate(
                    zip(_t3_log[:_t3_n], _t3_toks + [""] * _t3_n)):
                if not _step_top3: continue
                _r = [("?", 0.0)] * 3
                for _ri, (_sc, _tok) in enumerate(_step_top3[:3]):
                    _r[_ri] = (_tok, _sc)
                print(f"  {_t3i:<5}  {_chosen[:18]:<18}  "
                      f"{_r[0][0][:18]:<18}  {_r[0][1]:.3f}  "
                      f"{_r[1][0][:18]:<18}  {_r[1][1]:.3f}  "
                      f"{_r[2][0][:18]:<18}  {_r[2][1]:.3f}")
            continue

        if low.startswith("velplot"):
            # velplot [N] — sparkline of per-step context velocity magnitude
            _vp_res = getattr(model, "_last_gen_result", None)
            if _vp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _vp_vals = _vp_res.get("velocity_steps", [])
            _vp_toks = _vp_res.get("tokens", [])
            if not _vp_vals:
                print("  No velocity data in last result."); continue
            _vp_parts = low.split()
            _vp_n = (int(_vp_parts[1]) if len(_vp_parts) > 1
                     and _vp_parts[1].isdigit() else len(_vp_vals))
            import numpy as _np_vp
            _vp_arr = _np_vp.array(_vp_vals[:_vp_n], dtype=float)
            _vp_max = float(_vp_arr.max()) if _vp_arr.max() > 0 else 1.0
            _SPARKS6 = " ▁▂▃▄▅▆▇█"
            _vp_spark = "".join(
                _SPARKS6[min(8, int(v / _vp_max * 8))] for v in _vp_arr
            )
            print(f"\n  Context velocity EMA ({len(_vp_arr)} steps):")
            print(f"  {_vp_spark}")
            print(f"  {'Step':<5}  {'Token':<18}  VelEMA  Bar")
            for _vpi, (_vv, _vt) in enumerate(
                    zip(_vp_vals[:_vp_n], _vp_toks + [""] * _vp_n)):
                _vblen = int(20 * _vv / _vp_max)
                _vbar = "█" * _vblen + "░" * (20 - _vblen)
                print(f"  {_vpi:<5}  {_vt[:18]:<18}  {_vv:.4f}  {_vbar}")
            print(f"  avg={_vp_arr.mean():.4f}  max={_vp_max:.4f}  "
                  f"min={_vp_arr.min():.4f}")
            continue

        if low.startswith("sfbplot"):
            # sfbplot [N] — sparkline of per-step SFB strength
            _sb_res = getattr(model, "_last_gen_result", None)
            if _sb_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _sb_vals = _sb_res.get("sfb_strength_steps", [])
            _sb_toks = _sb_res.get("tokens", [])
            if not _sb_vals:
                print("  No SFB strength data in last result."); continue
            _sb_parts = low.split()
            _sb_n = (int(_sb_parts[1]) if len(_sb_parts) > 1
                     and _sb_parts[1].isdigit() else len(_sb_vals))
            import numpy as _np_sb
            _sb_arr = _np_sb.array(_sb_vals[:_sb_n], dtype=float)
            _sb_max = float(_sb_arr.max()) if _sb_arr.max() > 0 else 0.30
            _SPARKS7 = " ▁▂▃▄▅▆▇█"
            _sb_spark = "".join(
                _SPARKS7[min(8, int(s / _sb_max * 8))] for s in _sb_arr
            )
            print(f"\n  SFB strength per step ({len(_sb_arr)} steps):")
            print(f"  {_sb_spark}")
            print(f"  {'Step':<5}  {'Token':<18}  SFB    Bar")
            for _sbi, (_sv, _st) in enumerate(
                    zip(_sb_vals[:_sb_n], _sb_toks + [""] * _sb_n)):
                _sblen = int(20 * _sv / _sb_max)
                _sbar = "█" * _sblen + "░" * (20 - _sblen)
                print(f"  {_sbi:<5}  {_st[:18]:<18}  {_sv:.4f}  {_sbar}")
            print(f"  avg={_sb_arr.mean():.4f}  max={_sb_max:.4f}  "
                  f"min={_sb_arr.min():.4f}")
            continue

        if low.startswith("speedrun"):
            # speedrun [<prompt>] — run same prompt at 5,10,15,20,25 max_tokens
            _sr_rest = cmd[len("speedrun"):].strip()
            if not _sr_rest:
                print("  Usage: speedrun <prompt>"); continue
            import numpy as _np_sr
            _sr_lens = [5, 10, 15, 20, 25]
            print(f"\n  Speedrun: '{_sr_rest[:40]}' at {_sr_lens} max_tokens")
            print(f"  {'MaxTok':<7}  {'Toks':<5}  {'ConfEMA':<8}  "
                  f"{'Coh':<6}  {'PPL':<7}  {'Rhythm':<7}  Text")
            for _srl in _sr_lens:
                _srr = model.causal_generate(_sr_rest, max_tokens=_srl)
                _srt = len(_srr.get("tokens", []))
                _src = _srr.get("conf_ema_final", 0.0)
                _srch = _srr.get("coherence", 0.0)
                _srp = _srr.get("pseudo_ppl", 0.0)
                _srr2 = _srr.get("rhythm_score", 0.0)
                _srtx = _srr.get("text", "")[:30]
                print(f"  {_srl:<7}  {_srt:<5}  {_src:<8.4f}  "
                      f"{_srch:<6.3f}  {_srp:<7.1f}  {_srr2:<7.3f}  {_srtx}")
            continue

        if low == "fulldiag":
            # fulldiag — run all per-step trajectory diagnostics in sequence
            _fd_res = getattr(model, "_last_gen_result", None)
            if _fd_res is None:
                print("  No generation yet. Run a prompt first."); continue
            import numpy as _np_fd
            _fd_toks = _fd_res.get("tokens", [])
            _fd_text = _fd_res.get("text", "")[:60]
            print(f"\n  === FULLDIAG for: '{_fd_text}' ===")
            _SPKFD = " ▁▂▃▄▅▆▇█"
            def _fd_plot(label, key, scale=None):
                vals = _fd_res.get(key, [])
                if not vals:
                    print(f"  [{label}] no data"); return
                arr = _np_fd.array(vals, dtype=float)
                mx  = float(arr.max()) if arr.max() > 0 else 1.0
                sc  = scale if scale else mx
                spark = "".join(_SPKFD[min(8, int(v / sc * 8))] for v in arr)
                print(f"  {label:<14} μ={arr.mean():.3f}  "
                      f"σ={arr.std():.3f}  [{arr.min():.3f}–{arr.max():.3f}]")
                print(f"  {spark}")
            _fd_plot("Coh3",        "coh3_steps",      scale=1.0)
            _fd_plot("Entropy",     "entropy_steps",   scale=1.0)
            _fd_plot("Velocity",    "velocity_steps")
            _fd_plot("SFB strength","sfb_strength_steps")
            _fd_plot("Margins",     "top1_margins")
            _fd_plot("Percentile",  "percentile_steps", scale=1.0)
            _fd_sc = _fd_res.get("score_hist", [])
            if _fd_sc:
                print(f"  Score hist: " +
                      "  ".join(f"[{i/8:.2f}-{(i+1)/8:.2f}]:{v}"
                                for i, v in enumerate(_fd_sc) if v > 0))
            print(f"  conf_ema={_fd_res.get('conf_ema_final',0):.4f}  "
                  f"coherence={_fd_res.get('coherence',0):.3f}  "
                  f"rhythm={_fd_res.get('rhythm_score',0):.3f}  "
                  f"coh_dir={_fd_res.get('coh_direction',0):.3f}  "
                  f"tokens={len(_fd_toks)}")
            continue

        if low.startswith("benchprompts"):
            # benchprompts — run 5 standard prompts, compare quality metrics
            import numpy as _np_bp
            _bp_prompts = [
                "the purpose of language is",
                "science reveals that",
                "when we consider the nature of",
                "memory and identity are",
                "the future depends on",
            ]
            print(f"\n  Benchprompts: {len(_bp_prompts)} standard prompts")
            print(f"  {'#':<3}  {'ConfEMA':<8}  {'Coh':<6}  "
                  f"{'Rhythm':<7}  {'CohDir':<7}  {'Toks':<5}  Prompt → Text")
            _bp_scores = []
            for _bpi, _bpp in enumerate(_bp_prompts):
                _bpr = model.causal_generate(_bpp, max_tokens=16)
                _bpc = _bpr.get("conf_ema_final", 0.0)
                _bph = _bpr.get("coherence",      0.0)
                _bprh = _bpr.get("rhythm_score",  0.0)
                _bpcd = _bpr.get("coh_direction", 0.0)
                _bpt  = len(_bpr.get("tokens", []))
                _bptx = _bpr.get("text", "")[:28]
                _bp_scores.append(_bpc)
                print(f"  {_bpi+1:<3}  {_bpc:<8.4f}  {_bph:<6.3f}  "
                      f"{_bprh:<7.3f}  {_bpcd:<7.3f}  {_bpt:<5}  "
                      f"'{_bpp[:20]}' → '{_bptx}'")
            _bp_arr = _np_bp.array(_bp_scores, dtype=float)
            print(f"\n  ConfEMA: avg={_bp_arr.mean():.4f}  "
                  f"std={_bp_arr.std():.4f}  "
                  f"best={_bp_arr.max():.4f}  worst={_bp_arr.min():.4f}")
            continue

        if low.startswith("pctplot"):
            # pctplot [N] — sparkline of per-step score percentile of chosen token
            _pp_res = getattr(model, "_last_gen_result", None)
            if _pp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _pp_vals = _pp_res.get("percentile_steps", [])
            _pp_toks = _pp_res.get("tokens", [])
            if not _pp_vals:
                print("  No percentile data in last result."); continue
            _pp_parts = low.split()
            _pp_n = (int(_pp_parts[1]) if len(_pp_parts) > 1
                     and _pp_parts[1].isdigit() else len(_pp_vals))
            import numpy as _np_pp
            _pp_arr = _np_pp.array(_pp_vals[:_pp_n], dtype=float)
            _SPARKS8 = " ▁▂▃▄▅▆▇█"
            _pp_spark = "".join(_SPARKS8[min(8, int(v * 8))] for v in _pp_arr)
            print(f"\n  Per-step score percentile ({len(_pp_arr)} steps):")
            print(f"  {_pp_spark}")
            print(f"  {'Step':<5}  {'Token':<18}  Pctile  Bar")
            for _ppi, (_pv, _pt) in enumerate(
                    zip(_pp_vals[:_pp_n], _pp_toks + [""] * _pp_n)):
                _pblen = int(20 * max(_pv, 0.0))
                _pbar = "█" * _pblen + "░" * (20 - _pblen)
                print(f"  {_ppi:<5}  {_pt[:18]:<18}  {_pv:.3f}   {_pbar}")
            print(f"  avg={_pp_arr.mean():.3f}  "
                  f"min={_pp_arr.min():.3f}  max={_pp_arr.max():.3f}")
            continue

        if low == "dotscores":
            # dotscores — show which vocab tokens most dots top-1 predicted last step
            _ds_res = getattr(model, "_last_gen_result", None)
            if _ds_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ds_dt1 = _ds_res.get("_last_dot_top1", [])
            if not _ds_dt1:
                print("  No dot top-1 data in last result."); continue
            import numpy as _np_ds
            _ds_dt1_arr = _np_ds.array(_ds_dt1, dtype=int)
            _ds_vocab   = list(model.base_mapper._base_vocab.keys())
            _ds_V       = len(_ds_vocab)
            _ds_counts  = _np_ds.bincount(_ds_dt1_arr, minlength=_ds_V)
            _ds_top10   = _np_ds.argsort(_ds_counts)[::-1][:10]
            print(f"\n  Dot top-1 agreement (last generation step, {len(_ds_dt1)} dots):")
            print(f"  {'Token':<22}  Votes  Bar")
            for _di in _ds_top10:
                if _ds_counts[_di] == 0: break
                _dbar_len = int(20 * _ds_counts[_di] / max(_ds_counts))
                _dbar     = "█" * _dbar_len + "░" * (20 - _dbar_len)
                _dtok     = _ds_vocab[_di] if _di < _ds_V else "?"
                print(f"  {_dtok:<22}  {_ds_counts[_di]:<6}  {_dbar}")
            continue

        if low.startswith("tempplot"):
            # tempplot [N] — sparkline of per-step temperature used during sampling
            _tp_res = getattr(model, "_last_gen_result", None)
            if _tp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _tp_vals = _tp_res.get("temp_steps", [])
            _tp_toks = _tp_res.get("tokens", [])
            if not _tp_vals:
                print("  No temperature data in last result."); continue
            _tp_parts = low.split()
            _tp_n = (int(_tp_parts[1]) if len(_tp_parts) > 1
                     and _tp_parts[1].isdigit() else len(_tp_vals))
            import numpy as _np_tp
            _tp_arr = _np_tp.array(_tp_vals[:_tp_n], dtype=float)
            _tp_max = float(_tp_arr.max()) if _tp_arr.max() > 0 else 0.90
            _SPARKSTP = " ▁▂▃▄▅▆▇█"
            _tp_spark = "".join(
                _SPARKSTP[min(8, int(v / _tp_max * 8))] for v in _tp_arr
            )
            print(f"\n  Per-step temperature ({len(_tp_arr)} steps):")
            print(f"  {_tp_spark}")
            print(f"  {'Step':<5}  {'Token':<18}  Temp   Bar")
            for _tpi, (_tv, _tt) in enumerate(
                    zip(_tp_vals[:_tp_n], _tp_toks + [""] * _tp_n)):
                _tblen = int(20 * _tv / _tp_max)
                _tbar  = "█" * _tblen + "░" * (20 - _tblen)
                print(f"  {_tpi:<5}  {_tt[:18]:<18}  {_tv:.4f}  {_tbar}")
            print(f"  avg={_tp_arr.mean():.4f}  min={_tp_arr.min():.4f}  "
                  f"max={_tp_max:.4f}")
            continue

        if low.startswith("tokenmap"):
            # tokenmap — ASCII grid showing token length profile of last generation
            _tm_res = getattr(model, "_last_gen_result", None)
            if _tm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _tm_toks = _tm_res.get("tokens", [])
            if not _tm_toks:
                print("  No tokens in last result."); continue
            # Classify each token by char length
            _tm_cats = []
            for _tt in _tm_toks:
                _l = len(_tt)
                if _l <= 2:   _tm_cats.append("S")   # short (filler)
                elif _l <= 4: _tm_cats.append("M")   # medium
                elif _l <= 7: _tm_cats.append("L")   # long
                else:          _tm_cats.append("X")  # extra-long
            _tm_row_w = 40
            print(f"\n  Tokenmap: S=short(≤2)  M=med(3-4)  L=long(5-7)  X=extra(≥8)")
            print(f"  {len(_tm_toks)} tokens total:")
            for _row_s in range(0, len(_tm_cats), _tm_row_w):
                _chunk = _tm_cats[_row_s:_row_s + _tm_row_w]
                _row   = "".join(
                    "\033[33mS\033[0m" if c == "S" else
                    "\033[32mM\033[0m" if c == "M" else
                    "\033[36mL\033[0m" if c == "L" else
                    "\033[35mX\033[0m" for c in _chunk
                )
                print(f"  {_row}")
            from collections import Counter as _Ctr
            _tm_cnt = _Ctr(_tm_cats)
            print(f"  S:{_tm_cnt['S']}  M:{_tm_cnt['M']}  "
                  f"L:{_tm_cnt['L']}  X:{_tm_cnt['X']}")
            continue

        if low.startswith("gencompare"):
            # gencompare <prompt> — causal_generate vs causal_generate_nbest side-by-side
            _gc_prompt = cmd[len("gencompare"):].strip()
            if not _gc_prompt:
                print("  Usage: gencompare <prompt>"); continue
            print(f"\n  Running gencompare: '{_gc_prompt[:40]}' ...")
            import numpy as _np_gc
            _gc_sg = model.causal_generate(_gc_prompt, max_tokens=16)
            _gc_nb = model.causal_generate_nbest(_gc_prompt, n=3, max_tokens=16)
            def _gc_fmt(r, label):
                _c = r.get("confidences", [])
                print(f"\n  [{label}]")
                print(f"    Text:      {r.get('text','')[:55]}")
                print(f"    Tokens:    {len(_c)}")
                print(f"    ConfEMA:   {r.get('conf_ema_final', 0.0):.4f}")
                print(f"    Coherence: {r.get('coherence', 0.0):.4f}")
                print(f"    Rhythm:    {r.get('rhythm_score', 0.0):.4f}")
                print(f"    CohDir:    {r.get('coh_direction', 0.0):.4f}")
                print(f"    PPL:       {r.get('pseudo_ppl', 0.0):.1f}")
                if _c:
                    _ca = _np_gc.array(_c, dtype=float)
                    print(f"    ConfAvg:   {_ca.mean():.4f}  "
                          f"min:{_ca.min():.4f}  max:{_ca.max():.4f}")
            _gc_fmt(_gc_sg, "causal_generate (single)")
            _gc_fmt(_gc_nb, "causal_generate_nbest (n=3)")
            continue

        if low == "repdiag":
            # repdiag — repetition/uniqueness diagnostic for last generation
            _rd_res = getattr(model, "_last_gen_result", None)
            if _rd_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _rd_toks = _rd_res.get("tokens", [])
            if not _rd_toks:
                print("  No tokens in last result."); continue
            from collections import Counter as _Counter
            _rd_cnt  = _Counter(_rd_toks)
            _rd_uniq = sum(1 for c in _rd_cnt.values() if c == 1)
            _rd_reps = [(t, c) for t, c in _rd_cnt.most_common() if c > 1]
            _rd_short = [t for t in _rd_toks if len(t) <= 2]
            _rd_long  = [t for t in _rd_toks if len(t) >= 7]
            print(f"\n  Repetition diagnostics ({len(_rd_toks)} tokens):")
            print(f"  Unique: {_rd_uniq}/{len(_rd_toks)} "
                  f"({100*_rd_uniq/max(len(_rd_toks),1):.1f}%)")
            print(f"  Short (≤2 chars): {len(_rd_short)}  "
                  f"Long (≥7 chars): {len(_rd_long)}")
            if _rd_res.get("conf_declining"):
                print(f"  [!] Confidence declining: last 4 tokens all lost confidence")
            if _rd_reps:
                print(f"  Repeated tokens ({len(_rd_reps)}):")
                for _rt, _rc in _rd_reps[:8]:
                    print(f"    '{_rt}' × {_rc}")
            else:
                print(f"  No repeated tokens.")
            continue

        if low.startswith("confoverlay"):
            # confoverlay [N] — raw confidence + EMA overlay per step
            _co_res = getattr(model, "_last_gen_result", None)
            if _co_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _co_raw  = _co_res.get("confidences", [])
            _co_toks = _co_res.get("tokens", [])
            if not _co_raw:
                print("  No confidence data in last result."); continue
            _co_parts = low.split()
            _co_n = (int(_co_parts[1]) if len(_co_parts) > 1
                     and _co_parts[1].isdigit() else len(_co_raw))
            import numpy as _np_co
            # Re-compute EMA (α=0.35) from raw confs
            _co_ema_list = []
            _co_e = _co_raw[0] if _co_raw else 0.0
            for _cv in _co_raw[:_co_n]:
                _co_e = 0.65 * _co_e + 0.35 * _cv
                _co_ema_list.append(round(_co_e, 4))
            _SPARKS9 = " ▁▂▃▄▅▆▇█"
            _co_spark_r = "".join(_SPARKS9[min(8, int(v * 8))] for v in _co_raw[:_co_n])
            _co_spark_e = "".join(_SPARKS9[min(8, int(v * 8))] for v in _co_ema_list)
            _co_decl = _co_res.get("conf_declining", False)
            print(f"\n  Confidence overlay ({len(_co_raw[:_co_n])} steps)"
                  + ("  [!declining]" if _co_decl else ""))
            print(f"  Raw : {_co_spark_r}")
            print(f"  EMA : {_co_spark_e}")
            print(f"  {'Step':<5}  {'Token':<18}  Raw    EMA")
            for _coi, (_rv, _ev, _ct) in enumerate(
                    zip(_co_raw[:_co_n], _co_ema_list, _co_toks + [""] * _co_n)):
                _diff = _rv - _ev
                _tag  = " ↑" if _diff > 0.04 else (" ↓" if _diff < -0.04 else "")
                print(f"  {_coi:<5}  {_ct[:18]:<18}  {_rv:.3f}  {_ev:.3f}{_tag}")
            _co_arr = _np_co.array(_co_raw[:_co_n], dtype=float)
            print(f"  raw avg={_co_arr.mean():.3f}  "
                  f"ema_final={_co_ema_list[-1]:.3f}  "
                  f"trend={'↓' if _co_arr[-1] < _co_arr[0] else '↑'}")
            continue

        if low.startswith("coh6plot"):
            # coh6plot [N] — sparkline of per-step 6-token coherence window
            _c6_res = getattr(model, "_last_gen_result", None)
            if _c6_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _c6_vals = _c6_res.get("coh6_steps", [])
            _c6_toks = _c6_res.get("tokens", [])
            if not _c6_vals:
                print("  No coh6 data (need ≥6 tokens)."); continue
            _c6_parts = low.split()
            _c6_n = (int(_c6_parts[1]) if len(_c6_parts) > 1
                     and _c6_parts[1].isdigit() else len(_c6_vals))
            import numpy as _np_c6
            _c6_arr = _np_c6.array(_c6_vals[:_c6_n], dtype=float)
            _SPARKS_C6 = " ▁▂▃▄▅▆▇█"
            _c6_spark = "".join(_SPARKS_C6[min(8, int(c * 8))] for c in _c6_arr)
            print(f"\n  Per-step coh6 ({len(_c6_arr)} steps, starts at step 6):")
            print(f"  {_c6_spark}")
            print(f"  {'Step':<5}  {'Token':<18}  Coh6   Bar")
            # offset index by 6 to align with actual token positions
            for _c6i, (_c6v, _c6t) in enumerate(
                    zip(_c6_vals[:_c6_n],
                        (_c6_toks[5:] if len(_c6_toks) > 5 else []) + [""] * _c6_n)):
                _c6blen = int(20 * max(_c6v, 0.0))
                _c6bar  = "█" * _c6blen + "░" * (20 - _c6blen)
                print(f"  {_c6i+6:<5}  {_c6t[:18]:<18}  {_c6v:.3f}  {_c6bar}")
            print(f"  avg={_c6_arr.mean():.3f}  "
                  f"min={_c6_arr.min():.3f}  max={_c6_arr.max():.3f}")
            continue

        if low.startswith("topconfgen"):
            # topconfgen [N] <prompt> — run N times, auto-pick highest conf_ema_final
            _tcg_parts = cmd.split(None, 2)
            _tcg_n     = 3
            _tcg_pmt   = ""
            if len(_tcg_parts) >= 3 and _tcg_parts[1].isdigit():
                _tcg_n   = max(1, min(int(_tcg_parts[1]), 8))
                _tcg_pmt = _tcg_parts[2]
            elif len(_tcg_parts) >= 2:
                _tcg_pmt = " ".join(_tcg_parts[1:])
            if not _tcg_pmt:
                print("  Usage: topconfgen [N] <prompt>"); continue
            import numpy as _np_tcg
            print(f"\n  Topconfgen: '{_tcg_pmt[:40]}' × {_tcg_n} runs")
            _tcg_best = None
            _tcg_best_c = -1.0
            _tcg_rows = []
            for _ti in range(_tcg_n):
                _tr = model.causal_generate(_tcg_pmt, max_tokens=20)
                _tc = _tr.get("conf_ema_final", 0.0)
                _tcg_rows.append((_tc, _tr))
                if _tc > _tcg_best_c:
                    _tcg_best_c = _tc
                    _tcg_best = _tr
            _tcg_rows.sort(key=lambda x: x[0], reverse=True)
            print(f"  {'#':<3}  {'ConfEMA':<8}  {'Coh':<6}  Text")
            for _ri, (_rc, _rr) in enumerate(_tcg_rows):
                _best_tag = " ← BEST" if _ri == 0 else ""
                print(f"  {_ri+1:<3}  {_rc:<8.4f}  "
                      f"{_rr.get('coherence',0):<6.3f}  "
                      f"'{_rr.get('text','')[:40]}'{_best_tag}")
            print(f"\n  Best: {_tcg_best.get('text','') if _tcg_best else '?'}")
            continue

        if low.startswith("pivotgen"):
            # pivotgen <first_word> | <prompt> — generate with forced first token
            _pg_rest = cmd[len("pivotgen"):].strip()
            if " | " not in _pg_rest:
                print("  Usage: pivotgen <first_word> | <prompt>"); continue
            _pg_word, _pg_pmt = [s.strip() for s in _pg_rest.split(" | ", 1)]
            if not _pg_word or not _pg_pmt:
                print("  Both first_word and prompt required."); continue
            # Check word is in vocab
            _pg_vocab = model.base_mapper._base_vocab
            if _pg_word not in _pg_vocab:
                print(f"  '{_pg_word}' not in vocab. Try a common word."); continue
            # Run baseline without pivot
            _pg_base = model.causal_generate(_pg_pmt, max_tokens=16)
            # Run with pivot: prepend the pivot word to the prompt to steer context
            _pg_pivot_pmt = f"{_pg_pmt} {_pg_word}"
            _pg_piv = model.causal_generate(_pg_pivot_pmt, max_tokens=16)
            import numpy as _np_pg
            print(f"\n  Pivotgen: pivot='{_pg_word}'  prompt='{_pg_pmt[:35]}'")
            def _pg_show(label, r):
                _c = r.get("confidences", [])
                print(f"\n  [{label}]")
                print(f"    Text:    {r.get('text','')[:55]}")
                print(f"    ConfEMA: {r.get('conf_ema_final',0):.4f}  "
                      f"Coh: {r.get('coherence',0):.3f}  "
                      f"Rhythm: {r.get('rhythm_score',0):.3f}")
            _pg_show("Baseline", _pg_base)
            _pg_show(f"Pivot '{_pg_word}'", _pg_piv)
            continue

        if low == "stresstest":
            # stresstest — run 10 built-in prompts, report avg conf/coh/rhythm
            import numpy as _np_st
            _st_prompts = [
                "the nature of consciousness is",
                "language emerges from",
                "time is a measure of",
                "knowledge and belief differ because",
                "the universe began when",
                "intelligence requires the ability to",
                "memory shapes identity through",
                "science and philosophy both seek",
                "creativity is defined by its",
                "meaning arises from",
            ]
            print(f"\n  Stresstest: {len(_st_prompts)} prompts")
            print(f"  {'#':<3}  {'ConfEMA':<8}  {'Coh':<6}  {'Rhythm':<7}  "
                  f"{'PPL':<7}  {'Toks':<5}  Prompt")
            _st_ce, _st_co, _st_rh = [], [], []
            for _si, _sp in enumerate(_st_prompts):
                _sr = model.causal_generate(_sp, max_tokens=14)
                _sc = _sr.get("conf_ema_final", 0.0)
                _sh = _sr.get("coherence",      0.0)
                _sr2 = _sr.get("rhythm_score",  0.0)
                _sp2 = _sr.get("pseudo_ppl",    0.0)
                _st  = len(_sr.get("tokens",    []))
                _st_ce.append(_sc); _st_co.append(_sh); _st_rh.append(_sr2)
                print(f"  {_si+1:<3}  {_sc:<8.4f}  {_sh:<6.3f}  {_sr2:<7.3f}  "
                      f"{_sp2:<7.1f}  {_st:<5}  '{_sp[:30]}'")
            print(f"\n  avg  conf={float(_np_st.mean(_st_ce)):.4f}  "
                  f"coh={float(_np_st.mean(_st_co)):.3f}  "
                  f"rhythm={float(_np_st.mean(_st_rh)):.3f}")
            print(f"  std  conf={float(_np_st.std(_st_ce)):.4f}  "
                  f"coh={float(_np_st.std(_st_co)):.3f}  "
                  f"rhythm={float(_np_st.std(_st_rh)):.3f}")
            continue

        if low.startswith("varplot"):
            # varplot [N] — sparkline of per-step score variance
            _va_res = getattr(model, "_last_gen_result", None)
            if _va_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _va_vals = _va_res.get("score_var_steps", [])
            _va_toks = _va_res.get("tokens", [])
            if not _va_vals:
                print("  No score variance data in last result."); continue
            _va_parts = low.split()
            _va_n = (int(_va_parts[1]) if len(_va_parts) > 1
                     and _va_parts[1].isdigit() else len(_va_vals))
            import numpy as _np_va
            _va_arr = _np_va.array(_va_vals[:_va_n], dtype=float)
            _va_max = float(_va_arr.max()) if _va_arr.max() > 0 else 1.0
            _SPARKSVA = " ▁▂▃▄▅▆▇█"
            _va_spark = "".join(
                _SPARKSVA[min(8, int(v / _va_max * 8))] for v in _va_arr
            )
            print(f"\n  Per-step score variance ({len(_va_arr)} steps):")
            print(f"  {_va_spark}")
            print(f"  {'Step':<5}  {'Token':<18}  Var     Bar")
            for _vi, (_vv, _vt) in enumerate(
                    zip(_va_vals[:_va_n], _va_toks + [""] * _va_n)):
                _vblen = int(20 * _vv / _va_max)
                _vbar  = "█" * _vblen + "░" * (20 - _vblen)
                print(f"  {_vi:<5}  {_vt[:18]:<18}  {_vv:.5f}  {_vbar}")
            print(f"  avg={_va_arr.mean():.5f}  "
                  f"max={_va_max:.5f}  min={_va_arr.min():.5f}")
            continue

        if low == "genheatmap":
            # genheatmap — ASCII heat map of score histogram buckets over time
            _gh_res = getattr(model, "_last_gen_result", None)
            if _gh_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _gh_hist = _gh_res.get("score_hist", [])
            _gh_var  = _gh_res.get("score_var_steps", [])
            _gh_toks = _gh_res.get("tokens", [])
            if not _gh_hist:
                print("  No score hist data in last result."); continue
            # Show per-bucket fill as heat scale
            _GH_HEAT = " ░▒▓█"
            _gh_total = sum(_gh_hist)
            print(f"\n  Score histogram heatmap ({_gh_total} total steps):")
            print(f"  Bucket       Count  Heat   %")
            for _ghi, _ghc in enumerate(_gh_hist):
                _lo = _ghi / 8.0;  _hi = (_ghi + 1) / 8.0
                _pct = _ghc / max(_gh_total, 1)
                _hi_idx = min(4, int(_pct / 0.25 * 4))
                _heat_bar = _GH_HEAT[_hi_idx] * 12
                print(f"  [{_lo:.3f},{_hi:.3f})  {_ghc:<6}  {_heat_bar}  "
                      f"{100*_pct:.1f}%")
            # Variance summary
            if _gh_var:
                import numpy as _np_gh
                _ghv = _np_gh.array(_gh_var, dtype=float)
                print(f"  Score var: avg={_ghv.mean():.5f}  "
                      f"max={_ghv.max():.5f}  min={_ghv.min():.5f}")
            continue

        if low == "confmatrix":
            # confmatrix — 2D confidence×position heat map
            _cm_res = getattr(model, "_last_gen_result", None)
            if _cm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cm_confs = _cm_res.get("confidences", [])
            _cm_toks  = _cm_res.get("tokens", [])
            if not _cm_confs:
                print("  No confidence data in last result."); continue
            # 4 conf buckets × position thirds
            _cm_n     = len(_cm_confs)
            _cm_third = max(1, _cm_n // 3)
            _cm_bins  = {"low": [0,0,0], "mid": [0,0,0],
                         "high":[0,0,0], "peak":[0,0,0]}
            for _ci, _cv in enumerate(_cm_confs):
                _pos = min(2, _ci // _cm_third)
                if   _cv < 0.30: _cm_bins["low" ][_pos] += 1
                elif _cv < 0.55: _cm_bins["mid" ][_pos] += 1
                elif _cv < 0.80: _cm_bins["high"][_pos] += 1
                else:            _cm_bins["peak"][_pos] += 1
            _CM_HEAT2 = " ░▒▓█"
            print(f"\n  Confidence × position matrix ({_cm_n} tokens):")
            print(f"  {'Bucket':<7}  Early      Mid        Late")
            for _cml, _cmv in _cm_bins.items():
                _cells = ""
                for _cmx in _cmv:
                    _pct2 = _cmx / max(_cm_n // 3, 1)
                    _cells += _CM_HEAT2[min(4, int(_pct2 * 5))] * 8 + "  "
                _counts = f"({_cmv[0]},{_cmv[1]},{_cmv[2]})"
                print(f"  {_cml:<7}  {_cells}  {_counts}")
            print(f"  conf avg={sum(_cm_confs)/max(len(_cm_confs),1):.3f}  "
                  f"declining={_cm_res.get('conf_declining',False)}")
            continue

        if low.startswith("topkplot"):
            # topkplot [N] — sparkline of adaptive TopK k value per step
            _kp_res = getattr(model, "_last_gen_result", None)
            if _kp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _kp_vals = _kp_res.get("topk_steps", [])
            _kp_toks = _kp_res.get("tokens", [])
            if not _kp_vals:
                print("  No TopK step data in last result."); continue
            _kp_parts = low.split()
            _kp_n = (int(_kp_parts[1]) if len(_kp_parts) > 1
                     and _kp_parts[1].isdigit() else len(_kp_vals))
            import numpy as _np_kp
            _kp_arr = _np_kp.array(_kp_vals[:_kp_n], dtype=float)
            _kp_min = 20.0; _kp_max = 60.0
            _SPARKS_KP = " ▁▂▃▄▅▆▇█"
            _kp_spark = "".join(
                _SPARKS_KP[min(8, int((_v - _kp_min) / (_kp_max - _kp_min) * 8))]
                for _v in _kp_arr
            )
            print(f"\n  Adaptive TopK k per step ({len(_kp_arr)} steps, range 20-60):")
            print(f"  {_kp_spark}")
            print(f"  {'Step':<5}  {'Token':<18}  k    Bar")
            for _kpi, (_kv, _kt) in enumerate(
                    zip(_kp_vals[:_kp_n], _kp_toks + [""] * _kp_n)):
                _kblen = int(20 * (_kv - _kp_min) / (_kp_max - _kp_min))
                _kbar  = "█" * _kblen + "░" * (20 - _kblen)
                print(f"  {_kpi:<5}  {_kt[:18]:<18}  {int(_kv):<4}  {_kbar}")
            print(f"  avg={_kp_arr.mean():.1f}  min={_kp_arr.min():.0f}  "
                  f"max={_kp_arr.max():.0f}")
            continue

        if low.startswith("confgraph"):
            # confgraph [N] — ASCII sparkline of per-step confidences (last gen)
            _cg_res = getattr(model, "_last_gen_result", None)
            if _cg_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cg_confs = _cg_res.get("confidences", [])
            _cg_toks  = _cg_res.get("tokens", [])
            if not _cg_confs:
                print("  No confidences in last result."); continue
            _cg_parts = low.split()
            _cg_n = int(_cg_parts[1]) if len(_cg_parts) > 1 and _cg_parts[1].isdigit() else len(_cg_confs)
            _cg_confs = _cg_confs[:_cg_n]
            _cg_toks  = _cg_toks[:_cg_n]
            import numpy as _np_cg
            _SPARKS = " ▁▂▃▄▅▆▇█"
            print(f"\n  Confidence sparkline ({len(_cg_confs)} tokens):")
            _spark = "".join(_SPARKS[min(8, int(c * 8))] for c in _cg_confs)
            print(f"  {_spark}")
            print(f"  {'Step':<5}  {'Token':<18}  Conf   Bar")
            for _ci, (conf, tok) in enumerate(zip(_cg_confs, _cg_toks)):
                _blen = int(20 * conf)
                _bar  = "█" * _blen + "░" * (20 - _blen)
                print(f"  {_ci:<5}  {tok[:18]:<18}  {conf:.3f}  {_bar}")
            print(f"  avg={float(_np_cg.mean(_cg_confs)):.3f}  "
                  f"peak_step={_cg_res.get('peak_conf_step','?')}")
            continue

        if low.startswith("trendgen"):
            # trendgen [N] <prompt> — run same prompt N times, show conf_ema_final trend
            _tg_parts  = cmd.split(None, 2)
            _tg_n      = 3
            _tg_prompt = ""
            if len(_tg_parts) >= 3 and _tg_parts[1].isdigit():
                _tg_n      = max(1, min(int(_tg_parts[1]), 8))
                _tg_prompt = _tg_parts[2]
            elif len(_tg_parts) >= 2:
                _tg_prompt = " ".join(_tg_parts[1:])
            if not _tg_prompt:
                print("  Usage: trendgen [N] <prompt>"); continue
            import numpy as _np_tg
            print(f"\n  Trendgen: '{_tg_prompt}' × {_tg_n} runs")
            _tg_results = []
            for _ti in range(_tg_n):
                _tr = model.causal_generate(_tg_prompt, max_tokens=16)
                _tc = _tr.get("confidences", [])
                _tg_results.append({
                    "text":      _tr.get("text","")[:38],
                    "conf_ema":  _tr.get("conf_ema_final", 0.0),
                    "rhythm":    _tr.get("rhythm_score",   0.0),
                    "coh_dir":   _tr.get("coh_direction",  0.0),
                    "coh":       _tr.get("coherence",      0.0),
                    "toks":      len(_tr.get("tokens",[])),
                    "stop":      _tr.get("stop_reason",    "?"),
                })
                print(f"  [{_ti+1}] {_tg_results[-1]['text']}")
            print(f"\n  {'Run':<4}  {'ConfEMA':<8}  {'Coh':<6}  {'Rhythm':<7}"
                  f"  {'CohDir':<8}  {'Toks':<5}  {'Stop'}")
            for _ti, _tr in enumerate(_tg_results):
                print(f"  {_ti+1:<4}  {_tr['conf_ema']:<8.4f}  "
                      f"{_tr['coh']:<6.3f}  {_tr['rhythm']:<7.3f}  "
                      f"{_tr['coh_dir']:<+8.4f}  {_tr['toks']:<5}  {_tr['stop']}")
            _tg_emas = [r["conf_ema"] for r in _tg_results]
            print(f"\n  ConfEMA  mean:{float(_np_tg.mean(_tg_emas)):.4f}  "
                  f"std:{float(_np_tg.std(_tg_emas)):.4f}  "
                  f"range:[{min(_tg_emas):.4f}–{max(_tg_emas):.4f}]")
            continue

        if low.startswith("diffgen"):
            # diffgen A | B — run two prompts, show side-by-side stats comparison
            _dg_sep = " | "
            _dg_rest = cmd[len("diffgen"):].strip()
            if _dg_sep not in _dg_rest:
                print("  Usage: diffgen <prompt A> | <prompt B>"); continue
            _dg_pa, _dg_pb = [p.strip() for p in _dg_rest.split(_dg_sep, 1)]
            if not _dg_pa or not _dg_pb:
                print("  Both prompts must be non-empty."); continue
            import numpy as _np_dg
            print(f"\n  Running diffgen: '{_dg_pa}' vs '{_dg_pb}' ...")
            _dg_ra = model.causal_generate(_dg_pa, max_tokens=16)
            _dg_rb = model.causal_generate(_dg_pb, max_tokens=16)
            def _dg_stat(r):
                _c = r.get("confidences", [])
                return {
                    "text":    r.get("text","")[:40],
                    "avg":     float(_np_dg.mean(_c)) if _c else 0.0,
                    "coh":     r.get("coherence",  0.0),
                    "rhythm":  r.get("rhythm_score",0.0),
                    "ppl":     r.get("pseudo_ppl", 0.0),
                    "spread":  r.get("token_embed_spread",0.0),
                    "stop":    r.get("stop_reason","?"),
                    "toks":    len(r.get("tokens",[])),
                    "coh_dir": r.get("coh_direction",0.0),
                }
            _dg_sa = _dg_stat(_dg_ra)
            _dg_sb = _dg_stat(_dg_rb)
            _METRICS = ["text","avg","coh","rhythm","ppl","spread","stop","toks","coh_dir"]
            print(f"  {'Metric':<14}  {'A':<42}  {'B':<42}")
            print(f"  {'-'*14}  {'-'*42}  {'-'*42}")
            for _m in _METRICS:
                _va = _dg_sa[_m]
                _vb = _dg_sb[_m]
                if isinstance(_va, float):
                    _sa_str = f"{_va:.4f}"
                    _sb_str = f"{_vb:.4f}"
                else:
                    _sa_str = str(_va)
                    _sb_str = str(_vb)
                print(f"  {_m:<14}  {_sa_str:<42}  {_sb_str:<42}")
            continue

        if low == "scorehist":
            # scorehist — ASCII bar chart of per-step score gaps from last gen
            _sh_res = getattr(model, "_last_gen_result", None)
            if _sh_res is None:
                print("  No generation recorded yet. Run a prompt first."); continue
            _sh_gaps = _sh_res.get("sg_step_gaps", [])
            if not _sh_gaps:
                print("  No score-gap history in last result."); continue
            import numpy as _np_sh
            _sh_arr  = _np_sh.array(_sh_gaps, dtype=float)
            _sh_max  = float(_sh_arr.max()) if _sh_arr.max() > 0 else 1.0
            _sh_toks = _sh_res.get("tokens", [])
            _BAR_W   = 30
            print(f"\n  Score-gap histogram (last generation, {len(_sh_gaps)} steps):")
            print(f"  {'Step':<5}  {'Token':<18}  Gap    {'Bar'}")
            for _si, (_sg, _st) in enumerate(zip(_sh_gaps, _sh_toks + [""] * len(_sh_gaps))):
                _bar_len = int(_BAR_W * _sg / _sh_max)
                _bar = "█" * _bar_len + "░" * (_BAR_W - _bar_len)
                print(f"  {_si:<5}  {_st[:18]:<18}  {_sg:.3f}  {_bar}")
            _sh_avg = float(_sh_arr.mean())
            print(f"  avg={_sh_avg:.3f}  max={_sh_max:.3f}  "
                  f"min={float(_sh_arr.min()):.3f}")
            # Score histogram from result dict if present
            _sh_hist = _sh_res.get("score_hist", [])
            if _sh_hist:
                import numpy as _np_sh2
                _sh_hist_arr = _np_sh2.array(_sh_hist, dtype=int)
                _sh_hist_max = max(int(_sh_hist_arr.max()), 1)
                print(f"\n  Confidence bucket distribution (0=low, 7=high):")
                for _hi, _hc in enumerate(_sh_hist):
                    _hbar = "█" * int(20 * _hc / _sh_hist_max) + "░" * (20 - int(20 * _hc / _sh_hist_max))
                    _hlo  = _hi * 0.125
                    _hhi  = _hlo + 0.125
                    print(f"  [{_hlo:.3f}–{_hhi:.3f}]  {_hbar}  {_hc}")
            continue

        if low == "topconfs":
            # topconfs — show top-5 highest-confidence steps from the last generation
            _tc_res = getattr(model, "_last_gen_result", None)
            if _tc_res is None:
                print("  No generation recorded yet. Run a prompt first."); continue
            _tc_confs = _tc_res.get("confidences", [])
            _tc_toks  = _tc_res.get("tokens", [])
            if not _tc_confs:
                print("  No confidences found in last result."); continue
            import numpy as _np_tc
            _tc_arr  = _np_tc.array(_tc_confs)
            _tc_top5 = _np_tc.argsort(_tc_arr)[::-1][:5]
            print(f"\n  Top-5 highest-confidence steps (last generation):")
            print(f"  {'Step':<6}  {'Token':<22}  Conf")
            for _ti in _tc_top5:
                _tt = _tc_toks[_ti] if _ti < len(_tc_toks) else "?"
                print(f"  {_ti:<6}  {_tt:<22}  {_tc_confs[_ti]:.4f}")
            print(f"  Peak at step {_tc_res.get('peak_conf_step','?')}  "
                  f"avg={float(_np_tc.mean(_tc_confs)):.3f}")
            continue

        if low.startswith("topwords"):
            # topwords [N]  — show top-N vocab words by training frequency
            _tw_parts = low.split()
            _tw_n = int(_tw_parts[1]) if len(_tw_parts) > 1 and _tw_parts[1].isdigit() else 20
            _tw_vocab = model.base_mapper._base_vocab
            # Frequency info is stored in the brain's history; approximate by
            # checking the dot memory outcome counts for each word.
            _tw_dm = model.dot_memory
            _tw_freq: dict = {}
            for _tok in _tw_vocab:
                _count = 0
                for _dot in (_tw_dm.dots if _tw_dm else []):
                    _hist = getattr(_dot, "history", [])
                    _count += sum(1 for h in _hist
                                  if (h.get("outcome") or "") == _tok)
                _tw_freq[_tok] = _count
            _tw_sorted = sorted(_tw_freq.items(), key=lambda x: x[1], reverse=True)
            _tw_top = [p for p in _tw_sorted if p[1] > 0][:_tw_n]
            print(f"\n  Top-{_tw_n} vocab words by dot-outcome frequency:")
            for _twi, (_tww, _twc) in enumerate(_tw_top, 1):
                print(f"    {_twi:3}. {_tww:<22}  {_twc:,}")
            if not _tw_top:
                print("  No outcome history yet — train or cgen first.")
            continue

        if low.startswith("ctxvec "):
            # ctxvec <prompt>  — show top-10 vocab words nearest the context
            # direction produced by encoding the prompt.  Useful for diagnosing
            # whether the model's context is pointing at the right semantic region.
            _ctext = user[7:].strip()
            if not _ctext:
                print("  Usage: ctxvec <prompt>"); continue
            import numpy as _np_cv
            _cv_vocab = model.base_mapper._base_vocab
            _cv_words = list(_cv_vocab.keys())
            _cv_ctx   = model.encode(_ctext)   # returns (feature_dim,) vector
            _cv_ed    = 224                     # EMBED_DIM
            _cv_c     = _cv_ctx[:_cv_ed].astype(_np_cv.float32)
            _cv_cn    = _cv_c / (float(_np_cv.linalg.norm(_cv_c)) + 1e-9)
            # Build word_vecs_n on-the-fly for the ctxvec call
            _cv_mats  = []
            for _w in _cv_words:
                _e = _cv_vocab[_w]
                _n = float(_np_cv.linalg.norm(_e[:_cv_ed]))
                _cv_mats.append(_e[:_cv_ed] / (_n + 1e-9))
            _cv_wvn   = _np_cv.array(_cv_mats, dtype=_np_cv.float32)
            _cv_sims  = _cv_wvn @ _cv_cn
            _cv_top10 = _np_cv.argsort(_cv_sims)[::-1][:10]
            print(f"\n  Context direction for: '{_ctext}'")
            print(f"  Top-10 semantically nearest words:")
            for _ri, _wi in enumerate(_cv_top10, 1):
                print(f"    {_ri:2}. {_cv_words[_wi]:<20}  cos={_cv_sims[_wi]:+.4f}")
            continue

        if low == "rebuild":
            cmd_rebuild(); continue
        if low.startswith("calibrate"):
            parts = user.split(None, 1)
            corpus = parts[1].strip() if len(parts) > 1 else "/tmp/corpus_300k.txt"
            cmd_calibrate(corpus); continue
        if low.startswith("cgen "):
            prompt = user[5:].strip()
            if not prompt: print("  Usage: cgen <prompt>"); continue
            _run_cgen(model, prompt); continue
        if low.startswith("nbest10 "):
            prompt = user[8:].strip()
            if not prompt: print("  Usage: nbest10 <prompt>"); continue
            _run_nbest(model, prompt, n=10); continue
        if low.startswith("nbest5 "):
            prompt = user[7:].strip()
            if not prompt: print("  Usage: nbest5 <prompt>"); continue
            _run_nbest(model, prompt, n=5); continue
        if low.startswith("nbest "):
            prompt = user[6:].strip()
            if not prompt: print("  Usage: nbest <prompt>"); continue
            _run_nbest(model, prompt, n=3); continue
        if low.startswith("generate "):
            prompt = user[9:].strip()
            if not prompt: print("  Usage: generate <prompt>"); continue
            print("  [generating...]", end="", flush=True)
            out = model.generate(prompt)
            print(f"\r  Output: {out}            ")
            model.save_brain(); continue
        if low == "chat":
            cmd_chat(model); continue
        if low.startswith("encode "):
            text = user[7:].strip()
            if not text: print("  Usage: encode <text>"); continue
            res = model.run(text, verbose=False)
            v = res.output
            print(f"  norm={np.linalg.norm(v):.3f}  rounds={res.summary['rounds']}  "
                  f"stop='{res.stop_reason}'")
            print(f"  first8: {np.round(v[:8], 3)}")
            model.save_brain(); continue
        if low.startswith("compare ") and "|" in user:
            # compare A | B | C — pairwise similarity matrix in interactive mode
            from formulas.formulas import similarity_score as _ss
            _cmp_parts = user[8:].split("|")
            _cmp_texts = [p.strip() for p in _cmp_parts if p.strip()]
            if len(_cmp_texts) < 2:
                print("  Usage: compare <text1> | <text2> | ..."); continue
            _cmp_vecs = [(t, model.encode(t)) for t in _cmp_texts]
            _cw = max(25, max(len(t) for t in _cmp_texts[:6]) + 2)
            print(f"\n  Similarity matrix ({len(_cmp_texts)} texts):")
            _hdr = " " * (_cw + 2) + "  ".join(
                f"{t[:_cw]:>{_cw}}" for t, _ in _cmp_vecs)
            print(_hdr)
            for _i, (_ta, _va) in enumerate(_cmp_vecs):
                _row = "  ".join(
                    f"{_ss(_va, _vb):>{_cw}.4f}" for _, _vb in _cmp_vecs)
                print(f"  {_ta[:_cw]:<{_cw}}  {_row}")
            model.save_brain(); continue

        if low.startswith("train "):
            fpath = user[6:].strip()
            if not fpath: print("  Usage: train <filepath>"); continue
            try:
                model.fit_file(fpath, verbose=True)
            except FileNotFoundError as e:
                print(f"  Error: {e}")
            continue
        # Default: IECNN-native generation
        _run_cgen(model, user)


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
    if "--phase-coding" in args:
        _PHASE_CODING = True
        args = [a for a in args if a != "--phase-coding"]
    if not args:
        cmd_interactive()
    elif args[0] == "build":
        cmd_build()
    elif args[0] == "demo":
        cmd_demo()
    elif args[0] == "memory":
        cmd_memory()
    elif args[0] == "prune":
        dry = "--dry-run" in args
        min_outcomes = 2
        min_age = 2
        if "--min-outcomes" in args:
            i = args.index("--min-outcomes")
            if i + 1 < len(args):
                try: min_outcomes = int(args[i+1])
                except ValueError: pass
        if "--min-age" in args:
            i = args.index("--min-age")
            if i + 1 < len(args):
                try: min_age = int(args[i+1])
                except ValueError: pass
        cmd_prune(dry_run=dry, min_outcomes=min_outcomes, min_age_gens=min_age)
    elif args[0] == "encode" and len(args) >= 2:
        cmd_encode(" ".join(args[1:]))
    elif args[0] == "rebuild":
        cmd_rebuild()
    elif args[0] == "calibrate":
        corpus = args[1] if len(args) >= 2 else "/tmp/corpus_300k.txt"
        limit  = 0
        if "--limit" in args:
            i = args.index("--limit")
            if i + 1 < len(args):
                try: limit = int(args[i + 1])
                except ValueError: pass
        cmd_calibrate(corpus, limit=limit)
    elif args[0] == "cgen" and len(args) >= 2:
        cmd_cgen_oneshot(" ".join(args[1:]))
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
    elif args[0] == "chat":
        cmd_chat()
    elif args[0] == "train" and len(args) >= 2:
        # python main.py train <file> [--limit N] [--evolve] [--causal]
        #                             [--prune-every N] [--fast] [--workers N]
        filepath = args[1]
        limit = 0
        prune_every = 0
        evolve         = "--evolve"         in args
        causal         = "--causal"         in args
        fast           = "--fast"           in args
        shared_memory  = "--shared-memory"  in args
        ultra          = "--ultra"          in args
        workers        = None
        if "--limit" in args:
            i = args.index("--limit")
            if i + 1 < len(args):
                try: limit = int(args[i+1])
                except ValueError: limit = 0
        if "--prune-every" in args:
            i = args.index("--prune-every")
            if i + 1 < len(args):
                try: prune_every = int(args[i+1])
                except ValueError: prune_every = 0
            if not causal:
                evolve = True
        if "--workers" in args:
            i = args.index("--workers")
            if i + 1 < len(args):
                try: workers = int(args[i+1])
                except ValueError: workers = None
        cmd_train(filepath, limit=limit, evolve=evolve, causal=causal,
                  prune_every=prune_every, fast=fast, workers=workers,
                  shared_memory=shared_memory, ultra=ultra)
    else:
        print(__doc__)
