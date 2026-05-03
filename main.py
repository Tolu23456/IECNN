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
    coh_trend    = result.get("coh_trend",       0.0)
    sg_slope     = result.get("sg_slope",        0.0)
    vprec_ema    = result.get("vocab_prec_ema",  0.0)
    conf_decl    = result.get("conf_declining",  False)
    ent_ratio    = result.get("entropy_ratio",   1.0)
    vel_ratio    = result.get("vel_ratio",        1.0)
    flow_score   = result.get("flow_score",       0.0)
    bestseg      = result.get("bestseg",          {})
    worstseg     = result.get("worstseg",         {})
    topkdelta    = result.get("topkdelta",         0.0)
    scorevar     = result.get("scorevar",          0.0)
    perc_trend   = result.get("perc_trend",        0.0)
    sfb_trend    = result.get("sfb_trend",         0.0)
    perfindex    = result.get("perfindex",         0.0)
    uniq_ratio   = result.get("uniq_ratio",        0.0)
    conf_ema_del = result.get("conf_ema_delta",    0.0)
    conf_ema_mid = result.get("conf_ema_mid",      0.0)
    entdelta     = result.get("entdelta",           0.0)
    topk_mode    = result.get("topk_mode",          0)
    sfb_acc      = result.get("sfb_acc",            0.0)
    margin_trend = result.get("margin_trend",       0.0)
    entropy_trend = result.get("entropy_trend",     0.0)
    vel_trend         = result.get("vel_trend",         0.0)
    rhythm_trend      = result.get("rhythm_trend",      0.0)
    coh_entropy_corr  = result.get("coh_entropy_corr",  0.0)
    conf_entropy_corr = result.get("conf_entropy_corr", 0.0)
    topk_entropy_corr = result.get("topk_entropy_corr", 0.0)
    coh_conf_corr     = result.get("coh_conf_corr",     0.0)
    vel_conf_corr     = result.get("vel_conf_corr",     0.0)
    score_var_trend   = result.get("score_var_trend",   0.0)

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
          f"VPrecEMA:{vprec_ema:.3f}  EntRatio:{ent_ratio:.2f}  VelRatio:{vel_ratio:.2f}  "
          f"Flow:{flow_score:.2f}  "
          f"Best@{bestseg.get('start','?')}({bestseg.get('val',0):.3f})  "
          f"Worst@{worstseg.get('start','?')}({worstseg.get('val',0):.3f})  "
          f"TopKΔ:{topkdelta:.2f}  SVar:{scorevar:.4f}  "
          f"PercTrend:{perc_trend:+.4f}  SFBTrend:{sfb_trend:+.4f}  "
          f"PerfIdx:{perfindex:.3f}  "
          f"UniqR:{uniq_ratio:.2f}  ConfΔ:{conf_ema_del:+.4f}  "
          f"ConfMid:{conf_ema_mid:.4f}  EntΔ:{entdelta:+.4f}  "
          f"TopKMode:{topk_mode}  SFBAcc:{sfb_acc:+.5f}  "
          f"MarginTrend:{margin_trend:+.5f}  EntTrend:{entropy_trend:+.5f}  "
          f"VelTrend:{vel_trend:+.5f}  "
          f"TKECorr:{topk_entropy_corr:+.3f}  CohCConf:{coh_conf_corr:+.3f}  "
          f"VCCorr:{vel_conf_corr:+.3f}  SVTrend:{score_var_trend:+.5f}  "
          f"RhythmTrend:{rhythm_trend:+.5f}  CohEntCorr:{coh_entropy_corr:+.3f}  "
          f"ConfEntCorr:{conf_entropy_corr:+.3f}  "
          f"{'[!DECLINING]' if conf_decl else ''}  "
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
            print("    topicshift           steps where velocity > mean+1σ (topic jumps)")
            print("    lastgen              compact one-line summary of last generation")
            print("    genprofile <prompt>  generate + inline tokenmap/coh/var/shift report")
            print("    sgplot [N]           per-step score-gap EMA sparkline")
            print("    confrise [N]         steps where conf_ema rose fastest")
            print("    multiprofile         3 preset prompts compared in one metric table")
            print("    spikeplot [N]        steps where entropy ≥ 1.5× mean (spikes)")
            print("    velspike             steps where ctx velocity > mean+1σ")
            print("    confbands            per-token L/M/H confidence band strip")
            print("    repchain             detect 3+ same-token repeats in 6-step windows")
            print("    segplot              rolling 3-step coh3 window with best/worst marks")
            print("    flowbar              per-step flow indicator: B/C/H/N band strip")
            print("    confpeak             show peak + valley confidence steps with context")
            print("    trendcompare A | B   run two prompts, compare 9 trend metrics")
            print("    qualmap              per-step conf×coh3 composite quality heat map")
            print("    smoothplot [N]       raw conf_ema vs 3-pt smoothed overlay sparklines")
            print("    diagsummary          all key metrics in one compact diagnostic block")
            print("    accelplot [N]        confidence 2nd-derivative (▲accel ─flat ▼decel)")
            print("    cohrank              rank tokens by their coh3-window value")
            print("    slopechart [N]       confidence 1st-derivative (▲rising ─flat ▼falling)")
            print("    cohaccel             coherence 2nd-derivative (▲accel ─flat ▼decel)")
            print("    scorevarplot [N]     per-step within-step score variance sparkline")
            print("    rangeplot            confidence + coh3 range (spread) bars")
            print("    buckethist           10-bucket confidence histogram 0.0-1.0")
            print("    toklenplot           per-step token character length sparkline")
            print("    vocabtop10           top-10 tokens by frequency + dominant token")
            print("    phasemap             early/mid/late phase conf+coh comparison")
            print("    uniqplot             running unique-token ratio sparkline")
            print("    cohseries            full per-step coh3 table with sparkline")
            print("    midconf              early/mid/late conf_ema with direction arrows")
            print("    correlplot           coh3 vs velocity sparklines + Pearson r")
            print("    confdrop             steps where confidence dropped ≥0.05")
            print("    confrises            steps where confidence jumped ≥0.05")
            print("    topkmode             histogram of TopK k values used per step")
            print("    scoregaptrend        rolling score-gap (decisiveness) sparkline")
            print("    sfbaccel             SFB per-step slope and trend summary")
            print("    margintrend          top1−top2 score margin sparkline + slope")
            print("    confbuckets          5-bucket confidence histogram (0.0-1.0)")
            print("    cohdrop              steps where coh3 dropped ≥0.05")
            print("    entropytrend         entropy per-step sparkline + linear slope")
            print("    cohrises             steps where coh3 jumped ≥0.05")
            print("    veltrend             velocity per-step sparkline + slope")
            print("    topkentropyplot      TopK vs entropy sparklines + Pearson r")
            print("    cohconfcorr          coh3 vs confidence sparklines + Pearson r")
            print("    velconfcorr          velocity vs confidence sparklines + Pearson r")
            print("    scorevartrend        score variance per-step sparkline + slope")
            print("    rhythmtrend          running rhythm rate per-step sparkline + slope")
            print("    confemascatter       conf_ema slope scatter ▲/─/▼ per step")
            print("    confvarplot          running confidence variance sparkline")
            print("    cohentropycorr       coh3 vs entropy sparklines + Pearson r")
            print("    marginspikeplot      steps where top1-top2 margin spiked > mean+1.5σ")
            print("    cohvarplot           running coh3 variance sparkline")
            print("    confentropycorr      confidence vs entropy sparklines + Pearson r")
            print("    cohslopeplot         coh3 slope scatter ▲/─/▼ per step")
            print("    vprecslope           vprec_ema slope scatter ▲/─/▼ per step")
            print("    cohconfplot          coh3 × confidence quality product sparkline")
            print("    qualityplot          smoothed coh3×conf quality EMA sparkline")
            print("    confcohslopecorr     conf_ema slope vs coh3 slope scatter + Pearson r")
            print("    confvpreccorr        confidence vs vprec_ema sparklines + Pearson r")
            print("    qualspikeplot        steps where coh3×conf quality spiked > mean+1.5σ")
            print("    entropyvarplot       running entropy variance sparkline")
            print("    sgspikeplot          steps where ScoreGapEMA spiked > mean+1.5σ")
            print("    coh6varplot          running coh6 variance sparkline")
            print("    conftopkcorr         confidence vs TopK sparklines + Pearson r")
            print("    qualtrend            quality_steps sparkline with slope + direction")
            print("    vprecconfslopecorr   vprec slope vs conf_ema slope scatter + Pearson r")
            print("    qualvarplot          running quality variance sparkline")
            print("    sgconfcorr           ScoreGapEMA vs confidence sparklines + Pearson r")
            print("    marginvarplot        running top1 margin variance sparkline")
            print("    coh3sgcorr           coh3 vs ScoreGapEMA sparklines + Pearson r")
            print("    velvarplot           running velocity variance sparkline")
            print("    coh3vpreccorr        coh3 vs vprec_ema sparklines + Pearson r")
            print("    confvelcorr          confidence vs velocity sparklines + Pearson r")
            print("    topkvarplot          running adaptive TopK variance sparkline")
            print("    coh6confcorr         coh6 vs confidence sparklines + Pearson r")
            print("    topkvelcorr          TopK vs velocity sparklines + Pearson r")
            print("    marginconfcorr       top1 margin vs confidence sparklines + Pearson r")
            print("    entropyvelcorr       entropy vs velocity sparklines + Pearson r")
            print("    coh3marginslopecorr  coh3 slope vs margin slope scatter + Pearson r")
            print("    entropytopkcorr      entropy vs TopK sparklines + Pearson r")
            print("    vprecedntropycorr    vprec_ema vs entropy sparklines + Pearson r")
            print("    coh3entropyslopecorr coh3 slope vs entropy slope scatter + Pearson r")
            print("    healthscore          composite health score 0-4 with breakdown")
            print("    jointidealmeter      fraction of steps where coh3+vel+conf all ideal")
            print("    coh3slopetrend       coh3 1st+2nd derivative sparklines (momentum)")
            print("    confslopetrend       conf_ema 1st+2nd derivative sparklines (momentum)")
            print("    velslopetrend        velocity 1st+2nd derivative sparklines")
            print("    marginslopetrend     margin 1st+2nd derivative sparklines")
            print("    entropyslopetrend    entropy 1st+2nd derivative sparklines")
            print("    topkslopetrend       topk 1st+2nd derivative sparklines")
            print("    sgslopetrend         score-gap 1st+2nd derivative sparklines")
            print("    vprecslopetrend      vprec_ema 1st+2nd derivative sparklines")
            print("    coh6slopetrend       coh6 1st+2nd derivative sparklines")
            print("    marginveljoint       decisive+stable joint step meter")
            print("    confveljoint         confident+stable joint step meter")
            print("    coh3marginjoint      coherent+decisive joint step meter")
            print("    entropyveljoint      focused+stable joint step meter")
            print("    vpreccoh3joint       precise+coherent joint step meter")
            print("    quadrantmap          coh3×vel quadrant distribution")
            print("    idealrunlen          longest ideal streak + step-by-step map")
            print("    phasequalityscore    early/mid/late phase quality bar chart")
            print("    streakmap            sequence of quadrant streaks across gen")
            print("    confentropyratio     mean_conf / mean_entropy ratio + label")
            print("    phasetransitionmap   every quadrant change step with from→to")
            print("    confvelocityscore    mean_conf×(1−vel) composite quality bar")
            print("    quadentropy          per-quadrant mean entropy (focus table)")
            print("    quadconfmean         per-quadrant mean confidence table")
            print("    transitionrate       quadrant instability rate + label")
            print("    idealfrac            ideal-quadrant fraction bar + label")
            print("    quadvelocitymean     per-quadrant mean velocity table")
            print("    coh3veldivergence    |mean_coh3−(1−mean_vel)| alignment")
            print("    quadcoh3mean         per-quadrant mean coherence table")
            print("    confcoh3gap          mean_conf − mean_coh3 gap + label")
            print("    quadtransitionfrom   which quadrant was departed most often")
            print("    quadtransitionto     which quadrant was entered most often")
            print("    idealentryrate       ideal entries / total transitions bar")
            print("    driftingfrac         drifting-quadrant fraction bar + label")
            print("    quadbalancescore     (ideal−drifting)/n quality bias bar")
            print("    exploringfrac        exploring-quadrant fraction bar + label")
            print("    flatfrac             flat-quadrant fraction bar + label")
            print("    quadsummary          all-in-one quadrant dashboard (all 4 fracs + stats)")
            print("    quadvolatilityscore  normalised quadrant-flip rate bar + label")
            print("    quaddominancemargin  dominant_frac − second_frac decisiveness bar")
            print("    idealrundensity      ideal_frac × longest_run richness bar")
            print("    quadrecoveryrate     drifting→ideal/exploring recovery fraction bar")
            print("    quadpersistencescore avg steps per quadrant visit (stickiness) bar")
            print("    idealstabilityscore  ideal_frac − drifting_frac net quality index")
            print("    quadoscillationscore A→B→A ping-pong transition fraction bar")
            print("    quadidealentryvelocity mean velocity at ideal-quadrant entry")
            print("    quadcoh3entrymean    mean coh3 at ideal-quadrant entry bar")
            print("    quaddriftingexitcoh3 mean coh3 at drifting-quadrant exit bar")
            print("    quaddriftingentryvelocity mean velocity at drifting-entry bar")
            print("    quadidealdurationvariance std-dev of ideal-run lengths bar")
            print("    quadflatdurationvariance std-dev of flat-run lengths bar")
            print("    quadrecoveryvelocity     mean velocity at drifting-exit bar")
            print("    quaddriftingdurationvariance std-dev of drifting-run lengths bar")
            print("    quadexploringdurationvariance std-dev of exploring-run lengths bar")
            print("    quadidealentrycoh3variance std-dev of coh3 at ideal-entry bar")
            print("    quadexploringexitcoh3    mean coh3 at exploring-quadrant exit bar")
            print("    quadflatexitvelocity     mean velocity at flat-quadrant exit bar")
            print("    quadidealtoexploringrate fraction of ideal exits → exploring bar")
            print("    quaddriftingtoflatrate   fraction of drifting exits → flat bar")
            print("    quadexploringtoidealrate fraction of exploring exits → ideal bar")
            print("    quadflattodriftingrate   fraction of flat exits → drifting bar")
            print("    quadflattoexploringrate  fraction of flat exits → exploring bar")
            print("    quadflattoidealrate      fraction of flat exits → ideal bar")
            print("    quaddriftingtoexploringrate fraction of drifting exits → exploring bar")
            print("    quaddriftingtoidealrate  fraction of drifting exits → ideal bar")
            print("    quadexploringtoflatrate  fraction of exploring exits → flat bar")
            print("    quadexploringtodriftingrate fraction exploring exits → drifting bar")
            print("    quadidealtoflatrate      fraction of ideal exits → flat bar")
            print("    quadidealtodriftingrate  fraction of ideal exits → drifting bar")
            print("    quadtransitionentropy    Shannon entropy over all quad transitions")
            print("    quadselftransitionrate   fraction of steps quadrant unchanged bar")
            print("    quadtransitionmatrixskew max-minus-min row-prob transition skew bar")
            print("    quadidealrunconfidencemean mean conf during ideal-quadrant steps bar")
            print("    quaddriftingrunconfidencemean mean conf during drifting steps bar")
            print("    quadexploringrunconfidencemean mean conf during exploring steps bar")
            print("    quadflatrunconfidencemean mean conf during flat steps bar")
            print("    quadconfidencegap        ideal_conf minus drifting_conf gap bar")
            print("    quadconfidencespread     σ of 4 per-quadrant confidence means bar")
            print("    quadcoh3spread           σ of 4 per-quadrant coh3 means bar")
            print("    quadvelocityspread       σ of 4 per-quadrant velocity means bar")
            print("    quadcoh3idealvsflatratio ideal coh3 mean / flat coh3 mean bar")
            print("    quadvelocityidealvsdriftingratio ideal vel / drifting vel ratio bar")
            print("    quadshape                coh3/vel signal shape: arch/bowl/linear vs linear interp")
            print("    quadcounts               absolute ideal/drift/explore/flat step counts + bar")
            print("    quaddensity              ideal frac per quartile (Q1-Q4) + peak quartile")
            print("    quadpanorama             head/mid/tail scores side-by-side + overall weighted")
            print("    quadpeak                 peak/trough coh3 position + dynamic range")
            print("    quadmid                  middle-50% ideal/coh3/vel/conf + mid score")
            print("    quadarc                  head→tail score delta + direction labels (arc)")
            print("    quadhead                 first-25% ideal/coh3/vel/conf + head score")
            print("    quadtail                 last-25% ideal/coh3/vel/conf + tail score")
            print("    quadtransitions          P(ideal→ideal/drift) + P(drift→ideal/drift) + stability")
            print("    quadruns                 ideal/drift max-run + mean-run + run balance ratio")
            print("    quadcorrelations         Pearson r(coh3,vel) + r(coh3,conf) + r(vel,conf)")
            print("    quaduniformity           ideal/drift coh3+conf std + ideal uniformity score")
            print("    quadmomentum             coh3/vel/conf net momentum + curvature (2nd derivative)")
            print("    quadconfdeltas           conf +/- delta fracs + mean magnitudes + asymmetry")
            print("    quadveldeltas            vel +/- delta fracs + mean magnitudes + decel asymmetry")
            print("    quadcoh3deltas           coh3 +/- delta fracs + mean magnitudes + asymmetry")
            print("    quadconfpercentiles      conf p25/p50/p75/IQR + skew indicator")
            print("    quadvelpercentiles       vel p25/p50/p75/IQR + skew indicator")
            print("    quadcoh3percentiles      coh3 p25/p50/p75/IQR + skew indicator")
            print("    quadhealthscore          overall health + quality index + focus + grade A-F")
            print("    quadratiostats           ideal/explore ratio + drift/flat ratio + coherent fracs")
            print("    quadconfstats            conf min/max/range + frac>0.75 + frac>0.90")
            print("    quadvelstats             vel min/max/range + frac<0.25 + frac>mean")
            print("    quadcoh3stats            coh3 min/max/range + frac>0.75 + frac>0.90")
            print("    quadbursts               ideal/drift burst counts + max/mean burst + quality ratio")
            print("    quadvolatility           step-to-step Δ volatility for coh3/vel/conf + stability")
            print("    quadgaps                 ideal-vs-drift conf/coh3 gap + quality gap + separation")
            print("    quadconfprofile          per-quadrant confidence means + spread")
            print("    quadvelprofile           per-quadrant velocity means + spread")
            print("    quadquarters             Q1/Q4 ideal_frac + coh3_mean + arc score")
            print("    quadhalves               first/second half ideal_frac + coh3 + improvement score")
            print("    quadtrends               coh3/vel/conf linear slope + R2 + trend alignment")
            print("    quadentropy              label/coh3/vel/conf Shannon entropy + entropy index")
            print("    quadtransitions          Markov transition matrix + stability score")
            print("    quadpersistence          P(stay in ideal/drift) + lag-1 autocorr table")
            print("    quadautocorr             lag-1 autocorrelation for coh3/vel/conf signals")
            print("    quadcorrelations         coh3/conf/vel Pearson correlations + coupling score")
            print("    quadvariance             ideal+drift coh3/conf variance + variance ratio")
            print("    quadsignalquality        composite signal quality index bar + components")
            print("    quadkurtosis             coh3+vel+conf excess kurtosis + skewness table")
            print("    quadcentroids            ideal+drift temporal centroid step + gap label")
            print("    quadskew                 coh3 + velocity skewness of signal distribution")
            print("    quadzigzag               ideal↔other flip rate per step [0,1] bar + run counts")
            print("    quadexplorestreaks       exploring+flat max streak + run count + mean length")
            print("    quadrle                  opening RLE string + oscillations + gap + isolation")
            print("    quadoscillation          oscillation count + longest non-ideal gap + isolation")
            print("    quadstreaks              max ideal/drift streak + start index + concentration ratio")
            print("    quadbalance              coherent/focused ratios + health vector + quality score")
            print("    quadqualityscore         composite [0,1] quality score bar + components")
            print("    quadheadfracs            head/mid/tail ideal+drift frac table + trend")
            print("    quadpeaks                ideal peak coh3/conf + drifting worst coh3")
            print("    quadmidfracs             ideal/drift middle-50% fracs + tail comparison")
            print("    quadtailfracs            ideal/drift tail-25% fracs + improvement score")
            print("    quadtransitionentropy    Shannon entropy of all transitions + dominant transition")
            print("    quadfinalstate           final quadrant label + first/last ideal/drift indices")
            print("    quadseparation           composite ideal/drift separation score bar")
            print("    quadranges               ideal + drifting conf/vel max−min range bars")
            print("    quadvelstds              ideal + drifting vel σ + CV + contrast bar")
            print("    quadconfstds             ideal + drifting conf σ + CV + contrast bar")
            print("    quadcoh3stds             ideal + drifting coh3 σ + CV + contrast bar")
            print("    quadcoh3means            per-quadrant mean coh3 bars + gap")
            print("    quadrunlens              per-quadrant mean run length bars + ideal/drift ratio")
            print("    quadvelmeans             per-quadrant mean velocity bars + vel gap")
            print("    quadconfmeans            per-quadrant mean confidence + gap bar")
            print("    quadoverallefficiency    health/hi-vel composite efficiency bar")
            print("    quadconfcv               ideal/drifting σ/μ + ideal conf stability bar")
            print("    quadflowefficiency       ideal/hi-coh3 fraction bar (flow vs. exploration)")
            print("    quaddriftseverity        drift/hi-vel fraction bar (how bad fast steps are)")
            print("    quadrecoveryefficiency   drift→ideal × ideal_persistence product bar")
            print("    quadvelocityfracs        hi-vel + lo-vel fraction bars")
            print("    quadqualityarc           arc label + early/late/trend + health + fingerprint")
            print("    quaddeltas               ideal−drift + expl−flat signed bars + hi/lo coh3 fracs")
            print("    quadpersistence          all 4 quadrant self-loop fracs + fingerprint + sym")
            print("    quadsymmetry             normalised distribution entropy bar + fingerprint")
            print("    quadfingerprint          I>E>D>F dominance string + frac breakdown")
            print("    quadidealpersistence     ideal→ideal self-loop fraction bar")
            print("    quadnetrecovery          drift→ideal minus ideal→drift signed bar")
            print("    quadescaperates          flat/expl/drift→ideal + ideal→expl rate bars")
            print("    quadtransitionrates      ideal→drift / drift→ideal / exploring→ideal rates")
            print("    quadflatmomentum         coh3 + vel momentum bars for flat runs")
            print("    quadinterrungaps         mean steps between ideal + drifting run episodes")
            print("    quaddriftingmomentum     coh3 + vel momentum bars for drifting runs")
            print("    quadexploringcoh3momentum coh3 trend inside exploring runs signed bar")
            print("    quadtransitionmap        ASCII 4x4 transition matrix with row-probabilities")
            print("    quadvelocitymassratio    ideal |vel| mass / drifting |vel| mass bar")
            print("    quadidealvelocitymomentum last3 minus first3 ideal vel signed bar")
            print("    quadreport               comprehensive one-screen quadrant health report")
            print("    quadcoh3massratio        ideal coh3 mass / drifting coh3 mass bar")
            print("    quadidealcoh3momentum    last3 minus first3 ideal coh3 signed bar")
            print("    quadhealthscore          composite health score bar + decomposition")
            print("    quadthirds               ideal frac in early/mid/late thirds + arc label")
            print("    quadweightedscores       conf-mass ideal/drifting bars + ratio label")
            print("    quadconfidencemassratio  ideal conf mass / drifting conf mass bar")
            print("    quadfirststeps           first step index ideal + drifting reached")
            print("    quadearlylateideal       early/late ideal frac bars + trend label")
            print("    quadidealfractrend       late minus early ideal frac signed bar")
            print("    quadearlylatedrifting    early/late drifting frac bars + Δ label")
            print("    quadruncounts            ideal + drifting episode counts bar")
            print("    quadidealruncount        number of distinct ideal episodes bar")
            print("    quaddriftingruncount     number of distinct drifting episodes bar")
            print("    quadidealmeanstreak      mean length of ideal runs bar")
            print("    quaddriftingmeanstreak   mean length of drifting runs bar")
            print("    quadstreakvariability    σ of all run lengths across all quadrants bar")
            print("    quadmaxstreaks           all 4 quadrant max-streak bars + dominant label")
            print("    quadidealstreak          longest ideal-quadrant run bar")
            print("    quaddriftingstreak       longest drifting-quadrant run bar")
            print("    quadexploringstreak      longest exploring-quadrant run bar")
            print("    quadflatstreak           longest flat-quadrant run bar")
            print("    confplateau [win]    detect flat conf_ema windows (≥win steps)")
            print("    vocabjump [thr]      steps where vprec_ema dropped > thr")
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

        if low.startswith("topicshift"):
            # topicshift — steps where context velocity > mean+1std (topic jumps)
            _ts_res = getattr(model, "_last_gen_result", None)
            if _ts_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ts_vel = _ts_res.get("velocity_steps", [])
            _ts_tok = _ts_res.get("tokens", [])
            if not _ts_vel:
                print("  No velocity data in last result."); continue
            import numpy as _np_ts
            _ts_arr = _np_ts.array(_ts_vel, dtype=float)
            _ts_mean = float(_ts_arr.mean())
            _ts_std  = float(_ts_arr.std())
            _ts_thr  = _ts_mean + _ts_std
            print(f"\n  Topic-shift detector  (threshold = mean+1σ = {_ts_thr:.4f})")
            print(f"  mean={_ts_mean:.4f}  std={_ts_std:.4f}  steps={len(_ts_arr)}")
            _ts_shifts = [(i, v) for i, v in enumerate(_ts_arr) if v > _ts_thr]
            if not _ts_shifts:
                print("  No topic shifts detected — velocity was stable throughout.")
            else:
                print(f"  {'Step':<6}  {'Token':<18}  {'Velocity':<10}  Marker")
                for _tsi, _tsv in _ts_shifts:
                    _tst = _ts_tok[_tsi] if _tsi < len(_ts_tok) else "?"
                    _tsbar = "▲" * min(8, int((_tsv - _ts_mean) / (_ts_std + 1e-9) * 2))
                    print(f"  {_tsi:<6}  {_tst[:18]:<18}  {_tsv:<10.4f}  {_tsbar}")
            continue

        if low == "lastgen":
            # lastgen — compact one-line summary of the last generation
            _lg_res = getattr(model, "_last_gen_result", None)
            if _lg_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _lg_c  = _lg_res.get("conf_ema_final",  0.0)
            _lg_co = _lg_res.get("coherence",        0.0)
            _lg_c3 = _lg_res.get("coh3",             0.0)
            _lg_ct = _lg_res.get("coh_trend",        0.0)
            _lg_fl = _lg_res.get("fluency",          0.0)
            _lg_pp = _lg_res.get("pseudo_ppl",       0.0)
            _lg_rh = _lg_res.get("rhythm_score",     0.0)
            _lg_tx = _lg_res.get("text",             "(empty)")
            _lg_n   = len(_lg_res.get("tokens",       []))
            _lg_st  = _lg_res.get("stop_reason",      "?")
            _lg_fs  = _lg_res.get("flow_score",       0.0)
            _lg_bs  = _lg_res.get("bestseg",          {})
            _lg_ws  = _lg_res.get("worstseg",         {})
            print(f"\n  Last gen ({_lg_n} tok  stop={_lg_st}):")
            print(f"  Text: {_lg_tx[:72]}")
            print(f"  ConfEMA:{_lg_c:.4f}  Coh:{_lg_co:+.3f}  C3:{_lg_c3:+.3f}  "
                  f"CohTrend:{_lg_ct:+.4f}  Flu:{_lg_fl:.2f}  "
                  f"PPL:{_lg_pp:.1f}  Rhythm:{_lg_rh:.2f}  "
                  f"Flow:{_lg_fs:.2f}  "
                  f"Best@{_lg_bs.get('start','?')}({_lg_bs.get('val',0):.3f})  "
                  f"Worst@{_lg_ws.get('start','?')}({_lg_ws.get('val',0):.3f})")
            continue

        if low.startswith("genprofile"):
            # genprofile <prompt> — run prompt then inline tokenmap + coh + varplot
            _gp_pmt = cmd[len("genprofile"):].strip()
            if not _gp_pmt:
                print("  Usage: genprofile <prompt>"); continue
            print(f"\n  Genprofile: '{_gp_pmt[:50]}' ...")
            import sys as _gp_sys
            # ── run generation ──────────────────────────────────────────────
            _gp_old_last = getattr(model, "_last_gen_result", None)
            _gp_r = model.causal_generate(_gp_pmt, max_tokens=20)
            model._last_gen_result = _gp_r
            _gp_toks  = _gp_r.get("tokens", [])
            _gp_confs = _gp_r.get("confidences", [])
            _gp_c3s   = _gp_r.get("coh3_steps", [])
            _gp_vs    = _gp_r.get("score_var_steps", [])
            _gp_kps   = _gp_r.get("topk_steps", [])
            import numpy as _np_gp
            # ── header ──────────────────────────────────────────────────────
            print(f"  Output: {_gp_r.get('text','(empty)')}")
            print(f"  ConfEMA:{_gp_r.get('conf_ema_final',0):.4f}  "
                  f"Coh:{_gp_r.get('coherence',0):+.3f}  "
                  f"C3:{_gp_r.get('coh3',0):+.3f}  "
                  f"CohTrend:{_gp_r.get('coh_trend',0):+.4f}  "
                  f"Flu:{_gp_r.get('fluency',0):.2f}  "
                  f"PPL:{_gp_r.get('pseudo_ppl',0):.1f}  "
                  f"Rhythm:{_gp_r.get('rhythm_score',0):.2f}  "
                  f"Toks:{len(_gp_toks)}")
            # ── tokenmap ────────────────────────────────────────────────────
            if _gp_toks:
                _SPARKS_GP = " ▁▂▃▄▅▆▇█"
                _gp_cbar   = "".join(
                    _SPARKS_GP[min(8, int(c * 8))] for c in _gp_confs
                )
                print(f"\n  [Confidence] {_gp_cbar}")
                _gp_cats = []
                for _t in _gp_toks:
                    _l = len(_t)
                    if   _l <= 2: _gp_cats.append("\033[33mS\033[0m")
                    elif _l <= 4: _gp_cats.append("\033[32mM\033[0m")
                    elif _l <= 7: _gp_cats.append("\033[36mL\033[0m")
                    else:         _gp_cats.append("\033[35mX\033[0m")
                print(f"  [Tokenmap]   {''.join(_gp_cats)}")
            # ── coh3 sparkline ──────────────────────────────────────────────
            if _gp_c3s:
                _gp_c3arr = _np_gp.array(_gp_c3s, dtype=float)
                _gp_c3spark = "".join(
                    " ▁▂▃▄▅▆▇█"[min(8, int((max(c, 0)) * 8))]
                    for c in _gp_c3arr
                )
                print(f"  [Coh3]       {_gp_c3spark}  "
                      f"avg={_gp_c3arr.mean():.3f}  "
                      f"min={_gp_c3arr.min():.3f}  "
                      f"max={_gp_c3arr.max():.3f}")
            # ── score variance sparkline ────────────────────────────────────
            if _gp_vs:
                _gp_varr = _np_gp.array(_gp_vs, dtype=float)
                _gp_vmax = float(_gp_varr.max()) if _gp_varr.max() > 0 else 1.0
                _gp_vspark = "".join(
                    " ▁▂▃▄▅▆▇█"[min(8, int(_v / _gp_vmax * 8))]
                    for _v in _gp_varr
                )
                print(f"  [Var]        {_gp_vspark}  "
                      f"avg={_gp_varr.mean():.3f}")
            # ── topk sparkline ──────────────────────────────────────────────
            if _gp_kps:
                _kp_arr2 = _np_gp.array(_gp_kps, dtype=float)
                _kp_sp2  = "".join(
                    " ▁▂▃▄▅▆▇█"[min(8, int((_v - 20) / 40 * 8))]
                    for _v in _kp_arr2
                )
                print(f"  [TopK]       {_kp_sp2}  "
                      f"avg={_kp_arr2.mean():.1f}")
            # ── topic shifts ────────────────────────────────────────────────
            _gp_vel = _gp_r.get("velocity_steps", [])
            if _gp_vel:
                _gp_varr2 = _np_gp.array(_gp_vel, dtype=float)
                _gp_vthr  = float(_gp_varr2.mean()) + float(_gp_varr2.std())
                _gp_shifts = [
                    (i, _gp_toks[i] if i < len(_gp_toks) else "?", v)
                    for i, v in enumerate(_gp_varr2) if v > _gp_vthr
                ]
                if _gp_shifts:
                    _gp_stext = "  ".join(
                        f"step{i}:{t}({v:.3f})"
                        for i, t, v in _gp_shifts
                    )
                    print(f"  [Shifts]     {_gp_stext}")
                else:
                    print(f"  [Shifts]     none (velocity stable)")
            continue

        if low.startswith("velspike"):
            # velspike — steps where ctx velocity > mean+1std (rapid shifts)
            _vs_res = getattr(model, "_last_gen_result", None)
            if _vs_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _vs_vel = _vs_res.get("velocity_steps", [])
            _vs_tok = _vs_res.get("tokens", [])
            if not _vs_vel:
                print("  No velocity data in last result."); continue
            import numpy as _np_vs
            _vs_arr  = _np_vs.array(_vs_vel, dtype=float)
            _vs_mean = float(_vs_arr.mean())
            _vs_std  = float(_vs_arr.std())
            _vs_thr  = _vs_mean + _vs_std
            _vs_ratio = _vs_res.get("vel_ratio", 1.0)
            _SPARKS_VS = " ▁▂▃▄▅▆▇█"
            _vs_max = float(_vs_arr.max()) if _vs_arr.max() > 0 else 1.0
            _vs_spark = "".join(
                ("!" if v >= _vs_thr else
                 _SPARKS_VS[min(8, int(v / _vs_max * 8))])
                for v in _vs_arr
            )
            print(f"\n  Velocity spike detector  (threshold = mean+1σ = {_vs_thr:.4f})")
            print(f"  VelRatio={_vs_ratio:.2f}  mean={_vs_mean:.4f}  "
                  f"std={_vs_std:.4f}  max={_vs_arr.max():.4f}  '!' = spike")
            print(f"  {_vs_spark}")
            _vs_spikes = [(i, v) for i, v in enumerate(_vs_arr) if v >= _vs_thr]
            if not _vs_spikes:
                print("  No velocity spikes — context shifted smoothly.")
            else:
                print(f"  {'Step':<6}  {'Token':<18}  Velocity  ×mean")
                for _vsi, _vsv in _vs_spikes:
                    _vst = _vs_tok[_vsi] if _vsi < len(_vs_tok) else "?"
                    _vsb = "▲" * min(8, max(1, int((_vsv - _vs_mean) / (_vs_std + 1e-9) * 2)))
                    print(f"  {_vsi:<6}  {_vst[:18]:<18}  "
                          f"{_vsv:.4f}    {_vsv / (_vs_mean + 1e-9):.2f}×  {_vsb}")
            continue

        if low.startswith("confbands"):
            # confbands — compact L/M/H band strip for all tokens in last gen
            _cb_res = getattr(model, "_last_gen_result", None)
            if _cb_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cb_toks  = _cb_res.get("tokens", [])
            _cb_confs = _cb_res.get("confidences", [])
            if not _cb_toks:
                print("  No tokens in last result."); continue
            # Bands: L=<0.30  M=0.30-0.65  H=≥0.65
            _cb_bands = []
            for _cc in _cb_confs:
                if   _cc < 0.30: _cb_bands.append(("\033[31mL\033[0m", "L"))
                elif _cc < 0.65: _cb_bands.append(("\033[33mM\033[0m", "M"))
                else:             _cb_bands.append(("\033[32mH\033[0m", "H"))
            _cb_strip = "".join(b[0] for b in _cb_bands)
            _cb_raw   = "".join(b[1] for b in _cb_bands)
            _cb_nL = _cb_raw.count("L")
            _cb_nM = _cb_raw.count("M")
            _cb_nH = _cb_raw.count("H")
            print(f"\n  Confidence bands: L=red(<0.30)  M=yellow(0.30-0.65)  H=green(≥0.65)")
            print(f"  {_cb_strip}")
            print(f"  L:{_cb_nL}  M:{_cb_nM}  H:{_cb_nH}  "
                  f"(out of {len(_cb_toks)} tokens)")
            # Show per-token detail
            print(f"\n  {'Step':<5}  {'Band':<5}  {'Conf':<6}  Token")
            for _cbi, (_cbt, (_cbc_col, _cbc_l), _cbconf) in enumerate(
                    zip(_cb_toks, _cb_bands, _cb_confs)):
                print(f"  {_cbi:<5}  {_cbc_l:<5}  {_cbconf:.4f}  {_cbt}")
            continue

        if low.startswith("repchain"):
            # repchain — detect where same token repeated 3+ times within 6 steps
            _rc_res = getattr(model, "_last_gen_result", None)
            if _rc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _rc_toks = _rc_res.get("tokens", [])
            if len(_rc_toks) < 3:
                print("  Not enough tokens."); continue
            _rc_window = 6
            _rc_min_rep = 3
            _rc_chains = []
            for _rci in range(len(_rc_toks)):
                _window = _rc_toks[_rci: _rci + _rc_window]
                from collections import Counter as _RC_Ctr
                _wc = _RC_Ctr(_window)
                for _rct, _rcn in _wc.items():
                    if _rcn >= _rc_min_rep:
                        _rc_chains.append((_rci, _rct, _rcn, _rci + _rc_window))
            # Deduplicate: only keep first occurrence of each (token, count) chain
            _rc_seen = set()
            _rc_uniq = []
            for _rci, _rct, _rcn, _rce in _rc_chains:
                _key = (_rct, _rcn)
                if _key not in _rc_seen:
                    _rc_seen.add(_key)
                    _rc_uniq.append((_rci, _rct, _rcn, _rce))
            print(f"\n  Repetition chain detector  "
                  f"(window={_rc_window}, min_rep={_rc_min_rep})")
            print(f"  {len(_rc_toks)} tokens  {len(_rc_uniq)} chain(s) found")
            if not _rc_uniq:
                print("  No repetition chains — generation was lexically diverse.")
            else:
                print(f"  {'Start':<6}  {'End':<6}  {'Token':<18}  Count")
                for _rci, _rct, _rcn, _rce in _rc_uniq:
                    _rcbar = "●" * _rcn
                    print(f"  {_rci:<6}  {min(_rce,len(_rc_toks)):<6}  "
                          f"{_rct[:18]:<18}  {_rcn}×  {_rcbar}")
            continue

        if low.startswith("spikeplot"):
            # spikeplot [N] — steps where H_norm entropy ≥ 1.5× its mean (spikes)
            _sp_res = getattr(model, "_last_gen_result", None)
            if _sp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _sp_ent = _sp_res.get("entropy_steps", [])
            _sp_tok = _sp_res.get("tokens", [])
            if not _sp_ent:
                print("  No entropy step data in last result."); continue
            import numpy as _np_sp
            _sp_arr  = _np_sp.array(_sp_ent, dtype=float)
            _sp_mean = float(_sp_arr.mean())
            _sp_thr  = _sp_mean * 1.50
            _sp_parts = low.split()
            _sp_n = (int(_sp_parts[1]) if len(_sp_parts) > 1
                     and _sp_parts[1].isdigit() else len(_sp_ent))
            # Sparkline with spikes highlighted
            _SPARKS_SP = " ▁▂▃▄▅▆▇█"
            _sp_max = float(_sp_arr.max()) if _sp_arr.max() > 0 else 1.0
            _sp_spark = "".join(
                ("!" if v >= _sp_thr else
                 _SPARKS_SP[min(8, int(v / _sp_max * 8))])
                for v in _sp_arr[:_sp_n]
            )
            _sp_ratio = _sp_res.get("entropy_ratio", 1.0)
            print(f"\n  Entropy spike detector  (threshold = 1.5×mean = {_sp_thr:.4f})")
            print(f"  EntRatio={_sp_ratio:.2f}  mean={_sp_mean:.4f}  "
                  f"max={_sp_arr.max():.4f}  '!' = spike step")
            print(f"  {_sp_spark}")
            _sp_spikes = [(i, v) for i, v in enumerate(_sp_arr[:_sp_n]) if v >= _sp_thr]
            if not _sp_spikes:
                print("  No entropy spikes — distribution was stable throughout.")
            else:
                print(f"  {'Step':<6}  {'Token':<18}  H_norm   ×mean")
                for _spi, _spv in _sp_spikes:
                    _spt = _sp_tok[_spi] if _spi < len(_sp_tok) else "?"
                    print(f"  {_spi:<6}  {_spt[:18]:<18}  "
                          f"{_spv:.4f}   {_spv/(_sp_mean+1e-9):.2f}×")
            continue

        if low.startswith("confplateau"):
            # confplateau [win] — detect consecutive flat conf_ema windows
            _cp_res = getattr(model, "_last_gen_result", None)
            if _cp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cp_emas = _cp_res.get("conf_ema_steps", [])
            _cp_toks = _cp_res.get("tokens", [])
            if len(_cp_emas) < 3:
                print("  Not enough conf_ema data."); continue
            import numpy as _np_cp
            _cp_parts = low.split()
            _cp_win = (int(_cp_parts[1]) if len(_cp_parts) > 1
                       and _cp_parts[1].isdigit() else 3)
            _cp_arr = _np_cp.array(_cp_emas, dtype=float)
            _cp_diffs = _np_cp.abs(_np_cp.diff(_cp_arr))
            _CP_FLAT = 0.005  # ≤ 0.005 change = flat
            # Find runs of _cp_win or more consecutive flat steps
            _cp_plateaus = []
            _cp_run = 0
            for _cpi, _cpd in enumerate(_cp_diffs):
                if _cpd <= _CP_FLAT:
                    _cp_run += 1
                    if _cp_run >= _cp_win:
                        _cp_plateaus.append(_cpi - _cp_win + 2)  # start step
                else:
                    _cp_run = 0
            _cp_uniq_starts = sorted(set(_cp_plateaus))
            print(f"\n  Confidence-plateau detector  "
                  f"(window={_cp_win} steps, flat threshold=±{_CP_FLAT})")
            print(f"  {len(_cp_emas)} conf_ema steps  "
                  f"{len(_cp_uniq_starts)} plateau(s) found")
            if not _cp_uniq_starts:
                print("  No flat plateaus — confidence was always changing.")
            else:
                for _cps in _cp_uniq_starts:
                    _cpt_end = min(_cps + _cp_win - 1, len(_cp_toks) - 1)
                    _cpt_txt = _cp_toks[_cps] if _cps < len(_cp_toks) else "?"
                    _cp_val  = _cp_arr[_cps] if _cps < len(_cp_arr) else 0.0
                    print(f"  step {_cps}–{_cpt_end}  tok='{_cpt_txt}'  "
                          f"ema={float(_cp_val):.4f}")
            continue

        if low.startswith("segplot"):
            # segplot — rolling 3-step coh3 window for each position
            _seqp_res = getattr(model, "_last_gen_result", None)
            if _seqp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _seqp_c3s = _seqp_res.get("coh3_steps", [])
            _seqp_tok = _seqp_res.get("tokens", [])
            if len(_seqp_c3s) < 3:
                print("  Need ≥3 coh3 steps."); continue
            import numpy as _np_seqp
            _seqp_arr = _np_seqp.array(_seqp_c3s, dtype=float)
            _seqp_wins = [
                float(_np_seqp.mean(_seqp_arr[i: i + 3]))
                for i in range(len(_seqp_arr) - 2)
            ]
            _seqp_max = max(max(_seqp_wins), 1e-9)
            _SPARKS_SQ = " ▁▂▃▄▅▆▇█"
            _seqp_spark = "".join(
                _SPARKS_SQ[min(8, int(max(v, 0) / _seqp_max * 8))]
                for v in _seqp_wins
            )
            _bs  = _seqp_res.get("bestseg",  {})
            _ws  = _seqp_res.get("worstseg", {})
            print(f"\n  Rolling-3 coh3 window  ({len(_seqp_wins)} positions):")
            print(f"  {_seqp_spark}")
            print(f"  Best @step {_bs.get('start','?')}: {_bs.get('val',0):.4f}  "
                  f"Worst @step {_ws.get('start','?')}: {_ws.get('val',0):.4f}")
            print(f"  {'Pos':<5}  {'Token':<18}  Coh3-win  Bar")
            for _sqi, _sqv in enumerate(_seqp_wins):
                _sqt  = _seqp_tok[_sqi + 1] if _sqi + 1 < len(_seqp_tok) else "?"
                _sqbl = int(20 * max(_sqv, 0) / _seqp_max)
                _sqbar = "█" * _sqbl + "░" * (20 - _sqbl)
                _sqstar = " ★" if _sqi == _bs.get("start") else (
                          " ☆" if _sqi == _ws.get("start") else "")
                print(f"  {_sqi:<5}  {_sqt[:18]:<18}  {_sqv:.4f}    {_sqbar}{_sqstar}")
            continue

        if low.startswith("flowbar"):
            # flowbar — per-step B/C/H/N flow indicator strip with legend
            _fb_res = getattr(model, "_last_gen_result", None)
            if _fb_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _fb_steps = _fb_res.get("flow_steps", [])
            _fb_toks  = _fb_res.get("tokens", [])
            if not _fb_steps:
                print("  No flow step data."); continue
            _fb_map = {
                "B": "\033[32m▲▲\033[0m",   # both rising — green
                "C": "\033[36m▲=\033[0m",   # only conf rising — cyan
                "H": "\033[33m=▲\033[0m",   # only coh3 rising — yellow
                "N": "\033[31m▼▼\033[0m",   # neither — red
            }
            _fb_strip = "".join(_fb_map.get(s, "??") for s in _fb_steps)
            _fb_nB = _fb_steps.count("B")
            _fb_nC = _fb_steps.count("C")
            _fb_nH = _fb_steps.count("H")
            _fb_nN = _fb_steps.count("N")
            _fb_fs = _fb_res.get("flow_score", 0.0)
            print(f"\n  Flow bar: ▲▲=both rising  ▲==conf  =▲=coh3  ▼▼=neither")
            print(f"  {_fb_strip}")
            print(f"  B(both):{_fb_nB}  C(conf):{_fb_nC}  "
                  f"H(coh3):{_fb_nH}  N(neither):{_fb_nN}  "
                  f"FlowScore:{_fb_fs:.2f}")
            print(f"\n  {'Step':<5}  {'Code':<5}  {'Token':<18}  Meaning")
            _fb_meaning = {"B": "both↑", "C": "conf↑", "H": "coh3↑", "N": "both↓"}
            for _fbi, _fbs in enumerate(_fb_steps):
                _fbt = _fb_toks[_fbi + 1] if _fbi + 1 < len(_fb_toks) else "?"
                print(f"  {_fbi:<5}  {_fbs:<5}  {_fbt[:18]:<18}  {_fb_meaning[_fbs]}")
            continue

        if low.startswith("vocabjump"):
            # vocabjump [thr] — steps where vprec_ema dropped sharply
            _vj_res = getattr(model, "_last_gen_result", None)
            if _vj_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _vj_steps = _vj_res.get("vprec_ema_steps", [])
            _vj_toks  = _vj_res.get("tokens", [])
            if len(_vj_steps) < 2:
                print("  Not enough vprec_ema data."); continue
            import numpy as _np_vj
            _vj_parts = low.split()
            # threshold: default 0.05 drop per step
            _vj_thr = (float(_vj_parts[1]) if len(_vj_parts) > 1
                       else 0.05)
            _vj_arr  = _np_vj.array(_vj_steps, dtype=float)
            _vj_diffs = _np_vj.diff(_vj_arr)  # negative = drop
            _vj_drops = [(i + 1, float(-d)) for i, d in enumerate(_vj_diffs)
                         if d < -_vj_thr]
            _SPARKS_VJ = " ▁▂▃▄▅▆▇█"
            _vj_max = float(_vj_arr.max()) if _vj_arr.max() > 0 else 1.0
            _vj_spark = "".join(
                _SPARKS_VJ[min(8, int(v / _vj_max * 8))] for v in _vj_arr
            )
            print(f"\n  VocabJump detector  (drop threshold={_vj_thr:.3f}/step)")
            print(f"  {_vj_spark}  avg={_vj_arr.mean():.3f}  "
                  f"final={_vj_arr[-1]:.3f}")
            if not _vj_drops:
                print(f"  No sharp drops found (threshold {_vj_thr:.3f}).")
            else:
                print(f"  {'Step':<6}  {'Token':<18}  Drop")
                for _vji, _vjd in _vj_drops:
                    _vjt = _vj_toks[_vji] if _vji < len(_vj_toks) else "?"
                    _vjbar = "▼" * min(8, max(1, int(_vjd / _vj_thr)))
                    print(f"  {_vji:<6}  {_vjt[:18]:<18}  -{_vjd:.4f}  {_vjbar}")
            continue

        if low.startswith("vprecslope"):
            # vprecslope — vprec_ema slope scatter ▲/─/▼ per step
            _vps_res = getattr(model, "_last_gen_result", None)
            if _vps_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _vps_sl = _vps_res.get("vprec_slope_steps", [])
            _vps_vp = _vps_res.get("vprec_ema_steps",   [])
            if not _vps_sl and not _vps_vp:
                print("  No vprec_ema data."); continue
            if not _vps_sl and _vps_vp:
                _vps_sl = [0.0] + [round(_vps_vp[i] - _vps_vp[i-1], 6)
                                   for i in range(1, len(_vps_vp))]
            _vps_scatter = "".join(
                ("▲" if v > 0.002 else ("▼" if v < -0.002 else "─"))
                for v in _vps_sl
            )
            _vps_rises = sum(1 for v in _vps_sl if v > 0.002)
            _vps_drops = sum(1 for v in _vps_sl if v < -0.002)
            _vps_flat  = len(_vps_sl) - _vps_rises - _vps_drops
            _vps_net   = (_vps_vp[-1] - _vps_vp[0]) if _vps_vp else 0.0
            print(f"\n  VPrec_EMA slope scatter  ({len(_vps_sl)} steps):")
            print(f"  {_vps_scatter}")
            print(f"  ▲ rises={_vps_rises} ({100*_vps_rises/max(len(_vps_sl),1):.1f}%)  "
                  f"▼ drops={_vps_drops} ({100*_vps_drops/max(len(_vps_sl),1):.1f}%)  "
                  f"─ flat={_vps_flat}")
            print(f"  net Δvprec={_vps_net:+.5f}  "
                  f"(start={_vps_vp[0] if _vps_vp else 0.0:.5f}  "
                  f"end={_vps_vp[-1] if _vps_vp else 0.0:.5f})")
            print(f"  (▼ = model broadening vocab; ▲ = converging on familiar words)")
            continue

        if low.startswith("cohconfplot"):
            # cohconfplot — coh3 × confidence quality product sparkline
            _ccp_res = getattr(model, "_last_gen_result", None)
            if _ccp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ccp_steps = _ccp_res.get("coh_conf_steps", [])
            _ccp_c3    = _ccp_res.get("coh3_steps",     [])
            _ccp_cf    = _ccp_res.get("confidences",    [])
            if not _ccp_steps:
                if _ccp_c3 and _ccp_cf:
                    _ccp_steps = [round(_ccp_c3[i] * _ccp_cf[i], 5)
                                  for i in range(min(len(_ccp_c3), len(_ccp_cf)))]
                else:
                    print("  No coh3 or confidence data."); continue
            import numpy as _np_ccp
            _ccp_arr = _np_ccp.array(_ccp_steps, dtype=float)
            _ccp_max = max(float(_ccp_arr.max()), 1e-9)
            _SPARKS_CCP = " ▁▂▃▄▅▆▇█"
            _ccp_spark = "".join(
                _SPARKS_CCP[min(8, int(max(v, 0) / _ccp_max * 8))]
                for v in _ccp_steps
            )
            print(f"\n  Coh3 × Confidence quality product  ({len(_ccp_steps)} steps):")
            print(f"  avg={_ccp_arr.mean():.5f}  "
                  f"max={_ccp_arr.max():.5f}  "
                  f"min={_ccp_arr.min():.5f}")
            print(f"  {_ccp_spark}")
            print(f"  (high = coherent + confident; dips = weak quality moments)")
            continue

        if low.startswith("qualityplot"):
            # qualityplot — smoothed coh3 × confidence quality EMA sparkline
            _qp_res = getattr(model, "_last_gen_result", None)
            if _qp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qp_qs = _qp_res.get("quality_steps", [])
            _qp_cc = _qp_res.get("coh_conf_steps", [])
            if not _qp_qs and not _qp_cc:
                print("  No quality data."); continue
            _qp_data = _qp_qs or _qp_cc
            import numpy as _np_qp
            _qp_arr = _np_qp.array(_qp_data, dtype=float)
            _qp_max = max(float(_qp_arr.max()), 1e-9)
            _SPARKS_QP = " ▁▂▃▄▅▆▇█"
            _qp_spark = "".join(
                _SPARKS_QP[min(8, int(max(v, 0) / _qp_max * 8))]
                for v in _qp_data
            )
            _qp_label = "smoothed quality (coh3×conf EMA)" if _qp_qs else "raw quality (coh3×conf)"
            print(f"\n  Quality trajectory — {_qp_label}  ({len(_qp_data)} steps):")
            print(f"  avg={_qp_arr.mean():.5f}  "
                  f"max={_qp_arr.max():.5f}  "
                  f"min={_qp_arr.min():.5f}  "
                  f"final={_qp_arr[-1]:.5f}")
            print(f"  {_qp_spark}")
            print(f"  (high = coherent + confident moment; dips = uncertainty/drift)")
            continue

        if low.startswith("confcohslopecorr"):
            # confcohslopecorr — conf_ema slope vs coh3 slope sparklines + Pearson r
            _ccsc_res = getattr(model, "_last_gen_result", None)
            if _ccsc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ccsc_cfe = _ccsc_res.get("conf_ema_steps",  [])
            _ccsc_c3  = _ccsc_res.get("coh3_steps",      [])
            _ccsc_r   = _ccsc_res.get("conf_coh_slope_corr", 0.0)
            _ccsc_n   = min(len(_ccsc_cfe), len(_ccsc_c3))
            if _ccsc_n < 3:
                print("  Need ≥3 steps of both conf_ema and coh3."); continue
            # compute slopes
            _ccsc_cfe_sl = [0.0] + [_ccsc_cfe[i] - _ccsc_cfe[i-1] for i in range(1, _ccsc_n)]
            _ccsc_c3_sl  = [0.0] + [_ccsc_c3[i]  - _ccsc_c3[i-1]  for i in range(1, _ccsc_n)]
            _SPARKS_CCSC = " ▁▂▃▄▅▆▇█"
            _ccsc_cf_abs = [abs(v) for v in _ccsc_cfe_sl]
            _ccsc_c3_abs = [abs(v) for v in _ccsc_c3_sl]
            _ccsc_cfmax = max(max(_ccsc_cf_abs), 1e-9)
            _ccsc_c3max = max(max(_ccsc_c3_abs), 1e-9)
            _ccsc_cf_sp = "".join(
                ("▲" if v > 0.001 else ("▼" if v < -0.001 else "─"))
                for v in _ccsc_cfe_sl
            )
            _ccsc_c3_sp = "".join(
                ("▲" if v > 0.001 else ("▼" if v < -0.001 else "─"))
                for v in _ccsc_c3_sl
            )
            _ccsc_interp = ("both move together" if _ccsc_r > 0.40 else
                            ("anti-correlated (unusual)" if _ccsc_r < -0.40 else "decoupled"))
            print(f"\n  Conf_EMA slope vs Coh3 slope  (r={_ccsc_r:+.4f}  → {_ccsc_interp}):")
            print(f"  Conf slope: {_ccsc_cf_sp}")
            print(f"  Coh3 slope: {_ccsc_c3_sp}")
            print(f"  (+ r = confidence + coherence rise/fall together, typical healthy gen)")
            continue

        if low.startswith("confvpreccorr"):
            # confvpreccorr — confidence vs vprec_ema sparklines + Pearson r
            _cvp_res2 = getattr(model, "_last_gen_result", None)
            if _cvp_res2 is None:
                print("  No generation yet. Run a prompt first."); continue
            _cvp_cf2  = _cvp_res2.get("confidences",    [])
            _cvp_vp2  = _cvp_res2.get("vprec_ema_steps",[])
            _cvp_r2   = _cvp_res2.get("conf_vprec_corr", 0.0)
            _cvp_n2   = min(len(_cvp_cf2), len(_cvp_vp2))
            if _cvp_n2 < 3:
                print("  Need ≥3 steps of both confidence and vprec_ema."); continue
            _cvp_cf2  = _cvp_cf2[:_cvp_n2]
            _cvp_vp2  = _cvp_vp2[:_cvp_n2]
            _SPARKS_CVP2 = " ▁▂▃▄▅▆▇█"
            _cvp_c_max2 = max(max(_cvp_cf2), 1e-9)
            _cvp_v_max2 = max(max(_cvp_vp2), 1e-9)
            _cvp_c_sp2 = "".join(
                _SPARKS_CVP2[min(8, int(max(v, 0) / _cvp_c_max2 * 8))] for v in _cvp_cf2
            )
            _cvp_v_sp2 = "".join(
                _SPARKS_CVP2[min(8, int(max(v, 0) / _cvp_v_max2 * 8))] for v in _cvp_vp2
            )
            _cvp_interp2 = ("confident = precise vocab" if _cvp_r2 > 0.35 else
                            ("confident ≠ precise (decoupled)" if _cvp_r2 < -0.35 else "weak coupling"))
            print(f"\n  Confidence vs VPrec_EMA  (r={_cvp_r2:+.4f}  → {_cvp_interp2}):")
            print(f"  Conf:  {_cvp_c_sp2}  avg={sum(_cvp_cf2)/_cvp_n2:.4f}")
            print(f"  VPrec: {_cvp_v_sp2}  avg={sum(_cvp_vp2)/_cvp_n2:.4f}")
            print(f"  (positive r = model is confident when it's on familiar ground)")
            continue

        if low.startswith("qualspikeplot"):
            # qualspikeplot — annotated steps where coh3×conf quality spiked
            _qsp_res = getattr(model, "_last_gen_result", None)
            if _qsp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qsp_spikes = _qsp_res.get("qual_spike_steps", [])
            _qsp_c3   = _qsp_res.get("coh3_steps",  [])
            _qsp_cf   = _qsp_res.get("confidences", [])
            _qsp_toks = _qsp_res.get("tokens",      [])
            if not _qsp_c3 or not _qsp_cf:
                print("  No quality data."); continue
            import numpy as _np_qsp
            _qsp_raw = [_qsp_c3[i] * _qsp_cf[i]
                        if i < len(_qsp_cf) else 0.0
                        for i in range(len(_qsp_c3))]
            _qsp_arr = _np_qsp.array(_qsp_raw, dtype=float)
            _qsp_thr = (float(_qsp_arr.mean()) + 1.5 * float(_qsp_arr.std())
                        if len(_qsp_arr) >= 4 else 0.0)
            print(f"\n  Quality spike steps  (threshold={_qsp_thr:.5f}  "
                  f"mean={_qsp_arr.mean():.5f}  σ={_qsp_arr.std():.5f}):")
            if not _qsp_spikes:
                print("  No quality spikes detected.")
            else:
                for _qsi in _qsp_spikes:
                    _qst  = _qsp_toks[_qsi] if _qsi < len(_qsp_toks) else "?"
                    _qsv  = _qsp_raw[_qsi]
                    _qbar = "★" * min(5, max(1, int((_qsv - _qsp_thr) / max(_qsp_arr.max() - _qsp_thr, 1e-9) * 5)))
                    print(f"  step {_qsi:<4}  {_qst[:20]:<20}  {_qsv:.5f}  {_qbar}")
            print(f"  ({len(_qsp_spikes)} quality spikes in {len(_qsp_raw)} steps)")
            continue

        if low.startswith("quadshape"):
            _qsh_res = getattr(model, "_last_gen_result", None)
            if _qsh_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qsh_c3ld  = _qsh_res.get("quad_coh3_linear_deviation",    0.0)
            _qsh_c3mvl = _qsh_res.get("quad_coh3_midpoint_vs_linear",  0.0)
            _qsh_vld   = _qsh_res.get("quad_vel_linear_deviation",     0.0)
            _qsh_vmvl  = _qsh_res.get("quad_vel_midpoint_vs_linear",   0.0)
            _qsh_c3shp = ("arch" if _qsh_c3mvl > 0.01 else "bowl" if _qsh_c3mvl < -0.01 else "linear")
            _qsh_vshp  = ("peak" if _qsh_vmvl > 0.01 else "dip" if _qsh_vmvl < -0.01 else "linear")
            _qsh_vshp_lbl = ("velocity peaks mid (bad)" if _qsh_vshp == "peak"
                              else "velocity dips mid (good=focus episode)" if _qsh_vshp == "dip"
                              else "velocity is roughly linear")
            print(f"\n  Signal shape analysis (vs linear interpolation)")
            print(f"  coh3: linear_dev={_qsh_c3ld:.6f}  mid_vs_linear={_qsh_c3mvl:+.6f}  "
                  f"shape=[{_qsh_c3shp}]")
            print(f"  vel:  linear_dev={_qsh_vld:.6f}  mid_vs_linear={_qsh_vmvl:+.6f}  "
                  f"shape=[{_qsh_vshp}]  ({_qsh_vshp_lbl})")
            continue

        if low.startswith("quadcounts"):
            _qct_res = getattr(model, "_last_gen_result", None)
            if _qct_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qct_tot  = _qct_res.get("quad_total_step_count",     0)
            _qct_id   = _qct_res.get("quad_ideal_step_count",     0)
            _qct_dr   = _qct_res.get("quad_drifting_step_count",  0)
            _qct_ex   = _qct_res.get("quad_exploring_step_count", 0)
            _qct_fl   = _qct_res.get("quad_flat_step_count",      0)
            _qct_bar  = 25
            print(f"\n  Quadrant step counts  (total={_qct_tot})")
            for _qlbl, _qn in [("ideal    ", _qct_id), ("drifting ", _qct_dr),
                                ("exploring", _qct_ex), ("flat     ", _qct_fl)]:
                _qfrac = _qn / max(_qct_tot, 1)
                _qfill = int(_qfrac * _qct_bar)
                print(f"  {_qlbl}: {_qn:4d}  ({_qfrac:.3f})  {'█'*_qfill}{'░'*(_qct_bar-_qfill)}")
            continue

        if low.startswith("quaddensity"):
            _qdd_res = getattr(model, "_last_gen_result", None)
            if _qdd_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qdd_q1 = _qdd_res.get("quad_q1_ideal_frac", 0.0)
            _qdd_q2 = _qdd_res.get("quad_q2_ideal_frac", 0.0)
            _qdd_q3 = _qdd_res.get("quad_q3_ideal_frac", 0.0)
            _qdd_q4 = _qdd_res.get("quad_q4_ideal_frac", 0.0)
            _qdd_pk = _qdd_res.get("quad_ideal_peak_quartile", 0)
            _qdd_gf = _qdd_res.get("quad_ideal_frac", 0.0)
            _qdd_bar_len = 30
            print(f"\n  Ideal-state density across quartiles  (global avg={_qdd_gf:.4f})")
            for _qi, _qv in [("Q1 0-25%  ", _qdd_q1), ("Q2 25-50% ", _qdd_q2),
                              ("Q3 50-75% ", _qdd_q3), ("Q4 75-100%", _qdd_q4)]:
                _qfill = int(_qv * _qdd_bar_len)
                _qstar = " ★peak" if int(_qi[1]) == _qdd_pk else ""
                print(f"  {_qi} {_qv:.4f}  {'█'*_qfill}{'░'*(_qdd_bar_len-_qfill)}{_qstar}")
            continue

        if low.startswith("quadpanorama"):
            _qpn_res = getattr(model, "_last_gen_result", None)
            if _qpn_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qpn_hs  = _qpn_res.get("quad_head_score",            0.0)
            _qpn_ms  = _qpn_res.get("quad_mid_score",             0.0)
            _qpn_ts  = _qpn_res.get("quad_tail_score",            0.0)
            _qpn_oss = _qpn_res.get("quad_overall_segment_score", 0.0)
            _qpn_sr  = _qpn_res.get("quad_segment_range",         0.0)
            _qpn_scores = {"head": _qpn_hs, "mid": _qpn_ms, "tail": _qpn_ts}
            _qpn_best  = max(_qpn_scores, key=_qpn_scores.get)
            _qpn_worst = min(_qpn_scores, key=_qpn_scores.get)
            _qpn_bar = 30
            print(f"\n  Generation quality panorama (head / mid / tail)")
            for _seg, _sv in [("head (0-25%)  ", _qpn_hs), ("mid  (25-75%) ", _qpn_ms), ("tail (75-100%)", _qpn_ts)]:
                _sf = int(_sv * _qpn_bar)
                _sstar = " ★best" if _seg.strip().startswith(_qpn_best) else (" ▼worst" if _seg.strip().startswith(_qpn_worst) else "")
                print(f"  {_seg} {_sv:.4f}  {'█'*_sf}{'░'*(_qpn_bar-_sf)}{_sstar}")
            print(f"  overall (0.25h+0.5m+0.25t): {_qpn_oss:.4f}  segment_range: {_qpn_sr:.4f}")
            continue

        if low.startswith("quadpeak"):
            _qpk_res = getattr(model, "_last_gen_result", None)
            if _qpk_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qpk_ps  = _qpk_res.get("quad_peak_step",          0.0)
            _qpk_pc  = _qpk_res.get("quad_peak_coh3",          0.0)
            _qpk_cf  = _qpk_res.get("quad_peak_conf",          0.0)
            _qpk_tr  = _qpk_res.get("quad_trough_step",        0.0)
            _qpk_rng = _qpk_res.get("quad_peak_to_trough_range", 0.0)
            _qpk_ps_lbl = ("early" if _qpk_ps < 0.35 else "late" if _qpk_ps > 0.65 else "mid")
            _qpk_tr_lbl = ("early" if _qpk_tr < 0.35 else "late" if _qpk_tr > 0.65 else "mid")
            print(f"\n  Coh3 peak/trough analysis")
            print(f"  peak:   coh3={_qpk_pc:.6f}  pos={_qpk_ps:.4f} [{_qpk_ps_lbl}]  "
                  f"conf_at_peak={_qpk_cf:.6f}")
            print(f"  trough: pos={_qpk_tr:.4f} [{_qpk_tr_lbl}]  "
                  f"min_coh3={_qpk_pc - _qpk_rng:.6f}")
            print(f"  dynamic range (peak−trough): {_qpk_rng:.6f}")
            continue

        if low.startswith("quadmid"):
            _qm_res = getattr(model, "_last_gen_result", None)
            if _qm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qm_if  = _qm_res.get("quad_mid_ideal_frac", 0.0)
            _qm_c3  = _qm_res.get("quad_mid_coh3_mean",  0.0)
            _qm_vel = _qm_res.get("quad_mid_vel_mean",   0.0)
            _qm_cf  = _qm_res.get("quad_mid_conf_mean",  0.0)
            _qm_sc  = _qm_res.get("quad_mid_score",      0.0)
            _qm_gif = _qm_res.get("quad_ideal_frac",     0.0)
            _qm_sc_lbl = ("strong" if _qm_sc >= 0.65 else "adequate" if _qm_sc >= 0.45 else "weak")
            _qm_vs_lbl = ("better" if _qm_if > _qm_gif + 0.05
                           else "worse" if _qm_if < _qm_gif - 0.05 else "same as")
            print(f"\n  Mid analysis (middle 50% of steps)")
            print(f"  ideal_frac: {_qm_if:.4f}  [{_qm_vs_lbl} global {_qm_gif:.4f}]")
            print(f"  coh3_mean:  {_qm_c3:.6f}  vel_mean: {_qm_vel:.6f}  conf_mean: {_qm_cf:.6f}")
            print(f"  mid_score:  {_qm_sc:.4f}  [{_qm_sc_lbl}]")
            continue

        if low.startswith("quadarc"):
            _qa_res = getattr(model, "_last_gen_result", None)
            if _qa_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qa_ga   = _qa_res.get("quad_generation_arc",        0.0)
            _qa_sd   = _qa_res.get("quad_head_tail_score_delta",  0.0)
            _qa_id   = _qa_res.get("quad_head_tail_ideal_delta",  0.0)
            _qa_c3d  = _qa_res.get("quad_head_tail_coh3_delta",   0.0)
            _qa_vd   = _qa_res.get("quad_head_tail_vel_delta",    0.0)
            _qa_hs   = _qa_res.get("quad_head_score",             0.0)
            _qa_ts   = _qa_res.get("quad_tail_score",             0.0)
            _qa_arc_lbl = ("improving" if _qa_ga > 0.03 else "declining" if _qa_ga < -0.03 else "stable")
            print(f"\n  Generation arc (head → tail comparison)")
            print(f"  head_score: {_qa_hs:.4f}  →  tail_score: {_qa_ts:.4f}  "
                  f"delta={_qa_sd:+.4f}  [{_qa_arc_lbl}]")
            print(f"  ideal delta:  {_qa_id:+.4f}  "
                  f"({'↑ more ideal at end' if _qa_id > 0.05 else '↓ less ideal at end' if _qa_id < -0.05 else '~flat'})")
            print(f"  coh3 delta:   {_qa_c3d:+.6f}  "
                  f"({'↑ coherence grows' if _qa_c3d > 1e-5 else '↓ coherence falls' if _qa_c3d < -1e-5 else '~flat'})")
            print(f"  vel delta:    {_qa_vd:+.6f}  "
                  f"({'↓ decelerating (good)' if _qa_vd < -1e-5 else '↑ accelerating (bad)' if _qa_vd > 1e-5 else '~flat'})")
            continue

        if low.startswith("quadhead"):
            _qh_res = getattr(model, "_last_gen_result", None)
            if _qh_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qh_if  = _qh_res.get("quad_head_ideal_frac", 0.0)
            _qh_c3  = _qh_res.get("quad_head_coh3_mean",  0.0)
            _qh_vel = _qh_res.get("quad_head_vel_mean",   0.0)
            _qh_cf  = _qh_res.get("quad_head_conf_mean",  0.0)
            _qh_sc  = _qh_res.get("quad_head_score",      0.0)
            _qh_gif = _qh_res.get("quad_ideal_frac",      0.0)
            _qh_sc_lbl = ("strong" if _qh_sc >= 0.65 else "adequate" if _qh_sc >= 0.45 else "weak")
            _qh_vs_lbl = ("better" if _qh_if > _qh_gif + 0.05
                           else "worse" if _qh_if < _qh_gif - 0.05 else "same as")
            print(f"\n  Head analysis (first 25% of steps)")
            print(f"  ideal_frac: {_qh_if:.4f}  [{_qh_vs_lbl} global {_qh_gif:.4f}]")
            print(f"  coh3_mean:  {_qh_c3:.6f}  vel_mean: {_qh_vel:.6f}  conf_mean: {_qh_cf:.6f}")
            print(f"  head_score: {_qh_sc:.4f}  [{_qh_sc_lbl}]  "
                  f"(0.4×ideal + 0.3×coh3 + 0.3×(1−vel))")
            continue

        if low.startswith("quadtail"):
            _qt_res = getattr(model, "_last_gen_result", None)
            if _qt_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qt_if   = _qt_res.get("quad_tail_ideal_frac", 0.0)
            _qt_c3   = _qt_res.get("quad_tail_coh3_mean",  0.0)
            _qt_vel  = _qt_res.get("quad_tail_vel_mean",   0.0)
            _qt_cf   = _qt_res.get("quad_tail_conf_mean",  0.0)
            _qt_sc   = _qt_res.get("quad_tail_score",      0.0)
            _qt_gif  = _qt_res.get("quad_ideal_frac",      0.0)
            _qt_sc_lbl = ("strong" if _qt_sc >= 0.65 else "adequate" if _qt_sc >= 0.45 else "weak")
            _qt_vs_lbl = ("better" if _qt_if > _qt_gif + 0.05
                           else "worse" if _qt_if < _qt_gif - 0.05 else "same as")
            print(f"\n  Tail analysis (last 25% of steps)")
            print(f"  ideal_frac: {_qt_if:.4f}  [{_qt_vs_lbl} global {_qt_gif:.4f}]")
            print(f"  coh3_mean:  {_qt_c3:.6f}  vel_mean: {_qt_vel:.6f}  conf_mean: {_qt_cf:.6f}")
            print(f"  tail_score: {_qt_sc:.4f}  [{_qt_sc_lbl}]  "
                  f"(0.4×ideal + 0.3×coh3 + 0.3×(1−vel))")
            continue

        if low.startswith("quadtransitions"):
            _qtr_res = getattr(model, "_last_gen_result", None)
            if _qtr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qtr_ii  = _qtr_res.get("quad_ideal_to_ideal_frac",  0.0)
            _qtr_id  = _qtr_res.get("quad_ideal_to_drift_frac",  0.0)
            _qtr_di  = _qtr_res.get("quad_drift_to_ideal_frac",  0.0)
            _qtr_dd  = _qtr_res.get("quad_drift_to_drift_frac",  0.0)
            _qtr_st  = _qtr_res.get("quad_transition_stability", 0.0)
            _qtr_st_lbl  = ("sticky" if _qtr_st >= 0.6 else "fluid" if _qtr_st >= 0.35 else "chaotic")
            _qtr_rec_lbl = ("fast recovery" if _qtr_di >= 0.5
                             else "slow recovery" if _qtr_di >= 0.2 else "rarely recovers")
            print(f"\n  Quadrant transition matrix")
            print(f"  ideal  → ideal:   {_qtr_ii:.4f}  {'█'*int(_qtr_ii*30)}")
            print(f"  ideal  → drifting:{_qtr_id:.4f}  {'█'*int(_qtr_id*30)}")
            print(f"  drift  → ideal:   {_qtr_di:.4f}  {'█'*int(_qtr_di*30)}  [{_qtr_rec_lbl}]")
            print(f"  drift  → drift:   {_qtr_dd:.4f}  {'█'*int(_qtr_dd*30)}")
            print(f"  transition_stability: {_qtr_st:.4f}  [{_qtr_st_lbl}]")
            continue

        if low.startswith("quadruns"):
            _qrl_res = getattr(model, "_last_gen_result", None)
            if _qrl_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qrl_imx  = _qrl_res.get("quad_ideal_max_run",      0)
            _qrl_dmx  = _qrl_res.get("quad_drifting_max_run",   0)
            _qrl_imn  = _qrl_res.get("quad_ideal_mean_run",     0.0)
            _qrl_rb   = _qrl_res.get("quad_run_balance",        0.0)
            _qrl_if   = _qrl_res.get("quad_ideal_frac",         0.0)
            _qrl_df   = _qrl_res.get("quad_drifting_frac",      0.0)
            _qrl_rb_lbl = ("ideal clusters dominate" if _qrl_rb > 1.1
                            else "drift clusters dominate" if _qrl_rb < 0.9 else "balanced")
            print(f"\n  Quadrant run-length analysis")
            print(f"  ideal:    frac={_qrl_if:.4f}  max_run={_qrl_imx}  mean_run={_qrl_imn:.2f}")
            print(f"  drifting: frac={_qrl_df:.4f}  max_run={_qrl_dmx}")
            print(f"  run_balance (ideal_max/drift_max): {_qrl_rb:.4f}  [{_qrl_rb_lbl}]")
            continue

        if low.startswith("quadcorrelations"):
            _qcr_res = getattr(model, "_last_gen_result", None)
            if _qcr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcr_c3v  = _qcr_res.get("quad_coh3_vel_correlation",          0.0)
            _qcr_c3c  = _qcr_res.get("quad_coh3_conf_correlation",         0.0)
            _qcr_vc   = _qcr_res.get("quad_vel_conf_correlation",          0.0)
            _qcr_anti = _qcr_res.get("quad_coh3_vel_anticorrelation_score", 0.0)
            _qcr_c3v_lbl = ("good" if _qcr_c3v < -0.2 else "weak" if _qcr_c3v < 0.2 else "bad")
            _qcr_c3c_lbl = ("good" if _qcr_c3c >  0.2 else "weak" if _qcr_c3c > -0.2 else "bad")
            print(f"\n  Cross-signal Pearson correlations")
            print(f"  r(coh3, vel):  {_qcr_c3v:+.4f}  [{_qcr_c3v_lbl}]  "
                  f"anticorr_score={_qcr_anti:.4f}  (negative=ideal)")
            print(f"  r(coh3, conf): {_qcr_c3c:+.4f}  [{_qcr_c3c_lbl}]  (positive=good)")
            print(f"  r(vel,  conf): {_qcr_vc:+.4f}")
            continue

        if low.startswith("quaduniformity"):
            _qun_res = getattr(model, "_last_gen_result", None)
            if _qun_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qun_ic3s = _qun_res.get("quad_ideal_coh3_std",     0.0)
            _qun_dc3s = _qun_res.get("quad_drifting_coh3_std",  0.0)
            _qun_icfs = _qun_res.get("quad_ideal_conf_std",     0.0)
            _qun_iu   = _qun_res.get("quad_ideal_uniformity",   0.0)
            _qun_ic3m = _qun_res.get("quad_ideal_coh3_mean",    0.0)
            _qun_dcm  = _qun_res.get("quad_drifting_coh3_mean", 0.0)
            _qun_iu_lbl = ("uniform" if _qun_iu >= 0.90 else "varied" if _qun_iu >= 0.70 else "scattered")
            print(f"\n  Per-quadrant signal uniformity")
            print(f"  ideal coh3:    mean={_qun_ic3m:.6f}  std={_qun_ic3s:.6f}")
            print(f"  drifting coh3: mean={_qun_dcm:.6f}  std={_qun_dc3s:.6f}")
            print(f"  ideal conf:    std={_qun_icfs:.6f}")
            print(f"  ideal_uniformity: {_qun_iu:.4f}  [{_qun_iu_lbl}]  "
                  f"(1 − mean(ideal_coh3_std, ideal_conf_std))")
            continue

        if low.startswith("quadmomentum"):
            _qmom_res = getattr(model, "_last_gen_result", None)
            if _qmom_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qmom_c3  = _qmom_res.get("quad_coh3_momentum",        0.0)
            _qmom_vel = _qmom_res.get("quad_vel_momentum",         0.0)
            _qmom_cf  = _qmom_res.get("quad_conf_momentum",        0.0)
            _qmom_c3c = _qmom_res.get("quad_coh3_curvature_mean",  0.0)
            _qmom_vc  = _qmom_res.get("quad_vel_curvature_mean",   0.0)
            _qmom_c3_lbl  = ("rising" if _qmom_c3  > 1e-6 else "falling" if _qmom_c3  < -1e-6 else "flat")
            _qmom_vel_lbl = ("accel"  if _qmom_vel > 1e-6 else "decel"   if _qmom_vel < -1e-6 else "flat")
            _qmom_cf_lbl  = ("rising" if _qmom_cf  > 1e-6 else "falling" if _qmom_cf  < -1e-6 else "flat")
            print(f"\n  Signal momentum (mean signed Δ per step) and curvature")
            print(f"  coh3  momentum: {_qmom_c3:+.6f}  [{_qmom_c3_lbl}]  "
                  f"curvature={_qmom_c3c:.6f}")
            print(f"  vel   momentum: {_qmom_vel:+.6f}  [{_qmom_vel_lbl}]  "
                  f"curvature={_qmom_vc:.6f}")
            print(f"  conf  momentum: {_qmom_cf:+.6f}  [{_qmom_cf_lbl}]")
            continue

        if low.startswith("quadconfdeltas"):
            _qcfd_res = getattr(model, "_last_gen_result", None)
            if _qcfd_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcfd_pf  = _qcfd_res.get("quad_conf_positive_delta_frac",  0.0)
            _qcfd_nf  = _qcfd_res.get("quad_conf_negative_delta_frac",  0.0)
            _qcfd_mpd = _qcfd_res.get("quad_conf_mean_positive_delta",  0.0)
            _qcfd_mnd = _qcfd_res.get("quad_conf_mean_negative_delta",  0.0)
            _qcfd_da  = _qcfd_res.get("quad_conf_delta_asymmetry",      0.0)
            _qcfd_da_lbl = ("rises>falls" if _qcfd_da > 1.1
                             else "falls>rises" if _qcfd_da < 0.9 else "balanced")
            _qcfd_dir = ("rising" if _qcfd_pf > _qcfd_nf + 0.05
                          else "falling" if _qcfd_nf > _qcfd_pf + 0.05 else "oscillating")
            print(f"\n  Confidence delta analysis (signed step-to-step changes)")
            print(f"  positive delta frac: {_qcfd_pf:.4f}  {'█'*int(_qcfd_pf*40)}  (rising steps)")
            print(f"  negative delta frac: {_qcfd_nf:.4f}  {'█'*int(_qcfd_nf*40)}  (falling steps)")
            print(f"  mean +delta: {_qcfd_mpd:.6f}  mean -delta: {_qcfd_mnd:.6f}")
            print(f"  asymmetry: {_qcfd_da:.4f}  [{_qcfd_da_lbl}]  direction: [{_qcfd_dir}]")
            continue

        if low.startswith("quadveldeltas"):
            _qvd_res = getattr(model, "_last_gen_result", None)
            if _qvd_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qvd_pf  = _qvd_res.get("quad_vel_positive_delta_frac",  0.0)
            _qvd_nf  = _qvd_res.get("quad_vel_negative_delta_frac",  0.0)
            _qvd_mpd = _qvd_res.get("quad_vel_mean_positive_delta",  0.0)
            _qvd_mnd = _qvd_res.get("quad_vel_mean_negative_delta",  0.0)
            _qvd_da  = _qvd_res.get("quad_vel_delta_asymmetry",      0.0)
            _qvd_da_lbl = ("decel>accel" if _qvd_da > 1.1
                            else "accel>decel" if _qvd_da < 0.9 else "balanced")
            _qvd_dir = ("accelerating" if _qvd_pf > _qvd_nf + 0.05
                         else "decelerating" if _qvd_nf > _qvd_pf + 0.05 else "oscillating")
            print(f"\n  Velocity delta analysis (signed step-to-step changes)")
            print(f"  positive delta frac: {_qvd_pf:.4f}  {'█'*int(_qvd_pf*40)}  (accel steps)")
            print(f"  negative delta frac: {_qvd_nf:.4f}  {'█'*int(_qvd_nf*40)}  (decel steps)")
            print(f"  mean +delta: {_qvd_mpd:.6f}  mean -delta: {_qvd_mnd:.6f}")
            print(f"  asymmetry: {_qvd_da:.4f}  [{_qvd_da_lbl}]  direction: [{_qvd_dir}]")
            print(f"  (>1 asymmetry = decelerations bigger than accelerations = converging)")
            continue

        if low.startswith("quadcoh3deltas"):
            _qcd_res = getattr(model, "_last_gen_result", None)
            if _qcd_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcd_pf  = _qcd_res.get("quad_coh3_positive_delta_frac",  0.0)
            _qcd_nf  = _qcd_res.get("quad_coh3_negative_delta_frac",  0.0)
            _qcd_mpd = _qcd_res.get("quad_coh3_mean_positive_delta",  0.0)
            _qcd_mnd = _qcd_res.get("quad_coh3_mean_negative_delta",  0.0)
            _qcd_da  = _qcd_res.get("quad_coh3_delta_asymmetry",      0.0)
            _qcd_da_lbl = ("rises>falls" if _qcd_da > 1.1
                            else "falls>rises" if _qcd_da < 0.9 else "balanced")
            _qcd_dir = ("rising" if _qcd_pf > _qcd_nf + 0.05
                         else "falling" if _qcd_nf > _qcd_pf + 0.05 else "oscillating")
            print(f"\n  Coh3 delta analysis (signed step-to-step changes)")
            print(f"  positive delta frac: {_qcd_pf:.4f}  "
                  f"{'█'*int(_qcd_pf*40)}  (rising steps)")
            print(f"  negative delta frac: {_qcd_nf:.4f}  "
                  f"{'█'*int(_qcd_nf*40)}  (falling steps)")
            print(f"  mean +delta: {_qcd_mpd:.6f}  mean -delta: {_qcd_mnd:.6f}")
            print(f"  asymmetry: {_qcd_da:.4f}  [{_qcd_da_lbl}]  direction: [{_qcd_dir}]")
            continue

        if low.startswith("quadconfpercentiles"):
            _qcfpc_res = getattr(model, "_last_gen_result", None)
            if _qcfpc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcfpc_p25 = _qcfpc_res.get("quad_conf_p25", 0.0)
            _qcfpc_p50 = _qcfpc_res.get("quad_conf_p50", 0.0)
            _qcfpc_p75 = _qcfpc_res.get("quad_conf_p75", 0.0)
            _qcfpc_iqr = _qcfpc_res.get("quad_conf_iqr", 0.0)
            _qcfpc_mn  = _qcfpc_res.get("quad_conf_min",  0.0)
            _qcfpc_mx  = _qcfpc_res.get("quad_conf_max",  0.0)
            _qcfpc_mu  = _qcfpc_res.get("quad_conf_mean", 0.0)
            print(f"\n  Confidence percentile distribution")
            print(f"  min={_qcfpc_mn:.6f}  p25={_qcfpc_p25:.6f}  p50={_qcfpc_p50:.6f}  "
                  f"p75={_qcfpc_p75:.6f}  max={_qcfpc_mx:.6f}")
            print(f"  mean={_qcfpc_mu:.6f}  IQR={_qcfpc_iqr:.6f}")
            _qcfpc_sk = ("right-skewed" if _qcfpc_mu > _qcfpc_p50+1e-7
                          else "left-skewed" if _qcfpc_mu < _qcfpc_p50-1e-7 else "symmetric")
            print(f"  skew: mean-p50={_qcfpc_mu-_qcfpc_p50:+.6f}  [{_qcfpc_sk}]")
            continue

        if low.startswith("quadvelpercentiles"):
            _qvpc_res = getattr(model, "_last_gen_result", None)
            if _qvpc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qvpc_p25 = _qvpc_res.get("quad_vel_p25", 0.0)
            _qvpc_p50 = _qvpc_res.get("quad_vel_p50", 0.0)
            _qvpc_p75 = _qvpc_res.get("quad_vel_p75", 0.0)
            _qvpc_iqr = _qvpc_res.get("quad_vel_iqr", 0.0)
            _qvpc_mn  = _qvpc_res.get("quad_vel_min",  0.0)
            _qvpc_mx  = _qvpc_res.get("quad_vel_max",  0.0)
            _qvpc_mu  = _qvpc_res.get("quad_velocity_mean", 0.0)
            print(f"\n  Velocity percentile distribution")
            print(f"  min={_qvpc_mn:.6f}  p25={_qvpc_p25:.6f}  p50={_qvpc_p50:.6f}  "
                  f"p75={_qvpc_p75:.6f}  max={_qvpc_mx:.6f}")
            print(f"  mean={_qvpc_mu:.6f}  IQR={_qvpc_iqr:.6f}")
            _qvpc_sk = ("right-skewed" if _qvpc_mu > _qvpc_p50+1e-7
                         else "left-skewed" if _qvpc_mu < _qvpc_p50-1e-7 else "symmetric")
            print(f"  skew: mean-p50={_qvpc_mu-_qvpc_p50:+.6f}  [{_qvpc_sk}]")
            continue

        if low.startswith("quadcoh3percentiles"):
            _qc3pc_res = getattr(model, "_last_gen_result", None)
            if _qc3pc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qc3pc_p25 = _qc3pc_res.get("quad_coh3_p25", 0.0)
            _qc3pc_p50 = _qc3pc_res.get("quad_coh3_p50", 0.0)
            _qc3pc_p75 = _qc3pc_res.get("quad_coh3_p75", 0.0)
            _qc3pc_iqr = _qc3pc_res.get("quad_coh3_iqr", 0.0)
            _qc3pc_mn  = _qc3pc_res.get("quad_coh3_min",  0.0)
            _qc3pc_mx  = _qc3pc_res.get("quad_coh3_max",  0.0)
            _qc3pc_mu  = _qc3pc_res.get("quad_coh3_mean", 0.0)
            print(f"\n  Coh3 percentile distribution")
            print(f"  min={_qc3pc_mn:.6f}  p25={_qc3pc_p25:.6f}  p50={_qc3pc_p50:.6f}  "
                  f"p75={_qc3pc_p75:.6f}  max={_qc3pc_mx:.6f}")
            print(f"  mean={_qc3pc_mu:.6f}  IQR={_qc3pc_iqr:.6f}")
            _qc3pc_sk = ("right-skewed" if _qc3pc_mu > _qc3pc_p50+1e-7
                          else "left-skewed" if _qc3pc_mu < _qc3pc_p50-1e-7 else "symmetric")
            print(f"  skew: mean-p50={_qc3pc_mu-_qc3pc_p50:+.6f}  [{_qc3pc_sk}]")
            continue

        if low.startswith("quadhealthscore"):
            _qhs_res = getattr(model, "_last_gen_result", None)
            if _qhs_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qhs_ohs = _qhs_res.get("quad_overall_health_score",     0.0)
            _qhs_gqi = _qhs_res.get("quad_generation_quality_index", 0.0)
            _qhs_fs  = _qhs_res.get("quad_focus_score",              0.0)
            _qhs_cfs = _qhs_res.get("quad_coherence_focus_score",    0.0)
            _qhs_if  = _qhs_res.get("quad_ideal_frac",               0.0)
            _qhs_df  = _qhs_res.get("quad_drifting_frac",            0.0)
            _qhs_c3  = _qhs_res.get("quad_coh3_mean",                0.0)
            _qhs_cf  = _qhs_res.get("quad_conf_mean",                0.0)
            _qhs_grade = ("A" if _qhs_gqi >= 0.72 else "B" if _qhs_gqi >= 0.58
                           else "C" if _qhs_gqi >= 0.44 else "D" if _qhs_gqi >= 0.30 else "F")
            _qhs_ohs_lbl = ("excellent" if _qhs_ohs >= 0.70
                             else "good" if _qhs_ohs >= 0.50 else "fair" if _qhs_ohs >= 0.30 else "poor")
            print(f"\n  Generation health scores  [grade: {_qhs_grade}]")
            print(f"  overall_health:   {_qhs_ohs:.4f}  [{_qhs_ohs_lbl}]  "
                  f"(0.25×ideal + 0.25×(1-drift) + 0.25×coh3>0.75 + 0.25×conf>0.75)")
            print(f"  quality_index:    {_qhs_gqi:.4f}  "
                  f"(0.30×ideal + 0.20×coh3 + 0.20×(1-drift) + 0.15×conf + 0.15×(1-vel))")
            print(f"  focus_score:      {_qhs_fs:.4f}  (ideal_frac × (1-vel_mean))")
            print(f"  coherence_focus:  {_qhs_cfs:.6f}  (coh3_mean × (1-vel_mean))")
            print(f"  components: ideal={_qhs_if:.4f} drift={_qhs_df:.4f} "
                  f"coh3={_qhs_c3:.4f} conf={_qhs_cf:.4f}")
            continue

        if low.startswith("quadthirds"):
            _qth_res = getattr(model, "_last_gen_result", None)
            if _qth_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qth_t1i = _qth_res.get("quad_third1_ideal_frac", 0.0)
            _qth_t2i = _qth_res.get("quad_third2_ideal_frac", 0.0)
            _qth_t3i = _qth_res.get("quad_third3_ideal_frac", 0.0)
            _qth_tt  = _qth_res.get("quad_third_trend_ideal", 0.0)
            _qth_t1c = _qth_res.get("quad_third1_coh3_mean", 0.0)
            _qth_t2c = _qth_res.get("quad_third2_coh3_mean", 0.0)
            _qth_t3c = _qth_res.get("quad_third3_coh3_mean", 0.0)
            _qth_t1v = _qth_res.get("quad_third1_vel_mean", 0.0)
            _qth_t2v = _qth_res.get("quad_third2_vel_mean", 0.0)
            _qth_t3v = _qth_res.get("quad_third3_vel_mean", 0.0)
            print(f"\n  Thirds analysis")
            print(f"  ideal:  T1={_qth_t1i:.4f}  T2={_qth_t2i:.4f}  T3={_qth_t3i:.4f}  trend={_qth_tt:+.4f}")
            print(f"  coh3:   T1={_qth_t1c:.6f}  T2={_qth_t2c:.6f}  T3={_qth_t3c:.6f}")
            print(f"  vel:    T1={_qth_t1v:.6f}  T2={_qth_t2v:.6f}  T3={_qth_t3v:.6f}")
            continue

        if low.startswith("quadratiostats"):
            _qrs_res = getattr(model, "_last_gen_result", None)
            if _qrs_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qrs_ier = _qrs_res.get("quad_ideal_to_exploring_ratio", 0.0)
            _qrs_dfr = _qrs_res.get("quad_drift_to_flat_ratio",      0.0)
            _qrs_csf = _qrs_res.get("quad_coherent_states_frac",     0.0)
            _qrs_hvf = _qrs_res.get("quad_high_vel_states_frac",     0.0)
            _qrs_if  = _qrs_res.get("quad_ideal_frac",               0.0)
            _qrs_ef  = _qrs_res.get("quad_exploring_frac",           0.0)
            _qrs_df  = _qrs_res.get("quad_drifting_frac",            0.0)
            _qrs_ff  = _qrs_res.get("quad_flat_frac",                0.0)
            _qrs_ier_lbl = ("ideal-dominant" if _qrs_ier >= 1.5
                             else "balanced" if _qrs_ier >= 0.7 else "explore-dominant")
            print(f"\n  Quadrant ratio statistics")
            print(f"  ideal/exploring ratio: {_qrs_ier:.4f}  [{_qrs_ier_lbl}]")
            print(f"  drift/flat ratio:      {_qrs_dfr:.4f}  (>1=more drift than flat)")
            print(f"  coherent states frac:  {_qrs_csf:.4f}  (ideal+exploring, coh3>median)")
            print(f"  high-vel states frac:  {_qrs_hvf:.4f}  (drift+exploring, vel>=median)")
            print(f"  i={_qrs_if:.3f} e={_qrs_ef:.3f} d={_qrs_df:.3f} f={_qrs_ff:.3f}")
            continue

        if low.startswith("quadconfstats"):
            _qcfs_res = getattr(model, "_last_gen_result", None)
            if _qcfs_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcfs_mn  = _qcfs_res.get("quad_conf_min",           0.0)
            _qcfs_mx  = _qcfs_res.get("quad_conf_max",           0.0)
            _qcfs_rng = _qcfs_res.get("quad_conf_range",         0.0)
            _qcfs_75  = _qcfs_res.get("quad_conf_above_075_frac",0.0)
            _qcfs_90  = _qcfs_res.get("quad_conf_above_090_frac",0.0)
            _qcfs_mu  = _qcfs_res.get("quad_conf_mean",          0.0)
            print(f"\n  Confidence absolute statistics")
            print(f"  min={_qcfs_mn:.6f}  mean={_qcfs_mu:.6f}  max={_qcfs_mx:.6f}  range={_qcfs_rng:.6f}")
            print(f"  frac >0.75: {_qcfs_75:.4f}  {'█'*int(_qcfs_75*40)}")
            print(f"  frac >0.90: {_qcfs_90:.4f}  {'█'*int(_qcfs_90*40)}")
            continue

        if low.startswith("quadvelstats"):
            _qvs_res = getattr(model, "_last_gen_result", None)
            if _qvs_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qvs_mn  = _qvs_res.get("quad_vel_min",           0.0)
            _qvs_mx  = _qvs_res.get("quad_vel_max",           0.0)
            _qvs_rng = _qvs_res.get("quad_vel_range",         0.0)
            _qvs_b25 = _qvs_res.get("quad_vel_below_025_frac",0.0)
            _qvs_am  = _qvs_res.get("quad_vel_above_mean_frac",0.0)
            _qvs_mu  = _qvs_res.get("quad_velocity_mean",     0.0)
            print(f"\n  Velocity absolute statistics")
            print(f"  min={_qvs_mn:.6f}  mean={_qvs_mu:.6f}  max={_qvs_mx:.6f}  range={_qvs_rng:.6f}")
            print(f"  frac <0.25: {_qvs_b25:.4f}  {'█'*int(_qvs_b25*40)}  (low vel = focused steps)")
            print(f"  frac >mean: {_qvs_am:.4f}  (expect ~0.50; skewed = asymmetric vel dist)")
            continue

        if low.startswith("quadcoh3stats"):
            _qc3s_res = getattr(model, "_last_gen_result", None)
            if _qc3s_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qc3s_mn  = _qc3s_res.get("quad_coh3_min",           0.0)
            _qc3s_mx  = _qc3s_res.get("quad_coh3_max",           0.0)
            _qc3s_rng = _qc3s_res.get("quad_coh3_range",         0.0)
            _qc3s_75  = _qc3s_res.get("quad_coh3_above_075_frac",0.0)
            _qc3s_90  = _qc3s_res.get("quad_coh3_above_090_frac",0.0)
            _qc3s_mu  = _qc3s_res.get("quad_coh3_mean",          0.0)
            _qc3s_75_lbl = ("high" if _qc3s_75 >= 0.60 else "moderate" if _qc3s_75 >= 0.30 else "low")
            _qc3s_90_lbl = ("elite" if _qc3s_90 >= 0.30 else "present" if _qc3s_90 > 0 else "absent")
            print(f"\n  Coh3 absolute statistics")
            print(f"  min={_qc3s_mn:.6f}  mean={_qc3s_mu:.6f}  max={_qc3s_mx:.6f}  range={_qc3s_rng:.6f}")
            print(f"  frac >0.75: {_qc3s_75:.4f}  {'█'*int(_qc3s_75*40)}  [{_qc3s_75_lbl}]")
            print(f"  frac >0.90: {_qc3s_90:.4f}  {'█'*int(_qc3s_90*40)}  [{_qc3s_90_lbl}]")
            continue

        if low.startswith("quadbursts"):
            _qbu_res = getattr(model, "_last_gen_result", None)
            if _qbu_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qbu_ibc  = _qbu_res.get("quad_ideal_burst_count",    0)
            _qbu_ibx  = _qbu_res.get("quad_ideal_burst_max_len",  0)
            _qbu_ibm  = _qbu_res.get("quad_ideal_burst_mean_len", 0.0)
            _qbu_dbc  = _qbu_res.get("quad_drift_burst_count",    0)
            _qbu_bqr  = _qbu_res.get("quad_burst_quality_ratio",  0.0)
            _qbu_if   = _qbu_res.get("quad_ideal_frac",           0.0)
            _qbu_bqr_lbl = ("quality-rich" if _qbu_bqr >= 1.5
                             else "balanced"  if _qbu_bqr >= 0.8
                             else "drift-heavy")
            print(f"\n  Burst analysis (runs of ≥2 consecutive steps in same quadrant state)")
            print(f"  ideal bursts:  count={_qbu_ibc}  max_len={_qbu_ibx}  mean_len={_qbu_ibm:.2f}")
            print(f"  drift bursts:  count={_qbu_dbc}")
            print(f"  burst_quality_ratio: {_qbu_bqr:.4f}  [{_qbu_bqr_lbl}]  "
                  f"(ideal_bursts / (drift_bursts+1))")
            print(f"  overall ideal_frac={_qbu_if:.4f}")
            continue

        if low.startswith("quadvolatility"):
            _qvol_res = getattr(model, "_last_gen_result", None)
            if _qvol_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qvol_cv  = _qvol_res.get("quad_coh3_volatility",            0.0)
            _qvol_vv  = _qvol_res.get("quad_vel_volatility",             0.0)
            _qvol_fv  = _qvol_res.get("quad_conf_volatility",            0.0)
            _qvol_rat = _qvol_res.get("quad_coh3_vel_volatility_ratio",  0.0)
            _qvol_sc  = _qvol_res.get("quad_stability_composite",        0.0)
            _qvol_sc_lbl = ("stable"   if _qvol_sc >= 0.70
                             else "moderate" if _qvol_sc >= 0.40
                             else "volatile")
            _qvol_rat_lbl = ("coh3>vel" if _qvol_rat > 1.2
                              else "vel>coh3" if _qvol_rat < 0.8 else "balanced")
            print(f"\n  Signal volatility (std of |step-to-step deltas|)")
            print(f"  coh3:  {_qvol_cv:.6f}")
            print(f"  vel:   {_qvol_vv:.6f}")
            print(f"  conf:  {_qvol_fv:.6f}")
            print(f"  coh3/vel ratio: {_qvol_rat:.4f}  [{_qvol_rat_lbl}]")
            print(f"  stability_composite: {_qvol_sc:.4f}  [{_qvol_sc_lbl}]  "
                  f"(1 − mean_volatility, clipped)")
            continue

        if low.startswith("quadgaps"):
            _qgp_res = getattr(model, "_last_gen_result", None)
            if _qgp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qgp_cf  = _qgp_res.get("quad_ideal_vs_drift_conf_gap",    0.0)
            _qgp_c3  = _qgp_res.get("quad_ideal_vs_drift_coh3_gap",    0.0)
            _qgp_fc  = _qgp_res.get("quad_ideal_vs_flat_coh3_gap",     0.0)
            _qgp_qs  = _qgp_res.get("quad_quality_gap_score",          0.0)
            _qgp_si  = _qgp_res.get("quad_signal_separation_index",    0.0)
            _qgp_cf_lbl = ("high" if _qgp_cf > 0.01 else "low" if _qgp_cf < -0.005 else "neutral")
            _qgp_qs_lbl = ("separated" if _qgp_qs >= 0.50
                            else "partial"  if _qgp_qs >= 0.20 else "overlap")
            _qgp_si_lbl = ("strong" if _qgp_si >= 1.5 else "moderate" if _qgp_si >= 0.5 else "weak")
            print(f"\n  Ideal-vs-drift quality gap analysis")
            print(f"  conf gap  (ideal-drift):  {_qgp_cf:+.6f}  [{_qgp_cf_lbl}]")
            print(f"  coh3 gap  (ideal-drift):  {_qgp_c3:+.6f}")
            print(f"  coh3 gap  (ideal-flat):   {_qgp_fc:+.6f}")
            print(f"  quality_gap_score:        {_qgp_qs:.4f}  [{_qgp_qs_lbl}]  "
                  f"(0.4×conf + 0.6×coh3, normalized)")
            print(f"  signal_separation (Cohen d-like): {_qgp_si:+.4f}  [{_qgp_si_lbl}]")
            continue

        if low.startswith("quadconfprofile"):
            _qcp_res = getattr(model, "_last_gen_result", None)
            if _qcp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcp_ic  = _qcp_res.get("quad_ideal_conf_mean",       0.0)
            _qcp_ec  = _qcp_res.get("quad_exploring_conf_mean",   0.0)
            _qcp_dc  = _qcp_res.get("quad_drifting_conf_mean",    0.0)
            _qcp_fc  = _qcp_res.get("quad_flat_conf_mean",        0.0)
            _qcp_sp  = _qcp_res.get("quad_conf_quadrant_spread",  0.0)
            _qcp_cm  = _qcp_res.get("quad_conf_mean",             0.0)
            _qcp_scale = max(_qcp_ic, _qcp_ec, _qcp_dc, _qcp_fc, 1e-9)
            def _qcp_bar(v): return "█" * int(v / _qcp_scale * 30)
            print(f"\n  Per-quadrant confidence profile  (global_mean={_qcp_cm:.6f})")
            print(f"  ideal:     {_qcp_ic:.6f}  {_qcp_bar(_qcp_ic)}")
            print(f"  exploring: {_qcp_ec:.6f}  {_qcp_bar(_qcp_ec)}")
            print(f"  drifting:  {_qcp_dc:.6f}  {_qcp_bar(_qcp_dc)}")
            print(f"  flat:      {_qcp_fc:.6f}  {_qcp_bar(_qcp_fc)}")
            print(f"  spread (σ of 4 means): {_qcp_sp:.6f}")
            continue

        if low.startswith("quadvelprofile"):
            _qvp_res = getattr(model, "_last_gen_result", None)
            if _qvp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qvp_iv  = _qvp_res.get("quad_ideal_vel_mean",       0.0)
            _qvp_ev  = _qvp_res.get("quad_exploring_vel_mean",   0.0)
            _qvp_dv  = _qvp_res.get("quad_drifting_vel_mean",    0.0)
            _qvp_fv  = _qvp_res.get("quad_flat_vel_mean",        0.0)
            _qvp_sp  = _qvp_res.get("quad_vel_quadrant_spread",  0.0)
            _qvp_vm  = _qvp_res.get("quad_velocity_mean",        0.0)
            _qvp_scale = max(_qvp_iv, _qvp_ev, _qvp_dv, _qvp_fv, 1e-9)
            def _qvp_bar(v): return "█" * int(v / _qvp_scale * 30)
            print(f"\n  Per-quadrant velocity profile  (global_mean={_qvp_vm:.6f})")
            print(f"  ideal:     {_qvp_iv:.6f}  {_qvp_bar(_qvp_iv)}")
            print(f"  exploring: {_qvp_ev:.6f}  {_qvp_bar(_qvp_ev)}")
            print(f"  drifting:  {_qvp_dv:.6f}  {_qvp_bar(_qvp_dv)}")
            print(f"  flat:      {_qvp_fv:.6f}  {_qvp_bar(_qvp_fv)}")
            print(f"  spread (σ of 4 means): {_qvp_sp:.6f}")
            continue

        if low.startswith("quadquarters"):
            _qqt_res = getattr(model, "_last_gen_result", None)
            if _qqt_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qqt_q1i = _qqt_res.get("quad_q1_ideal_frac",     0.0)
            _qqt_q4i = _qqt_res.get("quad_q4_ideal_frac",     0.0)
            _qqt_q1c = _qqt_res.get("quad_q1_coh3_mean",      0.0)
            _qqt_q4c = _qqt_res.get("quad_q4_coh3_mean",      0.0)
            _qqt_arc = _qqt_res.get("quad_quarter_arc_score",  0.0)
            _qqt_hi  = _qqt_res.get("quad_half_improvement_score", 0.0)
            _qqt_n   = _qqt_res.get("quad_total_steps",        1)
            _qqt_arc_lbl = ("rising arc"  if _qqt_arc >= 0.15
                             else "flat arc"  if abs(_qqt_arc) < 0.15
                             else "falling arc")
            print(f"\n  Quarter-split analysis (Q1=first 25%, Q4=last 25% of {_qqt_n} steps)")
            print(f"  ideal_frac:  Q1={_qqt_q1i:.4f}  Q4={_qqt_q4i:.4f}  delta={_qqt_q4i-_qqt_q1i:+.4f}")
            print(f"  coh3_mean:   Q1={_qqt_q1c:.6f}  Q4={_qqt_q4c:.6f}  delta={_qqt_q4c-_qqt_q1c:+.6f}")
            print(f"  arc_score:   {_qqt_arc:+.4f}  [{_qqt_arc_lbl}]  "
                  f"half_improvement={_qqt_hi:+.4f}")
            continue

        if low.startswith("quadhalves"):
            _qhv_res = getattr(model, "_last_gen_result", None)
            if _qhv_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qhv_fi  = _qhv_res.get("quad_first_half_ideal_frac",   0.0)
            _qhv_si  = _qhv_res.get("quad_second_half_ideal_frac",  0.0)
            _qhv_fc3 = _qhv_res.get("quad_first_half_coh3_mean",    0.0)
            _qhv_sc3 = _qhv_res.get("quad_second_half_coh3_mean",   0.0)
            _qhv_imp = _qhv_res.get("quad_half_improvement_score",  0.0)
            _qhv_n   = _qhv_res.get("quad_total_steps",             1)
            _qhv_imp_lbl = ("improving" if _qhv_imp >= 0.10
                             else "flat"   if abs(_qhv_imp) < 0.10
                             else "declining")
            _qhv_id_delta = _qhv_si - _qhv_fi
            _qhv_c3_delta = _qhv_sc3 - _qhv_fc3
            print(f"\n  First vs second half split (split at step {_qhv_n//2}/{_qhv_n})")
            print(f"  ideal_frac:  1st={_qhv_fi:.4f}  2nd={_qhv_si:.4f}  delta={_qhv_id_delta:+.4f}")
            print(f"  coh3_mean:   1st={_qhv_fc3:.6f}  2nd={_qhv_sc3:.6f}  delta={_qhv_c3_delta:+.6f}")
            print(f"  half_improvement: {_qhv_imp:+.4f}  [{_qhv_imp_lbl}]")
            continue

        if low.startswith("quadtrends"):
            _qts_res = getattr(model, "_last_gen_result", None)
            if _qts_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qts_c3s = _qts_res.get("quad_coh3_trend_slope",      0.0)
            _qts_vs  = _qts_res.get("quad_vel_trend_slope",       0.0)
            _qts_cfs = _qts_res.get("quad_conf_trend_slope",      0.0)
            _qts_r2  = _qts_res.get("quad_coh3_trend_r2",         0.0)
            _qts_ta  = _qts_res.get("quad_trend_alignment_score", 0.0)
            _qts_c3s_lbl = ("rising" if _qts_c3s > 1e-5 else "falling" if _qts_c3s < -1e-5 else "flat")
            _qts_vs_lbl  = ("rising" if _qts_vs  > 1e-5 else "falling" if _qts_vs  < -1e-5 else "flat")
            _qts_cfs_lbl = ("rising" if _qts_cfs > 1e-5 else "falling" if _qts_cfs < -1e-5 else "flat")
            _qts_ta_lbl  = ("converging" if _qts_ta >= 0.50
                             else "weak"  if _qts_ta >= 0.20 else "flat")
            print(f"\n  Signal trend analysis (linear regression over step index)")
            print(f"  coh3:  slope={_qts_c3s:+.6f} [{_qts_c3s_lbl}]  R2={_qts_r2:.4f}")
            print(f"  vel:   slope={_qts_vs:+.6f} [{_qts_vs_lbl}]")
            print(f"  conf:  slope={_qts_cfs:+.6f} [{_qts_cfs_lbl}]")
            print(f"  trend_alignment: {_qts_ta:.4f}  [{_qts_ta_lbl}]  "
                  f"(high=coh3 rising + vel falling = ideal convergence)")
            continue

        if low.startswith("quadentropy"):
            _qen_res = getattr(model, "_last_gen_result", None)
            if _qen_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qen_le  = _qen_res.get("quad_label_entropy",   0.0)
            _qen_c3  = _qen_res.get("quad_coh3_entropy",    0.0)
            _qen_ve  = _qen_res.get("quad_vel_entropy",     0.0)
            _qen_cf  = _qen_res.get("quad_conf_entropy",    0.0)
            _qen_ei  = _qen_res.get("quad_entropy_index",   0.0)
            _qen_te  = _qen_res.get("quad_transition_entropy", 0.0)
            _qen_le_lbl = ("diverse" if _qen_le >= 0.85 else "mixed" if _qen_le >= 0.55 else "dominated")
            _qen_ei_lbl = ("high-entropy" if _qen_ei >= 0.65 else "moderate" if _qen_ei >= 0.40 else "low")
            print(f"\n  Entropy diagnostics (normalized [0,1]; 1.0=max diversity)")
            print(f"  label_entropy:      {_qen_le:.4f}  {'█'*int(_qen_le*40)}  [{_qen_le_lbl}]")
            print(f"  coh3_entropy:       {_qen_c3:.4f}  {'█'*int(_qen_c3*40)}")
            print(f"  vel_entropy:        {_qen_ve:.4f}  {'█'*int(_qen_ve*40)}")
            print(f"  conf_entropy:       {_qen_cf:.4f}  {'█'*int(_qen_cf*40)}")
            print(f"  transition_entropy: {_qen_te:.4f}")
            print(f"  entropy_index: {_qen_ei:.4f}  [{_qen_ei_lbl}]  "
                  f"(0.4×label + 0.3×coh3 + 0.3×(1-vel))")
            continue

        if low.startswith("quadtransitions"):
            _qtr_res = getattr(model, "_last_gen_result", None)
            if _qtr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qtr_ip  = _qtr_res.get("quad_ideal_persistence_score",   0.0)
            _qtr_dp  = _qtr_res.get("quad_drift_persistence_score",   0.0)
            _qtr_di  = _qtr_res.get("quad_drifting_to_ideal_rate",    0.0)
            _qtr_ie  = _qtr_res.get("quad_ideal_to_exploring_rate",   0.0)
            _qtr_id  = _qtr_res.get("quad_ideal_to_drifting_rate",    0.0)
            _qtr_ei  = _qtr_res.get("quad_exploring_to_ideal_rate",   0.0)
            _qtr_df  = _qtr_res.get("quad_drifting_to_flat_rate",     0.0)
            _qtr_ms  = _qtr_res.get("quad_markov_stability_score",    0.0)
            _qtr_ms_lbl = ("stable" if _qtr_ms >= 0.65 else "mixed" if _qtr_ms >= 0.40 else "unstable")
            print(f"\n  Markov transition matrix (conditional probabilities)")
            print(f"  FROM ideal:    stay={_qtr_ip:.3f}  ->explore={_qtr_ie:.3f}  ->drift={_qtr_id:.3f}")
            print(f"  FROM explore:  ->ideal={_qtr_ei:.3f}")
            print(f"  FROM drift:    stay={_qtr_dp:.3f}  ->ideal={_qtr_di:.3f}  ->flat={_qtr_df:.3f}")
            print(f"  markov_stability: {_qtr_ms:.4f}  [{_qtr_ms_lbl}]  "
                  f"(0.6x ideal_stay + 0.4x(1-drift_stay))")
            continue

        if low.startswith("quadpersistence"):
            _qps_res = getattr(model, "_last_gen_result", None)
            if _qps_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qps_ip  = _qps_res.get("quad_ideal_persistence_score",  0.0)
            _qps_dp  = _qps_res.get("quad_drift_persistence_score",  0.0)
            _qps_ca1 = _qps_res.get("quad_coh3_autocorr_lag1",       0.0)
            _qps_va1 = _qps_res.get("quad_vel_autocorr_lag1",        0.0)
            _qps_fa1 = _qps_res.get("quad_conf_autocorr_lag1",       0.0)
            _qps_ip_lbl = ("sticky"  if _qps_ip >= 0.70 else "moderate" if _qps_ip >= 0.40 else "fleeting")
            _qps_dp_lbl = ("sticky"  if _qps_dp >= 0.70 else "moderate" if _qps_dp >= 0.40 else "fleeting")
            print(f"\n  Quadrant persistence (P(stay | in state))")
            print(f"  ideal:   {_qps_ip:.4f}  {'█'*int(_qps_ip*40)}  [{_qps_ip_lbl}]")
            print(f"  drifting: {_qps_dp:.4f}  {'█'*int(_qps_dp*40)}  [{_qps_dp_lbl}]")
            print(f"  lag-1 autocorr: coh3={_qps_ca1:+.4f}  vel={_qps_va1:+.4f}  conf={_qps_fa1:+.4f}")
            print(f"  (>0=smooth/persistent signal; <0=oscillating; 1=perfectly smooth)")
            continue

        if low.startswith("quadautocorr"):
            _qac_res = getattr(model, "_last_gen_result", None)
            if _qac_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qac_c3  = _qac_res.get("quad_coh3_autocorr_lag1",  0.0)
            _qac_vl  = _qac_res.get("quad_vel_autocorr_lag1",   0.0)
            _qac_cf  = _qac_res.get("quad_conf_autocorr_lag1",  0.0)
            def _qac_lbl(r):
                if r >= 0.60: return "highly persistent"
                if r >= 0.30: return "moderately smooth"
                if r >= -0.10: return "near-random"
                return "anti-persistent (oscillating)"
            print(f"\n  Lag-1 autocorrelations (signal memory)")
            print(f"  coh3: {_qac_c3:+.4f}  [{_qac_lbl(_qac_c3)}]")
            print(f"  vel:  {_qac_vl:+.4f}  [{_qac_lbl(_qac_vl)}]")
            print(f"  conf: {_qac_cf:+.4f}  [{_qac_lbl(_qac_cf)}]")
            print(f"  (high=signal is smooth/memoryful; negative=signal anti-persists/oscillates)")
            continue

        if low.startswith("quadcorrelations"):
            _qcr_res = getattr(model, "_last_gen_result", None)
            if _qcr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcr_cc  = _qcr_res.get("quad_coh3_conf_correlation",          0.0)
            _qcr_cv  = _qcr_res.get("quad_coh3_vel_correlation",           0.0)
            _qcr_fv  = _qcr_res.get("quad_conf_vel_correlation",           0.0)
            _qcr_icc = _qcr_res.get("quad_ideal_coh3_conf_correlation",    0.0)
            _qcr_cs  = _qcr_res.get("quad_signal_coupling_score",          0.0)
            def _qcr_lbl(r):
                a = abs(r)
                return ("strong" if a >= 0.60 else "moderate" if a >= 0.30 else "weak") + (
                    "-pos" if r >= 0 else "-neg")
            _qcr_cs_lbl = "high-coupling" if _qcr_cs >= 0.70 else "moderate" if _qcr_cs >= 0.45 else "decoupled"
            print(f"\n  Cross-signal correlations (Pearson r)")
            print(f"  coh3↔conf: {_qcr_cc:+.4f} [{_qcr_lbl(_qcr_cc)}]   "
                  f"ideal-only: {_qcr_icc:+.4f} [{_qcr_lbl(_qcr_icc)}]")
            print(f"  coh3↔vel:  {_qcr_cv:+.4f} [{_qcr_lbl(_qcr_cv)}]   "
                  f"(expect negative: high coh3 → low vel)")
            print(f"  conf↔vel:  {_qcr_fv:+.4f} [{_qcr_lbl(_qcr_fv)}]")
            print(f"  signal_coupling_score: {_qcr_cs:.4f} [{_qcr_cs_lbl}]  "
                  f"(abs(c3_cf)×0.5 + (1−abs(c3_vel))×0.5)")
            continue

        if low.startswith("quadvariance"):
            _qvar_res = getattr(model, "_last_gen_result", None)
            if _qvar_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qvar_icv  = _qvar_res.get("quad_ideal_coh3_var",             0.0)
            _qvar_dcv  = _qvar_res.get("quad_drifting_coh3_var",          0.0)
            _qvar_icfv = _qvar_res.get("quad_ideal_conf_var",             0.0)
            _qvar_dcfv = _qvar_res.get("quad_drifting_conf_var",          0.0)
            _qvar_ratio= _qvar_res.get("quad_ideal_vs_drift_coh3_var_ratio", 0.0)
            _qvar_icm  = _qvar_res.get("quad_ideal_coh3_mean",            0.0)
            _qvar_dcm  = _qvar_res.get("quad_drifting_coh3_mean",         0.0)
            _qvar_ratio_lbl = ("ideal noisier" if _qvar_ratio > 1.5
                                else "similar"  if _qvar_ratio >= 0.67
                                else "ideal tighter")
            print(f"\n  Per-quadrant variance diagnostics")
            print(f"  coh3:  ideal_var={_qvar_icv:.6f}  (mean={_qvar_icm:.4f})   "
                  f"drift_var={_qvar_dcv:.6f}  (mean={_qvar_dcm:.4f})")
            print(f"  conf:  ideal_var={_qvar_icfv:.6f}   drift_var={_qvar_dcfv:.6f}")
            print(f"  ideal/drift coh3 var ratio: {_qvar_ratio:.4f}  [{_qvar_ratio_lbl}]")
            print(f"  (<1=ideal signal is tighter/more consistent than drift; >1=ideal is noisier)")
            continue

        if low.startswith("quadsignalquality"):
            _qsq_res = getattr(model, "_last_gen_result", None)
            if _qsq_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qsq_v   = _qsq_res.get("quad_signal_quality_index", 0.0)
            _qsq_qs  = _qsq_res.get("quad_quality_score",        0.0)
            _qsq_sep = _qsq_res.get("quad_separation_score",     0.0)
            _qsq_if  = _qsq_res.get("quad_ideal_frac",           0.0)
            _qsq_df  = _qsq_res.get("quad_drifting_frac",        0.0)
            _qsq_c3m = _qsq_res.get("quad_ideal_coh3_mean",      0.0)
            _qsq_cfm = _qsq_res.get("quad_ideal_conf_mean",      0.0)
            _qsq_lbl = ("excellent" if _qsq_v >= 0.70
                         else "good"    if _qsq_v >= 0.50
                         else "fair"    if _qsq_v >= 0.35
                         else "poor")
            _qsq_bar = "█" * int(_qsq_v * 50)
            print(f"\n  Signal quality index: {_qsq_v:.4f}  [{_qsq_lbl}]")
            print(f"  {_qsq_bar}")
            print(f"  components: ideal={_qsq_if:.3f}×0.35  (1−drift)={1-_qsq_df:.3f}×0.20  coh3_norm×0.25  conf×0.20")
            print(f"  quality_score={_qsq_qs:.4f}  separation={_qsq_sep:.4f}")
            print(f"  ideal_coh3_mean={_qsq_c3m:.4f}  ideal_conf_mean={_qsq_cfm:.4f}")
            continue

        if low.startswith("quadkurtosis"):
            _qku_res = getattr(model, "_last_gen_result", None)
            if _qku_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qku_c3  = _qku_res.get("quad_coh3_kurtosis",  0.0)
            _qku_v   = _qku_res.get("quad_vel_kurtosis",   0.0)
            _qku_cf  = _qku_res.get("quad_conf_kurtosis",  0.0)
            _qku_c3s = _qku_res.get("quad_coh3_skew",      0.0)
            _qku_vs  = _qku_res.get("quad_vel_skew",       0.0)
            _qku_cfs = _qku_res.get("quad_conf_skew",      0.0)
            def _qku_lbl(k): return "heavy-tailed" if k >= 1.0 else "flat" if k <= -1.0 else "normal"
            print(f"\n  Signal distribution shape (excess kurtosis + skewness)")
            print(f"  coh3:  kurtosis={_qku_c3:+.4f} [{_qku_lbl(_qku_c3)}]  skew={_qku_c3s:+.4f}")
            print(f"  vel:   kurtosis={_qku_v:+.4f} [{_qku_lbl(_qku_v)}]  skew={_qku_vs:+.4f}")
            print(f"  conf:  kurtosis={_qku_cf:+.4f} [{_qku_lbl(_qku_cf)}]  skew={_qku_cfs:+.4f}")
            print(f"  (excess kurtosis: >0=heavy tails/outliers; <0=flat/platykurtic; 0=normal)")
            continue

        if low.startswith("quadcentroids"):
            _qct_res = getattr(model, "_last_gen_result", None)
            if _qct_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qct_ic  = _qct_res.get("quad_ideal_centroid",     -1.0)
            _qct_dc  = _qct_res.get("quad_drifting_centroid",  -1.0)
            _qct_cg  = _qct_res.get("quad_centroid_gap",        0.0)
            _qct_n   = _qct_res.get("quad_total_steps",         1)
            _qct_lis = _qct_res.get("quad_late_ideal_start",    0)
            _qct_cg_lbl = ("ideal later"  if _qct_cg >= _qct_n * 0.10
                            else "ideal earlier" if _qct_cg <= -_qct_n * 0.10
                            else "co-located")
            _qct_mid = _qct_n / 2
            def _qct_arrow(c): return "early" if c < _qct_mid * 0.7 else "late" if c > _qct_mid * 1.3 else "mid"
            print(f"\n  Temporal centroids (weighted mean step index; total={_qct_n} steps)")
            print(f"  ideal_centroid:    step {_qct_ic:.1f}  ({_qct_arrow(_qct_ic) if _qct_ic >= 0 else 'n/a'})")
            print(f"  drifting_centroid: step {_qct_dc:.1f}  ({_qct_arrow(_qct_dc) if _qct_dc >= 0 else 'n/a'})")
            print(f"  centroid_gap: {_qct_cg:+.1f} steps  [{_qct_cg_lbl}]  late_start={bool(_qct_lis)}")
            continue

        if low.startswith("quadskew"):
            _qskw_res = getattr(model, "_last_gen_result", None)
            if _qskw_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qskw_c3 = _qskw_res.get("quad_coh3_skew", 0.0)
            _qskw_v  = _qskw_res.get("quad_vel_skew",  0.0)
            _qskw_c3_lbl = ("right-skewed(sparse hi-coh3)" if _qskw_c3 >= 0.5
                             else "left-skewed(dense hi-coh3)" if _qskw_c3 <= -0.5
                             else "near-symmetric")
            _qskw_v_lbl  = ("right-skewed(sparse hi-vel)"  if _qskw_v >= 0.5
                             else "left-skewed(dense hi-vel)"  if _qskw_v <= -0.5
                             else "near-symmetric")
            print(f"\n  Skewness of signal distributions")
            print(f"  coh3_skew:  {_qskw_c3:+.4f}  [{_qskw_c3_lbl}]")
            print(f"  vel_skew:   {_qskw_v:+.4f}  [{_qskw_v_lbl}]")
            print(f"  (negative=distribution is left-skewed=mode above mean; positive=right-skewed=long hi tail)")
            continue

        if low.startswith("quadzigzag"):
            _qzz_res = getattr(model, "_last_gen_result", None)
            if _qzz_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qzz_z   = _qzz_res.get("quad_zigzag_score",        0.0)
            _qzz_oc  = _qzz_res.get("quad_oscillation_count",   0)
            _qzz_irc = _qzz_res.get("quad_ideal_run_count",     0)
            _qzz_erc = _qzz_res.get("quad_exploring_run_count", 0)
            _qzz_drc = _qzz_res.get("quad_drifting_run_count",  0)
            _qzz_frc = _qzz_res.get("quad_flat_run_count",      0)
            _qzz_n   = _qzz_res.get("quad_total_steps",         1)
            _qzz_lbl = ("volatile"   if _qzz_z >= 0.40
                         else "active"  if _qzz_z >= 0.20
                         else "stable")
            _qzz_bar = "█" * int(_qzz_z * 50)
            print(f"\n  Zigzag score (ideal↔other flips/step): {_qzz_z:.4f}  [{_qzz_lbl}]")
            print(f"  {_qzz_bar}")
            print(f"  oscillations={_qzz_oc}  over {_qzz_n} steps")
            print(f"  run counts: ideal={_qzz_irc}  exploring={_qzz_erc}  drifting={_qzz_drc}  flat={_qzz_frc}")
            continue

        if low.startswith("quadexplorestreaks"):
            _qes_res = getattr(model, "_last_gen_result", None)
            if _qes_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qes_me  = _qes_res.get("quad_max_exploring_streak", 0)
            _qes_mf  = _qes_res.get("quad_max_flat_streak",      0)
            _qes_erc = _qes_res.get("quad_exploring_run_count",  0)
            _qes_frc = _qes_res.get("quad_flat_run_count",       0)
            _qes_eml = _qes_res.get("quad_exploring_run_mean_len", 0.0)
            _qes_fml = _qes_res.get("quad_flat_run_mean_len",      0.0)
            _qes_ef  = _qes_res.get("quad_exploring_frac",       0.0)
            _qes_ff  = _qes_res.get("quad_flat_frac",            0.0)
            _qes_scale = max(_qes_me, _qes_mf, 1)
            print(f"\n  Exploring and flat streak diagnostics")
            print(f"  exploring: max={_qes_me}  {'█'*int(_qes_me/_qes_scale*30)}  runs={_qes_erc}  mean={_qes_eml:.2f}  frac={_qes_ef:.4f}")
            print(f"  flat:      max={_qes_mf}  {'█'*int(_qes_mf/_qes_scale*30)}  runs={_qes_frc}  mean={_qes_fml:.2f}  frac={_qes_ff:.4f}")
            continue

        if low.startswith("quadrle"):
            _qrle_res = getattr(model, "_last_gen_result", None)
            if _qrle_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qrle_f5  = _qrle_res.get("quad_rle_first5",           "")
            _qrle_oc  = _qrle_res.get("quad_oscillation_count",    0)
            _qrle_lni = _qrle_res.get("quad_longest_non_ideal_run", 0)
            _qrle_iis = _qrle_res.get("quad_ideal_isolation_score", 0.0)
            _qrle_lis = _qrle_res.get("quad_late_ideal_start",     0)
            _qrle_fp  = _qrle_res.get("quad_fingerprint",          "")
            _qrle_oc_lbl = ("stable"     if _qrle_oc <= 4
                             else "active"  if _qrle_oc <= 12
                             else "volatile")
            print(f"\n  Quadrant sequence structure")
            print(f"  opening_rle:  [{_qrle_f5}]  (i=ideal e=exploring d=drifting f=flat  N=run_length)")
            print(f"  oscillations: {_qrle_oc}  [{_qrle_oc_lbl}]  longest_non_ideal_gap={_qrle_lni} steps")
            print(f"  isolation={_qrle_iis:.4f}  (ideal_run_count/total)  late_start={bool(_qrle_lis)}")
            if _qrle_fp:
                print(f"  fingerprint: {_qrle_fp}")
            continue

        if low.startswith("quadoscillation"):
            _qosc_res = getattr(model, "_last_gen_result", None)
            if _qosc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qosc_oc  = _qosc_res.get("quad_oscillation_count",    0)
            _qosc_lni = _qosc_res.get("quad_longest_non_ideal_run", 0)
            _qosc_iis = _qosc_res.get("quad_ideal_isolation_score", 0.0)
            _qosc_lis = _qosc_res.get("quad_late_ideal_start",      0)
            _qosc_mi  = _qosc_res.get("quad_max_ideal_streak",      0)
            _qosc_rc  = _qosc_res.get("quad_ideal_run_count",       0)
            _qosc_n   = _qosc_res.get("quad_total_steps",           1)
            _qosc_oc_lbl = ("stable" if _qosc_oc <= 4 else "active" if _qosc_oc <= 12 else "volatile")
            print(f"\n  Quadrant oscillation diagnostics")
            print(f"  oscillation_count:     {_qosc_oc}  [{_qosc_oc_lbl}]  (flips between ideal and other)")
            print(f"  longest_non_ideal_run: {_qosc_lni} steps  (max gap without quality flow)")
            print(f"  ideal_isolation:       {_qosc_iis:.4f}  runs/step  (high=fragmented ideal flow)")
            print(f"  ideal_run_count:       {_qosc_rc}  max_streak={_qosc_mi}  late_start={bool(_qosc_lis)}")
            continue

        if low.startswith("quadstreaks"):
            _qsk_res = getattr(model, "_last_gen_result", None)
            if _qsk_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qsk_mi  = _qsk_res.get("quad_max_ideal_streak",       0)
            _qsk_md  = _qsk_res.get("quad_max_drifting_streak",    0)
            _qsk_ms  = _qsk_res.get("quad_max_ideal_streak_start", -1)
            _qsk_isr = _qsk_res.get("quad_ideal_streak_ratio",     0.0)
            _qsk_dsr = _qsk_res.get("quad_drift_streak_ratio",     0.0)
            _qsk_iml = _qsk_res.get("quad_ideal_run_mean_len",     0.0)
            _qsk_dml = _qsk_res.get("quad_drifting_run_mean_len",  0.0)
            _qsk_isr_lbl = ("concentrated" if _qsk_isr >= 0.70
                             else "spread"  if _qsk_isr <= 0.30
                             else "mixed")
            _qsk_dsr_lbl = ("concentrated" if _qsk_dsr >= 0.70
                             else "spread"  if _qsk_dsr <= 0.30
                             else "mixed")
            _qsk_scale = max(_qsk_mi, _qsk_md, 1)
            print(f"\n  Per-quadrant max streaks")
            print(f"  ideal:   max_streak={_qsk_mi}  {'█'*int(_qsk_mi/_qsk_scale*30)}  start@{_qsk_ms}  "
                  f"ratio={_qsk_isr:.3f} [{_qsk_isr_lbl}]  mean={_qsk_iml:.2f}")
            print(f"  drifting: max_streak={_qsk_md}  {'█'*int(_qsk_md/_qsk_scale*30)}  "
                  f"ratio={_qsk_dsr:.3f} [{_qsk_dsr_lbl}]  mean={_qsk_dml:.2f}")
            print(f"  (ratio=max_streak/total_steps; concentrated=one big burst; spread=many small runs)")
            continue

        if low.startswith("quadbalance"):
            _qbal_res = getattr(model, "_last_gen_result", None)
            if _qbal_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qbal_amc = _qbal_res.get("quad_above_median_coh3_frac",       0.0)
            _qbal_bmv = _qbal_res.get("quad_below_median_vel_frac",        0.0)
            _qbal_cir = _qbal_res.get("quad_coherent_to_incoherent_ratio", 0.0)
            _qbal_fcr = _qbal_res.get("quad_focused_to_chaotic_ratio",     0.0)
            _qbal_hv  = _qbal_res.get("quad_health_vector",                "----")
            _qbal_qs  = _qbal_res.get("quad_quality_score",                0.0)
            _qbal_fp  = _qbal_res.get("quad_fingerprint",                  "")
            _qbal_cir_lbl = ("coherent"    if _qbal_cir >= 1.20
                              else "balanced" if _qbal_cir >= 0.80
                              else "incoherent")
            _qbal_fcr_lbl = ("focused"  if _qbal_fcr >= 1.20
                              else "balanced" if _qbal_fcr >= 0.80
                              else "chaotic")
            print(f"\n  Quadrant balance  health_vector=[{_qbal_hv}]  quality_score={_qbal_qs:.4f}")
            print(f"  above_coh3_mean: {_qbal_amc:.4f}  {'█'*int(_qbal_amc*40)}  (symmetric=0.50)")
            print(f"  below_vel_mean:  {_qbal_bmv:.4f}  {'█'*int(_qbal_bmv*40)}  (symmetric=0.50)")
            print(f"  coherent/incoherent: {_qbal_cir:.4f}  [{_qbal_cir_lbl}]   focused/chaotic: {_qbal_fcr:.4f}  [{_qbal_fcr_lbl}]")
            print(f"  [IEDX key: I=ideal E=exploring D=drifting F=flat | upper=dominant(≥35%/25%) lower=present(≥10%) .=absent]")
            if _qbal_fp:
                print(f"  fingerprint: {_qbal_fp}")
            continue

        if low.startswith("quadqualityscore"):
            _qqs_res = getattr(model, "_last_gen_result", None)
            if _qqs_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qqs_v   = _qqs_res.get("quad_quality_score",     0.0)
            _qqs_if  = _qqs_res.get("quad_ideal_frac",        0.0)
            _qqs_df  = _qqs_res.get("quad_drifting_frac",     0.0)
            _qqs_ii  = _qqs_res.get("quad_ideal_improvement", 0.0)
            _qqs_fs  = _qqs_res.get("quad_final_state",       "n/a")
            _qqs_lbl = ("excellent"   if _qqs_v >= 0.65
                         else "good"  if _qqs_v >= 0.45
                         else "fair"  if _qqs_v >= 0.28
                         else "poor")
            _qqs_bar = "█" * int(_qqs_v * 50)
            print(f"\n  Composite quality score: {_qqs_v:.4f}  [{_qqs_lbl}]")
            print(f"  {_qqs_bar}")
            print(f"  components: ideal_frac={_qqs_if:.4f}×0.50  (1−drift)={1-_qqs_df:.4f}×0.25  improve_norm×0.25")
            print(f"  final_state={_qqs_fs}  improvement={_qqs_ii:+.4f}")
            continue

        if low.startswith("quadheadfracs"):
            _qhf_res = getattr(model, "_last_gen_result", None)
            if _qhf_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qhf_ih  = _qhf_res.get("quad_ideal_head_frac",   0.0)
            _qhf_dh  = _qhf_res.get("quad_drift_head_frac",   0.0)
            _qhf_im  = _qhf_res.get("quad_ideal_mid_frac",    0.0)
            _qhf_it  = _qhf_res.get("quad_ideal_tail_frac",   0.0)
            _qhf_dm  = _qhf_res.get("quad_drift_mid_frac",    0.0)
            _qhf_dt  = _qhf_res.get("quad_drift_tail_frac",   0.0)
            _qhf_ii  = _qhf_res.get("quad_ideal_improvement", 0.0)
            _qhf_fs  = _qhf_res.get("quad_final_state",       "n/a")
            _qhf_ii_lbl = ("improving" if _qhf_ii >= 0.05 else "stable" if abs(_qhf_ii) < 0.05 else "degrading")
            print(f"\n  Generation quarters: head / mid / tail  (final_state={_qhf_fs})")
            print(f"  ideal: head={_qhf_ih:.3f}  mid={_qhf_im:.3f}  tail={_qhf_it:.3f}  trend=[{_qhf_ii_lbl}]  delta={_qhf_ii:+.3f}")
            print(f"  drift: head={_qhf_dh:.3f}  mid={_qhf_dm:.3f}  tail={_qhf_dt:.3f}")
            continue

        if low.startswith("quadpeaks"):
            _qpk_res = getattr(model, "_last_gen_result", None)
            if _qpk_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qpk_ipc = _qpk_res.get("quad_ideal_peak_coh3",      0.0)
            _qpk_ipf = _qpk_res.get("quad_ideal_peak_conf",      0.0)
            _qpk_dwc = _qpk_res.get("quad_drifting_worst_coh3",  0.0)
            _qpk_imc = _qpk_res.get("quad_ideal_coh3_mean",      0.0)
            _qpk_imf = _qpk_res.get("quad_ideal_conf_mean",      0.0)
            _qpk_dmc = _qpk_res.get("quad_drifting_coh3_mean",   0.0)
            _qpk_c3g = _qpk_res.get("quad_coh3_gap",             0.0)
            print(f"\n  Per-quadrant peak and floor values")
            print(f"  ideal_peak_coh3:       {_qpk_ipc:.4f}  (mean={_qpk_imc:.4f})")
            print(f"  ideal_peak_conf:       {_qpk_ipf:.4f}  (mean={_qpk_imf:.4f})")
            print(f"  drifting_worst_coh3:   {_qpk_dwc:.4f}  (mean={_qpk_dmc:.4f})")
            print(f"  coh3_gap (ideal−drift): {_qpk_c3g:+.4f}")
            continue

        if low.startswith("quadmidfracs"):
            _qmf_res = getattr(model, "_last_gen_result", None)
            if _qmf_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qmf_im  = _qmf_res.get("quad_ideal_mid_frac",    0.0)
            _qmf_dm  = _qmf_res.get("quad_drift_mid_frac",    0.0)
            _qmf_it  = _qmf_res.get("quad_ideal_tail_frac",   0.0)
            _qmf_if  = _qmf_res.get("quad_ideal_frac",        0.0)
            _qmf_ii  = _qmf_res.get("quad_ideal_improvement", 0.0)
            _qmf_dt  = _qmf_res.get("quad_drift_tail_frac",   0.0)
            _qmf_df  = _qmf_res.get("quad_drifting_frac",     0.0)
            print(f"\n  Generation thirds: head / mid / tail ideal + drift fracs")
            print(f"  ideal:  {'head=?':8}  mid={_qmf_im:.4f}  tail={_qmf_it:.4f}  global={_qmf_if:.4f}")
            print(f"  drift:  {'head=?':8}  mid={_qmf_dm:.4f}  tail={_qmf_dt:.4f}  global={_qmf_df:.4f}")
            print(f"  improvement={_qmf_ii:+.4f}  (tail−head ideal; positive=quality grew over time)")
            continue

        if low.startswith("quadtailfracs"):
            _qtf_res = getattr(model, "_last_gen_result", None)
            if _qtf_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qtf_it  = _qtf_res.get("quad_ideal_tail_frac",    0.0)
            _qtf_dt  = _qtf_res.get("quad_drift_tail_frac",    0.0)
            _qtf_ii  = _qtf_res.get("quad_ideal_improvement",  0.0)
            _qtf_if  = _qtf_res.get("quad_ideal_frac",         0.0)
            _qtf_df  = _qtf_res.get("quad_drifting_frac",      0.0)
            _qtf_fs  = _qtf_res.get("quad_final_state",        "n/a")
            _qtf_ii_lbl = ("improving"  if _qtf_ii >= 0.05
                            else "stable" if abs(_qtf_ii) < 0.05
                            else "degrading")
            print(f"\n  Tail vs overall fracs  (final_state={_qtf_fs})")
            print(f"  ideal_tail:  {_qtf_it:.4f}  (global={_qtf_if:.4f})  {'█'*int(_qtf_it*40)}")
            print(f"  drift_tail:  {_qtf_dt:.4f}  (global={_qtf_df:.4f})  {'█'*int(_qtf_dt*40)}")
            print(f"  improvement: {_qtf_ii:+.4f}  [{_qtf_ii_lbl}]  (tail−head ideal; positive=quality grew)")
            continue

        if low.startswith("quadtransitionentropy"):
            _qte_res = getattr(model, "_last_gen_result", None)
            if _qte_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qte_e   = _qte_res.get("quad_transition_entropy", 0.0)
            _qte_dt  = _qte_res.get("quad_dom_transition",     "none")
            _qte_ii  = _qte_res.get("quad_ideal_improvement",  0.0)
            _qte_fs  = _qte_res.get("quad_final_state",        "n/a")
            _qte_e_lbl = ("unpredictable"   if _qte_e >= 3.0
                           else "moderate"  if _qte_e >= 1.5
                           else "patterned")
            _qte_bar = "█" * int(_qte_e / 4.0 * 40)
            print(f"\n  Quadrant transition entropy: {_qte_e:.4f}  [{_qte_e_lbl}]")
            print(f"  {_qte_bar}")
            print(f"  dominant_transition: [{_qte_dt}]")
            print(f"  final_state={_qtf_fs if False else _qte_fs}  ideal_improvement={_qte_ii:+.4f}")
            print(f"  (max entropy=4.0 bits for 16 equally likely transitions)")
            continue

        if low.startswith("quadfinalstate"):
            _qfst_res = getattr(model, "_last_gen_result", None)
            if _qfst_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qfst_fs  = _qfst_res.get("quad_final_state",          "none")
            _qfst_li  = _qfst_res.get("quad_last_ideal_step",      -1)
            _qfst_ld  = _qfst_res.get("quad_last_drifting_step",   -1)
            _qfst_fi  = _qfst_res.get("quad_first_ideal_step_index", -1)
            _qfst_fd  = _qfst_res.get("quad_first_drifting_step",  -1)
            _qfst_icf = _qfst_res.get("quad_ideal_coverage_frac",  0.0)
            _qfst_n   = _qfst_res.get("quad_total_steps",          1)
            _qfst_lbl = {
                "ideal":     "generation ended in quality flow",
                "exploring": "generation ended in exploration",
                "drifting":  "generation ended in drift",
                "flat":      "generation ended in stagnation",
                "none":      "no data",
            }.get(_qfst_fs, _qfst_fs)
            print(f"\n  Quadrant first/last occurrence and final state")
            print(f"  final_state:   [{_qfst_fs}]  — {_qfst_lbl}")
            print(f"  ideal:  first={_qfst_fi}  last={_qfst_li}  coverage_frac={_qfst_icf:.4f}")
            print(f"  drift:  first={_qfst_fd}  last={_qfst_ld}")
            print(f"  (coverage_frac=ideal_steps/(last_ideal−first_ideal+1); high=ideal densely filled its span)")
            continue

        if low.startswith("quadseparation"):
            _qsep_res = getattr(model, "_last_gen_result", None)
            if _qsep_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qsep_v   = _qsep_res.get("quad_separation_score", 0.0)
            _qsep_c3g = _qsep_res.get("quad_coh3_gap",         0.0)
            _qsep_vg  = _qsep_res.get("quad_vel_gap",          0.0)
            _qsep_cfg = _qsep_res.get("quad_conf_gap",         0.0)
            _qsep_vc  = _qsep_res.get("quad_vel_contrast",     0.0)
            _qsep_lbl = ("well-separated"   if _qsep_v >= 0.55
                          else "moderate"   if _qsep_v >= 0.30
                          else "overlapping")
            _qsep_bar = "█" * int(_qsep_v * 40)
            print(f"\n  Quadrant separation score: {_qsep_v:.4f}  [{_qsep_lbl}]")
            print(f"  {_qsep_bar}")
            print(f"  coh3_gap={_qsep_c3g:+.4f}  vel_gap={_qsep_vg:+.5f}  conf_gap={_qsep_cfg:+.4f}  vel_contrast={_qsep_vc:.4f}")
            print(f"  (high=ideal and drifting quadrants are clearly distinct on all 3 axes=good)")
            continue

        if low.startswith("quadranges"):
            _qrg_res = getattr(model, "_last_gen_result", None)
            if _qrg_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qrg_icr = _qrg_res.get("quad_ideal_conf_range",     0.0)
            _qrg_dcr = _qrg_res.get("quad_drifting_conf_range",  0.0)
            _qrg_ivr = _qrg_res.get("quad_ideal_vel_range",      0.0)
            _qrg_dvr = _qrg_res.get("quad_drifting_vel_range",   0.0)
            _qrg_icv = _qrg_res.get("quad_ideal_conf_cv",        0.0)
            _qrg_ivs = _qrg_res.get("quad_ideal_vel_std",        0.0)
            print(f"\n  Per-quadrant confidence and velocity ranges (max−min)")
            print(f"  ideal_conf_range:    {_qrg_icr:.4f}  {'█'*int(_qrg_icr*40)}")
            print(f"  drifting_conf_range: {_qrg_dcr:.4f}  {'█'*int(_qrg_dcr*40)}")
            print(f"  ideal_vel_range:     {_qrg_ivr:.5f}  {'█'*int(_qrg_ivr*1000)}")
            print(f"  drifting_vel_range:  {_qrg_dvr:.5f}  {'█'*int(_qrg_dvr*1000)}")
            print(f"  (low ideal ranges=consistent quality flow; ideal_conf_cv={_qrg_icv:.4f}  ideal_vel_σ={_qrg_ivs:.5f})")
            continue

        if low.startswith("quadvelstds"):
            _qvs_res = getattr(model, "_last_gen_result", None)
            if _qvs_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qvs_is  = _qvs_res.get("quad_ideal_vel_std",     0.0)
            _qvs_ds  = _qvs_res.get("quad_drifting_vel_std",  0.0)
            _qvs_vc  = _qvs_res.get("quad_vel_contrast",      0.0)
            _qvs_vg  = _qvs_res.get("quad_vel_gap",           0.0)
            _qvs_im  = _qvs_res.get("quad_ideal_vel_mean",    0.0)
            _qvs_dm  = _qvs_res.get("quad_drifting_vel_mean", 0.0)
            _qvs_vc_lbl = ("high contrast"  if _qvs_vc >= 0.25
                            else "moderate" if _qvs_vc >= 0.10
                            else "low separation")
            print(f"\n  Velocity std devs  (vel_contrast={_qvs_vc:+.4f}  [{_qvs_vc_lbl}])")
            print(f"  ideal_vel:     mean={_qvs_im:.5f}  σ={_qvs_is:.5f}  cv={_qvs_is/max(_qvs_im,1e-9):.4f}")
            print(f"  drifting_vel:  mean={_qvs_dm:.5f}  σ={_qvs_ds:.5f}  cv={_qvs_ds/max(_qvs_dm,1e-9):.4f}")
            print(f"  vel_gap: {_qvs_vg:+.5f}  (drift−ideal; large=well-separated)")
            continue

        if low.startswith("quadconfstds"):
            _qcs_res = getattr(model, "_last_gen_result", None)
            if _qcs_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcs_is  = _qcs_res.get("quad_ideal_conf_std",     0.0)
            _qcs_ds  = _qcs_res.get("quad_drifting_conf_std",  0.0)
            _qcs_ct  = _qcs_res.get("quad_conf_contrast",      0.0)
            _qcs_im  = _qcs_res.get("quad_ideal_conf_mean",    0.0)
            _qcs_dm  = _qcs_res.get("quad_drifting_conf_mean", 0.0)
            _qcs_g   = _qcs_res.get("quad_conf_gap",           0.0)
            _qcs_ct_lbl = ("ideal>>flat"  if _qcs_ct >= 0.10
                            else "similar"  if abs(_qcs_ct) < 0.04
                            else "flat>>ideal")
            print(f"\n  Confidence std devs  (conf_contrast={_qcs_ct:+.4f}  [{_qcs_ct_lbl}])")
            print(f"  ideal_conf:    mean={_qcs_im:.4f}  σ={_qcs_is:.4f}  cv={_qcs_is/max(_qcs_im,1e-9):.4f}")
            print(f"  drifting_conf: mean={_qcs_dm:.4f}  σ={_qcs_ds:.4f}  cv={_qcs_ds/max(_qcs_dm,1e-9):.4f}")
            print(f"  conf_gap (ideal−drift): {_qcs_g:+.4f}")
            continue

        if low.startswith("quadcoh3stds"):
            _qc3s_res = getattr(model, "_last_gen_result", None)
            if _qc3s_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qc3s_i  = _qc3s_res.get("quad_ideal_coh3_std",     0.0)
            _qc3s_d  = _qc3s_res.get("quad_drifting_coh3_std",  0.0)
            _qc3s_cv_i = _qc3s_res.get("quad_ideal_coh3_cv",    0.0)
            _qc3s_cv_d = _qc3s_res.get("quad_drifting_coh3_cv", 0.0)
            _qc3s_ct = _qc3s_res.get("quad_coh3_contrast",      0.0)
            _qc3s_i_lbl = ("stable"    if _qc3s_cv_i <= 0.05
                            else "moderate" if _qc3s_cv_i <= 0.12
                            else "variable")
            _qc3s_d_lbl = ("erratic"   if _qc3s_cv_d >= 0.15
                            else "moderate" if _qc3s_cv_d >= 0.07
                            else "uniform")
            print(f"\n  Coh3 standard deviations  (contrast={_qc3s_ct:+.4f})")
            print(f"  ideal_coh3_std:    {_qc3s_i:.4f}  (cv={_qc3s_cv_i:.4f})  [{_qc3s_i_lbl}]")
            print(f"  drifting_coh3_std: {_qc3s_d:.4f}  (cv={_qc3s_cv_d:.4f})  [{_qc3s_d_lbl}]")
            print(f"  contrast: {_qc3s_ct:+.4f}  (ideal−flat)/(ideal+flat)  [+1=ideal>>flat; −1=flat>>ideal]")
            continue

        if low.startswith("quadcoh3means"):
            _qc3m_res = getattr(model, "_last_gen_result", None)
            if _qc3m_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qc3m_i  = _qc3m_res.get("quad_ideal_coh3_mean",     0.0)
            _qc3m_e  = _qc3m_res.get("quad_exploring_coh3_mean", 0.0)
            _qc3m_d  = _qc3m_res.get("quad_drifting_coh3_mean",  0.0)
            _qc3m_f  = _qc3m_res.get("quad_flat_coh3_mean",      0.0)
            _qc3m_g  = _qc3m_res.get("quad_coh3_gap",            0.0)
            _qc3m_g_lbl = ("ideal>>drift"   if _qc3m_g >= 0.10
                            else "similar"  if abs(_qc3m_g) < 0.05
                            else "drift>>ideal" if _qc3m_g < -0.05
                            else "moderate")
            _qc3m_scale = max(_qc3m_i, _qc3m_e, _qc3m_d, _qc3m_f, 0.01)
            def _qc3m_bar(v): return "█" * int(v / _qc3m_scale * 30)
            print(f"\n  Per-quadrant mean coh3  (coh3_gap={_qc3m_g:+.4f}  [{_qc3m_g_lbl}])")
            print(f"  ideal:     {_qc3m_i:.4f}  {_qc3m_bar(_qc3m_i)}")
            print(f"  exploring: {_qc3m_e:.4f}  {_qc3m_bar(_qc3m_e)}")
            print(f"  drifting:  {_qc3m_d:.4f}  {_qc3m_bar(_qc3m_d)}")
            print(f"  flat:      {_qc3m_f:.4f}  {_qc3m_bar(_qc3m_f)}")
            print(f"  (gap=ideal−drift; large=quadrants are coh3-distinct=good separation)")
            continue

        if low.startswith("quadrunlens"):
            _qrl_res = getattr(model, "_last_gen_result", None)
            if _qrl_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qrl_i  = _qrl_res.get("quad_ideal_run_mean_len",     0.0)
            _qrl_e  = _qrl_res.get("quad_exploring_run_mean_len", 0.0)
            _qrl_d  = _qrl_res.get("quad_drifting_run_mean_len",  0.0)
            _qrl_f  = _qrl_res.get("quad_flat_run_mean_len",      0.0)
            _qrl_r  = _qrl_res.get("quad_run_len_ratio",          0.0)
            _qrl_irc = _qrl_res.get("quad_ideal_run_count",       0)
            _qrl_drc = _qrl_res.get("quad_drifting_run_count",    0)
            _qrl_r_lbl = ("ideal runs longer"  if _qrl_r >= 1.5
                           else "similar"      if _qrl_r >= 0.7
                           else "drift runs longer")
            _qrl_scale = max(_qrl_i, _qrl_e, _qrl_d, _qrl_f, 0.1)
            def _qrl_bar(v): return "█" * int(v / _qrl_scale * 30)
            print(f"\n  Per-quadrant mean run length  (ratio={_qrl_r:.2f}  [{_qrl_r_lbl}])")
            print(f"  ideal:     {_qrl_i:.2f} steps  {_qrl_bar(_qrl_i)}  (runs={_qrl_irc})")
            print(f"  exploring: {_qrl_e:.2f} steps  {_qrl_bar(_qrl_e)}")
            print(f"  drifting:  {_qrl_d:.2f} steps  {_qrl_bar(_qrl_d)}  (runs={_qrl_drc})")
            print(f"  flat:      {_qrl_f:.2f} steps  {_qrl_bar(_qrl_f)}")
            print(f"  (ratio=ideal_len/drift_len; >1 ideal runs are longer than drift runs=good)")
            continue

        if low.startswith("quadvelmeans"):
            _qvm_res = getattr(model, "_last_gen_result", None)
            if _qvm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qvm_i  = _qvm_res.get("quad_ideal_vel_mean",     0.0)
            _qvm_e  = _qvm_res.get("quad_exploring_vel_mean", 0.0)
            _qvm_d  = _qvm_res.get("quad_drifting_vel_mean",  0.0)
            _qvm_f  = _qvm_res.get("quad_flat_vel_mean",      0.0)
            _qvm_g  = _qvm_res.get("quad_vel_gap",            0.0)
            _qvm_g_lbl = ("clear separation"  if _qvm_g >= 0.10
                           else "moderate"    if _qvm_g >= 0.04
                           else "overlapping")
            _qvm_scale = max(_qvm_i, _qvm_e, _qvm_d, _qvm_f, 0.01)
            def _qvm_bar(v): return "█" * int(v / _qvm_scale * 30)
            print(f"\n  Per-quadrant mean velocity  (vel_gap={_qvm_g:+.4f}  [{_qvm_g_lbl}])")
            print(f"  ideal:     {_qvm_i:.5f}  {_qvm_bar(_qvm_i)}")
            print(f"  exploring: {_qvm_e:.5f}  {_qvm_bar(_qvm_e)}")
            print(f"  drifting:  {_qvm_d:.5f}  {_qvm_bar(_qvm_d)}")
            print(f"  flat:      {_qvm_f:.5f}  {_qvm_bar(_qvm_f)}")
            print(f"  (vel_gap=drift−ideal; large=quadrants are velocity-distinct=good)")
            continue

        if low.startswith("quadconfmeans"):
            _qcm_res = getattr(model, "_last_gen_result", None)
            if _qcm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcm_i  = _qcm_res.get("quad_ideal_conf_mean",     0.0)
            _qcm_e  = _qcm_res.get("quad_exploring_conf_mean", 0.0)
            _qcm_d  = _qcm_res.get("quad_drifting_conf_mean",  0.0)
            _qcm_f  = _qcm_res.get("quad_flat_conf_mean",      0.0)
            _qcm_g  = _qcm_res.get("quad_conf_gap",            0.0)
            _qcm_g_lbl = ("ideal>>drift"  if _qcm_g >= 0.10
                           else "similar" if abs(_qcm_g) < 0.05
                           else "drift>>ideal")
            print(f"\n  Per-quadrant mean confidence")
            print(f"  ideal:     {_qcm_i:.4f}  {'█'*int(_qcm_i*40)}")
            print(f"  exploring: {_qcm_e:.4f}  {'█'*int(_qcm_e*40)}")
            print(f"  drifting:  {_qcm_d:.4f}  {'█'*int(_qcm_d*40)}")
            print(f"  flat:      {_qcm_f:.4f}  {'█'*int(_qcm_f*40)}")
            print(f"  conf_gap (ideal−drift): {_qcm_g:+.4f}  [{_qcm_g_lbl}]")
            continue

        if low.startswith("quadoverallefficiency"):
            _qoe_res = getattr(model, "_last_gen_result", None)
            if _qoe_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qoe_v   = _qoe_res.get("quad_overall_efficiency",    0.0)
            _qoe_hs  = _qoe_res.get("quad_health_score",          0.0)
            _qoe_hv  = _qoe_res.get("quad_hi_vel_frac",           0.0)
            _qoe_re  = _qoe_res.get("quad_recovery_efficiency",   0.0)
            _qoe_lbl = ("excellent"  if _qoe_v >= 0.40
                         else "good" if _qoe_v >= 0.20
                         else "fair" if _qoe_v >= 0.08
                         else "poor")
            _qoe_bar = "█" * int(_qoe_v * 40)
            print(f"\n  Overall efficiency: {_qoe_v:.4f}  [{_qoe_lbl}]")
            print(f"  {_qoe_bar[:40]}")
            print(f"  (health={_qoe_hs:.4f}  hi_vel_frac={_qoe_hv:.4f}  recovery_eff={_qoe_re:.4f})")
            print(f"  (quality per unit of fast motion: high=lots of ideal quality at low speed cost)")
            continue

        if low.startswith("quadconfcv"):
            _qcc_res = getattr(model, "_last_gen_result", None)
            if _qcc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcc_icv = _qcc_res.get("quad_ideal_conf_cv",    0.0)
            _qcc_ics = _qcc_res.get("quad_ideal_conf_stability", 0.0)
            _qcc_ic3 = _qcc_res.get("quad_ideal_coh3_cv",    0.0)
            _qcc_dc3 = _qcc_res.get("quad_drifting_coh3_cv", 0.0)
            _qcc_ics_lbl = ("very stable"  if _qcc_ics >= 0.85
                             else "stable" if _qcc_ics >= 0.70
                             else "moderate" if _qcc_ics >= 0.50
                             else "unstable")
            print(f"\n  Quadrant coefficient of variation (σ/μ)")
            print(f"  ideal_conf_cv:     {_qcc_icv:.4f}  (low=consistent ideal confidence)")
            print(f"  ideal_conf_stab:   {_qcc_ics:.4f}  [{_qcc_ics_lbl}]  {'█'*int(_qcc_ics*40)}")
            print(f"  ideal_coh3_cv:     {_qcc_ic3:.4f}  (low=coherence stable in ideal steps)")
            print(f"  drifting_coh3_cv:  {_qcc_dc3:.4f}  (high=drift is erratic)")
            continue

        if low.startswith("quadflowefficiency"):
            _qfe2_res = getattr(model, "_last_gen_result", None)
            if _qfe2_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qfe2_v  = _qfe2_res.get("quad_flow_efficiency",    0.0)
            _qfe2_pf = _qfe2_res.get("quad_positive_frac",      0.0)
            _qfe2_if = _qfe2_res.get("ideal_frac",              0.0)
            _qfe2_ef = _qfe2_res.get("exploring_frac",          0.0)
            _qfe2_lbl = ("pure flow"     if _qfe2_v >= 0.85
                          else "mostly flow" if _qfe2_v >= 0.65
                          else "mixed"       if _qfe2_v >= 0.40
                          else "exploration heavy")
            _qfe2_bar = "█" * int(_qfe2_v * 40)
            print(f"\n  Flow efficiency (ideal / hi-coh3): {_qfe2_v:.4f}  [{_qfe2_lbl}]")
            print(f"  {_qfe2_bar}")
            print(f"  (ideal_frac={_qfe2_if:.4f}  exploring_frac={_qfe2_ef:.4f}  hi_coh3={_qfe2_pf:.4f})")
            print(f"  (1=all hi-coh3 steps are controlled ideal; 0=all hi-coh3 are fast-exploring)")
            continue

        if low.startswith("quaddriftseverity"):
            _qds2_res = getattr(model, "_last_gen_result", None)
            if _qds2_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qds2_v  = _qds2_res.get("quad_drift_severity",  0.0)
            _qds2_hv = _qds2_res.get("quad_hi_vel_frac",     0.0)
            _qds2_df = _qds2_res.get("drifting_frac",        0.0)
            _qds2_lbl = ("severe drift"     if _qds2_v >= 0.75
                          else "notable"    if _qds2_v >= 0.50
                          else "moderate"   if _qds2_v >= 0.25
                          else "low drift severity")
            _qds2_bar = "█" * int(_qds2_v * 40)
            print(f"\n  Drift severity (drift / hi-vel): {_qds2_v:.4f}  [{_qds2_lbl}]")
            print(f"  {_qds2_bar}")
            print(f"  (hi_vel_frac={_qds2_hv:.4f}  drifting_frac={_qds2_df:.4f})")
            print(f"  (1=all fast steps are low-coh3 drift; 0=fast steps are creative exploring)")
            continue

        if low.startswith("quadrecoveryefficiency"):
            _qre2_res = getattr(model, "_last_gen_result", None)
            if _qre2_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qre2_v  = _qre2_res.get("quad_recovery_efficiency", 0.0)
            _qre2_dti = _qre2_res.get("quad_drift_to_ideal_rate", 0.0)
            _qre2_ip  = _qre2_res.get("quad_ideal_persistence",   0.0)
            _qre2_lbl = ("excellent"  if _qre2_v >= 0.20
                          else "good" if _qre2_v >= 0.10
                          else "fair" if _qre2_v >= 0.04
                          else "poor")
            _qre2_bar = "█" * int(_qre2_v * 100)
            print(f"\n  Recovery efficiency: {_qre2_v:.4f}  [{_qre2_lbl}]")
            print(f"  {_qre2_bar[:40]}")
            print(f"  (drift→ideal={_qre2_dti:.4f}  ideal_persistence={_qre2_ip:.4f})")
            print(f"  (product: recovers from drift AND sustains ideal flow)")
            continue

        if low.startswith("quadvelocityfracs"):
            _qvf_res = getattr(model, "_last_gen_result", None)
            if _qvf_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qvf_hi = _qvf_res.get("quad_hi_vel_frac", 0.0)
            _qvf_lo = _qvf_res.get("quad_lo_vel_frac", 0.0)
            _qvf_df = _qvf_res.get("drifting_frac",    0.0)
            _qvf_if = _qvf_res.get("ideal_frac",       0.0)
            print(f"\n  Velocity split fracs  (above/below median velocity)")
            print(f"  hi-vel (fast): {_qvf_hi:.4f}  {'█'*int(_qvf_hi*40)}")
            print(f"  lo-vel (slow): {_qvf_lo:.4f}  {'█'*int(_qvf_lo*40)}")
            print(f"  (hi-vel includes drifting+exploring; lo-vel includes ideal+flat)")
            print(f"  (drifting_frac={_qvf_df:.4f}  ideal_frac={_qvf_if:.4f})")
            continue

        if low.startswith("quadqualityarc"):
            _qqa2_res = getattr(model, "_last_gen_result", None)
            if _qqa2_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qqa2_arc = _qqa2_res.get("quad_quality_arc",           "none")
            _qqa2_ei  = _qqa2_res.get("quad_early_ideal_frac",      0.0)
            _qqa2_li  = _qqa2_res.get("quad_late_ideal_frac",       0.0)
            _qqa2_ift = _qqa2_res.get("quad_ideal_frac_trend",      0.0)
            _qqa2_hs  = _qqa2_res.get("quad_health_score",          0.0)
            _qqa2_fp  = _qqa2_res.get("quad_fingerprint",           "none")
            _qqa2_hs_lbl = ("excellent" if _qqa2_hs >= 0.25 else "good" if _qqa2_hs >= 0.12
                             else "fair" if _qqa2_hs >= 0.05 else "poor")
            print(f"\n  Quality arc: {_qqa2_arc}  fingerprint={_qqa2_fp}")
            print(f"  early_ideal={_qqa2_ei:.3f}  late_ideal={_qqa2_li:.3f}  trend={_qqa2_ift:+.3f}")
            print(f"  health={_qqa2_hs:.4f} [{_qqa2_hs_lbl}]")
            print(f"  arc labels: cold_start sustained warm_start collapse recovery oscillating improving degrading")
            continue

        if low.startswith("quaddeltas"):
            _qd_res = getattr(model, "_last_gen_result", None)
            if _qd_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qd_imd  = _qd_res.get("quad_ideal_minus_drifting_frac",  0.0)
            _qd_emf  = _qd_res.get("quad_exploring_minus_flat_frac",  0.0)
            _qd_pf   = _qd_res.get("quad_positive_frac",              0.0)
            _qd_nf   = _qd_res.get("quad_negative_frac",              0.0)
            _qd_imd_lbl = ("ideal dominates" if _qd_imd >= 0.15
                            else "balanced"  if _qd_imd >= -0.15
                            else "drift dominates")
            _qd_emf_lbl = ("creative>stagnant" if _qd_emf >= 0.10
                            else "balanced"     if _qd_emf >= -0.10
                            else "stagnant>creative")
            _qd_imd_bar = ("█" if _qd_imd >= 0 else "░") * min(int(abs(_qd_imd)*60), 40)
            _qd_emf_bar = ("█" if _qd_emf >= 0 else "░") * min(int(abs(_qd_emf)*60), 40)
            print(f"\n  Quadrant delta metrics")
            print(f"  ideal−drift: {_qd_imd:+.4f}  [{_qd_imd_lbl}]  {_qd_imd_bar}")
            print(f"  expl−flat:   {_qd_emf:+.4f}  [{_qd_emf_lbl}]  {_qd_emf_bar}")
            print(f"  hi-coh3(I+E)={_qd_pf:.4f}  lo-coh3(D+F)={_qd_nf:.4f}")
            continue

        if low.startswith("quadpersistence"):
            # quadpersistence — all 4 quadrant self-loop fractions
            _qpers_res = getattr(model, "_last_gen_result", None)
            if _qpers_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qpers_ip = _qpers_res.get("quad_ideal_persistence",     0.0)
            _qpers_ep = _qpers_res.get("quad_exploring_persistence", 0.0)
            _qpers_dp = _qpers_res.get("quad_drifting_persistence",  0.0)
            _qpers_fp = _qpers_res.get("quad_flat_persistence",      0.0)
            _qpers_sym = _qpers_res.get("quad_symmetry_score",       0.0)
            _qpers_fp2 = _qpers_res.get("quad_fingerprint",          "none")
            def _qp_bar(v): return "█" * int(v * 40)
            print(f"\n  Quadrant persistence (self-loop fracs)  fingerprint={_qpers_fp2}  sym={_qpers_sym:.3f}")
            print(f"  ideal     {_qpers_ip:.4f}  {_qp_bar(_qpers_ip)}")
            print(f"  exploring {_qpers_ep:.4f}  {_qp_bar(_qpers_ep)}")
            print(f"  drifting  {_qpers_dp:.4f}  {_qp_bar(_qpers_dp)}")
            print(f"  flat      {_qpers_fp:.4f}  {_qp_bar(_qpers_fp)}")
            print(f"  (sym=1 balanced; sym=0 one quadrant dominates)")
            continue

        if low.startswith("quadsymmetry"):
            _qsym_res = getattr(model, "_last_gen_result", None)
            if _qsym_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qsym_v  = _qsym_res.get("quad_symmetry_score", 0.0)
            _qsym_fp = _qsym_res.get("quad_fingerprint",    "none")
            _qsym_lbl = ("balanced"        if _qsym_v >= 0.85
                          else "moderate"  if _qsym_v >= 0.60
                          else "skewed"    if _qsym_v >= 0.35
                          else "dominated")
            _qsym_bar = "█" * int(_qsym_v * 40)
            print(f"\n  Quadrant symmetry: {_qsym_v:.4f}  [{_qsym_lbl}]  fingerprint={_qsym_fp}")
            print(f"  {_qsym_bar}")
            print(f"  (0=one quadrant dominates; 1=all four perfectly equal)")
            continue

        if low.startswith("quadfingerprint"):
            _qfp3_res = getattr(model, "_last_gen_result", None)
            if _qfp3_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qfp3_v  = _qfp3_res.get("quad_fingerprint",    "none")
            _qfp3_if = _qfp3_res.get("ideal_frac",          0.0)
            _qfp3_ef = _qfp3_res.get("exploring_frac",      0.0)
            _qfp3_df = _qfp3_res.get("drifting_frac",       0.0)
            _qfp3_ff = _qfp3_res.get("flat_frac",           0.0)
            _qfp3_hs = _qfp3_res.get("quad_health_score",   0.0)
            _qfp3_sym = _qfp3_res.get("quad_symmetry_score", 0.0)
            print(f"\n  Quadrant fingerprint: {_qfp3_v}")
            print(f"  I={_qfp3_if:.3f}  E={_qfp3_ef:.3f}  D={_qfp3_df:.3f}  F={_qfp3_ff:.3f}")
            print(f"  health={_qfp3_hs:.4f}  symmetry={_qfp3_sym:.4f}")
            print(f"  (I=ideal E=exploring D=drifting F=flat; sorted descending by fraction)")
            continue

        if low.startswith("quadidealpersistence"):
            _qip_res = getattr(model, "_last_gen_result", None)
            if _qip_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qip_v   = _qip_res.get("quad_ideal_persistence",    0.0)
            _qip_irc = _qip_res.get("quad_ideal_run_count",      0)
            _qip_ims = _qip_res.get("quad_ideal_max_streak",     0)
            _qip_lbl = ("very sticky"  if _qip_v >= 0.70
                         else "sticky" if _qip_v >= 0.50
                         else "moderate" if _qip_v >= 0.30
                         else "fragile")
            _qip_bar = "█" * int(_qip_v * 40)
            print(f"\n  Ideal persistence (ideal→ideal): {_qip_v:.4f}  [{_qip_lbl}]")
            print(f"  {_qip_bar}")
            print(f"  (ideal_runs={_qip_irc}  max_streak={_qip_ims})")
            print(f"  (high=quality flow is self-sustaining; low=ideal steps rarely chain)")
            continue

        if low.startswith("quadnetrecovery"):
            _qnrr_res = getattr(model, "_last_gen_result", None)
            if _qnrr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qnrr_v   = _qnrr_res.get("quad_net_recovery_rate",  0.0)
            _qnrr_dti = _qnrr_res.get("quad_drift_to_ideal_rate", 0.0)
            _qnrr_itd = _qnrr_res.get("quad_ideal_to_drift_rate", 0.0)
            _qnrr_lbl = ("net recovery"    if _qnrr_v >= 0.10
                          else "balanced"  if _qnrr_v >= -0.10
                          else "net decay")
            _qnrr_filled = int(abs(_qnrr_v) * 100)
            _qnrr_bar = ("█" if _qnrr_v >= 0 else "░") * min(_qnrr_filled, 40)
            print(f"\n  Net recovery rate: {_qnrr_v:+.4f}  [{_qnrr_lbl}]")
            print(f"  {_qnrr_bar}")
            print(f"  (drift→ideal={_qnrr_dti:.4f}  ideal→drift={_qnrr_itd:.4f})")
            print(f"  (█=net recovery  ░=net decay)")
            continue

        if low.startswith("quadescaperates"):
            _qer_res = getattr(model, "_last_gen_result", None)
            if _qer_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qer_fti = _qer_res.get("quad_flat_to_ideal_rate",       0.0)
            _qer_ite = _qer_res.get("quad_ideal_to_exploring_rate",  0.0)
            _qer_eti = _qer_res.get("quad_exploring_to_ideal_rate",  0.0)
            _qer_dti = _qer_res.get("quad_drift_to_ideal_rate",      0.0)
            print(f"\n  Escape / conversion rates")
            print(f"  flat→ideal:       {_qer_fti:.4f}  {'█'*int(_qer_fti*40)}")
            print(f"  exploring→ideal:  {_qer_eti:.4f}  {'█'*int(_qer_eti*40)}")
            print(f"  drift→ideal:      {_qer_dti:.4f}  {'█'*int(_qer_dti*40)}")
            print(f"  ideal→exploring:  {_qer_ite:.4f}  {'█'*int(_qer_ite*40)}")
            print(f"  (higher=more paths recover to ideal flow)")
            continue

        if low.startswith("quadtransitionrates"):
            _qtr_res = getattr(model, "_last_gen_result", None)
            if _qtr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qtr_itd = _qtr_res.get("quad_ideal_to_drift_rate",      0.0)
            _qtr_dti = _qtr_res.get("quad_drift_to_ideal_rate",      0.0)
            _qtr_eti = _qtr_res.get("quad_exploring_to_ideal_rate",  0.0)
            _qtr_str = _qtr_res.get("quad_self_transition_rate",     0.0)
            _qtr_itd_lbl = ("rare"     if _qtr_itd <= 0.10 else
                             "moderate" if _qtr_itd <= 0.30 else "frequent decay")
            _qtr_dti_lbl = ("strong recovery"  if _qtr_dti >= 0.30 else
                             "some recovery"    if _qtr_dti >= 0.10 else "rare recovery")
            _qtr_eti_lbl = ("exploration→ideal"  if _qtr_eti >= 0.30 else
                             "partial"           if _qtr_eti >= 0.10 else "exploration rarely converts")
            print(f"\n  Targeted transition rates  (self_loop={_qtr_str:.3f})")
            print(f"  ideal→drift:     {_qtr_itd:.4f}  [{_qtr_itd_lbl}]  {'█'*int(_qtr_itd*40)}")
            print(f"  drift→ideal:     {_qtr_dti:.4f}  [{_qtr_dti_lbl}]  {'█'*int(_qtr_dti*40)}")
            print(f"  exploring→ideal: {_qtr_eti:.4f}  [{_qtr_eti_lbl}]  {'█'*int(_qtr_eti*40)}")
            continue

        if low.startswith("quadflatmomentum"):
            _qfm_res = getattr(model, "_last_gen_result", None)
            if _qfm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qfm_cv = _qfm_res.get("quad_flat_coh3_momentum",     0.0)
            _qfm_vv = _qfm_res.get("quad_flat_velocity_momentum", 0.0)
            _qfm_c_lbl = ("recovering"  if _qfm_cv >= 0.01
                           else "stable"  if _qfm_cv >= -0.01
                           else "deepening stagnation")
            _qfm_v_lbl = ("velocity rising=pre-escape"  if _qfm_vv >= 0.01
                           else "stable"                 if _qfm_vv >= -0.01
                           else "velocity falling=stuck")
            print(f"\n  Flat quadrant momentum")
            print(f"  coh3_mom: {_qfm_cv:+.4f}  [{_qfm_c_lbl}]")
            print(f"  {'█' * min(int(abs(_qfm_cv)*200), 40)}")
            print(f"  vel_mom:  {_qfm_vv:+.4f}  [{_qfm_v_lbl}]")
            print(f"  {'█' * min(int(abs(_qfm_vv)*200), 40)}")
            continue

        if low.startswith("quadinterrungaps"):
            _qirg_res = getattr(model, "_last_gen_result", None)
            if _qirg_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qirg_i  = _qirg_res.get("quad_inter_ideal_gap",    0.0)
            _qirg_d  = _qirg_res.get("quad_inter_drifting_gap", 0.0)
            _qirg_irc = _qirg_res.get("quad_ideal_run_count",   0)
            _qirg_drc = _qirg_res.get("quad_drifting_run_count", 0)
            _qirg_i_lbl = ("frequent re-entry"  if _qirg_i <= 3
                            else "moderate gap"  if _qirg_i <= 8
                            else "rare re-entry" if _qirg_i > 8 else "n/a")
            _qirg_d_lbl = ("dense drift"  if _qirg_d <= 3
                            else "spaced"  if _qirg_d <= 8
                            else "rare drift" if _qirg_d > 8 else "n/a")
            print(f"\n  Inter-run gaps  (ideal_runs={_qirg_irc}  drifting_runs={_qirg_drc})")
            print(f"  ideal gap:   {_qirg_i:.2f} steps  [{_qirg_i_lbl}]")
            print(f"  {'█' * min(int(_qirg_i), 40)}")
            print(f"  drifting gap: {_qirg_d:.2f} steps  [{_qirg_d_lbl}]")
            print(f"  {'█' * min(int(_qirg_d), 40)}")
            print(f"  (small ideal gap = quality flow returns quickly; large drift gap = rare relapse)")
            continue

        if low.startswith("quaddriftingmomentum"):
            _qdm_res = getattr(model, "_last_gen_result", None)
            if _qdm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qdm_cv = _qdm_res.get("quad_drifting_coh3_momentum",     0.0)
            _qdm_vv = _qdm_res.get("quad_drifting_velocity_momentum", 0.0)
            _qdm_c_lbl = ("coh3 rising in drift"    if _qdm_cv >= 0.02
                           else "coh3 stable"        if _qdm_cv >= -0.02
                           else "coh3 falling=good")
            _qdm_v_lbl = ("accelerating=bad"         if _qdm_vv >= 0.02
                           else "stable"              if _qdm_vv >= -0.02
                           else "decelerating=good")
            print(f"\n  Drifting momentum")
            print(f"  coh3_mom: {_qdm_cv:+.4f}  [{_qdm_c_lbl}]")
            print(f"  {'█' * min(int(abs(_qdm_cv)*200), 40)}")
            print(f"  vel_mom:  {_qdm_vv:+.4f}  [{_qdm_v_lbl}]")
            print(f"  {'█' * min(int(abs(_qdm_vv)*200), 40)}")
            continue

        if low.startswith("quadexploringcoh3momentum"):
            _qecm2_res = getattr(model, "_last_gen_result", None)
            if _qecm2_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qecm2_v  = _qecm2_res.get("quad_exploring_coh3_momentum", 0.0)
            _qecm2_lbl = ("coh3 rising in exploration"  if _qecm2_v >= 0.02
                           else "coh3 stable"            if _qecm2_v >= -0.02
                           else "coh3 falling")
            _qecm2_bar = ("█" if _qecm2_v >= 0 else "░") * min(int(abs(_qecm2_v)*200), 40)
            print(f"\n  Exploring coh3 momentum: {_qecm2_v:+.4f}  [{_qecm2_lbl}]")
            print(f"  {_qecm2_bar}")
            print(f"  (positive=coherence improving during exploration; negative=declining)")
            continue

        if low.startswith("quadtransitionmap"):
            # quadtransitionmap — ASCII 4×4 transition matrix with counts and row-probs
            _qtmap_res = getattr(model, "_last_gen_result", None)
            if _qtmap_res is None:
                print("  No generation yet. Run a prompt first."); continue
            import itertools as _qtmap_it
            _qtmap_c3 = _qtmap_res.get("coh3_steps", [])
            _qtmap_vs = _qtmap_res.get("velocity_steps", [])
            if len(_qtmap_c3) < 4 or len(_qtmap_vs) < 4:
                print("  Not enough steps for transition map."); continue
            _qtmap_n  = min(len(_qtmap_c3), len(_qtmap_vs))
            _qtmap_c3m = sum(_qtmap_c3) / _qtmap_n
            _qtmap_vm  = sum(_qtmap_vs) / _qtmap_n
            def _qtmap_lab(c, v):
                if c > _qtmap_c3m and v < _qtmap_vm:  return "ideal"
                if c > _qtmap_c3m:                     return "expl"
                if v >= _qtmap_vm:                     return "drift"
                return "flat"
            _qtmap_labs = [_qtmap_lab(_qtmap_c3[i], _qtmap_vs[i]) for i in range(_qtmap_n)]
            _qtmap_qs   = ["ideal", "expl", "drift", "flat"]
            _qtmap_mat  = {
                (src, dst): sum(1 for i in range(1, _qtmap_n)
                                if _qtmap_labs[i-1] == src and _qtmap_labs[i] == dst)
                for src in _qtmap_qs for dst in _qtmap_qs
            }
            _qtmap_te = _qtmap_res.get("quad_transition_entropy", 0.0)
            print(f"\n  Transition matrix  (entropy={_qtmap_te:.3f} bits)")
            print(f"  {'':6s}  {'ideal':>5}  {'expl':>5}  {'drift':>5}  {'flat':>5}   total")
            for _src in _qtmap_qs:
                _row = [_qtmap_mat[(_src, _dst)] for _dst in _qtmap_qs]
                _tot = sum(_row)
                _probs = [f"{r/_tot:.2f}" if _tot > 0 else " .-- " for r in _row]
                print(f"  {_src:<6}  " + "  ".join(f"{p:>5}" for p in _probs) + f"  {_tot:>5}")
            continue

        if low.startswith("quadvelocitymassratio"):
            _qvmr_res = getattr(model, "_last_gen_result", None)
            if _qvmr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qvmr_v  = _qvmr_res.get("quad_velocity_mass_ratio", 0.0)
            _qvmr_i  = _qvmr_res.get("quad_weighted_ideal_velocity",    0.0)
            _qvmr_d  = _qvmr_res.get("quad_weighted_drifting_velocity", 0.0)
            _qvmr_mom = _qvmr_res.get("quad_ideal_velocity_momentum",   0.0)
            _qvmr_label = ("very controlled" if _qvmr_v <= 0.30
                            else "controlled"  if _qvmr_v <= 0.55
                            else "near-equal"  if _qvmr_v <= 0.80
                            else "ideal fast"  if _qvmr_v <= 1.20
                            else "drift slow")
            _qvmr_bar = "█" * min(int(_qvmr_v * 15), 40)
            print(f"\n  Velocity mass ratio (ideal/drift): {_qvmr_v:.4f}  [{_qvmr_label}]")
            print(f"  {_qvmr_bar}")
            print(f"  (ideal_vel_mass={_qvmr_i:.4f}  drifting_vel_mass={_qvmr_d:.4f})")
            print(f"  (ideal_vel_momentum={_qvmr_mom:+.4f}  negative=slowing down in ideal=good)")
            continue

        if low.startswith("quadidealvelocitymomentum"):
            _qivm_res = getattr(model, "_last_gen_result", None)
            if _qivm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qivm_v  = _qivm_res.get("quad_ideal_velocity_momentum", 0.0)
            _qivm_r  = _qivm_res.get("quad_velocity_mass_ratio",     0.0)
            _qivm_label = ("slowing nicely"  if _qivm_v <= -0.01
                            else "stable"    if _qivm_v <= 0.01
                            else "speeding up")
            _qivm_filled = int(abs(_qivm_v) * 200)
            _qivm_bar = ("░" if _qivm_v <= 0 else "█") * min(_qivm_filled, 40)
            print(f"\n  Ideal velocity momentum: {_qivm_v:+.4f}  [{_qivm_label}]")
            print(f"  {_qivm_bar}")
            print(f"  (vel_mass_ratio={_qivm_r:.4f})")
            print(f"  (░=slowing=good  █=speeding up=concerning within ideal runs)")
            continue

        if low.startswith("quadreport"):
            # quadreport — comprehensive single-screen quadrant health report
            _qr_res = getattr(model, "_last_gen_result", None)
            if _qr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            def _qr_bar(v, scale=40): return "█" * int(float(v or 0) * scale)
            def _qr_sbar(v, scale=80):
                vf = float(v or 0)
                filled = int(abs(vf) * scale)
                return ("█" if vf >= 0 else "░") * min(filled, 40)
            _qr_n    = max(_qr_res.get("output_len", 1) or 1, 1)
            _qr_hs   = _qr_res.get("quad_health_score",         0.0)
            _qr_if   = _qr_res.get("ideal_frac",                0.0)
            _qr_df   = _qr_res.get("drifting_frac",             0.0)
            _qr_ef   = _qr_res.get("exploring_frac",            0.0)
            _qr_ff   = _qr_res.get("flat_frac",                 0.0)
            _qr_ift  = _qr_res.get("quad_ideal_frac_trend",     0.0)
            _qr_te   = _qr_res.get("quad_transition_entropy",   0.0)
            _qr_str  = _qr_res.get("quad_self_transition_rate", 0.0)
            _qr_ims  = _qr_res.get("quad_ideal_max_streak",     0)
            _qr_dms  = _qr_res.get("quad_drifting_max_streak",  0)
            _qr_irc  = _qr_res.get("quad_ideal_run_count",      0)
            _qr_ic   = _qr_res.get("quad_ideal_run_confidence_mean", 0.0)
            _qr_dc   = _qr_res.get("quad_drifting_run_confidence_mean", 0.0)
            _qr_cmr  = _qr_res.get("quad_confidence_mass_ratio",  0.0)
            _qr_ifs  = _qr_res.get("quad_ideal_first_step",     -1)
            _qr_dfs  = _qr_res.get("quad_drifting_first_step",  -1)
            _qr_dom  = _qr_res.get("quad_dominant_streak_label","none")
            _qr_hs_lbl = ("excellent" if _qr_hs >= 0.25 else "good" if _qr_hs >= 0.12
                           else "fair" if _qr_hs >= 0.05 else "poor")
            _qr_trend_lbl = ("↑improving" if _qr_ift >= 0.05 else "↓degrading"
                              if _qr_ift <= -0.05 else "→stable")
            print(f"\n{'─'*60}")
            print(f"  QUADRANT REPORT  steps={_qr_n}  health={_qr_hs:.4f} [{_qr_hs_lbl}]  {_qr_trend_lbl}")
            print(f"{'─'*60}")
            print(f"  Fracs  ideal={_qr_if:.3f}  expl={_qr_ef:.3f}  drift={_qr_df:.3f}  flat={_qr_ff:.3f}")
            print(f"  ideal  {_qr_bar(_qr_if)}")
            print(f"  drift  {_qr_bar(_qr_df)}")
            print(f"  trend  {_qr_sbar(_qr_ift)} {_qr_ift:+.3f}")
            print(f"  Streaks  ideal_max={_qr_ims}  drift_max={_qr_dms}  dominant={_qr_dom}")
            print(f"  ideal_runs={_qr_irc}  self_loop={_qr_str:.3f}  entropy={_qr_te:.3f} bits")
            print(f"  Conf  ideal={_qr_ic:.3f}  drift={_qr_dc:.3f}  mass_ratio={_qr_cmr:.3f}")
            _qr_ifs_s = f"step {_qr_ifs}" if _qr_ifs >= 0 else "never"
            _qr_dfs_s = f"step {_qr_dfs}" if _qr_dfs >= 0 else "never"
            print(f"  First  ideal@{_qr_ifs_s}  drift@{_qr_dfs_s}")
            print(f"{'─'*60}")
            continue

        if low.startswith("quadcoh3massratio"):
            _qcm_res = getattr(model, "_last_gen_result", None)
            if _qcm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcm_v  = _qcm_res.get("quad_coh3_mass_ratio", 0.0)
            _qcm_i  = _qcm_res.get("quad_weighted_ideal_coh3",    0.0)
            _qcm_d  = _qcm_res.get("quad_weighted_drifting_coh3", 0.0)
            _qcm_mom = _qcm_res.get("quad_ideal_coh3_momentum",   0.0)
            _qcm_label = ("coh3 ideal dominates" if _qcm_v >= 2.0
                           else "coh3 ideal ahead" if _qcm_v >= 1.2
                           else "coh3 balanced"    if _qcm_v >= 0.8
                           else "coh3 drift heavy")
            _qcm_bar = "█" * min(int(_qcm_v * 10), 40)
            print(f"\n  Coh3 mass ratio (ideal/drift): {_qcm_v:.4f}  [{_qcm_label}]")
            print(f"  {_qcm_bar}")
            print(f"  (ideal_coh3_mass={_qcm_i:.4f}  drifting_coh3_mass={_qcm_d:.4f})")
            print(f"  (ideal_coh3_momentum={_qcm_mom:+.4f}  positive=coh3 rising in ideal runs)")
            continue

        if low.startswith("quadidealcoh3momentum"):
            _qicm_res = getattr(model, "_last_gen_result", None)
            if _qicm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qicm_v  = _qicm_res.get("quad_ideal_coh3_momentum", 0.0)
            _qicm_cr = _qicm_res.get("quad_coh3_mass_ratio", 0.0)
            _qicm_label = ("rising"     if _qicm_v >= 0.02
                            else "stable"  if _qicm_v >= -0.02
                            else "falling")
            _qicm_filled = int(abs(_qicm_v) * 200)
            _qicm_bar = ("█" if _qicm_v >= 0 else "░") * min(_qicm_filled, 40)
            print(f"\n  Ideal coh3 momentum: {_qicm_v:+.4f}  [{_qicm_label}]")
            print(f"  {_qicm_bar}")
            print(f"  (coh3_mass_ratio={_qicm_cr:.4f})")
            print(f"  (last3_ideal_coh3 minus first3_ideal_coh3; rising=coherence improving in flow)")
            continue

        if low.startswith("quadhealthscore"):
            _qhs_res = getattr(model, "_last_gen_result", None)
            if _qhs_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qhs_v  = _qhs_res.get("quad_health_score", 0.0)
            _qhs_if = _qhs_res.get("ideal_frac", 0.0)
            _qhs_df = _qhs_res.get("drifting_frac", 0.0)
            _qhs_ic = _qhs_res.get("quad_ideal_run_confidence_mean", 0.0)
            _qhs_label = ("excellent" if _qhs_v >= 0.25
                           else "good"  if _qhs_v >= 0.12
                           else "fair"  if _qhs_v >= 0.05
                           else "poor")
            _qhs_bar = "█" * int(_qhs_v * 80)
            print(f"\n  Quadrant health score: {_qhs_v:.4f}  [{_qhs_label}]")
            print(f"  {_qhs_bar[:40]}")
            print(f"  (ideal_frac={_qhs_if:.4f}  drifting_frac={_qhs_df:.4f}  ideal_conf={_qhs_ic:.4f})")
            print(f"  (composite: ideal_frac × ideal_conf × (1−drift_frac) × conf_gap_boost)")
            continue

        if low.startswith("quadthirds"):
            _qthirds_res = getattr(model, "_last_gen_result", None)
            if _qthirds_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qt_e  = _qthirds_res.get("quad_early_third_ideal_frac", 0.0)
            _qt_m  = _qthirds_res.get("quad_mid_third_ideal_frac",   0.0)
            _qt_l  = _qthirds_res.get("quad_late_third_ideal_frac",  0.0)
            _qt_de = _qthirds_res.get("quad_early_third_drifting_frac", 0.0)
            _qt_arc = ("warm-up" if _qt_l > _qt_e + 0.08
                        else "cool-down" if _qt_e > _qt_l + 0.08
                        else "mid-peak"  if _qt_m > _qt_e + 0.05 and _qt_m > _qt_l + 0.05
                        else "stable")
            print(f"\n  Ideal thirds  [{_qt_arc}]")
            print(f"  early  {_qt_e:.4f}  {'█' * int(_qt_e * 40)}")
            print(f"  mid    {_qt_m:.4f}  {'█' * int(_qt_m * 40)}")
            print(f"  late   {_qt_l:.4f}  {'█' * int(_qt_l * 40)}")
            print(f"  (early_drifting_frac={_qt_de:.4f})")
            continue

        if low.startswith("quadweightedscores"):
            # quadweightedscores — conf-weighted ideal/drifting scores + ratio
            _qws_res = getattr(model, "_last_gen_result", None)
            if _qws_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qws_i  = _qws_res.get("quad_weighted_ideal_score",    0.0)
            _qws_d  = _qws_res.get("quad_weighted_drifting_score", 0.0)
            _qws_r  = _qws_res.get("quad_confidence_mass_ratio",   0.0)
            _qws_if = _qws_res.get("ideal_frac", 0.0)
            _qws_df = _qws_res.get("drifting_frac", 0.0)
            _qws_label = ("ideal dominates"   if _qws_r >= 2.0
                           else "ideal ahead"  if _qws_r >= 1.2
                           else "balanced"     if _qws_r >= 0.8
                           else "drift heavy"  if _qws_r >= 0.4
                           else "drift dominates")
            print(f"\n  Confidence mass  ideal={_qws_i:.4f}  drifting={_qws_d:.4f}  ratio={_qws_r:.4f}  [{_qws_label}]")
            print(f"  ideal   {'█' * int(_qws_i * 40)}")
            print(f"  drifting {'█' * int(_qws_d * 40)}")
            print(f"  (ideal_frac={_qws_if:.4f}  drifting_frac={_qws_df:.4f})")
            print(f"  (ratio = ideal conf mass / drifting conf mass; >1 = ideal holds more weight)")
            continue

        if low.startswith("quadconfidencemasratio") or low.startswith("quadconfidencemassratio"):
            _qcmr2_res = getattr(model, "_last_gen_result", None)
            if _qcmr2_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcmr2_v  = _qcmr2_res.get("quad_confidence_mass_ratio", 0.0)
            _qcmr2_i  = _qcmr2_res.get("quad_weighted_ideal_score",    0.0)
            _qcmr2_d  = _qcmr2_res.get("quad_weighted_drifting_score", 0.0)
            _qcmr2_label = ("ideal dominates"  if _qcmr2_v >= 2.0
                             else "ideal ahead" if _qcmr2_v >= 1.2
                             else "balanced"    if _qcmr2_v >= 0.8
                             else "drift heavy")
            _qcmr2_bar = "█" * min(int(_qcmr2_v * 10), 40)
            print(f"\n  Confidence mass ratio (ideal/drift): {_qcmr2_v:.4f}  [{_qcmr2_label}]")
            print(f"  {_qcmr2_bar}")
            print(f"  (ideal_conf_mass={_qcmr2_i:.4f}  drifting_conf_mass={_qcmr2_d:.4f})")
            continue

        if low.startswith("quadfirststeps"):
            # quadfirststeps — first step index for ideal + drifting
            _qfst_res = getattr(model, "_last_gen_result", None)
            if _qfst_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qfst_i  = _qfst_res.get("quad_ideal_first_step",    -1)
            _qfst_d  = _qfst_res.get("quad_drifting_first_step", -1)
            _qfst_n  = max(_qfst_res.get("output_len", 1) or 1, 1)
            _qfst_i_label = (f"step {_qfst_i} ({100*_qfst_i//_qfst_n}% in)"
                              if _qfst_i >= 0 else "never")
            _qfst_d_label = (f"step {_qfst_d} ({100*_qfst_d//_qfst_n}% in)"
                              if _qfst_d >= 0 else "never")
            print(f"\n  First ideal step:    {_qfst_i_label}")
            print(f"  First drifting step: {_qfst_d_label}")
            print(f"  (total_steps={_qfst_n})")
            print(f"  (earlier ideal=good; earlier drifting=concerning)")
            continue

        if low.startswith("quadearlylateideal"):
            # quadearlylateideal — first/second-half ideal fractions + trend
            _qeli_res = getattr(model, "_last_gen_result", None)
            if _qeli_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qeli_e  = _qeli_res.get("quad_early_ideal_frac", 0.0)
            _qeli_l  = _qeli_res.get("quad_late_ideal_frac",  0.0)
            _qeli_t  = _qeli_res.get("quad_ideal_frac_trend", 0.0)
            _qeli_trend_label = ("improving" if _qeli_t >= 0.05
                                  else "stable" if _qeli_t >= -0.05
                                  else "degrading")
            print(f"\n  Ideal frac  early={_qeli_e:.4f}  late={_qeli_l:.4f}  trend={_qeli_t:+.4f}  [{_qeli_trend_label}]")
            print(f"  early  {'█' * int(_qeli_e * 40)}")
            print(f"  late   {'█' * int(_qeli_l * 40)}")
            print(f"  (positive trend = quality improving over generation)")
            continue

        if low.startswith("quadidealfractrend"):
            _qift2_res = getattr(model, "_last_gen_result", None)
            if _qift2_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qift2_v  = _qift2_res.get("quad_ideal_frac_trend", 0.0)
            _qift2_e  = _qift2_res.get("quad_early_ideal_frac", 0.0)
            _qift2_l  = _qift2_res.get("quad_late_ideal_frac",  0.0)
            _qift2_label = ("strong improvement"  if _qift2_v >= 0.15
                             else "improvement"   if _qift2_v >= 0.05
                             else "stable"        if _qift2_v >= -0.05
                             else "mild decline"  if _qift2_v >= -0.15
                             else "strong decline")
            _qift2_filled = int(abs(_qift2_v) * 160)
            _qift2_bar = ("█" * min(_qift2_filled, 40)) if _qift2_v >= 0 else ("░" * min(_qift2_filled, 40))
            print(f"\n  Ideal frac trend: {_qift2_v:+.4f}  [{_qift2_label}]")
            print(f"  {_qift2_bar}")
            print(f"  (early={_qift2_e:.4f}  late={_qift2_l:.4f})")
            print(f"  (█=improving ░=degrading; late_ideal minus early_ideal)")
            continue

        if low.startswith("quadearlylatedrifting"):
            _qeld_res = getattr(model, "_last_gen_result", None)
            if _qeld_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qeld_e  = _qeld_res.get("quad_early_drifting_frac", 0.0)
            _qeld_l  = _qeld_res.get("quad_late_drifting_frac",  0.0)
            _qeld_t  = round(_qeld_l - _qeld_e, 4)
            _qeld_label = ("worsening drift"    if _qeld_t >= 0.05
                            else "stable drift" if _qeld_t >= -0.05
                            else "recovering")
            print(f"\n  Drifting frac  early={_qeld_e:.4f}  late={_qeld_l:.4f}  Δ={_qeld_t:+.4f}  [{_qeld_label}]")
            print(f"  early  {'█' * int(_qeld_e * 40)}")
            print(f"  late   {'█' * int(_qeld_l * 40)}")
            print(f"  (positive Δ = drift worsening over generation)")
            continue

        if low.startswith("quadruncounts"):
            # quadruncounts — ideal + drifting + exploring + flat episode counts
            _qrc_res = getattr(model, "_last_gen_result", None)
            if _qrc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qrc_i  = _qrc_res.get("quad_ideal_run_count",    0)
            _qrc_d  = _qrc_res.get("quad_drifting_run_count", 0)
            _qrc_ibar = "█" * min(_qrc_i, 40)
            _qrc_dbar = "█" * min(_qrc_d, 40)
            _qrc_n   = max(_qrc_res.get("output_len", 1) or 1, 1)
            print(f"\n  Quadrant episode counts  (total_steps={_qrc_n})")
            print(f"  ideal runs:    {_qrc_i:3d}  {_qrc_ibar}")
            print(f"  drifting runs: {_qrc_d:3d}  {_qrc_dbar}")
            continue

        if low.startswith("quadidealruncount"):
            _qirc2_res = getattr(model, "_last_gen_result", None)
            if _qirc2_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qirc2_v  = _qirc2_res.get("quad_ideal_run_count", 0)
            _qirc2_ms = _qirc2_res.get("quad_ideal_mean_streak", 0.0)
            _qirc2_mx = _qirc2_res.get("quad_ideal_max_streak",  0)
            _qirc2_label = ("many runs"   if _qirc2_v >= 5
                             else "several" if _qirc2_v >= 3
                             else "few"     if _qirc2_v >= 1
                             else "none")
            print(f"\n  Ideal run count: {_qirc2_v}  [{_qirc2_label}]")
            print(f"  {'█' * min(_qirc2_v * 3, 40)}")
            print(f"  (mean_streak={_qirc2_ms:.2f}  max_streak={_qirc2_mx})")
            continue

        if low.startswith("quaddriftingruncount"):
            _qdrc2_res = getattr(model, "_last_gen_result", None)
            if _qdrc2_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qdrc2_v  = _qdrc2_res.get("quad_drifting_run_count", 0)
            _qdrc2_ms = _qdrc2_res.get("quad_drifting_mean_streak", 0.0)
            _qdrc2_mx = _qdrc2_res.get("quad_drifting_max_streak",  0)
            _qdrc2_label = ("frequent"   if _qdrc2_v >= 5
                             else "several" if _qdrc2_v >= 3
                             else "few"     if _qdrc2_v >= 1
                             else "none")
            print(f"\n  Drifting run count: {_qdrc2_v}  [{_qdrc2_label}]")
            print(f"  {'█' * min(_qdrc2_v * 3, 40)}")
            print(f"  (mean_streak={_qdrc2_ms:.2f}  max_streak={_qdrc2_mx})")
            continue

        if low.startswith("quadidealmeanstreak"):
            _qimsb_res = getattr(model, "_last_gen_result", None)
            if _qimsb_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qimsb_v  = _qimsb_res.get("quad_ideal_mean_streak", 0.0)
            _qimsb_mx = _qimsb_res.get("quad_ideal_max_streak",  0)
            _qimsb_rc = _qimsb_res.get("quad_ideal_run_count",   0)
            _qimsb_label = ("long avg runs"   if _qimsb_v >= 5
                             else "moderate"  if _qimsb_v >= 3
                             else "short"     if _qimsb_v >= 1.5
                             else "minimal")
            _qimsb_bar = "█" * int(_qimsb_v * 4)
            print(f"\n  Ideal mean streak: {_qimsb_v:.2f}  [{_qimsb_label}]")
            print(f"  {_qimsb_bar[:40]}")
            print(f"  (max_streak={_qimsb_mx}  run_count={_qimsb_rc})")
            continue

        if low.startswith("quaddriftingmeanstreak"):
            _qdmsb_res = getattr(model, "_last_gen_result", None)
            if _qdmsb_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qdmsb_v  = _qdmsb_res.get("quad_drifting_mean_streak", 0.0)
            _qdmsb_mx = _qdmsb_res.get("quad_drifting_max_streak",  0)
            _qdmsb_rc = _qdmsb_res.get("quad_drifting_run_count",   0)
            _qdmsb_label = ("long drift spells"  if _qdmsb_v >= 4
                             else "moderate"      if _qdmsb_v >= 2
                             else "brief"         if _qdmsb_v >= 1
                             else "no drift")
            _qdmsb_bar = "█" * int(_qdmsb_v * 5)
            print(f"\n  Drifting mean streak: {_qdmsb_v:.2f}  [{_qdmsb_label}]")
            print(f"  {_qdmsb_bar[:40]}")
            print(f"  (max_streak={_qdmsb_mx}  run_count={_qdmsb_rc})")
            continue

        if low.startswith("quadstreakvariability"):
            _qsv2_res = getattr(model, "_last_gen_result", None)
            if _qsv2_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qsv2_v  = _qsv2_res.get("quad_streak_variability", 0.0)
            _qsv2_im = _qsv2_res.get("quad_ideal_mean_streak",  0.0)
            _qsv2_dm = _qsv2_res.get("quad_drifting_mean_streak", 0.0)
            _qsv2_label = ("very irregular"  if _qsv2_v >= 4.0
                            else "irregular" if _qsv2_v >= 2.0
                            else "moderate"  if _qsv2_v >= 1.0
                            else "uniform")
            _qsv2_bar = "█" * int(_qsv2_v * 5)
            print(f"\n  Streak variability (σ of all run lengths): {_qsv2_v:.4f}  [{_qsv2_label}]")
            print(f"  {_qsv2_bar[:40]}")
            print(f"  (ideal_mean={_qsv2_im:.2f}  drifting_mean={_qsv2_dm:.2f})")
            print(f"  (low=pacing is uniform; high=some quadrants run much longer than others)")
            continue

        if low.startswith("quadmaxstreaks"):
            # quadmaxstreaks — all 4 quadrant max-streak values + dominant label
            _qms_res = getattr(model, "_last_gen_result", None)
            if _qms_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qms_i  = _qms_res.get("quad_ideal_max_streak",     0)
            _qms_d  = _qms_res.get("quad_drifting_max_streak",  0)
            _qms_e  = _qms_res.get("quad_exploring_max_streak", 0)
            _qms_f  = _qms_res.get("quad_flat_max_streak",      0)
            _qms_lbl = _qms_res.get("quad_dominant_streak_label", "none")
            _qms_n  = _qms_res.get("output_len", 1) or 1
            def _qms_bar(v): return "█" * min(v, 40)
            print(f"\n  Max streak by quadrant  (dominant: {_qms_lbl})")
            print(f"  ideal     {_qms_i:3d}  {_qms_bar(_qms_i)}")
            print(f"  exploring {_qms_e:3d}  {_qms_bar(_qms_e)}")
            print(f"  drifting  {_qms_d:3d}  {_qms_bar(_qms_d)}")
            print(f"  flat      {_qms_f:3d}  {_qms_bar(_qms_f)}")
            continue

        if low.startswith("quadidealstreak"):
            _qis_res = getattr(model, "_last_gen_result", None)
            if _qis_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qis_v  = _qis_res.get("quad_ideal_max_streak", 0)
            _qis_lbl = _qis_res.get("quad_dominant_streak_label", "none")
            _qis_n  = max(_qis_res.get("output_len", 1) or 1, 1)
            _qis_label = ("sustained flow" if _qis_v >= 8
                           else "moderate flow" if _qis_v >= 4
                           else "brief flow"    if _qis_v >= 2
                           else "no ideal run")
            print(f"\n  Ideal max streak: {_qis_v} steps  [{_qis_label}]")
            print(f"  {'█' * min(_qis_v, 40)}")
            print(f"  (dominant_quad={_qis_lbl}  total_steps={_qis_n})")
            continue

        if low.startswith("quaddriftingstreak"):
            _qdss_res = getattr(model, "_last_gen_result", None)
            if _qdss_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qdss_v  = _qdss_res.get("quad_drifting_max_streak", 0)
            _qdss_lbl = _qdss_res.get("quad_dominant_streak_label", "none")
            _qdss_n  = max(_qdss_res.get("output_len", 1) or 1, 1)
            _qdss_label = ("runaway drift"    if _qdss_v >= 6
                            else "sustained"  if _qdss_v >= 3
                            else "transient"  if _qdss_v >= 1
                            else "no drift")
            print(f"\n  Drifting max streak: {_qdss_v} steps  [{_qdss_label}]")
            print(f"  {'█' * min(_qdss_v, 40)}")
            print(f"  (dominant_quad={_qdss_lbl}  total_steps={_qdss_n})")
            continue

        if low.startswith("quadexploringstreak"):
            _qess_res = getattr(model, "_last_gen_result", None)
            if _qess_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qess_v  = _qess_res.get("quad_exploring_max_streak", 0)
            _qess_lbl = _qess_res.get("quad_dominant_streak_label", "none")
            _qess_n  = max(_qess_res.get("output_len", 1) or 1, 1)
            _qess_label = ("extended divergence" if _qess_v >= 7
                            else "notable"        if _qess_v >= 4
                            else "brief"          if _qess_v >= 2
                            else "no exploring run")
            print(f"\n  Exploring max streak: {_qess_v} steps  [{_qess_label}]")
            print(f"  {'█' * min(_qess_v, 40)}")
            print(f"  (dominant_quad={_qess_lbl}  total_steps={_qess_n})")
            continue

        if low.startswith("quadflatstreak"):
            _qfss_res = getattr(model, "_last_gen_result", None)
            if _qfss_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qfss_v  = _qfss_res.get("quad_flat_max_streak", 0)
            _qfss_lbl = _qfss_res.get("quad_dominant_streak_label", "none")
            _qfss_n  = max(_qfss_res.get("output_len", 1) or 1, 1)
            _qfss_label = ("deep stagnation"    if _qfss_v >= 6
                            else "stagnant"      if _qfss_v >= 3
                            else "occasional"    if _qfss_v >= 1
                            else "no flat run")
            print(f"\n  Flat max streak: {_qfss_v} steps  [{_qfss_label}]")
            print(f"  {'█' * min(_qfss_v, 40)}")
            print(f"  (dominant_quad={_qfss_lbl}  total_steps={_qfss_n})")
            continue

        if low.startswith("quadconfidencespread"):
            _qcsp_res = getattr(model, "_last_gen_result", None)
            if _qcsp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcsp_v  = _qcsp_res.get("quad_confidence_spread", 0.0)
            _qcsp_ic = _qcsp_res.get("quad_ideal_run_confidence_mean", 0.0)
            _qcsp_dc = _qcsp_res.get("quad_drifting_run_confidence_mean", 0.0)
            _qcsp_label = ("wide separation" if _qcsp_v >= 0.12
                            else "clear"      if _qcsp_v >= 0.07
                            else "moderate"   if _qcsp_v >= 0.03
                            else "indistinct")
            _qcsp_bar = "█" * int(_qcsp_v * 200)
            print(f"\n  Confidence spread (σ of 4 quadrant means): {_qcsp_v:.4f}  [{_qcsp_label}]")
            print(f"  {_qcsp_bar[:40]}")
            print(f"  (ideal_conf={_qcsp_ic:.4f}  drifting_conf={_qcsp_dc:.4f})")
            print(f"  (wide=quadrants clearly separated in conf space; narrow=all look alike)")
            continue

        if low.startswith("quadcoh3spread"):
            _qc3sp_res = getattr(model, "_last_gen_result", None)
            if _qc3sp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qc3sp_v  = _qc3sp_res.get("quad_coh3_spread", 0.0)
            _qc3sp_cs = _qc3sp_res.get("quad_confidence_spread", 0.0)
            _qc3sp_label = ("strong coh3 separation" if _qc3sp_v >= 0.10
                             else "clear"             if _qc3sp_v >= 0.05
                             else "moderate"          if _qc3sp_v >= 0.02
                             else "indistinct")
            _qc3sp_bar = "█" * int(_qc3sp_v * 200)
            print(f"\n  Coh3 spread (σ of 4 quadrant coh3 means): {_qc3sp_v:.4f}  [{_qc3sp_label}]")
            print(f"  {_qc3sp_bar[:40]}")
            print(f"  (conf_spread={_qc3sp_cs:.4f})")
            print(f"  (wide=coh3 strongly distinguishes quadrants; narrow=coh3 not discriminating)")
            continue

        if low.startswith("quadvelocityspread"):
            _qvsp_res = getattr(model, "_last_gen_result", None)
            if _qvsp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qvsp_v  = _qvsp_res.get("quad_velocity_spread", 0.0)
            _qvsp_cs = _qvsp_res.get("quad_coh3_spread", 0.0)
            _qvsp_label = ("strong velocity separation" if _qvsp_v >= 0.08
                            else "clear"                if _qvsp_v >= 0.04
                            else "moderate"             if _qvsp_v >= 0.01
                            else "indistinct")
            _qvsp_bar = "█" * int(_qvsp_v * 250)
            print(f"\n  Velocity spread (σ of 4 quadrant vel means): {_qvsp_v:.4f}  [{_qvsp_label}]")
            print(f"  {_qvsp_bar[:40]}")
            print(f"  (coh3_spread={_qvsp_cs:.4f})")
            print(f"  (wide=velocity clearly splits quadrants; narrow=speed undifferentiated)")
            continue

        if low.startswith("quadcoh3idealvsflatratio"):
            _qcifr_res = getattr(model, "_last_gen_result", None)
            if _qcifr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcifr_v  = _qcifr_res.get("quad_coh3_ideal_vs_flat_ratio", 0.0)
            _qcifr_cs = _qcifr_res.get("quad_coh3_spread", 0.0)
            _qcifr_label = ("strongly separated" if _qcifr_v >= 1.30
                             else "clear gap"     if _qcifr_v >= 1.10
                             else "slight gap"    if _qcifr_v >= 1.02
                             else "inverted" if _qcifr_v < 1.0 and _qcifr_v > 0
                             else "no data")
            _qcifr_bar = "█" * min(int(_qcifr_v * 15), 40)
            print(f"\n  Coh3 ideal/flat ratio: {_qcifr_v:.4f}  [{_qcifr_label}]")
            print(f"  {_qcifr_bar}")
            print(f"  (coh3_spread={_qcifr_cs:.4f})")
            print(f"  (>1.30=ideal clearly more coherent; <1.0=flat paradoxically more coherent)")
            continue

        if low.startswith("quadvelocityidealvsdriftingratio"):
            _qvidr_res = getattr(model, "_last_gen_result", None)
            if _qvidr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qvidr_v  = _qvidr_res.get("quad_velocity_ideal_vs_drifting_ratio", 0.0)
            _qvidr_vs = _qvidr_res.get("quad_velocity_spread", 0.0)
            _qvidr_label = ("ideal much slower"    if _qvidr_v <= 0.55
                             else "ideal slower"   if _qvidr_v <= 0.80
                             else "near equal"     if _qvidr_v <= 0.95
                             else "ideal as fast" if _qvidr_v > 0.95
                             else "no data")
            _qvidr_bar = "█" * min(int(_qvidr_v * 20), 40)
            print(f"\n  Velocity ideal/drifting ratio: {_qvidr_v:.4f}  [{_qvidr_label}]")
            print(f"  {_qvidr_bar}")
            print(f"  (velocity_spread={_qvidr_vs:.4f})")
            print(f"  (<0.55=ideal is slow & controlled; >0.95=ideal moves as fast as drifting)")
            continue

        if low.startswith("quadexploringrunconfidencemean"):
            # quadexploringrunconfidencemean — mean confidence during exploring-quadrant steps
            _qerc_res = getattr(model, "_last_gen_result", None)
            if _qerc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qerc_v  = _qerc_res.get("quad_exploring_run_confidence_mean", 0.0)
            _qerc_ef = _qerc_res.get("exploring_frac", 0.0)
            _qerc_ir = _qerc_res.get("quad_ideal_run_confidence_mean", 0.0)
            _qerc_label = ("near-ideal" if _qerc_ir > 0 and _qerc_v >= _qerc_ir * 0.85
                            else "moderate"   if _qerc_v >= 0.40
                            else "low")
            _qerc_bar = "█" * int(_qerc_v * 40)
            print(f"\n  Exploring-run confidence mean: {_qerc_v:.4f}  [{_qerc_label}]")
            print(f"  {_qerc_bar}")
            print(f"  (exploring_frac={_qerc_ef:.4f}  ideal_conf_mean={_qerc_ir:.4f})")
            print(f"  (mean confidence during exploring steps; near-ideal=good, low=volatile)")
            continue

        if low.startswith("quadflatrunconfidencemean"):
            # quadflatrunconfidencemean — mean confidence during flat-quadrant steps
            _qfrc_res = getattr(model, "_last_gen_result", None)
            if _qfrc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qfrc_v  = _qfrc_res.get("quad_flat_run_confidence_mean", 0.0)
            _qfrc_ff = _qfrc_res.get("flat_frac", 0.0)
            _qfrc_ir = _qfrc_res.get("quad_ideal_run_confidence_mean", 0.0)
            _qfrc_label = ("stagnation trap" if _qfrc_ir > 0 and _qfrc_v >= _qfrc_ir * 0.80
                            else "moderate stagnation" if _qfrc_v >= 0.40
                            else "low-confidence flat")
            _qfrc_bar = "█" * int(_qfrc_v * 40)
            print(f"\n  Flat-run confidence mean: {_qfrc_v:.4f}  [{_qfrc_label}]")
            print(f"  {_qfrc_bar}")
            print(f"  (flat_frac={_qfrc_ff:.4f}  ideal_conf_mean={_qfrc_ir:.4f})")
            print(f"  (high flat-conf = confident repetition trap; low = uncontrolled stagnation)")
            continue

        if low.startswith("quadconfidencegap"):
            # quadconfidencegap — ideal_conf_mean minus drifting_conf_mean
            _qcg_res = getattr(model, "_last_gen_result", None)
            if _qcg_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcg_v  = _qcg_res.get("quad_confidence_gap", 0.0)
            _qcg_ic = _qcg_res.get("quad_ideal_run_confidence_mean", 0.0)
            _qcg_dc = _qcg_res.get("quad_drifting_run_confidence_mean", 0.0)
            _qcg_label = ("strong separation"  if _qcg_v >= 0.15
                           else "clear gap"     if _qcg_v >= 0.07
                           else "weak gap"      if _qcg_v >= 0.01
                           else "inverted" if _qcg_v < 0
                           else "indistinct")
            _qcg_filled = int(abs(_qcg_v) * 200)
            _qcg_bar = ("█" * min(_qcg_filled, 40)) if _qcg_v >= 0 else ("░" * min(_qcg_filled, 40))
            print(f"\n  Confidence gap (ideal − drifting): {_qcg_v:+.4f}  [{_qcg_label}]")
            print(f"  {_qcg_bar}")
            print(f"  (ideal_conf={_qcg_ic:.4f}  drifting_conf={_qcg_dc:.4f})")
            print(f"  (positive=ideal more confident; inverted=drift paradoxically certain)")
            continue

        if low.startswith("quadidealrunconfidencemean"):
            # quadidealrunconfidencemean — mean confidence during ideal-quadrant steps
            _qirc_res = getattr(model, "_last_gen_result", None)
            if _qirc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qirc_v  = _qirc_res.get("quad_ideal_run_confidence_mean", 0.0)
            _qirc_if = _qirc_res.get("ideal_frac", 0.0)
            _qirc_ir = _qirc_res.get("ideal_run_density", 0.0)
            _qirc_label = ("very high" if _qirc_v >= 0.60
                            else "high"     if _qirc_v >= 0.45
                            else "medium"   if _qirc_v >= 0.30
                            else "low")
            _qirc_bar = "█" * int(_qirc_v * 40)
            print(f"\n  Ideal-run confidence mean: {_qirc_v:.4f}  [{_qirc_label}]")
            print(f"  {_qirc_bar}")
            print(f"  (ideal_frac={_qirc_if:.4f}  ideal_run_density={_qirc_ir:.4f})")
            print(f"  (mean confidence while in ideal state; higher = more stable quality flow)")
            continue

        if low.startswith("quaddriftingrunconfidencemean"):
            # quaddriftingrunconfidencemean — mean confidence during drifting-quadrant steps
            _qdrc_res = getattr(model, "_last_gen_result", None)
            if _qdrc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qdrc_v  = _qdrc_res.get("quad_drifting_run_confidence_mean", 0.0)
            _qdrc_df = _qdrc_res.get("drifting_frac", 0.0)
            _qdrc_ir = _qdrc_res.get("quad_ideal_run_confidence_mean", 0.0)
            _qdrc_label = ("surprisingly high" if _qdrc_v >= _qdrc_ir * 0.85
                            else "near-ideal"       if _qdrc_v >= _qdrc_ir * 0.65
                            else "low")
            _qdrc_bar = "█" * int(_qdrc_v * 40)
            print(f"\n  Drifting-run confidence mean: {_qdrc_v:.4f}  [{_qdrc_label}]")
            print(f"  {_qdrc_bar}")
            print(f"  (drifting_frac={_qdrc_df:.4f}  ideal_conf_mean={_qdrc_ir:.4f})")
            print(f"  (mean confidence while drifting; lower than ideal=expected; higher=unusual)")
            continue

        if low.startswith("quadselftransitionrate"):
            # quadselftransitionrate — fraction of steps where quadrant doesn't change
            _qstr_res = getattr(model, "_last_gen_result", None)
            if _qstr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qstr_v  = _qstr_res.get("quad_self_transition_rate", 0.0)
            _qstr_te = _qstr_res.get("quad_transition_entropy", 0.0)
            _qstr_label = ("very stuck" if _qstr_v >= 0.65
                            else "sticky"   if _qstr_v >= 0.45
                            else "mobile"   if _qstr_v >= 0.25
                            else "restless")
            _qstr_bar = "█" * int(_qstr_v * 40)
            print(f"\n  Self-transition rate: {_qstr_v:.4f}  [{_qstr_label}]")
            print(f"  {_qstr_bar}")
            print(f"  (transition_entropy={_qstr_te:.4f} bits)")
            print(f"  (fraction of steps where quadrant label is unchanged; higher=more stuck)")
            continue

        if low.startswith("quadtransitionmatrixskew"):
            # quadtransitionmatrixskew — skew of per-row transition probabilities
            _qtms_res = getattr(model, "_last_gen_result", None)
            if _qtms_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qtms_v  = _qtms_res.get("quad_transition_matrix_skew", 0.0)
            _qtms_te = _qtms_res.get("quad_transition_entropy", 0.0)
            _qtms_label = ("highly skewed" if _qtms_v >= 0.70
                            else "skewed"       if _qtms_v >= 0.45
                            else "moderate"     if _qtms_v >= 0.20
                            else "uniform")
            _qtms_bar = "█" * int(_qtms_v * 40)
            print(f"\n  Transition matrix skew: {_qtms_v:.4f}  [{_qtms_label}]")
            print(f"  {_qtms_bar}")
            print(f"  (transition_entropy={_qtms_te:.4f} bits)")
            print(f"  (0=uniform transitions per source  1=strongly biased toward one destination)")
            continue

        if low.startswith("quadidealtodriftingrate"):
            # quadidealtodriftingrate — fraction of ideal exits tipping into drifting
            _qitd_res = getattr(model, "_last_gen_result", None)
            if _qitd_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qitd_v  = _qitd_res.get("quad_ideal_to_drifting_rate", 0.0)
            _qitd_if = _qitd_res.get("ideal_frac",   0.0)
            _qitd_df = _qitd_res.get("drifting_frac", 0.0)
            _qitd_label = ("high risk" if _qitd_v >= 0.30
                            else "moderate" if _qitd_v >= 0.12
                            else "low")
            _qitd_bar = "█" * int(_qitd_v * 40)
            print(f"\n  Ideal→Drifting collapse rate: {_qitd_v:.4f}  [{_qitd_label}]")
            print(f"  {_qitd_bar}")
            print(f"  (ideal_frac={_qitd_if:.4f}  drifting_frac={_qitd_df:.4f})")
            print(f"  (fraction of ideal exits that tip into uncontrolled drift; lower=better)")
            continue

        if low.startswith("quadtransitionentropy"):
            # quadtransitionentropy — Shannon entropy over quadrant transition types
            _qte_res = getattr(model, "_last_gen_result", None)
            if _qte_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qte_v  = _qte_res.get("quad_transition_entropy", 0.0)
            _qte_qs = _qte_res.get("quadsummary",             "")
            _qte_label = ("chaotic"    if _qte_v >= 2.5
                           else "varied"     if _qte_v >= 1.5
                           else "patterned"  if _qte_v >= 0.8
                           else "rigid")
            _qte_bar = "█" * int(min(_qte_v / 3.6 * 40, 40))
            print(f"\n  Transition entropy: {_qte_v:.4f} bits  [{_qte_label}]")
            print(f"  {_qte_bar}")
            print(f"  (max possible ≈ 3.58 bits for 12 equal-prob transitions)")
            print(f"  (low=predictable path  high=unpredictable multi-path transitions)")
            if _qte_qs:
                print(f"  quad summary: {_qte_qs}")
            continue

        if low.startswith("quadexploringtodriftingrate"):
            # quadexploringtodriftingrate — fraction of exploring exits tipping into drifting
            _qetd_res = getattr(model, "_last_gen_result", None)
            if _qetd_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qetd_v  = _qetd_res.get("quad_exploring_to_drifting_rate", 0.0)
            _qetd_ef = _qetd_res.get("exploring_frac", 0.0)
            _qetd_df = _qetd_res.get("drifting_frac",  0.0)
            _qetd_label = ("high risk" if _qetd_v >= 0.40
                            else "moderate" if _qetd_v >= 0.18
                            else "low")
            _qetd_bar = "█" * int(_qetd_v * 40)
            print(f"\n  Exploring→Drifting collapse rate: {_qetd_v:.4f}  [{_qetd_label}]")
            print(f"  {_qetd_bar}")
            print(f"  (exploring_frac={_qetd_ef:.4f}  drifting_frac={_qetd_df:.4f})")
            print(f"  (fraction of exploring exits that tip into uncontrolled drift; lower=better)")
            continue

        if low.startswith("quadidealtoflatrate"):
            # quadidealtoflatrate — fraction of ideal exits collapsing into flat
            _qitf_res = getattr(model, "_last_gen_result", None)
            if _qitf_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qitf_v  = _qitf_res.get("quad_ideal_to_flat_rate", 0.0)
            _qitf_if = _qitf_res.get("ideal_frac", 0.0)
            _qitf_ff = _qitf_res.get("flat_frac",  0.0)
            _qitf_label = ("high risk" if _qitf_v >= 0.35
                            else "moderate" if _qitf_v >= 0.15
                            else "low")
            _qitf_bar = "█" * int(_qitf_v * 40)
            print(f"\n  Ideal→Flat collapse rate: {_qitf_v:.4f}  [{_qitf_label}]")
            print(f"  {_qitf_bar}")
            print(f"  (ideal_frac={_qitf_if:.4f}  flat_frac={_qitf_ff:.4f})")
            print(f"  (fraction of ideal exits that fall into stagnation; lower=better)")
            continue

        if low.startswith("quaddriftingtoidealrate"):
            # quaddriftingtoidealrate — fraction of drifting exits landing directly in ideal
            _qdti_res = getattr(model, "_last_gen_result", None)
            if _qdti_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qdti_v  = _qdti_res.get("quad_drifting_to_ideal_rate", 0.0)
            _qdti_df = _qdti_res.get("drifting_frac", 0.0)
            _qdti_if = _qdti_res.get("ideal_frac",   0.0)
            _qdti_label = ("excellent" if _qdti_v >= 0.30
                            else "good"     if _qdti_v >= 0.12
                            else "poor")
            _qdti_bar = "█" * int(_qdti_v * 40)
            print(f"\n  Drifting→Ideal direct rate: {_qdti_v:.4f}  [{_qdti_label}]")
            print(f"  {_qdti_bar}")
            print(f"  (drifting_frac={_qdti_df:.4f}  ideal_frac={_qdti_if:.4f})")
            print(f"  (fraction of drifting exits that jump directly to ideal; higher=better)")
            continue

        if low.startswith("quadexploringtoflatrate"):
            # quadexploringtoflatrate — fraction of exploring exits collapsing into flat
            _qetf_res = getattr(model, "_last_gen_result", None)
            if _qetf_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qetf_v  = _qetf_res.get("quad_exploring_to_flat_rate", 0.0)
            _qetf_ef = _qetf_res.get("exploring_frac", 0.0)
            _qetf_ff = _qetf_res.get("flat_frac",      0.0)
            _qetf_label = ("high risk" if _qetf_v >= 0.40
                            else "moderate" if _qetf_v >= 0.18
                            else "low")
            _qetf_bar = "█" * int(_qetf_v * 40)
            print(f"\n  Exploring→Flat collapse rate: {_qetf_v:.4f}  [{_qetf_label}]")
            print(f"  {_qetf_bar}")
            print(f"  (exploring_frac={_qetf_ef:.4f}  flat_frac={_qetf_ff:.4f})")
            print(f"  (fraction of exploring exits that collapse into stagnation; lower=better)")
            continue

        if low.startswith("quadflattoidealrate"):
            # quadflattoidealrate — fraction of flat exits landing directly in ideal
            _qfti_res = getattr(model, "_last_gen_result", None)
            if _qfti_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qfti_v  = _qfti_res.get("quad_flat_to_ideal_rate", 0.0)
            _qfti_ff = _qfti_res.get("flat_frac",  0.0)
            _qfti_if = _qfti_res.get("ideal_frac", 0.0)
            _qfti_label = ("excellent" if _qfti_v >= 0.35
                            else "good"     if _qfti_v >= 0.15
                            else "poor")
            _qfti_bar = "█" * int(_qfti_v * 40)
            print(f"\n  Flat→Ideal direct rate: {_qfti_v:.4f}  [{_qfti_label}]")
            print(f"  {_qfti_bar}")
            print(f"  (flat_frac={_qfti_ff:.4f}  ideal_frac={_qfti_if:.4f})")
            print(f"  (fraction of flat exits that jump directly to ideal state; higher=better)")
            continue

        if low.startswith("quaddriftingtoexploringrate"):
            # quaddriftingtoexploringrate — fraction of drifting exits landing in exploring
            _qdte_res = getattr(model, "_last_gen_result", None)
            if _qdte_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qdte_v  = _qdte_res.get("quad_drifting_to_exploring_rate", 0.0)
            _qdte_df = _qdte_res.get("drifting_frac",  0.0)
            _qdte_ef = _qdte_res.get("exploring_frac", 0.0)
            _qdte_label = ("excellent" if _qdte_v >= 0.45
                            else "good"     if _qdte_v >= 0.20
                            else "poor")
            _qdte_bar = "█" * int(_qdte_v * 40)
            print(f"\n  Drifting→Exploring transition rate: {_qdte_v:.4f}  [{_qdte_label}]")
            print(f"  {_qdte_bar}")
            print(f"  (drifting_frac={_qdte_df:.4f}  exploring_frac={_qdte_ef:.4f})")
            print(f"  (fraction of drifting exits that become creative divergence; higher=better)")
            continue

        if low.startswith("quadflattodriftingrate"):
            # quadflattodriftingrate — fraction of flat exits landing in drifting
            _qftd_res = getattr(model, "_last_gen_result", None)
            if _qftd_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qftd_v  = _qftd_res.get("quad_flat_to_drifting_rate", 0.0)
            _qftd_ff = _qftd_res.get("flat_frac",     0.0)
            _qftd_df = _qftd_res.get("drifting_frac", 0.0)
            _qftd_label = ("high risk" if _qftd_v >= 0.45
                            else "moderate" if _qftd_v >= 0.20
                            else "low")
            _qftd_bar = "█" * int(_qftd_v * 40)
            print(f"\n  Flat→Drifting transition rate: {_qftd_v:.4f}  [{_qftd_label}]")
            print(f"  {_qftd_bar}")
            print(f"  (flat_frac={_qftd_ff:.4f}  drifting_frac={_qftd_df:.4f})")
            print(f"  (fraction of flat exits that escalate to drifting; lower=better)")
            continue

        if low.startswith("quadflattoexploringrate"):
            # quadflattoexploringrate — fraction of flat exits landing in exploring
            _qfte_res = getattr(model, "_last_gen_result", None)
            if _qfte_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qfte_v  = _qfte_res.get("quad_flat_to_exploring_rate", 0.0)
            _qfte_ff = _qfte_res.get("flat_frac",     0.0)
            _qfte_ef = _qfte_res.get("exploring_frac", 0.0)
            _qfte_label = ("excellent" if _qfte_v >= 0.45
                            else "good"     if _qfte_v >= 0.20
                            else "poor")
            _qfte_bar = "█" * int(_qfte_v * 40)
            print(f"\n  Flat→Exploring transition rate: {_qfte_v:.4f}  [{_qfte_label}]")
            print(f"  {_qfte_bar}")
            print(f"  (flat_frac={_qfte_ff:.4f}  exploring_frac={_qfte_ef:.4f})")
            print(f"  (fraction of flat exits that jump into creative divergence; higher=better)")
            continue

        if low.startswith("quaddriftingtoflatrate"):
            # quaddriftingtoflatrate — fraction of drifting exits landing in flat
            _qdtf_res = getattr(model, "_last_gen_result", None)
            if _qdtf_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qdtf_v  = _qdtf_res.get("quad_drifting_to_flat_rate", 0.0)
            _qdtf_df = _qdtf_res.get("drifting_frac", 0.0)
            _qdtf_ff = _qdtf_res.get("flat_frac",     0.0)
            _qdtf_label = ("high risk" if _qdtf_v >= 0.50
                            else "moderate" if _qdtf_v >= 0.25
                            else "low")
            _qdtf_bar = "█" * int(_qdtf_v * 40)
            print(f"\n  Drifting→Flat transition rate: {_qdtf_v:.4f}  [{_qdtf_label}]")
            print(f"  {_qdtf_bar}")
            print(f"  (drifting_frac={_qdtf_df:.4f}  flat_frac={_qdtf_ff:.4f})")
            print(f"  (fraction of drifting exits that collapse into stagnation; lower=better)")
            continue

        if low.startswith("quadexploringtoidealrate"):
            # quadexploringtoidealrate — fraction of exploring exits landing in ideal
            _qeti_res = getattr(model, "_last_gen_result", None)
            if _qeti_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qeti_v  = _qeti_res.get("quad_exploring_to_ideal_rate", 0.0)
            _qeti_ef = _qeti_res.get("exploring_frac", 0.0)
            _qeti_if = _qeti_res.get("ideal_frac",     0.0)
            _qeti_label = ("excellent" if _qeti_v >= 0.60
                            else "good"     if _qeti_v >= 0.35
                            else "poor")
            _qeti_bar = "█" * int(_qeti_v * 40)
            print(f"\n  Exploring→Ideal transition rate: {_qeti_v:.4f}  [{_qeti_label}]")
            print(f"  {_qeti_bar}")
            print(f"  (exploring_frac={_qeti_ef:.4f}  ideal_frac={_qeti_if:.4f})")
            print(f"  (fraction of exploring exits that resolve into ideal; higher=better)")
            continue

        if low.startswith("quadflatexitvelocity"):
            # quadflatexitvelocity — mean velocity at flat-quadrant exit
            _qfev_res = getattr(model, "_last_gen_result", None)
            if _qfev_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qfev_v  = _qfev_res.get("quad_flat_exit_velocity", 0.0)
            _qfev_vm = _qfev_res.get("quad_velocity_mean",      0.0)
            _qfev_ff = _qfev_res.get("flat_frac",               0.0)
            _qfev_label = ("sharp"  if _qfev_v >= _qfev_vm * 1.15
                            else "normal" if _qfev_v >= _qfev_vm * 0.85
                            else "sluggish")
            _qfev_bar = "█" * int(min(_qfev_v * 60, 40))
            print(f"\n  Flat-exit velocity: {_qfev_v:.4f}  [{_qfev_label}]")
            print(f"  {_qfev_bar}")
            print(f"  (mean velocity baseline={_qfev_vm:.4f}  flat_frac={_qfev_ff:.4f})")
            print(f"  (higher = model snaps out of stagnation sharply; lower = sticky flat)")
            continue

        if low.startswith("quadidealtoexploringrate"):
            # quadidealtoexploringrate — fraction of ideal exits → exploring
            _qiter_res = getattr(model, "_last_gen_result", None)
            if _qiter_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qiter_v  = _qiter_res.get("quad_ideal_to_exploring_rate", 0.0)
            _qiter_if = _qiter_res.get("ideal_frac",    0.0)
            _qiter_ef = _qiter_res.get("exploring_frac", 0.0)
            _qiter_label = ("high"   if _qiter_v >= 0.50
                             else "medium" if _qiter_v >= 0.25
                             else "low")
            _qiter_bar = "█" * int(_qiter_v * 40)
            print(f"\n  Ideal→Exploring transition rate: {_qiter_v:.4f}  [{_qiter_label}]")
            print(f"  {_qiter_bar}")
            print(f"  (ideal_frac={_qiter_if:.4f}  exploring_frac={_qiter_ef:.4f})")
            print(f"  (fraction of ideal-quadrant exits that land in exploring; 1.0=always)")
            continue

        if low.startswith("quadidealentrycoh3variance"):
            # quadidealentrycoh3variance — std-dev of coh3 at ideal-quadrant entry
            _qiecv_res = getattr(model, "_last_gen_result", None)
            if _qiecv_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qiecv_v  = _qiecv_res.get("quad_ideal_entry_coh3_variance", 0.0)
            _qiecv_em = _qiecv_res.get("quad_coh3_entry_mean", 0.0)
            _qiecv_if = _qiecv_res.get("ideal_frac", 0.0)
            _qiecv_label = ("very noisy"  if _qiecv_v >= 0.12
                             else "noisy"     if _qiecv_v >= 0.07
                             else "moderate"  if _qiecv_v >= 0.03
                             else "stable")
            _qiecv_bar = "█" * int(min(_qiecv_v / 0.18 * 40, 40))
            print(f"\n  Ideal-entry coh3 variance: {_qiecv_v:.4f}  [{_qiecv_label}]")
            print(f"  {_qiecv_bar}")
            print(f"  (entry mean coh3={_qiecv_em:.4f}  ideal_frac={_qiecv_if:.4f})")
            print(f"  (low=consistent gate into ideal  high=noisy/unpredictable entry threshold)")
            continue

        if low.startswith("quadexploringexitcoh3"):
            # quadexploringexitcoh3 — mean coh3 at exploring-quadrant exit
            _qeec_res = getattr(model, "_last_gen_result", None)
            if _qeec_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qeec_v  = _qeec_res.get("quad_exploring_exit_coh3", 0.0)
            _qeec_ef = _qeec_res.get("exploring_frac", 0.0)
            _qeec_vm = _qeec_res.get("quad_coh3_entry_mean",     0.0)
            _qeec_label = ("graceful" if _qeec_v >= _qeec_vm * 1.05
                            else "normal"  if _qeec_v >= _qeec_vm * 0.90
                            else "abrupt")
            _qeec_bar = "█" * int(min(_qeec_v * 80, 40))
            print(f"\n  Exploring-exit coh3: {_qeec_v:.4f}  [{_qeec_label}]")
            print(f"  {_qeec_bar}")
            print(f"  (ideal-entry mean coh3={_qeec_vm:.4f}  exploring_frac={_qeec_ef:.4f})")
            print(f"  (higher = model leaves exploring quadrant with high coherence = graceful)")
            continue

        if low.startswith("quaddriftingdurationvariance"):
            # quaddriftingdurationvariance — std-dev of drifting-run lengths
            _qddv_res = getattr(model, "_last_gen_result", None)
            if _qddv_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qddv_v  = _qddv_res.get("quad_drifting_duration_variance", 0.0)
            _qddv_df = _qddv_res.get("drifting_frac", 0.0)
            _qddv_label = ("very erratic" if _qddv_v >= 4.0
                            else "erratic"    if _qddv_v >= 2.0
                            else "moderate"   if _qddv_v >= 1.0
                            else "consistent")
            _qddv_bar = "█" * int(min(_qddv_v / 6.0 * 40, 40))
            print(f"\n  Drifting duration variance: {_qddv_v:.4f}  [{_qddv_label}]")
            print(f"  {_qddv_bar}")
            print(f"  (drifting_frac={_qddv_df:.4f} — fraction of steps that were drifting)")
            print(f"  (0=uniform drift  ≥2=erratic drift bursts  ≥4=very erratic)")
            continue

        if low.startswith("quadexploringdurationvariance"):
            # quadexploringdurationvariance — std-dev of exploring-run lengths
            _qedv_res = getattr(model, "_last_gen_result", None)
            if _qedv_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qedv_v  = _qedv_res.get("quad_exploring_duration_variance", 0.0)
            _qedv_ef = _qedv_res.get("exploring_frac", 0.0)
            _qedv_label = ("very erratic" if _qedv_v >= 4.0
                            else "erratic"    if _qedv_v >= 2.0
                            else "moderate"   if _qedv_v >= 1.0
                            else "consistent")
            _qedv_bar = "█" * int(min(_qedv_v / 6.0 * 40, 40))
            print(f"\n  Exploring duration variance: {_qedv_v:.4f}  [{_qedv_label}]")
            print(f"  {_qedv_bar}")
            print(f"  (exploring_frac={_qedv_ef:.4f} — fraction of steps in creative divergence)")
            print(f"  (0=uniform exploring  ≥2=erratic creative bursts  ≥4=very erratic)")
            continue

        if low.startswith("quadflatdurationvariance"):
            # quadflatdurationvariance — std-dev of flat-run lengths
            _qfdv_res = getattr(model, "_last_gen_result", None)
            if _qfdv_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qfdv_v  = _qfdv_res.get("quad_flat_duration_variance", 0.0)
            _qfdv_ff = _qfdv_res.get("flat_frac",  0.0)
            _qfdv_label = ("very erratic" if _qfdv_v >= 4.0
                            else "erratic"    if _qfdv_v >= 2.0
                            else "moderate"   if _qfdv_v >= 1.0
                            else "consistent")
            _qfdv_bar = "█" * int(min(_qfdv_v / 6.0 * 40, 40))
            print(f"\n  Flat duration variance: {_qfdv_v:.4f}  [{_qfdv_label}]")
            print(f"  {_qfdv_bar}")
            print(f"  (flat_frac={_qfdv_ff:.4f} — fraction of steps in stagnant state)")
            print(f"  (0=uniform stagnation  ≥2=erratic flat bursts  ≥4=very erratic)")
            continue

        if low.startswith("quadrecoveryvelocity"):
            # quadrecoveryvelocity — mean velocity at drifting-quadrant exit
            _qrv_res = getattr(model, "_last_gen_result", None)
            if _qrv_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qrv_v  = _qrv_res.get("quad_recovery_velocity", 0.0)
            _qrv_vm = _qrv_res.get("quad_velocity_mean",     0.0)
            _qrv_rr = _qrv_res.get("quad_recovery_rate",     0.0)
            _qrv_label = ("smooth"  if _qrv_v <= _qrv_vm * 0.80
                           else "normal" if _qrv_v <= _qrv_vm * 1.20
                           else "abrupt")
            _qrv_bar = "█" * int(min(_qrv_v * 60, 40))
            print(f"\n  Recovery velocity: {_qrv_v:.4f}  [{_qrv_label}]")
            print(f"  {_qrv_bar}")
            print(f"  (mean velocity across all steps: {_qrv_vm:.4f}  recovery_rate={_qrv_rr:.4f})")
            print(f"  (lower = model exits drifting more smoothly)")
            continue

        if low.startswith("quaddriftingentryvelocity"):
            # quaddriftingentryvelocity — mean velocity at drifting-quadrant entry
            _qdv_res = getattr(model, "_last_gen_result", None)
            if _qdv_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qdv_v  = _qdv_res.get("quad_drifting_entry_velocity", 0.0)
            _qdv_vm = _qdv_res.get("quad_velocity_mean",           0.0)
            _qdv_label = ("hard fall"  if _qdv_v >= _qdv_vm * 1.20
                           else "normal"  if _qdv_v >= _qdv_vm * 0.80
                           else "soft fall")
            _qdv_bar = "█" * int(min(_qdv_v * 60, 40))
            print(f"\n  Drifting entry velocity: {_qdv_v:.4f}  [{_qdv_label}]")
            print(f"  {_qdv_bar}")
            print(f"  (mean velocity across all steps: {_qdv_vm:.4f})")
            print(f"  (higher = model falls harder into drifting state)")
            continue

        if low.startswith("quadidealdurationvariance"):
            # quadidealdurationvariance — std-dev of ideal run lengths
            _qidv_res = getattr(model, "_last_gen_result", None)
            if _qidv_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qidv_v  = _qidv_res.get("quad_ideal_duration_variance", 0.0)
            _qidv_rl = _qidv_res.get("ideal_run_len",    0)
            _qidv_if = _qidv_res.get("ideal_frac",       0.0)
            _qidv_label = ("very erratic" if _qidv_v >= 4.0
                            else "erratic"    if _qidv_v >= 2.0
                            else "moderate"   if _qidv_v >= 1.0
                            else "consistent")
            _qidv_bar = "█" * int(min(_qidv_v / 6.0 * 40, 40))
            print(f"\n  Ideal duration variance: {_qidv_v:.4f}  [{_qidv_label}]")
            print(f"  {_qidv_bar}")
            print(f"  (ideal_frac={_qidv_if:.4f}  longest_ideal_run={_qidv_rl})")
            print(f"  (0=perfectly consistent  ≥2=erratic  ≥4=very erratic)")
            continue

        if low.startswith("quadcoh3entrymean"):
            # quadcoh3entrymean — mean coh3 at ideal-quadrant entry
            _qce_res = getattr(model, "_last_gen_result", None)
            if _qce_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qce_v  = _qce_res.get("quad_coh3_entry_mean", 0.0)
            _qce_m  = _qce_res.get("quad_coh3_mean", {}).get("ideal", 0.0)
            _qce_gm = _qce_res.get("quad_coh3_mean", {})
            _qce_overall = (sum(_qce_gm.values()) / max(len(_qce_gm), 1)) if _qce_gm else 0.0
            _qce_label = ("strong entry" if _qce_v >= _qce_overall + 0.04
                           else "weak entry" if _qce_v <= _qce_overall - 0.04
                           else "neutral entry")
            _qce_bar = "█" * int(min(_qce_v * 50, 40))
            print(f"\n  Quad coh3 entry mean: {_qce_v:.4f}  [{_qce_label}]")
            print(f"  {_qce_bar}")
            print(f"  (overall mean coh3 = {_qce_overall:.4f}; ideal in-state mean = {_qce_m:.4f})")
            print(f"  (higher = model enters ideal from strong coherence foundation)")
            continue

        if low.startswith("quaddriftingexitcoh3"):
            # quaddriftingexitcoh3 — mean coh3 at drifting-quadrant exit
            _qdx_res = getattr(model, "_last_gen_result", None)
            if _qdx_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qdx_v  = _qdx_res.get("quad_drifting_exit_coh3", 0.0)
            _qdx_dm = _qdx_res.get("quad_coh3_mean", {}).get("drifting", 0.0)
            _qdx_gm = _qdx_res.get("quad_coh3_mean", {})
            _qdx_overall = (sum(_qdx_gm.values()) / max(len(_qdx_gm), 1)) if _qdx_gm else 0.0
            _qdx_label = ("healthy exit" if _qdx_v >= _qdx_overall
                           else "weak exit")
            _qdx_bar = "█" * int(min(_qdx_v * 50, 40))
            print(f"\n  Drifting exit coh3: {_qdx_v:.4f}  [{_qdx_label}]")
            print(f"  {_qdx_bar}")
            print(f"  (drifting in-state mean = {_qdx_dm:.4f}; overall = {_qdx_overall:.4f})")
            print(f"  (higher coh3 at exit = model recovers from drifting more cleanly)")
            continue

        if low.startswith("quadoscillationscore"):
            # quadoscillationscore — fraction of A→B→A ping-pong transitions
            _qos_res = getattr(model, "_last_gen_result", None)
            if _qos_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qos_s  = _qos_res.get("quad_oscillation_score", 0.0)
            _qos_tr = _qos_res.get("transition_rate",        0.0)
            _qos_label = ("severe"   if _qos_s >= 0.60
                           else "high"    if _qos_s >= 0.40
                           else "moderate" if _qos_s >= 0.20
                           else "low")
            _qos_bar = "█" * int(_qos_s * 40)
            print(f"\n  Quad oscillation score: {_qos_s:.4f}  [{_qos_label}]")
            print(f"  {_qos_bar}")
            print(f"  (transition_rate={_qos_tr:.4f} — how often model changes quadrant)")
            print(f"  (≥0.60=severe  ≥0.40=high  ≥0.20=moderate  <0.20=low)")
            continue

        if low.startswith("quadidealentryvelocity"):
            # quadidealentryvelocity — mean velocity at ideal-quadrant entry
            _qiev_res = getattr(model, "_last_gen_result", None)
            if _qiev_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qiev_v  = _qiev_res.get("quad_ideal_entry_velocity", 0.0)
            _qiev_vm = _qiev_res.get("quad_velocity_mean",        0.0)
            _qiev_label = ("smooth"  if _qiev_v <= _qiev_vm * 0.80
                            else "normal" if _qiev_v <= _qiev_vm * 1.20
                            else "abrupt")
            _qiev_bar = "█" * int(min(_qiev_v * 60, 40))
            print(f"\n  Ideal entry velocity: {_qiev_v:.4f}  [{_qiev_label}]")
            print(f"  {_qiev_bar}")
            print(f"  (mean velocity across all steps: {_qiev_vm:.4f})")
            print(f"  (lower = smoother entry into ideal quadrant)")
            continue

        if low.startswith("quadpersistencescore"):
            # quadpersistencescore — avg steps per quadrant visit (stickiness)
            _qps_res = getattr(model, "_last_gen_result", None)
            if _qps_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qps_s  = _qps_res.get("quad_persistence_score", 0.0)
            _qps_tr = _qps_res.get("transition_rate",        0.0)
            _qps_label = ("very sticky" if _qps_s >= 8.0
                           else "sticky"     if _qps_s >= 5.0
                           else "moderate"   if _qps_s >= 3.0
                           else "volatile")
            _qps_norm  = min(_qps_s / 10.0, 1.0)
            _qps_bar   = "█" * int(_qps_norm * 40)
            print(f"\n  Quad persistence score: {_qps_s:.2f}  [{_qps_label}]")
            print(f"  {_qps_bar}")
            print(f"  (avg {_qps_s:.2f} steps per quadrant visit; transition_rate={_qps_tr:.4f})")
            print(f"  (≥8=very sticky  ≥5=sticky  ≥3=moderate  <3=volatile)")
            continue

        if low.startswith("idealstabilityscore"):
            # idealstabilityscore — ideal_frac − drifting_frac net quality index
            _iss_res = getattr(model, "_last_gen_result", None)
            if _iss_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _iss_s  = _iss_res.get("ideal_stability_score", 0.0)
            _iss_if = _iss_res.get("ideal_frac",   0.0)
            _iss_df = _iss_res.get("drifting_frac", 0.0)
            _iss_sign  = "+" if _iss_s >= 0 else ""
            _iss_label = ("very stable"   if _iss_s >=  0.30
                           else "stable"       if _iss_s >=  0.10
                           else "neutral"      if abs(_iss_s) < 0.10
                           else "unstable"     if _iss_s >= -0.30
                           else "very unstable")
            _iss_bar_w = int(abs(_iss_s) * 40)
            _iss_bar   = ("→ " + "█" * _iss_bar_w) if _iss_s >= 0 else ("← " + "█" * _iss_bar_w)
            print(f"\n  Ideal stability score: {_iss_sign}{_iss_s:.4f}  [{_iss_label}]")
            print(f"  {_iss_bar}")
            print(f"  (ideal_frac={_iss_if:.4f}  drifting_frac={_iss_df:.4f})")
            print(f"  (≥+0.30=very stable  ≥+0.10=stable  ±0.10=neutral  ≤−0.10=unstable)")
            continue

        if low.startswith("idealrundensity"):
            # idealrundensity — ideal_frac × longest_ideal_run quality richness
            _ird_res = getattr(model, "_last_gen_result", None)
            if _ird_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ird_d  = _ird_res.get("ideal_run_density", 0.0)
            _ird_if = _ird_res.get("ideal_frac",        0.0)
            _ird_rl = _ird_res.get("ideal_run_len",     0)
            _ird_label = ("rich"     if _ird_d >= 0.20
                           else "good"    if _ird_d >= 0.10
                           else "sparse"  if _ird_d >= 0.04
                           else "minimal")
            _ird_bar = "█" * int(min(_ird_d, 1.0) * 40)
            print(f"\n  Ideal run density: {_ird_d:.4f}  [{_ird_label}]")
            print(f"  {_ird_bar}")
            print(f"  (ideal_frac={_ird_if:.4f}  ×  longest_ideal_run={_ird_rl})")
            print(f"  (≥0.20=rich  ≥0.10=good  ≥0.04=sparse  <0.04=minimal)")
            continue

        if low.startswith("quadrecoveryrate"):
            # quadrecoveryrate — how often drifting recovers to ideal/exploring
            _qrr_res = getattr(model, "_last_gen_result", None)
            if _qrr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qrr_r  = _qrr_res.get("quad_recovery_rate", 0.0)
            _qrr_df = _qrr_res.get("drifting_frac",      0.0)
            _qrr_label = ("excellent" if _qrr_r >= 0.70
                           else "good"      if _qrr_r >= 0.50
                           else "moderate"  if _qrr_r >= 0.30
                           else "poor")
            _qrr_bar = "█" * int(_qrr_r * 40)
            print(f"\n  Quad recovery rate: {_qrr_r:.4f}  [{_qrr_label}]")
            print(f"  {_qrr_bar}")
            print(f"  (drifting_frac={_qrr_df:.4f} — fraction of steps that were drifting)")
            print(f"  (≥0.70=excellent  ≥0.50=good  ≥0.30=moderate  <0.30=poor)")
            continue

        if low.startswith("quadvolatilityscore"):
            # quadvolatilityscore — normalised flip rate between quadrants
            _qv_res = getattr(model, "_last_gen_result", None)
            if _qv_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qv_s   = _qv_res.get("quad_volatility_score", 0.0)
            _qv_tr  = _qv_res.get("transition_rate",       0.0)
            _qv_label = ("very volatile" if _qv_s >= 0.60
                          else "volatile"    if _qv_s >= 0.40
                          else "moderate"    if _qv_s >= 0.20
                          else "stable")
            _qv_bar = "█" * int(_qv_s * 40)
            print(f"\n  Quad volatility score: {_qv_s:.4f}  [{_qv_label}]")
            print(f"  {_qv_bar}")
            print(f"  (transition_rate={_qv_tr:.4f}  — proportion of steps that change quadrant)")
            print(f"  (≥0.60=very volatile  ≥0.40=volatile  ≥0.20=moderate  <0.20=stable)")
            continue

        if low.startswith("quaddominancemargin"):
            # quaddominancemargin — how decisively one quadrant dominates
            _qdm_res = getattr(model, "_last_gen_result", None)
            if _qdm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qdm_m  = _qdm_res.get("quad_dominance_margin", 0.0)
            _qdm_if = _qdm_res.get("ideal_frac",     0.0)
            _qdm_ef = _qdm_res.get("exploring_frac", 0.0)
            _qdm_ff = _qdm_res.get("flat_frac",      0.0)
            _qdm_df = _qdm_res.get("drifting_frac",  0.0)
            _qdm_dom = max(
                [("ideal", _qdm_if), ("exploring", _qdm_ef),
                 ("flat",  _qdm_ff), ("drifting",  _qdm_df)],
                key=lambda t: t[1])[0]
            _qdm_label = ("decisive"  if _qdm_m >= 0.30
                           else "clear"    if _qdm_m >= 0.15
                           else "slight"   if _qdm_m >= 0.08
                           else "contested")
            _qdm_bar = "█" * int(_qdm_m * 40)
            print(f"\n  Quad dominance margin: {_qdm_m:.4f}  [{_qdm_label}]  dominant={_qdm_dom}")
            print(f"  {_qdm_bar}")
            print(f"  (margin = dominant_frac − second_frac; 0=tied  1=total dominance)")
            print(f"  (≥0.30=decisive  ≥0.15=clear  ≥0.08=slight  <0.08=contested)")
            continue

        if low.startswith("exploringfrac"):
            # exploringfrac — fraction of exploring-quadrant steps (coh3↑ vel↑)
            _ef_res = getattr(model, "_last_gen_result", None)
            if _ef_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ef_f   = _ef_res.get("exploring_frac", 0.0)
            _ef_qm  = _ef_res.get("quadrant_map", {})
            _ef_n   = min(len(_ef_res.get("coh3_steps",    [])),
                          len(_ef_res.get("velocity_steps", [])))
            _ef_label = ("dominant"  if _ef_f >= 0.40
                          else "high"     if _ef_f >= 0.25
                          else "moderate" if _ef_f >= 0.15
                          else "low")
            _ef_bar = "█" * int(_ef_f * 40)
            _ef_n2  = _ef_qm.get("exploring", 0)
            print(f"\n  Exploring-quadrant fraction: {_ef_f:.4f}  [{_ef_label}]")
            print(f"  {_ef_bar}")
            print(f"  ({_ef_n2} exploring steps out of {_ef_n} total)")
            print(f"  (coh3↑ vel↑ — creative divergence; ≥0.40=dominant)")
            continue

        if low.startswith("flatfrac"):
            # flatfrac — fraction of flat-quadrant steps (coh3↓ vel↓)
            _ff_res = getattr(model, "_last_gen_result", None)
            if _ff_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ff_f   = _ff_res.get("flat_frac", 0.0)
            _ff_qm  = _ff_res.get("quadrant_map", {})
            _ff_n   = min(len(_ff_res.get("coh3_steps",    [])),
                          len(_ff_res.get("velocity_steps", [])))
            _ff_label = ("dominant"  if _ff_f >= 0.40
                          else "high"     if _ff_f >= 0.25
                          else "moderate" if _ff_f >= 0.15
                          else "low")
            _ff_bar = "█" * int(_ff_f * 40)
            _ff_n2  = _ff_qm.get("flat", 0)
            print(f"\n  Flat-quadrant fraction: {_ff_f:.4f}  [{_ff_label}]")
            print(f"  {_ff_bar}")
            print(f"  ({_ff_n2} flat steps out of {_ff_n} total)")
            print(f"  (coh3↓ vel↓ — stagnant; ≥0.40=dominant)")
            continue

        if low.startswith("quadsummary"):
            # quadsummary — all-in-one quadrant dashboard
            _qs_res = getattr(model, "_last_gen_result", None)
            if _qs_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qs_if  = _qs_res.get("ideal_frac",      0.0)
            _qs_ef  = _qs_res.get("exploring_frac",  0.0)
            _qs_ff  = _qs_res.get("flat_frac",        0.0)
            _qs_df  = _qs_res.get("drifting_frac",    0.0)
            _qs_bal = _qs_res.get("quad_balance_score", 0.0)
            _qs_tr  = _qs_res.get("transition_rate",  0.0)
            _qs_irl = _qs_res.get("ideal_run_len",    0)
            _qs_ier = _qs_res.get("ideal_entry_rate", 0.0)
            _qs_dom = max(
                [("ideal", _qs_if), ("exploring", _qs_ef),
                 ("flat",  _qs_ff), ("drifting",  _qs_df)],
                key=lambda t: t[1])[0]
            _qs_bal_sign = "+" if _qs_bal >= 0 else ""
            def _qsbar(v, w=20): return "█" * int(v * w) + "░" * (w - int(v * w))
            print(f"\n  ── Quadrant Summary ──────────────────────────────────────")
            print(f"  ideal     {_qsbar(_qs_if)}  {_qs_if:.3f}")
            print(f"  exploring {_qsbar(_qs_ef)}  {_qs_ef:.3f}")
            print(f"  flat      {_qsbar(_qs_ff)}  {_qs_ff:.3f}")
            print(f"  drifting  {_qsbar(_qs_df)}  {_qs_df:.3f}")
            print(f"  ──────────────────────────────────────────────────────────")
            print(f"  dominant quadrant : {_qs_dom}")
            print(f"  balance score     : {_qs_bal_sign}{_qs_bal:.4f}"
                  f"  ({'quality-biased' if _qs_bal>=0.20 else 'neutral' if abs(_qs_bal)<0.20 else 'drift-biased'})")
            print(f"  transition rate   : {_qs_tr:.4f}"
                  f"  ({'volatile' if _qs_tr>=0.50 else 'stable'})")
            print(f"  ideal run len     : {_qs_irl}")
            print(f"  ideal entry rate  : {_qs_ier:.4f}")
            continue

        if low.startswith("driftingfrac"):
            # driftingfrac — fraction of drifting-quadrant steps + bar
            _df_res = getattr(model, "_last_gen_result", None)
            if _df_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _df_f   = _df_res.get("drifting_frac", 0.0)
            _df_qm  = _df_res.get("quadrant_map", {})
            _df_n   = min(len(_df_res.get("coh3_steps",    [])),
                          len(_df_res.get("velocity_steps", [])))
            _df_label = ("dominant"  if _df_f >= 0.40
                          else "high"     if _df_f >= 0.25
                          else "moderate" if _df_f >= 0.15
                          else "low")
            _df_bar = "█" * int(_df_f * 40)
            _df_drift_n = _df_qm.get("drifting", 0)
            print(f"\n  Drifting-quadrant fraction: {_df_f:.4f}  [{_df_label}]")
            print(f"  {_df_bar}")
            print(f"  ({_df_drift_n} drifting steps out of {_df_n} total)")
            print(f"  (≥0.40=dominant  ≥0.25=high  ≥0.15=moderate  <0.15=low)")
            continue

        if low.startswith("quadbalancescore"):
            # quadbalancescore — (ideal−drifting)/n quality bias indicator
            _qb_res = getattr(model, "_last_gen_result", None)
            if _qb_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qb_s   = _qb_res.get("quad_balance_score", 0.0)
            _qb_if  = _qb_res.get("ideal_frac",    0.0)
            _qb_df  = _qb_res.get("drifting_frac", 0.0)
            _qb_label = ("quality-biased" if _qb_s >=  0.20
                          else "neutral"   if abs(_qb_s) < 0.20
                          else "drift-biased")
            _qb_sign = "+" if _qb_s >= 0 else ""
            _qb_bar_r = int(abs(_qb_s) * 40)
            _qb_bar = ("→ " + "█" * _qb_bar_r) if _qb_s >= 0 else ("← " + "█" * _qb_bar_r)
            print(f"\n  Quad balance score: {_qb_sign}{_qb_s:.4f}  [{_qb_label}]")
            print(f"  {_qb_bar}")
            print(f"  ideal_frac={_qb_if:.4f}  drifting_frac={_qb_df:.4f}")
            print(f"  (≥+0.20=quality-biased  ±0.20=neutral  ≤−0.20=drift-biased)")
            continue

        if low.startswith("quadtransitionto"):
            # quadtransitionto — which quadrant was entered most often
            _qtt_res = getattr(model, "_last_gen_result", None)
            if _qtt_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qtt_qt  = _qtt_res.get("quad_transition_to", {})
            if not _qtt_qt:
                print("  No transitions found."); continue
            _qtt_sym = {"ideal": "●", "exploring": "◐", "flat": "○", "drifting": "◌"}
            _qtt_total = max(sum(_qtt_qt.values()), 1)
            print(f"\n  Quadrant entry counts  (most entered = most returned-to):")
            for _q in sorted(_qtt_qt, key=lambda k: _qtt_qt[k], reverse=True):
                _qv = _qtt_qt[_q]
                _qbar = "█" * int((_qv / _qtt_total) * 20)
                print(f"  {_qtt_sym.get(_q,'?')} {_q:<10} {_qv:4}  {_qbar}")
            _qtt_most = max(_qtt_qt, key=_qtt_qt.get)
            print(f"  most entered quadrant: {_qtt_most}")
            continue

        if low.startswith("idealentryrate"):
            # idealentryrate — fraction of transitions landing in ideal quadrant
            _ier_res = getattr(model, "_last_gen_result", None)
            if _ier_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ier_r  = _ier_res.get("ideal_entry_rate",    0.0)
            _ier_qtt = _ier_res.get("quad_transition_to", {})
            _ier_tot = sum(_ier_qtt.values()) if _ier_qtt else 0
            _ier_ide = _ier_qtt.get("ideal", 0)
            _ier_label = ("excellent" if _ier_r >= 0.40
                          else "good"  if _ier_r >= 0.25
                          else "fair"  if _ier_r >= 0.10
                          else "rare")
            _ier_bar = "█" * int(_ier_r * 40)
            print(f"\n  Ideal entry rate: {_ier_r:.4f}  [{_ier_label}]")
            print(f"  {_ier_bar}")
            print(f"  ({_ier_ide} ideal entries out of {_ier_tot} total transitions)")
            print(f"  (≥0.40=excellent  ≥0.25=good  ≥0.10=fair  <0.10=rare)")
            continue

        if low.startswith("confcoh3gap"):
            # confcoh3gap — mean_conf − mean_coh3 gap + interpretation
            _ccg_res = getattr(model, "_last_gen_result", None)
            if _ccg_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ccg_g  = _ccg_res.get("conf_coh3_gap", 0.0)
            _ccg_mc = _ccg_res.get("mean_conf",  0.0)
            _ccg_c3 = _ccg_res.get("coh3_steps", [])
            _ccg_mc3 = sum(_ccg_c3) / max(len(_ccg_c3), 1)
            _ccg_label = ("overconfident" if _ccg_g >  0.10
                          else "balanced" if abs(_ccg_g) <= 0.10
                          else "under-confident")
            _ccg_sign  = "+" if _ccg_g >= 0 else ""
            _ccg_bar_r = int(abs(_ccg_g) * 100)
            _ccg_bar   = ("→ " + "█" * _ccg_bar_r) if _ccg_g > 0 else ("← " + "█" * _ccg_bar_r)
            print(f"\n  Conf–Coh3 gap: {_ccg_sign}{_ccg_g:.5f}  [{_ccg_label}]")
            print(f"  {_ccg_bar}")
            print(f"  mean_conf={_ccg_mc:.4f}  mean_coh3={_ccg_mc3:.4f}")
            print(f"  (>+0.10=overconfident  ±0.10=balanced  <−0.10=under-confident)")
            continue

        if low.startswith("quadtransitionfrom"):
            # quadtransitionfrom — which quadrant was departed most often
            _qtf_res = getattr(model, "_last_gen_result", None)
            if _qtf_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qtf_qf  = _qtf_res.get("quad_transition_from", {})
            if not _qtf_qf:
                print("  No transitions found."); continue
            _qtf_sym = {"ideal": "●", "exploring": "◐", "flat": "○", "drifting": "◌"}
            _qtf_total = max(sum(_qtf_qf.values()), 1)
            print(f"\n  Quadrant departure counts  (most left = least stable):")
            for _q in sorted(_qtf_qf, key=lambda k: _qtf_qf[k], reverse=True):
                _qv = _qtf_qf[_q]
                _qbar = "█" * int((_qv / _qtf_total) * 20)
                print(f"  {_qtf_sym.get(_q,'?')} {_q:<10} {_qv:4}  {_qbar}")
            _qtf_most = max(_qtf_qf, key=_qtf_qf.get)
            print(f"  most abandoned quadrant: {_qtf_most}")
            continue

        if low.startswith("coh3veldivergence"):
            # coh3veldivergence — |mean_coh3 − (1 − mean_vel)| alignment measure
            _cvd_res = getattr(model, "_last_gen_result", None)
            if _cvd_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cvd_d  = _cvd_res.get("coh3_vel_divergence", 0.0)
            _cvd_c3 = _cvd_res.get("coh3_steps",    [])
            _cvd_vl = _cvd_res.get("velocity_steps", [])
            _cvd_n  = min(len(_cvd_c3), len(_cvd_vl))
            _cvd_mc3 = sum(_cvd_c3[:_cvd_n])  / max(_cvd_n, 1)
            _cvd_mv  = sum(_cvd_vl[:_cvd_n])  / max(_cvd_n, 1)
            _cvd_label = ("aligned"    if _cvd_d < 0.05
                          else "slight" if _cvd_d < 0.10
                          else "moderate" if _cvd_d < 0.15
                          else "misaligned")
            _cvd_bar = "█" * int(_cvd_d * 100)
            print(f"\n  Coh3/Vel divergence: {_cvd_d:.5f}  [{_cvd_label}]")
            print(f"  {_cvd_bar}")
            print(f"  mean_coh3={_cvd_mc3:.4f}  mean_vel={_cvd_mv:.4f}  (1−vel={1-_cvd_mv:.4f})")
            print(f"  (<0.05=aligned  <0.10=slight  <0.15=moderate  ≥0.15=misaligned)")
            continue

        if low.startswith("quadcoh3mean"):
            # quadcoh3mean — per-quadrant mean coherence (coh3) table
            _qcb_res = getattr(model, "_last_gen_result", None)
            if _qcb_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcb_qc = _qcb_res.get("quad_coh3_mean", {})
            if not _qcb_qc:
                print("  Need ≥2 steps."); continue
            _qcb_sym = {"ideal": "●", "exploring": "◐", "flat": "○", "drifting": "◌"}
            _qcb_max = max(_qcb_qc.values(), default=0.001)
            print(f"\n  Per-quadrant mean coh3  (higher = more coherent):")
            for _q in ["ideal", "exploring", "flat", "drifting"]:
                _qv = _qcb_qc.get(_q, 0.0)
                _qbar = "█" * int((_qv / max(_qcb_max, 1e-9)) * 20) if _qv > 0 else "(no steps)"
                print(f"  {_qcb_sym.get(_q,'?')} {_q:<10} {_qv:.5f}  {_qbar}")
            _qcb_best = max(_qcb_qc, key=lambda k: _qcb_qc[k])
            print(f"  most coherent quadrant: {_qcb_best}")
            continue

        if low.startswith("idealfrac"):
            # idealfrac — fraction of ideal-quadrant steps + bar
            _if_res = getattr(model, "_last_gen_result", None)
            if _if_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _if_f   = _if_res.get("ideal_frac", 0.0)
            _if_qm  = _if_res.get("quadrant_map", {})
            _if_n   = min(len(_if_res.get("coh3_steps",    [])),
                          len(_if_res.get("velocity_steps", [])))
            _if_label = ("dominant"  if _if_f >= 0.50
                          else "strong"   if _if_f >= 0.35
                          else "moderate" if _if_f >= 0.20
                          else "sparse")
            _if_bar = "█" * int(_if_f * 40)
            _if_ideal_n = _if_qm.get("ideal", 0)
            print(f"\n  Ideal-quadrant fraction: {_if_f:.4f}  [{_if_label}]")
            print(f"  {_if_bar}")
            print(f"  ({_if_ideal_n} ideal steps out of {_if_n} total)")
            print(f"  (≥0.50=dominant  ≥0.35=strong  ≥0.20=moderate  <0.20=sparse)")
            continue

        if low.startswith("quadvelocitymean"):
            # quadvelocitymean — per-quadrant mean semantic velocity table
            _qvm_res = getattr(model, "_last_gen_result", None)
            if _qvm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qvm_qv = _qvm_res.get("quad_velocity_mean", {})
            if not _qvm_qv:
                print("  Need ≥2 steps."); continue
            _qvm_sym = {"ideal": "●", "exploring": "◐", "flat": "○", "drifting": "◌"}
            _qvm_max = max(_qvm_qv.values(), default=0.001)
            print(f"\n  Per-quadrant mean velocity  (lower = more stable):")
            for _q in ["ideal", "exploring", "flat", "drifting"]:
                _qv = _qvm_qv.get(_q, 0.0)
                _qbar = "█" * int((_qv / max(_qvm_max, 1e-9)) * 20) if _qv > 0 else "(no steps)"
                print(f"  {_qvm_sym.get(_q,'?')} {_q:<10} {_qv:.5f}  {_qbar}")
            _qvm_min = min(_qvm_qv, key=lambda k: _qvm_qv[k] if _qvm_qv[k] > 0 else 99)
            print(f"  most stable quadrant: {_qvm_min}")
            continue

        if low.startswith("quadconfmean"):
            # quadconfmean — per-quadrant mean confidence table
            _qcm_res = getattr(model, "_last_gen_result", None)
            if _qcm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qcm_qc = _qcm_res.get("quad_conf_mean", {})
            if not _qcm_qc:
                print("  Need ≥2 steps."); continue
            _qcm_sym = {"ideal": "●", "exploring": "◐", "flat": "○", "drifting": "◌"}
            _qcm_max = max(_qcm_qc.values(), default=0.001)
            print(f"\n  Per-quadrant mean confidence  (higher = more certain):")
            for _q in ["ideal", "exploring", "flat", "drifting"]:
                _qv = _qcm_qc.get(_q, 0.0)
                _qbar = "█" * int((_qv / max(_qcm_max, 1e-9)) * 20) if _qv > 0 else "(no steps)"
                print(f"  {_qcm_sym.get(_q,'?')} {_q:<10} {_qv:.5f}  {_qbar}")
            _qcm_best = max(_qcm_qc, key=lambda k: _qcm_qc[k])
            print(f"  most confident quadrant: {_qcm_best}")
            continue

        if low.startswith("transitionrate"):
            # transitionrate — quadrant instability rate + label + adaptive note
            _tr_res = getattr(model, "_last_gen_result", None)
            if _tr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _tr_r   = _tr_res.get("transition_rate", 0.0)
            _tr_ptm = _tr_res.get("phase_transition_map", [])
            _tr_n   = min(len(_tr_res.get("coh3_steps", [])),
                          len(_tr_res.get("velocity_steps", [])))
            _tr_label = ("volatile"  if _tr_r >= 0.50
                          else "moderate" if _tr_r >= 0.25
                          else "stable")
            _tr_bar = "█" * int(_tr_r * 40)
            print(f"\n  Transition rate: {_tr_r:.4f}  [{_tr_label}]")
            print(f"  {_tr_bar}")
            print(f"  ({len(_tr_ptm)} transitions across {_tr_n} steps)")
            print(f"  (≥0.50=volatile  0.25–0.49=moderate  <0.25=stable)")
            continue

        if low.startswith("confvelocityscore"):
            # confvelocityscore — mean_conf × (1 − mean_vel) composite quality score
            _cvs_res = getattr(model, "_last_gen_result", None)
            if _cvs_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cvs_sc  = _cvs_res.get("conf_velocity_score", 0.0)
            _cvs_mc  = _cvs_res.get("mean_conf",    0.0)
            _cvs_vel = _cvs_res.get("velocity_steps", [])
            _cvs_mv  = sum(_cvs_vel) / max(len(_cvs_vel), 1)
            _cvs_label = ("excellent" if _cvs_sc >= 0.60
                          else "good" if _cvs_sc >= 0.45
                          else "fair" if _cvs_sc >= 0.30
                          else "weak")
            _cvs_bar = "█" * int(_cvs_sc * 40)
            print(f"\n  Conf×Velocity score: {_cvs_sc:.4f}  [{_cvs_label}]")
            print(f"  {_cvs_bar}")
            print(f"  mean_conf={_cvs_mc:.4f}  mean_vel={_cvs_mv:.4f}  (1−vel={1-_cvs_mv:.4f})")
            print(f"  (≥0.60=excellent  ≥0.45=good  ≥0.30=fair  <0.30=weak)")
            continue

        if low.startswith("quadentropy"):
            # quadentropy — per-quadrant mean entropy table
            _qe_res = getattr(model, "_last_gen_result", None)
            if _qe_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qe_qe = _qe_res.get("quad_entropy", {})
            if not _qe_qe:
                print("  Need ≥2 steps."); continue
            _qe_sym = {"ideal": "●", "exploring": "◐", "flat": "○", "drifting": "◌"}
            _qe_min = min((v for v in _qe_qe.values() if v > 0), default=0.001)
            _qe_max = max(_qe_qe.values(), default=0.001)
            print(f"\n  Per-quadrant mean entropy  (lower = more focused):")
            for _q in ["ideal", "exploring", "flat", "drifting"]:
                _qv = _qe_qe.get(_q, 0.0)
                _qbar = "█" * int((_qv / max(_qe_max, 1e-9)) * 20) if _qv > 0 else "(no steps)"
                print(f"  {_qe_sym.get(_q,'?')} {_q:<10} {_qv:.5f}  {_qbar}")
            _qe_focus = min(_qe_qe, key=lambda k: _qe_qe[k] if _qe_qe[k] > 0 else 99)
            print(f"  most focused quadrant: {_qe_focus}")
            continue

        if low.startswith("confentropyratio"):
            # confentropyratio — mean_conf / mean_entropy ratio + adaptive interpretation
            _cer_res = getattr(model, "_last_gen_result", None)
            if _cer_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cer_r = _cer_res.get("conf_entropy_ratio", 0.0)
            _cer_mc = _cer_res.get("mean_conf",    0.0)
            _cer_me = sum(_cer_res.get("entropy_steps", [])) / max(len(_cer_res.get("entropy_steps", [])), 1)
            _cer_label = ("overconfident"  if _cer_r > 1.4
                          else "balanced"  if _cer_r > 0.8
                          else "entropy-led")
            _cer_bar_len = min(40, max(1, int(_cer_r * 20)))
            _cer_bar = "█" * _cer_bar_len
            print(f"\n  Conf/Entropy ratio: {_cer_r:.4f}  [{_cer_label}]")
            print(f"  {_cer_bar}")
            print(f"  mean_conf={_cer_mc:.4f}  mean_entropy={_cer_me:.4f}")
            print(f"  (>1.4=overconfident  0.8-1.4=balanced  <0.8=entropy-led)")
            continue

        if low.startswith("phasetransitionmap"):
            # phasetransitionmap — every quadrant change step with from→to arrows
            _ptm_res = getattr(model, "_last_gen_result", None)
            if _ptm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ptm_map = _ptm_res.get("phase_transition_map", [])
            if not _ptm_map:
                print("  No quadrant transitions found (≥2 steps needed, or all steps same quadrant).")
                continue
            _ptm_sym = {"ideal": "●", "exploring": "◐", "flat": "○", "drifting": "◌"}
            _ptm_d2i = sum(1 for _, f, t in _ptm_map if f == "drifting"  and t == "ideal")
            _ptm_i2d = sum(1 for _, f, t in _ptm_map if f == "ideal"     and t == "drifting")
            _ptm_any_ideal = sum(1 for _, f, t in _ptm_map if t == "ideal")
            print(f"\n  Phase transition map  ({len(_ptm_map)} transitions):")
            for _pti, _ptf, _ptt in _ptm_map:
                _ptf_s = _ptm_sym.get(_ptf, "?"); _ptt_s = _ptm_sym.get(_ptt, "?")
                print(f"  step {_pti:<4}  {_ptf_s} {_ptf:<10} → {_ptt_s} {_ptt}")
            print(f"  drifting→ideal recoveries: {_ptm_d2i}  |  ideal→drifting falls: {_ptm_i2d}")
            continue

        if low.startswith("phasequalityscore"):
            # phasequalityscore — early/mid/late phase quality (avg_conf × avg_coh3)
            _pqs_res = getattr(model, "_last_gen_result", None)
            if _pqs_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _pqs_sc = _pqs_res.get("phase_quality_score", {})
            if not _pqs_sc:
                print("  Need ≥2 steps."); continue
            _pqs_max = max(_pqs_sc.values()) if _pqs_sc else 0.001
            print(f"\n  Phase quality score  (avg_conf × avg_coh3 per third):")
            for _ph in ["early", "mid", "late"]:
                _pv = _pqs_sc.get(_ph, 0.0)
                _bar = "█" * int(_pv / max(_pqs_max, 0.001) * 20)
                print(f"  {_ph:<6} {_pv:.4f}  {_bar}")
            _pqs_best = max(_pqs_sc, key=_pqs_sc.get) if _pqs_sc else "—"
            print(f"  best phase: {_pqs_best}")
            continue

        if low.startswith("streakmap"):
            # streakmap — sequence of quadrant streaks across generation
            _sm_res  = getattr(model, "_last_gen_result", None)
            if _sm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _sm_map  = _sm_res.get("streak_map",  [])
            if not _sm_map:
                print("  Need ≥2 steps."); continue
            _sm_sym  = {"ideal": "●", "exploring": "◐", "flat": "○", "drifting": "◌"}
            _sm_line = "  ".join(f"{_sm_sym.get(q,'?')}×{n}" for q, n in _sm_map)
            _sm_longest = max(_sm_map, key=lambda x: x[1]) if _sm_map else ("—", 0)
            _sm_ideal_total = sum(n for q, n in _sm_map if q == "ideal")
            _sm_total = sum(n for _, n in _sm_map)
            print(f"\n  Streak map  ({len(_sm_map)} runs  |  total {_sm_total} steps):")
            print(f"  {_sm_line}")
            print(f"  ● ideal  ◐ exploring  ○ flat  ◌ drifting")
            print(f"  longest run: {_sm_longest[0]}×{_sm_longest[1]}  |  ideal steps: {_sm_ideal_total}/{_sm_total} ({_sm_ideal_total/max(_sm_total,1):.1%})")
            continue

        if low.startswith("quadrantmap"):
            # quadrantmap — coh3×vel quadrant distribution (ideal/exploring/flat/drifting)
            _qm_res = getattr(model, "_last_gen_result", None)
            if _qm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qm_qmap = _qm_res.get("quadrant_map", {})
            _qm_irl  = _qm_res.get("ideal_run_len", 0)
            _qm_c3   = _qm_res.get("coh3_steps",    [])
            _qm_vel  = _qm_res.get("velocity_steps", [])
            _qm_n    = min(len(_qm_c3), len(_qm_vel))
            if _qm_n < 2 or not _qm_qmap:
                print("  Need ≥2 steps."); continue
            _qm_total = max(sum(_qm_qmap.values()), 1)
            _qm_ideal   = _qm_qmap.get("ideal",     0)
            _qm_expl    = _qm_qmap.get("exploring", 0)
            _qm_flat    = _qm_qmap.get("flat",       0)
            _qm_drift   = _qm_qmap.get("drifting",  0)
            _qm_dom = max(_qm_qmap, key=_qm_qmap.get)
            print(f"\n  Quadrant map  ({_qm_n} steps)")
            print(f"  ideal     (coh3↑ vel↓): {_qm_ideal:4}  {_qm_ideal/_qm_total:.1%}")
            print(f"  exploring (coh3↑ vel↑): {_qm_expl:4}  {_qm_expl/_qm_total:.1%}")
            print(f"  flat      (coh3↓ vel↓): {_qm_flat:4}  {_qm_flat/_qm_total:.1%}")
            print(f"  drifting  (coh3↓ vel↑): {_qm_drift:4}  {_qm_drift/_qm_total:.1%}")
            print(f"  dominant: {_qm_dom}  |  longest ideal streak: {_qm_irl} steps")
            continue

        if low.startswith("idealrunlen"):
            # idealrunlen — longest consecutive ideal streak detail
            _irl_res  = getattr(model, "_last_gen_result", None)
            if _irl_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _irl_len  = _irl_res.get("ideal_run_len",  0)
            _irl_c3   = _irl_res.get("coh3_steps",      [])
            _irl_vel  = _irl_res.get("velocity_steps",  [])
            _irl_cf   = _irl_res.get("confidences",     [])
            _irl_n    = min(len(_irl_c3), len(_irl_vel), len(_irl_cf))
            if _irl_n < 4:
                print("  Need ≥4 steps."); continue
            _irl_c3m  = sum(_irl_c3[:_irl_n])  / _irl_n
            _irl_vm   = sum(_irl_vel[:_irl_n]) / _irl_n
            _irl_cfm  = sum(_irl_cf[:_irl_n])  / _irl_n
            _irl_flags = [
                ("✓" if (_irl_c3[i] > _irl_c3m and _irl_vel[i] < _irl_vm and _irl_cf[i] > _irl_cfm)
                 else "─")
                for i in range(_irl_n)
            ]
            _irl_bar   = "".join(_irl_flags)
            _irl_label = ("excellent" if _irl_len >= 4 else
                          ("good" if _irl_len == 3 else
                           ("fair" if _irl_len == 2 else "none")))
            print(f"\n  Ideal run length: {_irl_len} steps  ({_irl_label})")
            print(f"  {_irl_bar}")
            print(f"  (✓ = coh3>avg ∧ vel<avg ∧ conf>avg  |  longest ✓ streak = {_irl_len})")
            continue

        if low.startswith("entropyveljoint"):
            # entropyveljoint — focused+stable joint step meter
            _evj_res  = getattr(model, "_last_gen_result", None)
            if _evj_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _evj_frac = _evj_res.get("entropy_vel_joint",  0.0)
            _evj_ent  = _evj_res.get("entropy_steps",       [])
            _evj_vel  = _evj_res.get("velocity_steps",      [])
            _evj_n    = min(len(_evj_ent), len(_evj_vel))
            if _evj_n < 4:
                print("  Need ≥4 steps."); continue
            _evj_em   = sum(_evj_ent[:_evj_n])  / _evj_n
            _evj_vm   = sum(_evj_vel[:_evj_n])  / _evj_n
            _evj_flags = [
                ("✓" if (_evj_ent[i] < _evj_em and _evj_vel[i] < _evj_vm) else "─")
                for i in range(_evj_n)
            ]
            _evj_bar   = "".join(_evj_flags)
            _evj_label = ("excellent" if _evj_frac > 0.50 else
                          ("good" if _evj_frac > 0.35 else
                           ("fair" if _evj_frac > 0.20 else "low")))
            print(f"\n  Entropy+vel joint meter  ({_evj_n} steps)  {_evj_frac:.1%}  ({_evj_label}):")
            print(f"  {_evj_bar}")
            print(f"  (✓ = entropy<avg ∧ vel<avg = focused + stable step)")
            continue

        if low.startswith("vpreccoh3joint"):
            # vpreccoh3joint — precise+coherent joint step meter
            _vcj_res  = getattr(model, "_last_gen_result", None)
            if _vcj_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _vcj_frac = _vcj_res.get("vprec_coh3_joint",    0.0)
            _vcj_vp   = _vcj_res.get("vprec_ema_steps",     [])
            _vcj_c3   = _vcj_res.get("coh3_steps",           [])
            _vcj_n    = min(len(_vcj_vp), len(_vcj_c3))
            if _vcj_n < 4:
                print("  Need ≥4 steps."); continue
            _vcj_vpm  = sum(_vcj_vp[:_vcj_n])  / _vcj_n
            _vcj_c3m  = sum(_vcj_c3[:_vcj_n])  / _vcj_n
            _vcj_flags = [
                ("✓" if (_vcj_vp[i] > _vcj_vpm and _vcj_c3[i] > _vcj_c3m) else "─")
                for i in range(_vcj_n)
            ]
            _vcj_bar   = "".join(_vcj_flags)
            _vcj_label = ("excellent" if _vcj_frac > 0.50 else
                          ("good" if _vcj_frac > 0.35 else
                           ("fair" if _vcj_frac > 0.20 else "low")))
            print(f"\n  Vprec+coh3 joint meter  ({_vcj_n} steps)  {_vcj_frac:.1%}  ({_vcj_label}):")
            print(f"  {_vcj_bar}")
            print(f"  (✓ = vprec>avg ∧ coh3>avg = precise + coherent step)")
            continue

        if low.startswith("confveljoint"):
            # confveljoint — confident+stable joint step meter
            _cvj_res  = getattr(model, "_last_gen_result", None)
            if _cvj_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cvj_frac = _cvj_res.get("conf_vel_joint",   0.0)
            _cvj_cf   = _cvj_res.get("confidences",      [])
            _cvj_vel  = _cvj_res.get("velocity_steps",   [])
            _cvj_n    = min(len(_cvj_cf), len(_cvj_vel))
            if _cvj_n < 4:
                print("  Need ≥4 steps."); continue
            _cvj_cfm  = sum(_cvj_cf[:_cvj_n])  / _cvj_n
            _cvj_vm   = sum(_cvj_vel[:_cvj_n]) / _cvj_n
            _cvj_flags = [
                ("✓" if (_cvj_cf[i] > _cvj_cfm and _cvj_vel[i] < _cvj_vm) else "─")
                for i in range(_cvj_n)
            ]
            _cvj_bar   = "".join(_cvj_flags)
            _cvj_label = ("excellent" if _cvj_frac > 0.50 else
                          ("good" if _cvj_frac > 0.35 else
                           ("fair" if _cvj_frac > 0.20 else "low")))
            print(f"\n  Conf+vel joint meter  ({_cvj_n} steps)  {_cvj_frac:.1%}  ({_cvj_label}):")
            print(f"  {_cvj_bar}")
            print(f"  (✓ = conf>avg ∧ vel<avg = confident + stable step)")
            continue

        if low.startswith("coh3marginjoint"):
            # coh3marginjoint — coherent+decisive joint step meter
            _cmj_res  = getattr(model, "_last_gen_result", None)
            if _cmj_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cmj_frac = _cmj_res.get("coh3_margin_joint",  0.0)
            _cmj_c3   = _cmj_res.get("coh3_steps",          [])
            _cmj_mg   = _cmj_res.get("top1_margins",         [])
            _cmj_n    = min(len(_cmj_c3), len(_cmj_mg))
            if _cmj_n < 4:
                print("  Need ≥4 steps."); continue
            _cmj_c3m  = sum(_cmj_c3[:_cmj_n])  / _cmj_n
            _cmj_mm   = sum(_cmj_mg[:_cmj_n])  / _cmj_n
            _cmj_flags = [
                ("✓" if (_cmj_c3[i] > _cmj_c3m and _cmj_mg[i] > _cmj_mm) else "─")
                for i in range(_cmj_n)
            ]
            _cmj_bar   = "".join(_cmj_flags)
            _cmj_label = ("excellent" if _cmj_frac > 0.50 else
                          ("good" if _cmj_frac > 0.35 else
                           ("fair" if _cmj_frac > 0.20 else "low")))
            print(f"\n  Coh3+margin joint meter  ({_cmj_n} steps)  {_cmj_frac:.1%}  ({_cmj_label}):")
            print(f"  {_cmj_bar}")
            print(f"  (✓ = coh3>avg ∧ margin>avg = coherent + decisive step)")
            continue

        if low.startswith("coh6slopetrend"):
            # coh6slopetrend — coh6 1st + 2nd derivative sparklines
            _c6s_res = getattr(model, "_last_gen_result", None)
            if _c6s_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _c6s_c6  = _c6s_res.get("coh6_steps",        [])
            _c6s_acc = _c6s_res.get("coh6_slope_trend",   0.0)
            if len(_c6s_c6) < 3:
                print("  Need ≥3 coh6 steps (generation ≥6 tokens)."); continue
            _c6s_d1 = [0.0] + [_c6s_c6[i] - _c6s_c6[i-1] for i in range(1, len(_c6s_c6))]
            _c6s_d2 = [0.0, 0.0] + [_c6s_d1[i] - _c6s_d1[i-1] for i in range(2, len(_c6s_d1))]
            _c6s_d1_sp = "".join(
                ("▲" if v > 0.002 else ("▼" if v < -0.002 else "─")) for v in _c6s_d1
            )
            _c6s_d2_sp = "".join(
                ("▲" if v > 0.001 else ("▼" if v < -0.001 else "─")) for v in _c6s_d2
            )
            _c6s_state = ("gaining momentum" if _c6s_acc > 0.001 else
                          ("losing momentum" if _c6s_acc < -0.001 else "steady"))
            print(f"\n  Coh6 slope trend  ({len(_c6s_c6)} steps)  acc={_c6s_acc:+.6f}  ({_c6s_state}):")
            print(f"  1st deriv: {_c6s_d1_sp}")
            print(f"  2nd deriv: {_c6s_d2_sp}")
            print(f"  (▲ in 2nd = 6-window coherence gaining momentum; ▼ = losing)")
            continue

        if low.startswith("marginveljoint"):
            # marginveljoint — decisive+stable joint step meter
            _mvj_res  = getattr(model, "_last_gen_result", None)
            if _mvj_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _mvj_frac = _mvj_res.get("margin_vel_joint",  0.0)
            _mvj_mg   = _mvj_res.get("top1_margins",       [])
            _mvj_vel  = _mvj_res.get("velocity_steps",    [])
            _mvj_n    = min(len(_mvj_mg), len(_mvj_vel))
            if _mvj_n < 4:
                print("  Need ≥4 steps."); continue
            _mvj_mm   = sum(_mvj_mg[:_mvj_n])  / _mvj_n
            _mvj_vm   = sum(_mvj_vel[:_mvj_n]) / _mvj_n
            _mvj_flags = [
                ("✓" if (_mvj_mg[i] > _mvj_mm and _mvj_vel[i] < _mvj_vm) else "─")
                for i in range(_mvj_n)
            ]
            _mvj_bar   = "".join(_mvj_flags)
            _mvj_label = ("excellent" if _mvj_frac > 0.50 else
                          ("good" if _mvj_frac > 0.35 else
                           ("fair" if _mvj_frac > 0.20 else "low")))
            print(f"\n  Margin+vel joint meter  ({_mvj_n} steps)  {_mvj_frac:.1%}  ({_mvj_label}):")
            print(f"  {_mvj_bar}")
            print(f"  (✓ = margin>avg ∧ vel<avg = decisive + stable step)")
            continue

        if low.startswith("sgslopetrend"):
            # sgslopetrend — score-gap EMA 1st + 2nd derivative sparklines
            _sgs_res = getattr(model, "_last_gen_result", None)
            if _sgs_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _sgs_sg  = _sgs_res.get("sg_steps",        [])
            _sgs_acc = _sgs_res.get("sg_slope_trend",   0.0)
            if len(_sgs_sg) < 3:
                print("  Need ≥3 score-gap steps."); continue
            _sgs_d1 = [0.0] + [_sgs_sg[i] - _sgs_sg[i-1] for i in range(1, len(_sgs_sg))]
            _sgs_d2 = [0.0, 0.0] + [_sgs_d1[i] - _sgs_d1[i-1] for i in range(2, len(_sgs_d1))]
            _sgs_d1_sp = "".join(
                ("▲" if v > 0.0005 else ("▼" if v < -0.0005 else "─")) for v in _sgs_d1
            )
            _sgs_d2_sp = "".join(
                ("▲" if v > 0.0008 else ("▼" if v < -0.0008 else "─")) for v in _sgs_d2
            )
            _sgs_state = ("sharpening faster" if _sgs_acc > 0.0008 else
                          ("softening" if _sgs_acc < -0.0008 else "steady"))
            print(f"\n  Score-gap slope trend  ({len(_sgs_sg)} steps)  acc={_sgs_acc:+.7f}  ({_sgs_state}):")
            print(f"  1st deriv: {_sgs_d1_sp}")
            print(f"  2nd deriv: {_sgs_d2_sp}")
            print(f"  (▲ in 2nd = competition sharpening faster; ▼ = gap softening)")
            continue

        if low.startswith("vprecslopetrend"):
            # vprecslopetrend — vprec_ema 1st + 2nd derivative sparklines
            _vps_res = getattr(model, "_last_gen_result", None)
            if _vps_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _vps_vp  = _vps_res.get("vprec_ema_steps",   [])
            _vps_acc = _vps_res.get("vprec_slope_trend",  0.0)
            if len(_vps_vp) < 3:
                print("  Need ≥3 vprec_ema steps."); continue
            _vps_d1 = [0.0] + [_vps_vp[i] - _vps_vp[i-1] for i in range(1, len(_vps_vp))]
            _vps_d2 = [0.0, 0.0] + [_vps_d1[i] - _vps_d1[i-1] for i in range(2, len(_vps_d1))]
            _vps_d1_sp = "".join(
                ("▲" if v > 0.001 else ("▼" if v < -0.001 else "─")) for v in _vps_d1
            )
            _vps_d2_sp = "".join(
                ("▲" if v > 0.0008 else ("▼" if v < -0.0008 else "─")) for v in _vps_d2
            )
            _vps_state = ("tightening faster" if _vps_acc > 0.0008 else
                          ("loosening" if _vps_acc < -0.0008 else "steady"))
            print(f"\n  Vprec_EMA slope trend  ({len(_vps_vp)} steps)  acc={_vps_acc:+.7f}  ({_vps_state}):")
            print(f"  1st deriv: {_vps_d1_sp}")
            print(f"  2nd deriv: {_vps_d2_sp}")
            print(f"  (▲ in 2nd = vocab precision tightening faster; ▼ = loosening)")
            continue

        if low.startswith("entropyslopetrend"):
            # entropyslopetrend — entropy 1st + 2nd derivative sparklines
            _est_res = getattr(model, "_last_gen_result", None)
            if _est_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _est_ent = _est_res.get("entropy_steps",     [])
            _est_acc = _est_res.get("entropy_slope_trend", 0.0)
            if len(_est_ent) < 3:
                print("  Need ≥3 entropy steps."); continue
            _est_d1 = [0.0] + [_est_ent[i] - _est_ent[i-1] for i in range(1, len(_est_ent))]
            _est_d2 = [0.0, 0.0] + [_est_d1[i] - _est_d1[i-1] for i in range(2, len(_est_d1))]
            _est_d1_sp = "".join(
                ("▲" if v > 0.0005 else ("▼" if v < -0.0005 else "─")) for v in _est_d1
            )
            _est_d2_sp = "".join(
                ("▲" if v > 0.0008 else ("▼" if v < -0.0008 else "─")) for v in _est_d2
            )
            _est_state = ("spreading faster" if _est_acc > 0.0008 else
                          ("focusing" if _est_acc < -0.0008 else "steady"))
            print(f"\n  Entropy slope trend  ({len(_est_ent)} steps)  acc={_est_acc:+.7f}  ({_est_state}):")
            print(f"  1st deriv: {_est_d1_sp}")
            print(f"  2nd deriv: {_est_d2_sp}")
            print(f"  (▲ in 2nd = distribution spreading faster; ▼ = focusing)")
            continue

        if low.startswith("topkslopetrend"):
            # topkslopetrend — topk 1st + 2nd derivative sparklines
            _tks_res = getattr(model, "_last_gen_result", None)
            if _tks_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _tks_tk  = _tks_res.get("topk_steps",       [])
            _tks_acc = _tks_res.get("topk_slope_trend",  0.0)
            if len(_tks_tk) < 3:
                print("  Need ≥3 topk steps."); continue
            _tks_d1 = [0.0] + [_tks_tk[i] - _tks_tk[i-1] for i in range(1, len(_tks_tk))]
            _tks_d2 = [0.0, 0.0] + [_tks_d1[i] - _tks_d1[i-1] for i in range(2, len(_tks_d1))]
            _tks_d1_sp = "".join(
                ("▲" if v > 0.3 else ("▼" if v < -0.3 else "─")) for v in _tks_d1
            )
            _tks_d2_sp = "".join(
                ("▲" if v > 0.2 else ("▼" if v < -0.2 else "─")) for v in _tks_d2
            )
            _tks_state = ("beam widening faster" if _tks_acc > 0.2 else
                          ("beam narrowing" if _tks_acc < -0.2 else "steady"))
            print(f"\n  TopK slope trend  ({len(_tks_tk)} steps)  acc={_tks_acc:+.6f}  ({_tks_state}):")
            print(f"  1st deriv: {_tks_d1_sp}")
            print(f"  2nd deriv: {_tks_d2_sp}")
            print(f"  (▲ in 2nd = sampling beam widening faster; ▼ = narrowing)")
            continue

        if low.startswith("velslopetrend"):
            # velslopetrend — velocity 1st + 2nd derivative sparklines
            _vst_res = getattr(model, "_last_gen_result", None)
            if _vst_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _vst_vel = _vst_res.get("velocity_steps",   [])
            _vst_acc = _vst_res.get("vel_slope_trend",   0.0)
            if len(_vst_vel) < 3:
                print("  Need ≥3 velocity steps."); continue
            _vst_d1 = [0.0] + [_vst_vel[i] - _vst_vel[i-1] for i in range(1, len(_vst_vel))]
            _vst_d2 = [0.0, 0.0] + [_vst_d1[i] - _vst_d1[i-1] for i in range(2, len(_vst_d1))]
            _vst_d1_sp = "".join(
                ("▲" if v > 0.00003 else ("▼" if v < -0.00003 else "─")) for v in _vst_d1
            )
            _vst_d2_sp = "".join(
                ("▲" if v > 0.00002 else ("▼" if v < -0.00002 else "─")) for v in _vst_d2
            )
            _vst_state = ("accelerating drift" if _vst_acc > 0.00002 else
                          ("settling" if _vst_acc < -0.00002 else "steady"))
            print(f"\n  Velocity slope trend  ({len(_vst_vel)} steps)  acc={_vst_acc:+.7f}  ({_vst_state}):")
            print(f"  1st deriv: {_vst_d1_sp}")
            print(f"  2nd deriv: {_vst_d2_sp}")
            print(f"  (▲ in 2nd = topic drift rate is accelerating)")
            continue

        if low.startswith("marginslopetrend"):
            # marginslopetrend — margin 1st + 2nd derivative sparklines
            _mst_res = getattr(model, "_last_gen_result", None)
            if _mst_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _mst_mg  = _mst_res.get("top1_margins",       [])
            _mst_acc = _mst_res.get("margin_slope_trend",  0.0)
            if len(_mst_mg) < 3:
                print("  Need ≥3 margin steps."); continue
            _mst_d1 = [0.0] + [_mst_mg[i] - _mst_mg[i-1] for i in range(1, len(_mst_mg))]
            _mst_d2 = [0.0, 0.0] + [_mst_d1[i] - _mst_d1[i-1] for i in range(2, len(_mst_d1))]
            _mst_d1_sp = "".join(
                ("▲" if v > 0.003 else ("▼" if v < -0.003 else "─")) for v in _mst_d1
            )
            _mst_d2_sp = "".join(
                ("▲" if v > 0.002 else ("▼" if v < -0.002 else "─")) for v in _mst_d2
            )
            _mst_state = ("decisiveness accelerating" if _mst_acc > 0.002 else
                          ("decisiveness decelerating" if _mst_acc < -0.002 else "steady"))
            print(f"\n  Margin slope trend  ({len(_mst_mg)} steps)  acc={_mst_acc:+.6f}  ({_mst_state}):")
            print(f"  1st deriv: {_mst_d1_sp}")
            print(f"  2nd deriv: {_mst_d2_sp}")
            print(f"  (▲ in 2nd = model becoming more decisive; ▼ = losing decisiveness)")
            continue

        if low.startswith("coh3slopetrend"):
            # coh3slopetrend — coh3 1st + 2nd derivative sparklines
            _c3st_res = getattr(model, "_last_gen_result", None)
            if _c3st_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _c3st_c3  = _c3st_res.get("coh3_steps",       [])
            _c3st_acc = _c3st_res.get("coh3_slope_trend",  0.0)
            if len(_c3st_c3) < 3:
                print("  Need ≥3 coh3 steps."); continue
            _c3st_d1 = [0.0] + [_c3st_c3[i] - _c3st_c3[i-1] for i in range(1, len(_c3st_c3))]
            _c3st_d2 = [0.0, 0.0] + [_c3st_d1[i] - _c3st_d1[i-1] for i in range(2, len(_c3st_d1))]
            _c3st_d1_sp = "".join(
                ("▲" if v > 0.002 else ("▼" if v < -0.002 else "─")) for v in _c3st_d1
            )
            _c3st_d2_sp = "".join(
                ("▲" if v > 0.001 else ("▼" if v < -0.001 else "─")) for v in _c3st_d2
            )
            _c3st_state = ("gaining momentum" if _c3st_acc > 0.001 else
                           ("losing momentum" if _c3st_acc < -0.001 else "steady"))
            print(f"\n  Coh3 slope trend  ({len(_c3st_c3)} steps)  acc={_c3st_acc:+.6f}  ({_c3st_state}):")
            print(f"  1st deriv: {_c3st_d1_sp}")
            print(f"  2nd deriv: {_c3st_d2_sp}")
            print(f"  (▲ in 2nd = coherence gaining momentum; ▼ = losing)")
            continue

        if low.startswith("confslopetrend"):
            # confslopetrend — conf_ema 1st + 2nd derivative sparklines
            _cst_res = getattr(model, "_last_gen_result", None)
            if _cst_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cst_cf  = _cst_res.get("conf_ema_steps",    [])
            _cst_acc = _cst_res.get("conf_slope_trend",   0.0)
            if len(_cst_cf) < 3:
                print("  Need ≥3 conf_ema steps."); continue
            _cst_d1 = [0.0] + [_cst_cf[i] - _cst_cf[i-1] for i in range(1, len(_cst_cf))]
            _cst_d2 = [0.0, 0.0] + [_cst_d1[i] - _cst_d1[i-1] for i in range(2, len(_cst_d1))]
            _cst_d1_sp = "".join(
                ("▲" if v > 0.001 else ("▼" if v < -0.001 else "─")) for v in _cst_d1
            )
            _cst_d2_sp = "".join(
                ("▲" if v > 0.0005 else ("▼" if v < -0.0005 else "─")) for v in _cst_d2
            )
            _cst_state = ("accelerating" if _cst_acc > 0.0005 else
                          ("decelerating" if _cst_acc < -0.0005 else "steady"))
            print(f"\n  Conf_EMA slope trend  ({len(_cst_cf)} steps)  acc={_cst_acc:+.6f}  ({_cst_state}):")
            print(f"  1st deriv: {_cst_d1_sp}")
            print(f"  2nd deriv: {_cst_d2_sp}")
            print(f"  (▲ in 2nd = confidence accelerating; ▼ = decelerating)")
            continue

        if low.startswith("healthscore"):
            # healthscore — composite health score breakdown (0-4)
            _hs_res = getattr(model, "_last_gen_result", None)
            if _hs_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _hs_sc  = _hs_res.get("conf_margin_vel_score", 0)
            _hs_mc  = _hs_res.get("margin_conf_corr",      0.0)
            _hs_cvc = _hs_res.get("conf_vel_corr",         0.0)
            _hs_vec = _hs_res.get("vprec_entropy_corr",    0.0)
            _hs_csg = _hs_res.get("coh3_sg_corr",          0.0)
            _hs_jt  = _hs_res.get("coh3_vel_conf_joint",   0.0)
            _hs_label = ("excellent" if _hs_sc == 4 else
                         ("good" if _hs_sc == 3 else
                          ("fair" if _hs_sc == 2 else
                           ("weak" if _hs_sc == 1 else "poor"))))
            print(f"\n  Health score: {_hs_sc}/4  ({_hs_label})")
            print(f"  Joint ideal (coh3>μ ∧ vel<μ ∧ conf>μ): {_hs_jt:.1%} of steps")
            print(f"  Component breakdown:")
            print(f"    margin↑∧conf↑   r={_hs_mc:+.4f}  {'✓' if _hs_mc > 0.20 else '✗'}  (decisive=confident)")
            print(f"    conf↑∧vel↓      r={_hs_cvc:+.4f}  {'✓' if _hs_cvc < -0.20 else '✗'}  (low-move=confident)")
            print(f"    vprec↑∧ent↓     r={_hs_vec:+.4f}  {'✓' if _hs_vec < -0.20 else '✗'}  (precise=focused)")
            print(f"    coh3↑∧sgap↑     r={_hs_csg:+.4f}  {'✓' if _hs_csg > 0.20 else '✗'}  (coherent=decisive)")
            continue

        if low.startswith("jointidealmeter"):
            # jointidealmeter — fraction of steps where coh3+vel+conf all ideal
            _jm_res = getattr(model, "_last_gen_result", None)
            if _jm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _jm_frac = _jm_res.get("coh3_vel_conf_joint", 0.0)
            _jm_c3   = _jm_res.get("coh3_steps",    [])
            _jm_vel  = _jm_res.get("velocity_steps", [])
            _jm_cf   = _jm_res.get("confidences",   [])
            _jm_n    = min(len(_jm_c3), len(_jm_vel), len(_jm_cf))
            if _jm_n < 4:
                print("  Need ≥4 steps."); continue
            _jm_c3m  = sum(_jm_c3[:_jm_n])  / _jm_n
            _jm_velm = sum(_jm_vel[:_jm_n]) / _jm_n
            _jm_cfm  = sum(_jm_cf[:_jm_n])  / _jm_n
            _jm_flags = [
                ("✓" if (_jm_c3[i] > _jm_c3m and _jm_vel[i] < _jm_velm and _jm_cf[i] > _jm_cfm)
                 else "─")
                for i in range(_jm_n)
            ]
            _jm_bar = "".join(_jm_flags)
            _jm_label = ("excellent" if _jm_frac > 0.50 else
                         ("good" if _jm_frac > 0.35 else
                          ("fair" if _jm_frac > 0.20 else "low")))
            print(f"\n  Joint ideal meter  ({_jm_n} steps)  {_jm_frac:.1%}  ({_jm_label}):")
            print(f"  {_jm_bar}")
            print(f"  (✓ = coh3>avg ∧ vel<avg ∧ conf>avg simultaneously = ideal step)")
            continue

        if low.startswith("vprecedntropycorr") or low.startswith("vprecedntropy") or low.startswith("vprec_entropy") or low == "vprecedntropycorr":
            low = "vprecedntropycorr"
            pass
        if low.startswith("vprecedntropycorr"):
            # vprecedntropycorr — vprec_ema vs entropy sparklines + Pearson r
            _vec_res = getattr(model, "_last_gen_result", None)
            if _vec_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _vec_vp  = _vec_res.get("vprec_ema_steps",    [])
            _vec_ent = _vec_res.get("entropy_steps",      [])
            _vec_r   = _vec_res.get("vprec_entropy_corr", 0.0)
            _vec_n   = min(len(_vec_vp), len(_vec_ent))
            if _vec_n < 3:
                print("  Need ≥3 steps of both vprec_ema and entropy."); continue
            _vec_vp  = _vec_vp[:_vec_n]
            _vec_ent = _vec_ent[:_vec_n]
            _SPARKS_VEC = " ▁▂▃▄▅▆▇█"
            _vec_v_max = max(max(_vec_vp), 1e-9)
            _vec_e_max = max(max(_vec_ent), 1e-9)
            _vec_v_sp = "".join(
                _SPARKS_VEC[min(8, int(max(v, 0) / _vec_v_max * 8))] for v in _vec_vp
            )
            _vec_e_sp = "".join(
                _SPARKS_VEC[min(8, int(max(v, 0) / _vec_e_max * 8))] for v in _vec_ent
            )
            _vec_interp = ("precise = entropic (unusual)" if _vec_r > 0.40 else
                           ("precise = focused (healthy)" if _vec_r < -0.40 else "decoupled"))
            print(f"\n  VPrec_EMA vs Entropy  (r={_vec_r:+.4f}  → {_vec_interp}):")
            print(f"  VPrec: {_vec_v_sp}  avg={sum(_vec_vp)/_vec_n:.4f}")
            print(f"  Ent:   {_vec_e_sp}  avg={sum(_vec_ent)/_vec_n:.4f}")
            print(f"  (negative r = when model is vocab-precise it is also low-entropy = good)")
            continue

        if low.startswith("coh3entropyslopecorr"):
            # coh3entropyslopecorr — coh3 slope vs entropy slope scatter + Pearson r
            _c3es_res = getattr(model, "_last_gen_result", None)
            if _c3es_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _c3es_c3  = _c3es_res.get("coh3_steps",              [])
            _c3es_ent = _c3es_res.get("entropy_steps",           [])
            _c3es_r   = _c3es_res.get("coh3_entropy_slope_corr", 0.0)
            _c3es_n   = min(len(_c3es_c3), len(_c3es_ent))
            if _c3es_n < 4:
                print("  Need ≥4 steps of both coh3 and entropy."); continue
            _c3es_c3_sl  = [0.0] + [_c3es_c3[i]  - _c3es_c3[i-1]  for i in range(1, _c3es_n)]
            _c3es_ent_sl = [0.0] + [_c3es_ent[i] - _c3es_ent[i-1] for i in range(1, _c3es_n)]
            _c3es_c_sp = "".join(
                ("▲" if v > 0.002 else ("▼" if v < -0.002 else "─"))
                for v in _c3es_c3_sl
            )
            _c3es_e_sp = "".join(
                ("▲" if v > 0.015 else ("▼" if v < -0.015 else "─"))
                for v in _c3es_ent_sl
            )
            _c3es_interp = ("coherence rises with entropy (unusual)" if _c3es_r > 0.35 else
                            ("coherence rises as entropy falls (healthy)" if _c3es_r < -0.35
                             else "decoupled"))
            print(f"\n  Coh3 slope vs Entropy slope  (r={_c3es_r:+.4f}  → {_c3es_interp}):")
            print(f"  Coh3Δ:  {_c3es_c_sp}")
            print(f"  EntΔ:   {_c3es_e_sp}")
            print(f"  (negative r = coherence and entropy move in opposite directions = ideal)")
            continue

        if low.startswith("coh3marginslopecorr"):
            # coh3marginslopecorr — coh3 slope vs margin slope scatter + Pearson r
            _c3ms_res = getattr(model, "_last_gen_result", None)
            if _c3ms_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _c3ms_c3  = _c3ms_res.get("coh3_steps",           [])
            _c3ms_mg  = _c3ms_res.get("top1_margins",         [])
            _c3ms_r   = _c3ms_res.get("coh3_margin_slope_corr", 0.0)
            _c3ms_n   = min(len(_c3ms_c3), len(_c3ms_mg))
            if _c3ms_n < 4:
                print("  Need ≥4 steps of both coh3 and top1_margins."); continue
            _c3ms_c3_sl = [0.0] + [_c3ms_c3[i] - _c3ms_c3[i-1] for i in range(1, _c3ms_n)]
            _c3ms_mg_sl = [0.0] + [_c3ms_mg[i] - _c3ms_mg[i-1] for i in range(1, _c3ms_n)]
            _c3ms_c_sp = "".join(
                ("▲" if v > 0.002 else ("▼" if v < -0.002 else "─"))
                for v in _c3ms_c3_sl
            )
            _c3ms_m_sp = "".join(
                ("▲" if v > 0.003 else ("▼" if v < -0.003 else "─"))
                for v in _c3ms_mg_sl
            )
            _c3ms_interp = ("coherence + margin move together" if _c3ms_r > 0.35 else
                            ("coherence rises as margin falls (unusual)" if _c3ms_r < -0.35
                             else "decoupled"))
            print(f"\n  Coh3 slope vs Margin slope  (r={_c3ms_r:+.4f}  → {_c3ms_interp}):")
            print(f"  Coh3Δ:  {_c3ms_c_sp}")
            print(f"  MgnΔ:   {_c3ms_m_sp}")
            print(f"  (positive r = coherence and score decisiveness evolve in sync)")
            continue

        if low.startswith("entropytopkcorr"):
            # entropytopkcorr — entropy vs TopK sparklines + Pearson r
            _etk_res = getattr(model, "_last_gen_result", None)
            if _etk_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _etk_ent = _etk_res.get("entropy_steps", [])
            _etk_tk  = _etk_res.get("topk_steps",    [])
            _etk_r   = _etk_res.get("entropy_topk_corr", 0.0)
            _etk_n   = min(len(_etk_ent), len(_etk_tk))
            if _etk_n < 3:
                print("  Need ≥3 steps of both entropy and topk."); continue
            _etk_ent = _etk_ent[:_etk_n]
            _etk_tk  = _etk_tk[:_etk_n]
            _SPARKS_ETK = " ▁▂▃▄▅▆▇█"
            _etk_e_max = max(max(_etk_ent), 1e-9)
            _etk_t_max = max(max(_etk_tk), 1e-9)
            _etk_e_sp = "".join(
                _SPARKS_ETK[min(8, int(max(v, 0) / _etk_e_max * 8))] for v in _etk_ent
            )
            _etk_t_sp = "".join(
                _SPARKS_ETK[min(8, int(max(v, 0) / _etk_t_max * 8))] for v in _etk_tk
            )
            _etk_interp = ("entropy widens beam (expected)" if _etk_r > 0.35 else
                           ("high entropy + narrow beam (tension)" if _etk_r < -0.35
                            else "decoupled"))
            print(f"\n  Entropy vs TopK  (r={_etk_r:+.4f}  → {_etk_interp}):")
            print(f"  Ent:  {_etk_e_sp}  avg={sum(_etk_ent)/_etk_n:.4f}")
            print(f"  TopK: {_etk_t_sp}  avg={sum(_etk_tk)/_etk_n:.1f}")
            print(f"  (positive r = higher entropy → wider beam = healthy adaptive response)")
            continue

        if low.startswith("marginconfcorr"):
            # marginconfcorr — top1 margin vs confidence sparklines + Pearson r
            _mcc_res = getattr(model, "_last_gen_result", None)
            if _mcc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _mcc_mg  = _mcc_res.get("top1_margins",    [])
            _mcc_cf  = _mcc_res.get("confidences",     [])
            _mcc_r   = _mcc_res.get("margin_conf_corr", 0.0)
            _mcc_n   = min(len(_mcc_mg), len(_mcc_cf))
            if _mcc_n < 3:
                print("  Need ≥3 steps of both top1_margins and confidence."); continue
            _mcc_mg  = _mcc_mg[:_mcc_n]
            _mcc_cf  = _mcc_cf[:_mcc_n]
            _SPARKS_MCC = " ▁▂▃▄▅▆▇█"
            _mcc_m_max = max(max(_mcc_mg), 1e-9)
            _mcc_c_max = max(max(_mcc_cf), 1e-9)
            _mcc_m_sp = "".join(
                _SPARKS_MCC[min(8, int(max(v, 0) / _mcc_m_max * 8))] for v in _mcc_mg
            )
            _mcc_c_sp = "".join(
                _SPARKS_MCC[min(8, int(max(v, 0) / _mcc_c_max * 8))] for v in _mcc_cf
            )
            _mcc_interp = ("decisive = confident (healthy)" if _mcc_r > 0.35 else
                           ("decisive = unconfident (tension)" if _mcc_r < -0.35
                            else "decoupled"))
            print(f"\n  Margin vs Confidence  (r={_mcc_r:+.4f}  → {_mcc_interp}):")
            print(f"  Margin: {_mcc_m_sp}  avg={sum(_mcc_mg)/_mcc_n:.4f}")
            print(f"  Conf:   {_mcc_c_sp}  avg={sum(_mcc_cf)/_mcc_n:.4f}")
            print(f"  (positive r = wide score margin correlates with higher confidence)")
            continue

        if low.startswith("entropyvelcorr"):
            # entropyvelcorr — entropy vs velocity sparklines + Pearson r
            _evc_res = getattr(model, "_last_gen_result", None)
            if _evc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _evc_ent = _evc_res.get("entropy_steps",   [])
            _evc_vel = _evc_res.get("velocity_steps",  [])
            _evc_r   = _evc_res.get("entropy_vel_corr", 0.0)
            _evc_n   = min(len(_evc_ent), len(_evc_vel))
            if _evc_n < 3:
                print("  Need ≥3 steps of both entropy and velocity."); continue
            _evc_ent = _evc_ent[:_evc_n]
            _evc_vel = _evc_vel[:_evc_n]
            _SPARKS_EVC = " ▁▂▃▄▅▆▇█"
            _evc_e_max = max(max(_evc_ent), 1e-9)
            _evc_v_max = max(max(_evc_vel), 1e-9)
            _evc_e_sp = "".join(
                _SPARKS_EVC[min(8, int(max(v, 0) / _evc_e_max * 8))] for v in _evc_ent
            )
            _evc_v_sp = "".join(
                _SPARKS_EVC[min(8, int(max(v, 0) / _evc_v_max * 8))] for v in _evc_vel
            )
            _evc_interp = ("entropy drives movement" if _evc_r > 0.35 else
                           ("entropic but static (unusual)" if _evc_r < -0.35
                            else "decoupled"))
            print(f"\n  Entropy vs Velocity  (r={_evc_r:+.4f}  → {_evc_interp}):")
            print(f"  Ent:  {_evc_e_sp}  avg={sum(_evc_ent)/_evc_n:.4f}")
            print(f"  Vel:  {_evc_v_sp}  avg={sum(_evc_vel)/_evc_n:.5f}")
            print(f"  (positive r = high entropy correlates with fast topic movement)")
            continue

        if low.startswith("coh6confcorr"):
            # coh6confcorr — 6-gram coherence vs confidence sparklines + Pearson r
            _c6cc_res = getattr(model, "_last_gen_result", None)
            if _c6cc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _c6cc_c6  = _c6cc_res.get("coh6_steps",    [])
            _c6cc_cf  = _c6cc_res.get("confidences",   [])
            _c6cc_r   = _c6cc_res.get("coh6_conf_corr", 0.0)
            _c6cc_n   = min(len(_c6cc_c6), len(_c6cc_cf))
            if _c6cc_n < 3:
                print("  Need ≥3 steps of both coh6 and confidence."); continue
            _c6cc_c6  = _c6cc_c6[:_c6cc_n]
            _c6cc_cf  = _c6cc_cf[:_c6cc_n]
            _SPARKS_C6CC = " ▁▂▃▄▅▆▇█"
            _c6cc_c6_max = max(max(_c6cc_c6), 1e-9)
            _c6cc_cf_max = max(max(_c6cc_cf), 1e-9)
            _c6cc_c6_sp = "".join(
                _SPARKS_C6CC[min(8, int(max(v, 0) / _c6cc_c6_max * 8))] for v in _c6cc_c6
            )
            _c6cc_cf_sp = "".join(
                _SPARKS_C6CC[min(8, int(max(v, 0) / _c6cc_cf_max * 8))] for v in _c6cc_cf
            )
            _c6cc_interp = ("long-range coherent + confident" if _c6cc_r > 0.35 else
                            ("coherent but uncertain (structured hedging)" if _c6cc_r < -0.35
                             else "decoupled"))
            print(f"\n  Coh6 vs Confidence  (r={_c6cc_r:+.4f}  → {_c6cc_interp}):")
            print(f"  Coh6: {_c6cc_c6_sp}  avg={sum(_c6cc_c6)/_c6cc_n:.4f}")
            print(f"  Conf: {_c6cc_cf_sp}  avg={sum(_c6cc_cf)/_c6cc_n:.4f}")
            print(f"  (positive r = long-range coherence and token confidence aligned)")
            continue

        if low.startswith("topkvelcorr"):
            # topkvelcorr — TopK k vs velocity sparklines + Pearson r
            _tkvc_res = getattr(model, "_last_gen_result", None)
            if _tkvc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _tkvc_tk  = _tkvc_res.get("topk_steps",    [])
            _tkvc_vel = _tkvc_res.get("velocity_steps", [])
            _tkvc_r   = _tkvc_res.get("topk_vel_corr",  0.0)
            _tkvc_n   = min(len(_tkvc_tk), len(_tkvc_vel))
            if _tkvc_n < 3:
                print("  Need ≥3 steps of both topk and velocity."); continue
            _tkvc_tk  = _tkvc_tk[:_tkvc_n]
            _tkvc_vel = _tkvc_vel[:_tkvc_n]
            _SPARKS_TKVC = " ▁▂▃▄▅▆▇█"
            _tkvc_tk_max  = max(max(_tkvc_tk), 1e-9)
            _tkvc_vel_max = max(max(_tkvc_vel), 1e-9)
            _tkvc_tk_sp  = "".join(
                _SPARKS_TKVC[min(8, int(max(v, 0) / _tkvc_tk_max * 8))] for v in _tkvc_tk
            )
            _tkvc_vel_sp = "".join(
                _SPARKS_TKVC[min(8, int(max(v, 0) / _tkvc_vel_max * 8))] for v in _tkvc_vel
            )
            _tkvc_interp = ("active exploration (wide beam + fast movement)" if _tkvc_r > 0.35 else
                            ("tension (narrow beam + fast movement)" if _tkvc_r < -0.35
                             else "decoupled"))
            print(f"\n  TopK vs Velocity  (r={_tkvc_r:+.4f}  → {_tkvc_interp}):")
            print(f"  TopK: {_tkvc_tk_sp}  avg={sum(_tkvc_tk)/_tkvc_n:.1f}")
            print(f"  Vel:  {_tkvc_vel_sp}  avg={sum(_tkvc_vel)/_tkvc_n:.5f}")
            print(f"  (positive r = wide beam during fast topic movement = active exploration)")
            continue

        if low.startswith("confvelcorr"):
            # confvelcorr — confidence vs velocity sparklines + Pearson r
            _cvc_res = getattr(model, "_last_gen_result", None)
            if _cvc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cvc_cf  = _cvc_res.get("confidences",    [])
            _cvc_vel = _cvc_res.get("velocity_steps", [])
            _cvc_r   = _cvc_res.get("conf_vel_corr",  0.0)
            _cvc_n   = min(len(_cvc_cf), len(_cvc_vel))
            if _cvc_n < 3:
                print("  Need ≥3 steps of both confidence and velocity."); continue
            _cvc_cf  = _cvc_cf[:_cvc_n]
            _cvc_vel = _cvc_vel[:_cvc_n]
            _SPARKS_CVC = " ▁▂▃▄▅▆▇█"
            _cvc_c_max = max(max(_cvc_cf), 1e-9)
            _cvc_v_max = max(max(_cvc_vel), 1e-9)
            _cvc_c_sp = "".join(
                _SPARKS_CVC[min(8, int(max(v, 0) / _cvc_c_max * 8))] for v in _cvc_cf
            )
            _cvc_v_sp = "".join(
                _SPARKS_CVC[min(8, int(max(v, 0) / _cvc_v_max * 8))] for v in _cvc_vel
            )
            _cvc_interp = ("overconfident drift (unusual)" if _cvc_r > 0.40 else
                           ("healthy exploration" if _cvc_r < -0.40 else "decoupled"))
            print(f"\n  Confidence vs Velocity  (r={_cvc_r:+.4f}  → {_cvc_interp}):")
            print(f"  Conf: {_cvc_c_sp}  avg={sum(_cvc_cf)/_cvc_n:.4f}")
            print(f"  Vel:  {_cvc_v_sp}  avg={sum(_cvc_vel)/_cvc_n:.5f}")
            print(f"  (negative r = fast topic movement = lower conf = healthy exploration)")
            continue

        if low.startswith("topkvarplot"):
            # topkvarplot — running TopK k variance sparkline
            _tkv_res = getattr(model, "_last_gen_result", None)
            if _tkv_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _tkv_tv  = _tkv_res.get("topk_var_steps", [])
            _tkv_tk  = _tkv_res.get("topk_steps",     [])
            if not _tkv_tv and _tkv_tk:
                import numpy as _np_tkvp
                _tkv_tv = [
                    float(_np_tkvp.var(_tkv_tk[:i+1])) if i >= 1 else 0.0
                    for i in range(len(_tkv_tk))
                ]
            if not _tkv_tv:
                print("  No TopK data."); continue
            import numpy as _np_tkv
            _tkv_arr = _np_tkv.array(_tkv_tv, dtype=float)
            _tkv_max = max(float(_tkv_arr.max()), 1e-9)
            _SPARKS_TKV = " ▁▂▃▄▅▆▇█"
            _tkv_spark = "".join(
                _SPARKS_TKV[min(8, int(max(v, 0) / _tkv_max * 8))]
                for v in _tkv_tv
            )
            _tkv_interp = ("erratic beam width" if _tkv_arr[-1] > 50 else
                           ("stable" if _tkv_arr[-1] < 5 else "moderate"))
            print(f"\n  TopK variance  ({len(_tkv_tv)} steps)  final: {_tkv_interp}:")
            print(f"  final={_tkv_arr[-1]:.2f}  max={_tkv_arr.max():.2f}  avg={_tkv_arr.mean():.2f}")
            print(f"  {_tkv_spark}")
            print(f"  (rising = adaptive TopK is swinging widely; flat = stable beam)")
            continue

        if low.startswith("velvarplot"):
            # velvarplot — running velocity variance sparkline
            _vvp_res = getattr(model, "_last_gen_result", None)
            if _vvp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _vvp_vv  = _vvp_res.get("vel_var_steps",   [])
            _vvp_vel = _vvp_res.get("velocity_steps",  [])
            if not _vvp_vv and _vvp_vel:
                import numpy as _np_vvpc
                _vvp_vv = [
                    float(_np_vvpc.var(_vvp_vel[:i+1])) if i >= 1 else 0.0
                    for i in range(len(_vvp_vel))
                ]
            if not _vvp_vv:
                print("  No velocity data."); continue
            import numpy as _np_vvp
            _vvp_arr = _np_vvp.array(_vvp_vv, dtype=float)
            _vvp_max = max(float(_vvp_arr.max()), 1e-9)
            _SPARKS_VVP = " ▁▂▃▄▅▆▇█"
            _vvp_spark = "".join(
                _SPARKS_VVP[min(8, int(max(v, 0) / _vvp_max * 8))]
                for v in _vvp_vv
            )
            _vvp_interp = ("erratic topic movement" if _vvp_arr[-1] > 0.0005 else
                           ("stable" if _vvp_arr[-1] < 0.00005 else "moderate"))
            print(f"\n  Velocity variance  ({len(_vvp_vv)} steps)  final: {_vvp_interp}:")
            print(f"  final={_vvp_arr[-1]:.7f}  max={_vvp_arr.max():.7f}  avg={_vvp_arr.mean():.7f}")
            print(f"  {_vvp_spark}")
            print(f"  (rising = topic-drift rate is inconsistent; flat = steady movement)")
            continue

        if low.startswith("coh3vpreccorr"):
            # coh3vpreccorr — coh3 vs vprec_ema sparklines + Pearson r
            _c3vp_res = getattr(model, "_last_gen_result", None)
            if _c3vp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _c3vp_c3  = _c3vp_res.get("coh3_steps",      [])
            _c3vp_vp  = _c3vp_res.get("vprec_ema_steps", [])
            _c3vp_r   = _c3vp_res.get("coh3_vprec_corr", 0.0)
            _c3vp_n   = min(len(_c3vp_c3), len(_c3vp_vp))
            if _c3vp_n < 3:
                print("  Need ≥3 steps of both coh3 and vprec_ema."); continue
            _c3vp_c3  = _c3vp_c3[:_c3vp_n]
            _c3vp_vp  = _c3vp_vp[:_c3vp_n]
            _SPARKS_C3VP = " ▁▂▃▄▅▆▇█"
            _c3vp_c3_max = max(max(_c3vp_c3), 1e-9)
            _c3vp_vp_max = max(max(_c3vp_vp), 1e-9)
            _c3vp_c3_sp = "".join(
                _SPARKS_C3VP[min(8, int(max(v, 0) / _c3vp_c3_max * 8))] for v in _c3vp_c3
            )
            _c3vp_vp_sp = "".join(
                _SPARKS_C3VP[min(8, int(max(v, 0) / _c3vp_vp_max * 8))] for v in _c3vp_vp
            )
            _c3vp_interp = ("coherent = precise (strong signal)" if _c3vp_r > 0.40 else
                            ("coherent but imprecise (unusual)" if _c3vp_r < -0.40
                             else "decoupled"))
            print(f"\n  Coh3 vs VPrec_EMA  (r={_c3vp_r:+.4f}  → {_c3vp_interp}):")
            print(f"  Coh3:  {_c3vp_c3_sp}  avg={sum(_c3vp_c3)/_c3vp_n:.4f}")
            print(f"  VPrec: {_c3vp_vp_sp}  avg={sum(_c3vp_vp)/_c3vp_n:.4f}")
            print(f"  (positive r = coherence and vocab precision rise together = quality run)")
            continue

        if low.startswith("marginvarplot"):
            # marginvarplot — running top1 margin variance sparkline
            _mvp_res = getattr(model, "_last_gen_result", None)
            if _mvp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _mvp_mv  = _mvp_res.get("margin_var_steps", [])
            _mvp_mg  = _mvp_res.get("top1_margins",     [])
            if not _mvp_mv and _mvp_mg:
                import numpy as _np_mvpc
                _mvp_mv = [
                    float(_np_mvpc.var(_mvp_mg[:i+1])) if i >= 1 else 0.0
                    for i in range(len(_mvp_mg))
                ]
            if not _mvp_mv:
                print("  No margin data."); continue
            import numpy as _np_mvp
            _mvp_arr = _np_mvp.array(_mvp_mv, dtype=float)
            _mvp_max = max(float(_mvp_arr.max()), 1e-9)
            _SPARKS_MVP = " ▁▂▃▄▅▆▇█"
            _mvp_spark = "".join(
                _SPARKS_MVP[min(8, int(max(v, 0) / _mvp_max * 8))]
                for v in _mvp_mv
            )
            _mvp_interp = ("erratic decisiveness" if _mvp_arr[-1] > 0.008 else
                           ("stable" if _mvp_arr[-1] < 0.001 else "moderate"))
            print(f"\n  Margin variance  ({len(_mvp_mv)} steps)  final: {_mvp_interp}:")
            print(f"  final={_mvp_arr[-1]:.6f}  max={_mvp_arr.max():.6f}  avg={_mvp_arr.mean():.6f}")
            print(f"  {_mvp_spark}")
            print(f"  (rising = model decisiveness is erratic; flat = consistent)")
            continue

        if low.startswith("coh3sgcorr"):
            # coh3sgcorr — coh3 vs ScoreGapEMA sparklines + Pearson r
            _c3sg_res = getattr(model, "_last_gen_result", None)
            if _c3sg_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _c3sg_c3  = _c3sg_res.get("coh3_steps",   [])
            _c3sg_sg  = _c3sg_res.get("sg_step_gaps", [])
            _c3sg_r   = _c3sg_res.get("coh3_sg_corr", 0.0)
            _c3sg_n   = min(len(_c3sg_c3), len(_c3sg_sg))
            if _c3sg_n < 3:
                print("  Need ≥3 steps of both coh3 and sg_step_gaps."); continue
            _c3sg_c3  = _c3sg_c3[:_c3sg_n]
            _c3sg_sg  = _c3sg_sg[:_c3sg_n]
            _SPARKS_C3SG = " ▁▂▃▄▅▆▇█"
            _c3sg_c3_max = max(max(_c3sg_c3), 1e-9)
            _c3sg_sg_max = max(max(_c3sg_sg), 1e-9)
            _c3sg_c3_sp = "".join(
                _SPARKS_C3SG[min(8, int(max(v, 0) / _c3sg_c3_max * 8))] for v in _c3sg_c3
            )
            _c3sg_sg_sp = "".join(
                _SPARKS_C3SG[min(8, int(max(v, 0) / _c3sg_sg_max * 8))] for v in _c3sg_sg
            )
            _c3sg_interp = ("coherent = decisive (healthy)" if _c3sg_r > 0.35 else
                            ("coherent = indecisive (unusual)" if _c3sg_r < -0.35
                             else "decoupled"))
            print(f"\n  Coh3 vs ScoreGap  (r={_c3sg_r:+.4f}  → {_c3sg_interp}):")
            print(f"  Coh3:  {_c3sg_c3_sp}  avg={sum(_c3sg_c3)/_c3sg_n:.4f}")
            print(f"  SGap:  {_c3sg_sg_sp}  avg={sum(_c3sg_sg)/_c3sg_n:.4f}")
            print(f"  (positive r = higher coherence correlates with wider score gaps)")
            continue

        if low.startswith("qualvarplot"):
            # qualvarplot — running quality variance sparkline
            _qvp_res = getattr(model, "_last_gen_result", None)
            if _qvp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qvp_qv  = _qvp_res.get("qual_var_steps", [])
            _qvp_cc  = _qvp_res.get("coh_conf_steps", [])
            if not _qvp_qv and not _qvp_cc:
                print("  No quality data."); continue
            if not _qvp_qv and _qvp_cc:
                import numpy as _np_qvpc
                _qvp_qv = [
                    float(_np_qvpc.var(_qvp_cc[:i+1])) if i >= 1 else 0.0
                    for i in range(len(_qvp_cc))
                ]
            import numpy as _np_qvp
            _qvp_arr = _np_qvp.array(_qvp_qv, dtype=float)
            _qvp_max = max(float(_qvp_arr.max()), 1e-9)
            _SPARKS_QVP = " ▁▂▃▄▅▆▇█"
            _qvp_spark = "".join(
                _SPARKS_QVP[min(8, int(max(v, 0) / _qvp_max * 8))]
                for v in _qvp_qv
            )
            print(f"\n  Running quality variance  ({len(_qvp_qv)} steps):")
            print(f"  final={_qvp_arr[-1]:.6f}  "
                  f"max={_qvp_arr.max():.6f}  "
                  f"avg={_qvp_arr.mean():.6f}")
            print(f"  {_qvp_spark}")
            print(f"  (rising = quality is inconsistent; flat = consistent quality)")
            continue

        if low.startswith("sgconfcorr"):
            # sgconfcorr — ScoreGapEMA vs confidence sparklines + Pearson r
            _sgcc_res = getattr(model, "_last_gen_result", None)
            if _sgcc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _sgcc_cf  = _sgcc_res.get("confidences",  [])
            _sgcc_sg  = _sgcc_res.get("sg_step_gaps", [])
            _sgcc_r   = _sgcc_res.get("sg_conf_corr", 0.0)
            _sgcc_n   = min(len(_sgcc_cf), len(_sgcc_sg))
            if _sgcc_n < 3:
                print("  Need ≥3 steps of both confidence and sg_step_gaps."); continue
            _sgcc_cf  = _sgcc_cf[:_sgcc_n]
            _sgcc_sg  = _sgcc_sg[:_sgcc_n]
            _SPARKS_SGCC = " ▁▂▃▄▅▆▇█"
            _sgcc_c_max = max(max(_sgcc_cf), 1e-9)
            _sgcc_s_max = max(max(_sgcc_sg), 1e-9)
            _sgcc_c_sp  = "".join(
                _SPARKS_SGCC[min(8, int(max(v, 0) / _sgcc_c_max * 8))] for v in _sgcc_cf
            )
            _sgcc_s_sp  = "".join(
                _SPARKS_SGCC[min(8, int(max(v, 0) / _sgcc_s_max * 8))] for v in _sgcc_sg
            )
            _sgcc_interp = ("decisive = confident (healthy)" if _sgcc_r > 0.35 else
                            ("decisive = less confident (unusual)" if _sgcc_r < -0.35
                             else "decoupled"))
            print(f"\n  ScoreGap vs Confidence  (r={_sgcc_r:+.4f}  → {_sgcc_interp}):")
            print(f"  Conf:  {_sgcc_c_sp}  avg={sum(_sgcc_cf)/_sgcc_n:.4f}")
            print(f"  SGap:  {_sgcc_s_sp}  avg={sum(_sgcc_sg)/_sgcc_n:.4f}")
            print(f"  (positive r = larger score gap correlates with higher confidence)")
            continue

        if low.startswith("qualtrend"):
            # qualtrend — quality_steps sparkline with slope + direction label
            _qt_res = getattr(model, "_last_gen_result", None)
            if _qt_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qt_qs   = _qt_res.get("quality_steps",  [])
            _qt_cc   = _qt_res.get("coh_conf_steps", [])
            _qt_sl   = _qt_res.get("qual_trend",     0.0)
            _qt_data = _qt_qs or _qt_cc
            if not _qt_data:
                print("  No quality data."); continue
            import numpy as _np_qt
            _qt_arr = _np_qt.array(_qt_data, dtype=float)
            _qt_max = max(float(_qt_arr.max()), 1e-9)
            _SPARKS_QT = " ▁▂▃▄▅▆▇█"
            _qt_spark = "".join(
                _SPARKS_QT[min(8, int(max(v, 0) / _qt_max * 8))]
                for v in _qt_data
            )
            _qt_dir = ("improving" if _qt_sl > 0.0001 else
                       ("degrading" if _qt_sl < -0.0001 else "stable"))
            print(f"\n  Quality trend  ({len(_qt_data)} steps)  "
                  f"slope={_qt_sl:+.6f}  ({_qt_dir}):")
            print(f"  avg={_qt_arr.mean():.5f}  "
                  f"max={_qt_arr.max():.5f}  "
                  f"final={_qt_arr[-1]:.5f}")
            print(f"  {_qt_spark}")
            print(f"  (+ slope = coherence+confidence both rising through gen)")
            continue

        if low.startswith("vprecconfslopecorr"):
            # vprecconfslopecorr — vprec slope vs conf_ema slope scatter + Pearson r
            _vpcs_res = getattr(model, "_last_gen_result", None)
            if _vpcs_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _vpcs_vp  = _vpcs_res.get("vprec_ema_steps",       [])
            _vpcs_cf  = _vpcs_res.get("conf_ema_steps",        [])
            _vpcs_r   = _vpcs_res.get("vprec_conf_slope_corr", 0.0)
            _vpcs_n   = min(len(_vpcs_vp), len(_vpcs_cf))
            if _vpcs_n < 3:
                print("  Need ≥3 steps of both vprec_ema and conf_ema."); continue
            _vpcs_vp_sl = [0.0] + [_vpcs_vp[i] - _vpcs_vp[i-1] for i in range(1, _vpcs_n)]
            _vpcs_cf_sl = [0.0] + [_vpcs_cf[i] - _vpcs_cf[i-1] for i in range(1, _vpcs_n)]
            _vpcs_v_sp = "".join(
                ("▲" if v > 0.002 else ("▼" if v < -0.002 else "─"))
                for v in _vpcs_vp_sl
            )
            _vpcs_c_sp = "".join(
                ("▲" if v > 0.001 else ("▼" if v < -0.001 else "─"))
                for v in _vpcs_cf_sl
            )
            _vpcs_interp = ("vocab precision + confidence rise together" if _vpcs_r > 0.40 else
                            ("anti-correlated (gain precision = lose conf)" if _vpcs_r < -0.40
                             else "decoupled"))
            print(f"\n  VPrec slope vs Conf_EMA slope  (r={_vpcs_r:+.4f}  → {_vpcs_interp}):")
            print(f"  VPrec slope: {_vpcs_v_sp}")
            print(f"  Conf slope:  {_vpcs_c_sp}")
            print(f"  (+ r = when model broadens/sharpens vocab, confidence follows)")
            continue

        if low.startswith("coh6varplot"):
            # coh6varplot — running coh6 variance sparkline
            _c6vp_res = getattr(model, "_last_gen_result", None)
            if _c6vp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _c6vp_cv = _c6vp_res.get("coh6_var_steps", [])
            _c6vp_c6 = _c6vp_res.get("coh6_steps",     [])
            if not _c6vp_cv and not _c6vp_c6:
                print("  No coh6 data (needs ≥6 tokens)."); continue
            if not _c6vp_cv and _c6vp_c6:
                import numpy as _np_c6vpc
                _c6vp_cv = [
                    float(_np_c6vpc.var(_c6vp_c6[:i+1])) if i >= 1 else 0.0
                    for i in range(len(_c6vp_c6))
                ]
            import numpy as _np_c6vp
            _c6vp_arr = _np_c6vp.array(_c6vp_cv, dtype=float)
            _c6vp_max = max(float(_c6vp_arr.max()), 1e-9)
            _SPARKS_C6VP = " ▁▂▃▄▅▆▇█"
            _c6vp_spark = "".join(
                _SPARKS_C6VP[min(8, int(max(v, 0) / _c6vp_max * 8))]
                for v in _c6vp_cv
            )
            print(f"\n  Running coh6 variance  ({len(_c6vp_cv)} steps):")
            print(f"  final={_c6vp_arr[-1]:.6f}  "
                  f"max={_c6vp_arr.max():.6f}  "
                  f"avg={_c6vp_arr.mean():.6f}")
            print(f"  {_c6vp_spark}")
            print(f"  (flat = long-range coherence stable; rising = 6-gram context drifting)")
            continue

        if low.startswith("conftopkcorr"):
            # conftopkcorr — confidence vs TopK sparklines + Pearson r
            _ctk_res = getattr(model, "_last_gen_result", None)
            if _ctk_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ctk_cf   = _ctk_res.get("confidences",    [])
            _ctk_tk   = _ctk_res.get("topk_steps",     [])
            _ctk_r    = _ctk_res.get("conf_topk_corr", 0.0)
            _ctk_n    = min(len(_ctk_cf), len(_ctk_tk))
            if _ctk_n < 3:
                print("  Need ≥3 steps of both confidence and topk_steps."); continue
            _ctk_cf = _ctk_cf[:_ctk_n]
            _ctk_tk = _ctk_tk[:_ctk_n]
            _SPARKS_CTK = " ▁▂▃▄▅▆▇█"
            _ctk_c_max = max(max(_ctk_cf), 1e-9)
            _ctk_t_max = max(max(_ctk_tk), 1e-9)
            _ctk_c_sp = "".join(
                _SPARKS_CTK[min(8, int(max(v, 0) / _ctk_c_max * 8))] for v in _ctk_cf
            )
            _ctk_t_sp = "".join(
                _SPARKS_CTK[min(8, int(max(v, 0) / _ctk_t_max * 8))] for v in _ctk_tk
            )
            _ctk_interp = ("wide beam = lower conf (healthy)" if _ctk_r < -0.35 else
                           ("wide beam = higher conf (unusual)" if _ctk_r > 0.35 else "decoupled"))
            print(f"\n  Confidence vs TopK  (r={_ctk_r:+.4f}  → {_ctk_interp}):")
            print(f"  Conf: {_ctk_c_sp}  avg={sum(_ctk_cf)/_ctk_n:.4f}")
            print(f"  TopK: {_ctk_t_sp}  avg={sum(_ctk_tk)/_ctk_n:.1f}")
            print(f"  (negative r = wider candidate beam correlates with lower confidence)")
            continue

        if low.startswith("entropyvarplot"):
            # entropyvarplot — running entropy variance sparkline
            _evp_res = getattr(model, "_last_gen_result", None)
            if _evp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _evp_ev  = _evp_res.get("entropy_var_steps", [])
            _evp_ent = _evp_res.get("entropy_steps",     [])
            if not _evp_ev and not _evp_ent:
                print("  No entropy data."); continue
            if not _evp_ev and _evp_ent:
                import numpy as _np_evpc
                _evp_ev = [
                    float(_np_evpc.var(_evp_ent[:i+1])) if i >= 1 else 0.0
                    for i in range(len(_evp_ent))
                ]
            import numpy as _np_evp
            _evp_arr = _np_evp.array(_evp_ev, dtype=float)
            _evp_max = max(float(_evp_arr.max()), 1e-9)
            _SPARKS_EVP = " ▁▂▃▄▅▆▇█"
            _evp_spark = "".join(
                _SPARKS_EVP[min(8, int(max(v, 0) / _evp_max * 8))]
                for v in _evp_ev
            )
            print(f"\n  Running entropy variance  ({len(_evp_ev)} steps):")
            print(f"  final={_evp_arr[-1]:.6f}  "
                  f"max={_evp_arr.max():.6f}  "
                  f"avg={_evp_arr.mean():.6f}")
            print(f"  {_evp_spark}")
            print(f"  (high/growing = distribution is chaotic; flat = entropy stable)")
            continue

        if low.startswith("sgspikeplot"):
            # sgspikeplot — annotated ScoreGapEMA spike steps
            _sgsp_res = getattr(model, "_last_gen_result", None)
            if _sgsp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _sgsp_spikes = _sgsp_res.get("sg_spike_steps",  [])
            _sgsp_gaps   = _sgsp_res.get("sg_step_gaps",    [])
            _sgsp_toks   = _sgsp_res.get("tokens",          [])
            if not _sgsp_gaps:
                print("  No sg_step_gaps data."); continue
            import numpy as _np_sgsp
            _sgsp_arr = _np_sgsp.array(_sgsp_gaps, dtype=float)
            _sgsp_thr = (float(_sgsp_arr.mean()) + 1.5 * float(_sgsp_arr.std())
                         if len(_sgsp_arr) >= 4 else 0.0)
            print(f"\n  ScoreGapEMA spike steps  (threshold={_sgsp_thr:.5f}  "
                  f"mean={_sgsp_arr.mean():.5f}  σ={_sgsp_arr.std():.5f}):")
            if not _sgsp_spikes:
                print("  No spikes detected.")
            else:
                for _sgi in _sgsp_spikes:
                    _sgt  = _sgsp_toks[_sgi] if _sgi < len(_sgsp_toks) else "?"
                    _sgv  = _sgsp_gaps[_sgi]
                    _sgb  = "◆" * min(6, max(1, int((_sgv - _sgsp_thr) / max(_sgsp_arr.max() - _sgsp_thr, 1e-9) * 6)))
                    print(f"  step {_sgi:<4}  {_sgt[:20]:<20}  gap={_sgv:.5f}  {_sgb}")
            print(f"  ({len(_sgsp_spikes)} spikes in {len(_sgsp_gaps)} steps)")
            continue

        if low.startswith("confentropycorr"):
            # confentropycorr — confidence vs entropy sparklines with Pearson r
            _cfec_res = getattr(model, "_last_gen_result", None)
            if _cfec_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cfec_cf  = _cfec_res.get("confidences",       [])
            _cfec_ent = _cfec_res.get("entropy_steps",     [])
            _cfec_r   = _cfec_res.get("conf_entropy_corr", 0.0)
            _cfec_n   = min(len(_cfec_cf), len(_cfec_ent))
            if _cfec_n < 3:
                print("  Need ≥3 steps of both confidence and entropy."); continue
            _cfec_cf  = _cfec_cf[:_cfec_n]
            _cfec_ent = _cfec_ent[:_cfec_n]
            _SPARKS_CFEC = " ▁▂▃▄▅▆▇█"
            _cfec_c_max = max(max(_cfec_cf), 1e-9)
            _cfec_e_max = max(max(_cfec_ent), 1e-9)
            _cfec_c_sp = "".join(
                _SPARKS_CFEC[min(8, int(max(v, 0) / _cfec_c_max * 8))] for v in _cfec_cf
            )
            _cfec_e_sp = "".join(
                _SPARKS_CFEC[min(8, int(max(v, 0) / _cfec_e_max * 8))] for v in _cfec_ent
            )
            _cfec_interp = ("confident = focused" if _cfec_r < -0.35 else
                            ("confident = chaotic (unusual)" if _cfec_r > 0.35 else "weak coupling"))
            print(f"\n  Confidence vs Entropy correlation  (r={_cfec_r:+.4f}  → {_cfec_interp}):")
            print(f"  Conf:    {_cfec_c_sp}  avg={sum(_cfec_cf)/_cfec_n:.4f}")
            print(f"  Entropy: {_cfec_e_sp}  avg={sum(_cfec_ent)/_cfec_n:.4f}")
            print(f"  (negative r = high confidence tokens are lower-entropy choices)")
            continue

        if low.startswith("cohslopeplot"):
            # cohslopeplot — coh3 slope scatter ▲/─/▼ per step
            _csp_res = getattr(model, "_last_gen_result", None)
            if _csp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _csp_sl = _csp_res.get("coh_slope_steps", [])
            _csp_c3 = _csp_res.get("coh3_steps",      [])
            if not _csp_sl and not _csp_c3:
                print("  No coh3 data."); continue
            if not _csp_sl and _csp_c3:
                _csp_sl = [0.0] + [round(_csp_c3[i] - _csp_c3[i-1], 6)
                                   for i in range(1, len(_csp_c3))]
            _csp_scatter = "".join(
                ("▲" if v > 0.003 else ("▼" if v < -0.003 else "─"))
                for v in _csp_sl
            )
            _csp_rises = sum(1 for v in _csp_sl if v > 0.003)
            _csp_drops = sum(1 for v in _csp_sl if v < -0.003)
            _csp_flat  = len(_csp_sl) - _csp_rises - _csp_drops
            _csp_net   = (_csp_c3[-1] - _csp_c3[0]) if _csp_c3 else 0.0
            print(f"\n  Coh3 slope scatter  ({len(_csp_sl)} steps):")
            print(f"  {_csp_scatter}")
            print(f"  ▲ rises={_csp_rises} ({100*_csp_rises/max(len(_csp_sl),1):.1f}%)  "
                  f"▼ drops={_csp_drops} ({100*_csp_drops/max(len(_csp_sl),1):.1f}%)  "
                  f"─ flat={_csp_flat}")
            print(f"  net Δcoh3={_csp_net:+.5f}  "
                  f"(start={_csp_c3[0] if _csp_c3 else 0.0:.4f}  "
                  f"end={_csp_c3[-1] if _csp_c3 else 0.0:.4f})")
            continue

        if low.startswith("marginspikeplot"):
            # marginspikeplot — annotated list of steps where margin spiked
            _msp_res = getattr(model, "_last_gen_result", None)
            if _msp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _msp_spikes = _msp_res.get("margin_spike_steps", [])
            _msp_margins = _msp_res.get("top1_margins", [])
            _msp_toks   = _msp_res.get("tokens", [])
            if not _msp_margins:
                print("  No top1_margins data."); continue
            import numpy as _np_msp
            _msp_arr = _np_msp.array(_msp_margins, dtype=float)
            _msp_thr = (float(_msp_arr.mean()) + 1.5 * float(_msp_arr.std())
                        if len(_msp_arr) >= 4 else 0.0)
            print(f"\n  Margin spike steps  (threshold={_msp_thr:.5f}  "
                  f"mean={_msp_arr.mean():.5f}  σ={_msp_arr.std():.5f}):")
            if not _msp_spikes:
                print("  No spikes detected.")
            else:
                for _msi in _msp_spikes:
                    _mst = _msp_toks[_msi] if _msi < len(_msp_toks) else "?"
                    _msv = _msp_margins[_msi]
                    _bar = "▲" * min(8, max(1, int((_msv - _msp_thr) / max(_msp_arr.max() - _msp_thr, 1e-9) * 8)))
                    print(f"  step {_msi:<4}  {_mst[:20]:<20}  +{_msv:.5f}  {_bar}")
            print(f"  ({len(_msp_spikes)} spikes in {len(_msp_margins)} steps)")
            continue

        if low.startswith("cohvarplot"):
            # cohvarplot — running coh3 variance sparkline
            _cvhp_res = getattr(model, "_last_gen_result", None)
            if _cvhp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cvhp_cv  = _cvhp_res.get("coh_var_steps", [])
            _cvhp_c3  = _cvhp_res.get("coh3_steps",    [])
            if not _cvhp_cv and not _cvhp_c3:
                print("  No coh3 data."); continue
            if not _cvhp_cv:
                import numpy as _np_cvhpc
                _cvhp_cv = [
                    float(_np_cvhpc.var(_cvhp_c3[:i+1])) if i >= 1 else 0.0
                    for i in range(len(_cvhp_c3))
                ]
            import numpy as _np_cvhp
            _cvhp_arr = _np_cvhp.array(_cvhp_cv, dtype=float)
            _cvhp_max = max(float(_cvhp_arr.max()), 1e-9)
            _SPARKS_CVH = " ▁▂▃▄▅▆▇█"
            _cvhp_spark = "".join(
                _SPARKS_CVH[min(8, int(max(v, 0) / _cvhp_max * 8))]
                for v in _cvhp_cv
            )
            print(f"\n  Running coh3 variance  ({len(_cvhp_cv)} steps):")
            print(f"  final={_cvhp_arr[-1]:.6f}  "
                  f"max={_cvhp_arr.max():.6f}  "
                  f"avg={_cvhp_arr.mean():.6f}")
            print(f"  {_cvhp_spark}")
            print(f"  (flat = stable coherence; rising = coherence fluctuating more)")
            continue

        if low.startswith("confvarplot"):
            # confvarplot — running confidence variance sparkline
            _cvp_res = getattr(model, "_last_gen_result", None)
            if _cvp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cvp_cv  = _cvp_res.get("conf_var_steps", [])
            _cvp_cf  = _cvp_res.get("confidences",    [])
            if not _cvp_cv and not _cvp_cf:
                print("  No confidence data."); continue
            if not _cvp_cv:
                import numpy as _np_cvpc
                _cvp_cv = [
                    float(_np_cvpc.var(_cvp_cf[:i+1])) if i >= 1 else 0.0
                    for i in range(len(_cvp_cf))
                ]
            import numpy as _np_cvp
            _cvp_arr = _np_cvp.array(_cvp_cv, dtype=float)
            _cvp_max = max(float(_cvp_arr.max()), 1e-9)
            _SPARKS_CVP = " ▁▂▃▄▅▆▇█"
            _cvp_spark = "".join(
                _SPARKS_CVP[min(8, int(max(v, 0) / _cvp_max * 8))]
                for v in _cvp_cv
            )
            print(f"\n  Running confidence variance  ({len(_cvp_cv)} steps):")
            print(f"  final={_cvp_arr[-1]:.6f}  "
                  f"max={_cvp_arr.max():.6f}  "
                  f"avg={_cvp_arr.mean():.6f}")
            print(f"  {_cvp_spark}")
            print(f"  (rising = spread growing; falling = confidence converging)")
            continue

        if low.startswith("cohentropycorr"):
            # cohentropycorr — coh3 vs entropy sparklines with Pearson r
            _cec_res = getattr(model, "_last_gen_result", None)
            if _cec_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cec_c3s = _cec_res.get("coh3_steps",     [])
            _cec_ent = _cec_res.get("entropy_steps",  [])
            _cec_r   = _cec_res.get("coh_entropy_corr", 0.0)
            _cec_n   = min(len(_cec_c3s), len(_cec_ent))
            if _cec_n < 3:
                print("  Need ≥3 steps of both coh3 and entropy."); continue
            _cec_c3s = _cec_c3s[:_cec_n]
            _cec_ent = _cec_ent[:_cec_n]
            import numpy as _np_cec
            _SPARKS_CEC = " ▁▂▃▄▅▆▇█"
            _cec_c_max = max(max(abs(v) for v in _cec_c3s), 1e-9)
            _cec_e_max = max(max(_cec_ent), 1e-9)
            _cec_c_sp = "".join(
                _SPARKS_CEC[min(8, int(max(v, 0) / _cec_c_max * 8))] for v in _cec_c3s
            )
            _cec_e_sp = "".join(
                _SPARKS_CEC[min(8, int(max(v, 0) / _cec_e_max * 8))] for v in _cec_ent
            )
            _cec_interp = ("coherent = focused" if _cec_r < -0.40 else
                           ("coherent = chaotic (unusual)" if _cec_r > 0.40 else "independent"))
            print(f"\n  Coh3 vs Entropy correlation  (r={_cec_r:+.4f}  → {_cec_interp}):")
            print(f"  Coh3:    {_cec_c_sp}  avg={sum(_cec_c3s)/_cec_n:+.4f}")
            print(f"  Entropy: {_cec_e_sp}  avg={sum(_cec_ent)/_cec_n:.4f}")
            print(f"  (negative r expected: higher coherence → lower entropy)")
            continue

        if low.startswith("rhythmtrend"):
            # rhythmtrend — running rhythm rate per step with slope
            _rt_res = getattr(model, "_last_gen_result", None)
            if _rt_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _rt_steps = _rt_res.get("rhythm_rate_steps", [])
            _rt_slope = _rt_res.get("rhythm_trend",      0.0)
            _rt_score = _rt_res.get("rhythm_score",      0.0)
            if not _rt_steps:
                print("  No rhythm_rate_steps data."); continue
            import numpy as _np_rt
            _rt_arr = _np_rt.array(_rt_steps, dtype=float)
            _SPARKS_RT = " ▁▂▃▄▅▆▇█"
            _rt_spark = "".join(
                _SPARKS_RT[min(8, int(v * 8))] for v in _rt_steps
            )
            _rt_dir = ("smoothing" if _rt_slope > 0.00005 else
                       ("roughening" if _rt_slope < -0.00005 else "stable"))
            print(f"\n  Rhythm rate trend  ({len(_rt_steps)} steps)  "
                  f"slope={_rt_slope:+.5f}  ({_rt_dir})  final={_rt_score:.4f}:")
            print(f"  {_rt_spark}")
            print(f"  avg={_rt_arr.mean():.4f}  "
                  f"min={_rt_arr.min():.4f}  max={_rt_arr.max():.4f}")
            print(f"  (1.0 = smooth confidence; slope + = generation getting smoother)")
            continue

        if low.startswith("confemascatter"):
            # confemascatter — conf_ema derivative scatter (+ ▲ / - ▼ / 0 ─ per step)
            _ces_res = getattr(model, "_last_gen_result", None)
            if _ces_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ces_sl = _ces_res.get("confslope_steps", [])
            _ces_em = _ces_res.get("conf_ema_steps",  [])
            _ces_tk = _ces_res.get("tokens",          [])
            if not _ces_sl:
                print("  No confslope_steps data."); continue
            # Build scatter string: ▲ positive, ▼ negative, ─ near zero
            _ces_scatter = "".join(
                ("▲" if v > 0.002 else ("▼" if v < -0.002 else "─"))
                for v in _ces_sl
            )
            _ces_rises  = sum(1 for v in _ces_sl if v > 0.002)
            _ces_drops  = sum(1 for v in _ces_sl if v < -0.002)
            _ces_flat   = len(_ces_sl) - _ces_rises - _ces_drops
            print(f"\n  Conf_EMA slope scatter  ({len(_ces_sl)} steps):")
            print(f"  {_ces_scatter}")
            print(f"  ▲ rises={_ces_rises} ({100*_ces_rises/max(len(_ces_sl),1):.1f}%)  "
                  f"▼ drops={_ces_drops} ({100*_ces_drops/max(len(_ces_sl),1):.1f}%)  "
                  f"─ flat={_ces_flat}")
            _ces_delta = _ces_res.get("conf_ema_delta", 0.0)
            print(f"  net Δ={_ces_delta:+.5f}  "
                  f"(start={_ces_em[0] if _ces_em else 0.0:.4f}  "
                  f"end={_ces_em[-1] if _ces_em else 0.0:.4f})")
            continue

        if low.startswith("velconfcorr"):
            # velconfcorr — velocity vs confidence sparklines with Pearson r
            _vcc_res = getattr(model, "_last_gen_result", None)
            if _vcc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _vcc_vel  = _vcc_res.get("velocity_steps", [])
            _vcc_conf = _vcc_res.get("confidences",    [])
            _vcc_r    = _vcc_res.get("vel_conf_corr",  0.0)
            _vcc_n    = min(len(_vcc_vel), len(_vcc_conf))
            if _vcc_n < 3:
                print("  Need ≥3 steps of both velocity and confidence."); continue
            _vcc_vel  = _vcc_vel[:_vcc_n]
            _vcc_conf = _vcc_conf[:_vcc_n]
            import numpy as _np_vcc
            _SPARKS_VCC = " ▁▂▃▄▅▆▇█"
            _vcc_v_max = max(max(_vcc_vel), 1e-9)
            _vcc_c_max = max(max(_vcc_conf), 1e-9)
            _vcc_v_sp = "".join(
                _SPARKS_VCC[min(8, int(v / _vcc_v_max * 8))] for v in _vcc_vel
            )
            _vcc_c_sp = "".join(
                _SPARKS_VCC[min(8, int(v / _vcc_c_max * 8))] for v in _vcc_conf
            )
            _vcc_interp = ("co-move (unusual)" if _vcc_r > 0.40 else
                           ("drift kills confidence" if _vcc_r < -0.40 else "independent"))
            print(f"\n  Velocity vs Confidence correlation  (r={_vcc_r:+.4f}  → {_vcc_interp}):")
            print(f"  Vel:  {_vcc_v_sp}  avg={sum(_vcc_vel)/_vcc_n:.5f}")
            print(f"  Conf: {_vcc_c_sp}  avg={sum(_vcc_conf)/_vcc_n:.4f}")
            print(f"  (negative r expected: faster context drift → lower confidence)")
            continue

        if low.startswith("scorevartrend"):
            # scorevartrend — score variance per-step sparkline with slope
            _svt_res = getattr(model, "_last_gen_result", None)
            if _svt_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _svt_sv   = _svt_res.get("score_var_steps",  [])
            _svt_sl   = _svt_res.get("score_var_trend",  0.0)
            _svt_mean = _svt_res.get("scorevar",         0.0)
            if not _svt_sv:
                print("  No score_var_steps data."); continue
            import numpy as _np_svt
            _svt_arr = _np_svt.array(_svt_sv, dtype=float)
            _svt_max = max(float(_svt_arr.max()), 1e-9)
            _SPARKS_SVT = " ▁▂▃▄▅▆▇█"
            _svt_spark = "".join(
                _SPARKS_SVT[min(8, int(max(v, 0) / _svt_max * 8))]
                for v in _svt_sv
            )
            _svt_dir = ("spreading" if _svt_sl > 0.0001 else
                        ("tightening" if _svt_sl < -0.0001 else "stable"))
            print(f"\n  Score variance trend  ({len(_svt_sv)} steps)  "
                  f"slope={_svt_sl:+.5f}  ({_svt_dir})  mean={_svt_mean:.5f}:")
            print(f"  {_svt_spark}")
            print(f"  max={_svt_arr.max():.5f}  min={_svt_arr.min():.5f}  "
                  f"std={_svt_arr.std():.5f}")
            print(f"  (rising slope = score spread widening = ranking less stable)")
            continue

        if low.startswith("topkentropyplot"):
            # topkentropyplot — side-by-side TopK + entropy sparklines with r
            _tep_res = getattr(model, "_last_gen_result", None)
            if _tep_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _tep_topk = _tep_res.get("topk_steps",         [])
            _tep_ent  = _tep_res.get("entropy_steps",      [])
            _tep_r    = _tep_res.get("topk_entropy_corr",  0.0)
            _tep_n    = min(len(_tep_topk), len(_tep_ent))
            if _tep_n < 3:
                print("  Need ≥3 steps of both topk and entropy."); continue
            _tep_topk = _tep_topk[:_tep_n]
            _tep_ent  = _tep_ent[:_tep_n]
            import numpy as _np_tep
            _SPARKS_TEP = " ▁▂▃▄▅▆▇█"
            _tep_tk_max = max(max(_tep_topk), 1)
            _tep_e_max  = max(max(_tep_ent), 1e-9)
            _tep_tk_sp = "".join(
                _SPARKS_TEP[min(8, int(v / _tep_tk_max * 8))] for v in _tep_topk
            )
            _tep_e_sp = "".join(
                _SPARKS_TEP[min(8, int(max(v, 0) / _tep_e_max * 8))] for v in _tep_ent
            )
            _tep_interp = ("co-move" if _tep_r > 0.40 else
                           ("opposite" if _tep_r < -0.40 else "unrelated"))
            print(f"\n  TopK vs Entropy correlation  (r={_tep_r:+.4f}  → {_tep_interp}):")
            print(f"  TopK:    {_tep_tk_sp}  mode={_tep_res.get('topk_mode', '?')}")
            print(f"  Entropy: {_tep_e_sp}")
            print(f"  TopK    avg={sum(_tep_topk)/_tep_n:.2f}  "
                  f"range={max(_tep_topk)-min(_tep_topk)}")
            print(f"  Entropy avg={sum(_tep_ent)/_tep_n:.4f}  "
                  f"range={max(_tep_ent)-min(_tep_ent):.4f}")
            continue

        if low.startswith("cohconfcorr"):
            # cohconfcorr — side-by-side coh3 + confidence sparklines with r
            _ccc_res = getattr(model, "_last_gen_result", None)
            if _ccc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ccc_c3s  = _ccc_res.get("coh3_steps",    [])
            _ccc_conf = _ccc_res.get("confidences",   [])
            _ccc_r    = _ccc_res.get("coh_conf_corr", 0.0)
            _ccc_n    = min(len(_ccc_c3s), len(_ccc_conf))
            if _ccc_n < 3:
                print("  Need ≥3 steps of both coh3 and confidence."); continue
            _ccc_c3s  = _ccc_c3s[:_ccc_n]
            _ccc_conf = _ccc_conf[:_ccc_n]
            import numpy as _np_ccc
            _SPARKS_CCC = " ▁▂▃▄▅▆▇█"
            _ccc_c_max = max(max(abs(v) for v in _ccc_c3s), 1e-9)
            _ccc_f_max = max(max(_ccc_conf), 1e-9)
            _ccc_c_sp = "".join(
                _SPARKS_CCC[min(8, int(max(v, 0) / _ccc_c_max * 8))] for v in _ccc_c3s
            )
            _ccc_f_sp = "".join(
                _SPARKS_CCC[min(8, int(v / _ccc_f_max * 8))] for v in _ccc_conf
            )
            _ccc_interp = ("coherent tokens = confident" if _ccc_r > 0.40 else
                           ("anti-correlated" if _ccc_r < -0.40 else "independent"))
            print(f"\n  Coh3 vs Confidence correlation  (r={_ccc_r:+.4f}  → {_ccc_interp}):")
            print(f"  Coh3: {_ccc_c_sp}  avg={sum(_ccc_c3s)/_ccc_n:+.4f}")
            print(f"  Conf: {_ccc_f_sp}  avg={sum(_ccc_conf)/_ccc_n:.4f}")
            continue

        if low.startswith("cohrises"):
            # cohrises — steps where coh3 jumped ≥0.05 in one step
            _chr_res = getattr(model, "_last_gen_result", None)
            if _chr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _chr_rises = _chr_res.get("coh_rise_steps", [])
            _chr_c3s   = _chr_res.get("coh3_steps",     [])
            _chr_toks  = _chr_res.get("tokens",         [])
            _CHR_SPARKS = " ▁▂▃▄▅▆▇█"
            _chr_max    = max((max(abs(v) for v in _chr_c3s), 1e-9)
                              if _chr_c3s else (1e-9,))
            _chr_rise_set = {r["step"] for r in _chr_rises}
            _chr_spark = "".join(
                ("^" if i in _chr_rise_set else
                 _CHR_SPARKS[min(8, int(max(v, 0) / _chr_max * 8))])
                for i, v in enumerate(_chr_c3s)
            )
            print(f"\n  Coherence rise events (coh3 ≥0.05 jump)  "
                  f"({len(_chr_rises)} event(s) in {len(_chr_c3s)} steps):")
            print(f"  {_chr_spark}  '^' = rise")
            if not _chr_rises:
                print("  No large coherence jumps detected.")
            else:
                print(f"  {'Step':<6}  {'Token':<18}  Rise      Before    After")
                for _rhv in _chr_rises:
                    _rhi  = _rhv["step"]
                    _rht  = _rhv["tok"]
                    _rhr  = _rhv["rise"]
                    _rbef = _chr_c3s[_rhi - 1] if _rhi >= 1 else 0.0
                    _raft = _chr_c3s[_rhi]     if _rhi < len(_chr_c3s) else 0.0
                    _rbar = "▲" * min(8, max(1, int(_rhr / 0.05)))
                    print(f"  {_rhi:<6}  {_rht[:18]:<18}  +{_rhr:.4f}   "
                          f"{_rbef:+.4f}  → {_raft:+.4f}  {_rbar}")
            continue

        if low.startswith("veltrend"):
            # veltrend — velocity per-step sparkline with linear slope
            _vt_res = getattr(model, "_last_gen_result", None)
            if _vt_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _vt_vel   = _vt_res.get("velocity_steps", [])
            _vt_slope = _vt_res.get("vel_trend",      0.0)
            _vt_velr  = _vt_res.get("vel_ratio",      0.0)
            if not _vt_vel:
                print("  No velocity_steps data."); continue
            import numpy as _np_vt
            _vt_arr = _np_vt.array(_vt_vel, dtype=float)
            _vt_max = max(float(_vt_arr.max()), 1e-9)
            _SPARKS_VT = " ▁▂▃▄▅▆▇█"
            _vt_spark = "".join(
                _SPARKS_VT[min(8, int(v / _vt_max * 8))]
                for v in _vt_vel
            )
            _vt_dir = ("drifting faster" if _vt_slope > 0.00005 else
                       ("slowing down" if _vt_slope < -0.00005 else "steady"))
            print(f"\n  Velocity trend  ({len(_vt_vel)} steps)  "
                  f"slope={_vt_slope:+.5f}  ({_vt_dir})  vel_ratio={_vt_velr:.4f}:")
            print(f"  {_vt_spark}")
            print(f"  avg={_vt_arr.mean():.5f}  "
                  f"min={_vt_arr.min():.5f}  max={_vt_arr.max():.5f}")
            print(f"  (rising slope = context shifting faster = more topic drift)")
            continue

        if low.startswith("cohdrop"):
            # cohdrop — steps where coh3 dropped ≥0.05 in one step
            _chd_res = getattr(model, "_last_gen_result", None)
            if _chd_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _chd_drops = _chd_res.get("coh_drop_steps", [])
            _chd_c3s   = _chd_res.get("coh3_steps",     [])
            _chd_toks  = _chd_res.get("tokens",         [])
            _CHD_SPARKS = " ▁▂▃▄▅▆▇█"
            _chd_max    = max((max(abs(v) for v in _chd_c3s), 1e-9)
                              if _chd_c3s else (1e-9,))
            _chd_drop_set = {d["step"] for d in _chd_drops}
            _chd_spark = "".join(
                ("!" if i in _chd_drop_set else
                 _CHD_SPARKS[min(8, int(max(v, 0) / _chd_max * 8))])
                for i, v in enumerate(_chd_c3s)
            )
            print(f"\n  Coherence drop events (coh3 ≥0.05 fall)  "
                  f"({len(_chd_drops)} event(s) in {len(_chd_c3s)} steps):")
            print(f"  {_chd_spark}  '!' = drop")
            if not _chd_drops:
                print("  No large coherence drops — coh3 was stable.")
            else:
                print(f"  {'Step':<6}  {'Token':<18}  Drop      Before    After")
                for _dhv in _chd_drops:
                    _dhi  = _dhv["step"]
                    _dht  = _dhv["tok"]
                    _dhd  = _dhv["drop"]
                    _dbef = _chd_c3s[_dhi - 1] if _dhi >= 1 else 0.0
                    _daft = _chd_c3s[_dhi]     if _dhi < len(_chd_c3s) else 0.0
                    _dhbar = "▼" * min(8, max(1, int(_dhd / 0.05)))
                    print(f"  {_dhi:<6}  {_dht[:18]:<18}  -{_dhd:.4f}   "
                          f"{_dbef:+.4f}  → {_daft:+.4f}  {_dhbar}")
            continue

        if low.startswith("entropytrend"):
            # entropytrend — entropy per-step sparkline with linear slope
            _et_res = getattr(model, "_last_gen_result", None)
            if _et_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _et_entr  = _et_res.get("entropy_steps",  [])
            _et_slope = _et_res.get("entropy_trend",  0.0)
            _et_delta = _et_res.get("entdelta",        0.0)
            if not _et_entr:
                print("  No entropy_steps data."); continue
            import numpy as _np_et
            _et_arr = _np_et.array(_et_entr, dtype=float)
            _et_max  = max(float(_et_arr.max()), 1e-9)
            _SPARKS_ET = " ▁▂▃▄▅▆▇█"
            _et_spark = "".join(
                _SPARKS_ET[min(8, int(max(v, 0) / _et_max * 8))]
                for v in _et_entr
            )
            _et_dir = ("rising (more random)" if _et_slope > 0.0001 else
                       ("falling (more focused)" if _et_slope < -0.0001 else "flat"))
            print(f"\n  Entropy trend  ({len(_et_entr)} steps)  "
                  f"slope={_et_slope:+.5f}  ({_et_dir})  Δtotal={_et_delta:+.4f}:")
            print(f"  {_et_spark}")
            print(f"  avg={_et_arr.mean():.4f}  "
                  f"min={_et_arr.min():.4f}  max={_et_arr.max():.4f}")
            print(f"  (H_norm ∈ [0,1]; falling = model getting more focused)")
            continue

        if low.startswith("margintrend"):
            # margintrend — per-step top1−top2 score margin with linear slope
            _mt_res = getattr(model, "_last_gen_result", None)
            if _mt_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _mt_margs = _mt_res.get("top1_margins",   [])
            _mt_slope = _mt_res.get("margin_trend",   0.0)
            _mt_toks  = _mt_res.get("tokens",         [])
            if not _mt_margs:
                print("  No margin data."); continue
            import numpy as _np_mt
            _mt_arr = _np_mt.array(_mt_margs, dtype=float)
            _mt_max = max(float(_mt_arr.max()), 1e-9)
            _SPARKS_MT = " ▁▂▃▄▅▆▇█"
            _mt_spark = "".join(
                _SPARKS_MT[min(8, int(max(v, 0) / _mt_max * 8))]
                for v in _mt_margs
            )
            _mt_dir = ("decisive" if _mt_slope > 0.0001 else
                       ("hesitant" if _mt_slope < -0.0001 else "steady"))
            print(f"\n  Score margin trend  ({len(_mt_margs)} steps)  "
                  f"slope={_mt_slope:+.5f}  ({_mt_dir}):")
            print(f"  {_mt_spark}")
            print(f"  avg={_mt_arr.mean():.5f}  "
                  f"min={_mt_arr.min():.5f}  max={_mt_arr.max():.5f}")
            print(f"  (higher = model more decisive; slope + = growing confidence in picks)")
            continue

        if low.startswith("confbuckets"):
            # confbuckets — 5-bucket confidence histogram 0-0.2 … 0.8-1.0
            _cb_res = getattr(model, "_last_gen_result", None)
            if _cb_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cb_hist = _cb_res.get("conf_bucket_hist", {})
            _cb_conf = _cb_res.get("confidences", [])
            if not _cb_hist and not _cb_conf:
                print("  No confidence data."); continue
            # recompute if missing
            if not _cb_hist:
                _cb_hist = {
                    f"{lo:.1f}-{hi:.1f}": sum(
                        1 for c in _cb_conf if lo <= c < hi
                    )
                    for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6),
                                   (0.6, 0.8), (0.8, 1.01)]
                }
            _cb_total = sum(_cb_hist.values()) or 1
            _cb_max   = max(_cb_hist.values(), default=1)
            print(f"\n  Confidence bucket histogram  ({_cb_total} steps):")
            print(f"  {'Bucket':<10}  {'Count':>6}  {'%':>6}  Bar")
            for _bk, _bv in sorted(_cb_hist.items()):
                _bp    = 100.0 * _bv / _cb_total
                _bblen = int(30 * _bv / _cb_max)
                _bbar  = "█" * _bblen + "░" * (30 - _bblen)
                print(f"  {_bk:<10}  {_bv:>6}  {_bp:>5.1f}%  {_bbar}")
            # dominant bucket
            _dom_bk = max(_cb_hist, key=_cb_hist.get)
            print(f"  dominant bucket: {_dom_bk}  "
                  f"({100*_cb_hist[_dom_bk]/_cb_total:.1f}% of steps)")
            continue

        if low.startswith("scoregaptrend"):
            # scoregaptrend — rolling-mean score gap per step sparkline
            _sgt_res = getattr(model, "_last_gen_result", None)
            if _sgt_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _sgt_steps = _sgt_res.get("score_gap_trend", [])
            _sgt_raw   = _sgt_res.get("sg_step_gaps",    [])
            _sgt_toks  = _sgt_res.get("tokens",          [])
            if not _sgt_steps and not _sgt_raw:
                print("  No score gap data."); continue
            _sgt_arr = _sgt_steps if _sgt_steps else _sgt_raw
            import numpy as _np_sgt
            _sgt_np   = _np_sgt.array(_sgt_arr, dtype=float)
            _sgt_max  = max(float(_sgt_np.max()), 1e-9)
            _SPARKS_SGT = " ▁▂▃▄▅▆▇█"
            _sgt_spark = "".join(
                _SPARKS_SGT[min(8, int(max(v, 0) / _sgt_max * 8))]
                for v in _sgt_arr
            )
            print(f"\n  Score gap trend  (rolling 4-step mean, {len(_sgt_arr)} pts):")
            print(f"  avg={_sgt_np.mean():.5f}  "
                  f"min={_sgt_np.min():.5f}  max={_sgt_np.max():.5f}")
            print(f"  {_sgt_spark}")
            print(f"  (higher gap = model is more decisive about its top choice)")
            continue

        if low.startswith("sfbaccel"):
            # sfbaccel — SFB per-step values with linear slope summary
            _sa_res = getattr(model, "_last_gen_result", None)
            if _sa_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _sa_sfbs = _sa_res.get("sfb_steps",  [])
            _sa_acc  = _sa_res.get("sfb_acc",    0.0)
            _sa_tr   = _sa_res.get("sfb_trend",  0.0)
            _sa_toks = _sa_res.get("tokens",     [])
            if not _sa_sfbs:
                print("  No sfb_steps data."); continue
            import numpy as _np_sa
            _sa_arr  = _np_sa.array(_sa_sfbs, dtype=float)
            _sa_max  = max(float(_sa_arr.max()), 1e-9)
            _SPARKS_SA = " ▁▂▃▄▅▆▇█"
            _sa_spark = "".join(
                _SPARKS_SA[min(8, int(max(v, 0) / _sa_max * 8))]
                for v in _sa_sfbs
            )
            _sa_dir = ("rising" if _sa_acc > 0.0001 else
                       ("falling" if _sa_acc < -0.0001 else "flat"))
            print(f"\n  SFB acceleration  ({len(_sa_sfbs)} steps)  "
                  f"slope={_sa_acc:+.5f}  ({_sa_dir})  trend={_sa_tr:+.4f}:")
            print(f"  {_sa_spark}")
            print(f"  avg={_sa_arr.mean():.4f}  "
                  f"min={_sa_arr.min():.4f}  max={_sa_arr.max():.4f}")
            print(f"  ('rising slope' = phrase variety improving over the generation)")
            continue

        if low.startswith("confrises"):
            # confrises — steps where confidence jumped ≥0.05 in one step
            _cr_res = getattr(model, "_last_gen_result", None)
            if _cr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cr_rises = _cr_res.get("conf_rise_steps", [])
            _cr_confs = _cr_res.get("confidences", [])
            _cr_toks  = _cr_res.get("tokens", [])
            _CR_SPARKS = " ▁▂▃▄▅▆▇█"
            _cr_max    = max(max(_cr_confs), 1e-9) if _cr_confs else 1.0
            _cr_rise_set = {r["step"] for r in _cr_rises}
            _cr_spark = "".join(
                ("^" if i in _cr_rise_set else
                 _CR_SPARKS[min(8, int(c / _cr_max * 8))])
                for i, c in enumerate(_cr_confs)
            )
            print(f"\n  Confidence rise events (≥0.05)  "
                  f"({len(_cr_rises)} rise(s) in {len(_cr_confs)} steps):")
            print(f"  {_cr_spark}  '^' = rise")
            if not _cr_rises:
                print("  No large confidence jumps detected.")
            else:
                print(f"  {'Step':<6}  {'Token':<18}  Rise      Before    After")
                for _rv in _cr_rises:
                    _ri   = _rv["step"]
                    _rt   = _rv["tok"]
                    _rr   = _rv["rise"]
                    _rbef = _cr_confs[_ri - 1] if _ri >= 1 else 0.0
                    _raft = _cr_confs[_ri]     if _ri < len(_cr_confs) else 0.0
                    _rbar = "▲" * min(8, max(1, int(_rr / 0.05)))
                    print(f"  {_ri:<6}  {_rt[:18]:<18}  +{_rr:.4f}   "
                          f"{_rbef:.4f}  → {_raft:.4f}  {_rbar}")
            continue

        if low.startswith("topkmode"):
            # topkmode — histogram of TopK k values used during generation
            _tm_res = getattr(model, "_last_gen_result", None)
            if _tm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _tm_steps = _tm_res.get("topk_steps", [])
            _tm_mode  = _tm_res.get("topk_mode",   0)
            if not _tm_steps:
                print("  No topk_steps data."); continue
            from collections import Counter as _TMCounter
            _tm_counts = _TMCounter(_tm_steps)
            _tm_total  = len(_tm_steps)
            _tm_sorted = sorted(_tm_counts.items())
            _tm_max    = max(_tm_counts.values(), default=1)
            print(f"\n  TopK usage histogram  ({_tm_total} steps, mode={_tm_mode}):")
            print(f"  {'k':<6}  {'Count':>6}  {'%':>6}  Bar")
            for _tk, _tc in _tm_sorted:
                _tp    = 100.0 * _tc / _tm_total
                _tblen = int(30 * _tc / _tm_max)
                _tbar  = "█" * _tblen + "░" * (30 - _tblen)
                _mark  = " ← mode" if _tk == _tm_mode else ""
                print(f"  {_tk:<6}  {_tc:>6}  {_tp:>5.1f}%  {_tbar}{_mark}")
            continue

        if low.startswith("correlplot"):
            # correlplot — side-by-side coh3 + velocity sparklines with Pearson r
            _cpl_res = getattr(model, "_last_gen_result", None)
            if _cpl_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cpl_c3s = _cpl_res.get("coh3_steps",     [])
            _cpl_vel = _cpl_res.get("velocity_steps",  [])
            _cpl_r   = _cpl_res.get("coh_vel_correlation", 0.0)
            _cpl_n   = min(len(_cpl_c3s), len(_cpl_vel))
            if _cpl_n < 3:
                print("  Need ≥3 steps of both coh3 and velocity."); continue
            _cpl_c3s = _cpl_c3s[:_cpl_n]
            _cpl_vel = _cpl_vel[:_cpl_n]
            import numpy as _np_cpl
            _SPARKS_CL = " ▁▂▃▄▅▆▇█"
            _cpl_c_max = max(max(abs(v) for v in _cpl_c3s), 1e-9)
            _cpl_v_max = max(max(_cpl_vel), 1e-9)
            _cpl_c_spark = "".join(
                _SPARKS_CL[min(8, int(max(v,0) / _cpl_c_max * 8))] for v in _cpl_c3s
            )
            _cpl_v_spark = "".join(
                _SPARKS_CL[min(8, int(v / _cpl_v_max * 8))] for v in _cpl_vel
            )
            _cpl_interp = ("move together" if _cpl_r > 0.40
                           else ("move oppositely" if _cpl_r < -0.40 else "uncorrelated"))
            print(f"\n  Coh3 vs Velocity correlation  (r={_cpl_r:+.4f}  "
                  f"→ {_cpl_interp}):")
            print(f"  Coh3:  {_cpl_c_spark}")
            print(f"  Vel:   {_cpl_v_spark}")
            print(f"  Coh3   avg={sum(_cpl_c3s)/_cpl_n:+.4f}  "
                  f"range={max(_cpl_c3s)-min(_cpl_c3s):.4f}")
            print(f"  Vel    avg={sum(_cpl_vel)/_cpl_n:.4f}  "
                  f"range={max(_cpl_vel)-min(_cpl_vel):.4f}")
            continue

        if low.startswith("confdrop"):
            # confdrop — show all steps where confidence dropped ≥0.05
            _cd_res = getattr(model, "_last_gen_result", None)
            if _cd_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cd_drops = _cd_res.get("conf_drop_steps", [])
            _cd_confs = _cd_res.get("confidences",     [])
            _cd_toks  = _cd_res.get("tokens",          [])
            # Build sparkline with drops marked
            _CD_SPARKS = " ▁▂▃▄▅▆▇█"
            _cd_max  = max(max(_cd_confs), 1e-9) if _cd_confs else 1.0
            _cd_drop_steps = {d["step"] for d in _cd_drops}
            _cd_spark = "".join(
                ("!" if i in _cd_drop_steps else
                 _CD_SPARKS[min(8, int(c / _cd_max * 8))])
                for i, c in enumerate(_cd_confs)
            )
            print(f"\n  Confidence drop events (≥0.05)  "
                  f"({len(_cd_drops)} drop(s) in {len(_cd_confs)} steps):")
            print(f"  {_cd_spark}  '!' = drop")
            if not _cd_drops:
                print("  No large drops — confidence was stable.")
            else:
                print(f"  {'Step':<6}  {'Token':<18}  Drop      Before    After")
                for _dv in _cd_drops:
                    _di   = _dv["step"]
                    _dt   = _dv["tok"]
                    _dd   = _dv["drop"]
                    _dbef = _cd_confs[_di - 1] if _di >= 1 else 0.0
                    _daft = _cd_confs[_di]     if _di < len(_cd_confs) else 0.0
                    _dbar = "▼" * min(8, max(1, int(_dd / 0.05)))
                    print(f"  {_di:<6}  {_dt[:18]:<18}  -{_dd:.4f}   "
                          f"{_dbef:.4f}  → {_daft:.4f}  {_dbar}")
            continue

        if low.startswith("midconf"):
            # midconf — early/mid/late conf_ema with comparison arrows
            _mc_res = getattr(model, "_last_gen_result", None)
            if _mc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _mc_emas = _mc_res.get("conf_ema_steps", [])
            _mc_toks = _mc_res.get("tokens", [])
            if len(_mc_emas) < 3:
                print("  Need ≥3 steps."); continue
            _mc_n   = len(_mc_emas)
            _mc_e   = float(_mc_emas[_mc_n // 6])      # ~16% in
            _mc_m   = float(_mc_emas[_mc_n // 2])      # midpoint
            _mc_l   = float(_mc_emas[-1])              # last
            _mc_start = float(_mc_emas[0])
            _mc_mid_key = _mc_res.get("conf_ema_mid", _mc_m)
            _mc_delta   = _mc_res.get("conf_ema_delta", 0.0)
            _mc_tok_e = _mc_toks[_mc_n // 6] if _mc_n // 6 < len(_mc_toks) else "?"
            _mc_tok_m = _mc_toks[_mc_n // 2] if _mc_n // 2 < len(_mc_toks) else "?"
            _mc_tok_l = _mc_toks[-1] if _mc_toks else "?"
            def _mc_arrow(a: float, b: float) -> str:
                return "▲" if b > a + 0.003 else ("▼" if b < a - 0.003 else "─")
            print(f"\n  Midpoint confidence analysis:")
            print(f"  start={_mc_start:.4f}  "
                  f"early={_mc_e:.4f}  "
                  f"mid={_mc_m:.4f}  "
                  f"final={_mc_l:.4f}  "
                  f"Δtotal={_mc_delta:+.4f}")
            print(f"  start→early {_mc_arrow(_mc_start, _mc_e)}  "
                  f"early→mid {_mc_arrow(_mc_e, _mc_m)}  "
                  f"mid→final {_mc_arrow(_mc_m, _mc_l)}")
            print(f"  early@step{_mc_n//6}: '{_mc_tok_e}'  "
                  f"mid@step{_mc_n//2}: '{_mc_tok_m}'  "
                  f"final@step{_mc_n-1}: '{_mc_tok_l}'")
            continue

        if low.startswith("uniqplot"):
            # uniqplot — running unique-token ratio per step
            _uq_res = getattr(model, "_last_gen_result", None)
            if _uq_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _uq_steps = _uq_res.get("uniq_steps", [])
            _uq_toks  = _uq_res.get("tokens",     [])
            _uq_ratio = _uq_res.get("uniq_ratio",  0.0)
            if not _uq_steps:
                # fallback: compute on the fly
                _uq_steps = [
                    len(set(_uq_toks[:i+1])) / max(i+1, 1)
                    for i in range(len(_uq_toks))
                ]
            if not _uq_steps:
                print("  No token data."); continue
            import numpy as _np_uq
            _uq_arr = _np_uq.array(_uq_steps, dtype=float)
            _SPARKS_UQ = " ▁▂▃▄▅▆▇█"
            _uq_spark = "".join(
                _SPARKS_UQ[min(8, int(v * 8))] for v in _uq_arr
            )
            print(f"\n  Running unique-token ratio  ({len(_uq_steps)} steps):")
            print(f"  final={_uq_ratio:.4f}  "
                  f"min={_uq_arr.min():.4f}  max={_uq_arr.max():.4f}")
            print(f"  {_uq_spark}")
            print(f"  (starts 1.0, drops as tokens repeat; higher=more diverse)")
            continue

        if low.startswith("cohseries"):
            # cohseries — full coh3_steps table with sparkline
            _csr_res = getattr(model, "_last_gen_result", None)
            if _csr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _csr_c3s = _csr_res.get("coh3_steps", [])
            _csr_tok = _csr_res.get("tokens",     [])
            if not _csr_c3s:
                print("  No coh3 data."); continue
            import numpy as _np_csr
            _csr_arr = _np_csr.array(_csr_c3s, dtype=float)
            _csr_max = max(float(_csr_arr.max()), 1e-9)
            _SPARKS_CSR = " ▁▂▃▄▅▆▇█"
            _csr_spark = "".join(
                _SPARKS_CSR[min(8, int(max(v, 0) / _csr_max * 8))]
                for v in _csr_arr
            )
            print(f"\n  Coh3 series  ({len(_csr_c3s)} steps)  "
                  f"avg={_csr_arr.mean():+.4f}  "
                  f"min={_csr_arr.min():+.4f}  max={_csr_arr.max():+.4f}")
            print(f"  {_csr_spark}")
            print(f"  {'Step':<5}  {'Token':<18}  Coh3      Bar")
            for _ci2, _cv2 in enumerate(_csr_c3s):
                _ct2  = _csr_tok[_ci2] if _ci2 < len(_csr_tok) else "?"
                _cb2l = int(20 * max(_cv2, 0) / _csr_max)
                _cb2  = "█" * _cb2l + "░" * (20 - _cb2l)
                print(f"  {_ci2:<5}  {_ct2[:18]:<18}  {_cv2:+.4f}   {_cb2}")
            continue

        if low.startswith("vocabtop10"):
            # vocabtop10 — top 10 tokens by frequency in last gen
            _vt_res = getattr(model, "_last_gen_result", None)
            if _vt_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _vt_toks = _vt_res.get("tokens", [])
            if not _vt_toks:
                print("  No tokens in last result."); continue
            from collections import Counter as _VT_Ctr
            _vt_ctr = _VT_Ctr(_vt_toks)
            _vt_total = len(_vt_toks)
            _vt_top   = _vt_ctr.most_common(10)
            _wintok   = _vt_res.get("wintok", "")
            _wincount = _vt_res.get("wintok_count", 0)
            print(f"\n  Top-10 tokens by frequency  ({_vt_total} total):")
            print(f"  Dominant token: '{_wintok}'  ({_wincount}×  "
                  f"{100*_wincount/_vt_total:.1f}%)")
            print(f"  {'Rank':<5}  {'Token':<20}  Count  %      Bar")
            for _rk2, (_tok2, _cnt2) in enumerate(_vt_top, 1):
                _pct2 = 100 * _cnt2 / _vt_total
                _bar2w = int(20 * _cnt2 / _vt_top[0][1])
                _bar2  = "█" * _bar2w + "░" * (20 - _bar2w)
                print(f"  {_rk2:<5}  {_tok2[:20]:<20}  {_cnt2:<5}  {_pct2:.1f}%   {_bar2}")
            continue

        if low.startswith("phasemap"):
            # phasemap — early/mid/late phase comparison of conf + coh3
            _pm_res = getattr(model, "_last_gen_result", None)
            if _pm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _pm_ps  = _pm_res.get("phasestats", {})
            _pm_tok = _pm_res.get("tokens", [])
            _pm_txt = _pm_res.get("text", "")
            if not _pm_ps:
                print("  No phase data."); continue
            print(f"\n  Phase map  ({len(_pm_tok)} tokens):")
            print(f"  {'Phase':<7}  {'Toks':<6}  AvgConf  AvgCoh3  "
                  f"ConfBar                   Coh3Bar")
            _pm_conf_max = max((v["avg_conf"] for v in _pm_ps.values()), default=1e-9)
            _pm_coh_max  = max((abs(v["avg_coh"]) for v in _pm_ps.values()), default=1e-9)
            for _ph in ("early", "mid", "late"):
                _pv = _pm_ps.get(_ph, {})
                _pc = _pv.get("avg_conf", 0.0)
                _ph2 = _pv.get("avg_coh",  0.0)
                _pn = _pv.get("n_toks",  0)
                _pc_bar = "█" * int(16 * _pc / max(_pm_conf_max, 1e-9)) + "░" * (16 - int(16 * _pc / max(_pm_conf_max, 1e-9)))
                _ph_bar = "█" * int(16 * max(_ph2, 0) / max(_pm_coh_max, 1e-9)) + "░" * (16 - int(16 * max(_ph2, 0) / max(_pm_coh_max, 1e-9)))
                print(f"  {_ph:<7}  {_pn:<6}  {_pc:.4f}   {_ph2:+.4f}   "
                      f"{_pc_bar}  {_ph_bar}")
            # Show token snippet for each phase
            _pm_n = len(_pm_tok)
            for _ph, _lo, _hi in [("early", 0, _pm_n//3),
                                   ("mid",   _pm_n//3, 2*_pm_n//3),
                                   ("late",  2*_pm_n//3, _pm_n)]:
                _snip = " ".join(_pm_tok[_lo:_hi][:6])
                print(f"  {_ph:7}  '{_snip[:40]}'{'…' if len(_snip)>40 else ''}")
            continue

        if low.startswith("rangeplot"):
            # rangeplot — conf and coh3 range (spread) bars for last generation
            _rp_res = getattr(model, "_last_gen_result", None)
            if _rp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _rp_confs = _rp_res.get("confidences",  [])
            _rp_c3s   = _rp_res.get("coh3_steps",   [])
            _rp_cr    = _rp_res.get("conf_range",   0.0)
            _rp_cohr  = _rp_res.get("coh_range",    0.0)
            if not _rp_confs:
                print("  Not enough data."); continue
            import numpy as _np_rp
            _rp_carr = _np_rp.array(_rp_confs, dtype=float)
            _rp_harr = _np_rp.array(_rp_c3s, dtype=float) if _rp_c3s else _np_rp.zeros(1)
            _rp_max  = max(_rp_cr, 1e-9)
            _rp_cbar = "█" * int(40 * _rp_cr / max(_rp_cr, _rp_cohr, 1e-9)) + "░" * (40 - int(40 * _rp_cr / max(_rp_cr, _rp_cohr, 1e-9)))
            _rp_hbar = "█" * int(40 * _rp_cohr / max(_rp_cr, _rp_cohr, 1e-9)) + "░" * (40 - int(40 * _rp_cohr / max(_rp_cr, _rp_cohr, 1e-9)))
            print(f"\n  Generation spread (range = max − min):")
            print(f"  Confidence  [{_rp_carr.min():.4f} … {_rp_carr.max():.4f}]  "
                  f"range={_rp_cr:.5f}")
            print(f"  Conf  {_rp_cbar}  {_rp_cr:.5f}")
            print(f"  Coh3  [{_rp_harr.min():.4f} … {_rp_harr.max():.4f}]  "
                  f"range={_rp_cohr:.5f}")
            print(f"  Coh3  {_rp_hbar}  {_rp_cohr:.5f}")
            print(f"  (wider bar = more spread in that signal)")
            continue

        if low.startswith("buckethist"):
            # buckethist — 10-bucket confidence histogram (0.0-0.1 … 0.9-1.0)
            _bh_res = getattr(model, "_last_gen_result", None)
            if _bh_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _bh_confs = _bh_res.get("confidences", [])
            if not _bh_confs:
                print("  No confidence data."); continue
            _bh_buckets = [0] * 10
            for _bhc in _bh_confs:
                _bh_buckets[min(9, int(_bhc * 10))] += 1
            _bh_max = max(max(_bh_buckets), 1)
            print(f"\n  Confidence distribution  ({len(_bh_confs)} tokens):")
            print(f"  {'Bucket':<14}  {'Count':<6}  Bar")
            for _bhi, _bhn in enumerate(_bh_buckets):
                _bh_lo = _bhi * 0.1
                _bh_hi = (_bhi + 1) * 0.1
                _bh_bar_w = int(30 * _bhn / _bh_max)
                _bh_bar = "█" * _bh_bar_w + "░" * (30 - _bh_bar_w)
                print(f"  [{_bh_lo:.1f} – {_bh_hi:.1f}]      "
                      f"{_bhn:<6}  {_bh_bar}  {100*_bhn/len(_bh_confs):.1f}%")
            continue

        if low.startswith("toklenplot"):
            # toklenplot — per-step token character length sparkline
            _tl_res = getattr(model, "_last_gen_result", None)
            if _tl_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _tl_lens = _tl_res.get("tokenlen_steps", [])
            _tl_toks = _tl_res.get("tokens", [])
            if not _tl_lens:
                # Fallback: compute from tokens list
                _tl_lens = [len(t) for t in _tl_toks]
            if not _tl_lens:
                print("  No token-length data."); continue
            import numpy as _np_tl
            _tl_arr = _np_tl.array(_tl_lens, dtype=float)
            _tl_max = max(float(_tl_arr.max()), 1.0)
            _SPARKS_TL = " ▁▂▃▄▅▆▇█"
            _tl_spark = "".join(
                _SPARKS_TL[min(8, int(v / _tl_max * 8))] for v in _tl_arr
            )
            print(f"\n  Token length per step  ({len(_tl_lens)} tokens):")
            print(f"  mean={_tl_arr.mean():.2f}  max={int(_tl_arr.max())}  "
                  f"min={int(_tl_arr.min())}  std={_tl_arr.std():.2f}")
            print(f"  {_tl_spark}")
            _tl_buckets = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for _tll in _tl_lens:
                _tl_buckets[min(5, max(1, _tll))] = _tl_buckets.get(min(5, max(1, _tll)), 0) + 1
            print(f"  Length distribution: " +
                  "  ".join(f"len{k}:{v}" for k, v in sorted(_tl_buckets.items())))
            continue

        if low.startswith("scorevarplot"):
            # scorevarplot [N] — sparkline of per-step within-step score variance
            _svp_res = getattr(model, "_last_gen_result", None)
            if _svp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _svp_vals = _svp_res.get("score_var_steps", [])
            _svp_toks = _svp_res.get("tokens", [])
            if not _svp_vals:
                print("  No score-variance data in last result."); continue
            _svp_parts = low.split()
            _svp_n = (int(_svp_parts[1]) if len(_svp_parts) > 1
                      and _svp_parts[1].isdigit() else len(_svp_vals))
            import numpy as _np_svp
            _svp_arr = _np_svp.array(_svp_vals[:_svp_n], dtype=float)
            _svp_max = float(_svp_arr.max()) if _svp_arr.max() > 0 else 1.0
            _SPARKS_SVP = " ▁▂▃▄▅▆▇█"
            _svp_spark = "".join(
                _SPARKS_SVP[min(8, int(v / _svp_max * 8))] for v in _svp_arr
            )
            _svp_mean  = _svp_res.get("scorevar", 0.0)
            print(f"\n  Score variance per step  ({len(_svp_arr)} steps):")
            print(f"  mean={_svp_mean:.5f}  max={_svp_arr.max():.5f}  "
                  f"min={_svp_arr.min():.5f}  std={_svp_arr.std():.5f}")
            print(f"  {_svp_spark}")
            print(f"  {'Step':<5}  {'Token':<18}  ScoreVar  Bar")
            for _svi, _svv in enumerate(_svp_arr):
                _svt  = _svp_toks[_svi] if _svi < len(_svp_toks) else "?"
                _svbl = int(20 * _svv / _svp_max)
                _svbar = "█" * _svbl + "░" * (20 - _svbl)
                print(f"  {_svi:<5}  {_svt[:18]:<18}  {_svv:.5f}  {_svbar}")
            continue

        if low.startswith("slopechart"):
            # slopechart [N] — per-step confidence slope (1st derivative) sparkline
            _sc_res = getattr(model, "_last_gen_result", None)
            if _sc_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _sc_sl  = _sc_res.get("confslope_steps", [])
            _sc_tok = _sc_res.get("tokens",          [])
            if len(_sc_sl) < 2:
                print("  Need ≥2 conf_ema steps."); continue
            _sc_parts = low.split()
            _sc_n = (int(_sc_parts[1]) if len(_sc_parts) > 1
                     and _sc_parts[1].isdigit() else len(_sc_sl))
            _sc_sl = _sc_sl[:_sc_n]
            import numpy as _np_sc
            _sc_arr = _np_sc.array(_sc_sl, dtype=float)
            _sc_spark = "".join(
                ("▲" if v > 0.001 else ("▼" if v < -0.001 else "─"))
                for v in _sc_arr
            )
            print(f"\n  Confidence slope (▲=rising ─=flat ▼=falling)  "
                  f"({len(_sc_sl)} steps):")
            print(f"  {_sc_spark}")
            print(f"  mean={_sc_arr.mean():+.5f}  std={_sc_arr.std():.5f}  "
                  f"max={_sc_arr.max():+.5f}  min={_sc_arr.min():+.5f}")
            _sc_rises = sum(1 for v in _sc_sl if v > 0.001)
            _sc_falls = sum(1 for v in _sc_sl if v < -0.001)
            _sc_flat  = len(_sc_sl) - _sc_rises - _sc_falls
            print(f"  ▲ rising:{_sc_rises}  ─ flat:{_sc_flat}  ▼ falling:{_sc_falls}  "
                  f"(net:{_sc_rises - _sc_falls:+d})")
            continue

        if low.startswith("cohaccel"):
            # cohaccel — ▲/─/▼ sparkline of coherence 2nd derivative
            _ca2_res = getattr(model, "_last_gen_result", None)
            if _ca2_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ca2_acc = _ca2_res.get("coh_acc",  [])
            _ca2_tok = _ca2_res.get("tokens",   [])
            if len(_ca2_acc) < 3:
                print("  Need ≥3 coh3 steps."); continue
            import numpy as _np_ca2
            _ca2_arr = _np_ca2.array(_ca2_acc, dtype=float)
            _ca2_spark = "".join(
                ("▲" if v > 0.001 else ("▼" if v < -0.001 else "─"))
                for v in _ca2_arr
            )
            print(f"\n  Coherence acceleration (▲=accel ─=flat ▼=decel):")
            print(f"  {_ca2_spark}")
            print(f"  mean={_ca2_arr.mean():+.5f}  std={_ca2_arr.std():.5f}  "
                  f"max={_ca2_arr.max():+.5f}  min={_ca2_arr.min():+.5f}")
            _ca2_pos = sorted(
                [(i, v) for i, v in enumerate(_ca2_arr) if v > 0.001],
                key=lambda x: x[1], reverse=True
            )[:5]
            _ca2_neg = sorted(
                [(i, v) for i, v in enumerate(_ca2_arr) if v < -0.001],
                key=lambda x: x[1]
            )[:5]
            for _lbl2, _lst2 in [("Top accel", _ca2_pos), ("Top decel", _ca2_neg)]:
                if not _lst2: continue
                print(f"  {_lbl2}:")
                for _cai, _cav in _lst2:
                    _cat = _ca2_tok[_cai] if _cai < len(_ca2_tok) else "?"
                    print(f"    step{_cai:<4}  {_cat[:16]:<16}  {_cav:+.5f}")
            continue

        if low.startswith("accelplot"):
            # accelplot [N] — sparkline of confidence 2nd derivative (acceleration)
            _ap_res = getattr(model, "_last_gen_result", None)
            if _ap_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ap_acc = _ap_res.get("conf_acc", [])
            _ap_tok = _ap_res.get("tokens",   [])
            if len(_ap_acc) < 3:
                print("  Need ≥3 steps of conf_ema data."); continue
            _ap_parts = low.split()
            _ap_n = (int(_ap_parts[1]) if len(_ap_parts) > 1
                     and _ap_parts[1].isdigit() else len(_ap_acc))
            _ap_acc = _ap_acc[:_ap_n]
            import numpy as _np_ap
            _ap_arr  = _np_ap.array(_ap_acc, dtype=float)
            _ap_max  = max(float(abs(_ap_arr).max()), 1e-9)
            _AP_POS  = "▲"
            _AP_NEG  = "▼"
            _ap_spark = "".join(
                (_AP_POS if v > 0.001 else (_AP_NEG if v < -0.001 else "─"))
                for v in _ap_arr
            )
            print(f"\n  Confidence acceleration (▲=accel ─=flat ▼=decel):")
            print(f"  {_ap_spark}")
            print(f"  mean={_ap_arr.mean():+.5f}  "
                  f"std={_ap_arr.std():.5f}  "
                  f"max={_ap_arr.max():+.5f}  min={_ap_arr.min():+.5f}")
            # Show top-5 acceleration and deceleration events
            _ap_sorted_pos = sorted(
                [(i, v) for i, v in enumerate(_ap_arr) if v > 0.001],
                key=lambda x: x[1], reverse=True
            )[:5]
            _ap_sorted_neg = sorted(
                [(i, v) for i, v in enumerate(_ap_arr) if v < -0.001],
                key=lambda x: x[1]
            )[:5]
            for _lbl, _lst in [("Top accel", _ap_sorted_pos), ("Top decel", _ap_sorted_neg)]:
                if not _lst: continue
                print(f"  {_lbl}:")
                for _api, _apv in _lst:
                    _apt = _ap_tok[_api] if _api < len(_ap_tok) else "?"
                    print(f"    step{_api:<4}  {_apt[:16]:<16}  {_apv:+.5f}")
            continue

        if low.startswith("cohrank"):
            # cohrank — rank tokens by the coh3-window value they appeared in
            _cr2_res = getattr(model, "_last_gen_result", None)
            if _cr2_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cr2_c3s  = _cr2_res.get("coh3_steps", [])
            _cr2_toks = _cr2_res.get("tokens",      [])
            if not _cr2_c3s or not _cr2_toks:
                print("  Not enough data."); continue
            # Each token at step i gets assigned coh3_steps[i] (or 0 if missing)
            _cr2_pairs = [
                (t, _cr2_c3s[i] if i < len(_cr2_c3s) else 0.0)
                for i, t in enumerate(_cr2_toks)
            ]
            _cr2_sorted = sorted(_cr2_pairs, key=lambda x: x[1], reverse=True)
            print(f"\n  Token coherence ranking  ({len(_cr2_sorted)} tokens):")
            print(f"  {'Rank':<5}  {'Token':<20}  Coh3      Bar")
            _cr2_max = max(max(v for _, v in _cr2_sorted), 1e-9)
            for _rk, (_crt, _crv) in enumerate(_cr2_sorted[:25], 1):
                _crbl = int(16 * max(_crv, 0) / _cr2_max)
                _crbar = "█" * _crbl + "░" * (16 - _crbl)
                print(f"  {_rk:<5}  {_crt[:20]:<20}  {_crv:+.4f}   {_crbar}")
            if len(_cr2_sorted) > 25:
                print(f"  … and {len(_cr2_sorted)-25} more")
            continue

        if low.startswith("smoothplot"):
            # smoothplot [N] — raw conf_ema vs 3-pt smoothed overlay sparklines
            _smp_res = getattr(model, "_last_gen_result", None)
            if _smp_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _smp_raw = _smp_res.get("conf_ema_steps", [])
            _smp_sm  = _smp_res.get("conf_smooth",    [])
            if not _smp_raw:
                print("  No conf_ema data."); continue
            _smp_parts = low.split()
            _smp_n = (int(_smp_parts[1]) if len(_smp_parts) > 1
                      and _smp_parts[1].isdigit() else len(_smp_raw))
            _smp_raw = _smp_raw[:_smp_n]
            _smp_sm  = _smp_sm[:_smp_n]
            _SPARKS_SM = " ▁▂▃▄▅▆▇█"
            _smp_max = max(max(_smp_raw), 1e-9)
            _smp_spark_r = "".join(
                _SPARKS_SM[min(8, int(v / _smp_max * 8))] for v in _smp_raw
            )
            _smp_spark_s = "".join(
                _SPARKS_SM[min(8, int(v / _smp_max * 8))] for v in _smp_sm
            )
            import numpy as _np_smp
            _smp_arr = _np_smp.array(_smp_raw, dtype=float)
            _smp_sarr = _np_smp.array(_smp_sm,  dtype=float)
            print(f"\n  Conf EMA vs smoothed (3-pt MA)  ({len(_smp_raw)} steps):")
            print(f"  Raw:      {_smp_spark_r}")
            print(f"  Smoothed: {_smp_spark_s}")
            print(f"  raw  avg={_smp_arr.mean():.4f}  "
                  f"std={_smp_arr.std():.4f}  "
                  f"min={_smp_arr.min():.4f}  max={_smp_arr.max():.4f}")
            print(f"  smth avg={_smp_sarr.mean():.4f}  "
                  f"std={_smp_sarr.std():.4f}  "
                  f"Δstd={_smp_sarr.std() - _smp_arr.std():+.4f}  "
                  f"(−= smoother)")
            continue

        if low.startswith("diagsummary"):
            # diagsummary — all key metrics in one compact diagnostic block
            _ds_res = getattr(model, "_last_gen_result", None)
            if _ds_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _ds_keys = [
                ("Text",         lambda r: r.get("text","")[:55]),
                ("Tokens",       lambda r: str(len(r.get("tokens",[])))),
                ("Stop",         lambda r: r.get("stop_reason","?")),
                ("ConfEMA",      lambda r: f"{r.get('conf_ema_final',0):.4f}"),
                ("Coherence",    lambda r: f"{r.get('coherence',0):+.4f}"),
                ("Coh3",         lambda r: f"{r.get('coh3',0):+.4f}"),
                ("CohTrend",     lambda r: f"{r.get('coh_trend',0):+.5f}"),
                ("FlowScore",    lambda r: f"{r.get('flow_score',0):.2f}"),
                ("Best@",        lambda r: (
                    f"step{r.get('bestseg',{}).get('start','?')}  "
                    f"val={r.get('bestseg',{}).get('val',0):.4f}"
                )),
                ("Worst@",       lambda r: (
                    f"step{r.get('worstseg',{}).get('start','?')}  "
                    f"val={r.get('worstseg',{}).get('val',0):.4f}"
                )),
                ("Fluency",      lambda r: f"{r.get('fluency',0):.3f}"),
                ("PPL",          lambda r: f"{r.get('pseudo_ppl',0):.1f}"),
                ("Rhythm",       lambda r: f"{r.get('rhythm_score',0):.3f}"),
                ("EntRatio",     lambda r: f"{r.get('entropy_ratio',1):.3f}"),
                ("VelRatio",     lambda r: f"{r.get('vel_ratio',1):.3f}"),
                ("TopKΔ",        lambda r: f"{r.get('topkdelta',0):.3f}"),
                ("VPrecEMA",     lambda r: f"{r.get('vocab_prec_ema',0):.4f}"),
                ("ConfVar",      lambda r: f"{r.get('conf_variance',0):.4f}"),
                ("ConfTrend",    lambda r: f"{r.get('conf_trend',0):+.5f}"),
                ("SGSlope",      lambda r: f"{r.get('sg_slope',0):+.5f}"),
                ("Spread",       lambda r: f"{r.get('token_embed_spread',0):.4f}"),
                ("Declining",    lambda r: str(r.get('conf_declining', False))),
                ("PromptType",   lambda r: r.get("prompt_type","?")),
            ]
            print(f"\n  ── DIAG SUMMARY ──────────────────────────────────")
            for _ds_lbl, _ds_fn in _ds_keys:
                try:
                    _ds_val = _ds_fn(_ds_res)
                except Exception:
                    _ds_val = "err"
                print(f"  {_ds_lbl:<14}  {_ds_val}")
            print(f"  ──────────────────────────────────────────────────")
            continue

        if low.startswith("confpeak"):
            # confpeak — show peak and valley confidence moments with their tokens
            _cpk_res = getattr(model, "_last_gen_result", None)
            if _cpk_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cpk_toks  = _cpk_res.get("tokens",        [])
            _cpk_confs = _cpk_res.get("confidences",   [])
            _cpk_pstep = _cpk_res.get("peak_conf_step", 0)
            _cpk_pval  = _cpk_res.get("peak_conf_val",  0.0)
            _cpk_mstep = _cpk_res.get("min_conf_step",  0)
            _cpk_mval  = _cpk_res.get("min_conf_val",   0.0)
            if not _cpk_confs:
                print("  No confidence data."); continue
            _cpk_avg = sum(_cpk_confs) / len(_cpk_confs)
            print(f"\n  Confidence peak & valley:")
            print(f"  avg={_cpk_avg:.4f}  "
                  f"peak=step{_cpk_pstep} val={_cpk_pval:.4f}  "
                  f"valley=step{_cpk_mstep} val={_cpk_mval:.4f}")
            # Show context window around peak
            _SPARKS_CPK = " ▁▂▃▄▅▆▇█"
            _cpk_spark = "".join(
                _SPARKS_CPK[min(8, int(c * 8))] for c in _cpk_confs
            )
            print(f"  {_cpk_spark}")
            # Show ±2 tokens around peak
            _cpk_win = 2
            for _lbl, _si in [("Peak", _cpk_pstep), ("Valley", _cpk_mstep)]:
                _lo = max(0, _si - _cpk_win)
                _hi = min(len(_cpk_toks), _si + _cpk_win + 1)
                _ctx_toks  = _cpk_toks[_lo:_hi]
                _ctx_confs = _cpk_confs[_lo:_hi]
                _ctx_str = "  ".join(
                    f"{'→' if _lo + i == _si else ''}{t}({c:.3f})"
                    for i, (t, c) in enumerate(zip(_ctx_toks, _ctx_confs))
                )
                print(f"  {_lbl:7}@step{_si}: {_ctx_str}")
            continue

        if low.startswith("trendcompare"):
            # trendcompare <pA> | <pB> — run two prompts, compare trend metrics
            _tc2_sep  = " | "
            _tc2_rest = cmd[len("trendcompare"):].strip()
            if _tc2_sep not in _tc2_rest:
                print("  Usage: trendcompare <prompt A> | <prompt B>"); continue
            _tc2_pa, _tc2_pb = [p.strip() for p in _tc2_rest.split(_tc2_sep, 1)]
            if not _tc2_pa or not _tc2_pb:
                print("  Both prompts must be non-empty."); continue
            import numpy as _np_tc2
            print(f"\n  Trendcompare:")
            print(f"  A: '{_tc2_pa[:45]}'")
            print(f"  B: '{_tc2_pb[:45]}'")
            _tc2_ra = model.causal_generate(_tc2_pa, max_tokens=16)
            _tc2_rb = model.causal_generate(_tc2_pb, max_tokens=16)
            _tc2_keys = [
                ("ConfEMA",   "conf_ema_final",  ".4f"),
                ("Coh",       "coherence",       "+.3f"),
                ("C3",        "coh3",            "+.3f"),
                ("CohTrend",  "coh_trend",       "+.5f"),
                ("Flow",      "flow_score",      ".2f"),
                ("PPL",       "pseudo_ppl",      ".1f"),
                ("Rhythm",    "rhythm_score",    ".3f"),
                ("EntRatio",  "entropy_ratio",   ".2f"),
                ("VelRatio",  "vel_ratio",       ".2f"),
            ]
            print(f"\n  {'Metric':<12}  {'A':<12}  {'B':<12}  Winner")
            for _tc2lbl, _tc2k, _tc2fmt in _tc2_keys:
                _tc2a = _tc2_ra.get(_tc2k, 0.0)
                _tc2b = _tc2_rb.get(_tc2k, 0.0)
                _tc2_win = "A" if float(_tc2a) > float(_tc2b) else "B"
                if _tc2k == "pseudo_ppl":  # lower is better for PPL
                    _tc2_win = "A" if float(_tc2a) < float(_tc2b) else "B"
                _ta_str = format(float(_tc2a), _tc2fmt.lstrip('+'))
                _tb_str = format(float(_tc2b), _tc2fmt.lstrip('+'))
                print(f"  {_tc2lbl:<12}  {_ta_str:<12}  {_tb_str:<12}  {_tc2_win}")
            print(f"\n  A text: {_tc2_ra.get('text','')[:55]}")
            print(f"  B text: {_tc2_rb.get('text','')[:55]}")
            continue

        if low.startswith("qualmap"):
            # qualmap — per-step composite quality heat map (conf × coh3)
            _qm_res = getattr(model, "_last_gen_result", None)
            if _qm_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _qm_confs = _qm_res.get("confidences",  [])
            _qm_c3s   = _qm_res.get("coh3_steps",   [])
            _qm_toks  = _qm_res.get("tokens",        [])
            if not _qm_confs or not _qm_c3s:
                print("  Not enough data."); continue
            import numpy as _np_qm
            _qm_n = min(len(_qm_confs), len(_qm_c3s), len(_qm_toks))
            _qm_qual = [
                float(_qm_confs[i]) * max(float(_qm_c3s[i]), 0.0)
                for i in range(_qm_n)
            ]
            _qm_max  = max(max(_qm_qual), 1e-9)
            _QM_HEAT = " ░▒▓█"
            _qm_spark = "".join(
                _QM_HEAT[min(4, int(_v / _qm_max * 4))]
                for _v in _qm_qual
            )
            _qm_avg = sum(_qm_qual) / max(len(_qm_qual), 1)
            print(f"\n  Quality heat map (conf × coh3 per step):")
            print(f"  {_qm_spark}  avg={_qm_avg:.4f}  max={_qm_max:.4f}")
            print(f"  {'Step':<5}  {'Token':<18}  Conf    Coh3    Quality  Bar")
            for _qmi in range(_qm_n):
                _qv = _qm_qual[_qmi]
                _qbl = int(20 * _qv / _qm_max)
                _qbar = "█" * _qbl + "░" * (20 - _qbl)
                print(f"  {_qmi:<5}  {_qm_toks[_qmi][:18]:<18}  "
                      f"{_qm_confs[_qmi]:.4f}  {_qm_c3s[_qmi]:+.4f}  "
                      f"{_qv:.4f}   {_qbar}")
            continue

        if low.startswith("sgplot"):
            # sgplot [N] — sparkline of per-step score-gap EMA
            _sg_res = getattr(model, "_last_gen_result", None)
            if _sg_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _sg_vals = _sg_res.get("sg_step_gaps", [])
            _sg_toks = _sg_res.get("tokens", [])
            if not _sg_vals:
                print("  No score-gap data in last result."); continue
            _sg_parts = low.split()
            _sg_n = (int(_sg_parts[1]) if len(_sg_parts) > 1
                     and _sg_parts[1].isdigit() else len(_sg_vals))
            import numpy as _np_sg
            _sg_arr  = _np_sg.array(_sg_vals[:_sg_n], dtype=float)
            _sg_max  = float(_sg_arr.max()) if _sg_arr.max() > 0 else 1.0
            _SPARKS_SG = " ▁▂▃▄▅▆▇█"
            _sg_spark = "".join(
                _SPARKS_SG[min(8, int(_v / _sg_max * 8))] for _v in _sg_arr
            )
            _sg_slope = float(_np_sg.polyfit(
                _np_sg.arange(len(_sg_arr)), _sg_arr, 1)[0])
            print(f"\n  Score-gap EMA per step ({len(_sg_arr)} steps):")
            print(f"  {_sg_spark}")
            print(f"  avg={_sg_arr.mean():.4f}  min={_sg_arr.min():.4f}  "
                  f"max={_sg_arr.max():.4f}  slope={_sg_slope:+.5f}")
            print(f"  {'Step':<5}  {'Token':<18}  Gap     Bar")
            for _sgi, (_sgv, _sgt) in enumerate(
                    zip(_sg_vals[:_sg_n], _sg_toks + [""] * _sg_n)):
                _sgblen = int(20 * _sgv / (_sg_max + 1e-9))
                _sgbar  = "█" * _sgblen + "░" * (20 - _sgblen)
                print(f"  {_sgi:<5}  {_sgt[:18]:<18}  {_sgv:.4f}  {_sgbar}")
            continue

        if low.startswith("confrise"):
            # confrise [N] — steps where conf_ema rose fastest (sharpest improvement)
            _cr_res = getattr(model, "_last_gen_result", None)
            if _cr_res is None:
                print("  No generation yet. Run a prompt first."); continue
            _cr_emas = _cr_res.get("conf_ema_steps", [])
            _cr_toks = _cr_res.get("tokens", [])
            if len(_cr_emas) < 2:
                print("  Not enough conf_ema data."); continue
            import numpy as _np_cr
            _cr_arr  = _np_cr.array(_cr_emas, dtype=float)
            _cr_diffs = _np_cr.diff(_cr_arr)  # step-on-step change
            _cr_mean = float(_cr_diffs.mean())
            _cr_std  = float(_cr_diffs.std())
            _cr_thr  = _cr_mean + _cr_std
            _cr_parts = low.split()
            _cr_n = (int(_cr_parts[1]) if len(_cr_parts) > 1
                     and _cr_parts[1].isdigit() else len(_cr_diffs))
            print(f"\n  Confidence-rise detector  (threshold = mean+1σ = {_cr_thr:+.4f})")
            print(f"  mean_Δ={_cr_mean:+.4f}  std_Δ={_cr_std:.4f}  steps={len(_cr_diffs)}")
            _cr_rises = [(i + 1, d) for i, d in enumerate(_cr_diffs[:_cr_n])
                         if d > _cr_thr]
            if not _cr_rises:
                print("  No sharp rises detected — confidence was smooth.")
            else:
                print(f"  {'Step':<6}  {'Token':<18}  {'Δ EMA':<10}  Bar")
                for _cri, _crd in _cr_rises:
                    _crt = _cr_toks[_cri] if _cri < len(_cr_toks) else "?"
                    _crb = "▲" * min(8, max(1, int(_crd / (_cr_std + 1e-9) * 3)))
                    print(f"  {_cri:<6}  {_crt[:18]:<18}  {_crd:+.4f}    {_crb}")
            continue

        if low.startswith("multiprofile"):
            # multiprofile — run 3 preset prompts, compare all metrics in a table
            import numpy as _np_mp
            _mp_prompts = [
                "language emerges from",
                "intelligence requires the ability to",
                "meaning arises from",
            ]
            print(f"\n  Multiprofile: {len(_mp_prompts)} preset prompts")
            print(f"  {'#':<3}  {'ConfEMA':<8}  {'Coh':<6}  {'C3':<6}  "
                  f"{'CohTrend':<10}  {'Flu':<5}  {'PPL':<7}  "
                  f"{'Rhythm':<7}  {'Toks':<5}  Prompt")
            _mp_rows = []
            for _mpi, _mpp in enumerate(_mp_prompts):
                _mpr = model.causal_generate(_mpp, max_tokens=16)
                _mp_rows.append(_mpr)
                _mpc  = _mpr.get("conf_ema_final", 0.0)
                _mpco = _mpr.get("coherence",       0.0)
                _mpc3 = _mpr.get("coh3",            0.0)
                _mpct = _mpr.get("coh_trend",       0.0)
                _mpfl = _mpr.get("fluency",         0.0)
                _mppp = _mpr.get("pseudo_ppl",      0.0)
                _mprh = _mpr.get("rhythm_score",    0.0)
                _mpn  = len(_mpr.get("tokens",      []))
                print(f"  {_mpi+1:<3}  {_mpc:<8.4f}  {_mpco:<6.3f}  {_mpc3:<6.3f}  "
                      f"{_mpct:<10.4f}  {_mpfl:<5.2f}  {_mppp:<7.1f}  "
                      f"{_mprh:<7.3f}  {_mpn:<5}  '{_mpp[:28]}'")
            # Summary row
            _mp_avg_conf  = _np_mp.mean([r.get("conf_ema_final", 0) for r in _mp_rows])
            _mp_avg_coh   = _np_mp.mean([r.get("coherence", 0)      for r in _mp_rows])
            _mp_avg_flu   = _np_mp.mean([r.get("fluency",   0)      for r in _mp_rows])
            _mp_avg_rh    = _np_mp.mean([r.get("rhythm_score", 0)   for r in _mp_rows])
            print(f"\n  avg  conf={float(_mp_avg_conf):.4f}  "
                  f"coh={float(_mp_avg_coh):.3f}  "
                  f"flu={float(_mp_avg_flu):.3f}  "
                  f"rhythm={float(_mp_avg_rh):.3f}")
            # Show generated texts
            print(f"\n  Outputs:")
            for _mpi, (_mpp, _mpr) in enumerate(zip(_mp_prompts, _mp_rows)):
                print(f"  {_mpi+1}. [{_mpp[:20]}] → {_mpr.get('text','')[:55]}")
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
