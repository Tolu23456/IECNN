"""
IECNN Pipeline — full 10-layer architecture (v6 SOTA).
"""

import numpy as np
import re
import regex
import unicodedata
import time
from typing import List, Dict, Optional, Tuple, Any
import sys, os
import ctypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── OpenMP thread configuration ───────────────────────────────────────────────
# Must be set before the C shared library first creates OMP threads.
# Defaults to all logical cores; caller can override env before importing.
import multiprocessing as _mp
os.environ.setdefault("OMP_NUM_THREADS", str(_mp.cpu_count() or 6))
os.environ.setdefault("OMP_PROC_BIND",   "close")
os.environ.setdefault("OMP_WAIT_POLICY", "active")

from basemapping.basemapping import BaseMapper, BaseMap, EMBED_DIM, FEATURE_DIM
from neural_dot.neural_dot   import BiasVector, DotGenerator, DotType, N_HEADS
from aim.aim                 import AIMLayer
from convergence.convergence import ConvergenceLayer, Cluster
from pruning.pruning         import PruningLayer
from iteration.iteration     import IterationController, StopReason
from memory.dot_memory       import DotMemory
from memory.cluster_memory   import ClusterMemory
from evolution.dot_evolution import DotEvolution, EvolutionConfig
from evaluation.metrics      import IECNNMetrics, RunMetrics
from formulas.formulas       import (
    similarity_score, cluster_entropy, adaptive_learning_rate, dominance_score,
    amplify_pressure, hierarchical_convergence_score, _fp
)

# ── Load C shared library ────────────────────────────────────────────
_lib_p = None

def _load_lib_p():
    global _lib_p
    if _lib_p is not None:
        return _lib_p
    here = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, "pipeline_c.so")
    if not os.path.exists(so_path):
        print(
            f"[IECNN] WARNING: {so_path} not found — pipeline will use slow Python path.\n"
            f"         Fix: run  python main.py build  to compile C extensions.",
            file=sys.stderr,
        )
        return _lib_p
    if os.path.exists(so_path):
        try:
            _lib_p = ctypes.CDLL(so_path)
            _lib_p.pipeline_run_c.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint),
                ctypes.c_float, ctypes.c_float, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,   # training_mode
            ]
            _lib_p.pipeline_run_c.restype = None
            _lib_p.pipeline_batch_run_c.argtypes = [
                ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_uint),
                ctypes.c_float, ctypes.c_float, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,   # training_mode
            ]
            _lib_p.pipeline_batch_run_c.restype = None
        except Exception:
            _lib_p = None
    return _lib_p


class IECNNResult:
    """Complete result from one IECNN run."""
    def __init__(self, output, basemap, top_cluster, summary, stop_reason,
                 rounds, metrics=None):
        self.output       = output
        self.basemap      = basemap
        self.top_cluster  = top_cluster
        self.summary      = summary
        self.stop_reason  = stop_reason
        self.rounds       = rounds
        self.metrics: Optional[RunMetrics] = metrics

    def __repr__(self):
        n_rnd = self.summary.get("rounds", 0)
        return (f"IECNNResult(rounds={n_rnd}, stop='{self.stop_reason}', "
                f"norm={np.linalg.norm(self.output):.3f})")


class IECNN:
    """
    Iterative Emergent Convergent Neural Network.
    """

    def __init__(self,
                 feature_dim:           int   = FEATURE_DIM,
                 num_dots:              int   = 128,
                 n_heads:               int   = N_HEADS,
                 max_iterations:        int   = 12,
                 micro_threshold:       float = 0.25,
                 macro_threshold:       float = 0.15,
                 dominance_threshold:   float = 0.35,
                 novelty_threshold:     float = 0.05,
                 stability_threshold:   float = 0.99,
                 alpha:                 float = 0.70,
                 gamma:                 float = 0.30,
                 soft_conf:             float = 0.06,
                 max_aim_variants:      int   = 4,
                 base_lr:               float = 0.10,
                 evolve:                bool  = True,
                 persistence_path:      Optional[str] = None,
                 phase_coding:          bool  = False,
                 seed:                  int   = 42):
        self.feature_dim = feature_dim
        self.num_dots    = num_dots
        self.n_heads     = n_heads
        self.alpha       = alpha
        self.seed        = seed
        self.do_evolve   = evolve
        self.phase_coding = phase_coding

        # Core components
        self.base_mapper = BaseMapper(feature_dim=feature_dim)
        self.persistence_path = persistence_path
        if persistence_path:
            self.base_mapper.load(persistence_path)
            self.base_mapper.persistence_path = persistence_path

        self.base_bias   = BiasVector(0.5, 0.5, 0.5, 0.3, 1.0)
        self.dot_gen     = DotGenerator(num_dots, feature_dim, self.base_bias,
                                        n_heads=n_heads, seed=seed)
        self.aim         = AIMLayer(max_aim_variants, seed)
        self.convergence = ConvergenceLayer(
            micro_threshold=micro_threshold, macro_threshold=macro_threshold,
            alpha=alpha, gamma=gamma, dominance_threshold=dominance_threshold,
            cross_type_bonus=0.15,
        )
        self.pruning     = PruningLayer(
            soft_conf=soft_conf, max_aim_per_dot=max_aim_variants, alpha=alpha
        )
        self.iter_ctrl   = IterationController(
            max_iterations=max_iterations, dominance_threshold=dominance_threshold,
            novelty_threshold=novelty_threshold, stability_threshold=stability_threshold,
            base_lr=base_lr, alpha=alpha,
        )

        # Memory and evolution
        self.dot_memory      = DotMemory(num_dots)
        if phase_coding:
            self.dot_memory.phase_bonus_weight = 0.50
        self.cluster_memory  = ClusterMemory(feature_dim, phase_coding=phase_coding)
        self.evolution       = DotEvolution(EvolutionConfig(), seed)
        self.evaluator       = IECNNMetrics(alpha)

        self.working_memory: Optional[np.ndarray] = None
        self.context_map: Optional[np.ndarray] = None
        self.narrative_base: Optional[np.ndarray] = None

        self._dots: Optional[list] = None
        self._call_count: int = 0
        self._rng = np.random.RandomState(seed)

        # LRU ctx cache: text → mean-pooled basemap float32 vector (dim,)
        # Eliminates re-transforming repeated causal prefixes (major speedup).
        self._ctx_cache: Dict[str, np.ndarray] = {}

        # Token embedding cache: token → normalised float32 (dim,)
        # Across a 200-sentence batch, target tokens repeat constantly (same
        # common words appear in many positions).  Caching avoids re-running
        # _token_embedding (complex polynomial + normalisation) per duplicate.
        self._tok_emb_cache: Dict[str, np.ndarray] = {}

        if persistence_path:
            try:
                self.load_brain(persistence_path)
            except Exception:
                pass

    def save_brain(self, persistence_path: Optional[str] = None) -> None:
        import pickle
        from neural_dot.neural_dot import _NEXT_DOT_ID
        path = persistence_path or self.persistence_path
        if not path: return
        self.base_mapper.save(path)
        p = self._learned_paths(path)
        if self._dots is not None:
            with open(p["dots"], "wb") as f: pickle.dump(self._dots, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(p["dotmem"], "wb") as f: pickle.dump(self.dot_memory.state_dict(), f)
        with open(p["clustmem"], "wb") as f: pickle.dump(self.cluster_memory.state_dict(), f)
        with open(p["evo"], "wb") as f: pickle.dump(self.evolution.state_dict(), f)
        with open(p["meta"], "wb") as f: pickle.dump({"call_count": self._call_count, "next_dot_id": _NEXT_DOT_ID}, f)

    def load_brain(self, persistence_path: Optional[str] = None) -> None:
        import pickle
        from neural_dot.neural_dot import set_next_dot_id
        path = persistence_path or self.persistence_path
        if not path: return
        p = self._learned_paths(path)
        if os.path.exists(p["meta"]):
            with open(p["meta"], "rb") as f:
                m = pickle.load(f)
                self._call_count = m.get("call_count", 0)
                set_next_dot_id(m.get("next_dot_id", 1000))
        if os.path.exists(p["dots"]):
            with open(p["dots"], "rb") as f: self._dots = pickle.load(f)
        if os.path.exists(p["dotmem"]):
            with open(p["dotmem"], "rb") as f: self.dot_memory.load_state(pickle.load(f))
        if os.path.exists(p["clustmem"]):
            with open(p["clustmem"], "rb") as f: self.cluster_memory.load_state(pickle.load(f))
        if os.path.exists(p["evo"]):
            with open(p["evo"], "rb") as f: self.evolution.load_state(pickle.load(f))

    def _learned_paths(self, base: str) -> dict:
        return {k: f"{base}.{k}.pkl" for k in ["dots", "dotmem", "clustmem", "evo", "meta"]}

    def _ensure_dots(self) -> list:
        if self._dots is None: self._dots = self.dot_gen.generate()
        return self._dots

    def run(self, input_data: Any, verbose: bool = False,
            mode: str = "text", causal: bool = False,
            use_c_pipeline: bool = True) -> IECNNResult:
        basemap = self.base_mapper.transform(input_data, mode=mode)
        if self.working_memory is not None:
            basemap = self._blend(basemap, self.working_memory, alpha=0.50)

        dots = self._ensure_dots()

        # Pipeline-C
        # Bypassed when phase_coding=True: the C kernel is real-valued only
        # and cannot propagate complex phase information through the forward pass.
        lib_p = _load_lib_p()
        if use_c_pipeline and lib_p and not verbose and not self.phase_coding:
            self.dot_gen._ensure_caches(dots)
            BM_mat = np.ascontiguousarray(np.real(basemap.matrix), dtype=np.float32)
            out_v = np.zeros(self.feature_dim, dtype=np.float32)
            assign = np.zeros(len(dots) * 8, dtype=np.int32)

            lib_p.pipeline_run_c(
                len(dots), self.feature_dim, BM_mat.shape[0],
                _fp(BM_mat)[0], _fp(self.dot_gen._cached_W_stack)[0],
                _fp(self.dot_gen._cached_HP_stack)[0], _fp(self.dot_gen._cached_B_stack)[0],
                _fp(self.dot_gen._cached_BIAS)[0],
                self.dot_gen._cached_TYPES.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                self.dot_gen._cached_N_HEADS.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                self.dot_gen._cached_SEEDS.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
                self.alpha, self.convergence.gamma, self.iter_ctrl.max_iterations,
                _fp(out_v)[0], assign.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                ctypes.c_int(0),   # training_mode=0 → full noise for inference diversity
            )

            # Outcome recording for C-pipeline
            for i, dot in enumerate(dots):
                for h in range(dot.n_heads):
                    idx = i * 8 + h
                    in_winner = (assign[idx] == 0)
                    self.dot_memory.record(dot.dot_id, out_v, in_winner)

            self._call_count += 1
            self.working_memory = out_v.copy()
            # Scope cluster memory to this call and commit the result pattern
            self.cluster_memory.reset_call()
            self.cluster_memory.commit_pattern(out_v, 1.0)
            return IECNNResult(out_v, basemap, None, {"rounds": 0}, "Pipeline-C", [])

        # Python Fallback (always used when phase_coding=True)
        # Clear per-call round snapshots so temporal_stability() measures THIS
        # run only, not whatever happened in the previous call.
        self.cluster_memory.reset_call()
        self.iter_ctrl.reset()
        cur_bmap = basemap
        surv: list = []
        while True:
            dot_preds = self.dot_gen.run_all(cur_bmap, dots, causal=causal)
            candidates = self.aim.transform(dot_preds, cur_bmap)
            filt, _ = self.pruning.stage1(candidates)
            clusters, assign = self.convergence.run(filt)
            surv, _ = self.pruning.stage3(clusters)
            if not surv: break

            # Record this round in cluster memory BEFORE advancing iter_ctrl
            # (so the round index in the snapshot matches iter_ctrl._round)
            round_centroids = [c.centroid for c in surv if c.centroid is not None]
            round_scores    = [c.score   for c in surv]
            self.cluster_memory.record_round(
                self.iter_ctrl._round, round_centroids, round_scores
            )

            self.iter_ctrl.record_round(surv, {})

            # Manual outcome recording for Python fallback.
            # assign maps indices into *filt* (post-pruning), not candidates.
            win_cid = surv[0].cluster_id
            for i, (_, _, info) in enumerate(filt):
                did = info.get("dot_id")
                if did is not None:
                    in_winner = bool(i < len(assign) and assign[i] == win_cid)
                    self.dot_memory.record(did, surv[0].centroid, in_winner)

            stop, reason = self.iter_ctrl.should_stop(surv)
            if stop: break
            refined = self.iter_ctrl.advance(surv, surv[0].centroid)
            cur_bmap = self._blend(basemap, refined)

        self._call_count += 1
        res_v = surv[0].centroid if (surv and surv[0].centroid is not None) else basemap.pool()
        self.working_memory = res_v.copy()
        summary = self.iter_ctrl.summary()
        # Commit the winning centroid to the cross-call pattern library
        if surv:
            self.cluster_memory.commit_pattern(res_v, float(surv[0].score))
        return IECNNResult(res_v, basemap, surv[0] if surv else None, summary, self.iter_ctrl.stop_reason or "Python", [])

    def fast_encode(self, text: str) -> np.ndarray:
        """
        Single-pass encoding without full convergence — designed for training.

        Does transform() → dot_gen.run_all() → weighted centroid of the top
        predictions.  Skips AIM inversion, the convergence loop, and pruning,
        so it is ~6-10x faster than run() at the cost of less-refined output.
        The signal is still good enough for next-token causal weight updates.
        """
        basemap = self.base_mapper.transform(text)
        dots    = self._ensure_dots()
        preds   = self.dot_gen.run_all(basemap, dots)
        if not preds:
            out = basemap.pool().astype(np.float32)
            n   = np.linalg.norm(out); return out / (n + 1e-10)
        # Weight by confidence; take top 16 predictions
        preds.sort(key=lambda x: x[1], reverse=True)
        top     = preds[:min(16, len(preds))]
        vecs    = np.stack([np.real(p[0]).astype(np.float32) for p in top])
        weights = np.array([max(p[1], 1e-10) for p in top], dtype=np.float32)
        weights /= weights.sum()
        out     = (weights[:, None] * vecs).sum(axis=0)
        n       = np.linalg.norm(out); return out / (n + 1e-10)

    def run_batch(self, inputs: List[str], use_c_pipeline: bool = True,
                  return_centroids: bool = False,
                  record_wins: bool = True) -> np.ndarray:
        """
        Batch forward pass through the IECNN pipeline.

        Parameters
        ----------
        inputs          : list of raw text sentences
        use_c_pipeline  : use the C batch kernel (fast); falls back to
                          fast_encode() when False or the .so is absent
        return_centroids: if True, return (out_v, ctxs) where ctxs is the
                          (n_sents, dim) array of per-sentence basemap
                          centroids — lets callers reuse them for per-dot
                          prediction without re-transforming the basemaps
        record_wins     : if False, skip writing to dot_memory (useful when
                          the caller will perform its own causal recording)
        """
        lib_p = _load_lib_p()
        if not use_c_pipeline or not lib_p:
            outs = np.stack([self.fast_encode(i) for i in inputs])
            if return_centroids:
                return outs, None
            return outs

        dots = self._ensure_dots()
        self.dot_gen._ensure_caches(dots)
        n_dots, dim, n_sents = len(dots), self.feature_dim, len(inputs)
        basemaps = [self.base_mapper.transform(inp) for inp in inputs]
        seq_lens = np.array([len(bm.tokens) for bm in basemaps], dtype=np.int32)
        offsets = np.zeros(n_sents, dtype=np.int32)
        for i in range(1, n_sents): offsets[i] = offsets[i-1] + seq_lens[i-1]
        flat_bm = np.ascontiguousarray(
            np.vstack([np.real(bm.matrix) for bm in basemaps]), dtype=np.float32
        )
        out_v   = np.zeros((n_sents, dim), dtype=np.float32)
        assigns = np.zeros((n_sents, n_dots * 8), dtype=np.int32)

        lib_p.pipeline_batch_run_c(
            n_sents, n_dots, dim, seq_lens.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            flat_bm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            _fp(self.dot_gen._cached_W_stack)[0], _fp(self.dot_gen._cached_HP_stack)[0],
            _fp(self.dot_gen._cached_B_stack)[0], _fp(self.dot_gen._cached_BIAS)[0],
            self.dot_gen._cached_TYPES.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.dot_gen._cached_N_HEADS.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.dot_gen._cached_SEEDS.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
            self.alpha, self.convergence.gamma, self.iter_ctrl.max_iterations,
            out_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            assigns.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(1),   # training_mode=1 → skip fast_randn (3x speedup)
        )

        # ── Per-sentence context centroids: (n_sents, dim) ────────────────────
        # mean-pool each sentence's basemap rows into a single context vector
        ctxs = np.stack([
            flat_bm[int(offsets[s]):int(offsets[s]) + max(int(seq_lens[s]), 1)].mean(axis=0)
            for s in range(n_sents)
        ])   # (n_sents, dim)

        if record_wins:
            # ── Vectorised per-dot win via W-matrix alignment ─────────────────
            #
            # New criterion: cosine_sim(W[i] @ ctx[s], out_v[s]) > 0
            #   W[i] is the dot's weight matrix (256×256).  W[i] @ ctx[s] is
            #   the dot's preferred direction given the sentence context.
            #   Positive cosine sim with the convergence output = "win".
            #
            #   By symmetry of Gaussian init, ~50 % of dots win per sentence.
            #   Dots that specialise on recurring contexts drift above 50 %
            #   → effectiveness > 0.5 → MaxEff > 0.5 after even one evolution.
            #
            # Implementation: explicit BLAS SGEMM via matmul (faster than
            # einsum for this tensor shape because BLAS is memory-bandwidth
            # optimal):
            #   W_flat = W_stack.reshape(n_dots*dim, dim)    (32768, 256)
            #   W_flat @ ctxs.T  →  (32768, n_sents) via SGEMM
            #   reshape + transpose  →  (n_sents, n_dots, dim)
            W_flat = self.dot_gen._cached_W_stack.reshape(n_dots * dim, dim)

            # (n_dots*dim, n_sents) via BLAS SGEMM
            dp_flat = W_flat @ ctxs.T                   # (32768, n_sents)
            # (n_sents, n_dots, dim)
            dot_preds = dp_flat.reshape(n_dots, dim, n_sents).transpose(2, 0, 1)

            # normalise dot predictions: (n_sents, n_dots, 1)
            pn   = np.linalg.norm(dot_preds, axis=2, keepdims=True).clip(1e-10)
            dp_n = dot_preds / pn                       # (n_sents, n_dots, dim)

            # normalise convergence outputs: (n_sents, dim)
            on    = np.linalg.norm(out_v, axis=1, keepdims=True).clip(1e-10)
            out_n = out_v / on

            # cosine sim via batch matmul: (n_sents, 1, dim) @ (n_sents, dim, n_dots)
            # = (n_sents, 1, n_dots) → squeeze → (n_sents, n_dots)
            sims       = np.squeeze(out_n[:, None, :] @ dp_n.transpose(0, 2, 1), 1)
            in_winners = sims > 0.0                     # (n_sents, n_dots) bool

            # ── batch_record replaces the per-dot Python loop ─────────────────
            # One call per sentence: 128-iteration inner loop vs 128 function
            # calls with hash lookups, optional-param evaluation, etc.
            dot_ids = [d.dot_id for d in dots]
            for s in range(n_sents):
                self.dot_memory.batch_record(
                    dot_ids,
                    out_v[s][None, :].repeat(n_dots, axis=0),  # (n_dots, dim)
                    in_winners[s],                              # (n_dots,) bool
                )

        self._call_count += n_sents
        if return_centroids:
            return out_v, ctxs
        return out_v

    def _get_ctx_cached(self, text: str) -> np.ndarray:
        """Return mean-pooled token-embedding context for text.

        Fast path for the causal training hot loop — bypasses the full
        basemapping.transform() pipeline (_apply_aaf costs ~129 ms,
        _segment regex/enum costs ~105 ms) and goes straight to
        _token_embedding (~0.19 ms/token).  For training we only need
        the static mean-pool of token vectors; the attention filter is
        only useful for inference refinement.

        Cache is a plain dict capped at 100 k entries to avoid unbounded
        memory growth on large corpora (~100 MB at dim=256 float32).
        """
        ctx = self._ctx_cache.get(text)
        if ctx is not None:
            return ctx

        dim    = self.feature_dim
        tokens = self.base_mapper._tokenize(text)
        if not tokens:
            ctx = np.zeros(dim, dtype=np.float32)
        else:
            vecs = []
            for tok in tokens:
                te = self.base_mapper._token_embedding(
                    tok, self.base_mapper._base_types.get(tok, "word")
                )
                n = min(len(te), dim)
                v = np.zeros(dim, dtype=np.float32)
                v[:n] = np.real(te[:n])
                vecs.append(v)
            ctx = np.mean(vecs, axis=0).astype(np.float32)

        if len(self._ctx_cache) < 100_000:
            self._ctx_cache[text] = ctx
        return ctx

    def causal_train_pass(self, sentences: List[str], max_pos: int = 6,
                          verbose: bool = True, prune_every: int = 0,
                          causal_batch: int = 200,
                          save_every: int = 5000) -> None:
        """
        Causal next-token prediction training (fast path).

        Win criterion:
          per-dot win = cosine_sim(W[i] @ ctx, target_token) > 0
          W[i] maps context onto prediction direction; ~50 % of dots win by
          Gaussian symmetry; dots that specialise on common tokens drift above
          50 % → effectiveness > 0.5 → MaxEff > 0.5.

        Speed design (v2 — skip convergence in training):
          The C convergence pipeline output was NEVER used by this function —
          only the mean-pooled basemap context (ctxs) is needed for the
          W-matrix alignment.  Skipping pipeline_batch_run_c saves ~248 ms of
          redundant C work per sentence.  We now compute ctxs directly via
          _get_ctx_cached(), which also memoises frequent short prefixes.

          Expected improvement: 0.67 sent/s → 30+ sent/s (causal_batch=200).
        """
        if not sentences:
            return
        t0          = time.time()
        total_steps = 0
        dim         = self.feature_dim
        n_done      = 0

        # ── Pre-load W stack once outside the batch loop ──────────────────────
        # Building _cached_W_stack requires stacking 128 dot.W matrices (33 MB)
        # PLUS rebuilding HP_stack (np.pad × 128 = expensive).  We maintain a
        # LOCAL W array that persists across batches and is only rebuilt when
        # evolution creates a new dot list (every 2000 sents).
        # _ensure_caches is NEVER called here; it would rebuild HP_stack which
        # is unused during training.
        dots   = self._ensure_dots()
        n_dots = len(dots)
        W      = np.ascontiguousarray(
            np.stack([d.W for d in dots]), dtype=np.float32
        )                                       # (n_dots, dim, dim) — our live copy
        _W_dots_id = id(dots)                   # track list identity for evolution
        dot_ids_list = [d.dot_id for d in dots]
        idx    = np.arange(n_dots, dtype=np.int64)

        for batch_start in range(0, len(sentences), causal_batch):
            batch_sents = sentences[batch_start:batch_start + causal_batch]

            # ── PHASE 1: Collect ALL prefixes from ALL sentences in batch ─────
            all_prefixes: List[str] = []
            all_tgt_toks: List[str] = []

            for sentence in batch_sents:
                tokens = self.base_mapper._tokenize(sentence)
                if len(tokens) < 2:
                    continue
                n_pos = min(len(tokens), max_pos + 1)
                for pos in range(1, n_pos):
                    if pos < len(tokens):
                        all_prefixes.append(" ".join(tokens[:pos]))
                        all_tgt_toks.append(tokens[pos])

            if not all_prefixes:
                n_done += len(batch_sents)
                continue

            # ── PHASE 2: Context centroids (fast token-embedding mean pool) ───
            n_total  = len(all_prefixes)
            all_ctxs = np.empty((n_total, dim), dtype=np.float32)
            for j_pfx, pfx in enumerate(all_prefixes):
                all_ctxs[j_pfx] = self._get_ctx_cached(pfx)

            # ── PHASE 3: Build normalised target vectors ──────────────────────
            # _tok_emb_cache avoids re-running _token_embedding for tokens that
            # appeared in earlier batches (common words appear thousands of times).
            tvs = np.zeros((n_total, dim), dtype=np.float32)
            for j, tok in enumerate(all_tgt_toks):
                cached = self._tok_emb_cache.get(tok)
                if cached is not None:
                    tvs[j] = cached
                else:
                    te = self.base_mapper._token_embedding(
                        tok, self.base_mapper._base_types.get(tok, "word")
                    )
                    nc = min(len(te), dim)
                    v  = np.zeros(dim, dtype=np.float32)
                    v[:nc] = np.real(te[:nc])
                    tn = float(np.linalg.norm(v))
                    if tn > 1e-10:
                        v /= tn
                    if len(self._tok_emb_cache) < 50_000:
                        self._tok_emb_cache[tok] = v
                    tvs[j] = v
            tvs_n = tvs                                     # already normalised

            # ── PHASE 4: BLAS win-criterion ───────────────────────────────────
            # dot_pred[p, d] = W[d] @ ctx[p]  →  (n_total, n_dots, dim)
            # win[p, d]  = (W[d] @ ctx[p]) · tv_n[p] > 0
            #            = ctx[p] · (W[d].T @ tv_n[p]) > 0
            #
            # Route: ctx @ W_flat.T  →  (n_total, n_dots*dim)
            #        reshape          →  (n_total, n_dots, dim)
            #        batched @        →  causal_sims (n_total, n_dots)
            #
            # Memory note: W_flat view; no transpose copy (C-contiguous already).
            W_flat        = W.reshape(n_dots * dim, dim)    # (32768, 256) — view
            dot_preds_all = (all_ctxs @ W_flat.T).reshape(n_total, n_dots, dim)
            causal_sims   = (dot_preds_all @ tvs_n[:, :, None]).squeeze(2)
            causal_wins   = causal_sims > 0.0               # (n_total, n_dots) bool

            # ── PHASE 5a: Vectorized W-row update ────────────────────────────
            lr       = 0.005
            lr_eff   = 1.0 - (1.0 - lr) ** n_total          # compound lr
            win_counts_f = causal_wins.sum(axis=0).astype(np.float32).clip(1.0)
            mean_targets = (causal_wins.T.astype(np.float32) @ tvs_n) / win_counts_f[:, None]
            mt_norms     = np.linalg.norm(mean_targets, axis=1, keepdims=True).clip(1e-10)
            mean_targets /= mt_norms                         # (n_dots, dim)

            rows = np.array([d._rng.randint(0, dim) for d in dots], dtype=np.int64)
            W[idx, rows] = (1.0 - lr_eff) * W[idx, rows] + lr_eff * mean_targets
            row_vecs     = W[idx, rows]
            W[idx, rows] = row_vecs / np.linalg.norm(row_vecs, axis=1, keepdims=True).clip(1e-10)

            # Sync back to dot objects (once per batch, not per-position)
            for i, dot in enumerate(dots):
                dot.W = W[i].copy()

            # ── PHASE 5b: Vectorized memory record ───────────────────────────
            # One BLAS call computes ALL 128 mean predictions simultaneously:
            #   mean_ctx (dim,) @ W_flat.T (dim, n_dots*dim) → (n_dots*dim,)
            #   reshape → (n_dots, dim) — replaces 128 separate matmuls.
            mean_ctx      = all_ctxs.mean(axis=0)            # (dim,)
            all_mean_preds = (mean_ctx @ W_flat.T).reshape(n_dots, dim)  # (n_dots, dim)
            wins_per_dot  = causal_wins.sum(axis=0)          # (n_dots,) int
            decay         = float(0.9 ** n_total)
            for d_idx, did in enumerate(dot_ids_list):
                self.dot_memory._ensure_id(did)
                n_wins   = int(wins_per_dot[d_idx])
                self.dot_memory._total_counts[did]   += float(n_total)
                self.dot_memory._success_counts[did] += float(n_wins)
                win_rate = n_wins / n_total
                h = self.dot_memory._surprise_history[did]
                self.dot_memory._surprise_history[did] = decay * h + (1.0 - decay) * win_rate
                mean_pred = all_mean_preds[d_idx]            # (dim,) — from bulk BLAS
                self.dot_memory._windows[did].append(mean_pred)
                old_mean, M2, count = self.dot_memory._var_stats[did]
                count  += 1
                delta   = mean_pred - old_mean
                new_mean = old_mean + delta / count
                self.dot_memory._var_stats[did] = (new_mean, M2 + delta * (mean_pred - new_mean), count)

            total_steps += n_total
            n_done      += len(batch_sents)

            # ── Periodic evolution (every 2000 sentences) ────────────────────
            if n_done % 2000 < causal_batch:
                self._dots = self.evolution.evolve(
                    dots, self.dot_memory, call_count=self._call_count
                )
                dots         = self._dots
                n_dots       = len(dots)
                W            = np.ascontiguousarray(
                    np.stack([d.W for d in dots]), dtype=np.float32
                )
                _W_dots_id   = id(dots)
                dot_ids_list = [d.dot_id for d in dots]
                idx          = np.arange(n_dots, dtype=np.int64)

            if save_every > 0 and n_done % save_every < causal_batch:
                self.save_brain()

            if prune_every > 0 and n_done % prune_every < causal_batch:
                self.prune_dots()

            if verbose:
                elapsed = time.time() - t0
                rate    = n_done / max(elapsed, 1e-6)
                effs    = self.dot_memory.all_effectivenesses(dot_ids_list)
                max_eff = float(np.max(effs)) if len(effs) > 0 else 0.0
                words_per_s = total_steps / max(elapsed, 1e-6)
                print(
                    f"\r[causal] {n_done}/{len(sentences)} sents"
                    f" | {rate:.1f} s/s ({words_per_s:.0f} w/s)"
                    f" | cache={len(self._ctx_cache)}"
                    f" | MaxEff: {max_eff:.4f}",
                    end="", flush=True,
                )

        if verbose:
            print()

    def causal_train_file(
        self,
        path: str,
        chunk_size: int = 5000,
        max_pos: int = 6,
        causal_batch: int = 200,
        save_every: int = 10000,
        encoding: str = "utf-8",
        verbose: bool = True,
    ) -> None:
        """Stream-train causal_train_pass from a corpus file without loading it all
        into memory.  Reads *chunk_size* non-blank lines at a time, calls
        causal_train_pass on each chunk, then discards the chunk.

        Args:
            path:         Path to a plain-text corpus file (one sentence per line).
            chunk_size:   Lines per chunk (trades RAM for progress-granularity).
            max_pos:      Passed through to causal_train_pass.
            causal_batch: Passed through to causal_train_pass.
            save_every:   Save brain every N *total* lines processed (0 = never).
            encoding:     File encoding (default utf-8, falls back to latin-1).
            verbose:      Print per-chunk throughput line.
        """
        import os, time as _time
        total_lines = 0
        chunk: List[str] = []
        t_start = _time.time()
        t_chunk  = _time.time()

        # Estimate file line count cheaply (for progress display)
        try:
            file_bytes = os.path.getsize(path)
        except OSError:
            file_bytes = 0

        try:
            fh = open(path, encoding=encoding, errors="replace")
        except OSError as exc:
            raise OSError(f"causal_train_file: cannot open {path!r}: {exc}") from exc

        with fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                chunk.append(line)
                if len(chunk) < chunk_size:
                    continue

                # ── Process chunk ────────────────────────────────────────────
                self.causal_train_pass(
                    chunk,
                    max_pos=max_pos,
                    causal_batch=causal_batch,
                    verbose=False,
                )
                total_lines += len(chunk)

                if verbose:
                    elapsed    = _time.time() - t_start
                    chunk_time = _time.time() - t_chunk
                    wps        = total_lines / max(elapsed, 1e-6)
                    dots       = self._dots or []
                    effs       = self.dot_memory.all_effectivenesses(
                        [d.dot_id for d in dots]
                    )
                    max_eff = float(np.max(effs)) if len(effs) > 0 else 0.0
                    print(
                        f"\r[file-train] {total_lines:,} lines | "
                        f"{wps:,.0f} w/s | "
                        f"MaxEff={max_eff:.4f} | "
                        f"elapsed={elapsed:.0f}s",
                        end="", flush=True,
                    )

                if save_every > 0 and (total_lines % save_every) < chunk_size:
                    self.save_brain()

                chunk = []
                t_chunk = _time.time()

            # ── Tail chunk (< chunk_size lines) ─────────────────────────────
            if chunk:
                self.causal_train_pass(
                    chunk,
                    max_pos=max_pos,
                    causal_batch=causal_batch,
                    verbose=False,
                )
                total_lines += len(chunk)
                self.save_brain()

        if verbose:
            elapsed = _time.time() - t_start
            dots    = self._dots or []
            effs    = self.dot_memory.all_effectivenesses(
                [d.dot_id for d in dots]
            )
            max_eff = float(np.max(effs)) if len(effs) > 0 else 0.0
            print(
                f"\n[file-train] DONE — {total_lines:,} lines in {elapsed:.1f}s "
                f"({total_lines/max(elapsed,1e-6):,.0f} w/s avg) | MaxEff={max_eff:.4f}"
            )

    def train_pass(self, sentences: List[str], use_c_pipeline: bool = True,
                   verbose: bool = True, prune_every: int = 0,
                   batch_size: int = 200, save_every: int = 5000):
        if not sentences: return
        t0 = time.time()
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            self.run_batch(batch, use_c_pipeline=use_c_pipeline)
            dots = self._ensure_dots()
            self._dots = self.evolution.evolve(dots, self.dot_memory, call_count=self._call_count)
            if prune_every > 0 and (i + len(batch)) % prune_every == 0:
                self.prune_dots()
            if save_every > 0 and (i + len(batch)) % save_every < batch_size:
                self.save_brain()
            if verbose:
                elapsed = time.time() - t0
                rate = (i + len(batch)) / max(elapsed, 1e-6)
                effs = self.dot_memory.all_effectivenesses([d.dot_id for d in self._dots])
                max_eff = np.max(effs) if len(effs) > 0 else 0.0
                print(f"\r[train] {i+len(batch)}/{len(sentences)} | {rate:.2f} lines/s | MaxEff: {max_eff:.4f}", end="", flush=True)
        if prune_every > 0:
            self.prune_dots()
        if verbose: print()

    def _blend(self, original: BaseMap, refined: np.ndarray, alpha: float = 0.85) -> BaseMap:
        mat = original.matrix.copy()
        r_real = np.real(refined).astype(np.float32)
        n = np.linalg.norm(r_real)
        if n > 1e-10:
            r = r_real / n
            mat = (alpha * mat + (1.0 - alpha) * r[None, :]).astype(np.float32)
        return BaseMap(mat, original.tokens, original.token_types, original.modifiers, original.metadata)

    def fit(self, texts: List[str]) -> "IECNN":
        self.base_mapper.fit(texts)
        return self

    def encode(self, text: str) -> np.ndarray: return self.run(text).output

    def chat(self, message: str, history: List = None, verbose: bool = False,
             beam_width: int = 4) -> str:
        """Generate a chat reply using beam-search decoding (beam_width=4)."""
        from decoding.decoder import IECNNDecoder
        context = ""
        if history:
            turns = history[-3:]
            context = " ".join(f"{u} {r}" for u, r in turns) + " "
        prompt = context + message
        res = self.run(prompt)
        decoder = IECNNDecoder(self)
        return decoder.decode(res.output, max_tokens=12, iterations=40,
                              beam_width=beam_width)

    def generate(self, prompt: str, max_tokens: int = 8, iterations: int = 20,
                 beam_width: int = 4) -> str:
        """Encode prompt → decode output text using beam-search (beam_width=4)."""
        from decoding.decoder import IECNNDecoder
        res = self.run(prompt)
        decoder = IECNNDecoder(self)
        return decoder.decode(res.output, max_tokens=max_tokens,
                              iterations=iterations, beam_width=beam_width)

    def compare(self, texts: List[str]) -> np.ndarray:
        """Return n×n pairwise similarity matrix for the given texts."""
        from formulas.formulas import similarity_score
        vecs = [self.encode(t) for t in texts]
        n = len(vecs)
        mat = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            mat[i, i] = 1.0
            for j in range(i + 1, n):
                s = similarity_score(vecs[i], vecs[j])
                mat[i, j] = s
                mat[j, i] = s
        return mat

    def memory_status(self) -> dict:
        """Return a summary dict of dot memory, cluster memory, and evolution state."""
        dots = self._ensure_dots()
        dot_ids = [d.dot_id for d in dots]
        dm = self.dot_memory.summary(dot_ids)
        cm = self.cluster_memory.summary()
        ev = self.evolution.stats(self.dot_memory, dot_ids)
        return {
            "call_count":     self._call_count,
            "dot_memory":     dm,
            "cluster_memory": cm,
            "evolution":      ev,
        }

    def fit_file(self, filepath: str, verbose: bool = True) -> "IECNN":
        """Stream one sentence per line from filepath and fit the BaseMapper vocabulary."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Corpus not found: {filepath}")
        sentences = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    sentences.append(line)
        if verbose:
            print(f"[IECNN] fit_file: {len(sentences)} sentences from {filepath}")
        self.base_mapper.fit(sentences)
        self._call_count += 1
        if verbose:
            print(f"[IECNN] fit_file: vocab size = {len(self.base_mapper._base_vocab)}")
        self.save_brain()
        return self

    def prune_dots(self, min_outcomes: int = 2, min_age_gens: int = 2,
                   dry_run: bool = False) -> dict:
        """
        Remove under-performing dots and their orphaned memory history.

        A dot is eligible for removal when:
          - It is old enough: current_gen - birth_generation >= min_age_gens
          - Its total prediction count < min_outcomes

        Returns a stats dict with removed/kept counts.
        """
        dots = self._ensure_dots()
        cur_gen = self.evolution.generation
        keep_dots = []
        remove_ids = []
        for d in dots:
            age = cur_gen - getattr(d, "birth_generation", 0)
            total = self.dot_memory._total_counts.get(d.dot_id, 0.0)
            if age >= min_age_gens and total < min_outcomes:
                remove_ids.append(d.dot_id)
            else:
                keep_dots.append(d)

        removed_history = 0
        if not dry_run:
            self._dots = keep_dots
            before = sum(len(v) for v in self.dot_memory._windows.values())
            keep_ids = [d.dot_id for d in keep_dots]
            self.dot_memory.prune(keep_ids)
            after = sum(len(v) for v in self.dot_memory._windows.values())
            removed_history = before - after

        return {
            "generation":       cur_gen,
            "removed_dots":     len(remove_ids),
            "kept_dots":        len(keep_dots),
            "removed_history":  removed_history,
            "kept_history":     sum(len(v) for v in self.dot_memory._windows.values()),
        }
