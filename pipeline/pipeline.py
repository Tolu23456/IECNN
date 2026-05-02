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
from peep.peep               import PeepMechanism
from grammar.grammar         import GrammarGuide
from generation              import (
    SemanticFieldBias, VocabFrequencyPrior,
    RepetitionPenalty, NoRepeatNGram,
    DegenerationPenalty, MinLengthGuard, ExponentialDecayLength,
    TypicalFilter,
    NucleusFilter, MinPFilter, EtaFilter,
    DynamicTemperature, MirostatScheduler,
    BigramContinuationBonus, SemanticProximityPenalty,
    TailFreeFilter, TopKFilter,
    PromptDriftPenalty, LocalSemanticFilter, DotVariancePenalty, SurpriseBonus,
    ScoreProcessorList, softmax_sample,
    ContextHistory, ContextAnchor, MultiHeadConvergence,
)
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

        # LRU ctx cache: text → recency-weighted token-embedding vector (dim,)
        # Eliminates re-transforming repeated causal prefixes (major speedup).
        self._ctx_cache: Dict[str, np.ndarray] = {}

        # Token embedding cache: token → normalised float32 (dim,)
        # Across a 200-sentence batch, target tokens repeat constantly (same
        # common words appear in many positions).  Caching avoids re-running
        # _token_embedding (complex polynomial + normalisation) per duplicate.
        self._tok_emb_cache: Dict[str, np.ndarray] = {}

        # Peep Mechanism — calibrated during causal_train_pass, loaded from disk.
        # None until first calibration batch or successful load_brain.
        self.peep: Optional[PeepMechanism] = None

        # GrammarGuide — built lazily in causal_generate, cached per vocab size.
        self._grammar_guide: Optional[GrammarGuide] = None
        self._grammar_vocab_size: int = 0

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
        if self.peep is not None:
            self.peep.save(path + ".peep.pkl")

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
        peep_path = path + ".peep.pkl"
        if os.path.exists(peep_path):
            self.peep = PeepMechanism.load(peep_path)

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
        """Return recency-weighted token-embedding context for text.

        Fast path for the causal training hot loop — bypasses the full
        basemapping.transform() pipeline and goes straight to
        _token_embedding (~0.19 ms/token).

        Encoding strategy — recency-weighted pooling
        ---------------------------------------------
        Instead of a plain mean, later tokens are weighted more heavily:
          weight[i] = 0.70^(n-1-i)   (last token = 1.0, each step back × 0.70)
        This means a single-word prompt ("Hello") produces a very different
        vector from a multi-word prompt ("The president signed the bill"),
        because the last word dominates.  Plain mean-pooling washes this out
        and causes all multi-word prompts to cluster together regardless of
        topic.

        The result is L2-normalised to unit length so cosine comparisons
        are meaningful across prompts of different lengths.

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

            n_toks = len(vecs)
            if n_toks == 1:
                ctx = vecs[0].copy()
            else:
                decay   = 0.70
                weights = np.array(
                    [decay ** (n_toks - 1 - i) for i in range(n_toks)],
                    dtype=np.float32,
                )
                weights /= weights.sum()
                ctx = (np.stack(vecs) * weights[:, None]).sum(axis=0).astype(np.float32)

        ctx_norm = float(np.linalg.norm(ctx))
        if ctx_norm > 1e-9:
            ctx = ctx / ctx_norm

        if len(self._ctx_cache) < 100_000:
            self._ctx_cache[text] = ctx
        return ctx

    def predict_word(
        self,
        prefix: str,
        top_k: int = 5,
        n_cands: int = 3000,
    ) -> List[str]:
        """Predict the next word after *prefix* using nearest-neighbour lookup.

        Unlike the previous vote-counting approach, this computes a single
        effectiveness-weighted mean prediction vector and returns the vocab
        tokens whose normalised embeddings are most cosine-similar to it.

        Args:
            prefix:   The input text (tokenised internally).
            top_k:    Number of candidate tokens to return (ranked).
            n_cands:  How many vocab tokens to score (top-n by vocab order,
                      which approximates frequency rank after fit_fast).
        Returns:
            List of token strings, highest-scoring first.
        """
        ctx  = self._get_ctx_cached(prefix)           # (dim,)
        dots = self._ensure_dots()
        dim  = self.feature_dim
        nd   = len(dots)

        W_loc = np.ascontiguousarray(
            np.stack([d.W for d in dots]), dtype=np.float32
        )                                             # (nd, dim, dim)

        # All dot predictions in one BLAS call
        preds = (ctx @ W_loc.reshape(nd * dim, dim).T).reshape(nd, dim)  # (nd, dim)

        # Effectiveness-weighted mean — specialised dots get more weight
        effs = np.array(
            self.dot_memory.all_effectivenesses([d.dot_id for d in dots]),
            dtype=np.float32,
        )
        w = effs / effs.sum() if effs.sum() > 1e-10 else np.ones(nd, dtype=np.float32) / nd
        wmean = (w[:, None] * preds).sum(axis=0)     # (dim,)
        wn    = float(np.linalg.norm(wmean))
        if wn > 1e-10:
            wmean /= wn

        # Build/look-up candidate embedding matrix
        vocab    = list(self.base_mapper._base_vocab.keys())[:n_cands]
        n_v      = len(vocab)
        cand_mat = np.zeros((n_v, dim), dtype=np.float32)
        for j, tok in enumerate(vocab):
            v = self._tok_emb_cache.get(tok)
            if v is None:
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
            cand_mat[j] = v

        scores = cand_mat @ wmean                    # (n_v,) cosine similarities
        top_i  = np.argsort(scores)[::-1][:top_k]
        return [vocab[i] for i in top_i]

    def causal_train_pass(self, sentences: List[str], max_pos: int = 6,
                          verbose: bool = True, prune_every: int = 0,
                          causal_batch: int = 200,
                          save_every: int = 5000,
                          max_sub_pos: int = 0) -> None:
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

        # max_sub_pos=0 means "use max_pos" (no subsampling)
        _sub = max_sub_pos if max_sub_pos > 0 else max_pos

        for batch_start in range(0, len(sentences), causal_batch):
            batch_sents = sentences[batch_start:batch_start + causal_batch]

            # ── PHASE 1: Collect prefixes — with optional position subsampling ─
            # Subsampling: if a sentence has more candidate positions than _sub,
            # randomly draw _sub of them.  Reduces n_total when sentences are
            # long → shrinks Phase-4 BLAS cost proportionally (the dominant op).
            all_prefixes: List[str] = []
            all_tgt_toks: List[str] = []

            for sentence in batch_sents:
                tokens = self.base_mapper._tokenize(sentence)
                if len(tokens) < 2:
                    continue
                n_pos = min(len(tokens), max_pos + 1)
                positions = list(range(1, n_pos))
                if len(positions) > _sub:
                    positions = sorted(self._rng.choice(
                        len(positions), _sub, replace=False
                    ).tolist())
                for pos in positions:
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
            # dot_pred[p, d] = ctx[p] @ W[d].T  →  (n_total, n_dots, dim)
            # raw_win[p, d]  = dot_pred[p,d] · tv_n[p] > 0
            #
            # COMPETITIVE FIX — WINNER-TAKE-ALL per position:
            # For each context position, only the SINGLE highest-scoring dot
            # (by causal cosine) trains on that position.  Every other dot is
            # excluded.  This is the strongest possible specialisation pressure:
            # 128 dots partition the n_total positions into 128 disjoint subsets,
            # so each dot must learn to handle its unique slice of the corpus.
            #
            # Why not top-50%?  Empirically, top-50% still collapsed to
            # inter-dot cosine ≈ 0.66 after 200 sentences because all dots
            # share ~50% of positions and converge to the same mean direction.
            # Winner-take-all gives ~1/128 = 0.78% of positions per dot, which
            # forces genuine divergence.
            #
            # Route: ctx @ W_flat.T → (n_total, n_dots*dim) → reshape → sims
            W_flat        = W.reshape(n_dots * dim, dim)        # (32768, 256) view
            dot_preds_all = (all_ctxs @ W_flat.T).reshape(n_total, n_dots, dim)
            causal_sims   = (dot_preds_all @ tvs_n[:, :, None]).squeeze(2)
                                                                # (n_total, n_dots)

            # Winner-take-all: each position → one dot (the argmax scorer)
            best_dot     = np.argmax(causal_sims, axis=1)      # (n_total,) int
            causal_wins  = np.zeros((n_total, n_dots), dtype=np.float32)
            causal_wins[np.arange(n_total), best_dot] = 1.0    # one-hot rows

            # ── Peep calibration: update dot specialisations ──────────────────
            # best_dot already tells us which dot won each position — we just
            # need to teach Peep which context directions each dot specialises in.
            # Cost is negligible: one groupby over n_total rows.
            if self.peep is None:
                self.peep = PeepMechanism(n_dots, self.feature_dim)
            self.peep.observe_batch(all_ctxs, best_dot)

            # ── PHASE 5a: Vectorized W-row update with diversity push ─────────
            # DIVERSITY FIX: after computing each dot's mean target direction,
            # subtract the global average direction (50% weight) and re-normalise.
            # This amplifies each dot's unique direction and suppresses the shared
            # "average corpus direction" that causes collapse.
            #
            # F14 ADAPTIVE LR: each dot's effective learning rate is scaled by
            # (1 - 0.8 * dominance²) so dominant dots slow down and allow
            # weaker dots to catch up — prevents winner-take-all monopoly.
            base_lr     = 0.005
            dominance_arr = np.array(
                [getattr(d, 'dominance', 0.5) for d in dots], dtype=np.float32
            )                                                   # (n_dots,)
            lr_arr      = base_lr * (1.0 - 0.8 * dominance_arr ** 2)  # (n_dots,) F14
            lr_arr      = np.clip(lr_arr, 1e-5, base_lr)
            lr_eff_arr  = 1.0 - (1.0 - lr_arr) ** n_total     # (n_dots,) effective

            win_counts_f = causal_wins.sum(axis=0).clip(1.0)   # (n_dots,)
            mean_targets = (causal_wins.T @ tvs_n) / win_counts_f[:, None]
                                                                # (n_dots, dim)
            mt_norms     = np.linalg.norm(mean_targets, axis=1, keepdims=True).clip(1e-10)
            mean_targets /= mt_norms                            # normalised

            # Diversity push: subtract 70% of the shared centroid, then renorm.
            # (was 50% — increased to 70% to fight W-matrix collapse harder)
            global_dir   = mean_targets.mean(axis=0)           # (dim,) centroid
            div_targets  = mean_targets - 0.70 * global_dir[None, :]
            dt_norms     = np.linalg.norm(div_targets, axis=1, keepdims=True).clip(1e-10)
            eff_targets  = div_targets / dt_norms               # (n_dots, dim)

            rows = np.array([d._rng.randint(0, dim) for d in dots], dtype=np.int64)
            lre  = lr_eff_arr[:, None]                          # (n_dots, 1) broadcast
            W[idx, rows] = (1.0 - lre) * W[idx, rows] + lre * eff_targets
            row_vecs     = W[idx, rows]
            W[idx, rows] = row_vecs / np.linalg.norm(row_vecs, axis=1, keepdims=True).clip(1e-10)

            # ── Inter-dot W repulsion (vectorised) ───────────────────────────
            # Push W-matrix rows that are too similar apart.  Threshold 0.70 so
            # only genuinely similar rows are repelled; lr=0.008 keeps it gentle.
            WREP_THRESH = 0.70
            WREP_LR     = 0.008
            cur_rows    = W[idx, rows]                          # (n_dots, dim)
            row_norms   = np.linalg.norm(cur_rows, axis=1, keepdims=True).clip(1e-9)
            rows_n      = cur_rows / row_norms                  # unit vectors
            row_sims    = rows_n @ rows_n.T                     # (n_dots, n_dots)
            np.fill_diagonal(row_sims, 0.0)
            too_close   = np.clip(row_sims - WREP_THRESH, 0.0, None)
            if too_close.sum() > 0.0:
                rep_dir     = too_close @ rows_n                # (n_dots, dim)
                pair_cnt    = (too_close > 0).sum(axis=1, keepdims=True).clip(1)
                cur_rows   -= WREP_LR * rep_dir / pair_cnt
                row_norms2  = np.linalg.norm(cur_rows, axis=1, keepdims=True).clip(1e-9)
                W[idx, rows] = cur_rows / row_norms2

            # Sync back to dot objects (once per batch, not per-position)
            for i, dot in enumerate(dots):
                dot.W = W[i].copy()

            # ── PHASE 5b: Memory record + F17 dominance update ───────────────
            # Win scoring: normalise relative to the expected baseline (1/n_dots).
            #   score = 0.5 * min(2, actual_rate / expected_rate)
            #   → 0.5  for an average dot   (ratio = 1.0)
            #   → 1.0  for 2× average       (ratio = 2.0, clipped)
            #   → 0.0  for a dot that never wins
            # This ensures effectiveness converges to ~0.5 for random dots and
            # rises above 0.5 as dots specialise — symmetry breaking is possible.
            #
            # F17 DOMINANCE EMA: updated per-batch on the same above/below signal.
            # dominant dots (dominance > 0.5) had their lr reduced via F14 above.
            mean_ctx      = all_ctxs.mean(axis=0)            # (dim,)
            all_mean_preds = (mean_ctx @ W_flat.T).reshape(n_dots, dim)  # (n_dots, dim)
            wins_per_dot  = causal_wins.sum(axis=0)          # (n_dots,) float
            expected_wins = float(n_total) / float(n_dots)   # baseline per dot
            decay         = float(0.9 ** n_total)
            for d_idx, did in enumerate(dot_ids_list):
                self.dot_memory._ensure_id(did)
                n_wins    = float(wins_per_dot[d_idx])
                ratio     = n_wins / max(expected_wins, 1e-6)   # 1.0 = average
                norm_score = 0.5 * min(2.0, ratio)              # [0, 1], 0.5=average
                # One trial per batch — effectiveness = mean(norm_score) over batches
                self.dot_memory._total_counts[did]   += 1.0
                self.dot_memory._success_counts[did] += norm_score

                # F17: update dot's dominance EMA (above/below expected baseline)
                dot = dots[d_idx]
                dot.dominance = 0.9 * dot.dominance + 0.1 * min(1.0, ratio)
                dot.dominance = max(0.0, min(1.0, dot.dominance))

                win_rate = n_wins / max(float(n_total), 1.0)
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
        max_sub_pos: int = 4,
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
                    max_sub_pos=max_sub_pos,
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
                    max_sub_pos=max_sub_pos,
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

    def causal_generate(self,
                        prompt:               str,
                        max_tokens:           int   = 20,
                        ctx_alpha:            float = 0.55,
                        confidence_threshold: float = 0.08,
                        variation:            float = 0.60,
                        # ── Generation quality controls ────────────────
                        top_p:                float = 0.90,
                        min_p:                float = 0.05,
                        rep_presence:         float = 0.15,
                        rep_frequency:        float = 0.08,
                        degeneration_alpha:   float = 0.12,
                        no_repeat_ngram_size: int   = 3,
                        min_tokens:           int   = 5,
                        semantic_bias:        float = 0.10,
                        n_heads:              int   = 8,
                        history_window:       int   = 24,
                        ) -> dict:
        """
        IECNN-native generation loop with full transformer-equivalent quality pipeline.

        Architecture
        ============
        The loop is driven by two voting paths (Peep-guided / Multi-Head Convergence)
        followed by a ten-stage score-processor pipeline applied every step.

        Voting paths
        ------------
        PATH A  (Peep calibrated)
            Top-5 Peep-specialised dots produce a weighted residual signal.
            Combined 70/30 with the Multi-Head Convergence signal for stability.

        PATH B  (Peep not yet calibrated)
            8 heads of ~16 dots each vote independently; scores are fused by
            softmax-weighted head confidence.  Replaces the old flat majority-vote.

        Score processor pipeline (applied every step, in order)
        ---------------------------------------------------------
        1.  SemanticFieldBias      — cosine bonus toward prompt-topic vocabulary
        2.  RepetitionPenalty      — presence (0.15) + frequency (0.08) subtraction
        3.  NoRepeatNGram (n=3)    — hard-∞ block on repeat trigrams (and bigrams)
        4.  DegenerationPenalty    — SimCTG α=0.12: penalise embedding-similar tokens
        5.  MinLengthGuard         — block stop tokens before min_tokens steps
        6.  NucleusFilter          — top-p=0.90 nucleus filter
        7.  MinPFilter             — min-p=0.05 adaptive filter
        8.  DynamicTemperature     — adaptive temperature from confidence trend
        9.  softmax_sample()       — temperature-scaled categorical sample

        Context enrichment (applied before every voting step)
        -------------------------------------------------------
        • ContextHistory  — rolling 24-token ring buffer with attention-weighted
                            context aggregation (IECNN's KV-cache equivalent)
        • ContextAnchor   — prompt-anchored anti-drift correction
        • Exclusion EMA   — steers context away from already-visited directions

        Returns
        -------
        dict with keys:
          text          – space-joined output string
          tokens        – list of token strings
          confidences   – per-token winning-cluster score
          stop_reason   – "max_tokens" | "low_confidence(x.xxx)"
                          | "natural_stop" | "no_dots" | "no_vocab"
        """
        import re as _re
        from collections import Counter as _Counter

        dots = self._ensure_dots()
        if not dots:
            return {"text": "", "tokens": [], "confidences": [],
                    "stop_reason": "no_dots"}

        vocab = self.base_mapper._base_vocab
        if not vocab:
            return {"text": "", "tokens": [], "confidences": [],
                    "stop_reason": "no_vocab"}

        # ── Vocab pool: real, reasonably-common words only ───────────────────
        # Filtering philosophy (mirrors GPT-2 vocabulary curation):
        #   1. Whole-word tokens only (no subword fragments)
        #   2. Pure-alpha words ≥ 3 chars, containing at least one vowel
        #   3. Minimum training frequency — removes hapax legomena and
        #      rare proper nouns that produce cosine noise (e.g. "wanyan",
        #      "tjoa", "hazael").  Floor = 3 occurrences.
        #   4. Fallback to unfiltered set if filtering leaves < 50 words.
        _WORD_RE    = _re.compile(r'^[A-Za-z][a-z]{2,}$')
        _HAS_VOWEL  = _re.compile(r'[aeiou]', _re.I)
        base_types  = getattr(self.base_mapper, "_base_types", {})
        _word_freq_map = getattr(self.base_mapper, "_word_freq", {})
        _MIN_FREQ   = 3   # minimum training-corpus occurrences

        words = [
            w for w in vocab
            if " " not in w
            and base_types.get(w, "subword") == "word"
            and _WORD_RE.match(w)
            and _HAS_VOWEL.search(w)
            and _word_freq_map.get(w, 0) >= _MIN_FREQ
        ]
        # Fallback: relax frequency filter then relax type filter
        if len(words) < 200:
            words = [
                w for w in vocab
                if " " not in w
                and _WORD_RE.match(w)
                and _HAS_VOWEL.search(w)
                and _word_freq_map.get(w, 0) >= _MIN_FREQ
            ]
        if len(words) < 50:
            words = [w for w in vocab if " " not in w]

        word_vecs   = np.stack([vocab[w].astype(np.float32) for w in words])  # (V, 224)
        wv_norms    = np.linalg.norm(word_vecs, axis=1, keepdims=True) + 1e-10
        word_vecs_n = word_vecs / wv_norms                                      # unit (V, 224)

        # Fast word→index lookup for n-gram and repetition processors
        word_index: Dict[str, int] = {w: i for i, w in enumerate(words)}

        # Padded+normalised word embeddings for lookahead (256-dim context space).
        # Each word vector is zero-padded from EMBED_DIM (224) to FEATURE_DIM (256)
        # so it can be compared directly with the context vector.
        _wvp = np.zeros((len(words), FEATURE_DIM), dtype=np.float32)
        _wvp[:, :EMBED_DIM] = word_vecs
        _wvp_n = _wvp / (np.linalg.norm(_wvp, axis=1, keepdims=True) + 1e-10)  # (V, 256)

        # ── Initial context ────────────────────────────────────────────────
        ctx = self._get_ctx_cached(prompt).astype(np.float32)   # (256,)
        ctx = ctx / (np.linalg.norm(ctx) + 1e-10)

        # ── W-matrix stack ────────────────────────────────────────────────
        W_stack  = np.stack([d.W for d in dots]).astype(np.float32)  # (D, 256, 256)

        # ── Grammar guide (cached) ────────────────────────────────────────
        if (self._grammar_guide is None
                or self._grammar_vocab_size != len(words)):
            self._grammar_guide      = GrammarGuide(words, bias_strength=0.50)
            self._grammar_vocab_size = len(words)
        _grammar_guide = self._grammar_guide

        # ── Generation-quality components ─────────────────────────────────

        # Context history: rolling attention-weighted ring buffer (IECNN KV-cache)
        # Adaptive history window: longer generations benefit from wider context
        _hist_win_eff = max(history_window, min(history_window + max_tokens // 4, 40))
        ctx_hist = ContextHistory(window=_hist_win_eff, dim=len(ctx),
                                  decay=0.88, ctx_alpha=0.68)

        # ── Prompt-type detection ─────────────────────────────────────────
        # Adjust CONF_STOP and min_tokens per detected prompt intent:
        #   question   → tighter confidence stop, one less min token
        #   imperative → slightly softer stop, two more min tokens
        #   statement  → defaults unchanged
        _Q_STARTS   = {"what","who","when","where","why","how","is","are","was",
                       "were","does","do","did","can","could","would","should",
                       "have","has"}
        _IMP_STARTS = {"check","find","tell","list","describe","explain","show",
                       "write","generate","give","make","define","analyze",
                       "compare","summarize","outline"}
        _pw = prompt.strip().lower().split()
        if _pw and _pw[0] in _Q_STARTS:
            CONF_STOP  = confidence_threshold * 0.85   # questions: sharper cutoff
            min_tokens = max(3, min_tokens - 1)
            _ptype     = "question"
        elif _pw and _pw[0] in _IMP_STARTS:
            CONF_STOP  = confidence_threshold * 1.10   # imperatives: softer cutoff
            min_tokens = min_tokens + 2
            _ptype     = "imperative"
        else:
            CONF_STOP  = confidence_threshold
            _ptype     = "statement"

        # Context anchor: prompt-anchored anti-drift (IECNN prompt-following)
        # Prompt-length adaptive strength: longer prompts carry more semantic
        # intent, so we anchor more firmly to them.
        _prompt_words = len(prompt.split())
        _anc_strength = float(np.clip(0.15 + 0.03 * (_prompt_words - 1), 0.15, 0.36))
        anchor = ContextAnchor(ctx.copy(), drift_threshold=0.20,
                               correction_strength=_anc_strength)

        # Multi-head convergence voter (IECNN multi-head attention equivalent)
        mhc = MultiHeadConvergence(n_heads=n_heads, embed_dim=EMBED_DIM)
        mhc.build(len(dots))

        # Semantic field bias — prompt-topic coherence (pre-computed once)
        prompt_embed = ctx[:EMBED_DIM]
        # SFB strength reset: ensure each generation starts with a fresh strength,
        # not a mutated value left over from a previous generation's dynamics.
        _sfb = SemanticFieldBias(word_vecs_n, prompt_embed,
                                 strength=semantic_bias, threshold=0.20, decay=0.97)

        # Prompt drift penalty — symmetric push-pull with SemanticFieldBias.
        # While SFB adds bonus for prompt-close tokens, PromptDriftPenalty
        # subtracts from prompt-distant tokens (strength=0.06, threshold=0.08).
        _pdrift = PromptDriftPenalty(word_vecs_n, prompt_embed,
                                     strength=0.06, threshold=0.08)

        # Vocabulary frequency prior — penalise ultra-rare words
        _word_freq  = getattr(self.base_mapper, "_word_freq", {})
        _freq_prior = VocabFrequencyPrior(words, _word_freq, strength=0.04)

        # Bigram continuation bonus — data-driven collocation scoring
        # Boosts candidates that co-occurred with the last token in training.
        _phrase_vocab   = getattr(self.base_mapper, "_base_vocab", {})
        _bigram_bonus   = BigramContinuationBonus(words, _phrase_vocab, strength=0.07)

        # Semantic proximity penalty — embedding-space near-synonym suppression.
        # Catches synonymous repetition ("big … large … huge") that bypasses
        # string-based repetition penalties.  Threshold 0.82 targets near-duplicates
        # while leaving loosely-related words (~0.50 similarity) unaffected.
        _prox_pen = SemanticProximityPenalty(
            window=4, sim_threshold=0.82, proximity_scale=3.5)

        # Presence + frequency repetition penalty (Keskar et al. 2019 / CTRL)
        _rep = RepetitionPenalty(presence=rep_presence, frequency=rep_frequency)

        # No-repeat n-gram — hard-block repeated trigrams + bigrams
        _ngram3 = NoRepeatNGram(n=no_repeat_ngram_size, word_index=word_index)
        _ngram2 = NoRepeatNGram(n=2,                    word_index=word_index)

        # SimCTG degeneration penalty (Su & Collier, NeurIPS '22)
        _deg = DegenerationPenalty(alpha=degeneration_alpha, window=16)

        # Stop token IDs for length-control processors
        STOP_TOKENS = {".", "!", "?", "\n", "<eos>", "[eos]", "</s>"}
        stop_ids    = {word_index[t] for t in STOP_TOKENS if t in word_index}

        # Minimum length guard — block early stops
        _minlen = MinLengthGuard(min_tokens=min_tokens, stop_ids=stop_ids)

        # Exponential decay length penalty — smooth natural stopping
        _expdecay = ExponentialDecayLength(
            start_idx=max(min_tokens, 8), factor=1.06, stop_ids=stop_ids)

        # Adaptive top-K filter — hard cap before the softer filters.
        # k=40, adaptive: k scales from 40 (confident) to 60 (uncertain)
        # based on normalised entropy. Applied first in the filter chain.
        _topk = TopKFilter(k=40, min_keep=1, k_adaptive=True, entropy_scale=0.50)

        # Dot variance penalty — penalise words where dots disagree strongly.
        # strength=0.08: applied after voting, before the filter chain.
        _dot_var_pen = DotVariancePenalty(strength=0.08)

        # Local semantic filter — restrict vocab to top-200 cosine-close words.
        # Applied as the very first score filter so downstream processors work
        # on a tightly focused pool; fast: O(V × embed_dim) one matmul/step.
        _local_sem = LocalSemanticFilter(top_k=200)

        # Surprise bonus — give non-top-50 but context-close tokens a nudge.
        # Opens the beam slightly beyond the dot-consensus ranking, preventing
        # the model from being trapped in a high-frequency attractor.
        _surprise = SurpriseBonus(strength=0.04, top_skip=50, ctx_threshold=0.15)

        # Typical filter (Meister et al. 2023) — removes tokens whose surprisal
        # deviates too far from the distribution entropy in either direction.
        # Applied after top-K so the combined chain is:
        #   top-K → typical → TFS → nucleus → eta → min-p
        _typical = TypicalFilter(p=0.95, min_keep=1)

        # Tail-free sampling (TFS) filter (Phénix & Egan 2019) — cuts the
        # statistical tail by examining the 2nd derivative of sorted probs.
        # Placed after TypicalFilter and before nucleus as a complementary
        # stage: typical removes distribution outliers, TFS cuts the flat tail.
        _tfs = TailFreeFilter(z=0.97, min_keep=1)

        # Nucleus (top-p) + Eta (entropy-adaptive) + Min-p filters
        _nucleus = NucleusFilter(p=top_p,       min_keep=1)
        _eta     = EtaFilter(epsilon=3e-4,       min_keep=1)
        _minp    = MinPFilter(min_p=min_p,       min_keep=1)

        # Mirostat v2 scheduler (Basu et al. 2020) — target confidence 0.38
        # τ-warmup: ceiling starts at 0.85 for exploratory early tokens, then
        # decays to 0.65 over 6 steps as context constrains the distribution.
        _mirostat = MirostatScheduler(target=0.38, lr=0.08,
                                      base=0.15, max_temp=0.65,
                                      warmup_max_temp=0.85, warmup_steps=6)
        # DynamicTemperature as fallback / first steps
        _dtemp    = DynamicTemperature(base=0.15, max_temp=0.65, window=4)

        # ── State trackers ────────────────────────────────────────────────
        output_tokens:  List[str]   = []
        output_confs:   List[float] = []
        generated_ids:  List[int]   = []   # vocab indices for rep penalty
        prev_vecs:      List[np.ndarray] = []  # unit word embeddings for deg penalty

        stop_reason  = "max_tokens"
        _ema_entropy = 0.0   # EMA of per-step normalised entropy (0..1)
        _H_norm      = 0.50  # normalised entropy of current dist (initialised to mid)
        _vocab_prec_hits = 0   # steps where chosen token was in local_sem pool
        _vocab_prec_total = 0  # total steps counted for precision metric
        _vprec_ema       = 0.50  # EMA of vocab precision (α=0.20); initialised to mid
        _VPREC_EMA_ALPHA = 0.20
        CONF_STOP    = confidence_threshold
        HARD_BLOCK   = 8      # max surface-level hard-block (adaptive per step)

        # Soft anchor restart detector:
        # When score-gap has been very low (< 0.03) for 3+ consecutive steps,
        # the model is stuck in a flat, indecisive plateau.  Re-warm the anchor
        # to the current context direction so it doesn't keep pulling toward a
        # stale position.  One-time per generation to avoid oscillation.
        _low_gap_streak  = 0
        _LOW_GAP_THR     = 0.03
        _LOW_GAP_WIN     = 3
        _anchor_restarted = False

        # Penalty momentum: EMA of the per-step repetition-penalty delta vector.
        # Carries burst-repetition memory across steps so a word repeated twice
        # in a row accumulates more penalty than a single distant repeat.
        _rep_mom      = np.zeros(len(words), dtype=np.float32)
        _REP_MOM_B1   = 0.78   # EMA decay for penalty momentum
        _REP_MOM_S    = 0.30   # momentum contribution to next step's penalty

        # Repetition burst detector:
        # If 2+ of the last 4 output tokens are the same word, we are in a
        # burst-repetition loop.  On detection, reset rep_mom to zero (clear
        # momentum state) and fire a one-step temperature pulse (+0.20) to
        # force the model off the attractor.
        _BURST_WIN     = 4     # window to check for bursts
        _BURST_THRESH  = 2     # how many identical tokens trigger burst-clear
        _BURST_TEMP    = 0.20  # additive temp pulse on burst detection
        _burst_pulse   = 0.0   # one-shot temp addition (decays to 0 after use)

        # Low-margin burst trigger:
        # If top-1 margin has been < 0.02 for 3+ consecutive steps, the model
        # is in a flat, indecisive plateau — arm a burst pulse to escape.
        _LOW_MARGIN_WIN   = 3
        _LOW_MARGIN_THR   = 0.02
        _low_margin_count = 0    # consecutive steps below threshold

        # Margin-adaptive temperature:
        # Track EMA of the top-1 margin; high EMA → allow slight temp reduction.
        _margin_ema     = 0.10   # initialised to mid-range
        _MARGIN_EMA_A   = 0.25

        # Adaptive temperature lower bound:
        # When the model has been consistently confident (RW conf EMA ≥ 0.50),
        # the temperature lower bound is allowed to shrink from 0.15 to 0.08,
        # making confident generations tighter/more decisive.
        # When confidence is low (< 0.30), the lower bound rises to 0.20 to
        # force more exploration.
        _rw_conf_ema   = 0.50  # rolling recency-weighted confidence EMA
        _RW_CONF_ALPHA = 0.25  # EMA blend rate (fast, tracks recent steps)

        # Confidence trend window — last 5 confidences for slope detection.
        # If the slope is sharply declining (< -0.04/step) we add a temperature
        # boost so the sampler explores a wider beam rather than collapsing.
        from collections import deque as _deque
        _conf_window  : "deque[float]" = _deque(maxlen=5)
        _TREND_SLOPE  = -0.04  # threshold: decline steeper than this triggers boost
        _TREND_BOOST  = 0.12   # additive temperature bonus when trend is declining

        # Context momentum (Adam β1 = 0.85):
        # Running mean of context-update deltas; smooths the trajectory and
        # prevents single-token outliers from jerking the semantic direction.
        # Applied as a small additive correction to the EMA context update.
        _ctx_vel     = np.zeros_like(ctx)  # velocity vector
        _CTX_MOM_B1  = 0.85               # momentum decay coefficient
        _CTX_MOM_S   = 0.12               # velocity contribution strength

        # Context oscillation dampener:
        # Stores the previous delta (direction of last context move).
        # If the new delta is pointing OPPOSITE (cosine < 0), the context is
        # oscillating — subtract a fraction of the velocity to resist the swing.
        _ctx_prev_delta = np.zeros_like(ctx)   # delta from previous step
        _CTX_OSC_THR    = -0.10                # cosine threshold to detect reversal
        _CTX_OSC_DAMP   = 0.30                 # fraction of velocity to subtract

        # Semantic field momentum:
        # A slow-moving centroid tracking the "topic" of the generation so far.
        # Updated every step with a very small blend (α=0.05) so it lags behind
        # the fast ctx vector.  When ctx has drifted more than 0.35 from the
        # centroid, a gentle pull (0.08) brings it back — preventing runaway
        # topic drift while still allowing natural sentence development.
        _sem_centroid   = ctx.copy()        # slow topic tracker (feature_dim)
        _SEM_CENT_ALPHA = 0.05              # blend rate per step (very slow)
        _SEM_CENT_THR   = 0.35             # cosine-distance threshold to trigger pull
        _SEM_CENT_STR   = 0.08             # pull strength when triggered

        # Prompt-type anchor multiplier:
        # Questions need stronger topic anchoring (they must stay relevant to the
        # question being asked); continuations can drift more freely.
        _ptype_anchor_mult = {
            "question":     1.20,   # 20 % stronger — stay on-topic
            "statement":    1.00,   # baseline
            "continuation": 0.85,   # allow more drift
            "imperative":   1.10,   # commands stay focused
        }.get(_ptype, 1.00)

        # Per-step score-gap tracker (top1 − top2 raw score):
        # Tracks the decision certainty at each step.  Low gap = uncertain.
        # Accumulated as EMA → `avg_score_gap` in the result dict.
        _ema_score_gap   = 0.0
        _SG_EMA_ALPHA    = 0.20             # fast EMA so recent steps dominate

        # Entropy budget guard:
        # Track cumulative per-step normalised entropy.  If the total budget
        # (max_tokens × 0.65) is exhausted, switch to near-greedy sampling
        # (temperature capped at 0.10) for the remaining steps to stabilise
        # the generation toward a clean ending.
        _entropy_budget     = float(max_tokens) * 0.65  # generous budget
        _entropy_spent      = 0.0
        _entropy_budget_out = False         # flag: budget exhausted

        # Peep hit-rate tracker:
        # Fraction of steps where the final sampled token was in Peep's top-5
        # candidates.  High hit-rate = Peep specialisations are well-calibrated.
        _peep_top5_hits  = 0
        _peep_top5_steps = 0

        # Coherence direction tracker:
        # Rolling slope of coh3 over last 4 steps.
        # Positive = coherence improving; negative = degrading.
        _coh_dir_buf: "deque[float]" = _deque(maxlen=4)
        _coh_dir = 0.0  # most-recent slope estimate

        # Step-wise confidence EMA: smooths single-step noise in confidence
        # before it is used by trend/floor/early-stop logic.
        # α=0.35 — responsive but smoothed (not used for sampling itself).
        _CONF_EMA_ALPHA = 0.35
        _conf_ema       = 0.0   # initialised at step 0

        # Score histogram accumulator (8 buckets [0,0.125), …, [0.875,1.0])
        # for result dict key "score_hist".
        _score_hist = np.zeros(8, dtype=np.int32)

        # Per-step trajectory logs (for cohplot / entropyplot / pctplot CLI):
        _coh3_steps:      list = []   # per-step coh3 values
        _entropy_steps:   list = []   # per-step H_norm values
        _percentile_steps: list = []  # per-step score percentile of chosen token
        _score_var_steps:  list = []  # per-step variance of finite scores

        # Score centroid floor lift:
        # Track the EMA of the mean top-K score.  If it drops sharply
        # (slope < -0.10 over last 3 steps) we add a small additive lift
        # (+0.04) to every finite score to rescue fading candidates before
        # the filter chain zeros them out.
        _score_centroid_ema  = 0.0
        _SC_ALPHA            = 0.30
        _sc_hist             : "deque[float]" = _deque(maxlen=3)
        _SC_DROP_SLOPE       = -0.10   # threshold: sharper drop triggers lift
        _SC_LIFT             = 0.04   # default; overridden adaptively below

        # Per-step top-1 margin tracker:
        # At each step record (score_top1 - score_top2) after all filters.
        # High margin = confident/decisive step; low = wavering.
        _top1_margins: list = []

        # Context velocity magnitude tracker → added to result dict.
        _ctx_vel_mag_ema = 0.0
        _VEL_MAG_ALPHA   = 0.20

        # Per-step velocity + SFB strength + temperature trajectory logs:
        _velocity_steps:    list = []   # per-step ctx_vel_mag_ema values
        _sfb_strength_steps: list = []  # per-step SFB strength values
        _temp_steps:         list = []  # per-step final temperature value
        _topk_steps:         list = []  # per-step adaptive TopK k value

        # Confidence floor early stop:
        # If the last _CF_WIN tokens ALL had confidence below _CF_THR, the model
        # is consistently failing to find reliable continuations — stop early
        # instead of appending low-quality tokens.  Only fires after min_tokens.
        _CF_WIN   = 4
        _CF_THR   = 0.12
        _conf_floor_buf: "deque[float]" = _deque(maxlen=_CF_WIN)

        # Peak confidence step tracker:
        # Record which step index had the highest peak score.
        _peak_conf_val  = 0.0
        _peak_conf_step = 0

        # Generation rhythm tracker:
        # Measures alternation rate in the sign of confidence deltas.
        # +1 if conf rose from prev step, -1 if fell.  High alternation
        # (zigzag ±±±) = low rhythm; consistent run (+++ or ---) = high rhythm.
        # rhythm_score ∈ [0,1]: 1 = perfectly smooth, 0 = every step alternates.
        _rhythm_prev_conf = 0.0
        _rhythm_alts = 0         # alternation count
        _rhythm_steps = 0        # steps counted

        # Last-step dot top-1 index for each dot (for dotscores CLI):
        _last_dot_top1: list = []

        # Per-step top-3 token log (for top3log CLI):
        # Each step stores [(score, word), (score, word), (score, word)]
        _top3_log: list = []

        # Rhythm-adaptive temperature:
        # Tracks rolling alternation rate (3-step window).  When the last
        # 3 steps all alternated sign (zigzag), add a small +0.04 nudge
        # to break out of the pattern.
        _rhythm_delta_signs: "deque[int]" = _deque(maxlen=3)

        # Score-gap oscillation damper:
        # If score gap alternates high-low-high-low (abs diff > 0.08 each flip),
        # the sampler is in an oscillating attractor.  Track sign flips in the
        # last 4 steps; if 3+ flips detected, damp the gap-to-temp mapping by
        # halving the delta for 2 steps.
        _sg_osc_buf: "deque[float]" = _deque(maxlen=4)
        _sg_osc_damp_steps = 0      # steps remaining for damping

        # C6 sustained-drift temperature pulse:
        # Track rolling C6 coherence across steps.  If C6 < 0.05 for 3 straight
        # steps the model has drifted far from topic → fire a one-time +0.12
        # temperature pulse to shake out of the drift attractor.
        _c6_low_streak  = 0
        _C6_LOW_THR     = 0.05
        _C6_LOW_WIN     = 3
        _c6_drift_fired = False   # one-time guard
        _C6_DRIFT_PULSE = 0.12

        # Per-step score-gap history: used by `scorehist` CLI command.
        _sg_step_gaps: list = []

        # Per-step coh6 log: 6-token coherence window at every step ≥6.
        _coh6_steps: list = []

        # Exclusion signal: EMA of visited directions, pre-warmed with prompt
        exclusion = 0.40 * ctx

        # ── Main generation loop ──────────────────────────────────────────
        for _step in range(max_tokens):

            # Step-adaptive EMA coefficient (coherence-augmented):
            # Base: tight hold (0.72) at early steps → loose at late steps.
            # Coherence modifier: if recent tokens are very coherent (≥ 0.30)
            # we can afford a slightly firmer hold (adds 0.04) because the
            # context is moving predictably.  If incoherent (< 0.08) we loosen
            # slightly (-0.04) to let the model explore a new direction.
            _step_frac = float(_step) / max(max_tokens - 1, 1)
            _ca = ctx_alpha + (0.72 - ctx_alpha) * max(0.0, 1.0 - _step_frac * 2.5)
            if len(prev_vecs) >= 2:
                _local_coh = float(prev_vecs[-2] @ prev_vecs[-1])
                if _local_coh >= 0.30:
                    _ca = min(_ca + 0.04, 0.78)   # coherent: hold context firmer
                elif _local_coh < 0.08:
                    _ca = max(_ca - 0.04, ctx_alpha)  # incoherent: let go
            else:
                _local_coh = 0.0
            # Gap-adaptive ctx EMA: high score gap this step means the model is
            # certain about the next token — tighten context hold slightly.
            if _ema_score_gap > 0.15:
                _ca = min(_ca + 0.02, 0.80)
            elif _ema_score_gap < 0.04:
                _ca = max(_ca - 0.02, ctx_alpha)
            _ca = float(np.clip(_ca, ctx_alpha, 0.80))
            _1m_ca = 1.0 - _ca

            # ── 1. Context enrichment ──────────────────────────────────────
            # a) Attention-weighted history aggregation
            ctx_eff = ctx_hist.attend(ctx)
            # b) Prompt-topic anti-drift correction with step-decaying strength.
            # Strong correction at step 0 (0.20) ensures first tokens are
            # on-topic; correction fades to 0.06 at step 10+ to allow natural
            # sentence development without over-constraining the continuation.
            # Adaptive anchor strength: base = 0.20 × 0.88^step → 0.06 floor.
            # Boost by +0.05 if the last 3-step coherence is falling (C3 < 0.10)
            # so the anchor resists semantic drift more aggressively when needed.
            # Also scaled by the prompt-type multiplier so questions/commands
            # stay more tightly anchored than open continuations.
            _anchor_base   = float(max(0.06, 0.20 * (0.88 ** _step)))
            _anchor_c3_mod = 0.05 if (len(prev_vecs) >= 3 and
                                       float(prev_vecs[-3] @ prev_vecs[-2]) +
                                       float(prev_vecs[-2] @ prev_vecs[-1]) < 0.20)  \
                             else 0.0
            anchor.strength = float(np.clip(
                (_anchor_base + _anchor_c3_mod) * _ptype_anchor_mult,
                0.06, 0.35))
            ctx_eff = anchor.correct(ctx_eff)
            # Margin-adaptive anchor boost: indecisive steps → pull back to prompt
            if _margin_ema < 0.03 and anchor.strength < 0.35:
                anchor.strength = float(np.clip(anchor.strength + 0.04, 0.0, 0.35))
            # b2) Soft anchor restart on sustained low score-gap plateau:
            if not _anchor_restarted:
                if _ema_score_gap < _LOW_GAP_THR:
                    _low_gap_streak += 1
                else:
                    _low_gap_streak = 0
                if _low_gap_streak >= _LOW_GAP_WIN:
                    # Re-warm anchor to current context to escape stale pull
                    anchor.anchor = _normalise(ctx_eff[:len(anchor.anchor)].astype(np.float32))
                    _low_gap_streak  = 0
                    _anchor_restarted = True   # allow only one restart per generation

            # c) Exclusion: push away already-visited embedding space
            # Exclusion coefficient is entropy-adaptive: stronger when the
            # distribution is broad/uncertain (prevents random walks), gentler
            # when the model is confident (allows refinement).
            # Additionally: scale exclusion radius by rep_mom magnitude so that
            # a model caught in a repetition loop gets a stronger push away from
            # the visited semantic region it keeps circling.
            # Adaptive warmup: no exclusion for the first 2 steps (free exploration).
            ex_norm = np.linalg.norm(exclusion)
            if ex_norm > 1e-6 and _step >= 2:
                _excl_coeff = getattr(self, "_last_excl_coeff", 0.42)
                # Rep-momentum magnitude: average l1-norm over non-zero entries
                _rep_mom_mag = float(np.mean(np.abs(_rep_mom[_rep_mom > 0.001]))
                                     if np.any(_rep_mom > 0.001) else 0.0)
                _excl_rep_bonus = float(np.clip(_rep_mom_mag * 0.50, 0.0, 0.15))
                _excl_coeff_eff = float(np.clip(_excl_coeff + _excl_rep_bonus,
                                                 0.30, 0.70))
                ctx_eff = ctx_eff - _excl_coeff_eff * (exclusion / ex_norm)
                ctx_eff = ctx_eff / (np.linalg.norm(ctx_eff) + 1e-10)

            # ── 2. Dot predictions ─────────────────────────────────────────
            raw_preds = W_stack @ ctx_eff               # (D, 256)
            mean_pred = raw_preds.mean(axis=0)          # (256,)

            # Dot agreement bonus (pre-scoring):
            # Find the top-1 vocab token for each dot's prediction.
            # Tokens that multiple dots independently predict get a small
            # additive bonus (+0.012 per extra dot beyond the first).
            # Cost: O(D × 256) ≈ 128×256 = 32k — cheap pre-pass.
            _da_top1_per_dot = np.argmax(word_vecs_n @ raw_preds[:, :EMBED_DIM].T, axis=0)  # (D,)
            _last_dot_top1 = _da_top1_per_dot.tolist()  # stash for dotscores
            _da_counts = np.bincount(_da_top1_per_dot, minlength=len(words))  # (V,)
            _da_bonus  = np.where(_da_counts >= 2,
                                  np.minimum(
                                      (_da_counts - 1).astype(np.float32) * 0.012,
                                      0.06),   # cap per-token agreement bonus
                                  0.0)
            # Will apply _da_bonus after raw scoring (added at 6-0 position below)

            # ── 2.5 Speculative two-pass vocab pre-filter ──────────────────
            # Draft pass: score all V words using the top-DRAFT_D dots most
            # aligned with the current context (proxy for most-relevant dots).
            # Keep only the top-DRAFT_KEEP survivors; hard-block the rest.
            # Full pass: run normal voting on all D dots but with only the
            # DRAFT_KEEP survivors eligible — reduces (V,224)@(224,D) cost.
            #
            # Break-even: V × DRAFT_D + DRAFT_KEEP × D ≪ V × D
            # With V=5000, D=128, DRAFT_D=32, DRAFT_KEEP=350:
            #   Draft: 5000×32 = 160k  |  Full: 350×128 = 45k  → 3× speedup
            # Dynamic DRAFT_D: use more draft dots when entropy is high
            # (uncertain steps benefit from wider dot sampling), fewer when low.
            # Range: 24 (confident) to 48 (very uncertain, H_norm > 0.70).
            DRAFT_D    = int(np.clip(24 + 24 * _H_norm, 24, 48))
            DRAFT_KEEP = min(350, len(words))

            if len(words) > DRAFT_KEEP:
                # Select top-DRAFT_D dots by raw alignment with ctx_eff
                _dot_align   = raw_preds @ ctx_eff                           # (D,)
                _draft_idx   = np.argpartition(_dot_align, -DRAFT_D)[-DRAFT_D:]
                _draft_preds = raw_preds[_draft_idx]                         # (32, 256)

                # Draft scoring: mean cosine sim over 32 dots
                _d_sims   = word_vecs_n @ _draft_preds[:, :EMBED_DIM].T     # (V, 32)
                _d_scores = _d_sims.mean(axis=1)                             # (V,)

                # Survivors mask: top-DRAFT_KEEP by draft score
                _surv_idx  = np.argpartition(_d_scores, -DRAFT_KEEP)[-DRAFT_KEEP:]
                _spec_mask = np.ones(len(words), dtype=bool)
                _spec_mask[_surv_idx] = False   # True = blocked

                # Merge with existing block_mask (will be applied in voting)
                _block_merged = (_spec_mask | block_mask
                                 if block_mask is not None else _spec_mask)
            else:
                _block_merged = block_mask   # vocab small enough — skip draft

            # ── 3. Grammar weights ─────────────────────────────────────────
            last_tok = output_tokens[-1] if output_tokens else ""
            prev_tok = output_tokens[-2] if len(output_tokens) >= 2 else ""
            gram_w   = _grammar_guide.weights(last_tok, prev_tok)          # (V,)
            gram_w   = gram_w + _grammar_guide.anti_rep_mask(output_tokens[-4:])

            # ── 4. Adaptive hard-block mask ───────────────────────────────
            # Window grows with step: 4 early (open exploration) → 8 late
            # (tight loop prevention).  Formula: 4 + step // 4, capped at 8.
            _hb_win    = max(4, min(HARD_BLOCK, 4 + _step // 4))
            blocked    = set(output_tokens[-_hb_win:])
            block_mask = (np.array([w in blocked for w in words], dtype=bool)
                          if blocked else None)

            # ── 5. Voting paths → raw score distribution ──────────────────
            if self.peep is not None and self.peep.calibrated:
                # PATH A: Peep top-5 + Multi-Head fusion (70/30)
                top5_idx  = self.peep.top_k(ctx_eff, raw_preds, k=5)       # (5,)
                ctx_n     = ctx_eff / (np.linalg.norm(ctx_eff) + 1e-9)
                top5_sims = self.peep.specialisations[top5_idx] @ ctx_n     # (5,)
                top5_sims = np.clip(top5_sims, 0.0, None)
                sim_sum   = top5_sims.sum()
                top5_w    = (top5_sims / sim_sum if sim_sum > 1e-9
                             else np.ones(5, dtype=np.float32) / 5.0)
                top5_w    = top5_w.astype(np.float32)

                # Peep top-k adaptation: wider candidate set on uncertain steps
                _peep_k_eff = 7 if _H_norm > 0.60 else 5
                if _peep_k_eff != 5:
                    _top_idx_ext  = self.peep.top_k(ctx_eff, raw_preds, k=_peep_k_eff)
                    _sims_ext     = self.peep.specialisations[_top_idx_ext] @ ctx_n
                    _sims_ext     = np.clip(_sims_ext, 0.0, None)
                    _sum_ext      = _sims_ext.sum()
                    _w_ext        = (_sims_ext / _sum_ext if _sum_ext > 1e-9
                                     else np.ones(_peep_k_eff, dtype=np.float32) / _peep_k_eff)
                    top5_idx  = _top_idx_ext
                    top5_w    = _w_ext.astype(np.float32)

                scores, confidence = mhc.peep_forward(
                    raw_preds   = raw_preds,
                    word_vecs_n = word_vecs_n,
                    gram_w      = gram_w,
                    peep_top_k  = top5_idx,
                    peep_weights= top5_w,
                    mean_pred   = mean_pred,
                    block_mask  = _block_merged,
                )

                # ── 5.75 Two-path consensus blend (Path A + lightweight B) ──
                # Compute a lightweight contrastive Path B score and blend
                # 88/12 with Path A.  Tokens that score well on both paths
                # (genuine consensus) are rewarded; tokens only one path
                # likes get slightly diluted.  Adds a diversity cross-check
                # without the full contrastive overhead (uses same raw_preds).
                # Context-weighted: only reward tokens whose embedding is
                # within cos threshold of ctx_eff (context relevance gate).
                _b_alpha  = float(max(0.06, 0.12 * (0.90 ** _step)))
                _b_scores, _ = mhc.contrastive_forward(
                    raw_preds   = raw_preds,
                    ctx_eff     = ctx_eff,
                    word_vecs_n = word_vecs_n,
                    gram_w      = gram_w,
                    block_mask  = _block_merged,
                    alpha       = _b_alpha,
                )
                # Context-weighted blend: attenuate Path-B contribution for
                # tokens whose embedding is far from the current context direction.
                # cos(word, ctx_eff) < 0 → gate = 0.5 (half weight); ≥ 0 → 1.0.
                _ctx_gate = np.where(
                    word_vecs_n @ ctx_eff[:EMBED_DIM] >= 0.0,
                    1.0, 0.5
                ).astype(np.float32)
                _b_scores_gated = _b_scores * _ctx_gate
                scores = 0.88 * scores + 0.12 * _b_scores_gated

                # Peep hit-rate tracking: record whether the final token (to be
                # determined later) is in the Peep top-5 candidate set.
                # We store the top-5 idx set and check after sampling.
                _peep_top5_set_this_step = set(top5_idx.tolist())
                _peep_top5_steps += 1
            else:
                _peep_top5_set_this_step = None
                # PATH B: Contrastive multi-head convergence voting
                # Top-½ alignment dots (expert) vs bottom-½ (amateur).
                # Score = (1+α)*expert − α*amateur  amplifies context-specific signal.
                # Alpha decays with step: starts high (0.18) for early guidance,
                # settles to floor (0.08) once context is established.
                # Decay rate adaptive to generation length: longer runs need a
                # slower decay so contrast doesn't vanish by step 8.
                _alpha_decay = 0.88 if max_tokens <= 16 else max(0.92, 0.88 + 0.005 * (max_tokens - 16))
                _c_alpha = max(0.08, 0.18 * (_alpha_decay ** _step))

                # Dot dropout (attention-dropout analog, dropout_rate=0.10):
                # Randomly zero-out ~10% of dot predictions before voting.
                # Prevents any single dot from dominating across identical steps;
                # adds natural variance that improves N-best diversity.
                # Applied only when D > 16 so small pools are left intact.
                _n_dots = raw_preds.shape[0]
                if _n_dots > 16:
                    _drop_mask = np.random.rand(_n_dots) > 0.10   # True = keep
                    _rp_b      = raw_preds * _drop_mask[:, None]  # zero out 10%
                else:
                    _rp_b = raw_preds

                scores, confidence = mhc.contrastive_forward(
                    raw_preds   = _rp_b,
                    ctx_eff     = ctx_eff,
                    word_vecs_n = word_vecs_n,
                    gram_w      = gram_w,
                    block_mask  = _block_merged,
                    alpha       = _c_alpha,
                )

            # ── 5.5 Cross-head agreement bonus ────────────────────────────
            # Words ranked in top-K by many independent heads are more reliably
            # correct than words favoured by only one head.  Add a small bonus
            # proportional to the fraction of heads that agree on each word.
            # This is the IECNN ensemble-agreement signal (no gradient needed).
            _agree_bonus = mhc.agreement_bonus(
                raw_preds=raw_preds, word_vecs_n=word_vecs_n,
                top_k=10, strength=0.10)
            scores = scores + _agree_bonus

            # ── 5.55 Head-spread penalty ───────────────────────────────────
            # Words where the per-head cosine scores span a large range
            # (heads strongly disagree) receive a penalty.  Complements the
            # agreement bonus by also penalising high-spread unreliable tokens.
            _spread_pen = mhc.head_spread_penalty(
                raw_preds=raw_preds, word_vecs_n=word_vecs_n, strength=0.05)
            scores = scores - _spread_pen

            # ── 5.6 Dot variance penalty ──────────────────────────────────
            # Penalise words where the per-dot cosine scores span a wide range
            # (high variance = dot disagreement = unreliable signal).
            # Applied directly on raw_preds so it runs before the filter chain.
            scores = _dot_var_pen(scores,
                                  raw_preds=raw_preds,
                                  word_vecs_n=word_vecs_n)

            # ── 6. Score processor pipeline ───────────────────────────────

            # 6-0. Local semantic restriction — keep top-200 context-close words
            scores = _local_sem(scores,
                                word_vecs_n=word_vecs_n,
                                ctx_eff=ctx_eff)

            # 6-0a. Dot agreement bonus application:
            # Add the pre-computed agreement bonus (≥2 dots agree → +0.012 each).
            # Only apply to tokens with a valid score.
            _da_valid = scores > -1e9
            scores = np.where(_da_valid, scores + _da_bonus[:len(scores)], scores)

            # 6-0b. Score centroid floor lift:
            # Compute mean of top-10 valid scores, track EMA, apply lift if
            # the centroid is declining — rescues a fading candidate pool.
            _sc_valid = scores[scores > -1e9]
            if len(_sc_valid) >= 10:
                _sc_top10_mean = float(np.partition(_sc_valid, -10)[-10:].mean())
            elif len(_sc_valid) > 0:
                _sc_top10_mean = float(_sc_valid.mean())
            else:
                _sc_top10_mean = 0.0
            _score_centroid_ema = ((1.0 - _SC_ALPHA) * _score_centroid_ema
                                   + _SC_ALPHA * _sc_top10_mean)
            _sc_hist.append(_score_centroid_ema)
            if len(_sc_hist) == 3:
                _sc_x = np.arange(3, dtype=np.float32)
                _sc_y = np.array(list(_sc_hist), dtype=np.float32)
                _sc_slope = float(np.polyfit(_sc_x, _sc_y, 1)[0])
                if _sc_slope < _SC_DROP_SLOPE:
                    # coh_dir-adaptive lift: stronger correction when coherence is falling
                    _sc_lift_now = (0.05 if _coh_dir < -0.04 else
                                    0.03 if _coh_dir > 0.04 else _SC_LIFT)
                    scores = np.where(scores > -1e9, scores + _sc_lift_now, scores)

            # 6-0c. Coherence-variance SFB boost:
            # Measure std-dev of recent cosine similarities (last 4 steps).
            # High variance = topic zigzagging → add a small SFB boost to
            # pull back toward the prompt topic.
            if len(prev_vecs) >= 4:
                _coh_win_sims = np.array([
                    float(prev_vecs[-(i+2)] @ prev_vecs[-(i+1)])
                    for i in range(3)
                ], dtype=np.float32)
                _coh_var = float(np.std(_coh_win_sims))
                if _coh_var > 0.15:
                    _sfb.strength = min(_sfb.strength + 0.04, semantic_bias + 0.10)

            # 6-1. Semantic field bias (prompt-topic coherence)
            # Coherence-guided SFB strength:
            # When C3 is strong (≥ 0.30) the model is already on-topic —
            # reduce SFB strength to avoid over-constraining.
            # When C3 is weak (< 0.08) the model is drifting — boost SFB.
            if len(prev_vecs) >= 3:
                _sfb_coh3 = (float(prev_vecs[-3] @ prev_vecs[-2]) +
                             float(prev_vecs[-2] @ prev_vecs[-1])) / 2.0
                if _sfb_coh3 >= 0.30:
                    _sfb.strength = max(semantic_bias * 0.70, semantic_bias - 0.04)
                elif _sfb_coh3 < 0.08:
                    _sfb.strength = min(semantic_bias * 1.40, semantic_bias + 0.06)
                else:
                    _sfb.strength = semantic_bias
            scores = _sfb(scores)

            # 6-1b. Prompt drift penalty — penalise prompt-distant tokens
            scores = _pdrift(scores)

            # 6-2. Vocabulary frequency prior (de-rank ultra-rare tokens)
            scores = _freq_prior(scores)

            # 6-2a. Surprise bonus — reward context-close non-top-50 tokens
            scores = _surprise(scores,
                               word_vecs_n=word_vecs_n,
                               ctx_eff=ctx_eff)

            # 6-2b. Bigram continuation bonus (data-driven collocation boost)
            scores = _bigram_bonus(scores, last_token=last_tok)

            # 6-2c. Semantic proximity penalty (near-synonym suppression)
            # Builds recent_vecs_n from the last 4 entries of prev_vecs.
            # prev_vecs stores unit-norm 224-dim token embeddings, so the
            # matrix multiply (V,224) @ (K,224).T is correct without padding.
            _rv = (np.stack(prev_vecs[-4:]).astype(np.float32)
                   if prev_vecs else None)
            scores = _prox_pen(scores,
                               word_vecs_n=word_vecs_n,
                               recent_vecs_n=_rv)

            # 6-3. Presence + frequency repetition penalty (CTRL-style)
            # Recency decay 0.88: presence penalty for tokens generated 8+
            # steps ago decays to ~37 %, so distant tokens are penalised less.
            _scores_pre_rep = scores.copy()
            scores = _rep(scores, generated_ids=generated_ids,
                          recency_decay=0.88)

            # 6-3a. Near-window hard-boost repetition:
            # Tokens from the last 3 steps get an extra presence penalty
            # on top of the recency-decayed penalty above.  This specifically
            # targets burst-repetition (same word twice in a row) which the
            # EMA momentum alone may not suppress fast enough.
            # Entropy-adaptive magnitude: high entropy → stronger penalty.
            _nw_pen = 0.08 * (0.70 + 0.60 * _H_norm)   # range ~0.056 – 0.104
            for _rw_tok in output_tokens[-3:]:
                if _rw_tok in word_index:
                    _rw_i = word_index[_rw_tok]
                    if scores[_rw_i] > -1e9:
                        scores[_rw_i] -= _nw_pen

            # 6-3b. Penalty momentum carry-over:
            # Compute how much this step's rep penalty added, then blend into
            # the momentum buffer and apply a fraction next step.
            _rep_delta  = np.maximum(0.0, _scores_pre_rep - scores)   # (V,)
            _rep_mom    = _REP_MOM_B1 * _rep_mom + (1.0 - _REP_MOM_B1) * _rep_delta
            scores      = scores - _REP_MOM_S * _rep_mom

            # 6-3c. Repetition burst detector:
            # Count how many unique tokens in the last window are duplicated.
            # Low-margin burst trigger: update count and arm pulse if threshold met.
            _cur_margin = _top1_margins[-1] if _top1_margins else 0.0
            _margin_ema = (1.0 - _MARGIN_EMA_A) * _margin_ema + _MARGIN_EMA_A * _cur_margin
            if _cur_margin < _LOW_MARGIN_THR:
                _low_margin_count += 1
                if _low_margin_count >= _LOW_MARGIN_WIN and _burst_pulse == 0.0:
                    _rep_mom[:] = 0.0
                    _burst_pulse = _BURST_TEMP * 0.75   # lighter than full burst
                    _low_margin_count = 0
            else:
                _low_margin_count = 0

            # If a burst is detected, clear rep_mom (so penalties don't compound
            # stale history) and arm a one-step temperature pulse.
            if len(output_tokens) >= _BURST_WIN:
                _bwin = output_tokens[-_BURST_WIN:]
                from collections import Counter as _BCounter
                _bcnt = _BCounter(_bwin)
                if any(v >= _BURST_THRESH for v in _bcnt.values()):
                    _rep_mom[:] = 0.0          # clear stale momentum
                    _burst_pulse = _BURST_TEMP  # arm temperature pulse

            # 6-4. No-repeat trigram + bigram hard blocking
            scores = _ngram3(scores, output_tokens=output_tokens,
                             word_index=word_index)
            scores = _ngram2(scores, output_tokens=output_tokens,
                             word_index=word_index)

            # 6-5. SimCTG degeneration penalty — embedding-level anti-repeat
            # Adaptive alpha: when C3 coherence is strong the model is on-topic
            # so soften the penalty; when coherence is poor, tighten it.
            pv_arr = (np.stack(prev_vecs[-16:]).astype(np.float32)
                      if prev_vecs else None)
            if len(prev_vecs) >= 3:
                _deg_coh3 = (float(prev_vecs[-3] @ prev_vecs[-2]) +
                             float(prev_vecs[-2] @ prev_vecs[-1])) / 2.0
                _deg.alpha = float(np.clip(0.50 - 0.30 * _deg_coh3, 0.20, 0.65))
            scores = _deg(scores, word_vecs_n=word_vecs_n, prev_vecs=pv_arr)

            # 6-6. Minimum length guard (block stop tokens early)
            scores = _minlen(scores, step=_step, stop_ids=stop_ids)

            # 6-7. Exponential decay length penalty (smooth natural ending)
            scores = _expdecay(scores, step=_step, stop_ids=stop_ids)

            # 6-8. Depth-2 lookahead confidence pre-scorer
            # Depth-1: simulate one step ahead for top-K candidates.
            # Depth-2 (top-2 only): simulate a second step from the depth-1
            # best successor to score how promising the path continues.
            # Tokens leading to strong 2-step paths get a higher bonus.
            # Cost: depth-1 O(K × D × 256) + depth-2 O(2 × D × 256)
            _LH_K      = 3
            _LH_WEIGHT = 0.06
            _LH_D2_W   = 0.03   # smaller weight for depth-2 (noisier signal)
            _lh_valid  = scores > -1e9
            if _lh_valid.sum() > _LH_K and _step < max_tokens - 1:
                _lh_top   = np.argpartition(
                    np.where(_lh_valid, scores, -1e18), -_LH_K)[-_LH_K:]
                _lh_bonus = np.zeros_like(scores)
                _lh_d1_strengths: list = []
                for _ci in _lh_top:
                    _nx = _ca * ctx_eff + _1m_ca * _wvp_n[_ci]
                    _nx = _nx / (np.linalg.norm(_nx) + 1e-10)
                    _d1_str = float(np.linalg.norm((W_stack @ _nx).mean(axis=0)))
                    _lh_bonus[_ci] = _d1_str * _LH_WEIGHT
                    _lh_d1_strengths.append((_ci, _nx, _d1_str))

                # Depth-2: take the 2 best depth-1 paths, simulate one more step.
                # Skip on final 2 steps — no planning benefit near end of generation.
                _lh_d1_sorted = sorted(_lh_d1_strengths, key=lambda x: x[2], reverse=True)[:2]
                for _ci, _nx1, _ in _lh_d1_sorted if _step < max_tokens - 2 else []:
                    # Predict top-1 successor from depth-1 context
                    _nx1_preds  = W_stack @ _nx1                       # (D, 256)
                    _nx1_mean   = _nx1_preds.mean(axis=0)              # (256,)
                    _nx1_sims   = word_vecs_n @ _nx1_mean[:EMBED_DIM]  # (V,)
                    _nx1_best_i = int(np.argmax(_nx1_sims))
                    # Simulate the context after depth-2 token
                    _wvp_d2 = word_vecs_n[_nx1_best_i]                # (embed_dim,)
                    _nx2_pad = np.zeros(len(_nx1), dtype=np.float32)
                    _nx2_pad[:EMBED_DIM] = _wvp_d2.astype(np.float32)
                    _nx2 = _ca * _nx1 + _1m_ca * _nx2_pad
                    _nx2 /= (np.linalg.norm(_nx2) + 1e-10)
                    _d2_str = float(np.linalg.norm((W_stack @ _nx2).mean(axis=0)))
                    _lh_bonus[_ci] += _d2_str * _LH_D2_W

                scores = scores + _lh_bonus

            # Score spread clipping:
            # If the spread of valid scores (max − min) exceeds 1.6, the
            # scoring distribution is skewed by an outlier-high token.
            # Clip the maximum score down to min + 1.6 to prevent any single
            # token from dominating the softmax and collapsing diversity.
            _ss_valid = scores > -1e9
            if _ss_valid.sum() >= 3:
                _ss_min = float(scores[_ss_valid].min())
                _ss_max = float(scores[_ss_valid].max())
                if _ss_max - _ss_min > 1.6:
                    scores = np.where(_ss_valid,
                                      np.minimum(scores, _ss_min + 1.6),
                                      scores)

            # Score floor clamp: after all additive penalties, ensure no
            # finite score falls below -3.0.  Stacked penalties (rep_mom +
            # prox_pen + deg + pdrift) can push tail scores to -5 or lower,
            # making softmax_sample numerically unstable (exp overflow).
            # -3.0 is well below every real signal score so ranking is unchanged.
            # Rhythm-adaptive floor: low rhythm (<0.4) → broaden pool by +0.02.
            _rt_now = 1.0 - (_rhythm_alts / max(_rhythm_steps, 1))
            _floor_base = -3.0 + (0.02 if _rt_now < 0.40 else 0.0)
            _valid_fin = scores > -1e9
            scores     = np.where(_valid_fin,
                                  np.maximum(scores, _floor_base),
                                  scores)

            # ── 7. Confidence check (on processed scores) ─────────────────
            valid_mask = scores > -1e9
            if not valid_mask.any():
                stop_reason = "no_valid_candidates"
                break
            peak_score = float(scores[valid_mask].max())
            confidence = float(np.clip(peak_score, 0.0, 1.0))

            # Step-wise confidence EMA smoothing (for tracking; does not alter sampling)
            if _step == 0:
                _conf_ema = confidence
            else:
                _conf_ema = (1.0 - _CONF_EMA_ALPHA) * _conf_ema + _CONF_EMA_ALPHA * confidence

            # Accumulate score histogram bucket
            _sh_bucket = min(7, int(confidence * 8))
            _score_hist[_sh_bucket] += 1

            # Rhythm tracker: count confidence delta sign alternations
            if _step > 0:
                _rhythm_steps += 1
                _conf_delta = confidence - _rhythm_prev_conf
                _prev_delta = _rhythm_prev_conf - (output_confs[-1] if output_confs else 0.0)
                if _rhythm_steps > 1 and (_conf_delta * _prev_delta < 0):
                    _rhythm_alts += 1
            _rhythm_prev_conf = confidence

            # Track peak confidence step
            if confidence > _peak_conf_val:
                _peak_conf_val  = confidence
                _peak_conf_step = _step

            # Confidence floor early stop: uses EMA-smoothed confidence so that
            # a single noisy low-confidence step cannot prematurely terminate.
            _conf_floor_buf.append(_conf_ema)
            if (_step >= min_tokens and len(_conf_floor_buf) == _CF_WIN
                    and all(c < _CF_THR for c in _conf_floor_buf)):
                stop_reason = f"conf_floor(ema={_conf_ema:.3f})"
                break

            if _step >= min_tokens and confidence < CONF_STOP:
                stop_reason = f"low_confidence({confidence:.3f})"
                break

            # High-quality early stop:
            # If we are already in a very confident + locally coherent state
            # past min_tokens + 2, additional tokens are likely to lower
            # quality (regression to the mean / overextension).
            if (_step >= min_tokens + 2
                    and confidence >= 0.78
                    and len(prev_vecs) >= 2):
                _hq_coh = float(prev_vecs[-2] @ prev_vecs[-1])
                if _hq_coh >= 0.55 and confidence * _hq_coh > 0.50:
                    stop_reason = "high_quality_stop"
                    break

            # Local coherence extension gate:
            # If coh3 ≥ 0.50 on the first step past min_tokens, allow
            # one extra token — the generation is on an excellent track.
            # Fires at most once per generation.
            if (len(prev_vecs) >= 3 and _step == min_tokens
                    and not getattr(self, "_coh_ext_fired_thisgen", False)):
                _ceh_coh = (float(prev_vecs[-3] @ prev_vecs[-2]) +
                            float(prev_vecs[-2] @ prev_vecs[-1])) / 2.0
                if _ceh_coh >= 0.50:
                    self._coh_ext_fired_thisgen = True
                    min_tokens += 1   # allow one extra token

            # Early C3 strength relaxation:
            # If C3 is very strong (≥ 0.40, or 0.35 when vprec_ema is high)
            # already by step 3, the model is in a very coherent run —
            # allow stopping 1 step before min_tokens if confidence is also
            # high (≥ 0.70).  Only fires once.
            _ec3_thr = 0.35 if _vprec_ema >= 0.65 else 0.40
            if (_step >= max(1, min_tokens - 1)
                    and not getattr(self, "_early_c3_fired_thisgen", False)
                    and len(prev_vecs) >= 3):
                _ec3 = (float(prev_vecs[-3] @ prev_vecs[-2]) +
                        float(prev_vecs[-2] @ prev_vecs[-1])) / 2.0
                if _ec3 >= _ec3_thr and confidence >= 0.70:
                    self._early_c3_fired_thisgen = True
                    self._coh_ext_fired_thisgen  = False  # reset for safety
                    stop_reason = "early_c3_stop"
                    break

            # Multi-window coherence degeneration guard.
            # Two independent windows catch different failure modes:
            #   3-step (sensitive)  — catches abrupt cosine collapse (noise burst)
            #   6-step (persistent) — catches slow-rolling semantic drift
            # A single-window guard misses gradual drift; dual-trigger is robust.
            if len(prev_vecs) >= 3 and _step >= min_tokens:
                _c3a = float(prev_vecs[-3] @ prev_vecs[-2])
                _c3b = float(prev_vecs[-2] @ prev_vecs[-1])
                _coh3 = 0.5 * _c3a + 0.5 * _c3b
                _coh3_steps.append(round(_coh3, 5))  # log for cohplot

                _coh6 = 1.0   # default: no signal when buffer < 6
                if len(prev_vecs) >= 6:
                    _coh6 = float(np.mean([
                        float(prev_vecs[i] @ prev_vecs[i + 1])
                        for i in range(-6, -1)
                    ]))
                    _coh6_steps.append(round(_coh6, 5))  # log for coh6plot

                # Coherence direction: update rolling slope
                _coh_dir_buf.append(_coh3)
                if len(_coh_dir_buf) >= 4:
                    _cd = list(_coh_dir_buf)
                    _coh_dir = float(
                        sum((_cd[i+1] - _cd[i]) for i in range(len(_cd)-1))
                        / max(len(_cd)-1, 1)
                    )

                # Adaptive degeneration threshold:
                # Strong falling coh_dir (< -0.06) → tighten guard (0.06).
                # Gentle or rising → keep default (0.04).
                _degen_coh3_thr = 0.06 if _coh_dir < -0.06 else 0.04
                _degenerate = (_coh3 < _degen_coh3_thr) or (len(prev_vecs) >= 6 and _coh6 < 0.10)
                if _degenerate:
                    stop_reason = (
                        f"degeneration(coh3={_coh3:.3f},coh6={_coh6:.3f})"
                    )
                    break

            # Entropy-adaptive exclusion: compute entropy of current distribution
            # to scale the exclusion coefficient at the next step.
            # Also update the EMA entropy tracker for the result dict.
            _valid = scores[scores > -1e9]
            if len(_valid) > 1:
                _s = _valid - _valid.max()
                _p = np.exp(_s); _p /= _p.sum()
                _H  = float(-(_p * np.log(_p + 1e-30)).sum())
                _H_norm = float(np.clip(_H / np.log(max(len(_p), 2)), 0.0, 1.0))
                _entropy_steps.append(round(_H_norm, 5))   # log for entropyplot
                # High entropy → stronger exclusion (0.50); low → gentler (0.35)
                # Velocity-adaptive exclusion: high ctx velocity means the model
                # is already moving away from visited space — loosen exclusion.
                _vel_adj = float(np.clip(0.08 * _ctx_vel_mag_ema, 0.0, 0.08))
                self._last_excl_coeff = 0.35 + 0.15 * _H_norm - _vel_adj
                # EMA entropy: α=0.15 — slow tracker of generation uncertainty
                _ema_entropy = 0.85 * _ema_entropy + 0.15 * _H_norm
            else:
                self._last_excl_coeff = 0.42

            # ── 8. Four-layer filtering: typical → nucleus → eta → min-p ─────
            # Typical removes over-predictable AND over-surprising tokens.
            # Nucleus caps the tail by probability mass with an adaptive p:
            #   early steps (0-3): p=0.92  — wider beam, richer exploration
            #   late steps (8+):   p=0.87  — tighter focus once context set
            # Eta trims entropy-adaptive stragglers.
            # Min-p removes anything below min_p × peak_probability.
            _nuc_p = float(np.clip(0.92 - 0.05 * min(_step / 8.0, 1.0), 0.80, 0.95))
            # Margin-adaptive nucleus: very low margin_ema → widen nucleus (+0.03)
            # to surface more diverse candidates and escape flat score plateaus.
            if _margin_ema < 0.03:
                _nuc_p = float(np.clip(_nuc_p + 0.03, 0.80, 0.97))
            # Adaptive min_p: higher entropy → smaller threshold (more permissive).
            # Also gated by vocab precision EMA: high vocab_prec means the model
            # is consistently picking context-close words → widen threshold slightly
            # (allow more diversity); low vprec = struggling → tighten filter.
            # Update vocab-prec EMA (α=0.20) for smoother adaptive min_p
            _vprec_raw = float(_vocab_prec_hits / max(_vocab_prec_total, 1))
            _vprec_ema = (1.0 - _VPREC_EMA_ALPHA) * _vprec_ema + _VPREC_EMA_ALPHA * _vprec_raw
            _vprec_now = _vprec_ema   # use EMA instead of raw
            _min_p_vprec_adj = float(np.clip(0.50 - _vprec_now, -0.20, 0.20)) * 0.01
            _min_p_eff = float(np.clip(
                0.05 * (1.0 - 0.40 * _H_norm) + _min_p_vprec_adj,
                0.02, 0.06))
            _scores_before_filters = scores.copy()   # saved for beam entropy gate
            # Adaptive TopK from entropy: high H_norm → more candidates (k→50);
            # low H_norm → fewer (k→30) since distribution is already peaked.
            _topk.k = int(np.clip(30 + int(20 * _H_norm), 20, 60))
            _topk_steps.append(_topk.k)  # log for topkplot
            scores = _topk(scores)
            # Adaptive typical p from coh3: high coherence → tighter (0.92);
            # low coherence → wider (0.97) to keep more candidates.
            if len(prev_vecs) >= 3:
                _typ_coh = (float(prev_vecs[-3] @ prev_vecs[-2]) +
                            float(prev_vecs[-2] @ prev_vecs[-1])) / 2.0
                _typical.p = float(np.clip(0.95 - 0.03 * _typ_coh, 0.88, 0.97))
            scores = _typical(scores)
            # Adaptive TFS z from entropy: high H_norm → tighter z (cut more tail);
            # low H_norm → looser (distribution already focused).
            _tfs.z = float(np.clip(0.97 - 0.04 * _H_norm, 0.90, 0.98))
            scores = _tfs(scores)
            scores = _nucleus(scores, p=_nuc_p)
            # Adaptive eta: high H_norm → loosen epsilon (wider net);
            # low H_norm → tighten (fewer survivors needed).
            _eta.epsilon = float(np.clip(3e-4 * (1.0 + 2.0 * _H_norm), 3e-4, 9e-4))
            scores = _eta(scores)
            scores = _minp(scores, min_p=_min_p_eff)

            # Short-token filler penalty: tokens ≤ 2 chars are likely function
            # words ("a", "in", "of") that dominate when the model is uncertain.
            # Apply a small score demerit (-0.025) to discourage filler run-ons.
            _filler_valid = scores > -1e9
            if _filler_valid.sum() > 3:   # only when we have genuine alternatives
                for _fi, _fw in enumerate(words):
                    if scores[_fi] > -1e9 and len(_fw) <= 2:
                        scores[_fi] -= 0.025

            # ── 8.5 Beam entropy gate (too few survivors) ─────────────────
            # If the filter chain removed too aggressively (< 3 valid survivors),
            # fall back to the top-3 pre-filter scores.  This prevents the model
            # from getting stuck with only 1 candidate on every step, which
            # produces extremely deterministic and repetitive output.
            _hb_block = np.array([w in blocked for w in words], dtype=bool) \
                        if blocked else np.zeros(len(words), dtype=bool)
            _n_surv = int((scores > -1e9).sum())
            if _n_surv < 3:
                _gate_top3 = np.argpartition(
                    np.where(_scores_before_filters > -1e9,
                             _scores_before_filters, -1e18), -3)[-3:]
                for _gi in _gate_top3:
                    if not _hb_block[_gi]:
                        scores[_gi] = _scores_before_filters[_gi]
                _n_surv = int((scores > -1e9).sum())

            # ── 8.6 Low-entropy collapse gate (too few effective survivors) ─
            # Opposite problem: if only 1 or 2 tokens survived AND they all
            # have nearly-identical scores (post-filter entropy ≈ 0), the model
            # has collapsed to near-deterministic output.  Widen by restoring
            # the top-5 pre-filter scores, giving diversity a chance.
            # Guard: only trigger when we have fewer than 2 meaningful survivors
            # (score spread < 0.02) to avoid interfering with genuine confidence.
            if _n_surv <= 2:
                _gate_top5 = np.argpartition(
                    np.where(_scores_before_filters > -1e9,
                             _scores_before_filters, -1e18), -5)[-5:]
                for _gi in _gate_top5:
                    if not _hb_block[_gi]:
                        scores[_gi] = _scores_before_filters[_gi]
            else:
                # Check if surviving scores are suspiciously flat
                _surv_scores = scores[scores > -1e9]
                if len(_surv_scores) >= 2:
                    _surv_spread = float(_surv_scores.max() - _surv_scores.min())
                    if _surv_spread < 0.02 and len(_surv_scores) <= 3:
                        _gate_top5 = np.argpartition(
                            np.where(_scores_before_filters > -1e9,
                                     _scores_before_filters, -1e18), -5)[-5:]
                        for _gi in _gate_top5:
                            if not _hb_block[_gi]:
                                scores[_gi] = _scores_before_filters[_gi]

            # ── 8.7 Score-gap EMA tracker ──────────────────────────────────
            # Track top1−top2 score gap each step; high gap = confident step.
            _sg_valid = scores[scores > -1e9]
            if len(_sg_valid) >= 2:
                _sg_top2   = np.partition(_sg_valid, -2)[-2:]
                _sg_gap    = float(_sg_top2[1] - _sg_top2[0])
                _ema_score_gap = ((1.0 - _SG_EMA_ALPHA) * _ema_score_gap
                                  + _SG_EMA_ALPHA * max(0.0, _sg_gap))

            # Dynamic local-semantic top_k:
            # Wider pool (250) when uncertain, narrower (150) when confident.
            # Final 3 steps: tighten to 120 to force focused ending candidates.
            _steps_remaining = max_tokens - _step - 1
            if _steps_remaining <= 2:
                _local_sem.top_k = 120   # tight focus for clean ending
            else:
                _local_sem.top_k = int(np.clip(150 + 100 * _H_norm, 150, 250))

            # ── 9. Temperature scheduling ──────────────────────────────────
            # First 3 steps: DynamicTemperature (warms up with trend).
            # After that: Mirostat v2 (feedback loop toward target confidence).
            if _step < 3:
                temp = _dtemp.get(confidence, output_confs)
            else:
                # Mirostat target adaptation from local coherence (C3):
                # If recent tokens are very coherent (coh≥0.30), the model is
                # on a good track — lower the target slightly (0.35) for a
                # tighter distribution.  If incoherent (coh<0.08), raise it
                # (0.42) to force more exploration.
                if len(prev_vecs) >= 3:
                    _miro_coh3 = (float(prev_vecs[-3] @ prev_vecs[-2]) +
                                  float(prev_vecs[-2] @ prev_vecs[-1])) / 2.0
                    if _miro_coh3 >= 0.30:
                        _mirostat.target = 0.35
                    elif _miro_coh3 < 0.08:
                        _mirostat.target = 0.42
                    else:
                        _mirostat.target = 0.38  # baseline
                    # vprec_ema adjustment: high precision → tighter target (0.32)
                    if _vprec_ema >= 0.70:
                        _mirostat.target = min(_mirostat.target, 0.32)
                temp = _mirostat.get()

            # Confidence trend boost: if the last 4+ confidences have a
            # slope steeper than _TREND_SLOPE (declining), add a temperature
            # boost so the sampler explores rather than collapsing on a rut.
            _conf_window.append(confidence)
            if len(_conf_window) >= 4:
                _cw_arr  = np.array(_conf_window, dtype=np.float32)
                _cw_x    = np.arange(len(_cw_arr), dtype=np.float32)
                _trend   = float(np.polyfit(_cw_x, _cw_arr, 1)[0])
                if _trend < _TREND_SLOPE:
                    temp = float(np.clip(temp + _TREND_BOOST, 0.0, 0.90))

            # Score-gap guided temperature:
            # High score gap (confident step) → lower temp (be decisive).
            # Low score gap (uncertain step)  → raise temp slightly (explore).
            # Applied as a small delta to avoid fighting the Mirostat feedback.
            # Score-gap oscillation damper: halve the delta when oscillating.
            _sg_step_gaps.append(_ema_score_gap)
            _sg_osc_buf.append(_ema_score_gap)
            _sg_osc_delta_scale = 0.5 if _sg_osc_damp_steps > 0 else 1.0
            if _sg_osc_damp_steps > 0:
                _sg_osc_damp_steps -= 1
            elif len(_sg_osc_buf) == 4:
                _signs = [1 if g > 0.08 else -1 for g in _sg_osc_buf]
                _flips = sum(1 for i in range(1, len(_signs)) if _signs[i] != _signs[i-1])
                if _flips >= 3:
                    _sg_osc_damp_steps = 2
                    _sg_osc_delta_scale = 0.5
            if _ema_score_gap > 0.15:
                temp = float(np.clip(
                    temp - 0.05 * min(_ema_score_gap / 0.15, 1.0) * _sg_osc_delta_scale,
                    0.0, 0.90))
            elif _ema_score_gap < 0.05 and _step >= 3:
                temp = float(np.clip(temp + 0.05 * _sg_osc_delta_scale, 0.0, 0.90))

            # C6 sustained-drift temperature pulse:
            # If C6 has been below threshold for 3 straight steps, fire a one-time
            # temperature pulse to break out of the drift attractor.
            if len(prev_vecs) >= 6:
                _c6_now = float(np.mean([
                    float(prev_vecs[-(i+2)] @ prev_vecs[-(i+1)])
                    for i in range(min(5, len(prev_vecs)-1))
                ]))
                if _c6_now < _C6_LOW_THR:
                    _c6_low_streak += 1
                else:
                    _c6_low_streak = 0
                if _c6_low_streak >= _C6_LOW_WIN and not _c6_drift_fired:
                    temp = float(np.clip(temp + _C6_DRIFT_PULSE, 0.0, 0.90))
                    _c6_drift_fired = True   # one-time only

            # Margin-adaptive temperature:
            # High margin EMA (decisive) → gentle temp reduction (-0.02).
            # Low margin EMA (indecisive) → gentle temp increase (+0.02).
            if _margin_ema > 0.15:
                temp = float(np.clip(temp - 0.02, 0.0, 0.90))
            elif _margin_ema < 0.03:
                temp = float(np.clip(temp + 0.02, 0.0, 0.90))

            # Rhythm-adaptive temperature:
            # If last 3 confidence deltas all alternated sign, add +0.04
            # to break the zigzag pattern.
            _rhythm_delta_signs.append(
                1 if confidence > _rhythm_prev_conf else -1
            )
            if len(_rhythm_delta_signs) == 3:
                _rs = list(_rhythm_delta_signs)
                if _rs[0] != _rs[1] and _rs[1] != _rs[2]:   # alternating: +-+ or -+-
                    temp = float(np.clip(temp + 0.04, 0.0, 0.90))

            # Burst-pulse: fire once (one step), then clear.
            if _burst_pulse > 0.0:
                temp = float(np.clip(temp + _burst_pulse, 0.0, 0.90))
                _burst_pulse = 0.0

            # Low-confidence recovery pulse:
            # If this step's confidence is extremely low (< 0.15), the model
            # is very uncertain.  Add a small temperature nudge (+0.08) to
            # escape the current low-confidence attractor next step.
            if confidence < 0.15 and _step >= 2:
                temp = float(np.clip(temp + 0.08, 0.0, 0.90))

            # Adaptive temperature lower bound from RW confidence EMA:
            # Confident generations can use a tighter (lower) temperature floor.
            _rw_conf_ema = ((1.0 - _RW_CONF_ALPHA) * _rw_conf_ema
                            + _RW_CONF_ALPHA * confidence)
            if _rw_conf_ema >= 0.50:
                _temp_lb = 0.08
            elif _rw_conf_ema < 0.30:
                _temp_lb = 0.20
            else:
                _temp_lb = 0.15
            temp = float(max(temp, _temp_lb))

            # Entropy budget guard:
            # If cumulative entropy spending has exceeded budget, cap temp.
            _entropy_spent += _H_norm
            if not _entropy_budget_out and _entropy_spent >= _entropy_budget:
                _entropy_budget_out = True
            if _entropy_budget_out:
                temp = float(min(temp, 0.10))   # near-greedy finishing

            _temp_steps.append(round(temp, 4))   # log for tempplot

            # Per-step top-1 margin: difference between top-1 and top-2 scores.
            _m_valid = scores > -1e9
            if _m_valid.sum() >= 2:
                _m_top2 = np.partition(scores[_m_valid], -2)[-2:]
                _top1_margins.append(round(float(_m_top2[-1] - _m_top2[-2]), 5))
            else:
                _top1_margins.append(0.0)

            # Per-step score percentile: where does the winning token rank among
            # finite scores? 1.0 = top of distribution, 0.0 = worst.
            if _m_valid.sum() >= 2:
                _m_fin = scores[_m_valid]
                _m_win = float(scores[next_idx]) if scores[next_idx] > -1e9 else float(_m_fin.max())
                _m_pct = float(np.mean(_m_fin <= _m_win))
                _percentile_steps.append(round(_m_pct, 4))
                _score_var_steps.append(round(float(np.var(_m_fin)), 5))
            else:
                _percentile_steps.append(1.0)
                _score_var_steps.append(0.0)

            # Per-step top-3 token log: record top-3 candidates by score.
            _t3_valid = scores > -1e9
            if _t3_valid.sum() >= 3:
                _t3_top3 = np.argpartition(
                    np.where(_t3_valid, scores, -1e18), -3)[-3:]
                _t3_top3 = _t3_top3[np.argsort(scores[_t3_top3])[::-1]]
                _top3_log.append([
                    (float(scores[_ti]), words[_ti]) for _ti in _t3_top3
                ])
            else:
                _top3_log.append([])

            next_idx   = softmax_sample(scores, temperature=temp)
            next_token = words[next_idx]

            # Peep hit-rate: check if sampled token was in Peep's top-5.
            if _peep_top5_set_this_step is not None:
                if next_idx in _peep_top5_set_this_step:
                    _peep_top5_hits += 1

            # Vocabulary precision: track if chosen token was in local_sem pool.
            # A high precision (→1.0) means the model consistently picks words
            # that are semantically close to the current context direction.
            _vocab_prec_total += 1
            if _scores_before_filters[next_idx] > -1e9:
                # token was in the pre-filter pool (proxy for local_sem survival)
                _vocab_prec_hits += 1

            # Mirostat update: adjust temperature estimate for next step
            _mirostat.update(confidence)

            # ── 10. Emit token ─────────────────────────────────────────────
            output_tokens.append(next_token)
            output_confs.append(confidence)
            generated_ids.append(next_idx)

            # Natural stop check
            if next_token in STOP_TOKENS:
                stop_reason = "natural_stop"
                break

            # ── 11. Context update ─────────────────────────────────────────
            tok_emb = vocab.get(next_token)
            if tok_emb is None:
                tok_emb = np.zeros(EMBED_DIM, dtype=np.float32)
            else:
                tok_emb = tok_emb.astype(np.float32)

            tok_padded             = np.zeros(len(ctx), dtype=np.float32)
            tok_padded[:EMBED_DIM] = tok_emb
            tok_n = tok_padded / (np.linalg.norm(tok_padded) + 1e-10)

            # Score-margin gating for context update:
            # If the top score and 2nd score are very close (uncertain choice),
            # blend tok_n with the 2nd-best token embedding to soften the
            # context commitment.  Prevents a low-confidence token from
            # hard-overwriting the context direction.
            _post_valid = scores > -1e9
            if _post_valid.sum() >= 2:
                _top2_idx    = np.argpartition(
                    np.where(_post_valid, scores, -1e18), -2)[-2:]
                _top2_scores = scores[_top2_idx]
                _top2_sorted = np.sort(_top2_scores)[::-1]
                _score_gap   = float(_top2_sorted[0] - _top2_sorted[1])
                if _score_gap < 0.05:
                    # Uncertain — soft-blend with 2nd-best embedding
                    _second_i   = _top2_idx[0] if _top2_idx[1] == next_idx else _top2_idx[1]
                    _second_tok = words[_second_i]
                    _sec_emb    = vocab.get(_second_tok,
                                            np.zeros(EMBED_DIM, dtype=np.float32))
                    _sec_pad    = np.zeros(len(ctx), dtype=np.float32)
                    _sec_pad[:EMBED_DIM] = _sec_emb.astype(np.float32)
                    _sec_n   = _sec_pad / (np.linalg.norm(_sec_pad) + 1e-10)
                    # Weight: 0.70..1.00 depending on gap (0.00 → 0.70, 0.05 → 1.00)
                    _w_top1  = 0.70 + 0.60 * min(_score_gap / 0.05, 1.0)
                    _w_top1  = float(np.clip(_w_top1, 0.70, 1.00))
                    tok_n    = _w_top1 * tok_n + (1.0 - _w_top1) * _sec_n
                    tok_n    = tok_n / (np.linalg.norm(tok_n) + 1e-10)

            # Step-adaptive EMA context roll + momentum correction
            _delta       = tok_n - ctx
            _ctx_vel     = _CTX_MOM_B1 * _ctx_vel + (1.0 - _CTX_MOM_B1) * _delta

            # Oscillation dampening: if this delta reverses the previous one,
            # reduce the velocity contribution to prevent the context from
            # ping-ponging between two semantic poles.
            _osc_cos = float(np.dot(_delta, _ctx_prev_delta) /
                             (np.linalg.norm(_delta) * np.linalg.norm(_ctx_prev_delta) + 1e-9))
            if _osc_cos < _CTX_OSC_THR:
                _ctx_vel *= (1.0 - _CTX_OSC_DAMP)

            _ctx_prev_delta = _delta.copy()
            ctx          = _ca * ctx + _1m_ca * tok_n + _CTX_MOM_S * _ctx_vel
            ctx          = ctx / (np.linalg.norm(ctx) + 1e-10)

            # Context velocity magnitude EMA: tracks how fast the context is
            # moving.  High magnitude = rapid semantic shift; near-zero = stable.
            _vel_mag = float(np.linalg.norm(_ctx_vel))
            _ctx_vel_mag_ema = ((1.0 - _VEL_MAG_ALPHA) * _ctx_vel_mag_ema
                                + _VEL_MAG_ALPHA * _vel_mag)
            _velocity_steps.append(round(_ctx_vel_mag_ema, 5))  # log for velplot
            _sfb_strength_steps.append(round(float(_sfb.strength), 5))  # SFB log

            # Adaptive SFB decay from coh3: high coherence → slow decay (stay on
            # topic); low coherence → faster decay (allow semantic drift).
            if len(prev_vecs) >= 3:
                _sfb_coh_now = (float(prev_vecs[-3] @ prev_vecs[-2]) +
                                float(prev_vecs[-2] @ prev_vecs[-1])) / 2.0
                if _sfb_coh_now >= 0.45:
                    _sfb._decay = max(0.96, _sfb._decay)   # slow decay when coherent
                elif _sfb_coh_now < 0.20:
                    _sfb._decay = min(0.90, _sfb._decay)   # faster when drifting

            # Semantic field momentum: update slow topic centroid and pull ctx
            # back if it has drifted more than _SEM_CENT_THR from the centroid.
            _sem_centroid = ((1.0 - _SEM_CENT_ALPHA) * _sem_centroid
                             + _SEM_CENT_ALPHA * ctx)
            _sem_centroid /= (np.linalg.norm(_sem_centroid) + 1e-10)
            _sem_dist = float(1.0 - np.dot(ctx[:len(_sem_centroid)],
                                           _sem_centroid[:len(ctx)]))
            if _sem_dist > _SEM_CENT_THR:
                # Pull ctx toward centroid proportional to how far it has drifted
                _pull     = _SEM_CENT_STR * (_sem_dist - _SEM_CENT_THR)
                ctx       = ((1.0 - _pull) * ctx
                             + _pull * _sem_centroid[:len(ctx)])
                ctx       = ctx / (np.linalg.norm(ctx) + 1e-10)

            # Push to attention-weighted history
            ctx_hist.push(tok_padded)

            # Track prev_vecs for degeneration penalty
            emb_n = tok_emb / (np.linalg.norm(tok_emb) + 1e-9)
            prev_vecs.append(emb_n)

            # Dynamic topic-tracking: every 3 steps blend SemanticFieldBias
            # anchor toward the current context vector (224-dim prefix of ctx).
            # blend=0.12 — slow enough to follow the topic without losing the
            # prompt signal, fast enough to stay relevant to new developments.
            if _step % 3 == 2 and _step > 0:
                _sfb.update(ctx[:EMBED_DIM], blend=0.12)

            # Coherence-adaptive history decay:
            # If recent 3-step coherence is strong (≥ 0.20), allow history
            # to reach back further by using a slow decay (0.92).
            # If coherence is weak (< 0.10), shorten effective window with
            # faster decay (0.70) so stale incoherent tokens have less pull.
            if len(prev_vecs) >= 3:
                _r3a = float(prev_vecs[-3] @ prev_vecs[-2])
                _r3b = float(prev_vecs[-2] @ prev_vecs[-1])
                _rcoh = 0.5 * _r3a + 0.5 * _r3b
                if _rcoh >= 0.20:
                    ctx_hist.decay = 0.92    # coherent: long effective window
                elif _rcoh < 0.10:
                    ctx_hist.decay = 0.70    # incoherent: short effective window
                else:
                    ctx_hist.decay = 0.88    # default

            # Exclusion EMA
            exclusion = 0.80 * exclusion + 0.20 * tok_n
            ex_n = np.linalg.norm(exclusion)
            if ex_n > 1e-10:
                exclusion /= ex_n

        # ── Post-generation quality metrics ───────────────────────────────
        coherence = 0.0
        coh3      = 0.0   # 3-token window coherence
        coh6      = 0.0   # 6-token window coherence
        diversity = 0.0
        if len(prev_vecs) >= 2:
            # Coherence: average cosine similarity between consecutive token embeddings
            # Higher = more semantically consistent across consecutive words
            cosines = [float(prev_vecs[i] @ prev_vecs[i + 1])
                       for i in range(len(prev_vecs) - 1)]
            coherence = float(np.mean(cosines))
            # Multi-scale window coherence (last N pairs)
            coh3 = float(np.mean(cosines[-3:])) if len(cosines) >= 3 else coherence
            coh6 = float(np.mean(cosines[-6:])) if len(cosines) >= 6 else coherence
        if output_tokens:
            diversity = len(set(output_tokens)) / len(output_tokens)

        # Shannon diversity index (normalised to [0,1]):
        # More informative than type-token ratio because it weights by frequency.
        # H = -sum(p_i × log(p_i)),  H_norm = H / log(N_tokens)
        shannon_diversity = 0.0
        if len(output_tokens) > 1:
            from collections import Counter as _Ctr
            _ctok  = _Ctr(output_tokens)
            _probs = np.array(list(_ctok.values()), dtype=np.float32)
            _probs /= _probs.sum()
            _H_tok  = float(-np.dot(_probs, np.log(_probs + 1e-30)))
            shannon_diversity = float(_H_tok / np.log(max(len(output_tokens), 2)))

        # Bigram + trigram fluency: fraction of consecutive n-grams attested in vocab.
        # Trigrams are rare in a word-level vocab so we score them separately and
        # blend (80 % bigram, 20 % trigram) to avoid dominating by zero-matches.
        fluency = 0.0
        if len(output_tokens) > 1:
            _pv_f  = getattr(self.base_mapper, "_base_vocab", {})
            _f_bi_hits = sum(
                1 for a, b in zip(output_tokens, output_tokens[1:])
                if (a + " " + b) in _pv_f
            )
            _f_bi = _f_bi_hits / (len(output_tokens) - 1)

            _f_tri = 0.0
            if len(output_tokens) > 2:
                _f_tri_hits = sum(
                    1 for a, b, c in zip(output_tokens,
                                         output_tokens[1:],
                                         output_tokens[2:])
                    if (a + " " + b + " " + c) in _pv_f
                )
                _f_tri = _f_tri_hits / (len(output_tokens) - 2)

            fluency = 0.80 * _f_bi + 0.20 * _f_tri

        # Confidence variance (std dev): measures consistency of model certainty.
        # Low variance + high mean = reliably confident; high variance = patchy.
        conf_variance = 0.0
        if len(output_confs) > 1:
            conf_variance = float(np.std(output_confs))

        # Confidence trend: linear regression slope over all step confidences.
        # Positive = gaining confidence across the generation (good).
        # Negative = losing confidence (model unsure at the end).
        conf_trend = 0.0
        if len(output_confs) >= 3:
            _ct_x = np.arange(len(output_confs), dtype=np.float32)
            _ct_y = np.array(output_confs, dtype=np.float32)
            conf_trend = float(np.polyfit(_ct_x, _ct_y, 1)[0])

        # Coherence trend: linear slope of coh3_steps.
        # Positive = coherence growing through the generation (good).
        # Negative = losing coherence (semantic drift).
        coh_trend = 0.0
        if len(_coh3_steps) >= 3:
            _ctr_x = np.arange(len(_coh3_steps), dtype=np.float32)
            _ctr_y = np.array(_coh3_steps, dtype=np.float32)
            coh_trend = float(np.polyfit(_ctr_x, _ctr_y, 1)[0])

        # Confidence declining flag: True when last ≥4 confidences each
        # fall below their predecessor (strict monotone decline).
        conf_declining = False
        if len(output_confs) >= 4:
            _cd_last = output_confs[-4:]
            conf_declining = all(
                _cd_last[i] > _cd_last[i + 1] for i in range(3)
            )

        # Pseudo-perplexity: 2 ^ (avg_entropy_norm × log2(vocab_size)).
        # Scaled to [1, vocab_size] — lower is better.
        # Uses the per-step EMA entropy tracked during generation.
        _ppl_bits = _ema_entropy * np.log2(max(len(words), 2))
        pseudo_ppl = float(np.clip(2.0 ** _ppl_bits, 1.0, float(len(words))))

        # Token embedding spread: mean pairwise cosine distance among generated
        # token embeddings (1 − cosine similarity).  Higher = more semantically
        # diverse set of chosen tokens.  Uses prev_vecs[:, :EMBED_DIM].
        token_embed_spread = 0.0
        if len(prev_vecs) >= 2:
            _spread_vecs = np.stack(prev_vecs[-16:]).astype(np.float32)  # (N, 256)
            _spread_vecs_e = _spread_vecs[:, :EMBED_DIM]
            _sv_norms = np.linalg.norm(_spread_vecs_e, axis=1, keepdims=True) + 1e-9
            _sv_n = _spread_vecs_e / _sv_norms
            _sv_sims = _sv_n @ _sv_n.T   # (N, N)
            N_sv = _sv_sims.shape[0]
            _sv_upper = _sv_sims[np.triu_indices(N_sv, k=1)]
            token_embed_spread = float(1.0 - float(np.mean(_sv_upper)))

        return {
            "text":        " ".join(output_tokens),
            "tokens":      output_tokens,
            "confidences": output_confs,
            "stop_reason": stop_reason,
            "coherence":   round(coherence,  4),
            "diversity":    round(diversity,        4),
            "shannon_div":  round(shannon_diversity, 4),
            "fluency":      round(fluency,           4),  # 0..1 bigram/trigram blend
            "avg_entropy":  round(_ema_entropy,      4),
            "coh3":         round(coh3,              4),
            "coh6":         round(coh6,              4),
            "vocab_prec":     round((_vocab_prec_hits / max(_vocab_prec_total, 1)), 4),
            "vocab_prec_ema": round(_vprec_ema, 4),  # EMA-smoothed vocab precision
            "conf_variance": round(conf_variance,    4),  # std dev of step confidences
            "conf_trend":   round(conf_trend,        5),  # slope of conf over steps
            "pseudo_ppl":   round(pseudo_ppl,        2),  # 2^(H×log2V) — lower=better
            "avg_score_gap": round(_ema_score_gap,   4),  # EMA top1−top2 score gap
            "ctx_vel_mag":  round(_ctx_vel_mag_ema,  4),  # EMA ctx velocity magnitude
            "peep_hit_rate": round(
                _peep_top5_hits / max(_peep_top5_steps, 1), 4
            ),   # fraction of steps Peep top-5 included the chosen token
            "peak_conf_step": _peak_conf_step,  # step index of highest confidence
            "sg_step_gaps":  _sg_step_gaps,     # per-step score gaps (for scorehist)
            "sg_slope":      round(float(np.polyfit(
                np.arange(len(_sg_step_gaps), dtype=np.float32),
                np.array(_sg_step_gaps, dtype=np.float32), 1
            )[0]) if len(_sg_step_gaps) >= 2 else 0.0, 5),  # decisiveness trend
            "coh6_steps":    _coh6_steps,       # per-step coh6 values (step≥6)
            "token_embed_spread": round(token_embed_spread, 4),  # mean pairwise embedding distance
            "rhythm_score": round(
                1.0 - (_rhythm_alts / max(_rhythm_steps, 1)), 4
            ),   # 1=smooth, 0=fully alternating confidence
            "_last_dot_top1":  _last_dot_top1,   # for dotscores CLI
            "_top3_log":       _top3_log,         # per-step top-3 candidates
            "top1_margins":    _top1_margins,     # per-step top1−top2 score gap
            "margin_ema_final": round(_margin_ema, 5),  # EMA of margin at end of gen
            "coh3_steps":        _coh3_steps,         # per-step coh3 values
            "entropy_steps":     _entropy_steps,     # per-step H_norm values
            "velocity_steps":     _velocity_steps,      # per-step ctx velocity EMA
            "sfb_strength_steps": _sfb_strength_steps,  # per-step SFB strength
            "percentile_steps":   _percentile_steps,    # per-step score percentile
            "temp_steps":         _temp_steps,          # per-step temperature value
            "vocab_prec_ema":     round(_vprec_ema, 4), # vocab-precision EMA at end
            "score_var_steps":    _score_var_steps,     # per-step score variance
            "topk_steps":         _topk_steps,          # per-step adaptive TopK k
            "score_hist":      _score_hist.tolist(),  # 8-bucket confidence histogram
            "conf_ema_final":  round(_conf_ema, 4),  # EMA-smoothed confidence final step
            "anchor_strength": round(_anc_strength, 4),  # prompt-adaptive anchor strength
            "prompt_type":     _ptype,            # detected prompt intent
            "coh_direction":   round(_coh_dir, 4),  # coherence slope over last 4 steps
            "conf_declining":  conf_declining,       # True if last 4 confs all falling
            "coh_trend":       round(coh_trend, 5),  # linear slope of coh3 across steps
            "vocab_size":  len(words),
        }

    def causal_generate_nbest(self,
                               prompt:    str,
                               n:         int   = 3,
                               max_tokens: int  = 20,
                               **kwargs) -> dict:
        """N-best reranking generation.

        Runs ``causal_generate`` N times with different random states and
        returns the single best result by a composite reranking score:

            score = avg_confidence × (0.80 + 0.20 × coherence)
                    × length_bonus × (1.0 + diversity × 0.10)

        This is IECNN's lightweight equivalent of beam search + LM scoring:
        no backtracking, just forward sampling followed by best-of-N selection.

        Parameters
        ----------
        n          : number of candidates to generate (default 3)
        max_tokens : passed through to causal_generate
        **kwargs   : any causal_generate parameter
        """
        candidates: list = []
        # Diversity forcing: after each run, block its first token so the
        # next run is forced to explore a different beginning.
        # Builds up an exclusion set run-by-run; limit to first token only
        # so we don't over-constrain later tokens.
        _run_excl_first: set = set()

        for _run in range(n):
            # Pass the exclusion-first set as an extra hint via variation seed
            # We implement this by temporarily monkey-patching the random seed
            # with a per-run offset so each sample path differs.
            import random as _rnd
            _rnd.seed(_run * 37 + 1)
            np.random.seed(_run * 37 + 1)
            result = self.causal_generate(prompt, max_tokens=max_tokens, **kwargs)
            np.random.seed(None)   # restore non-determinism for subsequent calls
            _rnd.seed(None)

            if not result["tokens"]:
                continue

            confs     = result["confidences"]
            avg_conf  = float(np.mean(confs)) if confs else 0.0
            coherence = result.get("coherence", 0.0)
            diversity = result.get("diversity", 0.0)
            ntok      = len(result["tokens"])

            # Penalise very short outputs (< 4 tokens)
            length_bonus = float(np.tanh(ntok / 6.0))

            # Diversity penalty: penalise outputs whose first token is a
            # near-duplicate of a first token already seen in this batch.
            first_tok = result["tokens"][0]
            _excl_pen = 1.0
            if first_tok in _run_excl_first:
                _excl_pen = 0.85   # 15% penalty for first-token repetition
            _run_excl_first.add(first_tok)

            # Bigram fluency bonus: fraction of consecutive token pairs that
            # are attested bigrams in the training vocab (proxy for linguistic
            # naturalness beyond cosine coherence).
            _fluency = 0.0
            if ntok > 1:
                _pv   = getattr(self.base_mapper, "_base_vocab", {})
                _hits = sum(
                    1 for a, b in zip(result["tokens"], result["tokens"][1:])
                    if (a + " " + b) in _pv
                )
                _fluency = _hits / (ntok - 1)

            # Enriched scoring: incorporate recency-weighted confidence,
            # vocab precision, local coherence (C3), pseudo-perplexity, and
            # confidence variance for a richer composite signal.
            _rw_arr    = np.arange(1, len(confs) + 1, dtype=float)
            _rw_conf   = float(np.average(confs, weights=_rw_arr)) if confs else avg_conf
            _vprec     = result.get("vocab_prec",   0.5)
            _c3        = max(result.get("coh3",     coherence), 0.0)
            _shannon   = result.get("shannon_div",  0.0)
            _conf_var  = result.get("conf_variance", 0.1)   # lower = more consistent
            _ppl       = result.get("pseudo_ppl",   1000.0) # lower = better
            _ppl_norm  = float(np.clip(1.0 / (1.0 + np.log1p(_ppl - 1.0 + 1e-6)), 0.5, 1.0))

            # Peak-step bonus: generations that peaked later (coherent run
            # continues after peak) get a small reward; early-peaking ones
            # (peaked at step 0 = noisy start) are slightly penalised.
            _pcs      = result.get("peak_conf_step", 0)
            _pcs_norm = float(np.tanh(_pcs / max(ntok, 1)))  # 0→1 as pcs grows
            _pcs_fac  = 1.0 + 0.05 * _pcs_norm

            score = (avg_conf
                     * (0.75 + 0.15 * max(coherence, 0.0) + 0.10 * _c3)
                     * length_bonus
                     * (1.0 + diversity    * 0.08)
                     * (1.0 + _shannon     * 0.06)
                     * (1.0 + _fluency     * 0.10)
                     * (1.0 + (_rw_conf - avg_conf) * 0.20)
                     * (1.0 + _vprec       * 0.04)
                     * (1.0 - _conf_var    * 0.10)   # penalise patchy confidence
                     * _ppl_norm                      # reward lower pseudo-perplexity
                     * _excl_pen
                     * _pcs_fac                       # reward late-peaking paths
                     * (1.0 + result.get("rhythm_score",   0.5) * 0.04)    # smooth rhythm bonus
                     * (1.0 + max(result.get("coh_direction", 0.0), 0.0) * 0.06))  # coh_dir bonus

            candidates.append((score, result))

        if not candidates:
            return {"text": "", "tokens": [], "confidences": [],
                    "stop_reason": "no_candidates", "n_best": n}

        # Stash quality history on the model object (last 8 scores)
        if not hasattr(self, "_quality_history"):
            from collections import deque as _dq2
            self._quality_history = _dq2(maxlen=8)
        self._quality_history.append({
            "score": round(candidates[0][0], 4),
            "text":  candidates[0][1].get("text","")[:30],
        })

        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]
        best["n_best_scores"] = [round(s, 4) for s, _ in candidates]
        best["n_best"] = n
        return best

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
