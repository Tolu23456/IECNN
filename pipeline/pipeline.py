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
        # Base factor 1.06; adapted per-step from perc_trend at step 8+.
        _expdecay_factor = 1.06
        _expdecay = ExponentialDecayLength(
            start_idx=max(min_tokens, 8), factor=_expdecay_factor, stop_ids=stop_ids)

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
        # _TREND_SLOPE is also adaptive: quad_flat_exit_velocity modifies it
        # (see quad_flat_exit_velocity adaptive hook in the generation loop)

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
            "question":     1.30,   # 30 % stronger — questions need tight focus
            "statement":    1.00,   # baseline
            "continuation": 0.85,   # allow more drift
            "imperative":   1.15,   # commands stay focused
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
        _conf_ema_prev        = 0.0  # previous-step conf_ema (decline tracking)
        _conf_ema_decline_cnt = 0    # consecutive steps where conf_ema fell

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
        _rhythm_rate_steps: list = []   # per-step running rhythm rate

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

        # Per-step confidence EMA log: smoothed confidence at every step.
        # Enables confrise CLI (rising-confidence step detector).
        _conf_ema_steps: list = []

        # Per-step vocab precision EMA log: enables vocabjump CLI.
        _vprec_ema_steps: list = []

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
            # quad_coh3_entry_mean-adaptive ContextAnchor strength:
            # when the mean coh3 at ideal-quadrant entry is high (≥ mean+0.05),
            # the model glides into ideal from a strong coherence base — ease
            # anchor (−0.02) to let it continue that natural trajectory; when
            # entry coh3 is very low (≤ mean−0.05), the model forces its way
            # into ideal without coherence support — strengthen anchor (+0.02)
            # to guard against the fragile ideal episodes collapsing.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qce_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qce_c3m = sum(_coh3_steps[:_qce_n]) / _qce_n
                _qce_vm  = sum(_velocity_steps[:_qce_n]) / _qce_n
                _qce_entries = [
                    _coh3_steps[i]
                    for i in range(1, _qce_n)
                    if _coh3_steps[i] > _qce_c3m and _velocity_steps[i] < _qce_vm
                    and not (_coh3_steps[i-1] > _qce_c3m and _velocity_steps[i-1] < _qce_vm)
                ]
                if _qce_entries:
                    _qce_mean = sum(_qce_entries) / len(_qce_entries)
                    if _qce_mean >= _qce_c3m + 0.05:
                        anchor.strength = float(np.clip(anchor.strength - 0.02, 0.05, 0.35))
                    elif _qce_mean <= _qce_c3m - 0.05:
                        anchor.strength = float(np.clip(anchor.strength + 0.02, 0.05, 0.35))
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
            # coh3_vprec_corr-adaptive dot-agreement bonus cap:
            # when coh3 and vocab precision are rising together (+ corr > 0.40),
            # boost the cap to 0.068 — model is coherent + precise, strengthen
            # consensus signal; when anti-correlated (< -0.40, coherence and
            # vocab precision diverge), ease cap to 0.054 to avoid over-rewarding
            # a noisy consensus signal.
            _da_cap = 0.06
            if len(_coh3_steps) >= 3 and len(_vprec_ema_steps) >= 3:
                _da_cvn = min(len(_coh3_steps), len(_vprec_ema_steps))
                _da_cvr = float(np.corrcoef(
                    np.array(_coh3_steps[:_da_cvn],     dtype=np.float32),
                    np.array(_vprec_ema_steps[:_da_cvn], dtype=np.float32)
                )[0, 1])
                if _da_cvr > 0.40:
                    _da_cap = 0.068
                elif _da_cvr < -0.40:
                    _da_cap = 0.054
            _da_bonus  = np.where(_da_counts >= 2,
                                  np.minimum(
                                      (_da_counts - 1).astype(np.float32) * 0.012,
                                      _da_cap),  # cap per-token agreement bonus
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

            _hb_win    = max(4, min(HARD_BLOCK, 4 + _step // 4))
            blocked    = set(output_tokens[-_hb_win:])
            block_mask = (np.array([w in blocked for w in words], dtype=bool)
                          if blocked else None)

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
            # exploring_frac-adaptive DotVariancePenalty strength:
            # when the model is mostly in the exploring quadrant (coh3↑ vel↑),
            # it is actively diverging — strengthen the variance penalty (+0.02)
            # to discourage further disagreement amplification; when exploring
            # is rare (≤0.10) the model stays grounded, ease back (−0.01).
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _dvp_n   = min(len(_coh3_steps), len(_velocity_steps))
                _dvp_c3m = sum(_coh3_steps[:_dvp_n])    / _dvp_n
                _dvp_vm  = sum(_velocity_steps[:_dvp_n]) / _dvp_n
                _dvp_ef  = sum(1 for i in range(_dvp_n)
                               if _coh3_steps[i] > _dvp_c3m and _velocity_steps[i] >= _dvp_vm
                               ) / max(_dvp_n, 1)
                if _dvp_ef >= 0.35:
                    _dot_var_pen.strength = min(0.16, _dot_var_pen.strength + 0.02)
                elif _dvp_ef <= 0.10:
                    _dot_var_pen.strength = max(0.04, _dot_var_pen.strength - 0.01)
            scores = _dot_var_pen(scores,
                                  raw_preds=raw_preds,
                                  word_vecs_n=word_vecs_n)

            # ── 6. Score processor pipeline ───────────────────────────────

            # 6-0. Local semantic restriction — keep top-200 context-close words
            # vel_trend-adaptive top_k: when velocity is trending upward fast
            # (context drifting), shrink LocalSem window to 150 to avoid
            # anchoring to stale semantics; when velocity is falling (stable
            # context), open up to 220 to allow richer exploration.
            if len(_velocity_steps) >= 2 and (("_steps_remaining" not in dir()) or _steps_remaining > 2):
                _lsem_vel_slope = float(
                    np.polyfit(range(len(_velocity_steps)), _velocity_steps, 1)[0]
                )
                if _lsem_vel_slope > 0.00008:
                    _local_sem.top_k = max(150, _local_sem.top_k - 10)
                elif _lsem_vel_slope < -0.00008:
                    _local_sem.top_k = min(220, _local_sem.top_k + 10)
            # vprec_entropy_corr-adaptive LocalSem top_k: when vocab precision
            # and entropy are negatively correlated (precise = low entropy =
            # healthy focus), tighten LocalSem (−8) to reinforce local context;
            # when positively correlated (precise but high entropy, unusual),
            # widen (+8) to expand the candidate pool.
            if len(_vprec_ema_steps) >= 4 and len(_entropy_steps) >= 4:
                _lsem_ven = min(len(_vprec_ema_steps), len(_entropy_steps))
                _lsem_ver = float(np.corrcoef(
                    np.array(_vprec_ema_steps[:_lsem_ven], dtype=np.float32),
                    np.array(_entropy_steps[:_lsem_ven],   dtype=np.float32)
                )[0, 1])
                if _lsem_ver < -0.40:
                    _local_sem.top_k = max(130, _local_sem.top_k - 8)
                elif _lsem_ver > 0.40:
                    _local_sem.top_k = min(240, _local_sem.top_k + 8)
            # quad_balance_score-adaptive LocalSemanticFilter top_k:
            # balance = (ideal_steps − drifting_steps) / n_steps;
            # when strongly quality-biased (≥0.30), widen top_k (+10) so the
            # model can freely pick from a larger focused set while it's on a
            # good run; when strongly drift-biased (≤−0.30), tighten top_k
            # (−12) to constrain the model toward the nearest semantic region.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qbs_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qbs_c3m = sum(_coh3_steps[:_qbs_n])    / _qbs_n
                _qbs_vm  = sum(_velocity_steps[:_qbs_n]) / _qbs_n
                _qbs_ideal   = sum(1 for i in range(_qbs_n)
                                   if _coh3_steps[i] > _qbs_c3m and _velocity_steps[i] < _qbs_vm)
                _qbs_drifting = sum(1 for i in range(_qbs_n)
                                    if _coh3_steps[i] <= _qbs_c3m and _velocity_steps[i] >= _qbs_vm)
                _qbs_score = (_qbs_ideal - _qbs_drifting) / max(_qbs_n, 1)
                if _qbs_score >= 0.30:
                    _local_sem.top_k = min(250, _local_sem.top_k + 10)
                elif _qbs_score <= -0.30:
                    _local_sem.top_k = max(120, _local_sem.top_k - 12)
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
                    # margin_conf_corr-adaptive centroid lift: when margins and
                    # confidence are positively correlated (decisive + confident
                    # = healthy), ease lift (−0.003) — no rescue needed; when
                    # negatively correlated (decisive but unconfident), add
                    # extra lift (+0.004) to pull the pool up harder.
                    if len(_top1_margins) >= 3 and len(output_confs) >= 3:
                        _sc_mcn = min(len(_top1_margins), len(output_confs))
                        _sc_mcr = float(np.corrcoef(
                            np.array(_top1_margins[:_sc_mcn], dtype=np.float32),
                            np.array(output_confs[:_sc_mcn],  dtype=np.float32)
                        )[0, 1])
                        if _sc_mcr > 0.40:
                            _sc_lift_now = max(0.008, _sc_lift_now - 0.003)
                        elif _sc_mcr < -0.40:
                            _sc_lift_now = min(0.08, _sc_lift_now + 0.004)
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

            # sfb_acc-adaptive SFB strength: when SFB slope is strongly positive
            # (phrase variety already improving) ease off the boost to avoid
            # over-constraining; when strongly negative (variety falling) add
            # extra nudge upward.
            _sfb_steps = globals().get("_sfb_steps", None)
            if _sfb_steps is not None and len(_sfb_steps) >= 4:
                _sfb_slope_now = float(
                    np.polyfit(range(len(_sfb_steps)), _sfb_steps, 1)[0]
                )
                if _sfb_slope_now > 0.002:
                    _sfb.strength = max(_sfb.strength - 0.015, semantic_bias * 0.60)
                elif _sfb_slope_now < -0.002:
                    _sfb.strength = min(_sfb.strength + 0.015, semantic_bias + 0.08)

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
            # conf_vel_corr-adaptive strength: when confidence and velocity are
            # negatively correlated (high movement = low confidence — healthy
            # exploration), ease drift penalty (−0.012) so topic-moving tokens
            # aren't overly suppressed; when positively correlated (movement and
            # confidence both rise, unusual), strengthen it (+0.010) to resist
            # drifting away from prompt when the model is overconfident.
            if len(output_confs) >= 4 and len(_velocity_steps) >= 4:
                _pdr_n = min(len(output_confs), len(_velocity_steps))
                _pdr_r = float(np.corrcoef(
                    np.array(output_confs[:_pdr_n],    dtype=np.float32),
                    np.array(_velocity_steps[:_pdr_n], dtype=np.float32)
                )[0, 1])
                if _pdr_r < -0.40:
                    _pdrift.strength = max(0.024, _pdrift.strength - 0.012)
                elif _pdr_r > 0.40:
                    _pdrift.strength = min(0.090, _pdrift.strength + 0.010)
            # coh3_margin_joint-adaptive PromptDrift: when coh3 and margin are
            # both above average at a high fraction of steps (coherent+decisive),
            # ease drift penalty (−0.004) — model is navigating well and can be
            # trusted to drift intentionally; when joint fraction is low (<0.20),
            # tighten (+0.004) to anchor it back toward the prompt.
            if len(_coh3_steps) >= 4 and len(_top1_margins) >= 4:
                _pdr_cmj_n  = min(len(_coh3_steps), len(_top1_margins))
                _pdr_cmj_c3m = sum(_coh3_steps[:_pdr_cmj_n])    / _pdr_cmj_n
                _pdr_cmj_mm  = sum(_top1_margins[:_pdr_cmj_n]) / _pdr_cmj_n
                _pdr_cmj_frac = sum(
                    1 for _xi in range(_pdr_cmj_n)
                    if (_coh3_steps[_xi] > _pdr_cmj_c3m
                        and _top1_margins[_xi] > _pdr_cmj_mm)
                ) / max(_pdr_cmj_n, 1)
                if _pdr_cmj_frac > 0.50:
                    _pdrift.strength = max(0.020, _pdrift.strength - 0.004)
                elif _pdr_cmj_frac < 0.20:
                    _pdrift.strength = min(0.095, _pdrift.strength + 0.004)
            # phase_transition-adaptive PromptDrift: scan the quadrant sequence
            # for transitions; if ≥2 drifting→ideal recoveries occurred, ease
            # penalty (−0.004) — the model self-corrects reliably; if ≥2
            # ideal→drifting falls occurred, tighten (+0.004) to actively
            # resist the repeated quality drops.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _pt_n   = min(len(_coh3_steps), len(_velocity_steps))
                _pt_c3m = sum(_coh3_steps[:_pt_n])    / _pt_n
                _pt_vm  = sum(_velocity_steps[:_pt_n]) / _pt_n
                _pt_prev_q: str = ""; _pt_d2i = 0; _pt_i2d = 0
                for _xi in range(_pt_n):
                    _pt_q = ("ideal"
                              if _coh3_steps[_xi] > _pt_c3m and _velocity_steps[_xi] < _pt_vm
                              else ("exploring"
                                    if _coh3_steps[_xi] > _pt_c3m
                                    else ("drifting"
                                          if _velocity_steps[_xi] >= _pt_vm else "flat")))
                    if _pt_prev_q:
                        if _pt_prev_q == "drifting" and _pt_q == "ideal":
                            _pt_d2i += 1
                        elif _pt_prev_q == "ideal" and _pt_q == "drifting":
                            _pt_i2d += 1
                    _pt_prev_q = _pt_q
                if _pt_d2i >= 2:
                    _pdrift.strength = max(0.018, _pdrift.strength - 0.004)
                elif _pt_i2d >= 2:
                    _pdrift.strength = min(0.098, _pdrift.strength + 0.004)
            scores = _pdrift(scores)

            # 6-2. Vocabulary frequency prior (de-rank ultra-rare tokens)
            # entropy_trend-adaptive strength: when entropy is rising (generation
            # becoming more random), strengthen freq_prior to de-rank ultra-rares
            # more aggressively and pull the distribution back toward common words;
            # when entropy is falling (model focusing), ease off so rare but
            # context-apt tokens can still surface.
            if len(_entropy_steps) >= 4:
                _fp_ent_slope = float(
                    np.polyfit(range(len(_entropy_steps)), _entropy_steps, 1)[0]
                )
                if _fp_ent_slope > 0.002:
                    _freq_prior.strength = min(0.08, _freq_prior.strength + 0.008)
                elif _fp_ent_slope < -0.002:
                    _freq_prior.strength = max(0.02, _freq_prior.strength - 0.006)
            # margin_var-adaptive FreqPrior strength: when top1_margin variance
            # is high (erratic decisiveness — occasional spikes), increase
            # freq_prior strength (+0.006) to anchor toward common words and
            # smooth out the spikes; when variance is low (consistently
            # decisive), ease off (−0.005) to allow rarer context-apt tokens.
            if len(_top1_margins) >= 5:
                _fp_mv = float(np.var(
                    np.array(_top1_margins[-min(len(_top1_margins), 8):], dtype=np.float32)
                ))
                if _fp_mv > 0.008:
                    _freq_prior.strength = min(0.10, _freq_prior.strength + 0.006)
                elif _fp_mv < 0.001:
                    _freq_prior.strength = max(0.018, _freq_prior.strength - 0.005)
            # streak-adaptive FreqPrior: track the current coh3×vel quadrant streak;
            # in a run of ≥3 "ideal" steps (coh3↑ vel↓), ease strength (−0.003) to
            # let rare precise words surface; in a run of ≥3 "drifting" steps
            # (coh3↓ vel↑), tighten (+0.003) to anchor toward stable common words.
            if len(_coh3_steps) >= 3 and len(_velocity_steps) >= 3:
                _strk_n   = min(len(_coh3_steps), len(_velocity_steps))
                _strk_c3m = sum(_coh3_steps[:_strk_n])    / _strk_n
                _strk_vm  = sum(_velocity_steps[:_strk_n]) / _strk_n
                _strk_cur_q: str = ""; _strk_cur_len = 0
                for _xi in range(_strk_n):
                    _strk_q = ("ideal"
                                if _coh3_steps[_xi] > _strk_c3m and _velocity_steps[_xi] < _strk_vm
                                else ("exploring"
                                      if _coh3_steps[_xi] > _strk_c3m
                                      else ("drifting"
                                            if _velocity_steps[_xi] >= _strk_vm else "flat")))
                    if _strk_q == _strk_cur_q:
                        _strk_cur_len += 1
                    else:
                        _strk_cur_q   = _strk_q
                        _strk_cur_len = 1
                if _strk_cur_q == "ideal" and _strk_cur_len >= 3:
                    _freq_prior.strength = max(0.015, _freq_prior.strength - 0.003)
                elif _strk_cur_q == "drifting" and _strk_cur_len >= 3:
                    _freq_prior.strength = min(0.11, _freq_prior.strength + 0.003)
            # quad_oscillation_score-adaptive VocabFrequencyPrior strength:
            # oscillation = fraction of transitions that are A→B→A (ping-pong);
            # when the model oscillates heavily (≥ 0.55), it is flip-flopping
            # between two states — ease the frequency prior (−0.004) to stop
            # over-correcting; when oscillation is very low (≤ 0.15), the model
            # is exploring freely — tighten (+0.003) to keep word choice grounded.
            if len(_coh3_steps) >= 6 and len(_velocity_steps) >= 6:
                _qos_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qos_c3m = sum(_coh3_steps[:_qos_n])    / _qos_n
                _qos_vm  = sum(_velocity_steps[:_qos_n]) / _qos_n
                _qos_seq = [
                    ("ideal"      if _coh3_steps[i] > _qos_c3m and _velocity_steps[i] < _qos_vm
                     else "exploring" if _coh3_steps[i] > _qos_c3m
                     else "drifting"  if _velocity_steps[i] >= _qos_vm
                     else "flat")
                    for i in range(_qos_n)
                ]
                _qos_trans_total = sum(1 for j in range(1, len(_qos_seq))
                                       if _qos_seq[j] != _qos_seq[j-1])
                if _qos_trans_total >= 3:
                    _qos_osc = sum(
                        1 for j in range(2, len(_qos_seq))
                        if _qos_seq[j] != _qos_seq[j-1]
                        and _qos_seq[j] == _qos_seq[j-2]
                    ) / max(_qos_trans_total - 1, 1)
                    if _qos_osc >= 0.55:
                        _freq_prior.strength = max(0.012, _freq_prior.strength - 0.004)
                    elif _qos_osc <= 0.15:
                        _freq_prior.strength = min(0.12, _freq_prior.strength + 0.003)
            scores = _freq_prior(scores)

            # 6-2a. Surprise bonus — reward context-close non-top-50 tokens
            # margin_trend-adaptive strength: growing margin (model is decisive)
            # → reduce surprise strength (it already knows what it wants);
            # shrinking margin → increase surprise to push exploration.
            if len(_top1_margins) >= 4:
                _surp_slope = float(
                    np.polyfit(range(len(_top1_margins)), _top1_margins, 1)[0]
                )
                if _surp_slope > 0.0002:
                    _surprise.strength = max(0.02, _surprise.strength - 0.005)
                elif _surp_slope < -0.0002:
                    _surprise.strength = min(0.07, _surprise.strength + 0.005)
            # coh_var-adaptive Surprise: when coherence is fluctuating heavily
            # (coh3 variance high), boost Surprise to inject variety and help
            # the model escape the incoherent zone; when variance is low
            # (stable coherence), ease off so the model can exploit its groove.
            if len(_coh3_steps) >= 4:
                _coh3_var_now = float(np.var(_coh3_steps[-min(8, len(_coh3_steps)):]))
                if _coh3_var_now > 0.018:
                    _surprise.strength = min(0.08, _surprise.strength + 0.004)
                elif _coh3_var_now < 0.006:
                    _surprise.strength = max(0.02, _surprise.strength - 0.003)
            # vel_slope_trend-adaptive Surprise: when velocity is accelerating
            # (2nd derivative > 0.00002 — model actively exploring), ease surprise
            # (−0.003) to avoid compounding exploration with extra novelty injection;
            # when decelerating (< -0.00002 — model settling), boost it (+0.003)
            # to keep injecting novelty and prevent premature settling.
            if len(_velocity_steps) >= 3:
                _vel_d1a = _velocity_steps[-1] - _velocity_steps[-2]
                _vel_d1b = _velocity_steps[-2] - _velocity_steps[-3]
                _vel_acc = _vel_d1a - _vel_d1b
                if _vel_acc > 0.00002:
                    _surprise.strength = max(0.018, _surprise.strength - 0.003)
                elif _vel_acc < -0.00002:
                    _surprise.strength = min(0.09, _surprise.strength + 0.003)
            # coh3_vel_divergence-adaptive Surprise: divergence = |mean_coh3 −
            # (1 − mean_vel)|; when low (<0.05, signals are aligned — high coh3
            # and low vel pulling together), ease surprise (−0.001) to avoid
            # over-injecting novelty into an already well-directed generation;
            # when high (>0.15, coh3 and vel are pulling in opposite directions),
            # boost (+0.002) to help the model break out of the misaligned state.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _cvd_n   = min(len(_coh3_steps), len(_velocity_steps))
                _cvd_mc3 = sum(_coh3_steps[:_cvd_n])    / _cvd_n
                _cvd_mv  = sum(_velocity_steps[:_cvd_n]) / _cvd_n
                _cvd_div = abs(_cvd_mc3 - (1.0 - _cvd_mv))
                if _cvd_div < 0.05:
                    _surprise.strength = max(0.016, _surprise.strength - 0.001)
                elif _cvd_div > 0.15:
                    _surprise.strength = min(0.10, _surprise.strength + 0.002)
            # quad_recovery_rate-adaptive SurpriseBonus strength:
            # when the model frequently recovers from drifting back to good states
            # (recovery_rate ≥ 0.60), reward that self-correction with a slightly
            # higher surprise bonus (+0.003) to encourage continued exploration of
            # recovery paths; when the model rarely recovers (≤ 0.20), suppress
            # surprise (−0.002) to avoid pulling it even further off track.
            if len(_coh3_steps) >= 6 and len(_velocity_steps) >= 6:
                _qrr_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qrr_c3m = sum(_coh3_steps[:_qrr_n])    / _qrr_n
                _qrr_vm  = sum(_velocity_steps[:_qrr_n]) / _qrr_n
                _qrr_seq = [
                    ("ideal"      if _coh3_steps[i] > _qrr_c3m and _velocity_steps[i] < _qrr_vm
                     else "exploring" if _coh3_steps[i] > _qrr_c3m
                     else "drifting"  if _velocity_steps[i] >= _qrr_vm
                     else "flat")
                    for i in range(_qrr_n)
                ]
                _qrr_denom = sum(1 for j in range(1, len(_qrr_seq))
                                 if _qrr_seq[j-1] == "drifting")
                if _qrr_denom > 0:
                    _qrr_rate = sum(1 for j in range(1, len(_qrr_seq))
                                    if _qrr_seq[j-1] == "drifting"
                                    and _qrr_seq[j] in ("ideal", "exploring")
                                    ) / _qrr_denom
                    if _qrr_rate >= 0.60:
                        _surprise.strength = min(0.11, _surprise.strength + 0.003)
                    elif _qrr_rate <= 0.20:
                        _surprise.strength = max(0.015, _surprise.strength - 0.002)
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
            # conf_ema_delta-adaptive prox_pen strength: when conf_ema has
            # improved a lot since step-0 (delta>0.05) ease off near-synonym
            # suppression — the model is finding good tokens on its own; when
            # conf_ema is declining (delta<−0.05) tighten the penalty to push
            # the model toward more semantically distant (fresh) choices.
            if len(_conf_ema_steps) >= 2:
                _px_ced = _conf_ema_steps[-1] - _conf_ema_steps[0]
                if _px_ced > 0.05:
                    _prox_pen.strength = max(0.04, _prox_pen.strength - 0.01)
                elif _px_ced < -0.05:
                    _prox_pen.strength = min(0.16, _prox_pen.strength + 0.01)
            # coh_slope-adaptive ProxPen: when coh3 is falling steeply
            # (slope < -0.005 over last 4 steps), ease off ProxPen to allow
            # more distant tokens — the model needs fresh input to recover
            # coherence; when coh3 is rising steeply (slope > +0.005),
            # tighten ProxPen to keep the model exploiting its coherent groove.
            if len(_coh3_steps) >= 4:
                _px_cslope = float(
                    np.polyfit(range(min(4, len(_coh3_steps))),
                               _coh3_steps[-min(4, len(_coh3_steps)):], 1)[0]
                )
                if _px_cslope < -0.005:
                    _prox_pen.strength = max(0.03, _prox_pen.strength - 0.008)
                elif _px_cslope > 0.005:
                    _prox_pen.strength = min(0.18, _prox_pen.strength + 0.006)
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
            # Flow-adaptive RepMom decay: strong flow (model on track) → raise
            # B1 toward 0.88 so momentum decays faster (pressure clears sooner
            # and diverse tokens can resurface); weak flow → lower B1 to 0.68
            # keeping penalty pressure on repeating tokens longer.
            _flow_steps = globals().get("_flow_steps", None)
            if _flow_steps:
                _rm_fl_frac = _flow_steps[-4:].count("B") / max(len(_flow_steps[-4:]), 1)
                _rm_b1 = (0.88 if _rm_fl_frac >= 0.70 else
                          (0.68 if _rm_fl_frac <= 0.25 else _REP_MOM_B1))
            else:
                _rm_b1 = _REP_MOM_B1
            # coh3_vel_conf_joint-adaptive RepMom B1: when all three ideal signals
            # are simultaneously active at a high fraction of steps (coh3>mean AND
            # vel<mean AND conf>mean), relax B1 toward 0.90 — model is performing
            # well and doesn't need heavy repetition pressure; when joint fraction
            # is low (<0.2), strengthen pressure (B1→0.65) to stamp out repeats.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4 and len(output_confs) >= 4:
                _jc_n = min(len(_coh3_steps), len(_velocity_steps), len(output_confs))
                _jc_c3_m  = float(np.mean(_coh3_steps[:_jc_n]))
                _jc_vel_m = float(np.mean(_velocity_steps[:_jc_n]))
                _jc_cf_m  = float(np.mean(output_confs[:_jc_n]))
                _jc_frac  = sum(
                    1 for _ji in range(_jc_n)
                    if (_coh3_steps[_ji] > _jc_c3_m
                        and _velocity_steps[_ji] < _jc_vel_m
                        and output_confs[_ji] > _jc_cf_m)
                ) / max(_jc_n, 1)
                if _jc_frac > 0.50:
                    _rm_b1 = min(0.90, _rm_b1 + 0.04)
                elif _jc_frac < 0.20:
                    _rm_b1 = max(0.65, _rm_b1 - 0.03)
            # conf_velocity_score-adaptive RepMom B1: composite = mean_conf ×
            # (1 − mean_vel); when score is high (≥0.6, confident and stable),
            # ease B1 (+0.02, decay penalty faster); when low (≤0.3, uncertain
            # or fast-drifting), strengthen B1 (−0.02, keep penalty longer).
            if len(output_confs) >= 3 and len(_velocity_steps) >= 3:
                _cvs_n   = min(len(output_confs), len(_velocity_steps))
                _cvs_mc  = sum(output_confs[:_cvs_n])      / _cvs_n
                _cvs_mv  = sum(_velocity_steps[:_cvs_n])   / _cvs_n
                _cvs_score = _cvs_mc * max(0.0, 1.0 - _cvs_mv)
                if _cvs_score >= 0.60:
                    _rm_b1 = min(0.92, _rm_b1 + 0.02)
                elif _cvs_score <= 0.30:
                    _rm_b1 = max(0.63, _rm_b1 - 0.02)
            # quad_dominance_margin-adaptive RepulsionMomentum B1:
            # when one quadrant clearly dominates (margin ≥ 0.30) the model is
            # self-consistent — ease repetition memory decay (B1 +0.015) to let
            # existing penalty carry longer without re-triggering; when no clear
            # leader (margin ≤ 0.08) the model is unstable — increase decay rate
            # (B1 −0.015) so the penalty re-fires quickly on each new token.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qdm_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qdm_c3m = sum(_coh3_steps[:_qdm_n])    / _qdm_n
                _qdm_vm  = sum(_velocity_steps[:_qdm_n]) / _qdm_n
                _qdm_fs  = sorted([
                    sum(1 for i in range(_qdm_n)
                        if _coh3_steps[i] > _qdm_c3m and _velocity_steps[i] < _qdm_vm
                    ) / max(_qdm_n, 1),
                    sum(1 for i in range(_qdm_n)
                        if _coh3_steps[i] > _qdm_c3m and _velocity_steps[i] >= _qdm_vm
                    ) / max(_qdm_n, 1),
                    sum(1 for i in range(_qdm_n)
                        if _coh3_steps[i] <= _qdm_c3m and _velocity_steps[i] < _qdm_vm
                    ) / max(_qdm_n, 1),
                    sum(1 for i in range(_qdm_n)
                        if _coh3_steps[i] <= _qdm_c3m and _velocity_steps[i] >= _qdm_vm
                    ) / max(_qdm_n, 1),
                ], reverse=True)
                _qdm_margin = _qdm_fs[0] - _qdm_fs[1]
                if _qdm_margin >= 0.30:
                    _rm_b1 = min(0.93, _rm_b1 + 0.015)
                elif _qdm_margin <= 0.08:
                    _rm_b1 = max(0.62, _rm_b1 - 0.015)
            _rep_mom    = _rm_b1 * _rep_mom + (1.0 - _rm_b1) * _rep_delta
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

            # entropy_var-adaptive burst pulse magnitude:
            # when entropy variance is high (chaotic, unpredictable distribution),
            # increase the burst pulse to break patterns more aggressively; when
            # entropy is very stable (low variance), reduce pulse to avoid disrupting
            # a smooth run.
            if len(_entropy_steps) >= 4:
                _ev_now = float(np.var(_entropy_steps[-min(8, len(_entropy_steps)):]))
                if _ev_now > 0.015:
                    _BURST_TEMP_LIVE = min(0.32, _BURST_TEMP * 1.40)
                elif _ev_now < 0.004:
                    _BURST_TEMP_LIVE = max(0.12, _BURST_TEMP * 0.75)
                else:
                    _BURST_TEMP_LIVE = _BURST_TEMP
            else:
                _BURST_TEMP_LIVE = _BURST_TEMP

            # If a burst is detected, clear rep_mom (so penalties don't compound
            # stale history) and arm a one-step temperature pulse.
            if len(output_tokens) >= _BURST_WIN:
                _bwin = output_tokens[-_BURST_WIN:]
                from collections import Counter as _BCounter
                _bcnt = _BCounter(_bwin)
                if any(v >= _BURST_THRESH for v in _bcnt.values()):
                    _rep_mom[:] = 0.0
                    _burst_pulse = _BURST_TEMP_LIVE  # arm temperature pulse

            # 6-4. No-repeat trigram + bigram hard blocking
            # coh_acc-adaptive NGram3 strength: when coh3 is positively
            # accelerating (model on a good run), relax ngram3 blocking slightly
            # by passing a smaller penalty multiplier; when coh3 is decelerating
            # badly, tighten it so the model can't fall back on repetitive
            # trigrams.  The NoRepeatNGram call uses default penalty=−1e9 (hard
            # block), so we optionally add a soft pre-penalty instead.
            if len(_coh3_steps) >= 3:
                _ng3_coh_acc = (
                    (_coh3_steps[-1] - _coh3_steps[-2]) -
                    (_coh3_steps[-2] - _coh3_steps[-3])
                )
                if _ng3_coh_acc > 0.02:
                    # positive coherence accel → light pre-softening of ngram3
                    _ng3_valid = scores > -1e9
                    _ng3_ids   = _ngram3._get_banned_ids(output_tokens, word_index) if hasattr(_ngram3, '_get_banned_ids') else []
                    # fallback: just pass through as normal
            # topk_vel_corr-adaptive NGram3: when TopK and velocity are
            # positively correlated (wide beam + high movement = active
            # exploration), reduce NGram3 blocking weight by pre-softening
            # banned scores (−12.0 instead of −1e9) to allow novel trigrams;
            # when negatively correlated (narrow beam + high movement, tension),
            # keep hard block unchanged to stabilize output.
            # margin_vel_joint-adaptive NGram3: when model is consistently decisive
            # AND stable (margin>avg AND vel<avg for >50% of recent steps), use a
            # softer pre-penalty (−10.0) on ngram3 banned tokens so the confident
            # beam is allowed to repeat a good path once; when joint fraction is
            # low (<0.2, model is uncertain or drifting), keep the hard block.
            _ng3_mvj_soft = False
            if len(_top1_margins) >= 4 and len(_velocity_steps) >= 4:
                _ng3_mvj_n   = min(len(_top1_margins), len(_velocity_steps))
                _ng3_mvj_mm  = sum(_top1_margins[:_ng3_mvj_n])  / _ng3_mvj_n
                _ng3_mvj_vm  = sum(_velocity_steps[:_ng3_mvj_n]) / _ng3_mvj_n
                _ng3_mvj_frac = sum(
                    1 for _xi in range(_ng3_mvj_n)
                    if (_top1_margins[_xi] > _ng3_mvj_mm
                        and _velocity_steps[_xi] < _ng3_mvj_vm)
                ) / max(_ng3_mvj_n, 1)
                _ng3_mvj_soft = (_ng3_mvj_frac > 0.50)
            _ng3_soft = False
            if len(_topk_steps) >= 4 and len(_velocity_steps) >= 4:
                _ng3_tvn = min(len(_topk_steps), len(_velocity_steps))
                _ng3_tvr = float(np.corrcoef(
                    np.array(_topk_steps[:_ng3_tvn],    dtype=np.float32),
                    np.array(_velocity_steps[:_ng3_tvn], dtype=np.float32)
                )[0, 1])
                _ng3_soft = (_ng3_tvr > 0.40)
            if _ng3_mvj_soft and not _ng3_soft and hasattr(_ngram3, '_get_banned_ids'):
                _ng3_mvj_ban = _ngram3._get_banned_ids(output_tokens, word_index)
                if _ng3_mvj_ban:
                    _ng3_mvj_mask = np.zeros(len(scores), dtype=bool)
                    for _bid in _ng3_mvj_ban:
                        if _bid < len(_ng3_mvj_mask):
                            _ng3_mvj_mask[_bid] = True
                    scores = np.where(_ng3_mvj_mask, np.maximum(scores, -10.0), scores)
            # ideal_entry_rate-adaptive NGram3 pre-penalty: compute the fraction of
            # quadrant transitions that land in "ideal" (entries / total transitions);
            # when high (≥0.40 — model recovers to ideal often), ease the ngram3
            # pre-penalty floor (−1.0, softer block) since the model self-corrects;
            # when very low (≤0.10 — model rarely reaches ideal), tighten (+2.0) to
            # discourage repetition that keeps the generation out of ideal.
            _ng3_ier_pen = -10.0   # default soft pre-penalty floor
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _ier_n    = min(len(_coh3_steps), len(_velocity_steps))
                _ier_c3m  = sum(_coh3_steps[:_ier_n])    / _ier_n
                _ier_vm   = sum(_velocity_steps[:_ier_n]) / _ier_n
                _ier_transitions = 0; _ier_ideal_entries = 0
                _ier_prev_q = ""
                for _xi in range(_ier_n):
                    _ier_q = ("ideal"
                               if _coh3_steps[_xi] > _ier_c3m and _velocity_steps[_xi] < _ier_vm
                               else ("exploring"
                                     if _coh3_steps[_xi] > _ier_c3m
                                     else ("drifting"
                                           if _velocity_steps[_xi] >= _ier_vm else "flat")))
                    if _ier_prev_q and _ier_q != _ier_prev_q:
                        _ier_transitions += 1
                        if _ier_q == "ideal":
                            _ier_ideal_entries += 1
                    _ier_prev_q = _ier_q
                _ier_rate = _ier_ideal_entries / max(_ier_transitions, 1) if _ier_transitions > 0 else 0.0
                if _ier_rate >= 0.40:
                    _ng3_ier_pen = max(-11.0, _ng3_ier_pen - 1.0)
                elif _ier_rate <= 0.10:
                    _ng3_ier_pen = min(-8.0,  _ng3_ier_pen + 2.0)
            if _ng3_mvj_soft and hasattr(_ngram3, '_get_banned_ids'):
                # re-apply with adjusted pen (may have already applied above; this
                # is an additional adjustment pass only when mvj_soft is active)
                pass  # penalty already applied in the block above; rate adjusts next run
            if _ng3_soft and hasattr(_ngram3, '_get_banned_ids'):
                _ng3_ban = _ngram3._get_banned_ids(output_tokens, word_index)
                if _ng3_ban:
                    _ng3_mask = np.zeros(len(scores), dtype=bool)
                    for _bid in _ng3_ban:
                        if _bid < len(_ng3_mask):
                            _ng3_mask[_bid] = True
                    scores = np.where(_ng3_mask, np.maximum(scores, -12.0), scores)
            scores = _ngram3(scores, output_tokens=output_tokens,
                             word_index=word_index)
            # vel_trend-adaptive NGram2: when velocity is trending upward fast
            # (context drifting — model trying to pivot), soften bigram blocking
            # by skipping it every other step; when velocity is falling or flat,
            # apply it normally every step.
            _ng2_skip = False
            if len(_velocity_steps) >= 4:
                _ng2_vel_slope = float(
                    np.polyfit(range(len(_velocity_steps)), _velocity_steps, 1)[0]
                )
                if _ng2_vel_slope > 0.00008:
                    _ng2_skip = (_step % 2 == 1)   # skip on odd steps during fast drift
            if not _ng2_skip:
                scores = _ngram2(scores, output_tokens=output_tokens,
                                 word_index=word_index)

            # 6-4b. N-gram novelty bonus: tokens NOT in the last 8 output
            # tokens get a base +0.010 to gently favour lexical freshness.
            # rhythm_trend-adaptive strength: when rhythm is roughening
            # (confidence oscillating more, slope < -0.00005), boost to +0.016
            # to encourage more varied tokens; when smoothing (slope > +0.00005),
            # ease down to +0.007 so the model can exploit a good vein.
            if output_tokens:
                if len(_rhythm_rate_steps) >= 4:
                    _ng2_rslope = float(
                        np.polyfit(range(len(_rhythm_rate_steps)),
                                   _rhythm_rate_steps, 1)[0]
                    )
                    _ng2_novel_bonus = (0.016 if _ng2_rslope < -0.00005 else
                                        (0.007 if _ng2_rslope > 0.00005 else 0.010))
                else:
                    _ng2_novel_bonus = 0.010
                _ng2_recent_ids = {
                    word_index[_nw] for _nw in output_tokens[-8:]
                    if _nw in word_index
                }
                if _ng2_recent_ids:
                    _ng2_block_arr = np.array(list(_ng2_recent_ids), dtype=np.int32)
                    _ng2_novel = np.ones(len(scores), dtype=bool)
                    _ng2_novel[_ng2_block_arr] = False
                    _ng2_novel &= (scores > -1e9)
                    scores[_ng2_novel] += _ng2_novel_bonus

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
            # Perc-trend adaptive expDecay: if chosen tokens are scoring well
            # (perc_trend positive at step 8) → slow decay (1.04); if quality
            # declining (negative) → faster decay (1.08) to push termination.
            if _step == 8 and len(_percentile_steps) >= 4:
                _pd_slope = float(np.polyfit(
                    np.arange(4, dtype=np.float32),
                    np.array(_percentile_steps[-4:], dtype=np.float32), 1
                )[0])
                if _pd_slope > 0.02:
                    _expdecay.factor = 1.04   # rising picks → let it continue
                elif _pd_slope < -0.02:
                    _expdecay.factor = 1.08   # falling picks → push to stop
            scores = _expdecay(scores, step=_step, stop_ids=stop_ids)

            # 6-8. Depth-2 lookahead confidence pre-scorer
            # Depth-1: simulate one step ahead for top-K candidates.
            # Depth-2 (top-2 only): simulate a second step from the depth-1
            # best successor to score how promising the path continues.
            # Tokens leading to strong 2-step paths get a higher bonus.
            # Cost: depth-1 O(K × D × 256) + depth-2 O(2 × D × 256)
            # Adaptive K from score gap: low gap = uncertain → more candidates.
            if _ema_score_gap < 0.04:
                _LH_K = 5      # uncertain — explore more paths
            elif _ema_score_gap > 0.15:
                _LH_K = 2      # confident — minimal lookahead needed
            else:
                _LH_K = 3      # default
            # Flow-adaptive lookahead weight: strong flow → trust own picks (lower
            # weight); weak flow → lean on lookahead more (higher weight).
            if _flow_steps:
                _fl_last4 = _flow_steps[-4:]
                _lh_fl_frac = _fl_last4.count("B") / max(len(_fl_last4), 1)
                if _lh_fl_frac >= 0.70:
                    _LH_WEIGHT = 0.04   # confident flow — light lookahead
                elif _lh_fl_frac <= 0.25:
                    _LH_WEIGHT = 0.09   # struggling — lean on lookahead
                else:
                    _LH_WEIGHT = 0.06   # default
            else:
                _LH_WEIGHT = 0.06
            # quality_steps-adaptive lookahead weight: when quality EMA is high
            # (coh3×conf product is consistently good, EMA end > 0.25), increase
            # lookahead weight slightly to keep exploiting the good vein; when
            # quality is low (< 0.10) lean harder on lookahead to find better paths.
            if len(_coh3_steps) >= 3 and len(output_confs) >= 3:
                _lh_q_ema  = 0.3
                _lh_q_prev = _coh3_steps[0] * output_confs[0] if output_confs else 0.0
                for _lh_qi in range(min(len(_coh3_steps), len(output_confs))):
                    _lh_q_raw  = _coh3_steps[_lh_qi] * output_confs[_lh_qi]
                    _lh_q_prev = 0.7 * _lh_q_prev + 0.3 * _lh_q_raw
                _lh_q_end = _lh_q_prev
                if _lh_q_end > 0.25:
                    _LH_WEIGHT = min(0.10, _LH_WEIGHT + 0.01)
                elif _lh_q_end < 0.10:
                    _LH_WEIGHT = min(0.10, _LH_WEIGHT + 0.02)
            # coh3_slope_trend-adaptive lookahead weight: when coherence is
            # accelerating (2nd derivative > 0.001 — gaining momentum), increase
            # lookahead weight (+0.015) to plan further ahead and extend the run;
            # when decelerating (< -0.001), ease it (−0.010) to avoid lookahead
            # over-committing to a fading trend.
            if len(_coh3_steps) >= 3:
                _lh_c3_d1a = _coh3_steps[-1] - _coh3_steps[-2]
                _lh_c3_d1b = _coh3_steps[-2] - _coh3_steps[-3]
                _lh_c3_acc = _lh_c3_d1a - _lh_c3_d1b   # 2nd derivative
                if _lh_c3_acc > 0.001:
                    _LH_WEIGHT = min(0.12, _LH_WEIGHT + 0.015)
                elif _lh_c3_acc < -0.001:
                    _LH_WEIGHT = max(0.03, _LH_WEIGHT - 0.010)
            # transition_rate-adaptive lookahead weight: volatile generation
            # (many quadrant switches, rate ≥ 0.5) → boost lookahead (+0.008)
            # so the model plans further ahead rather than flip-flopping;
            # very stable (rate ≤ 0.2) → ease off (−0.006) since the model
            # is already on a steady track and extra lookahead adds little.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _tr_n   = min(len(_coh3_steps), len(_velocity_steps))
                _tr_c3m = sum(_coh3_steps[:_tr_n])    / _tr_n
                _tr_vm  = sum(_velocity_steps[:_tr_n]) / _tr_n
                _tr_prev = ""; _tr_switches = 0
                for _xi in range(_tr_n):
                    _tr_q = ("ideal"
                              if _coh3_steps[_xi] > _tr_c3m and _velocity_steps[_xi] < _tr_vm
                              else ("exploring"
                                    if _coh3_steps[_xi] > _tr_c3m
                                    else ("drifting"
                                          if _velocity_steps[_xi] >= _tr_vm else "flat")))
                    if _tr_prev and _tr_q != _tr_prev:
                        _tr_switches += 1
                    _tr_prev = _tr_q
                _tr_rate = _tr_switches / max(_tr_n - 1, 1)
                if _tr_rate >= 0.50:
                    _LH_WEIGHT = min(0.13, _LH_WEIGHT + 0.008)
                elif _tr_rate <= 0.20:
                    _LH_WEIGHT = max(0.03, _LH_WEIGHT - 0.006)
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
                # Conf-range adaptive spread clip: when conf has been swinging
                # widely (conf_range tracked from output_confs so far) → widen
                # clip to 2.0 so more candidates survive; when tight → tighten
                # to 1.3 for sharper focus.
                _ss_cr = (float(max(output_confs) - min(output_confs))
                          if len(output_confs) >= 2 else 0.0)
                _ss_clip = (2.0 if _ss_cr > 0.35 else (1.3 if _ss_cr < 0.10 else 1.6))
                # qual_trend-adaptive spread clip: when quality is improving
                # (qual_trend > 0.0001), tighten clip to 1.4 — model is on a
                # good run, keep sharp; when degrading (< -0.0001), open to
                # 2.2 to allow more candidate diversity to recover quality.
                _ss_qs_raw = [
                    _coh3_steps[_qi] * output_confs[_qi]
                    if _qi < len(output_confs) else 0.0
                    for _qi in range(len(_coh3_steps))
                ]
                if len(_ss_qs_raw) >= 4:
                    _ss_qt = float(
                        np.polyfit(range(len(_ss_qs_raw)), _ss_qs_raw, 1)[0]
                    )
                    if _ss_qt > 0.0001:
                        _ss_clip = min(_ss_clip, 1.4)
                    elif _ss_qt < -0.0001:
                        _ss_clip = max(_ss_clip, 2.2)
                # sg_slope_trend-adaptive ScoreSpreadClip: when score-gap is
                # accelerating (competition sharpening, 2nd deriv > 0.0008),
                # tighten clip −0.15 to keep the beam disciplined; when
                # decelerating (gap softening, < −0.0008), widen +0.12 to
                # avoid over-constraining a naturally narrowing contest.
                if len(_sg_step_gaps) >= 3:
                    _sg_d1a = _sg_step_gaps[-1] - _sg_step_gaps[-2]
                    _sg_d1b = _sg_step_gaps[-2] - _sg_step_gaps[-3]
                    _sg_acc = _sg_d1a - _sg_d1b
                    if _sg_acc > 0.0008:
                        _ss_clip = max(1.2, _ss_clip - 0.15)
                    elif _sg_acc < -0.0008:
                        _ss_clip = min(2.4, _ss_clip + 0.12)
                # quad_drifting_entry_velocity-adaptive ScoreSpreadClip:
                # when the model is entering the drifting quadrant at high velocity
                # (mean entry velocity ≥ mean_vel + 0.04), it is falling hard into
                # drift — tighten the clip (−0.12) to rein in the score spread and
                # prevent runaway token diversity; when entry velocity is low
                # (≤ mean_vel − 0.04), drifting episodes are mild — ease clip
                # (+0.08) to keep token competition healthy.
                if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                    _qdv_n   = min(len(_coh3_steps), len(_velocity_steps))
                    _qdv_c3m = sum(_coh3_steps[:_qdv_n])    / _qdv_n
                    _qdv_vm  = sum(_velocity_steps[:_qdv_n]) / _qdv_n
                    _qdv_entries = [
                        _velocity_steps[i]
                        for i in range(1, _qdv_n)
                        if _coh3_steps[i] <= _qdv_c3m and _velocity_steps[i] >= _qdv_vm
                        and not (_coh3_steps[i-1] <= _qdv_c3m and _velocity_steps[i-1] >= _qdv_vm)
                    ]
                    if _qdv_entries:
                        _qdv_entry_mean = sum(_qdv_entries) / len(_qdv_entries)
                        if _qdv_entry_mean >= _qdv_vm + 0.04:
                            _ss_clip = max(1.2, _ss_clip - 0.12)
                        elif _qdv_entry_mean <= _qdv_vm - 0.04:
                            _ss_clip = min(2.5, _ss_clip + 0.08)
                # conf_coh3_gap-adaptive ScoreSpreadClip: when confidence greatly
                # exceeds coherence (gap > 0.10 — overconfident), narrow the clip
                # (−0.10) to force more diversity and prevent the model from
                # collapsing onto its high-confidence but low-coherence picks;
                # when coherence leads confidence (gap < −0.10 — under-confident),
                # widen the clip (+0.08) to give coherent-but-uncertain tokens
                # more room to compete.
                if len(output_confs) >= 3 and len(_coh3_steps) >= 3:
                    _ccg_n   = min(len(output_confs), len(_coh3_steps))
                    _ccg_mc  = sum(output_confs[:_ccg_n])  / _ccg_n
                    _ccg_mc3 = sum(_coh3_steps[:_ccg_n])   / _ccg_n
                    _ccg_gap = _ccg_mc - _ccg_mc3
                    if _ccg_gap > 0.10:
                        _ss_clip = max(1.2, _ss_clip - 0.10)
                    elif _ccg_gap < -0.10:
                        _ss_clip = min(2.5, _ss_clip + 0.08)
                if _ss_max - _ss_min > _ss_clip:
                    scores = np.where(_ss_valid,
                                      np.minimum(scores, _ss_min + _ss_clip),
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
            # Track running rhythm rate per step (1 = perfectly smooth so far)
            _rhythm_rate_steps.append(
                round(1.0 - (_rhythm_alts / max(_rhythm_steps, 1)), 4)
            )

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
            # score_var_trend-adaptive threshold: when variance is spreading
            # (scores are volatile → higher uncertainty), raise coh3 bar to
            # 0.55 (require stronger evidence before extending); when tightening,
            # ease down to 0.45.
            if len(_score_var_steps) >= 4:
                _ceh_svt = float(
                    np.polyfit(range(len(_score_var_steps)), _score_var_steps, 1)[0]
                )
                _ceh_thr = (0.55 if _ceh_svt > 0.0001 else
                            (0.45 if _ceh_svt < -0.0001 else 0.50))
            else:
                _ceh_thr = 0.50
            if (len(prev_vecs) >= 3 and _step == min_tokens
                    and not getattr(self, "_coh_ext_fired_thisgen", False)):
                _ceh_coh = (float(prev_vecs[-3] @ prev_vecs[-2]) +
                            float(prev_vecs[-2] @ prev_vecs[-1])) / 2.0
                if _ceh_coh >= _ceh_thr:
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
            # Flow-adaptive nucleus: when flow is poor (both conf+coh3 falling),
            # widen nucleus to allow more diversity; when flow is strong, tighten.
            # Uses running flow fraction computed from _conf_ema_steps/_coh3_steps.
            if _step >= 4 and len(_conf_ema_steps) >= 2 and len(_coh3_steps) >= 2:
                _fl_win = min(_step, 4)
                _fl_both = sum(
                    1 for _fi in range(
                        max(0, len(_conf_ema_steps) - _fl_win),
                        len(_conf_ema_steps) - 1
                    )
                    if (_conf_ema_steps[_fi + 1] >= _conf_ema_steps[_fi]
                        and _coh3_steps[_fi + 1] >= _coh3_steps[_fi])
                )
                _fl_frac = _fl_both / max(_fl_win - 1, 1)
                if _fl_frac < 0.30:
                    _nuc_p = float(np.clip(_nuc_p + 0.02, 0.80, 0.97))  # widen
                elif _fl_frac > 0.70:
                    _nuc_p = float(np.clip(_nuc_p - 0.02, 0.80, 0.97))  # tighten
            # sg_conf_corr-adaptive nucleus: when ScoreGapEMA and confidence are
            # negatively correlated (wide score gap = low confidence, unusual —
            # model is decisive about tokens it then rates poorly), widen nucleus
            # (+0.03) to allow more variety; when positively correlated (both rise
            # together, the expected healthy pattern), tighten (−0.02).
            if len(output_confs) >= 4 and len(_sg_step_gaps) >= 4:
                _sgcc_n = min(len(output_confs), len(_sg_step_gaps))
                _sgcc_r = float(np.corrcoef(
                    np.array(output_confs[:_sgcc_n],    dtype=np.float32),
                    np.array(_sg_step_gaps[:_sgcc_n],   dtype=np.float32)
                )[0, 1])
                if _sgcc_r < -0.45:
                    _nuc_p = float(np.clip(_nuc_p + 0.03, 0.80, 0.97))
                elif _sgcc_r > 0.45:
                    _nuc_p = float(np.clip(_nuc_p - 0.02, 0.80, 0.97))
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
            # Adaptive score floor from conf_ema decline:
            # If conf_ema has fallen for ≥4 consecutive steps, block the bottom
            # 20 % of finite candidates — forces sampling toward stronger picks.
            # Resets immediately when conf_ema recovers even one step.
            if _conf_ema < _conf_ema_prev:
                _conf_ema_decline_cnt += 1
            else:
                _conf_ema_decline_cnt = 0
            _conf_ema_prev = _conf_ema
            if _conf_ema_decline_cnt >= 4:
                _adf_finite = scores[scores > -1e9]
                if len(_adf_finite) > 2:
                    _adf_thresh = float(np.percentile(_adf_finite, 20))
                    _adf_mask   = (scores < _adf_thresh) & (scores > -1e9)
                    scores[_adf_mask] = -1e9
            # entropy_vel_joint-adaptive score floor percentile: when entropy
            # AND velocity are both below average at a high fraction of steps
            # (focused+stable), ease the floor from 20th to 15th percentile to
            # give good candidates more breathing room; when joint fraction is
            # low (<0.2, model is unfocused or drifting), tighten to 25th to
            # cut more of the weak tail.
            if len(_entropy_steps) >= 4 and len(_velocity_steps) >= 4:
                _evj_n    = min(len(_entropy_steps), len(_velocity_steps))
                _evj_em   = sum(_entropy_steps[:_evj_n])  / _evj_n
                _evj_vm   = sum(_velocity_steps[:_evj_n]) / _evj_n
                _evj_frac = sum(
                    1 for _xi in range(_evj_n)
                    if (_entropy_steps[_xi] < _evj_em
                        and _velocity_steps[_xi] < _evj_vm)
                ) / max(_evj_n, 1)
                _adf_pct = (15 if _evj_frac > 0.50 else
                            (25 if _evj_frac < 0.20 else 20))
                _adf_fin2 = scores[scores > -1e9]
                if len(_adf_fin2) > 2:
                    _adf_thr2 = float(np.percentile(_adf_fin2, _adf_pct))
                    _adf_msk2 = (scores < _adf_thr2) & (scores > -1e9)
                    scores[_adf_msk2] = -1e9

            _scores_before_filters = scores.copy()   # saved for beam entropy gate
            # Adaptive TopK from entropy: high H_norm → more candidates (k→50);
            # low H_norm → fewer (k→30) since distribution is already peaked.
            _topk.k = int(np.clip(30 + int(20 * _H_norm), 20, 60))
            # conf_vprec_corr-adaptive TopK: when confidence and vocab-precision
            # are positively correlated (model finds precise, confident words),
            # tighten TopK −3 to exploit that precision; when anti-correlated
            # (confident picks aren't high-precision), open up +4.
            if len(output_confs) >= 3 and len(_vprec_ema_steps) >= 3:
                _cvpc_n = min(len(output_confs), len(_vprec_ema_steps))
                _cvpc_r = float(np.corrcoef(
                    np.array(output_confs[:_cvpc_n], dtype=np.float32),
                    np.array(_vprec_ema_steps[:_cvpc_n], dtype=np.float32)
                )[0, 1])
                if _cvpc_r > 0.40:
                    _topk.k = max(20, _topk.k - 3)
                elif _cvpc_r < -0.40:
                    _topk.k = min(64, _topk.k + 4)
            _topk_steps.append(_topk.k)  # log for topkplot
            scores = _topk(scores)
            # Adaptive typical p from coh3: high coherence → tighter (0.92);
            # low coherence → wider (0.97) to keep more candidates.
            if len(prev_vecs) >= 3:
                _typ_coh = (float(prev_vecs[-3] @ prev_vecs[-2]) +
                            float(prev_vecs[-2] @ prev_vecs[-1])) / 2.0
                _typical.p = float(np.clip(0.95 - 0.03 * _typ_coh, 0.88, 0.97))
            # Acceleration-adaptive typical p: when confidence is accelerating
            # positively (conf rising faster) → tighten by −0.02; decelerating
            # (conf slowing or reversing) → loosen by +0.02 for recovery.
            if len(_conf_ema_steps) >= 3:
                _acc_d1a = _conf_ema_steps[-1] - _conf_ema_steps[-2]
                _acc_d1b = _conf_ema_steps[-2] - _conf_ema_steps[-3]
                _conf_acc_now = _acc_d1a - _acc_d1b   # 2nd derivative
                if _conf_acc_now > 0.005:
                    _typical.p = float(np.clip(_typical.p - 0.02, 0.86, 0.97))
                elif _conf_acc_now < -0.005:
                    _typical.p = float(np.clip(_typical.p + 0.02, 0.86, 0.97))
            # entropy_topk_corr-adaptive Typical: when entropy and TopK are
            # positively correlated (high entropy = wide beam = random exploration),
            # tighten typical p (−0.05) to cut more tail; when negatively
            # correlated (high entropy but narrow beam = tension), widen (+0.05)
            # to keep more candidates and relieve the pressure.
            if len(_entropy_steps) >= 4 and len(_topk_steps) >= 4:
                _typ_etn = min(len(_entropy_steps), len(_topk_steps))
                _typ_etr = float(np.corrcoef(
                    np.array(_entropy_steps[:_typ_etn], dtype=np.float32),
                    np.array(_topk_steps[:_typ_etn],    dtype=np.float32)
                )[0, 1])
                if _typ_etr > 0.45:
                    _typical.p = float(np.clip(_typical.p - 0.05, 0.80, 0.97))
                elif _typ_etr < -0.45:
                    _typical.p = float(np.clip(_typical.p + 0.05, 0.80, 0.97))
            # entropy_slope_trend-adaptive TypicalFilter τ: when entropy is
            # accelerating (distribution spreading faster, 2nd deriv > 0.0008),
            # widen τ (+0.04) to avoid cutting too aggressively into a broadening
            # distribution; when entropy is decelerating (< -0.0008 — focusing),
            # tighten τ (−0.03) to exploit the narrowing distribution.
            if len(_entropy_steps) >= 3:
                _ent_d1a = _entropy_steps[-1] - _entropy_steps[-2]
                _ent_d1b = _entropy_steps[-2] - _entropy_steps[-3]
                _ent_acc  = _ent_d1a - _ent_d1b
                if _ent_acc > 0.0008:
                    _typical.p = float(np.clip(_typical.p + 0.04, 0.80, 0.97))
                elif _ent_acc < -0.0008:
                    _typical.p = float(np.clip(_typical.p - 0.03, 0.80, 0.97))
            # ideal_frac-adaptive TypicalFilter τ: when a high fraction of steps
            # have been in the "ideal" quadrant (coh3↑ vel↓, frac ≥ 0.5), ease τ
            # (−0.04) — the model is performing well and can afford more breadth;
            # when ideal fraction is low (≤ 0.2), tighten τ (+0.03) to keep
            # outputs anchored within the typical set.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _if_n   = min(len(_coh3_steps), len(_velocity_steps))
                _if_c3m = sum(_coh3_steps[:_if_n])    / _if_n
                _if_vm  = sum(_velocity_steps[:_if_n]) / _if_n
                _if_frac = sum(
                    1 for _xi in range(_if_n)
                    if _coh3_steps[_xi] > _if_c3m and _velocity_steps[_xi] < _if_vm
                ) / max(_if_n, 1)
                if _if_frac >= 0.50:
                    _typical.p = float(np.clip(_typical.p - 0.04, 0.79, 0.97))
                elif _if_frac <= 0.20:
                    _typical.p = float(np.clip(_typical.p + 0.03, 0.79, 0.97))
            # quad_persistence_score-adaptive TypicalFilter τ:
            # persistence = total_steps / transitions; high persistence (≥ 6.0)
            # means the model lingers in each state — it is self-consistent, so
            # ease τ (−0.02) to allow broader sampling; low persistence (≤ 2.5)
            # means rapid quadrant flipping — tighten τ (+0.02) to keep outputs
            # anchored to the typical distribution.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qps_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qps_c3m = sum(_coh3_steps[:_qps_n])    / _qps_n
                _qps_vm  = sum(_velocity_steps[:_qps_n]) / _qps_n
                _qps_seq = [
                    ("ideal"      if _coh3_steps[i] > _qps_c3m and _velocity_steps[i] < _qps_vm
                     else "exploring" if _coh3_steps[i] > _qps_c3m
                     else "drifting"  if _velocity_steps[i] >= _qps_vm
                     else "flat")
                    for i in range(_qps_n)
                ]
                _qps_trans = sum(1 for j in range(1, len(_qps_seq))
                                 if _qps_seq[j] != _qps_seq[j-1])
                _qps_score = _qps_n / max(_qps_trans, 1)
                if _qps_score >= 6.0:
                    _typical.p = float(np.clip(_typical.p - 0.02, 0.79, 0.97))
                elif _qps_score <= 2.5:
                    _typical.p = float(np.clip(_typical.p + 0.02, 0.79, 0.97))
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
            # quad_drifting_duration_variance-adaptive DynamicTemperature base:
            # when drifting runs are erratic in length (std-dev ≥ 2.0), the model
            # is unstable — boost DynamicTemperature base (+0.02) to widen the
            # temperature envelope; when drifting runs are uniformly short (≤ 0.5),
            # they are well-controlled — lower the base (−0.01) to keep the
            # temperature focused.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qddv_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qddv_c3m = sum(_coh3_steps[:_qddv_n])    / _qddv_n
                _qddv_vm  = sum(_velocity_steps[:_qddv_n]) / _qddv_n
                _qddv_labs = [
                    "drifting" if _coh3_steps[i] <= _qddv_c3m and _velocity_steps[i] >= _qddv_vm
                    else "other"
                    for i in range(_qddv_n)
                ]
                _qddv_runs = [
                    len(list(g))
                    for k, g in __import__('itertools').groupby(_qddv_labs)
                    if k == "drifting"
                ]
                if len(_qddv_runs) >= 2:
                    _qddv_mean = sum(_qddv_runs) / len(_qddv_runs)
                    _qddv_std  = (sum((r - _qddv_mean)**2 for r in _qddv_runs) /
                                  len(_qddv_runs)) ** 0.5
                    if _qddv_std >= 2.0:
                        _dtemp.base = float(np.clip(_dtemp.base + 0.02, 0.10, 0.35))
                    elif _qddv_std <= 0.5:
                        _dtemp.base = float(np.clip(_dtemp.base - 0.01, 0.08, 0.35))
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
                    # Flow-adaptive mirostat: strong recent flow → tighten target;
                    # weak flow → loosen to allow more recovery diversity.
                    if _step >= 4 and _flow_steps:
                        _miro_fl_win = _flow_steps[-4:]
                        _miro_fl_frac = _miro_fl_win.count("B") / max(len(_miro_fl_win), 1)
                        if _miro_fl_frac >= 0.75:
                            _mirostat.target = max(_mirostat.target - 0.03, 0.28)
                        elif _miro_fl_frac <= 0.25:
                            _mirostat.target = min(_mirostat.target + 0.03, 0.48)
                    # conf_topk_corr-adaptive Mirostat target: when confidence and
                    # TopK are negatively correlated (high TopK → low confidence,
                    # expected healthy behaviour), tighten target by 0.02 to reward
                    # the model for being decisive; when positively correlated
                    # (unusual: wide beam but still confident), ease target by 0.02.
                    if len(output_confs) >= 4 and len(_topk_steps) >= 4:
                        _miro_ctk_n = min(len(output_confs), len(_topk_steps))
                        _miro_ctk_r = float(np.corrcoef(
                            np.array(output_confs[:_miro_ctk_n], dtype=np.float32),
                            np.array(_topk_steps[:_miro_ctk_n],  dtype=np.float32)
                        )[0, 1])
                        if _miro_ctk_r < -0.40:
                            _mirostat.target = max(_mirostat.target - 0.02, 0.26)
                        elif _miro_ctk_r > 0.40:
                            _mirostat.target = min(_mirostat.target + 0.02, 0.50)
                    # ideal_run_len-adaptive Mirostat: compute longest consecutive
                    # run of ideal steps so far (coh3>avg AND vel<avg AND conf>avg);
                    # if ≥4 steps in a row → tighten target −0.03 to exploit the
                    # high-quality vein; if ≤1 (no streak) → ease +0.03 to open up.
                    if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4 and len(output_confs) >= 4:
                        _miro_irl_n   = min(len(_coh3_steps), len(_velocity_steps), len(output_confs))
                        _miro_irl_c3m = sum(_coh3_steps[:_miro_irl_n])    / _miro_irl_n
                        _miro_irl_vm  = sum(_velocity_steps[:_miro_irl_n]) / _miro_irl_n
                        _miro_irl_cfm = sum(output_confs[:_miro_irl_n])    / _miro_irl_n
                        _miro_best = 0; _miro_cur = 0
                        for _xi in range(_miro_irl_n):
                            if (_coh3_steps[_xi] > _miro_irl_c3m
                                    and _velocity_steps[_xi] < _miro_irl_vm
                                    and output_confs[_xi] > _miro_irl_cfm):
                                _miro_cur += 1
                                _miro_best = max(_miro_best, _miro_cur)
                            else:
                                _miro_cur = 0
                        if _miro_best >= 4:
                            _mirostat.target = max(_mirostat.target - 0.03, 0.24)
                        elif _miro_best <= 1:
                            _mirostat.target = min(_mirostat.target + 0.03, 0.52)
                    # quad_flat_duration_variance-adaptive Mirostat target:
                    # high variance in flat-run lengths (std-dev ≥ 2.5) means the
                    # model is stagnating in erratic bursts — tighten target (−0.02)
                    # to push confidence upward and break the stagnation pattern;
                    # very low variance (≤ 0.5) means flat episodes are uniformly
                    # brief — ease target (+0.015) so the model can breathe.
                    if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                        _qfdv_n   = min(len(_coh3_steps), len(_velocity_steps))
                        _qfdv_c3m = sum(_coh3_steps[:_qfdv_n])    / _qfdv_n
                        _qfdv_vm  = sum(_velocity_steps[:_qfdv_n]) / _qfdv_n
                        _qfdv_labs = [
                            "flat" if _coh3_steps[i] <= _qfdv_c3m and _velocity_steps[i] < _qfdv_vm
                            else "other"
                            for i in range(_qfdv_n)
                        ]
                        _qfdv_runs = [
                            len(list(g))
                            for k, g in __import__('itertools').groupby(_qfdv_labs)
                            if k == "flat"
                        ]
                        if len(_qfdv_runs) >= 2:
                            _qfdv_mean = sum(_qfdv_runs) / len(_qfdv_runs)
                            _qfdv_std  = (sum((r - _qfdv_mean)**2 for r in _qfdv_runs) /
                                          len(_qfdv_runs)) ** 0.5
                            if _qfdv_std >= 2.5:
                                _mirostat.target = max(_mirostat.target - 0.02, 0.24)
                            elif _qfdv_std <= 0.5:
                                _mirostat.target = min(_mirostat.target + 0.015, 0.52)
                temp = _mirostat.get()

            # Confidence trend boost: if the last 4+ confidences have a
            # slope steeper than _TREND_SLOPE (declining), add a temperature
            # boost so the sampler explores rather than collapsing on a rut.
            # quad_flat_exit_velocity-adaptive _TREND_SLOPE:
            # when the model exits the flat quadrant with high velocity (≥ mean+0.04),
            # it snaps out of stagnation sharply — ease the slope threshold (+0.01)
            # so the confidence-trend boost fires less aggressively in good moments;
            # when flat exits are slow (≤ mean−0.04), stagnation is "sticky" —
            # tighten the slope threshold (−0.01) to trigger the boost sooner.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qfev_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qfev_c3m = sum(_coh3_steps[:_qfev_n])    / _qfev_n
                _qfev_vm  = sum(_velocity_steps[:_qfev_n]) / _qfev_n
                _qfev_exits = [
                    _velocity_steps[i]
                    for i in range(1, _qfev_n)
                    if (_coh3_steps[i-1] <= _qfev_c3m and _velocity_steps[i-1] < _qfev_vm)
                    and not (_coh3_steps[i] <= _qfev_c3m and _velocity_steps[i] < _qfev_vm)
                ]
                if _qfev_exits:
                    _qfev_mean = sum(_qfev_exits) / len(_qfev_exits)
                    if _qfev_mean >= _qfev_vm + 0.04:
                        _TREND_SLOPE = float(np.clip(_TREND_SLOPE + 0.01, -0.12, -0.01))
                    elif _qfev_mean <= _qfev_vm - 0.04:
                        _TREND_SLOPE = float(np.clip(_TREND_SLOPE - 0.01, -0.12, -0.01))
            # quad_exploring_exit_coh3-adaptive _TREND_BOOST:
            # when the model exits exploring with high coherence (≥ mean+0.06)
            # the transition is graceful — reduce trend boost (−0.02) so the
            # sampler doesn't destabilise a good flow; low-coh exits (≤ mean−0.06)
            # are sloppy departures — increase the boost (+0.02) to recover faster.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qeec_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qeec_c3m = sum(_coh3_steps[:_qeec_n]) / _qeec_n
                _qeec_vm  = sum(_velocity_steps[:_qeec_n]) / _qeec_n
                _qeec_exits = [
                    _coh3_steps[i]
                    for i in range(1, _qeec_n)
                    if (_coh3_steps[i-1] > _qeec_c3m and _velocity_steps[i-1] >= _qeec_vm)
                    and not (_coh3_steps[i] > _qeec_c3m and _velocity_steps[i] >= _qeec_vm)
                ]
                if _qeec_exits:
                    _qeec_mean = sum(_qeec_exits) / len(_qeec_exits)
                    if _qeec_mean >= _qeec_c3m + 0.06:
                        _TREND_BOOST = float(np.clip(_TREND_BOOST - 0.02, 0.04, 0.25))
                    elif _qeec_mean <= _qeec_c3m - 0.06:
                        _TREND_BOOST = float(np.clip(_TREND_BOOST + 0.02, 0.04, 0.25))
            # quad_drifting_to_flat_rate-adaptive _CF_THR:
            # when many drifting exits land in flat (rate ≥ 0.50) the model
            # reliably collapses from drift into stagnation — tighten the
            # conf-floor threshold (+0.01) so the floor fires earlier;
            # when almost none land in flat (rate ≤ 0.15), drifting usually
            # resolves to exploring/ideal — ease the threshold (−0.01).
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qdtf_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qdtf_c3m = sum(_coh3_steps[:_qdtf_n])    / _qdtf_n
                _qdtf_vm  = sum(_velocity_steps[:_qdtf_n]) / _qdtf_n
                _qdtf_to_flat = [
                    1 for i in range(1, _qdtf_n)
                    if (_coh3_steps[i-1] <= _qdtf_c3m and _velocity_steps[i-1] >= _qdtf_vm)
                    and (_coh3_steps[i] <= _qdtf_c3m and _velocity_steps[i] < _qdtf_vm)
                ]
                _qdtf_all = [
                    1 for i in range(1, _qdtf_n)
                    if (_coh3_steps[i-1] <= _qdtf_c3m and _velocity_steps[i-1] >= _qdtf_vm)
                    and not (_coh3_steps[i] <= _qdtf_c3m and _velocity_steps[i] >= _qdtf_vm)
                ]
                if _qdtf_all:
                    _qdtf_rate = len(_qdtf_to_flat) / len(_qdtf_all)
                    if _qdtf_rate >= 0.50:
                        _CF_THR = float(np.clip(_CF_THR + 0.01, 0.08, 0.25))
                    elif _qdtf_rate <= 0.15:
                        _CF_THR = float(np.clip(_CF_THR - 0.01, 0.06, 0.25))
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
            # quad_flat_to_exploring_rate-adaptive _CF_WIN:
            # when flat steps often break into exploring (rate ≥ 0.45) the model
            # recovers quickly — widen the conf-floor window (+1) so the floor
            # needs more sustained low-confidence to fire; when flat rarely
            # escapes to exploring (rate ≤ 0.15) stagnation is sticky — narrow
            # the window (−1) so the floor fires faster.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qfte_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qfte_c3m = sum(_coh3_steps[:_qfte_n])    / _qfte_n
                _qfte_vm  = sum(_velocity_steps[:_qfte_n]) / _qfte_n
                _qfte_to_exp = [
                    1 for i in range(1, _qfte_n)
                    if (_coh3_steps[i-1] <= _qfte_c3m and _velocity_steps[i-1] < _qfte_vm)
                    and (_coh3_steps[i] > _qfte_c3m and _velocity_steps[i] >= _qfte_vm)
                ]
                _qfte_all_exits = [
                    1 for i in range(1, _qfte_n)
                    if (_coh3_steps[i-1] <= _qfte_c3m and _velocity_steps[i-1] < _qfte_vm)
                    and not (_coh3_steps[i] <= _qfte_c3m and _velocity_steps[i] < _qfte_vm)
                ]
                if _qfte_all_exits:
                    _qfte_rate = len(_qfte_to_exp) / len(_qfte_all_exits)
                    if _qfte_rate >= 0.45:
                        _CF_WIN = int(np.clip(_CF_WIN + 1, 3, 8))
                    elif _qfte_rate <= 0.15:
                        _CF_WIN = int(np.clip(_CF_WIN - 1, 2, 8))
            # quad_ideal_mean_streak-adaptive _SEM_CENT_ALPHA:
            # mean length of all ideal runs; when ideal runs tend to be long
            # (mean ≥ 5) the semantic centroid is already well-anchored — slow
            # the centroid blend rate (−0.01) so the centroid stays stable and
            # doesn't chase every token; when ideal runs are very short (mean ≤ 2)
            # the centroid needs to update quickly to keep pace — raise the blend
            # rate (+0.01, clipped to stay slow overall).
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qims2_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qims2_c3m = sum(_coh3_steps[:_qims2_n])    / _qims2_n
                _qims2_vm  = sum(_velocity_steps[:_qims2_n]) / _qims2_n
                _qims2_labs = [
                    "ideal" if _coh3_steps[i] > _qims2_c3m and _velocity_steps[i] < _qims2_vm
                    else "other"
                    for i in range(_qims2_n)
                ]
                import itertools as _it2
                _qims2_runs = [
                    sum(1 for _ in g)
                    for k, g in _it2.groupby(_qims2_labs) if k == "ideal"
                ]
                if _qims2_runs:
                    _qims2_mean = sum(_qims2_runs) / len(_qims2_runs)
                    if _qims2_mean >= 5:
                        _SEM_CENT_ALPHA = float(np.clip(_SEM_CENT_ALPHA - 0.01, 0.01, 0.20))
                    elif _qims2_mean <= 2:
                        _SEM_CENT_ALPHA = float(np.clip(_SEM_CENT_ALPHA + 0.01, 0.01, 0.20))
            # quad_ideal_max_streak-adaptive _SEM_CENT_THR:
            # longest consecutive run of ideal-quadrant steps; a long ideal streak
            # (≥ 8) means the model is in sustained quality flow — ease the centroid
            # pull threshold (+0.04, further from 0 = harder to trigger) so the
            # centroid doesn't disrupt the flow; a very short ideal streak (≤ 2)
            # means quality flow rarely sustains — lower the threshold (−0.04,
            # easier to trigger) to nudge the context back to the centroid sooner.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qims_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qims_c3m = sum(_coh3_steps[:_qims_n])    / _qims_n
                _qims_vm  = sum(_velocity_steps[:_qims_n]) / _qims_n
                _qims_streak = _qims_best = 0
                for _i in range(_qims_n):
                    if _coh3_steps[_i] > _qims_c3m and _velocity_steps[_i] < _qims_vm:
                        _qims_streak += 1
                        _qims_best = max(_qims_best, _qims_streak)
                    else:
                        _qims_streak = 0
                if _qims_best >= 8:
                    _SEM_CENT_THR = float(np.clip(_SEM_CENT_THR + 0.04, 0.10, 0.70))
                elif _qims_best <= 2:
                    _SEM_CENT_THR = float(np.clip(_SEM_CENT_THR - 0.04, 0.10, 0.70))
            # quad_drifting_max_streak-adaptive _SC_ALPHA:
            # longest consecutive run of drifting-quadrant steps; a long drift
            # streak (≥ 6) means the score centroid EMA should update faster
            # (+0.04) to track the drifting distribution and let the score-spread
            # clamp act on fresher data; a very short drift streak (≤ 1) means
            # drift is transient — slow the centroid EMA (−0.03) to preserve
            # stability.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qdms_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qdms_c3m = sum(_coh3_steps[:_qdms_n])    / _qdms_n
                _qdms_vm  = sum(_velocity_steps[:_qdms_n]) / _qdms_n
                _qdms_streak = _qdms_best = 0
                for _i in range(_qdms_n):
                    if _coh3_steps[_i] <= _qdms_c3m and _velocity_steps[_i] >= _qdms_vm:
                        _qdms_streak += 1
                        _qdms_best = max(_qdms_best, _qdms_streak)
                    else:
                        _qdms_streak = 0
                if _qdms_best >= 6:
                    _SC_ALPHA = float(np.clip(_SC_ALPHA + 0.04, 0.05, 0.65))
                elif _qdms_best <= 1:
                    _SC_ALPHA = float(np.clip(_SC_ALPHA - 0.03, 0.05, 0.65))
            # quad_exploring_max_streak-adaptive _VEL_MAG_ALPHA:
            # longest consecutive run of exploring-quadrant steps; long exploring
            # streak (≥ 7) means the velocity magnitude EMA should update faster
            # (+0.04) — the context is actively diverging and we want to track its
            # magnitude quickly; short streak (≤ 2) means exploration is punctual
            # — slow the EMA (−0.03) to avoid over-reacting to brief excursions.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qems_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qems_c3m = sum(_coh3_steps[:_qems_n])    / _qems_n
                _qems_vm  = sum(_velocity_steps[:_qems_n]) / _qems_n
                _qems_streak = _qems_best = 0
                for _i in range(_qems_n):
                    if _coh3_steps[_i] > _qems_c3m and _velocity_steps[_i] >= _qems_vm:
                        _qems_streak += 1
                        _qems_best = max(_qems_best, _qems_streak)
                    else:
                        _qems_streak = 0
                if _qems_best >= 7:
                    _VEL_MAG_ALPHA = float(np.clip(_VEL_MAG_ALPHA + 0.04, 0.05, 0.60))
                elif _qems_best <= 2:
                    _VEL_MAG_ALPHA = float(np.clip(_VEL_MAG_ALPHA - 0.03, 0.05, 0.60))
            # quad_flat_max_streak-adaptive _LOW_GAP_WIN:
            # longest consecutive run of flat-quadrant steps; a long flat streak
            # (≥ 6) means the low-gap detection window should be wider (+1) to
            # catch the sustained stagnation pattern; a very short flat streak
            # (≤ 1) means flat is transient — shrink the window (−1) to avoid
            # false alarms on isolated flat steps.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qfms_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qfms_c3m = sum(_coh3_steps[:_qfms_n])    / _qfms_n
                _qfms_vm  = sum(_velocity_steps[:_qfms_n]) / _qfms_n
                _qfms_streak = _qfms_best = 0
                for _i in range(_qfms_n):
                    if _coh3_steps[_i] <= _qfms_c3m and _velocity_steps[_i] < _qfms_vm:
                        _qfms_streak += 1
                        _qfms_best = max(_qfms_best, _qfms_streak)
                    else:
                        _qfms_streak = 0
                if _qfms_best >= 6:
                    _LOW_GAP_WIN = int(np.clip(_LOW_GAP_WIN + 1, 2, 8))
                elif _qfms_best <= 1:
                    _LOW_GAP_WIN = int(np.clip(_LOW_GAP_WIN - 1, 2, 8))
            # quad_confidence_spread-adaptive _MARGIN_EMA_A:
            # std-dev of the 4 per-quadrant confidence means; when the spread is
            # wide (≥ 0.12) the sampler clearly distinguishes good from bad states
            # — speed up the margin EMA (+0.04) to track those sharp differences;
            # when the spread is narrow (≤ 0.04) all quadrants look alike in
            # confidence space — slow the EMA (−0.03) to avoid chasing noise.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4 and len(output_confs) >= 4:
                _qcsp_n   = min(len(_coh3_steps), len(_velocity_steps), len(output_confs))
                _qcsp_c3m = sum(_coh3_steps[:_qcsp_n])    / _qcsp_n
                _qcsp_vm  = sum(_velocity_steps[:_qcsp_n]) / _qcsp_n
                _qcsp_pools = {
                    q: [output_confs[i] for i in range(_qcsp_n)
                        if ("ideal"     if _coh3_steps[i] > _qcsp_c3m and _velocity_steps[i] < _qcsp_vm else
                            "exploring" if _coh3_steps[i] > _qcsp_c3m else
                            "drifting"  if _velocity_steps[i] >= _qcsp_vm else "flat") == q]
                    for q in ["ideal", "exploring", "drifting", "flat"]
                }
                _qcsp_means = [sum(v)/len(v) for v in _qcsp_pools.values() if v]
                if len(_qcsp_means) >= 2:
                    _qcsp_mu  = sum(_qcsp_means) / len(_qcsp_means)
                    _qcsp_std = (sum((x - _qcsp_mu)**2 for x in _qcsp_means) / len(_qcsp_means)) ** 0.5
                    if _qcsp_std >= 0.12:
                        _MARGIN_EMA_A = float(np.clip(_MARGIN_EMA_A + 0.04, 0.05, 0.60))
                    elif _qcsp_std <= 0.04:
                        _MARGIN_EMA_A = float(np.clip(_MARGIN_EMA_A - 0.03, 0.05, 0.60))
            # quad_coh3_spread-adaptive _SG_EMA_ALPHA:
            # std-dev of the 4 per-quadrant coh3 means; wide coh3 spread (≥ 0.10)
            # means the score-gap EMA should be faster (+0.04) to track those
            # peaks; narrow spread (≤ 0.03) means coh3 barely distinguishes
            # quadrants — slow the score-gap EMA (−0.03) to stabilise.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qc3sp_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qc3sp_c3m = sum(_coh3_steps[:_qc3sp_n])    / _qc3sp_n
                _qc3sp_vm  = sum(_velocity_steps[:_qc3sp_n]) / _qc3sp_n
                _qc3sp_pools = {
                    q: [_coh3_steps[i] for i in range(_qc3sp_n)
                        if ("ideal"     if _coh3_steps[i] > _qc3sp_c3m and _velocity_steps[i] < _qc3sp_vm else
                            "exploring" if _coh3_steps[i] > _qc3sp_c3m else
                            "drifting"  if _velocity_steps[i] >= _qc3sp_vm else "flat") == q]
                    for q in ["ideal", "exploring", "drifting", "flat"]
                }
                _qc3sp_means = [sum(v)/len(v) for v in _qc3sp_pools.values() if v]
                if len(_qc3sp_means) >= 2:
                    _qc3sp_mu  = sum(_qc3sp_means) / len(_qc3sp_means)
                    _qc3sp_std = (sum((x - _qc3sp_mu)**2 for x in _qc3sp_means) / len(_qc3sp_means)) ** 0.5
                    if _qc3sp_std >= 0.10:
                        _SG_EMA_ALPHA = float(np.clip(_SG_EMA_ALPHA + 0.04, 0.05, 0.60))
                    elif _qc3sp_std <= 0.03:
                        _SG_EMA_ALPHA = float(np.clip(_SG_EMA_ALPHA - 0.03, 0.05, 0.60))
            # quad_velocity_spread-adaptive _CTX_OSC_DAMP:
            # std-dev of the 4 per-quadrant velocity means; when velocity varies
            # widely across quadrants (≥ 0.08) the context oscillation damping
            # should be stronger (+0.04) to prevent over-correction; when all
            # quadrants move at similar speed (≤ 0.02) ease the damping (−0.03)
            # to let the context explore more freely.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qvsp_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qvsp_c3m = sum(_coh3_steps[:_qvsp_n])    / _qvsp_n
                _qvsp_vm  = sum(_velocity_steps[:_qvsp_n]) / _qvsp_n
                _qvsp_pools = {
                    q: [_velocity_steps[i] for i in range(_qvsp_n)
                        if ("ideal"     if _coh3_steps[i] > _qvsp_c3m and _velocity_steps[i] < _qvsp_vm else
                            "exploring" if _coh3_steps[i] > _qvsp_c3m else
                            "drifting"  if _velocity_steps[i] >= _qvsp_vm else "flat") == q]
                    for q in ["ideal", "exploring", "drifting", "flat"]
                }
                _qvsp_means = [sum(v)/len(v) for v in _qvsp_pools.values() if v]
                if len(_qvsp_means) >= 2:
                    _qvsp_mu  = sum(_qvsp_means) / len(_qvsp_means)
                    _qvsp_std = (sum((x - _qvsp_mu)**2 for x in _qvsp_means) / len(_qvsp_means)) ** 0.5
                    if _qvsp_std >= 0.08:
                        _CTX_OSC_DAMP = float(np.clip(_CTX_OSC_DAMP + 0.04, 0.05, 0.70))
                    elif _qvsp_std <= 0.02:
                        _CTX_OSC_DAMP = float(np.clip(_CTX_OSC_DAMP - 0.03, 0.05, 0.70))
            # quad_coh3_ideal_vs_flat_ratio-adaptive _RW_CONF_ALPHA:
            # ratio ideal_coh3_mean / flat_coh3_mean; when the ratio is high
            # (≥ 1.30) ideal steps are clearly more coherent — increase the
            # reward-weighted conf EMA rate (+0.04) to track those peaks faster;
            # when the ratio is near 1 (≤ 1.05) ideal and flat are
            # indistinguishable in coh3 — slow the EMA (−0.03) to stabilise.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qcifr_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qcifr_c3m = sum(_coh3_steps[:_qcifr_n])    / _qcifr_n
                _qcifr_vm  = sum(_velocity_steps[:_qcifr_n]) / _qcifr_n
                _qcifr_ideal = [_coh3_steps[i] for i in range(_qcifr_n)
                                 if _coh3_steps[i] > _qcifr_c3m and _velocity_steps[i] < _qcifr_vm]
                _qcifr_flat  = [_coh3_steps[i] for i in range(_qcifr_n)
                                 if _coh3_steps[i] <= _qcifr_c3m and _velocity_steps[i] < _qcifr_vm]
                if _qcifr_ideal and _qcifr_flat:
                    _qcifr_ratio = (sum(_qcifr_ideal)/len(_qcifr_ideal)) / max(sum(_qcifr_flat)/len(_qcifr_flat), 1e-6)
                    if _qcifr_ratio >= 1.30:
                        _RW_CONF_ALPHA = float(np.clip(_RW_CONF_ALPHA + 0.04, 0.05, 0.60))
                    elif _qcifr_ratio <= 1.05:
                        _RW_CONF_ALPHA = float(np.clip(_RW_CONF_ALPHA - 0.03, 0.05, 0.60))
            # quad_velocity_ideal_vs_drifting_ratio-adaptive _CTX_OSC_THR:
            # ratio ideal_velocity_mean / drifting_velocity_mean; if ideal steps
            # are much slower than drifting (ratio ≤ 0.55) the oscillation
            # detector threshold should be less negative (closer to 0 → +0.015)
            # to catch reversals sooner; if ideal is nearly as fast as drifting
            # (ratio ≥ 0.85) loosen the threshold (more negative → −0.015) to
            # avoid false positives.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qvidr_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qvidr_c3m = sum(_coh3_steps[:_qvidr_n])    / _qvidr_n
                _qvidr_vm  = sum(_velocity_steps[:_qvidr_n]) / _qvidr_n
                _qvidr_iv = [_velocity_steps[i] for i in range(_qvidr_n)
                              if _coh3_steps[i] > _qvidr_c3m and _velocity_steps[i] < _qvidr_vm]
                _qvidr_dv = [_velocity_steps[i] for i in range(_qvidr_n)
                              if _coh3_steps[i] <= _qvidr_c3m and _velocity_steps[i] >= _qvidr_vm]
                if _qvidr_iv and _qvidr_dv:
                    _qvidr_ratio = (sum(_qvidr_iv)/len(_qvidr_iv)) / max(sum(_qvidr_dv)/len(_qvidr_dv), 1e-6)
                    if _qvidr_ratio <= 0.55:
                        _CTX_OSC_THR = float(np.clip(_CTX_OSC_THR + 0.015, -0.30, -0.01))
                    elif _qvidr_ratio >= 0.85:
                        _CTX_OSC_THR = float(np.clip(_CTX_OSC_THR - 0.015, -0.30, -0.01))
            # quad_flat_run_confidence_mean-adaptive _REP_MOM_S:
            # mean confidence during flat (stagnant) steps; if flat steps are
            # paradoxically high-confidence (≥ 0.55) the model is confidently
            # repeating itself — raise repulsion-momentum penalty (+0.02) to push
            # it away from that attractor; if flat confidence is very low (≤ 0.25)
            # the model is already struggling — ease repulsion (−0.01) so it doesn't
            # punish the few candidates it has.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4 and len(output_confs) >= 4:
                _qfrc_n   = min(len(_coh3_steps), len(_velocity_steps), len(output_confs))
                _qfrc_c3m = sum(_coh3_steps[:_qfrc_n])    / _qfrc_n
                _qfrc_vm  = sum(_velocity_steps[:_qfrc_n]) / _qfrc_n
                _qfrc_flat_confs = [
                    output_confs[i]
                    for i in range(_qfrc_n)
                    if _coh3_steps[i] <= _qfrc_c3m and _velocity_steps[i] < _qfrc_vm
                ]
                if _qfrc_flat_confs:
                    _qfrc_mean = sum(_qfrc_flat_confs) / len(_qfrc_flat_confs)
                    if _qfrc_mean >= 0.55:
                        _REP_MOM_S = float(np.clip(_REP_MOM_S + 0.02, 0.05, 0.60))
                    elif _qfrc_mean <= 0.25:
                        _REP_MOM_S = float(np.clip(_REP_MOM_S - 0.01, 0.05, 0.60))
            # quad_transition_matrix_skew-adaptive _LOW_MARGIN_THR:
            # when transitions are highly skewed (≥ 0.65) the model is trapped in
            # one pathway — raise the low-margin threshold (+0.005) to demand more
            # decisive score gaps, forcing the sampler to look further; when skew
            # is near-uniform (≤ 0.20) lower the threshold (−0.003) to allow tighter
            # margins and preserve the natural diversity already present.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qtmsa_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qtmsa_c3m = sum(_coh3_steps[:_qtmsa_n])    / _qtmsa_n
                _qtmsa_vm  = sum(_velocity_steps[:_qtmsa_n]) / _qtmsa_n
                _qtmsa_labs = [
                    ("ideal" if _coh3_steps[i] > _qtmsa_c3m and _velocity_steps[i] < _qtmsa_vm
                     else "exploring" if _coh3_steps[i] > _qtmsa_c3m
                     else "drifting" if _velocity_steps[i] >= _qtmsa_vm
                     else "flat")
                    for i in range(_qtmsa_n)
                ]
                _qtmsa_src = ["ideal", "exploring", "drifting", "flat"]
                _qtmsa_rows = [
                    [sum(1 for i in range(1, _qtmsa_n)
                         if _qtmsa_labs[i-1] == src and _qtmsa_labs[i] == dst)
                     for dst in _qtmsa_src]
                    for src in _qtmsa_src
                ]
                _qtmsa_skew_vals = [
                    max(row) / max(sum(row), 1)
                    for row in _qtmsa_rows if sum(row) > 0
                ]
                if _qtmsa_skew_vals:
                    _qtmsa_skew = max(_qtmsa_skew_vals) - min([
                        min(row) / max(sum(row), 1)
                        for row in _qtmsa_rows if sum(row) > 0
                    ])
                    if _qtmsa_skew >= 0.65:
                        _LOW_MARGIN_THR = float(np.clip(_LOW_MARGIN_THR + 0.005, 0.005, 0.08))
                    elif _qtmsa_skew <= 0.20:
                        _LOW_MARGIN_THR = float(np.clip(_LOW_MARGIN_THR - 0.003, 0.005, 0.08))
            # quad_ideal_run_confidence_mean-adaptive _CTX_MOM_S:
            # mean confidence during ideal runs tells us how robustly the model
            # sustains quality flow; high confidence (≥ 0.60) means the ideal
            # state is stable — boost context-momentum contribution (+0.015) so
            # the trajectory stays on that track; low confidence (≤ 0.30) means
            # ideal runs are fragile — ease momentum (−0.01) to let the context
            # update more freely and find a better path.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4 and len(output_confs) >= 4:
                _qirc_n   = min(len(_coh3_steps), len(_velocity_steps), len(output_confs))
                _qirc_c3m = sum(_coh3_steps[:_qirc_n])    / _qirc_n
                _qirc_vm  = sum(_velocity_steps[:_qirc_n]) / _qirc_n
                _qirc_ideal_confs = [
                    output_confs[i]
                    for i in range(_qirc_n)
                    if _coh3_steps[i] > _qirc_c3m and _velocity_steps[i] < _qirc_vm
                ]
                if _qirc_ideal_confs:
                    _qirc_mean = sum(_qirc_ideal_confs) / len(_qirc_ideal_confs)
                    if _qirc_mean >= 0.60:
                        _CTX_MOM_S = float(np.clip(_CTX_MOM_S + 0.015, 0.04, 0.28))
                    elif _qirc_mean <= 0.30:
                        _CTX_MOM_S = float(np.clip(_CTX_MOM_S - 0.01, 0.04, 0.28))
            # quad_self_transition_rate-adaptive _SEM_CENT_STR:
            # when the model rarely changes quadrant (self-loop rate ≥ 0.55) it is
            # stuck in one state — increase semantic centroid pull strength (+0.02)
            # to tug the context toward the centroid and unstick it; when transitions
            # are frequent (self-loop rate ≤ 0.25), the model is already mobile —
            # ease the pull (−0.015) to avoid disrupting natural flow.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qstr_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qstr_c3m = sum(_coh3_steps[:_qstr_n])    / _qstr_n
                _qstr_vm  = sum(_velocity_steps[:_qstr_n]) / _qstr_n
                def _qstr_lab(ci, vi, c3m, vm):
                    if ci > c3m and vi < vm: return "ideal"
                    if ci > c3m:             return "exploring"
                    if vi >= vm:             return "drifting"
                    return "flat"
                _qstr_self = sum(
                    1 for i in range(1, _qstr_n)
                    if _qstr_lab(_coh3_steps[i-1], _velocity_steps[i-1], _qstr_c3m, _qstr_vm)
                    == _qstr_lab(_coh3_steps[i],   _velocity_steps[i],   _qstr_c3m, _qstr_vm)
                )
                _qstr_rate = _qstr_self / max(_qstr_n - 1, 1)
                if _qstr_rate >= 0.55:
                    _SEM_CENT_STR = float(np.clip(_SEM_CENT_STR + 0.02, 0.02, 0.20))
                elif _qstr_rate <= 0.25:
                    _SEM_CENT_STR = float(np.clip(_SEM_CENT_STR - 0.015, 0.02, 0.20))
            # quad_transition_entropy-adaptive _BURST_TEMP:
            # transition entropy measures how evenly the model spreads transitions
            # across all 12 quadrant pairs; high entropy (≥ 1.8 bits) means
            # transitions are unpredictable and bursty — raise _BURST_TEMP (+0.02)
            # to sharpen the burst-clear response; low entropy (≤ 0.8 bits) means
            # the model follows one or two dominant paths — lower _BURST_TEMP
            # (−0.02) since bursts are less likely to be random noise.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qte_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qte_c3m = sum(_coh3_steps[:_qte_n])    / _qte_n
                _qte_vm  = sum(_velocity_steps[:_qte_n]) / _qte_n
                def _qte_label(ci, vi, c3m, vm):
                    if ci > c3m and vi < vm:   return "ideal"
                    if ci > c3m:               return "exploring"
                    if vi >= vm:               return "drifting"
                    return "flat"
                _qte_pairs = {}
                for i in range(1, _qte_n):
                    _qte_k = (_qte_label(_coh3_steps[i-1], _velocity_steps[i-1], _qte_c3m, _qte_vm),
                              _qte_label(_coh3_steps[i],   _velocity_steps[i],   _qte_c3m, _qte_vm))
                    _qte_pairs[_qte_k] = _qte_pairs.get(_qte_k, 0) + 1
                _qte_total = sum(_qte_pairs.values())
                if _qte_total > 0:
                    _qte_entropy = -sum(
                        (v / _qte_total) * __import__('math').log2(v / _qte_total)
                        for v in _qte_pairs.values() if v > 0
                    )
                    if _qte_entropy >= 1.8:
                        _BURST_TEMP = float(np.clip(_BURST_TEMP + 0.02, 0.08, 0.45))
                    elif _qte_entropy <= 0.8:
                        _BURST_TEMP = float(np.clip(_BURST_TEMP - 0.02, 0.08, 0.45))
            # quad_exploring_to_drifting_rate-adaptive _VPREC_EMA_ALPHA:
            # when exploring often tips into drifting (rate ≥ 0.40) the model's
            # vocab precision is unreliable after divergence — speed up the VPREC
            # EMA (+0.04) so it tracks the deterioration faster; when exploring
            # rarely drifts (rate ≤ 0.12), precision holds — slow the EMA (−0.03)
            # for smoother tracking.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qetd_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qetd_c3m = sum(_coh3_steps[:_qetd_n])    / _qetd_n
                _qetd_vm  = sum(_velocity_steps[:_qetd_n]) / _qetd_n
                _qetd_to_drift = [
                    1 for i in range(1, _qetd_n)
                    if (_coh3_steps[i-1] > _qetd_c3m and _velocity_steps[i-1] >= _qetd_vm)
                    and (_coh3_steps[i] <= _qetd_c3m and _velocity_steps[i] >= _qetd_vm)
                ]
                _qetd_all_exits = [
                    1 for i in range(1, _qetd_n)
                    if (_coh3_steps[i-1] > _qetd_c3m and _velocity_steps[i-1] >= _qetd_vm)
                    and not (_coh3_steps[i] > _qetd_c3m and _velocity_steps[i] >= _qetd_vm)
                ]
                if _qetd_all_exits:
                    _qetd_rate = len(_qetd_to_drift) / len(_qetd_all_exits)
                    if _qetd_rate >= 0.40:
                        _VPREC_EMA_ALPHA = float(np.clip(_VPREC_EMA_ALPHA + 0.04, 0.08, 0.50))
                    elif _qetd_rate <= 0.12:
                        _VPREC_EMA_ALPHA = float(np.clip(_VPREC_EMA_ALPHA - 0.03, 0.08, 0.50))
            # quad_drifting_to_ideal_rate-adaptive _CONF_EMA_ALPHA:
            # when drift often resolves directly to ideal (rate ≥ 0.30), the model
            # is self-correcting quickly — slow the EMA (−0.03) so it smooths over
            # transient noise and stays near the quality level; when drift rarely
            # reaches ideal (rate ≤ 0.08), the model is stuck — speed up the EMA
            # (+0.03) so confidence tracking responds faster to any improvement.
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qdti_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qdti_c3m = sum(_coh3_steps[:_qdti_n])    / _qdti_n
                _qdti_vm  = sum(_velocity_steps[:_qdti_n]) / _qdti_n
                _qdti_to_ideal = [
                    1 for i in range(1, _qdti_n)
                    if (_coh3_steps[i-1] <= _qdti_c3m and _velocity_steps[i-1] >= _qdti_vm)
                    and (_coh3_steps[i] > _qdti_c3m and _velocity_steps[i] < _qdti_vm)
                ]
                _qdti_all_exits = [
                    1 for i in range(1, _qdti_n)
                    if (_coh3_steps[i-1] <= _qdti_c3m and _velocity_steps[i-1] >= _qdti_vm)
                    and not (_coh3_steps[i] <= _qdti_c3m and _velocity_steps[i] >= _qdti_vm)
                ]
                if _qdti_all_exits:
                    _qdti_rate = len(_qdti_to_ideal) / len(_qdti_all_exits)
                    if _qdti_rate >= 0.30:
                        _CONF_EMA_ALPHA = float(np.clip(_CONF_EMA_ALPHA - 0.03, 0.15, 0.60))
                    elif _qdti_rate <= 0.08:
                        _CONF_EMA_ALPHA = float(np.clip(_CONF_EMA_ALPHA + 0.03, 0.15, 0.60))
            # quad_flat_to_ideal_rate-adaptive _LOW_GAP_THR:
            # when flat steps often jump directly into ideal (rate ≥ 0.35) the
            # model has a strong recovery signal even from stagnation — raise the
            # low-gap threshold (+0.005) so weaker score gaps also trigger the
            # exploration boost; when flat rarely reaches ideal (rate ≤ 0.10) the
            # model needs a genuine weak signal — lower the threshold (−0.005).
            if len(_coh3_steps) >= 4 and len(_velocity_steps) >= 4:
                _qfti_n   = min(len(_coh3_steps), len(_velocity_steps))
                _qfti_c3m = sum(_coh3_steps[:_qfti_n])    / _qfti_n
                _qfti_vm  = sum(_velocity_steps[:_qfti_n]) / _qfti_n
                _qfti_to_ideal = [
                    1 for i in range(1, _qfti_n)
                    if (_coh3_steps[i-1] <= _qfti_c3m and _velocity_steps[i-1] < _qfti_vm)
                    and (_coh3_steps[i] > _qfti_c3m and _velocity_steps[i] < _qfti_vm)
                ]
                _qfti_all_exits = [
                    1 for i in range(1, _qfti_n)
                    if (_coh3_steps[i-1] <= _qfti_c3m and _velocity_steps[i-1] < _qfti_vm)
                    and not (_coh3_steps[i] <= _qfti_c3m and _velocity_steps[i] < _qfti_vm)
                ]
                if _qfti_all_exits:
                    _qfti_rate = len(_qfti_to_ideal) / len(_qfti_all_exits)
                    if _qfti_rate >= 0.35:
                        _LOW_GAP_THR = float(np.clip(_LOW_GAP_THR + 0.005, 0.01, 0.08))
                    elif _qfti_rate <= 0.10:
                        _LOW_GAP_THR = float(np.clip(_LOW_GAP_THR - 0.005, 0.01, 0.08))
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

            # conf_ema_delta adaptive temperature: at step 6+ compare current
            # conf_ema to step-0 value.  Strong gain (>0.05) → cool temp by
            # 0.03 (exploit the good trajectory); strong decline (<−0.05) →
            # warm temp by 0.05 (push exploration to recover).
            if _step >= 6 and len(_conf_ema_steps) >= 2:
                _ced_now = _conf_ema_steps[-1] - _conf_ema_steps[0]
                if _ced_now > 0.05:
                    temp = float(max(temp - 0.03, _temp_lb))
                elif _ced_now < -0.05:
                    temp = float(min(temp + 0.05, 0.90))

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
                _m_best_idx = int(np.argmax(scores))
                _m_win = float(scores[_m_best_idx]) if scores[_m_best_idx] > -1e9 else float(_m_fin.max())
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
            _sfb_strength_steps.append(round(float(_sfb._strength), 5))  # SFB log

            # Per-step conf_ema log (for confrise CLI)
            _conf_ema_steps.append(round(_conf_ema, 5))

            # Per-step vprec_ema log (for vocabjump CLI)
            _vprec_ema_steps.append(round(_vprec_ema, 5))

            # Adaptive SFB decay from coh3: high coherence → slow decay (stay on
            # topic); low coherence → faster decay (allow semantic drift).
            if len(prev_vecs) >= 3:
                _sfb_coh_now = (float(prev_vecs[-3] @ prev_vecs[-2]) +
                                float(prev_vecs[-2] @ prev_vecs[-1])) / 2.0
                if _sfb_coh_now >= 0.45:
                    _sfb._decay = max(0.96, _sfb._decay)   # slow decay when coherent
                elif _sfb_coh_now < 0.20:
                    _sfb._decay = min(0.90, _sfb._decay)   # faster when drifting

            # Adaptive SFB decay from coh_trend: if coherence has been falling
            # across the generation so far (negative running slope), accelerate
            # SFB decay to loosen the semantic anchor and allow recovery.
            if len(_coh3_steps) >= 4:
                _sfb_trend_x = np.arange(len(_coh3_steps), dtype=np.float32)
                _sfb_trend_y = np.array(_coh3_steps, dtype=np.float32)
                _sfb_live_trend = float(np.polyfit(_sfb_trend_x, _sfb_trend_y, 1)[0])
                if _sfb_live_trend < -0.02:
                    # Coherence is declining — speed decay to break the attractor
                    _sfb._decay = min(float(_sfb._decay), 0.91)

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

        # Best / worst coherence segment: find the 3-step window with the
        # highest and lowest mean coh3.  Useful for pinpointing where the
        # generation peaked and where it struggled most.
        bestseg_start, bestseg_val  = 0, 0.0
        worstseg_start, worstseg_val = 0, 0.0
        _SEG_W = 3
        if len(_coh3_steps) >= _SEG_W:
            _seg_best_v, _seg_worst_v = -1e9, 1e9
            for _si in range(len(_coh3_steps) - _SEG_W + 1):
                _sv = float(np.mean(_coh3_steps[_si: _si + _SEG_W]))
                if _sv > _seg_best_v:
                    _seg_best_v  = _sv;  bestseg_start = _si;  bestseg_val = _sv
                if _sv < _seg_worst_v:
                    _seg_worst_v = _sv; worstseg_start = _si; worstseg_val = _sv

        # Flow score: fraction of steps where both conf_ema AND coh3 improved
        # simultaneously (both non-decreasing vs prior step).  Perfect flow = 1.0.
        flow_score = 0.0
        if len(_conf_ema_steps) >= 2 and len(_coh3_steps) >= 2:
            _fl_n = min(len(_conf_ema_steps), len(_coh3_steps)) - 1
            _fl_both = sum(
                1 for _i in range(_fl_n)
                if (_conf_ema_steps[_i + 1] >= _conf_ema_steps[_i]
                    and _coh3_steps[_i + 1] >= _coh3_steps[_i])
            )
            flow_score = round(_fl_both / _fl_n, 4) if _fl_n > 0 else 0.0

        # Per-step flow indicators for flowbar CLI:
        # 'B'=both rising, 'C'=only conf, 'H'=only coh3, 'N'=neither
        _flow_steps: list = []
        if len(_conf_ema_steps) >= 2 and len(_coh3_steps) >= 2:
            for _fi in range(min(len(_conf_ema_steps), len(_coh3_steps)) - 1):
                _fc = _conf_ema_steps[_fi + 1] >= _conf_ema_steps[_fi]
                _fh = _coh3_steps[_fi + 1]     >= _coh3_steps[_fi]
                _flow_steps.append("B" if _fc and _fh else
                                   "C" if _fc else
                                   "H" if _fh else "N")

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
            "entropy_ratio": round(
                (max(_entropy_steps) / (_ema_entropy + 1e-9))
                if _entropy_steps and _ema_entropy > 1e-9 else 1.0, 4
            ),   # max_entropy / avg_entropy; spikiness measure (>2 = one step much noisier)
            "entropy_acc": [
                round(float(
                    (_entropy_steps[i] - _entropy_steps[i-1]) -
                    (_entropy_steps[i-1] - _entropy_steps[i-2])
                ), 6) if i >= 2 else 0.0
                for i in range(len(_entropy_steps))
            ],   # per-step entropy 2nd derivative (acceleration of entropy change)
            "conf_acc": [
                round(float(
                    (_conf_ema_steps[i] - _conf_ema_steps[i-1]) -
                    (_conf_ema_steps[i-1] - _conf_ema_steps[i-2])
                ), 6) if i >= 2 else 0.0
                for i in range(len(_conf_ema_steps))
            ],   # per-step confidence 2nd derivative (acceleration)
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
            "peak_conf_val":  round(_peak_conf_val, 4),  # value of highest confidence
            "min_conf_step":  int(np.argmin(output_confs)) if output_confs else 0,
            "min_conf_val":   round(float(min(output_confs)), 4) if output_confs else 0.0,
            "sg_step_gaps":    _sg_step_gaps,     # per-step score gaps (for sgplot)
            "conf_ema_steps":   _conf_ema_steps,  # per-step smoothed confidence EMA
            "vprec_ema_steps":  _vprec_ema_steps, # per-step vocab-precision EMA
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
            "vel_ratio": round(
                (max(_velocity_steps) / (sum(_velocity_steps) / max(len(_velocity_steps), 1) + 1e-9))
                if _velocity_steps else 1.0, 4
            ),   # max_velocity / mean_velocity; spikiness of context shifts
            "sfb_strength_steps": _sfb_strength_steps,  # per-step SFB strength
            "percentile_steps":   _percentile_steps,    # per-step score percentile
            "temp_steps":         _temp_steps,          # per-step temperature value
            "confslope_steps": [
                round(float(_conf_ema_steps[i] - _conf_ema_steps[i-1]), 6) if i >= 1 else 0.0
                for i in range(len(_conf_ema_steps))
            ],   # per-step 1st derivative of conf_ema (instantaneous slope)
            "coh_acc": [
                round(float(
                    (_coh3_steps[i] - _coh3_steps[i-1]) -
                    (_coh3_steps[i-1] - _coh3_steps[i-2])
                ), 6) if i >= 2 else 0.0
                for i in range(len(_coh3_steps))
            ],   # per-step coherence 2nd derivative (acceleration of coh3 change)
            "vocab_prec_ema":     round(_vprec_ema, 4), # vocab-precision EMA at end
            "score_var_steps":    _score_var_steps,     # per-step score variance
            "scorevar": round(
                float(np.mean(_score_var_steps)) if _score_var_steps else 0.0, 5
            ),   # mean within-step score variance across generation
            "perc_trend": round(
                float(np.polyfit(np.arange(len(_percentile_steps), dtype=np.float32),
                                 np.array(_percentile_steps, dtype=np.float32), 1)[0])
                if len(_percentile_steps) >= 3 else 0.0, 5
            ),   # linear slope of score-percentile of chosen tokens (rising = better picks)
            "sfb_trend": round(
                float(np.polyfit(np.arange(len(_sfb_strength_steps), dtype=np.float32),
                                 np.array(_sfb_strength_steps, dtype=np.float32), 1)[0])
                if len(_sfb_strength_steps) >= 3 else 0.0, 5
            ),   # linear slope of SFB strength (rising = anchor growing over gen)
            "perfindex": round(float(np.clip(
                # composite: conf_ema × (1+coh3) × fluency × (1/pseudo_ppl × vocab_size)
                # normalised to 0..1 range (approx)
                min(1.0,
                    (np.mean(output_confs) if output_confs else 0.5) *
                    (1.0 + max(coh3, 0)) *
                    max(fluency, 0.1) *
                    (1.0 - min(flow_score if flow_score else 0.0, 1.0) * 0.0 + flow_score * 0.2)
                ), 0.0, 1.0)), 4
            ),   # single composite quality index ∈ [0, 1]
            "conf_range": round(
                float(max(output_confs) - min(output_confs)) if len(output_confs) >= 2 else 0.0, 5
            ),   # max−min confidence across generation (spread)
            "coh_range": round(
                float(max(_coh3_steps) - min(_coh3_steps)) if len(_coh3_steps) >= 2 else 0.0, 5
            ),   # max−min coh3 across generation (coherence spread)
            "tokenlen_steps": [len(t) for t in output_tokens],  # per-step token char length
            "wintok": (
                max(set(output_tokens), key=output_tokens.count)
                if output_tokens else ""
            ),   # most-repeated token in the generation
            "wintok_count": (
                output_tokens.count(max(set(output_tokens), key=output_tokens.count))
                if output_tokens else 0
            ),   # count of most-repeated token
            "uniq_ratio": round(
                len(set(output_tokens)) / max(len(output_tokens), 1), 4
            ),   # unique token types / total tokens (lexical uniqueness ∈ [0,1])
            "conf_ema_delta": round(
                float(_conf_ema_steps[-1] - _conf_ema_steps[0])
                if len(_conf_ema_steps) >= 2 else 0.0, 5
            ),   # total conf_ema change: last − first step (positive = improving)
            "uniq_steps": [
                round(len(set(output_tokens[:i+1])) / max(i+1, 1), 4)
                for i in range(len(output_tokens))
            ],   # per-step running unique-token ratio
            "conf_ema_mid": round(
                float(_conf_ema_steps[len(_conf_ema_steps) // 2])
                if _conf_ema_steps else 0.0, 5
            ),   # conf_ema at the midpoint step (early vs late comparisons)
            "entdelta": round(
                float(_entropy_steps[-1] - _entropy_steps[0])
                if len(_entropy_steps) >= 2 else 0.0, 5
            ),   # total entropy change: last − first step (negative = focused)
            "coh_vel_correlation": round(float(
                np.corrcoef(
                    np.array(_coh3_steps[:min(len(_coh3_steps),len(_velocity_steps))], dtype=np.float32),
                    np.array(_velocity_steps[:min(len(_coh3_steps),len(_velocity_steps))], dtype=np.float32)
                )[0, 1]
            ) if min(len(_coh3_steps), len(_velocity_steps)) >= 3 else 0.0, 4),
            # Pearson r between coh3 and velocity (positive = they co-move)
            "conf_drop_steps": [
                {"step": i, "tok": output_tokens[i] if i < len(output_tokens) else "?",
                 "drop": round(float(output_confs[i-1] - output_confs[i]), 5)}
                for i in range(1, len(output_confs))
                if output_confs[i-1] - output_confs[i] >= 0.05
            ],   # steps where raw confidence dropped ≥0.05
            "conf_rise_steps": [
                {"step": i, "tok": output_tokens[i] if i < len(output_tokens) else "?",
                 "rise": round(float(output_confs[i] - output_confs[i-1]), 5)}
                for i in range(1, len(output_confs))
                if output_confs[i] - output_confs[i-1] >= 0.05
            ],   # steps where raw confidence rose ≥0.05 (positive jumps)
            "topk_mode": (
                max(set(_topk_steps), key=_topk_steps.count)
                if _topk_steps else 0
            ),   # most frequently used TopK value across all steps
            "score_gap_trend": [
                round(float(np.mean(_sg_step_gaps[max(0, i-3):i+1])), 5)
                if _sg_step_gaps else 0.0
                for i in range(len(_sg_step_gaps))
            ] if (hasattr(np, 'mean') and _sg_step_gaps) else [],
            # per-step 4-step rolling mean of top1−top2 score gap (confidence in margin)
            "sfb_acc": round(float(
                np.polyfit(range(len(_sfb_steps)), _sfb_steps, 1)[0]
                if len(_sfb_steps) >= 4 else 0.0
            ), 6),  # linear slope of SFB per step (positive = SFB rising = phrase variety improving)
            "margin_trend": round(float(
                np.polyfit(range(len(_top1_margins)), _top1_margins, 1)[0]
                if len(_top1_margins) >= 4 else 0.0
            ), 6),  # linear slope of top1-top2 margin (positive = model getting more decisive)
            "conf_bucket_hist": {
                f"{lo:.1f}-{hi:.1f}": sum(
                    1 for c in output_confs if lo <= c < hi
                )
                for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6),
                               (0.6, 0.8), (0.8, 1.01)]
            },  # 5-bucket confidence histogram for distribution shape
            "coh_drop_steps": [
                {"step": i, "tok": output_tokens[i] if i < len(output_tokens) else "?",
                 "drop": round(float(_coh3_steps[i-1] - _coh3_steps[i]), 5)}
                for i in range(1, len(_coh3_steps))
                if _coh3_steps[i-1] - _coh3_steps[i] >= 0.05
            ],   # steps where coh3 dropped ≥0.05 (coherence collapses)
            "entropy_trend": round(float(
                np.polyfit(range(len(_entropy_steps)), _entropy_steps, 1)[0]
                if len(_entropy_steps) >= 4 else 0.0
            ), 6),  # linear slope of entropy over time (neg = distribution tightening)
            "coh_rise_steps": [
                {"step": i, "tok": output_tokens[i] if i < len(output_tokens) else "?",
                 "rise": round(float(_coh3_steps[i] - _coh3_steps[i-1]), 5)}
                for i in range(1, len(_coh3_steps))
                if _coh3_steps[i] - _coh3_steps[i-1] >= 0.05
            ],   # steps where coh3 jumped ≥0.05 (coherence surges)
            "vel_trend": round(float(
                np.polyfit(range(len(_velocity_steps)), _velocity_steps, 1)[0]
                if len(_velocity_steps) >= 4 else 0.0
            ), 6),  # linear slope of velocity_steps (positive = context drifting faster)
            "topk_entropy_corr": round(float(
                np.corrcoef(
                    np.array(_topk_steps[:min(len(_topk_steps), len(_entropy_steps))], dtype=np.float32),
                    np.array(_entropy_steps[:min(len(_topk_steps), len(_entropy_steps))], dtype=np.float32)
                )[0, 1]
            ) if min(len(_topk_steps), len(_entropy_steps)) >= 3 else 0.0, 4),
            # Pearson r between TopK k and entropy (positive = wider k → more entropy)
            "coh_conf_corr": round(float(
                np.corrcoef(
                    np.array(_coh3_steps[:min(len(_coh3_steps), len(output_confs))], dtype=np.float32),
                    np.array(output_confs[:min(len(_coh3_steps), len(output_confs))], dtype=np.float32)
                )[0, 1]
            ) if min(len(_coh3_steps), len(output_confs)) >= 3 else 0.0, 4),
            # Pearson r between coh3 and confidence (positive = coherent tokens are also confident)
            "vel_conf_corr": round(float(
                np.corrcoef(
                    np.array(_velocity_steps[:min(len(_velocity_steps), len(output_confs))], dtype=np.float32),
                    np.array(output_confs[:min(len(_velocity_steps), len(output_confs))], dtype=np.float32)
                )[0, 1]
            ) if min(len(_velocity_steps), len(output_confs)) >= 3 else 0.0, 4),
            # Pearson r between velocity and confidence (neg = faster drift → lower conf, typical)
            "score_var_trend": round(float(
                np.polyfit(range(len(_score_var_steps)), _score_var_steps, 1)[0]
                if len(_score_var_steps) >= 4 else 0.0
            ), 6),  # linear slope of per-step score variance (+ = scores spreading out)
            "rhythm_trend": round(float(
                np.polyfit(range(len(_rhythm_rate_steps)), _rhythm_rate_steps, 1)[0]
                if len(_rhythm_rate_steps) >= 4 else 0.0
            ), 6),  # linear slope of running rhythm rate (+ = becoming smoother)
            "rhythm_rate_steps": _rhythm_rate_steps,
            # per-step running rhythm rate (1 = smooth, 0 = fully alternating)
            "margin_spike_steps": (lambda _ms=_top1_margins: (
                [i for i in range(len(_ms))
                 if len(_ms) >= 4 and _ms[i] > float(np.mean(_ms)) + 1.5 * float(np.std(_ms))]
            ) if len(_top1_margins) >= 4 else [])(),
            # steps where top1−top2 margin > mean+1.5σ (sudden decisive clarity bursts)
            "coh_var_steps": [
                round(float(np.var(_coh3_steps[:i+1])), 6)
                if i >= 1 else 0.0
                for i in range(len(_coh3_steps))
            ],  # per-step running variance of coh3 values (how stable coherence has been)
            "conf_var_steps": [
                round(float(np.var(output_confs[:i+1])), 6)
                if i >= 1 else 0.0
                for i in range(len(output_confs))
            ],  # per-step running variance of confidence values seen so far
            "coh_entropy_corr": round(float(
                np.corrcoef(
                    np.array(_coh3_steps[:min(len(_coh3_steps), len(_entropy_steps))], dtype=np.float32),
                    np.array(_entropy_steps[:min(len(_coh3_steps), len(_entropy_steps))], dtype=np.float32)
                )[0, 1]
            ) if min(len(_coh3_steps), len(_entropy_steps)) >= 3 else 0.0, 4),
            # Pearson r between coh3 and entropy (negative = coherent → focused, expected)
            "conf_entropy_corr": round(float(
                np.corrcoef(
                    np.array(output_confs[:min(len(output_confs), len(_entropy_steps))], dtype=np.float32),
                    np.array(_entropy_steps[:min(len(output_confs), len(_entropy_steps))], dtype=np.float32)
                )[0, 1]
            ) if min(len(output_confs), len(_entropy_steps)) >= 3 else 0.0, 4),
            # Pearson r between confidence and entropy (negative = high confidence → low entropy)
            "coh_slope_steps": [
                round(float(_coh3_steps[i] - _coh3_steps[i-1]), 6) if i >= 1 else 0.0
                for i in range(len(_coh3_steps))
            ],  # per-step 1st derivative of coh3 trajectory (instantaneous slope)
            "vprec_slope_steps": [
                round(float(_vprec_ema_steps[i] - _vprec_ema_steps[i-1]), 6) if i >= 1 else 0.0
                for i in range(len(_vprec_ema_steps))
            ],  # per-step 1st derivative of vprec_ema (vocab precision slope)
            "coh_conf_steps": [
                round(float(_coh3_steps[i] * output_confs[i]), 5)
                if i < len(output_confs) else 0.0
                for i in range(len(_coh3_steps))
            ],  # per-step coh3 × confidence quality product
            "quality_steps": (lambda _qs_raw=(
                [_coh3_steps[_qi] * output_confs[_qi]
                 if _qi < len(output_confs) else 0.0
                 for _qi in range(len(_coh3_steps))]
            ): [
                round(float(
                    sum(0.7 ** k * _qs_raw[max(0, _n - k)]
                        * (0.3 if k > 0 else 1.0)
                        for k in range(min(_n + 1, 6)))
                ), 5)
                for _n in range(len(_qs_raw))
            ])(),
            # per-step EMA-smoothed coh3×conf quality signal (0.7 decay)
            "conf_coh_slope_corr": round(float(
                np.corrcoef(
                    np.array([(_conf_ema_steps[i] - _conf_ema_steps[i-1]) if i >= 1 else 0.0
                               for i in range(len(_conf_ema_steps))], dtype=np.float32),
                    np.array([(_coh3_steps[i] - _coh3_steps[i-1]) if i >= 1 else 0.0
                               for i in range(len(_coh3_steps[:len(_conf_ema_steps)]))],
                              dtype=np.float32)
                )[0, 1]
            ) if min(len(_conf_ema_steps), len(_coh3_steps)) >= 3 else 0.0, 4),
            # Pearson r between conf_ema slope and coh3 slope (+ = both move together)
            "conf_vprec_corr": round(float(
                np.corrcoef(
                    np.array(output_confs[:min(len(output_confs), len(_vprec_ema_steps))], dtype=np.float32),
                    np.array(_vprec_ema_steps[:min(len(output_confs), len(_vprec_ema_steps))], dtype=np.float32)
                )[0, 1]
            ) if min(len(output_confs), len(_vprec_ema_steps)) >= 3 else 0.0, 4),
            # Pearson r between confidence and vprec_ema (+ = confident = precise vocab)
            "qual_spike_steps": (lambda _qss=(
                [_coh3_steps[_qi] * output_confs[_qi]
                 if _qi < len(output_confs) else 0.0
                 for _qi in range(len(_coh3_steps))]
            ): (
                [i for i in range(len(_qss))
                 if len(_qss) >= 4 and _qss[i] > float(np.mean(_qss)) + 1.5 * float(np.std(_qss))]
            ) if len(_coh3_steps) >= 4 else [])(),
            # steps where coh3×conf quality > mean+1.5σ (moments of peak quality)
            "entropy_var_steps": [
                round(float(np.var(_entropy_steps[:i+1])), 6)
                if i >= 1 else 0.0
                for i in range(len(_entropy_steps))
            ],  # per-step running variance of entropy values (how chaotic distribution has been)
            "sg_spike_steps": (lambda _sgs=_sg_step_gaps: (
                [i for i in range(len(_sgs))
                 if len(_sgs) >= 4 and _sgs[i] > float(np.mean(_sgs)) + 1.5 * float(np.std(_sgs))]
            ) if len(_sg_step_gaps) >= 4 else [])(),
            # steps where ScoreGapEMA > mean+1.5σ (sudden decisive-gap spikes)
            "coh6_var_steps": [
                round(float(np.var(_coh6_steps[:i+1])), 6)
                if i >= 1 else 0.0
                for i in range(len(_coh6_steps))
            ],  # per-step running variance of coh6 values (long-range coherence stability)
            "conf_topk_corr": round(float(
                np.corrcoef(
                    np.array(output_confs[:min(len(output_confs), len(_topk_steps))], dtype=np.float32),
                    np.array(_topk_steps[:min(len(output_confs), len(_topk_steps))],  dtype=np.float32)
                )[0, 1]
            ) if min(len(output_confs), len(_topk_steps)) >= 3 else 0.0, 4),
            # Pearson r between confidence and TopK k (negative = wide beam → lower conf, healthy)
            "qual_trend": round(float(
                np.polyfit(range(len(
                    [_coh3_steps[_qi] * output_confs[_qi]
                     if _qi < len(output_confs) else 0.0
                     for _qi in range(len(_coh3_steps))]
                )), [_coh3_steps[_qi] * output_confs[_qi]
                     if _qi < len(output_confs) else 0.0
                     for _qi in range(len(_coh3_steps))], 1)[0]
                if len(_coh3_steps) >= 4 else 0.0
            ), 6),  # linear slope of coh3×conf quality product (+ = quality improving)
            "vprec_conf_slope_corr": round(float(
                np.corrcoef(
                    np.array([(_vprec_ema_steps[i] - _vprec_ema_steps[i-1]) if i >= 1 else 0.0
                               for i in range(len(_vprec_ema_steps))], dtype=np.float32),
                    np.array([(_conf_ema_steps[i] - _conf_ema_steps[i-1]) if i >= 1 else 0.0
                               for i in range(len(_conf_ema_steps[:len(_vprec_ema_steps)]))],
                              dtype=np.float32)
                )[0, 1]
            ) if min(len(_vprec_ema_steps), len(_conf_ema_steps)) >= 3 else 0.0, 4),
            # Pearson r between vprec slope and conf_ema slope (+ = vocab precision and confidence rise together)
            "qual_var_steps": (lambda _qv_raw=(
                [_coh3_steps[_qi] * output_confs[_qi]
                 if _qi < len(output_confs) else 0.0
                 for _qi in range(len(_coh3_steps))]
            ): [
                round(float(np.var(_qv_raw[:i+1])), 6) if i >= 1 else 0.0
                for i in range(len(_qv_raw))
            ])(),
            # per-step running variance of quality signal (how much quality has spread)
            "sg_conf_corr": round(float(
                np.corrcoef(
                    np.array(output_confs[:min(len(output_confs), len(_sg_step_gaps))], dtype=np.float32),
                    np.array(_sg_step_gaps[:min(len(output_confs), len(_sg_step_gaps))], dtype=np.float32)
                )[0, 1]
            ) if min(len(output_confs), len(_sg_step_gaps)) >= 3 else 0.0, 4),
            # Pearson r between confidence and ScoreGapEMA (+ = larger gap = higher conf, healthy)
            "vel_var_steps": [
                round(float(np.var(_velocity_steps[:i+1])), 6) if i >= 1 else 0.0
                for i in range(len(_velocity_steps))
            ],  # per-step running variance of velocity EMA (high = erratic topic movement)
            "coh3_vprec_corr": round(float(
                np.corrcoef(
                    np.array(_coh3_steps[:min(len(_coh3_steps), len(_vprec_ema_steps))], dtype=np.float32),
                    np.array(_vprec_ema_steps[:min(len(_coh3_steps), len(_vprec_ema_steps))], dtype=np.float32)
                )[0, 1]
            ) if min(len(_coh3_steps), len(_vprec_ema_steps)) >= 3 else 0.0, 4),
            # Pearson r between coh3 and vprec_ema (+ = coherence and vocab precision rise together)
            "margin_var_steps": [
                round(float(np.var(_top1_margins[:i+1])), 6) if i >= 1 else 0.0
                for i in range(len(_top1_margins))
            ],  # per-step running variance of top1 score margins (high = erratic decisiveness)
            "coh3_sg_corr": round(float(
                np.corrcoef(
                    np.array(_coh3_steps[:min(len(_coh3_steps), len(_sg_step_gaps))], dtype=np.float32),
                    np.array(_sg_step_gaps[:min(len(_coh3_steps), len(_sg_step_gaps))], dtype=np.float32)
                )[0, 1]
            ) if min(len(_coh3_steps), len(_sg_step_gaps)) >= 3 else 0.0, 4),
            # Pearson r between coh3 and ScoreGapEMA (+ = more coherent = wider score gap, healthy)
            "conf_vel_corr": round(float(
                np.corrcoef(
                    np.array(output_confs[:min(len(output_confs), len(_velocity_steps))], dtype=np.float32),
                    np.array(_velocity_steps[:min(len(output_confs), len(_velocity_steps))], dtype=np.float32)
                )[0, 1]
            ) if min(len(output_confs), len(_velocity_steps)) >= 3 else 0.0, 4),
            # Pearson r between confidence and velocity EMA (negative = moving fast = lower conf, healthy)
            "topk_var_steps": [
                round(float(np.var(_topk_steps[:i+1])), 4) if i >= 1 else 0.0
                for i in range(len(_topk_steps))
            ],  # per-step running variance of adaptive TopK k (high = beam width is erratic)
            "coh6_conf_corr": round(float(
                np.corrcoef(
                    np.array(_coh6_steps[:min(len(_coh6_steps), len(output_confs))], dtype=np.float32),
                    np.array(output_confs[:min(len(_coh6_steps), len(output_confs))], dtype=np.float32)
                )[0, 1]
            ) if min(len(_coh6_steps), len(output_confs)) >= 3 else 0.0, 4),
            # Pearson r between 6-gram coherence and confidence (+ = long-range coherence and conf rise together)
            "topk_vel_corr": round(float(
                np.corrcoef(
                    np.array(_topk_steps[:min(len(_topk_steps), len(_velocity_steps))], dtype=np.float32),
                    np.array(_velocity_steps[:min(len(_topk_steps), len(_velocity_steps))], dtype=np.float32)
                )[0, 1]
            ) if min(len(_topk_steps), len(_velocity_steps)) >= 3 else 0.0, 4),
            # Pearson r between TopK k and velocity EMA (+ = wide beam during fast topic movement)
            "margin_conf_corr": round(float(
                np.corrcoef(
                    np.array(_top1_margins[:min(len(_top1_margins), len(output_confs))], dtype=np.float32),
                    np.array(output_confs[:min(len(_top1_margins), len(output_confs))],  dtype=np.float32)
                )[0, 1]
            ) if min(len(_top1_margins), len(output_confs)) >= 3 else 0.0, 4),
            # Pearson r between top1 margins and confidence (+ = decisive = confident, healthy)
            "entropy_vel_corr": round(float(
                np.corrcoef(
                    np.array(_entropy_steps[:min(len(_entropy_steps), len(_velocity_steps))], dtype=np.float32),
                    np.array(_velocity_steps[:min(len(_entropy_steps), len(_velocity_steps))], dtype=np.float32)
                )[0, 1]
            ) if min(len(_entropy_steps), len(_velocity_steps)) >= 3 else 0.0, 4),
            # Pearson r between entropy and velocity EMA (+ = high entropy = fast topic movement)
            "coh3_margin_slope_corr": round(float(
                np.corrcoef(
                    np.array(
                        [(_coh3_steps[i] - _coh3_steps[i-1]) if i >= 1 else 0.0
                          for i in range(len(_coh3_steps))], dtype=np.float32),
                    np.array(
                        [(_top1_margins[i] - _top1_margins[i-1]) if i >= 1 else 0.0
                          for i in range(min(len(_top1_margins), len(_coh3_steps)))],
                        dtype=np.float32)
                )[0, 1]
            ) if min(len(_coh3_steps), len(_top1_margins)) >= 4 else 0.0, 4),
            # Pearson r between coh3 slope and margin slope (+ = coherence and margin move together)
            "entropy_topk_corr": round(float(
                np.corrcoef(
                    np.array(_entropy_steps[:min(len(_entropy_steps), len(_topk_steps))], dtype=np.float32),
                    np.array(_topk_steps[:min(len(_entropy_steps), len(_topk_steps))],    dtype=np.float32)
                )[0, 1]
            ) if min(len(_entropy_steps), len(_topk_steps)) >= 3 else 0.0, 4),
            # Pearson r between entropy and TopK k (+ = high entropy opens wider beam, as expected)
            "vprec_entropy_corr": round(float(
                np.corrcoef(
                    np.array(_vprec_ema_steps[:min(len(_vprec_ema_steps), len(_entropy_steps))], dtype=np.float32),
                    np.array(_entropy_steps[:min(len(_vprec_ema_steps), len(_entropy_steps))],   dtype=np.float32)
                )[0, 1]
            ) if min(len(_vprec_ema_steps), len(_entropy_steps)) >= 3 else 0.0, 4),
            # Pearson r between vprec_ema and entropy (negative = precise = focused = healthy)
            "coh3_entropy_slope_corr": round(float(
                np.corrcoef(
                    np.array(
                        [(_coh3_steps[i] - _coh3_steps[i-1]) if i >= 1 else 0.0
                          for i in range(len(_coh3_steps))], dtype=np.float32),
                    np.array(
                        [(_entropy_steps[i] - _entropy_steps[i-1]) if i >= 1 else 0.0
                          for i in range(min(len(_entropy_steps), len(_coh3_steps)))],
                        dtype=np.float32)
                )[0, 1]
            ) if min(len(_coh3_steps), len(_entropy_steps)) >= 4 else 0.0, 4),
            # Pearson r between coh3 slope and entropy slope (- = coherence rises as entropy falls)
            "conf_margin_vel_score": (lambda: (
                (1 if (
                    min(len(_top1_margins), len(output_confs)) >= 3 and
                    float(np.corrcoef(
                        np.array(_top1_margins[:min(len(_top1_margins), len(output_confs))], dtype=np.float32),
                        np.array(output_confs[:min(len(_top1_margins), len(output_confs))], dtype=np.float32)
                    )[0, 1]) > 0.20
                ) else 0) +
                (1 if (
                    min(len(output_confs), len(_velocity_steps)) >= 3 and
                    float(np.corrcoef(
                        np.array(output_confs[:min(len(output_confs), len(_velocity_steps))], dtype=np.float32),
                        np.array(_velocity_steps[:min(len(output_confs), len(_velocity_steps))], dtype=np.float32)
                    )[0, 1]) < -0.20
                ) else 0) +
                (1 if (
                    min(len(_vprec_ema_steps), len(_entropy_steps)) >= 3 and
                    float(np.corrcoef(
                        np.array(_vprec_ema_steps[:min(len(_vprec_ema_steps), len(_entropy_steps))], dtype=np.float32),
                        np.array(_entropy_steps[:min(len(_vprec_ema_steps), len(_entropy_steps))], dtype=np.float32)
                    )[0, 1]) < -0.20
                ) else 0) +
                (1 if (
                    min(len(_coh3_steps), len(_sg_step_gaps)) >= 3 and
                    float(np.corrcoef(
                        np.array(_coh3_steps[:min(len(_coh3_steps), len(_sg_step_gaps))], dtype=np.float32),
                        np.array(_sg_step_gaps[:min(len(_coh3_steps), len(_sg_step_gaps))], dtype=np.float32)
                    )[0, 1]) > 0.20
                ) else 0)
            ))(),
            # composite health score 0-4: counts active healthy correlation patterns
            "coh3_vel_conf_joint": round(float(
                sum(
                    1 for _ji in range(min(len(_coh3_steps), len(_velocity_steps), len(output_confs)))
                    if (_coh3_steps[_ji] > (sum(_coh3_steps) / max(len(_coh3_steps), 1))
                        and _velocity_steps[_ji] < (sum(_velocity_steps) / max(len(_velocity_steps), 1))
                        and output_confs[_ji] > (sum(output_confs) / max(len(output_confs), 1)))
                ) / max(min(len(_coh3_steps), len(_velocity_steps), len(output_confs)), 1)
            ) if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) >= 4 else 0.0, 4),
            # fraction of steps where coh3>mean AND vel<mean AND conf>mean simultaneously
            "coh3_slope_trend": round(float(
                ((_coh3_steps[-1] - _coh3_steps[-2]) - (_coh3_steps[-2] - _coh3_steps[-3]))
                if len(_coh3_steps) >= 3 else 0.0
            ), 6),
            # 2nd derivative of coh3 (+ = coherence gaining momentum, - = slowing)
            "conf_slope_trend": round(float(
                ((_conf_ema_steps[-1] - _conf_ema_steps[-2]) - (_conf_ema_steps[-2] - _conf_ema_steps[-3]))
                if len(_conf_ema_steps) >= 3 else 0.0
            ), 6),
            # 2nd derivative of conf_ema (+ = confidence acceleration, - = deceleration)
            "vel_slope_trend": round(float(
                ((_velocity_steps[-1] - _velocity_steps[-2]) - (_velocity_steps[-2] - _velocity_steps[-3]))
                if len(_velocity_steps) >= 3 else 0.0
            ), 7),
            # 2nd derivative of velocity EMA (+ = topic drift accelerating)
            "margin_slope_trend": round(float(
                ((_top1_margins[-1] - _top1_margins[-2]) - (_top1_margins[-2] - _top1_margins[-3]))
                if len(_top1_margins) >= 3 else 0.0
            ), 6),
            # 2nd derivative of top1 margins (+ = decisiveness accelerating)
            "entropy_slope_trend": round(float(
                ((_entropy_steps[-1] - _entropy_steps[-2]) - (_entropy_steps[-2] - _entropy_steps[-3]))
                if len(_entropy_steps) >= 3 else 0.0
            ), 7),
            # 2nd derivative of entropy (+ = distribution spreading faster)
            "topk_slope_trend": round(float(
                ((_topk_steps[-1] - _topk_steps[-2]) - (_topk_steps[-2] - _topk_steps[-3]))
                if len(_topk_steps) >= 3 else 0.0
            ), 6),
            # 2nd derivative of top-k (+ = beam widening faster)
            "sg_slope_trend": round(float(
                ((_sg_step_gaps[-1] - _sg_step_gaps[-2]) - (_sg_step_gaps[-2] - _sg_step_gaps[-3]))
                if len(_sg_step_gaps) >= 3 else 0.0
            ), 7),
            # 2nd derivative of score-gap EMA (+ = competition sharpening faster)
            "vprec_slope_trend": round(float(
                ((_vprec_ema_steps[-1] - _vprec_ema_steps[-2]) - (_vprec_ema_steps[-2] - _vprec_ema_steps[-3]))
                if len(_vprec_ema_steps) >= 3 else 0.0
            ), 7),
            # 2nd derivative of vprec_ema (+ = vocab precision tightening faster)
            "coh6_slope_trend": round(float(
                ((_coh6_steps[-1] - _coh6_steps[-2]) - (_coh6_steps[-2] - _coh6_steps[-3]))
                if len(_coh6_steps) >= 3 else 0.0
            ), 6),
            # 2nd derivative of coh6 (+ = long-window coherence gaining momentum)
            "margin_vel_joint": round(float(
                sum(
                    1 for _xi in range(min(len(_top1_margins), len(_velocity_steps)))
                    if (_top1_margins[_xi] > (sum(_top1_margins) / max(len(_top1_margins), 1))
                        and _velocity_steps[_xi] < (sum(_velocity_steps) / max(len(_velocity_steps), 1)))
                ) / max(min(len(_top1_margins), len(_velocity_steps)), 1)
            ) if min(len(_top1_margins), len(_velocity_steps)) >= 4 else 0.0, 4),
            # fraction of steps where margin>avg AND vel<avg (decisive+stable)
            "conf_vel_joint": round(float(
                sum(
                    1 for _xi in range(min(len(output_confs), len(_velocity_steps)))
                    if (output_confs[_xi] > (sum(output_confs) / max(len(output_confs), 1))
                        and _velocity_steps[_xi] < (sum(_velocity_steps) / max(len(_velocity_steps), 1)))
                ) / max(min(len(output_confs), len(_velocity_steps)), 1)
            ) if min(len(output_confs), len(_velocity_steps)) >= 4 else 0.0, 4),
            # fraction of steps where conf>avg AND vel<avg (confident+stable)
            "coh3_margin_joint": round(float(
                sum(
                    1 for _xi in range(min(len(_coh3_steps), len(_top1_margins)))
                    if (_coh3_steps[_xi] > (sum(_coh3_steps) / max(len(_coh3_steps), 1))
                        and _top1_margins[_xi] > (sum(_top1_margins) / max(len(_top1_margins), 1)))
                ) / max(min(len(_coh3_steps), len(_top1_margins)), 1)
            ) if min(len(_coh3_steps), len(_top1_margins)) >= 4 else 0.0, 4),
            # fraction of steps where coh3>avg AND margin>avg (coherent+decisive)
            "entropy_vel_joint": round(float(
                sum(
                    1 for _xi in range(min(len(_entropy_steps), len(_velocity_steps)))
                    if (_entropy_steps[_xi] < (sum(_entropy_steps) / max(len(_entropy_steps), 1))
                        and _velocity_steps[_xi] < (sum(_velocity_steps) / max(len(_velocity_steps), 1)))
                ) / max(min(len(_entropy_steps), len(_velocity_steps)), 1)
            ) if min(len(_entropy_steps), len(_velocity_steps)) >= 4 else 0.0, 4),
            # fraction of steps where entropy<avg AND vel<avg (focused+stable)
            "vprec_coh3_joint": round(float(
                sum(
                    1 for _xi in range(min(len(_vprec_ema_steps), len(_coh3_steps)))
                    if (_vprec_ema_steps[_xi] > (sum(_vprec_ema_steps) / max(len(_vprec_ema_steps), 1))
                        and _coh3_steps[_xi] > (sum(_coh3_steps) / max(len(_coh3_steps), 1)))
                ) / max(min(len(_vprec_ema_steps), len(_coh3_steps)), 1)
            ) if min(len(_vprec_ema_steps), len(_coh3_steps)) >= 4 else 0.0, 4),
            # fraction of steps where vprec>avg AND coh3>avg (precise+coherent)
            "quadrant_map": (lambda: (
                lambda _qn=min(len(_coh3_steps), len(_velocity_steps)):
                    ({"ideal": 0, "exploring": 0, "flat": 0, "drifting": 0}
                     if _qn < 2 else
                     {k: v for k, v in zip(
                         ["ideal", "exploring", "flat", "drifting"],
                         [
                             sum(1 for i in range(_qn)
                                 if _coh3_steps[i] > sum(_coh3_steps[:_qn])/_qn
                                 and _velocity_steps[i] < sum(_velocity_steps[:_qn])/_qn),
                             sum(1 for i in range(_qn)
                                 if _coh3_steps[i] > sum(_coh3_steps[:_qn])/_qn
                                 and _velocity_steps[i] >= sum(_velocity_steps[:_qn])/_qn),
                             sum(1 for i in range(_qn)
                                 if _coh3_steps[i] <= sum(_coh3_steps[:_qn])/_qn
                                 and _velocity_steps[i] < sum(_velocity_steps[:_qn])/_qn),
                             sum(1 for i in range(_qn)
                                 if _coh3_steps[i] <= sum(_coh3_steps[:_qn])/_qn
                                 and _velocity_steps[i] >= sum(_velocity_steps[:_qn])/_qn),
                         ]
                     )})()
            ))(),
            # step quadrant counts: ideal/exploring/flat/drifting (coh3×vel grid)
            "ideal_run_len": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qn=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps), len(output_confs))]) /
                                max(min(len(_coh3_steps), len(_velocity_steps), len(output_confs)), 1),
                           _vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps), len(output_confs))]) /
                               max(min(len(_coh3_steps), len(_velocity_steps), len(output_confs)), 1),
                           _cfm=sum(output_confs[:min(len(_coh3_steps), len(_velocity_steps), len(output_confs))]) /
                                max(min(len(_coh3_steps), len(_velocity_steps), len(output_confs)), 1):
                    __import__('functools').reduce(
                        lambda acc, i: (
                            (acc[0] + 1, max(acc[1], acc[0] + 1))
                            if (_coh3_steps[i] > _c3m
                                and _velocity_steps[i] < _vm
                                and output_confs[i] > _cfm)
                            else (0, acc[1])
                        ),
                        range(_qn), (0, 0)
                    )[1]
                )()
            ))(),
            # longest consecutive run of ideal steps (coh3+vel+conf all above/below avg)
            "phase_quality_score": (lambda: (
                lambda _ps=({} if min(len(_coh3_steps), len(output_confs)) < 2 else {
                    ph: round(
                        float(np.mean(output_confs[lo:hi])) *
                        float(np.mean(_coh3_steps[lo:hi])), 4)
                    if output_confs[lo:hi] and _coh3_steps[lo:hi] else 0.0
                    for ph, lo, hi in [
                        ("early", 0,                     max(len(output_confs)//3, 1)),
                        ("mid",   len(output_confs)//3,  2*len(output_confs)//3),
                        ("late",  2*len(output_confs)//3, len(output_confs)),
                    ]
                }): _ps
            ))(),
            # per-phase quality: avg_conf × avg_coh3 for early/mid/late thirds
            "streak_map": (lambda: (
                (lambda _sn=min(len(_coh3_steps), len(_velocity_steps)):
                    [] if _sn < 2 else
                    (lambda _sc3m=sum(_coh3_steps[:_sn])/_sn,
                             _svm=sum(_velocity_steps[:_sn])/_sn:
                        (lambda _quads=[
                            ("ideal"      if _coh3_steps[i] > sum(_coh3_steps[:_sn])/_sn
                                          and _velocity_steps[i] < sum(_velocity_steps[:_sn])/_sn
                             else "exploring" if _coh3_steps[i] > sum(_coh3_steps[:_sn])/_sn
                             else "drifting"  if _velocity_steps[i] >= sum(_velocity_steps[:_sn])/_sn
                             else "flat")
                            for i in range(_sn)
                        ]:
                            __import__('itertools').groupby(_quads)
                            and [(q, sum(1 for _ in g))
                                 for q, g in __import__('itertools').groupby(_quads)]
                        )()
                    )()
                )()
            ))(),
            # sequence of (quadrant, streak_length) pairs across generation
            "phase_transition_map": (lambda: (
                [] if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _ptn=min(len(_coh3_steps), len(_velocity_steps)),
                           _ptc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _ptvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                 max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    [
                        (i, (
                            "ideal"      if _coh3_steps[i-1] > _ptc3m and _velocity_steps[i-1] < _ptvm
                            else "exploring" if _coh3_steps[i-1] > _ptc3m
                            else "drifting"  if _velocity_steps[i-1] >= _ptvm else "flat"),
                         (
                            "ideal"      if _coh3_steps[i] > _ptc3m and _velocity_steps[i] < _ptvm
                            else "exploring" if _coh3_steps[i] > _ptc3m
                            else "drifting"  if _velocity_steps[i] >= _ptvm else "flat"))
                        for i in range(1, _ptn)
                        if (("ideal"      if _coh3_steps[i-1] > _ptc3m and _velocity_steps[i-1] < _ptvm
                             else "exploring" if _coh3_steps[i-1] > _ptc3m
                             else "drifting"  if _velocity_steps[i-1] >= _ptvm else "flat") !=
                            ("ideal"      if _coh3_steps[i] > _ptc3m and _velocity_steps[i] < _ptvm
                             else "exploring" if _coh3_steps[i] > _ptc3m
                             else "drifting"  if _velocity_steps[i] >= _ptvm else "flat"))
                    ]
                )()
            ))(),
            # list of (step_idx, from_quadrant, to_quadrant) for every quadrant change
            "conf_entropy_ratio": round(
                (sum(output_confs) / max(len(output_confs), 1)) /
                max(sum(_entropy_steps) / max(len(_entropy_steps), 1), 1e-6), 4
            ) if output_confs and _entropy_steps else 0.0,
            # mean_conf / mean_entropy: >1.0 → confidence dominates, <1.0 → entropy dominates
            "conf_velocity_score": round(
                (sum(output_confs) / max(len(output_confs), 1)) *
                max(0.0, 1.0 - (sum(_velocity_steps) / max(len(_velocity_steps), 1))), 4
            ) if output_confs and _velocity_steps else 0.0,
            # mean_conf × (1 − mean_vel): high quality when both confident and velocity-stable
            "quad_entropy": (lambda: (
                {} if min(len(_coh3_steps), len(_velocity_steps), len(_entropy_steps)) < 2
                else (lambda _qen=min(len(_coh3_steps), len(_velocity_steps), len(_entropy_steps)),
                           _qec3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps), len(_entropy_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps), len(_entropy_steps)), 1),
                           _qevm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps), len(_entropy_steps))]) /
                                 max(min(len(_coh3_steps), len(_velocity_steps), len(_entropy_steps)), 1):
                    {q: round(float(np.mean([_entropy_steps[i] for i in range(_qen)
                        if ("ideal"      if _coh3_steps[i] > _qec3m and _velocity_steps[i] < _qevm
                            else "exploring" if _coh3_steps[i] > _qec3m
                            else "drifting"  if _velocity_steps[i] >= _qevm else "flat") == q]
                    )), 5) if any(
                        ("ideal"      if _coh3_steps[i] > _qec3m and _velocity_steps[i] < _qevm
                         else "exploring" if _coh3_steps[i] > _qec3m
                         else "drifting"  if _velocity_steps[i] >= _qevm else "flat") == q
                        for i in range(_qen)
                    ) else 0.0
                    for q in ["ideal", "exploring", "flat", "drifting"]}
                )()
            ))(),
            # per-quadrant mean entropy: which quadrant had the most focused steps?
            "quad_conf_mean": (lambda: (
                {} if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qcn=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qcc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps), len(output_confs))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps), len(output_confs)), 1),
                           _qcvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps), len(output_confs))]) /
                                 max(min(len(_coh3_steps), len(_velocity_steps), len(output_confs)), 1):
                    {q: round(float(np.mean([output_confs[i] for i in range(_qcn)
                        if ("ideal"      if _coh3_steps[i] > _qcc3m and _velocity_steps[i] < _qcvm
                            else "exploring" if _coh3_steps[i] > _qcc3m
                            else "drifting"  if _velocity_steps[i] >= _qcvm else "flat") == q]
                    )), 5) if any(
                        ("ideal"      if _coh3_steps[i] > _qcc3m and _velocity_steps[i] < _qcvm
                         else "exploring" if _coh3_steps[i] > _qcc3m
                         else "drifting"  if _velocity_steps[i] >= _qcvm else "flat") == q
                        for i in range(_qcn)
                    ) else 0.0
                    for q in ["ideal", "exploring", "flat", "drifting"]}
                )()
            ))(),
            # per-quadrant mean confidence: which quadrant had the most certain steps?
            "transition_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _trn=min(len(_coh3_steps), len(_velocity_steps)),
                           _trc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _trvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                 max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(
                        sum(1 for i in range(1, _trn)
                            if ("ideal"      if _coh3_steps[i] > _trc3m and _velocity_steps[i] < _trvm
                                else "exploring" if _coh3_steps[i] > _trc3m
                                else "drifting"  if _velocity_steps[i] >= _trvm else "flat") !=
                               ("ideal"      if _coh3_steps[i-1] > _trc3m and _velocity_steps[i-1] < _trvm
                                else "exploring" if _coh3_steps[i-1] > _trc3m
                                else "drifting"  if _velocity_steps[i-1] >= _trvm else "flat")
                        ) / max(_trn - 1, 1), 4)
                )()
            ))(),
            # fraction of step-to-step transitions that change quadrant (0=stable, 1=volatile)
            "ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _ifn=min(len(_coh3_steps), len(_velocity_steps)),
                           _ifc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _ifvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                 max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_ifn)
                              if _coh3_steps[i] > _ifc3m and _velocity_steps[i] < _ifvm
                    ) / max(_ifn, 1), 4)
                )()
            ))(),
            # fraction of steps that are in the ideal quadrant (coh3↑ vel↓)
            "quad_velocity_mean": (lambda: (
                {} if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qvn=min(len(_coh3_steps), len(_velocity_steps)),
                           _qvc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qvvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                 max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    {q: round(float(np.mean([_velocity_steps[i] for i in range(_qvn)
                        if ("ideal"      if _coh3_steps[i] > _qvc3m and _velocity_steps[i] < _qvvm
                            else "exploring" if _coh3_steps[i] > _qvc3m
                            else "drifting"  if _velocity_steps[i] >= _qvvm else "flat") == q]
                    )), 5) if any(
                        ("ideal"      if _coh3_steps[i] > _qvc3m and _velocity_steps[i] < _qvvm
                         else "exploring" if _coh3_steps[i] > _qvc3m
                         else "drifting"  if _velocity_steps[i] >= _qvvm else "flat") == q
                        for i in range(_qvn)
                    ) else 0.0
                    for q in ["ideal", "exploring", "flat", "drifting"]}
                )()
            ))(),
            # per-quadrant mean semantic velocity
            "coh3_vel_divergence": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else round(abs(
                    sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                    max(min(len(_coh3_steps), len(_velocity_steps)), 1) -
                    (1.0 - sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                           max(min(len(_coh3_steps), len(_velocity_steps)), 1))
                ), 5)
            ))(),
            # |mean_coh3 − (1 − mean_vel)|: 0=perfectly aligned, >0.15=misaligned
            "quad_coh3_mean": (lambda: (
                {} if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qcbn=min(len(_coh3_steps), len(_velocity_steps)),
                           _qcbc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qcbvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    {q: round(float(np.mean([_coh3_steps[i] for i in range(_qcbn)
                        if ("ideal"      if _coh3_steps[i] > _qcbc3m and _velocity_steps[i] < _qcbvm
                            else "exploring" if _coh3_steps[i] > _qcbc3m
                            else "drifting"  if _velocity_steps[i] >= _qcbvm else "flat") == q]
                    )), 5) if any(
                        ("ideal"      if _coh3_steps[i] > _qcbc3m and _velocity_steps[i] < _qcbvm
                         else "exploring" if _coh3_steps[i] > _qcbc3m
                         else "drifting"  if _velocity_steps[i] >= _qcbvm else "flat") == q
                        for i in range(_qcbn)
                    ) else 0.0
                    for q in ["ideal", "exploring", "flat", "drifting"]}
                )()
            ))(),
            # per-quadrant mean coherence (coh3)
            "conf_coh3_gap": round(
                (sum(output_confs) / max(len(output_confs), 1)) -
                (sum(_coh3_steps)  / max(len(_coh3_steps),  1)), 5
            ) if output_confs and _coh3_steps else 0.0,
            # mean_conf − mean_coh3: >0=overconfident vs coherence, <0=under-confident
            "quad_transition_from": (lambda: (
                {} if not (
                    min(len(_coh3_steps), len(_velocity_steps)) >= 2
                    and any(True for _ in range(1, min(len(_coh3_steps), len(_velocity_steps))))
                )
                else (lambda _qtfn=min(len(_coh3_steps), len(_velocity_steps)),
                           _qtfc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qtfvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    {q: sum(
                        1 for i in range(1, _qtfn)
                        if ("ideal"      if _coh3_steps[i-1] > _qtfc3m and _velocity_steps[i-1] < _qtfvm
                            else "exploring" if _coh3_steps[i-1] > _qtfc3m
                            else "drifting"  if _velocity_steps[i-1] >= _qtfvm else "flat") == q
                        and ("ideal"     if _coh3_steps[i] > _qtfc3m and _velocity_steps[i] < _qtfvm
                            else "exploring" if _coh3_steps[i] > _qtfc3m
                            else "drifting"  if _velocity_steps[i] >= _qtfvm else "flat") != q
                    )
                    for q in ["ideal", "exploring", "flat", "drifting"]}
                )()
            ))(),
            # dict of {quadrant: times_left}: which quadrant was departed most often
            "quad_transition_to": (lambda: (
                {} if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qttn=min(len(_coh3_steps), len(_velocity_steps)),
                           _qttc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qttvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    {q: sum(
                        1 for i in range(1, _qttn)
                        if ("ideal"      if _coh3_steps[i] > _qttc3m and _velocity_steps[i] < _qttvm
                            else "exploring" if _coh3_steps[i] > _qttc3m
                            else "drifting"  if _velocity_steps[i] >= _qttvm else "flat") == q
                        and ("ideal"     if _coh3_steps[i-1] > _qttc3m and _velocity_steps[i-1] < _qttvm
                            else "exploring" if _coh3_steps[i-1] > _qttc3m
                            else "drifting"  if _velocity_steps[i-1] >= _qttvm else "flat") != q
                    )
                    for q in ["ideal", "exploring", "flat", "drifting"]}
                )()
            ))(),
            # dict of {quadrant: times_entered}: which quadrant was entered most often
            "ideal_entry_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _iern=min(len(_coh3_steps), len(_velocity_steps)):
                    (lambda _qseq=[
                        ("ideal"
                         if _coh3_steps[i] > sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                            max(min(len(_coh3_steps), len(_velocity_steps)), 1)
                         and _velocity_steps[i] < sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                             max(min(len(_coh3_steps), len(_velocity_steps)), 1)
                         else "exploring"
                         if _coh3_steps[i] > sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                            max(min(len(_coh3_steps), len(_velocity_steps)), 1)
                         else "drifting"
                         if _velocity_steps[i] >= sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                            max(min(len(_coh3_steps), len(_velocity_steps)), 1)
                         else "flat")
                        for i in range(_iern)
                    ]:
                        round(
                            sum(1 for j in range(1, len(_qseq))
                                if _qseq[j] == "ideal" and _qseq[j-1] != "ideal") /
                            max(sum(1 for j in range(1, len(_qseq))
                                    if _qseq[j] != _qseq[j-1]), 1),
                            4)
                    )()
                )()
            ))(),
            # fraction of quadrant transitions that land in ideal (0=never, 1=always ideal entry)
            "drifting_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _dfn=min(len(_coh3_steps), len(_velocity_steps)),
                           _dfc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _dfvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                 max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_dfn)
                              if _coh3_steps[i] <= _dfc3m and _velocity_steps[i] >= _dfvm
                    ) / max(_dfn, 1), 4)
                )()
            ))(),
            # fraction of steps in the drifting quadrant (coh3↓ vel↑)
            "quad_balance_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qbn=min(len(_coh3_steps), len(_velocity_steps)),
                           _qbc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qbvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                 max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round((
                        sum(1 for i in range(_qbn)
                            if _coh3_steps[i] > _qbc3m and _velocity_steps[i] < _qbvm) -
                        sum(1 for i in range(_qbn)
                            if _coh3_steps[i] <= _qbc3m and _velocity_steps[i] >= _qbvm)
                    ) / max(_qbn, 1), 4)
                )()
            ))(),
            # (ideal_count − drifting_count) / n: quality bias (+=quality, −=drift)
            "exploring_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _efn=min(len(_coh3_steps), len(_velocity_steps)),
                           _efc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _efvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                 max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_efn)
                              if _coh3_steps[i] > _efc3m and _velocity_steps[i] >= _efvm
                    ) / max(_efn, 1), 4)
                )()
            ))(),
            # fraction of steps in the exploring quadrant (coh3↑ vel↑: creative drift)
            "flat_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _ffn=min(len(_coh3_steps), len(_velocity_steps)),
                           _ffc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _ffvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                 max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_ffn)
                              if _coh3_steps[i] <= _ffc3m and _velocity_steps[i] < _ffvm
                    ) / max(_ffn, 1), 4)
                )()
            ))(),
            # fraction of steps in the flat quadrant (coh3↓ vel↓: stagnant generation)
            "quad_volatility_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qvn=min(len(_coh3_steps), len(_velocity_steps)),
                           _qvc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qvvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                 max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qvseq=[
                        ("ideal"
                         if _coh3_steps[i] > sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                            max(min(len(_coh3_steps), len(_velocity_steps)), 1)
                         and _velocity_steps[i] < sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                             max(min(len(_coh3_steps), len(_velocity_steps)), 1)
                         else "exploring"
                         if _coh3_steps[i] > sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                            max(min(len(_coh3_steps), len(_velocity_steps)), 1)
                         else "drifting"
                         if _velocity_steps[i] >= sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                            max(min(len(_coh3_steps), len(_velocity_steps)), 1)
                         else "flat")
                        for i in range(_qvn)
                    ]:
                        round(
                            sum(1 for j in range(1, len(_qvseq)) if _qvseq[j] != _qvseq[j-1]) /
                            max(len(_qvseq) - 1, 1),
                            4)
                    )()
                )()
            ))(),
            # normalised transition rate per step (0=constant; 1=flip every step)
            "quad_dominance_margin": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qdm_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdm_fracs=sorted([
                        sum(1 for i in range(_qdm_n)
                            if _coh3_steps[i] > _qdm_c3m and _velocity_steps[i] < _qdm_vm
                        ) / max(_qdm_n, 1),
                        sum(1 for i in range(_qdm_n)
                            if _coh3_steps[i] > _qdm_c3m and _velocity_steps[i] >= _qdm_vm
                        ) / max(_qdm_n, 1),
                        sum(1 for i in range(_qdm_n)
                            if _coh3_steps[i] <= _qdm_c3m and _velocity_steps[i] < _qdm_vm
                        ) / max(_qdm_n, 1),
                        sum(1 for i in range(_qdm_n)
                            if _coh3_steps[i] <= _qdm_c3m and _velocity_steps[i] >= _qdm_vm
                        ) / max(_qdm_n, 1),
                    ], reverse=True):
                        round(_qdm_fracs[0] - _qdm_fracs[1], 4)
                    )()
                )()
            ))(),
            # dominant_frac − second_frac: how decisively one quadrant dominates (0=tied)
            "ideal_run_density": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _irdn=min(len(_coh3_steps), len(_velocity_steps)),
                           _irdc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _irdvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _ird_labs=[
                        "ideal" if _coh3_steps[i] > _irdc3m and _velocity_steps[i] < _irdvm
                        else "other"
                        for i in range(_irdn)
                    ]:
                        round(
                            (sum(1 for x in _ird_labs if x == "ideal") / max(_irdn, 1)) *
                            max((len(list(g))
                                 for k, g in __import__('itertools').groupby(_ird_labs)
                                 if k == "ideal"),
                                default=0),
                            4)
                    )()
                )()
            ))(),
            # ideal_frac × longest_ideal_run: combined quality richness score
            "quad_recovery_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qrn=min(len(_coh3_steps), len(_velocity_steps)),
                           _qrc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qrvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                 max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qrseq=[
                        ("ideal"      if _coh3_steps[i] > _qrc3m and _velocity_steps[i] < _qrvm
                         else "exploring" if _coh3_steps[i] > _qrc3m
                         else "drifting"  if _velocity_steps[i] >= _qrvm
                         else "flat")
                        for i in range(_qrn)
                    ]:
                        round(
                            sum(1 for j in range(1, len(_qrseq))
                                if _qrseq[j-1] == "drifting"
                                and _qrseq[j] in ("ideal", "exploring")) /
                            max(sum(1 for j in range(1, len(_qrseq))
                                    if _qrseq[j-1] == "drifting"), 1),
                            4)
                    )()
                )()
            ))(),
            # when drifting: fraction of next-steps that recover to ideal/exploring
            "quad_persistence_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qpsn=min(len(_coh3_steps), len(_velocity_steps)),
                           _qpsc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qpsvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qpsseq=[
                        ("ideal"      if _coh3_steps[i] > _qpsc3m and _velocity_steps[i] < _qpsvm
                         else "exploring" if _coh3_steps[i] > _qpsc3m
                         else "drifting"  if _velocity_steps[i] >= _qpsvm
                         else "flat")
                        for i in range(_qpsn)
                    ]:
                        round(_qpsn / max(
                            sum(1 for j in range(1, len(_qpsseq))
                                if _qpsseq[j] != _qpsseq[j-1]),
                            1), 2)
                    )()
                )()
            ))(),
            # total_steps / transitions: avg steps per quadrant visit (≥6=sticky)
            "ideal_stability_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _issn=min(len(_coh3_steps), len(_velocity_steps)),
                           _issc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _issvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _iss_ideal=sum(
                        1 for i in range(_issn)
                        if _coh3_steps[i] > _issc3m and _velocity_steps[i] < _issvm
                    ) / max(_issn, 1),
                           _iss_drift=sum(
                        1 for i in range(_issn)
                        if _coh3_steps[i] <= _issc3m and _velocity_steps[i] >= _issvm
                    ) / max(_issn, 1):
                        round(_iss_ideal - _iss_drift, 4)
                    )()
                )()
            ))(),
            # ideal_frac − drifting_frac: net quality stability (+=ideal-dominant)
            "quad_oscillation_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 6
                else (lambda _qosn=min(len(_coh3_steps), len(_velocity_steps)),
                           _qosc3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qosvm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                  max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qosseq=[
                        ("ideal"      if _coh3_steps[i] > _qosc3m and _velocity_steps[i] < _qosvm
                         else "exploring" if _coh3_steps[i] > _qosc3m
                         else "drifting"  if _velocity_steps[i] >= _qosvm
                         else "flat")
                        for i in range(_qosn)
                    ]:
                        (lambda _qos_tt=sum(1 for j in range(1, len(_qosseq))
                                            if _qosseq[j] != _qosseq[j-1]):
                            round(
                                sum(1 for j in range(2, len(_qosseq))
                                    if _qosseq[j] != _qosseq[j-1]
                                    and _qosseq[j] == _qosseq[j-2]) /
                                max(_qos_tt - 1, 1),
                                4)
                            if _qos_tt >= 3 else 0.0
                        )()
                    )()
                )()
            ))(),
            # fraction of transitions that are A→B→A ping-pong oscillations
            "quad_ideal_entry_velocity": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qiev_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qiev_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qiev_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qiev_entries=[
                        _velocity_steps[i]
                        for i in range(1, _qiev_n)
                        if _coh3_steps[i] > _qiev_c3m and _velocity_steps[i] < _qiev_vm
                        and not (_coh3_steps[i-1] > _qiev_c3m and _velocity_steps[i-1] < _qiev_vm)
                    ]:
                        round(sum(_qiev_entries) / max(len(_qiev_entries), 1), 4)
                        if _qiev_entries else 0.0
                    )()
                )()
            ))(),
            # mean velocity at the moment of entering the ideal quadrant (lower=smoother)
            "quad_coh3_entry_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qce_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qce_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qce_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qce_ents=[
                        _coh3_steps[i]
                        for i in range(1, _qce_n)
                        if _coh3_steps[i] > _qce_c3m and _velocity_steps[i] < _qce_vm
                        and not (_coh3_steps[i-1] > _qce_c3m and _velocity_steps[i-1] < _qce_vm)
                    ]:
                        round(sum(_qce_ents) / max(len(_qce_ents), 1), 4)
                        if _qce_ents else 0.0
                    )()
                )()
            ))(),
            # mean coh3 at ideal-quadrant entry (higher = entering from strong coherence base)
            "quad_drifting_exit_coh3": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdx_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdx_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdx_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdx_exits=[
                        _coh3_steps[i]
                        for i in range(1, _qdx_n)
                        if _coh3_steps[i-1] <= _qdx_c3m and _velocity_steps[i-1] >= _qdx_vm
                        and not (_coh3_steps[i] <= _qdx_c3m and _velocity_steps[i] >= _qdx_vm)
                    ]:
                        round(sum(_qdx_exits) / max(len(_qdx_exits), 1), 4)
                        if _qdx_exits else 0.0
                    )()
                )()
            ))(),
            # mean coh3 at drifting-quadrant exit (higher = exiting from better position)
            "quad_drifting_entry_velocity": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdv_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdv_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdv_ents=[
                        _velocity_steps[i]
                        for i in range(1, _qdv_n)
                        if _coh3_steps[i] <= _qdv_c3m and _velocity_steps[i] >= _qdv_vm
                        and not (_coh3_steps[i-1] <= _qdv_c3m and _velocity_steps[i-1] >= _qdv_vm)
                    ]:
                        round(sum(_qdv_ents) / max(len(_qdv_ents), 1), 4)
                        if _qdv_ents else 0.0
                    )()
                )()
            ))(),
            # mean velocity at moment of entering drifting quadrant (higher=harder fall)
            "quad_ideal_duration_variance": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qidv_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qidv_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qidv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qidv_labs=[
                        "ideal" if _coh3_steps[i] > _qidv_c3m and _velocity_steps[i] < _qidv_vm
                        else "other"
                        for i in range(_qidv_n)
                    ]:
                        (lambda _qidv_runs=[
                            len(list(g))
                            for k, g in __import__('itertools').groupby(_qidv_labs)
                            if k == "ideal"
                        ]:
                            round(
                                (sum((r - sum(_qidv_runs)/max(len(_qidv_runs),1))**2
                                     for r in _qidv_runs) /
                                 max(len(_qidv_runs), 1)) ** 0.5,
                                4) if _qidv_runs else 0.0
                        )()
                    )()
                )()
            ))(),
            # std-dev of ideal-run lengths (0=perfectly consistent; high=erratic)
            "quad_flat_duration_variance": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qfdv_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfdv_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfdv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfdv_labs=[
                        "flat" if _coh3_steps[i] <= _qfdv_c3m and _velocity_steps[i] < _qfdv_vm
                        else "other"
                        for i in range(_qfdv_n)
                    ]:
                        (lambda _qfdv_runs=[
                            len(list(g))
                            for k, g in __import__('itertools').groupby(_qfdv_labs)
                            if k == "flat"
                        ]:
                            round(
                                (sum((r - sum(_qfdv_runs)/max(len(_qfdv_runs),1))**2
                                     for r in _qfdv_runs) /
                                 max(len(_qfdv_runs), 1)) ** 0.5,
                                4) if _qfdv_runs else 0.0
                        )()
                    )()
                )()
            ))(),
            # std-dev of flat-run lengths (0=uniform stagnation; high=erratic)
            "quad_recovery_velocity": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qrv_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qrv_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qrv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qrv_exits=[
                        _velocity_steps[i]
                        for i in range(1, _qrv_n)
                        if _coh3_steps[i-1] <= _qrv_c3m and _velocity_steps[i-1] >= _qrv_vm
                        and not (_coh3_steps[i] <= _qrv_c3m and _velocity_steps[i] >= _qrv_vm)
                    ]:
                        round(sum(_qrv_exits) / max(len(_qrv_exits), 1), 4)
                        if _qrv_exits else 0.0
                    )()
                )()
            ))(),
            # mean velocity at moment of exiting drifting quadrant (lower=smoother recovery)
            "quad_drifting_duration_variance": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qddv_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qddv_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qddv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qddv_labs=[
                        "drifting" if _coh3_steps[i] <= _qddv_c3m and _velocity_steps[i] >= _qddv_vm
                        else "other"
                        for i in range(_qddv_n)
                    ]:
                        (lambda _qddv_runs=[
                            len(list(g))
                            for k, g in __import__('itertools').groupby(_qddv_labs)
                            if k == "drifting"
                        ]:
                            round(
                                (sum((r - sum(_qddv_runs)/max(len(_qddv_runs),1))**2
                                     for r in _qddv_runs) /
                                 max(len(_qddv_runs), 1)) ** 0.5,
                                4) if _qddv_runs else 0.0
                        )()
                    )()
                )()
            ))(),
            # std-dev of drifting-run lengths (0=uniform; high=erratic drift bursts)
            "quad_exploring_duration_variance": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qedv_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qedv_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qedv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qedv_labs=[
                        "exploring" if _coh3_steps[i] > _qedv_c3m and _velocity_steps[i] >= _qedv_vm
                        else "other"
                        for i in range(_qedv_n)
                    ]:
                        (lambda _qedv_runs=[
                            len(list(g))
                            for k, g in __import__('itertools').groupby(_qedv_labs)
                            if k == "exploring"
                        ]:
                            round(
                                (sum((r - sum(_qedv_runs)/max(len(_qedv_runs),1))**2
                                     for r in _qedv_runs) /
                                 max(len(_qedv_runs), 1)) ** 0.5,
                                4) if _qedv_runs else 0.0
                        )()
                    )()
                )()
            ))(),
            # std-dev of exploring-run lengths (0=uniform; high=erratic creative bursts)
            "quad_ideal_entry_coh3_variance": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qiecv_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qiecv_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qiecv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qiecv_entries=[
                        _coh3_steps[i]
                        for i in range(1, _qiecv_n)
                        if not (_coh3_steps[i-1] > _qiecv_c3m and _velocity_steps[i-1] < _qiecv_vm)
                        and (_coh3_steps[i] > _qiecv_c3m and _velocity_steps[i] < _qiecv_vm)
                    ]:
                        (lambda _qiecv_mean=(sum(_qiecv_entries)/max(len(_qiecv_entries),1)):
                            round(
                                (sum((v - _qiecv_mean)**2 for v in _qiecv_entries) /
                                 max(len(_qiecv_entries), 1)) ** 0.5,
                                4) if _qiecv_entries else 0.0
                        )()
                    )()
                )()
            ))(),
            # std-dev of coh3 values at ideal-quadrant entry (low=stable gate; high=noisy)
            "quad_exploring_exit_coh3": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qeec_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qeec_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qeec_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qeec_exits=[
                        _coh3_steps[i]
                        for i in range(1, _qeec_n)
                        if (_coh3_steps[i-1] > _qeec_c3m and _velocity_steps[i-1] >= _qeec_vm)
                        and not (_coh3_steps[i] > _qeec_c3m and _velocity_steps[i] >= _qeec_vm)
                    ]:
                        round(sum(_qeec_exits) / max(len(_qeec_exits), 1), 4)
                        if _qeec_exits else 0.0
                    )()
                )()
            ))(),
            # mean coh3 at moment of exiting exploring quadrant (higher=graceful exit)
            "quad_flat_exit_velocity": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qfev_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfev_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfev_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfev_exits=[
                        _velocity_steps[i]
                        for i in range(1, _qfev_n)
                        if (_coh3_steps[i-1] <= _qfev_c3m and _velocity_steps[i-1] < _qfev_vm)
                        and not (_coh3_steps[i] <= _qfev_c3m and _velocity_steps[i] < _qfev_vm)
                    ]:
                        round(sum(_qfev_exits) / max(len(_qfev_exits), 1), 4)
                        if _qfev_exits else 0.0
                    )()
                )()
            ))(),
            # mean velocity at moment of exiting flat quadrant (higher=sharper snap-out)
            "quad_ideal_to_exploring_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qiter_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qiter_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qiter_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qiter_exits=[
                        1 for i in range(1, _qiter_n)
                        if (_coh3_steps[i-1] > _qiter_c3m and _velocity_steps[i-1] < _qiter_vm)
                        and (_coh3_steps[i] > _qiter_c3m and _velocity_steps[i] >= _qiter_vm)
                    ],
                    _qiter_all=[
                        1 for i in range(1, _qiter_n)
                        if (_coh3_steps[i-1] > _qiter_c3m and _velocity_steps[i-1] < _qiter_vm)
                        and not (_coh3_steps[i] > _qiter_c3m and _velocity_steps[i] < _qiter_vm)
                    ]:
                        round(len(_qiter_exits) / max(len(_qiter_all), 1), 4)
                    )()
                )()
            ))(),
            # fraction of ideal exits that land in exploring (vs. other quadrants)
            "quad_drifting_to_flat_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdtf_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdtf_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdtf_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdtf_to=[
                        1 for i in range(1, _qdtf_n)
                        if (_coh3_steps[i-1] <= _qdtf_c3m and _velocity_steps[i-1] >= _qdtf_vm)
                        and (_coh3_steps[i] <= _qdtf_c3m and _velocity_steps[i] < _qdtf_vm)
                    ],
                    _qdtf_all=[
                        1 for i in range(1, _qdtf_n)
                        if (_coh3_steps[i-1] <= _qdtf_c3m and _velocity_steps[i-1] >= _qdtf_vm)
                        and not (_coh3_steps[i] <= _qdtf_c3m and _velocity_steps[i] >= _qdtf_vm)
                    ]:
                        round(len(_qdtf_to) / max(len(_qdtf_all), 1), 4)
                    )()
                )()
            ))(),
            # fraction of drifting exits that land in flat (0=never; 1=always)
            "quad_exploring_to_ideal_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qeti_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qeti_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qeti_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qeti_to=[
                        1 for i in range(1, _qeti_n)
                        if (_coh3_steps[i-1] > _qeti_c3m and _velocity_steps[i-1] >= _qeti_vm)
                        and (_coh3_steps[i] > _qeti_c3m and _velocity_steps[i] < _qeti_vm)
                    ],
                    _qeti_all=[
                        1 for i in range(1, _qeti_n)
                        if (_coh3_steps[i-1] > _qeti_c3m and _velocity_steps[i-1] >= _qeti_vm)
                        and not (_coh3_steps[i] > _qeti_c3m and _velocity_steps[i] >= _qeti_vm)
                    ]:
                        round(len(_qeti_to) / max(len(_qeti_all), 1), 4)
                    )()
                )()
            ))(),
            # fraction of exploring exits that land in ideal (0=never; 1=always)
            "quad_flat_to_drifting_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qftd_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qftd_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qftd_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qftd_to=[
                        1 for i in range(1, _qftd_n)
                        if (_coh3_steps[i-1] <= _qftd_c3m and _velocity_steps[i-1] < _qftd_vm)
                        and (_coh3_steps[i] <= _qftd_c3m and _velocity_steps[i] >= _qftd_vm)
                    ],
                    _qftd_all=[
                        1 for i in range(1, _qftd_n)
                        if (_coh3_steps[i-1] <= _qftd_c3m and _velocity_steps[i-1] < _qftd_vm)
                        and not (_coh3_steps[i] <= _qftd_c3m and _velocity_steps[i] < _qftd_vm)
                    ]:
                        round(len(_qftd_to) / max(len(_qftd_all), 1), 4)
                    )()
                )()
            ))(),
            # fraction of flat exits that land in drifting (0=never; 1=always)
            "quad_flat_to_exploring_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qfte_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfte_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfte_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfte_to=[
                        1 for i in range(1, _qfte_n)
                        if (_coh3_steps[i-1] <= _qfte_c3m and _velocity_steps[i-1] < _qfte_vm)
                        and (_coh3_steps[i] > _qfte_c3m and _velocity_steps[i] >= _qfte_vm)
                    ],
                    _qfte_all=[
                        1 for i in range(1, _qfte_n)
                        if (_coh3_steps[i-1] <= _qfte_c3m and _velocity_steps[i-1] < _qfte_vm)
                        and not (_coh3_steps[i] <= _qfte_c3m and _velocity_steps[i] < _qfte_vm)
                    ]:
                        round(len(_qfte_to) / max(len(_qfte_all), 1), 4)
                    )()
                )()
            ))(),
            # fraction of flat exits that land in exploring (0=never; 1=always)
            "quad_flat_to_ideal_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qfti_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfti_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfti_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfti_to=[
                        1 for i in range(1, _qfti_n)
                        if (_coh3_steps[i-1] <= _qfti_c3m and _velocity_steps[i-1] < _qfti_vm)
                        and (_coh3_steps[i] > _qfti_c3m and _velocity_steps[i] < _qfti_vm)
                    ],
                    _qfti_all=[
                        1 for i in range(1, _qfti_n)
                        if (_coh3_steps[i-1] <= _qfti_c3m and _velocity_steps[i-1] < _qfti_vm)
                        and not (_coh3_steps[i] <= _qfti_c3m and _velocity_steps[i] < _qfti_vm)
                    ]:
                        round(len(_qfti_to) / max(len(_qfti_all), 1), 4)
                    )()
                )()
            ))(),
            # fraction of flat exits that land directly in ideal (0=never; 1=always)
            "quad_drifting_to_exploring_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdte_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdte_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdte_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdte_to=[
                        1 for i in range(1, _qdte_n)
                        if (_coh3_steps[i-1] <= _qdte_c3m and _velocity_steps[i-1] >= _qdte_vm)
                        and (_coh3_steps[i] > _qdte_c3m and _velocity_steps[i] >= _qdte_vm)
                    ],
                    _qdte_all=[
                        1 for i in range(1, _qdte_n)
                        if (_coh3_steps[i-1] <= _qdte_c3m and _velocity_steps[i-1] >= _qdte_vm)
                        and not (_coh3_steps[i] <= _qdte_c3m and _velocity_steps[i] >= _qdte_vm)
                    ]:
                        round(len(_qdte_to) / max(len(_qdte_all), 1), 4)
                    )()
                )()
            ))(),
            # fraction of drifting exits that land in exploring (0=never; 1=always)
            "quad_drifting_to_ideal_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdti_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdti_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdti_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdti_to=[
                        1 for i in range(1, _qdti_n)
                        if (_coh3_steps[i-1] <= _qdti_c3m and _velocity_steps[i-1] >= _qdti_vm)
                        and (_coh3_steps[i] > _qdti_c3m and _velocity_steps[i] < _qdti_vm)
                    ],
                    _qdti_all=[
                        1 for i in range(1, _qdti_n)
                        if (_coh3_steps[i-1] <= _qdti_c3m and _velocity_steps[i-1] >= _qdti_vm)
                        and not (_coh3_steps[i] <= _qdti_c3m and _velocity_steps[i] >= _qdti_vm)
                    ]:
                        round(len(_qdti_to) / max(len(_qdti_all), 1), 4)
                    )()
                )()
            ))(),
            # fraction of drifting exits that land directly in ideal (0=never; 1=always)
            "quad_exploring_to_flat_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qetf_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qetf_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qetf_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qetf_to=[
                        1 for i in range(1, _qetf_n)
                        if (_coh3_steps[i-1] > _qetf_c3m and _velocity_steps[i-1] >= _qetf_vm)
                        and (_coh3_steps[i] <= _qetf_c3m and _velocity_steps[i] < _qetf_vm)
                    ],
                    _qetf_all=[
                        1 for i in range(1, _qetf_n)
                        if (_coh3_steps[i-1] > _qetf_c3m and _velocity_steps[i-1] >= _qetf_vm)
                        and not (_coh3_steps[i] > _qetf_c3m and _velocity_steps[i] >= _qetf_vm)
                    ]:
                        round(len(_qetf_to) / max(len(_qetf_all), 1), 4)
                    )()
                )()
            ))(),
            # fraction of exploring exits that collapse into flat (0=never; 1=always)
            "quad_exploring_to_drifting_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qetd_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qetd_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qetd_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qetd_to=[
                        1 for i in range(1, _qetd_n)
                        if (_coh3_steps[i-1] > _qetd_c3m and _velocity_steps[i-1] >= _qetd_vm)
                        and (_coh3_steps[i] <= _qetd_c3m and _velocity_steps[i] >= _qetd_vm)
                    ],
                    _qetd_all=[
                        1 for i in range(1, _qetd_n)
                        if (_coh3_steps[i-1] > _qetd_c3m and _velocity_steps[i-1] >= _qetd_vm)
                        and not (_coh3_steps[i] > _qetd_c3m and _velocity_steps[i] >= _qetd_vm)
                    ]:
                        round(len(_qetd_to) / max(len(_qetd_all), 1), 4)
                    )()
                )()
            ))(),
            # fraction of exploring exits that tip into drifting (0=never; 1=always)
            "quad_ideal_to_flat_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qitf_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qitf_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qitf_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qitf_to=[
                        1 for i in range(1, _qitf_n)
                        if (_coh3_steps[i-1] > _qitf_c3m and _velocity_steps[i-1] < _qitf_vm)
                        and (_coh3_steps[i] <= _qitf_c3m and _velocity_steps[i] < _qitf_vm)
                    ],
                    _qitf_all=[
                        1 for i in range(1, _qitf_n)
                        if (_coh3_steps[i-1] > _qitf_c3m and _velocity_steps[i-1] < _qitf_vm)
                        and not (_coh3_steps[i] > _qitf_c3m and _velocity_steps[i] < _qitf_vm)
                    ]:
                        round(len(_qitf_to) / max(len(_qitf_all), 1), 4)
                    )()
                )()
            ))(),
            # fraction of ideal exits that collapse into flat (0=never; 1=always)
            "quad_ideal_to_drifting_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qitd_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qitd_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qitd_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qitd_to=[
                        1 for i in range(1, _qitd_n)
                        if (_coh3_steps[i-1] > _qitd_c3m and _velocity_steps[i-1] < _qitd_vm)
                        and (_coh3_steps[i] <= _qitd_c3m and _velocity_steps[i] >= _qitd_vm)
                    ],
                    _qitd_all=[
                        1 for i in range(1, _qitd_n)
                        if (_coh3_steps[i-1] > _qitd_c3m and _velocity_steps[i-1] < _qitd_vm)
                        and not (_coh3_steps[i] > _qitd_c3m and _velocity_steps[i] < _qitd_vm)
                    ]:
                        round(len(_qitd_to) / max(len(_qitd_all), 1), 4)
                    )()
                )()
            ))(),
            # fraction of ideal exits that tip into drifting (0=never; 1=always)
            "quad_transition_entropy": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qte_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qte_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qte_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qte_labs=[
                        ("ideal" if _coh3_steps[i] > _qte_c3m and _velocity_steps[i] < _qte_vm
                         else "exploring" if _coh3_steps[i] > _qte_c3m
                         else "drifting" if _velocity_steps[i] >= _qte_vm
                         else "flat")
                        for i in range(_qte_n)
                    ]:
                        (lambda _qte_tpairs=[
                            (_qte_labs[i-1], _qte_labs[i])
                            for i in range(1, _qte_n)
                        ]:
                            (lambda _qte_uniq=list(set(
                                (_qte_labs[i-1], _qte_labs[i])
                                for i in range(1, _qte_n)
                            )):
                                (lambda _qte_cnts=[
                                    sum(1 for p in _qte_tpairs if p == u)
                                    for u in _qte_uniq
                                ],
                                _qte_tot=len(_qte_tpairs):
                                    round(
                                        -sum(
                                            (c / _qte_tot) * __import__('math').log2(c / _qte_tot)
                                            for c in _qte_cnts if c > 0
                                        ) if _qte_tot > 0 else 0.0,
                                        4)
                                )()
                            )()
                        )()
                    )()
                )()
            ))(),
            # Shannon entropy over observed quadrant-transition types (bits; max≈3.58)
            "quad_self_transition_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qstr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qstr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qstr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qstr_labs=[
                        ("ideal" if _coh3_steps[i] > _qstr_c3m and _velocity_steps[i] < _qstr_vm
                         else "exploring" if _coh3_steps[i] > _qstr_c3m
                         else "drifting" if _velocity_steps[i] >= _qstr_vm
                         else "flat")
                        for i in range(_qstr_n)
                    ]:
                        round(
                            sum(1 for i in range(1, _qstr_n)
                                if _qstr_labs[i-1] == _qstr_labs[i]) /
                            max(_qstr_n - 1, 1),
                            4)
                    )()
                )()
            ))(),
            # fraction of steps where the quadrant label doesn't change (self-loops)
            "quad_transition_matrix_skew": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qtms_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qtms_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qtms_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qtms_labs=[
                        ("ideal" if _coh3_steps[i] > _qtms_c3m and _velocity_steps[i] < _qtms_vm
                         else "exploring" if _coh3_steps[i] > _qtms_c3m
                         else "drifting" if _velocity_steps[i] >= _qtms_vm
                         else "flat")
                        for i in range(_qtms_n)
                    ]:
                        (lambda _qtms_src=["ideal", "exploring", "drifting", "flat"]:
                            (lambda _qtms_rows=[
                                [sum(1 for i in range(1, _qtms_n)
                                     if _qtms_labs[i-1] == src and _qtms_labs[i] == dst)
                                 for dst in ["ideal", "exploring", "drifting", "flat"]]
                                for src in _qtms_src
                            ]:
                                (lambda _qtms_rmax=[
                                    max(row) / max(sum(row), 1)
                                    for row in _qtms_rows if sum(row) > 0
                                ],
                                _qtms_rmin=[
                                    min(row) / max(sum(row), 1)
                                    for row in _qtms_rows if sum(row) > 0
                                ]:
                                    round(
                                        (max(_qtms_rmax) - min(_qtms_rmin))
                                        if _qtms_rmax else 0.0,
                                        4)
                                )()
                            )()
                        )()
                    )()
                )()
            ))(),
            # max row-prob minus min row-prob in transition matrix (0=uniform; 1=maximally skewed)
            "quad_ideal_run_confidence_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qirc_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qirc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qirc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qirc_confs=[
                        output_confs[i]
                        for i in range(_qirc_n)
                        if _coh3_steps[i] > _qirc_c3m and _velocity_steps[i] < _qirc_vm
                    ]:
                        round(sum(_qirc_confs) / max(len(_qirc_confs), 1), 4)
                        if _qirc_confs else 0.0
                    )()
                )()
            ))(),
            # mean confidence score during ideal-quadrant steps
            "quad_drifting_run_confidence_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qdrc_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qdrc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdrc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdrc_confs=[
                        output_confs[i]
                        for i in range(_qdrc_n)
                        if _coh3_steps[i] <= _qdrc_c3m and _velocity_steps[i] >= _qdrc_vm
                    ]:
                        round(sum(_qdrc_confs) / max(len(_qdrc_confs), 1), 4)
                        if _qdrc_confs else 0.0
                    )()
                )()
            ))(),
            # mean confidence score during drifting-quadrant steps
            "quad_exploring_run_confidence_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qerc_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qerc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qerc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qerc_confs=[
                        output_confs[i]
                        for i in range(_qerc_n)
                        if _coh3_steps[i] > _qerc_c3m and _velocity_steps[i] >= _qerc_vm
                    ]:
                        round(sum(_qerc_confs) / max(len(_qerc_confs), 1), 4)
                        if _qerc_confs else 0.0
                    )()
                )()
            ))(),
            # mean confidence score during exploring-quadrant steps
            "quad_flat_run_confidence_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qfrc_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qfrc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfrc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfrc_confs=[
                        output_confs[i]
                        for i in range(_qfrc_n)
                        if _coh3_steps[i] <= _qfrc_c3m and _velocity_steps[i] < _qfrc_vm
                    ]:
                        round(sum(_qfrc_confs) / max(len(_qfrc_confs), 1), 4)
                        if _qfrc_confs else 0.0
                    )()
                )()
            ))(),
            # mean confidence score during flat-quadrant steps
            "quad_confidence_gap": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qcg_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qcg_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qcg_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qcg_ic=[
                        output_confs[i]
                        for i in range(_qcg_n)
                        if _coh3_steps[i] > _qcg_c3m and _velocity_steps[i] < _qcg_vm
                    ],
                    _qcg_dc=[
                        output_confs[i]
                        for i in range(_qcg_n)
                        if _coh3_steps[i] <= _qcg_c3m and _velocity_steps[i] >= _qcg_vm
                    ]:
                        round(
                            (sum(_qcg_ic) / max(len(_qcg_ic), 1)
                             - sum(_qcg_dc) / max(len(_qcg_dc), 1))
                            if _qcg_ic and _qcg_dc else 0.0,
                            4)
                    )()
                )()
            ))(),
            # ideal_conf_mean minus drifting_conf_mean (positive = ideal is more confident)
            "quad_confidence_spread": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qcsp_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qcsp_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qcsp_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qcsp_pools=[
                        [output_confs[i] for i in range(_qcsp_n)
                         if ("ideal"     if _coh3_steps[i] > _qcsp_c3m and _velocity_steps[i] < _qcsp_vm else
                             "exploring" if _coh3_steps[i] > _qcsp_c3m else
                             "drifting"  if _velocity_steps[i] >= _qcsp_vm else "flat") == q]
                        for q in ["ideal", "exploring", "drifting", "flat"]
                    ]:
                        (lambda _qcsp_means=[sum(p)/len(p) for p in _qcsp_pools if p]:
                            round(
                                (sum((x - sum(_qcsp_means)/len(_qcsp_means))**2
                                     for x in _qcsp_means) / len(_qcsp_means)) ** 0.5
                                if len(_qcsp_means) >= 2 else 0.0,
                                4)
                        )()
                    )()
                )()
            ))(),
            # std-dev of the 4 per-quadrant confidence means
            "quad_coh3_spread": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qc3sp_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qc3sp_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qc3sp_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qc3sp_pools=[
                        [_coh3_steps[i] for i in range(_qc3sp_n)
                         if ("ideal"     if _coh3_steps[i] > _qc3sp_c3m and _velocity_steps[i] < _qc3sp_vm else
                             "exploring" if _coh3_steps[i] > _qc3sp_c3m else
                             "drifting"  if _velocity_steps[i] >= _qc3sp_vm else "flat") == q]
                        for q in ["ideal", "exploring", "drifting", "flat"]
                    ]:
                        (lambda _qc3sp_means=[sum(p)/len(p) for p in _qc3sp_pools if p]:
                            round(
                                (sum((x - sum(_qc3sp_means)/len(_qc3sp_means))**2
                                     for x in _qc3sp_means) / len(_qc3sp_means)) ** 0.5
                                if len(_qc3sp_means) >= 2 else 0.0,
                                4)
                        )()
                    )()
                )()
            ))(),
            # std-dev of the 4 per-quadrant coh3 means
            "quad_velocity_spread": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qvsp_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qvsp_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qvsp_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qvsp_pools=[
                        [_velocity_steps[i] for i in range(_qvsp_n)
                         if ("ideal"     if _coh3_steps[i] > _qvsp_c3m and _velocity_steps[i] < _qvsp_vm else
                             "exploring" if _coh3_steps[i] > _qvsp_c3m else
                             "drifting"  if _velocity_steps[i] >= _qvsp_vm else "flat") == q]
                        for q in ["ideal", "exploring", "drifting", "flat"]
                    ]:
                        (lambda _qvsp_means=[sum(p)/len(p) for p in _qvsp_pools if p]:
                            round(
                                (sum((x - sum(_qvsp_means)/len(_qvsp_means))**2
                                     for x in _qvsp_means) / len(_qvsp_means)) ** 0.5
                                if len(_qvsp_means) >= 2 else 0.0,
                                4)
                        )()
                    )()
                )()
            ))(),
            # std-dev of the 4 per-quadrant velocity means
            "quad_coh3_ideal_vs_flat_ratio": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qcifr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qcifr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qcifr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qcifr_ideal=[_coh3_steps[i] for i in range(_qcifr_n)
                                          if _coh3_steps[i] > _qcifr_c3m and _velocity_steps[i] < _qcifr_vm],
                           _qcifr_flat=[_coh3_steps[i] for i in range(_qcifr_n)
                                         if _coh3_steps[i] <= _qcifr_c3m and _velocity_steps[i] < _qcifr_vm]:
                        round(
                            (sum(_qcifr_ideal)/max(len(_qcifr_ideal), 1)) /
                            max(sum(_qcifr_flat)/max(len(_qcifr_flat), 1), 1e-6)
                            if _qcifr_ideal and _qcifr_flat else 0.0,
                            4)
                    )()
                )()
            ))(),
            # ratio of ideal coh3 mean to flat coh3 mean (higher = clearer coh3 separation)
            "quad_velocity_ideal_vs_drifting_ratio": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qvidr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qvidr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qvidr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qvidr_iv=[_velocity_steps[i] for i in range(_qvidr_n)
                                        if _coh3_steps[i] > _qvidr_c3m and _velocity_steps[i] < _qvidr_vm],
                           _qvidr_dv=[_velocity_steps[i] for i in range(_qvidr_n)
                                       if _coh3_steps[i] <= _qvidr_c3m and _velocity_steps[i] >= _qvidr_vm]:
                        round(
                            (sum(_qvidr_iv)/max(len(_qvidr_iv), 1)) /
                            max(sum(_qvidr_dv)/max(len(_qvidr_dv), 1), 1e-6)
                            if _qvidr_iv and _qvidr_dv else 0.0,
                            4)
                    )()
                )()
            ))(),
            # ratio of ideal velocity mean to drifting velocity mean (≤1 = ideal is slower = good)
            "quad_ideal_max_streak": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qims_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qims_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qims_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qims_labs=[
                        ("ideal" if _coh3_steps[i] > _qims_c3m and _velocity_steps[i] < _qims_vm
                         else "other")
                        for i in range(_qims_n)
                    ]:
                        max(
                            (sum(1 for _ in g) for k, g in
                             __import__('itertools').groupby(_qims_labs) if k == "ideal"),
                            default=0)
                    )()
                )()
            ))(),
            # longest consecutive run of ideal-quadrant steps
            "quad_drifting_max_streak": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdms_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdms_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdms_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdms_labs=[
                        ("drifting" if _coh3_steps[i] <= _qdms_c3m and _velocity_steps[i] >= _qdms_vm
                         else "other")
                        for i in range(_qdms_n)
                    ]:
                        max(
                            (sum(1 for _ in g) for k, g in
                             __import__('itertools').groupby(_qdms_labs) if k == "drifting"),
                            default=0)
                    )()
                )()
            ))(),
            # longest consecutive run of drifting-quadrant steps
            "quad_exploring_max_streak": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qems_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qems_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qems_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qems_labs=[
                        ("exploring" if _coh3_steps[i] > _qems_c3m and _velocity_steps[i] >= _qems_vm
                         else "other")
                        for i in range(_qems_n)
                    ]:
                        max(
                            (sum(1 for _ in g) for k, g in
                             __import__('itertools').groupby(_qems_labs) if k == "exploring"),
                            default=0)
                    )()
                )()
            ))(),
            # longest consecutive run of exploring-quadrant steps
            "quad_flat_max_streak": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qfms_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfms_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfms_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfms_labs=[
                        ("flat" if _coh3_steps[i] <= _qfms_c3m and _velocity_steps[i] < _qfms_vm
                         else "other")
                        for i in range(_qfms_n)
                    ]:
                        max(
                            (sum(1 for _ in g) for k, g in
                             __import__('itertools').groupby(_qfms_labs) if k == "flat"),
                            default=0)
                    )()
                )()
            ))(),
            # longest consecutive run of flat-quadrant steps
            "quad_dominant_streak_label": (lambda: (
                "none" if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdsl_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdsl_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdsl_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdsl_streaks={
                        q: max(
                            (sum(1 for _ in g)
                             for k, g in __import__('itertools').groupby([
                                 (q if ("ideal"     if _coh3_steps[i] > _qdsl_c3m and _velocity_steps[i] < _qdsl_vm else
                                        "exploring" if _coh3_steps[i] > _qdsl_c3m else
                                        "drifting"  if _velocity_steps[i] >= _qdsl_vm else "flat") == q
                                  else "other")
                                 for i in range(_qdsl_n)
                             ]) if k == q),
                            default=0)
                        for q in ["ideal", "exploring", "drifting", "flat"]
                    }:
                        max(_qdsl_streaks, key=lambda x: _qdsl_streaks[x])
                    )()
                )()
            ))(),
            # which quadrant has the longest max streak ("ideal"/"exploring"/"drifting"/"flat")
            "quad_ideal_run_count": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qirc2_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qirc2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qirc2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    sum(1 for k, _ in
                        __import__('itertools').groupby([
                            "ideal" if _coh3_steps[i] > _qirc2_c3m and _velocity_steps[i] < _qirc2_vm
                            else "other"
                            for i in range(_qirc2_n)
                        ]) if k == "ideal")
                )()
            ))(),
            # number of distinct ideal-quadrant episodes
            "quad_drifting_run_count": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdrc2_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdrc2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdrc2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    sum(1 for k, _ in
                        __import__('itertools').groupby([
                            "drifting" if _coh3_steps[i] <= _qdrc2_c3m and _velocity_steps[i] >= _qdrc2_vm
                            else "other"
                            for i in range(_qdrc2_n)
                        ]) if k == "drifting")
                )()
            ))(),
            # number of distinct drifting-quadrant episodes
            "quad_ideal_mean_streak": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qimsb_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qimsb_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qimsb_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qimsb_runs=[
                        sum(1 for _ in g)
                        for k, g in __import__('itertools').groupby([
                            "ideal" if _coh3_steps[i] > _qimsb_c3m and _velocity_steps[i] < _qimsb_vm
                            else "other"
                            for i in range(_qimsb_n)
                        ]) if k == "ideal"
                    ]:
                        round(sum(_qimsb_runs) / max(len(_qimsb_runs), 1), 4)
                        if _qimsb_runs else 0.0
                    )()
                )()
            ))(),
            # mean length of ideal-quadrant runs
            "quad_drifting_mean_streak": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdmsb_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdmsb_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdmsb_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdmsb_runs=[
                        sum(1 for _ in g)
                        for k, g in __import__('itertools').groupby([
                            "drifting" if _coh3_steps[i] <= _qdmsb_c3m and _velocity_steps[i] >= _qdmsb_vm
                            else "other"
                            for i in range(_qdmsb_n)
                        ]) if k == "drifting"
                    ]:
                        round(sum(_qdmsb_runs) / max(len(_qdmsb_runs), 1), 4)
                        if _qdmsb_runs else 0.0
                    )()
                )()
            ))(),
            # mean length of drifting-quadrant runs
            "quad_streak_variability": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qsv2_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qsv2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qsv2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qsv2_labs=[
                        ("ideal"     if _coh3_steps[i] > _qsv2_c3m and _velocity_steps[i] < _qsv2_vm else
                         "exploring" if _coh3_steps[i] > _qsv2_c3m else
                         "drifting"  if _velocity_steps[i] >= _qsv2_vm else "flat")
                        for i in range(_qsv2_n)
                    ]:
                        (lambda _qsv2_runs=[
                            sum(1 for _ in g)
                            for k, g in __import__('itertools').groupby(_qsv2_labs)
                            if k != "other"
                        ]:
                            (lambda _qsv2_mu=sum(_qsv2_runs)/max(len(_qsv2_runs),1):
                                round(
                                    (sum((x-_qsv2_mu)**2 for x in _qsv2_runs)/max(len(_qsv2_runs),1))**0.5
                                    if len(_qsv2_runs) >= 2 else 0.0,
                                    4)
                            )()
                        )()
                    )()
                )()
            ))(),
            # std-dev of all quadrant run lengths (high=runs vary wildly; low=uniform pacing)
            "quad_early_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qei_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qei_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qei_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qei_half=max(_qei_n // 2, 1):
                        round(
                            sum(1 for i in range(_qei_half)
                                if _coh3_steps[i] > _qei_c3m and _velocity_steps[i] < _qei_vm)
                            / _qei_half,
                            4)
                    )()
                )()
            ))(),
            # fraction of ideal steps in the first half of generation
            "quad_late_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qli_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qli_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qli_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qli_half=max(_qli_n // 2, 1):
                        round(
                            sum(1 for i in range(_qli_half, _qli_n)
                                if _coh3_steps[i] > _qli_c3m and _velocity_steps[i] < _qli_vm)
                            / max(_qli_n - _qli_half, 1),
                            4)
                    )()
                )()
            ))(),
            # fraction of ideal steps in the second half of generation
            "quad_ideal_frac_trend": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qift_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qift_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qift_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qift_half=max(_qift_n // 2, 1):
                        round(
                            sum(1 for i in range(_qift_half, _qift_n)
                                if _coh3_steps[i] > _qift_c3m and _velocity_steps[i] < _qift_vm)
                            / max(_qift_n - _qift_half, 1)
                            -
                            sum(1 for i in range(_qift_half)
                                if _coh3_steps[i] > _qift_c3m and _velocity_steps[i] < _qift_vm)
                            / _qift_half,
                            4)
                    )()
                )()
            ))(),
            # late_ideal_frac − early_ideal_frac (positive = improving; negative = degrading)
            "quad_early_drifting_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qed_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qed_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qed_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qed_half=max(_qed_n // 2, 1):
                        round(
                            sum(1 for i in range(_qed_half)
                                if _coh3_steps[i] <= _qed_c3m and _velocity_steps[i] >= _qed_vm)
                            / _qed_half,
                            4)
                    )()
                )()
            ))(),
            # fraction of drifting steps in the first half of generation
            "quad_late_drifting_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qld_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qld_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qld_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qld_half=max(_qld_n // 2, 1):
                        round(
                            sum(1 for i in range(_qld_half, _qld_n)
                                if _coh3_steps[i] <= _qld_c3m and _velocity_steps[i] >= _qld_vm)
                            / max(_qld_n - _qld_half, 1),
                            4)
                    )()
                )()
            ))(),
            # fraction of drifting steps in the second half of generation
            "quad_weighted_ideal_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qwis_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qwis_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qwis_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qwis_tot=sum(output_confs[:_qwis_n]):
                        round(
                            sum(output_confs[i] for i in range(_qwis_n)
                                if _coh3_steps[i] > _qwis_c3m and _velocity_steps[i] < _qwis_vm)
                            / max(_qwis_tot, 1e-9),
                            4)
                    )()
                )()
            ))(),
            # fraction of total confidence mass that falls on ideal-quadrant steps
            "quad_weighted_drifting_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qwds_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qwds_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qwds_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qwds_tot=sum(output_confs[:_qwds_n]):
                        round(
                            sum(output_confs[i] for i in range(_qwds_n)
                                if _coh3_steps[i] <= _qwds_c3m and _velocity_steps[i] >= _qwds_vm)
                            / max(_qwds_tot, 1e-9),
                            4)
                    )()
                )()
            ))(),
            # fraction of total confidence mass that falls on drifting-quadrant steps
            "quad_confidence_mass_ratio": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qcmr_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qcmr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qcmr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qcmr_tot=sum(output_confs[:_qcmr_n]):
                        (lambda _qcmr_ideal=sum(output_confs[i] for i in range(_qcmr_n)
                                                if _coh3_steps[i] > _qcmr_c3m and _velocity_steps[i] < _qcmr_vm),
                               _qcmr_drift=sum(output_confs[i] for i in range(_qcmr_n)
                                               if _coh3_steps[i] <= _qcmr_c3m and _velocity_steps[i] >= _qcmr_vm):
                            round(_qcmr_ideal / max(_qcmr_drift, 1e-9), 4)
                        )()
                    )()
                )()
            ))(),
            # ratio of ideal conf mass to drifting conf mass (>1 = ideal holds more confidence weight)
            "quad_ideal_first_step": (lambda: (
                -1 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qifs_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qifs_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qifs_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    next((i for i in range(_qifs_n)
                          if _coh3_steps[i] > _qifs_c3m and _velocity_steps[i] < _qifs_vm), -1)
                )()
            ))(),
            # first step index at which ideal quadrant is reached (-1 if never)
            "quad_drifting_first_step": (lambda: (
                -1 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdfs_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdfs_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdfs_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    next((i for i in range(_qdfs_n)
                          if _coh3_steps[i] <= _qdfs_c3m and _velocity_steps[i] >= _qdfs_vm), -1)
                )()
            ))(),
            # first step index at which drifting quadrant is reached (-1 if never)
            "quad_early_third_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 6
                else (lambda _qet_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qet_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qet_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qet_t=max(_qet_n // 3, 1):
                        round(
                            sum(1 for i in range(_qet_t)
                                if _coh3_steps[i] > _qet_c3m and _velocity_steps[i] < _qet_vm)
                            / _qet_t, 4)
                    )()
                )()
            ))(),
            # fraction of ideal steps in first third of generation
            "quad_mid_third_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 6
                else (lambda _qmt_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qmt_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qmt_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qmt_t=max(_qmt_n // 3, 1):
                        round(
                            sum(1 for i in range(_qmt_t, 2 * _qmt_t)
                                if _coh3_steps[i] > _qmt_c3m and _velocity_steps[i] < _qmt_vm)
                            / _qmt_t, 4)
                    )()
                )()
            ))(),
            # fraction of ideal steps in middle third of generation
            "quad_late_third_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 6
                else (lambda _qlt_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qlt_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qlt_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qlt_t=max(_qlt_n // 3, 1):
                        round(
                            sum(1 for i in range(2 * _qlt_t, _qlt_n)
                                if _coh3_steps[i] > _qlt_c3m and _velocity_steps[i] < _qlt_vm)
                            / max(_qlt_n - 2 * _qlt_t, 1), 4)
                    )()
                )()
            ))(),
            # fraction of ideal steps in last third of generation
            "quad_health_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qhs_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qhs_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qhs_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qhs_ideal=[
                            output_confs[i] for i in range(_qhs_n)
                            if _coh3_steps[i] > _qhs_c3m and _velocity_steps[i] < _qhs_vm
                         ],
                         _qhs_drift=[
                            output_confs[i] for i in range(_qhs_n)
                            if _coh3_steps[i] <= _qhs_c3m and _velocity_steps[i] >= _qhs_vm
                         ]:
                        round(
                            # ideal_frac × avg_ideal_conf × (1 − drifting_frac) × conf_gap_boost
                            (len(_qhs_ideal) / _qhs_n)
                            * (sum(_qhs_ideal) / max(len(_qhs_ideal), 1))
                            * (1.0 - len(_qhs_drift) / _qhs_n)
                            * (1.0 + max(
                                (sum(_qhs_ideal) / max(len(_qhs_ideal), 1))
                                - (sum(_qhs_drift) / max(len(_qhs_drift), 1))
                                if _qhs_drift else 0.0, 0.0))
                            if _qhs_ideal else 0.0,
                            4)
                    )()
                )()
            ))(),
            # composite quadrant health: ideal_frac × ideal_conf × (1−drift_frac) × conf_gap_boost
            "quad_early_third_drifting_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 6
                else (lambda _qetd_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qetd_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qetd_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qetd_t=max(_qetd_n // 3, 1):
                        round(
                            sum(1 for i in range(_qetd_t)
                                if _coh3_steps[i] <= _qetd_c3m and _velocity_steps[i] >= _qetd_vm)
                            / _qetd_t, 4)
                    )()
                )()
            ))(),
            # fraction of drifting steps in first third of generation
            "quad_weighted_ideal_coh3": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qwic_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qwic_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qwic_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qwic_tot=sum(_coh3_steps[:_qwic_n]):
                        round(
                            sum(_coh3_steps[i] for i in range(_qwic_n)
                                if _coh3_steps[i] > _qwic_c3m and _velocity_steps[i] < _qwic_vm)
                            / max(_qwic_tot, 1e-9), 4)
                    )()
                )()
            ))(),
            # fraction of total coh3 mass that falls on ideal-quadrant steps
            "quad_weighted_drifting_coh3": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qwdc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qwdc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qwdc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qwdc_tot=sum(_coh3_steps[:_qwdc_n]):
                        round(
                            sum(_coh3_steps[i] for i in range(_qwdc_n)
                                if _coh3_steps[i] <= _qwdc_c3m and _velocity_steps[i] >= _qwdc_vm)
                            / max(_qwdc_tot, 1e-9), 4)
                    )()
                )()
            ))(),
            # fraction of total coh3 mass that falls on drifting-quadrant steps
            "quad_coh3_mass_ratio": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qcmr2_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qcmr2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qcmr2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qcmr2_i=sum(_coh3_steps[i] for i in range(_qcmr2_n)
                                         if _coh3_steps[i] > _qcmr2_c3m and _velocity_steps[i] < _qcmr2_vm),
                           _qcmr2_d=sum(_coh3_steps[i] for i in range(_qcmr2_n)
                                         if _coh3_steps[i] <= _qcmr2_c3m and _velocity_steps[i] >= _qcmr2_vm):
                        round(_qcmr2_i / max(_qcmr2_d, 1e-9), 4)
                    )()
                )()
            ))(),
            # ratio of ideal coh3 mass to drifting coh3 mass (>1 = ideal holds more coh3 weight)
            "quad_ideal_coh3_momentum": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 6
                else (lambda _qicm_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qicm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qicm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qicm_ideal=[
                        (i, _coh3_steps[i]) for i in range(_qicm_n)
                        if _coh3_steps[i] > _qicm_c3m and _velocity_steps[i] < _qicm_vm
                    ]:
                        round(
                            (sum(v for _, v in _qicm_ideal[-3:]) / max(len(_qicm_ideal[-3:]), 1)
                             - sum(v for _, v in _qicm_ideal[:3])  / max(len(_qicm_ideal[:3]),  1))
                            if len(_qicm_ideal) >= 3 else 0.0,
                            4)
                    )()
                )()
            ))(),
            # mean coh3 of last 3 ideal steps minus mean coh3 of first 3 ideal steps (positive=improving)
            "quad_weighted_ideal_velocity": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qwiv_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qwiv_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qwiv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qwiv_tot=sum(abs(v) for v in _velocity_steps[:_qwiv_n]):
                        round(
                            sum(abs(_velocity_steps[i]) for i in range(_qwiv_n)
                                if _coh3_steps[i] > _qwiv_c3m and _velocity_steps[i] < _qwiv_vm)
                            / max(_qwiv_tot, 1e-9), 4)
                    )()
                )()
            ))(),
            # fraction of total |velocity| mass that falls on ideal-quadrant steps (should be LOW = good)
            "quad_weighted_drifting_velocity": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qwdv_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qwdv_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qwdv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qwdv_tot=sum(abs(v) for v in _velocity_steps[:_qwdv_n]):
                        round(
                            sum(abs(_velocity_steps[i]) for i in range(_qwdv_n)
                                if _coh3_steps[i] <= _qwdv_c3m and _velocity_steps[i] >= _qwdv_vm)
                            / max(_qwdv_tot, 1e-9), 4)
                    )()
                )()
            ))(),
            # fraction of total |velocity| mass that falls on drifting-quadrant steps (should be HIGH)
            "quad_velocity_mass_ratio": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qvmr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qvmr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qvmr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qvmr_i=sum(abs(_velocity_steps[i]) for i in range(_qvmr_n)
                                         if _coh3_steps[i] > _qvmr_c3m and _velocity_steps[i] < _qvmr_vm),
                           _qvmr_d=sum(abs(_velocity_steps[i]) for i in range(_qvmr_n)
                                         if _coh3_steps[i] <= _qvmr_c3m and _velocity_steps[i] >= _qvmr_vm):
                        round(_qvmr_i / max(_qvmr_d, 1e-9), 4)
                    )()
                )()
            ))(),
            # ratio of ideal |vel| mass to drifting |vel| mass (<1 = ideal is slower = good)
            "quad_ideal_velocity_momentum": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 6
                else (lambda _qivm_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qivm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qivm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qivm_ideal=[
                        (i, _velocity_steps[i]) for i in range(_qivm_n)
                        if _coh3_steps[i] > _qivm_c3m and _velocity_steps[i] < _qivm_vm
                    ]:
                        round(
                            (sum(v for _, v in _qivm_ideal[-3:]) / max(len(_qivm_ideal[-3:]), 1)
                             - sum(v for _, v in _qivm_ideal[:3])  / max(len(_qivm_ideal[:3]),  1))
                            if len(_qivm_ideal) >= 3 else 0.0,
                            4)
                    )()
                )()
            ))(),
            # mean vel of last 3 ideal steps minus first 3 (positive=speeding up; negative=slowing=good)
            "quad_drifting_coh3_momentum": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 6
                else (lambda _qdcm_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdcm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdcm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdcm_drift=[
                        (i, _coh3_steps[i]) for i in range(_qdcm_n)
                        if _coh3_steps[i] <= _qdcm_c3m and _velocity_steps[i] >= _qdcm_vm
                    ]:
                        round(
                            (sum(v for _, v in _qdcm_drift[-3:]) / max(len(_qdcm_drift[-3:]), 1)
                             - sum(v for _, v in _qdcm_drift[:3])  / max(len(_qdcm_drift[:3]),  1))
                            if len(_qdcm_drift) >= 3 else 0.0,
                            4)
                    )()
                )()
            ))(),
            # coh3 trend in drifting steps: last3 minus first3 (positive=coh3 worsening in drift)
            "quad_drifting_velocity_momentum": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 6
                else (lambda _qdvm_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdvm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdvm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdvm_drift=[
                        (i, _velocity_steps[i]) for i in range(_qdvm_n)
                        if _coh3_steps[i] <= _qdvm_c3m and _velocity_steps[i] >= _qdvm_vm
                    ]:
                        round(
                            (sum(v for _, v in _qdvm_drift[-3:]) / max(len(_qdvm_drift[-3:]), 1)
                             - sum(v for _, v in _qdvm_drift[:3])  / max(len(_qdvm_drift[:3]),  1))
                            if len(_qdvm_drift) >= 3 else 0.0,
                            4)
                    )()
                )()
            ))(),
            # vel trend in drifting steps: last3 minus first3 (positive=drift accelerating=bad)
            "quad_exploring_coh3_momentum": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 6
                else (lambda _qecm_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qecm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qecm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qecm_expl=[
                        (i, _coh3_steps[i]) for i in range(_qecm_n)
                        if _coh3_steps[i] > _qecm_c3m and _velocity_steps[i] >= _qecm_vm
                    ]:
                        round(
                            (sum(v for _, v in _qecm_expl[-3:]) / max(len(_qecm_expl[-3:]), 1)
                             - sum(v for _, v in _qecm_expl[:3])  / max(len(_qecm_expl[:3]),  1))
                            if len(_qecm_expl) >= 3 else 0.0,
                            4)
                    )()
                )()
            ))(),
            # coh3 trend in exploring steps: last3 minus first3 (positive=coherence rising during exploration)
            "quad_inter_ideal_gap": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qiig_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qiig_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qiig_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qiig_labs=[
                        "ideal" if _coh3_steps[i] > _qiig_c3m and _velocity_steps[i] < _qiig_vm
                        else "other"
                        for i in range(_qiig_n)
                    ]:
                        (lambda _qiig_ends=[
                            i for i in range(1, _qiig_n)
                            if _qiig_labs[i-1] == "ideal" and _qiig_labs[i] != "ideal"
                        ],
                        _qiig_starts=[
                            i for i in range(1, _qiig_n)
                            if _qiig_labs[i-1] != "ideal" and _qiig_labs[i] == "ideal"
                        ]:
                            round(
                                sum(_qiig_starts[j] - _qiig_ends[j-1] if j > 0 else _qiig_starts[0]
                                    for j in range(min(len(_qiig_starts), len(_qiig_ends) + 1)))
                                / max(len(_qiig_starts), 1), 4)
                            if _qiig_starts else 0.0
                        )()
                    )()
                )()
            ))(),
            # mean number of steps between the end of one ideal run and the start of the next
            "quad_inter_drifting_gap": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qidg_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qidg_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qidg_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qidg_labs=[
                        "drift" if _coh3_steps[i] <= _qidg_c3m and _velocity_steps[i] >= _qidg_vm
                        else "other"
                        for i in range(_qidg_n)
                    ]:
                        (lambda _qidg_ends=[
                            i for i in range(1, _qidg_n)
                            if _qidg_labs[i-1] == "drift" and _qidg_labs[i] != "drift"
                        ],
                        _qidg_starts=[
                            i for i in range(1, _qidg_n)
                            if _qidg_labs[i-1] != "drift" and _qidg_labs[i] == "drift"
                        ]:
                            round(
                                sum(_qidg_starts[j] - _qidg_ends[j-1] if j > 0 else _qidg_starts[0]
                                    for j in range(min(len(_qidg_starts), len(_qidg_ends) + 1)))
                                / max(len(_qidg_starts), 1), 4)
                            if _qidg_starts else 0.0
                        )()
                    )()
                )()
            ))(),
            # mean number of steps between the end of one drifting run and the start of the next
            "quad_flat_coh3_momentum": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 6
                else (lambda _qfcm_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfcm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfcm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfcm_flat=[
                        (i, _coh3_steps[i]) for i in range(_qfcm_n)
                        if _coh3_steps[i] <= _qfcm_c3m and _velocity_steps[i] < _qfcm_vm
                    ]:
                        round(
                            (sum(v for _, v in _qfcm_flat[-3:]) / max(len(_qfcm_flat[-3:]), 1)
                             - sum(v for _, v in _qfcm_flat[:3])  / max(len(_qfcm_flat[:3]),  1))
                            if len(_qfcm_flat) >= 3 else 0.0,
                            4)
                    )()
                )()
            ))(),
            # coh3 trend in flat steps: last3 minus first3 (positive=coh3 recovering; negative=worsening)
            "quad_flat_velocity_momentum": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 6
                else (lambda _qfvm_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfvm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfvm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfvm_flat=[
                        (i, _velocity_steps[i]) for i in range(_qfvm_n)
                        if _coh3_steps[i] <= _qfvm_c3m and _velocity_steps[i] < _qfvm_vm
                    ]:
                        round(
                            (sum(v for _, v in _qfvm_flat[-3:]) / max(len(_qfvm_flat[-3:]), 1)
                             - sum(v for _, v in _qfvm_flat[:3])  / max(len(_qfvm_flat[:3]),  1))
                            if len(_qfvm_flat) >= 3 else 0.0,
                            4)
                    )()
                )()
            ))(),
            # vel trend in flat steps: last3 minus first3 (positive=velocity creeping up=pre-escape signal)
            "quad_ideal_to_drift_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qitd_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qitd_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qitd_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qitd_labs=[
                        ("ideal"     if _coh3_steps[i] > _qitd_c3m and _velocity_steps[i] < _qitd_vm else
                         "exploring" if _coh3_steps[i] > _qitd_c3m else
                         "drifting"  if _velocity_steps[i] >= _qitd_vm else "flat")
                        for i in range(_qitd_n)
                    ]:
                        (lambda _qitd_exits=sum(1 for i in range(1, _qitd_n)
                                                if _qitd_labs[i-1] == "ideal" and _qitd_labs[i] != "ideal"),
                               _qitd_to_d=sum(1 for i in range(1, _qitd_n)
                                              if _qitd_labs[i-1] == "ideal" and _qitd_labs[i] == "drifting"):
                            round(_qitd_to_d / max(_qitd_exits, 1), 4) if _qitd_exits else 0.0
                        )()
                    )()
                )()
            ))(),
            # fraction of ideal exits that go directly to drifting (low=better)
            "quad_drift_to_ideal_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdti_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdti_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdti_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdti_labs=[
                        ("ideal"     if _coh3_steps[i] > _qdti_c3m and _velocity_steps[i] < _qdti_vm else
                         "exploring" if _coh3_steps[i] > _qdti_c3m else
                         "drifting"  if _velocity_steps[i] >= _qdti_vm else "flat")
                        for i in range(_qdti_n)
                    ]:
                        (lambda _qdti_exits=sum(1 for i in range(1, _qdti_n)
                                                if _qdti_labs[i-1] == "drifting" and _qdti_labs[i] != "drifting"),
                               _qdti_to_i=sum(1 for i in range(1, _qdti_n)
                                              if _qdti_labs[i-1] == "drifting" and _qdti_labs[i] == "ideal"):
                            round(_qdti_to_i / max(_qdti_exits, 1), 4) if _qdti_exits else 0.0
                        )()
                    )()
                )()
            ))(),
            # fraction of drifting exits that go directly to ideal (high=recovery rate is good)
            "quad_exploring_to_ideal_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qeti_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qeti_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qeti_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qeti_labs=[
                        ("ideal"     if _coh3_steps[i] > _qeti_c3m and _velocity_steps[i] < _qeti_vm else
                         "exploring" if _coh3_steps[i] > _qeti_c3m else
                         "drifting"  if _velocity_steps[i] >= _qeti_vm else "flat")
                        for i in range(_qeti_n)
                    ]:
                        (lambda _qeti_exits=sum(1 for i in range(1, _qeti_n)
                                                if _qeti_labs[i-1] == "exploring" and _qeti_labs[i] != "exploring"),
                               _qeti_to_i=sum(1 for i in range(1, _qeti_n)
                                              if _qeti_labs[i-1] == "exploring" and _qeti_labs[i] == "ideal"):
                            round(_qeti_to_i / max(_qeti_exits, 1), 4) if _qeti_exits else 0.0
                        )()
                    )()
                )()
            ))(),
            # fraction of exploring exits that go to ideal (high=exploration converts to quality)
            "quad_flat_to_ideal_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qfti_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfti_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfti_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfti_labs=[
                        ("ideal"     if _coh3_steps[i] > _qfti_c3m and _velocity_steps[i] < _qfti_vm else
                         "exploring" if _coh3_steps[i] > _qfti_c3m else
                         "drifting"  if _velocity_steps[i] >= _qfti_vm else "flat")
                        for i in range(_qfti_n)
                    ]:
                        (lambda _qfti_exits=sum(1 for i in range(1, _qfti_n)
                                                if _qfti_labs[i-1] == "flat" and _qfti_labs[i] != "flat"),
                               _qfti_to_i=sum(1 for i in range(1, _qfti_n)
                                              if _qfti_labs[i-1] == "flat" and _qfti_labs[i] == "ideal"):
                            round(_qfti_to_i / max(_qfti_exits, 1), 4) if _qfti_exits else 0.0
                        )()
                    )()
                )()
            ))(),
            # fraction of flat exits going directly to ideal (stagnation escape rate)
            "quad_ideal_to_exploring_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qite_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qite_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qite_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qite_labs=[
                        ("ideal"     if _coh3_steps[i] > _qite_c3m and _velocity_steps[i] < _qite_vm else
                         "exploring" if _coh3_steps[i] > _qite_c3m else
                         "drifting"  if _velocity_steps[i] >= _qite_vm else "flat")
                        for i in range(_qite_n)
                    ]:
                        (lambda _qite_exits=sum(1 for i in range(1, _qite_n)
                                                if _qite_labs[i-1] == "ideal" and _qite_labs[i] != "ideal"),
                               _qite_to_e=sum(1 for i in range(1, _qite_n)
                                              if _qite_labs[i-1] == "ideal" and _qite_labs[i] == "exploring"):
                            round(_qite_to_e / max(_qite_exits, 1), 4) if _qite_exits else 0.0
                        )()
                    )()
                )()
            ))(),
            # fraction of ideal exits going to exploring (partial destabilisation vs. full drift)
            "quad_ideal_persistence": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qip_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qip_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qip_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qip_labs=[
                        ("ideal"     if _coh3_steps[i] > _qip_c3m and _velocity_steps[i] < _qip_vm else
                         "exploring" if _coh3_steps[i] > _qip_c3m else
                         "drifting"  if _velocity_steps[i] >= _qip_vm else "flat")
                        for i in range(_qip_n)
                    ]:
                        (lambda _qip_ideal_trans=sum(1 for i in range(1, _qip_n)
                                                     if _qip_labs[i-1] == "ideal"),
                               _qip_ideal_self=sum(1 for i in range(1, _qip_n)
                                                   if _qip_labs[i-1] == "ideal" and _qip_labs[i] == "ideal"):
                            round(_qip_ideal_self / max(_qip_ideal_trans, 1), 4) if _qip_ideal_trans else 0.0
                        )()
                    )()
                )()
            ))(),
            # ideal→ideal self-transition fraction (how sticky ideal quadrant is; high=sustained flow)
            "quad_net_recovery_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qnrr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qnrr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qnrr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qnrr_labs=[
                        ("ideal"     if _coh3_steps[i] > _qnrr_c3m and _velocity_steps[i] < _qnrr_vm else
                         "exploring" if _coh3_steps[i] > _qnrr_c3m else
                         "drifting"  if _velocity_steps[i] >= _qnrr_vm else "flat")
                        for i in range(_qnrr_n)
                    ]:
                        (lambda _qnrr_ie=sum(1 for i in range(1, _qnrr_n)
                                             if _qnrr_labs[i-1] == "ideal" and _qnrr_labs[i] != "ideal"),
                               _qnrr_itd=sum(1 for i in range(1, _qnrr_n)
                                             if _qnrr_labs[i-1] == "ideal" and _qnrr_labs[i] == "drifting"),
                               _qnrr_de=sum(1 for i in range(1, _qnrr_n)
                                            if _qnrr_labs[i-1] == "drifting" and _qnrr_labs[i] != "drifting"),
                               _qnrr_dti=sum(1 for i in range(1, _qnrr_n)
                                             if _qnrr_labs[i-1] == "drifting" and _qnrr_labs[i] == "ideal"):
                            round(
                                (_qnrr_dti / max(_qnrr_de, 1) if _qnrr_de else 0.0)
                                - (_qnrr_itd / max(_qnrr_ie, 1) if _qnrr_ie else 0.0),
                                4)
                        )()
                    )()
                )()
            ))(),
            # drift_to_ideal_rate − ideal_to_drift_rate (positive=recovers more than it decays)
            "quad_exploring_persistence": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qep_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qep_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qep_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qep_labs=[
                        ("ideal"     if _coh3_steps[i] > _qep_c3m and _velocity_steps[i] < _qep_vm else
                         "exploring" if _coh3_steps[i] > _qep_c3m else
                         "drifting"  if _velocity_steps[i] >= _qep_vm else "flat")
                        for i in range(_qep_n)
                    ]:
                        (lambda _qep_trans=sum(1 for i in range(1, _qep_n)
                                               if _qep_labs[i-1] == "exploring"),
                               _qep_self=sum(1 for i in range(1, _qep_n)
                                             if _qep_labs[i-1] == "exploring" and _qep_labs[i] == "exploring"):
                            round(_qep_self / max(_qep_trans, 1), 4) if _qep_trans else 0.0
                        )()
                    )()
                )()
            ))(),
            # exploring→exploring self-loop fraction (how sticky the exploring quadrant is)
            "quad_drifting_persistence": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdp_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdp_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdp_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdp_labs=[
                        ("ideal"     if _coh3_steps[i] > _qdp_c3m and _velocity_steps[i] < _qdp_vm else
                         "exploring" if _coh3_steps[i] > _qdp_c3m else
                         "drifting"  if _velocity_steps[i] >= _qdp_vm else "flat")
                        for i in range(_qdp_n)
                    ]:
                        (lambda _qdp_trans=sum(1 for i in range(1, _qdp_n)
                                               if _qdp_labs[i-1] == "drifting"),
                               _qdp_self=sum(1 for i in range(1, _qdp_n)
                                             if _qdp_labs[i-1] == "drifting" and _qdp_labs[i] == "drifting"):
                            round(_qdp_self / max(_qdp_trans, 1), 4) if _qdp_trans else 0.0
                        )()
                    )()
                )()
            ))(),
            # drifting→drifting self-loop fraction (how sticky the drifting quadrant is)
            "quad_flat_persistence": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qfp_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfp_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfp_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfp_labs=[
                        ("ideal"     if _coh3_steps[i] > _qfp_c3m and _velocity_steps[i] < _qfp_vm else
                         "exploring" if _coh3_steps[i] > _qfp_c3m else
                         "drifting"  if _velocity_steps[i] >= _qfp_vm else "flat")
                        for i in range(_qfp_n)
                    ]:
                        (lambda _qfp_trans=sum(1 for i in range(1, _qfp_n)
                                               if _qfp_labs[i-1] == "flat"),
                               _qfp_self=sum(1 for i in range(1, _qfp_n)
                                             if _qfp_labs[i-1] == "flat" and _qfp_labs[i] == "flat"):
                            round(_qfp_self / max(_qfp_trans, 1), 4) if _qfp_trans else 0.0
                        )()
                    )()
                )()
            ))(),
            # flat→flat self-loop fraction (how sticky the flat/stagnant quadrant is)
            "quad_symmetry_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qss_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qss_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qss_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qss_counts={
                        q: sum(1 for i in range(_qss_n) if (
                            ("ideal"     if _coh3_steps[i] > _qss_c3m and _velocity_steps[i] < _qss_vm else
                             "exploring" if _coh3_steps[i] > _qss_c3m else
                             "drifting"  if _velocity_steps[i] >= _qss_vm else "flat") == q))
                        for q in ["ideal", "exploring", "drifting", "flat"]
                    }:
                        (lambda _qss_fracs=[c / _qss_n for c in _qss_counts.values()]:
                            round(
                                -sum(p * __import__('math').log2(max(p, 1e-9)) for p in _qss_fracs)
                                / __import__('math').log2(4),
                                4)
                        )()
                    )()
                )()
            ))(),
            # quadrant distribution entropy normalised to [0,1] (1=perfectly balanced; 0=one quadrant dominates)
            "quad_fingerprint": (lambda: (
                "none" if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qfp2_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfp2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfp2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfp2_fracs={
                        "I": sum(1 for i in range(_qfp2_n)
                                 if _coh3_steps[i] > _qfp2_c3m and _velocity_steps[i] < _qfp2_vm) / _qfp2_n,
                        "E": sum(1 for i in range(_qfp2_n)
                                 if _coh3_steps[i] > _qfp2_c3m and _velocity_steps[i] >= _qfp2_vm) / _qfp2_n,
                        "D": sum(1 for i in range(_qfp2_n)
                                 if _coh3_steps[i] <= _qfp2_c3m and _velocity_steps[i] >= _qfp2_vm) / _qfp2_n,
                        "F": sum(1 for i in range(_qfp2_n)
                                 if _coh3_steps[i] <= _qfp2_c3m and _velocity_steps[i] < _qfp2_vm) / _qfp2_n,
                    }:
                        ">".join(k for k, _ in sorted(_qfp2_fracs.items(), key=lambda x: -x[1]))
                    )()
                )()
            ))(),
            # "I>F>D>E" string — quadrant labels sorted by fraction descending (dominant first)
            "quad_ideal_minus_drifting_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qimd_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qimd_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qimd_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(
                        sum(1 for i in range(_qimd_n)
                            if _coh3_steps[i] > _qimd_c3m and _velocity_steps[i] < _qimd_vm) / _qimd_n
                        - sum(1 for i in range(_qimd_n)
                              if _coh3_steps[i] <= _qimd_c3m and _velocity_steps[i] >= _qimd_vm) / _qimd_n,
                        4)
                )()
            ))(),
            # ideal_frac − drifting_frac (positive=ideal dominates; negative=drift dominates)
            "quad_exploring_minus_flat_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qemf_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qemf_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qemf_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(
                        sum(1 for i in range(_qemf_n)
                            if _coh3_steps[i] > _qemf_c3m and _velocity_steps[i] >= _qemf_vm) / _qemf_n
                        - sum(1 for i in range(_qemf_n)
                              if _coh3_steps[i] <= _qemf_c3m and _velocity_steps[i] < _qemf_vm) / _qemf_n,
                        4)
                )()
            ))(),
            # exploring_frac − flat_frac (positive=creative > stagnant)
            "quad_positive_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qpf_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qpf_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qpf_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(
                        sum(1 for i in range(_qpf_n)
                            if _coh3_steps[i] > _qpf_c3m) / _qpf_n,
                        4)
                )()
            ))(),
            # fraction of steps with above-median coh3 (ideal + exploring = "high coherence" quadrants)
            "quad_negative_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qnf_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qnf_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qnf_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(
                        sum(1 for i in range(_qnf_n)
                            if _coh3_steps[i] <= _qnf_c3m) / _qnf_n,
                        4)
                )()
            ))(),
            # fraction of steps with below-median coh3 (drifting + flat = "low coherence" quadrants)
            "quad_quality_arc": (lambda: (
                "none" if min(len(_coh3_steps), len(_velocity_steps)) < 6
                else (lambda _qqa_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qqa_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qqa_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qqa_half=max(_qqa_n // 2, 1):
                        (lambda _qqa_ei=sum(1 for i in range(_qqa_half)
                                            if _coh3_steps[i] > _qqa_c3m and _velocity_steps[i] < _qqa_vm) /
                                          _qqa_half,
                               _qqa_li=sum(1 for i in range(_qqa_half, _qqa_n)
                                           if _coh3_steps[i] > _qqa_c3m and _velocity_steps[i] < _qqa_vm) /
                                         max(_qqa_n - _qqa_half, 1),
                               _qqa_ed=sum(1 for i in range(_qqa_half)
                                           if _coh3_steps[i] <= _qqa_c3m and _velocity_steps[i] >= _qqa_vm) /
                                         _qqa_half,
                               _qqa_ld=sum(1 for i in range(_qqa_half, _qqa_n)
                                           if _coh3_steps[i] <= _qqa_c3m and _velocity_steps[i] >= _qqa_vm) /
                                         max(_qqa_n - _qqa_half, 1):
                            ("cold_start" if _qqa_ei <= 0.05 and _qqa_li >= 0.20
                             else "warm_start"   if _qqa_ei >= 0.20 and _qqa_li <= 0.10
                             else "collapse"     if _qqa_ei >= 0.15 and _qqa_ld >= _qqa_ed + 0.15
                             else "recovery"     if _qqa_ed >= 0.15 and _qqa_li >= _qqa_ei + 0.10
                             else "sustained"    if _qqa_ei >= 0.15 and _qqa_li >= 0.15
                             else "oscillating"  if abs(_qqa_li - _qqa_ei) <= 0.05
                             else "degrading"    if _qqa_li < _qqa_ei - 0.08
                             else "improving")
                        )()
                    )()
                )()
            ))(),
            # quality arc label: cold_start/warm_start/sustained/collapse/recovery/oscillating/improving/degrading
            "quad_hi_vel_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qhv_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qhv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_qhv_n) if _velocity_steps[i] >= _qhv_vm) / _qhv_n, 4)
                )()
            ))(),
            # fraction of steps with above-median velocity (drifting + exploring)
            "quad_lo_vel_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qlv_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qlv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_qlv_n) if _velocity_steps[i] < _qlv_vm) / _qlv_n, 4)
                )()
            ))(),
            # fraction of steps with below-median velocity (ideal + flat)
            "quad_flow_efficiency": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qfe_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfe_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfe_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfe_ideal=sum(1 for i in range(_qfe_n)
                                           if _coh3_steps[i] > _qfe_c3m and _velocity_steps[i] < _qfe_vm),
                           _qfe_pos=sum(1 for i in range(_qfe_n) if _coh3_steps[i] > _qfe_c3m):
                        round(_qfe_ideal / max(_qfe_pos, 1), 4)
                    )()
                )()
            ))(),
            # fraction of high-coh3 steps that are also low-velocity (ideal vs exploring); 1=all hi-coh3=ideal
            "quad_drift_severity": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qds_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qds_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qds_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qds_drift=sum(1 for i in range(_qds_n)
                                           if _coh3_steps[i] <= _qds_c3m and _velocity_steps[i] >= _qds_vm),
                           _qds_hv=sum(1 for i in range(_qds_n) if _velocity_steps[i] >= _qds_vm):
                        round(_qds_drift / max(_qds_hv, 1), 4)
                    )()
                )()
            ))(),
            # fraction of high-velocity steps that are low-coh3 (pure drift severity; 1=all fast=drift)
            "quad_recovery_efficiency": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qre_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qre_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qre_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qre_labs=[
                        ("ideal"     if _coh3_steps[i] > _qre_c3m and _velocity_steps[i] < _qre_vm else
                         "exploring" if _coh3_steps[i] > _qre_c3m else
                         "drifting"  if _velocity_steps[i] >= _qre_vm else "flat")
                        for i in range(_qre_n)
                    ]:
                        (lambda _qre_de=sum(1 for i in range(1, _qre_n)
                                            if _qre_labs[i-1] == "drifting" and _qre_labs[i] != "drifting"),
                               _qre_dti=sum(1 for i in range(1, _qre_n)
                                            if _qre_labs[i-1] == "drifting" and _qre_labs[i] == "ideal"),
                               _qre_ideal_trans=sum(1 for i in range(1, _qre_n)
                                                    if _qre_labs[i-1] == "ideal"),
                               _qre_ideal_self=sum(1 for i in range(1, _qre_n)
                                                   if _qre_labs[i-1] == "ideal" and _qre_labs[i] == "ideal"):
                            round(
                                (_qre_dti / max(_qre_de, 1) if _qre_de else 0.0)
                                * (_qre_ideal_self / max(_qre_ideal_trans, 1) if _qre_ideal_trans else 0.0),
                                4)
                        )()
                    )()
                )()
            ))(),
            # drift_to_ideal_rate × ideal_persistence (both recovers AND sustains quality flow)
            "quad_ideal_conf_cv": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qicc_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qicc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qicc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qicc_vals=[
                        output_confs[i] for i in range(_qicc_n)
                        if _coh3_steps[i] > _qicc_c3m and _velocity_steps[i] < _qicc_vm
                    ]:
                        (lambda _qicc_mu=sum(_qicc_vals)/max(len(_qicc_vals),1):
                            round(
                                (sum((v-_qicc_mu)**2 for v in _qicc_vals)/max(len(_qicc_vals),1))**0.5
                                / max(_qicc_mu, 1e-9)
                                if len(_qicc_vals) >= 2 else 0.0,
                                4)
                        )()
                    )()
                )()
            ))(),
            # coefficient of variation (σ/μ) of confidence on ideal steps (low=reliable ideal confidence)
            "quad_ideal_coh3_cv": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qic3c_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qic3c_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qic3c_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qic3c_vals=[
                        _coh3_steps[i] for i in range(_qic3c_n)
                        if _coh3_steps[i] > _qic3c_c3m and _velocity_steps[i] < _qic3c_vm
                    ]:
                        (lambda _qic3c_mu=sum(_qic3c_vals)/max(len(_qic3c_vals),1):
                            round(
                                (sum((v-_qic3c_mu)**2 for v in _qic3c_vals)/max(len(_qic3c_vals),1))**0.5
                                / max(_qic3c_mu, 1e-9)
                                if len(_qic3c_vals) >= 2 else 0.0,
                                4)
                        )()
                    )()
                )()
            ))(),
            # coefficient of variation (σ/μ) of coh3 on ideal steps (low=consistent coherence in flow)
            "quad_drifting_coh3_cv": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdc3c_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdc3c_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdc3c_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdc3c_vals=[
                        _coh3_steps[i] for i in range(_qdc3c_n)
                        if _coh3_steps[i] <= _qdc3c_c3m and _velocity_steps[i] >= _qdc3c_vm
                    ]:
                        (lambda _qdc3c_mu=sum(_qdc3c_vals)/max(len(_qdc3c_vals),1):
                            round(
                                (sum((v-_qdc3c_mu)**2 for v in _qdc3c_vals)/max(len(_qdc3c_vals),1))**0.5
                                / max(_qdc3c_mu, 1e-9)
                                if len(_qdc3c_vals) >= 2 else 0.0,
                                4)
                        )()
                    )()
                )()
            ))(),
            # coefficient of variation (σ/μ) of coh3 on drifting steps (high=drift is erratic)
            "quad_ideal_conf_stability": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qics_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qics_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qics_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qics_vals=[
                        output_confs[i] for i in range(_qics_n)
                        if _coh3_steps[i] > _qics_c3m and _velocity_steps[i] < _qics_vm
                    ]:
                        (lambda _qics_mu=sum(_qics_vals)/max(len(_qics_vals),1):
                            round(
                                1.0 / (1.0 + (
                                    (sum((v-_qics_mu)**2 for v in _qics_vals)/max(len(_qics_vals),1))**0.5
                                    / max(_qics_mu, 1e-9)
                                ) if len(_qics_vals) >= 2 else 0.0),
                                4)
                        )()
                    )()
                )()
            ))(),
            # 1/(1+ideal_conf_cv) mapped to (0,1]; high=ideal confidence is very consistent
            "quad_ideal_conf_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qicm_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qicm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qicm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qicm_v=[output_confs[i] for i in range(_qicm_n)
                                     if _coh3_steps[i] > _qicm_c3m and _velocity_steps[i] < _qicm_vm]:
                        round(sum(_qicm_v)/max(len(_qicm_v),1), 4) if _qicm_v else 0.0
                    )()
                )()
            ))(),
            # mean confidence on ideal steps
            "quad_exploring_conf_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qecm2_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qecm2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qecm2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qecm2_v=[output_confs[i] for i in range(_qecm2_n)
                                      if _coh3_steps[i] > _qecm2_c3m and _velocity_steps[i] >= _qecm2_vm]:
                        round(sum(_qecm2_v)/max(len(_qecm2_v),1), 4) if _qecm2_v else 0.0
                    )()
                )()
            ))(),
            # mean confidence on exploring steps
            "quad_drifting_conf_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qdcm2_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qdcm2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdcm2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdcm2_v=[output_confs[i] for i in range(_qdcm2_n)
                                      if _coh3_steps[i] <= _qdcm2_c3m and _velocity_steps[i] >= _qdcm2_vm]:
                        round(sum(_qdcm2_v)/max(len(_qdcm2_v),1), 4) if _qdcm2_v else 0.0
                    )()
                )()
            ))(),
            # mean confidence on drifting steps
            "quad_flat_conf_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qfcm2_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qfcm2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfcm2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfcm2_v=[output_confs[i] for i in range(_qfcm2_n)
                                      if _coh3_steps[i] <= _qfcm2_c3m and _velocity_steps[i] < _qfcm2_vm]:
                        round(sum(_qfcm2_v)/max(len(_qfcm2_v),1), 4) if _qfcm2_v else 0.0
                    )()
                )()
            ))(),
            # mean confidence on flat steps
            "quad_conf_gap": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qcg_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qcg_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qcg_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qcg_id=[output_confs[i] for i in range(_qcg_n)
                                     if _coh3_steps[i] > _qcg_c3m and _velocity_steps[i] < _qcg_vm],
                           _qcg_dr=[output_confs[i] for i in range(_qcg_n)
                                    if _coh3_steps[i] <= _qcg_c3m and _velocity_steps[i] >= _qcg_vm]:
                        round(
                            sum(_qcg_id)/max(len(_qcg_id),1) - sum(_qcg_dr)/max(len(_qcg_dr),1)
                            if _qcg_id and _qcg_dr else 0.0,
                            4)
                    )()
                )()
            ))(),
            # ideal_conf_mean − drifting_conf_mean (positive=ideal steps are more confident=good)
            "quad_ideal_vel_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qivm_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qivm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qivm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qivm_v=[_velocity_steps[i] for i in range(_qivm_n)
                                     if _coh3_steps[i] > _qivm_c3m and _velocity_steps[i] < _qivm_vm]:
                        round(sum(_qivm_v)/max(len(_qivm_v),1), 4) if _qivm_v else 0.0
                    )()
                )()
            ))(),
            # mean velocity on ideal steps
            "quad_exploring_vel_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qevm_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qevm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qevm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qevm_v=[_velocity_steps[i] for i in range(_qevm_n)
                                     if _coh3_steps[i] > _qevm_c3m and _velocity_steps[i] >= _qevm_vm]:
                        round(sum(_qevm_v)/max(len(_qevm_v),1), 4) if _qevm_v else 0.0
                    )()
                )()
            ))(),
            # mean velocity on exploring steps
            "quad_drifting_vel_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qdvm2_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdvm2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdvm2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdvm2_v=[_velocity_steps[i] for i in range(_qdvm2_n)
                                      if _coh3_steps[i] <= _qdvm2_c3m and _velocity_steps[i] >= _qdvm2_vm]:
                        round(sum(_qdvm2_v)/max(len(_qdvm2_v),1), 4) if _qdvm2_v else 0.0
                    )()
                )()
            ))(),
            # mean velocity on drifting steps
            "quad_flat_vel_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qfvm2_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfvm2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfvm2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfvm2_v=[_velocity_steps[i] for i in range(_qfvm2_n)
                                      if _coh3_steps[i] <= _qfvm2_c3m and _velocity_steps[i] < _qfvm2_vm]:
                        round(sum(_qfvm2_v)/max(len(_qfvm2_v),1), 4) if _qfvm2_v else 0.0
                    )()
                )()
            ))(),
            # mean velocity on flat steps
            "quad_vel_gap": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qvg_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qvg_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qvg_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qvg_id=[_velocity_steps[i] for i in range(_qvg_n)
                                     if _coh3_steps[i] > _qvg_c3m and _velocity_steps[i] < _qvg_vm],
                           _qvg_dr=[_velocity_steps[i] for i in range(_qvg_n)
                                    if _coh3_steps[i] <= _qvg_c3m and _velocity_steps[i] >= _qvg_vm]:
                        round(
                            sum(_qvg_dr)/max(len(_qvg_dr),1) - sum(_qvg_id)/max(len(_qvg_id),1)
                            if _qvg_id and _qvg_dr else 0.0,
                            4)
                    )()
                )()
            ))(),
            # drifting_vel_mean − ideal_vel_mean (positive=drift is faster; large=clear velocity separation)
            "quad_ideal_run_mean_len": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qirml_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qirml_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qirml_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qirml_labs=[
                        "ideal" if _coh3_steps[i] > _qirml_c3m and _velocity_steps[i] < _qirml_vm
                        else "other" for i in range(_qirml_n)
                    ]:
                        (lambda _qirml_runs=sum(1 for i in range(1, _qirml_n)
                                               if _qirml_labs[i-1] != "ideal" and _qirml_labs[i] == "ideal"),
                               _qirml_steps=sum(1 for l in _qirml_labs if l == "ideal"):
                            round(_qirml_steps / max(_qirml_runs, 1), 2) if _qirml_runs else 0.0
                        )()
                    )()
                )()
            ))(),
            # mean number of consecutive ideal steps per ideal run
            "quad_drifting_run_mean_len": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdrml_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdrml_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdrml_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdrml_labs=[
                        "drifting" if _coh3_steps[i] <= _qdrml_c3m and _velocity_steps[i] >= _qdrml_vm
                        else "other" for i in range(_qdrml_n)
                    ]:
                        (lambda _qdrml_runs=sum(1 for i in range(1, _qdrml_n)
                                               if _qdrml_labs[i-1] != "drifting" and _qdrml_labs[i] == "drifting"),
                               _qdrml_steps=sum(1 for l in _qdrml_labs if l == "drifting"):
                            round(_qdrml_steps / max(_qdrml_runs, 1), 2) if _qdrml_runs else 0.0
                        )()
                    )()
                )()
            ))(),
            # mean number of consecutive drifting steps per drifting run
            "quad_exploring_run_mean_len": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qerml_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qerml_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qerml_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qerml_labs=[
                        "exploring" if _coh3_steps[i] > _qerml_c3m and _velocity_steps[i] >= _qerml_vm
                        else "other" for i in range(_qerml_n)
                    ]:
                        (lambda _qerml_runs=sum(1 for i in range(1, _qerml_n)
                                               if _qerml_labs[i-1] != "exploring" and _qerml_labs[i] == "exploring"),
                               _qerml_steps=sum(1 for l in _qerml_labs if l == "exploring"):
                            round(_qerml_steps / max(_qerml_runs, 1), 2) if _qerml_runs else 0.0
                        )()
                    )()
                )()
            ))(),
            # mean number of consecutive exploring steps per exploring run
            "quad_flat_run_mean_len": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qfrml_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfrml_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfrml_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfrml_labs=[
                        "flat" if _coh3_steps[i] <= _qfrml_c3m and _velocity_steps[i] < _qfrml_vm
                        else "other" for i in range(_qfrml_n)
                    ]:
                        (lambda _qfrml_runs=sum(1 for i in range(1, _qfrml_n)
                                               if _qfrml_labs[i-1] != "flat" and _qfrml_labs[i] == "flat"),
                               _qfrml_steps=sum(1 for l in _qfrml_labs if l == "flat"):
                            round(_qfrml_steps / max(_qfrml_runs, 1), 2) if _qfrml_runs else 0.0
                        )()
                    )()
                )()
            ))(),
            # mean number of consecutive flat steps per flat run
            "quad_run_len_ratio": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qrlr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qrlr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qrlr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qrlr_il=[
                        "ideal"    if _coh3_steps[i] > _qrlr_c3m and _velocity_steps[i] < _qrlr_vm
                        else "other" for i in range(_qrlr_n)
                    ],
                    _qrlr_dl=[
                        "drifting" if _coh3_steps[i] <= _qrlr_c3m and _velocity_steps[i] >= _qrlr_vm
                        else "other" for i in range(_qrlr_n)
                    ]:
                        (lambda _qrlr_ir=sum(1 for i in range(1, _qrlr_n)
                                            if _qrlr_il[i-1] != "ideal" and _qrlr_il[i] == "ideal"),
                               _qrlr_is=sum(1 for l in _qrlr_il if l == "ideal"),
                               _qrlr_dr=sum(1 for i in range(1, _qrlr_n)
                                            if _qrlr_dl[i-1] != "drifting" and _qrlr_dl[i] == "drifting"),
                               _qrlr_ds=sum(1 for l in _qrlr_dl if l == "drifting"):
                            round(
                                (_qrlr_is / max(_qrlr_ir, 1)) / max(_qrlr_ds / max(_qrlr_dr, 1), 0.1),
                                4)
                            if _qrlr_ir and _qrlr_dr else 0.0
                        )()
                    )()
                )()
            ))(),
            # ideal_run_mean_len / drifting_run_mean_len (>1 = ideal runs are longer than drift runs = good)
            "quad_ideal_coh3_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qic3m_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qic3m_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qic3m_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qic3m_v=[_coh3_steps[i] for i in range(_qic3m_n)
                                      if _coh3_steps[i] > _qic3m_c3m and _velocity_steps[i] < _qic3m_vm]:
                        round(sum(_qic3m_v)/max(len(_qic3m_v),1), 4) if _qic3m_v else 0.0
                    )()
                )()
            ))(),
            # mean coh3 on ideal steps
            "quad_drifting_coh3_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qdc3m_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdc3m_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdc3m_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdc3m_v=[_coh3_steps[i] for i in range(_qdc3m_n)
                                      if _coh3_steps[i] <= _qdc3m_c3m and _velocity_steps[i] >= _qdc3m_vm]:
                        round(sum(_qdc3m_v)/max(len(_qdc3m_v),1), 4) if _qdc3m_v else 0.0
                    )()
                )()
            ))(),
            # mean coh3 on drifting steps
            "quad_exploring_coh3_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qec3m_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qec3m_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qec3m_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qec3m_v=[_coh3_steps[i] for i in range(_qec3m_n)
                                      if _coh3_steps[i] > _qec3m_c3m and _velocity_steps[i] >= _qec3m_vm]:
                        round(sum(_qec3m_v)/max(len(_qec3m_v),1), 4) if _qec3m_v else 0.0
                    )()
                )()
            ))(),
            # mean coh3 on exploring steps
            "quad_flat_coh3_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qfc3m_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfc3m_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfc3m_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfc3m_v=[_coh3_steps[i] for i in range(_qfc3m_n)
                                      if _coh3_steps[i] <= _qfc3m_c3m and _velocity_steps[i] < _qfc3m_vm]:
                        round(sum(_qfc3m_v)/max(len(_qfc3m_v),1), 4) if _qfc3m_v else 0.0
                    )()
                )()
            ))(),
            # mean coh3 on flat steps
            "quad_ideal_coh3_std": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 3
                else (lambda _qic3s_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qic3s_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qic3s_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qic3s_v=[_coh3_steps[i] for i in range(_qic3s_n)
                                      if _coh3_steps[i] > _qic3s_c3m and _velocity_steps[i] < _qic3s_vm]:
                        (lambda _qic3s_mu=sum(_qic3s_v)/max(len(_qic3s_v),1):
                            round(
                                (sum((v-_qic3s_mu)**2 for v in _qic3s_v)/max(len(_qic3s_v)-1,1))**0.5
                                if len(_qic3s_v) >= 2 else 0.0,
                                4)
                        )()
                    )()
                )()
            ))(),
            # std of coh3 on ideal steps (low=coherence is stable during quality flow)
            "quad_drifting_coh3_std": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 3
                else (lambda _qdc3s_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdc3s_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdc3s_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdc3s_v=[_coh3_steps[i] for i in range(_qdc3s_n)
                                      if _coh3_steps[i] <= _qdc3s_c3m and _velocity_steps[i] >= _qdc3s_vm]:
                        (lambda _qdc3s_mu=sum(_qdc3s_v)/max(len(_qdc3s_v),1):
                            round(
                                (sum((v-_qdc3s_mu)**2 for v in _qdc3s_v)/max(len(_qdc3s_v)-1,1))**0.5
                                if len(_qdc3s_v) >= 2 else 0.0,
                                4)
                        )()
                    )()
                )()
            ))(),
            # std of coh3 on drifting steps (high=drift is erratic in coherence)
            "quad_coh3_contrast": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qcc3_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qcc3_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qcc3_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qcc3_id=[_coh3_steps[i] for i in range(_qcc3_n)
                                      if _coh3_steps[i] > _qcc3_c3m and _velocity_steps[i] < _qcc3_vm],
                           _qcc3_fl=[_coh3_steps[i] for i in range(_qcc3_n)
                                     if _coh3_steps[i] <= _qcc3_c3m and _velocity_steps[i] < _qcc3_vm]:
                        (lambda _qcc3_im=sum(_qcc3_id)/max(len(_qcc3_id),1),
                               _qcc3_fm=sum(_qcc3_fl)/max(len(_qcc3_fl),1):
                            round(
                                (_qcc3_im - _qcc3_fm) / max(_qcc3_im + _qcc3_fm, 1e-9)
                                if _qcc3_id and _qcc3_fl else 0.0,
                                4)
                        )()
                    )()
                )()
            ))(),
            # (ideal_coh3_mean − flat_coh3_mean)/(ideal+flat) — normalised coh3 contrast [−1,1]
            "quad_ideal_conf_std": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 3
                else (lambda _qics2_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qics2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qics2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qics2_v=[output_confs[i] for i in range(_qics2_n)
                                      if _coh3_steps[i] > _qics2_c3m and _velocity_steps[i] < _qics2_vm]:
                        (lambda _qics2_mu=sum(_qics2_v)/max(len(_qics2_v),1):
                            round(
                                (sum((v-_qics2_mu)**2 for v in _qics2_v)/max(len(_qics2_v)-1,1))**0.5
                                if len(_qics2_v) >= 2 else 0.0, 4)
                        )()
                    )()
                )()
            ))(),
            # std dev of confidence on ideal steps
            "quad_drifting_conf_std": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 3
                else (lambda _qdcs_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qdcs_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdcs_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdcs_v=[output_confs[i] for i in range(_qdcs_n)
                                     if _coh3_steps[i] <= _qdcs_c3m and _velocity_steps[i] >= _qdcs_vm]:
                        (lambda _qdcs_mu=sum(_qdcs_v)/max(len(_qdcs_v),1):
                            round(
                                (sum((v-_qdcs_mu)**2 for v in _qdcs_v)/max(len(_qdcs_v)-1,1))**0.5
                                if len(_qdcs_v) >= 2 else 0.0, 4)
                        )()
                    )()
                )()
            ))(),
            # std dev of confidence on drifting steps
            "quad_conf_contrast": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qcc2_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qcc2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qcc2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qcc2_id=[output_confs[i] for i in range(_qcc2_n)
                                      if _coh3_steps[i] > _qcc2_c3m and _velocity_steps[i] < _qcc2_vm],
                           _qcc2_fl=[output_confs[i] for i in range(_qcc2_n)
                                     if _coh3_steps[i] <= _qcc2_c3m and _velocity_steps[i] < _qcc2_vm]:
                        (lambda _qcc2_im=sum(_qcc2_id)/max(len(_qcc2_id),1),
                               _qcc2_fm=sum(_qcc2_fl)/max(len(_qcc2_fl),1):
                            round(
                                (_qcc2_im - _qcc2_fm) / max(_qcc2_im + _qcc2_fm, 1e-9)
                                if _qcc2_id and _qcc2_fl else 0.0, 4)
                        )()
                    )()
                )()
            ))(),
            # (ideal_conf_mean − flat_conf_mean)/(ideal+flat) normalised confidence contrast [−1,+1]
            "quad_ideal_vel_std": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 3
                else (lambda _qivs_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qivs_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qivs_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qivs_v=[_velocity_steps[i] for i in range(_qivs_n)
                                     if _coh3_steps[i] > _qivs_c3m and _velocity_steps[i] < _qivs_vm]:
                        (lambda _qivs_mu=sum(_qivs_v)/max(len(_qivs_v),1):
                            round(
                                (sum((v-_qivs_mu)**2 for v in _qivs_v)/max(len(_qivs_v)-1,1))**0.5
                                if len(_qivs_v) >= 2 else 0.0, 4)
                        )()
                    )()
                )()
            ))(),
            # std dev of velocity on ideal steps (low=ideal is consistently slow=good focus)
            "quad_drifting_vel_std": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 3
                else (lambda _qdvs_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdvs_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdvs_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdvs_v=[_velocity_steps[i] for i in range(_qdvs_n)
                                     if _coh3_steps[i] <= _qdvs_c3m and _velocity_steps[i] >= _qdvs_vm]:
                        (lambda _qdvs_mu=sum(_qdvs_v)/max(len(_qdvs_v),1):
                            round(
                                (sum((v-_qdvs_mu)**2 for v in _qdvs_v)/max(len(_qdvs_v)-1,1))**0.5
                                if len(_qdvs_v) >= 2 else 0.0, 4)
                        )()
                    )()
                )()
            ))(),
            # std dev of velocity on drifting steps (high=drift speed is erratic)
            "quad_ideal_conf_range": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qicr_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qicr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qicr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qicr_v=[output_confs[i] for i in range(_qicr_n)
                                     if _coh3_steps[i] > _qicr_c3m and _velocity_steps[i] < _qicr_vm]:
                        round(max(_qicr_v) - min(_qicr_v), 4) if len(_qicr_v) >= 2 else 0.0
                    )()
                )()
            ))(),
            # max − min confidence on ideal steps (low=very consistent confidence during flow)
            "quad_drifting_conf_range": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qdcr_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qdcr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdcr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdcr_v=[output_confs[i] for i in range(_qdcr_n)
                                     if _coh3_steps[i] <= _qdcr_c3m and _velocity_steps[i] >= _qdcr_vm]:
                        round(max(_qdcr_v) - min(_qdcr_v), 4) if len(_qdcr_v) >= 2 else 0.0
                    )()
                )()
            ))(),
            # max − min confidence on drifting steps
            "quad_ideal_vel_range": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qivr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qivr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qivr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qivr_v=[_velocity_steps[i] for i in range(_qivr_n)
                                     if _coh3_steps[i] > _qivr_c3m and _velocity_steps[i] < _qivr_vm]:
                        round(max(_qivr_v) - min(_qivr_v), 4) if len(_qivr_v) >= 2 else 0.0
                    )()
                )()
            ))(),
            # max − min velocity on ideal steps (low=ideal is narrow and steady=good)
            "quad_drifting_vel_range": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qdvr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdvr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdvr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdvr_v=[_velocity_steps[i] for i in range(_qdvr_n)
                                     if _coh3_steps[i] <= _qdvr_c3m and _velocity_steps[i] >= _qdvr_vm]:
                        round(max(_qdvr_v) - min(_qdvr_v), 4) if len(_qdvr_v) >= 2 else 0.0
                    )()
                )()
            ))(),
            # max − min velocity on drifting steps
            "quad_separation_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qss2_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qss2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qss2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qss2_id_c3=[_coh3_steps[i] for i in range(_qss2_n)
                                         if _coh3_steps[i] > _qss2_c3m and _velocity_steps[i] < _qss2_vm],
                           _qss2_dr_c3=[_coh3_steps[i] for i in range(_qss2_n)
                                        if _coh3_steps[i] <= _qss2_c3m and _velocity_steps[i] >= _qss2_vm],
                           _qss2_id_v=[_velocity_steps[i] for i in range(_qss2_n)
                                       if _coh3_steps[i] > _qss2_c3m and _velocity_steps[i] < _qss2_vm],
                           _qss2_dr_v=[_velocity_steps[i] for i in range(_qss2_n)
                                       if _coh3_steps[i] <= _qss2_c3m and _velocity_steps[i] >= _qss2_vm],
                           _qss2_id_cf=[output_confs[i] for i in range(_qss2_n)
                                        if _coh3_steps[i] > _qss2_c3m and _velocity_steps[i] < _qss2_vm],
                           _qss2_dr_cf=[output_confs[i] for i in range(_qss2_n)
                                        if _coh3_steps[i] <= _qss2_c3m and _velocity_steps[i] >= _qss2_vm]:
                        (lambda _qss2_c3g=abs(
                                    sum(_qss2_id_c3)/max(len(_qss2_id_c3),1)
                                    - sum(_qss2_dr_c3)/max(len(_qss2_dr_c3),1)),
                               _qss2_vg=abs(
                                    sum(_qss2_dr_v)/max(len(_qss2_dr_v),1)
                                    - sum(_qss2_id_v)/max(len(_qss2_id_v),1)),
                               _qss2_cfg=abs(
                                    sum(_qss2_id_cf)/max(len(_qss2_id_cf),1)
                                    - sum(_qss2_dr_cf)/max(len(_qss2_dr_cf),1))
                            if _qss2_id_c3 and _qss2_dr_c3 else (0.0, 0.0, 0.0):
                            round(
                                (_qss2_c3g / max(_qss2_c3g + 0.2, 1e-9) * 0.5
                                 + _qss2_vg / max(_qss2_vg + 0.05, 1e-9) * 0.3
                                 + _qss2_cfg / max(_qss2_cfg + 0.1, 1e-9) * 0.2)
                                if _qss2_id_c3 and _qss2_dr_c3 else 0.0,
                                4)
                        )()
                    )()
                )()
            ))(),
            # composite separation score: weighted coh3_gap + vel_gap + conf_gap [0,1]
            "quad_first_drifting_step": (lambda: (
                -1 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qfds_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfds_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfds_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    next((i for i in range(_qfds_n)
                          if _coh3_steps[i] <= _qfds_c3m and _velocity_steps[i] >= _qfds_vm),
                         -1)
                )()
            ))(),
            # index of first drifting step (−1 if none)
            "quad_last_ideal_step": (lambda: (
                -1 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qlis_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qlis_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qlis_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    next((i for i in range(_qlis_n-1, -1, -1)
                          if _coh3_steps[i] > _qlis_c3m and _velocity_steps[i] < _qlis_vm),
                         -1)
                )()
            ))(),
            # index of last ideal step (−1 if none); large=ideal persisted to the end
            "quad_last_drifting_step": (lambda: (
                -1 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qlds_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qlds_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qlds_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    next((i for i in range(_qlds_n-1, -1, -1)
                          if _coh3_steps[i] <= _qlds_c3m and _velocity_steps[i] >= _qlds_vm),
                         -1)
                )()
            ))(),
            # index of last drifting step (−1 if none); large=drift continued until end
            "quad_ideal_coverage_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qicf_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qicf_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qicf_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qicf_labs=[
                        "ideal" if _coh3_steps[i] > _qicf_c3m and _velocity_steps[i] < _qicf_vm
                        else "other" for i in range(_qicf_n)
                    ]:
                        (lambda _qicf_fi=next((i for i, l in enumerate(_qicf_labs) if l == "ideal"), -1),
                               _qicf_li=next((i for i in range(_qicf_n-1,-1,-1) if _qicf_labs[i] == "ideal"), -1):
                            round(
                                sum(1 for l in _qicf_labs if l == "ideal") /
                                max(_qicf_li - _qicf_fi + 1, 1)
                                if _qicf_fi >= 0 and _qicf_li >= 0 else 0.0,
                                4)
                        )()
                    )()
                )()
            ))(),
            # ideal_steps / (last_ideal − first_ideal + 1) — how densely ideal fills its span [0,1]
            "quad_final_state": (lambda: (
                "none" if min(len(_coh3_steps), len(_velocity_steps)) < 1
                else (lambda _qfs_c3m=sum(_coh3_steps) / max(len(_coh3_steps), 1),
                           _qfs_vm=sum(_velocity_steps) / max(len(_velocity_steps), 1),
                           _qfs_n=min(len(_coh3_steps), len(_velocity_steps)):
                    ("ideal"     if _coh3_steps[_qfs_n-1] > _qfs_c3m and _velocity_steps[_qfs_n-1] < _qfs_vm
                     else "exploring" if _coh3_steps[_qfs_n-1] > _qfs_c3m
                     else "drifting"  if _velocity_steps[_qfs_n-1] >= _qfs_vm
                     else "flat")
                )()
            ))(),
            # label of the last step: which quadrant was the model in when generation ended?
            "quad_transition_entropy": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qte_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qte_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qte_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qte_labs=[
                        "ideal"     if _coh3_steps[i] > _qte_c3m and _velocity_steps[i] < _qte_vm
                        else "exploring" if _coh3_steps[i] > _qte_c3m
                        else "drifting"  if _velocity_steps[i] >= _qte_vm
                        else "flat" for i in range(_qte_n)
                    ]:
                        (lambda _qte_counts={
                            (a, b): sum(1 for j in range(1, _qte_n)
                                        if _qte_labs[j-1] == a and _qte_labs[j] == b)
                            for a in ("ideal","exploring","drifting","flat")
                            for b in ("ideal","exploring","drifting","flat")
                        }:
                            (lambda _qte_total=sum(_qte_counts.values()):
                                round(
                                    -sum(
                                        (v/_qte_total) * __import__("math").log2(v/_qte_total)
                                        for v in _qte_counts.values() if v > 0
                                    ) if _qte_total > 0 else 0.0,
                                    4)
                            )()
                        )()
                    )()
                )()
            ))(),
            # Shannon entropy of all observed quadrant transitions (high=unpredictable; low=patterned)
            "quad_dom_transition": (lambda: (
                "none" if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdt_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdt_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdt_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdt_labs=[
                        "ideal"     if _coh3_steps[i] > _qdt_c3m and _velocity_steps[i] < _qdt_vm
                        else "exploring" if _coh3_steps[i] > _qdt_c3m
                        else "drifting"  if _velocity_steps[i] >= _qdt_vm
                        else "flat" for i in range(_qdt_n)
                    ]:
                        (lambda _qdt_counts={
                            f"{a}→{b}": sum(1 for j in range(1, _qdt_n)
                                            if _qdt_labs[j-1] == a and _qdt_labs[j] == b)
                            for a in ("ideal","exploring","drifting","flat")
                            for b in ("ideal","exploring","drifting","flat")
                        }:
                            max(_qdt_counts, key=lambda k: _qdt_counts[k])
                            if any(v > 0 for v in _qdt_counts.values()) else "none"
                        )()
                    )()
                )()
            ))(),
            # string label of most frequent single-step transition (e.g. "ideal→ideal")
            "quad_ideal_tail_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qitf_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qitf_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qitf_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qitf_tail_start=max(_qitf_n - max(_qitf_n // 4, 1), 0):
                        (lambda _qitf_tail_n=_qitf_n - _qitf_tail_start:
                            round(
                                sum(1 for i in range(_qitf_tail_start, _qitf_n)
                                    if _coh3_steps[i] > _qitf_c3m and _velocity_steps[i] < _qitf_vm)
                                / max(_qitf_tail_n, 1),
                                4)
                        )()
                    )()
                )()
            ))(),
            # fraction of last-25% steps that are ideal (high=quality persisted to end)
            "quad_drift_tail_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdtf_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdtf_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdtf_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdtf_tail_start=max(_qdtf_n - max(_qdtf_n // 4, 1), 0):
                        (lambda _qdtf_tail_n=_qdtf_n - _qdtf_tail_start:
                            round(
                                sum(1 for i in range(_qdtf_tail_start, _qdtf_n)
                                    if _coh3_steps[i] <= _qdtf_c3m and _velocity_steps[i] >= _qdtf_vm)
                                / max(_qdtf_tail_n, 1),
                                4)
                        )()
                    )()
                )()
            ))(),
            # fraction of last-25% steps that are drifting (high=drift dominated the ending)
            "quad_ideal_improvement": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 8
                else (lambda _qii_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qii_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qii_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qii_q=max(_qii_n // 4, 1):
                        (lambda _qii_head=sum(1 for i in range(_qii_q)
                                              if _coh3_steps[i] > _qii_c3m and _velocity_steps[i] < _qii_vm),
                               _qii_tail=sum(1 for i in range(_qii_n - _qii_q, _qii_n)
                                              if _coh3_steps[i] > _qii_c3m and _velocity_steps[i] < _qii_vm):
                            round((_qii_tail - _qii_head) / max(_qii_q, 1), 4)
                        )()
                    )()
                )()
            ))(),
            # (ideal_tail_frac - ideal_head_frac): positive=quality improved; negative=quality degraded
            "quad_ideal_mid_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 8
                else (lambda _qimf_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qimf_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qimf_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qimf_q=max(_qimf_n // 4, 1):
                        round(
                            sum(1 for i in range(_qimf_q, _qimf_n - _qimf_q)
                                if _coh3_steps[i] > _qimf_c3m and _velocity_steps[i] < _qimf_vm)
                            / max(_qimf_n - 2 * _qimf_q, 1),
                            4)
                    )()
                )()
            ))(),
            # fraction of middle-50% steps that are ideal
            "quad_drift_mid_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 8
                else (lambda _qdmf_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdmf_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdmf_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdmf_q=max(_qdmf_n // 4, 1):
                        round(
                            sum(1 for i in range(_qdmf_q, _qdmf_n - _qdmf_q)
                                if _coh3_steps[i] <= _qdmf_c3m and _velocity_steps[i] >= _qdmf_vm)
                            / max(_qdmf_n - 2 * _qdmf_q, 1),
                            4)
                    )()
                )()
            ))(),
            # fraction of middle-50% steps that are drifting
            "quad_ideal_peak_coh3": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qipc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qipc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qipc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qipc_v=[_coh3_steps[i] for i in range(_qipc_n)
                                     if _coh3_steps[i] > _qipc_c3m and _velocity_steps[i] < _qipc_vm]:
                        round(max(_qipc_v), 4) if _qipc_v else 0.0
                    )()
                )()
            ))(),
            # max coh3 on ideal steps — best coherence moment during quality flow
            "quad_ideal_peak_conf": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qipf_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qipf_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qipf_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qipf_v=[output_confs[i] for i in range(_qipf_n)
                                     if _coh3_steps[i] > _qipf_c3m and _velocity_steps[i] < _qipf_vm]:
                        round(max(_qipf_v), 4) if _qipf_v else 0.0
                    )()
                )()
            ))(),
            # max confidence on ideal steps — peak confidence during quality flow
            "quad_drifting_worst_coh3": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qdwc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdwc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdwc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdwc_v=[_coh3_steps[i] for i in range(_qdwc_n)
                                     if _coh3_steps[i] <= _qdwc_c3m and _velocity_steps[i] >= _qdwc_vm]:
                        round(min(_qdwc_v), 4) if _qdwc_v else 0.0
                    )()
                )()
            ))(),
            # min coh3 on drifting steps — floor of coherence during worst drift
            "quad_ideal_head_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qihf_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qihf_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qihf_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qihf_q=max(_qihf_n // 4, 1):
                        round(
                            sum(1 for i in range(_qihf_q)
                                if _coh3_steps[i] > _qihf_c3m and _velocity_steps[i] < _qihf_vm)
                            / max(_qihf_q, 1),
                            4)
                    )()
                )()
            ))(),
            # fraction of first-25% steps that are ideal (high=generation started strong)
            "quad_drift_head_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdhf_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdhf_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdhf_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdhf_q=max(_qdhf_n // 4, 1):
                        round(
                            sum(1 for i in range(_qdhf_q)
                                if _coh3_steps[i] <= _qdhf_c3m and _velocity_steps[i] >= _qdhf_vm)
                            / max(_qdhf_q, 1),
                            4)
                    )()
                )()
            ))(),
            # fraction of first-25% steps that are drifting (high=started in drift=bad opening)
            "quad_quality_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qqs_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qqs_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qqs_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qqs_labs=[
                        "ideal"     if _coh3_steps[i] > _qqs_c3m and _velocity_steps[i] < _qqs_vm
                        else "drifting" if _coh3_steps[i] <= _qqs_c3m and _velocity_steps[i] >= _qqs_vm
                        else "other" for i in range(_qqs_n)
                    ]:
                        (lambda _qqs_if=sum(1 for l in _qqs_labs if l == "ideal") / max(_qqs_n, 1),
                               _qqs_df=sum(1 for l in _qqs_labs if l == "drifting") / max(_qqs_n, 1):
                            (lambda _qqs_q=max(_qqs_n // 4, 1):
                                (lambda _qqs_head=sum(1 for i in range(_qqs_q)
                                                     if _qqs_labs[i] == "ideal"),
                                       _qqs_tail=sum(1 for i in range(_qqs_n - _qqs_q, _qqs_n)
                                                     if _qqs_labs[i] == "ideal"):
                                    round(
                                        0.50 * _qqs_if
                                        + 0.25 * (1.0 - _qqs_df)
                                        + 0.25 * min(max((_qqs_tail - _qqs_head) / max(_qqs_q, 1) * 0.5 + 0.5, 0.0), 1.0),
                                        4)
                                )()
                            )()
                        )()
                    )()
                )()
            ))(),
            # composite quality score [0,1]: 50%×ideal_frac + 25%×(1−drift_frac) + 25%×improvement_norm
            "quad_ideal_conf_floor": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qicfl_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qicfl_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qicfl_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qicfl_v=[output_confs[i] for i in range(_qicfl_n)
                                      if _coh3_steps[i] > _qicfl_c3m and _velocity_steps[i] < _qicfl_vm]:
                        round(min(_qicfl_v), 4) if _qicfl_v else 0.0
                    )()
                )()
            ))(),
            # min confidence on ideal steps (low=even ideal steps had confidence dips)
            "quad_drifting_peak_conf": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qdpc_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qdpc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdpc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdpc_v=[output_confs[i] for i in range(_qdpc_n)
                                     if _coh3_steps[i] <= _qdpc_c3m and _velocity_steps[i] >= _qdpc_vm]:
                        round(max(_qdpc_v), 4) if _qdpc_v else 0.0
                    )()
                )()
            ))(),
            # max confidence on drifting steps (high=drift had confident moments=surprising)
            "quad_above_median_coh3_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qamc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qamc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_qamc_n) if _coh3_steps[i] > _qamc_c3m) / max(_qamc_n, 1), 4)
                )()
            ))(),
            # fraction of steps with coh3 above the run mean (ideal+exploring combined; ~0.50 if symmetric)
            "quad_below_median_vel_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qbmv_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qbmv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_qbmv_n) if _velocity_steps[i] < _qbmv_vm) / max(_qbmv_n, 1), 4)
                )()
            ))(),
            # fraction of steps below velocity mean (ideal+flat combined)
            "quad_coherent_to_incoherent_ratio": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qcir_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qcir_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qcir_hi=sum(1 for i in range(_qcir_n) if _coh3_steps[i] > _qcir_c3m),
                           _qcir_lo=sum(1 for i in range(_qcir_n) if _coh3_steps[i] <= _qcir_c3m):
                        round(_qcir_hi / max(_qcir_lo, 1), 4)
                    )()
                )()
            ))(),
            # above-coh3-mean / below-coh3-mean steps ratio; >1=more coherent than not
            "quad_focused_to_chaotic_ratio": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qfcr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfcr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfcr_lo=sum(1 for i in range(_qfcr_n) if _velocity_steps[i] < _qfcr_vm),
                           _qfcr_hi=sum(1 for i in range(_qfcr_n) if _velocity_steps[i] >= _qfcr_vm):
                        round(_qfcr_lo / max(_qfcr_hi, 1), 4)
                    )()
                )()
            ))(),
            # below-vel-mean / above-vel-mean steps ratio; >1=more focused (slow) than chaotic (fast)
            "quad_health_vector": (lambda: (
                "----" if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qhv_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qhv_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qhv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qhv_i=sum(1 for i in range(_qhv_n)
                                       if _coh3_steps[i] > _qhv_c3m and _velocity_steps[i] < _qhv_vm),
                           _qhv_e=sum(1 for i in range(_qhv_n)
                                      if _coh3_steps[i] > _qhv_c3m and _velocity_steps[i] >= _qhv_vm),
                           _qhv_d=sum(1 for i in range(_qhv_n)
                                      if _coh3_steps[i] <= _qhv_c3m and _velocity_steps[i] >= _qhv_vm),
                           _qhv_f=sum(1 for i in range(_qhv_n)
                                      if _coh3_steps[i] <= _qhv_c3m and _velocity_steps[i] < _qhv_vm):
                        (lambda _qhv_t=max(_qhv_n, 1):
                            "".join([
                                "I" if _qhv_i/_qhv_t >= 0.35 else "i" if _qhv_i/_qhv_t >= 0.15 else ".",
                                "E" if _qhv_e/_qhv_t >= 0.25 else "e" if _qhv_e/_qhv_t >= 0.10 else ".",
                                "D" if _qhv_d/_qhv_t >= 0.25 else "d" if _qhv_d/_qhv_t >= 0.10 else ".",
                                "F" if _qhv_f/_qhv_t >= 0.25 else "f" if _qhv_f/_qhv_t >= 0.10 else ".",
                            ])
                        )()
                    )()
                )()
            ))(),
            # compact health vector e.g. "I..f" — uppercase=dominant, lowercase=present, .=absent
            "quad_max_ideal_streak": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qmis_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qmis_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qmis_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qmis_labs=[
                        1 if _coh3_steps[i] > _qmis_c3m and _velocity_steps[i] < _qmis_vm else 0
                        for i in range(_qmis_n)
                    ]:
                        (lambda _qmis_best=0, _qmis_cur=0:
                            max(
                                (lambda _r=[0, 0]:
                                    [_r.__setitem__(0, _r[0]+v) or _r.__setitem__(1, max(_r[1], _r[0]))
                                     or _r.__setitem__(0, 0) if v == 0 else None
                                     for v in _qmis_labs] or _r[1]
                                )(),
                                sum(_qmis_labs[-k:]) if _qmis_labs else 0
                            )
                        )()
                    )()
                )()
            ))(),
            # max consecutive ideal steps in any single run
            "quad_max_drifting_streak": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qmds_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qmds_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qmds_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qmds_labs=[
                        1 if _coh3_steps[i] <= _qmds_c3m and _velocity_steps[i] >= _qmds_vm else 0
                        for i in range(_qmds_n)
                    ]:
                        (lambda _r=[0, 0]:
                            [_r.__setitem__(0, _r[0]+v) or _r.__setitem__(1, max(_r[1], _r[0]))
                             or _r.__setitem__(0, 0) if v == 0 else None
                             for v in _qmds_labs] or max(_r[1], _r[0])
                        )()
                    )()
                )()
            ))(),
            # max consecutive drifting steps in any single run
            "quad_max_ideal_streak_start": (lambda: (
                -1 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qmiss_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qmiss_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qmiss_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qmiss_labs=[
                        1 if _coh3_steps[i] > _qmiss_c3m and _velocity_steps[i] < _qmiss_vm else 0
                        for i in range(_qmiss_n)
                    ]:
                        (lambda _qmiss_state=[0, 0, -1, 0]:
                            [
                                (
                                    _qmiss_state.__setitem__(0, _qmiss_state[0]+v) or
                                    (_qmiss_state.__setitem__(2, i - _qmiss_state[0] + 1)
                                     or _qmiss_state.__setitem__(3, _qmiss_state[0])
                                     if _qmiss_state[0] > _qmiss_state[3] else None)
                                ) if v == 1
                                else _qmiss_state.__setitem__(0, 0)
                                for i, v in enumerate(_qmiss_labs)
                            ] or _qmiss_state[2]
                        )()
                    )()
                )()
            ))(),
            # index where the longest ideal streak started (−1 if no ideal steps)
            "quad_ideal_streak_ratio": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qisr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qisr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qisr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qisr_labs=[
                        1 if _coh3_steps[i] > _qisr_c3m and _velocity_steps[i] < _qisr_vm else 0
                        for i in range(_qisr_n)
                    ]:
                        (lambda _qisr_tot=sum(_qisr_labs):
                            (lambda _qisr_r=[0, 0]:
                                round(
                                    max(
                                        ([_qisr_r.__setitem__(0, _qisr_r[0]+v)
                                          or _qisr_r.__setitem__(1, max(_qisr_r[1], _qisr_r[0]))
                                          or _qisr_r.__setitem__(0, 0) if v == 0 else None
                                          for v in _qisr_labs] or _qisr_r[1]),
                                        _qisr_r[0]) / max(_qisr_tot, 1),
                                    4) if _qisr_tot > 0 else 0.0
                            )()
                        )()
                    )()
                )()
            ))(),
            # max_ideal_streak / total_ideal_steps — high=ideal concentrated in one burst
            "quad_drift_streak_ratio": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qdsr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdsr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdsr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdsr_labs=[
                        1 if _coh3_steps[i] <= _qdsr_c3m and _velocity_steps[i] >= _qdsr_vm else 0
                        for i in range(_qdsr_n)
                    ]:
                        (lambda _qdsr_tot=sum(_qdsr_labs):
                            (lambda _qdsr_r=[0, 0]:
                                round(
                                    max(
                                        ([_qdsr_r.__setitem__(0, _qdsr_r[0]+v)
                                          or _qdsr_r.__setitem__(1, max(_qdsr_r[1], _qdsr_r[0]))
                                          or _qdsr_r.__setitem__(0, 0) if v == 0 else None
                                          for v in _qdsr_labs] or _qdsr_r[1]),
                                        _qdsr_r[0]) / max(_qdsr_tot, 1),
                                    4) if _qdsr_tot > 0 else 0.0
                            )()
                        )()
                    )()
                )()
            ))(),
            # max_drifting_streak / total_drifting_steps — high=drift concentrated in one burst
            "quad_oscillation_count": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qoc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qoc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qoc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qoc_labs=[
                        1 if _coh3_steps[i] > _qoc_c3m and _velocity_steps[i] < _qoc_vm else 0
                        for i in range(_qoc_n)
                    ]:
                        sum(1 for j in range(1, _qoc_n) if _qoc_labs[j] != _qoc_labs[j-1])
                    )()
                )()
            ))(),
            # number of label flips between ideal and not-ideal (each ideal↔other switch)
            "quad_longest_non_ideal_run": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qlni_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qlni_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qlni_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qlni_labs=[
                        0 if _coh3_steps[i] > _qlni_c3m and _velocity_steps[i] < _qlni_vm else 1
                        for i in range(_qlni_n)
                    ]:
                        (lambda _r=[0, 0]:
                            max(
                                ([_r.__setitem__(0, _r[0]+v) or _r.__setitem__(1, max(_r[1], _r[0]))
                                  or _r.__setitem__(0, 0) if v == 0 else None
                                  for v in _qlni_labs] or _r[1]),
                                _r[0])
                        )()
                    )()
                )()
            ))(),
            # max consecutive non-ideal steps (max gap without quality flow)
            "quad_rle_first5": (lambda: (
                "" if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qrle_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qrle_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qrle_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qrle_labs=[
                        "i" if _coh3_steps[i] > _qrle_c3m and _velocity_steps[i] < _qrle_vm
                        else "e" if _coh3_steps[i] > _qrle_c3m
                        else "d" if _velocity_steps[i] >= _qrle_vm
                        else "f" for i in range(_qrle_n)
                    ]:
                        (lambda _qrle_runs=[(k, sum(1 for _ in g))
                                            for k, g in __import__("itertools").groupby(_qrle_labs)]:
                            "".join(f"{k}{n}" for k, n in _qrle_runs[:5])
                        )()
                    )()
                )()
            ))(),
            # first 5 RLE segments as string e.g. "i3d2e1" — shows how generation opened
            "quad_ideal_isolation_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qiis_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qiis_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qiis_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qiis_labs=[
                        1 if _coh3_steps[i] > _qiis_c3m and _velocity_steps[i] < _qiis_vm else 0
                        for i in range(_qiis_n)
                    ]:
                        round(
                            sum(1 for j in range(1, _qiis_n)
                                if _qiis_labs[j-1] == 0 and _qiis_labs[j] == 1)
                            / max(_qiis_n, 1),
                            4)
                    )()
                )()
            ))(),
            # ideal_run_count / total_steps — high=ideal is fragmented into many tiny runs
            "quad_late_ideal_start": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qlis2_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qlis2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qlis2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qlis2_fi=next((i for i in range(_qlis2_n)
                                            if _coh3_steps[i] > _qlis2_c3m
                                            and _velocity_steps[i] < _qlis2_vm), -1):
                        1 if _qlis2_fi >= _qlis2_n // 2 else 0
                    )()
                )()
            ))(),
            # 1 if the first ideal step is in the second half of the generation (late quality start)
            "quad_exploring_run_count": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qerc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qerc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qerc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qerc_labs=[
                        1 if _coh3_steps[i] > _qerc_c3m and _velocity_steps[i] >= _qerc_vm else 0
                        for i in range(_qerc_n)
                    ]:
                        sum(1 for j in range(1, _qerc_n) if _qerc_labs[j-1] == 0 and _qerc_labs[j] == 1)
                    )()
                )()
            ))(),
            # number of exploring runs (each entry into the exploring quadrant)
            "quad_flat_run_count": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qfrc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfrc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfrc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfrc_labs=[
                        1 if _coh3_steps[i] <= _qfrc_c3m and _velocity_steps[i] < _qfrc_vm else 0
                        for i in range(_qfrc_n)
                    ]:
                        sum(1 for j in range(1, _qfrc_n) if _qfrc_labs[j-1] == 0 and _qfrc_labs[j] == 1)
                    )()
                )()
            ))(),
            # number of flat runs (each entry into the flat quadrant)
            "quad_max_exploring_streak": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qmes_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qmes_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qmes_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qmes_labs=[
                        1 if _coh3_steps[i] > _qmes_c3m and _velocity_steps[i] >= _qmes_vm else 0
                        for i in range(_qmes_n)
                    ]:
                        (lambda _r=[0, 0]:
                            max(
                                ([_r.__setitem__(0, _r[0]+v) or _r.__setitem__(1, max(_r[1], _r[0]))
                                  or _r.__setitem__(0, 0) if v == 0 else None
                                  for v in _qmes_labs] or _r[1]),
                                _r[0])
                        )()
                    )()
                )()
            ))(),
            # max consecutive exploring steps
            "quad_max_flat_streak": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qmfs_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qmfs_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qmfs_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qmfs_labs=[
                        1 if _coh3_steps[i] <= _qmfs_c3m and _velocity_steps[i] < _qmfs_vm else 0
                        for i in range(_qmfs_n)
                    ]:
                        (lambda _r=[0, 0]:
                            max(
                                ([_r.__setitem__(0, _r[0]+v) or _r.__setitem__(1, max(_r[1], _r[0]))
                                  or _r.__setitem__(0, 0) if v == 0 else None
                                  for v in _qmfs_labs] or _r[1]),
                                _r[0])
                        )()
                    )()
                )()
            ))(),
            # max consecutive flat steps
            "quad_zigzag_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qzz_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qzz_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qzz_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qzz_labs=[
                        1 if _coh3_steps[i] > _qzz_c3m and _velocity_steps[i] < _qzz_vm else 0
                        for i in range(_qzz_n)
                    ]:
                        round(
                            sum(1 for j in range(1, _qzz_n) if _qzz_labs[j] != _qzz_labs[j-1])
                            / max(_qzz_n - 1, 1),
                            4)
                    )()
                )()
            ))(),
            # oscillation_count / (total_steps-1) — ideal↔other flips per step [0,1]; high=very volatile
            "quad_coh3_skew": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _qcs_n=len(_coh3_steps),
                           _qcs_mu=sum(_coh3_steps)/max(len(_coh3_steps),1):
                    (lambda _qcs_var=sum((v-_qcs_mu)**2 for v in _coh3_steps)/max(_qcs_n-1,1):
                        (lambda _qcs_std=_qcs_var**0.5:
                            round(
                                sum((v-_qcs_mu)**3 for v in _coh3_steps) / max(_qcs_n*(_qcs_std**3), 1e-12)
                                if _qcs_std > 1e-9 else 0.0,
                                4)
                        )()
                    )()
                )()
            ))(),
            # skewness of coh3 distribution (negative=left-skewed=most steps below mean; positive=right-skewed)
            "quad_vel_skew": (lambda: (
                0.0 if len(_velocity_steps) < 4
                else (lambda _qvsk_n=len(_velocity_steps),
                           _qvsk_mu=sum(_velocity_steps)/max(len(_velocity_steps),1):
                    (lambda _qvsk_var=sum((v-_qvsk_mu)**2 for v in _velocity_steps)/max(_qvsk_n-1,1):
                        (lambda _qvsk_std=_qvsk_var**0.5:
                            round(
                                sum((v-_qvsk_mu)**3 for v in _velocity_steps) / max(_qvsk_n*(_qvsk_std**3), 1e-12)
                                if _qvsk_std > 1e-9 else 0.0,
                                4)
                        )()
                    )()
                )()
            ))(),
            # skewness of velocity distribution (positive=right-skewed=most steps slow but tail of fast)
            "quad_ideal_centroid": (lambda: (
                -1.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qic2_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qic2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qic2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qic2_mask=[
                        1 if _coh3_steps[i] > _qic2_c3m and _velocity_steps[i] < _qic2_vm else 0
                        for i in range(_qic2_n)
                    ]:
                        (lambda _qic2_tot=sum(_qic2_mask):
                            round(
                                sum(i * _qic2_mask[i] for i in range(_qic2_n)) / max(_qic2_tot, 1),
                                2) if _qic2_tot > 0 else -1.0
                        )()
                    )()
                )()
            ))(),
            # weighted mean step index of ideal steps — where in the generation ideal flow occurs
            "quad_drifting_centroid": (lambda: (
                -1.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qdc2_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdc2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdc2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdc2_mask=[
                        1 if _coh3_steps[i] <= _qdc2_c3m and _velocity_steps[i] >= _qdc2_vm else 0
                        for i in range(_qdc2_n)
                    ]:
                        (lambda _qdc2_tot=sum(_qdc2_mask):
                            round(
                                sum(i * _qdc2_mask[i] for i in range(_qdc2_n)) / max(_qdc2_tot, 1),
                                2) if _qdc2_tot > 0 else -1.0
                        )()
                    )()
                )()
            ))(),
            # weighted mean step index of drifting steps
            "quad_centroid_gap": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qcg2_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qcg2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qcg2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qcg2_im=[
                        1 if _coh3_steps[i] > _qcg2_c3m and _velocity_steps[i] < _qcg2_vm else 0
                        for i in range(_qcg2_n)
                    ],
                    _qcg2_dm=[
                        1 if _coh3_steps[i] <= _qcg2_c3m and _velocity_steps[i] >= _qcg2_vm else 0
                        for i in range(_qcg2_n)
                    ]:
                        (lambda _qcg2_it=sum(_qcg2_im), _qcg2_dt=sum(_qcg2_dm):
                            round(
                                sum(i * _qcg2_im[i] for i in range(_qcg2_n)) / max(_qcg2_it, 1)
                                - sum(i * _qcg2_dm[i] for i in range(_qcg2_n)) / max(_qcg2_dt, 1),
                                2) if _qcg2_it > 0 and _qcg2_dt > 0 else 0.0
                        )()
                    )()
                )()
            ))(),
            # ideal_centroid − drifting_centroid (positive=ideal later than drift; negative=ideal earlier)
            "quad_coh3_kurtosis": (lambda: (
                0.0 if len(_coh3_steps) < 5
                else (lambda _qck_n=len(_coh3_steps),
                           _qck_mu=sum(_coh3_steps)/max(len(_coh3_steps),1):
                    (lambda _qck_var=sum((v-_qck_mu)**2 for v in _coh3_steps)/max(_qck_n-1,1):
                        (lambda _qck_std=_qck_var**0.5:
                            round(
                                sum((v-_qck_mu)**4 for v in _coh3_steps) / max(_qck_n*(_qck_std**4), 1e-12)
                                - 3.0
                                if _qck_std > 1e-9 else 0.0, 4)
                        )()
                    )()
                )()
            ))(),
            # excess kurtosis of coh3 (0=normal; >0=heavy tails; <0=flat; high=outlier-prone)
            "quad_vel_kurtosis": (lambda: (
                0.0 if len(_velocity_steps) < 5
                else (lambda _qvk_n=len(_velocity_steps),
                           _qvk_mu=sum(_velocity_steps)/max(len(_velocity_steps),1):
                    (lambda _qvk_var=sum((v-_qvk_mu)**2 for v in _velocity_steps)/max(_qvk_n-1,1):
                        (lambda _qvk_std=_qvk_var**0.5:
                            round(
                                sum((v-_qvk_mu)**4 for v in _velocity_steps) / max(_qvk_n*(_qvk_std**4), 1e-12)
                                - 3.0
                                if _qvk_std > 1e-9 else 0.0, 4)
                        )()
                    )()
                )()
            ))(),
            # excess kurtosis of velocity (>0=rare extreme velocities; <0=velocity is uniformly spread)
            "quad_conf_skew": (lambda: (
                0.0 if len(output_confs) < 4
                else (lambda _qcsk_n=len(output_confs),
                           _qcsk_mu=sum(output_confs)/max(len(output_confs),1):
                    (lambda _qcsk_var=sum((v-_qcsk_mu)**2 for v in output_confs)/max(_qcsk_n-1,1):
                        (lambda _qcsk_std=_qcsk_var**0.5:
                            round(
                                sum((v-_qcsk_mu)**3 for v in output_confs) / max(_qcsk_n*(_qcsk_std**3), 1e-12)
                                if _qcsk_std > 1e-9 else 0.0, 4)
                        )()
                    )()
                )()
            ))(),
            # skewness of confidence distribution (negative=most conf above mean; positive=long low-conf tail)
            "quad_conf_kurtosis": (lambda: (
                0.0 if len(output_confs) < 5
                else (lambda _qcku_n=len(output_confs),
                           _qcku_mu=sum(output_confs)/max(len(output_confs),1):
                    (lambda _qcku_var=sum((v-_qcku_mu)**2 for v in output_confs)/max(_qcku_n-1,1):
                        (lambda _qcku_std=_qcku_var**0.5:
                            round(
                                sum((v-_qcku_mu)**4 for v in output_confs) / max(_qcku_n*(_qcku_std**4), 1e-12)
                                - 3.0
                                if _qcku_std > 1e-9 else 0.0, 4)
                        )()
                    )()
                )()
            ))(),
            # excess kurtosis of confidence distribution
            "quad_signal_quality_index": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qsqi_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qsqi_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qsqi_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qsqi_i=sum(1 for i in range(_qsqi_n)
                                        if _coh3_steps[i] > _qsqi_c3m and _velocity_steps[i] < _qsqi_vm),
                           _qsqi_d=sum(1 for i in range(_qsqi_n)
                                       if _coh3_steps[i] <= _qsqi_c3m and _velocity_steps[i] >= _qsqi_vm):
                        (lambda _qsqi_cf=sum(output_confs[:_qsqi_n])/_qsqi_n,
                               _qsqi_c3=sum(_coh3_steps[:_qsqi_n])/_qsqi_n:
                            round(
                                0.35*(_qsqi_i/_qsqi_n)
                                + 0.20*(1.0-_qsqi_d/_qsqi_n)
                                + 0.25*min(_qsqi_c3*4.0, 1.0)
                                + 0.20*min(_qsqi_cf, 1.0),
                                4)
                        )()
                    )()
                )()
            ))(),
            # composite signal quality: 35%×ideal_frac + 20%×(1−drift_frac) + 25%×norm_coh3 + 20%×conf [0,1]
            "quad_ideal_coh3_var": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qicv_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qicv_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qicv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qicv_v=[_coh3_steps[i] for i in range(_qicv_n)
                                     if _coh3_steps[i] > _qicv_c3m and _velocity_steps[i] < _qicv_vm]:
                        (lambda _qicv_mu=sum(_qicv_v)/max(len(_qicv_v),1):
                            round(sum((x-_qicv_mu)**2 for x in _qicv_v)/max(len(_qicv_v)-1,1), 6)
                            if len(_qicv_v) >= 2 else 0.0
                        )()
                    )()
                )()
            ))(),
            # variance of coh3 on ideal steps (low=consistent quality; high=noisy ideal signal)
            "quad_drifting_coh3_var": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qdcv_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdcv_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdcv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdcv_v=[_coh3_steps[i] for i in range(_qdcv_n)
                                     if _coh3_steps[i] <= _qdcv_c3m and _velocity_steps[i] >= _qdcv_vm]:
                        (lambda _qdcv_mu=sum(_qdcv_v)/max(len(_qdcv_v),1):
                            round(sum((x-_qdcv_mu)**2 for x in _qdcv_v)/max(len(_qdcv_v)-1,1), 6)
                            if len(_qdcv_v) >= 2 else 0.0
                        )()
                    )()
                )()
            ))(),
            # variance of coh3 on drifting steps
            "quad_ideal_conf_var": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qicfv_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qicfv_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qicfv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qicfv_v=[output_confs[i] for i in range(_qicfv_n)
                                      if _coh3_steps[i] > _qicfv_c3m and _velocity_steps[i] < _qicfv_vm]:
                        (lambda _qicfv_mu=sum(_qicfv_v)/max(len(_qicfv_v),1):
                            round(sum((x-_qicfv_mu)**2 for x in _qicfv_v)/max(len(_qicfv_v)-1,1), 6)
                            if len(_qicfv_v) >= 2 else 0.0
                        )()
                    )()
                )()
            ))(),
            # variance of confidence on ideal steps
            "quad_drifting_conf_var": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qdcfv_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qdcfv_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdcfv_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdcfv_v=[output_confs[i] for i in range(_qdcfv_n)
                                      if _coh3_steps[i] <= _qdcfv_c3m and _velocity_steps[i] >= _qdcfv_vm]:
                        (lambda _qdcfv_mu=sum(_qdcfv_v)/max(len(_qdcfv_v),1):
                            round(sum((x-_qdcfv_mu)**2 for x in _qdcfv_v)/max(len(_qdcfv_v)-1,1), 6)
                            if len(_qdcfv_v) >= 2 else 0.0
                        )()
                    )()
                )()
            ))(),
            # variance of confidence on drifting steps
            "quad_ideal_vs_drift_coh3_var_ratio": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qivdcvr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qivdcvr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                        max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qivdcvr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                       max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qivdcvr_iv=[_coh3_steps[i] for i in range(_qivdcvr_n)
                                         if _coh3_steps[i] > _qivdcvr_c3m and _velocity_steps[i] < _qivdcvr_vm],
                           _qivdcvr_dv=[_coh3_steps[i] for i in range(_qivdcvr_n)
                                         if _coh3_steps[i] <= _qivdcvr_c3m and _velocity_steps[i] >= _qivdcvr_vm]:
                        (lambda _qivdcvr_ivar=(sum((x-sum(_qivdcvr_iv)/max(len(_qivdcvr_iv),1))**2
                                                   for x in _qivdcvr_iv)/max(len(_qivdcvr_iv)-1,1))
                                              if len(_qivdcvr_iv) >= 2 else 0.0,
                               _qivdcvr_dvar=(sum((x-sum(_qivdcvr_dv)/max(len(_qivdcvr_dv),1))**2
                                                   for x in _qivdcvr_dv)/max(len(_qivdcvr_dv)-1,1))
                                              if len(_qivdcvr_dv) >= 2 else 0.0:
                            round(_qivdcvr_ivar / max(_qivdcvr_dvar, 1e-12), 4)
                            if _qivdcvr_dvar > 1e-12 else 0.0
                        )()
                    )()
                )()
            ))(),
            # ideal_coh3_var / drifting_coh3_var — <1=ideal tighter than drift; >1=ideal noisier than drift
            "quad_coh3_conf_correlation": (lambda: (
                0.0 if min(len(_coh3_steps), len(output_confs)) < 4
                else (lambda _qccc_n=min(len(_coh3_steps), len(output_confs)),
                           _qccc_c3=_coh3_steps[:min(len(_coh3_steps), len(output_confs))],
                           _qccc_cf=output_confs[:min(len(_coh3_steps), len(output_confs))]:
                    (lambda _qccc_mc3=sum(_qccc_c3)/_qccc_n,
                           _qccc_mcf=sum(_qccc_cf)/_qccc_n:
                        (lambda _qccc_num=sum((_qccc_c3[i]-_qccc_mc3)*(_qccc_cf[i]-_qccc_mcf)
                                              for i in range(_qccc_n)),
                               _qccc_dc3=(sum((_qccc_c3[i]-_qccc_mc3)**2 for i in range(_qccc_n)))**0.5,
                               _qccc_dcf=(sum((_qccc_cf[i]-_qccc_mcf)**2 for i in range(_qccc_n)))**0.5:
                            round(_qccc_num / max(_qccc_dc3*_qccc_dcf, 1e-12), 4)
                            if _qccc_dc3 > 1e-9 and _qccc_dcf > 1e-9 else 0.0
                        )()
                    )()
                )()
            ))(),
            # Pearson correlation between coh3 and confidence across all steps (high=move together)
            "quad_coh3_vel_correlation": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qcvc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qcvc_c3=_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))],
                           _qcvc_vl=_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]:
                    (lambda _qcvc_mc3=sum(_qcvc_c3)/_qcvc_n,
                           _qcvc_mvl=sum(_qcvc_vl)/_qcvc_n:
                        (lambda _qcvc_num=sum((_qcvc_c3[i]-_qcvc_mc3)*(_qcvc_vl[i]-_qcvc_mvl)
                                              for i in range(_qcvc_n)),
                               _qcvc_dc3=(sum((_qcvc_c3[i]-_qcvc_mc3)**2 for i in range(_qcvc_n)))**0.5,
                               _qcvc_dvl=(sum((_qcvc_vl[i]-_qcvc_mvl)**2 for i in range(_qcvc_n)))**0.5:
                            round(_qcvc_num / max(_qcvc_dc3*_qcvc_dvl, 1e-12), 4)
                            if _qcvc_dc3 > 1e-9 and _qcvc_dvl > 1e-9 else 0.0
                        )()
                    )()
                )()
            ))(),
            # Pearson correlation between coh3 and velocity (expect negative: high coh3 → low vel)
            "quad_conf_vel_correlation": (lambda: (
                0.0 if min(len(output_confs), len(_velocity_steps)) < 4
                else (lambda _qcfvc_n=min(len(output_confs), len(_velocity_steps)),
                           _qcfvc_cf=output_confs[:min(len(output_confs), len(_velocity_steps))],
                           _qcfvc_vl=_velocity_steps[:min(len(output_confs), len(_velocity_steps))]:
                    (lambda _qcfvc_mcf=sum(_qcfvc_cf)/_qcfvc_n,
                           _qcfvc_mvl=sum(_qcfvc_vl)/_qcfvc_n:
                        (lambda _qcfvc_num=sum((_qcfvc_cf[i]-_qcfvc_mcf)*(_qcfvc_vl[i]-_qcfvc_mvl)
                                               for i in range(_qcfvc_n)),
                               _qcfvc_dcf=(sum((_qcfvc_cf[i]-_qcfvc_mcf)**2 for i in range(_qcfvc_n)))**0.5,
                               _qcfvc_dvl=(sum((_qcfvc_vl[i]-_qcfvc_mvl)**2 for i in range(_qcfvc_n)))**0.5:
                            round(_qcfvc_num / max(_qcfvc_dcf*_qcfvc_dvl, 1e-12), 4)
                            if _qcfvc_dcf > 1e-9 and _qcfvc_dvl > 1e-9 else 0.0
                        )()
                    )()
                )()
            ))(),
            # Pearson correlation between confidence and velocity
            "quad_ideal_coh3_conf_correlation": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qicc_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qicc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qicc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qicc_ic3=[_coh3_steps[i] for i in range(_qicc_n)
                                       if _coh3_steps[i] > _qicc_c3m and _velocity_steps[i] < _qicc_vm],
                           _qicc_icf=[output_confs[i] for i in range(_qicc_n)
                                      if _coh3_steps[i] > _qicc_c3m and _velocity_steps[i] < _qicc_vm]:
                        (lambda _qicc_k=min(len(_qicc_ic3), len(_qicc_icf)):
                            (lambda _qicc_mc3=sum(_qicc_ic3[:_qicc_k])/max(_qicc_k,1),
                                   _qicc_mcf=sum(_qicc_icf[:_qicc_k])/max(_qicc_k,1):
                                (lambda _qicc_num=sum((_qicc_ic3[i]-_qicc_mc3)*(_qicc_icf[i]-_qicc_mcf)
                                                      for i in range(_qicc_k)),
                                       _qicc_dc3=(sum((_qicc_ic3[i]-_qicc_mc3)**2 for i in range(_qicc_k)))**0.5,
                                       _qicc_dcf=(sum((_qicc_icf[i]-_qicc_mcf)**2 for i in range(_qicc_k)))**0.5:
                                    round(_qicc_num / max(_qicc_dc3*_qicc_dcf, 1e-12), 4)
                                    if _qicc_k >= 2 and _qicc_dc3 > 1e-9 and _qicc_dcf > 1e-9 else 0.0
                                )()
                            )()
                        )()
                    )()
                )()
            ))(),
            # coh3–conf correlation restricted to ideal steps only
            "quad_signal_coupling_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qscs_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qscs_c3=_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))],
                           _qscs_vl=_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))],
                           _qscs_cf=output_confs[:min(len(_coh3_steps), len(_velocity_steps), len(output_confs))]:
                    (lambda _qscs_mc3=sum(_qscs_c3)/_qscs_n,
                           _qscs_mvl=sum(_qscs_vl)/_qscs_n,
                           _qscs_mcf=sum(_qscs_cf)/_qscs_n:
                        (lambda _qscs_dc3=(sum((v-_qscs_mc3)**2 for v in _qscs_c3))**0.5,
                               _qscs_dvl=(sum((v-_qscs_mvl)**2 for v in _qscs_vl))**0.5,
                               _qscs_dcf=(sum((v-_qscs_mcf)**2 for v in _qscs_cf))**0.5:
                            (lambda _qscs_r_c3cf=(sum((_qscs_c3[i]-_qscs_mc3)*(_qscs_cf[i]-_qscs_mcf)
                                                      for i in range(_qscs_n)) /
                                                  max(_qscs_dc3*_qscs_dcf, 1e-12))
                                                 if _qscs_dc3 > 1e-9 and _qscs_dcf > 1e-9 else 0.0,
                                   _qscs_r_c3vl=(sum((_qscs_c3[i]-_qscs_mc3)*(_qscs_vl[i]-_qscs_mvl)
                                                      for i in range(_qscs_n)) /
                                                  max(_qscs_dc3*_qscs_dvl, 1e-12))
                                                 if _qscs_dc3 > 1e-9 and _qscs_dvl > 1e-9 else 0.0:
                                round(abs(_qscs_r_c3cf)*0.5 + (1.0-abs(_qscs_r_c3vl))*0.5, 4)
                            )()
                        )()
                    )()
                )()
            ))(),
            # coupling: abs(coh3_conf_corr)×0.5 + (1−abs(coh3_vel_corr))×0.5 [0,1]; high=coh3+conf aligned, vel decoupled
            "quad_coh3_autocorr_lag1": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _qca1_n=len(_coh3_steps),
                           _qca1_mu=sum(_coh3_steps)/max(len(_coh3_steps),1):
                    (lambda _qca1_num=sum((_coh3_steps[i]-_qca1_mu)*(_coh3_steps[i+1]-_qca1_mu)
                                          for i in range(_qca1_n-1)),
                           _qca1_den=sum((_coh3_steps[i]-_qca1_mu)**2 for i in range(_qca1_n)):
                        round(_qca1_num / max(_qca1_den, 1e-12), 4)
                        if _qca1_den > 1e-9 else 0.0
                    )()
                )()
            ))(),
            # lag-1 autocorrelation of coh3 series (high=smooth/persistent; low=choppy)
            "quad_vel_autocorr_lag1": (lambda: (
                0.0 if len(_velocity_steps) < 4
                else (lambda _qva1_n=len(_velocity_steps),
                           _qva1_mu=sum(_velocity_steps)/max(len(_velocity_steps),1):
                    (lambda _qva1_num=sum((_velocity_steps[i]-_qva1_mu)*(_velocity_steps[i+1]-_qva1_mu)
                                          for i in range(_qva1_n-1)),
                           _qva1_den=sum((_velocity_steps[i]-_qva1_mu)**2 for i in range(_qva1_n)):
                        round(_qva1_num / max(_qva1_den, 1e-12), 4)
                        if _qva1_den > 1e-9 else 0.0
                    )()
                )()
            ))(),
            # lag-1 autocorrelation of velocity series
            "quad_conf_autocorr_lag1": (lambda: (
                0.0 if len(output_confs) < 4
                else (lambda _qcfa1_n=len(output_confs),
                           _qcfa1_mu=sum(output_confs)/max(len(output_confs),1):
                    (lambda _qcfa1_num=sum((output_confs[i]-_qcfa1_mu)*(output_confs[i+1]-_qcfa1_mu)
                                           for i in range(_qcfa1_n-1)),
                           _qcfa1_den=sum((output_confs[i]-_qcfa1_mu)**2 for i in range(_qcfa1_n)):
                        round(_qcfa1_num / max(_qcfa1_den, 1e-12), 4)
                        if _qcfa1_den > 1e-9 else 0.0
                    )()
                )()
            ))(),
            # lag-1 autocorrelation of confidence series (high=conf is smooth/persistent)
            "quad_ideal_persistence_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qips_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qips_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qips_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qips_labs=[
                        1 if _coh3_steps[i] > _qips_c3m and _velocity_steps[i] < _qips_vm else 0
                        for i in range(_qips_n)
                    ]:
                        (lambda _qips_ii=sum(1 for j in range(1, _qips_n)
                                             if _qips_labs[j-1]==1 and _qips_labs[j]==1),
                               _qips_i=sum(_qips_labs[:-1]):
                            round(_qips_ii / max(_qips_i, 1), 4) if _qips_i > 0 else 0.0
                        )()
                    )()
                )()
            ))(),
            # P(ideal at t+1 | ideal at t) — probability ideal state persists for one more step
            "quad_drift_persistence_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdps_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdps_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdps_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdps_labs=[
                        1 if _coh3_steps[i] <= _qdps_c3m and _velocity_steps[i] >= _qdps_vm else 0
                        for i in range(_qdps_n)
                    ]:
                        (lambda _qdps_dd=sum(1 for j in range(1, _qdps_n)
                                             if _qdps_labs[j-1]==1 and _qdps_labs[j]==1),
                               _qdps_d=sum(_qdps_labs[:-1]):
                            round(_qdps_dd / max(_qdps_d, 1), 4) if _qdps_d > 0 else 0.0
                        )()
                    )()
                )()
            ))(),
            # P(drift at t+1 | drift at t) — probability drift state persists for one more step
            "quad_ideal_to_exploring_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qiter_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qiter_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qiter_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qiter_labs=[
                        "i" if _coh3_steps[j] > _qiter_c3m and _velocity_steps[j] < _qiter_vm
                        else "e" if _coh3_steps[j] > _qiter_c3m
                        else "d" if _velocity_steps[j] >= _qiter_vm
                        else "f" for j in range(_qiter_n)
                    ]:
                        (lambda _qiter_ie=sum(1 for j in range(1, _qiter_n)
                                              if _qiter_labs[j-1]=="i" and _qiter_labs[j]=="e"),
                               _qiter_i=sum(1 for j in range(_qiter_n-1)
                                            if _qiter_labs[j]=="i"):
                            round(_qiter_ie / max(_qiter_i, 1), 4) if _qiter_i > 0 else 0.0
                        )()
                    )()
                )()
            ))(),
            # P(exploring at t+1 | ideal at t)
            "quad_ideal_to_drifting_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qitdr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qitdr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qitdr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qitdr_labs=[
                        "i" if _coh3_steps[j] > _qitdr_c3m and _velocity_steps[j] < _qitdr_vm
                        else "e" if _coh3_steps[j] > _qitdr_c3m
                        else "d" if _velocity_steps[j] >= _qitdr_vm
                        else "f" for j in range(_qitdr_n)
                    ]:
                        (lambda _qitdr_id=sum(1 for j in range(1, _qitdr_n)
                                              if _qitdr_labs[j-1]=="i" and _qitdr_labs[j]=="d"),
                               _qitdr_i=sum(1 for j in range(_qitdr_n-1)
                                            if _qitdr_labs[j]=="i"):
                            round(_qitdr_id / max(_qitdr_i, 1), 4) if _qitdr_i > 0 else 0.0
                        )()
                    )()
                )()
            ))(),
            # P(drifting at t+1 | ideal at t) — direct quality-to-chaos transition
            "quad_exploring_to_ideal_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qetir_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qetir_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qetir_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qetir_labs=[
                        "i" if _coh3_steps[j] > _qetir_c3m and _velocity_steps[j] < _qetir_vm
                        else "e" if _coh3_steps[j] > _qetir_c3m
                        else "d" if _velocity_steps[j] >= _qetir_vm
                        else "f" for j in range(_qetir_n)
                    ]:
                        (lambda _qetir_ei=sum(1 for j in range(1, _qetir_n)
                                              if _qetir_labs[j-1]=="e" and _qetir_labs[j]=="i"),
                               _qetir_e=sum(1 for j in range(_qetir_n-1)
                                            if _qetir_labs[j]=="e"):
                            round(_qetir_ei / max(_qetir_e, 1), 4) if _qetir_e > 0 else 0.0
                        )()
                    )()
                )()
            ))(),
            # P(ideal at t+1 | exploring at t) — exploring becoming quality
            "quad_drifting_to_flat_rate": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdtfr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdtfr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdtfr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdtfr_labs=[
                        "i" if _coh3_steps[j] > _qdtfr_c3m and _velocity_steps[j] < _qdtfr_vm
                        else "e" if _coh3_steps[j] > _qdtfr_c3m
                        else "d" if _velocity_steps[j] >= _qdtfr_vm
                        else "f" for j in range(_qdtfr_n)
                    ]:
                        (lambda _qdtfr_df=sum(1 for j in range(1, _qdtfr_n)
                                              if _qdtfr_labs[j-1]=="d" and _qdtfr_labs[j]=="f"),
                               _qdtfr_d=sum(1 for j in range(_qdtfr_n-1)
                                            if _qdtfr_labs[j]=="d"):
                            round(_qdtfr_df / max(_qdtfr_d, 1), 4) if _qdtfr_d > 0 else 0.0
                        )()
                    )()
                )()
            ))(),
            # P(flat at t+1 | drifting at t) — drift decelerating into stagnation
            "quad_markov_stability_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qmss_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qmss_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qmss_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qmss_labs=[
                        "i" if _coh3_steps[j] > _qmss_c3m and _velocity_steps[j] < _qmss_vm
                        else "d" if _velocity_steps[j] >= _qmss_vm and _coh3_steps[j] <= _qmss_c3m
                        else "x" for j in range(_qmss_n)
                    ]:
                        (lambda _qmss_ii=sum(1 for j in range(1, _qmss_n)
                                             if _qmss_labs[j-1]=="i" and _qmss_labs[j]=="i"),
                               _qmss_i=sum(1 for j in range(_qmss_n-1) if _qmss_labs[j]=="i"),
                               _qmss_dd=sum(1 for j in range(1, _qmss_n)
                                            if _qmss_labs[j-1]=="d" and _qmss_labs[j]=="d"),
                               _qmss_d=sum(1 for j in range(_qmss_n-1) if _qmss_labs[j]=="d"):
                            round(
                                0.6*(_qmss_ii/max(_qmss_i,1) if _qmss_i > 0 else 0.0)
                                + 0.4*(1.0-_qmss_dd/max(_qmss_d,1)) if _qmss_d > 0 else
                                0.6*(_qmss_ii/max(_qmss_i,1) if _qmss_i > 0 else 0.0) + 0.4,
                                4)
                        )()
                    )()
                )()
            ))(),
            # 0.6×P(ideal stays) + 0.4×(1−P(drift stays)); high=ideal sticky and drift short-lived
            "quad_label_entropy": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qle_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qle_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qle_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qle_fi=sum(1 for j in range(_qle_n)
                                        if _coh3_steps[j] > _qle_c3m and _velocity_steps[j] < _qle_vm),
                           _qle_fe=sum(1 for j in range(_qle_n)
                                       if _coh3_steps[j] > _qle_c3m and _velocity_steps[j] >= _qle_vm),
                           _qle_fd=sum(1 for j in range(_qle_n)
                                       if _coh3_steps[j] <= _qle_c3m and _velocity_steps[j] >= _qle_vm),
                           _qle_ff=sum(1 for j in range(_qle_n)
                                       if _coh3_steps[j] <= _qle_c3m and _velocity_steps[j] < _qle_vm):
                        round(
                            -sum(
                                (p/_qle_n)*__import__("math").log2(p/_qle_n)
                                for p in [_qle_fi, _qle_fe, _qle_fd, _qle_ff] if p > 0
                            ) / __import__("math").log2(4),
                            4)
                    )()
                )()
            ))(),
            # Shannon entropy of {i,e,d,f} distribution normalized by log2(4) ∈[0,1]; high=all used equally
            "quad_coh3_entropy": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _qce_n=len(_coh3_steps),
                           _qce_lo=min(_coh3_steps), _qce_hi=max(_coh3_steps):
                    (lambda _qce_w=max(_qce_hi-_qce_lo, 1e-12):
                        (lambda _qce_bins=[0]*8:
                            (lambda _qce_fill=[
                                _qce_bins.__setitem__(min(7, int((v-_qce_lo)/_qce_w*8)), 1)
                                or _qce_bins.__setitem__(min(7, int((v-_qce_lo)/_qce_w*8)),
                                    _qce_bins[min(7, int((v-_qce_lo)/_qce_w*8))]+1)
                                for v in _coh3_steps] or True:
                                round(
                                    -sum(
                                        (c/_qce_n)*__import__("math").log2(c/_qce_n)
                                        for c in _qce_bins if c > 0
                                    ) / __import__("math").log2(8),
                                    4)
                            )()
                        )()
                    )()
                )()
            ))(),
            # normalized Shannon entropy of coh3 values binned into 8 buckets [0,1]
            "quad_vel_entropy": (lambda: (
                0.0 if len(_velocity_steps) < 4
                else (lambda _qve_n=len(_velocity_steps),
                           _qve_lo=min(_velocity_steps), _qve_hi=max(_velocity_steps):
                    (lambda _qve_w=max(_qve_hi-_qve_lo, 1e-12):
                        (lambda _qve_bins=[0]*8:
                            (lambda _qve_fill=[
                                _qve_bins.__setitem__(min(7, int((v-_qve_lo)/_qve_w*8)), 1)
                                or _qve_bins.__setitem__(min(7, int((v-_qve_lo)/_qve_w*8)),
                                    _qve_bins[min(7, int((v-_qve_lo)/_qve_w*8))]+1)
                                for v in _velocity_steps] or True:
                                round(
                                    -sum(
                                        (c/_qve_n)*__import__("math").log2(c/_qve_n)
                                        for c in _qve_bins if c > 0
                                    ) / __import__("math").log2(8),
                                    4)
                            )()
                        )()
                    )()
                )()
            ))(),
            # normalized Shannon entropy of velocity values binned into 8 buckets [0,1]
            "quad_conf_entropy": (lambda: (
                0.0 if len(output_confs) < 4
                else (lambda _qcfe_n=len(output_confs),
                           _qcfe_lo=min(output_confs), _qcfe_hi=max(output_confs):
                    (lambda _qcfe_w=max(_qcfe_hi-_qcfe_lo, 1e-12):
                        (lambda _qcfe_bins=[0]*8:
                            (lambda _qcfe_fill=[
                                _qcfe_bins.__setitem__(min(7, int((v-_qcfe_lo)/_qcfe_w*8)), 1)
                                or _qcfe_bins.__setitem__(min(7, int((v-_qcfe_lo)/_qcfe_w*8)),
                                    _qcfe_bins[min(7, int((v-_qcfe_lo)/_qcfe_w*8))]+1)
                                for v in output_confs] or True:
                                round(
                                    -sum(
                                        (c/_qcfe_n)*__import__("math").log2(c/_qcfe_n)
                                        for c in _qcfe_bins if c > 0
                                    ) / __import__("math").log2(8),
                                    4)
                            )()
                        )()
                    )()
                )()
            ))(),
            # normalized Shannon entropy of confidence values binned into 8 buckets [0,1]
            "quad_entropy_index": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qei_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qei_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qei_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qei_fi=sum(1 for j in range(_qei_n)
                                        if _coh3_steps[j] > _qei_c3m and _velocity_steps[j] < _qei_vm),
                           _qei_fe=sum(1 for j in range(_qei_n)
                                       if _coh3_steps[j] > _qei_c3m and _velocity_steps[j] >= _qei_vm),
                           _qei_fd=sum(1 for j in range(_qei_n)
                                       if _coh3_steps[j] <= _qei_c3m and _velocity_steps[j] >= _qei_vm),
                           _qei_ff=sum(1 for j in range(_qei_n)
                                       if _coh3_steps[j] <= _qei_c3m and _velocity_steps[j] < _qei_vm),
                           _qei_c3e=(lambda _n=len(_coh3_steps), _lo=min(_coh3_steps),
                                          _hi=max(_coh3_steps):
                                (lambda _w=max(_hi-_lo,1e-12):
                                    (lambda _b=[0]*8:
                                        [_b.__setitem__(min(7,int((v-_lo)/_w*8)),_b[min(7,int((v-_lo)/_w*8))]+1)
                                         for v in _coh3_steps] or
                                        -sum((c/_n)*__import__("math").log2(c/_n) for c in _b if c>0) /
                                        __import__("math").log2(8)
                                    )()
                                )()
                            )() if len(_coh3_steps) >= 4 else 0.0,
                           _qei_ve=(lambda _n=len(_velocity_steps), _lo=min(_velocity_steps),
                                         _hi=max(_velocity_steps):
                                (lambda _w=max(_hi-_lo,1e-12):
                                    (lambda _b=[0]*8:
                                        [_b.__setitem__(min(7,int((v-_lo)/_w*8)),_b[min(7,int((v-_lo)/_w*8))]+1)
                                         for v in _velocity_steps] or
                                        -sum((c/_n)*__import__("math").log2(c/_n) for c in _b if c>0) /
                                        __import__("math").log2(8)
                                    )()
                                )()
                            )() if len(_velocity_steps) >= 4 else 0.0:
                        round(
                            0.4*(
                                -sum((p/_qei_n)*__import__("math").log2(p/_qei_n)
                                     for p in [_qei_fi, _qei_fe, _qei_fd, _qei_ff] if p > 0)
                                / __import__("math").log2(4)
                            )
                            + 0.3*_qei_c3e
                            + 0.3*(1.0-_qei_ve),
                            4)
                    )()
                )()
            ))(),
            # 0.4×label_entropy_norm + 0.3×coh3_entropy_norm + 0.3×(1−vel_entropy_norm)
            "quad_coh3_trend_slope": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _qcts_n=len(_coh3_steps),
                           _qcts_xs=list(range(len(_coh3_steps))):
                    (lambda _qcts_mx=sum(_qcts_xs)/_qcts_n,
                           _qcts_my=sum(_coh3_steps)/_qcts_n:
                        (lambda _qcts_num=sum((_qcts_xs[i]-_qcts_mx)*(_coh3_steps[i]-_qcts_my)
                                               for i in range(_qcts_n)),
                               _qcts_den=sum((_qcts_xs[i]-_qcts_mx)**2 for i in range(_qcts_n)):
                            round(_qcts_num / max(_qcts_den, 1e-12), 6)
                            if _qcts_den > 1e-9 else 0.0
                        )()
                    )()
                )()
            ))(),
            # linear regression slope of coh3 over step index (positive=rising coherence)
            "quad_vel_trend_slope": (lambda: (
                0.0 if len(_velocity_steps) < 4
                else (lambda _qvts_n=len(_velocity_steps),
                           _qvts_xs=list(range(len(_velocity_steps))):
                    (lambda _qvts_mx=sum(_qvts_xs)/_qvts_n,
                           _qvts_my=sum(_velocity_steps)/_qvts_n:
                        (lambda _qvts_num=sum((_qvts_xs[i]-_qvts_mx)*(_velocity_steps[i]-_qvts_my)
                                               for i in range(_qvts_n)),
                               _qvts_den=sum((_qvts_xs[i]-_qvts_mx)**2 for i in range(_qvts_n)):
                            round(_qvts_num / max(_qvts_den, 1e-12), 6)
                            if _qvts_den > 1e-9 else 0.0
                        )()
                    )()
                )()
            ))(),
            # linear regression slope of velocity over step index (negative=slowing=good)
            "quad_conf_trend_slope": (lambda: (
                0.0 if len(output_confs) < 4
                else (lambda _qcfts_n=len(output_confs),
                           _qcfts_xs=list(range(len(output_confs))):
                    (lambda _qcfts_mx=sum(_qcfts_xs)/_qcfts_n,
                           _qcfts_my=sum(output_confs)/_qcfts_n:
                        (lambda _qcfts_num=sum((_qcfts_xs[i]-_qcfts_mx)*(output_confs[i]-_qcfts_my)
                                                for i in range(_qcfts_n)),
                               _qcfts_den=sum((_qcfts_xs[i]-_qcfts_mx)**2 for i in range(_qcfts_n)):
                            round(_qcfts_num / max(_qcfts_den, 1e-12), 6)
                            if _qcfts_den > 1e-9 else 0.0
                        )()
                    )()
                )()
            ))(),
            # linear regression slope of confidence over step index
            "quad_coh3_trend_r2": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _qctr2_n=len(_coh3_steps),
                           _qctr2_xs=list(range(len(_coh3_steps))):
                    (lambda _qctr2_mx=sum(_qctr2_xs)/_qctr2_n,
                           _qctr2_my=sum(_coh3_steps)/_qctr2_n:
                        (lambda _qctr2_num=sum((_qctr2_xs[i]-_qctr2_mx)*(_coh3_steps[i]-_qctr2_my)
                                                for i in range(_qctr2_n)),
                               _qctr2_denx=sum((_qctr2_xs[i]-_qctr2_mx)**2 for i in range(_qctr2_n)),
                               _qctr2_deny=sum((_coh3_steps[i]-_qctr2_my)**2 for i in range(_qctr2_n)):
                            round((_qctr2_num**2) / max(_qctr2_denx*_qctr2_deny, 1e-24), 4)
                            if _qctr2_denx > 1e-9 and _qctr2_deny > 1e-9 else 0.0
                        )()
                    )()
                )()
            ))(),
            # R² of coh3 linear fit (high=strong linear trend; low=noisy/non-linear)
            "quad_trend_alignment_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qtas_nc3=len(_coh3_steps),
                           _qtas_nv=len(_velocity_steps),
                           _qtas_xsc3=list(range(len(_coh3_steps))),
                           _qtas_xsv=list(range(len(_velocity_steps))):
                    (lambda _qtas_mc3x=sum(_qtas_xsc3)/_qtas_nc3,
                           _qtas_mc3y=sum(_coh3_steps)/_qtas_nc3,
                           _qtas_mvx=sum(_qtas_xsv)/_qtas_nv,
                           _qtas_mvy=sum(_velocity_steps)/_qtas_nv:
                        (lambda _qtas_sc3=(sum((_qtas_xsc3[i]-_qtas_mc3x)*(_coh3_steps[i]-_qtas_mc3y)
                                               for i in range(_qtas_nc3)) /
                                           max(sum((_qtas_xsc3[i]-_qtas_mc3x)**2
                                                   for i in range(_qtas_nc3)), 1e-12)),
                               _qtas_sv=(sum((_qtas_xsv[i]-_qtas_mvx)*(_velocity_steps[i]-_qtas_mvy)
                                              for i in range(_qtas_nv)) /
                                          max(sum((_qtas_xsv[i]-_qtas_mvx)**2
                                                  for i in range(_qtas_nv)), 1e-12)):
                            round(
                                0.5*min(max(_qtas_sc3*100, 0.0), 1.0)
                                + 0.5*min(max(-_qtas_sv*100, 0.0), 1.0),
                                4)
                        )()
                    )()
                )()
            ))(),
            # 0.5×clamp(coh3_slope×100,[0,1]) + 0.5×clamp(-vel_slope×100,[0,1]); high=coh3 rising+vel falling
            "quad_first_half_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qfhi_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfhi_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfhi_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfhi_h=_qfhi_n//2:
                        round(sum(1 for i in range(_qfhi_h)
                                  if _coh3_steps[i] > _qfhi_c3m and _velocity_steps[i] < _qfhi_vm)
                              / max(_qfhi_h, 1), 4)
                    )()
                )()
            ))(),
            # ideal_frac in first half of the generation
            "quad_second_half_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qshi_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qshi_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qshi_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qshi_h=_qshi_n//2:
                        round(sum(1 for i in range(_qshi_h, _qshi_n)
                                  if _coh3_steps[i] > _qshi_c3m and _velocity_steps[i] < _qshi_vm)
                              / max(_qshi_n - _qshi_h, 1), 4)
                    )()
                )()
            ))(),
            # ideal_frac in second half of the generation
            "quad_first_half_coh3_mean": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _qfhc_n=len(_coh3_steps):
                    (lambda _qfhc_h=_qfhc_n//2:
                        round(sum(_coh3_steps[:_qfhc_h]) / max(_qfhc_h, 1), 6)
                    )()
                )()
            ))(),
            # mean coh3 in first half of the generation
            "quad_second_half_coh3_mean": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _qshc_n=len(_coh3_steps):
                    (lambda _qshc_h=_qshc_n//2:
                        round(sum(_coh3_steps[_qshc_h:]) / max(_qshc_n - _qshc_h, 1), 6)
                    )()
                )()
            ))(),
            # mean coh3 in second half of the generation
            "quad_half_improvement_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qhis_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qhis_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qhis_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qhis_h=_qhis_n//2:
                        (lambda _qhis_fi=sum(1 for i in range(_qhis_h)
                                             if _coh3_steps[i] > _qhis_c3m and _velocity_steps[i] < _qhis_vm)
                                          / max(_qhis_h, 1),
                               _qhis_si=sum(1 for i in range(_qhis_h, _qhis_n)
                                            if _coh3_steps[i] > _qhis_c3m and _velocity_steps[i] < _qhis_vm)
                                          / max(_qhis_n - _qhis_h, 1),
                               _qhis_fc3=sum(_coh3_steps[:_qhis_h]) / max(_qhis_h, 1),
                               _qhis_sc3=sum(_coh3_steps[_qhis_h:_qhis_n]) / max(_qhis_n - _qhis_h, 1):
                            round(
                                0.6*(_qhis_si - _qhis_fi)
                                + 0.4*min(max((_qhis_sc3 - _qhis_fc3)*10, -1.0), 1.0),
                                4)
                        )()
                    )()
                )()
            ))(),
            # 0.6×(2nd_ideal_frac - 1st_ideal_frac) + 0.4×norm(coh3_delta); positive=improving generation
            "quad_q1_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qq1i_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qq1i_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qq1i_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qq1i_q=max(_qq1i_n//4, 1):
                        round(sum(1 for i in range(_qq1i_q)
                                  if _coh3_steps[i] > _qq1i_c3m and _velocity_steps[i] < _qq1i_vm)
                              / _qq1i_q, 4)
                    )()
                )()
            ))(),
            # ideal_frac in first quarter of generation
            "quad_q4_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qq4i_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qq4i_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qq4i_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qq4i_q=max(_qq4i_n//4, 1):
                        round(sum(1 for i in range(_qq4i_n - _qq4i_q, _qq4i_n)
                                  if _coh3_steps[i] > _qq4i_c3m and _velocity_steps[i] < _qq4i_vm)
                              / _qq4i_q, 4)
                    )()
                )()
            ))(),
            # ideal_frac in last quarter of generation
            "quad_q1_coh3_mean": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _qq1c_n=len(_coh3_steps):
                    (lambda _qq1c_q=max(_qq1c_n//4, 1):
                        round(sum(_coh3_steps[:_qq1c_q]) / _qq1c_q, 6)
                    )()
                )()
            ))(),
            # mean coh3 in first quarter
            "quad_q4_coh3_mean": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _qq4c_n=len(_coh3_steps):
                    (lambda _qq4c_q=max(_qq4c_n//4, 1):
                        round(sum(_coh3_steps[_qq4c_n - _qq4c_q:]) / _qq4c_q, 6)
                    )()
                )()
            ))(),
            # mean coh3 in last quarter
            "quad_quarter_arc_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 8
                else (lambda _qqas_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qqas_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qqas_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qqas_q=max(_qqas_n//4, 1):
                        (lambda _qqas_qi=sum(1 for i in range(_qqas_q)
                                             if _coh3_steps[i] > _qqas_c3m and _velocity_steps[i] < _qqas_vm)
                                          / _qqas_q,
                               _qqas_q4i=sum(1 for i in range(_qqas_n - _qqas_q, _qqas_n)
                                             if _coh3_steps[i] > _qqas_c3m and _velocity_steps[i] < _qqas_vm)
                                           / _qqas_q,
                               _qqas_qc3=sum(_coh3_steps[:_qqas_q]) / _qqas_q,
                               _qqas_q4c3=sum(_coh3_steps[_qqas_n - _qqas_q:_qqas_n]) / _qqas_q:
                            round(
                                0.6*(_qqas_q4i - _qqas_qi)
                                + 0.4*min(max((_qqas_q4c3 - _qqas_qc3)*10, -1.0), 1.0),
                                4)
                        )()
                    )()
                )()
            ))(),
            # 0.6×(Q4_ideal - Q1_ideal) + 0.4×norm_coh3_delta; positive=generation has rising arc
            "quad_ideal_vel_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qivm_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qivm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qivm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qivm_v=[_velocity_steps[i] for i in range(_qivm_n)
                                     if _coh3_steps[i] > _qivm_c3m and _velocity_steps[i] < _qivm_vm]:
                        round(sum(_qivm_v)/max(len(_qivm_v),1), 6) if _qivm_v else 0.0
                    )()
                )()
            ))(),
            # mean velocity on ideal steps (low=ideal steps are slow/focused as expected)
            "quad_drifting_vel_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qdvm_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdvm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdvm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdvm_v=[_velocity_steps[i] for i in range(_qdvm_n)
                                     if _coh3_steps[i] <= _qdvm_c3m and _velocity_steps[i] >= _qdvm_vm]:
                        round(sum(_qdvm_v)/max(len(_qdvm_v),1), 6) if _qdvm_v else 0.0
                    )()
                )()
            ))(),
            # mean velocity on drifting steps (high=drifting steps are fast/chaotic as expected)
            "quad_exploring_vel_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qevm_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qevm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qevm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qevm_v=[_velocity_steps[i] for i in range(_qevm_n)
                                     if _coh3_steps[i] > _qevm_c3m and _velocity_steps[i] >= _qevm_vm]:
                        round(sum(_qevm_v)/max(len(_qevm_v),1), 6) if _qevm_v else 0.0
                    )()
                )()
            ))(),
            # mean velocity on exploring steps (high=exploring is fast but coherent)
            "quad_flat_vel_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qfvm_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfvm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfvm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfvm_v=[_velocity_steps[i] for i in range(_qfvm_n)
                                     if _coh3_steps[i] <= _qfvm_c3m and _velocity_steps[i] < _qfvm_vm]:
                        round(sum(_qfvm_v)/max(len(_qfvm_v),1), 6) if _qfvm_v else 0.0
                    )()
                )()
            ))(),
            # mean velocity on flat steps (low=flat steps are slow but incoherent)
            "quad_vel_quadrant_spread": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qvqs_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qvqs_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qvqs_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qvqs_iv=[_velocity_steps[i] for i in range(_qvqs_n)
                                      if _coh3_steps[i] > _qvqs_c3m and _velocity_steps[i] < _qvqs_vm],
                           _qvqs_ev=[_velocity_steps[i] for i in range(_qvqs_n)
                                     if _coh3_steps[i] > _qvqs_c3m and _velocity_steps[i] >= _qvqs_vm],
                           _qvqs_dv=[_velocity_steps[i] for i in range(_qvqs_n)
                                     if _coh3_steps[i] <= _qvqs_c3m and _velocity_steps[i] >= _qvqs_vm],
                           _qvqs_fv=[_velocity_steps[i] for i in range(_qvqs_n)
                                     if _coh3_steps[i] <= _qvqs_c3m and _velocity_steps[i] < _qvqs_vm]:
                        (lambda _qvqs_means=[
                            sum(v)/max(len(v),1) for v in [_qvqs_iv, _qvqs_ev, _qvqs_dv, _qvqs_fv] if v
                        ]:
                            (lambda _qvqs_mu=sum(_qvqs_means)/max(len(_qvqs_means),1):
                                round(
                                    (sum((m-_qvqs_mu)**2 for m in _qvqs_means)/max(len(_qvqs_means)-1,1))**0.5,
                                    6) if len(_qvqs_means) >= 2 else 0.0
                            )()
                        )()
                    )()
                )()
            ))(),
            # std of the 4 per-quadrant velocity means (high=quadrants differ greatly in speed)
            "quad_ideal_conf_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qicm_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qicm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qicm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qicm_v=[output_confs[i] for i in range(_qicm_n)
                                     if _coh3_steps[i] > _qicm_c3m and _velocity_steps[i] < _qicm_vm]:
                        round(sum(_qicm_v)/max(len(_qicm_v),1), 6) if _qicm_v else 0.0
                    )()
                )()
            ))(),
            # mean confidence on ideal steps
            "quad_drifting_conf_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qdcm_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qdcm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdcm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdcm_v=[output_confs[i] for i in range(_qdcm_n)
                                     if _coh3_steps[i] <= _qdcm_c3m and _velocity_steps[i] >= _qdcm_vm]:
                        round(sum(_qdcm_v)/max(len(_qdcm_v),1), 6) if _qdcm_v else 0.0
                    )()
                )()
            ))(),
            # mean confidence on drifting steps
            "quad_exploring_conf_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qecm_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qecm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qecm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qecm_v=[output_confs[i] for i in range(_qecm_n)
                                     if _coh3_steps[i] > _qecm_c3m and _velocity_steps[i] >= _qecm_vm]:
                        round(sum(_qecm_v)/max(len(_qecm_v),1), 6) if _qecm_v else 0.0
                    )()
                )()
            ))(),
            # mean confidence on exploring steps
            "quad_flat_conf_mean": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qfcm_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qfcm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfcm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qfcm_v=[output_confs[i] for i in range(_qfcm_n)
                                     if _coh3_steps[i] <= _qfcm_c3m and _velocity_steps[i] < _qfcm_vm]:
                        round(sum(_qfcm_v)/max(len(_qfcm_v),1), 6) if _qfcm_v else 0.0
                    )()
                )()
            ))(),
            # mean confidence on flat steps
            "quad_conf_quadrant_spread": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qcqs_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qcqs_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qcqs_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qcqs_iv=[output_confs[i] for i in range(_qcqs_n)
                                      if _coh3_steps[i] > _qcqs_c3m and _velocity_steps[i] < _qcqs_vm],
                           _qcqs_ev=[output_confs[i] for i in range(_qcqs_n)
                                     if _coh3_steps[i] > _qcqs_c3m and _velocity_steps[i] >= _qcqs_vm],
                           _qcqs_dv=[output_confs[i] for i in range(_qcqs_n)
                                     if _coh3_steps[i] <= _qcqs_c3m and _velocity_steps[i] >= _qcqs_vm],
                           _qcqs_fv=[output_confs[i] for i in range(_qcqs_n)
                                     if _coh3_steps[i] <= _qcqs_c3m and _velocity_steps[i] < _qcqs_vm]:
                        (lambda _qcqs_means=[
                            sum(v)/max(len(v),1) for v in [_qcqs_iv, _qcqs_ev, _qcqs_dv, _qcqs_fv] if v
                        ]:
                            (lambda _qcqs_mu=sum(_qcqs_means)/max(len(_qcqs_means),1):
                                round(
                                    (sum((m-_qcqs_mu)**2 for m in _qcqs_means)/max(len(_qcqs_means)-1,1))**0.5,
                                    6) if len(_qcqs_means) >= 2 else 0.0
                            )()
                        )()
                    )()
                )()
            ))(),
            # σ of the 4 per-quadrant confidence means (high=quadrants differ greatly in confidence)
            "quad_ideal_vs_drift_conf_gap": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qidc_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qidc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qidc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qidc_iv=[output_confs[i] for i in range(_qidc_n)
                                      if _coh3_steps[i] > _qidc_c3m and _velocity_steps[i] < _qidc_vm],
                           _qidc_dv=[output_confs[i] for i in range(_qidc_n)
                                     if _coh3_steps[i] <= _qidc_c3m and _velocity_steps[i] >= _qidc_vm]:
                        round(
                            sum(_qidc_iv)/max(len(_qidc_iv),1)
                            - sum(_qidc_dv)/max(len(_qidc_dv),1),
                            6)
                        if _qidc_iv and _qidc_dv else 0.0
                    )()
                )()
            ))(),
            # ideal_conf_mean − drifting_conf_mean; positive=ideal steps are more confident
            "quad_ideal_vs_drift_coh3_gap": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qidc3_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qidc3_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qidc3_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qidc3_iv=[_coh3_steps[i] for i in range(_qidc3_n)
                                       if _coh3_steps[i] > _qidc3_c3m and _velocity_steps[i] < _qidc3_vm],
                           _qidc3_dv=[_coh3_steps[i] for i in range(_qidc3_n)
                                      if _coh3_steps[i] <= _qidc3_c3m and _velocity_steps[i] >= _qidc3_vm]:
                        round(
                            sum(_qidc3_iv)/max(len(_qidc3_iv),1)
                            - sum(_qidc3_dv)/max(len(_qidc3_dv),1),
                            6)
                        if _qidc3_iv and _qidc3_dv else 0.0
                    )()
                )()
            ))(),
            # ideal_coh3_mean − drifting_coh3_mean; positive=ideal more coherent (expected)
            "quad_ideal_vs_flat_coh3_gap": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qifc3_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qifc3_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qifc3_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qifc3_iv=[_coh3_steps[i] for i in range(_qifc3_n)
                                       if _coh3_steps[i] > _qifc3_c3m and _velocity_steps[i] < _qifc3_vm],
                           _qifc3_fv=[_coh3_steps[i] for i in range(_qifc3_n)
                                      if _coh3_steps[i] <= _qifc3_c3m and _velocity_steps[i] < _qifc3_vm]:
                        round(
                            sum(_qifc3_iv)/max(len(_qifc3_iv),1)
                            - sum(_qifc3_fv)/max(len(_qifc3_fv),1),
                            6)
                        if _qifc3_iv and _qifc3_fv else 0.0
                    )()
                )()
            ))(),
            # ideal_coh3_mean − flat_coh3_mean; positive=ideal is coherent, flat is not
            "quad_quality_gap_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qqgs_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qqgs_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qqgs_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qqgs_ic=[output_confs[i] for i in range(_qqgs_n)
                                      if _coh3_steps[i] > _qqgs_c3m and _velocity_steps[i] < _qqgs_vm],
                           _qqgs_dc=[output_confs[i] for i in range(_qqgs_n)
                                     if _coh3_steps[i] <= _qqgs_c3m and _velocity_steps[i] >= _qqgs_vm],
                           _qqgs_ic3=[_coh3_steps[i] for i in range(_qqgs_n)
                                      if _coh3_steps[i] > _qqgs_c3m and _velocity_steps[i] < _qqgs_vm],
                           _qqgs_dc3=[_coh3_steps[i] for i in range(_qqgs_n)
                                      if _coh3_steps[i] <= _qqgs_c3m and _velocity_steps[i] >= _qqgs_vm]:
                        (lambda _qqgs_cf_gap=(sum(_qqgs_ic)/max(len(_qqgs_ic),1)
                                              - sum(_qqgs_dc)/max(len(_qqgs_dc),1))
                                             if _qqgs_ic and _qqgs_dc else 0.0,
                               _qqgs_c3_gap=(sum(_qqgs_ic3)/max(len(_qqgs_ic3),1)
                                              - sum(_qqgs_dc3)/max(len(_qqgs_dc3),1))
                                             if _qqgs_ic3 and _qqgs_dc3 else 0.0,
                               _qqgs_c3_range=max(_coh3_steps)-min(_coh3_steps)+1e-9,
                               _qqgs_cf_range=max(output_confs[:_qqgs_n])-min(output_confs[:_qqgs_n])+1e-9:
                            round(
                                0.4*min(max(_qqgs_cf_gap/_qqgs_cf_range, 0.0), 1.0)
                                + 0.6*min(max(_qqgs_c3_gap/_qqgs_c3_range, 0.0), 1.0),
                                4)
                        )()
                    )()
                )()
            ))(),
            # 0.4×norm_conf_gap + 0.6×norm_coh3_gap [0,1]; high=ideal and drift clearly separated
            "quad_signal_separation_index": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qssi_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qssi_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qssi_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qssi_ic3=[_coh3_steps[i] for i in range(_qssi_n)
                                       if _coh3_steps[i] > _qssi_c3m and _velocity_steps[i] < _qssi_vm],
                           _qssi_dc3=[_coh3_steps[i] for i in range(_qssi_n)
                                      if _coh3_steps[i] <= _qssi_c3m and _velocity_steps[i] >= _qssi_vm]:
                        (lambda _qssi_mu=sum(_coh3_steps[:_qssi_n])/_qssi_n:
                            (lambda _qssi_std=max(
                                (sum((_coh3_steps[i]-_qssi_mu)**2 for i in range(_qssi_n))/_qssi_n)**0.5,
                                1e-9):
                                round(
                                    (sum(_qssi_ic3)/max(len(_qssi_ic3),1)
                                     - sum(_qssi_dc3)/max(len(_qssi_dc3),1))
                                    / _qssi_std,
                                    4)
                                if _qssi_ic3 and _qssi_dc3 else 0.0
                            )()
                        )()
                    )()
                )()
            ))(),
            # (ideal_coh3_mean − drift_coh3_mean) / coh3_std — Cohen's d analogue; high=well-separated
            "quad_coh3_volatility": (lambda: (
                0.0 if len(_coh3_steps) < 3
                else (lambda _qcv_d=[abs(_coh3_steps[i+1]-_coh3_steps[i])
                                     for i in range(len(_coh3_steps)-1)]:
                    (lambda _qcv_mu=sum(_qcv_d)/max(len(_qcv_d),1):
                        round((sum((x-_qcv_mu)**2 for x in _qcv_d)/max(len(_qcv_d)-1,1))**0.5, 6)
                    )()
                )()
            ))(),
            # std of step-to-step |Δcoh3| (high=coh3 is erratic; low=coh3 changes smoothly)
            "quad_vel_volatility": (lambda: (
                0.0 if len(_velocity_steps) < 3
                else (lambda _qvv_d=[abs(_velocity_steps[i+1]-_velocity_steps[i])
                                     for i in range(len(_velocity_steps)-1)]:
                    (lambda _qvv_mu=sum(_qvv_d)/max(len(_qvv_d),1):
                        round((sum((x-_qvv_mu)**2 for x in _qvv_d)/max(len(_qvv_d)-1,1))**0.5, 6)
                    )()
                )()
            ))(),
            # std of step-to-step |Δvelocity| (high=velocity is erratic)
            "quad_conf_volatility": (lambda: (
                0.0 if len(output_confs) < 3
                else (lambda _qcfv_d=[abs(output_confs[i+1]-output_confs[i])
                                      for i in range(len(output_confs)-1)]:
                    (lambda _qcfv_mu=sum(_qcfv_d)/max(len(_qcfv_d),1):
                        round((sum((x-_qcfv_mu)**2 for x in _qcfv_d)/max(len(_qcfv_d)-1,1))**0.5, 6)
                    )()
                )()
            ))(),
            # std of step-to-step |Δconfidence| (high=confidence is erratic)
            "quad_coh3_vel_volatility_ratio": (lambda: (
                0.0 if len(_coh3_steps) < 3 or len(_velocity_steps) < 3
                else (lambda _qcvr_cd=[abs(_coh3_steps[i+1]-_coh3_steps[i])
                                       for i in range(len(_coh3_steps)-1)],
                           _qcvr_vd=[abs(_velocity_steps[i+1]-_velocity_steps[i])
                                     for i in range(len(_velocity_steps)-1)]:
                    (lambda _qcvr_cs=(sum((x-sum(_qcvr_cd)/max(len(_qcvr_cd),1))**2
                                         for x in _qcvr_cd)/max(len(_qcvr_cd)-1,1))**0.5,
                           _qcvr_vs=(sum((x-sum(_qcvr_vd)/max(len(_qcvr_vd),1))**2
                                         for x in _qcvr_vd)/max(len(_qcvr_vd)-1,1))**0.5:
                        round(_qcvr_cs/max(_qcvr_vs, 1e-9), 4)
                    )()
                )()
            ))(),
            # coh3_volatility / vel_volatility; >1=coh3 is more erratic than velocity (unusual)
            "quad_stability_composite": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 3
                else (lambda _qsc_cd=[abs(_coh3_steps[i+1]-_coh3_steps[i])
                                      for i in range(len(_coh3_steps)-1)],
                           _qsc_vd=[abs(_velocity_steps[i+1]-_velocity_steps[i])
                                    for i in range(len(_velocity_steps)-1)],
                           _qsc_fd=[abs(output_confs[i+1]-output_confs[i])
                                    for i in range(len(output_confs)-1)]:
                    (lambda _qsc_cv=(sum((x-sum(_qsc_cd)/max(len(_qsc_cd),1))**2
                                        for x in _qsc_cd)/max(len(_qsc_cd)-1,1))**0.5,
                           _qsc_vv=(sum((x-sum(_qsc_vd)/max(len(_qsc_vd),1))**2
                                        for x in _qsc_vd)/max(len(_qsc_vd)-1,1))**0.5,
                           _qsc_fv=(sum((x-sum(_qsc_fd)/max(len(_qsc_fd),1))**2
                                        for x in _qsc_fd)/max(len(_qsc_fd)-1,1))**0.5:
                        round(
                            max(0.0, 1.0 - (_qsc_cv + _qsc_vv + _qsc_fv) / 3.0),
                            4)
                    )()
                )()
            ))(),
            # 1 − mean(coh3_vol, vel_vol, conf_vol) clipped to [0,1]; high=smooth stable generation
            "quad_ideal_burst_count": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qibc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qibc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qibc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qibc_lbl=["i" if _coh3_steps[i] > _qibc_c3m and _velocity_steps[i] < _qibc_vm
                                        else "x" for i in range(_qibc_n)]:
                        sum(1 for i in range(1, _qibc_n)
                            if _qibc_lbl[i] == "i" and _qibc_lbl[i-1] != "i"
                            and sum(1 for j in range(i, _qibc_n)
                                    if _qibc_lbl[j] == "i" and all(_qibc_lbl[k] == "i"
                                                                    for k in range(i, j+1))) >= 2)
                        + (1 if _qibc_lbl[0] == "i" and _qibc_n >= 2 and _qibc_lbl[1] == "i" else 0)
                    )()
                )()
            ))(),
            # number of contiguous ideal runs of length ≥2
            "quad_ideal_burst_max_len": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qibm_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qibm_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qibm_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qibm_lbl=["i" if _coh3_steps[i] > _qibm_c3m and _velocity_steps[i] < _qibm_vm
                                        else "x" for i in range(_qibm_n)]:
                        (lambda _qibm_runs=
                            [sum(1 for _ in g)
                             for k, g in __import__("itertools").groupby(_qibm_lbl) if k == "i"]:
                            max(_qibm_runs) if _qibm_runs else 0
                        )()
                    )()
                )()
            ))(),
            # length of the longest consecutive ideal run
            "quad_ideal_burst_mean_len": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qibml_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qibml_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qibml_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qibml_lbl=["i" if _coh3_steps[i] > _qibml_c3m and _velocity_steps[i] < _qibml_vm
                                         else "x" for i in range(_qibml_n)]:
                        (lambda _qibml_runs=
                            [sum(1 for _ in g)
                             for k, g in __import__("itertools").groupby(_qibml_lbl) if k == "i"]:
                            round(sum(_qibml_runs)/max(len(_qibml_runs),1), 4) if _qibml_runs else 0.0
                        )()
                    )()
                )()
            ))(),
            # mean length of all consecutive ideal runs
            "quad_drift_burst_count": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qdbc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdbc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdbc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdbc_lbl=["d" if _coh3_steps[i] <= _qdbc_c3m and _velocity_steps[i] >= _qdbc_vm
                                        else "x" for i in range(_qdbc_n)]:
                        len([1 for k, g in __import__("itertools").groupby(_qdbc_lbl)
                             if k == "d" and sum(1 for _ in g) >= 2])
                    )()
                )()
            ))(),
            # number of contiguous drifting runs of length ≥2
            "quad_burst_quality_ratio": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qbqr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qbqr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qbqr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qbqr_il=["i" if _coh3_steps[i] > _qbqr_c3m and _velocity_steps[i] < _qbqr_vm
                                       else "x" for i in range(_qbqr_n)],
                           _qbqr_dl=["d" if _coh3_steps[i] <= _qbqr_c3m and _velocity_steps[i] >= _qbqr_vm
                                      else "x" for i in range(_qbqr_n)]:
                        (lambda _qbqr_ib=len([1 for k, g in __import__("itertools").groupby(_qbqr_il)
                                               if k == "i" and sum(1 for _ in g) >= 2]),
                               _qbqr_db=len([1 for k, g in __import__("itertools").groupby(_qbqr_dl)
                                              if k == "d" and sum(1 for _ in g) >= 2]):
                            round(_qbqr_ib / max(_qbqr_db + 1, 1), 4)
                        )()
                    )()
                )()
            ))(),
            # ideal_burst_count / (drift_burst_count + 1); high=many quality bursts, few drift bursts
            "quad_coh3_min": (lambda: (
                0.0 if not _coh3_steps
                else round(min(_coh3_steps), 6)
            ))(),
            # global minimum coh3 value across the generation
            "quad_coh3_max": (lambda: (
                0.0 if not _coh3_steps
                else round(max(_coh3_steps), 6)
            ))(),
            # global maximum coh3 value across the generation
            "quad_coh3_range": (lambda: (
                0.0 if not _coh3_steps
                else round(max(_coh3_steps) - min(_coh3_steps), 6)
            ))(),
            # coh3 value range (max − min); high=coh3 spans a wide band
            "quad_coh3_above_075_frac": (lambda: (
                0.0 if not _coh3_steps
                else round(sum(1 for v in _coh3_steps if v > 0.75) / len(_coh3_steps), 4)
            ))(),
            # fraction of steps with coh3 > 0.75 (high=generation is consistently high-coherence)
            "quad_coh3_above_090_frac": (lambda: (
                0.0 if not _coh3_steps
                else round(sum(1 for v in _coh3_steps if v > 0.90) / len(_coh3_steps), 4)
            ))(),
            # fraction of steps with coh3 > 0.90 (high=elite coherence throughout)
            "quad_vel_min": (lambda: (
                0.0 if not _velocity_steps
                else round(min(_velocity_steps), 6)
            ))(),
            # global minimum velocity across the generation
            "quad_vel_max": (lambda: (
                0.0 if not _velocity_steps
                else round(max(_velocity_steps), 6)
            ))(),
            # global maximum velocity across the generation
            "quad_vel_range": (lambda: (
                0.0 if not _velocity_steps
                else round(max(_velocity_steps) - min(_velocity_steps), 6)
            ))(),
            # velocity range (max − min)
            "quad_vel_below_025_frac": (lambda: (
                0.0 if not _velocity_steps
                else round(sum(1 for v in _velocity_steps if v < 0.25) / len(_velocity_steps), 4)
            ))(),
            # fraction of steps with velocity < 0.25 (low vel = focused/deep steps; high frac=good)
            "quad_vel_above_mean_frac": (lambda: (
                0.0 if not _velocity_steps
                else (lambda _qvamf_mu=sum(_velocity_steps)/len(_velocity_steps):
                    round(sum(1 for v in _velocity_steps if v > _qvamf_mu) / len(_velocity_steps), 4)
                )()
            ))(),
            # fraction of velocity steps above their own mean (expect ~0.5; skewed=asymmetric vel dist)
            "quad_conf_min": (lambda: (
                0.0 if not output_confs
                else round(min(output_confs), 6)
            ))(),
            # global minimum confidence across the generation
            "quad_conf_max": (lambda: (
                0.0 if not output_confs
                else round(max(output_confs), 6)
            ))(),
            # global maximum confidence across the generation
            "quad_conf_range": (lambda: (
                0.0 if not output_confs
                else round(max(output_confs) - min(output_confs), 6)
            ))(),
            # confidence range (max − min)
            "quad_conf_above_075_frac": (lambda: (
                0.0 if not output_confs
                else round(sum(1 for v in output_confs if v > 0.75) / len(output_confs), 4)
            ))(),
            # fraction of steps with confidence > 0.75 (high=model is consistently confident)
            "quad_conf_above_090_frac": (lambda: (
                0.0 if not output_confs
                else round(sum(1 for v in output_confs if v > 0.90) / len(output_confs), 4)
            ))(),
            # fraction of steps with confidence > 0.90 (high=elite model confidence)
            "quad_ideal_to_exploring_ratio": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qier_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qier_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qier_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(
                        sum(1 for i in range(_qier_n)
                            if _coh3_steps[i] > _qier_c3m and _velocity_steps[i] < _qier_vm) /
                        max(sum(1 for i in range(_qier_n)
                                if _coh3_steps[i] > _qier_c3m and _velocity_steps[i] >= _qier_vm), 1),
                        4)
                )()
            ))(),
            # ideal_count / max(exploring_count, 1); >1=more ideal than exploring
            "quad_drift_to_flat_ratio": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qdfr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdfr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdfr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(
                        sum(1 for i in range(_qdfr_n)
                            if _coh3_steps[i] <= _qdfr_c3m and _velocity_steps[i] >= _qdfr_vm) /
                        max(sum(1 for i in range(_qdfr_n)
                                if _coh3_steps[i] <= _qdfr_c3m and _velocity_steps[i] < _qdfr_vm), 1),
                        4)
                )()
            ))(),
            # drifting_count / max(flat_count, 1); high=more drifting than flat (chaos>stagnation)
            "quad_coherent_states_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qcsf_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qcsf_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_qcsf_n) if _coh3_steps[i] > _qcsf_c3m) / _qcsf_n, 4)
                )()
            ))(),
            # (ideal+exploring) / total — fraction of steps with above-median coherence
            "quad_high_vel_states_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qhvsf_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qhvsf_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_qhvsf_n) if _velocity_steps[i] >= _qhvsf_vm) / _qhvsf_n, 4)
                )()
            ))(),
            # (drifting+exploring) / total — fraction of steps with above-median velocity
            "quad_focus_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qfs_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfs_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfs_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfs_vmean=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(
                        (sum(1 for i in range(_qfs_n)
                             if _coh3_steps[i] > _qfs_c3m and _velocity_steps[i] < _qfs_vm) / _qfs_n)
                        * min(max(1.0 - _qfs_vmean, 0.0), 1.0),
                        4)
                )()
            ))(),
            # ideal_frac × (1 − vel_mean); high=high ideal fraction AND low velocity
            "quad_coherence_focus_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qcfs2_nc3=len(_coh3_steps),
                           _qcfs2_nv=len(_velocity_steps):
                    round(
                        (sum(_coh3_steps) / _qcfs2_nc3)
                        * (1.0 - sum(_velocity_steps) / _qcfs2_nv),
                        6)
                )()
            ))(),
            # coh3_mean × (1 − vel_mean); pure signal quality score
            "quad_overall_health_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qohs_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qohs_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qohs_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(
                        0.25*(sum(1 for i in range(_qohs_n)
                                  if _coh3_steps[i] > _qohs_c3m and _velocity_steps[i] < _qohs_vm)
                              / _qohs_n)
                        + 0.25*(1.0 - sum(1 for i in range(_qohs_n)
                                          if _coh3_steps[i] <= _qohs_c3m and _velocity_steps[i] >= _qohs_vm)
                                / _qohs_n)
                        + 0.25*sum(1 for v in _coh3_steps if v > 0.75) / len(_coh3_steps)
                        + 0.25*sum(1 for v in output_confs[:_qohs_n] if v > 0.75) / _qohs_n,
                        4)
                )()
            ))(),
            # 0.25×ideal_frac + 0.25×(1-drift_frac) + 0.25×coh3>0.75 + 0.25×conf>0.75 [0,1]
            "quad_generation_quality_index": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qgqi_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qgqi_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qgqi_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qgqi_nc3=len(_coh3_steps):
                    round(
                        0.30*(sum(1 for i in range(_qgqi_n)
                                  if _coh3_steps[i] > _qgqi_c3m and _velocity_steps[i] < _qgqi_vm)
                              / _qgqi_n)
                        + 0.20*(sum(_coh3_steps) / _qgqi_nc3)
                        + 0.20*(1.0 - sum(1 for i in range(_qgqi_n)
                                          if _coh3_steps[i] <= _qgqi_c3m and _velocity_steps[i] >= _qgqi_vm)
                                / _qgqi_n)
                        + 0.15*(sum(output_confs[:_qgqi_n]) / _qgqi_n)
                        + 0.15*(1.0 - sum(_velocity_steps[:_qgqi_n]) / _qgqi_n),
                        4)
                )()
            ))(),
            # 0.30×ideal_frac + 0.20×coh3_mean + 0.20×(1-drift_frac) + 0.15×conf_mean + 0.15×(1-vel_mean)
            "quad_coh3_p25": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _c3p25_s=sorted(_coh3_steps):
                    round(_c3p25_s[max(0, int(len(_c3p25_s)*0.25)-1)], 6)
                )()
            ))(),
            # 25th percentile of coh3 values
            "quad_coh3_p50": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _c3p50_s=sorted(_coh3_steps), _c3p50_n=len(_coh3_steps):
                    round((_c3p50_s[_c3p50_n//2-1]+_c3p50_s[_c3p50_n//2])/2.0
                          if _c3p50_n % 2 == 0 else _c3p50_s[_c3p50_n//2], 6)
                )()
            ))(),
            # median (50th percentile) of coh3 values
            "quad_coh3_p75": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _c3p75_s=sorted(_coh3_steps), _c3p75_n=len(_coh3_steps):
                    round(_c3p75_s[min(_c3p75_n-1, int(_c3p75_n*0.75))], 6)
                )()
            ))(),
            # 75th percentile of coh3 values
            "quad_coh3_iqr": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _c3iqr_s=sorted(_coh3_steps), _c3iqr_n=len(_coh3_steps):
                    round(
                        _c3iqr_s[min(_c3iqr_n-1, int(_c3iqr_n*0.75))]
                        - _c3iqr_s[max(0, int(_c3iqr_n*0.25)-1)],
                        6)
                )()
            ))(),
            # inter-quartile range of coh3 (p75−p25); high=coh3 is widely spread
            "quad_vel_p25": (lambda: (
                0.0 if len(_velocity_steps) < 4
                else (lambda _vp25_s=sorted(_velocity_steps):
                    round(_vp25_s[max(0, int(len(_vp25_s)*0.25)-1)], 6)
                )()
            ))(),
            # 25th percentile of velocity values
            "quad_vel_p50": (lambda: (
                0.0 if len(_velocity_steps) < 4
                else (lambda _vp50_s=sorted(_velocity_steps), _vp50_n=len(_velocity_steps):
                    round((_vp50_s[_vp50_n//2-1]+_vp50_s[_vp50_n//2])/2.0
                          if _vp50_n % 2 == 0 else _vp50_s[_vp50_n//2], 6)
                )()
            ))(),
            # median (50th percentile) of velocity values
            "quad_vel_p75": (lambda: (
                0.0 if len(_velocity_steps) < 4
                else (lambda _vp75_s=sorted(_velocity_steps), _vp75_n=len(_velocity_steps):
                    round(_vp75_s[min(_vp75_n-1, int(_vp75_n*0.75))], 6)
                )()
            ))(),
            # 75th percentile of velocity values
            "quad_vel_iqr": (lambda: (
                0.0 if len(_velocity_steps) < 4
                else (lambda _viqr_s=sorted(_velocity_steps), _viqr_n=len(_velocity_steps):
                    round(
                        _viqr_s[min(_viqr_n-1, int(_viqr_n*0.75))]
                        - _viqr_s[max(0, int(_viqr_n*0.25)-1)],
                        6)
                )()
            ))(),
            # inter-quartile range of velocity (p75−p25)
            "quad_conf_p25": (lambda: (
                0.0 if len(output_confs) < 4
                else (lambda _cfp25_s=sorted(output_confs):
                    round(_cfp25_s[max(0, int(len(_cfp25_s)*0.25)-1)], 6)
                )()
            ))(),
            # 25th percentile of confidence values
            "quad_conf_p50": (lambda: (
                0.0 if len(output_confs) < 4
                else (lambda _cfp50_s=sorted(output_confs), _cfp50_n=len(output_confs):
                    round((_cfp50_s[_cfp50_n//2-1]+_cfp50_s[_cfp50_n//2])/2.0
                          if _cfp50_n % 2 == 0 else _cfp50_s[_cfp50_n//2], 6)
                )()
            ))(),
            # median (50th percentile) of confidence values
            "quad_conf_p75": (lambda: (
                0.0 if len(output_confs) < 4
                else (lambda _cfp75_s=sorted(output_confs), _cfp75_n=len(output_confs):
                    round(_cfp75_s[min(_cfp75_n-1, int(_cfp75_n*0.75))], 6)
                )()
            ))(),
            # 75th percentile of confidence values
            "quad_conf_iqr": (lambda: (
                0.0 if len(output_confs) < 4
                else (lambda _cfiqr_s=sorted(output_confs), _cfiqr_n=len(output_confs):
                    round(
                        _cfiqr_s[min(_cfiqr_n-1, int(_cfiqr_n*0.75))]
                        - _cfiqr_s[max(0, int(_cfiqr_n*0.25)-1)],
                        6)
                )()
            ))(),
            # inter-quartile range of confidence (p75−p25)
            "quad_coh3_positive_delta_frac": (lambda: (
                0.0 if len(_coh3_steps) < 3
                else (lambda _qcpd_n=len(_coh3_steps)-1:
                    round(sum(1 for i in range(_qcpd_n)
                              if _coh3_steps[i+1] > _coh3_steps[i]) / _qcpd_n, 4)
                )()
            ))(),
            # fraction of step-pairs where coh3 rises (positive delta); >0.5=coh3 mostly rising
            "quad_coh3_negative_delta_frac": (lambda: (
                0.0 if len(_coh3_steps) < 3
                else (lambda _qcnd_n=len(_coh3_steps)-1:
                    round(sum(1 for i in range(_qcnd_n)
                              if _coh3_steps[i+1] < _coh3_steps[i]) / _qcnd_n, 4)
                )()
            ))(),
            # fraction of step-pairs where coh3 falls (negative delta)
            "quad_coh3_mean_positive_delta": (lambda: (
                0.0 if len(_coh3_steps) < 3
                else (lambda _qcmpd=[_coh3_steps[i+1]-_coh3_steps[i]
                                     for i in range(len(_coh3_steps)-1)
                                     if _coh3_steps[i+1] > _coh3_steps[i]]:
                    round(sum(_qcmpd)/max(len(_qcmpd),1), 6) if _qcmpd else 0.0
                )()
            ))(),
            # mean magnitude of positive coh3 deltas (how much coh3 jumps up per rising step)
            "quad_coh3_mean_negative_delta": (lambda: (
                0.0 if len(_coh3_steps) < 3
                else (lambda _qcmnd=[_coh3_steps[i]-_coh3_steps[i+1]
                                     for i in range(len(_coh3_steps)-1)
                                     if _coh3_steps[i+1] < _coh3_steps[i]]:
                    round(sum(_qcmnd)/max(len(_qcmnd),1), 6) if _qcmnd else 0.0
                )()
            ))(),
            # mean magnitude of negative coh3 deltas (how much coh3 drops per falling step)
            "quad_coh3_delta_asymmetry": (lambda: (
                0.0 if len(_coh3_steps) < 3
                else (lambda _qcda_p=[_coh3_steps[i+1]-_coh3_steps[i]
                                      for i in range(len(_coh3_steps)-1)
                                      if _coh3_steps[i+1] > _coh3_steps[i]],
                           _qcda_n=[_coh3_steps[i]-_coh3_steps[i+1]
                                    for i in range(len(_coh3_steps)-1)
                                    if _coh3_steps[i+1] < _coh3_steps[i]]:
                    round(
                        (sum(_qcda_p)/max(len(_qcda_p),1))
                        / max(sum(_qcda_n)/max(len(_qcda_n),1), 1e-9),
                        4)
                    if _qcda_p else 0.0
                )()
            ))(),
            # mean_pos_delta / mean_neg_delta; >1=rises bigger than falls (coh3 net-positive)
            "quad_vel_positive_delta_frac": (lambda: (
                0.0 if len(_velocity_steps) < 3
                else (lambda _qvpd_n=len(_velocity_steps)-1:
                    round(sum(1 for i in range(_qvpd_n)
                              if _velocity_steps[i+1] > _velocity_steps[i]) / _qvpd_n, 4)
                )()
            ))(),
            # fraction of step-pairs where velocity rises (bad=acceleration)
            "quad_vel_negative_delta_frac": (lambda: (
                0.0 if len(_velocity_steps) < 3
                else (lambda _qvnd_n=len(_velocity_steps)-1:
                    round(sum(1 for i in range(_qvnd_n)
                              if _velocity_steps[i+1] < _velocity_steps[i]) / _qvnd_n, 4)
                )()
            ))(),
            # fraction of step-pairs where velocity falls (good=deceleration toward focus)
            "quad_vel_mean_positive_delta": (lambda: (
                0.0 if len(_velocity_steps) < 3
                else (lambda _qvmpd=[_velocity_steps[i+1]-_velocity_steps[i]
                                     for i in range(len(_velocity_steps)-1)
                                     if _velocity_steps[i+1] > _velocity_steps[i]]:
                    round(sum(_qvmpd)/max(len(_qvmpd),1), 6) if _qvmpd else 0.0
                )()
            ))(),
            # mean magnitude of positive velocity jumps (acceleration episodes)
            "quad_vel_mean_negative_delta": (lambda: (
                0.0 if len(_velocity_steps) < 3
                else (lambda _qvmnd=[_velocity_steps[i]-_velocity_steps[i+1]
                                     for i in range(len(_velocity_steps)-1)
                                     if _velocity_steps[i+1] < _velocity_steps[i]]:
                    round(sum(_qvmnd)/max(len(_qvmnd),1), 6) if _qvmnd else 0.0
                )()
            ))(),
            # mean magnitude of negative velocity drops (deceleration episodes)
            "quad_vel_delta_asymmetry": (lambda: (
                0.0 if len(_velocity_steps) < 3
                else (lambda _qvda_p=[_velocity_steps[i+1]-_velocity_steps[i]
                                      for i in range(len(_velocity_steps)-1)
                                      if _velocity_steps[i+1] > _velocity_steps[i]],
                           _qvda_n=[_velocity_steps[i]-_velocity_steps[i+1]
                                    for i in range(len(_velocity_steps)-1)
                                    if _velocity_steps[i+1] < _velocity_steps[i]]:
                    round(
                        (sum(_qvda_n)/max(len(_qvda_n),1))
                        / max(sum(_qvda_p)/max(len(_qvda_p),1), 1e-9),
                        4)
                    if _qvda_n else 0.0
                )()
            ))(),
            # neg_delta_mean / pos_delta_mean; >1=velocity falls more than rises (net deceleration=good)
            "quad_conf_positive_delta_frac": (lambda: (
                0.0 if len(output_confs) < 3
                else (lambda _qcfpd_n=len(output_confs)-1:
                    round(sum(1 for i in range(_qcfpd_n)
                              if output_confs[i+1] > output_confs[i]) / _qcfpd_n, 4)
                )()
            ))(),
            # fraction of step-pairs where confidence rises
            "quad_conf_negative_delta_frac": (lambda: (
                0.0 if len(output_confs) < 3
                else (lambda _qcfnd_n=len(output_confs)-1:
                    round(sum(1 for i in range(_qcfnd_n)
                              if output_confs[i+1] < output_confs[i]) / _qcfnd_n, 4)
                )()
            ))(),
            # fraction of step-pairs where confidence falls
            "quad_conf_mean_positive_delta": (lambda: (
                0.0 if len(output_confs) < 3
                else (lambda _qcfmpd=[output_confs[i+1]-output_confs[i]
                                      for i in range(len(output_confs)-1)
                                      if output_confs[i+1] > output_confs[i]]:
                    round(sum(_qcfmpd)/max(len(_qcfmpd),1), 6) if _qcfmpd else 0.0
                )()
            ))(),
            # mean magnitude of positive confidence jumps
            "quad_conf_mean_negative_delta": (lambda: (
                0.0 if len(output_confs) < 3
                else (lambda _qcfmnd=[output_confs[i]-output_confs[i+1]
                                      for i in range(len(output_confs)-1)
                                      if output_confs[i+1] < output_confs[i]]:
                    round(sum(_qcfmnd)/max(len(_qcfmnd),1), 6) if _qcfmnd else 0.0
                )()
            ))(),
            # mean magnitude of negative confidence drops
            "quad_conf_delta_asymmetry": (lambda: (
                0.0 if len(output_confs) < 3
                else (lambda _qcfda_p=[output_confs[i+1]-output_confs[i]
                                       for i in range(len(output_confs)-1)
                                       if output_confs[i+1] > output_confs[i]],
                           _qcfda_n=[output_confs[i]-output_confs[i+1]
                                     for i in range(len(output_confs)-1)
                                     if output_confs[i+1] < output_confs[i]]:
                    round(
                        (sum(_qcfda_p)/max(len(_qcfda_p),1))
                        / max(sum(_qcfda_n)/max(len(_qcfda_n),1), 1e-9),
                        4)
                    if _qcfda_p else 0.0
                )()
            ))(),
            # conf_pos_delta_mean / conf_neg_delta_mean; >1=confidence rises bigger than it falls
            "quad_coh3_momentum": (lambda: (
                0.0 if len(_coh3_steps) < 3
                else round(
                    sum(_coh3_steps[i+1]-_coh3_steps[i] for i in range(len(_coh3_steps)-1))
                    / (len(_coh3_steps)-1), 6)
            ))(),
            # mean signed Δcoh3 = net drift of coherence per step (positive=overall rising)
            "quad_vel_momentum": (lambda: (
                0.0 if len(_velocity_steps) < 3
                else round(
                    sum(_velocity_steps[i+1]-_velocity_steps[i] for i in range(len(_velocity_steps)-1))
                    / (len(_velocity_steps)-1), 6)
            ))(),
            # mean signed Δvelocity per step (negative=decelerating/converging=good)
            "quad_conf_momentum": (lambda: (
                0.0 if len(output_confs) < 3
                else round(
                    sum(output_confs[i+1]-output_confs[i] for i in range(len(output_confs)-1))
                    / (len(output_confs)-1), 6)
            ))(),
            # mean signed Δconfidence per step (positive=confidence building)
            "quad_coh3_curvature_mean": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else round(
                    sum(abs(_coh3_steps[i+2] - 2*_coh3_steps[i+1] + _coh3_steps[i])
                        for i in range(len(_coh3_steps)-2))
                    / (len(_coh3_steps)-2), 6)
            ))(),
            # mean |second difference| of coh3 = curvature/acceleration of coherence signal
            "quad_vel_curvature_mean": (lambda: (
                0.0 if len(_velocity_steps) < 4
                else round(
                    sum(abs(_velocity_steps[i+2] - 2*_velocity_steps[i+1] + _velocity_steps[i])
                        for i in range(len(_velocity_steps)-2))
                    / (len(_velocity_steps)-2), 6)
            ))(),
            # mean |second difference| of velocity = curvature/acceleration of velocity signal
            "quad_ideal_coh3_std": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qics_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qics_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qics_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qics_v=[_coh3_steps[i] for i in range(_qics_n)
                                     if _coh3_steps[i] > _qics_c3m and _velocity_steps[i] < _qics_vm]:
                        (lambda _qics_mu=sum(_qics_v)/max(len(_qics_v),1):
                            round((sum((x-_qics_mu)**2 for x in _qics_v)/max(len(_qics_v)-1,1))**0.5, 6)
                        )() if len(_qics_v) >= 2 else 0.0
                    )()
                )()
            ))(),
            # std of coh3 on ideal steps (low=ideal steps are internally consistent)
            "quad_drifting_coh3_std": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qdcs_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdcs_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdcs_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdcs_v=[_coh3_steps[i] for i in range(_qdcs_n)
                                     if _coh3_steps[i] <= _qdcs_c3m and _velocity_steps[i] >= _qdcs_vm]:
                        (lambda _qdcs_mu=sum(_qdcs_v)/max(len(_qdcs_v),1):
                            round((sum((x-_qdcs_mu)**2 for x in _qdcs_v)/max(len(_qdcs_v)-1,1))**0.5, 6)
                        )() if len(_qdcs_v) >= 2 else 0.0
                    )()
                )()
            ))(),
            # std of coh3 on drifting steps
            "quad_ideal_conf_std": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 2
                else (lambda _qicfs_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qicfs_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qicfs_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qicfs_v=[output_confs[i] for i in range(_qicfs_n)
                                      if _coh3_steps[i] > _qicfs_c3m and _velocity_steps[i] < _qicfs_vm]:
                        (lambda _qicfs_mu=sum(_qicfs_v)/max(len(_qicfs_v),1):
                            round((sum((x-_qicfs_mu)**2 for x in _qicfs_v)/max(len(_qicfs_v)-1,1))**0.5, 6)
                        )() if len(_qicfs_v) >= 2 else 0.0
                    )()
                )()
            ))(),
            # std of confidence on ideal steps (low=ideal steps are consistently confident)
            "quad_ideal_uniformity": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qiu_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qiu_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qiu_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qiu_c3v=[_coh3_steps[i] for i in range(_qiu_n)
                                      if _coh3_steps[i] > _qiu_c3m and _velocity_steps[i] < _qiu_vm],
                           _qiu_cfv=[output_confs[i] for i in range(_qiu_n)
                                     if _coh3_steps[i] > _qiu_c3m and _velocity_steps[i] < _qiu_vm]:
                        (lambda _qiu_c3mu=sum(_qiu_c3v)/max(len(_qiu_c3v),1),
                               _qiu_cfmu=sum(_qiu_cfv)/max(len(_qiu_cfv),1):
                            round(
                                1.0 - (
                                    (sum((x-_qiu_c3mu)**2 for x in _qiu_c3v)/max(len(_qiu_c3v)-1,1))**0.5
                                    + (sum((x-_qiu_cfmu)**2 for x in _qiu_cfv)/max(len(_qiu_cfv)-1,1))**0.5
                                ) / 2.0,
                                4)
                            if len(_qiu_c3v) >= 2 else 0.0
                        )()
                    )()
                )()
            ))(),
            # 1 − mean(ideal_coh3_std, ideal_conf_std); high=ideal steps are internally homogeneous
            "quad_coh3_vel_correlation": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qcvc_n=min(len(_coh3_steps), len(_velocity_steps)):
                    (lambda _qcvc_c3=_coh3_steps[:_qcvc_n],
                            _qcvc_vl=_velocity_steps[:_qcvc_n]:
                        (lambda _qcvc_c3m=sum(_qcvc_c3)/_qcvc_n,
                                _qcvc_vlm=sum(_qcvc_vl)/_qcvc_n:
                            (lambda _qcvc_num=sum((_qcvc_c3[i]-_qcvc_c3m)*(_qcvc_vl[i]-_qcvc_vlm)
                                                  for i in range(_qcvc_n)),
                                    _qcvc_dc3=(sum((_qcvc_c3[i]-_qcvc_c3m)**2 for i in range(_qcvc_n)))**0.5,
                                    _qcvc_dvl=(sum((_qcvc_vl[i]-_qcvc_vlm)**2 for i in range(_qcvc_n)))**0.5:
                                round(_qcvc_num / max(_qcvc_dc3 * _qcvc_dvl, 1e-9), 4)
                            )()
                        )()
                    )()
                )()
            ))(),
            # Pearson r(coh3, vel); negative=ideal: high coh3 paired with low vel
            "quad_coh3_conf_correlation": (lambda: (
                0.0 if min(len(_coh3_steps), len(output_confs)) < 4
                else (lambda _qccc_n=min(len(_coh3_steps), len(output_confs)):
                    (lambda _qccc_c3=_coh3_steps[:_qccc_n],
                            _qccc_cf=output_confs[:_qccc_n]:
                        (lambda _qccc_c3m=sum(_qccc_c3)/_qccc_n,
                                _qccc_cfm=sum(_qccc_cf)/_qccc_n:
                            (lambda _qccc_num=sum((_qccc_c3[i]-_qccc_c3m)*(_qccc_cf[i]-_qccc_cfm)
                                                  for i in range(_qccc_n)),
                                    _qccc_dc3=(sum((_qccc_c3[i]-_qccc_c3m)**2 for i in range(_qccc_n)))**0.5,
                                    _qccc_dcf=(sum((_qccc_cf[i]-_qccc_cfm)**2 for i in range(_qccc_n)))**0.5:
                                round(_qccc_num / max(_qccc_dc3 * _qccc_dcf, 1e-9), 4)
                            )()
                        )()
                    )()
                )()
            ))(),
            # Pearson r(coh3, conf); positive=good: high coh3 paired with high confidence
            "quad_vel_conf_correlation": (lambda: (
                0.0 if min(len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qvcc_n=min(len(_velocity_steps), len(output_confs)):
                    (lambda _qvcc_vl=_velocity_steps[:_qvcc_n],
                            _qvcc_cf=output_confs[:_qvcc_n]:
                        (lambda _qvcc_vlm=sum(_qvcc_vl)/_qvcc_n,
                                _qvcc_cfm=sum(_qvcc_cf)/_qvcc_n:
                            (lambda _qvcc_num=sum((_qvcc_vl[i]-_qvcc_vlm)*(_qvcc_cf[i]-_qvcc_cfm)
                                                  for i in range(_qvcc_n)),
                                    _qvcc_dvl=(sum((_qvcc_vl[i]-_qvcc_vlm)**2 for i in range(_qvcc_n)))**0.5,
                                    _qvcc_dcf=(sum((_qvcc_cf[i]-_qvcc_cfm)**2 for i in range(_qvcc_n)))**0.5:
                                round(_qvcc_num / max(_qvcc_dvl * _qvcc_dcf, 1e-9), 4)
                            )()
                        )()
                    )()
                )()
            ))(),
            # Pearson r(vel, conf)
            "quad_coh3_vel_anticorrelation_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qcva_n=min(len(_coh3_steps), len(_velocity_steps)):
                    (lambda _qcva_c3=_coh3_steps[:_qcva_n],
                            _qcva_vl=_velocity_steps[:_qcva_n]:
                        (lambda _qcva_c3m=sum(_qcva_c3)/_qcva_n,
                                _qcva_vlm=sum(_qcva_vl)/_qcva_n:
                            (lambda _qcva_num=sum((_qcva_c3[i]-_qcva_c3m)*(_qcva_vl[i]-_qcva_vlm)
                                                  for i in range(_qcva_n)),
                                    _qcva_dc3=(sum((_qcva_c3[i]-_qcva_c3m)**2 for i in range(_qcva_n)))**0.5,
                                    _qcva_dvl=(sum((_qcva_vl[i]-_qcva_vlm)**2 for i in range(_qcva_n)))**0.5:
                                round(max(0.0, -_qcva_num / max(_qcva_dc3 * _qcva_dvl, 1e-9)), 4)
                            )()
                        )()
                    )()
                )()
            ))(),
            # max(0, -r(coh3,vel)); high=strong anticorrelation; ideal generation signal
            "quad_ideal_max_run": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qimr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qimr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qimr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qimr_labels=[("ideal" if _coh3_steps[i] > _qimr_c3m and _velocity_steps[i] < _qimr_vm
                                           else "other") for i in range(_qimr_n)]:
                        max(
                            (lambda _qimr_runs, _qimr_cur=0:
                                [_qimr_runs.__setitem__(0, max(_qimr_runs[0], _qimr_cur))
                                 or 0 for _ in [None]]
                                and _qimr_runs[0]
                            )([0]),
                            (lambda: (
                                lambda _best=[0], _cur=[0]: [
                                    (_cur.__setitem__(0, _cur[0]+1) or _best.__setitem__(0, max(_best[0], _cur[0])))
                                    if lbl == "ideal" else _cur.__setitem__(0, 0)
                                    for lbl in _qimr_labels
                                ] and _best[0]
                            ))()
                        )
                    )()
                )()
            ))(),
            # longest consecutive run of ideal steps
            "quad_drifting_max_run": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qdmr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdmr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdmr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdmr_labels=[("drifting" if _coh3_steps[i] <= _qdmr_c3m and _velocity_steps[i] >= _qdmr_vm
                                           else "other") for i in range(_qdmr_n)]:
                        (lambda: (
                            lambda _best=[0], _cur=[0]: [
                                (_cur.__setitem__(0, _cur[0]+1) or _best.__setitem__(0, max(_best[0], _cur[0])))
                                if lbl == "drifting" else _cur.__setitem__(0, 0)
                                for lbl in _qdmr_labels
                            ] and _best[0]
                        ))()
                    )()
                )()
            ))(),
            # longest consecutive run of drifting steps
            "quad_ideal_mean_run": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qimrn_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qimrn_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qimrn_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qimrn_labels=[("ideal" if _coh3_steps[i] > _qimrn_c3m and _velocity_steps[i] < _qimrn_vm
                                            else "other") for i in range(_qimrn_n)]:
                        (lambda: (
                            lambda _runs=[], _cur=[0]: [
                                _cur.__setitem__(0, _cur[0]+1) if lbl == "ideal"
                                else (_runs.append(_cur[0]) or _cur.__setitem__(0, 0))
                                if _cur[0] > 0 else None
                                for lbl in _qimrn_labels + ["other"]
                            ] and (round(sum(_runs)/max(len(_runs),1), 2) if _runs else 0.0)
                        ))()
                    )()
                )()
            ))(),
            # mean length of ideal runs (>1 = ideal comes in clusters)
            "quad_run_balance": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qrb_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qrb_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qrb_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qrb_lbl=[("ideal" if _coh3_steps[i] > _qrb_c3m and _velocity_steps[i] < _qrb_vm
                                        else "drifting" if _coh3_steps[i] <= _qrb_c3m and _velocity_steps[i] >= _qrb_vm
                                        else "other") for i in range(_qrb_n)]:
                        (lambda: (
                            lambda _bi=[0], _bc=[0], _di=[0], _dc=[0]: [
                                (_bc.__setitem__(0,_bc[0]+1) or _bi.__setitem__(0,max(_bi[0],_bc[0])))
                                if lbl == "ideal"
                                else ((_bc.__setitem__(0,0)) if lbl != "ideal" else None) or
                                     ((_dc.__setitem__(0,_dc[0]+1) or _di.__setitem__(0,max(_di[0],_dc[0])))
                                      if lbl == "drifting" else _dc.__setitem__(0,0))
                                for lbl in _qrb_lbl
                            ] and round(_bi[0] / max(_di[0], 1e-9), 4)
                        ))()
                    )()
                )()
            ))(),
            # ideal_max_run / drifting_max_run; >1=ideal clusters longer than drift
            "quad_ideal_to_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 3
                else (lambda _qiti_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qiti_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qiti_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qiti_lbl=["ideal" if _coh3_steps[i] > _qiti_c3m and _velocity_steps[i] < _qiti_vm
                                        else "other" for i in range(_qiti_n)]:
                        (lambda _qiti_from=[i for i in range(_qiti_n-1) if _qiti_lbl[i] == "ideal"]:
                            round(sum(1 for i in _qiti_from if _qiti_lbl[i+1] == "ideal") /
                                  max(len(_qiti_from), 1), 4)
                            if _qiti_from else 0.0
                        )()
                    )()
                )()
            ))(),
            # P(ideal → ideal): prob of staying ideal given currently ideal
            "quad_ideal_to_drift_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 3
                else (lambda _qitd_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qitd_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qitd_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qitd_lbl=["ideal" if _coh3_steps[i] > _qitd_c3m and _velocity_steps[i] < _qitd_vm
                                        else "drifting" if _coh3_steps[i] <= _qitd_c3m and _velocity_steps[i] >= _qitd_vm
                                        else "other" for i in range(_qitd_n)]:
                        (lambda _qitd_from=[i for i in range(_qitd_n-1) if _qitd_lbl[i] == "ideal"]:
                            round(sum(1 for i in _qitd_from if _qitd_lbl[i+1] == "drifting") /
                                  max(len(_qitd_from), 1), 4)
                            if _qitd_from else 0.0
                        )()
                    )()
                )()
            ))(),
            # P(ideal → drifting): prob of falling to drift given currently ideal
            "quad_drift_to_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 3
                else (lambda _qdti_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdti_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdti_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdti_lbl=["ideal" if _coh3_steps[i] > _qdti_c3m and _velocity_steps[i] < _qdti_vm
                                        else "drifting" if _coh3_steps[i] <= _qdti_c3m and _velocity_steps[i] >= _qdti_vm
                                        else "other" for i in range(_qdti_n)]:
                        (lambda _qdti_from=[i for i in range(_qdti_n-1) if _qdti_lbl[i] == "drifting"]:
                            round(sum(1 for i in _qdti_from if _qdti_lbl[i+1] == "ideal") /
                                  max(len(_qdti_from), 1), 4)
                            if _qdti_from else 0.0
                        )()
                    )()
                )()
            ))(),
            # P(drifting → ideal): recovery rate from drift
            "quad_drift_to_drift_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 3
                else (lambda _qdtd_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdtd_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdtd_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdtd_lbl=["drifting" if _coh3_steps[i] <= _qdtd_c3m and _velocity_steps[i] >= _qdtd_vm
                                        else "other" for i in range(_qdtd_n)]:
                        (lambda _qdtd_from=[i for i in range(_qdtd_n-1) if _qdtd_lbl[i] == "drifting"]:
                            round(sum(1 for i in _qdtd_from if _qdtd_lbl[i+1] == "drifting") /
                                  max(len(_qdtd_from), 1), 4)
                            if _qdtd_from else 0.0
                        )()
                    )()
                )()
            ))(),
            # P(drifting → drifting): prob of staying in drift once there
            "quad_transition_stability": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 3
                else (lambda _qts_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qts_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qts_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qts_lbl=["ideal" if _coh3_steps[i] > _qts_c3m and _velocity_steps[i] < _qts_vm
                                       else "drifting" if _coh3_steps[i] <= _qts_c3m and _velocity_steps[i] >= _qts_vm
                                       else "other" for i in range(_qts_n)]:
                        (lambda _qts_if=[i for i in range(_qts_n-1) if _qts_lbl[i] == "ideal"],
                               _qts_df=[i for i in range(_qts_n-1) if _qts_lbl[i] == "drifting"]:
                            round((
                                (sum(1 for i in _qts_if if _qts_lbl[i+1] == "ideal") / max(len(_qts_if),1)
                                 if _qts_if else 0.0)
                                + (sum(1 for i in _qts_df if _qts_lbl[i+1] == "drifting") / max(len(_qts_df),1)
                                   if _qts_df else 0.0)
                            ) / 2.0, 4)
                        )()
                    )()
                )()
            ))(),
            # mean(P(ideal→ideal), P(drift→drift)) = how sticky each state is
            "quad_tail_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qtif_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qtif_t=max(1, int(_qtif_n * 0.75)),
                           _qtif_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qtif_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qtif_tail_n=_qtif_n - _qtif_t:
                        round(sum(1 for i in range(_qtif_t, _qtif_n)
                                  if _coh3_steps[i] > _qtif_c3m and _velocity_steps[i] < _qtif_vm) /
                              max(_qtif_tail_n, 1), 4)
                    )()
                )()
            ))(),
            # ideal frac in the last 25% of steps (tail health)
            "quad_tail_coh3_mean": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _qtc3_n=len(_coh3_steps), _qtc3_t=max(1, int(len(_coh3_steps)*0.75)):
                    round(sum(_coh3_steps[_qtc3_t:]) / max(_qtc3_n - _qtc3_t, 1), 6)
                )()
            ))(),
            # mean coh3 in last 25% of steps
            "quad_tail_vel_mean": (lambda: (
                0.0 if len(_velocity_steps) < 4
                else (lambda _qtv_n=len(_velocity_steps), _qtv_t=max(1, int(len(_velocity_steps)*0.75)):
                    round(sum(_velocity_steps[_qtv_t:]) / max(_qtv_n - _qtv_t, 1), 6)
                )()
            ))(),
            # mean velocity in last 25% of steps
            "quad_tail_conf_mean": (lambda: (
                0.0 if len(output_confs) < 4
                else (lambda _qtcf_n=len(output_confs), _qtcf_t=max(1, int(len(output_confs)*0.75)):
                    round(sum(output_confs[_qtcf_t:]) / max(_qtcf_n - _qtcf_t, 1), 6)
                )()
            ))(),
            # mean confidence in last 25% of steps
            "quad_tail_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qtsco_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qtsco_t=max(1, int(_qtsco_n * 0.75)),
                           _qtsco_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qtsco_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qtsco_tail_n=_qtsco_n - _qtsco_t:
                        round(
                            0.4 * (sum(1 for i in range(_qtsco_t, _qtsco_n)
                                       if _coh3_steps[i] > _qtsco_c3m and _velocity_steps[i] < _qtsco_vm) /
                                   max(_qtsco_tail_n, 1))
                            + 0.3 * (sum(_coh3_steps[_qtsco_t:_qtsco_n]) / max(_qtsco_tail_n, 1))
                            + 0.3 * (1.0 - sum(_velocity_steps[_qtsco_t:_qtsco_n]) / max(_qtsco_tail_n, 1)),
                            4)
                    )()
                )()
            ))(),
            # composite tail score: 0.4×tail_ideal + 0.3×tail_coh3 + 0.3×(1−tail_vel)
            "quad_head_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qhif_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qhif_e=max(1, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.25)),
                           _qhif_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qhif_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_qhif_e)
                              if _coh3_steps[i] > _qhif_c3m and _velocity_steps[i] < _qhif_vm) /
                          max(_qhif_e, 1), 4)
                )()
            ))(),
            # ideal frac in the first 25% of steps (head quality)
            "quad_head_coh3_mean": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _qhc3_e=max(1, int(len(_coh3_steps)*0.25)):
                    round(sum(_coh3_steps[:_qhc3_e]) / _qhc3_e, 6)
                )()
            ))(),
            # mean coh3 in first 25% of steps
            "quad_head_vel_mean": (lambda: (
                0.0 if len(_velocity_steps) < 4
                else (lambda _qhv_e=max(1, int(len(_velocity_steps)*0.25)):
                    round(sum(_velocity_steps[:_qhv_e]) / _qhv_e, 6)
                )()
            ))(),
            # mean velocity in first 25% of steps
            "quad_head_conf_mean": (lambda: (
                0.0 if len(output_confs) < 4
                else (lambda _qhcf_e=max(1, int(len(output_confs)*0.25)):
                    round(sum(output_confs[:_qhcf_e]) / _qhcf_e, 6)
                )()
            ))(),
            # mean confidence in first 25% of steps
            "quad_head_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qhsc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qhsc_e=max(1, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.25)),
                           _qhsc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qhsc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(
                        0.4 * (sum(1 for i in range(_qhsc_e)
                                   if _coh3_steps[i] > _qhsc_c3m and _velocity_steps[i] < _qhsc_vm) /
                               max(_qhsc_e, 1))
                        + 0.3 * (sum(_coh3_steps[:_qhsc_e]) / max(_qhsc_e, 1))
                        + 0.3 * (1.0 - sum(_velocity_steps[:_qhsc_e]) / max(_qhsc_e, 1)),
                        4)
                )()
            ))(),
            # composite head score: 0.4×head_ideal + 0.3×head_coh3 + 0.3×(1−head_vel)
            "quad_head_tail_ideal_delta": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qarc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qarc_h=max(1, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.25)),
                           _qarc_t=max(1, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.75)),
                           _qarc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qarc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(
                        sum(1 for i in range(_qarc_t, _qarc_n)
                            if _coh3_steps[i] > _qarc_c3m and _velocity_steps[i] < _qarc_vm) /
                        max(_qarc_n - _qarc_t, 1)
                        - sum(1 for i in range(_qarc_h)
                              if _coh3_steps[i] > _qarc_c3m and _velocity_steps[i] < _qarc_vm) /
                        max(_qarc_h, 1),
                        4)
                )()
            ))(),
            # tail_ideal_frac − head_ideal_frac; positive=generation improves in ideal density
            "quad_head_tail_coh3_delta": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _qacd_n=len(_coh3_steps),
                           _qacd_h=max(1, int(len(_coh3_steps)*0.25)),
                           _qacd_t=max(1, int(len(_coh3_steps)*0.75)):
                    round(
                        sum(_coh3_steps[_qacd_t:]) / max(_qacd_n - _qacd_t, 1)
                        - sum(_coh3_steps[:_qacd_h]) / _qacd_h,
                        6)
                )()
            ))(),
            # tail_coh3_mean − head_coh3_mean; positive=coherence grows over generation
            "quad_head_tail_vel_delta": (lambda: (
                0.0 if len(_velocity_steps) < 4
                else (lambda _qavd_n=len(_velocity_steps),
                           _qavd_h=max(1, int(len(_velocity_steps)*0.25)),
                           _qavd_t=max(1, int(len(_velocity_steps)*0.75)):
                    round(
                        sum(_velocity_steps[_qavd_t:]) / max(_qavd_n - _qavd_t, 1)
                        - sum(_velocity_steps[:_qavd_h]) / _qavd_h,
                        6)
                )()
            ))(),
            # tail_vel_mean − head_vel_mean; negative=velocity decelerates (good)
            "quad_head_tail_score_delta": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qasd_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qasd_h=max(1, int(_qasd_n * 0.25)),
                           _qasd_t=max(1, int(_qasd_n * 0.75)),
                           _qasd_c3m=sum(_coh3_steps[:_qasd_n]) / max(_qasd_n, 1),
                           _qasd_vm=sum(_velocity_steps[:_qasd_n]) / max(_qasd_n, 1):
                    round(
                        (0.4*(sum(1 for i in range(_qasd_t,_qasd_n)
                                  if _coh3_steps[i]>_qasd_c3m and _velocity_steps[i]<_qasd_vm)/max(_qasd_n-_qasd_t,1))
                         + 0.3*(sum(_coh3_steps[_qasd_t:_qasd_n])/max(_qasd_n-_qasd_t,1))
                         + 0.3*(1.0-sum(_velocity_steps[_qasd_t:_qasd_n])/max(_qasd_n-_qasd_t,1)))
                        - (0.4*(sum(1 for i in range(_qasd_h)
                                   if _coh3_steps[i]>_qasd_c3m and _velocity_steps[i]<_qasd_vm)/max(_qasd_h,1))
                           + 0.3*(sum(_coh3_steps[:_qasd_h])/max(_qasd_h,1))
                           + 0.3*(1.0-sum(_velocity_steps[:_qasd_h])/max(_qasd_h,1))),
                        4)
                )()
            ))(),
            # tail_score − head_score; positive=generation improves overall
            "quad_generation_arc": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qga_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qga_h=max(1, int(_qga_n * 0.25)),
                           _qga_t=max(1, int(_qga_n * 0.75)),
                           _qga_c3m=sum(_coh3_steps[:_qga_n]) / max(_qga_n, 1),
                           _qga_vm=sum(_velocity_steps[:_qga_n]) / max(_qga_n, 1):
                    (lambda _qga_delta=
                        (0.4*(sum(1 for i in range(_qga_t,_qga_n)
                                  if _coh3_steps[i]>_qga_c3m and _velocity_steps[i]<_qga_vm)/max(_qga_n-_qga_t,1))
                         + 0.3*(sum(_coh3_steps[_qga_t:_qga_n])/max(_qga_n-_qga_t,1))
                         + 0.3*(1.0-sum(_velocity_steps[_qga_t:_qga_n])/max(_qga_n-_qga_t,1)))
                        - (0.4*(sum(1 for i in range(_qga_h)
                                   if _coh3_steps[i]>_qga_c3m and _velocity_steps[i]<_qga_vm)/max(_qga_h,1))
                           + 0.3*(sum(_coh3_steps[:_qga_h])/max(_qga_h,1))
                           + 0.3*(1.0-sum(_velocity_steps[:_qga_h])/max(_qga_h,1))):
                        round(_qga_delta, 4)
                    )()
                )()
            ))(),
            # signed (tail_score − head_score); positive=improving arc, negative=declining arc
            "quad_mid_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qmif_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qmif_s=max(1, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.25)),
                           _qmif_e=max(2, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.75)),
                           _qmif_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qmif_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_qmif_s, _qmif_e)
                              if _coh3_steps[i] > _qmif_c3m and _velocity_steps[i] < _qmif_vm) /
                          max(_qmif_e - _qmif_s, 1), 4)
                )()
            ))(),
            # ideal frac in the middle 50% of steps
            "quad_mid_coh3_mean": (lambda: (
                0.0 if len(_coh3_steps) < 4
                else (lambda _qmc3_n=len(_coh3_steps),
                           _qmc3_s=max(1, int(len(_coh3_steps)*0.25)),
                           _qmc3_e=max(2, int(len(_coh3_steps)*0.75)):
                    round(sum(_coh3_steps[_qmc3_s:_qmc3_e]) / max(_qmc3_e - _qmc3_s, 1), 6)
                )()
            ))(),
            # mean coh3 in middle 50% of steps
            "quad_mid_vel_mean": (lambda: (
                0.0 if len(_velocity_steps) < 4
                else (lambda _qmv_n=len(_velocity_steps),
                           _qmv_s=max(1, int(len(_velocity_steps)*0.25)),
                           _qmv_e=max(2, int(len(_velocity_steps)*0.75)):
                    round(sum(_velocity_steps[_qmv_s:_qmv_e]) / max(_qmv_e - _qmv_s, 1), 6)
                )()
            ))(),
            # mean velocity in middle 50% of steps
            "quad_mid_conf_mean": (lambda: (
                0.0 if len(output_confs) < 4
                else (lambda _qmcf_n=len(output_confs),
                           _qmcf_s=max(1, int(len(output_confs)*0.25)),
                           _qmcf_e=max(2, int(len(output_confs)*0.75)):
                    round(sum(output_confs[_qmcf_s:_qmcf_e]) / max(_qmcf_e - _qmcf_s, 1), 6)
                )()
            ))(),
            # mean confidence in middle 50% of steps
            "quad_mid_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qmsc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qmsc_s=max(1, int(_qmsc_n * 0.25)),
                           _qmsc_e=max(2, int(_qmsc_n * 0.75)),
                           _qmsc_c3m=sum(_coh3_steps[:_qmsc_n]) / max(_qmsc_n, 1),
                           _qmsc_vm=sum(_velocity_steps[:_qmsc_n]) / max(_qmsc_n, 1):
                    round(
                        0.4 * (sum(1 for i in range(_qmsc_s, _qmsc_e)
                                   if _coh3_steps[i] > _qmsc_c3m and _velocity_steps[i] < _qmsc_vm) /
                               max(_qmsc_e - _qmsc_s, 1))
                        + 0.3 * (sum(_coh3_steps[_qmsc_s:_qmsc_e]) / max(_qmsc_e - _qmsc_s, 1))
                        + 0.3 * (1.0 - sum(_velocity_steps[_qmsc_s:_qmsc_e]) / max(_qmsc_e - _qmsc_s, 1)),
                        4)
                )()
            ))(),
            # composite mid score: 0.4×mid_ideal + 0.3×mid_coh3 + 0.3×(1−mid_vel)
            "quad_peak_step": (lambda: (
                0.0 if len(_coh3_steps) < 2
                else (lambda _qpk_i=max(range(len(_coh3_steps)), key=lambda i: _coh3_steps[i]):
                    round(_qpk_i / max(len(_coh3_steps) - 1, 1), 4)
                )()
            ))(),
            # normalized position (0=start,1=end) of max coh3 step
            "quad_peak_coh3": (lambda: (
                0.0 if not _coh3_steps else round(max(_coh3_steps), 6)
            ))(),
            # max coh3 value in the generation
            "quad_peak_conf": (lambda: (
                0.0 if not _coh3_steps or not output_confs
                else (lambda _qpkc_i=max(range(min(len(_coh3_steps), len(output_confs))),
                                         key=lambda i: _coh3_steps[i]):
                    round(output_confs[_qpkc_i] if _qpkc_i < len(output_confs) else 0.0, 6)
                )()
            ))(),
            # confidence at the peak coh3 step
            "quad_trough_step": (lambda: (
                0.0 if len(_coh3_steps) < 2
                else (lambda _qtr_i=min(range(len(_coh3_steps)), key=lambda i: _coh3_steps[i]):
                    round(_qtr_i / max(len(_coh3_steps) - 1, 1), 4)
                )()
            ))(),
            # normalized position (0=start,1=end) of min coh3 step
            "quad_peak_to_trough_range": (lambda: (
                0.0 if len(_coh3_steps) < 2
                else round(max(_coh3_steps) - min(_coh3_steps), 6)
            ))(),
            # max_coh3 − min_coh3: dynamic range of coherence across the generation
            "quad_overall_segment_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qoss_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qoss_h=max(1, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.25)),
                           _qoss_t=max(1, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.75)),
                           _qoss_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qoss_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qoss_hs=(
                                0.4*(sum(1 for i in range(_qoss_h) if _coh3_steps[i]>_qoss_c3m and _velocity_steps[i]<_qoss_vm)/max(_qoss_h,1))
                                +0.3*(sum(_coh3_steps[:_qoss_h])/max(_qoss_h,1))
                                +0.3*(1.0-sum(_velocity_steps[:_qoss_h])/max(_qoss_h,1))),
                           _qoss_ms=(
                                0.4*(sum(1 for i in range(_qoss_h,_qoss_t) if _coh3_steps[i]>_qoss_c3m and _velocity_steps[i]<_qoss_vm)/max(_qoss_t-_qoss_h,1))
                                +0.3*(sum(_coh3_steps[_qoss_h:_qoss_t])/max(_qoss_t-_qoss_h,1))
                                +0.3*(1.0-sum(_velocity_steps[_qoss_h:_qoss_t])/max(_qoss_t-_qoss_h,1))),
                           _qoss_ts=(
                                0.4*(sum(1 for i in range(_qoss_t,_qoss_n) if _coh3_steps[i]>_qoss_c3m and _velocity_steps[i]<_qoss_vm)/max(_qoss_n-_qoss_t,1))
                                +0.3*(sum(_coh3_steps[_qoss_t:_qoss_n])/max(_qoss_n-_qoss_t,1))
                                +0.3*(1.0-sum(_velocity_steps[_qoss_t:_qoss_n])/max(_qoss_n-_qoss_t,1))):
                        round(0.25*_qoss_hs + 0.50*_qoss_ms + 0.25*_qoss_ts, 4)
                    )()
                )()
            ))(),
            # 0.25×head_score + 0.50×mid_score + 0.25×tail_score; center-weighted quality
            "quad_segment_range": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qsr_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qsr_h=max(1, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.25)),
                           _qsr_t=max(1, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.75)),
                           _qsr_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qsr_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qsr_hs=(
                                0.4*(sum(1 for i in range(_qsr_h) if _coh3_steps[i]>_qsr_c3m and _velocity_steps[i]<_qsr_vm)/max(_qsr_h,1))
                                +0.3*(sum(_coh3_steps[:_qsr_h])/max(_qsr_h,1))
                                +0.3*(1.0-sum(_velocity_steps[:_qsr_h])/max(_qsr_h,1))),
                           _qsr_ms=(
                                0.4*(sum(1 for i in range(_qsr_h,_qsr_t) if _coh3_steps[i]>_qsr_c3m and _velocity_steps[i]<_qsr_vm)/max(_qsr_t-_qsr_h,1))
                                +0.3*(sum(_coh3_steps[_qsr_h:_qsr_t])/max(_qsr_t-_qsr_h,1))
                                +0.3*(1.0-sum(_velocity_steps[_qsr_h:_qsr_t])/max(_qsr_t-_qsr_h,1))),
                           _qsr_ts=(
                                0.4*(sum(1 for i in range(_qsr_t,_qsr_n) if _coh3_steps[i]>_qsr_c3m and _velocity_steps[i]<_qsr_vm)/max(_qsr_n-_qsr_t,1))
                                +0.3*(sum(_coh3_steps[_qsr_t:_qsr_n])/max(_qsr_n-_qsr_t,1))
                                +0.3*(1.0-sum(_velocity_steps[_qsr_t:_qsr_n])/max(_qsr_n-_qsr_t,1))):
                        round(max(_qsr_hs,_qsr_ms,_qsr_ts) - min(_qsr_hs,_qsr_ms,_qsr_ts), 4)
                    )()
                )()
            ))(),
            # max(head,mid,tail) − min(head,mid,tail): quality spread across segments
            "quad_q1_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qq1_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qq1_e=max(1, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.25)),
                           _qq1_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qq1_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(0, _qq1_e)
                              if _coh3_steps[i] > _qq1_c3m and _velocity_steps[i] < _qq1_vm) /
                          max(_qq1_e, 1), 4)
                )()
            ))(),
            # ideal frac in Q1 (first 25%)
            "quad_q2_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qq2_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qq2_s=max(1, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.25)),
                           _qq2_e=max(2, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.50)),
                           _qq2_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qq2_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_qq2_s, _qq2_e)
                              if _coh3_steps[i] > _qq2_c3m and _velocity_steps[i] < _qq2_vm) /
                          max(_qq2_e - _qq2_s, 1), 4)
                )()
            ))(),
            # ideal frac in Q2 (25-50%)
            "quad_q3_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qq3_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qq3_s=max(2, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.50)),
                           _qq3_e=max(3, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.75)),
                           _qq3_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qq3_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_qq3_s, _qq3_e)
                              if _coh3_steps[i] > _qq3_c3m and _velocity_steps[i] < _qq3_vm) /
                          max(_qq3_e - _qq3_s, 1), 4)
                )()
            ))(),
            # ideal frac in Q3 (50-75%)
            "quad_q4_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qq4_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qq4_s=max(3, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.75)),
                           _qq4_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qq4_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_qq4_s, _qq4_n)
                              if _coh3_steps[i] > _qq4_c3m and _velocity_steps[i] < _qq4_vm) /
                          max(_qq4_n - _qq4_s, 1), 4)
                )()
            ))(),
            # ideal frac in Q4 (75-100%)
            "quad_ideal_peak_quartile": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qipq_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qipq_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qipq_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qipq_q=[
                        sum(1 for i in range(0, max(1,int(_qipq_n*0.25)))
                            if _coh3_steps[i]>_qipq_c3m and _velocity_steps[i]<_qipq_vm) /
                        max(int(_qipq_n*0.25), 1),
                        sum(1 for i in range(max(1,int(_qipq_n*0.25)), max(2,int(_qipq_n*0.50)))
                            if _coh3_steps[i]>_qipq_c3m and _velocity_steps[i]<_qipq_vm) /
                        max(int(_qipq_n*0.25), 1),
                        sum(1 for i in range(max(2,int(_qipq_n*0.50)), max(3,int(_qipq_n*0.75)))
                            if _coh3_steps[i]>_qipq_c3m and _velocity_steps[i]<_qipq_vm) /
                        max(int(_qipq_n*0.25), 1),
                        sum(1 for i in range(max(3,int(_qipq_n*0.75)), _qipq_n)
                            if _coh3_steps[i]>_qipq_c3m and _velocity_steps[i]<_qipq_vm) /
                        max(_qipq_n - int(_qipq_n*0.75), 1)
                    ]:
                        _qipq_q.index(max(_qipq_q)) + 1
                    )()
                )()
            ))(),
            # quartile (1-4) with highest ideal density
            "quad_total_step_count": (lambda: (
                min(len(_coh3_steps), len(_velocity_steps))
            ))(),
            # total steps analyzed in the generation
            "quad_ideal_step_count": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 1
                else (lambda _qisc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qisc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qisc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    sum(1 for i in range(_qisc_n)
                        if _coh3_steps[i] > _qisc_c3m and _velocity_steps[i] < _qisc_vm)
                )()
            ))(),
            # raw count of ideal steps
            "quad_drifting_step_count": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 1
                else (lambda _qdsc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdsc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdsc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    sum(1 for i in range(_qdsc_n)
                        if _coh3_steps[i] <= _qdsc_c3m and _velocity_steps[i] >= _qdsc_vm)
                )()
            ))(),
            # raw count of drifting steps
            "quad_exploring_step_count": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 1
                else (lambda _qesc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qesc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qesc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    sum(1 for i in range(_qesc_n)
                        if _coh3_steps[i] > _qesc_c3m and _velocity_steps[i] >= _qesc_vm)
                )()
            ))(),
            # raw count of exploring steps
            "quad_flat_step_count": (lambda: (
                0 if min(len(_coh3_steps), len(_velocity_steps)) < 1
                else (lambda _qfsc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qfsc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qfsc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    sum(1 for i in range(_qfsc_n)
                        if _coh3_steps[i] <= _qfsc_c3m and _velocity_steps[i] < _qfsc_vm)
                )()
            ))(),
            # raw count of flat steps
            "quad_coh3_linear_deviation": (lambda: (
                0.0 if len(_coh3_steps) < 3
                else (lambda _qcld_n=len(_coh3_steps):
                    round(
                        sum(abs(_coh3_steps[i] -
                                (_coh3_steps[0] + i * (_coh3_steps[-1] - _coh3_steps[0]) / max(_qcld_n-1,1)))
                            for i in range(_qcld_n)) / _qcld_n,
                        6)
                )()
            ))(),
            # mean |coh3[i] − linear_interpolation[i]|; high=nonlinear shape
            "quad_coh3_midpoint_vs_linear": (lambda: (
                0.0 if len(_coh3_steps) < 3
                else (lambda _qcmvl_n=len(_coh3_steps), _qcmvl_m=len(_coh3_steps)//2:
                    round(
                        _coh3_steps[_qcmvl_m]
                        - (_coh3_steps[0] + _qcmvl_m * (_coh3_steps[-1] - _coh3_steps[0]) / max(_qcmvl_n-1,1)),
                        6)
                )()
            ))(),
            # coh3[mid] − linear interp at mid; positive=arch(peak in middle), negative=bowl(trough in middle)
            "quad_vel_linear_deviation": (lambda: (
                0.0 if len(_velocity_steps) < 3
                else (lambda _qvld_n=len(_velocity_steps):
                    round(
                        sum(abs(_velocity_steps[i] -
                                (_velocity_steps[0] + i * (_velocity_steps[-1] - _velocity_steps[0]) / max(_qvld_n-1,1)))
                            for i in range(_qvld_n)) / _qvld_n,
                        6)
                )()
            ))(),
            # mean |vel[i] − linear_interpolation[i]|; high=nonlinear velocity shape
            "quad_vel_midpoint_vs_linear": (lambda: (
                0.0 if len(_velocity_steps) < 3
                else (lambda _qvmvl_n=len(_velocity_steps), _qvmvl_m=len(_velocity_steps)//2:
                    round(
                        _velocity_steps[_qvmvl_m]
                        - (_velocity_steps[0] + _qvmvl_m * (_velocity_steps[-1] - _velocity_steps[0]) / max(_qvmvl_n-1,1)),
                        6)
                )()
            ))(),
            # vel[mid] − linear interp at mid; negative=velocity dips in middle (good=focus episode)
            "quad_third1_vel_mean": (lambda: (
                0.0 if len(_velocity_steps) < 3
                else (lambda _qt1v_e=max(1, int(len(_velocity_steps) * 0.333)):
                    round(sum(_velocity_steps[:_qt1v_e]) / _qt1v_e, 6)
                )()
            ))(),
            # mean velocity in first 33% of steps
            "quad_third2_vel_mean": (lambda: (
                0.0 if len(_velocity_steps) < 3
                else (lambda _qt2v_n=len(_velocity_steps),
                           _qt2v_s=max(1, int(len(_velocity_steps) * 0.333)),
                           _qt2v_e=max(2, int(len(_velocity_steps) * 0.667)):
                    round(sum(_velocity_steps[_qt2v_s:_qt2v_e]) / max(_qt2v_e - _qt2v_s, 1), 6)
                )()
            ))(),
            # mean velocity in middle 33% of steps
            "quad_third3_vel_mean": (lambda: (
                0.0 if len(_velocity_steps) < 3
                else (lambda _qt3v_n=len(_velocity_steps),
                           _qt3v_s=max(2, int(len(_velocity_steps) * 0.667)):
                    round(sum(_velocity_steps[_qt3v_s:]) / max(_qt3v_n - _qt3v_s, 1), 6)
                )()
            ))(),
            # mean velocity in last 33% of steps
            "quad_vel_third_trend": (lambda: (
                0.0 if len(_velocity_steps) < 3
                else (lambda _qvtt_n=len(_velocity_steps),
                           _qvtt_s1=max(1, int(len(_velocity_steps) * 0.333)),
                           _qvtt_s2=max(2, int(len(_velocity_steps) * 0.667)):
                    round(
                        sum(_velocity_steps[_qvtt_s2:]) / max(_qvtt_n - _qvtt_s2, 1)
                        - sum(_velocity_steps[:_qvtt_s1]) / _qvtt_s1,
                        6)
                )()
            ))(),
            # third3_vel − third1_vel; negative=decelerating across generation (good)
            "quad_vel_convergence_score": (lambda: (
                0.0 if len(_velocity_steps) < 3
                else (lambda _qvcs_n=len(_velocity_steps),
                           _qvcs_s1=max(1, int(len(_velocity_steps) * 0.333)),
                           _qvcs_s2=max(2, int(len(_velocity_steps) * 0.667)):
                    (lambda _qvcs_t1=sum(_velocity_steps[:_qvcs_s1]) / _qvcs_s1,
                            _qvcs_t3=sum(_velocity_steps[_qvcs_s2:]) / max(_qvcs_n - _qvcs_s2, 1):
                        round(max(0.0, (_qvcs_t1 - _qvcs_t3) / max(_qvcs_t1, 1e-9)), 4)
                    )()
                )()
            ))(),
            # max(0, (third1_vel - third3_vel) / third1_vel); high=strong deceleration=converging
            "quad_third1_coh3_mean": (lambda: (
                0.0 if len(_coh3_steps) < 3
                else (lambda _qt1c_e=max(1, int(len(_coh3_steps) * 0.333)):
                    round(sum(_coh3_steps[:_qt1c_e]) / _qt1c_e, 6)
                )()
            ))(),
            # mean coh3 in first 33% of steps
            "quad_third2_coh3_mean": (lambda: (
                0.0 if len(_coh3_steps) < 3
                else (lambda _qt2c_n=len(_coh3_steps),
                           _qt2c_s=max(1, int(len(_coh3_steps) * 0.333)),
                           _qt2c_e=max(2, int(len(_coh3_steps) * 0.667)):
                    round(sum(_coh3_steps[_qt2c_s:_qt2c_e]) / max(_qt2c_e - _qt2c_s, 1), 6)
                )()
            ))(),
            # mean coh3 in middle 33% of steps
            "quad_third3_coh3_mean": (lambda: (
                0.0 if len(_coh3_steps) < 3
                else (lambda _qt3c_n=len(_coh3_steps),
                           _qt3c_s=max(2, int(len(_coh3_steps) * 0.667)):
                    round(sum(_coh3_steps[_qt3c_s:]) / max(_qt3c_n - _qt3c_s, 1), 6)
                )()
            ))(),
            # mean coh3 in last 33% of steps
            "quad_coh3_third_trend": (lambda: (
                0.0 if len(_coh3_steps) < 3
                else (lambda _qctt_n=len(_coh3_steps),
                           _qctt_s1=max(1, int(len(_coh3_steps) * 0.333)),
                           _qctt_s2=max(2, int(len(_coh3_steps) * 0.667)):
                    round(
                        sum(_coh3_steps[_qctt_s2:]) / max(_qctt_n - _qctt_s2, 1)
                        - sum(_coh3_steps[:_qctt_s1]) / _qctt_s1,
                        6)
                )()
            ))(),
            # third3_coh3 − third1_coh3; positive=coherence building across generation
            "quad_coh3_convergence_score": (lambda: (
                0.0 if len(_coh3_steps) < 3
                else (lambda _qccs_n=len(_coh3_steps),
                           _qccs_s1=max(1, int(len(_coh3_steps) * 0.333)),
                           _qccs_s2=max(2, int(len(_coh3_steps) * 0.667)):
                    (lambda _qccs_t1=sum(_coh3_steps[:_qccs_s1]) / _qccs_s1,
                            _qccs_t3=sum(_coh3_steps[_qccs_s2:]) / max(_qccs_n - _qccs_s2, 1):
                        round(max(0.0, (_qccs_t3 - _qccs_t1) / max(_qccs_t1, 1e-9)), 4)
                    )()
                )()
            ))(),
            # max(0, (third3_coh3 - third1_coh3) / third1_coh3); high=strong coherence growth
            "quad_conf_third1_mean": (lambda: (
                0.0 if len(output_confs) < 3
                else (lambda _qcf1_e=max(1, int(len(output_confs) * 0.333)):
                    round(sum(output_confs[:_qcf1_e]) / _qcf1_e, 6)
                )()
            ))(),
            # mean confidence in first 33% of steps
            "quad_conf_third2_mean": (lambda: (
                0.0 if len(output_confs) < 3
                else (lambda _qcf2_n=len(output_confs),
                           _qcf2_s=max(1, int(len(output_confs) * 0.333)),
                           _qcf2_e=max(2, int(len(output_confs) * 0.667)):
                    round(sum(output_confs[_qcf2_s:_qcf2_e]) / max(_qcf2_e - _qcf2_s, 1), 6)
                )()
            ))(),
            # mean confidence in middle 33% of steps
            "quad_conf_third3_mean": (lambda: (
                0.0 if len(output_confs) < 3
                else (lambda _qcf3_n=len(output_confs),
                           _qcf3_s=max(2, int(len(output_confs) * 0.667)):
                    round(sum(output_confs[_qcf3_s:]) / max(_qcf3_n - _qcf3_s, 1), 6)
                )()
            ))(),
            # mean confidence in last 33% of steps
            "quad_conf_third_trend": (lambda: (
                0.0 if len(output_confs) < 3
                else (lambda _qcft_n=len(output_confs),
                           _qcft_s1=max(1, int(len(output_confs) * 0.333)),
                           _qcft_s2=max(2, int(len(output_confs) * 0.667)):
                    round(
                        sum(output_confs[_qcft_s2:]) / max(_qcft_n - _qcft_s2, 1)
                        - sum(output_confs[:_qcft_s1]) / _qcft_s1,
                        6)
                )()
            ))(),
            # third3_conf − third1_conf; positive=confidence building (good)
            "quad_triple_convergence_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 3
                else (lambda _qtcs_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qtcs_s1=max(1, int(_qtcs_n * 0.333)),
                           _qtcs_s2=max(2, int(_qtcs_n * 0.667)):
                    (lambda _qtcs_c3t1=sum(_coh3_steps[:_qtcs_s1])/_qtcs_s1,
                            _qtcs_c3t3=sum(_coh3_steps[_qtcs_s2:_qtcs_n])/max(_qtcs_n-_qtcs_s2,1),
                            _qtcs_vt1=sum(_velocity_steps[:_qtcs_s1])/_qtcs_s1,
                            _qtcs_vt3=sum(_velocity_steps[_qtcs_s2:_qtcs_n])/max(_qtcs_n-_qtcs_s2,1),
                            _qtcs_cft1=sum(output_confs[:_qtcs_s1])/_qtcs_s1,
                            _qtcs_cft3=sum(output_confs[_qtcs_s2:_qtcs_n])/max(_qtcs_n-_qtcs_s2,1):
                        round(
                            (max(0.0, (_qtcs_c3t3-_qtcs_c3t1)/max(_qtcs_c3t1,1e-9))
                             + max(0.0, (_qtcs_vt1-_qtcs_vt3)/max(_qtcs_vt1,1e-9))
                             + max(0.0, (_qtcs_cft3-_qtcs_cft1)/max(_qtcs_cft1,1e-9))) / 3.0,
                            4)
                    )()
                )()
            ))(),
            # mean(coh3_convergence, vel_convergence, conf_convergence) — all three improving = high score
            "quad_third1_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 3
                else (lambda _qt1i_e=max(1, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.333)),
                           _qt1i_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qt1i_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_qt1i_e)
                              if _coh3_steps[i] > _qt1i_c3m and _velocity_steps[i] < _qt1i_vm) /
                          max(_qt1i_e, 1), 4)
                )()
            ))(),
            # ideal frac in first 33% of steps
            "quad_third2_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 3
                else (lambda _qt2i_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qt2i_s=max(1, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.333)),
                           _qt2i_e=max(2, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.667)),
                           _qt2i_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qt2i_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_qt2i_s, _qt2i_e)
                              if _coh3_steps[i] > _qt2i_c3m and _velocity_steps[i] < _qt2i_vm) /
                          max(_qt2i_e - _qt2i_s, 1), 4)
                )()
            ))(),
            # ideal frac in middle 33% of steps
            "quad_third3_ideal_frac": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 3
                else (lambda _qt3i_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qt3i_s=max(2, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.667)),
                           _qt3i_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qt3i_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(sum(1 for i in range(_qt3i_s, _qt3i_n)
                              if _coh3_steps[i] > _qt3i_c3m and _velocity_steps[i] < _qt3i_vm) /
                          max(_qt3i_n - _qt3i_s, 1), 4)
                )()
            ))(),
            # ideal frac in last 33% of steps
            "quad_third_trend_ideal": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 3
                else (lambda _qttri_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qttri_s1=max(1, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.333)),
                           _qttri_s2=max(2, int(min(len(_coh3_steps), len(_velocity_steps)) * 0.667)),
                           _qttri_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qttri_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    round(
                        sum(1 for i in range(_qttri_s2, _qttri_n)
                            if _coh3_steps[i] > _qttri_c3m and _velocity_steps[i] < _qttri_vm) /
                        max(_qttri_n - _qttri_s2, 1)
                        - sum(1 for i in range(_qttri_s1)
                              if _coh3_steps[i] > _qttri_c3m and _velocity_steps[i] < _qttri_vm) /
                        max(_qttri_s1, 1),
                        4)
                )()
            ))(),
            # last33% ideal frac − first33% ideal frac; positive=ideal becomes denser over time
            "quad_third_shape_score": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 3
                else (lambda _qtsh_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qtsh_s1=max(1, int(_qtsh_n * 0.333)),
                           _qtsh_s2=max(2, int(_qtsh_n * 0.667)),
                           _qtsh_c3m=sum(_coh3_steps[:_qtsh_n]) / max(_qtsh_n, 1),
                           _qtsh_vm=sum(_velocity_steps[:_qtsh_n]) / max(_qtsh_n, 1):
                    round(
                        (sum(_coh3_steps[_qtsh_s2:_qtsh_n]) / max(_qtsh_n - _qtsh_s2, 1)
                         - sum(_coh3_steps[:_qtsh_s1]) / _qtsh_s1)
                        + (sum(output_confs[_qtsh_s2:_qtsh_n]) / max(_qtsh_n - _qtsh_s2, 1)
                           - sum(output_confs[:_qtsh_s1]) / _qtsh_s1)
                        - (sum(_velocity_steps[_qtsh_s2:_qtsh_n]) / max(_qtsh_n - _qtsh_s2, 1)
                           - sum(_velocity_steps[:_qtsh_s1]) / _qtsh_s1),
                        4)
                )()
            ))(),
            # combined third-based shape score: coh3/conf growth minus velocity growth
            "quad_third_velocity_trend": (lambda: (
                0.0 if len(_velocity_steps) < 3
                else (lambda _qtvt_n=len(_velocity_steps),
                           _qtvt_s1=max(1, int(_qtvt_n * 0.333)),
                           _qtvt_s2=max(2, int(_qtvt_n * 0.667)),
                           _qtvt_m1=sum(_velocity_steps[:_qtvt_s1]) / max(_qtvt_s1, 1),
                           _qtvt_m2=sum(_velocity_steps[_qtvt_s1:_qtvt_s2]) / max(_qtvt_s2 - _qtvt_s1, 1),
                           _qtvt_m3=sum(_velocity_steps[_qtvt_s2:]) / max(_qtvt_n - _qtvt_s2, 1):
                    round((_qtvt_m3 - _qtvt_m1) - abs(_qtvt_m2 - ((_qtvt_m1 + _qtvt_m3) / 2.0)), 4)
                )()
            ))(),
            # third-wise velocity arc: last-third minus first-third, with middle penalty
            "quad_third_coh3_trend": (lambda: (
                0.0 if len(_coh3_steps) < 3
                else (lambda _qtct_n=len(_coh3_steps),
                           _qtct_s1=max(1, int(_qtct_n * 0.333)),
                           _qtct_s2=max(2, int(_qtct_n * 0.667)),
                           _qtct_m1=sum(_coh3_steps[:_qtct_s1]) / max(_qtct_s1, 1),
                           _qtct_m2=sum(_coh3_steps[_qtct_s1:_qtct_s2]) / max(_qtct_s2 - _qtct_s1, 1),
                           _qtct_m3=sum(_coh3_steps[_qtct_s2:]) / max(_qtct_n - _qtct_s2, 1):
                    round((_qtct_m3 - _qtct_m1) - abs(_qtct_m2 - ((_qtct_m1 + _qtct_m3) / 2.0)), 6)
                )()
            ))(),
            # third-wise coh3 arc: last-third minus first-third, with middle penalty
            "quad_vel_contrast": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qvc_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qvc_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qvc_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qvc_id=[_velocity_steps[i] for i in range(_qvc_n)
                                     if _coh3_steps[i] > _qvc_c3m and _velocity_steps[i] < _qvc_vm],
                           _qvc_dr=[_velocity_steps[i] for i in range(_qvc_n)
                                    if _coh3_steps[i] <= _qvc_c3m and _velocity_steps[i] >= _qvc_vm]:
                        (lambda _qvc_im=sum(_qvc_id)/max(len(_qvc_id),1),
                               _qvc_dm=sum(_qvc_dr)/max(len(_qvc_dr),1):
                            round(
                                (_qvc_dm - _qvc_im) / max(_qvc_dm + _qvc_im, 1e-9)
                                if _qvc_id and _qvc_dr else 0.0, 4)
                        )()
                    )()
                )()
            ))(),
            # (drifting_vel_mean − ideal_vel_mean)/(drift+ideal) normalised velocity contrast [0,1]
            "quad_coh3_gap": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 2
                else (lambda _qcg3_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qcg3_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qcg3_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qcg3_id=[_coh3_steps[i] for i in range(_qcg3_n)
                                      if _coh3_steps[i] > _qcg3_c3m and _velocity_steps[i] < _qcg3_vm],
                           _qcg3_dr=[_coh3_steps[i] for i in range(_qcg3_n)
                                     if _coh3_steps[i] <= _qcg3_c3m and _velocity_steps[i] >= _qcg3_vm]:
                        round(
                            sum(_qcg3_id)/max(len(_qcg3_id),1) - sum(_qcg3_dr)/max(len(_qcg3_dr),1)
                            if _qcg3_id and _qcg3_dr else 0.0,
                            4)
                    )()
                )()
            ))(),
            # ideal_coh3_mean − drifting_coh3_mean (positive=ideal is more coherent; large=quadrants are coh3-distinct)
            "quad_ideal_run_len_std": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qirls_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qirls_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qirls_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qirls_labs=[
                        "ideal" if _coh3_steps[i] > _qirls_c3m and _velocity_steps[i] < _qirls_vm
                        else "other" for i in range(_qirls_n)
                    ]:
                        (lambda _qirls_lens=(lambda _qirls_l=[], _qirls_cur=0:
                            [_qirls_l.append(_qirls_cur) or 0 if _qirls_labs[i-1] == "ideal" and _qirls_labs[i] != "ideal"
                             else _qirls_l.__setitem__(-1, _qirls_cur + 1) or 0
                             if _qirls_labs[i] == "ideal" and len(_qirls_l) > 0 and _qirls_labs[i-1] == "ideal"
                             else 0 for i in range(1, _qirls_n)] or _qirls_l
                        )():
                            (lambda _qirls_runs=[
                                sum(1 for j in range(i, _qirls_n)
                                    if _qirls_labs[j] == "ideal"
                                    and (j == 0 or _qirls_labs[j-1] == "other" or j == i))
                                for i in range(_qirls_n)
                                if _qirls_labs[i] == "ideal" and (i == 0 or _qirls_labs[i-1] != "ideal")
                            ]:
                                0.0
                            )()
                        )()
                    )()
                )()
            ))(),
            # placeholder — replaced by simpler implementation below
            "quad_drifting_run_len_std": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps)) < 4
                else (lambda _qdrls_n=min(len(_coh3_steps), len(_velocity_steps)),
                           _qdrls_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                      max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qdrls_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                     max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qdrls_labs=[
                        "drifting" if _coh3_steps[i] <= _qdrls_c3m and _velocity_steps[i] >= _qdrls_vm
                        else "other" for i in range(_qdrls_n)
                    ]:
                        (lambda _qdrls_runs=[
                            sum(1 for j in range(i, _qdrls_n)
                                if _qdrls_labs[j] == "drifting"
                                and (j == i or (_qdrls_labs[j-1] == "drifting")))
                            for i in range(_qdrls_n)
                            if _qdrls_labs[i] == "drifting" and (i == 0 or _qdrls_labs[i-1] != "drifting")
                        ]:
                            0.0
                        )()
                    )()
                )()
            ))(),
            # placeholder for drifting run length std — computed simply in CLI
            "quad_overall_efficiency": (lambda: (
                0.0 if min(len(_coh3_steps), len(_velocity_steps), len(output_confs)) < 4
                else (lambda _qoe_n=min(len(_coh3_steps), len(_velocity_steps), len(output_confs)),
                           _qoe_c3m=sum(_coh3_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                    max(min(len(_coh3_steps), len(_velocity_steps)), 1),
                           _qoe_vm=sum(_velocity_steps[:min(len(_coh3_steps), len(_velocity_steps))]) /
                                   max(min(len(_coh3_steps), len(_velocity_steps)), 1):
                    (lambda _qoe_ideal_c=[
                            output_confs[i] for i in range(_qoe_n)
                            if _coh3_steps[i] > _qoe_c3m and _velocity_steps[i] < _qoe_vm],
                         _qoe_drift_c=[
                            output_confs[i] for i in range(_qoe_n)
                            if _coh3_steps[i] <= _qoe_c3m and _velocity_steps[i] >= _qoe_vm]:
                        (lambda _qoe_hs=(
                                (len(_qoe_ideal_c) / _qoe_n)
                                * (sum(_qoe_ideal_c)/max(len(_qoe_ideal_c),1))
                                * (1.0 - len(_qoe_drift_c)/_qoe_n)
                                if _qoe_ideal_c else 0.0),
                               _qoe_hv_frac=sum(1 for i in range(_qoe_n)
                                                if _velocity_steps[i] >= _qoe_vm) / _qoe_n:
                            round(_qoe_hs / max(_qoe_hv_frac, 0.01), 4)
                        )()
                    )()
                )()
            ))(),
            # health_score / max(hi_vel_frac,0.01) — quality per unit of fast motion (composite)
            "phasestats": (lambda _ot=output_tokens, _oc=output_confs, _cs=_coh3_steps: {
                phase: {
                    "avg_conf": round(float(np.mean(_oc[lo:hi])), 4) if _oc[lo:hi] else 0.0,
                    "avg_coh":  round(float(np.mean(_cs[lo:hi])), 4) if _cs[lo:hi] else 0.0,
                    "n_toks":   hi - lo,
                }
                for phase, lo, hi in [
                    ("early", 0,         max(len(_ot)//3, 1)),
                    ("mid",   len(_ot)//3, 2*len(_ot)//3),
                    ("late",  2*len(_ot)//3, len(_ot)),
                ]
            })(),  # early/mid/late phase avg conf + coh
            "topk_steps":         _topk_steps,          # per-step adaptive TopK k
            "topkdelta": round(
                float(np.std(_topk_steps)) if len(_topk_steps) >= 2 else 0.0, 4
            ),   # std-dev of topk values; high = beam width fluctuated a lot
            "conf_smooth": [
                round(float(np.mean(_conf_ema_steps[max(0, i-1): i+2])), 5)
                for i in range(len(_conf_ema_steps))
            ],   # 3-point centred moving average of conf_ema_steps
            "score_hist":      _score_hist.tolist(),  # 8-bucket confidence histogram
            "conf_ema_final":  round(_conf_ema, 4),  # EMA-smoothed confidence final step
            "anchor_strength": round(_anc_strength, 4),  # prompt-adaptive anchor strength
            "prompt_type":     _ptype,            # detected prompt intent
            "coh_direction":   round(_coh_dir, 4),  # coherence slope over last 4 steps
            "conf_declining":  conf_declining,       # True if last 4 confs all falling
            "coh_trend":       round(coh_trend, 5),  # linear slope of coh3 across steps
            "bestseg":  {"start": bestseg_start,  "val": round(bestseg_val,  4)},
            "worstseg": {"start": worstseg_start, "val": round(worstseg_val, 4)},
            "flow_score":  flow_score,   # fraction of steps both conf+coh3 improved
            "flow_steps":  _flow_steps,  # per-step B/C/H/N indicators
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
