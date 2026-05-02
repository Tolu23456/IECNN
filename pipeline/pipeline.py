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
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)
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
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)
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
                _fp(out_v)[0], assign.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
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
            assigns.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
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

    def causal_train_pass(self, sentences: List[str], max_pos: int = 6,
                          verbose: bool = True, prune_every: int = 0,
                          causal_batch: int = 20) -> None:
        """
        Causal next-token prediction training (fast path).

        Win criterion:
          per-dot win = cosine_sim(W[i] @ ctx, target_token) > 0
          W[i] maps context onto prediction direction; ~50 % of dots win by
          Gaussian symmetry; dots that specialise on common tokens drift above
          50 % → effectiveness > 0.5 → MaxEff > 0.5.

        Speed design:
          The C pipeline has ~1750 ms of FIXED OVERHEAD per call regardless of
          batch size.  Processing 1 sentence/call wastes 99 % of each call.
          This implementation batches `causal_batch` source sentences into ONE
          run_batch() call (collecting all their prefix strings), then applies
          BLAS-efficient W-matrix alignment and batch_record() updates.

          With causal_batch=20 and ~6 prefixes/sentence:
            Old: 20 calls × ~2000 ms = 40 s for 20 sentences → 0.5 sent/s
            New:  1 call  × ~6000 ms = 6 s for 20 sentences  → 3+ sent/s
        """
        if not sentences:
            return
        t0         = time.time()
        total_steps = 0
        dim         = self.feature_dim
        n_done      = 0                 # sentences successfully processed

        for batch_start in range(0, len(sentences), causal_batch):
            batch_sents = sentences[batch_start:batch_start + causal_batch]

            # ── PHASE 1: Collect ALL prefixes from ALL sentences in batch ─────
            all_prefixes:  List[str] = []
            all_tgt_toks:  List[str] = []
            # per_sent_slices[s] = (start, stop) into all_prefixes for sentence s
            per_sent_slices: List[tuple] = []

            for sentence in batch_sents:
                tokens = self.base_mapper._tokenize(sentence)
                if len(tokens) < 2:
                    per_sent_slices.append(None)
                    continue
                n_pos  = min(len(tokens), max_pos + 1)
                start  = len(all_prefixes)
                for pos in range(1, n_pos):
                    if pos < len(tokens):
                        all_prefixes.append(" ".join(tokens[:pos]))
                        all_tgt_toks.append(tokens[pos])
                stop = len(all_prefixes)
                per_sent_slices.append((start, stop) if stop > start else None)

            if not all_prefixes:
                n_done += len(batch_sents)
                continue

            # ── PHASE 2: ONE C pipeline call for ALL prefixes ─────────────────
            # This amortises the ~1750 ms fixed overhead over causal_batch sents
            all_outputs, all_ctxs = self.run_batch(
                all_prefixes, return_centroids=True, record_wins=False
            )   # all_outputs: (n_total, dim), all_ctxs: (n_total, dim) or None

            # ── PHASE 3: Build ALL target vectors at once ─────────────────────
            n_total = len(all_prefixes)
            tvs     = np.zeros((n_total, dim), dtype=np.float32)
            for j, tok in enumerate(all_tgt_toks):
                te = self.base_mapper._token_embedding(
                    tok, self.base_mapper._base_types.get(tok, "word")
                )
                n = min(len(te), dim)
                tvs[j, :n] = np.real(te[:n]).astype(np.float32)
                tn = float(np.linalg.norm(tvs[j]))
                if tn > 1e-10:
                    tvs[j] /= tn

            # ── PHASE 4: BLAS-efficient per-dot predictions ───────────────────
            dots   = self._ensure_dots()
            n_dots = len(dots)
            self.dot_gen._ensure_caches(dots)
            W3d    = self.dot_gen._cached_W_stack          # (n_dots, dim, dim)
            ctxs   = all_ctxs if all_ctxs is not None else all_outputs

            W_flat = W3d.reshape(n_dots * dim, dim)        # (32768, 256) — view
            dp_flat = W_flat @ ctxs.T                      # (32768, n_total) BLAS
            dot_preds_all = np.ascontiguousarray(
                dp_flat.reshape(n_dots, dim, n_total).transpose(2, 0, 1)
            )   # (n_total, n_dots, dim)

            # normalise predictions and targets for cosine similarity
            dp_norms  = np.linalg.norm(dot_preds_all, axis=2, keepdims=True).clip(1e-10)
            dp_all_n  = dot_preds_all / dp_norms            # (n_total, n_dots, dim)
            tv_norms  = np.linalg.norm(tvs, axis=1, keepdims=True).clip(1e-10)
            tvs_n     = tvs / tv_norms                      # (n_total, dim)

            # causal_sims[p, i] = dp_all_n[p, i] · tvs_n[p]
            # batch matmul: (n_total, 1, dim) @ (n_total, dim, n_dots) = (n_total, n_dots)
            causal_sims = np.squeeze(
                tvs_n[:, None, :] @ dp_all_n.transpose(0, 2, 1), 1
            )
            causal_wins = causal_sims > 0.0                 # (n_total, n_dots) bool

            # ── PHASE 5: Record + weight update in causal order ───────────────
            # For causal integrity, positions within each sentence are processed
            # in order.  Positions from different sentences are interleaved in
            # prefix-list order, which is already sentence-by-sentence.
            dot_ids_list = [d.dot_id for d in dots]
            for p_idx in range(n_total):
                self.dot_memory.batch_record(
                    dot_ids_list,
                    dot_preds_all[p_idx],   # (n_dots, dim) contiguous
                    causal_wins[p_idx],     # (n_dots,) bool
                )
                self.dot_gen.batch_local_update(dots, tvs[p_idx], lr=0.005)
                total_steps += 1

            # ── Periodic evolution ────────────────────────────────────────────
            n_done += len(batch_sents)
            if n_done % 10 < causal_batch:
                dots = self._ensure_dots()
                self._dots = self.evolution.evolve(
                    dots, self.dot_memory, call_count=self._call_count
                )

            if prune_every > 0 and n_done % prune_every < causal_batch:
                self.prune_dots()

            if verbose:
                elapsed = time.time() - t0
                rate    = n_done / max(elapsed, 1e-6)
                effs    = self.dot_memory.all_effectivenesses(
                    [d.dot_id for d in self._ensure_dots()]
                )
                max_eff = float(np.max(effs)) if len(effs) > 0 else 0.0
                print(
                    f"\r[causal] {n_done}/{len(sentences)} sents"
                    f" | {total_steps} steps"
                    f" | {rate:.2f} sent/s"
                    f" | MaxEff: {max_eff:.4f}",
                    end="", flush=True,
                )

        if verbose:
            print()

    def train_pass(self, sentences: List[str], use_c_pipeline: bool = True,
                   verbose: bool = True, prune_every: int = 0,
                   batch_size: int = 200):
        if not sentences: return
        t0 = time.time()
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            self.run_batch(batch, use_c_pipeline=use_c_pipeline)
            dots = self._ensure_dots()
            self._dots = self.evolution.evolve(dots, self.dot_memory, call_count=self._call_count)
            if prune_every > 0 and (i + len(batch)) % prune_every == 0:
                self.prune_dots()
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
