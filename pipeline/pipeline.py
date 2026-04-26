"""
IECNN Pipeline — full 10-layer architecture.

Layers:
  1.  Input              — receives raw data
  2.  BaseMapping        — converts to structured token maps (one row per token)
  3.  Dot Generation     — creates diverse pool (6 types, multi-head)
  4.  Prediction         — each dot independently predicts (N_HEADS each)
  5.  AIM                — 9 inversions run in parallel with originals
  6.  Pruning Stage 1    — soft filter (conf, duplicates, AIM cap)
  7.  Convergence        — two-level hierarchical clustering (F2, F13, F15)
  8.  Pruning Stage 2+3  — cluster compression + dynamic hard selection
  9.  Iteration Control  — 5 stopping conditions + rollback + adaptive LR (F14)
  10. Output             — normalized final vector

Evolution (across calls):
  After each run(), DotMemory records which dots contributed to the winner.
  DotEvolution evolves the pool between runs (mutation, crossover, selection).

Memory guidance:
  Memory hints (recent centroids) are passed to dots for attention guidance.
  ClusterMemory records temporal stability across rounds (F12).
"""

import numpy as np
import re
from typing import List, Dict, Optional, Tuple, Any
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basemapping.basemapping import BaseMapper, BaseMap, EMBED_DIM
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
    amplify_pressure, hierarchical_convergence_score
)


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

    Instantiate once, call fit() on a corpus, then run()/encode()/similarity().
    The dot pool evolves across calls via DotEvolution.
    """

    def __init__(self,
                 feature_dim:           int   = 256,
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
            self.dot_memory.phase_bonus_weight = 0.50 # Boosted for new task
        self.cluster_memory  = ClusterMemory(feature_dim, phase_coding=phase_coding)
        self.evolution       = DotEvolution(EvolutionConfig(), seed)
        self.evaluator       = IECNNMetrics(alpha)

        # Working Memory: stores result of previous run() for context
        self.working_memory: Optional[np.ndarray] = None

        # AGI Control & Long-term Memory
        from cognition.control import SelfModel, CognitiveStateVector
        from memory.graph import WorldGraph
        self.self_model = SelfModel(seed=seed)
        self.world_graph = WorldGraph(feature_dim=feature_dim)
        self.context_map: Optional[np.ndarray] = None # Rolling document context

        # Dot pool (generated lazily on first use; evolved across calls)
        self._dots: Optional[list] = None
        self._call_count: int = 0
        self._rng = np.random.RandomState(seed)

        # Auto-load learned state (dots + memory + evolution) if present
        if persistence_path:
            try:
                self.load_brain(persistence_path)
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[IECNN] Could not load learned state: {e}")

    # ── Setup ────────────────────────────────────────────────────────

    def fit(self, texts: List[str]) -> "IECNN":
        """Discover word and phrase bases from a corpus."""
        self.base_mapper.fit(texts)
        return self

    def _learned_paths(self, persistence_path: str) -> dict:
        base = persistence_path
        return {
            "dots":      base + ".dots.pkl",
            "dotmem":    base + ".dotmem.pkl",
            "clustmem":  base + ".clustmem.pkl",
            "evo":       base + ".evo.pkl",
            "meta":      base + ".meta.pkl",
        }

    def save_brain(self, persistence_path: Optional[str] = None) -> None:
        """Persist all learned state: vocab + dot pool + memory + evolution."""
        import pickle
        from neural_dot.neural_dot import _NEXT_DOT_ID
        path = persistence_path or self.persistence_path
        if not path:
            raise ValueError("No persistence_path set; cannot save brain.")
        # 1. Vocabulary (existing)
        self.base_mapper.save(path)
        # 2. Dot pool, memories, evolution, and metadata
        paths = self._learned_paths(path)
        if self._dots is not None:
            with open(paths["dots"], "wb") as fh:
                pickle.dump(self._dots, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with open(paths["dotmem"], "wb") as fh:
            pickle.dump(self.dot_memory.state_dict(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        with open(paths["clustmem"], "wb") as fh:
            pickle.dump(self.cluster_memory.state_dict(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        with open(paths["evo"], "wb") as fh:
            pickle.dump(self.evolution.state_dict(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        with open(paths["meta"], "wb") as fh:
            pickle.dump({
                "call_count":   self._call_count,
                "next_dot_id":  _NEXT_DOT_ID,
                "feature_dim":  self.feature_dim,
                "num_dots":     self.num_dots,
                "phase_coding": self.phase_coding,
            }, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load_brain(self, persistence_path: Optional[str] = None) -> None:
        """Load all learned state previously written by save_brain()."""
        import pickle
        from neural_dot.neural_dot import set_next_dot_id
        path = persistence_path or self.persistence_path
        if not path:
            raise ValueError("No persistence_path set; cannot load brain.")
        paths = self._learned_paths(path)
        if not os.path.exists(paths["meta"]):
            raise FileNotFoundError(paths["meta"])
        with open(paths["meta"], "rb") as fh:
            meta = pickle.load(fh)
        self._call_count = int(meta.get("call_count", 0))
        if "next_dot_id" in meta:
            set_next_dot_id(int(meta["next_dot_id"]))
        # If the saved brain was trained with phase coding, propagate the
        # flag so we don't silently corrupt phase accumulators on the next
        # commit_pattern call. If the constructor explicitly enabled phase
        # coding for a legacy (no-phase) brain, that's also fine — new
        # patterns will start accumulating phase from this point onward.
        if "phase_coding" in meta:
            self.phase_coding = bool(meta["phase_coding"]) or self.phase_coding
            self.cluster_memory.phase_coding = self.phase_coding
        if os.path.exists(paths["dots"]):
            with open(paths["dots"], "rb") as fh:
                self._dots = pickle.load(fh)
        if os.path.exists(paths["dotmem"]):
            with open(paths["dotmem"], "rb") as fh:
                self.dot_memory.load_state(pickle.load(fh))
            # Reflect current phase_coding setting onto the dot memory so a
            # brain trained without phase coding still gets the bonus enabled
            # when re-launched with --phase-coding (and vice-versa is harmless
            # because the bonus is multiplied by zero-concentration anyway).
            if self.phase_coding and self.dot_memory.phase_bonus_weight <= 0.0:
                self.dot_memory.phase_bonus_weight = 0.30
        if os.path.exists(paths["clustmem"]):
            with open(paths["clustmem"], "rb") as fh:
                self.cluster_memory.load_state(pickle.load(fh))
        if os.path.exists(paths["evo"]):
            with open(paths["evo"], "rb") as fh:
                self.evolution.load_state(pickle.load(fh))

    def prune_dots(self, min_outcomes: int = 2, min_age_gens: int = 2,
                   dry_run: bool = False) -> dict:
        """
        Joint pruner for dot pool + dot memory.

        A dot is removed when it is BOTH:
          - old enough (current_gen - birth_generation >= min_age_gens), and
          - has accumulated fewer than `min_outcomes` recorded predictions.

        After culling dead dots from `self._dots`, dot_memory is pruned to
        the surviving id set, which also removes orphaned history left
        behind by previous generations of dead dots.

        When `dry_run=True`, no state is mutated — only the stats dict is
        returned so callers can preview the impact.

        Returns a small stats dict for logging.
        """
        if not self._dots:
            return {"removed_dots": 0, "removed_history": 0,
                    "kept_dots": 0, "kept_history": 0,
                    "generation": int(self.evolution.generation),
                    "dry_run": dry_run}

        current_gen = int(self.evolution.generation)
        before_dots = len(self._dots)
        before_hist = len(self.dot_memory._total_counts)

        survivors = []
        for d in self._dots:
            birth = int(getattr(d, "birth_generation", 0))
            age = current_gen - birth
            outcomes = self.dot_memory._total_counts.get(d.dot_id, 0.0)
            # Newborns and dots with enough recorded outcomes survive.
            if age < min_age_gens or outcomes >= min_outcomes:
                survivors.append(d)

        # Always keep at least one dot to avoid a degenerate empty pool.
        if not survivors and self._dots:
            survivors = [self._dots[0]]

        keep_ids = [d.dot_id for d in survivors]
        keep_set = set(int(i) for i in keep_ids)
        kept_history = sum(1 for did in self.dot_memory._total_counts
                           if int(did) in keep_set)

        if not dry_run:
            self._dots = survivors
            self.dot_memory.prune(keep_ids)

        return {
            "removed_dots":    before_dots - len(survivors),
            "kept_dots":       len(survivors),
            "removed_history": before_hist - kept_history,
            "kept_history":    kept_history,
            "generation":      current_gen,
            "dry_run":         dry_run,
        }

    def train_pass(self, sentences: List[str], max_iterations: int = 2,
                   max_aim_variants: int = 1, verbose: bool = True,
                   save_every: int = 500,
                   prune_every: int = 0,
                   prune_min_outcomes: int = 2,
                   prune_min_age_gens: int = 2,
                   mask_ratio: float = 0.0) -> "IECNN":
        """
        Run the full pipeline over each sentence so the dot pool, dot memory,
        cluster memory, and evolution all get updated. This is what makes
        the model actually 'learn' from the corpus (not just count vocab).

        Uses reduced max_iterations / aim variants for speed during training.

        When `prune_every > 0`, an explicit `prune_dots(...)` is called every
        N sentences (and once before each periodic save_brain). This bounds
        on-disk growth during long training runs even if the per-evolve
        auto-prune is too lenient for the chosen cadence.
        """
        import time
        # Save & temporarily lower iteration / aim budgets for speed
        orig_max_iter   = self.iter_ctrl.max_iterations
        orig_max_aim    = self.aim.max_variants if hasattr(self.aim, "max_variants") else None
        self.iter_ctrl.max_iterations = max_iterations
        if orig_max_aim is not None:
            self.aim.max_variants = max_aim_variants

        n = len(sentences)
        t0 = time.time()
        try:
            for i, sent in enumerate(sentences, 1):
                if not sent:
                    continue
                try:
                    self.run(sent, verbose=False, mask_ratio=mask_ratio)
                except Exception as e:
                    if verbose:
                        print(f"\n[train] skip line {i}: {e}")
                    continue
                if verbose and (i % 25 == 0 or i == n):
                    elapsed = time.time() - t0
                    rate    = i / max(elapsed, 1e-6)
                    eta     = (n - i) / max(rate, 1e-6)
                    dm      = self.dot_memory.summary()
                    print(f"\r[train] {i:>6}/{n}  "
                          f"({rate:5.1f} ex/s, ETA {eta/60:5.1f}m)  "
                          f"gen={self.evolution.generation:>3}  "
                          f"mean_eff={dm['mean_eff']:.3f}  "
                          f"max_eff={dm['max_eff']:.3f}",
                          end="", flush=True)
                # Periodic deep prune (in addition to the per-evolve auto-prune).
                if prune_every and i % prune_every == 0:
                    stats = self.prune_dots(min_outcomes=prune_min_outcomes,
                                            min_age_gens=prune_min_age_gens)
                    if verbose and (stats["removed_dots"] or stats["removed_history"]):
                        print(f"\n[train] prune @ {i}: "
                              f"-{stats['removed_dots']} dots, "
                              f"-{stats['removed_history']} history "
                              f"(kept {stats['kept_dots']} dots)")
                if self.persistence_path and save_every and i % save_every == 0:
                    # Always prune right before a save so the file we write is compact.
                    if prune_every:
                        self.prune_dots(min_outcomes=prune_min_outcomes,
                                        min_age_gens=prune_min_age_gens)
                    self.save_brain(self.persistence_path)
            if verbose:
                print()
        finally:
            self.iter_ctrl.max_iterations = orig_max_iter
            if orig_max_aim is not None:
                self.aim.max_variants = orig_max_aim

        if self.persistence_path:
            if prune_every:
                self.prune_dots(min_outcomes=prune_min_outcomes,
                                min_age_gens=prune_min_age_gens)
            self.save_brain(self.persistence_path)
        return self

    def fit_file(self, filepath: str, batch_size: int = 500,
                 verbose: bool = True) -> "IECNN":
        """
        Train on a large text file in streaming batches.

        Reads the file line by line so arbitrarily large corpora fit in
        memory.  Each batch_size lines are fed to fit() which enriches the
        BaseMapper vocabulary with word/phrase co-occurrences and IDF weights.
        After the last batch the persistence path is saved (if set).

        Args:
          filepath   — path to a UTF-8 plain-text file (one sentence per line)
          batch_size — lines per training batch (default 500)
          verbose    — print progress to stdout

        Returns self for chaining.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Training file not found: {filepath}")

        total_lines = 0
        batch: List[str] = []

        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                batch.append(line)
                total_lines += 1
                if len(batch) >= batch_size:
                    self.base_mapper.fit(batch)
                    batch = []
                    if verbose:
                        n_bases = len(self.base_mapper._base_vocab)
                        print(f"[IECNN] Trained on {total_lines:>8} lines "
                              f"│ vocab: {n_bases:>6} bases", end="\r")

        if batch:
            self.base_mapper.fit(batch)

        if self.base_mapper.persistence_path:
            self.base_mapper.save(self.base_mapper.persistence_path)

        if verbose:
            n_bases = len(self.base_mapper._base_vocab)
            print(f"\n[IECNN] Vocabulary fit — {total_lines} lines "
                  f"│ vocab: {n_bases} word/phrase bases")
            print(f"[IECNN] Running learning pass over corpus to update dots…")

        # ── Phase 2: real learning pass over the corpus ────────────────
        # Read sentences once more so the dot pool, dot memory, cluster
        # memory and evolution all actually update from the data.
        sentences: List[str] = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                sentences.append(line)

        self.train_pass(sentences, verbose=verbose)
        return self

    def _ensure_dots(self) -> list:
        if self._dots is None:
            self._dots = self.dot_gen.generate()
        return self._dots

    # ── Main run ─────────────────────────────────────────────────────

    def run(self, input_data: Any, verbose: bool = False,
            update_brain: bool = False, mode: str = "text",
            mask_ratio: float = 0.0) -> IECNNResult:
        """
        Run the full 10-layer IECNN pipeline.

        Args:
          input_data   — input data (text, img_path, etc.)
          verbose      — print round-by-round progress
          update_brain — if True, the text enriches the global BaseMapping knowledge
          mode         — 'text' | 'image' | 'audio' | 'video'
          mask_ratio   — fraction of BaseMap rows to 'mask' for MBM Pretraining (v4 SOTA)
        """
        if update_brain and mode == "text":
            self.base_mapper.fit([input_data])
            if self.base_mapper.persistence_path:
                self.base_mapper.save(self.base_mapper.persistence_path)

        self.iter_ctrl.reset()
        self.cluster_memory.reset_call()

        basemap  = self.base_mapper.transform(input_data, mode=mode)

        # ── Layer 2.1.5: Masked BaseMap Modeling (MBM) ───────────
        # For unsupervised pretraining: hide some rows, force dots to predict them.
        if mask_ratio > 0.0:
            basemap = self._apply_mbm(basemap, mask_ratio)

        # ── Layer 2.2: Rolling Context Injection ──────────────────
        if self.context_map is not None:
            # Guide the new sentence with the high-level context of previous ones
            basemap = self._blend(basemap, self.context_map, alpha=0.70)

        # ── Phase-Coded Encoding: walk the token stream once ──
        # Assign phases to dots based on the token position they 'win'
        if self.phase_coding:
            self._apply_phase_coding(basemap)

        # ── Layer 2.5: Working Memory Injection ──────────────────
        # If we have a working memory from the previous call, blend it
        # into the initial basemap to provide narrative context.
        # We use a 50/50 blend to ensure context survives the fresh input noise.
        if self.working_memory is not None:
            basemap = self._blend(basemap, self.working_memory, alpha=0.50)
        dots     = self._ensure_dots()
        cur_bmap = basemap
        all_rounds: List[Dict] = []
        top_cluster: Optional[Cluster] = None
        final_out   = basemap.pool("mean")
        prev_centroid: Optional[np.ndarray] = None
        cur_entropy   = 0.5  # initial entropy (unknown)

        if verbose:
            self._print_header(str(input_data), basemap, dots)

        while True:
            rnd = self.iter_ctrl.current_round

            # ── Memory hints for adaptive attention ──────────────────
            memory_hints = {}
            for dot in dots:
                # 1. Start with episodic hint (v4 SOTA)
                hint = self.dot_memory.episodic_hint(dot.dot_id, cur_bmap.matrix, self.alpha)

                # 2. Fallback to rolling centroid
                if hint is None:
                    hint = self.dot_memory.recent_centroid(dot.dot_id)

                # 3. Last fallback: pattern library
                if hint is None and rnd == 0:
                    hint = self.cluster_memory.closest_pattern(basemap.pool("mean"), self.alpha)

                memory_hints[dot.dot_id] = hint

            # ── Layers 3+4: Dot Prediction ───────────────────────────
            # Provide hazy consensus (previous round centroid) for 'Social Interaction'
            consensus = prev_centroid if prev_centroid is not None else None

            dot_preds = self.dot_gen.run_all(
                cur_bmap, dots,
                memory_hints=memory_hints,
                context_entropy=cur_entropy,
                consensus=consensus,
            )

            # ── Layer 5: AIM (9 inversions in parallel) ───────────────
            candidates = self.aim.transform(dot_preds, cur_bmap)

            # ── Pruning Stage 1 ───────────────────────────────────────
            dom_prev = self.iter_ctrl.current_dominance()
            filt, s1 = self.pruning.stage1(candidates, dom_prev)
            if not filt:
                break

            # ── Layer 6: Convergence (hierarchical, 2-level) ──────────
            clusters, assign = self.convergence.run(filt)

            # ── Pruning Stages 2+3 ────────────────────────────────────
            _, surv, pruning_stats = self.pruning.run(filt, clusters, dom_prev)
            if not surv:
                surv = clusters[:1] if clusters else []

            # ── Compute metrics for this round ────────────────────────
            scores = [c.score for c in surv]
            dom    = dominance_score(scores[0], scores) if scores else 0.0
            ent    = cluster_entropy(scores)
            cur_entropy = ent  # feed forward to next round

            conv_sum = self.convergence.summarize(surv, prev_centroid)
            conv_sum["centroid"] = surv[0].centroid if surv else None

            # ── Record round ──────────────────────────────────────────
            self.iter_ctrl.record_round(surv, pruning_stats)
            self.cluster_memory.record_round(
                rnd,
                [c.centroid for c in surv if c.centroid is not None],
                scores,
            )

            rnd_info = {
                "round":       rnd,
                "candidates":  len(candidates),
                "filtered":    len(filt),
                "clusters":    len(surv),
                "micro":       conv_sum.get("num_micro", 0),
                "dominance":   float(dom),
                "top_score":   float(surv[0].score) if surv else 0.0,
                "entropy":     float(ent),
                "stability":   float(self.iter_ctrl.current_stability()),
                "objective":   float(self.iter_ctrl.current_objective()),
                "energy":      float(self.iter_ctrl.current_energy()),
                "lr":          float(self.iter_ctrl.current_lr()),
                "eug":         float(self.iter_ctrl.current_eug()),
                "delta_u":     float(self.iter_ctrl.utility_acceleration()),
                "centroid":    surv[0].centroid.copy() if surv and surv[0].centroid is not None else None,
            }
            all_rounds.append(rnd_info)

            if verbose:
                self._print_round(rnd_info)

            if surv:
                top_cluster   = surv[0]
                final_out     = top_cluster.centroid.copy()
                prev_centroid = final_out.copy()

            # ── Check stopping conditions ────────────────────────────
            stop, reason = self.iter_ctrl.should_stop(surv)
            if stop:
                if verbose: print(f"\n  ⟹  Stop: [{reason}]")
                break

            # ── Update state for next round ───────────────────────────
            refined = self.iter_ctrl.advance(surv, final_out)

            # ── Adaptive exploration (F16-driven stagnation response) ────
            # When EUG is near-zero the system is in a flat basin: inject
            # noise into the refined vector AND raise context_entropy so
            # dots explore broader temperature / inversion settings next round.
            eug_val = self.iter_ctrl.current_eug()
            if abs(eug_val) < 0.01:
                noise   = self._rng.randn(len(refined)).astype(np.float32) * 0.05
                refined = refined + noise
                cur_entropy = min(1.0, cur_entropy + 0.30)

            cur_bmap = self._blend(basemap, refined)
            self._record_dot_outcomes(surv, candidates, assign, dots, cur_bmap)

            # ── Layer 4.5: Hebbian Local Plasticity ──────────────────
            # Nudge winning dots toward consensus immediately
            if surv:
                win_cid = surv[0].cluster_id
                win_centroid = surv[0].centroid
                for i, (_, _, info) in enumerate(candidates):
                    if i < len(assign) and assign[i] == win_cid:
                        did = info.get("dot_id")
                        head = info.get("head", 0)
                        # Find the actual dot object
                        dot_obj = next((d for d in dots if d.dot_id == did), None)
                        if dot_obj:
                            dot_obj.local_update(win_centroid, head, lr=0.01)

            # ── F17 Dot Reinforcement Pressure ───────────────────────────
            # Dots now respond to normalized global objective J(t)
            obj     = self.iter_ctrl.current_objective()
            # Crude normalization for now: tanh(obj)
            j_norm  = float(np.tanh(obj))

            dot_ids = [d.dot_id for d in dots]
            drp     = self.dot_memory.drp_scores(dot_ids, j_norm)

            # ── F23 Memory Decay ─────────────────────────────────────────
            from formulas.formulas import memory_plasticity
            rho = memory_plasticity(self.iter_ctrl.current_stability())
            self.dot_memory.apply_memory_decay(dot_ids, rho)

            # Step 1 — floor pressure on raw scores (gentle, catches near-zero)
            self.dot_memory.apply_floor_pressure(dot_ids, drp)

            # Step 2 — nonlinear amplification: sign(R) × |R|^1.5
            # Monotone, so ranking is preserved; extremes are stretched further apart.
            drp_amp = (np.sign(drp) * np.abs(drp) ** 1.5).astype(np.float32)

            # Step 3 — competition decay on amplified scores (bottom 30%)
            self.dot_memory.competition_decay(dot_ids, drp_amp)

            # Step 4 — hard selection: bottom 40% cut in half (not just nudged)
            self.dot_memory.hard_selection(dot_ids, drp_amp)

            # Step 5 — inline mutation: weak dots transform structurally
            # Use a wider std when EUG is stagnant (more exploration needed)
            eff          = self.dot_memory.all_effectivenesses(dot_ids)
            mutation_std = 0.10 if abs(eug_val) < 0.01 else 0.05

            # Stable dot_id preservation: mutate in-place if effectiveness is low
            dots         = self.evolution.mutate_weak_dots(dots, eff,
                                                           mutation_std=mutation_std)

            # Step 6 — diversity constraint: if type distribution is too skewed,
            # raise temperature of underrepresented dot types
            if self._compute_diversity(dots) < 0.60:
                self._boost_underrepresented(dots)

            # Step 7 — Dynamic Head Allocation: Grant extra heads to elites
            self._allocate_heads(dots, eff)

            self._learn_bias(surv, candidates, assign, basemap)

            # ── Layer 9.5: AGI Control & Meta-Learning ──────────────────
            # Update Cognitive State Vector (CSV)
            from cognition.control import CognitiveStateVector
            current_csv = CognitiveStateVector(
                entropy=ent, dominance=dom,
                stability=self.iter_ctrl.current_stability(),
                energy=self.iter_ctrl.current_energy(),
                eug=eug_val, call_count=self._call_count,
                reasoning_depth=2 # default
            )

            # Ego deciding internal actions
            actions = self.self_model.decide(current_csv)

            # Apply Self-Model decisions:
            # 1. Modulate mutation pressure
            mutation_std = 0.05 * actions.mutation_pressure
            # 2. Add exploration noise to refined vector
            if actions.exploration_noise > 0.01:
                noise = self._rng.randn(len(refined)).astype(np.float32) * actions.exploration_noise
                refined = refined + noise

            # Learn from the delta in system energy/utility (simplified update)
            u_delta = self.iter_ctrl.utility_acceleration()
            e_delta = self.iter_ctrl.current_energy() - (all_rounds[-2]["energy"] if len(all_rounds) > 1 else 0.5)
            self.self_model.learn(current_csv, actions, u_delta, -e_delta)

            # ── Meta-Learning: Auto-tune DRP and LR ──────────────────────
            # Adjust meta-parameters based on EUG history.
            # If EUG is consistently low, we increase failure penalty (lambda4)
            # and potentially base_lr to escape the local basin.
            self._optimize_meta_params(eug_val)

        # ── Layer 11: Reflection & Self-Correction ────────────────────
        # Winning cluster is vetted by specialized dots to ensure it
        # doesn't violate learned constraints.
        if top_cluster:
            final_out = self._reflect(final_out, dots, basemap)

        # ── Rollback if last round was worse ─────────────────────────
        best = self.iter_ctrl.best_clusters()
        if (best and top_cluster and
                best[0].score > top_cluster.score * 1.05):
            top_cluster = best[0]
            final_out   = best[0].centroid.copy()

        # ── Normalize output vector ───────────────────────────────────
        n = np.linalg.norm(final_out)
        if n > 1e-10: final_out = final_out / n * np.sqrt(self.feature_dim)

        # ── Post-run: update memory + evolve dots ─────────────────────
        # Store output in working memory for next call
        self.working_memory = final_out.copy()

        if top_cluster:
            top_phase = None
            if self.phase_coding:
                mp, mc = top_cluster.mean_phase()
                if mc > 0.0:
                    top_phase = mp
            self.cluster_memory.commit_pattern(final_out, top_cluster.score,
                                               alpha=self.alpha, phase=top_phase)

            # Recursive Base Composition: High-score winners become new Bases
            if top_cluster.score > 0.65:
                # Heuristic naming for composite concept
                if mode == "text":
                    # For text, we can use the top tokens as a name hint
                    # (simplified for now)
                    concept_name = f"concept_{self._call_count % 100}"
                    self.base_mapper.register_composite_base(concept_name, final_out)
        if self.do_evolve:
            self._dots = self.evolution.evolve(dots, self.dot_memory)
            # Joint prune: drop dots that have lived long enough to prove
            # themselves useless, and drop their history from dot_memory so
            # the on-disk brain stops growing unboundedly.
            self.prune_dots()
        self._call_count += 1

        result = IECNNResult(
            output=final_out, basemap=basemap, top_cluster=top_cluster,
            summary=self.iter_ctrl.summary(),
            stop_reason=self.iter_ctrl.stop_reason or StopReason.BUDGET,
            rounds=all_rounds,
        )
        result.metrics = self.evaluator.evaluate(result)
        return result

    # ── Public API ───────────────────────────────────────────────────

    def encode(self, text: str, verbose: bool = False) -> np.ndarray:
        """Encode text to a 256-dim vector (hierarchical for multi-sentence)."""
        sentences = self._split_sentences(text)
        if len(sentences) > 1:
            if verbose: print(f"[IECNN] Processing {len(sentences)} sentences hierarchically...")
            return self.run_hierarchical(sentences, verbose=verbose)
        return self.run(text, verbose=verbose).output

    def run_hierarchical(self, sentences: List[str], verbose: bool = False) -> np.ndarray:
        """
        Rolling Hierarchical Convergence (SOTA upgrade):
        Processes document as a sequence of sentences, where each sentence
        centroid enriches a persistent 'Context Map' that guides the next.
        """
        self.context_map = None # Reset for new document
        final_centroids = []

        for i, sent in enumerate(sentences):
            if verbose: print(f"  [Document] Sentence {i+1}/{len(sentences)}")

            # Process sentence with rolling context injection
            res = self.run(sent, verbose=verbose)

            if res.top_cluster:
                # Update Context Map: EMA blend
                if self.context_map is None:
                    self.context_map = res.top_cluster.centroid.copy()
                else:
                    self.context_map = 0.7 * self.context_map + 0.3 * res.top_cluster.centroid

                final_centroids.append(res.top_cluster.centroid)

                # Consolidate into World Graph periodically
                if i % 5 == 0:
                    self.world_graph.consolidate([(res.top_cluster.centroid, res.top_cluster.score)], alpha=self.alpha)

        if not final_centroids:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Final Document Vector is the final Context Map (sum of narrative arc)
        out = self.context_map
        n = np.linalg.norm(out)
        if n > 1e-10:
            out = out / n * np.sqrt(self.feature_dim)
        return out

    def _apply_mbm(self, basemap: BaseMap, ratio: float) -> BaseMap:
        """
        Hide a portion of the BaseMap matrix for pretraining.
        Masked rows are set to zero (absent).
        """
        n = len(basemap.tokens)
        if n < 2: return basemap

        # Select indices to mask
        mask_count = max(1, int(n * ratio))
        mask_idx = self._rng.choice(n, mask_count, replace=False)

        new_matrix = basemap.matrix.copy()
        for idx in mask_idx:
            new_matrix[idx, :] = 0.0 # Zero out the embedding

        from basemapping.basemapping import BaseMap as BM
        return BM(new_matrix, basemap.tokens, basemap.token_types,
                  basemap.modifiers, {**basemap.metadata, "masked": True, "mask_idx": mask_idx})

    def _apply_phase_coding(self, basemap: BaseMap):
        """
        Walk the token stream and assign phases to dots.
        The dot that 'wins' (most similar) a token position t gets phase 2pi * t / L.
        """
        dots = self._ensure_dots()
        n_tokens = len(basemap.tokens)
        if n_tokens == 0: return

        for t, token_vec in enumerate(basemap.matrix):
            phase = 2.0 * np.pi * t / n_tokens

            # Simple competition: which dot's base projection fits this token best?
            best_dot = None
            best_sim = -1.0

            for dot in dots:
                # Use a simplified projection for fast phase assignment
                # Use only EMBED_DIM for semantic/structural alignment
                sim = similarity_score(token_vec[:EMBED_DIM], dot.W[:EMBED_DIM, 0], self.alpha)
                if sim > best_sim:
                    best_sim = sim
                    best_dot = dot

            if best_dot:
                # Assign phase to dot for this run
                best_dot.current_phase = phase
                # Record this phase sample in dot memory for fitness
                self.dot_memory.record_phase_sample(best_dot.dot_id, phase)

    def _split_sentences(self, text: str) -> List[str]:
        """Simple heuristic sentence splitter."""
        # Split on . ! ? followed by space or end of string
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]

    def generate(self, prompt: str, max_tokens: int = 12,
                 iterations: int = 40) -> str:
        """
        Encode a prompt to a latent vector, then decode back to text.

        Uses a fast BaseMapper pooled-vector as the target latent (no full
        pipeline call) so generation completes in under a second regardless
        of model size.  For maximum semantic fidelity on a small model, pass
        use_pipeline=True to the underlying decoder directly.

        Args:
          prompt     — input text to encode and then regenerate
          max_tokens — maximum output tokens (default 12)
          iterations — decoder search budget: higher = more thorough but slower

        Returns a string, or the original prompt if the decoder cannot improve.
        """
        from decoding.decoder import IECNNDecoder
        # Fast path: get latent from BaseMapper (no full pipeline call)
        if self.base_mapper.is_fitted:
            bm = self.base_mapper.fit_transform([prompt])
            latent = bm.pool("mean").astype(np.float32)
            # Pad to FEATURE_DIM if the pool output is shorter
            if len(latent) < self.feature_dim:
                padded = np.zeros(self.feature_dim, dtype=np.float32)
                padded[:len(latent)] = latent
                latent = padded
        else:
            latent = self.encode(prompt)

        decoder = IECNNDecoder(self)
        out = decoder.decode(latent, target_mode="text",
                             max_tokens=max_tokens, iterations=iterations)
        return out if out and out != "..." else prompt

    def similarity(self, a: str, b: str, update_brain: bool = False) -> float:
        """Similarity between two texts (F1)."""
        # If we update brain, we do it for both first to ensure mutual knowledge
        if update_brain:
            self.base_mapper.fit([a, b])
            if self.base_mapper.persistence_path:
                self.base_mapper.save(self.base_mapper.persistence_path)

        return float(similarity_score(
            self.encode(a),
            self.encode(b),
            self.alpha
        ))

    def compare(self, texts: List[str]) -> np.ndarray:
        """Return n×n similarity matrix for a list of texts."""
        vecs = [self.encode(t) for t in texts]
        n = len(vecs)
        mat = np.zeros((n, n), np.float32)
        for i in range(n):
            for j in range(i, n):
                s = similarity_score(vecs[i], vecs[j], self.alpha)
                mat[i, j] = mat[j, i] = s
        return mat

    def memory_status(self) -> Dict:
        """Summary of dot memory and cluster memory state."""
        dots = self._ensure_dots()
        dot_ids = [d.dot_id for d in dots]
        return {
            "dot_memory":     self.dot_memory.summary(dot_ids),
            "cluster_memory": self.cluster_memory.summary(),
            "evolution":      self.evolution.stats(self.dot_memory),
            "call_count":     self._call_count,
        }

    # ── Internal helpers ─────────────────────────────────────────────

    def _reflect(self, output: np.ndarray, dots: list, basemap: BaseMap) -> np.ndarray:
        """
        Counterfactual Reasoning Layer (v2):
        Resolves semantic contradictions by simulating alternative interpretations.

        1. Identifies 'veto' dots (LOGIC/SEMANTIC) that strongly disagree with the output.
        2. For each veto, runs an AIM inversion to see if a counterfactual
           interpretation provides a better fit for the conflicting dots.
        3. Fuses the most consistent counterfactuals back into the final output.
        """
        # Select specialized dots for reflection
        veto_dots = [d for d in dots if d.dot_type in (DotType.LOGIC, DotType.SEMANTIC)]
        if not veto_dots:
            return output

        # Sample a few dots for efficiency
        sample_size = min(12, len(veto_dots))
        idx = self._rng.choice(len(veto_dots), sample_size, replace=False)
        selected = [veto_dots[i] for i in idx]

        corrections = []

        for dot in selected:
            # Get dot's internal predictions for the current output context
            preds = dot.predict(basemap, memory_hint=output, context_entropy=0.05)

            for p, conf, _ in preds:
                sim = similarity_score(output, p, self.alpha)

                # If dot strongly disagrees (sim < 0.35), simulate counterfactuals
                if sim < 0.35:
                    # 1. Choose inversion type based on dot's requested inversion or default to feature
                    inv_type = dot.dot_type.name.lower() if hasattr(dot, "requested_inversion") and dot.requested_inversion else "feature"

                    # 2. Transform the dot's prediction using AIM to explore 'the other side'
                    # We use the internal dispatch from aim.aim
                    from aim.aim import _apply_inversion
                    p_inv = _apply_inversion(inv_type, p, basemap.matrix, self._rng)

                    # p_hat = Attention(output, context, Invert(p))
                    from formulas.formulas import aim_transform
                    ctx2d = basemap.matrix.reshape(1, -1) if basemap.matrix.ndim == 1 else basemap.matrix
                    p_hat = aim_transform(p, ctx2d, lambda _: p_inv)

                    # 3. If the counterfactual interpretation aligns better with the rest of the context
                    # or resolves the contradiction, we use it as a correction signal.
                    hat_sim = similarity_score(output, p_hat, self.alpha)
                    if hat_sim > sim:
                        # The inverted interpretation is more plausible than the direct contradiction
                        corrections.append(conf * (p_hat - output))
                    else:
                        # Direct correction (nudge toward dot's original prediction)
                        corrections.append(conf * (p - output))

        if corrections:
            # Apply weighted corrections (conservative blend to maintain stability)
            nudge = np.mean(np.stack(corrections), axis=0)
            return output + 0.15 * nudge

        return output

    def _optimize_meta_params(self, current_eug: float):
        """Adjust DRP weights and Base LR based on utility gradient."""
        # Simple heuristic: if EUG is low, increase selection pressure
        if abs(current_eug) < 0.005:
            # Increase failure penalty to flush out weak dots
            self.dot_memory.lambda4 = min(0.30, self.dot_memory.lambda4 + 0.01)
            # Increase base learning rate to explore more
            self.iter_ctrl.base_lr = min(0.20, self.iter_ctrl.base_lr + 0.005)
        elif current_eug > 0.05:
            # System is doing well, potentially reduce pressure to stabilize
            self.dot_memory.lambda4 = max(0.05, self.dot_memory.lambda4 - 0.005)
            self.iter_ctrl.base_lr = max(0.05, self.iter_ctrl.base_lr - 0.002)

    def _blend(self, original: BaseMap, refined: np.ndarray, alpha: float = 0.85) -> BaseMap:
        """Blend the refined vector back into the basemap for the next round."""
        mat = original.matrix.copy()
        n = np.linalg.norm(refined)
        if n > 1e-10:
            r = refined / n
            mat = alpha * mat + (1.0 - alpha) * r[None, :]
        from basemapping.basemapping import BaseMap as BM
        return BM(mat, original.tokens, original.token_types,
                  original.modifiers, {**original.metadata, "blended": True})

    def _record_dot_outcomes(self, surviving: List[Cluster], candidates: List[Tuple],
                             assignments: List[int], dots: list, basemap: BaseMap):
        """Update DotMemory: which dots contributed to the winning cluster."""
        if not surviving: return
        win_cid = {surviving[0].cluster_id}

        # Compute initial agreement for surprise tracking
        # (Simplified: mean confidence of all candidates in this round)
        avg_conf = np.mean([c[1] for c in candidates])

        for i, (pred, conf, info) in enumerate(candidates):
            in_winner = (i < len(assignments) and assignments[i] in win_cid)
            dot_id = info.get("dot_id", -1)
            if dot_id >= 0:
                # Get input slice context for episodic memory
                sl_start, sl_end = info.get("slice", (0, 0))
                ctx = basemap.matrix[sl_start:sl_end]

                self.dot_memory.record(dot_id, pred, in_winner,
                                       phase=info.get("phase"),
                                       initial_agreement=avg_conf,
                                       input_context=ctx)

    def _learn_bias(self, surviving: List[Cluster], candidates: List[Tuple],
                    assignments: List[int], basemap: BaseMap):
        """Update global base_bias and refine BaseMapper embeddings based on consensus."""
        if not surviving: return
        win_cid = {surviving[0].cluster_id}
        win_biases = []
        aim_wins = total = 0
        for i, (_, _, info) in enumerate(candidates):
            if i < len(assignments) and assignments[i] in win_cid:
                total += 1
                if info.get("inversion_type") is not None:
                    aim_wins += 1
                b = info.get("bias")
                if b is not None:
                    win_biases.append(b.to_array())
        if not win_biases: return

        dom = self.iter_ctrl.current_dominance()
        lr  = self.iter_ctrl.current_lr()

        win_mean = np.mean(np.stack(win_biases), axis=0).astype(np.float32)
        self.base_bias = BiasVector.from_array(
            self.base_bias.to_array() + lr * (win_mean - self.base_bias.to_array())
        )

        # Consensus-Driven Learning: Nudge BaseMapper embeddings
        # Identify tokens that were in the winning cluster
        win_cid = {surviving[0].cluster_id}
        winning_centroid = surviving[0].centroid

        # Simple back-attribution: tokens used by winning dots get refined
        active_tokens = set()
        for i, (_, _, info) in enumerate(candidates):
            if i < len(assignments) and assignments[i] in win_cid:
                sl_start, sl_end = info.get("slice", (0, 0))
                for t_idx in range(sl_start, sl_end):
                    if t_idx < len(basemap.tokens):
                        active_tokens.add(basemap.tokens[t_idx])

        # Nudge these bases toward the consensus
        if active_tokens:
            refine_lr = 0.02 * lr
            for tok in active_tokens:
                if tok in self.base_mapper._base_vocab:
                    old_emb = self.base_mapper._base_vocab[tok]
                    # Winning centroid is complex, we take real part for base storage
                    target = np.real(winning_centroid[:EMBED_DIM]).astype(np.float32)
                    new_emb = old_emb + refine_lr * (target - old_emb)
                    n = np.linalg.norm(new_emb)
                    self.base_mapper._base_vocab[tok] = new_emb / n if n > 1e-10 else new_emb
        # AIM feedback: if inversions won often, raise inversion_bias
        if total > 0:
            ratio = aim_wins / total
            arr = self.base_bias.to_array()
            arr[3] = float(np.clip(arr[3] + ratio * lr * 0.5, 0.05, 0.95))
            self.base_bias = BiasVector.from_array(arr)
        self.dot_gen.base_bias = self.base_bias

    def _compute_diversity(self, dots: list) -> float:
        """
        Compute type diversity using the Simpson index:
            Diversity = 1 - Σ p(type)²
        where p(type) is the fraction of dots of each type.

        Returns 1.0 (fully diverse) for a uniform distribution across all 6
        types, and 0.0 if all dots share the same type.
        """
        if not dots:
            return 1.0
        counts: dict = {}
        for d in dots:
            t = d.dot_type
            counts[t] = counts.get(t, 0) + 1
        n = len(dots)
        return float(1.0 - sum((c / n) ** 2 for c in counts.values()))

    def _allocate_heads(self, dots: list, effectivenesses: np.ndarray):
        """
        Dynamically allocate prediction heads based on effectiveness.
        Top 20% get 6 heads, bottom 40% get 2 heads, others get default (4).
        """
        n = len(dots)
        if n < 5: return

        # effectivenesses corresponds to the dots list order
        order = np.argsort(effectivenesses)[::-1]

        top_k = max(1, int(n * 0.20))
        bot_k = max(1, int(n * 0.40))

        elites = set(order[:top_k])
        weak   = set(order[-bot_k:])

        for i, dot in enumerate(dots):
            if i in elites:
                dot.n_heads = 6
            elif i in weak:
                dot.n_heads = 2
            else:
                dot.n_heads = self.n_heads # Default (4)

    def _boost_underrepresented(self, dots: list) -> None:
        """
        Raise the temperature in the bias vector of underrepresented dot types
        so they explore more aggressively and can re-establish their niche.

        A type is considered underrepresented when its count is less than half
        the expected count for a perfectly uniform distribution across all types.
        """
        if not dots:
            return
        counts: dict = {}
        for d in dots:
            t = d.dot_type
            counts[t] = counts.get(t, 0) + 1
        n       = len(dots)
        n_types = len(DotType.__members__)
        expected = n / n_types
        for dot in dots:
            if counts.get(dot.dot_type, 0) < expected * 0.50:
                arr    = dot.bias.to_array()
                arr[4] = float(np.clip(arr[4] * 1.5, 0.30, 2.0))  # index 4 = temperature
                dot.bias = BiasVector.from_array(arr)

    # ── Verbose output ────────────────────────────────────────────────

    def _print_header(self, text: str, basemap: BaseMap, dots: list):
        trunc = text[:55] + ("..." if len(text) > 55 else "")
        type_dist = self.dot_gen.type_distribution(dots)
        type_str  = " ".join(f"{k[:3]}={v}" for k, v in sorted(type_dist.items()))
        print(f"\n{'═'*74}")
        print(f"  IECNN  │  input: '{trunc}'")
        print(f"  {basemap}  │  dots: {len(dots)} ({type_str})")
        print(f"{'═'*74}")
        print(f"  {'Rnd':>3}  {'cands':>5}  {'kept':>5}  {'cls':>4}  "
              f"{'dom':>6}  {'obj':>7}  {'ent':>6}  {'stab':>6}  {'enrg':>6}  {'lr':>6}  {'eug':>7}")
        print(f"  {'─'*71}")

    def _print_round(self, r: Dict):
        eug = r.get("eug", float("nan"))
        eug_str = f"{eug:>+7.4f}" if eug == eug else "    n/a"
        print(f"  {r['round']:>3}  {r['candidates']:>5}  {r['filtered']:>5}  "
              f"{r['clusters']:>4}  "
              f"{r['dominance']:>6.3f}  {r['objective']:>7.4f}  "
              f"{r['entropy']:>6.3f}  {r['stability']:>6.3f}  "
              f"{r['energy']:>6.3f}  "
              f"{r['lr']:>6.4f}  {eug_str}")
