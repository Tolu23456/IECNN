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
from typing import List, Dict, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basemapping.basemapping import BaseMapper, BaseMap
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
    amplify_pressure,
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
                 micro_threshold:       float = 0.60,
                 macro_threshold:       float = 0.40,
                 dominance_threshold:   float = 0.70,
                 novelty_threshold:     float = 0.05,
                 stability_threshold:   float = 0.99,
                 alpha:                 float = 0.70,
                 gamma:                 float = 0.30,
                 soft_conf:             float = 0.06,
                 max_aim_variants:      int   = 4,
                 base_lr:               float = 0.10,
                 evolve:                bool  = True,
                 persistence_path:      Optional[str] = None,
                 seed:                  int   = 42):
        self.feature_dim = feature_dim
        self.num_dots    = num_dots
        self.n_heads     = n_heads
        self.alpha       = alpha
        self.seed        = seed
        self.do_evolve   = evolve

        # Core components
        self.base_mapper = BaseMapper(feature_dim=feature_dim)
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
        self.cluster_memory  = ClusterMemory(feature_dim)
        self.evolution       = DotEvolution(EvolutionConfig(), seed)
        self.evaluator       = IECNNMetrics(alpha)

        # Dot pool (generated lazily on first use; evolved across calls)
        self._dots: Optional[list] = None
        self._call_count: int = 0
        self._rng = np.random.RandomState(seed)

    # ── Setup ────────────────────────────────────────────────────────

    def fit(self, texts: List[str]) -> "IECNN":
        """Discover word and phrase bases from a corpus."""
        self.base_mapper.fit(texts)
        return self

    def _ensure_dots(self) -> list:
        if self._dots is None:
            self._dots = self.dot_gen.generate()
        return self._dots

    # ── Main run ─────────────────────────────────────────────────────

    def run(self, text: str, verbose: bool = False, update_brain: bool = False) -> IECNNResult:
        """
        Run the full 10-layer IECNN pipeline on a text input.

        Args:
          text         — input text
          verbose      — print round-by-round progress
          update_brain — if True, the text enriches the global BaseMapping knowledge
        """
        if update_brain:
            self.base_mapper.fit([text])
            if self.base_mapper.persistence_path:
                self.base_mapper.save(self.base_mapper.persistence_path)

        self.iter_ctrl.reset()
        self.cluster_memory.reset_call()

        basemap  = self.base_mapper.transform(text)
        dots     = self._ensure_dots()
        cur_bmap = basemap
        all_rounds: List[Dict] = []
        top_cluster: Optional[Cluster] = None
        final_out   = basemap.pool("mean")
        prev_centroid: Optional[np.ndarray] = None
        cur_entropy   = 0.5  # initial entropy (unknown)

        if verbose:
            self._print_header(text, basemap, dots)

        while True:
            rnd = self.iter_ctrl.current_round

            # ── Memory hints for adaptive attention ──────────────────
            memory_hints = {}
            for dot in dots:
                hint = self.dot_memory.recent_centroid(dot.dot_id)
                if hint is None and rnd == 0:
                    # Seed round 0 with cluster pattern library hints
                    # if dot memory is empty
                    hint = self.cluster_memory.closest_pattern(basemap.pool("mean"), self.alpha)
                memory_hints[dot.dot_id] = hint

            # ── Layers 3+4: Dot Prediction ───────────────────────────
            dot_preds = self.dot_gen.run_all(
                cur_bmap, dots,
                memory_hints=memory_hints,
                context_entropy=cur_entropy,
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
                "stability":   float(conv_sum.get("stability", 0.0)),
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
            self._record_dot_outcomes(surv, candidates, assign, dots)

            # ── F17 Dot Reinforcement Pressure ───────────────────────────
            delta_u = self.iter_ctrl.utility_acceleration()
            dot_ids = [d.dot_id for d in dots]
            drp     = self.dot_memory.drp_scores(dot_ids, eug_val, delta_u)

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
            dots         = self.evolution.mutate_weak_dots(dots, eff,
                                                           mutation_std=mutation_std)

            # Step 6 — diversity constraint: if type distribution is too skewed,
            # raise temperature of underrepresented dot types
            if self._compute_diversity(dots) < 0.60:
                self._boost_underrepresented(dots)

            self._learn_bias(surv, candidates, assign)

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
        if top_cluster:
            self.cluster_memory.commit_pattern(final_out, top_cluster.score, self.alpha)
        if self.do_evolve:
            self._dots = self.evolution.evolve(dots, self.dot_memory)
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

    def encode(self, text: str) -> np.ndarray:
        """Encode text to a 128-dim vector."""
        return self.run(text).output

    def similarity(self, a: str, b: str, update_brain: bool = False) -> float:
        """Similarity between two texts (F1)."""
        # If we update brain, we do it for both first to ensure mutual knowledge
        if update_brain:
            self.base_mapper.fit([a, b])
            if self.base_mapper.persistence_path:
                self.base_mapper.save(self.base_mapper.persistence_path)

        return float(similarity_score(
            self.run(a, update_brain=False).output,
            self.run(b, update_brain=False).output,
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
        Self-Correction: Veto/Refine output based on logical consistency.
        Uses LOGIC and SEMANTIC dots to 're-predict' from the final output.
        """
        # Select specialized dots for reflection
        veto_dots = [d for d in dots if d.dot_type in (DotType.LOGIC, DotType.SEMANTIC)]
        if not veto_dots:
            return output

        # Sample a few dots
        sample_size = min(10, len(veto_dots))
        idx = self._rng.choice(len(veto_dots), sample_size, replace=False)
        selected = [veto_dots[i] for i in idx]

        # Each selected dot 'critiques' the output by measuring its
        # consistency with the original basemap.
        corrections = []
        for dot in selected:
            # We treat the final output as a 'prediction' and see if the dot agrees
            preds = dot.predict(basemap, memory_hint=output, context_entropy=0.1)
            for p, conf, _ in preds:
                sim = similarity_score(output, p, self.alpha)
                if sim < 0.4: # Potential contradiction
                    # Add a correction nudge toward the dot's perspective
                    corrections.append(conf * (p - output))

        if corrections:
            # Apply weighted corrections
            nudge = np.mean(np.stack(corrections), axis=0)
            return output + 0.2 * nudge

        return output

    def _blend(self, original: BaseMap, refined: np.ndarray) -> BaseMap:
        """Blend the refined vector back into the basemap for the next round."""
        mat = original.matrix.copy()
        n = np.linalg.norm(refined)
        if n > 1e-10:
            r = refined / n
            mat = 0.85 * mat + 0.15 * r[None, :]
        from basemapping.basemapping import BaseMap as BM
        return BM(mat, original.tokens, original.token_types,
                  original.modifiers, {**original.metadata, "blended": True})

    def _record_dot_outcomes(self, surviving: List[Cluster], candidates: List[Tuple],
                             assignments: List[int], dots: list):
        """Update DotMemory: which dots contributed to the winning cluster."""
        if not surviving: return
        win_cid = {surviving[0].cluster_id}
        for i, (pred, conf, info) in enumerate(candidates):
            in_winner = (i < len(assignments) and assignments[i] in win_cid)
            dot_id = info.get("dot_id", -1)
            if dot_id >= 0:
                self.dot_memory.record(dot_id, pred, in_winner)

    def _learn_bias(self, surviving: List[Cluster], candidates: List[Tuple],
                    assignments: List[int]):
        """Update global base_bias based on winning prediction characteristics."""
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
        print(f"  {'Rnd':>3}  {'cands':>5}  {'kept':>5}  {'cls':>4}  {'mic':>4}  "
              f"{'dom':>6}  {'top':>7}  {'ent':>6}  {'stab':>6}  {'lr':>6}  {'eug':>7}")
        print(f"  {'─'*71}")

    def _print_round(self, r: Dict):
        eug = r.get("eug", float("nan"))
        eug_str = f"{eug:>+7.4f}" if eug == eug else "    n/a"
        print(f"  {r['round']:>3}  {r['candidates']:>5}  {r['filtered']:>5}  "
              f"{r['clusters']:>4}  {r['micro']:>4}  "
              f"{r['dominance']:>6.3f}  {r['top_score']:>7.4f}  "
              f"{r['entropy']:>6.3f}  {r['stability']:>6.3f}  "
              f"{r['lr']:>6.4f}  {eug_str}")
