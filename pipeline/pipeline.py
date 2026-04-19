"""
IECNN Pipeline — full 10-layer architecture.

Layers:
  1.  Input              — receives raw data
  2.  BaseMapping        — converts to structured token maps
  3.  Dot Generation     — creates diverse neural dot pool
  4.  Prediction         — each dot independently predicts
  5.  AIM                — generates inverted variants (parallel with originals)
  6.  Convergence        — clusters all candidates by similarity + agreement
  7.  Merge              — centroid of winning cluster
  8.  Pruning            — 3-stage removal of weak candidates/clusters
  9.  Iteration Control  — checks 3 stopping conditions, loops back if needed
  10. Output             — returns final stable vector
"""

import numpy as np
from typing import List, Dict, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basemapping.basemapping import BaseMapper, BaseMap
from neural_dot.neural_dot   import BiasVector, DotGenerator
from aim.aim                 import AIMLayer
from convergence.convergence import ConvergenceLayer, Cluster
from pruning.pruning         import PruningLayer
from iteration.iteration     import IterationController, StopReason
from formulas.formulas       import similarity_score


class IECNNResult:
    def __init__(self, output, basemap, top_cluster, summary, stop_reason, rounds):
        self.output       = output
        self.basemap      = basemap
        self.top_cluster  = top_cluster
        self.summary      = summary
        self.stop_reason  = stop_reason
        self.rounds       = rounds

    def __repr__(self):
        return (f"IECNNResult(rounds={self.summary.get('rounds',0)}, "
                f"stop='{self.stop_reason}', norm={np.linalg.norm(self.output):.3f})")


class IECNN:
    def __init__(self, feature_dim=128, num_dots=64, max_iterations=10,
                 similarity_threshold=0.45, dominance_threshold=0.70,
                 novelty_threshold=0.05, alpha=0.7, max_aim_variants=3,
                 soft_conf=0.08, seed=42):
        self.feature_dim = feature_dim
        self.num_dots    = num_dots
        self.seed        = seed

        self.base_mapper  = BaseMapper(feature_dim=feature_dim)
        self.base_bias    = BiasVector(0.5, 0.5, 0.5, 0.3, 1.0)
        self.dot_gen      = DotGenerator(num_dots, feature_dim, self.base_bias, seed)
        self.aim          = AIMLayer(max_aim_variants, seed)
        self.convergence  = ConvergenceLayer(similarity_threshold, alpha, dominance_threshold)
        self.pruning      = PruningLayer(soft_conf=soft_conf, max_aim_per_dot=max_aim_variants, alpha=alpha)
        self.iter_ctrl    = IterationController(max_iterations, dominance_threshold, novelty_threshold)

    def fit(self, texts: List[str]) -> "IECNN":
        self.base_mapper.fit(texts)
        return self

    def run(self, text: str, verbose: bool = False) -> IECNNResult:
        self.iter_ctrl.reset()
        basemap  = self.base_mapper.transform(text)
        dots     = self.dot_gen.generate()
        cur_bmap = basemap
        all_rounds = []
        top_cluster: Optional[Cluster] = None
        final_out   = basemap.pool("mean")

        if verbose:
            print(f"\n{'='*62}")
            print(f"  IECNN | input: '{text[:55]}{'...' if len(text)>55 else ''}'")
            print(f"  {basemap}  |  dots: {len(dots)}")
            print(f"{'='*62}")

        while True:
            rnd = self.iter_ctrl.current_round

            dot_preds   = self.dot_gen.run_all(cur_bmap, dots)
            candidates  = self.aim.transform(dot_preds, cur_bmap)
            filt, s1    = self.pruning.stage1(candidates)

            if not filt:
                break

            clusters, assign = self.convergence.run(filt)
            _, surv, pruning_stats = self.pruning.run(filt, clusters)

            if not surv: surv = clusters[:1] if clusters else []

            self.iter_ctrl.record_round(surv, pruning_stats)
            conv_sum = self.convergence.summarize(surv)
            dom, _   = self.convergence.dominance(surv)

            rnd_info = {
                "round": rnd, "candidates": len(candidates),
                "filtered": len(filt), "clusters": len(surv),
                "dominance": float(dom), "top_score": conv_sum.get("top_score", 0.0),
            }
            all_rounds.append(rnd_info)

            if verbose:
                print(f"  Round {rnd}: {len(candidates):>3} cands → "
                      f"{len(filt):>3} kept → {len(surv):>2} clusters | "
                      f"dom={dom:.3f}  top={conv_sum.get('top_score',0):.4f}")

            if surv:
                top_cluster = surv[0]
                final_out   = top_cluster.centroid.copy()

            stop, reason = self.iter_ctrl.should_stop(surv)
            if stop:
                if verbose: print(f"\n  Stop: {reason}")
                break

            refined  = self.iter_ctrl.advance(surv, final_out)
            cur_bmap = self._blend(basemap, refined)
            self._learn(surv, candidates, assign)

        n = np.linalg.norm(final_out)
        if n > 1e-10: final_out = final_out / n * np.sqrt(self.feature_dim)

        return IECNNResult(
            output=final_out, basemap=basemap, top_cluster=top_cluster,
            summary=self.iter_ctrl.summary(),
            stop_reason=self.iter_ctrl.stop_reason or StopReason.BUDGET,
            rounds=all_rounds,
        )

    def encode(self, text: str) -> np.ndarray:
        return self.run(text).output

    def similarity(self, a: str, b: str) -> float:
        return float(similarity_score(self.encode(a), self.encode(b)))

    def _blend(self, original: BaseMap, refined: np.ndarray) -> BaseMap:
        mat = original.matrix.copy()
        n   = np.linalg.norm(refined)
        if n > 1e-10:
            r = refined / n
            mat = 0.8 * mat + 0.2 * r[None, :]
        from basemapping.basemapping import BaseMap as BM
        return BM(mat, original.tokens, original.token_types,
                  original.modifiers, {**original.metadata, "blended": True})

    def _learn(self, surviving, candidates, assignments):
        if not surviving: return
        win_ids = {surviving[0].cluster_id}
        aim_wins = total = 0
        for i, (_, _, info) in enumerate(candidates):
            if i < len(assignments) and assignments[i] in win_ids:
                total += 1
                if info.get("inversion_type") is not None: aim_wins += 1
        if total > 0:
            ratio = aim_wins / total
            arr   = self.base_bias.to_array()
            arr[3] = min(0.9, arr[3] + ratio * 0.1)
            self.base_bias = BiasVector.from_array(arr)
            self.dot_gen.base_bias = self.base_bias
