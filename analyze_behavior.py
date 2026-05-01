import sys
import os
import numpy as np
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from pipeline.pipeline import IECNN

def deep_behavioral_analysis(text: str):
    print(f"--- DEEP BEHAVIORAL ANALYSIS: '{text}' ---")
    model = IECNN(num_dots=128, persistence_path="global_brain.pkl")
    res = model.run(text, verbose=False)

    # 1. Convergence Dynamics
    print("\n[1] Convergence Dynamics")
    rounds = res.rounds
    print(f"Total Rounds: {len(rounds)}")
    for r in rounds:
        print(f"  R{r['round']}: Dom={r['dominance']:.4f}, Obj={r['objective']:.4f}, Energy={r['energy']:.4f}, EUG={r['eug']:.4f}")

    # 2. Neural Dot Specialization
    print("\n[2] Neural Dot Contribution")
    if res.top_cluster:
        types = res.top_cluster.dot_types()
        modalities = res.top_cluster.modalities()
        sources = res.top_cluster.sources()

        print("  Winning Dot Types:")
        for t, c in sorted(types.items(), key=lambda x: -x[1]):
            print(f"    - {t}: {c} dots")

        print("  Winning Modalities:")
        for m, c in modalities.items():
            print(f"    - {m}: {c} dots")

        print("  Winning Sources (Original vs AIM):")
        for s, c in sources.items():
            print(f"    - {s}: {c} candidates")

    # 3. Memory & Graph
    print("\n[3] Knowledge Integration")
    status = model.memory_status()
    dm = status["dot_memory"]
    cm = status["cluster_memory"]
    print(f"  Active Dots: {dm['active_dots']}")
    print(f"  Mean Effectiveness: {dm['mean_eff']:.4f}")
    print(f"  World Graph Nodes: {len(model.world_graph.nodes)}")

    # 4. Self-Model Actions
    print("\n[4] Meta-Cognitive Actions (Final Round)")
    csv = model.self_model.history[-1] if hasattr(model.self_model, 'history') and model.self_model.history else None
    # Since I didn't store history in the SM class exactly like that, let's just run a decision
    from cognition.control import CognitiveStateVector
    last_r = rounds[-1]
    current_csv = CognitiveStateVector(
        entropy=last_r['entropy'], dominance=last_r['dominance'],
        stability=last_r['stability'], energy=last_r['energy'],
        eug=last_r['eug'], call_count=model._call_count
    )
    actions = model.self_model.decide(current_csv)
    print(f"  Mutation Pressure: {actions.mutation_pressure:.4f}")
    print(f"  Reasoning Depth Delta: {actions.reasoning_depth_delta:.4f}")
    print(f"  Iteration Budget Delta: {actions.iteration_budget_delta}")

if __name__ == "__main__":
    deep_behavioral_analysis("The quick brown fox jumps over the lazy dog.")
    deep_behavioral_analysis("If the input is logical, the convergence should be high.")
