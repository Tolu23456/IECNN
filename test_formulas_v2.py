import numpy as np
from formulas.formulas import (
    global_energy, system_objective, memory_plasticity,
    dot_fitness, stability_energy, exploration_pressure,
    emergent_utility_gradient, dot_reinforcement_pressure
)

def test_new_formulas():
    print("Testing F21: Global Energy")
    e = global_energy(0.5, 0.7, 0.1)
    print(f"Energy: {e}")
    assert isinstance(e, float)

    print("\nTesting F22: System Objective")
    j = system_objective(0.8, 0.1, e)
    print(f"J(t): {j}")
    assert isinstance(j, float)

    print("\nTesting F23: Memory Plasticity")
    rho = memory_plasticity(0.9)
    print(f"rho: {rho}")
    assert 0 < rho < 1

    print("\nTesting F24: Dot Fitness")
    f = dot_fitness(0.5, 0.8, 0.9, 0.1, 0.2)
    print(f"Fitness: {f}")
    assert isinstance(f, float)

    print("\nTesting F25: Stability Energy")
    s = stability_energy(0.5, 0.1)
    print(f"Stability S(t): {s}")
    assert isinstance(s, float)

    print("\nTesting F26: Exploration Pressure")
    x = exploration_pressure(s, 0.7)
    print(f"Exploration X(t): {x}")
    assert isinstance(x, float)

    print("\nTesting Fixed F16: EUG")
    eug = emergent_utility_gradient([0.5, 0.6], [0.5, 0.4], [0.8, 0.85])
    print(f"EUG: {eug}")
    # ΔC = 0.1, ΔH = -0.1, ΔS = 0.05
    # EUG = 0.1 + 0.3*(-0.1) + 0.3*(0.05) = 0.1 - 0.03 + 0.015 = 0.085
    assert abs(eug - 0.085) < 1e-6

    print("\nTesting Fixed F17: DRP")
    drp = dot_reinforcement_pressure(0.8, 0.9, 0.5, 0.2)
    print(f"DRP: {drp}")
    assert isinstance(drp, float)

    print("\nAll new formulas tested successfully!")

if __name__ == "__main__":
    test_new_formulas()
