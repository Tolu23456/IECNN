import numpy as np
import time
from neural_dot.neural_dot import NeuralDot, DotType

def benchmark_relational_pool():
    print("=== NeuralDot Optimization Benchmark ===")

    # Create a dummy dot
    dot = NeuralDot(dot_id=1, feature_dim=256, dot_type=DotType.RELATIONAL)

    # Test with different sequence lengths
    seq_lengths = [10, 50, 200, 500]

    for n in seq_lengths:
        mat = np.random.randn(n, 256).astype(np.float32)

        # Benchmark optimized version
        start = time.time()
        for _ in range(100):
            res_opt = dot._relational_pool(mat)
        end = time.time()
        opt_time = (end - start) / 100

        # Benchmark naive version (simulation of old loop)
        def _naive_pool(mat):
            n = mat.shape[0]
            diffs = []
            for i in range(n):
                for j in range(i+1, n):
                    diffs.append(mat[i] - mat[j])
            return np.mean(np.stack(diffs), axis=0) if diffs else mat[0]

        start = time.time()
        # For large n, only run once to avoid timeout
        iters = 1 if n > 200 else 100
        for _ in range(iters):
            res_naive = _naive_pool(mat)
        end = time.time()
        naive_time = (end - start) / iters

        print(f"Sequence length: {n:>3} | Optimized: {opt_time*1000:7.3f}ms | Naive: {naive_time*1000:7.3f}ms | Speedup: {naive_time/opt_time:6.1f}x")

        # Verify correctness
        assert np.allclose(res_opt, res_naive, atol=1e-5), f"Correctness check failed at n={n}"

    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    benchmark_relational_pool()
