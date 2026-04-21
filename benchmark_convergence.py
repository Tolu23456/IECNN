import numpy as np
import time
from convergence.convergence import ConvergenceLayer, MicroCluster

def benchmark_micro_clustering():
    print("=== ConvergenceLayer Optimization Benchmark ===")

    cl = ConvergenceLayer(micro_threshold=0.60, alpha=0.7)

    # Test with different candidate pool sizes
    pool_sizes = [100, 500, 2000]

    for n in pool_sizes:
        preds = [np.random.randn(256).astype(np.float32) for _ in range(n)]
        # Normalize
        for p in preds: p /= np.linalg.norm(p)
        confs = [0.8] * n
        infos = [{"id": i} for i in range(n)]

        # Benchmark optimized version
        start = time.time()
        for _ in range(10):
            res_opt = cl._micro_cluster(preds, confs, infos)
        end = time.time()
        opt_time = (end - start) / 10

        # Benchmark naive version (simulation of old loop)
        def _naive_cluster(preds, confs, infos, threshold):
            clusters = []
            centroids = []
            from formulas.formulas import similarity_score
            for i, (p, c, info) in enumerate(zip(preds, confs, infos)):
                best_cid, best_sim = -1, threshold
                for cid, cent in enumerate(centroids):
                    s = similarity_score(p, cent, 0.7)
                    if s > best_sim: best_sim, best_cid = s, cid
                if best_cid == -1:
                    mc = MicroCluster(len(clusters))
                    mc.add(p, c, info)
                    clusters.append(mc); centroids.append(p.copy())
                else:
                    clusters[best_cid].add(p, c, info)
                    centroids[best_cid] = clusters[best_cid].centroid
            return clusters

        start = time.time()
        # For large n, only run once
        iters = 1 if n > 500 else 10
        for _ in range(iters):
            res_naive = _naive_cluster(preds, confs, infos, 0.60)
        end = time.time()
        naive_time = (end - start) / iters

        print(f"Candidates: {n:>5} | Optimized: {opt_time*1000:7.3f}ms | Naive: {naive_time*1000:7.3f}ms | Speedup: {naive_time/opt_time:6.1f}x")

        # Verify correctness (number of clusters should be identical for greedy approach)
        assert len(res_opt) == len(res_naive), f"Correctness check failed: cluster count mismatch {len(res_opt)} vs {len(res_naive)}"

    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    benchmark_micro_clustering()
