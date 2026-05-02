import time
import numpy as np
from pipeline.pipeline import IECNN

def benchmark():
    model = IECNN()
    text = "The quick brown fox jumps over the lazy dog."

    # Warmup
    model.run(text, use_c_pipeline=False)
    model.run(text, use_c_pipeline=True)

    n = 100

    t0 = time.time()
    for _ in range(n):
        model.run(text, use_c_pipeline=False)
    t_py = time.time() - t0
    print(f"Python Pipeline: {t_py:.4f}s ({n/t_py:.2f} iterations/s)")

    t0 = time.time()
    for _ in range(n):
        model.run(text, use_c_pipeline=True)
    t_c = time.time() - t0
    print(f"C Pipeline: {t_c:.4f}s ({n/t_c:.2f} iterations/s)")
    print(f"Speedup: {t_py/t_c:.2f}x")

if __name__ == "__main__":
    benchmark()
