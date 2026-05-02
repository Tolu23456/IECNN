from pipeline.pipeline import IECNN
import numpy as np

def main():
    model = IECNN()
    dots = model._ensure_dots()
    ids = [d.dot_id for d in dots]
    for d_id in ids: model.dot_memory._ensure_id(d_id)

    # Force Win
    for _ in range(20):
        model.dot_memory.record(ids[0], np.zeros(256), True)

    # Force Loss
    for _ in range(20):
        model.dot_memory.record(ids[1], np.zeros(256), False)

    effs = model.dot_memory.all_effectivenesses(ids)
    print(f"Manual MaxEff: {np.max(effs):.4f}")
    print(f"Manual MinEff: {np.min(effs):.4f}")

if __name__ == "__main__":
    main()
