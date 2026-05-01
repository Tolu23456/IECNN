import time
import numpy as np
from pipeline.pipeline import IECNN
import os

def noise_test(model, text, noise_level=0.1):
    v_orig = model.run(text, use_c_pipeline=True).output
    chars = list(text)
    for _ in range(max(1, int(len(chars) * noise_level))):
        i = np.random.randint(len(chars))
        chars[i] = chr(np.random.randint(32, 126))
    noisy_text = "".join(chars)
    v_noisy = model.run(noisy_text, use_c_pipeline=True).output

    from formulas.formulas import similarity_score
    sim = similarity_score(v_orig, v_noisy, alpha=model.alpha)
    return sim, noisy_text

def main():
    corpus_path = "corpus_300k.txt"
    persistence_path = "v6_brain.pkl"
    model = IECNN(persistence_path=persistence_path)

    print("--- IECNN V6 SOTA Massive Training ---")

    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f if line.strip()][:500]

        print(f"Training on {len(lines)} lines...")
        model.train_pass(lines, use_c_pipeline=True, verbose=True)
        model.save_brain(persistence_path)

    effs = model.dot_memory.all_effectivenesses()
    max_eff = np.max(effs) if len(effs) > 0 else 0.0
    print(f"\nFinal MaxEff: {max_eff:.4f}")

    # Stability Test
    text = "The emergence of intelligence in gradient-free systems is inevitable."
    sim, noisy = noise_test(model, text, noise_level=0.2)
    print(f"\nStability Test (20% noise):")
    print(f"Original: {text}")
    print(f"Noisy:    {noisy}")
    print(f"Similarity: {sim:.4f}")

    # Multilingual Check
    ar_text = "أنا أحبك"
    zh_text = "我爱你"
    v_ar = model.run(ar_text, use_c_pipeline=True).output
    v_zh = model.run(zh_text, use_c_pipeline=True).output
    from formulas.formulas import similarity_score
    m_sim = similarity_score(v_ar, v_zh, alpha=model.alpha)
    print(f"\nCross-lingual Semantic Alignment (AR vs ZH): {m_sim:.4f}")

if __name__ == "__main__":
    main()
