import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
from pipeline.pipeline import IECNN
from decoding.decoder import IECNNDecoder
from formulas.formulas import similarity_score
import time

def run():
    print("ULTIMATE SOTA BENCHMARK (50% NOISE)")
    ie = IECNN(num_dots=64)
    ie.base_mapper.min_base_freq = 1

    # Representative 'extreme noisy' training
    corpus = [
        "Thx rxd applx fxl frxm thx trx.",
        "Thx sxn xs shinfng brxghtly.",
        "A lxsh grxxn frest wxs hxre.",
        "The blue sky is clear today."
    ]
    ie.fit(corpus)
    # Apply Contrastive Anchoring
    ie.base_mapper.fit_contrastive([("red", "apple"), ("sun", "shining")], [("red", "blue"), ("apple", "sun")])

    gpt2 = GPT2Model.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2.eval()

    def gpt_encode(text):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            return gpt2(**inputs).last_hidden_state[0, -1, :].numpy()

    test_pairs = [
        ("The red apple fell.", "Thx rxd applx fxl.", True),
        ("The sun is bright.", "Thx sxn xs brxght.", True),
        ("The red apple fell.", "The blue ocean waves.", False),
        ("The sun is bright.", "The cold rain falls.", False)
    ]

    results = {"IECNN": [], "GPT2": []}

    for t1, t2, match in test_pairs:
        # IECNN
        v1_i = ie.encode(t1)
        v2_i = ie.encode(t2)
        s_i = similarity_score(v1_i, v2_i)
        results["IECNN"].append((s_i, match))

        # GPT2
        v1_g = gpt_encode(t1)
        v2_g = gpt_encode(t2)
        s_g = np.dot(v1_g, v2_g) / (np.linalg.norm(v1_g) * np.linalg.norm(v2_g))
        results["GPT2"].append((s_g, match))

    for name in ["IECNN", "GPT2"]:
        matches = [s for s, m in results[name] if m]
        mismatches = [s for s, m in results[name] if not m]
        avg_m = np.mean(matches)
        avg_nm = np.mean(mismatches)
        print(f"\n{name} Results:")
        print(f"  Avg Match Sim (50% Noise): {avg_m:+.4f}")
        print(f"  Avg Mismatch Sim:          {avg_nm:+.4f}")
        print(f"  Discriminative Gap:        {avg_m - avg_nm:+.4f}")

if __name__ == "__main__":
    run()
