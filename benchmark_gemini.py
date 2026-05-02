import os
import time
import numpy as np
import google.generativeai as genai
from pipeline.pipeline import IECNN
from formulas.formulas import similarity_score
from cognition.reasoning import DeepReasoningLayer

# Gemini Setup
GEMINI_API_KEY = "AIzaSyAjIOaz-4xS-gdTEhp1Dbr4HF3RvsthbQ4"
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel('gemini-flash-latest')

def get_gemini_response(prompt):
    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"

def run_benchmark():
    print("=== IECNN V5 SOTA vs GEMINI PRO BENCHMARK ===")

    # Initialize IECNN (V5 SOTA)
    # Using smaller dots for benchmark speed
    iecnn = IECNN(num_dots=32, persistence_path="global_brain.pkl")
    reasoner = DeepReasoningLayer(iecnn)

    tasks = [
        {
            "name": "Light Reasoning: Synonym Consistency",
            "iecnn_input": "The sun is hot.",
            "iecnn_compare": "The star is warm.",
            "prompt": "Are 'The sun is hot.' and 'The star is warm.' semantically similar? Answer with only similarity score 0 to 1."
        },
        {
            "name": "Deep Reasoning: Causal Shift",
            "text": "I pushed the door open.",
            "token_idx": 2,
            "replacement": "locked",
            "prompt": "How does meaning change from 'I pushed the door open' to 'I locked the door open'?"
        }
    ]

    results = []

    for task in tasks:
        print(f"\nTask: {task['name']}")

        # 1. IECNN Performance
        t0 = time.time()
        if "iecnn_input" in task:
            res = iecnn.run(task['iecnn_input'])
            v1 = res.output
            v2 = iecnn.encode(task['iecnn_compare'])
            sim = similarity_score(v1, v2)
            ie_out = f"Similarity: {sim:.4f}"
        elif "text" in task:
            impact = reasoner.simulate_intervention(task['text'], task['token_idx'], task['replacement'])
            ie_out = f"Causal Impact Shift: {impact:.4f}"

        ie_time = time.time() - t0

        # 2. Gemini Performance
        t0 = time.time()
        gemini_out = get_gemini_response(task['prompt'])
        gemini_time = time.time() - t0

        print(f"IECNN [{ie_time:.2f}s]: {ie_out}")
        print(f"Gemini [{gemini_time:.2f}s]: {gemini_out.strip()}")

        # Behavior Analysis during task
        if task['name'] == "Light Reasoning: Synonym Consistency":
            analyze_behavior(res)

def analyze_behavior(res):
    print("\n--- IECNN Internal Behavior Analysis ---")
    sm = res.summary
    print(f"Convergence Rounds: {sm['rounds']}")
    print(f"Final Dominance: {sm['history'][-1]['dominance']:.4f}")
    print(f"System Energy: {sm['history'][-1]['energy']:.4f}")

    # Dot behavior
    print("Neural Dot Specialization Distribution:")
    types = res.top_cluster.dot_types()
    for t, count in types.items():
        print(f"  - {t}: {count} dots in winning cluster")

    # Entropy vs Stability
    print(f"Final Entropy: {sm['history'][-1].get('entropy', 0.0):.4f}")
    print(f"Temporal Stability: {sm['history'][-1]['stability']:.4f}")

if __name__ == "__main__":
    run_benchmark()
