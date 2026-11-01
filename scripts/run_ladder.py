"""Run the experiment ladder at equal compute, 3 seeds per rung (spec 08, section 5)."""
RUNGS = [
    "b0_gru.yaml",
    "b1_transformer.yaml",
    "e0_plain_ensemble.yaml",
    "e1_weights_passes.yaml",
    "e2_regions.yaml",
    "e3_messages.yaml",
    "e4_decorrelation.yaml",
    "e5_ubm_promotion.yaml",
]

if __name__ == "__main__":
    raise SystemExit("TODO(agent): estimate FLOPs per step, match total compute, run each rung "
                     "for seeds 0-2, write results/ladder_results.csv")
