# Spec vs build audit — 2026-11-14

Source: `spec/IECNN_FULL_SPEC.md` (files 00–08). Status marks: ✅ built + tested, 🟡 partial, ❌ not built, ⬜ stub (file exists but empty).

## 1. What we've built (inventory)

### Core model (spec 00–04)
| Spec item | Status | Where |
|---|---|---|
| Hyperparameters table (d=128, d_q=32, d_v=64, d_s=128, N=8, R=4, P=3, d_m=16, hard m=4) | ✅ exact | `configs/base.yaml` |
| Dot params (q_k, optional U_k, s0, GRU, P_k, c_k, g_k) + round signal + batched-GEMM implementation | ✅ | `models/dot.py` (fast path pinned by equivalence tests) |
| Error feature = 3 scalars vs leave-one-out crowd (C3) | ✅ | `models/dot.py`, `models/consensus.py` |
| Shared memory: region-1 = bare embeddings (no pos), region r = LayerNorm(Proj(h)+m) residual, rel-distance embedding | ✅ | `models/memory.py` |
| Banded causal window, spans [32,64,128,256] as config | ✅ (used only at R≥4; E2 pilot ran 1 region) | `configs/base.yaml` |
| Passes with shared params + round embedding, tau_p = tau0·kappa^(p−1) | ✅ | `models/region.py` |
| Consensus: label-free scores (α·agree + β·conf + gate), trust ρ, weight floor 0.1/N with final-phase zeroing | ✅ | `models/consensus.py`, `train/schedules.py` |
| Log-linear pooling + mixture-pooling switch | ✅ | `models/consensus.py`, E0 uses mixture |
| Cross-region competition D5 (entropy + gate, tau_region floor 0.5) | ✅ | `models/iecnn.py` |
| Hard top-m inference (m=4) + mode gap measured every eval | ✅ | `train/evaluate.py`, mode_gap in every eval JSON |
| Messages: ring s=2, d_m=16, one-pass delay, dropout 0.2, off ⇒ bit-identical | ✅ | `models/messages.py` + 4 tests |
| Constraints C1–C5, decisions D1–D10 | ✅ all honored | code + tests |
| Halting (`halting: dispersion`) | ❌ knob accepted, only `none` implemented | `models/iecnn.py` |
| Inference sliding key/value cache (spec 02 §5) | ❌ (eval/generate are batch-parallel over blocks) | — |

### Training (spec 06)
| Spec item | Status |
|---|---|
| All 6 loss terms (main, region, boosted-dot, decorrelation+ramp, confidence, weight-entropy) | ✅ `train/losses.py` |
| Schedules (AdamW warmup+cosine, tau cosine, λ_c ramp, hard batches w/ straight-through, dot dropout) | ✅ all as knobs |
| 3-seed protocol | 🟡 matrix exists in train.yml (B0/E0/E1-eqc have 3 seeds); headline E1-vocab runs are seed-0 only |
| FLOPs per position measured | 🟡 crude 6·params·tokens estimate logged; `train/profile_step.py` gives per-op timing |
| Dot revival | ❌ (optional per spec) |
| Monitoring (weight entropy, per-pass bpc, mode gap) | ✅ in CSV + eval JSONs |

### UBM (spec 07)
| Spec item | Status |
|---|---|
| v0 byte-level BPE (pretokenizer, NFC, digit groups, max len 32, determinism, ID stability, tombstone rollback) | ✅ `UBM/` + `tests/test_ubm.py`; vocab_v1–v4 trained, compression JSONs committed |
| bpc fair metric everywhere | ✅ all evals |
| v1 checkpoint promotion (candidate log → rank → create → mean-of-parts init → settle → rollback test) | ✅ built 2026-11-14: `UBM/promotion.py` (CandidateLog = agreement × confidence × log(1+count); promote() with mean-of-parts init, new-rows-only settle, batch rollback → tombstone+restore) + `scripts/promote_vocab.py` driver + `promotion.yml` workflow; first real run dispatched on the vocab3 warmstart checkpoint |
| Fertility / fairness test (Yoruba, Igbo, Hausa) | 🟡 `UBM/fertility.py` exists; no language samples collected, never run |

### Evaluation (spec 08)
| Spec item | Status |
|---|---|
| bpc evaluator (full-val, per-pass, hard/soft) | ✅ corpus-fingerprinted eval JSONs |
| B0 GRU baseline, 3 seeds | ✅ trained (`results/train_b0_gru_seed{0,1,2}.csv`) |
| B1 transformer baseline | 🟡 implemented + tested (`models/baselines.py`) but never trained |
| B2 = E0 plain ensemble | ✅ trained, multi-seed, equal-compute pair (`*_eqc` configs, seeds 0–2) |
| Ladder rungs trained | E0 ✅, E1 ✅ (deep: vocab2/3/4, warmstart, λ ablation), E2 pilot only (180 steps), E3/E4/E5 configs ready but untrained, E6/E6b beyond-spec extensions |
| Diagnostics suite §6 (ablation damage + Gini, win map, error correlation, dispersion AUROC, calibration, region-weights-by-token-type) | ✅ built 2026-11-14: all 8 modules implemented (`diagnostics/*.py` + `common.py` collector pinned to evaluate_bpc's accounting) with `scripts/diagnose_all.py` + `diagnostics.yml` workflow; first real run dispatched on the vocab3 warmstart checkpoint |
| Statistical protocol (mean±std over 3 seeds) | 🟡 infra exists, applied only to B0/E0/E1-eqc |
| Reporting template | 🟡 eval JSONs carry bpc/mode-gap/vocab-sha; no params/FLOPs fields |

### Beyond spec (built on our own initiative)
- Speed stack opts 1–7 (bf16 fast paths → fused compiled consensus → threads-4 → copy-free head → shared-attention cache) with **paired same-runner A/B benching** as the evidence standard (`scripts/bench.py`, all `results/bench_ab_*.json`).
- E6 consensus-v2 + verifier head with six-metric acceptance audit (`models/verifier.py`, `train/evaluate.py`).
- Warmstart training, chunked CI chains with cancel-guard, artifact recovery.
- Explorer dashboard (10 routes) + screenshots workflow; playground generation with fixed top-k/nucleus sampling (`scripts/generate.py`).
- Reload-diagnosis workflow with corpus fingerprinting.

## 2. Gaps, ranked by value

**P0 — cheap, high value**
1. ~~Implement the 8 diagnostic modules~~ **DONE 2026-11-14** (`diagnose-all` workflow dispatched on the vocab3 warmstart checkpoint).
2. ~~Wire UBM v1 promotion~~ **DONE 2026-11-14** (`promotion.yml` driver workflow dispatched on the vocab3 warmstart checkpoint).
3. **Train B1 transformer** at matched params/FLOPs — without it there is no honest "how good is this really" anchor.

**P1 — completes the ladder**
4. E2 full rung: 4 regions with real spans, 3 seeds, plus the span ablation ([32,64,128,256] vs all-256).
5. E3 messages and E4 decorrelation rungs (configs already exist).
6. Seeds 1–2 on the headline E1-vocab3 config + mean±std reporting.
7. FLOPs-per-position field in the reporting template (measured via profile_step).

**P2 — optional / later**
8. Inference sliding cache (faster, lower-memory generation).
9. Remaining ablation-matrix axes: shared vs unshared passes, static vs dynamic query, N∈{4,8,16}, P∈{1,2,3,4}, weight floor on/off, message topologies.
10. Halting — only if the dispersion AUROC diagnostic clears ~0.7.
11. Fairness corpus collection (Yoruba/Igbo/Hausa samples) + fertility run.
12. Dot revival (spec marks it optional; revisit only if ablation shows dead dots).

## 3. Standing quality note
The best model (vocab3 warmstart, 2.27183 bpc full-val @1680) has seen ~0.18 epochs of the 39.2 MB corpus — coherence is a *data-quantity* constraint, not an architecture bug. The two paths on the table (long chains ~1.5–2.5 days CI vs 30-min domain-slice fine-tune) both remain pending Tolu's choice.
