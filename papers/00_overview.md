# IECNN: Specification Overview (v0.1)

**Iterative Emergent Convergent Neural Network**
Status: design draft for a first prototype. Nothing here has been implemented or tested yet.

---

## 1. What IECNN is

IECNN is a next-token prediction architecture built from a **population of small learners called dots**.

- Every dot reads the same shared memory of recent context, but asks it a different question (its own learned query).
- Every dot holds a tiny recurrent state and produces a probability distribution over the next base (token), plus a confidence.
- Dots are grouped into **regions**. Regions are stages: early regions work on spelling-level patterns, later regions on meaning-level patterns.
- Inside a region, dots **rethink** their answer over a few **passes** (the "layers"). The passes share parameters, so the same dots think again rather than being replaced by new ones.
- After each pass, dots receive **weights**. Dots that agree with the crowd, are confident, or have earned trust gain influence; the rest fade. This fading is called **pruning**, but no dot is ever deleted.
- Regions also compete, through region weights.
- The weighted consensus of the surviving dots becomes the output distribution, and its top choice is the next base.

The vocabulary comes from **UBM (Universal Basemapping)**, a byte-level tokenizer whose bases can be promoted over time.

## 2. The core claim being tested

> Iterated, pruned consensus among many small specialists, with stage-wise regions, beats a plain average of the same dots at equal compute, and the disagreement among dots carries useful information (for example about which tokens are hard).

This is a claim to **test**, not assume. A population of small independent networks is mathematically a wide network with block-diagonal weights, so the entire value of IECNN has to come from the **interaction rule** (weights, rounds, regions, messages). Every rung of the experiment ladder (section 9) exists to check that interaction rule earns its cost.

## 3. Glossary

| Term | Meaning |
|---|---|
| Base | A vocabulary item produced by UBM (a byte, a merged piece, or a promoted word). |
| Dot | One small learner: own query, own recurrent state, own output projection. |
| Pass | One round of rethinking inside a region. Passes share parameters. Also called a layer. |
| Region | A group of N dots that form one stage of processing. Regions are stacked. |
| Memory | A fixed-size window table with one vector per recent position. |
| Query | A learned vector owned by a dot, used to read the memory. |
| Pooled opinion | The region's weighted summary of dot states from the previous pass. |
| Weight | A number in [0, 1] per dot per pass; weights in a region sum to 1. |
| Pruning | Driving a dot's weight toward zero for the current token. Temporary, not deletion. |
| Temperature (tau) | Controls how sharp the weights are. Annealed over training and within a token. |
| Consensus | The weighted log-linear pool of dot distributions. |
| Message | A small vector a dot can send to nearby dots in the same region (optional). |
| Round signal | A learned vector added per pass so a dot knows which pass it is on. |

## 4. Architecture at a glance

```
text -> UBM -> base ids x_1 .. x_T
                   |
        embedding + position
                   v
 +-------------- Region 1 (spelling-level) --------------+
 | memory M1 = embeddings of last w1 positions           |
 | N dots: query -> read -> GRU -> (distribution, conf)  |
 | passes 1..P (shared params, round signal)             |
 | weights w -> consensus b1, region summary h1_t        |
 +------------------------+------------------------------+
                          | h1 per position becomes memory M2
                          v
 +-------------- Region 2 ... Region R (meaning-level) --+
                          |
        region weights omega -> final distribution y_t
                          v
                   next base x_{t+1}
```

## 5. Data flow for one position t (prediction of x_{t+1})

1. Look up the memory window for the current region (rows for positions t-w+1 .. t).
2. Each dot builds its query and reads the memory, getting a context vector.
3. Each dot updates its state from: context vector, pooled opinion from the previous pass, an error feature, the round signal, and (if enabled) messages.
4. Each dot outputs a distribution and a confidence.
5. The region computes weights from label-free signals, forms the weighted consensus, and sharpens the temperature.
6. Repeat steps 2 to 5 for P passes.
7. The region emits its final consensus and a summary vector per position (used as the next region's memory).
8. After the last region, region weights combine the regions' consensus distributions into the final output.

Training runs all positions in parallel (teacher forcing). Inference runs one position at a time with a sliding-window cache.

## 6. Default hyperparameters for the first prototype

| Name | Symbol | Default | Notes |
|---|---|---|---|
| Vocabulary | V | 264 to start | 256 bytes + 8 specials; merges added by UBM later |
| Model width | d | 128 | embedding and memory dimension |
| Query / key dim | d_q | 32 | |
| Value dim | d_v | 64 | |
| Dot state dim | d_s | 128 | |
| Dots per region | N | 8 | |
| Regions | R | 4 | 32 dots total |
| Passes per region | P | 3 | shared parameters across passes |
| Region attention spans | w_r | 32, 64, 128, 256 | proposed, not decided |
| Memory window cap | W | 256 | decided: fixed window |
| Message dim | d_m | 16 | only when messages are enabled |
| Initial temperature | tau_0 | 1.0 annealed to 0.2 over training | |
| In-token sharpening | kappa | 0.7 per pass | |
| Trust decay | rho | 0.5 | |
| Hard-mode survivors | m | 4 of 8 per region | inference only |

Rough size: each dot is about 150k parameters (its own GRU, projection, query), so 32 dots is about 5M parameters. Small enough to train on a single GPU.

## 7. Design decisions log

Decided by you in discussion:

| ID | Decision |
|---|---|
| D1 | Dots read a **shared memory**, not private windows. |
| D2 | Diversity comes from each dot's own learned query, plus a penalty on correlated errors. |
| D3 | Dots are **not regenerated** per token. The same dots come back; pruning is a temporary mask within one token's decision. |
| D4 | Regions are **stages**: spelling-level first, meaning-level later. |
| D5 | Dots from different stages **also compete**, through region-level weights. |
| D6 | Passes (layers) **share parameters**: each dot rethinks its decision. |
| D7 | A dot sees **how its own previous guess scored** on each pass. |
| D8 | **Fixed-window** memory first; long-range memory later. |
| D9 | Build the thin communication channel first (weights + pooled opinion); **messages** come later as an on/off switch. |
| D10 | UBM's main purpose is **handling new words**; layers and regions drive vocabulary promotions at checkpoints. |

Proposed by me, open to change: region spans, the exact scoring formula, log-linear pooling, hard top-m at inference, small settings in the table above.

## 8. Clarifications that tighten the earlier discussion

These fix gaps that were glossed over in conversation. They are binding for v0.1.

- **C1. Recurrence is over depth, not time.** A dot's state resets at the start of each position and evolves only across the P passes. All temporal context comes from the memory window. This is what allows parallel training over positions.
- **C2. Weights must be label-free.** At inference there is no true next token, so the weighting rule can only use signals computed from dot outputs. The idea of rewarding "dots that were right when others were wrong" can therefore live only in **training losses**, never in the inference-time weighting rule. See file 04 and file 06.
- **C3. The error feature is measured against the consensus, not the truth.** The thing a dot sees about its previous guess is how far it was from the leave-one-out crowd, never the true label and never the crowd's full distribution (which would encourage copying).
- **C4. No dot is deleted.** All N dots exist for every token. Pruning is a mask.
- **C5. Hard voting is inference-only.** Majority voting has no gradient, so training uses soft weights and a log-linear pool. Plurality agreement is logged only as a diagnostic.

## 9. Experiment ladder

Each rung must beat the previous one at equal compute (details in file 08).

| Rung | Adds | Question |
|---|---|---|
| E0 | Plain ensemble of dots, one pass, uniform weights | Baseline. |
| E1 | Weights and multiple passes | Do rounds beat a plain average? |
| E2 | Regions as stages and region competition | Does staging help? |
| E3 | Messages between nearby dots | Does extra communication earn its cost? |
| E4 | Decorrelation penalty | Do dots specialize more? |
| E5 | UBM promotions at checkpoints | Does the vocabulary improve bits-per-character? |

## 10. Non-goals for v0.1

- Beating transformers at scale. The goal is a working, learnable prototype and an honest comparison.
- Long-range memory, retrieval, or tools.
- Multilingual evaluation beyond a small fairness test (file 07).
- Any claim of novelty before prior-art checks are done properly for a paper.

## 11. Prior art pointers (verify before citing)

From searches done during design. These are pointers, not verified citations:

- Routing by agreement in capsule networks (iterative votes between units).
- Mixture-of-experts specialization losses (divergence, orthogonality, variance).
- Token-wise adaptive computation (ACT, Universal Transformer, pondering and early-exit language models).
- Dynamic chunking and byte-level modeling (H-Net, Byte Latent Transformer).
- Vocabulary expansion with embeddings initialized from constituent tokens.
- Hierarchical and tree-structured vocabularies, hierarchical softmax.

The specific combination (shrinking-population consensus plus agreement-driven vocabulary promotion) was not found in the searches, but absence in a few searches is not proof of novelty.

## 12. Top risks

1. **Collapse**: dots become near copies; IECNN degrades to an expensive average. Mitigation: per-dot queries, correlation penalty, diagnostics from day one.
2. **Soft-to-hard gap**: model trained soft behaves worse when pruned hard. Mitigation: temperature annealing, occasional hard-mode training batches.
3. **Compute cost**: N x R x P multiplies work. Mitigation: tiny dots, grouped matmuls, equal-FLOP baselines.
4. **Vocabulary instability**: promotions scramble learned dots. Mitigation: checkpoint-only batched promotions, settling phase, rollback.
5. **Interaction does not help**: the honest outcome if rungs E1 to E3 fail. The ladder lets you find out cheaply.

## 13. Document map

| File | Contents |
|---|---|
| `00_overview.md` | This file. |
| `01_dot.md` | A single dot: parameters, equations, shapes, failure modes. |
| `02_memory.md` | The fixed-window shared memory and how each region builds it. |
| `03_regions_and_passes.md` | Regions, passes, cross-region competition. |
| `04_consensus_and_pruning.md` | Weights, pooling, temperature, soft vs hard. |
| `05_messages.md` | Optional within-region messages. |
| `06_training.md` | Losses, schedules, configuration, monitoring. |
| `07_ubm.md` | Universal Basemapping tokenizer and checkpoint promotion. |
| `08_evaluation.md` | Metrics, baselines, ablations, kill criteria. |
