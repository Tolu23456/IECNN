# IECNN

**Iterative Emergent Convergent Neural Network**: a next-token prediction architecture built from a population of small learners ("dots") that rethink their answers over several passes and settle on a weighted consensus. Includes **UBM (Universal Basemapping)**, a byte-level tokenizer with model-driven vocabulary growth.

> **Status: design stage.** The full specification and a project skeleton exist. Nothing has been trained yet, and no performance claims are made. See [Status and honesty notes](#status-and-honesty-notes).

---

## The idea in one minute

Instead of one large network (or a handful of routed experts), IECNN uses **many small dots**:

1. Every dot reads the same shared memory of recent context, but through its **own learned query**, so each looks at the context differently.
2. Every dot keeps a tiny recurrent state and outputs a **probability distribution over the next base (token)** plus a **confidence**.
3. Dots are grouped into **regions**, which are stages: early regions work on spelling-level patterns, later regions on meaning-level patterns.
4. Inside a region, dots **rethink** their answer over a few **passes**. The passes share parameters, so it is the same dots thinking again.
5. After each pass, dots get **weights**. Dots that agree with the crowd, are confident, or have earned trust gain influence; the rest fade. This temporary fading is the "pruning". No dot is deleted.
6. Regions compete with each other through region weights.
7. The **weighted consensus** of the surviving dots is the output, and its top choice is the next base.

The central claim to be tested:

> Iterated, pruned consensus among many small specialists, arranged in stages, beats a plain average of the same dots at equal compute, and the disagreement among dots carries useful information (for example about which tokens are hard).

A population of small independent networks is mathematically just a wide network with block-diagonal weights, so all of IECNN's value has to come from the **interaction rule**: weights, passes, regions, and optional messages. The experiment ladder below exists to check that honestly.

## Architecture

```
text -> UBM -> base ids x_1 .. x_T
                   |
        embedding (+ relative position in the keys)
                   v
 +-------------- Region 1 (spelling-level) ------------+
 | memory: last w1 positions                            |
 | N dots: query -> read -> GRU -> (distribution, conf) |
 | passes 1..P (shared parameters, round signal)        |
 | weights -> consensus, per-position summary           |
 +------------------------+-----------------------------+
                          | summary becomes memory for the next region
                          v
 +-------------- Region 2 ... Region R (meaning-level) -+
                          |
        region weights -> final distribution
                          v
                   next base x_{t+1}
```

### Key design decisions

| Decision | Choice |
|---|---|
| Memory | One **shared**, fixed-window memory; dots differ by their learned queries |
| Recurrence | Over **passes (depth)**, not time; a dot's state resets at each position, which allows parallel training |
| Weights | **Label-free**: computed only from dot outputs (agreement with the leave-one-out crowd, confidence, a learned gate) |
| Pooling | Weighted **log-linear** pooling of dot distributions (mixture pooling as an ablation) |
| Pruning | A temporary weight or mask within one token's decision; all dots return for the next token |
| Training vs inference | Soft weights with an annealed temperature in training; hard top-m survivors at inference |
| Regions | Stages with growing memory spans (proposed 32, 64, 128, 256); regions also compete |
| Passes | Share parameters; a learned round signal tells a dot which pass it is on |
| Error feedback | Each dot sees three label-free numbers about how its last guess compared with the crowd |
| Communication | Thin channel first (weights and pooled opinion); small within-region **messages** as an on/off switch |
| Anti-collapse | Per-dot queries, a decorrelation penalty on errors, boosted per-dot loss, dot dropout |

### Why a literal majority vote is not used

A vote over token ids discards probability information, has no gradient, and breaks ties poorly over large vocabularies. Plurality agreement is only logged as a diagnostic.

## UBM: Universal Basemapping

UBM turns text into numbered **bases**: raw bytes at the bottom, merged pieces above them, and promoted frequent words at the top. Every base records the two parts it was built from, so any base expands to bytes and **a new word is never "unknown"**: `catography` becomes the base for `cat` plus bases for the rest.

| ID range | Contents |
|---|---|
| 0 to 255 | Raw bytes |
| 256 to 263 | Special tokens |
| 264 onward | Learned bases, in order of creation (never reused or removed) |

- **v0**: standard frequency-based merging. As described, this is essentially byte-level BPE, a solved baseline.
- **v1** (the research part): **checkpoint promotion driven by the model**. Candidates are short base sequences that region 1 predicts with high agreement, repeatedly. A small batch is promoted at each checkpoint, new embeddings start as the mean of their parts, a short settling phase trains only the new rows, and a rollback test on held-out **bits per character** undoes any promotion that hurts.
- **Fairness check**: tokens-per-word (fertility) for English, Yoruba, Igbo, and Hausa, to see whether including those languages in training and promotion helps text that general tokenizers split into many pieces.

## Experiment ladder

Each rung changes exactly one thing and must beat the previous rung at **equal compute**, over **3 seeds**, by more than 2 standard deviations of seed noise.

| Rung | Adds | Question |
|---|---|---|
| B0 / B1 | Small GRU / small transformer baselines | Reference points |
| E0 | Plain ensemble of dots, one pass, uniform weights | The floor |
| E1 | Learned weights and 3 passes | Do rounds beat a plain average? |
| E2 | Regions as stages and region competition | Does staging help? |
| E3 | Within-region messages | Does extra communication earn its cost? |
| E4 | Decorrelation penalty (and late hard-mode batches) | Do dots specialize more, without losing accuracy? |
| E5 | UBM checkpoint promotion | Does the vocabulary improve bits per character? |

Rules: a rung that fails after a fixed tuning budget is **cut**. If E1 fails, the central iterative claim is in doubt and work stops there.

## Diagnostics

| Diagnostic | Question it answers |
|---|---|
| Per-dot ablation damage and its spread | Do dots specialize? |
| Win map by token type | Do different dots win on different kinds of tokens? |
| Pairwise error correlation | Are dots collapsing into copies? |
| Dispersion versus loss (AUROC) | Does dot disagreement predict hard tokens? |
| Loss by pass | Do extra passes actually help? |
| Region weights by token type | Do early regions handle spelling and later ones meaning? |
| Mode gap | How much is lost moving from soft training to hard inference? |
| Confidence calibration | Does a dot's confidence mean anything? |
| Message influence | Do receivers actually use messages? |

## Repository layout

```
IECNN/  (the repository root)
  README.md
  requirements.txt
  config.py  utils.py
  DECISIONS.md       every choice the spec did not fix
  STATUS.md          running log, newest first
  RUNBOOK.md         exact commands to set up, run, resume
  spec/              full specification (IECNN_FULL_SPEC.md; git-ignored for now)
  configs/           base.yaml, tiny.yaml, one file per experiment rung
  UBM/               tokenizer, merge training, promotion, fertility
  data/              corpus preparation, packing, dataloaders
  models/            baselines, dot, memory, region, consensus, messages, iecnn
  train/             losses, schedules, training loop, evaluation
  diagnostics/       ablation, win map, correlation, dispersion, and more
  scripts/           run_ladder.py, make_paste_back.py
  tests/             one test file per module
  results/  reports/  checkpoints/  data/
```

## Getting started

Requires Python 3.10 or newer. A GPU is helpful but not required; `configs/tiny.yaml` is an overlay for slow computers.

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                             # works today: config, utils, data split, UBM tokenizer suite
```

Training commands (**planned**; they work once the training phases are built):

```bash
python -m train.train --config configs/e0_plain_ensemble.yaml --seed 0
python -m train.train --config configs/e1_weights_passes.yaml --overlay tiny.yaml --seed 0
python scripts/run_ladder.py
```

Config files inherit from each other (`e1` builds on `e0`, `e2` on `e1`, and so on), so each rung changes only what the specification says it changes.

## Building it with an agent

This project is designed to be built by a coding agent working from the specification. The agent prompt, the phase plan, and the reporting format (a short **PASTE-BACK SUMMARY** after every phase) are in `IECNN_AGENT_PROMPT.md`. Most Python files in the skeleton are stubs that raise `NotImplementedError`; stub tests marked `skip` are the checklist of tests to implement.

## Roadmap

| Phase | Goal |
|---|---|
| 0 | Environment and layout |
| 1 | UBM v0 tokenizer with full test suite |
| 2 | Data pipeline, baselines B0 and B1, bits-per-character evaluation |
| 3 | Core IECNN: dots, memory, region, passes, weights (E0, E1) |
| 4 | Regions as stages and cross-region competition (E2) |
| 5 | Full loss suite and schedules |
| 6 | Messages (E3) |
| 7 | Diagnostics suite |
| 8 | The full experiment ladder (E0 to E4) |
| 9 | UBM checkpoint promotion and fairness test (E5) |
| 10 | Final report |

Deferred beyond v0.1: long-range memory (compressive or strided rows, state-space summary, retrieval), dispersion-based early halting, tree-structured vocabularies, scaling studies.

## Status and honesty notes

- **No results yet.** Every claim above is a hypothesis to test, not a finding.
- **Novelty is unverified.** Searches during design found close relatives of individual pieces but not the exact combination. Absence in a few searches is not proof of novelty; do a proper literature review before claiming anything in a paper. Pointers (verify before citing):
  - routing by agreement in capsule networks;
  - mixture-of-experts specialization losses;
  - adaptive computation and token-wise early exit (ACT, Universal Transformer, pondering language models);
  - dynamic chunking and byte-level models (H-Net, Byte Latent Transformer);
  - vocabulary expansion with embeddings initialized from constituent tokens;
  - hierarchical and tree-structured vocabularies, hierarchical softmax.
- **UBM v0 is BPE.** Only the v1 promotion rule and its experiments are potentially new.
- **Expect small scale.** The first prototype is about 5M parameters on character-level text. Beating a matched transformer is a stretch goal, not a requirement.
- **The interaction rule is the whole bet.** If E1 (rounds and weights) does not beat a plain ensemble at equal compute, the iterative claim fails, and the honest report says so.

## Known risks

| Risk | Mitigation |
|---|---|
| Dots collapse into copies | Per-dot queries, decorrelation penalty, collapse diagnostics from day one |
| Soft-to-hard gap | Temperature annealing, late hard-mode training batches, mode-gap reporting |
| Compute cost of N x R x P | Tiny dots, grouped matrix operations, equal-compute comparisons |
| Vocabulary changes destabilize training | Checkpoint-only small batches, settling phase, rollback |
| Interaction does not help | The ladder finds out cheaply; cut what does not earn its cost |

## Glossary

| Term | Meaning |
|---|---|
| Base | A vocabulary item produced by UBM |
| Dot | One small learner with its own query, state, and output projection |
| Pass | One round of rethinking inside a region (also called a layer) |
| Region | A group of dots forming one stage |
| Memory | Fixed window of recent positions that dots read |
| Weight | How much a dot's opinion counts for the current token |
| Pruning | Driving a dot's weight toward zero for one token; not deletion |
| Temperature | Controls how sharp the weights are; annealed over training |
| Consensus | Weighted log-linear pool of dot distributions |
| bpc | Bits per character; lower is better; the fair metric across vocabularies |
| Ablation | Switching something off to measure how much it mattered |

## License and citation

License: to be decided. Citation: to be added once there is a paper or release.
