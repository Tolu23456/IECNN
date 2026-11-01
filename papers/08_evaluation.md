# 08. Evaluation

How to tell whether IECNN works, whether each mechanism earns its place, and when to cut something.

Status: draft v0.1. Loss is the **gate**; specialization diagnostics are the **explanation**.

---

## 1. Principles

1. **Equal compute.** Every comparison is at matched training FLOPs (or matched wall-clock), and parameters and FLOPs are reported for every run.
2. **Several seeds.** At least 3 seeds per configuration. Report mean and standard deviation.
3. **Ablate one thing at a time.** Each ladder rung changes exactly one mechanism.
4. **Report failures.** A rung that does not help is a result, not an embarrassment.
5. **Fair metric across vocabularies.** Use bits-per-character (file 07, section 7).

## 2. Datasets (start small)

| Stage | Data | Purpose |
|---|---|---|
| Debug | A few MB of text | Overfit tests, unit-level sanity |
| Main prototype | A small character-level corpus (tens to hundreds of MB) | Rung comparisons |
| Later | A small story or web-text corpus | Check behaviour beyond characters |
| Fairness | Small held-out English, Yoruba, Igbo, Hausa samples | Tokenizer fertility (file 07) |

Hold out validation text that is never used for tuning, plus a separate test set used only for final reports. Check licenses before using any corpus.

## 3. Headline metrics

| Metric | Definition |
|---|---|
| Validation bpc | Total NLL (nats) divided by (ln 2 times characters) on held-out text |
| Mode gap | Hard-mode bpc minus soft-mode bpc |
| Parameters | Trainable parameter count |
| FLOPs per position | Estimated from file 06, section 9, and measured |
| Throughput | Tokens per second, training and inference |

## 4. Baselines

| ID | Baseline | Purpose |
|---|---|---|
| B0 | Small GRU or LSTM, matched parameters | Weak sanity baseline |
| B1 | Small transformer, matched parameters and FLOPs | The real comparison; the goal is competitive, not necessarily winning |
| B2 | Plain ensemble of N x R dots, one pass, uniform weights, no regions (rung E0) | Isolates the value of interaction |

## 5. Experiment ladder with pass criteria

Each rung is compared with the one before it at equal compute.

| Rung | Change | Comparison | Pass criterion |
|---|---|---|---|
| E0 | Plain ensemble | vs B0 and B1 | Sets the floor |
| E1 | Weights plus P = 3 passes | vs E0 | Validation bpc improves by more than 2 seed standard deviations |
| E2 | Regions as stages and region competition | vs E1 | Same criterion, **and** region weights differ by token type |
| E3 | Messages | vs E2 | Same criterion, **and** message influence is clearly above zero |
| E4 | Decorrelation penalty | vs E3 | Lower error correlation **and** no loss in bpc |
| E5 | UBM checkpoint promotions | vs fixed vocabulary | bpc improves after rollback tests; fertility does not get worse |

### Kill criteria

- If a rung fails its criterion after a fixed tuning budget (for example 10 runs of hyperparameter search), **cut it** and carry forward the previous rung.
- If E1 fails (rounds do not beat a plain ensemble), the central iterative claim fails; stop and rethink the interaction rule before adding anything else.

## 6. Diagnostics for the core claims

### 6.1 Do dots specialize?

| Check | Method | Healthy sign |
|---|---|---|
| Ablation damage | Mask one dot at a time, measure bpc increase | Wide spread: some dots matter a lot, some little |
| Spread statistic | Gini coefficient of ablation damage | Clearly above that of the E0 ensemble |
| Win map | For each token, which dot had the highest weight; group by token type | Different dots win on different token types |
| Pairwise error correlation | Weighted correlation of dot errors (file 06) | Falls during training, stays low |
| Query similarity | Cosine between learned queries | Not converging to one direction |

### 6.2 Do rounds help?

| Check | Method | Healthy sign |
|---|---|---|
| Loss by pass | bpc of the consensus at pass 1, 2, 3 | Monotonic improvement |
| P=1 vs P=3 | Train both | Gap larger than seed noise |
| Weight entropy by pass | Mean entropy of weights | Decreasing with pass index |

### 6.3 Is disagreement useful?

| Check | Method | Healthy sign |
|---|---|---|
| Dispersion vs token loss | AUROC of dot dispersion at pass 1 predicting whether the consensus is wrong | Clearly above 0.5 (for example above 0.7 is encouraging) |
| Calibration | Compare confidence with accuracy | Reliability curve near diagonal |
| Halting simulation | Stop early where dispersion is low; measure loss and savings | Meaningful savings with little loss |

This is the cheapest first experiment on a trained model and needs no training changes.

### 6.4 Do regions specialize by stage?

| Check | Method | Healthy sign |
|---|---|---|
| Region weight by token type | Mean region weight for capitalized words, rare words, within-word positions, punctuation, common phrases | Early regions up-weighted on spelling-hard tokens, later regions on meaning-hard |
| Single-region performance | bpc of each region's own consensus | Later regions better on long-context tokens |
| Span ablation | Compare proposed spans with all regions at 256 | The proposed spans are not worse |

### 6.5 Soft vs hard

Report both bpc values and the gap on every evaluation (file 04, section 5).

## 7. Tokenizer evaluation (UBM)

| Metric | Meaning |
|---|---|
| Round-trip pass rate | Must be 100% |
| Compression | Characters per token |
| Fertility | Tokens per word, by language |
| Fertility ratio vs English | For Yoruba, Igbo, Hausa |
| bpc before and after each promotion batch | The rollback test |
| Fraction of promotions rolled back | Health of the promotion rule |

## 8. Statistical protocol

1. For each configuration, run the same set of seeds.
2. Report mean and standard deviation of validation bpc.
3. A change counts as real if the gap exceeds 2 standard deviations of the seed noise **and** is seen on at least 2 of 3 seeds individually.
4. Do not tune on the test set. Tune on validation only, run the test set once at the end.
5. Record config, code commit, vocabulary version, and seed with every number.

## 9. Reporting template

For every experiment record:

```
experiment id:
config file / commit:
vocabulary version:
seeds:
parameters:
FLOPs per position (est. / measured):
validation bpc (mean ± std):
hard-mode bpc and mode gap:
ablation damage spread (Gini):
mean pairwise error correlation:
dispersion-vs-loss AUROC:
region weight by token type (table):
notes / failures:
decision (keep / cut / retune):
```

## 10. Ablation matrix

| Ablation | Question |
|---|---|
| Uniform weights vs learned weights | Is weighting useful? |
| Log-linear vs mixture pooling | Which pooling is better? |
| Shared vs unshared pass parameters | Does sharing cost accuracy? |
| Static vs dynamic query | Do state-dependent queries help? |
| Error feature on vs off | Does self-correction help? |
| Ring vs full-region message topology | Cost vs benefit |
| Per-region spans vs all 256 | Do short early spans help spelling? |
| N = 4, 8, 16 dots per region | Scaling of the population |
| P = 1, 2, 3, 4 passes | Diminishing returns |
| Correlation penalty strength | Specialization vs accuracy |
| Weight floor on/off | Dead-dot behaviour |

## 11. What success looks like

A modest, honest success for v0.1 is:

1. IECNN with rounds and regions beats the plain-ensemble baseline at equal compute (E1 and E2 pass).
2. Dots measurably specialize.
3. Dot disagreement predicts hard tokens.
4. The mode gap is small.
5. UBM v1 is not worse than UBM v0 on bits-per-character, and shows a fertility improvement on the under-served languages if they are included in training.

Beating a matched transformer is a stretch goal, not a requirement for the prototype.

## 12. Threats to validity

| Threat | Mitigation |
|---|---|
| Hyperparameter tuning favors IECNN over baselines | Give baselines the same tuning budget |
| Seed noise larger than effects | More seeds; only claim effects beyond noise |
| Compute mismatch hides cost | Report FLOPs and wall-clock for all runs |
| Small-scale results do not transfer | State scale limits clearly in any paper |
| Test-set reuse | One final test run per experiment family |
| Metric leakage across vocabularies | Use bpc, never per-token loss |
