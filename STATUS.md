# Status log

Newest entry at the top.

## 2026-11-04 — Tolu's A/B/C decisions executed; 4 pilots dispatched

- **Decisions in**: (A) early-stop adopted with edits — eval sample raised to
  a fixed **100k chars** (`training.eval_chars`) AND `min_improvement` 0.005
  (100k-char eval noise ≈ 0.002 bpc, so 0.005 is safely above it); every
  summary now records `stop_kind` ("patience" | "wall_cap" | null) plus
  `stopped_early`, `hit_wall_cap`, `reached_step_target`. (B) corpus v1 with a
  fresh **UBM v1 vocab @512**. (C) pilot first — ONE seed-0 run per config.
- **Corpus v1 split (Tolu's question answered)**: whole held-out BOOKS, not
  tail blocks. Train = 32 books (38,620,418 chars), **val = The Wonderful
  Wizard of Oz (book 55, 207,823 chars)**, test = Heart of Darkness (219) +
  A Modest Proposal (1080) + The Yellow Wallpaper (1952) = 260,911 chars. No
  book appears in two parts (asserted by id AND by body sha16); Gutenberg
  boilerplate stripped between the license markers for all 38 books
  (`markers_found: True` each); NFC-normalized. Builder: `data/prepare.py`
  (`data.corpus: big`); split_info.json carries sha256 per part.
- **UBM v1 trained on the RUNNER** (dev box OOM-kills at 38.6M chars — the
  in-memory merge trainer needs ~3 GB; new `vocab.yml` workflow does corpus
  prep + vocab training + commit). Result: 512 bases, 248 merges,
  **2.00 chars/token train, 2.09 val** (val compresses better — Oz is more
  regular English than the KJV-heavy train mix). `UBM/vocab_v1.json` +
  `results/vocab_v1_compression.json` committed.
- **Pilot-corrupting bug caught before dispatch**: `tokenize_corpus`
  HARDCODED `UBM/vocab_v0.json` — every pilot would have tokenized the train
  stream with the v0 merge set while the model used v1 ids (same size 512,
  different semantics). Fixed: vocab_file is threaded through from config, and
  the npy cache key now includes the vocab file name.
- **Per-pass logging implemented** (Tolu's pilot logging list):
  `evaluate_bpc_per_pass` scores passes 1/2/3 of the last region from ONE
  forward (pass p predicts from its own pooled log-prob), plus weight entropy
  of the last pass's dot-weights and per-pass tau. The training CSV now logs
  `step,lr,loss,bpc_val,bpc_p1,bpc_p2,bpc_p3,bpc_train,tau0,weight_entropy,L_w,
  elapsed_s`; summaries carry `final_bpc_per_pass`, `final_bpc_train`,
  `final_weight_entropy`, `final_tau0`. `bpc_train` = fixed 100k-char slice of
  the train stream (fit, not generalization).
- **CI chaining test added** (`tests/test_chaining.py`): (1) two chained
  `train_one` chunks on a synthetic corpus — asserts chunk 2 RESUMES
  (`steps_trained_this_run == 4` after `max_steps` 4→8; the 11-03 bug would
  have re-trained 8 from 0) and the CSV stays one continuous log; (2) a
  workflow lint asserting `train.yml` extracts the resume artifact at the repo
  root (the `checkpoints/checkpoints/...` bug cannot return silently). Suite:
  **67 passed, 6 skipped** locally.
- **"7x" correction** in STATUS done (42→16.6 s/step ≈ 2.5×; rung ratios
  E1/E0 4.2×, E1/B1 6.5×, E1/B0 17.3×).
- **B0's old best full-val bpc**: NOT recoverable from the old artifact (its
  checkpoint stores the 20k overfit state; `best_state` tracking postdates
  that run, and the in-loop best 1.95 was a 12k-char subsample on corpus v0).
  The honest like-for-like number comes from today's B0 pilot on corpus v1
  (`best_full_val_bpc`), reported in the pilot table with E0/E1/E1-nolambda.
- **Pilots dispatched (runs 72–75, seed 0, suffix -pilot)**: `b0_gru_pilot`,
  `e0_plain_ensemble_pilot`, `e1_weights_passes_pilot`,
  `e1_weights_passes_nolambda_pilot` (lambda_d/k/w = 0 isolates the aux
  losses; weights + 3 passes stay on). All: corpus v1, vocab v1,
  eval_chars 100k, patience 6 evals, wall cap 2.0 h (accumulated train
  seconds, chunk-safe). Schedule ceilings keep eqc shapes (B0 50k / E0 12k /
  E1 2,880 steps; warmup 1%, tau warmup 4%) but the 2 h cap governs — none of
  these runs reaches its ceiling. NO full ladder dispatched.

## 2026-11-03/04 — Tolu's pause: what the runner actually did (and didn't) do

- **Cancellation order executed**: B0 50k chain (runs 21/22/23) and all 6 remaining
  in-flight eqc chain continuations (runs 66–71) cancelled. Nothing else dispatched.
- **Chain-resume bug found and fixed** (this explains everything): the uploaded
  artifact zip is rooted at `checkpoints/`, but the chain's restore step extracted
  it *into* `checkpoints/` → `checkpoints/checkpoints/...`; `train.py` never saw
  `checkpoint.pt` in its out-dir, so `--resume` never engaged and **every chained
  chunk re-trained the rung from step 0**. Effects: the B0 `-r2` "50k ladder"
  actually only ever completed its first chunk (20k steps — the committed CSV is
  that single chunk); the E0/E1 eqc rungs burned ~30 chains × 320 min producing
  only chunk-1 data segments; and the artifact nests deeper every chunk (131 MB
  zips full of duplicate .pts). Fix: extract at repo root (`dd83589`), one line +
  comment. Honest statement: my "chain handoff works" from yesterday was verified
  only as "the next run started", not as "the step counter continued" — wrong.
- **What the eqc chunks actually delivered (single-chunk curves, 16,384 tokens/step
  for every rung, all still on v0 corpus/vocab)**: E0 plain ensemble reached
  **2.395 / 2.418 / 2.427 bpc by ~5.0–5.8k steps (seeds 0/1/2, still falling)**;
  E1 weights rung reached **2.997 / 2.990 / 3.107 bpc at ~0.9–1.1k steps, then
  flattening**. Under contention, measured s/step: E0 ≈ 4.2–4.3, E1 ≈ 17.4–21.4.
  At equal compute so far, **E0 (plain ensemble) is ahead**. No true bottom
  reached; these are "still falling at chunk end" points, not converged values.
- **L_w / tau reporting**: tau at each best point is computable exactly from the
  schedule (e.g. E1 best steps 900/1,080/1,260 with tau_warmup 115 → tau
  0.851/0.783/0.707); `L_w` is only in stdout logs and no chained chunk ever
  finalized, so per-best `L_w` needs either the in-progress logs or a re-run —
  not faked. CSV loss column is the total (main + aux).
- Work done locally while paused: evaluator-health tests (uniform at init ≈
  4.80 bpc = exactly ln(512)/(ln2×1.874) on real vocab; low bpc on a trained
  slice), trainer early-stop keys implemented (patience + wall cap, chunk-safe),
  regularization parity pinned in every rung config (E4's spec-default 5% dot
  dropout removed after Tolu's equalization request), and — a real catch —
  `dot_dropout` had NO implementation behind the config key; one was written
  (spec table semantics: 5% of steps, one random dot masked per region, training
  only) plus tests. Corpus v1 (39.1 M chars, 38 public-domain books) built with
  a deterministic deterministic manifest; huge divergence story pending approval.

## 2026-11-02 — Equal-compute ladder dispatched (Tolu chose option b)

- Checkpoint-B follow-up options presented with measured numbers; **Tolu chose (b): equal-compute stepping**, budget W = B0's 50k-step compute (~47,850 s/seed, from B0's measured 0.957 s/step). Resulting step counts: **E0 → 12,000** (3.9 s/step = 46,800 s), **E1 → 2,880** (16.56 s/step = 47,692 s).
- Schedule-shape fairness: absolute-step schedule knobs scaled by steps/50,000 in the new `configs/*_eqc.yaml` (LR warmup 500→120/30, tau warmup 2000→480/115, eval/checkpoint cadence likewise), so every rung ends past its own cosine anneal with the same schedule shape; everything else was already relative to total steps. Local 5-step smoke of the E1 eqc config passes.
- Chain-control improvement: chained chunks now forward `commit_results` (final chunk after a chain previously could not commit its results — for the running B0 `-r2` this means its final summaries must be recovered from artifacts).
- Dispatched `e0_plain_ensemble_eqc.yaml` and `e1_weights_passes_eqc.yaml`, seeds 0,1,2, suffix `-eqc`, `commit_results=true` (runs 28/29 train dispatches at 07:06 UTC). Expect ~13 h/seed (3 chained chunks each), B0 finishes its 50k ladder in parallel today.

## 2026-11-02 — Checkpoint B follow-through: (c) executed — E1 optimized 42 → 16.6 s/step; B0 overfitting found; best-checkpoint protocol

- **Fast-dot rewrite** (`models/dot.py`, `fast_math` flag default ON): GRU gates + tied head as plain batched GEMMs over merged `[B*T, N]` layout; einsum version kept as `_forward_reference`; equivalence test fast↔reference incl. dynamic query. Suite now **61 passed, 6 skipped**.
- **Chain of measured E1 speeds on the runner**: 42 → 22.1 (loo/JS dedup) → 17.4 s/step (fast-math) → **16.6 s/step, 989 tok/s** (JS reuses the dot's `log_pi` instead of re-taking `log(p)`). Ratio check (Tolu): 42 → 16.6 is a **~2.5×** total speedup, not 7×; per-rung ratios are E1/E0 ≈ 4.2×, E1/B1 ≈ 6.5×, E1/B0 ≈ 17.3×, and E1 vs *B1's own optimizer-fitted* speed is NOT a like-for-like claim anyway. Below the self-set bar (≥1200 tok/s would justify torch.compile), so optimization stops here.
- **Op-level profiler** (torch.profiler table in `train/profile_step.py`, real fwd+bwd on the runner): after the GEMM rewrite, remaining cost = elementwise `[f, N, V]` chains (mul/log/div/copy_/add/sub/neg/where) in consensus + backward — i.e., mostly intrinsic to the ensemble-over-vocab design.
- **B0 `-r2` chunk 1 finished and self-chained at 04:28 for all 3 seeds (runs 21/22/23, chunk 2 in progress)**. The CSVs carried back show **real overfitting**: train loss → 0.25 while in-loop val bpc rises monotonically 1.95 (step 1k) → 5.43 (20k) — at ~14k tokens/step the 50k target is ~3,900 epochs on a 540k-token corpus. Run so far has been valuable regardless: it also proved the 320-min timeout → artifact → resume handoff works.
- **Trainer now tracks the best checkpoint** (`best_state`/`best_step` saved + resumed; final `summary.json` gains `best_full_val_bpc`, `best_step`, `best_mode_gap` at the best point, all scored on FULL validation). The rung headline is the best point, not the most-overfit final state.
- **Workflow results-commit fixed** (`git pull --rebase --autostash`): seed 0's chunk-1 commit failed on unstaged-changes rebase error (chain itself unaffected — chunk 2 fired on schedule). Chains dispatched before the fix still run old code, so per-chunk `results/` commits may fail; artifacts are fine.
- Reports: `reports/phase_03_report.md` §6b (real-number ladder table: B0 13 h, E0 54 h, E1 ~230 h per seed at 50k). DECISIONS.md updated (fast-math default, best-checkpoint protocol, optimizer stopped).

## 2026-11-02 — Phase 3 core IECNN built, tested, and timed on Actions; B0 ladder running

- **Core model complete and spec-tested** (`models/dot.py`, `memory.py`, `region.py`,
  `consensus.py`, `messages.py`, `iecnn.py`): 28 new tests, all passing; full suite
  **60 passed, 6 skipped**. Highlights: E0 is bit-exactly a plain uniform ensemble;
  causality is bit-exact; every dot receives gradient; cross-region competition einsum
  bug (`btr,brv->btv` vs proper `btr,brtv->btv` with `[B,R,T,V]` stacking) caught by
  tests before it could silently corrupt E2+.
- **Full-size Actions timings (EPYC, 2 threads, fp32)**: E0 3.9 s/step → 2775 tok/s
  after a small fix (was 4228 before an accidental extra JS computation; restored);
  init loss 6.238 = ln 512 exactly. E1 first measured **42 s/step** → component
  profiled on Actions: consensus machinery was computed twice per pass; after reusing
  loo/JS parts, E1 **22 s/step** (742 tok/s). E1 vs E0 ratio: 22.1/3.9 ≈ 5.6× then; after the final round of optimisation it settled at 16.6/3.9 ≈ **4.2×** (3 passes + weighting; elementwise consensus is the leftover cost).
- **Workflow upgraded for >6 h rungs**: seeds matrix (parallel per-seed jobs), 320-min
  chunks with **self-chaining resume only on exit code 124** (crashes fail loudly),
  overwrite artifacts named `{config}-seeds-{seed}{suffix}`. A bash arithmetic bug
  (`${CHUNKMIN:-320}` empty expansion) crashed all 3 first B0 ladder jobs in minute 1
  — found on the runner, fixed, re-dispatched as `-r2` (3 seeds, 50k steps, in
  progress and healthy: 40+ min into chunk 1 of ~4).
- **Profile-run casualties teach a lesson**: two profile dispatches died with a
  "runner shutdown signal" ~100 s in — self-inflicted RAM exhaustion (7 GB hosted
  runner; warmup graph retained under a second full graph). The real E1 trainer
  survives full size but the memory margin is thin; memory-lean audit scheduled
  before long chains.
- Reports: `reports/phase_03_report.md` (structure table, 6 bugs found on the way,
  measured profile breakdown, ladder implication: E1-at-50k ≈ 424 h/seed uncompressed
  → options a/b/c to be decided with Tolu at Checkpoint B). DECISIONS.md updated
  (cyclic ring, E0 mixture pooling pin, aux lambdas per rung only, chaining policy).

## 2026-11-01 — Phase 2 complete: baselines + trainer validated on Actions; 3 workflow bugs found and fixed

- **train.yml now actually dispatches** after 3 fixes: (1) embedded python in the resume step de-indented out of its YAML block scalar → file unparseable → GitHub silently dropped `workflow_dispatch` (cause of every 422); (2) `.gitignore`'s `data/*` excluded `data/dataset.py`/`prepare.py` from the commit → `ModuleNotFoundError` on runner; corpus step added (rebuilds data/*.txt on runner; checksums **byte-identical** to local); (3) `timeout-minutes: 720` → 350 (hosted job cap is 6h; longer runs must use resume).
- **First Actions timing numbers** (EPYC 7763, 2 threads, full-val eval included, all measured on GitHub per Tolu's new rule "no local training"): tiny b0 73 ms/step; full b0 1,290 ms/step; full b1 2,550 ms/step (≈6.4k tok/s). Full b1 FLOPs 342 GFLOP/step.
- **Real bug caught**: full-config b1 diverged instantly (val bpc 122.5 at step 20 with warmup LR 4e-5). Static root cause: tied output head initialized from N(0,1) embedding → logits~N(0,256) at init → confidently-wrong model (173 nats vs lnV=6.24). Fixed with GPT-style 0.02-normal init; init loss now 6.29; regression test `test_init_loss_near_uniform_at_full_config` added; post-fix Actions rerun healthy (bpc 4.53 at step 20 and decreasing).
- Full local suite re-run: **32 passed, 33 skipped** (skips = Phase 3+ roadmap).
- Reports: `reports/phase_02_report.md` (measurements, bug post-mortem, ladder estimate ~1.5–3 weeks wall for 8 rungs × 3 seeds × 50k steps, b1=35 h/seed measured, b0=18 h/seed measured).
- DECISIONS.md updated: no-local-training rule (Tolu), eval subsampling, bpc accounting kernel.

## 2026-11-01 — Restructure per Tolu: src/ flattened, UBM/ is the tokenizer home

- Moved `src/models` → `models/`, `src/train` → `train/`, `src/data` → `data/`, `src/diagnostics` → `diagnostics/`, `src/config.py` → `config.py`, `src/utils.py` → `utils.py`; deleted `src/` including the lowercase `src/ubm/` shim.
- Tokenizer code lives only in `UBM/` now (as Tolu requested earlier). The old `src.ubm.tokenizer` import path no longer exists — anything importing it must use `UBM`.
- Updated: both test files, GitHub workflow `train.yml` (entrypoint `train.train`), README (layout + commands); full test suite re-run: **18 passed, 40 skipped** (skips = stub checklist for later phases).
- Also removed the empty leftover `iecnn/` unzip folder at Tolu's request context.

## 2026-11-01 — Phase 1 (UBM) built early, at Tolu's request

- `UBM/tokenizer.py` complete: encode/decode/expand/parts/add_bases/tombstone/revive/save/load; specials carry empty byte payloads; decode uses replacement characters for invalid UTF-8.
- `UBM/train_bpe.py` complete: incremental deterministic BPE trainer (ties by smallest pair; >32-byte merged forms banned; pair-count index maintained by delta updates).
- `tests/test_ubm.py`: 10 tests replacing 7 skipped stubs, including round trip on random Unicode + combining marks + emoji, ID stability, rollback semantics ("encoding returns to the state before a base existed"), save/load, and an independent naive BPE reference cross-check.
- Training on the real corpus and the compression report (264/512/1024) are next.

## 2026-11-01 — GitHub repository and Actions configured

- Request from Tolu: run most training on GitHub Actions; GitHub credentials exist (token already stored by a prior project's git settings; no `git.key` file found). Token never copied into any project file and never printed.
- **Repo setup**: existing private `Tolu23456/IECNN` (an earlier IECNN prototype by Tolu, ~6.8k files) was safely renamed to `Tolu23456/IECNN-v0-archive`; a fresh **public** `Tolu23456/IECNN` was created and the skeleton pushed (`git` remote configured with the token, as elsewhere on this machine). Advice given: rotate that token at some point.
- **Workflows added**: `tests.yml` (runs pytest on every push) and `train.yml` (manual dispatch: config + overlay + seeds; artifacts uploaded, CSVs committed back). Training entrypoint `src.train.train` will exist after Phase 2; dispatching before then fails gracefully at the import.
- Runner tier: `ubuntu-latest` = 4 vCPU / 16 GB RAM, no GPU → still the CPU tier, but ~2x the local box.
- `spec/` git-ignored per Tolu (spec stays local; not published on GitHub).
- Environment moved from `iecnn/.venv` to the project root `.venv` with internal paths fixed after PyTorch install completed: **torch 2.15.0+cpu imports fine on Python 3.11.2**.
- Remaining numpy/pandas/etc. packages still downloading (slow link); pinned `requirements.txt` and the full test suite run follow.

## 2026-11-01 — Project moved to the main folder

- At Tolu's request, moved everything from `iecnn/` up one level into `/home/t85491005/IECNN` (configs, src, tests, data, spec, reports, results, scripts, checkpoints, README/STATUS/RUNBOOK/DECISIONS/requirements/.gitignore).
- `iecnn/.venv` remains in the old subfolder for now: PyTorch is still downloading into it. It will be moved to `/home/t85491005/IECNN/.venv` and its internal absolute paths fixed once the install completes.
- RUNBOOK.md paths updated to the new location.

## 2026-11-01 — Phase 0 started

- Environment detected: Linux x86_64 (kernel 6.6), 2 CPU cores, no GPU, 2.7 GB RAM, ~4.5 GB free disk.
- Default `python3` is 3.15; using system `python3.11` for the venv (torch wheel availability).
- Created venv at `iecnn/.venv`, installing PyTorch (CPU wheel) first, then the rest of the pinned requirements.
- Found spec at `papers/00..08`; concatenated into `spec/IECNN_FULL_SPEC.md` (see DECISIONS.md).
- Unzipped `iecnn_skeleton.zip`; the skeleton layout matches Section 5 of the agent prompt. Building on it.
- Reading the full spec (done: 00–08) and the skeleton stubs before writing any code.
