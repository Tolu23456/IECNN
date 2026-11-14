# Status log

Newest entry at the top.

## 2026-11-14 (night) — Pilot wave 2 landed: vocab 2048 wins big, E2 fails at pilot scale

All three 2h-wall pilots completed success and were re-scored corpus-verified
(val sha256 vs each run's own split_info; best-step tau; FULL val = Oz
207,823 chars):

| config | best full-val bpc | step | tau0 | verdict |
|---|---|---|---|---|
| **E1 + vocab_v2 @2048** | **2.59667** | 780 | 0.891 | **still descending at cap** |
| E1 @512 seed 0 (ref) | 3.20294 | 540 | 0.954 | still descending at cap |
| E1 @512 seed 1 | 3.49401 | 360 | 0.985 | still descending at cap |
| E2 regions @ batch 8 | 3.54218 | 180 | 1.000 | loss flat ~12.0 — staging does not help at pilot scale |
| (E1 @512 no-lambdas, from earlier) | 3.47421 | 360 | 0.985 | aux lambdas ≈ 0.27 bpc |

- **Vocab is the biggest lever found so far**: 2048-token vocab cuts E1's
  full-val bpc by **0.61** (2.59667 vs 3.20294) at the same 2h wall class —
  bigger than the aux-lambda effect (0.27) and much bigger than any rung
  delta measured so far. Fertility curve: 512 → 2.00, 2048 → 2.86,
  4096 → 3.20 chars/token (2048 is the sweet spot on the curve).
- **Seed variance at EQUAL step is tiny**: seed 0 @360 in-loop = 3.50881,
  seed 1 @360 = 3.50789 (Δ 0.001). The apparent 3.21-vs-3.49 gap between
  seeds was purely step count (540 vs 360 at the wall cap), not noise.
  Equal-step comparisons are the honest basis for ladder criteria.
- **E2 verdict (pilot)**: 3.54218 with train loss flat at ~12.0 (vs E1's
  ~5.2) — 4-region staging at batch_size 8 learned almost nothing in 2h.
  Fails the spec's E2 criterion (must beat E1 by 2 seed SDs). Confound to
  record: batch 8 vs E1's 64, and ~4x per-token compute meant ~360 steps vs
  E1's 540. Next step if pursued: larger batch via memory work (gradient
  checkpointing / per-pass frees), not more wall time.
- **E0 warm-protocol (patience-only) chunk 1**: 6000 steps, bpc 2.29172
  @6240 and still descending at the 320-min chunk cap — auto-chained into
  chunk 2 (run 43537908132). E0's plateau is forming below 2.3.
- **Playground demo from the vocab2 checkpoint** ("Dorothy followed the
  yellow brick road", 160 tokens @ T 0.8/top-k 40/seed 0):
  results/generation/gen-20261114-005810.json. Still incoherent — 2.6 bpc
  is far above the ~1.5–2.0 needed for readable text. Coherence is a
  plateau-run outcome, not a pilot outcome. Honest caveat recorded:
  same-seed generations from different prompts stayed near-identical —
  consistent with unigram-dominated early training.

## 2026-11-14 (early morning) — Playground: send prompts, get model output

- **Tolu asked for "a space for sending prompts and receiving outputs from
  the model."** Shipped as the Explorer's **Playground page** (subagent,
  commit 3d794c6) + runner-side backend (agent, commits 771fcf1/e33e2ea):
  `scripts/generate.py` + `.github/workflows/generate.yml` — downloads a
  checkpoint artifact, verifies the corpus (same guard as eval/viz), encodes
  the prompt with the run's own UBM vocabulary (NFC + merges by birth order),
  samples a continuation from `final_log_probs` (temperature 0.8, top-k 40,
  seed 0; all configurable), commits `results/generation/gen-<ts>.json`.
- Server endpoints: `POST /api/generate` (dispatch; ambient credential from
  git config, token only ever in the Authorization header — never logged/
  stored/returned), `GET /api/generate/runs`, `GET /api/generate/result`
  (run's head_sha commit → the generation file; honest pending/failed
  states), `GET /api/generations` (local history). Tests 29/29; GitHub layer
  mocked so no test can dispatch a real run; CI green.
- **Bugs hunted and fixed (Tolu's rule):** workflow YAML broke on an inline
  `python -c` (heredoc now); hardcoded checkpoint dir would have generated
  from the wrong run (now derived from the artifact's contents); generate.py
  used `out["log_b"]` which only exists inside `region_outputs[r]` — top-level
  forward returns `final_log_probs` — KeyError killed the first two e2e runs
  (caught by the subagent's endpoint verification, fixed, re-run green).
- **E2E verified twice** (runs 43526108853, 43526557467): real 120-token
  generations committed. Quality is gibberish at E1@step-540 (undertrained,
  3.20 bpc) — expected; the page says so. Re-run the Playground against the
  plateau checkpoints when they land. Noted: two different prompts with the
  same seed produced near-identical text — consistent with an undertrained
  model dominated by unigram frequencies; re-check with the plateau
  checkpoint.

## 2026-11-13 (late evening) — E2 memory lesson, vocab pilots, consensus audit

- **E2 pilot killed 3× by runner RAM (~7 GB usable, not the nominal 16 GB).**
  Bisected with a 5-step probe: full-dim E2 died ~85 s in every time; a
  tiny-overlay probe **completed** → the code path is correct, memory is the
  cause. E2's four regions materialize windowed memory over spans
  32+64+128+256 = 480 slots; retained-for-backward tensors at batch 64/seq 256
  exceed the runner. **Agent bug found during the bisect:** the first "fix"
  wrote `batch:` — a key train.py ignores (it reads `training.batch_size`,
  default 16, base.yaml sets 64) — so the "batch 16" attempt silently ran at
  64. Real fix: `batch_size: 8` in `configs/e2_regions_pilot.yaml`. E2 pilot
  (run 43520535363) has now survived far past the old kill window.
- **Bigger vocab per Tolu's question ("can't we increase the vocab from
  256…?"): yes.** 512 of the vocab are 256 mandatory byte-fallback tokens +
  248 merges (fertility 2.00 chars/token train / 2.09 val — the "words in
  chunks" observation is by design at this scale, not a bug). Trained
  **vocab_v2 @ 2048** (`UBM/vocab_v2.json`): fertility **2.856 train / 2.975
  val** (+43% compression, 19.3M → 13.5M train tokens). Dispatched
  **vocab_v3 @ 4096** for the fertility curve. Memory catch: the per-pass
  per-dot log-prob block `[B,T,N,V]` quadruples with V → the E1+vocab2 pilot
  (`configs/e1_weights_passes_vocab2.yaml`) runs at `batch_size: 16` to keep
  the E1@512/batch-64 memory class.
- **Warm-protocol (patience-only, constant LR) runs in flight:** E0 warm
  (43494318309), E1 warm (43508978701), E1 seed-1 pilot (43515821271),
  E2 pilot batch_size 8 (43520535363), E1 vocab2 pilot (43521585038),
  vocab v3 (43521589596). Also repaired 3 eqc CSVs that had stacked
  stash-pop conflict markers committed (dedupe by step, sorted;
  commit 8295859).
- **Consensus audit (Tolu's viz observation, quantified on the 47-position
  export, E1@step 540):** pooled top-1 **3/47 (6%)**; oracle best-dot top-1
  **6/47 (13%)**, oracle top-5 **11/47 (23%)**; 3 positions had a dot's
  TOP-1 correct yet outvoted (one muted to weight 5.9e-7; one held
  0.13–0.24 and lost to a sharper wrong dot); 8/47 had it in some dot's
  top-5. Small sample / early checkpoint — a trend, not a verdict. Fix
  underway on two fronts: viz export now samples **256 positions**
  (n_tokens 48 → 256), and the Explorer gains a consensus-audit panel
  (outvoted badges + oracle stats).
- **Explorer follow-ups shipped by the viz subagent:** readable hex escapes
  for byte-fallback tokens (no more � walls), visible token boundaries,
  fertility explainer on the Vocab page (commit 4c943ec; tests green).

## 2026-11-13 (evening) — Reload anomaly SOLVED: it was the wrong corpus, not the reload

- **Root cause (proven by probe + log forensics):** the `eval`/`viz` re-score
  workflows rebuilt the corpus with a **base rung config**
  (`e1_weights_passes.yaml`) whose default `data.corpus` is **v0** (tiny
  Shakespeare). Every "re-scored" number yesterday — val 3.777, train 3.738,
  passes 4.226/4.112/3.777 — was computed on the **v0 Shakespeare val
  (55,769 chars)**, not corpus v1's Oz book (207,823 chars). The job log even
  printed it: `train 1003856 / val 55769 / test 55769`. The reload path,
  dropout mode, seq_len (128 vs 256), and fast_math are **all innocent**.
- **One-process factorial probes** (`scripts/diagnose_reload.py`, run
  43501937855, results in
  `results/diagnosis/e1_weights_passes_pilot_seed0_reload.json`) on the
  training run's own `best_state`, with the corpus **verified against the
  run's `split_info.json`** (val sha16 b94503186e683737):
  - A1 (train-mode, seq 128, head-100k) = **3.21328** [3.60097, 3.49581,
    3.21328] — the in-loop number reproduced EXACTLY with reloaded weights.
  - A2 (eval-mode) = A1; A3 (seq 256) = A1 → dropout & window irrelevant
    (E1's forward has no stochastic eval path: `use_messages: false`,
    `dot_dropout: 0.0`).
  - T1 (train slice 100k) = **3.28431** [3.73774, 3.63683, 3.28431] — matches
    in-loop train exactly.
  - **Full-val Oz = 3.20294** [3.5845, 3.48395, 3.20294] — identical to 5
    decimals for live (A6), fresh rebuild + `load_state_dict` (B2), and
    torch.save/torch.load round trip (C2). B1/C1 max param diffs = 0.0;
    forward max|Δ| live vs round-trip = 0.0 (bit-identical).
  - D1–D4 (fast_math True/False, live + reloaded) all = 3.21328.
  - **Causality probe on trained weights** (B=4, T=256, fast path, tokens
    randomized after position 200): log-probs at positions ≤ 200
    **bit-identical** on live AND reloaded models (max|Δ| = 0.0).
  - Oz chars 100k–207.8k tail scores 3.19468 (even easier than the head).
- **Retraction:** the earlier "full-val re-score" numbers 3.777/3.967/4.451
  and "train 3.738" are **wrong-corpus artifacts**; E0/B0 pilot numbers are
  unaffected (computed in-process on the right corpus).
- **Guard baked into every workflow** (`eval.yml`, `viz.yml`, `diagnose.yml`):
  corpus_config defaults to the `*_pilot.yaml`; the artifact is downloaded
  FIRST and its `data/split_info.json` (carried by training artifacts) is used
  to verify the rebuilt val's sha256 before any scoring; a val < 150k chars
  fails loudly (diagnose script raises too). Verified locally: local val
  (207,823 chars, b94503186e683737) == the E1 training run's recorded split.
- **train.py hygiene:** `model.eval()` around in-loop evals, train mode
  restored after — numerically neutral today (A1==A2), correct for any rung
  whose forward later gains stochastic eval behavior.

## 2026-11-13 (later) — IECNN Explorer: viz server for everything

- **`server/server.py` + `server/static/`**: a zero-dependency (stdlib-only,
  no torch) read-only dashboard, `python3 server/server.py` →
  http://127.0.0.1:8765. Pages: Dashboard (ladder + runs + re-score reports),
  How-it-works (11 clickable stages with shapes/formulas, numbers pulled live
  from `configs/e1_weights_passes.yaml` so they can't drift), Training runs
  (overlay val/train bpc, per-pass, tau + weight entropy, per-component grad
  norms, matched-step table computed from CSVs), Weights & dots (checkpoint
  inspector), Vocab explorer, Corpus & evals. Chart.js vendored locally
  (MIT, offline-safe). Malformed trailing CSV rows (the old 5-column bug) are
  filtered server-side so old runs plot honestly.
- **`scripts/export_viz.py` + `viz.yml`**: the inspector's data source.
  Real weights/dots live only in torch checkpoints (CI artifacts), so the
  export runs ON THE RUNNER: download artifact → vocab fingerprint check →
  one sample forward → commit `results/viz/<run>.json` with a 2-D embedding
  projection (SVD), per-position × per-pass × per-dot weights, gamma, top-5
  predictions per dot, pooled answers, weight entropy, learned
  alpha/beta scalars. No torch on the dev box, ever.
- **`tests/test_server.py`**: server endpoints, CSV parsing (incl. the
  malformed-row filter and a clean 404), and static files — stdlib-only,
  runs in CI.
- Nothing dispatched yet: viz.yml exports await Tolu picking a checkpoint
  artifact (the E1 pilots are the interesting ones).

## 2026-11-13 — Tolu's six items implemented; trainer defects found by pilots fixed; CI green

- **Item 1 — fresh-clone pip fix**: `requirements.txt` no longer pins torch
  (the `+cpu` build only exists on PyTorch's own index, so plain
  `pip install -r requirements.txt` failed from a fresh clone). Torch moved to
  `requirements-torch.txt` (CPU index pinned); `train.yml`/`tests.yml` install
  both files; README's Getting Started updated. The `vocab.yml` workflow needs
  no torch at all.
- **Item 2 — vocab name + checksum everywhere**: `UBM.tokenizer.fingerprint()`
  (sha256 of the vocab file's bytes). Every `checkpoint.pt` and `summary.json`
  now records `vocab_file` + `vocab_sha256`; resume **raises loudly** on a
  mismatch (exact error names both hashes and the file to point back at);
  `scripts/eval_checkpoint.py` exercises the same guard before evaluating.
- **Item 3 — one identical LR schedule for the next wave**:
  `train/schedules.py::lr_at_constant` (linear warmup then FLAT peak) with
  `training.lr_schedule: "constant"` (default stays cosine). The warm-start E0
  config uses constant 1e-3 after warmup 500.
- **Item 4 — per-component gradient norms + zero-init test**:
  `grad_norms_by_component()` logs pre-clip L2 norms per bucket —
  emb / regions / messages / chain / gate / other — as CSV columns
  `gn_total,gn_emb,gn_regions,gn_chain,gn_gate,gn_messages,gn_other`.
  `tests/test_zero_init.py`: (a) E1 with the learned-weight pathway at zero
  init produces EXACTLY uniform dot-weights and its output equals E0's at
  step 0 (forward-only, no training); (b) the grad buckets sum to the model's
  total gradient norm. NOTE: first CI run failed because the score-path
  params (`alpha`/`beta` scalars + `c_w/c_b/g_w/g_b` heads) are shared names
  across E0/E1 and got copied instead of zeroed; fixed by explicitly zeroing
  the whole score pathway in both builds — CI **green** (tests run 40).
- **Three trainer defects the pilots exposed, fixed**:
  1. `steps_trained_this_run`/`steps_final` reported the SCHEDULE ceiling
     (e.g. 2880) instead of the ACTUAL last step under early stop — now
     tracked from the loop (`last_step`); summaries add `steps_target`.
  2. The final CSV row was written 5-columns-wide into a 12-column file
     (malformed row) — now full width at the real last step.
  3. The best-point full-val re-score used the FINAL tau (0.2) instead of the
     tau annealed to the BEST step — E1 pilots' best points read ~4.45/4.69
     instead of ~3.21/3.49 on full val, purely from the tau change. The
     re-score now uses `temperature_at(best_step, ...)`.
- **Pilot CSVs recovered to `results/` and committed** (runs 72/73's results-bot
  push had 500'd; artifacts were intact) — commit 525669b.
- **`eval.yml` workflow + `scripts/eval_checkpoint.py`**: runner-side
  checkpoint re-scoring (download artifact by name → fingerprint check →
  re-score best point on FULL validation at the best-step tau → per-pass bpc +
  weight entropy + hard mode → commit JSON to `results/`). Dispatched for both
  E1 pilots to get honest best_full_val_bpc numbers.

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
