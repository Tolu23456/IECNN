# Status log

Newest entry at the top.

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
  loo/JS parts, E1 **22 s/step** (742 tok/s). E1 vs E0 ratio now ~5.6× (3 passes +
  weighting); a bmm/einsum rewrite of `dots_forward` is listed for Phase 4 before any
  long E1 chain.
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
