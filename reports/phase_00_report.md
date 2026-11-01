# Phase 0 Report — Environment and layout

**Date:** 2026-11-01. **Status:** done, with one environment deviation (CPU-only tier).

## What this means for IECNN (plain English)

The tools are installed and working, the data is prepared and verified clean, the
repository is on GitHub with automatic test running, and the UBM tokenizer (Phase
1's product) is already built and tested at Tolu's request. All training will run
on GitHub's free computers (4 cores, 16 GB), which are about twice as fast as the
local machine. Everything is CPU-only — no GPU for free plans.

## Claim markers

- Everything below is **measured** on this machine unless marked otherwise.
- Numbers with `±` are means over the stated number of runs; single measurements
  are reported as exact values.

## Environment (measured)

| Item | Value |
|---|---|
| OS | Linux x86_64, kernel 6.6.119 |
| Python | 3.11.2 (system), venv at `.venv/` using it; default `python3` is 3.15 (avoided — no torch wheels) |
| CPU | 2 cores |
| RAM | 2.7 GiB total |
| Disk free at start | ~4.5 GB |
| GPU | none (no NVIDIA device, no CUDA) |
| PyTorch | 2.15.0+cpu — smoke test: matrix multiply ran; numpy interop warning appeared only before numpy was installed, resolved after |
| Packages | pinned in `requirements.txt`: numpy 2.4.6, pandas 3.1.0, regex 2026.9.29, pytest 9.2.0, pyyaml 6.0.3, matplotlib 3.11.2, scikit-learn 1.9.1, torch 2.15.0+cpu, tqdm 4.70.1 |
| Compute tier | **CPU-only** → `tiny` config overlay for local runs; GitHub Actions runners (4 vCPU/16 GB, `ubuntu-latest`) carry the main training at Tolu's request |

## Data (measured)

- Corpus: Tiny Shakespeare (public domain), 1,115,394 chars, 40,000 lines, 65 unique characters.
- **Cleaning audit**: 0 changes from NFC normalization; only newline/tab as control characters; no format/unassigned characters; no mojibake. Cleaning applied: NFC, CRLF→LF, strip ASCII EOF char. Nothing else.
- **Split** (contiguous blocks, 90/5/5): train 1,003,856 / val 55,769 / test 55,769 chars.
- Checksums (sha256, first 16 hex): train `9e2b074a547cbfd3`, val `c6666eacd4d7f5c1`, test `be...` see `data/split_info.json`.
- A second corpus (Project Gutenberg) failed to download (connection cut mid-file); removed the partial file. Internet works, so this is a source-side limitation, not a setup problem. Not a blocker.

## Repository / infrastructure (measured)

- Skeleton unzipped and used as the base; layout restructured at Tolu's request: `src/` flattened to the project root; `UBM/` is the single tokenizer package (lowercase `src/ubm/` deleted).
- Git: initialized, `main` branch. Remote `Tolu23456/IECNN` (public, new). The **old** private repo of the same name (Tolu's earlier prototype, ~6.8k files) was renamed `Tolu23456/IECNN-v0-archive` instead of being overwritten.
- `spec/` is git-ignored per Tolu; a full combined spec is built locally from `papers/00`–`08`.
- GitHub Actions: `tests.yml` (pytest on every push), `train.yml` (manual dispatch: config + overlay + seeds; artifacts + CSVs committed back). Runner: 4 vCPU / 16 GB, no GPU.

## Tests (measured)

- **18 passed, 40 skipped** (skips are the skeleton's stub checklist for phases 2–7, not failures).
- Of the skips, **10 former UBM skips were implemented and now pass** as part of the early Phase 1 build (see below).

## Phase 1 (UBM) — built early at Tolu's request (measured)

- Files: `UBM/tokenizer.py` (complete), `UBM/train_bpe.py` (complete).
- Trainer: deterministic byte-level BPE over pre-tokenizer chunks; ties by smallest pair; merged bases longer than 32 bytes excluded; incremental pair counting (trained the full corpus in **20.4 s** on 2 cores).
- **Compression on validation** (chars per token; round trip verified):
| Vocabulary | Tokens for 55,769 chars | chars/token |
|---|---|---|
| 264 (bytes only) | 55,769 | 1.000 |
| 512 | 29,754 | 1.874 |
| 1024 | 24,708 | 2.257 |
First merges learned: `' t'`, `'he'`, `' a'`, `'ou'`, `' s'`, `' m'`, `'in'`, `' w'`, `'re'`, `'ha'`, `' the'`, `'nd'` (full list of first 20 in `results/ubm_first20_merges.txt`).
- Vocabulary saved to `UBM/vocab_v0.json`.

## Deviations from the agent prompt (also in DECISIONS.md)

1. Spec arrived as `papers/00`–`08`, not `spec/IECNN_FULL_SPEC.md`; contents combined unchanged.
2. Python 3.11 instead of default 3.15 (torch wheel availability).
3. Layout flattened (`src/` removed) and tokenizer moved to `UBM/` at Tolu's request — skeleton folder names/signatures for data/models kept.
4. A second corpus was not obtained (download failure); not required by spec.
5. Phase 1 substantially completed during Phase 0 on Tolu's explicit request ("in the meantime build UBM").

## Open items for the next phases

- 40 stub tests remain as the roadmap (memory, dots, region, consensus, losses, baselines, training, diagnostics).
- `data/dataset.py` (packing/dataloaders) still stubbed — Phase 2.
- Fairness (Yoruba/Igbo/Hausa) samples: not supplied; skipped for now, will ask Tolu when needed.
