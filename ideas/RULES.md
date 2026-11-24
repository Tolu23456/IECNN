# RULES — the fixed contract every idea, probe and arm is judged against

Written by the campaign owner. **The FIXED section may not be edited by idea
agents.** The judge agent may append to the DYNAMIC section only, one dated line
each, and every rejection must cite the rule it rests on.

Repo: `/home/t85491005/IECNN`. Evidence ledger: `DECISIONS.md` (one row per
measured batch). Live map: `STATUS.md`. Read both before proposing anything.

---

## FIXED

**F1 — A mechanism pays for its own compute, or it is closed.**
Every claim reports three numbers: `bpc` at equal wall clock (the headline
metric), `bpc` at matched total MACs (per-token efficiency), and the step-time
delta. A gain that exists only at matched MACs is a **wash**, not a win. MACs
alone are not a cost model: launch-bound ops and dtype temporaries have cost
1.5–2.2% of step time while being ~0.06% of MACs. Price the launches too.

**F2 — Resolution bars.**
900 s: a mechanism is admitted at **≥0.005 bpc with both seeds the same sign**,
and the effect must be stated against that session's own anchor. 360 s: the
floor is **~0.01**; a 360 s result inside ±0.01 is a *resolution failure*, not a
rejection and not an acceptance. Quote the seed spread you are comparing against.

**F3 — Flag-gated, off by default, baseline byte-identical when off.**
New mechanism ⇒ a config flag whose default reproduces today's model exactly
(a test asserts bit-identical log-probs), a unit test of the exact mechanism, and
a one-step smoke run before any timed arm. `py_compile` locally is not a test.

**F4 — No GPU time without an offline price.**
Price it first on **trained checkpoints with the base frozen**: fit any free
parameters on one parity half of the val windows, score the other, both
directions. Report the per-position oracle as the ceiling, and the deployed
mechanism's number as the comparison.

**F5 — The closed list. Reopening requires a MATERIALLY different
implementation, not a retune.** Closed: `query_from_input`; uniform `k_lag`;
extra read heads; N=2 / prior N=4; `gate_agree`; `global_exact`; linear-attention
global read; lr tuning; `dist_bias`; `d_s 64`; span 12/16; `ngram_bias`;
`id_lags`; `head_rank48`; `head_skip`; `q_shift=1`; `k_lag=1`; `dynamic_query`;
geometric/quad bands; `seclusion`; plain vote over duplicate matches; choosing a
single k (k=3 is the elbow); additive parallel-k in **both** implementations
(`suffix_copy_ks`, `suffix_copy_ext4` — each ~+1.7–2.2% step for ~0.005/token);
the pointer-rule replacement `ext1 else maxext` (63.6% at `maxext`'s 63.8%);
depth-down at the endpoint (`x3` +0.028); gate tuning of `(a,b,cap)` (the learned
point is the joint optimum); `sc3e42`.

**F6 — Protocol.** Canonical split: train = enwik8 bytes `[0, 90_000_000)`, val =
`[90_000_000, 95_000_000)`, last 5 MB untouched. `B=32`, `T=256`, `V=256` byte
level, 160 val windows, `stride=122`, `ARM_MARK_S=30`. Claims are **in-session
only**; the anchor moves between sessions (`tf-swa-c` 1.7960 → 1.7917), so quote
`data_fp` (currently `a23bcd55c7/afc1109404`) and never compare across sessions.

**F7 — Faithfulness before deltas.** Any offline table must first recompute the
model's own `bpc_final` to 4 dp. If it cannot, its deltas are not believed.
`bpc_final` is the SWA average — the reference — never the last raw marker.

**F8 — Alignment before interpretation.** Any pointer/index/precomputed table is
asserted against a brute-force reference on at least one position, or a
misalignment will look like a mechanism result.

**F9 — Isolate.** One change at a time, with params and step cost reported. An
ablation that moves two things cannot be attributed.

**F10 — Pre-registration.** Write the predicted effect and the falsifier *before*
the run. A verdict may not be retro-fitted to whatever the queue returned.

**F11 — Say what changed the model's inputs.** Two different claims: a
like-for-like same-input win, versus a **capability change** (e.g. exact-match
retrieval over a corpus the transformer never sees, at a one-off build cost).
Never blur them.

**F12 — Prune loudly.** Something that does not pay is closed in `DECISIONS.md`
with its number, and the reason it failed is stated as a mechanism fact.

**F13 — Do not double-run the campaign.** Anything already measured in
`DECISIONS.md` is a fact, not a hypothesis. Check before proposing; cite the row.

---

## DYNAMIC (judge appends here; dated, one line each)

_(empty)_
