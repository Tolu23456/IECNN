"""Reload diagnosis: why does E1 score worse outside the training process?

Tolu (2026-11-13): E1 pilot @ step 540, tau 0.954 scored in-loop at
val 3.213 / train 3.284 / passes 3.601,3.496,3.213 — but the runner re-score
gave val 3.777 / train 3.738 / passes 4.226,4.112,3.777. The gap hits train
text and every pass, so it is not the val sample.

ALL probes run in ONE process on the trained checkpoint (inference only):

  (a) replicate the TRAINING-LOOP eval protocol exactly on the live model,
      then toggle ONE factor at a time:
        A1 live, train()-mode dropout, seq 128, first-100k  (= the in-loop way)
        A2 live, eval()-mode,           seq 128, first-100k (dropout off)
        A3 live, eval()-mode,           seq 256, first-100k (context length)
        A4 live, eval()-mode,           seq 128, TAIL-100k   (val sample)
  (b) state_dict -> fresh build_model -> load_state_dict(strict=True) -> score
  (c) torch.save -> torch.load round trip -> score
  (d) live vs reloaded with fast_math True and False
plus: exact state-dict equality, bit-identity of forward outputs, and the
CAUSALITY PROBE on trained weights (B>1, T=256, fast path): change tokens
after position t, confirm log-probs at positions <= t are bit-identical —
on live AND reloaded models.

Writes results/diagnosis/<run>_reload.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from UBM import UBM                                    # noqa: E402
from UBM.tokenizer import chunk_text, fingerprint      # noqa: E402
from train.schedules import temperature_at             # noqa: E402
from train.train import build_model                    # noqa: E402
from train.evaluate import (evaluate_bpc, evaluate_bpc_per_pass,  # noqa: E402
                            prepare_text)


def score(model, ubm, text, prepared, *, seq_len, tau0, max_chars=None,
          train_mode=False, per_pass=False):
    """Score exactly like evaluate_* but with an explicit mode toggle."""
    was_training = model.training
    model.train(train_mode)
    try:
        if per_pass:
            out = evaluate_bpc_per_pass(model, ubm, text, seq_len=seq_len,
                                        max_chars=max_chars, tau0=tau0,
                                        prepared=prepared)
            return {"bpc": out["final"], "passes": out["passes"]}
        return {"bpc": evaluate_bpc(model, ubm, text, seq_len=seq_len,
                                    max_chars=max_chars, tau0=tau0,
                                    prepared=prepared)}
    finally:
        model.train(was_training)


def fast_math_flags(model, value: bool):
    for region in getattr(model, "regions", []):
        if hasattr(region, "dots"):
            region.dots.fast_math = value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--vocab", default="UBM/vocab_v1.json")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = Path(args.ckpt_dir)
    summary = json.load(open(d / "summary.json"))
    config = json.load(open(d / "config_used.json"))
    data_dir = Path(args.data)
    val_text = (data_dir / "val.txt").read_text(encoding="utf-8")
    train_text = (data_dir / "train.txt").read_text(encoding="utf-8")

    ubm = UBM.load(args.vocab)
    V = len(ubm.bases)
    seq_train = int(config.get("data", {}).get("seq_len", 256))
    eval_chars = int(config.get("training", {}).get("eval_chars", 100_000))

    # Corpus v1 guard (2026-11-13): the original re-scores silently used the
    # v0 Shakespeare val (55,769 chars) because the workflow's corpus_config
    # was a base rung config without data.corpus: big. Refuse to diagnose on
    # a corpus that cannot be corpus v1.
    if len(val_text) < 150_000:
        raise RuntimeError(
            f"CORPUS GUARD: val.txt has only {len(val_text)} chars — this is "
            "the v0 Shakespeare split, not corpus v1 (val = a whole held-out "
            "book, >200k chars). Re-run with a *_pilot.yaml corpus_config.")

    # trained weights = best_state (what the re-score used)
    ck = torch.load(d / "checkpoint.pt", weights_only=False)
    state = ck.get("best_state") or ck["model"]
    live = build_model(config, V)
    live.load_state_dict(state, strict=True)
    live.eval()

    best_step = int(summary.get("best_step", ck.get("step", 0)))
    steps_target = int(summary.get("steps_target",
                                   config["training"]["steps"]))
    cons = config.get("consensus", {})
    tau0 = temperature_at(best_step, steps_target,
                          int(cons.get("tau_warmup", 2000)),
                          float(cons.get("tau_max", 1.0)),
                          float(cons.get("tau_min", 0.2)))
    R = {"meta": {"run": d.name, "best_step": best_step, "tau0": round(tau0, 5),
                  "seq_len_trained": seq_train,
                  "eval_chars": eval_chars,
                  "vocab_sha256": fingerprint(args.vocab)},
         "probes": {}, "causality": {}}

    # head replicates the in-loop sample EXACTLY: the loop prepared the FULL
    # val text once and cut at eval_chars; a token crossing the 100k boundary
    # behaves identically in a 200k-char prefix (merges are <=32 bytes, local).
    head = val_text[: eval_chars * 2]
    tail = val_text[eval_chars:]          # everything after the in-loop sample
    prepared_head = prepare_text(ubm, head)
    prepared_tail = prepare_text(ubm, tail)
    train_slice = train_text[: eval_chars * 6]
    prepared_train = prepare_text(ubm, train_slice)

    # ---------------- (a) one-factor-at-a-time on the LIVE model -----------
    def probe(key, **kw):
        torch.manual_seed(0)
        r = score(live, ubm, kw.pop("text"), kw.pop("prepared"), tau0=tau0, **kw)
        R["probes"][key] = r
        print(f"  {key:34} bpc={r['bpc']:.5f} "
              f"passes={r.get('passes', '-')}", flush=True)
        return r

    print("(a) live model, one factor at a time:", flush=True)
    probe("A1_live_trainmode_seq128_head100k", text=head, prepared=prepared_head,
          seq_len=seq_train, max_chars=eval_chars, train_mode=True, per_pass=True)
    probe("A2_live_evalmode_seq128_head100k", text=head, prepared=prepared_head,
          seq_len=seq_train, max_chars=eval_chars, train_mode=False, per_pass=True)
    probe("A3_live_evalmode_seq256_head100k", text=head, prepared=prepared_head,
          seq_len=256, max_chars=eval_chars, train_mode=False, per_pass=True)
    probe("A4_live_evalmode_seq128_after100k", text=tail, prepared=prepared_tail,
          seq_len=seq_train, train_mode=False)
    probe("A5_live_evalmode_seq256_after100k", text=tail, prepared=prepared_tail,
          seq_len=256, train_mode=False)
    R["probes"]["A1_in_loop_reference"] = {              # from the pilot CSV/summary
        "bpc": 3.21328, "passes": [3.60097, 3.49581, 3.21328],
        "note": "in-loop at step 540 (train-mode, seq128, head-100k)"}
    R["probes"]["rescore_reference"] = {
        "bpc": 3.77721, "passes": [4.22628, 4.11179, 3.77721],
        "note": "runner re-score (eval-mode, seq256 default, FULL val)"}

    # train slice, both protocols (the 'affects train text' datum)
    probe("T1_live_trainmode_seq128_train100k", text=train_slice,
          prepared=prepared_train, seq_len=seq_train, max_chars=eval_chars,
          train_mode=True, per_pass=True)
    probe("T2_live_evalmode_seq256_train100k", text=train_slice,
          prepared=prepared_train, seq_len=256, max_chars=eval_chars,
          train_mode=False, per_pass=True)

    # ---------------- canonical protocol used by (b)-(d) -------------------
    canon = dict(text=val_text, prepared=prepare_text(ubm, val_text),
                 seq_len=seq_train, max_chars=None, train_mode=False)
    # NOTE: canonical = eval-mode + the TRAINED seq_len + FULL val.
    # (The re-score's extra deltas vs A2/A3 tell us about full-vs-sample and
    # seq, already factored above.)

    # ---------------- (b) state_dict -> fresh build -> load ----------------
    print("(b) fresh build + load_state_dict(strict=True):", flush=True)
    fresh = build_model(config, V)
    fresh.load_state_dict(live.state_dict(), strict=True)
    fresh.eval()
    max_param_diff = max(float((a - b).abs().max())
                         for a, b in zip(live.state_dict().values(),
                                         fresh.state_dict().values()))
    print(f"  max |param diff| live vs fresh-loaded: {max_param_diff}", flush=True)
    R["probes"]["B1_max_param_diff"] = max_param_diff
    R["probes"]["B2_fresh_full_val"] = score(fresh, ubm, canon["text"],
                                             canon["prepared"], tau0=tau0,
                                             seq_len=canon["seq_len"],
                                             per_pass=True)
    print(f"  B2 fresh full-val: {R['probes']['B2_fresh_full_val']['bpc']:.5f}",
          flush=True)

    # ---------------- (c) torch.save / torch.load round trip ---------------
    print("(c) torch.save -> torch.load round trip:", flush=True)
    tmp = Path("/tmp/opencode/_roundtrip.pt")
    torch.save({"model": live.state_dict()}, tmp)
    rt_state = torch.load(tmp, weights_only=False)["model"]
    rt = build_model(config, V)
    rt.load_state_dict(rt_state, strict=True)
    rt.eval()
    max_rt_diff = max(float((a - b).abs().max())
                      for a, b in zip(live.state_dict().values(), rt_state.values()))
    print(f"  max |param diff| live vs round-trip: {max_rt_diff}", flush=True)
    R["probes"]["C1_max_param_diff_roundtrip"] = max_rt_diff
    R["probes"]["C2_roundtrip_full_val"] = score(rt, ubm, canon["text"],
                                                 canon["prepared"], tau0=tau0,
                                                 seq_len=canon["seq_len"],
                                                 per_pass=True)
    print(f"  C2 roundtrip full-val: "
          f"{R['probes']['C2_roundtrip_full_val']['bpc']:.5f}", flush=True)

    # live full-val on the SAME canonical protocol for the (b)/(c) baseline
    R["probes"]["A6_live_full_val_canonical"] = score(
        live, ubm, canon["text"], canon["prepared"], tau0=tau0,
        seq_len=canon["seq_len"], per_pass=True)
    print(f"  A6 live full-val (canonical): "
          f"{R['probes']['A6_live_full_val_canonical']['bpc']:.5f}", flush=True)

    # ---------------- (d) fast_math True vs False --------------------------
    print("(d) fast_math True vs False (live + reloaded):", flush=True)
    fast_math_flags(live, False)
    R["probes"]["D1_live_fastOFF_head100k"] = score(
        live, ubm, head, prepared_head, seq_len=seq_train, max_chars=eval_chars)
    fast_math_flags(live, True)
    R["probes"]["D2_live_fastON_head100k"] = score(
        live, ubm, head, prepared_head, seq_len=seq_train, max_chars=eval_chars)
    fast_math_flags(rt, False)
    R["probes"]["D3_rt_fastOFF_head100k"] = score(
        rt, ubm, head, prepared_head, seq_len=seq_train, max_chars=eval_chars)
    fast_math_flags(rt, True)
    R["probes"]["D4_rt_fastON_head100k"] = score(
        rt, ubm, head, prepared_head, seq_len=seq_train, max_chars=eval_chars)
    for k in ("D1_live_fastOFF_head100k", "D2_live_fastON_head100k",
              "D3_rt_fastOFF_head100k", "D4_rt_fastON_head100k"):
        print(f"  {k:34} bpc={R['probes'][k]['bpc']:.5f}", flush=True)

    # ---------------- bit-identity of forwards (live vs reloaded) ----------
    torch.manual_seed(1)
    xb = torch.randint(0, V, (2, seq_train))
    with torch.no_grad():
        o_live = live(xb, tau0=tau0)
        o_rt = rt(xb, tau0=tau0)
    R["probes"]["E1_forward_max_abs_diff_live_vs_roundtrip"] = float(
        (o_live["final_log_probs"] - o_rt["final_log_probs"]).abs().max())
    print(f"  forward max|Δ| live vs round-trip: "
          f"{R['probes']['E1_forward_max_abs_diff_live_vs_roundtrip']:.3e}",
          flush=True)

    # ---------------- causality probe (B>1, T=256, fast path) --------------
    print("causality probe (trained weights, B=4, T=256, fast_math=True):",
          flush=True)
    B, T, t0 = 4, 256, 200
    torch.manual_seed(2)
    chunk = chunk_text(train_text[:20_000])[0]
    ids_all = torch.tensor(ubm.encode(chunk), dtype=torch.long)
    idx = torch.randint(0, len(ids_all) - T, (B,))
    x = torch.stack([ids_all[i:i + T] for i in idx])

    def causality(model, tag):
        fast_math_flags(model, True)
        model.eval()
        with torch.no_grad():
            base = model(x, tau0=tau0)["final_log_probs"]
            worst = 0.0
            bit_identical = True
            for b in range(B):
                x2 = x.clone()
                x2[b, t0:] = torch.randint(0, V, (T - t0,))
                alt = model(x2, tau0=tau0)["final_log_probs"]
                diff = (alt[b, :t0] - base[b, :t0]).abs().max().item()
                worst = max(worst, diff)
                bit_identical &= torch.equal(base[b, :t0], alt[b, :t0])
        R["causality"][tag] = {"max_abs_diff_leq_t0": worst,
                               "bit_identical": bool(bit_identical)}
        print(f"  {tag:12} max|Δ| positions<={t0}: {worst:.3e} "
              f"bit-identical: {bit_identical}", flush=True)

    causality(live, "live")
    causality(rt, "reloaded")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(R, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
