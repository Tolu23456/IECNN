"""Multi-choice benchmarks for IECNN (Tolu, 2026-11-16): WinoGrande,
HellaSwag, ARC-Challenge — zero-shot, option scored by NLL (the standard
LM protocol; no fine-tuning, no prompt engineering beyond the formats
below). Chance: 50% / 25% / 25%. GPT2-XL reference (phi-1.5 paper):
0.583 / 0.400 / 0.250.

Formats (kept minimal and fixed — the same protocol scores every future
checkpoint, so rungs are comparable):
  WinoGrande: sentence with "_"; score NLL of (option + suffix) given the
              prefix (partial scoring, standard practice).
  HellaSwag:  score NLL of each ending given the full ctx; report acc and
              acc_norm (NLL normalized by ending char length).
  ARC-C:      "Question: {q}\nAnswer: {option}" — score NLL of the answer
              span given the question prefix.

Datasets: parquet under data/mc/ (winogrande_xl / hellaswag / ARC-Challenge
validation splits, allenai HF repos).
"""
from __future__ import annotations
import sys, os, json, math
sys.path.insert(0, "/content/IECNN"); os.chdir("/content/IECNN")
import torch
import pyarrow.parquet as pq
from pathlib import Path
from UBM import UBM
from config import load_config
from train.train import build_model
from train.schedules import temperature_at

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/content/e6_cpu/checkpoint.pt")
ap.add_argument("--config", default="configs/e6_consensus_v2_vocab3_warmstart.yaml")
ap.add_argument("--overlay", nargs="*", default=["overlay_cpu_fp32.yaml"])
ap.add_argument("--limit-hs", type=int, default=1000)
ap.add_argument("--only", default="", help="comma list: winogrande,hellaswag,arc_challenge (default all)")
ap.add_argument("--limit-wg", type=int, default=0, help="0 = all 1267")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--context", type=int, default=256)
ap.add_argument("--out", default="results/mc_bench.json")
args = ap.parse_args()

overlays = [o for o in (args.overlay or []) if o and o != "none"]
cfg = load_config(args.config, *overlays)
ubm = UBM.load(cfg["ubm"]["vocab_file"])
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
state = ck.get("best_state") or ck["model"]
model = build_model(cfg, len(ubm.bases))
model.load_state_dict(state, strict=False)
model.eval()
torch.set_grad_enabled(False)

cons = cfg.get("consensus", {})
tcfg = cfg["training"]
best_step = int(ck.get("best_step") or ck.get("step") or 0)
tau0 = temperature_at(best_step, int(tcfg.get("steps", 50000)),
                      int(cons.get("tau_warmup", 2000)),
                      float(cons.get("tau_max", 1.0)),
                      float(cons.get("tau_min", 0.2)))
print(f"ckpt {args.ckpt} best_step {best_step} tau0 {tau0:.4f}", flush=True)

CTX = args.context

def nll_batch(seqs: list[tuple[list[int], list[int]]]) -> list[float]:
    """seqs = [(prefix_ids, score_ids)] -> total NLL (nats) of score_ids.
    Prefix truncated on the LEFT to fit one window; rows batched."""
    out = [0.0] * len(seqs)
    device = next(model.parameters()).device
    for i in range(0, len(seqs), args.batch):
        chunk = seqs[i:i + args.batch]
        rows_x, rows_y = [], []
        for pre, sc in chunk:
            room = CTX - len(sc)
            pre = pre[-max(room, 1):] if room < len(pre) else pre
            x = (pre + sc)[-CTX:]
            rows_x.append(x[:-1])
            rows_y.append(x[1:])
        L = min(len(r) for r in rows_x)
        x = torch.tensor([r[-L:] for r in rows_x], device=device)
        y = torch.tensor([r[-L:] for r in rows_y], device=device)
        o = model(x, tau0=tau0, floor=0.0) if hasattr(model, "regions") else model(x)
        logp = o["final_log_probs"] if isinstance(o, dict) else o
        nll = -logp.gather(2, y.unsqueeze(2)).squeeze(2)          # [B, L]
        for b, (pre, sc) in enumerate(chunk):
            # the scored span is the LAST len(sc)-1 tokens of the window
            k = min(len(sc) - 1, L)
            out[i + b] = float(nll[b, -k:].sum().item()) if k > 0 else 0.0
    return out

def enc(s: str) -> list[int]:
    return ubm.encode(s)

def run(name: str, items: list[dict], acc_key: str):
    # drop degenerate items (no options / empty scored spans score 0 NLL)
    items = [it for it in items
             if it["options"] and all(sc for _, sc in it["options"])]
    seqs, meta = [], []
    # ARC has 3-5 options per question (WG fixed 2, HS fixed 4) — track the
    # cumulative base per item so item->nll indices never desync.
    bases = []
    for it in items:
        bases.append(len(seqs))
        for j, (pre, sc) in enumerate(it["options"]):
            seqs.append((pre, sc))
            meta.append((it["answer"], j, len(sc), it.get("chars", [0]*len(it["options"]))[j]))
    nlls = nll_batch(seqs)
    n_items = len(items)
    correct = 0
    correct_norm = 0
    for i, it in enumerate(items):
        idx = list(range(bases[i], bases[i] + len(it["options"])))
        scores = [nlls[j] for j in idx]
        pick = scores.index(min(scores))
        chars = it.get("chars") or [1] * len(scores)
        pickn = min(range(len(scores)), key=lambda j: scores[j] / max(chars[j], 1))
        correct += int(pick == it["answer"])
        correct_norm += int(pickn == it["answer"])
    acc = correct / n_items
    res = {"acc": round(acc, 4), "n": n_items}
    if name == "hellaswag":
        res["acc_norm"] = round(correct_norm / n_items, 4)
    print(f"{name}: {res}", flush=True)
    return res

results = {}
ONLY = [x.strip() for x in args.only.split(',') if x.strip()] or     ["winogrande", "hellaswag", "arc_challenge"]

# ---- WinoGrande ----
wg = pq.read_table("data/mc/wg.parquet").to_pylist() if "winogrande" in ONLY else []
if args.limit_wg:
    wg = wg[:args.limit_wg]
items = []
for r in wg:
    sent = r["sentence"]
    assert "_" in sent
    pre_s, suf_s = sent.split("_", 1)
    opts = []
    for key in ("option1", "option2"):
        s = pre_s + " " + r[key] + suf_s
        full = enc(s)
        # score tokens after the blank: prefix tokens = enc(pre_s)
        np_ = len(enc(pre_s)) or 1
        opts.append((full[:np_], full[np_:]))
    items.append({"options": opts,
                  "answer": int(r["answer"]) - 1})   # answer is "1"/"2"
if "winogrande" in ONLY:
    results["winogrande"] = run("winogrande", items, "acc")

# ---- HellaSwag ----
hs = pq.read_table("data/mc/hs.parquet").to_pylist()[:args.limit_hs] if "hellaswag" in ONLY else []
items = []
for r in hs:
    ctx = r["ctx"]
    endings = r["endings"]
    if isinstance(endings, str):
        endings = json.loads(endings)
    pre = enc(ctx)
    opts, chars = [], []
    for e in endings:
        full = enc(ctx + " " + e)
        np_ = len(pre) or 1
        opts.append((full[:np_], full[np_:]))
        chars.append(len(e))
    if len(opts) != 4:
        continue
    items.append({"options": opts, "chars": chars,
                  "answer": int(r["label"])})
if "hellaswag" in ONLY:
    results["hellaswag"] = run("hellaswag", items, "acc")

# ---- ARC-Challenge ----
arc = pq.read_table("data/mc/arc.parquet").to_pylist() if "arc_challenge" in ONLY else []
items = []
for r in arc:
    texts = r["choices"]["text"]
    labels = r["choices"]["label"]
    if r["answerKey"] not in labels:
        continue
    pre = enc(f"Question: {r['question']}\nAnswer:")
    opts = [(pre, enc(" " + t)) for t in texts]
    items.append({"options": opts, "answer": labels.index(r["answerKey"])})
if "arc_challenge" in ONLY:
    results["arc_challenge"] = run("arc_challenge", items, "acc")

res = {"meta": {"ckpt": args.ckpt, "best_step": best_step,
                "tau0": round(tau0, 5), "limit_hs": args.limit_hs},
       "results": results,
       "chance": {"winogrande": 0.5, "hellaswag": 0.25, "arc_challenge": 0.25},
       "gpt2_xl_reference": {"winogrande": 0.583, "hellaswag": 0.400,
                             "arc_challenge": 0.250}}
print(json.dumps(res, indent=1))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=1))
print("saved", args.out)
