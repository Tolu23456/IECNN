"""Export a checkpoint's internals to JSON for the viz server (runner job).

Loads checkpoints/<dir>/checkpoint.pt, runs ONE sample text through the real
model, and writes results/viz/<run>.json with:
  - 2-D projection of the tied embedding table (numpy SVD),
  - per-position, per-pass, per-dot: mixing weights, gamma confidence,
    top-5 predictions of each dot and of the pooled region answer,
  - cross-region omega, weight entropy per position,
  - learned score-pathway scalars (alpha/beta per region).
Inference only — no training. Torch use is allowed ONLY here (on GitHub
Actions runners), never on the dev box.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from UBM import UBM                                    # noqa: E402
from UBM.tokenizer import fingerprint                  # noqa: E402
from train.schedules import temperature_at             # noqa: E402
from train.train import build_model                    # noqa: E402


def _id_bytes(ubm: UBM) -> dict:
    """id -> raw bytes via the tokenizer's own recursive expansion."""
    return {b.id: ubm.expand(b.id) for b in ubm.bases}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--vocab", default="UBM/vocab_v1.json")
    ap.add_argument("--data", default="data/val.txt")
    ap.add_argument("--n-tokens", type=int, default=256)   # 48 was too few for consensus-audit stats (Tolu, 2026-11-13)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = Path(args.ckpt_dir)
    ck = torch.load(d / "checkpoint.pt", weights_only=False)
    summary = json.load(open(d / "summary.json"))
    config = json.load(open(d / "config_used.json"))
    run = d.name

    rec_sha = str(ck.get("vocab_sha256", ""))
    if rec_sha and fingerprint(args.vocab) != rec_sha:
        raise RuntimeError("VOCAB MISMATCH in export (see eval_checkpoint.py)")
    ubm = UBM.load(args.vocab)
    V = len(ubm.bases)

    model = build_model(config, V)
    model.load_state_dict(ck.get("best_state") or ck["model"])
    model.eval()

    # ---- sample text -> ids (truncate by tokens) --------------------------
    text = Path(args.data).read_text(encoding="utf-8")
    # a real 4000-char slice of story text: chunk_text()[0] is a single
    # pre-tokenizer chunk (often the book's leading blank line!) — the
    # 2026-11-13 first export produced a one-token '\n' sample.
    chunk = text.strip()[:4000]
    ids = ubm.encode(chunk)[: args.n_tokens]
    decode_bytes = _id_bytes(ubm)
    toks = [decode_bytes.get(int(i), b"?").decode("utf-8", "replace") for i in ids]

    # ---- eval tau at the checkpoint's best step ---------------------------
    best_step = int(summary.get("best_step", ck.get("step", 0)))
    steps_target = int(summary.get("steps_target",
                                   config.get("training", {}).get("steps", 1)))
    cons = config.get("consensus", {})
    tau0 = temperature_at(best_step, steps_target,
                          int(cons.get("tau_warmup", 2000)),
                          float(cons.get("tau_max", 1.0)),
                          float(cons.get("tau_min", 0.2)))
    floor = 0.0
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model(x, tau0=tau0, floor=floor)
    regions = out["region_outputs"]
    R = len(regions)
    last = regions[-1]
    passes = last["per_pass"]                       # list of dicts
    P = len(passes)
    N = passes[0]["weights"].shape[-1]
    T = x.shape[1]

    # ---- per-position capture --------------------------------------------
    positions = []
    for t in range(T - 1):
        nxt = int(ids[t + 1])
        dots = []
        for k in range(N):
            wts, gms, tops = [], [], []
            for p in range(P):
                wts.append(float(passes[p]["weights"][0, t, k]))
                gms.append(float(passes[p]["gamma"][0, t, k]))
                lp = passes[p]["log_pi"][0, t, k]
                top = torch.topk(lp, args.top)
                tops.append([(decode_bytes.get(int(ix), "?").decode("utf-8", "replace"),
                              float(probs)) for ix, probs in
                             zip(top.indices, top.values.softmax(0))])
            dots.append({"weights": wts, "gamma": gms, "top": tops})
        pooled_lp = passes[-1]["log_b"][0, t]
        top = torch.topk(pooled_lp, args.top)
        pooled_top = [(decode_bytes.get(int(ix), "?").decode("utf-8", "replace"),
                       float(pr)) for ix, pr in
                      zip(top.indices, top.values.softmax(0))]
        positions.append({"token": toks[t], "next": toks[t + 1], "next_id": nxt,
                          "dots": dots, "pooled_top": pooled_top})

    w_last = passes[-1]["weights"][0]               # [T, N]
    ent = -(w_last * (w_last + 1e-12).log()).sum(-1)

    # ---- embedding projection --------------------------------------------
    E = model.embed.weight.detach().cpu().numpy().astype(np.float64)
    E = E - E.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(E, full_matrices=False)
    pts = (U[:, :2] * S[:2]).tolist()

    # ---- learned score-pathway scalars -----------------------------------
    scalars = {}
    for r, region in enumerate(model.regions):
        if hasattr(region, "alpha"):
            scalars[f"region{r}_alpha"] = float(region.alpha.detach())
            scalars[f"region{r}_beta"] = float(region.beta.detach())
    if hasattr(model, "alpha_region"):
        scalars["alpha_region"] = float(model.alpha_region.detach())

    export = {
        "meta": {"run": run, "step": best_step, "stop_kind": summary.get("stop_kind"),
                 "params": summary.get("parameters"), "tau0": round(tau0, 5),
                 "vocab_sha256": fingerprint(args.vocab),
                 "R": R, "n_dots": N, "n_passes": P},
        "scalars": scalars,
        "embeddings": {"labels": [decode_bytes.get(i, "?").decode("utf-8", "replace")
                                  .replace("\n", "\\n") for i in range(V)],
                       "points": pts},
        "sample": {"text": chunk[:600], "tokens": toks,
                   "passes": [f"pass {p+1}" for p in range(P)],
                   "positions": positions},
        "weight_entropy_over_positions": [round(float(e), 5) for e in ent],
        "omega_over_positions": [[round(float(o), 4) for o in
                                  out["region_weights"][0, t]] for t in range(T)],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(export, indent=1))
    print("exported", args.out, f"({len(positions)} positions, "
          f"{R} regions x {N} dots x {P} passes)")


if __name__ == "__main__":
    main()
