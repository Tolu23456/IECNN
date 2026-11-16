"""Committee-combination study (the new central research question, Tolu +
ChatGPT, 2026-11-16): 32.1% oracle top-1 vs 22.6% pooled top-1 says the
pool throws away ~10 accuracy points the dots already hold. Can a better
COMBINATION/SELECTION rule recover them WITHOUT more parameters?

Method: one eval forward over the val slice; capture the LAST pass's
per-dot log_pi [.., N, V] and consensus weights [.., N]. Every combination
rule is then pure post-hoc math on those two tensors — no retraining:

  log_linear   the model's current rule (must equal the model's own bpc —
               a built-in correctness check for the capture)
  mixture      linear opinion pool: sum w_k * pi_k
  uniform      plain mean of the dots' distributions
  argmax_w     hard selection: trust ONLY the weight-argmax dot (this is
               the model's own "which dot do I believe" judgment)
  sharpen3     weights cubed + renormalized, then log-linear (sharper trust)
  best_fixed   the single best dot overall (post-hoc fixed choice)
  oracle       per-position: a correct dot when one exists, else pooled
               (upper bound for ANY selection rule)

Each rule reports bpc (bits/char), top-1 and top-5 accuracy over the same
positions with the same char accounting as train/evaluate.py.

Inference only; CPU-friendly. Run on any trained checkpoint dir.
"""
from __future__ import annotations
import sys, os, json, math
sys.path.insert(0, "/content/IECNN"); os.chdir("/content/IECNN")
import torch
import torch.nn.functional as F
from pathlib import Path
from UBM import UBM
from config import load_config
from train.train import build_model
from train.evaluate import prepare_text
from train.schedules import temperature_at

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/content/e6_cpu/checkpoint.pt")
ap.add_argument("--ckpt-dir", default="")
ap.add_argument("--config", default="configs/e6_consensus_v2_vocab3_warmstart.yaml")
ap.add_argument("--overlay", nargs="*", default=["overlay_cpu_fp32.yaml"],
                help="overlays on top of --config (default: the CPU recipe)")
ap.add_argument("--chars", type=int, default=100000,
                help="char cap, matching training's eval_chars")
ap.add_argument("--context", type=int, default=256)
ap.add_argument("--out", default="results/consensus_ablation.json")
args = ap.parse_args()

overlays = [o for o in (args.overlay or []) if o and o != "none"]
cfg = load_config(args.config, *overlays)
ubm = UBM.load(cfg["ubm"]["vocab_file"])
V = len(ubm.bases)
ck_path = args.ckpt_dir and str(Path(args.ckpt_dir) / "checkpoint.pt") or args.ckpt
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
state = ck.get("best_state") or ck["model"]
model = build_model(cfg, V)
model.load_state_dict(state, strict=False)
model.eval()
torch.set_grad_enabled(False)

tcfg = cfg["training"]
cons = cfg.get("consensus", {})
best_step = int(ck.get("best_step") or ck.get("step") or 0)
tau0 = temperature_at(best_step, int(tcfg.get("steps", 50000)),
                      int(cons.get("tau_warmup", 2000)),
                      float(cons.get("tau_max", 1.0)),
                      float(cons.get("tau_min", 0.2)))
print(f"ckpt {ck_path} best_step {best_step} tau0 {tau0:.4f}", flush=True)

val = Path("data/val.txt").read_text()
ids, expand_len = prepare_text(ubm, val)
L = args.context
starts = list(range(0, len(ids) - 1, L))
starts = [s for s in starts if s + L + 1 <= len(ids)]   # full windows only

N = cfg["model"]["n_dots"]
RULES = ["log_linear", "mixture", "uniform", "argmax_w", "sharpen3", "oracle"]
nll = {r: 0.0 for r in RULES}
top1 = {r: 0 for r in RULES}
top5 = {r: 0 for r in RULES}
nll_dot = [0.0] * N
hit_dot = [0] * N
chars_total = 0
n_pos = 0

for s in starts:
    x = torch.tensor([ids[s:s + L]], dtype=torch.long)
    y = torch.tensor([ids[s + 1:s + 1 + L]], dtype=torch.long)
    out = model(x, tau0=tau0, floor=0.0)
    pp = out["per_pass_last_region"][-1]
    log_pi = pp["log_pi"].float()[0]        # [L, N, V]
    w = pp["weights"].float()[0]            # [L, N]
    t = y[0]                                # [L]

    # ---- combination rules (post-hoc on log_pi, w) ----
    pooled = torch.einsum("pn,pnv->pv", w, log_pi)                 # log-linear
    mix = torch.log(torch.einsum("pn,pnv->pv", w, log_pi.exp())
                    .clamp_min(1e-12))
    uni = log_pi.mean(dim=1)                                        # [P, V]
    k_star = w.argmax(dim=-1, keepdim=True)                         # [P, 1]
    argw = log_pi.gather(1, k_star[:, :, None].expand(-1, 1, V)
                         ).squeeze(1)                               # [P, V]
    w3 = w.pow(3)
    w3 = w3 / w3.sum(-1, keepdim=True).clamp_min(1e-9)
    sharp = torch.einsum("pn,pnv->pv", w3, log_pi)
    # oracle: a correct dot when one exists, else the pooled rule
    hit = (log_pi.argmax(-1) == t[:, None])                         # [P, N]
    any_hit = hit.any(-1)
    ko = hit.float().argmax(dim=-1)                                 # first correct
    orc = log_pi.gather(1, ko[:, None, None].expand(-1, 1, V)).squeeze(1)
    # stronger selection bound: among correct dots take the one whose
    # distribution has the LOWEST NLL at the truth token
    pd = -log_pi.gather(2, t[:, None, None].expand(L, N, 1)).squeeze(-1)  # [L, N]
    kn = torch.where(hit, pd, torch.full_like(pd, float("inf"))).argmin(-1)
    orcn = log_pi.gather(1, kn[:, None, None].expand(-1, 1, V)).squeeze(1)
    orc = torch.where(any_hit[:, None], orcn, pooled)

    outs = {"log_linear": pooled, "mixture": mix, "uniform": uni,
            "argmax_w": argw, "sharpen3": sharp, "oracle": orc}
    for r, lb in outs.items():
        nrm = lb - lb.logsumexp(dim=-1, keepdim=True)               # proper dist
        nll[r] += float(-nrm.gather(1, t[:, None]).sum().item())
        top1[r] += int((nrm.argmax(-1) == t).sum().item())
        top5[r] += int((nrm.topk(5, -1).indices == t[:, None]).any(-1).sum().item())
    for k in range(N):
        nll_dot[k] += float(pd[:, k].sum().item())
        hit_dot[k] += int(hit[:, k].sum().item())


    for i in range(L):
        chars_total += expand_len[s + 1 + i]
        n_pos += 1
        if chars_total >= args.chars:
            break
    if chars_total >= args.chars:
        break

denom = math.log(2) * chars_total
res = {
    "meta": {"ckpt": ck_path, "best_step": best_step, "tau0": round(tau0, 5),
             "chars": chars_total, "positions": n_pos, "vocab": V, "n_dots": N},
    "model_bpc_reference": round(nll["log_linear"] / denom, 5),
    "rules": {r: {"bpc": round(nll[r] / denom, 5),
                  "top1": round(top1[r] / n_pos, 5),
                  "top5": round(top5[r] / n_pos, 5)} for r in RULES},
    "best_fixed_dot": {
        "dot": int(max(range(N), key=lambda k: hit_dot[k])),
        "bpc": round(min(nll_dot) / denom, 5),
        "top1": round(max(hit_dot) / n_pos, 5),
        "acc_per_dot": [round(h / n_pos, 5) for h in hit_dot]},
}
print(json.dumps(res, indent=1))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=1))
print("saved", args.out)
