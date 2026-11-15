"""Per-dot deep audit: what each dot learnt, how it votes, when it is
trusted, what it could rescue, and how correlated the committee is
(Tolu's 2026-11-15 dot-observatory ask). Inference only; CPU-friendly.
"""
from __future__ import annotations
import sys, os, json, math
from collections import Counter
sys.path.insert(0, "/content/IECNN"); os.chdir("/content/IECNN")
import torch, numpy as np
from pathlib import Path
from UBM import UBM
from config import load_config
from train.train import build_model
from train.evaluate import prepare_text

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/vocab3_gpu_seed0_checkpoint.pt")
ap.add_argument("--tau", type=float, default=0.894)
ap.add_argument("--windows", type=int, default=255)
ap.add_argument("--out", default="results/dot_audit.json")
args = ap.parse_args()
CKPT = args.ckpt
TAU = args.tau
MAX_WINDOWS = args.windows

cfg = load_config("configs/e1_weights_passes_vocab3_warmstart.yaml")
ubm = UBM.load(cfg["ubm"]["vocab_file"])
V = len(ubm.bases)
tok_str = [str(b) for b in ubm.bases]
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
state = ck.get("best_state") or ck["model"]
model = build_model(cfg, V)
model.load_state_dict(state, strict=True)
model.eval()
torch.set_grad_enabled(False)

val = Path("data/val.txt").read_text()
ids, expand_len = prepare_text(ubm, val)
context = 256
# only FULL windows: the multi-char vocab makes val ~57k tokens,
# so keep s only if ids[s+1 : s+context+1] is also complete
starts = [s for s in range(0, len(ids) - 1, context)
          if s + context + 1 <= len(ids)][:MAX_WINDOWS]
N = 8

# accumulators
top1_hits = [0]*N; top5_hits = [0]*N
w_sum = np.zeros(N); w_right = np.zeros(N); w_wrong = np.zeros(N)
n_right = np.zeros(N); n_wrong = np.zeros(N)
argmax_cnt = np.zeros(N)
solo_ce = np.zeros(N)          # -log p(truth) under dot k alone (nats)
rescue = np.zeros(N)           # dot right & pooled wrong
n_pos = 0
pooled_hit = 0; oracle1 = 0; oracle5 = 0; outvoted = 0
overlap = np.zeros((N, N)); err_same = np.zeros((N, N))
cls_names = ["letter", "upper", "digit", "punct", "space", "other"]
cls_hits = np.zeros((N, 6)); cls_cnt = np.zeros(6)
cal_p_truth = np.zeros(N)
top_pred = [Counter() for _ in range(N)]
w_entropy_sum = 0.0; peaked_cnt = 0
omega_sum = None

def cls_of(tok: str) -> int:
    """Classify a DECODED token. Multi-char merges (v3: ' the', ',\n') made
    the first audit's classifier lump most tokens into 'other' (recorded in
    DECISIONS.md 2026-11-15) — classify on the STRIPPED body instead, and
    give leading-uppercase words their own class."""
    body = tok.strip()
    if not body:
        return 4 if tok.isspace() else 3          # whitespace vs pure punct
    if body.isdigit(): return 2
    if body[0].isupper(): return 1
    if body.isalpha(): return 0
    if all(not c.isalnum() for c in body): return 3
    return 5

B = 4
for gi in range(0, len(starts), B):
    group = starts[gi:gi+B]
    L = context
    xs = [ids[s:s+L] for s in group]; ys = [ids[s+1:s+1+L] for s in group]
    x = torch.tensor(xs, dtype=torch.long); y = torch.tensor(ys, dtype=torch.long)
    out = model(x, tau0=TAU, floor=0.0)
    pp = out["per_pass_last_region"]
    log_pi = pp[-1]["log_pi"]                      # [B,T,N,V]
    w = pp[-1]["weights"]                          # [B,T,N]
    final = out["final_log_probs"]                 # [B,T,V]
    omega = out["region_weights"]                  # [B,T,R]
    if omega_sum is None: omega_sum = np.zeros(omega.shape[-1])
    omega_sum += omega.mean(dim=(0,1)).numpy() * len(group)
    truth = y                                      # [B,T]
    top5 = log_pi.topk(5, dim=-1).indices          # [B,T,N,5]
    hit = (top5[...,0] == truth[:,:,None])         # [B,T,N]
    in5 = (top5 == truth[:,:,None,None]).any(-1)   # [B,T,N]
    p_hit = (final.argmax(-1) == truth)            # [B,T]
    n_pos += truth.numel()
    pooled_hit += int(p_hit.sum()); oracle1 += int(hit.any(-1).sum())
    oracle5 += int(in5.any(-1).sum())
    outvoted += int((~p_hit & hit.any(-1)).sum())
    # per-dot
    for k in range(N):
        h = hit[:,:,k]; f5 = in5[:,:,k]
        top1_hits[k] += int(h.sum()); top5_hits[k] += int(f5.sum())
        wk = w[:,:,k].numpy()
        w_sum[k] += wk.sum()
        w_right[k] += wk[h.numpy().astype(bool)].sum()
        w_wrong[k] += wk[(~h.numpy()).astype(bool)].sum()
        n_right[k] += int(h.sum()); n_wrong[k] += int((~h).sum())
        am = (w.argmax(-1) == k)
        argmax_cnt[k] += int(am.sum())
        lp = log_pi[:,:,k,:]                       # [B,T,V]
        solo_ce[k] += float((-lp.gather(-1, truth[...,None]).squeeze(-1)).sum())
        rescue[k] += int((~p_hit & h).sum())
        cal_p_truth[k] += float(lp.exp().gather(-1, truth[...,None]).squeeze(-1).sum())
        am_idx = lp.argmax(-1).reshape(-1)
        for i in am_idx.tolist(): top_pred[k][tok_str[i]] += 1
    # overlap + error correlation (token-count identity for top-5)
    f5flat = top5.reshape(-1, N, 5).numpy()
    for k in range(N):
        for j in range(k+1, N):
            for t in range(len(group)*L):
                ov = len(set(f5flat[t,k]) & set(f5flat[t,j]))
                overlap[k,j] += ov; overlap[j,k] += ov
    eh = hit.numpy()                               # [B,T,N]
    for k in range(N):
        for j in range(k+1, N):
            same = ((eh[:,:,k] == eh[:,:,j])).sum()
            err_same[k,j] += same; err_same[j,k] += same
    # weight entropy + peaked
    wp = w / (w.sum(-1, keepdim=True) + 1e-12)
    we = -(wp * wp.clamp_min(1e-12).log()).sum(-1)
    w_entropy_sum += float(we.sum())
    peaked_cnt += int((w.max(-1).values > 0.5).sum())
    # token-class specialization
    tr = truth.reshape(-1).numpy()
    hh = eh.reshape(-1, N)
    for t, tid in enumerate(tr):
        c = cls_of(tok_str[tid]); cls_cnt[c] += 1
        cls_hits[:, c] += hh[t].astype(float)

rep = {}
for k in range(N):
    acc1 = top1_hits[k]/n_pos; acc5 = top5_hits[k]/n_pos
    rep[k] = {
        "top1": round(acc1,4), "top5": round(acc5,4),
        "mean_w": round(float(w_sum[k]/n_pos),4),
        "w_when_right": round(float(w_right[k]/max(n_right[k],1)),4),
        "w_when_wrong": round(float(w_wrong[k]/max(n_wrong[k],1)),4),
        "argmax_rate": round(float(argmax_cnt[k]/n_pos),4),
        "solo_bpc": round(float(solo_ce[k]/n_pos/math.log(2)),4),
        "rescues": int(rescue[k]),
        "p_truth": round(float(cal_p_truth[k]/n_pos),5),
        "top_preds": top_pred[k].most_common(8),
    }
    rep[k]["cls_acc"] = {cls_names[c]: round(cls_hits[k,c]/max(cls_cnt[c],1),4)
                         for c in range(6) if cls_cnt[c] > 0}

pairs = [(k,j) for k in range(N) for j in range(k+1,N)]
res = {
  "n_positions": n_pos, "tau0": TAU, "checkpoint_step": int(ck.get("step",-1)),
  "pooled_top1": round(float(pooled_hit)/n_pos,4),
  "oracle_top1": round(float(oracle1)/n_pos,4),
  "oracle_top5": round(float(oracle5)/n_pos,4),
  "outvoted": int(outvoted),
  "w_entropy": round(w_entropy_sum/n_pos,4),
  "peaked_positions": round(peaked_cnt/n_pos,4),
  "region_omega": (omega_sum/len(starts)).tolist(),
  "overlap5_matrix": [[round(float(overlap[k,j])/(n_pos*5),3) for j in range(N)] for k in range(N)],
  "err_agree_matrix": [[round(float(err_same[k,j])/n_pos,3) for j in range(N)] for k in range(N)],
  "dots": rep,
}
Path(args.out).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
