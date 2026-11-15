"""Selective-activation experiment (Tolu's roadmap, 2026-11-16): can the
verifier's PRIOR-pass scores pick a top-r sub-committee whose pooled vote
beats the all-8 pooled vote?

Strategies compared at each r = 1..N:
  all       — the model's own pooled output (baseline, r = N)
  verifier  — pool only the r dots with the highest PASS-1 verifier scores
              (prior-pass features only: legal for a real selective pass)
  random    — random r dots (control, averaged over draws)
  oracle    — the best r dots for THIS position by final correctness
              (upper bound; not realizable)

Also reports AUROC(pass-1 verifier score -> final dot correctness) per dot.

Inference only; CPU-friendly. Rerun on every E6 checkpoint — the verifier
head is fresh at step 0 and needs training before the numbers mean anything.
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, "/content/IECNN"); os.chdir("/content/IECNN")
import torch, numpy as np
from pathlib import Path
from UBM import UBM
from config import load_config
from train.train import build_model
from train.evaluate import prepare_text
from models import consensus

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/vocab3_gpu_seed0_checkpoint.pt")
ap.add_argument("--config", default="configs/e6_consensus_v2_vocab3_warmstart.yaml")
ap.add_argument("--tau", type=float, default=0.894)
ap.add_argument("--windows", type=int, default=200)
ap.add_argument("--context", type=int, default=256)
ap.add_argument("--n-random", type=int, default=16)
ap.add_argument("--out", default="results/selective_topr.json")
args = ap.parse_args()

cfg = load_config(args.config)
ubm = UBM.load(cfg["ubm"]["vocab_file"])
V = len(ubm.bases)
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
state = ck.get("best_state") or ck["model"]
model = build_model(cfg, V)
model.load_state_dict(state, strict=False)
model.eval()
torch.set_grad_enabled(False)

val = Path("data/val.txt").read_text()
ids, _ = prepare_text(ubm, val)
L = args.context
starts = [s for s in range(0, len(ids) - 1, L)
          if s + L + 1 <= len(ids)][:args.windows]
N = 8

# accumulators per r: hits for each strategy
R = list(range(1, N + 1))
hits = {s: {r: 0 for r in R} for s in ("all", "verifier", "random", "oracle")}
rnd_hits_sum = {r: 0.0 for r in R}
n_pos = 0
v_correct = np.zeros(N)     # pass-1 verifier score summed where dot final-correct
v_wrong = np.zeros(N)
c_correct = np.zeros(N)     # count correct / wrong per dot (for AUROC)
c_wrong = np.zeros(N)
pooled_base = 0

B = 4
rng = np.random.default_rng(0)
for gi in range(0, len(starts), B):
    group = starts[gi:gi + B]
    xs = [ids[s:s + L] for s in group]; ys = [ids[s + 1:s + 1 + L] for s in group]
    x = torch.tensor(xs, dtype=torch.long); y = torch.tensor(ys, dtype=torch.long)
    out = model(x, tau0=args.tau, floor=0.0)
    pp = out["per_pass_last_region"]
    log_pi = pp[-1]["log_pi"]                      # [B,T,N,V]
    w = pp[-1]["weights"]                          # [B,T,N]
    v0 = pp[0]["verifier"]                         # [B,T,N] pass-1 scores
    final = out["final_log_probs"]
    truth = y
    hit = (log_pi.argmax(-1) == truth[:, :, None])   # [B,T,N] final correctness
    p_hit = (final.argmax(-1) == truth)
    n_pos += truth.numel()
    pooled_base += int(p_hit.sum())
    # verifier AUROC bookkeeping (per dot): mean score where correct vs wrong
    vn = v0.numpy(); hn = hit.numpy()
    for k in range(N):
        v_correct[k] += vn[:, :, k][hn[:, :, k]].sum()
        v_wrong[k] += vn[:, :, k][~hn[:, :, k]].sum()
        c_correct[k] += int(hn[:, :, k].sum())
        c_wrong[k] += int((~hn[:, :, k]).sum())
    # strategies
    hits["all"][N] += int(p_hit.sum())
    order = np.argsort(-vn, axis=-1)                 # [B,T,N] dot ranks
    for r in R:
        # verifier top-r: renormalized log-linear pool of selected dots
        sel = order[:, :, :r]                        # [B,T,r]
        w_sel = torch.gather(w, -1, torch.tensor(sel))  # [B,T,r]
        w_sel = w_sel / w_sel.sum(-1, keepdim=True).clamp_min(1e-9)
        lp_sel = torch.gather(
            log_pi, 2, torch.tensor(sel)[:, :, :, None].expand(-1, -1, -1, V))
        log_b = consensus.pool_log_linear(lp_sel, w_sel)   # [B,T,V]
        hits["verifier"][r] += int((log_b.argmax(-1) == truth).sum())
        # random-r control (averaged over draws)
        acc = 0.0
        for _ in range(args.n_random):
            rnd = np.argsort(rng.random((B, L, N)), axis=-1)[:, :, :r]
            wr = torch.gather(w, -1, torch.tensor(rnd))
            wr = wr / wr.sum(-1, keepdim=True).clamp_min(1e-9)
            lr = torch.gather(
                log_pi, 2, torch.tensor(rnd)[:, :, :, None].expand(-1, -1, -1, V))
            acc += float((consensus.pool_log_linear(lr, wr).argmax(-1)
                          == truth).sum())
        rnd_hits_sum[r] += acc / args.n_random
        # oracle-r: for each position pick the r dots that are correct
        # (tie-broken by weight desc); deterministic ordering, upper bound
        hnp = hn.astype(np.int64)
        okey = hnp * 1000 + np.argsort(-w.numpy(), axis=-1).argsort(-1)
        osel = np.argsort(-okey, axis=-1)[:, :, :r]
        wo = torch.gather(w, -1, torch.tensor(osel))
        wo = wo / wo.sum(-1, keepdim=True).clamp_min(1e-9)
        lo = torch.gather(
            log_pi, 2, torch.tensor(osel)[:, :, :, None].expand(-1, -1, -1, V))
        hits["oracle"][r] += int((consensus.pool_log_linear(lo, wo).argmax(-1)
                                  == truth).sum())

def auroc(pos_sum, neg_sum, npos, nneg):
    """AUROC from the pooled score distributions' means is not exact; report
    the simpler and more honest separation metric: mean(score | correct) -
    mean(score | wrong), plus per-dot hit-rate lift."""
    if npos == 0 or nneg == 0:
        return None
    return pos_sum / npos - neg_sum / nneg

verifier_sep = [auroc(v_correct[k], v_wrong[k], c_correct[k], c_wrong[k])
                for k in range(N)]
res = {
    "ckpt": args.ckpt,
    "n_positions": n_pos,
    "pooled_all_top1": pooled_base / n_pos,
    "strategies": {s: {str(r): h / n_pos for r, h in d.items()}
                   for s, d in hits.items()},
    "verifier_separation_mean_score_correct_minus_wrong": verifier_sep,
    "note": "verifier = pass-1 scores (prior-pass features only). 'verifier' "
            "rows are the selective-activation numbers; 'oracle' is the "
            "upper bound. AUROC-like separation >0 means the verifier "
            "scores dots that end up correct higher than wrong ones.",
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "note"}, indent=1)[:2000])
