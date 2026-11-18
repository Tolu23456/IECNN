"""Per-dot specialisation + next-token accuracy for an A/B arm.

Tolu's question (2026-11-19): "are the dots just becoming clones, or are they
specialising? And how is the next-token accuracy?" Trains one arm with the
standard protocol, then answers both with numbers.

WHY THIS VERSION EXISTS (the first one measured the wrong thing)
----------------------------------------------------------------
The first audit judged each dot by its own logits (per_dot_logits). Those are
a TRAIN-TIME instrument: with per_dot_logits=False the training loss never
touches them, so per-dot bpc/top-1/agreement described an untrained side
output, not what the dot believes. It reported per-dot bpc 7.3-8.3 while the
committee was at 2.02 — meaningless.

The honest instrument is a ONE-HOT POOL (`pool_only_dot=k`): force the gate to
pick dot k and run the SAME trained pool->head path. "bpc if the gate had to
use dot k alone" is exactly what a clone test needs, and it needs no probe
model, no extra parameters and no retraining.

What this reports:
  * committee: bpc, top1, top5 on the val slice;
  * per-dot quality ALONE (one-hot pool): bpc, top1, top5;
  * clone test on the trained path: top-1 agreement and JS between the three
    one-hot pooled distributions (clones => agreement ~1, JS ~0);
  * drop-one: bpc when dot k is removed from the pool and the rest renormalise
    (how much the committee needs that dot);
  * where each dot LOOKS: read entropy, mean distance back, share of positions
    parked on the current token, and pairwise JS between read distributions;
  * does the gate know: mean weight per dot, and the weight it puts on the dot
    that had the lowest loss at that position, against uniform 1/N;
  * accuracy by confidence decile (is the model calibrated?).

Runs anywhere torch does; on Kaggle it is `scripts/kaggle_audit.py`.
Env: ARM_NAME, K_BUDGET, AUDIT_WINDOWS, plus ab_small's ARM_* data knobs.
"""
from __future__ import annotations

import json
import math
import os
import sys

ARM = os.environ.get("ARM_NAME", "iecnn-n3-x5-p1-ln-decay")

# ab_small reads its budget/arm list from argv at import time; give it a
# budget and no arm list so it loads data and defines everything but does not
# train anything by itself.
sys.argv = [sys.argv[0], os.environ.get("K_BUDGET", "180"), ""]

import torch                                                   # noqa: E402
import torch.nn.functional as F                                 # noqa: E402

import scripts.ab_small as A                                    # noqa: E402
from models.iecnn import build_iecnn                            # noqa: E402

N_WINDOWS = int(os.environ.get("AUDIT_WINDOWS", "96"))
T, V, DEV = A.T, A.V, A.DEV


def js(p, q, eps=1e-12):
    """Jensen-Shannon divergence between two batches of distributions.

    Masked reads can be exactly zero in some slots, and `0 * log(0)` is nan,
    so the first factor is clamped: eps*log(eps) is ~1e-11 and cannot move a
    divergence that is O(0.1). Clamped at 0 because JS is non-negative and
    the eps term can otherwise push it a hair below.
    """
    m = 0.5 * (p + q)
    kl = lambda a, b: (a.clamp_min(eps)
                       * ((a + eps) / (b + eps)).log()).sum(-1)
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).clamp_min(0.0).mean().item()


def nll_of(model, x, y, **kw):
    lp = model(x, **kw)["final_log_probs"]                     # [1,T,V]
    n = F.nll_loss(lp.reshape(-1, V), y.reshape(-1), reduction="none")
    top = lp.topk(5, dim=-1).indices
    return (n.reshape(-1),                                          # [T]
            top[..., 0].eq(y).reshape(-1),                          # [T]
            top.eq(y.unsqueeze(-1)).any(-1).reshape(-1))            # [T]


def main():
    kind, cfg_fn, opts = A.ARMS[ARM]
    if kind != "iecnn":
        raise SystemExit("this audit is for iecnn arms only")
    model = build_iecnn(cfg_fn(), V).to(DEV)
    print(f"=== AUDIT {ARM} ({kind}) budget {A.BUDGET}s", flush=True)
    row = A.run(name=ARM + "-audit", kind=kind, model=model, **opts)
    print(f"trained: bpc_val {row['bpc_final']:.4f} in {row['steps']} steps",
          flush=True)
    model.eval()

    reg = model.regions[0]
    dots = reg.dots
    n = dots.N

    wins = []
    for w in range(N_WINDOWS):
        i = w * T * A.VALSTRIDE
        x = A.val[i:i + T].unsqueeze(0).to(DEV)
        y = A.val[i + 1:i + 1 + T].unsqueeze(0).to(DEV)
        if x.shape[1] < T:
            break
        wins.append((x, y))

    # ---- committee + gate + where the dots look -------------------------
    tot_nll = 0.0
    ntok = 0
    top1 = top5 = 0
    weight = torch.zeros(n)
    best_weight_hit = 0.0
    nbest = 0
    gate_argmax_is_best = 0
    read_ent = torch.zeros(n)
    dist_back = torch.zeros(n)
    frac_end = torch.zeros(n)
    read_js = torch.zeros(n, n)
    conf_p, conf_ok = [], []

    # one-hot pool per dot, and drop-one, accumulated over the same windows
    one_nll = torch.zeros(n)
    one_top1 = torch.zeros(n)
    one_top5 = torch.zeros(n)
    drop_nll = torch.zeros(n)
    one_arg = [[] for _ in range(n)]
    one_prob = [[] for _ in range(n)]
    gate_rows = []                 # [W][T,N]

    with torch.no_grad():
        for x, y in wins:
            out = model(x)
            lp = out["final_log_probs"]
            wt = out["per_pass_last_region"][-1]["weights"].reshape(-1, n)
            nll, hit, h5 = nll_of(model, x, y)
            tot_nll += nll.sum().item()
            ntok += nll.numel()
            top1 += hit.sum().item()
            top5 += h5.sum().item()
            weight += wt.sum(0).cpu()
            gate_rows.append(wt.cpu())
            prob = lp.exp().max(-1).values.reshape(-1)
            conf_p.append(prob.cpu())
            conf_ok.append(hit.cpu())

            # --- where each dot looks (the trained read, same softmax) ----
            rows_r = A.region1_rows(model.embed(x))
            wk, _wv, valid = reg._window_memory(rows_r, x.device)
            q = dots._sel(0, dots.q)
            sc = torch.einsum("bfwj,nj->bfnw", wk, q) / math.sqrt(dots.d_q)
            sc = sc.masked_fill(~valid[None, :, None, :], float("-inf"))
            p = torch.nan_to_num(sc.softmax(-1), nan=0.0)
            p = p.reshape(-1, n, p.shape[-1])                # [T,N,w]
            ent = -(p.clamp_min(1e-12).log() * p).sum(-1)     # [T,N]
            read_ent += ent.sum(0).cpu()
            offs = reg._window_offsets(x.device, 0)
            idx = p.argmax(-1)                                # [T,N]
            dist_back += (-offs)[idx].clamp_min(0).sum(0).cpu()
            frac_end += (idx == p.shape[-1] - 1).sum(0).cpu()
            for a in range(n):
                for b in range(a + 1, n):
                    v = js(p[:, a], p[:, b]) * p.shape[0]
                    read_js[a, b] += v
                    read_js[b, a] = read_js[a, b]

            # --- each dot ALONE through the trained path ------------------
            # ONE forward per dot buys the loss, the top-5 and the full
            # distribution; the first version ran three.
            row_nll = torch.zeros(T, n)
            for k in range(n):
                lk = model(x, pool_only_dot=k)["final_log_probs"]  # [1,T,V]
                nk = F.nll_loss(lk.reshape(-1, V), y.reshape(-1),
                                reduction="none").reshape(-1)
                topk = lk.topk(5, dim=-1).indices
                one_nll[k] += nk.sum().item()
                one_top1[k] += topk[..., 0].eq(y).sum().item()
                one_top5[k] += topk.eq(y.unsqueeze(-1)).any(-1).sum().item()
                row_nll[:, k] = nk.cpu()
                one_arg[k].append(lk.argmax(-1).reshape(-1).cpu())
                one_prob[k].append(lk.exp().reshape(-1, V).cpu())
                # drop-one on the same window (renormalised pool)
                dk = nll_of(model, x, y, ablate_dot=k)[0]
                drop_nll[k] += dk.sum().item()

            best = row_nll.argmin(-1).to(DEV)                   # [T]
            best_weight_hit += wt.gather(1, best[:, None]).sum().item()
            nbest += T
            gate_argmax_is_best += int((wt.argmax(-1) == best).sum().item())

    if not ntok:
        print("no val windows", flush=True)
        return

    # ---- clone test on the trained path ---------------------------------
    agree = torch.zeros(n, n)
    js_vocab = torch.zeros(n, n)
    for a in range(n):
        aa = torch.cat(one_arg[a])
        for b in range(n):
            bb = torch.cat(one_arg[b])
            agree[a, b] = aa.eq(bb).float().mean().item()
    for a in range(n):
        for b in range(a + 1, n):
            pa = torch.cat(one_prob[a])
            pb = torch.cat(one_prob[b])
            v = js(pa, pb)
            js_vocab[a, b] = js_vocab[b, a] = v

    rep = {
        "arm": ARM, "params": row["params"], "steps": row["steps"],
        "bpc": tot_nll / ntok / math.log(2),
        "top1": top1 / ntok, "top5": top5 / ntok,
        "bpc_at_eq": row["bpc_final"],
        "dots_alone": {
            "bpc": (one_nll / ntok / math.log(2)).tolist(),
            "top1": (one_top1 / ntok).tolist(),
            "top5": (one_top5 / ntok).tolist(),
        },
        "drop_one_bpc": (drop_nll / ntok / math.log(2)).tolist(),
        "mean_gate_weight": (weight / ntok).tolist(),
        "gate_weight_on_best_dot": best_weight_hit / max(nbest, 1),
        "gate_argmax_eq_best_frac": gate_argmax_is_best / max(nbest, 1),
        "agree_top1": agree.tolist(),
        "js_vocab_nats": js_vocab.tolist(),
        "read_entropy_nats": (read_ent / ntok).tolist(),
        "mean_dist_back": (dist_back / ntok).tolist(),
        "frac_read_at_current_token": (frac_end / ntok).tolist(),
        "js_read_nats": (read_js / ntok).tolist(),
    }
    cp = torch.cat(conf_p)
    co = torch.cat(conf_ok).float()
    order = cp.argsort()
    dec = []
    for qd in range(10):
        sl = order[int(qd * len(order) / 10):int((qd + 1) * len(order) / 10)]
        dec.append(round(co[sl].mean().item(), 3))
    rep["accuracy_by_confidence_decile"] = dec

    # ---- print -----------------------------------------------------------
    print("\n=== DOT REPORT", flush=True)
    print(f"{ARM}  params {rep['params']}  bpc {rep['bpc']:.4f}  "
          f"top1 {rep['top1']:.3f}  top5 {rep['top5']:.3f}", flush=True)
    print("dot  bpc_alone  top1  top5   drop_one  read_ent  dist_back  "
          "frac_at_t  gate_w", flush=True)
    for a in range(n):
        print(f"{a:>3}  {rep['dots_alone']['bpc'][a]:>9.4f}  "
              f"{rep['dots_alone']['top1'][a]:.3f} "
              f"{rep['dots_alone']['top5'][a]:.3f}  "
              f"{rep['drop_one_bpc'][a]:>9.4f}  "
              f"{rep['read_entropy_nats'][a]:>8.3f}  "
              f"{rep['mean_dist_back'][a]:>9.2f}  "
              f"{rep['frac_read_at_current_token'][a]:>9.3f}  "
              f"{rep['mean_gate_weight'][a]:.3f}", flush=True)
    print("\nclone test, trained path (upper: top1 agreement, "
          "lower: JS nats)", flush=True)
    for a in range(n):
        print("  ".join(
            f"{(rep['agree_top1'][a][b] if b >= a else rep['js_vocab_nats'][a][b]):.3f}"
            for b in range(n)), flush=True)
    print("\nread JS nats (how differently they look at the window)", flush=True)
    for a in range(n):
        print("  ".join(f"{rep['js_read_nats'][a][b]:.3f}"
                        for b in range(n)), flush=True)
    print(f"gate weight on the best dot {rep['gate_weight_on_best_dot']:.3f} "
          f"(uniform {1.0 / n:.3f}); gate argmax == best dot "
          f"{rep['gate_argmax_eq_best_frac']:.3f} "
          f"(chance {1.0 / n:.3f})", flush=True)
    print("accuracy by confidence decile", dec, flush=True)

    with open(os.environ.get("AUDIT_OUT", "dot_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("JSON " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
