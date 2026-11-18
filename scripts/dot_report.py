"""Per-dot specialisation + next-token accuracy for an A/B arm.

Tolu's question (2026-11-19): "are the dots just becoming clones, or are they
specialising? And how is the next-token accuracy?" The pointer audit in
ab_small only reports argmax diversity, which can look perfectly diverse while
every dot computes the same function. This trains one arm with the standard
protocol and then answers both questions with numbers:

  * the committee's own quality: bpc, top-1, top-5 on the val slice;
  * read-level diversity: per-dot slot entropy, Jensen-Shannon between dots'
    read distributions, where each dot looks (distance histogram);
  * prediction-level diversity: top-1 agreement between every pair of dots,
    JS between their vocab distributions (a clone pair has JS ~ 0 and
    agreement ~ 1);
  * per-dot quality: bpc and top-1 of each dot ALONE, against the committee;
  * does the gate know: mean pooling weight per dot, and the weight the gate
    puts on the dot that happened to be right;
  * specialisation by target type: copy / space / digit / other, per dot.

Runs anywhere torch does; on Kaggle it is `scripts/kaggle_audit.py`.
Env: ARM_NAME (default the best arm on record), K_BUDGET, plus the usual
ARM_* data knobs that ab_small reads.
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
    """Jensen-Shannon divergence between two batches of distributions."""
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * ((a + eps) / (b + eps)).log()).sum(-1)
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).mean().item()


def main():
    kind, cfg_fn, opts = A.ARMS[ARM]
    if kind != "iecnn":
        raise SystemExit("this audit is for iecnn arms only")
    cfg = cfg_fn()
    model = build_iecnn(cfg, V).to(DEV)
    print(f"=== AUDIT {ARM} ({kind}) budget {A.BUDGET}s", flush=True)
    row = A.run(name=ARM + "-audit", kind=kind, model=model, **opts)
    print(f"trained: bpc_val {row['bpc_final']:.4f} in {row['steps']} steps",
          flush=True)

    # Per-dot logits are a train-time instrument (one extra GEMM); the
    # parameter set is identical, so the trained weights load straight in and
    # the POOLED path is bit-identical — only the diagnostic head is added.
    cfg_pi = cfg_fn()
    cfg_pi["model"]["per_dot_logits"] = True
    cfg_pi["model"]["head_last_only"] = False
    probe = build_iecnn(cfg_pi, V).to(DEV)
    probe.load_state_dict(model.state_dict(), strict=True)
    probe.eval()
    reg = probe.regions[0]
    dots = reg.dots
    n = dots.N

    # ---- accumulate over the val slice --------------------------------
    acc = dict(tot_nll=0.0, ntok=0, top1=0, top5=0,
               dot_nll=torch.zeros(n), dot_top1=torch.zeros(n),
               agree=torch.zeros(n, n), js_vocab=torch.zeros(n, n),
               read_ent=torch.zeros(n), read_js=torch.zeros(n, n),
               weight=torch.zeros(n), weight_on_best=0.0, nbest=0,
               by_type={})
    types = {"copy": [], "space": [], "digit": [], "other": []}
    conf_probs, conf_ok = [], []

    with torch.no_grad():
        for w in range(N_WINDOWS):
            i = w * T * A.VALSTRIDE
            x = A.val[i:i + T].unsqueeze(0).to(DEV)
            y = A.val[i + 1:i + 1 + T].unsqueeze(0).to(DEV)
            if x.shape[1] < T:
                break
            out = probe(x)
            lp = out["final_log_probs"]                        # [1,T,V]
            pp = out["per_pass_last_region"][-1]
            log_pi = pp["log_pi"]                              # [1,T,N,V]
            wt = pp["weights"].reshape(-1, n)                  # [T,N]

            acc["tot_nll"] += F.nll_loss(lp.reshape(-1, V), y.reshape(-1),
                                         reduction="sum").item()
            acc["ntok"] += y.numel()
            top = lp.topk(5, dim=-1).indices                   # [1,T,5]
            hit = top.eq(y.unsqueeze(-1))
            acc["top1"] += hit[..., 0].sum().item()
            acc["top5"] += hit.any(-1).sum().item()
            prob = lp.exp()
            conf_probs.append(prob.max(-1).values.reshape(-1).cpu())
            conf_ok.append(hit[..., 0].reshape(-1).cpu())

            # per-dot quality
            # [1,T,N,V] flattens t-major, so repeat the target per dot
            # ALONG T (dot-major repeat would pair every dot with the wrong
            # position and make every dot look equally bad)
            y_flat = y.reshape(-1, 1).repeat(1, n).reshape(-1)
            dnll = F.nll_loss(log_pi.reshape(-1, V), y_flat,
                              reduction="none").reshape(1, T, n)
            acc["dot_nll"] += dnll.sum(dim=(0, 1)).cpu()
            dhit = log_pi.argmax(-1).eq(y.unsqueeze(-1))        # [1,T,N]
            acc["dot_top1"] += dhit.sum(dim=(0, 1)).cpu()

            # prediction-level diversity
            dp = log_pi.exp().reshape(-1, n, V)                 # [T,N,V]
            arg = dhit[0].float()                               # [T,N]
            for a in range(n):
                acc["agree"][a] += (arg == arg[:, a:a + 1]).sum(0).cpu()
            for a in range(n):
                for b in range(a + 1, n):
                    acc["js_vocab"][a, b] += js(dp[:, a], dp[:, b]) * dp.shape[0]
                    acc["js_vocab"][b, a] = acc["js_vocab"][a, b]

            # read-level diversity: the same softmax the dots actually use
            rows = A.region1_rows(probe.embed(x))
            wk, _wv, valid = reg._window_memory(rows, x.device)
            q = dots._sel(0, dots.q)
            sc = torch.einsum("bfwj,nj->bfnw", wk, q) / math.sqrt(dots.d_q)
            sc = sc.masked_fill(~valid[None, :, None, :], float("-inf"))
            p = torch.nan_to_num(sc.softmax(-1), nan=0.0)        # [1,T,N,w]
            p = p.reshape(-1, n, -1)
            ent = -(p.clamp_min(1e-12).log() * p).sum(-1)       # [T,N]
            acc["read_ent"] += ent.sum(0).cpu()
            for a in range(n):
                for b in range(a + 1, n):
                    acc["read_js"][a, b] += js(p[:, a], p[:, b]) * p.shape[0]
                    acc["read_js"][b, a] = acc["read_js"][a, b]

            # does the gate put its weight on the dot that was right?
            acc["weight"] += wt.sum(0).cpu()
            per_dot_ok = dhit[0].float()                        # [T,N]
            any_ok = per_dot_ok.sum(-1) > 0
            if any_ok.any():
                best = dnll[0][any_ok].argmin(-1)
                acc["weight_on_best"] += wt[any_ok].gather(
                    1, best[:, None]).sum().item()
                acc["nbest"] += int(any_ok.sum())

            # specialisation by target type
            ymask = {"copy": y[0] == x[0], "space": y[0] == 32,
                     "digit": (y[0] >= 48) & (y[0] <= 57)}
            ymask["other"] = ~(ymask["copy"] | ymask["space"] | ymask["digit"])
            for name, m in ymask.items():
                if m.any():
                    d = types[name]
                    d.append((dnll[0][m].mean(0).cpu(), dhit[0][m].float()
                              .mean(0).cpu()))
    if not acc["ntok"]:
        print("no val windows", flush=True)
        return

    ntok = acc["ntok"]
    rep = {
        "arm": ARM,
        "params": row["params"], "steps": row["steps"],
        "bpc": acc["tot_nll"] / ntok / math.log(2),
        "top1": acc["top1"] / ntok, "top5": acc["top5"] / ntok,
        "bpc_at_eq": row["bpc_final"],
    }
    rep["dots"] = {
        "bpc_alone": (acc["dot_nll"] / ntok / math.log(2)).tolist(),
        "top1_alone": (acc["dot_top1"] / ntok).tolist(),
        "read_entropy_nats": (acc["read_ent"] / ntok).tolist(),
        "mean_gate_weight": (acc["weight"] / ntok).tolist(),
        "agree_top1": (acc["agree"] / ntok).tolist(),
        "js_vocab_nats": (acc["js_vocab"] / ntok).tolist(),
        "js_read_nats": (acc["read_js"] / ntok).tolist(),
    }
    rep["gate_weight_on_best_dot"] = (acc["weight_on_best"]
                                     / max(acc["nbest"], 1))
    rep["types"] = {}
    for name, d in types.items():
        if d:
            loss = torch.stack([t[0] for t in d]).mean(0)
            hit = torch.stack([t[1] for t in d]).mean(0)
            rep["types"][name] = {"bpc": (loss / math.log(2)).tolist(),
                                  "top1": hit.tolist()}
    cp = torch.cat(conf_probs)
    co = torch.cat(conf_ok).float()
    order = cp.argsort()
    dec = []
    for q in range(10):
        idx = order[int(q * len(order) / 10):int((q + 1) * len(order) / 10)]
        dec.append(round(co[idx].mean().item(), 3))
    rep["accuracy_by_confidence_decile"] = dec

    print("\n=== DOT REPORT", flush=True)
    print(f"{ARM}  params {rep['params']}  bpc {rep['bpc']:.4f}  "
          f"top1 {rep['top1']:.3f}  top5 {rep['top5']:.3f}", flush=True)
    hdr = "dot  bpc_alone top1_alone read_ent gate_w"
    print(hdr, flush=True)
    for a in range(n):
        print(f"{a:>3}  {rep['dots']['bpc_alone'][a]:>9.4f} "
              f"{rep['dots']['top1_alone'][a]:>10.3f} "
              f"{rep['dots']['read_entropy_nats'][a]:>8.3f} "
              f"{rep['dots']['mean_gate_weight'][a]:>6.3f}", flush=True)
    print("\npairwise (upper: top1 agreement, lower: JS nats)", flush=True)
    for a in range(n):
        print("  ".join(
            f"{(rep['dots']['agree_top1'][a][b] if b >= a else rep['dots']['js_vocab_nats'][a][b]):.3f}"
            for b in range(n)), flush=True)
    print("\nread JS (how different their attention is)", flush=True)
    for a in range(n):
        print("  ".join(f"{rep['dots']['js_read_nats'][a][b]:.3f}"
                        for b in range(n)), flush=True)
    print(f"gate weight on the best dot {rep['gate_weight_on_best_dot']:.3f} "
          f"(uniform = {1.0 / n:.3f})", flush=True)
    print("accuracy by confidence decile", dec, flush=True)
    for name, d in rep["types"].items():
        print(f"  {name:<6} bpc {[round(v, 3) for v in d['bpc']]} "
              f"top1 {[round(v, 3) for v in d['top1']]}", flush=True)
    with open(os.environ.get("AUDIT_OUT", "dot_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("JSON " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
