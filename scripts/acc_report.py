"""Next-token accuracy for ANY arm, on the standard protocol val slice.

Tolu's question "how is accuracy next token" deserves a reference point: a top-1
of 59% means nothing without the transformer's top-1 on the same bytes. This
trains one arm with the exact A/B recipe (`A.run`) and reports

    bpc, top-1, top-5, accuracy by confidence decile,
    top-1 split by how much context the model has (first tokens vs the rest),

so the accuracy of the dot model and of tf-swa are directly comparable.

Deliberately arm-agnostic: no dot-specific machinery, so it runs on a
transformer too. Env: ARM_NAME, K_BUDGET, AUDIT_WINDOWS, plus ab_small's
ARM_* data knobs.
"""
from __future__ import annotations

import json
import math
import os
import sys

ARM = os.environ.get("ARM_NAME", "tf-swa")

sys.argv = [sys.argv[0], os.environ.get("K_BUDGET", "180"), ""]

import torch                                                   # noqa: E402
import torch.nn.functional as F                                 # noqa: E402

import scripts.ab_small as A                                    # noqa: E402
from models.iecnn import build_iecnn                            # noqa: E402
from train.train import build_model                             # noqa: E402

N_WINDOWS = int(os.environ.get("AUDIT_WINDOWS", "96"))
T, V, DEV = A.T, A.V, A.DEV


def main():
    kind, cfg_fn, opts = A.ARMS[ARM]
    cfg = cfg_fn()
    model = (build_iecnn(cfg, V) if kind == "iecnn"
             else build_model(cfg, V)).to(DEV)
    print(f"=== ACC {ARM} ({kind}) budget {A.BUDGET}s", flush=True)
    row = A.run(name=ARM + "-acc", kind=kind, model=model, **opts)
    print(f"trained: bpc_val {row['bpc_final']:.4f} in {row['steps']} steps",
          flush=True)
    model.eval()

    wins = []
    for w in range(N_WINDOWS):
        i = w * T * A.VALSTRIDE
        x = A.val[i:i + T].unsqueeze(0).to(DEV)
        y = A.val[i + 1:i + 1 + T].unsqueeze(0).to(DEV)
        if x.shape[1] < T:
            break
        wins.append((x, y))

    nll_sum = 0.0
    ntok = 0
    hit1 = hit5 = 0
    conf, ok = [], []
    # top-1 by amount of context available: positions 0..15 of a window have
    # almost no history, positions 32+ have plenty. A model that is only good
    # because of long context shows it here.
    buckets = {"0-15": [0, 0], "16-63": [0, 0], "64+": [0, 0]}
    with torch.no_grad():
        for x, y in wins:
            lp = model(x)["final_log_probs"]
            n = F.nll_loss(lp.reshape(-1, V), y.reshape(-1),
                           reduction="none").reshape(-1)
            nll_sum += n.sum().item()
            ntok += n.numel()
            top = lp.topk(5, dim=-1).indices
            h1 = top[..., 0].eq(y).reshape(-1)
            hit1 += h1.sum().item()
            hit5 += top.eq(y.unsqueeze(-1)).any(-1).sum().item()
            conf.append(lp.exp().max(-1).values.reshape(-1).cpu())
            ok.append(h1.cpu())
            for name, (lo, hi) in buckets.items():
                sl = slice(lo, min(hi, T))
                buckets[name][0] += h1[sl].sum().item()
                buckets[name][1] += h1[sl].numel()

    cp = torch.cat(conf)
    co = torch.cat(ok).float()
    order = cp.argsort()
    dec = []
    for q in range(10):
        sl = order[int(q * len(order) / 10):int((q + 1) * len(order) / 10)]
        dec.append(round(co[sl].mean().item(), 3))

    rep = {
        "arm": ARM, "kind": kind, "params": row["params"],
        "steps": row["steps"], "bpc": nll_sum / ntok / math.log(2),
        "bpc_at_eq": row["bpc_final"],
        "top1": hit1 / ntok, "top5": hit5 / ntok,
        "top1_by_context": {k: round(v[0] / max(v[1], 1), 3)
                            for k, v in buckets.items()},
        "accuracy_by_confidence_decile": dec,
    }
    print(f"\n=== ACC REPORT {ARM}  params {rep['params']}  "
          f"bpc {rep['bpc']:.4f}  top1 {rep['top1']:.3f}  "
          f"top5 {rep['top5']:.3f}", flush=True)
    print(f"top1 by context {rep['top1_by_context']}", flush=True)
    print(f"accuracy by confidence decile {dec}", flush=True)
    with open(os.environ.get("AUDIT_OUT", "acc_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("JSON " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
