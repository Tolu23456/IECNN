"""Does SELECTION among earlier occurrences pay on the TRAINED channel?

`iecnn-copy-vote` priced extension-based selection (extpick/extvote) at +0.0132
out-of-fold over the single pointer, but on the checkpoint that never trained
the channel and with a gate fitted to the channel. That is a ceiling on an
unadapted model, and the ledger's history says such a number can over-predict
the delivered gain by 4.5x.

This probe asks the question that actually decides whether `sc3ev` is worth a
GPU session, on the SAME trained `sc3e` weights the slope ablation used, with
every parameter frozen:

  * `p` is the model's own distribution with the gate forced shut (identity
    mixture), so the recomposition below is exact, not an approximation;
  * the gate (a, b, cap) is the checkpoint's learned gate, unchanged;
  * the only thing that varies is WHICH earlier occurrence's follower is
    copied -- the model's own pointer, or a selector.

That is the same shape of measurement as the slope ablation, which predicted
the trained two-seed difference to 0.001, so its verdict is worth acting on. It
is still an eval-time substitution: a trained `sc3ev` would also adapt `p` and
the gate to the selector, so a small positive here is a LOWER bound while a ZERO
is a real reason not to spend the session.

Selectors (from `probe_copy_vote`, reused rather than reimplemented):

  single   the most recent earlier occurrence (what the model does)
  extpick  the occurrence whose match extends furthest back
  extvote  majority among the occurrences at the maximum extension
  oracle   NOT deployable: the right byte if ANY occurrence predicts it. This
           bounds the whole family, so "selection is exhausted" and "the
           selector is the problem" stay distinguishable.

Each selector is reported with two gate inputs: the model's own extension (what
a drop-in replacement feeds the frozen gate) and the SELECTED occurrence's
extension (what the gate would see if it were computed from the chosen pointer).

Usage: PYTHONPATH=. python3 scripts/probe_selection_trained.py ckpt.pt [...]
Reads ARM_T / ARM_VAL / ARM_NVAL / ARM_VALW / ARM_VALSTRIDE / COPY_K / VOTE_M /
COPY_EXTRA_CAP.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch

from models.iecnn import build_iecnn
from models.baselines import build_transformer_baseline
from scripts.probe_copy_vote import derive, window_candidates

V = 256
T = int(os.environ.get("ARM_T", "256"))
K = int(os.environ.get("COPY_K", "3"))
M = int(os.environ.get("VOTE_M", "3"))
CAP = int(os.environ.get("COPY_EXTRA_CAP", "8"))
VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", "5_000_000"))
VALW = int(os.environ.get("ARM_VALW", "160"))
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", "122"))
SELECTORS = ("single", "extpick", "extvote", "oracle")


def bpc(p_true: torch.Tensor, mass: torch.Tensor, g: torch.Tensor) -> float:
    """bpc of the mixture with the model's own 1e-9 clamp."""
    mix = (1 - g) * p_true + g * mass
    return float(-torch.log2(mix.clamp_min(1e-9)).mean())


def gate_of(extra: torch.Tensor, a: float, b: float | None,
            valid: torch.Tensor) -> torch.Tensor:
    logit = (torch.full_like(extra, a, dtype=torch.float32) if b is None
             else a + b * extra.clamp(max=CAP).to(torch.float32))
    return torch.sigmoid(logit) * valid.to(torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    raw = open(VAL_PATH, "rb").read(N_VAL)
    val = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.int64)
    starts = [w * T * VALSTRIDE for w in range(VALW)
              if w * T * VALSTRIDE + T + 1 <= val.numel()]
    y = torch.cat([val[i + 1:i + 1 + T] for i in starts])
    print(f"val {len(starts)}x{T} = {len(starts) * T} positions | stride "
          f"{VALSTRIDE} | k={K} M={M} cap={CAP} | {len(raw)} bytes", flush=True)

    for path in args.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        build = (build_transformer_baseline if ck["kind"] == "transformer"
                 else build_iecnn)
        model = build(ck["cfg"], V).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        name = os.path.basename(path).replace(".pt", "")
        r = model.regions[0] if getattr(model, "regions", None) else None
        print(f"\n===== {name}", flush=True)
        if r is None or getattr(r, "sc_gate", None) is None:
            print("  no copy channel (base only)", flush=True)
            continue

        a = float(r.sc_gate.detach())
        b = float(r.sc_slope.detach()) if r.sc_slope is not None else None
        print(f"  learned gate a={a:.4f} b="
              f"{'none' if b is None else f'{b:.4f}'} cap={CAP}", flush=True)

        # (1) the model's own channel, and the faithful reference bpc
        m_copy, m_valid, m_extra = [], [], []
        with torch.no_grad():
            for i in starts:
                x = val[i:i + T].unsqueeze(0).to(device)
                c, v, e = r._suffix_copy(x, with_extra=True)
                m_copy.append(c[0]); m_valid.append(v[0]); m_extra.append(e[0])
        m_copy = torch.cat(m_copy)
        m_valid = torch.cat(m_valid)
        m_extra = torch.cat(m_extra)
        with torch.no_grad():
            lp = torch.cat([model(val[i:i + T].unsqueeze(0).to(device))
                            ["final_log_probs"][0] for i in starts])
        model_bpc = float(torch.nn.functional.nll_loss(lp, y) / math.log(2))

        # (2) base p: same weights, gate forced shut (identity mixture)
        saved_a = r.sc_gate.detach().clone()
        saved_b = r.sc_slope.detach().clone() if r.sc_slope is not None else None
        if r.sc_slope is not None:
            r.sc_slope.data.zero_()
        r.sc_gate.data.fill_(-50.0)
        with torch.no_grad():
            p = torch.cat([model(val[i:i + T].unsqueeze(0).to(device))
                           ["final_log_probs"][0].exp() for i in starts])
        r.sc_gate.data.copy_(saved_a)
        if saved_b is not None:
            r.sc_slope.data.copy_(saved_b)

        p_true = p.gather(1, y.unsqueeze(1)).squeeze(1)
        mass_model = (m_copy == y).to(torch.float32)
        shut = bpc(p_true, mass_model, gate_of(m_extra, -50.0, 0.0, m_valid))
        recon = bpc(p_true, mass_model, gate_of(m_extra, a, b, m_valid))
        print(f"  model {model_bpc:.4f} | analytic {recon:.4f} | gate shut "
              f"{shut:.4f}  (channel {shut - recon:+.4f})", flush=True)
        assert abs(recon - model_bpc) < 2e-3, (recon, model_bpc)

        # (3) corpus candidates on the same windows, flattened the same way
        cand = [window_candidates(val[i:i + T].numpy()) for i in starts]
        der = [derive(*c[:3]) for c in cand]
        CVAL = torch.from_numpy(np.concatenate([c[0] for c in cand]))
        EXT = torch.from_numpy(np.concatenate([c[1] for c in cand]))
        CN = torch.from_numpy(np.concatenate([c[2] for c in cand])).long()

        pred_of, valid_of = {}, {}
        for k in ("single", "extpick", "extvote"):
            pred_of[k] = torch.from_numpy(
                np.concatenate([d[k][0] for d in der])).long()
            valid_of[k] = torch.from_numpy(
                np.concatenate([d[k][1] for d in der])).bool()

        # chosen column per selector, for the "gate on selected extension" row
        idx = torch.arange(CVAL.shape[1])[None, :]
        live = idx < CN[:, None]
        ext_live = torch.where(live, EXT, torch.full_like(EXT, -1))
        col = {
            "single": torch.clamp(CN - 1, min=0),
            # derive's extpick breaks ties to the LARGEST j, so take the last
            # column attaining the maximum, not the first
            "extpick": (torch.where(
                ext_live == ext_live.max(1, keepdim=True).values,
                idx, torch.zeros_like(idx)).max(1).values),
        }
        col["extvote"] = col["extpick"]
        ext_of = {k: EXT.gather(1, col[k][:, None]).squeeze(1) for k in col}

        print(f"\n  {'selector':<9}{'cover':>8}{'acc':>7}{'vs model':>10}"
              f"{'sel-only':>10}{'mod-only':>9}{'bpc(model ext)':>16}"
              f"{'bpc(sel ext)':>14}", flush=True)
        for k in SELECTORS:
            if k == "oracle":
                hit = ((CVAL == y[:, None])
                       & (torch.arange(M)[None, :] < CN[:, None])).any(1)
                pred = torch.where(hit, y, (y + 1) % V)
                valid = CN >= 1
            else:
                pred, valid = pred_of[k], valid_of[k]
            mass = (pred == y).to(torch.float32)
            diff = m_valid & valid & (pred != m_copy)
            sel_only = int((diff & (pred == y) & (m_copy != y)).sum())
            mod_only = int((diff & (m_copy == y) & (pred != y)).sum())
            print(f"  {k:<9}{100 * float(valid.float().mean()):>7.1f}%"
                  f"{100 * float((pred == y)[valid].float().mean()):>6.1f}%"
                  f"{int(diff.sum()):>10}{sel_only:>10}{mod_only:>9}"
                  f"{bpc(p_true, mass, gate_of(m_extra, a, b, valid)):>16.4f}"
                  f"{bpc(p_true, mass, gate_of(ext_of[k], a, b, valid)):>14.4f}",
                  flush=True)

        print(f"\n  model pointer: coverage "
              f"{100 * float(m_valid.float().mean()):.1f}% | accuracy "
              f"{100 * float((m_copy == y)[m_valid].float().mean()):.1f}%", flush=True)


if __name__ == "__main__":
    main()
