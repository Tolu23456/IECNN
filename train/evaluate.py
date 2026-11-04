"""Evaluation: bits per character, soft and hard mode (spec 07, section 7).

bpc = total NLL (nats) of predicting each token / (ln 2 * characters of the
original text). Characters are credited per predicted token as that token's
expansion length; over a full text that sum equals len(text) exactly because
decode(encode(x)) == normalize(x) is a proven invariant of UBM.

`evaluate_bpc_per_pass` additionally scores each IECNN pass of the last region
separately (Tolu's pilot logging request: "bpc at each pass (1, 2, 3),
weight entropy, and tau") from the SAME forward — pass p predicts from its own
pooled log-prob, and the pass-weights' entropy is averaged over the eval.
"""
from __future__ import annotations

import math

import torch


def prepare_text(ubm, text: str) -> tuple[list[int], list[int]]:
    """Encode once: (ids, per-token expansion lengths). Callers can reuse the
    pair for repeated evals of the same text (train.py caches val/train)."""
    ids = ubm.encode(text)
    expand_len = [len(ubm.expand(int(i))) for i in ids]
    return ids, expand_len


def _model_forward(model, x, hard_mode: bool, has_regions: bool, kwargs: dict):
    if has_regions:
        return model(x, hard_mode=hard_mode, **kwargs)
    return model(x)


@torch.no_grad()
def evaluate_bpc(model, ubm, text, hard_mode: bool = False,
                 seq_len: int | None = None, max_chars: int | None = None,
                 tau0: float = 1.0, floor: float = 0.0,
                 prepared: tuple | None = None) -> float:
    """Bits per character of `text` under the model, optionally truncated.

    Truncation (`max_chars`) stops after the first prediction token whose
    cumulative expansion length reaches the cap, so the numerator NLL and the
    denominator characters stay consistent at any cut point. tau0/floor are
    forwarded to IECNN models (annealed temperature + weight floor at eval).
    `prepared` may pass a cached (ids, expand_len) pair from prepare_text.
    """
    device = next(model.parameters()).device
    ids, expand_len = prepared if prepared is not None else prepare_text(ubm, text)
    if not ids:
        raise ValueError("cannot evaluate empty text")
    context = seq_len or getattr(model, "context_length", 256) or 256
    has_regions = hasattr(model, "regions")
    kwargs = {"tau0": tau0, "floor": floor} if has_regions else {}

    nll_total = 0.0
    chars_total = 0
    for start in range(0, len(ids) - 1, context):
        end = min(start + context + 1, len(ids))
        window = ids[start:end]
        x = torch.tensor([window[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([window[1:]], dtype=torch.long, device=device)
        out = _model_forward(model, x, hard_mode, has_regions, kwargs)
        logp = out["final_log_probs"] if isinstance(out, dict) else out
        nll = -logp[0].gather(1, y[0].unsqueeze(1)).squeeze(1)
        done = False
        for i in range(len(y[0])):
            chars_total += expand_len[start + 1 + i]
            nll_total += float(nll[i].item())
            if max_chars is not None and chars_total >= max_chars:
                done = True
                break
        if done:
            break
    if chars_total == 0:
        raise ValueError("evaluation produced zero characters")
    return nll_total / (math.log(2) * chars_total)


@torch.no_grad()
def evaluate_bpc_per_pass(model, ubm, text, seq_len: int | None = None,
                          max_chars: int | None = None, tau0: float = 1.0,
                          floor: float = 0.0,
                          prepared: tuple | None = None) -> dict:
    """Per-pass bpc of the last IECNN region, from ONE forward per window.

    Returns {"passes": [bpc_p1, ..., bpc_pP], "final": bpc, "tau": [...],
    "weight_entropy": float}. "final" equals the model's actual pooled
    prediction (identical to evaluate_bpc in soft mode); "passes" are the
    per-pass consensus predictors log_b of the last region. weight_entropy is
    the entropy of the last pass's dot-weights (mean over batch/time/windows),
    the quantity L_w's bonus pushes UP. Baselines (no regions) report a single
    pass equal to the final bpc.
    """
    device = next(model.parameters()).device
    ids, expand_len = prepared if prepared is not None else prepare_text(ubm, text)
    if not ids:
        raise ValueError("cannot evaluate empty text")
    context = seq_len or getattr(model, "context_length", 256) or 256
    has_regions = hasattr(model, "regions")
    kwargs = {"tau0": tau0, "floor": floor} if has_regions else {}

    n_passes = 0
    nll_pass: list[float] = []
    nll_final = 0.0
    chars_total = 0
    ent_sum, ent_n = 0.0, 0
    taus: list[float] | None = None
    for start in range(0, len(ids) - 1, context):
        end = min(start + context + 1, len(ids))
        window = ids[start:end]
        x = torch.tensor([window[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([window[1:]], dtype=torch.long, device=device)
        out = _model_forward(model, x, False, has_regions, kwargs)
        logp = out["final_log_probs"] if isinstance(out, dict) else out
        pp = out.get("per_pass_last_region") if isinstance(out, dict) else None
        if pp is None:                      # baseline: one pass == final
            pp = [{"log_b": logp, "weights": None, "tau": None}]
        if not nll_pass:
            n_passes = len(pp)
            nll_pass = [0.0] * n_passes
            taus = [p.get("tau") for p in pp]
        done = False
        for i in range(len(y[0])):
            ch = expand_len[start + 1 + i]
            chars_total += ch
            nll_final += float(-logp[0, i].gather(
                0, y[0, i].unsqueeze(0)).item())
            for p, entry in enumerate(pp):
                nll_p = -entry["log_b"][0, i].gather(
                    0, y[0, i].unsqueeze(0)).item()
                nll_pass[p] += nll_p
            w = pp[-1].get("weights")
            if w is not None:
                ent = -(w * w.clamp_min(1e-9).log()).sum(-1)
                ent_sum += float(ent.sum().item())
                ent_n += ent.numel()
            if max_chars is not None and chars_total >= max_chars:
                done = True
                break
        if done:
            break
    if chars_total == 0:
        raise ValueError("evaluation produced zero characters")
    denom = math.log(2) * chars_total
    passes = [nl / denom for nl in nll_pass]
    return {"passes": [round(b, 5) for b in passes],
            "final": round(nll_final / denom, 5),
            "tau": taus,
            "weight_entropy": round(ent_sum / max(1, ent_n), 5)}


def mode_gap(model, ubm, text: str) -> dict:
    """Soft bpc, hard bpc, and the gap (hard minus soft)."""
    soft = evaluate_bpc(model, ubm, text, hard_mode=False)
    hard = evaluate_bpc(model, ubm, text, hard_mode=True)
    return {"soft_bpc": soft, "hard_bpc": hard, "gap": hard - soft}


def bpc_from_nll(nll_nats: float, n_chars: int) -> float:
    """The exact identity bpc = nll / (ln2 * chars), for tests and callers."""
    if n_chars <= 0:
        raise ValueError("n_chars must be positive")
    return nll_nats / (math.log(2) * n_chars)
