"""Evaluation: bits per character, soft and hard mode (spec 07, section 7).

bpc = total NLL (nats) of predicting each token / (ln 2 * characters of the
original text). Characters are credited per predicted token as that token's
expansion length; over a full text that sum equals len(text) exactly because
decode(encode(x)) == normalize(x) is a proven invariant of UBM.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def evaluate_bpc(model, ubm, text: str, hard_mode: bool = False,
                 seq_len: int | None = None, max_chars: int | None = None) -> float:
    """Bits per character of `text` under the model, optionally truncated.

    Truncation (`max_chars`) stops after the first prediction token whose
    cumulative expansion length reaches the cap, so the numerator NLL and the
    denominator characters stay consistent at any cut point.
    """
    device = next(model.parameters()).device
    ids = ubm.encode(text)
    if not ids:
        raise ValueError("cannot evaluate empty text")
    expand_len = [len(ubm.expand(int(i))) for i in ids]
    context = seq_len or getattr(model, "context_length", 256) or 256

    nll_total = 0.0
    chars_total = 0
    for start in range(0, len(ids) - 1, context):
        end = min(start + context + 1, len(ids))
        window = ids[start:end]
        x = torch.tensor([window[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([window[1:]], dtype=torch.long, device=device)
        out = model(x, hard_mode=hard_mode)
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
