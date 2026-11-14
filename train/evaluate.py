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
import os

import torch

# Rows per eval forward (speed goal opt 3, 2026-11-14): eval used to run one
# window per forward (batch 1), which wastes the GEMMs on tiny matrices.
# Window rows are independent (states reset per position, window memory is
# per row), so batching them changes nothing per token — pinned by
# test_batched_eval_matches_batch1. Keep the default small: activations at
# [B*T, N, V] are the memory class; 4 keeps ~1.1 GB at vocab2.
EVAL_BATCH = int(os.environ.get("IECNN_EVAL_BATCH", "4"))


def prepare_text(ubm, text: str) -> tuple[list[int], list[int]]:
    """Encode once: (ids, per-token expansion lengths). Callers can reuse the
    pair for repeated evals of the same text (train.py caches val/train)."""
    ids = ubm.encode(text)
    expand_len = [len(ubm.expand(int(i))) for i in ids]
    return ids, expand_len


def _equal_length_batches(starts: list[int], ids: list[int], context: int,
                          batch: int):
    """Group window starts whose rows share one token length.

    Full windows have `context` input tokens; the tail window is shorter and
    must run alone (padding it would score tokens the sequential accounting
    never counted). Groups preserve order, so the sequential char-count cut
    logic is unchanged.
    """
    cur: list[int] = []
    cur_len = 0
    for s in starts:
        length = min(context, len(ids) - s - 1)
        if cur and (length != cur_len or len(cur) >= batch):
            yield cur
            cur = []
        cur_len = length
        cur.append(s)
    if cur:
        yield cur


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
    starts = list(range(0, len(ids) - 1, context))
    for group in _equal_length_batches(starts, ids, context, EVAL_BATCH):
        L = min(context, len(ids) - group[0] - 1)
        xs = [ids[s:s + L] for s in group]
        ys = [ids[s + 1:s + 1 + L] for s in group]
        x = torch.tensor(xs, dtype=torch.long, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        out = _model_forward(model, x, hard_mode, has_regions, kwargs)
        logp = out["final_log_probs"] if isinstance(out, dict) else out
        nll = -logp.gather(2, y.unsqueeze(2)).squeeze(2)          # [K, T]
        stop = False
        for row, s in enumerate(group):          # sequential accounting, in order
            for i in range(L):
                chars_total += expand_len[s + 1 + i]
                nll_total += float(nll[row, i].item())
                if max_chars is not None and chars_total >= max_chars:
                    stop = True
                    break
            if stop:
                break
        if stop:
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
    starts = list(range(0, len(ids) - 1, context))
    for group in _equal_length_batches(starts, ids, context, EVAL_BATCH):
        L = min(context, len(ids) - group[0] - 1)
        xs = [ids[s:s + L] for s in group]
        ys = [ids[s + 1:s + 1 + L] for s in group]
        x = torch.tensor(xs, dtype=torch.long, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        out = _model_forward(model, x, False, has_regions, kwargs)
        logp = out["final_log_probs"] if isinstance(out, dict) else out
        pp = out.get("per_pass_last_region") if isinstance(out, dict) else None
        if pp is None:                      # baseline: one pass == final
            pp = [{"log_b": logp, "weights": None, "tau": None}]
        if not nll_pass:
            n_passes = len(pp)
            nll_pass = [0.0] * n_passes
            taus = [p.get("tau") for p in pp]
        final_nll = -logp.gather(2, y.unsqueeze(2)).squeeze(2)    # [K, T]
        pass_nll = [-entry["log_b"].gather(2, y.unsqueeze(2)).squeeze(2)
                    for entry in pp]
        w = pp[-1].get("weights")
        ent_all = None
        if w is not None:
            ent_all = -(w * w.clamp_min(1e-9).log()).sum(-1)      # [K, T]
        stop = False
        for row, s in enumerate(group):          # sequential accounting, in order
            for i in range(L):
                chars_total += expand_len[s + 1 + i]
                nll_final += float(final_nll[row, i].item())
                for p_idx in range(n_passes):
                    nll_pass[p_idx] += float(pass_nll[p_idx][row, i].item())
                if ent_all is not None:
                    # per-position weight entropy; the mean over counted
                    # positions equals the old per-window accounting (the old
                    # loop added each window's row T times and divided by T^2)
                    ent_sum += float(ent_all[row, i].sum().item())
                    ent_n += ent_all[row, i].numel()
                if max_chars is not None and chars_total >= max_chars:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break
    if chars_total == 0:
        raise ValueError("evaluation produced zero characters")
    denom = math.log(2) * chars_total
    passes = [nl / denom for nl in nll_pass]
    return {"passes": [round(b, 5) for b in passes],
            "final": round(nll_final / denom, 5),
            "tau": taus,
            "weight_entropy": round(ent_sum / max(1, ent_n), 5)}


@torch.no_grad()
def evaluate_consensus_metrics(model, ubm, text, seq_len: int | None = None,
                               max_chars: int | None = None, tau0: float = 1.0,
                               floor: float = 0.0,
                               prepared: tuple | None = None) -> dict:
    """E6 consensus health metrics (Tolu's six-metric acceptance test).

    One forward per window over the eval text, LAST pass of the last region:

      pooled_top1   fraction where the pooled argmax is the truth (today ~2%)
      oracle_top1   fraction where SOME dot's top-1 is the truth (~9%)
      oracle_top5   fraction where the truth is in some dot's top-5 (~30%)
      outvote_rate  fraction where a dot was right and the pooled answer
                    was wrong (the consensus-audit "outvoted" number)
      verifier_sel  top-verifier dot's top-1 accuracy (None without a head)
      overlap5      mean pairwise top-5 overlap of dots (diversity guard;
                    0.58/5 measured at E1@512 — must stay flat or fall)
      acc_per_dot   per-dot top-1 accuracy list (muted dots must revive)

    Uses the same forward as scoring, so it costs one extra pass over the
    eval subset at eval steps only.
    """
    device = next(model.parameters()).device
    ids, expand_len = prepared if prepared is not None else prepare_text(ubm, text)
    if not ids:
        raise ValueError("cannot evaluate empty text")
    context = seq_len or getattr(model, "context_length", 256) or 256
    kwargs = {"tau0": tau0, "floor": floor}

    n_pos = 0
    pooled_hit = oracle1 = oracle5 = outvoted = verif_hit = 0
    overlap_sum = 0.0
    acc_hits: list[int] | None = None
    pp_last = None
    starts = list(range(0, len(ids) - 1, context))
    for group in _equal_length_batches(starts, ids, context, EVAL_BATCH):
        L = min(context, len(ids) - group[0] - 1)
        xs = [ids[s:s + L] for s in group]
        ys = [ids[s + 1:s + 1 + L] for s in group]
        x = torch.tensor(xs, dtype=torch.long, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        out = model(x, **kwargs)
        pp = out.get("per_pass_last_region")
        if not pp or "log_pi" not in pp[-1]:
            break                       # baseline: metrics undefined
        pp_last = pp[-1]
        log_pi = pp[-1]["log_pi"]                         # [K, T, N, V]
        truth = y                                         # [K, T]
        top5 = log_pi.topk(5, dim=-1).indices             # [K, T, N, 5]
        top1 = top5[..., 0]                               # [K, T, N]
        hit = (top1 == truth[:, :, None])                 # [K, T, N]
        n_dots = top1.shape[-1]
        if acc_hits is None:
            acc_hits = [0] * n_dots
        for k in range(n_dots):
            acc_hits[k] += int(hit[..., k].sum().item())
        n_pos += truth.numel()
        oracle1 += int(hit.any(-1).sum().item())
        in5 = (top5 == truth[:, :, None, None]).any(-1)   # [K, T, N]
        oracle5 += int(in5.any(-1).sum().item())
        pooled_top = out["final_log_probs"].argmax(-1)    # [K, T]
        p_hit = (pooled_top == truth)
        pooled_hit += int(p_hit.sum().item())
        outvoted += int((~p_hit & hit.any(-1)).sum().item())
        # pairwise top-5 overlap via token-count identity:
        # sum over pairs of |A_i ∩ A_j| = sum over tokens C(c_v, 2)
        counts = torch.zeros(top5.shape[0], top5.shape[1], log_pi.shape[-1],
                             dtype=torch.long, device=device)
        flat5 = top5.reshape(top5.shape[0], top5.shape[1], -1)
        counts.scatter_add_(2, flat5,
                            torch.ones_like(flat5, dtype=torch.long))
        npairs = n_dots * (n_dots - 1) / 2
        v = pp[-1].get("verifier")
        sel = None
        if v is not None:
            sel = v.argmax(-1, keepdim=True)              # [K, T, 1]
        stop = False
        for row, s in enumerate(group):
            overlap_sum += float(
                (counts[row] * (counts[row] - 1) / 2).sum(-1).sum().item()
            ) / (npairs * 5)
            if sel is not None:
                verif_hit += int(hit[row].gather(1, sel[row]).sum().item())
            if max_chars is not None:
                chars_here = sum(expand_len[s + 1: s + 1 + L])
                if chars_here >= max_chars:
                    stop = True
        if stop:
            break
    if n_pos == 0 or acc_hits is None:
        raise ValueError("no IECNN positions found for consensus metrics")
    return {
        "n_positions": n_pos,
        "pooled_top1": pooled_hit / n_pos,
        "oracle_top1": oracle1 / n_pos,
        "oracle_top5": oracle5 / n_pos,
        "outvote_rate": outvoted / n_pos,
        "verifier_sel": (verif_hit / n_pos) if pp[-1].get("verifier") is not None else None,
        "overlap5": overlap_sum / n_pos,
        "acc_per_dot": [h / n_pos for h in acc_hits],
    }


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
