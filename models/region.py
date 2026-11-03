"""One region: N dots, P passes, weights, consensus, summary (spec 03).

The parameters are shared across passes (D6); the round signal, the carried
state, the pooled opinion and the error feature give each pass its behavior.
The consensus machinery lives in `consensus.py` (label-free, C2).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from models import consensus
from models.dot import DotPopulation
from models.memory import gather_windows


class Region(nn.Module):
    """Rung-building block. One region reads a fixed window of memory rows."""

    def __init__(self, region_index: int, config: dict, vocab_size: int):
        super().__init__()
        m = config["model"]
        d = int(m.get("d", 128)); d_q = int(m.get("d_q", 32))
        d_v = int(m.get("d_v", 64)); d_s = int(m.get("d_s", 128))
        self.index = region_index
        self.d = d; self.d_q = d_q; self.d_v = d_v; self.d_s = d_s
        self.span = int((m.get("spans") or [256])[region_index % len(m.get("spans") or [256])])
        self.n_dots = int(m.get("n_dots", 8))
        self.n_passes = int(m.get("n_passes", 3))
        self.use_weights = bool(m.get("use_weights", True))
        self.use_messages = bool(m.get("use_messages", False))
        self.pooling = m.get("pooling", "log_linear")
        self.rho = float(config.get("consensus", {}).get("rho", 0.5))
        self.kappa = float(config.get("consensus", {}).get("kappa", 0.7))
        self.hard_top_m = int(m.get("hard_top_m", 4))
        self.dynamic_query = bool(m.get("dynamic_query", False))
        self.share_pass_params = bool(m.get("share_pass_params", True))
        self.dot_dropout = float(m.get("dot_dropout", 0.0))

        self.dots = DotPopulation(
            self.n_dots, d, d_q, d_v, d_s, vocab_size,
            message_dim=(16 if self.use_messages else 0),
            n_passes=self.n_passes, dynamic_query=self.dynamic_query,
            share_pass_params=self.share_pass_params,
            seed=1000 + region_index,
            fast_math=bool(m.get("fast_dot_math", True)))
        self.w_k = nn.Linear(d, d_q, bias=False)
        self.w_v = nn.Linear(d, d_v, bias=False)
        self.rel_emb = nn.Embedding(self.span, d_q)
        nn.init.normal_(self.rel_emb.weight, std=1.0 / math.sqrt(d_q))

        # learned scale/trust scalars per region (spec 04 section 3)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        # messages: write at pass p, read at pass p+1 (spec 05)
        self.messages = None
        if self.use_messages:
            from models.messages import MessageChannel
            self.messages = MessageChannel(
                d_s, 16, self.n_dots, radius=2,
                dropout=float(m.get("message_dropout", 0.2)))

    # ------------------------------------------------------------------
    def _window_memory(self, rows: torch.Tensor, device):
        """Rows [B,T,d] -> (wind_k [B,T,w,d_q], wind_v [B,T,w,d_v], valid [T,w])."""
        k_all = self.w_k(rows)                       # [B, T, d_q]
        v_all = self.w_v(rows)                       # [B, T, d_v]
        wind_k_raw, valid = gather_windows(k_all, self.span)
        wind_v, _ = gather_windows(v_all, self.span)
        # slot i of the window holds position t-(w-1-i): distance = w-1-i
        slot_idx = torch.arange(self.span, device=device)
        dist_idx = self.span - 1 - slot_idx
        rel = self.rel_emb(dist_idx)                 # [w, d_q], distance 0 at t itself
        wind_k = wind_k_raw + rel[None, None]
        return wind_k, wind_v, valid.to(device)

    # ------------------------------------------------------------------
    def forward(self, rows: torch.Tensor, emb_table: torch.Tensor,
                tau0: float = 1.0, hard_mode: bool = False,
                floor: float = 0.0, training: bool | None = None) -> dict:
        """rows [B, T, d] -> the region's consensus and diagnostics.

        floor: the per-dot weight floor eps (spec 04 section 3); callers hand
        0.1/N during training and 0 for the final phase / inference.
        """
        if training is None:
            training = self.training
        device = rows.device
        b, t = rows.shape[0], rows.shape[1]
        wind_k, wind_v, valid = self._window_memory(rows, device)
        n = self.n_dots

        state = self.dots.initial_state((b, t))
        trust = None
        pi_prev = None
        prev_crowd = prev_js = None
        w_prev = None
        state_prev = None
        msg_read = None
        log_b_prev = None
        per_pass = []

        # dot dropout (spec table at E4: "5% of steps, one random dot masked
        # per region"): active only in training; the SAME dot is masked across
        # this region's passes so the region behaves as if the dot were absent.
        dot_mask = None
        if training and self.dot_dropout > 0 and torch.rand(()) < self.dot_dropout:
            k = int(torch.randint(0, n, ()))
            dot_mask = torch.ones(n, device=device, dtype=rows.dtype)
            dot_mask[k] = 0.0

        for p in range(1, self.n_passes + 1):
            tau_p = tau0 * (self.kappa ** (p - 1))
            if p == 1 or not self.use_weights:
                h_pool = torch.zeros(b, t, n, self.d_s, device=device, dtype=rows.dtype)
            else:
                pooled = torch.einsum("bfn,bfnd->bfd", w_prev, state_prev)  # [B,T,d_s]
                h_pool = pooled[:, :, None, :].expand(b, t, n, self.d_s)
                # ^ every dot reads the SAME pooled opinion (spec 01, input table)
            if p == 1:
                err = torch.zeros(b, t, n, 3, device=device, dtype=rows.dtype)
            else:
                # error features from the numbers already computed at pass p-1
                err = consensus.error_features_from_parts(pi_prev, prev_crowd, prev_js)
            out = self.dots(wind_k, wind_v, valid, h_pool, err, round_idx=p - 1,
                            msg_in=msg_read if p > 1 else None, state=state,
                            emb_table=emb_table)
            state_prev = out.state
            pi = out.log_pi.exp()

            if self.use_weights:
                # loo/JS are computed ONCE here and reused for the error
                # feature of the next pass (spec 04 = same numbers either way;
                # computing them twice cost ~30% of the forward).
                scores, loo, js = consensus.compute_scores(
                    pi, out.gamma, out.trust_term, self.alpha, self.beta,
                    return_parts=True, log_pi=out.log_pi)
                trust = consensus.update_trust(trust, scores, self.rho)
                w = consensus.compute_weights(trust, tau_p, floor=0.0 if hard_mode else floor)
                crowd, js_own = loo, js
            else:
                w = consensus.uniform_weights(n, pi)          # rung E0: plain ensemble
                if p < self.n_passes:      # only a LATER pass needs these numbers
                    crowd = consensus.leave_one_out_mean(pi)
                    js_own = consensus.js_divergence(pi, crowd)
                else:
                    crowd = js_own = None

            if dot_mask is not None:      # dot dropout: excluded dot gets 0
                w = w * dot_mask
                w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)

            if self.pooling == "mixture":
                b_mix = consensus.pool_mixture(pi, w)
                log_b = (b_mix / b_mix.sum(-1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12).log()
            else:
                log_b = consensus.pool_log_linear(out.log_pi, w)
            per_pass.append({"log_pi": out.log_pi, "weights": w,
                             "log_b": log_b, "gamma": out.gamma, "tau": tau_p})
            log_b_prev = log_b
            pi_prev = pi
            prev_crowd, prev_js = crowd, js_own
            w_prev = w
            if self.messages is not None:
                msg_write = self.messages(out.state, training=training)   # read at p+1
                msg_read = msg_write

        w_final = per_pass[-1]["weights"]
        summary = torch.einsum("bfn,bfnd->bfd", w_final, state_prev)  # [B, T, d_s]

        if hard_mode and self.use_weights:
            w_hard = consensus.hard_top_m(w_final, self.hard_top_m)
            log_final = consensus.pool_log_linear(per_pass[-1]["log_pi"], w_hard)
            log_b_prev = log_final
            w_out = w_hard
        else:
            log_final = log_b_prev
            w_out = w_final

        return {"log_b": log_final,              # [B, T, V] region consensus
                "region_summary": summary,       # [B, T, d_s] for the next memory
                "weights": w_out,                # [B, T, N]
                "per_pass": per_pass,
                "state_final": state_prev}
