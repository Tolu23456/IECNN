"""One region: N dots, P passes, weights, consensus, summary (spec 03).

The parameters are shared across passes (D6); the round signal, the carried
state, the pooled opinion and the error feature give each pass its behavior.
The consensus machinery lives in `consensus.py` (label-free, C2).
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import consensus
from models.dot import DotPopulation
from models.memory import gather_windows

_COMPILED_CORE = None
_COMPILE_MODE = None


def _compiled_consensus_core(*args, mode=None):
    """Lazily compiled consensus_core (torch.compile/inductor). The whole
    per-pass [., N, V] chain — exp, leave-one-out, JS identity, scores,
    trust, floor-softmax, error features, matmul pooling — as one graph so
    inductor can fuse the bandwidth-bound elementwise chain. dynamic=False:
    shapes are fixed per config; recompiles only on (rare) shape changes.

    mode (GPU speed pack, 2026-11-15): None = default inductor; "cuda_graphs"
    wraps the compiled core in torch.cuda.graphs-style reduce-overhead
    (kills kernel-launch overhead — the suspected bottleneck on small-SM
    GPUs like the T4); "max-autotune-no-cudnn" = inductor autotuning.
    """
    global _COMPILED_CORE, _COMPILE_MODE
    if _COMPILED_CORE is None or _COMPILE_MODE != mode:
        kw = {}
        if mode == "cuda_graphs":
            kw["mode"] = "reduce-overhead"
        elif mode:
            kw["mode"] = mode
        _COMPILED_CORE = torch.compile(consensus.consensus_core,
                                       dynamic=False, fullgraph=False, **kw)
        _COMPILE_MODE = mode
    return _COMPILED_CORE(*args)


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
        # speed goal round 2: fuse the per-pass [., N, V] consensus chain
        # with torch.compile (training.compile_consensus: true). Default OFF
        # — eager stays the bit-for-bit reference path.
        self.compile_consensus = bool(config.get("training", {})
                                      .get("compile_consensus", False))
        # GPU speed pack (2026-11-15): torch.compile mode for the fused core
        # ("cuda_graphs" -> reduce-overhead, i.e. CUDA graphs, for
        # launch-bound small-SM GPUs; any other string passes to inductor).
        self.compile_mode = str(config.get("training", {})
                                .get("compile_mode", "")) or None
        # torch 2.14 removed "max-autotune-no-cudnn" (renamed to
        # "max-autotune-no-cudagraphs") — map it so configs work on both.
        if self.compile_mode == "max-autotune-no-cudnn":
            self.compile_mode = "max-autotune-no-cudagraphs"
        self.hard_top_m = int(m.get("hard_top_m", 4))
        self.dynamic_query = bool(m.get("dynamic_query", False))
        self.share_pass_params = bool(m.get("share_pass_params", True))
        self.dot_dropout = float(m.get("dot_dropout", 0.0))
        # Rung E11 (Tolu, 2026-11-17: "instead of every dot having its own
        # head they all share one head and only their state differs"):
        # "prob" (default) = the original per-dot vocab heads + consensus
        # pooling over [., N, V]; "hidden" = dots emit hidden states only,
        # a learned gate weights them, and ONE shared tied head reads the
        # pooled state. Prunes: per-dot logits on P-1 of P passes, the whole
        # JS/leave-one-out/verifier chain, and the [B,T,N,V] elementwise
        # traffic — with the committee read intact in hidden space.
        self.head_mode = str(m.get("head_mode", "prob"))

        self.dots = DotPopulation(
            self.n_dots, d, d_q, d_v, d_s, vocab_size,
            message_dim=(16 if self.use_messages else 0),
            n_passes=self.n_passes, dynamic_query=self.dynamic_query,
            share_pass_params=self.share_pass_params,
            seed=1000 + region_index,
            fast_math=bool(m.get("fast_dot_math", True)))
        self.dots.head_layout = str(m.get("head_layout", "shared_gemm"))
        # speed opt 7 (Tolu's "don't compute the same op twice"): with
        # share_pass_params + static queries the attention block is
        # bit-identical across passes; one fresh dict per region-forward
        # lets pass 1 compute it and later passes reuse ctx. Default ON;
        # model.cache_attention: false restores the recompute-every-pass
        # behavior (paired A/B knob).
        self.cache_attention = bool(m.get("cache_attention", True))
        self.w_k = nn.Linear(d, d_q, bias=False)
        self.w_v = nn.Linear(d, d_v, bias=False)
        self.rel_emb = nn.Embedding(self.span, d_q)
        nn.init.normal_(self.rel_emb.weight, std=1.0 / math.sqrt(d_q))

        # learned scale/trust scalars per region (spec 04 section 3)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        # E6 consensus-v2: verifier head replaces pure scalars with a learned
        # oracle-supervised trust judgment. Off (absent) = bit-identical E1.
        self.verifier = None
        if bool(m.get("verifier", False)):
            from models.verifier import VerifierHead
            self.verifier = VerifierHead(seed=2000 + region_index)
            self.v_scale = nn.Parameter(torch.tensor(0.05))

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
    def _verifier_features(self, pi: torch.Tensor, gamma: torch.Tensor,
                           state: torch.Tensor, err: torch.Tensor,
                           w_prev: torch.Tensor | None,
                           log_b_prev: torch.Tensor | None,
                           p: int, device) -> torch.Tensor:
        """Per-dot features for the verifier head: [..., N, 9].

        All label-free and all already computed inside the pass loop (the
        oracle only ever TRAINS the head — see train/losses.py — it is not an
        input). Pass 1 uses neutral values for the not-yet-existing weight
        and region entropy. Leading dims are agnostic: callers pass either
        [B, T, N, ...] or a row chunk [c, N, ...].
        """
        from models.verifier import verifier_features
        n = gamma.shape[-1]
        top = pi.argmax(-1)                                  # [..., N]
        same = (top.unsqueeze(-1) == top.unsqueeze(-2)).float()
        agree = (same.sum(-1) - 1.0) / max(n - 1, 1)         # [..., N]
        state_norm = state.norm(dim=-1) / math.sqrt(self.d_s)
        if w_prev is None:
            w_feat = torch.full_like(gamma, 1.0 / n)
        else:
            w_feat = w_prev
        if log_b_prev is not None:
            h_region = consensus.entropy(log_b_prev)         # [...] leading dims
        else:
            h_region = torch.zeros(gamma.shape[:-1], device=device)
        pass_frac = torch.tensor(p / self.n_passes, device=device)
        return verifier_features(gamma, w_feat, agree, state_norm, err,
                                 h_region, pass_frac)

    # ------------------------------------------------------------------
    def _pass_consensus(self, log_pi_c, gamma_c, trust_term_c, trust_prev_c,
                        state_c, err_c, w_prev_c, log_b_prev_c, dot_mask,
                        p: int, tau_p: float, hard_mode: bool, floor: float):
        """One row-chunk of the per-pass consensus (see the chunking note in
        `forward`). Inputs/outputs carry a leading chunk dim instead of
        [B, T]: log_pi_c [c, N, V], gamma/trust_term/trust/w [c, N],
        state [c, N, d_s], err [c, N, 3], log_b_prev [c, V]. Returns
        (trust_c, w_c, log_b_c, v_out_c, err_next_c)."""
        pi = log_pi_c.exp()
        v_out = None
        if self.use_weights:
            # loo/JS are computed ONCE here and reused for the error feature
            # of the next pass (spec 04 = same numbers either way; computing
            # them twice cost ~30% of the forward). The _fast variant reuses
            # log_pi for the JS entropy terms (last-bit differences only).
            if (self.compile_consensus and self.n_dots >= 2
                    and self.pooling == "log_linear" and dot_mask is None):
                # fused path (torch.compile): the whole [c, N, V] chain in
                # one graph. tau/floor as 0-dim tensors so per-step annealed
                # floats are data, not recompile-triggering guards. Verifier
                # features stay eager (they need pi before the core); their
                # verdict folds in as v_scale * v_out (zeros when absent —
                # bit-exact x + 0 = x).
                dtype, device = log_pi_c.dtype, log_pi_c.device
                if self.verifier is not None:
                    feats = self._verifier_features(
                        pi, gamma_c, state_c, err_c, w_prev_c, log_b_prev_c,
                        p, device)
                    v_out = self.verifier(feats).squeeze(-1)     # [c, N]
                    v_scale_t, v_out_t = self.v_scale, v_out
                else:
                    v_scale_t = torch.zeros((), dtype=dtype, device=device)
                    v_out_t = torch.zeros(gamma_c.shape, dtype=dtype,
                                          device=device)
                tau_t = torch.tensor(tau_p, dtype=dtype, device=device)
                floor_t = torch.tensor(0.0 if hard_mode else floor,
                                       dtype=dtype, device=device)
                trust_in = trust_prev_c if trust_prev_c is not None \
                    else torch.zeros_like(gamma_c)
                trust_c, w_c, log_b_c, err_next = _compiled_consensus_core(
                    log_pi_c, gamma_c, trust_term_c, trust_in,
                    self.alpha, self.beta, v_scale_t, v_out_t,
                    self.rho, tau_t, floor_t, self.n_dots,
                    pi if self.verifier is not None else None,
                    mode=self.compile_mode)
                return trust_c, w_c, log_b_c, v_out, err_next
            scores, loo, js = consensus.compute_scores_fast(
                pi, log_pi_c, gamma_c, trust_term_c, self.alpha, self.beta,
                return_parts=True)
            if self.verifier is not None:
                # E6: the verifier's trust judgment joins the consensus
                # scores (scaled — it must EARN influence from small).
                feats = self._verifier_features(
                    pi, gamma_c, state_c, err_c, w_prev_c, log_b_prev_c,
                    p, log_pi_c.device)
                v_out = self.verifier(feats).squeeze(-1)         # [c, N]
                scores = scores + self.v_scale * v_out
            trust_c = consensus.update_trust(trust_prev_c, scores, self.rho)
            w_c = consensus.compute_weights(trust_c, tau_p,
                                            floor=0.0 if hard_mode else floor)
            err_next = consensus.error_features_from_parts(pi, loo, js)
        else:
            trust_c = None
            w_c = consensus.uniform_weights(self.n_dots, pi)   # rung E0
            if p < self.n_passes:      # only a LATER pass needs these numbers
                crowd = consensus.leave_one_out_mean(pi)
                js_own = consensus.js_from_parts_fast(pi, log_pi_c, crowd)
                err_next = consensus.error_features_from_parts(pi, crowd, js_own)
            else:
                err_next = None

        if dot_mask is not None:      # dot dropout: excluded dot gets 0
            w_c = w_c * dot_mask
            w_c = w_c / w_c.sum(-1, keepdim=True).clamp_min(1e-9)

        if self.pooling == "mixture":
            b_mix = consensus.pool_mixture(pi, w_c)
            log_b_c = (b_mix / b_mix.sum(-1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12).log()
        else:
            log_b_c = consensus.pool_log_linear(log_pi_c, w_c)
        return trust_c, w_c, log_b_c, v_out, err_next

    # ------------------------------------------------------------------
    def _pass_hidden(self, gamma_c, trust_term_c, trust_prev_c, dot_mask,
                     tau_p: float, floor: float):
        """One pass of the shared-head gate (hidden mode). All ops are
        [., N]-scale — no vocab-space tensor exists in this path. Weights
        come from the same learned scalar projections as the prob-space
        consensus (gamma / trust_term + trust accumulation), with the
        JS-disagreement and verifier terms dropped (the committee study
        bounded their value at ~0.047 bpc; they were the dominant cost).

        Returns (trust_c, w_c)."""
        scores = self.beta * gamma_c + trust_term_c
        trust_c = consensus.update_trust(trust_prev_c, scores, self.rho)
        t = trust_c.clamp(-20.0, 20.0)
        w = F.softmax(t / max(tau_p, 1e-6), dim=-1)
        if floor > 0.0:
            wf = w.clamp_min(floor)
            w = wf / wf.sum(-1, keepdim=True).clamp_min(1e-9)
        if dot_mask is not None:
            w = w * dot_mask
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
        return trust_c, w

    # ------------------------------------------------------------------
    def _forward_hidden(self, rows: torch.Tensor, emb_table: torch.Tensor,
                        tau0: float, hard_mode: bool, floor: float,
                        training: bool) -> dict:
        """Shared-head mode (rung E11): dots produce hidden states; a
        learned gate pools them; ONE tied head reads the pooled state.

        Passes 1..P-1 skip the per-dot vocab GEMM entirely; the LAST pass
        materializes per-dot logits once (the oracle/metric machinery and
        the boosted-dot losses consume [., N, V] there — train-time only
        cost, and a single GEMM + log-softmax instead of the old chain)."""
        device = rows.device
        b, t = rows.shape[0], rows.shape[1]
        wind_k, wind_v, valid = self._window_memory(rows, device)
        n = self.n_dots
        f = b * t

        state = self.dots.initial_state((b, t))
        attn_cache = {} if self.cache_attention else None
        trust = None
        w_prev = None
        state_prev = None
        msg_read = None
        per_pass = []
        err = torch.zeros(b, t, n, 3, device=device, dtype=rows.dtype)
        e_state_last = None

        dot_mask = None
        if training and self.dot_dropout > 0 and torch.rand(()) < self.dot_dropout:
            k = int(torch.randint(0, n, ()))
            dot_mask = torch.ones(n, device=device, dtype=rows.dtype)
            dot_mask[k] = 0.0

        for p in range(1, self.n_passes + 1):
            tau_p = tau0 * (self.kappa ** (p - 1))
            if p == 1:
                h_pool = torch.zeros(b, t, n, self.d_s, device=device,
                                     dtype=rows.dtype)
            else:
                pooled = torch.einsum("bfn,bfnd->bfd", w_prev, state_prev)
                h_pool = pooled[:, :, None, :].expand(b, t, n, self.d_s)
            last = (p == self.n_passes)
            out = self.dots(wind_k, wind_v, valid, h_pool, err,
                            round_idx=p - 1,
                            msg_in=msg_read if p > 1 else None, state=state,
                            emb_table=emb_table, attn_cache=attn_cache,
                            want_log_pi=last)
            state_prev = out.state
            if last:
                e_state_last = out.e_state

            tr_c, w_c = self._pass_hidden(
                out.gamma, out.trust_term,
                trust.reshape(b, t, n) if trust is not None else None,
                dot_mask, tau_p, 0.0 if hard_mode else floor)
            if self.use_weights:
                trust = tr_c
            w = w_c

            # ONE shared head on the pooled hidden state
            h_mix = torch.einsum("bfn,bfnd->bfd", w, out.e_state)   # [B, T, d]
            logits = h_mix.reshape(f, self.d).matmul(emb_table.t())  # [f, V]
            log_b = F.log_softmax(logits, dim=-1).reshape(b, t, -1)

            # hidden-space error features for the next pass (3 dims, cheap):
            # own weight, gate entropy, top weight
            w_ent = -(w * w.clamp_min(1e-12).log()).sum(-1, keepdim=True)
            w_top = w.max(-1, keepdim=True).values
            err = torch.cat([w.unsqueeze(-1), w_ent.expand(b, t, n, 1),
                             w_top.expand(b, t, n, 1)], dim=-1)

            per_pass.append({"weights": w, "log_b": log_b,
                             "gamma": out.gamma, "tau": tau_p,
                             "verifier": None})
            w_prev = w
            if self.messages is not None:
                msg_write = self.messages(out.state, training=training)
                msg_read = msg_write

        w_final = per_pass[-1]["weights"]
        summary = torch.einsum("bfn,bfnd->bfd", w_final, state_prev)

        if hard_mode and self.use_weights:
            w_hard = consensus.hard_top_m(w_final, self.hard_top_m)
            h_mix = torch.einsum("bfn,bfnd->bfd", w_hard, e_state_last)
            logits = h_mix.reshape(f, self.d).matmul(emb_table.t())
            log_final = F.log_softmax(logits, dim=-1).reshape(b, t, -1)
            w_out = w_hard
        else:
            log_final = per_pass[-1]["log_b"]
            w_out = w_final

        # per-dot logits, ONE GEMM, last pass only (oracle/metrics/losses)
        e_f = e_state_last.reshape(f * n, self.d)
        log_pi = F.log_softmax(e_f.matmul(emb_table.t()),
                               dim=-1).reshape(b, t, n, -1)
        per_pass[-1]["log_pi"] = log_pi

        return {"log_b": log_final,
                "region_summary": summary,
                "weights": w_out,
                "per_pass": per_pass,
                "state_final": state_prev}

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
        if self.head_mode == "hidden":
            return self._forward_hidden(rows, emb_table, tau0, hard_mode,
                                        floor, training)
        device = rows.device
        b, t = rows.shape[0], rows.shape[1]
        wind_k, wind_v, valid = self._window_memory(rows, device)
        n = self.n_dots

        state = self.dots.initial_state((b, t))
        attn_cache = {} if self.cache_attention else None
        trust = None
        w_prev = None
        state_prev = None
        msg_read = None
        log_b_prev = None
        per_pass = []

        # chunked consensus (speed opt 4): rows per consensus slice; 0 or
        # >= f disables chunking (single slice = the pre-opt-4 math exactly).
        # DEFAULT OFF: measured on the bench (results/bench_opt4_bf16.json,
        # vocab2/batch16/V=2048), chunk 64 was a 1.9x SLOWDOWN (10.13 vs
        # 5.43 s/step) — the [B,T,N,V] temporaries fit RAM, so slicing only
        # breaks GEMM batching. Keep the knob for memory-tight big-vocab
        # runs where the same block approaches GB scale.
        f_total = b * t
        chunk = f_total
        try:
            cz = int(os.environ.get("IECNN_CONS_CHUNK", "0"))
        except ValueError:
            cz = 0
        if 0 < cz < f_total:
            chunk = cz

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
            if p == 1:
                err = torch.zeros(b, t, n, 3, device=device, dtype=rows.dtype)
            if p == 1 or not self.use_weights:
                h_pool = torch.zeros(b, t, n, self.d_s, device=device, dtype=rows.dtype)
            else:
                pooled = torch.einsum("bfn,bfnd->bfd", w_prev, state_prev)  # [B,T,d_s]
                h_pool = pooled[:, :, None, :].expand(b, t, n, self.d_s)
                # ^ every dot reads the SAME pooled opinion (spec 01, input table)
            out = self.dots(wind_k, wind_v, valid, h_pool, err, round_idx=p - 1,
                            msg_in=msg_read if p > 1 else None, state=state,
                            emb_table=emb_table, attn_cache=attn_cache)
            state_prev = out.state

            # ---- consensus + pooling, chunked over rows (speed opt 4) ----
            # The [., N, V] chain (exp, loo, JS, scores, weights, pool) is
            # memory-bound: whole-tensor it streams hundreds of MB through a
            # small L3 several times per pass. In `chunk`-row slices the
            # working set stays cache-hot and the big temporaries (loo, m,
            # logm) exist only per chunk. chunk == f reproduces the previous
            # single-slice math op-for-op (same ops, same order); only the
            # pooling GEMM's reassociation differs (last bit).
            f = b * t
            log_pi_f = out.log_pi.reshape(f, n, -1)          # contiguous views
            gamma_f = out.gamma.reshape(f, n)
            trust_term_f = out.trust_term.reshape(f, n)
            state_f = out.state.reshape(f, n, self.d_s)
            err_f = err.reshape(f, n, 3)
            w_prev_f = w_prev.reshape(f, n) if w_prev is not None else None
            trs, ws, lbs, vos, ens = [], [], [], [], []
            for i in range(0, f, chunk):
                sl = slice(i, min(i + chunk, f))
                tr_c, w_c, lb_c, v_c, en_c = self._pass_consensus(
                    log_pi_f[sl], gamma_f[sl], trust_term_f[sl],
                    trust[sl] if trust is not None else None,
                    state_f[sl], err_f[sl],
                    w_prev_f[sl] if w_prev_f is not None else None,
                    log_b_prev[sl] if log_b_prev is not None else None,
                    dot_mask, p, tau_p, hard_mode, floor)
                trs.append(tr_c); ws.append(w_c); lbs.append(lb_c)
                vos.append(v_c); ens.append(en_c)
            if self.use_weights:
                trust = torch.cat(trs, 0)
            w_f = torch.cat(ws, 0)
            log_b_f = torch.cat(lbs, 0)
            v_out = torch.cat(vos, 0) if self.verifier is not None else None
            # error features for the NEXT pass, computed now (chunk-wise) so
            # this pass's pi/loo never have to be kept whole
            if p < self.n_passes:
                err = torch.cat(ens, 0).reshape(b, t, n, 3)
            w = w_f.reshape(b, t, n)
            log_b = log_b_f.reshape(b, t, -1)
            per_pass.append({"log_pi": out.log_pi, "weights": w,
                             "log_b": log_b, "gamma": out.gamma, "tau": tau_p,
                             "verifier": (v_out.reshape(b, t, n)
                                          if v_out is not None else None)})
            log_b_prev = log_b_f
            w_prev = w
            if self.messages is not None:
                msg_write = self.messages(out.state, training=training)   # read at p+1
                msg_read = msg_write

        w_final = per_pass[-1]["weights"]
        summary = torch.einsum("bfn,bfnd->bfd", w_final, state_prev)  # [B, T, d_s]

        if hard_mode and self.use_weights:
            w_hard = consensus.hard_top_m(w_final, self.hard_top_m)
            log_final = consensus.pool_log_linear(per_pass[-1]["log_pi"], w_hard)
            w_out = w_hard
        else:
            log_final = per_pass[-1]["log_b"]   # [B, T, V] (log_b_prev is the
            w_out = w_final                     # chunk-layout [f, V] twin)

        return {"log_b": log_final,              # [B, T, V] region consensus
                "region_summary": summary,       # [B, T, d_s] for the next memory
                "weights": w_out,                # [B, T, N]
                "per_pass": per_pass,
                "state_final": state_prev}
