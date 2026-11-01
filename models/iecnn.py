"""The full IECNN model: regions stacked, region competition, final output (spec 03).

Data flow (file 00, section 5): embeddings -> region 1 (window of 32) -> its
summary builds region 2's memory -> ... -> region weights omega pool the
regions' consensus distributions into the final next-token distribution.

Config switches honored: n_regions, n_passes, use_weights, use_cross_region,
use_messages, pooling, share_pass_params, dynamic_query, halting (none in
v0.1), spans, hard_top_m. Rung E0 (n_regions=1, n_passes=1, use_weights=false)
must be exactly a plain uniform ensemble — checked by a test.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models import consensus
from models.memory import region1_rows, next_region_rows
from models.region import Region


def build_iecnn(config: dict, vocab_size: int) -> "IECNN":
    return IECNN(config, vocab_size)


class IECNN(nn.Module):
    def __init__(self, config: dict, vocab_size: int):
        super().__init__()
        m = config["model"]
        self.cfg = config
        self.d = int(m.get("d", 128))
        self.d_s = int(m.get("d_s", 128))
        self.vocab_size = vocab_size
        self.R = int(m.get("n_regions", 4))
        self.use_cross_region = bool(m.get("use_cross_region", True))
        self.hard_top_m = int(m.get("hard_top_m", 4))

        # tied embedding E [V, d]: memory rows and dot output heads share it
        self.embed = nn.Embedding(vocab_size, self.d)
        nn.init.normal_(self.embed.weight, std=0.02)

        self.regions = nn.ModuleList(
            [Region(r, config, vocab_size) for r in range(self.R)])

        # region r>1 chaining: Proj_r + LayerNorm build region r's rows from
        # region r-1's summary + residual (spec 02, section 2 — both learnable)
        self.chain_proj = nn.ModuleList(
            [nn.Linear(self.d_s, self.d, bias=False) for _ in range(self.R - 1)])
        self.chain_ln = nn.ModuleList(
            [nn.LayerNorm(self.d) for _ in range(self.R - 1)])

        # cross-region competition (D5): gate_r from the region summary,
        # plus one learned scale on the confidence term
        self.region_gate = nn.ModuleList(
            [nn.Linear(self.d_s, 1) for _ in range(self.R)])
        self.alpha_region = nn.Parameter(torch.tensor(1.0))

    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor, tau0: float = 1.0,
                hard_mode: bool = False, floor: float = 0.0):
        """input_ids [B, T] -> dict with final_log_probs [B, T, V] and details.

        `floor` is the per-dot weight floor eps (0.1/N in training; 0 late).
        Region temperature floor is 0.5, higher than the dot floor (spec 03).
        """
        tau_region = max(tau0, 0.5)
        emb_table = self.embed.weight
        rows = region1_rows(self.embed(input_ids))     # [B, T, d]

        region_outputs = []
        for r, region in enumerate(self.regions):
            out = region(rows, emb_table, tau0=tau0, hard_mode=hard_mode,
                         floor=floor)
            region_outputs.append(out)
            if r + 1 < self.R:
                rows = next_region_rows(
                    out["region_summary"], rows,
                    self.chain_proj[r], self.chain_ln[r])      # [B, T, d]

        # ---- cross-region competition (D5) --------------------------------
        if self.use_cross_region and self.R > 1:
            scores = []
            for r, out in enumerate(region_outputs):
                log_b = out["log_b"]
                conf = -consensus.entropy(log_b)               # [B, T]
                gate = self.region_gate[r](out["region_summary"]).squeeze(-1)
                scores.append(self.alpha_region * conf + gate)
            stacked = torch.stack(scores, dim=-1)              # [B, T, R]
            omega = torch.softmax(stacked / tau_region, dim=-1)
            log_stack = torch.stack([o["log_b"] for o in region_outputs],
                                    dim=1)                     # [B, R, T, V]
            # log y[t] proportional to sum_r omega_r(t) log b_r[t]
            log_final = torch.einsum("btr,brtv->btv", omega, log_stack)
            log_y = log_final - torch.logsumexp(log_final, dim=-1, keepdim=True)
        else:
            omega = torch.ones(input_ids.shape[0], input_ids.shape[1], self.R) / self.R
            log_y = region_outputs[-1]["log_b"]

        return {"final_log_probs": log_y,
                "logits": log_y,
                "region_outputs": region_outputs,
                "region_weights": omega,
                "per_pass_last_region": region_outputs[-1]["per_pass"]}
