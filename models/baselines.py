"""Baselines B0 (small GRU) and B1 (small causal transformer) (spec 08, section 4).

Both must match IECNN's parameter count roughly (about 5M by default, smaller
with the tiny overlay) and behave causally (perturbing a future token cannot
change earlier outputs).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRULanguageModel(nn.Module):
    """B0: embedding -> GRU -> tied-output logits over next token."""

    context_length = 256

    def __init__(self, vocab_size: int, d: int, hidden: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.embed = nn.Embedding(vocab_size, d)
        self.cell = nn.GRU(input_size=d, hidden_size=hidden, batch_first=True)
        self.out = nn.Parameter(torch.zeros(vocab_size, hidden))
        nn.init.xavier_uniform_(self.out)

    def forward(self, x: torch.Tensor, hard_mode: bool = False) -> dict:
        """x [B, T] integer ids -> dict with logits and final_log_probs."""
        h0 = torch.zeros(1, x.shape[0], self.hidden, device=x.device)
        h, _state = self.cell(self.embed(x), h0)
        logits = F.linear(h, self.out)                     # [B, T, V]
        return {"logits": logits,
                "final_log_probs": F.log_softmax(logits, dim=-1)}


class TransformerLanguageModel(nn.Module):
    """B1: small pre-LN causal transformer with learned absolute positions."""

    context_length = 256

    def __init__(self, vocab_size: int, d: int, n_heads: int, n_layers: int,
                 seq_len: int = 256, ngram_r: int = 0, window: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(seq_len, d)
        # GPT-style init: small-normal embeddings keep tied logits near zero.
        # (Default N(0,1) embeds + tied head gave init logits ~N(0, d_model),
        #  i.e. a confidently-wrong model: init loss 173 nats on the full config.)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)
        # window: DIAGNOSTIC ONLY, default 0 = full causal attention.
        #
        # `swa` in the arm registry is stochastic weight averaging, not
        # sliding-window attention, so the baseline has always had full access
        # to all T=256 positions. window=k restricts every query to the k most
        # recent positions (k includes itself). That makes the transformer
        # read the same 8-slot horizon the dot model's window has, at the same
        # parameter count, which is the only clean way to ask how much of the
        # bpc gap is the READ HORIZON rather than the interaction rule.
        self.window = int(window)
        self.blocks = nn.ModuleList([
            _CausalBlock(d, n_heads, window=self.window) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d)
        self.out = nn.Parameter(torch.zeros(vocab_size, d))
        with torch.no_grad():
            self.out.copy_(self.embed.weight)
        # fairness control: the SAME rank-r exact order-1 prior the IECNN
        # arms get (see models/region.py). Identical compute and params, so a
        # head-to-head stays like-for-like.
        self.ngram_r = int(ngram_r)
        if self.ngram_r > 0:
            e1 = torch.empty(vocab_size, self.ngram_r)
            e2 = torch.empty(vocab_size, self.ngram_r)
            nn.init.normal_(e1, std=0.02)
            nn.init.normal_(e2, std=0.02)
            self.ng_e1 = nn.Parameter(e1)
            self.ng_e2 = nn.Parameter(e2)
        else:
            self.ng_e1 = None
            self.ng_e2 = None

    def forward(self, x: torch.Tensor, hard_mode: bool = False) -> dict:
        """Causal forward; returns dict with logits and final_log_probs."""
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device)
        h = self.embed(x) + self.pos(pos_ids)[None]
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = F.linear(h, self.out)
        if self.ng_e1 is not None:
            logits = logits + self.ng_e1[x].matmul(self.ng_e2.t())
        return {"logits": logits,
                "final_log_probs": F.log_softmax(logits, dim=-1)}


class _CausalBlock(nn.Module):
    """Pre-LN attention + MLP block with a strictly causal mask."""

    def __init__(self, d: int, n_heads: int, window: int = 0):
        super().__init__()
        self.window = int(window)
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.mlp_in = nn.Linear(d, 4 * d)
        self.mlp_out = nn.Linear(4 * d, d)
        self.n_heads = n_heads
        self.d = d

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h [B, T, d] -> h [B, T, d]; each position sees only the past."""
        B, T, d = h.shape
        q, k, v = self.qkv(self.ln1(h)).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, d // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, d // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, d // self.n_heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d // self.n_heads)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=h.device))
        if 0 < self.window < T:
            # keep j >= i - window + 1 as well as j <= i (window=1 -> self only)
            recent = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                           device=h.device),
                                diagonal=-(self.window - 1))
            causal = causal & recent
        att = att.masked_fill(~causal[None, None], float("-inf"))
        att = F.softmax(att, dim=-1)
        h = h + self.proj((att @ v).transpose(1, 2).reshape(B, T, d))
        h = h + self.mlp_out(F.gelu(self.mlp_in(self.ln2(h))))
        return h


def build_gru_baseline(config: dict, vocab_size: int):
    """GRU baseline sized from the config (model.d and model.gru_hidden)."""
    m = config["model"]
    return GRULanguageModel(vocab_size, d=int(m.get("d", 128)),
                            hidden=int(m.get("gru_hidden", 4 * m.get("d", 128))))


def build_transformer_baseline(config: dict, vocab_size: int):
    """Transformer baseline sized from the config (model.d_model etc.)."""
    m = config["model"]
    d_model = int(m.get("d_model", 2 * m.get("d", 128)))
    return TransformerLanguageModel(
        vocab_size, d=d_model,
        n_heads=int(m.get("n_heads", 4)),
        n_layers=int(m.get("n_layers", 4)),
        seq_len=int(config.get("data", {}).get("seq_len", 256)),
        ngram_r=int(m.get("ngram_bias", 0)),
        window=int(m.get("window", 0)),
    )
