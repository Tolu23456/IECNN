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

    def __init__(self, vocab_size: int, d: int, n_heads: int, n_layers: int, seq_len: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(seq_len, d)
        self.blocks = nn.ModuleList([
            _CausalBlock(d, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d)
        self.out = nn.Parameter(torch.zeros(vocab_size, d))
        with torch.no_grad():
            self.out.copy_(self.embed.weight)

    def forward(self, x: torch.Tensor, hard_mode: bool = False) -> dict:
        """Causal forward; returns dict with logits and final_log_probs."""
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device)
        h = self.embed(x) + self.pos(pos_ids)[None]
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = F.linear(h, self.out)
        return {"logits": logits,
                "final_log_probs": F.log_softmax(logits, dim=-1)}


class _CausalBlock(nn.Module):
    """Pre-LN attention + MLP block with a strictly causal mask."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
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
    )
