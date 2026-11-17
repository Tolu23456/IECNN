"""Slot-table toy (Tolu, 2026-11-17): does a per-dot persistent slot table
(cross-position memory, gated DETACHED writes) beat the same model without
it, at matched window? The window read is attention's KV-cache analogue
(raw past-token recordings, write-once); the table is a learned notebook.

Arms (same seed, same params except the table's own):
  none   : window W=8, no table
  table  : window W=8 + S private slots per dot, carried across positions
  wide   : window W=32, no table  (control: notebook vs longer recording)

Run: python3 scripts/slot_toy.py [steps]
"""
import sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

D, N, S, T = 32, 4, 4, 48
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0

text = open("data/tinyshakespeare.txt", encoding="utf-8").read()
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
tr = torch.tensor([stoi[c] for c in text[:250000]])
va = torch.tensor([stoi[c] for c in text[250000:300000]])


def batch(pool, bs):
    ix = torch.randint(0, len(pool) - T - 1, (bs,))
    x = torch.stack([pool[i:i + T] for i in ix])
    y = torch.stack([pool[i + 1:i + 1 + T] for i in ix])
    return x, y


class Model(nn.Module):
    """Minimal recurrent dot population. Each position t:
    read window of past raw embeddings (KV-cache analogue) -> context c;
    read own slot slice -> r (differentiable wrt read proj only);
    GRU update s = GRU([c ; r ; s]); gated detached write into slots.
    Slots differentiate via per-slot write heads (S distinct heads)."""

    def __init__(self, use_table, W):
        super().__init__()
        self.use_table, self.W = use_table, W
        self.emb = nn.Embedding(V, D)
        self.q = nn.Parameter(torch.randn(N, D) * 0.1)   # per-dot queries
        self.k = nn.Linear(D, D, bias=False)             # window keys
        self.gru = nn.GRUCell(D + (D if use_table else 0) + D, D)
        if use_table:
            self.r_proj = nn.Linear(S * D, D)            # slot read
            self.w_gate = nn.Linear(D, S)                # per-slot write gate
            self.w_val = nn.Linear(D, S * D)             # per-slot write value
        self.head = nn.Linear(D, D, bias=False)          # pooled -> tied emb
        self.gate_sum, self.gate_n = 0.0, 0

    def forward(self, x):
        B = x.shape[0]
        E = self.emb(x)                                   # [B,T,D] (grad path)
        Et = self.emb.weight                              # tied output table
        s = torch.zeros(B, N, D, device=x.device)
        M = torch.zeros(B, N, S, D, device=x.device)      # slots: no grad ever
        logits = []
        for t in range(T):
            lo = max(0, t - self.W + 1)
            win = E[:, lo:t + 1]                          # [B,w,D] raw past
            att = torch.einsum('nd,bwd->bnw', self.q, self.k(win))
            w = (att / D ** 0.5).softmax(-1)
            c = torch.einsum('bnw,bwd->bnd', w, win)      # [B,N,D]
            if self.use_table:
                r = self.r_proj(M.reshape(B, N, S * D))   # grad -> r_proj only
                inp = torch.cat([c, r, s], -1)
            else:
                inp = torch.cat([c, s], -1)
            s = self.gru(inp.reshape(B * N, -1),
                         s.reshape(B * N, D)).reshape(B, N, D)
            if self.use_table:
                with torch.no_grad():                     # fast-weight write
                    g = torch.sigmoid(self.w_gate(s))     # [B,N,S]
                    v = torch.tanh(self.w_val(s)).reshape(B, N, S, D)
                    M = 0.5 * M + 0.5 * g.unsqueeze(3) * v
                if self.training:
                    self.gate_sum += g.mean().item(); self.gate_n += 1
            logits.append(self.head(s.mean(1)) @ Et.T)    # pooled dots -> tied
        return torch.stack(logits, 1)                     # [B,T,V]


def run(arm, steps):
    use_table, W = {"none": (False, 8), "table": (True, 8),
                    "wide": (False, 32)}[arm]
    torch.manual_seed(SEED)
    m = Model(use_table, W)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    t0 = time.time()
    curve = []
    for step in range(1, steps + 1):
        x, y = batch(tr, 16)
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % (steps // 4) == 0 or step == 1:
            m.eval()
            with torch.no_grad():
                vx, vy = batch(va, 32)
                vb = F.cross_entropy(m(vx).reshape(-1, V), vy.reshape(-1))
            m.train()
            curve.append(vb.item() / torch.log(torch.tensor(2.0)).item())
    div = (f"  write-gate avg {m.gate_sum / max(m.gate_n, 1):.3f}"
           if use_table else "")
    return curve, time.time() - t0, div


t0 = time.time()
for arm in ("none", "table", "wide"):
    torch.manual_seed(SEED)
    curve, dt, div = run(arm, STEPS)
    print(f"[{arm:6s}] val bpc: {curve[0]:.3f} -> {curve[-1]:.3f}  "
          f"curve {['%.3f' % b for b in curve]}  ({dt:.0f}s){div}")
print(f"total {time.time() - t0:.0f}s")
