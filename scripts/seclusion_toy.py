"""Small-scale seclusion A/B (Tolu, 2026-11-17): does the seclusion read
(hard top-1, straight-through backward) learn on toy data, and what do the
dots pick? Soft vs secluded, same data/seed/shapes, tinyshakespeare
char-vocab, 1 region, 2 passes, 4 dots. Also reports per-dot pick stats.
Run: python3 scripts/seclusion_toy.py [steps]
"""
import sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from models.iecnn import build_iecnn

D, SEQ = 32, 32
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400

# ---- data: tinyshakespeare char vocab -> the model's own tokenizer is
# overkill here; a char-vocab keeps the toy interpretable.
text = open("data/tinyshakespeare.txt", encoding="utf-8").read()
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
ids = torch.tensor([stoi[c] for c in text[:200000]], dtype=torch.long)


def batch(bs):
    ix = torch.randint(0, len(ids) - SEQ - 1, (bs,))
    x = torch.stack([ids[i:i + SEQ] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + SEQ] for i in ix])
    return x, y


def make(read_mode, seed=0):
    torch.manual_seed(seed)
    cfg = {
        "model": {"n_regions": 1, "n_dots": 4, "n_passes": 2, "d": D,
                  "d_s": D, "d_v": 16, "d_q": 8, "read_mode": read_mode,
                  "use_weights": True, "verifier": False,
                  "spans": [8]},
        "consensus": {"rho": 0.5, "kappa": 0.7},
    }
    return build_iecnn(cfg, V)


def train(m, steps, lr=3e-3):
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    out = []
    for s in range(steps):
        x, y = batch(16)
        lp = m(x, tau0=1.0, floor=0.0)["final_log_probs"]
        loss = -lp[:, :-1].gather(2, y[:, 1:, None]).squeeze(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % (steps // 5) == 0:
            m.eval()
            with torch.no_grad():
                x, y = batch(32)
                lp = m(x, tau0=1.0, floor=0.0)["final_log_probs"]
                bpc = -lp[:, :-1].gather(2, y[:, 1:, None]).squeeze(2).mean() / \
                    torch.log(torch.tensor(2.0))
            m.train()
            out.append((s + 1, bpc.item()))
    return out


def pick_stats(m, n_batches=4):
    """What does each dot commit to? Chosen window-offset histogram +
    distinct picks per position (seclusion's core claim: each dot reads
    exactly ONE token, and different dots pick differently)."""
    r = m.regions[0]
    hist = None                    # sized to the window
    m.eval()
    for _ in range(n_batches):
        x, _ = batch(8)
        with torch.no_grad():
            rows = m.embed(x)                  # [B, T, d] — ids -> embeddings
            wind_k, wind_v, valid = r._window_memory(rows, rows.device)
            w = wind_k.shape[2]
            scores = torch.tensordot(wind_k.reshape(-1, w, r.d_q),
                                     r.dots._sel(0, r.dots.q), dims=([2], [1]))
            scores = scores.permute(0, 2, 1)                        # [f, N, w]
            if valid is not None:
                Bm = wind_k.shape[0]
                mask = (~valid).repeat(Bm, 1, 1).reshape(-1, 1, w)  # [f,1,w]
                scores = scores.masked_fill(mask, float("-inf"))
            idx = scores.argmax(-1)            # [f, N]
            bc = torch.bincount(idx.flatten(), minlength=w)[:w].float()
            hist = bc if hist is None else hist + bc
    tot = hist.sum()
    return (hist / tot).tolist(), tot.item()


t0 = time.time()
for mode in ("soft", "secluded"):
    m = make(mode)
    curve = train(m, STEPS)
    bpc0, bpcN = curve[0][1], curve[-1][1]
    print(f"[{mode:9s}] bpc: {bpc0:.4f} -> {bpcN:.4f}  "
          f"curve: {['%.3f' % b for _, b in curve]}  ({time.time()-t0:.0f}s)")
    h, n = pick_stats(m)
    top = sorted(range(len(h)), key=lambda i: -h[i])[:3]
    print(f"[{mode:9s}] picks: {n:.0f} reads; offset hist "
          f"{['%.2f' % v for v in h]}; top offsets {top}")
    # causality sanity: secluded model must still be causal
print("done")
