"""Is the model writing the PROMPT, or its TRAINING DATA's idiolect?

The observation that prompted this: the dot model's generations look like the
corpus it was trained on — its diction, names, markup habits — more than the
transformer's do, even when the prompt is about something else. That is exactly
what a type-level read predicts. With a constant query the read is a
conditional-probability table over fixed horizons, so the only thing that can
condition the output beyond the 8-byte window is the CORPUS PRIOR; a
content-addressed read can instead copy a specific earlier span (the prompt) and
hold on to a plan.

Four byte-4-gram statistics per 200-byte generation make that testable:

  copy   4-grams that occur in the PROMPT but NOWHERE in the training corpus
         (only a model that actually reads the prompt can emit them)
  hot    4-grams that occur >= HOT times in the training corpus (the corpus's
         own idiolect, i.e. "flavour")
  novel  4-grams that occur nowhere in train or prompt (invention)
  truth  fraction matching the true continuation (a reference, not a score)

and one distributional number, the log-likelihood ratio

  LR = mean over generated trigrams t of log[(c_prompt(t)+a)/(c_corpus(t)+a)]

which is positive when the generation's trigrams are more typical of the prompt
than of the corpus at large. LR > 0 means "wrote the prompt"; LR < 0 means
"wrote the training data".

Prompts are drawn from the VAL split at several offsets, so they were never
trained on and their rare 4-grams are genuinely prompt-only evidence.

Run (same env as ab_small):
    ARM_TRAIN=... ARM_VAL=... ARM_NTRAIN=... ARM_NVAL=... \
    ARM_B=32 ARM_T=256 ARM_VALW=160 \
    python3 -u scripts/flavour_probe.py 360 tf-swa,iecnn-...-w512

    SC_PROMPTS=12 SC_LEN=200 SC_TEMP=0.7 SC_TOPK=40 SC_HOT=50
    ARM_LOAD=/kaggle/input/<dataset>   # skip training, load <arm>.pt weights
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
ARMS = sys.argv[2].split(",") if len(sys.argv) > 2 else [
    "tf-swa", "iecnn-n3-x3-p1-ln-decay-d128-w512"]
N_PROMPT = int(os.environ.get("SC_PROMPTS", "12"))
PROMPT_N = int(os.environ.get("SC_PROMPT", "200"))
GEN_N = int(os.environ.get("SC_LEN", "200"))
TEMP = float(os.environ.get("SC_TEMP", "0.7"))
TOPK = int(os.environ.get("SC_TOPK", "40"))
HOT = int(os.environ.get("SC_HOT", "50"))
SC_SEED = int(os.environ.get("SC_SEED", "1234"))
N_OOD = int(os.environ.get("SC_OOD_PROMPTS", "6"))

# Out-of-domain prompts: the model was trained on enwik8 (or Gutenberg); this
# is Shakespeare, embedded so the test is identical on every runner. If the
# model's output on an OOD prompt is still corpus-typical, it is writing its
# training data's flavour; if it drifts toward the prompt's trigrams, it is
# conditioning on the prompt.
OOD_TEXT = b"""To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream: ay, there's the rub,
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of dispriz'd love, the law's delay,
The insolence of office, and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry
And lose the name of action."""


_argv = sys.argv
sys.argv = ["ab_small.py", str(BUDGET)]
_spec = importlib.util.spec_from_file_location(
    "ab_small", os.path.join(HERE, "scripts", "ab_small.py"))
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)
sys.argv = _argv

DEV = ab.DEV
T = ab.T


def _keys4(b: np.ndarray) -> np.ndarray:
    """4-gram keys as uint32 (byte-level, so a 4-gram is exactly 4 bytes)."""
    if b.size < 4:
        return np.zeros(0, dtype=np.uint32)
    b = b.astype(np.uint32)
    return (b[:-3] << 24) | (b[1:-2] << 16) | (b[2:-1] << 8) | b[3:]


class CorpusCounts:
    """4-gram counts over the training bytes (uint32 keys, sorted)."""

    def __init__(self, train: np.ndarray):
        k = _keys4(train)
        self.keys, self.counts = np.unique(k, return_counts=True)

    def count(self, k: np.ndarray) -> np.ndarray:
        if k.size == 0:
            return np.zeros(0, dtype=np.int64)
        i = np.searchsorted(self.keys, k)
        i = np.clip(i, 0, len(self.keys) - 1)
        return np.where(self.keys[i] == k, self.counts[i], 0).astype(np.int64)


def _tri_counts(b: np.ndarray) -> dict:
    d = {}
    for i in range(len(b) - 2):
        k = (int(b[i]) << 16) | (int(b[i + 1]) << 8) | int(b[i + 2])
        d[k] = d.get(k, 0) + 1
    return d


@torch.no_grad()
def generate(model, prompt: torch.Tensor, n: int, seed: int) -> bytes:
    model.eval()
    g = torch.Generator(device=prompt.device).manual_seed(seed)
    seq = prompt.clone()
    out = []
    for _ in range(n):
        lp = model(seq[:, -T:])["final_log_probs"][:, -1]
        logits = lp / max(TEMP, 1e-6)
        if 0 < TOPK < logits.shape[-1]:
            th = torch.topk(logits, TOPK, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < th, float("-inf"))
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=g)
        out.append(int(nxt))
        seq = torch.cat([seq, nxt.reshape(1, 1)], dim=1)
    return bytes(out)


def scores(gen: bytes, prompt: bytes, truth: bytes, cc: "CorpusCounts",
           corpus_tri: dict, alpha: float = 0.5) -> dict:
    g = np.frombuffer(gen, dtype=np.uint8)
    p = np.frombuffer(prompt, dtype=np.uint8)
    gk, pk = _keys4(g), _keys4(p)
    gc, pc = cc.count(gk), cc.count(pk)
    prompt_set = set(int(x) for x in pk)
    in_prompt = np.array([int(x) in prompt_set for x in gk], dtype=bool) \
        if gk.size else np.zeros(0, dtype=bool)
    tr = np.frombuffer(truth, dtype=np.uint8) if truth else b""
    tk = _keys4(tr)
    tset = set(int(x) for x in tk)
    match = np.mean([int(x) in tset for x in gk]) if gk.size and tset else 0.0
    # distributional test: are the generation's trigrams more typical of the
    # prompt than of the corpus? positive => wrote the prompt.
    gt = _tri_counts(g)
    n_corp = sum(corpus_tri.values())
    pt = _tri_counts(p)
    lr = []
    for k, c in gt.items():
        cp = pt.get(k, 0) / max(1, sum(pt.values()))
        cq = corpus_tri.get(k, 0) / max(1, n_corp)
        lr.append(np.log((cp + alpha) / (cq + alpha)))
    return {
        "copy": float(np.mean(in_prompt & (gc == 0))) if gk.size else 0.0,
        "hot": float(np.mean(gc >= HOT)) if gk.size else 0.0,
        "novel": float(np.mean(gc == 0)) if gk.size else 0.0,
        "match": float(match),
        "lr": float(np.mean(lr)) if lr else 0.0,
    }


def main():
    va_path = os.environ.get("ARM_VAL", "data/toy_val.txt")
    raw_val = open(va_path, "rb").read()
    train_np = ab.train.numpy().astype(np.uint8)
    print(f"=== FLAVOUR probe: corpus {train_np.size} bytes, val {len(raw_val)}, "
          f"{N_PROMPT} prompts x {PROMPT_N}B -> {GEN_N}B, T={TEMP} top-k {TOPK}, "
          f"hot>={HOT}", flush=True)
    cc = CorpusCounts(train_np)
    corpus_tri = _tri_counts(train_np[:2_000_000])
    print(f"train 4-gram vocabulary {len(cc.keys)}", flush=True)

    # prompts spread over the val split (never trained on)
    span = max(1, len(raw_val) - PROMPT_N - GEN_N - 1)
    starts = [int(span * (i + 0.5) / N_PROMPT) for i in range(N_PROMPT)]
    prompts = [("val", raw_val[s:s + PROMPT_N],
                raw_val[s + PROMPT_N:s + PROMPT_N + GEN_N]) for s in starts]
    if N_OOD > 0:
        ood = OOD_TEXT
        ostarts = [int((len(ood) - PROMPT_N - GEN_N) * (i + 0.5) / N_OOD)
                   for i in range(N_OOD)]
        prompts += [("ood", ood[s:s + PROMPT_N],
                     ood[s + PROMPT_N:s + PROMPT_N + GEN_N]) for s in ostarts]

    out = {}
    for name in ARMS:
        load_dir = os.environ.get("ARM_LOAD")
        if load_dir:
            blob = torch.load(os.path.join(load_dir, name.replace('/', '_') + ".pt"),
                              map_location=DEV, weights_only=False)
            cfg, kind = blob["cfg"], blob["kind"]
            model = (ab.build_transformer_baseline(cfg, ab.V) if kind == "transformer"
                     else ab.build_iecnn(cfg, ab.V))
            model.load_state_dict(blob["state_dict"])
            model = model.to(DEV)
            print(f"--- loaded {name} ({kind})", flush=True)
        else:
            entry = ab.ARMS[name]
            kind, mk = entry[0], entry[1]
            opts = dict(entry[2]) if len(entry) > 2 else {}
            opts.pop("compile", None)
            cfg = mk()
            torch.manual_seed(opts.get("seed", ab.SEED))
            model = (ab.build_transformer_baseline(cfg, ab.V) if kind == "transformer"
                     else ab.build_iecnn(cfg, ab.V)).to(DEV)
            print(f"--- training {name} ({sum(p.numel() for p in model.parameters())} "
                  f"params, {BUDGET:.0f}s)", flush=True)
            ab.run(name, kind, model, **opts)
        per_domain = {}
        for i, (dom, pr, tr) in enumerate(prompts):
            pt = torch.tensor(list(pr), dtype=torch.long, device=DEV).unsqueeze(0)
            gen = generate(model, pt, GEN_N, SC_SEED + i)
            per_domain.setdefault(dom, []).append(scores(gen, pr, tr, cc, corpus_tri))
        out[name] = {}
        for dom, rows in per_domain.items():
            agg = {k: (float(np.mean([r[k] for r in rows])),
                       float(np.std([r[k] for r in rows]))) for k in rows[0]}
            out[name][dom] = agg
            print(f"=== {name} [{dom}]\n" + "  ".join(
                f"{k} {m:.3f}±{sd:.3f}" for k, (m, sd) in agg.items()), flush=True)
        del model
        if DEV.type == "cuda":
            torch.cuda.empty_cache()

    names = list(out)
    if len(names) >= 2:
        a, b = names[0], names[1]
        for dom in out[a]:
            print(f"=== {a} vs {b} [{dom}]", flush=True)
            for k in out[a][dom]:
                print(f"  {k:<6} {out[a][dom][k][0]:>8.3f}   "
                      f"{out[b][dom][k][0]:>8.3f}   "
                      f"diff {out[a][dom][k][0] - out[b][dom][k][0]:+.3f}",
                      flush=True)
    print("FLAVOUR DONE", flush=True)


if __name__ == "__main__":
    main()
