"""E5 driver: UBM v1 checkpoint promotion on a trained checkpoint (runner job).

Implements spec 07 §6.2 end-to-end:
  1. harvest candidates — sequences (2..6 bases) that REGION 1 predicted with
     high plurality agreement, streamed window-by-window over a train slice
     (never materializes more than one window's [., N, V] block),
  2. rank by agreement x confidence x log(1+count), keep top K,
  3. promote() creates bases (mean-of-parts embedding init), settles ONLY the
     new rows for S steps, and runs the held-out bpc rollback test,
  4. writes: updated vocab JSON (version bumped), promotion report JSON, and
     a promoted checkpoint (checkpoint_promoted.pt) next to the original.

Usage:
  python scripts/promote_vocab.py --ckpt-dir checkpoints/<run>_seed0 \
      --vocab UBM/vocab_v3.json --out-vocab UBM/vocab_v4_promoted.json \
      [--top-k 16] [--settle-steps 500] [--rollback-epsilon 0.002] \
      [--candidate-chars 100000] [--val-chars 60000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.common import load_run                    # noqa: E402
from train.evaluate import _equal_length_batches, prepare_text, EVAL_BATCH  # noqa: E402
from train.losses import main_loss                         # noqa: E402
from data.dataset import Batcher                           # noqa: E402
from UBM.promotion import CandidateLog, promote            # noqa: E402
from models import consensus                               # noqa: E402


@torch.no_grad()
def harvest_candidates(model, ubm, train_slice, *, tau0, seq_len, max_chars):
    """Stream region-1 predictions over the slice; update a CandidateLog."""
    from train.evaluate import prepare_text as _pt
    ids, expand_len = _pt(ubm, train_slice)
    context = seq_len or getattr(model, "context_length", 256) or 256
    device = next(model.parameters()).device
    log = CandidateLog()
    chars = 0
    n_windows = 0
    starts = list(range(0, max(len(ids) - 1, 1), context))
    for group in _equal_length_batches(starts, ids, context, EVAL_BATCH):
        L = min(context, len(ids) - group[0] - 1)
        x = torch.tensor([ids[s:s + L] for s in group],
                         dtype=torch.long, device=device)
        out = model(x, tau0=tau0)
        ro = out["region_outputs"][0]["per_pass"][-1]     # region 1, last pass
        log_pi = ro["log_pi"].float()                     # [K, T, N, V]
        log_b = ro["log_b"].float()                       # [K, T, V]
        agree = consensus.plurality_agreement(log_pi.exp(), log_b.exp())
        conf = log_b.exp().amax(-1)
        pred = log_b.argmax(-1).reshape(-1)
        a_f = agree.reshape(-1).tolist()
        c_f = conf.reshape(-1).tolist()
        m = pred.numel()
        for n in range(2, 7):
            if m < n:
                break
            grams = pred.unfold(0, n, 1).tolist()
            for gram, a, c in zip(grams, a_f[:m - n + 1], c_f[:m - n + 1]):
                log.update(gram, a, c)
        n_windows += 1
        for s in group:
            chars += sum(expand_len[s + 1 + i] for i in range(L))
        if chars >= max_chars:
            break
    return log, {"n_windows": n_windows, "n_chars": chars}


def make_settle_fn(model, train_slice, seq_len, batch, lr, tau0):
    """Trains ONLY the new embedding rows for S steps (spec 6.2 step 6).

    Freeze-by-grad-zeroing: the whole weight tensor keeps requires_grad, but
    after each backward the grad rows of pre-existing bases are zeroed, so
    only promoted rows move. weight_decay=0 so AdamW's decoupled decay cannot
    touch the frozen rows either.
    """
    def settle_fn(m, ubm, new_ids, steps):
        before = {n: p.requires_grad for n, p in m.named_parameters()}
        for p in m.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW([m.embed.weight], lr=lr, weight_decay=0.0)
        ids = np.asarray(ubm.encode(train_slice), dtype=np.int64)
        b = Batcher(ids, seq_len, batch, seed=0, shuffle=True)
        rows = torch.zeros(m.embed.weight.shape[0])
        rows[torch.tensor(new_ids, dtype=torch.long)] = 1.0
        m.train()
        losses = []
        for _ in range(int(steps)):
            x, y = b.next_batch()
            xb = torch.from_numpy(np.ascontiguousarray(x))
            yb = torch.from_numpy(np.ascontiguousarray(y))
            out = m(xb, tau0=tau0)
            loss = main_loss(out["final_log_probs"], yb)
            opt.zero_grad()
            loss.backward()
            with torch.no_grad():
                g = m.embed.weight.grad
                if g is not None:
                    g *= rows[:, None]
            opt.step()
            losses.append(float(loss.item()))
        m.eval()
        for n, p in m.named_parameters():
            p.requires_grad_(before.get(n, True))
        print(f"settle: {steps} steps, loss {losses[0]:.4f} -> "
              f"{losses[-1]:.4f} (new rows only)", flush=True)
    return settle_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--config-fallback", default=None)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out-vocab", required=True)
    ap.add_argument("--out-report", default=None)
    ap.add_argument("--train-file", default="data/train.txt")
    ap.add_argument("--val-file", default="data/val.txt")
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--settle-steps", type=int, default=500)
    ap.add_argument("--settle-lr", type=float, default=1e-3)
    ap.add_argument("--settle-batch", type=int, default=8)
    ap.add_argument("--rollback-epsilon", type=float, default=0.002)
    ap.add_argument("--candidate-chars", type=int, default=100_000)
    ap.add_argument("--candidate-start", type=int, default=0)
    ap.add_argument("--val-chars", type=int, default=60_000)
    ap.add_argument("--seq-len", type=int, default=None)
    args = ap.parse_args()

    r = load_run(args.ckpt_dir, args.config_fallback, args.vocab)
    model, ubm, tau0 = r["model"], r["ubm"], r["tau0"]
    seq_len = args.seq_len or int(r["config"].get("data", {}).get("seq_len", 256))
    train_slice = Path(args.train_file).read_text(encoding="utf-8")
    train_slice = train_slice[args.candidate_start:
                              args.candidate_start + args.candidate_chars * 3]
    val_full = Path(args.val_file).read_text(encoding="utf-8")
    val_slice = val_full[:args.val_chars]

    print(f"harvesting candidates from {len(train_slice)} chars of train "
          f"(vocab {len(ubm.bases)} bases, tau0 {tau0:.4f})", flush=True)
    t0 = time.perf_counter()
    log, harvest_stats = harvest_candidates(model, ubm, train_slice,
                                            tau0=tau0, seq_len=seq_len,
                                            max_chars=args.candidate_chars)
    print(f"  {len(log._stats)} distinct n-grams from {harvest_stats['n_windows']} "
          f"windows ({time.perf_counter() - t0:.0f}s)", flush=True)

    candidates = log.top_candidates(args.top_k)
    if not candidates:
        print("no candidates harvested — nothing to promote", flush=True)
        return
    for c in candidates[:5]:
        print(f"  cand {''.join(ubm.expand(i).decode('utf-8', 'replace') for i in c['base_ids'])!r} "
              f"n={c['count']} agree={c['mean_plurality_agreement']:.3f} "
              f"conf={c['mean_confidence']:.3f}", flush=True)

    def eval_bpc(text, _m=model, _u=ubm):
        from train.evaluate import evaluate_bpc
        return evaluate_bpc(_m, _u, text, tau0=tau0,
                            prepared=prepare_text(_u, text))

    report = promote(ubm, model, candidates, args.settle_steps,
                     args.rollback_epsilon, val_slice, eval_bpc=eval_bpc,
                     settle_fn=make_settle_fn(model, train_slice, seq_len,
                                              args.settle_batch,
                                              args.settle_lr, tau0)
                     if args.settle_steps > 0 else None,
                     birth_step=int(r["best_step"]))
    report["harvest"] = harvest_stats
    report["source_vocab"] = args.vocab
    report["ckpt_dir"] = args.ckpt_dir

    ubm.vocab_version += 1
    ubm.save(args.out_vocab)
    report["vocab_version_after"] = ubm.vocab_version
    report["vocab_file"] = args.out_vocab

    out_report = args.out_report or str(Path("results") /
                                         f"promotion_step{r['best_step']}.json")
    Path(out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(out_report).write_text(json.dumps(report, indent=1, default=str))

    from train.train import save_checkpoint
    save_checkpoint(Path(args.ckpt_dir) / "checkpoint_promoted.pt", {
        "step": int(r["best_step"]), "model": model.state_dict(),
        "best_state": model.state_dict(), "best_step": int(r["best_step"]),
        "promoted_from": str(args.ckpt_dir), "vocab_file": args.out_vocab,
        "promotion_report": report})

    print(f"promotion {'ROLLED BACK' if report['rolled_back'] else 'kept'}: "
          f"bpc {report['bpc_before']:.5f} -> {report['bpc_after']:.5f} "
          f"(eps {args.rollback_epsilon}) | promoted {len(report['promoted_ids'])} "
          f"bases -> {args.out_vocab}", flush=True)
    print(f"report: {out_report}", flush=True)


if __name__ == "__main__":
    main()
