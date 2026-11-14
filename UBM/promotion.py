"""UBM v1: model-driven checkpoint promotion (spec 07, section 6.2).

Built 2026-11-14. The loop:
  1. candidates = short base sequences (2..6) region 1 predicted with high
     plurality agreement, many times (CandidateLog; fed by the training loop
     or the standalone driver).
  2. rank by frequency x agreement x confidence, keep top K.
  3. create one new base per survivor, `parts` = the two-part split of its
     CURRENT encoding that reproduces the sequence. Candidates whose current
     encoding is not exactly two bases are skipped (a `parts` entry must be
     two existing base ids; a >2 encoding would need intermediate bases —
     recorded honestly in the report as `skipped`).
  4. embedding init = MEAN of the two constituent rows (tied embedding: one
     table serves memory rows and every dot's output head, so behaviour
     barely changes at the moment of promotion).
  5. settle: `settle_fn` trains ONLY the new embedding rows for S steps
     (everything else frozen).
  6. rollback test: held-out bpc after vs before; if worse by more than
     `rollback_epsilon`, tombstone every promoted base and restore the
     original embedding table.

`promote` never touches the optimizer: callers manage training around it and
should rebuild their optimizer if they continue training with a grown table.
"""
from __future__ import annotations

import math

import torch

from UBM.tokenizer import BaseRecord

CAND_MIN_LEN = 2
CAND_MAX_LEN = 6


class CandidateLog:
    """Logs short base sequences that region 1 predicted with high agreement."""

    def __init__(self, min_len: int = CAND_MIN_LEN,
                 max_len: int = CAND_MAX_LEN):
        self.min_len = min_len
        self.max_len = max_len
        self._stats: dict[tuple[int, ...], list] = {}   # key -> [count, sum_agree, sum_conf]

    def update(self, base_ids, plurality_agreement, confidence) -> None:
        """Record one observation of a predicted base sequence.

        base_ids: the sequence (2..6 ids); plurality_agreement/confidence:
        the label-free signals at the position the sequence was predicted.
        """
        key = tuple(int(b) for b in base_ids)
        if not (self.min_len <= len(key) <= self.max_len):
            return
        st = self._stats.setdefault(key, [0, 0.0, 0.0])
        st[0] += 1
        st[1] += float(plurality_agreement)
        st[2] += float(confidence)

    def top_candidates(self, k: int) -> list:
        """Rank by frequency, agreement, and confidence together.

        score = mean_agreement x mean_confidence x log(1 + count) — all
        three label-free (plurality agreement is a diagnostic per C5, the
        consensus confidence is computed from dot outputs only per C2).
        """
        out = []
        for key, (count, s_agree, s_conf) in self._stats.items():
            if count == 0:
                continue
            agree = s_agree / count
            conf = s_conf / count
            out.append({
                "base_ids": list(key),
                "count": count,
                "mean_plurality_agreement": agree,
                "mean_confidence": conf,
                "score": agree * conf * math.log1p(count),
            })
        out.sort(key=lambda c: -c["score"])
        return out[:max(0, int(k))]


def _plan_promotions(ubm, candidates: list) -> tuple[list, list]:
    """Keep candidates whose CURRENT encoding is exactly two bases."""
    plan, skipped = [], []
    seen: set[bytes] = set()
    for c in candidates:
        ids = [int(i) for i in c["base_ids"]]
        try:
            payload = b"".join(ubm.expand(i) for i in ids)
            text = payload.decode("utf-8")
        except (UnicodeDecodeError, IndexError):
            skipped.append({"candidate": ids, "reason": "not valid utf-8"})
            continue
        if not payload or payload in seen:
            skipped.append({"candidate": ids, "reason": "empty or duplicate"})
            continue
        cur = ubm.encode(text)
        if len(cur) != 2:
            skipped.append({"candidate": ids, "reason":
                            f"encodes to {len(cur)} bases, need exactly 2"})
            continue
        seen.add(payload)
        plan.append({**c, "bytes": payload, "parts": (int(cur[0]), int(cur[1]))})
    return plan, skipped


def promote(ubm, model, candidates, settle_steps: int, rollback_epsilon: float,
            val_text: str, *, eval_bpc=None, settle_fn=None,
            birth_step: int = 0) -> dict:
    """Create bases, init embeddings as mean of parts, settle, roll back if
    worse (spec 07 §6.2 steps 3-7).

    eval_bpc(text) -> bpc (defaults to train.evaluate.evaluate_bpc with the
    model's own settings; callers with an annealed tau should pass a closure).
    settle_fn(model, ubm, new_ids, steps) trains only the new rows (required
    for a non-zero settle_steps — the library never trains on its own).
    """
    if eval_bpc is None:
        from train.evaluate import evaluate_bpc

        def eval_bpc(text, _m=model, _u=ubm):    # noqa: E704
            return evaluate_bpc(_m, _u, text)

    plan, skipped = _plan_promotions(ubm, candidates)

    old_embed = getattr(model, "embed", None)
    old_weight = None if old_embed is None else old_embed.weight.detach().clone()
    bpc_before = float(eval_bpc(val_text))

    new_ids: list[int] = []
    records = []
    for rec in plan:
        rec_id = len(ubm.bases)
        ubm.add_bases([BaseRecord(id=rec_id, bytes=rec["bytes"],
                                  parts=rec["parts"], origin="promotion",
                                  birth_step=birth_step,
                                  count=int(rec["count"]))])
        new_ids.append(rec_id)
        records.append({"id": rec_id,
                        "text": rec["bytes"].decode("utf-8"),
                        "parts": list(rec["parts"]),
                        "count": int(rec["count"]),
                        "mean_plurality_agreement":
                            round(float(rec["mean_plurality_agreement"]), 5),
                        "mean_confidence":
                            round(float(rec["mean_confidence"]), 5),
                        "score": round(float(rec["score"]), 6)})

    # grow the tied embedding: new rows = mean of the two constituent rows
    if new_ids and old_embed is not None:
        v0, d = old_weight.shape
        grown = torch.zeros(v0 + len(new_ids), d)
        grown[:v0] = old_weight
        for j, rec in zip(new_ids, records):
            a, b = rec["parts"]
            grown[j] = (old_weight[a] + old_weight[b]) / 2.0
        new_embed = torch.nn.Embedding(v0 + len(new_ids), d)
        with torch.no_grad():
            new_embed.weight.copy_(grown)
        model.embed = new_embed
        if hasattr(model, "vocab_size"):
            model.vocab_size = v0 + len(new_ids)

    settle_skipped = False
    if new_ids and settle_steps > 0:
        if settle_fn is not None:
            settle_fn(model, ubm, new_ids, int(settle_steps))
        else:
            settle_skipped = True

    bpc_after = float(eval_bpc(val_text))
    rolled_back = bool(new_ids) and bpc_after > bpc_before + rollback_epsilon
    if rolled_back:
        for i in new_ids:
            ubm.tombstone(i)
        if old_embed is not None:
            model.embed = old_embed          # original module object back
            if hasattr(model, "vocab_size"):
                model.vocab_size = old_weight.shape[0]
        bpc_final = float(eval_bpc(val_text))
    else:
        bpc_final = bpc_after

    return {
        "bpc_before": round(bpc_before, 6),
        "bpc_after": round(bpc_after, 6),
        "bpc_final": round(bpc_final, 6),
        "bpc_delta": round(bpc_after - bpc_before, 6),
        "rollback_epsilon": rollback_epsilon,
        "rolled_back": rolled_back,
        "promoted": records,
        "promoted_ids": new_ids if not rolled_back else [],
        "tombstoned_ids": new_ids if rolled_back else [],
        "skipped": skipped,
        "settle_steps": int(settle_steps) if new_ids else 0,
        "settle_skipped": settle_skipped,
        "note": "promotion = spec 07 6.2: candidates from region-1 agreement, "
                "mean-of-parts init, settle new rows only, batch rollback "
                "test on held-out bpc",
    }
