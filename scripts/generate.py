"""Runner-side text generation from a trained checkpoint (inference only).

Loads checkpoints/<dir>/checkpoint.pt, encodes a prompt with the run's own
UBM vocabulary, and autoregressively samples a continuation from the model's
next-token distribution (temperature + top-k, seeded). Torch use is allowed
ONLY on GitHub Actions runners — never on the dev box.

Output: results/generation/<slug>.json with the prompt, the generated text,
per-token ids, and full parameter provenance (vocab sha, tau, seed) so any
generation can be reproduced exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from UBM import UBM                                    # noqa: E402
from UBM.tokenizer import fingerprint                  # noqa: E402
from train.schedules import temperature_at             # noqa: E402
from train.train import build_model                    # noqa: E402


def _next_log_probs(model, out: dict, kind: str) -> torch.Tensor:
    """Last-position next-token log-probs for either model family."""
    if kind == "iecnn":
        return out["log_b"][0, -1]                     # region consensus
    return out["final_log_probs"][0, -1]               # GRU / transformer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--vocab", default="UBM/vocab_v1.json")
    ap.add_argument("--prompt", default="",
                    help="prompt text; if empty, read --prompt-file")
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40,
                    help="0 = sample from the full distribution")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompt_text = args.prompt or (Path(args.prompt_file).read_text("utf-8")
                                  if args.prompt_file else "")
    if not prompt_text.strip():
        raise SystemExit("empty prompt: pass --prompt or --prompt-file")

    d = Path(args.ckpt_dir)
    ck = torch.load(d / "checkpoint.pt", weights_only=False)
    summary = json.load(open(d / "summary.json"))
    config = json.load(open(d / "config_used.json"))
    run = d.name
    kind = config["model"]["kind"]

    rec_sha = str(ck.get("vocab_sha256", ""))
    if rec_sha and fingerprint(args.vocab) != rec_sha:
        raise RuntimeError(
            f"VOCAB MISMATCH: checkpoint recorded sha {rec_sha[:16]} but "
            f"{args.vocab} hashes {fingerprint(args.vocab)[:16]}. Generation "
            "with the wrong vocabulary is silently gibberish — fail loudly.")
    ubm = UBM.load(args.vocab)
    V = len(ubm.bases)

    model = build_model(config, V)
    model.load_state_dict(ck.get("best_state") or ck["model"])
    model.eval()

    # eval tau at the checkpoint's best step (same rule as eval/export)
    best_step = int(summary.get("best_step", ck.get("step", 0)))
    steps_target = int(summary.get("steps_target",
                                   config.get("training", {}).get("steps", 1)))
    cons = config.get("consensus", {})
    tau0 = temperature_at(best_step, steps_target,
                          int(cons.get("tau_warmup", 2000)),
                          float(cons.get("tau_max", 1.0)),
                          float(cons.get("tau_min", 0.2)))

    prompt_ids = ubm.encode(prompt_text)
    if not prompt_ids:
        raise SystemExit("prompt normalizes to zero tokens")
    seq_len = int(config.get("data", {}).get("seq_len", 256))

    torch.manual_seed(args.seed)
    ids = list(prompt_ids)
    gen_ids: list[int] = []
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            ctx = torch.tensor([ids[-seq_len:]], dtype=torch.long)
            out = model(ctx, tau0=tau0, floor=0.0) if kind == "iecnn" \
                else model(ctx)
            lp = _next_log_probs(model, out, kind) / args.temperature
            if args.top_k > 0:
                k = min(args.top_k, lp.shape[-1])
                kth = lp.topk(k).values[-1]
                lp = lp.clamp_max(kth)
            probs = lp.softmax(-1)
            nxt = int(torch.multinomial(probs, 1))
            ids.append(nxt)
            gen_ids.append(nxt)
    elapsed = time.time() - t0

    generated = ubm.decode(gen_ids)
    payload = {
        "meta": {
            "run": run,
            "kind": kind,
            "step": best_step,
            "stop_kind": summary.get("stop_kind"),
            "params": summary.get("parameters"),
            "tau0": round(tau0, 5),
            "vocab_sha256": fingerprint(args.vocab),
            "prompt_sha256": hashlib.sha256(
                prompt_text.encode("utf-8")).hexdigest(),
            "temperature": args.temperature,
            "top_k": args.top_k,
            "seed": args.seed,
            "seq_len": seq_len,
        },
        "prompt": prompt_text,
        "prompt_tokens": len(prompt_ids),
        "generated": generated,
        "generated_ids": gen_ids,
        "elapsed_s": round(elapsed, 2),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"generated {len(gen_ids)} tokens in {elapsed:.1f}s -> {out_path}")
    print("PROMPT:", prompt_text[:200].replace("\n", " "))
    print("GENERATED:", generated[:400].replace("\n", " "))


if __name__ == "__main__":
    main()
