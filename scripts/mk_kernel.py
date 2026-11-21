"""Emit a Kaggle kernel dir for one ab_small arm batch on the long corpus.

This generator lives in the repo because it is the protocol: it pins the corpus
split (canonical enwik8 90 MB train / 5 MB dev, last 5 MB never touched), the
run env, the pre-flight gate and the post-run population audit. It used to live
only in a scratch directory and was lost twice.

Usage: mk_kernel.py <tag> <budget_s> <arms> <seed> <n_train> <val_off> <n_val>
                    <mark_s> <gen 0|1> <dir>

  tag       kernel name suffix: voidmaster000/iecnn-<tag>-gpu
  budget_s  seconds per arm (180 is retired for claims; 360 floor, 900 confirm)
  arms      comma-separated ARM names from scripts/ab_small.py
  seed      ARM_SEED
  n_train / val_off / n_val   corpus split, in bytes
  mark_s    ARM_MARK_S (mid-run val marker; 0 = none)
  gen       1 = run the generation probe during the dry pass
  dir       output directory for the two kernel files

The emitted kernel:
  * clones main at run time (so commit+push first),
  * runs ARM_DRY=1 over EVERY arm -- build, one fwd/bwd, one val pass, pointer
    audit, optional generation -- and refuses the timed run if any arm fails,
  * runs the timed arms, then `probe_population.py` on each saved iecnn
    checkpoint so every dot arm reports its drop-one numbers at its own budget.

Glob-free paths on purpose: Kaggle's dynamo/compile path breaks on unordered
globs in some environments, and this keeps the run reproducible.
"""
import json
import os
import sys

TAG, BUDGET, ARMS, SEED, N_TRAIN, VAL_OFF, N_VAL, MARK_S, GEN, OUT = sys.argv[1:]

TEMPLATE = '''"""Kaggle T4 runner for scripts/ab_small.py (long-corpus protocol).

Corpus: enwik8 with the CANONICAL split -- train = bytes [0, {n_train}), val =
bytes [{val_off}, {val_off} + {n_val}), last 5 MB never touched. This is the
standard enwik8 dev split, unlike the 60 MB / 5 MB window used by the first 173
ledger rows, so absolute bpc is NOT comparable with those rows; every session
re-runs its own transformer anchor and claims are in-session only.

Before the timed run the kernel executes `ARM_DRY=1`, which builds every arm,
does one fwd/bwd, one val pass, the pointer audit and (when ARM_GEN=1) a
generation -- so a config or shape error costs seconds, not hours. The timed
run is skipped if any arm fails the dry pass.

Env is fixed here on purpose: Kaggle does not pass the local shell's env.
"""
import json
import os
import subprocess
import sys
import urllib.request
import zipfile

import torch                    # only for reading the checkpoints at the end

TAG = {tag!r}
BUDGET = {budget!r}
ARM_B = "32"
ARM_T = "256"
VALW = "160"
ARMS = {arms!r}
SEED = {seed!r}
MARK_S = {mark_s!r}
GEN = {gen!r}

# canonical enwik8: 90 MB train / 5 MB dev / 5 MB test untouched
N_TRAIN = {n_train}
VAL_OFF = {val_off}
N_VAL = {n_val}

W = "/kaggle/working"
REPO = "https://github.com/Tolu23456/IECNN.git"


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


os.makedirs(W, exist_ok=True)
os.chdir(W)

if not os.path.isdir("IECNN"):
    sh(["git", "clone", "--depth", "1", REPO, "IECNN"], check=True)
os.chdir("IECNN")
sh(["git", "pull", "--ff-only", "origin", "main"], check=False)
sh(["git", "log", "--oneline", "-1"])

sh([sys.executable, "-c",
    "import torch; p=torch.cuda.get_device_properties(0); "
    "print('GPU', torch.cuda.get_device_name(0), p.total_memory//2**30, 'GiB', "
    "'torch', torch.__version__)"])

tr_path = os.path.join(W, "enwik_train.txt")
va_path = os.path.join(W, "enwik_val.txt")

if not (os.path.exists(tr_path) and os.path.exists(va_path)):
    zpath = os.path.join(W, "enwik8.zip")
    if not os.path.exists(zpath):
        for url in ("http://mattmahoney.net/dc/enwik8.zip",
                    "https://mattmahoney.net/dc/enwik8.zip"):
            try:
                urllib.request.urlretrieve(url, zpath)
                break
            except Exception as e:                          # noqa: BLE001
                print("download failed:", url, e, flush=True)
    with zipfile.ZipFile(zpath) as z:
        raw = z.read("enwik8")
    print("enwik8 bytes", len(raw), flush=True)
    if VAL_OFF + N_VAL > len(raw):
        raise SystemExit(f"split {{VAL_OFF}}+{{N_VAL}} exceeds {{len(raw)}}")
    with open(tr_path, "wb") as f:
        f.write(raw[:N_TRAIN])
    with open(va_path, "wb") as f:
        f.write(raw[VAL_OFF:VAL_OFF + N_VAL])

n_val = os.path.getsize(va_path)
stride = max(1, (n_val - 300) // (int(VALW) * int(ARM_T)))
print(f"train={{os.path.getsize(tr_path)}} val={{n_val}} stride={{stride}} "
      f"mark={{MARK_S}}s gen={{GEN}}", flush=True)

os.makedirs(os.path.join(W, "ckpt"), exist_ok=True)
env = dict(os.environ)
env.update({{
    "PYTHONPATH": ".",
    "ARM_B": ARM_B,
    "ARM_T": ARM_T,
    "ARM_THREADS": "4",
    "ARM_TRAIN": tr_path,
    "ARM_VAL": va_path,
    "ARM_NTRAIN": str(N_TRAIN),
    "ARM_NVAL": str(N_VAL),
    "ARM_VALW": VALW,
    "ARM_VALSTRIDE": str(stride),
    "ARM_MARK_S": MARK_S,
    "ARM_GEN": GEN,
    "ARM_GEN_N": "240",
    "ARM_COMPILE": "0",
    "ARM_BREAK_SCAN": "1",
    "ARM_SEED": SEED,
    "ARM_SAVE": os.path.join(W, "ckpt"),
    # abort the session at the first heartbeat if the host is throttled
    # (healthy tf is 0.0185 s/step, healthy x5 0.027; the bad signature is
    # 0.33-0.84). Costs ~30 s and saves the remaining arms.
    "ARM_SLOW_GATE": "0.10",
}})

names = ARMS.split(",")
print("=== TAG", TAG, "arms", ARMS, "budget", BUDGET, flush=True)

# --- pre-flight: every arm builds, trains one step, evals, samples ----------
dry_env = dict(env)
dry_env["ARM_DRY"] = "1"
dry_env["ARM_SLOW_GATE"] = "0"          # the dry pass is 1 s, not a rate test
d = subprocess.run([sys.executable, "-u", "scripts/ab_small.py", "1", ARMS],
                   env=dry_env, cwd=os.path.join(W, "IECNN"),
                   capture_output=True, text=True)
print(d.stdout[-20000:], flush=True)
print(d.stderr[-8000:], flush=True)
ok_line = f"DRY {{len(names)}}/{{len(names)}} arms ok"
if d.returncode != 0 or ok_line not in d.stdout:
    print("=== PREFLIGHT FAILED -- not starting the timed run", flush=True)
    with open(os.path.join(W, f"results_{{TAG}}.jsonl"), "w") as f:
        f.write(json.dumps({{"tag": TAG, "arms": ARMS, "budget": BUDGET,
                            "exit": "preflight", "stdout": d.stdout[-4000:]}}) + "\\n")
    raise SystemExit(1)

# --- timed run -------------------------------------------------------------
proc = subprocess.run(
    [sys.executable, "-u", "scripts/ab_small.py", BUDGET, ARMS],
    env=env, cwd=os.path.join(W, "IECNN"))

out = os.path.join(W, f"results_{{TAG}}.jsonl")
with open(out, "a") as f:
    f.write(json.dumps({{"tag": TAG, "arms": ARMS, "budget": BUDGET,
                        "exit": proc.returncode}}) + "\\n")
print("WROTE", out, "exit", proc.returncode, flush=True)

# --- population audit on the checkpoints this run just wrote ---------------
# Every dot arm now answers "what does each dot add" at its OWN budget, in the
# same session, with no checkpoint transfer. Cheap: a few hundred forwards.
ck_dir = os.path.join(W, "ckpt")
if os.path.isdir(ck_dir):
    pop_env = dict(env)
    pop_env["POP_OUT"] = os.path.join(W, f"pop_{{TAG}}.json")
    found = sorted(f for f in os.listdir(ck_dir) if f.endswith(".pt"))
    print("=== POP AUDIT over", found, flush=True)
    for name in found:
        path = os.path.join(ck_dir, name)
        try:
            _ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:                                  # noqa: BLE001
            print("skip", name, "unreadable:", e, flush=True)
            continue
        if str(_ck.get("kind")) != "iecnn":
            print("skip", name, "(kind", _ck.get("kind"), ")", flush=True)
            continue
        a = subprocess.run([sys.executable, "-u", "scripts/probe_population.py",
                            path, "--windows", "96"],
                           env=pop_env, cwd=os.path.join(W, "IECNN"))
        print("POP", name, "exit", a.returncode, flush=True)
'''

os.makedirs(OUT, exist_ok=True)
src = TEMPLATE.format(tag=TAG, budget=BUDGET, arms=ARMS, seed=SEED,
                      n_train=N_TRAIN, val_off=VAL_OFF, n_val=N_VAL,
                      mark_s=MARK_S, gen=GEN)
with open(os.path.join(OUT, "iecnn_gpu.py"), "w") as f:
    f.write(src)
meta = {"id": f"voidmaster000/iecnn-{TAG}-gpu", "title": f"iecnn {TAG} gpu",
        "code_file": "iecnn_gpu.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
        "enable_tpu": "false", "enable_internet": "true",
        "machine_shape": "", "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": []}
with open(os.path.join(OUT, "kernel-metadata.json"), "w") as f:
    json.dump(meta, f)
print("wrote", OUT, "id", meta["id"])
