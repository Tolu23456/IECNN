"""Kaggle T4 runner for scripts/ab_small.py.

This file is the kernel entry point: `kaggle kernels push -p <dir>` uploads it
with a kernel-metadata.json that points `code_file` here. It clones the repo,
prepares byte-level enwik8 (60 MB train / 5 MB val), and runs the arm list at
B=32 T=256 for K_BUDGET seconds per arm. All arms share the session, so the
comparison is within-session and the transformer anchor is always included.

Val slice: the 5 MB immediately AFTER the 60 MB train window
(md5 6d6c2819c2). Batches up to `combo1` used a different 5 MB window
(afc1109404) whose provenance is unknown; it is ~0.04 bpc HARDER than this one
(tf-swa 2.0219 there vs 2.0612 here), which is why only in-session comparisons
are used and every batch re-runs the transformer anchor.

Why 60/5 MB: the full 100 MB corpus makes one epoch unreachable at this budget
and the val split would be scored near the end of the file; 60 MB lets the
180 s arms see ~1 epoch and keeps `bpc_final` comparable across batches.

Env (all optional): K_TAG, K_BUDGET, K_B, K_T, K_VALW, K_ARMS.
"""
import json
import os
import subprocess
import sys
import urllib.request
import zipfile

TAG = os.environ.get("K_TAG", "comb2")
BUDGET = os.environ.get("K_BUDGET", "180")
ARM_B = os.environ.get("K_B", "32")
ARM_T = os.environ.get("K_T", "256")
VALW = os.environ.get("K_VALW", "160")
ARMS = os.environ.get(
    "K_ARMS", "tf-swa,iecnn-n3-x4-p1-ln-decay,iecnn-n3-x5-p1-ln-decay,"
              "iecnn-n3-x4-p1-ln-decay-w256,"
              "iecnn-n3-x4-p1-ln-decay-d128,"
              "iecnn-n3-x4-p1-ln-decay-h2")

W = "/kaggle/working"
REPO = "https://github.com/Tolu23456/IECNN.git"
N_TRAIN = 60_000_000
N_VAL = 5_000_000


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


os.makedirs(W, exist_ok=True)
os.chdir(W)

if not os.path.isdir("IECNN"):
    sh(["git", "clone", "--depth", "1", REPO, "IECNN"], check=True)
os.chdir("IECNN")

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
    with open(tr_path, "wb") as f:
        f.write(raw[:N_TRAIN])
    with open(va_path, "wb") as f:
        f.write(raw[N_TRAIN:N_TRAIN + N_VAL])

n_val = os.path.getsize(va_path)
stride = max(1, (n_val - 300) // (int(VALW) * int(ARM_T)))
print(f"val={n_val} stride={stride}", flush=True)

env = dict(os.environ)
env.update({
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
    "ARM_COMPILE": "0",          # Inductor dies on the CUDA prefix scan
    "ARM_BREAK_SCAN": "0",
    "ARM_SEED": os.environ.get("K_SEED", "0"),   # 2-seed confirmation runs
})

print(f"=== TAG {TAG} arms {ARMS}", flush=True)
proc = subprocess.run(
    [sys.executable, "-u", "scripts/ab_small.py", BUDGET, ARMS],
    env=env, cwd=os.path.join(W, "IECNN"))

out = os.path.join(W, f"results_{TAG}.jsonl")
with open(out, "w") as f:
    f.write(json.dumps({"tag": TAG, "arms": ARMS, "budget": BUDGET,
                        "exit": proc.returncode}) + "\n")
print("WROTE", out, "exit", proc.returncode, flush=True)
