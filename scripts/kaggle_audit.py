"""Kaggle runner for scripts/dot_report.py (per-dot audit of one arm).

Same data preparation and protocol as scripts/kaggle_gpu.py so the audited
model is exactly the model the A/B compares: 60 MB train / 5 MB val enwik8,
byte level, same val stride. Trains ONE arm for K_BUDGET seconds, then runs
the dot report.

Env: K_TAG, K_BUDGET, K_B, K_T, K_VALW, ARM_NAME, AUDIT_WINDOWS.
"""
import os
import subprocess
import sys
import urllib.request
import zipfile

TAG = os.environ.get("K_TAG", "audit1")
BUDGET = os.environ.get("K_BUDGET", "180")
ARM = os.environ.get("ARM_NAME", "iecnn-n3-x5-p1-ln-decay")
ARM_B = os.environ.get("K_B", "32")
ARM_T = os.environ.get("K_T", "256")
VALW = os.environ.get("K_VALW", "160")

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
    with open(tr_path, "wb") as f:
        f.write(raw[:N_TRAIN])
    with open(va_path, "wb") as f:
        f.write(raw[N_TRAIN:N_TRAIN + N_VAL])

n_val = os.path.getsize(va_path)
stride = max(1, (n_val - 300) // (int(VALW) * int(ARM_T)))
print(f"val={n_val} stride={stride} arm={ARM}", flush=True)

env = dict(os.environ)
env.update({
    "PYTHONPATH": ".",
    "ARM_B": ARM_B, "ARM_T": ARM_T, "ARM_THREADS": "4",
    "ARM_TRAIN": tr_path, "ARM_VAL": va_path,
    "ARM_NTRAIN": str(N_TRAIN), "ARM_NVAL": str(N_VAL),
    "ARM_VALW": VALW, "ARM_VALSTRIDE": str(stride),
    "ARM_COMPILE": "0", "ARM_BREAK_SCAN": "0",
    "ARM_SEED": os.environ.get("K_SEED", "0"),
    "ARM_NAME": ARM, "K_BUDGET": BUDGET,
    "AUDIT_OUT": os.path.join(W, f"dot_report_{TAG}.json"),
})
print(f"=== AUDIT {TAG} arm {ARM}", flush=True)
proc = subprocess.run([sys.executable, "-u", "scripts/dot_report.py"],
                      env=env, cwd=os.path.join(W, "IECNN"))
print("AUDIT EXIT", proc.returncode, flush=True)
