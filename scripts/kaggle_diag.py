"""Kaggle runner: compile/step-time diagnostics for two arms, no training."""
import json
import os
import subprocess
import sys
import urllib.request
import zipfile

TAG = os.environ.get("K_TAG", "diag1")
ARMS = os.environ.get("K_ARMS",
                      "iecnn-n3-x3-p1-ln-decay-d128-w512,"
                      "iecnn-n3-x3-p1-ln-decay-d128-w512-gx,tf-swa")
W = "/kaggle/working"
REPO = "https://github.com/Tolu23456/IECNN.git"
N_TRAIN, N_VAL = 60_000_000, 5_000_000

os.makedirs(W, exist_ok=True)
os.chdir(W)
if not os.path.isdir("IECNN"):
    subprocess.run(["git", "clone", "--depth", "1", REPO, "IECNN"], check=True)
os.chdir("IECNN")
tr, va = os.path.join(W, "enwik_train.txt"), os.path.join(W, "enwik_val.txt")
if not (os.path.exists(tr) and os.path.exists(va)):
    z = os.path.join(W, "enwik8.zip")
    if not os.path.exists(z):
        urllib.request.urlretrieve("http://mattmahoney.net/dc/enwik8.zip", z)
    raw = zipfile.ZipFile(z).read("enwik8")
    open(tr, "wb").write(raw[:N_TRAIN])
    open(va, "wb").write(raw[N_TRAIN:N_TRAIN + N_VAL])

for arm in [a for a in ARMS.split(",") if a]:
    env = dict(os.environ)
    env.update({"PYTHONPATH": ".", "ARM_B": "32", "ARM_T": "256",
                "ARM_THREADS": "4", "ARM_TRAIN": tr, "ARM_VAL": va,
                "ARM_NTRAIN": str(N_TRAIN), "ARM_NVAL": str(N_VAL),
                "ARM_VALW": "16", "ARM_VALSTRIDE": "122",
                "ARM_COMPILE": "0", "ARM_BREAK_SCAN": "0",
                "ARM_SEED": "0", "ARM_NAME": arm, "K_BUDGET": "5",
                "DIAG_STEPS": os.environ.get("DIAG_STEPS", "20")})
    print(f"=== DIAG {TAG} arm {arm}", flush=True)
    p = subprocess.run([sys.executable, "-u", "scripts/compile_diag.py"],
                       env=env, cwd=os.path.join(W, "IECNN"), timeout=1800)
    print("DIAG EXIT", arm, p.returncode, flush=True)
print("DIAG DONE", json.dumps({"tag": TAG, "arms": ARMS}), flush=True)
