# Runbook

Exact commands to set up, run tests, run an experiment, resume a run, and view
results. The agent keeps this file up to date.

All commands are run from the project folder `/home/t85491005/IECNN`.

Setup (one time, already done by the agent):

```bash
cd /home/t85491005/IECNN
uv venv --python /usr/bin/python3.11 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
```

Run the test suite:

```bash
cd /home/t85491005/IECNN
.venv/bin/python -m pytest tests/ -v
```

Run training on GitHub's computers (GitHub Actions), instead of this slow local box:

1. Open https://github.com/Tolu23456/IECNN/-/ — actually the Actions tab: https://github.com/Tolu23456/IECNN/actions
2. Click "train" in the left sidebar, then "Run workflow".
3. Fill in: config file name (e.g. `e1_weights_passes.yaml`), optional overlay (e.g. `tiny.yaml`), seeds (`0,1,2`).
4. Click "Run workflow". Each seed runs as one job (4 vCPU, 16 GB RAM). Results come back as download artifacts plus committed CSV files.

Every push to GitHub also runs the full test suite automatically (workflow "tests").
