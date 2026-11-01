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
