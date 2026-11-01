# Decisions log

Every choice the spec did not fix. One entry per decision.

| Date | Decision | Why | Alternatives considered |
|---|---|---|---|
| 2026-11-01 | The spec was supplied as nine files `papers/00..08`, not as `spec/IECNN_FULL_SPEC.md`. The agent concatenated them into `spec/IECNN_FULL_SPEC.md` (with a comment marking each source file) and uses that as the source of truth. Neither the content nor the order was changed. | The required path did not exist, but the complete spec content was present. | Stop and ask Tolu to move the files; rejected because the content was unambiguous and available. |
| 2026-11-01 | Use Python 3.11 (system `/usr/bin/python3.11`) for the virtual environment, not the default `python3` (3.15). | PyTorch wheel availability for 3.11 is guaranteed; wheels for a fresh 3.15 may be missing. Python 3.11 is well within the "3.10 or newer" requirement. | Use 3.15 and hope for wheels; rejected as an avoidable risk. |
| 2026-11-01 | Install PyTorch from the CPU wheel index (`--index-url https://download.pytorch.org/whl/cpu`). | The machine has no GPU; the full wheel pulls several GB of unused CUDA libraries, and only ~4.5 GB of disk is free. | Standard index; rejected for disk cost. |
