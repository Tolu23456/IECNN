# IECNN Project

## Overview
A conceptual research documentation project for IECNN (Iterative Emergent Convergent Neural Network) — a novel neural network architecture aimed at serving as a foundation for AGI.

## Architecture
- **Frontend**: Static HTML (`index.html`) served by a Python HTTP server
- **Server**: `server.py` — Python `http.server` on port 5000, host `0.0.0.0`
- **Deployment**: Configured as a static site

## Key Files
- `index.html` — Web-based documentation viewer
- `server.py` — Simple HTTP server (Python standard library)
- `iecnn_notes.txt` — Core concept notes (source of truth for the documentation)
- `README.md` — Project overview

## Running
```bash
python server.py
```
Serves on `http://0.0.0.0:5000`.

## Workflow
- **Start application**: `python server.py` on port 5000 (webview)
