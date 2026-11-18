# VINGAGE – Real-Time Audience Engagement Detection System

Aggregate-only engagement estimation for a group of people in front of a camera. Per-person signals (head pose, eye openness, gaze, blendshapes, motion) are computed in memory, scored, and discarded. The dashboard shows only group statistics, and only when at least **k** people (default 3) are in view. No frames, faces or identities are stored or sent.

## Run the live demo (laptop, CPU only)
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows;  source .venv/bin/activate on Mac/Linux
pip install -r requirements.txt
# put vingage_model.json (from notebook 4) into app/models/
python app/server.py --source 0 --kmin 3            # webcam (use --kmin 1 to test alone)
```
Open http://localhost:8000. The face model (~4 MB) downloads on first run.

**Pages:** intro (`/`), Room (live), Replays (recorded sessions, CSV export), Evidence (pipeline, results, SHAP, limits), Privacy, Setup (camera source, minimum group size, alert level).

**Cameras:** Setup page. Laptop webcam, a phone browser (scan the QR on the Setup page; the phone sends its camera over your Wi-Fi to a camera-only HTTPS page), an IP-camera app URL, or a video file.

UI test without a camera or model (random data, labelled "Simulation"):
```bash
python tests/sim_server.py --port 8000
```

## Layout
- `src/` feature extraction, pipeline, baselines, GRU (`ed_*.py`)
- `notebooks/` Colab notebooks 1–4 (extraction + baselines, SHAP + GRU, dark/CLAHE test, model export)
- `app/` live engine (`engine.py`), FastAPI + WebSocket server (`server.py`), dashboard (`static/index.html`)
- `tests/` engine test with a fake multi-face landmarker

## Data
DAiSEE (access by request; not included). Do not commit the dataset or the download link.
