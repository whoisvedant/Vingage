"""Developer tool: run the full app with a SIMULATED audience (no camera, no face model). Output is random test data.
   python tests/sim_server.py [--faces 6]"""
import os, sys, time, types, json, tempfile, argparse
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "app")); sys.path.insert(0, os.path.join(ROOT, "src"))
import engine as EN, ed_features as E, uvicorn
from server import make_app
ap = argparse.ArgumentParser(); ap.add_argument("--faces", type=int, default=6); ap.add_argument("--port", type=int, default=8000); ap.add_argument("--warm", type=int, default=12); a = ap.parse_args()
rng = np.random.default_rng(3)
class P:
    def __init__(s, x, y): s.x, s.y = x, y
class BS:
    def __init__(s, n, v): s.category_name, s.score = n, v
def rot(yaw, pitch):
    y, p = np.radians(yaw), np.radians(pitch); Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]); Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    M = np.eye(4); M[:3, :3] = Ry @ Rx; return M
class FakeLM:
    def __init__(s): s.t0 = time.time(); s.mood = rng.random(a.faces); s.ph = rng.random(a.faces) * 6
    def detect_for_video(s, img, ts): return s.detect(img)
    def detect(s, img):
        el = time.time() - s.t0; n = 1 if el < a.warm else a.faces; fl, M, B = [], [], []
        drift = 0.5 + 0.45 * np.sin(el / 20.0)                     # slow rise and fall of the whole room's attention
        for f in range(n):
            cx = 0.12 + 0.15 * f; base = rng.normal(0, 0.004, (478, 2)) + [cx, 0.5]
            away = (s.mood[f] > drift)                              # people with a high 'mood' value look away when the room drifts
            yaw = (55 if away else 6) * np.sin(el / (3 + f) + s.ph[f]) + (30 if away else 0)
            fl.append([P(*b) for b in base]); M.append(rot(yaw, 4 * np.sin(el / 5 + f)))
            B.append([BS(k, float(np.clip(rng.normal(0.3, 0.25 * (1.2 - drift)), 0, 1))) for k in E.BLEND])
        return types.SimpleNamespace(face_landmarks=fl, facial_transformation_matrixes=M, face_blendshapes=B)
cols = [f"{s}_{k}" for s in E.SIGNALS for k in ("mean", "std", "min", "max", "slope")] + ["perclos", "detect_rate", "n_frames"]
w = rng.normal(0, 0.15, (1, len(cols)))
m = {"task": "binary", "cols": cols, "median": [0.0] * len(cols), "mean": [0.0] * len(cols), "scale": [1.0] * len(cols), "coef": w.tolist(), "intercept": [0.1], "classes": [0, 1]}
d = tempfile.mkdtemp(); json.dump(m, open(os.path.join(d, "m.json"), "w"))
import math
_t0 = time.time()
def _p(feats):                                   # simulated per-person probability: a slow room-wide swing plus individual noise
    return float(np.clip(0.5 + 0.28 * math.sin((time.time() - _t0) / 22.0) + rng.normal(0, 0.22), 0.02, 0.98))
eng = EN.Engine(os.path.join(d, "m.json"), source="simulated", k_min=3, data_dir=d, mode="simulated"); eng._landmarker = lambda: FakeLM(); eng.model.p_engaged = _p
class FakeCap:
    def __init__(s, *_): pass
    def read(s): time.sleep(0.03); return True, np.full((480, 640, 3), 120, np.uint8)
    def get(s, *_): return 30.0
    def isOpened(s): return True
    def release(s): pass
    def set(s, *_): pass
EN.cv2.VideoCapture = FakeCap
print(f"SIMULATION on http://localhost:{a.port}"); uvicorn.run(make_app(eng, 8443), host="127.0.0.1", port=a.port, log_level="warning")
