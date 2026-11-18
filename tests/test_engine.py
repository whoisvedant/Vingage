"""Engine test with a fake multi-face landmarker and a generated video. Run:  python tests/test_engine.py"""
import sys, os, time, types, json, tempfile, threading
import numpy as np, cv2
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "app")); sys.path.insert(0, os.path.join(ROOT, "src"))
import engine as EN, ed_features as E
rng = np.random.default_rng(1)
class P:
    def __init__(s, x, y): s.x, s.y = x, y
class BS:
    def __init__(s, n): s.category_name, s.score = n, float(rng.random())
class FakeLM:
    def __init__(s, n): s.n = n
    def detect_for_video(s, img, ts):
        fl, M, B = [], [], []
        for f in range(s.n):
            base = rng.normal(0, 0.03, (478, 2)) + [0.2 + 0.3 * f, 0.5]
            fl.append([P(*b) for b in base]); M.append(np.eye(4)); B.append([BS(n) for n in EN.E.BLEND])
        return types.SimpleNamespace(face_landmarks=fl, facial_transformation_matrixes=M, face_blendshapes=B)
d = tempfile.mkdtemp(); vid = os.path.join(d, "t.avi")
w = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"MJPG"), 15, (320, 240))
for i in range(60): w.write(np.full((240, 320, 3), 100, np.uint8))
w.release()
cols = [f"{s}_{k}" for s in E.SIGNALS for k in ("mean", "std", "min", "max", "slope")] + ["perclos", "detect_rate", "n_frames"]
json.dump({"task": "binary", "cols": cols, "median": [0.0] * len(cols), "mean": [0.0] * len(cols), "scale": [1.0] * len(cols),
           "coef": rng.normal(0, 0.1, (1, len(cols))).tolist(), "intercept": [0.0], "classes": [0, 1]}, open(os.path.join(d, "m.json"), "w"))
for n_faces, expect in ((3, "ok"), (2, "insufficient")):
    e = EN.Engine(os.path.join(d, "m.json"), source=vid, k_min=3, data_dir=d); e._landmarker = lambda n=n_faces: FakeLM(n)
    threading.Thread(target=e.run, daemon=True).start(); time.sleep(16); e.running = False
    s = e.snapshot(); print(n_faces, "faces ->", s["status"], "fps", s["fps"], "err", s["error"])
    assert s["status"] == expect and s["error"] is None
    if expect == "insufficient": assert s["engaged_share"] is None and s["history"] == []
print("engine ok")
