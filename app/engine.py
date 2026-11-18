"""Live engagement engine. Per-person data lives only in RAM; only aggregates leave this module."""
import os, sys, time, json, threading, urllib.request
from collections import deque
from types import SimpleNamespace
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import ed_features as E
from sessions import SessionStore

TASK_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
WIN, MIN_WIN, SAMPLE_FPS = 50, 25, 5   # model sees a 10 s window sampled at 5 fps (as in training); first score after 5 s
PROC_FPS = 60                          # upper cap for the analysis loop (real speed depends on the CPU and the number of faces)
TOL_YAW, TOL_PITCH_K = 30.0, 0.75     # attention cone: degrees of head turn / tilt from the focus direction that still count as 'facing the focus'
MAX_W = 640                            # frames are downscaled to this width before detection

class LRModel:
    def __init__(self, path):
        o = json.load(open(path)); self.cols = o["cols"]; self.med = np.array(o["median"], float)
        self.mean = np.array(o["mean"]); self.scale = np.array(o["scale"])
        self.coef = np.array(o["coef"]); self.b = np.array(o["intercept"]); self.classes = o["classes"]
    def p_engaged(self, feats):
        x = np.array([feats.get(c, np.nan) for c in self.cols], float)
        x = np.where(np.isnan(x), self.med, x)
        z = ((x - self.mean) / self.scale) @ self.coef.T + self.b
        return float(1 / (1 + np.exp(-z[0]))) if self.coef.shape[0] == 1 else float(np.exp(z).max() / np.exp(z).sum())

class Track:
    def __init__(self, c, t):
        self.c, self.seen, self.rows, self.prev, self.score = c, t, deque(maxlen=WIN), None, None
        self.att_f, self.att, self.bbox = 0.5, None, None   # smoothed 'facing the focus' value, its on/off state, last box

class Engine:
    def __init__(self, model_path, source=0, k_min=3, max_faces=12, data_dir=None, mode="live"):
        self.model = LRModel(model_path); self.source = source; self.k_min = k_min; self.max_faces = max_faces
        self.mode, self.alert = mode, 0.5; self.events = deque(maxlen=40); self._low_since = None; self._low_flag = False; self._last_status = None
        self.store = SessionStore(os.path.join(data_dir or os.path.join(os.path.dirname(__file__), "data"), "sessions.json"))
        self.tracks, self.lock, self.running = [], threading.Lock(), False
        self.hist = deque(maxlen=180); self.fps = 0.0; self._frame = None; self._frame_id = 0; self._src_gen = 0; self._preview = None; self.preview_on = False; self._last_sample = 0.0; self.cam_err = None; self.det_ms = 0.0; self.phone_frames = 0; self.phone_seen = 0.0; self.cam_fps = 0.0; self.viewers = 0; self._overlay = []; self._ov_size = (640, 480); self.ref = (0.0, 0.0); self.tol = TOL_YAW; self.blur = False; self._calib = None; self.ready = False; self.faces = 0; self.start = time.time(); self.agg = None; self.err = None

    def _landmarker(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        p = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")
        if not os.path.exists(p):
            print("downloading face model (one time, ~4 MB)..."); urllib.request.urlretrieve(TASK_URL, p)
        o = vision.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=p), running_mode=vision.RunningMode.VIDEO,
                                         num_faces=self.max_faces, output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        return vision.FaceLandmarker.create_from_options(o)

    def _match(self, sigs, t, w):
        used, out = set(), []
        for s in sigs:
            x0, y0, x1, y1 = s["_bbox"]; c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            best, bd = None, 0.2 * w
            for tr in self.tracks:
                d = np.linalg.norm(tr.c - c)
                if id(tr) not in used and d < bd: best, bd = tr, d
            if best is None:
                best = Track(c, t); self.tracks.append(best)
            used.add(id(best)); best.c, best.seen = c, t; out.append((best, s))
        return out, used

    def _detect(self, frame, lm, ts_ms):
        import mediapipe as mp
        h, w = frame.shape[:2]
        if w > MAX_W:
            frame = cv2.resize(frame, (MAX_W, int(h * MAX_W / w)), interpolation=cv2.INTER_AREA); h, w = frame.shape[:2]
        frame, _ = E.clahe_if_dark(frame)
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts_ms)
        return frame, res, w, h

    def _step(self, frame, lm, t, ts_ms):
        frame, res, w, h = self._detect(frame, lm, ts_ms)
        sample = (t - self._last_sample) >= 1.0 / SAMPLE_FPS
        found = []
        for i in range(len(res.face_landmarks)):
            pts = np.array([[p.x, p.y] for p in res.face_landmarks[i]]) * [w, h]
            x0, y0 = pts.min(0); x1, y1 = pts.max(0)
            found.append((i, {"_bbox": (int(max(x0, 0)), int(max(y0, 0)), int(min(x1, w - 1)), int(min(y1, h - 1)))}))
        self.faces = len(found)
        matched, used = self._match([s for _, s in found], t, w)
        # --- attention: is this head turned towards the focus direction (presenter / screen)? updated on every frame ---
        poses, ov = [], []
        for (i, _), (tr, s) in zip(found, matched):
            yaw, pitch, _r = E.head_pose_deg(np.asarray(res.facial_transformation_matrixes[i])); poses.append((yaw, pitch))
            dy, dp = (yaw - self.ref[0]) / self.tol, (pitch - self.ref[1]) / (self.tol * TOL_PITCH_K)
            inside = 1.0 if dy * dy + dp * dp < 1.0 else 0.0
            tr.att_f += 0.25 * (inside - tr.att_f)                      # smoothing so a quick glance does not flip the state
            tr.att = (tr.att_f > 0.4) if tr.att else (tr.att_f > 0.6) if tr.att is not None else tr.att_f > 0.5
            tr.bbox = s["_bbox"]; ov.append({"b": s["_bbox"], "att": bool(tr.att)})
        self._overlay, self._ov_size = ov, (w, h)
        if self._calib is not None:
            self._calib["yaw"] += [p[0] for p in poses]; self._calib["pitch"] += [p[1] for p in poses]
            if t >= self._calib["until"]: self._finish_calibration()
        if sample:
            self._last_sample = t
            for (i, _), (tr, s) in zip(found, matched):
                one = SimpleNamespace(face_landmarks=[res.face_landmarks[i]], facial_transformation_matrixes=[res.facial_transformation_matrixes[i]],
                                      face_blendshapes=[res.face_blendshapes[i]] if res.face_blendshapes else [])
                sig = E.frame_signals(one, w, h)
                if sig is None: tr.rows.append([np.nan] * len(E.SIGNALS)); tr.prev = None; continue
                g = E.face_gray(frame, sig["_bbox"]); m = np.nan
                if g is not None and tr.prev is not None:
                    f = cv2.calcOpticalFlowFarneback(tr.prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0); m = float(np.linalg.norm(f, axis=2).mean())
                tr.prev = g; sig["motion"] = m; tr.rows.append([sig[k] for k in E.SIGNALS])
            for tr in self.tracks:
                if id(tr) not in used: tr.rows.append([np.nan] * len(E.SIGNALS)); tr.prev = None
        self.tracks = [tr for tr in self.tracks if t - tr.seen < 3.0]

    # ---- focus calibration ----------------------------------------------------------------------------------------
    def calibrate(self, seconds=3.0):
        """Ask the audience to look at the presenter / screen; the median head direction becomes the 'focus' direction."""
        self._calib = {"until": time.time() + seconds, "yaw": [], "pitch": []}

    def reset_focus(self):
        self.ref = (0.0, 0.0); self._calib = None
        self.events.appendleft({"t": round(time.time() - self.start), "type": "info", "msg": "Focus reset: the camera direction counts as the focus"})

    def _finish_calibration(self):
        c, self._calib = self._calib, None; now = round(time.time() - self.start)
        if len(c["yaw"]) < 15:
            self.events.appendleft({"t": now, "type": "warn", "msg": "Calibration failed: no faces seen. Face the camera and try again"}); return
        self.ref = (float(np.median(c["yaw"])), float(np.median(c["pitch"])))
        self.events.appendleft({"t": now, "type": "good", "msg": f"Focus set from {len(c['yaw'])} face readings"})

    # ---- live view (operator only) -----------------------------------------------------------------------------------
    def render_view(self):
        """Latest raw frame with anonymous boxes: green = facing the focus, red = looking elsewhere. Never stored."""
        f = self._frame
        if f is None: return None
        vis = f
        if vis.shape[1] > 960: vis = cv2.resize(vis, (960, int(vis.shape[0] * 960 / vis.shape[1])), interpolation=cv2.INTER_AREA)
        else: vis = vis.copy()
        H, W = vis.shape[:2]; sx, sy = W / self._ov_size[0], H / self._ov_size[1]
        for o in self._overlay:
            x0, y0, x1, y1 = o["b"]; x0, x1, y0, y1 = int(x0 * sx), int(x1 * sx), int(y0 * sy), int(y1 * sy)
            if self.blur and x1 > x0 + 4 and y1 > y0 + 4:
                roi = vis[y0:y1, x0:x1]; sm = cv2.resize(roi, (max(2, (x1 - x0) // 14), max(2, (y1 - y0) // 14)), interpolation=cv2.INTER_AREA)
                vis[y0:y1, x0:x1] = cv2.resize(sm, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (110, 190, 60) if o["att"] else (60, 80, 230), 2)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 72]); return buf.tobytes() if ok else None

    def _score(self):
        now_t = time.time(); att = [tr.att_f > 0.5 for tr in self.tracks if now_t - tr.seen < 1.0]
        scores = []
        for tr in self.tracks:
            if len(tr.rows) < MIN_WIN: continue
            A = np.array(tr.rows, float); det = ~np.isnan(A[:, 0])
            if det.mean() < 0.5: continue
            for j in range(A.shape[1]): A[:, j] = E.median3(A[:, j])
            f = E.window_features(A, det); f["n_frames"] = WIN
            tr.score = self.model.p_engaged(f); scores.append(tr.score)
        n = len(att); nm = len(scores); now = time.time() - self.start
        with self.lock:
            if n >= self.k_min:
                a_share = float(np.mean(att)); p = np.array(scores) if nm >= self.k_min else None
                self.agg = {"status": "ok", "n": n, "att_share": a_share, "engaged_share": float((p >= 0.5).mean()) if p is not None else None, "mean_score": float(p.mean()) if p is not None else None}
                self.hist.append({"t": round(now, 1), "att": a_share, "share": self.agg["engaged_share"], "score": self.agg["mean_score"], "n": n})
                self.store.record(a_share, self.agg["engaged_share"], self.agg["mean_score"], n)
                self._events(now, a_share)
            else:
                self.agg = {"status": "insufficient", "n": None, "att_share": None, "engaged_share": None, "mean_score": None}   # k-anonymity: nothing reported below k
                self._low_since = None
            if self.agg["status"] != self._last_status:
                msg = {"ok": "Audience large enough: statistics are live", "insufficient": f"Fewer than {self.k_min} people in view: statistics hidden"}[self.agg["status"]]
                self.events.appendleft({"t": round(now), "type": "info", "msg": msg}); self._last_status = self.agg["status"]

    def _events(self, now, share):
        if share < self.alert:
            self._low_since = self._low_since or now
            if not self._low_flag and now - self._low_since >= 10:
                self._low_flag = True; self.events.appendleft({"t": round(now), "type": "warn", "msg": f"Attention below {round(self.alert*100)}% for 10 s"})
        else:
            self._low_since = None
            if self._low_flag:
                self._low_flag = False; self.events.appendleft({"t": round(now), "type": "good", "msg": "Attention recovered above the alert level"})

    # ---- sources -------------------------------------------------------------------------------------------------
    def set_source(self, source):
        """Switch camera at runtime: int index, URL / file path, or 'phone' (frames pushed by the phone page)."""
        with self.lock:
            self.source = source; self._src_gen += 1; self.cam_err = None; self.tracks = []; self.hist.clear(); self.agg = None; self._last_status = None
            self._frame = None

    def phone_live(self):
        return time.time() - self.phone_seen < 3.0

    def push_frame(self, jpeg_bytes):
        """Frame from the phone page (JPEG). Only the latest frame is kept."""
        if str(self.source) != "phone": return
        img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is not None: self._frame, self._frame_id = img, self._frame_id + 1

    def _open(self, src):
        s = str(src)
        if s == "phone": return None, False
        if s.isdigit():
            idx = int(s)
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
            for be in backends:                                  # try each backend, first with fast settings, then plain
                for tuned in (True, False):
                    cap = cv2.VideoCapture(idx, be)
                    if not cap.isOpened(): cap.release(); continue
                    if tuned:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1); cap.set(cv2.CAP_PROP_FPS, 60)
                    ok = False
                    for _ in range(12):                          # cameras often need a moment before the first frame
                        ok, _f = cap.read()
                        if ok: break
                        time.sleep(0.1)
                    if ok: return cap, False
                    cap.release()
            return cv2.VideoCapture(), False                     # closed capture: reported as an error below
        return cv2.VideoCapture(s), not s.lower().startswith(("http", "rtsp"))

    def _capture(self):
        """Reads the camera as fast as it delivers and keeps only the newest frame, so a slow detector never builds a backlog."""
        gen = -1; cap = None; is_file = False; src_fps = 30.0; retry_at = 0.0; cn = 0; c0 = time.time()
        while self.running:
            if gen != self._src_gen or (cap is not None and not cap.isOpened() and time.time() >= retry_at):
                if cap is not None: cap.release()
                gen = self._src_gen; cap, is_file = self._open(self.source)
                if cap is not None:
                    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    if cap.isOpened(): self.cam_err = None
                    else:
                        retry_at = time.time() + 3.0
                        self.cam_err = (f"Could not open camera '{self.source}'. Another program may be using it (close other VINGAGE servers with Ctrl+C, "
                                        "Zoom, Teams, the Camera app), or choose another camera on the Setup page. Retrying every 3 seconds.")
            if cap is None: time.sleep(0.05); continue
            if not cap.isOpened(): time.sleep(0.2); continue
            ok, frame = cap.read()
            if not ok:
                if is_file: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                time.sleep(0.05); continue
            if is_file: time.sleep(1.0 / src_fps)
            self._frame, self._frame_id = frame, self._frame_id + 1; cn += 1
            if time.time() - c0 >= 1.0: self.cam_fps = cn / (time.time() - c0); cn = 0; c0 = time.time()
        if cap is not None: cap.release()

    def run(self):
        self.running = True
        try:
            threading.Thread(target=self._capture, daemon=True).start()   # camera first, so video appears while the face model loads
            lm = self._landmarker(); self.ready = True
            seen = -1; last_score = 0.0; n = 0; t0 = time.time(); t_prev = 0.0; ts = 0; gen = self._src_gen
            while self.running:
                if gen != self._src_gen: gen = self._src_gen; seen = -1
                f = self._frame
                if f is None or self._frame_id == seen: time.sleep(0.004); continue
                seen = self._frame_id; now = time.time()
                if now - t_prev < 1.0 / PROC_FPS: continue
                t_prev = now; ts = max(ts + 1, int(now * 1000))
                t_a = time.time(); self._step(f, lm, now, ts); self.det_ms = 0.9 * self.det_ms + 0.1 * (time.time() - t_a) * 1000; n += 1
                if now - t0 >= 2: self.fps = n / (now - t0); n = 0; t0 = now
                if now - last_score >= 1.0: last_score = now; self._score()
        except Exception as e:
            self.err = repr(e); raise

    def snapshot(self):
        with self.lock:
            a = dict(self.agg or {"status": "warming_up", "n": None, "att_share": None, "engaged_share": None, "mean_score": None})
            return {**a, "k_min": self.k_min, "faces_in_view": self.faces, "fps": round(self.fps, 1), "uptime": round(time.time() - self.start),
                    "history": list(self.hist), "error": self.err or self.cam_err, "mode": self.mode, "source": str(self.source), "det_ms": round(self.det_ms), "alert": self.alert,
                    "events": list(self.events), "session": self.store.state(), "cam_fps": round(self.cam_fps, 1), "ready": self.ready,
                    "focus": {"yaw": round(self.ref[0], 1), "pitch": round(self.ref[1], 1), "tol": self.tol, "calibrating": self._calib is not None}, "blur": self.blur,
                    "privacy": "group statistics only: nothing is stored, and the live view never leaves this computer"}
