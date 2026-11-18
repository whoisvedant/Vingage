"""Per-clip feature extraction for DAiSEE. No pixels are written to disk."""
import numpy as np, cv2

R_EYE = [33, 160, 158, 133, 153, 144]     # p1..p6, right eye (image left)
L_EYE = [362, 385, 387, 263, 373, 380]    # p1..p6, left eye
R_IRIS, L_IRIS = 468, 473
BLEND = ["eyeBlinkLeft", "eyeBlinkRight", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookInRight",
         "eyeLookOutRight", "eyeLookUpLeft", "eyeLookDownLeft", "browDownLeft", "browDownRight",
         "browInnerUp", "jawOpen", "mouthSmileLeft", "mouthSmileRight"]
SIGNALS = ["yaw", "pitch", "roll", "ear", "gaze_x", "gaze_y", "motion"] + BLEND

def eye_aspect_ratio(eye):
    v1 = np.linalg.norm(eye[1] - eye[5]); v2 = np.linalg.norm(eye[2] - eye[4])
    h = np.linalg.norm(eye[0] - eye[3])
    return (v1 + v2) / (2.0 * h + 1e-8)

def head_pose_deg(M):
    R = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0)
    sy = np.hypot(R[0, 0], R[1, 0])
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll

def clahe_if_dark(bgr, thresh=90, clip=2.0):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    if lab[..., 0].mean() >= thresh:
        return bgr, False
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), True

def frame_signals(res, w, h):
    """res: FaceLandmarkerResult. Returns dict of signals or None if no face."""
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    pts = np.array([[p.x * w, p.y * h] for p in lm])
    out = {}
    out["yaw"], out["pitch"], out["roll"] = head_pose_deg(np.asarray(res.facial_transformation_matrixes[0]))
    out["ear"] = 0.5 * (eye_aspect_ratio(pts[R_EYE]) + eye_aspect_ratio(pts[L_EYE]))
    gx, gy = [], []
    for iris, (a, b, up, lo) in ((R_IRIS, (33, 133, 159, 145)), (L_IRIS, (362, 263, 386, 374))):
        c = 0.5 * (pts[a] + pts[b]); wid = np.linalg.norm(pts[a] - pts[b]) + 1e-8
        gx.append((pts[iris][0] - c[0]) / wid)
        gy.append((pts[iris][1] - 0.5 * (pts[up][1] + pts[lo][1])) / wid)
    out["gaze_x"], out["gaze_y"] = float(np.mean(gx)), float(np.mean(gy))
    bs = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
    for k in BLEND:
        out[k] = float(bs.get(k, np.nan))
    x0, y0 = np.clip(pts.min(0), 0, [w - 1, h - 1]); x1, y1 = np.clip(pts.max(0), 0, [w - 1, h - 1])
    out["_bbox"] = (int(x0), int(y0), int(x1), int(y1))
    return out

def face_gray(bgr, bbox, size=64):
    x0, y0, x1, y1 = bbox
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return cv2.resize(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (size, size))

def median3(a):
    a = np.asarray(a, float)
    if len(a) < 3:
        return a
    out = a.copy()
    for i in range(1, len(a) - 1):
        win = a[i - 1:i + 2]; win = win[~np.isnan(win)]
        out[i] = np.median(win) if len(win) else np.nan
    return out

def extract_clip(path, landmarker, target_fps=5, darken=None, use_clahe=True):
    """Return (T, len(SIGNALS)) array with NaN rows where no face; plus n_frames and n_clahe."""
    import mediapipe as mp
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / target_fps)))
    rows, prev, i, n_clahe = [], None, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            if darken:                                    # simulate a dark room: scale pixel values
                frame = np.clip(frame.astype(np.float32) * darken, 0, 255).astype(np.uint8)
            if use_clahe:
                frame, used = clahe_if_dark(frame); n_clahe += used
            h, w = frame.shape[:2]
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            sig = frame_signals(res, w, h)
            if sig is None:
                rows.append([np.nan] * len(SIGNALS)); prev = None
            else:
                g = face_gray(frame, sig["_bbox"])
                motion = np.nan
                if g is not None and prev is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    motion = float(np.linalg.norm(flow, axis=2).mean())
                prev = g
                sig["motion"] = motion
                rows.append([sig[k] for k in SIGNALS])
        i += 1
    cap.release()
    A = np.array(rows, float).reshape(-1, len(SIGNALS))
    det = ~np.isnan(A[:, 0]) if len(A) else np.zeros(0, bool)   # raw detections, before filtering
    for j in range(A.shape[1]):
        A[:, j] = median3(A[:, j])
    return A, det, n_clahe

def window_features(A, det, ear_closed=0.2):
    """One row of features for one 10 s clip (the whole clip is one window). det = raw detection mask."""
    T = len(A)
    feats = {"n_frames": T, "detect_rate": float(det.mean()) if T else 0.0}
    t = np.arange(T)
    for j, name in enumerate(SIGNALS):
        x = A[:, j] if T else np.array([])
        m = ~np.isnan(x)
        if m.sum() >= 3:
            feats[f"{name}_mean"] = x[m].mean(); feats[f"{name}_std"] = x[m].std()
            feats[f"{name}_min"] = x[m].min(); feats[f"{name}_max"] = x[m].max()
            feats[f"{name}_slope"] = np.polyfit(t[m], x[m], 1)[0]
        else:
            for s in ("mean", "std", "min", "max", "slope"):
                feats[f"{name}_{s}"] = np.nan
    ear = A[:, SIGNALS.index("ear")] if T else np.array([])
    feats["perclos"] = float((ear[~np.isnan(ear)] < ear_closed).mean()) if (~np.isnan(ear)).any() else np.nan
    return feats

def aggregate(probs, k_min=3):
    if len(probs) < k_min:
        return None
    p = np.asarray(probs)
    return {"mean": float(p.mean()), "share_engaged": float((p >= 0.5).mean()), "n": len(p)}
