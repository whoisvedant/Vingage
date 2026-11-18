"""Session log. Stores aggregate statistics only (time, engaged share, mean score, count); never anything per person."""
import os, json, time, threading, uuid, io, csv

class SessionStore:
    def __init__(self, path):
        self.path, self.lock, self.active = path, threading.Lock(), None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.items = json.load(open(path)) if os.path.exists(path) else []

    def _save(self):
        tmp = self.path + ".tmp"; json.dump(self.items, open(tmp, "w")); os.replace(tmp, self.path)

    def start(self, name):
        with self.lock:
            if self.active: return self.active["id"]
            self.active = {"id": uuid.uuid4().hex[:8], "name": (name or "Untitled session")[:60], "start": time.time(), "end": None, "points": [], "markers": []}
            return self.active["id"]

    def record(self, att, share, score, n):
        with self.lock:
            if self.active: self.active["points"].append({"t": round(time.time() - self.active["start"], 1), "att": round(att, 4), "share": None if share is None else round(share, 4), "score": None if score is None else round(score, 4), "n": n})

    def mark(self, label):
        with self.lock:
            if not self.active: return False
            self.active.setdefault("markers", []).append({"t": round(time.time() - self.active["start"], 1), "label": (label or "Marker")[:80]}); return True

    def stop(self):
        with self.lock:
            if not self.active: return None
            s = self.active; s["end"] = time.time(); self.active = None
            self.items.append(s); self._save(); return s["id"]

    @staticmethod
    def summary(s):
        p = s["points"]; dur = (s["end"] or time.time()) - s["start"]
        if not p: return {"id": s["id"], "name": s["name"], "start": s["start"], "duration": round(dur), "samples": 0, "mean_share": None, "peak": None, "low": None, "mean_n": None}
        sh = [x.get("att", x.get("share")) for x in p]; sh = [v for v in sh if v is not None]          # headline metric: attention (older files: engaged share)
        return {"id": s["id"], "name": s["name"], "start": s["start"], "duration": round(dur), "samples": len(p),
                "mean_share": sum(sh) / len(sh), "peak": max(sh), "low": min(sh), "mean_n": sum(x["n"] for x in p) / len(p)}

    def list(self):
        with self.lock: return [self.summary(s) for s in reversed(self.items)]

    def get(self, sid):
        with self.lock:
            for s in self.items:
                if s["id"] == sid: return {**self.summary(s), "points": s["points"], "markers": s.get("markers", [])}
        return None

    def delete(self, sid):
        with self.lock:
            n = len(self.items); self.items = [s for s in self.items if s["id"] != sid]
            if len(self.items) != n: self._save(); return True
        return False

    def clear(self):
        with self.lock: self.items = []; self._save()

    def to_csv(self, sid):
        s = self.get(sid)
        if not s: return None
        b = io.StringIO(); w = csv.writer(b); w.writerow(["seconds", "attention_share", "engaged_share_model", "mean_score_model", "people_counted"])
        for p in s["points"]: w.writerow([p["t"], p.get("att", ""), p.get("share", ""), p.get("score", ""), p["n"]])
        return b.getvalue()

    def state(self):
        with self.lock:
            a = self.active
            return {"active": bool(a), "id": a["id"] if a else None, "name": a["name"] if a else None, "elapsed": round(time.time() - a["start"]) if a else 0}
