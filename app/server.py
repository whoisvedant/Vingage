"""VINGAGE server:  python app/server.py --model app/models/vingage_model.json --source 0 --kmin 3"""
import os, sys, json, asyncio, argparse, threading, socket, secrets, io
from contextlib import asynccontextmanager
sys.path.insert(0, os.path.dirname(__file__))
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from engine import Engine
import uvicorn

HERE = os.path.dirname(os.path.abspath(__file__)); STATIC = os.path.join(HERE, "static")

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models", "vingage_model.json"))
    ap.add_argument("--source", default="0", help="webcam index or a video file")
    ap.add_argument("--kmin", type=int, default=3, help="minimum people before any statistic is shown")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--https-port", type=int, default=8443, help="port the phone connects to (camera page only)")
    return ap.parse_args(argv)

def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: s.connect(("10.255.255.255", 1)); return s.getsockname()[0]
    except Exception: return "127.0.0.1"
    finally: s.close()

def make_app(eng, https_port=8443, token=None):
    token = token or secrets.token_urlsafe(6)
    @asynccontextmanager
    async def life(app):
        threading.Thread(target=eng.run, daemon=True).start(); yield
        eng.running = False
    app = FastAPI(title="VINGAGE", lifespan=life)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def phone_port_guard(request, call_next):         # the phone-facing HTTPS port serves the camera page only
        if request.scope.get("server", (None, None))[1] == https_port and not (request.url.path in ("/phone", "/favicon.ico") or request.url.path.startswith("/static/")):
            return Response("Not found", status_code=404)
        return await call_next(request)

    for route, page in (("/", "index"), ("/room", "room"), ("/replays", "replays"), ("/evidence", "evidence"), ("/privacy", "privacy"), ("/setup", "setup"), ("/phone", "phone")):
        app.get(route, include_in_schema=False)(lambda page=page: FileResponse(os.path.join(STATIC, page + ".html")))

    class Cfg(BaseModel):
        k_min: int | None = None
        alert: float | None = None
        focus_tol: float | None = None
    class Mark(BaseModel):
        label: str = ""

    class Start(BaseModel):
        name: str = ""

    @app.get("/api/snapshot")
    def snap(): return eng.snapshot()

    @app.get("/api/info")
    def info():
        return {"mode": eng.mode, "source": str(eng.source), "model": {"type": "logistic regression", "task": "engaged vs not engaged", "features": len(eng.model.cols),
                "trained_on": "DAiSEE Train split"}, "k_min": eng.k_min, "alert": eng.alert, "max_faces": eng.max_faces, "window_s": 10, "fps_target": 5, "focus_tol": eng.tol}

    class Src(BaseModel):
        source: str
    @app.post("/api/source")
    def source(b: Src):
        v = b.source.strip()
        if not v: raise HTTPException(400, "empty source")
        eng.set_source(v); return {"source": v}

    @app.get("/api/connect")
    def connect():
        ip = lan_ip(); return {"ip": ip, "phone_url": f"https://{ip}:{https_port}/phone?t={token}", "source": str(eng.source),
                               "phone_frames": eng.phone_frames, "phone_live": eng.phone_live()}

    @app.get("/api/qr.svg")
    def qr(text: str):
        import segno
        b = io.BytesIO(); segno.make(text, error="m").save(b, kind="svg", scale=6, border=1, dark="#14110d", light="#ffffff"); return Response(b.getvalue(), media_type="image/svg+xml")

    @app.get("/api/stream.mjpg")
    def stream(request: Request):
        """Live view for the operator (local port only): newest camera frame with anonymous boxes. Frames are never stored."""
        if request.scope.get("server", (None, None))[1] == https_port: raise HTTPException(404)
        def gen():
            import time as _t
            eng.viewers += 1; last = -1
            try:
                while eng.running:
                    if eng._frame_id == last: _t.sleep(0.004); continue
                    last = eng._frame_id; jpg = eng.render_view()
                    if jpg: yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
                    _t.sleep(max(0.0, 1 / 60 - 0.002))
            finally:
                eng.viewers -= 1
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store"})

    @app.post("/api/blur/{state}")
    def blur(state: str): eng.blur = state == "on"; return {"blur": eng.blur}

    @app.post("/api/focus/{action}")
    def focus(action: str):
        if action == "calibrate": eng.calibrate()
        elif action == "reset": eng.reset_focus()
        else: raise HTTPException(404)
        return {"ok": True}

    @app.websocket("/ws/phone")
    async def ws_phone(sock: WebSocket, t: str = ""):
        if t != token: await sock.close(code=4403); return
        await sock.accept(); eng.phone_seen = __import__("time").time()
        try:
            while True:
                data = await sock.receive_bytes(); eng.phone_frames += 1; eng.phone_seen = __import__("time").time(); eng.push_frame(data)
        except Exception:
            pass

    @app.post("/api/config")
    def config(c: Cfg):
        if c.k_min is not None:
            if not 1 <= c.k_min <= 20: raise HTTPException(400, "k_min must be 1-20")
            eng.k_min = c.k_min
        if c.alert is not None:
            if not 0.05 <= c.alert <= 0.95: raise HTTPException(400, "alert must be 0.05-0.95")
            eng.alert = c.alert
        if c.focus_tol is not None:
            if not 10 <= c.focus_tol <= 70: raise HTTPException(400, "focus_tol must be 10-70")
            eng.tol = float(c.focus_tol)
        return {"k_min": eng.k_min, "alert": eng.alert, "focus_tol": eng.tol}

    @app.post("/api/session/start")
    def start(s: Start): return {"id": eng.store.start(s.name)}

    @app.post("/api/session/mark")
    def mark(m: Mark):
        if not eng.store.mark(m.label): raise HTTPException(409, "No session is being recorded")
        return {"ok": True}

    @app.post("/api/session/stop")
    def stop(): return {"id": eng.store.stop()}

    @app.get("/api/sessions")
    def sessions(): return eng.store.list()

    @app.get("/api/sessions/{sid}")
    def one(sid: str):
        s = eng.store.get(sid)
        if not s: raise HTTPException(404)
        return s

    @app.get("/api/sessions/{sid}/csv")
    def one_csv(sid: str):
        t = eng.store.to_csv(sid)
        if t is None: raise HTTPException(404)
        return Response(t, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="vingage_{sid}.csv"'})

    @app.delete("/api/sessions/{sid}")
    def delete(sid: str):
        if not eng.store.delete(sid): raise HTTPException(404)
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                await sock.send_json(eng.snapshot()); await asyncio.sleep(1.0)
        except (WebSocketDisconnect, Exception):
            pass
    return app

def serve(eng, port=8000, https_port=8443):
    """HTTP on localhost for the operator; HTTPS on the LAN (camera page only) so a phone can send its camera."""
    app = make_app(eng, https_port)
    try:
        from certs import ensure_cert
        ip = lan_ip(); crt, key = ensure_cert(os.path.join(HERE, "data"), ip)
        cfg = uvicorn.Config(app, host="0.0.0.0", port=https_port, ssl_certfile=crt, ssl_keyfile=key, log_level="warning", timeout_graceful_shutdown=2)
        srv = uvicorn.Server(cfg); srv.install_signal_handlers = lambda: None
        threading.Thread(target=srv.run, daemon=True).start()
        print(f"phone camera: open the QR on the Setup page, or https://{ip}:{https_port}/phone")
    except Exception as e:
        print("phone camera disabled:", e)
    print(f"open http://localhost:{port}")
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", timeout_graceful_shutdown=2)
    except KeyboardInterrupt:
        pass
    finally:
        eng.running = False       # ends the live-view stream loop so Ctrl+C exits at once
        os._exit(0)

if __name__ == "__main__":
    a = parse_args(); serve(Engine(a.model, source=a.source, k_min=a.kmin), a.port, a.https_port)
