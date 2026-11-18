"""Index, audit and resumable extraction for DAiSEE."""
import os, re, glob, time
from pathlib import Path
import numpy as np, pandas as pd

VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv"}
_FACTORY = None; _LM = None; _SEQDIR = None; _OPTS = {}

def build_index(root):
    """Return DataFrame: clip, path, split, subject, Boredom/Engagement/Confusion/Frustration."""
    vids = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if Path(f).suffix.lower() in VIDEO_EXT:
                vids[Path(f).stem] = os.path.join(dp, f)
    labs = []
    for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        n = os.path.basename(p).lower()
        if "label" not in n or "all" in n:
            continue
        split = "Train" if "train" in n else "Validation" if ("valid" in n or "val" in n) else "Test" if "test" in n else None
        if split is None:
            continue
        d = pd.read_csv(p); d.columns = [c.strip() for c in d.columns]; d["split"] = split; labs.append(d)
    if not labs:
        raise RuntimeError("No Train/Validation/Test label CSVs found under " + root)
    L = pd.concat(labs, ignore_index=True)
    L["clip"] = L["ClipID"].astype(str).map(lambda x: Path(x.strip()).stem)
    L["path"] = L["clip"].map(vids)
    L["subject"] = L["clip"].str[:6]
    return L.drop(columns=["ClipID"]), len(vids)

def audit(L, n_videos):
    print(f"video files found: {n_videos}; labelled rows: {len(L)}; rows with a video: {L['path'].notna().sum()}")
    print("\nclips per split:\n", L["split"].value_counts())
    print("\nEngagement level share per split (%):")
    print((pd.crosstab(L["split"], L["Engagement"], normalize="index") * 100).round(2))
    subs = {s: set(L.loc[L.split == s, "subject"]) for s in L.split.unique()}
    names = list(subs)
    print("\nsubjects per split:", {k: len(v) for k, v in subs.items()})
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"subjects shared by {names[i]} and {names[j]}:", len(subs[names[i]] & subs[names[j]]))
    print("\n(subject = first 6 characters of the clip ID: an assumption, check that the counts above look like ~112 users)")

def _init(factory, seqdir, opts):
    global _LM, _SEQDIR, _OPTS
    _LM = factory(); _SEQDIR = seqdir; _OPTS = opts

def _work(args):
    import ed_features as E
    clip, path = args
    t = time.time()
    try:
        A, det, nc = E.extract_clip(path, _LM, darken=_OPTS.get("darken"), use_clahe=_OPTS.get("use_clahe", True))
        seq = np.concatenate([A, det[:, None].astype(float)], axis=1).astype(np.float32)   # signals + raw detection flag
        np.save(os.path.join(_SEQDIR, clip + ".npy"), seq)
        f = E.window_features(A, det); f["clahe_frames"] = nc; f["error"] = ""
    except Exception as e:
        f = {"error": repr(e)[:200]}
    f["clip"] = clip; f["secs"] = time.time() - t
    return f

def run_extraction(L, out_dir, factory, workers=2, chunk=100, limit=None, darken=None, use_clahe=True):
    """Resumable: each chunk of clips is saved to out_dir/chunk_XXXX.csv and skipped if already present."""
    import multiprocessing as mp
    os.makedirs(out_dir, exist_ok=True)
    seqdir = os.path.join(out_dir, "seq"); os.makedirs(seqdir, exist_ok=True)
    todo = L.dropna(subset=["path"]).reset_index(drop=True)
    if limit: todo = todo.iloc[:limit]
    n_chunks = (len(todo) + chunk - 1) // chunk
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(factory, seqdir, {"darken": darken, "use_clahe": use_clahe})) as pool:
        for c in range(n_chunks):
            fn = os.path.join(out_dir, f"chunk_{c:04d}.csv")
            if os.path.exists(fn):
                continue
            part = todo.iloc[c * chunk:(c + 1) * chunk]
            t = time.time()
            rows = pool.map(_work, list(zip(part["clip"], part["path"])), chunksize=2)
            pd.DataFrame(rows).to_csv(fn + ".tmp", index=False); os.replace(fn + ".tmp", fn)
            print(f"chunk {c + 1}/{n_chunks} done in {time.time() - t:.0f}s", flush=True)

def collect(L, out_dir):
    F = pd.concat([pd.read_csv(p, dtype={"clip": str}) for p in sorted(glob.glob(os.path.join(out_dir, "chunk_*.csv")))], ignore_index=True)
    F["error"] = F["error"].fillna("")
    return L.drop(columns=["path"]).merge(F, on="clip", how="inner")
