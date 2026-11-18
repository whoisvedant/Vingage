"""Baselines on the clip feature table. Reports accuracy, macro-F1 and per-class recall."""
import numpy as np, pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

NON_FEATURE = {"clip", "split", "subject", "Boredom", "Engagement", "Confusion", "Frustration", "error", "secs", "clahe_frames"}

def feature_cols(F):
    return [c for c in F.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(F[c])]

def make_xy(F, task):
    y = F["Engagement"].astype(int)
    if task == "binary":
        y = (y >= 2).astype(int)
    return y

def run_baselines(F, task="four", min_detect=0.0, seed=0):
    from xgboost import XGBClassifier
    F = F[(F["error"] == "") & (F["detect_rate"] >= min_detect)]
    cols = feature_cols(F)
    tr, va, te = (F[F.split == s] for s in ("Train", "Validation", "Test"))
    med = tr[cols].median()
    X = {k: d[cols].fillna(med).fillna(0).values for k, d in (("val", va), ("test", te), ("train", tr))}
    ytr = make_xy(tr, task).values; classes = sorted(set(ytr)); remap = {c: i for i, c in enumerate(classes)}
    labels = [0, 1] if task == "binary" else [0, 1, 2, 3]
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000)),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                 random_state=seed, n_jobs=2, eval_metric="mlogloss"),
    }
    out, fitted = [], {}
    for name, m in models.items():
        yt = np.array([remap[v] for v in ytr])
        if name == "xgboost":
            m.fit(X["train"], yt, sample_weight=compute_sample_weight("balanced", yt))
        else:
            m.fit(X["train"], yt)
        fitted[name] = m
        for split, d in (("Validation", va), ("Test", te)):
            y = make_xy(d, task).values
            p = np.array([classes[i] for i in m.predict(X["val" if split == "Validation" else "test"])])
            row = {"model": name, "split": split, "n": len(y), "accuracy": accuracy_score(y, p),
                   "macro_F1": f1_score(y, p, labels=labels, average="macro", zero_division=0)}
            for l, r in zip(labels, recall_score(y, p, labels=labels, average=None, zero_division=0)):
                row[f"recall_{l}"] = r
            out.append(row)
    return pd.DataFrame(out).round(3), fitted, cols


def fit_models(F, task="four", seed=0):
    """Train the three feature models on the Train split and return what is needed to score other feature tables."""
    from xgboost import XGBClassifier
    F = F[F["error"] == ""]
    cols = feature_cols(F); tr = F[F.split == "Train"]; med = tr[cols].median()
    ytr = make_xy(tr, task).values; classes = sorted(set(ytr)); remap = {c: i for i, c in enumerate(classes)}
    yt = np.array([remap[v] for v in ytr]); Xtr = tr[cols].fillna(med).fillna(0).values
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000)),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                 random_state=seed, n_jobs=2, eval_metric="mlogloss"),
    }
    for name, mod in models.items():
        if name == "xgboost": mod.fit(Xtr, yt, sample_weight=compute_sample_weight("balanced", yt))
        else: mod.fit(Xtr, yt)
    return {"models": models, "cols": cols, "med": med, "classes": classes, "task": task}

def score(bundle, Fe, condition):
    """Score a feature table with already-trained models. Returns accuracy and macro-F1 per model."""
    task, cols, med, classes = bundle["task"], bundle["cols"], bundle["med"], bundle["classes"]
    Fe = Fe[Fe["error"] == ""]; X = Fe[cols].fillna(med).fillna(0).values; y = make_xy(Fe, task).values
    labels = [0, 1] if task == "binary" else [0, 1, 2, 3]; rows = []
    for name, mod in bundle["models"].items():
        p = np.array([classes[i] for i in mod.predict(X)])
        rows.append({"task": task, "condition": condition, "model": name, "n": len(y), "face_detect_rate": round(Fe["detect_rate"].mean(), 3),
                     "accuracy": accuracy_score(y, p), "macro_F1": f1_score(y, p, labels=labels, average="macro", zero_division=0)})
    return pd.DataFrame(rows).round(3)

def export_lr_json(F, task="binary", path="vingage_model.json"):
    """Train the logistic-regression model on Train and save it as plain JSON (no sklearn needed to run it)."""
    import json
    b = fit_models(F, task=task)
    pipe = b["models"]["logistic_regression"]; sc, lr = pipe[0], pipe[1]
    obj = {"task": task, "cols": b["cols"], "median": [float(b["med"][c]) for c in b["cols"]],
           "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": lr.coef_.tolist(),
           "intercept": lr.intercept_.tolist(), "classes": [int(c) for c in lr.classes_]}
    json.dump(obj, open(path, "w"))
    return obj
