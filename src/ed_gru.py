"""GRU on per-frame signal sequences (run on a Colab GPU). Same splits and metrics as the baselines."""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

T = 50

def load_split(F, seq_dir, split, task):
    d = F[(F.split == split) & (F.error == "")]
    X, y, keep = [], [], []
    for clip, e in zip(d["clip"], d["Engagement"].astype(int)):
        p = os.path.join(seq_dir, f"{clip}.npy")
        if not os.path.exists(p): continue
        a = np.load(p)[:T]
        if len(a) < T: a = np.vstack([a, np.full((T - len(a), a.shape[1]), np.nan, np.float32)])
        X.append(a); y.append(int(e >= 2) if task == "binary" else e); keep.append(clip)
    return np.stack(X), np.array(y), keep

def prep(X, mu=None, sd=None):
    """Standardise signals; every NaN becomes 0 with a per-signal 'valid' flag so no NaN reaches the network."""
    sig = X[..., :-1]
    if mu is None:
        flat = sig.reshape(-1, sig.shape[-1])
        mu = np.nan_to_num(np.nanmean(flat, axis=0)); sd = np.nan_to_num(np.nanstd(flat, axis=0), nan=1.0) + 1e-6
    Z = (sig - mu) / sd
    valid = ~np.isnan(Z)
    Z = np.clip(np.where(valid, Z, 0.0), -10, 10)
    det = np.nan_to_num(X[..., -1:], nan=0.0)
    out = np.concatenate([Z, valid.astype(np.float32), det], axis=-1).astype(np.float32)
    assert np.isfinite(out).all(), "non-finite values reached the GRU input"
    return out, mu, sd

class GRUNet(nn.Module):
    def __init__(self, d_in, n_cls, h=64):
        super().__init__(); self.gru = nn.GRU(d_in, h, batch_first=True); self.drop = nn.Dropout(0.3); self.fc = nn.Linear(h, n_cls)
    def forward(self, x):
        _, hN = self.gru(x); return self.fc(self.drop(hN[-1]))

def focal(logits, y, w, gamma=2.0):
    lp = torch.log_softmax(logits, 1); p = lp.exp()
    return (-(w[y] * (1 - p.gather(1, y[:, None])[:, 0]) ** gamma * lp.gather(1, y[:, None])[:, 0])).mean()

def metrics(y, p, labels):
    r = {"accuracy": accuracy_score(y, p), "macro_F1": f1_score(y, p, labels=labels, average="macro", zero_division=0)}
    for l, v in zip(labels, recall_score(y, p, labels=labels, average=None, zero_division=0)): r[f"recall_{l}"] = v
    return r

def train_gru(F, seq_dir, task="four", epochs=40, seed=0, patience=8):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, ytr, _ = load_split(F, seq_dir, "Train", task); Xva, yva, _ = load_split(F, seq_dir, "Validation", task); Xte, yte, _ = load_split(F, seq_dir, "Test", task)
    Xtr, mu, sd = prep(Xtr); Xva, _, _ = prep(Xva, mu, sd); Xte, _, _ = prep(Xte, mu, sd)
    labels = [0, 1] if task == "binary" else [0, 1, 2, 3]
    cw = np.ones(len(labels)); present = np.unique(ytr)
    cw[present] = compute_class_weight("balanced", classes=present, y=ytr)
    w = torch.tensor(cw, dtype=torch.float32, device=dev)
    net = GRUNet(Xtr.shape[-1], len(labels)).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    tt = lambda a: torch.tensor(a, device=dev)
    xtr, ytr_t = tt(Xtr), tt(ytr).long()
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]; opt.zero_grad(); loss = focal(net(xtr[idx]), ytr_t[idx], w)
            if not torch.isfinite(loss): raise RuntimeError('loss became NaN/inf')
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad(): pv = net(tt(Xva)).argmax(1).cpu().numpy()
        f1 = f1_score(yva, pv, labels=labels, average="macro", zero_division=0)
        if f1 > best: best, bad, best_state = f1, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    net.load_state_dict(best_state); net.eval(); out = []
    with torch.no_grad():
        for name, X, y in (("Validation", Xva, yva), ("Test", Xte, yte)):
            p = net(tt(X)).argmax(1).cpu().numpy(); print(name, "predicted class counts:", dict(zip(*np.unique(p, return_counts=True)))); out.append({"model": "gru", "split": name, "n": len(y), **metrics(y, p, labels)})
    return pd.DataFrame(out).round(3), net
