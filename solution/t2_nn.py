"""Task 2 corridor network: a dilated 1D convolutional net along the corridor (links in travel order).

Same windows, same leak-free inputs (v11 features + corridor level 2, link identity blanked, location priors rolled
in half of the training windows) and same February holdout as the LightGBM / CatBoost bags; the net sees every link
of a window at once, so the extent and fronts of a jam are learned instead of hand-built. A second learner family
for the ensemble. Trained on a GPU (Kaggle), evaluated and applied on CPU.

  python t2_nn.py train --tag n14 --seed 0 [--full]
  python t2_nn.py predict --tag n14 [--full]       -> work/t2/nnprobs_<tag>[_full].npz (validation + private windows)
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import t2_train as T
from common import QUEUE_PANELS

CFG = {"aug": True, "corr": 2, "data": "11", "drop": ["link_pos", "panel"], "prior_shift": 0.5}


@functools.lru_cache(maxsize=None)
def travel_order(panel: str) -> np.ndarray:
    """Links from the most upstream to the most downstream, read from link_pos of the (small) validation file."""
    z = np.load(f"{T.OUT}/validation{CFG['data']}_{panel}.npz", allow_pickle=True)
    return np.argsort(z["X"][0][:, list(z["names"]).index("link_pos")], kind="stable")


class Net(nn.Module):
    def __init__(self, cin: int, out: int, ch: int = 128, dil=(1, 2, 4, 8, 16, 32, 1, 2)):
        super().__init__()
        self.inp = nn.Conv1d(cin, ch, 1)
        self.norms = nn.ModuleList([nn.GroupNorm(8, ch) for _ in dil])
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(ch, ch, 3, padding=d, dilation=d), nn.GELU(),
                                                   nn.Dropout(0.1), nn.Conv1d(ch, ch, 1)) for d in dil])
        self.out = nn.Conv1d(ch, out, 1)

    def forward(self, x):
        h = self.inp(x)
        for n, b in zip(self.norms, self.blocks):
            h = h + b(n(h))
        return self.out(h)


class Scaler:
    """Per-feature standardization fitted on the training links; NaN -> 0 plus a missing flag per feature."""

    def __init__(self, mu=None, sd=None):
        self.mu, self.sd = mu, sd

    def fit(self, Xs):
        flat = np.concatenate([x.reshape(-1, x.shape[-1]) for x in Xs]).astype(np.float64)
        self.mu = np.nan_to_num(np.nanmean(flat, 0)); self.sd = np.nan_to_num(np.nanstd(flat, 0)) + 1e-3
        return self

    def __call__(self, X):          # (L, F) -> (2F, L)
        miss = ~np.isfinite(X)
        Z = np.where(miss, 0.0, (X - self.mu) / self.sd).clip(-10, 10)
        return np.concatenate([Z, miss], axis=1).T.astype(np.float32)


def windows(parts, kind):
    """(X (L,F) in travel order, y (6,L), mask (6,L) of supervised cells, hold) per window of one condition."""
    out = []
    for z in parts:
        o = travel_order(z["panel"])
        names = list(z["names"])
        for X, y, m, cond, hold in zip(z["X"], z["y"], z["m"], z["meta_cond"], z["meta_hold"]):
            if (cond == "queue_ongoing") != (kind == "ongoing"):
                continue
            if kind == "ongoing":
                mask = m & T.candidates(X, names)[None, :]
            else:
                mask = np.zeros_like(m); mask[5] = m[5]
            out.append((X[o], y[:, o].astype(np.float32), mask[:, o], bool(hold)))
    return out


STATIC = ["vf", "lanes", "cap_lane", "cap_ratio_down", "lane_ratio_down", "length"]


def crop_jitter(w, rng, crop, jitter, static_idx):
    """Training-only regularization against memorizing a corridor's layout: a random stretch of the corridor and
    multiplicative noise on the static per-link attributes."""
    X, y, m, h = w
    L = X.shape[0]
    if crop and L > crop:
        c = int(rng.integers(crop // 2, crop + 1)); s = int(rng.integers(0, L - c + 1))
        X, y, m = X[s:s + c], y[:, s:s + c], m[:, s:s + c]
    if jitter:
        X = X.copy()
        X[:, static_idx] *= np.exp(jitter * rng.standard_normal((X.shape[0], len(static_idx)))).astype(np.float32)
    return X, y, m, h


def batches(ws, scaler, bs, rng, shuffle, aug=None):
    idx = rng.permutation(len(ws)) if shuffle else np.arange(len(ws))
    for s in range(0, len(idx), bs):
        chunk = [ws[i] for i in idx[s:s + bs]]
        if aug is not None:
            chunk = [aug(w) for w in chunk]
        Lm = max(w[0].shape[0] for w in chunk)
        F = 2 * chunk[0][0].shape[1]
        xb = np.zeros((len(chunk), F, Lm), np.float32); yb = np.zeros((len(chunk), 6, Lm), np.float32)
        mb = np.zeros((len(chunk), 6, Lm), np.float32)
        for k, (X, y, m, _) in enumerate(chunk):
            L = X.shape[0]
            xb[k, :, :L] = scaler(X); yb[k, :, :L] = y; mb[k, :, :L] = m / max(m.sum(), 1)
        yield torch.from_numpy(xb), torch.from_numpy(yb), torch.from_numpy(mb)


def window_iou(P, y, m, kind, thr=0.5):
    pred = np.zeros(P.shape, bool)
    if kind == "ongoing":
        pred = P >= thr
    else:
        s = P[5] >= thr
        if not s.any():
            s[np.argmax(P[5])] = True
        pred[5] = s
    return T.iou(pred, y.astype(bool), m.astype(bool))


def predict_windows(net, scaler, ws, device, bs=32):
    net.eval(); out = []
    with torch.no_grad():
        for xb, _, _ in batches(ws, scaler, bs, None, False):
            p = torch.sigmoid(net(xb.to(device))).cpu().numpy()
            out += [p[k] for k in range(len(p))]
    return [p[:, :w[0].shape[0]] for p, w in zip(out, ws)]


def train(a):
    for k, v in (("AUG", True), ("CORR", CFG["corr"]), ("DATA", CFG["data"]), ("DROP", CFG["drop"]),
                 ("PRIOR_SHIFT", CFG["prior_shift"])):
        setattr(T, k, v)
    parts = T.load_all(seed=a.seed)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    names = list(parts[0]["names"])
    static_idx = [names.index(n) for n in STATIC if n in names]
    aug = (lambda w: crop_jitter(w, rng, a.crop, a.jitter, static_idx)) if (a.crop or a.jitter) else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sfx = "_full" if a.full else ""
    for kind in ("ongoing", "onset"):
        ws = windows(parts, kind)
        fit = [w for w in ws if a.full or not w[3]]; hold = [w for w in ws if w[3]]
        scaler = Scaler().fit([w[0] for w in fit])
        net = Net(2 * fit[0][0].shape[1], 6).to(device)
        epochs = json.load(open(f"{T.OUT}/nnrounds_{a.tag}.json"))[kind] if a.full else a.epochs
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=epochs * ((len(fit) + a.bs - 1) // a.bs))
        lossf = nn.BCEWithLogitsLoss(reduction="none")
        best, best_ep, best_state = -1.0, epochs, None
        for ep in range(epochs):
            net.train(); t0 = time.time(); tot = 0.0
            for xb, yb, mb in batches(fit, scaler, a.bs, rng, True, aug):
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                loss = (lossf(net(xb), yb) * mb).sum() / xb.shape[0]
                opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss)
            msg = f"{kind} ep {ep + 1}/{epochs} loss {tot:.4f} {time.time() - t0:.0f}s"
            if not a.full:
                P = predict_windows(net, scaler, hold, device)
                iou = float(np.mean([window_iou(p, w[1], w[2], kind) for p, w in zip(P, hold)]))
                msg += f" hold IoU {iou:.4f}"
                if iou > best:
                    best, best_ep = iou, ep + 1
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            print(msg, flush=True)
        state = best_state if best_state is not None else {k: v.detach().cpu() for k, v in net.state_dict().items()}
        torch.save({"state": state, "mu": scaler.mu, "sd": scaler.sd, "cin": 2 * fit[0][0].shape[1]},
                   f"{T.OUT}/nn_{kind}_{a.tag}{sfx}.pt")
        if not a.full:
            print(f"{kind}: best hold IoU {best:.4f} at epoch {best_ep}", flush=True)
            r = json.load(open(f"{T.OUT}/nnrounds_{a.tag}.json")) if os.path.exists(f"{T.OUT}/nnrounds_{a.tag}.json") else {}
            r[kind] = best_ep
            json.dump(r, open(f"{T.OUT}/nnrounds_{a.tag}.json", "w"))
    json.dump(dict(CFG, nn=True), open(f"{T.OUT}/config_{a.tag}.json", "w"))


class NNModels:
    """Inference on CPU for one tag: per window (L, F) model inputs -> (6, L) probabilities."""

    def __init__(self, tag, full=False):
        sfx = "_full" if full else ""
        self.nets, self.scalers = {}, {}
        for kind in ("ongoing", "onset"):
            ck = torch.load(f"{T.OUT}/nn_{kind}_{tag}{sfx}.pt", map_location="cpu", weights_only=False)
            net = Net(ck["cin"], 6); net.load_state_dict(ck["state"]); net.eval()
            self.nets[kind], self.scalers[kind] = net, Scaler(ck["mu"], ck["sd"])

    def window(self, X, names, panel, cond):
        o = travel_order(panel)
        kind = "ongoing" if cond == "queue_ongoing" else "onset"
        with torch.no_grad():
            p = torch.sigmoid(self.nets[kind](torch.from_numpy(self.scalers[kind](X[o]))[None])).numpy()[0]
        P = np.zeros_like(p); P[:, o] = p
        if kind == "ongoing":
            P[:, ~T.candidates(X, names)] = 0.0
        else:
            P[:5] = 0.0
        return P


def predict(tag, full=False, splits=("validation", "private")):
    """Probabilities for the official windows, the same input path as t2_predict.predict."""
    nm = NNModels(tag, full)
    probs = {}
    for p in QUEUE_PANELS:
        for s in splits:
            z = np.load(f"{T.OUT}/{s}{CFG['data']}_{p}.npz", allow_pickle=True)
            Xall, names0 = T.add_corridor(z["X"], list(z["names"]), CFG["corr"])
            Xall = T.blank(Xall, names0, CFG["drop"])
            for X, wid, cond in zip(Xall, z["wid"], z["cond"]):
                Xa, names = T.augment(X, names0)
                probs[str(wid)] = nm.window(Xa, names, p, cond)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("--tag", default="n14")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--crop", type=int, default=0, help="train on random corridor stretches of crop/2..crop links")
    ap.add_argument("--jitter", type=float, default=0.0, help="log-normal noise on static link attributes")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    else:
        P = predict(a.tag, a.full)
        np.savez_compressed(f"{T.OUT}/nnprobs_{a.tag}{'_full' if a.full else ''}.npz", **P)


if __name__ == "__main__":
    main()
