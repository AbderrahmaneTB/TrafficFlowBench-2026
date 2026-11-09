"""Task 2 windows and features.

Training windows are cut from the train split with the organizer's published
selection procedure (synthetic_release_v1.json): history coverage >= 0.7, the
visible-history condition (>= 2 queued observations with >= 2 on one link ->
ongoing, else onset), a queue in the horizon, persistence IoU <= 0.9 for ongoing,
and the earliest qualifying origin with 360-minute spacing per condition.

Every feature of a window is computed from data stamped at or before its origin T:
the 60-minute raw history (T-60..T-5), the released masked row at T, and earlier
masked rows of the same day. Nothing after T is read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RELEASE, PANELS, network, free_speed

HIST = 12          # T-60 .. T-5
CTX = 36           # extra masked context before the history (3 hours)
SPACING = 72       # 360 minutes
PATCH_J = 6        # +- links in the raw patch
PATCH_K = 6        # steps back (T, T-5, ..., T-30)


def queued_obs(v, vcut):
    return np.isfinite(v) & (v <= vcut)


def select_windows(v, el, vcut, phase=0, max_persist=0.9, spacing=SPACING):
    """Emulate the published selector on one panel's continuous train timeline.

    v, el: (N, L) observed speed and eligibility over the whole split (time flattened).
    Returns list of (origin_index, condition).
    """
    N, L = v.shape
    Q = queued_obs(v, vcut) & el
    E = el & np.isfinite(v)
    cumQ = np.concatenate([np.zeros((1, L), np.int32), np.cumsum(Q, axis=0, dtype=np.int32)])
    cumE = np.concatenate([np.zeros((1, L), np.int32), np.cumsum(E, axis=0, dtype=np.int32)])
    anyQ = Q.any(axis=1)
    out = []
    last = {"queue_onset": -10**9, "queue_ongoing": -10**9}
    # state at T for persistence: last eligible observation, carried forward up to the history
    for i in range(HIST + phase, N - 6):
        hq = anyQ[i + 1:i + 7]
        if not hq.any():
            continue
        cov = (cumE[i] - cumE[i - HIST]).sum() / (HIST * L)
        if cov < 0.7:
            continue
        per_link = cumQ[i] - cumQ[i - HIST]
        ongoing = per_link.sum() >= 2 and per_link.max() >= 2
        cond = "queue_ongoing" if ongoing else "queue_onset"
        if i - last[cond] < spacing:
            continue
        if cond == "queue_onset":
            if hq[:5].any():          # the earliest qualifying origin puts the queue at T+30
                continue
        else:
            s = np.zeros(L, bool); seen = np.zeros(L, bool)
            for k in range(0, HIST + 1):
                row = i - k
                new = E[row] & ~seen
                s[new] = Q[row][new]; seen |= E[row]
            fut = Q[i + 1:i + 7]; fe = E[i + 1:i + 7]
            pa = s[None, :] & fe; tb = fut & fe
            u = (pa | tb).sum()
            iou = (pa & tb).sum() / u if u else 1.0
            if iou > max_persist:
                continue
        out.append((i, cond))
        last[cond] = i
    return out


class PanelFeatures:
    """Static per-link quantities and the feature builder for one panel."""

    def __init__(self, panel: str, prof_qu: np.ndarray, head_rate: np.ndarray | None = None):
        self.panel = panel
        net = network(panel)
        self.L = len(net)
        self.sign = net.attrs["direction"]
        self.vf = net.free_speed_kmh.to_numpy(np.float32)
        self.vcut = net.vcut.to_numpy(np.float32)
        self.vfm = free_speed(panel)
        self.lanes = net.lanes.to_numpy(np.float32)
        self.cap = net.capacity_vph.to_numpy(np.float32)
        self.kc = (net.capacity_vph / net.free_speed_kmh).to_numpy(np.float32)
        self.length = net.length_km.to_numpy(np.float32)
        self.mp = net.milepost.to_numpy(np.float32)
        self.prof_qu = prof_qu      # (288, L) queued share by slot (train)
        self.head_rate = head_rate if head_rate is not None else np.zeros(self.L, np.float32)
        # travel-direction order: position 0 = most upstream
        self.order = np.arange(self.L) if self.sign > 0 else np.arange(self.L)[::-1]
        self.rank = np.empty(self.L, int); self.rank[self.order] = np.arange(self.L)
        cap_o = self.cap[self.order]
        down_cap = np.r_[cap_o[1:], cap_o[-1]]
        self.cap_ratio_down = (down_cap / cap_o)[self.rank]
        lan_o = self.lanes[self.order]
        self.lane_ratio_down = (np.r_[lan_o[1:], lan_o[-1]] / lan_o)[self.rank]

    def features(self, V, Qf, O, T_slot, dow, condition):
        """V, Qf, O: (CTX+HIST+1, L) speed, flow, occupancy ending at row T (last row).

        Returns per-link feature matrix (L, F) and names, in the panel's link order.
        """
        L = self.L
        n = V.shape[0]
        vr = V / self.vfm[None, :]
        load = Qf / self.cap[None, :]
        dens = O * 285.0 / self.lanes[None, :] / (self.kc / self.lanes)[None, :]
        fin = np.isfinite(vr)
        q = np.where(fin, vr <= 0.6 * self.vf / self.vfm, False)
        feats, names = [], []

        def add(name, arr):
            names.append(name); feats.append(np.asarray(arr, np.float32))

        # last observation at or before T (within the history+T)
        rows = np.arange(n)[:, None]
        last_idx = np.where(fin, rows, -1).max(axis=0)
        has = last_idx >= 0
        li = np.clip(last_idx, 0, n - 1)
        cols = np.arange(L)
        vr_last = np.where(has, vr[li, cols], np.nan)
        age = np.where(has, n - 1 - last_idx, 99)
        add("vr_last", vr_last); add("age", age)
        add("load_last", np.where(has, load[li, cols], np.nan))
        add("dens_last", np.where(has, dens[li, cols], np.nan))
        qlast = np.where(has, q[li, cols], False)
        add("q_last", qlast)
        h = slice(n - 1 - HIST, n)  # history + T (13 rows)
        with np.errstate(all="ignore"):
            for w in (3, 6, 13):
                sl = slice(n - w, n)
                add(f"vr_mean{w}", np.nanmean(vr[sl], axis=0))
                add(f"load_mean{w}", np.nanmean(load[sl], axis=0))
                add(f"dens_mean{w}", np.nanmean(dens[sl], axis=0))
            add("vr_min13", np.nanmin(vr[h], axis=0))
            add("vr_trend", np.nanmean(vr[n - 3:], axis=0) - np.nanmean(vr[n - 9:n - 3], axis=0))
            add("load_trend", np.nanmean(load[n - 3:], axis=0) - np.nanmean(load[n - 9:n - 3], axis=0))
            add("dens_trend", np.nanmean(dens[n - 3:], axis=0) - np.nanmean(dens[n - 9:n - 3], axis=0))
            add("vr_ctx_min", np.nanmin(vr[:n - 13], axis=0) if n > 13 else np.full(L, np.nan))
            add("q_ctx_n", q[:n - 13].sum(axis=0))
        add("q_hist_n", q[h].sum(axis=0))
        add("obs_hist_n", fin[h].sum(axis=0))
        # consecutive queued run ending at the last observation
        run = np.zeros(L)
        alive = np.ones(L, bool)
        for k in range(n - 1, -1, -1):
            obs = fin[k]
            alive &= ~(obs & ~q[k])
            run += alive & obs & q[k]
        add("q_run", run)
        # state at T: last obs within 3 steps, else unknown
        st = np.where(has & (age <= 3), qlast.astype(float), np.nan)
        # direction-ordered helpers
        so = st[self.order]
        vro = vr_last[self.order]
        Lr = np.arange(L)
        qpos = np.flatnonzero(so == 1)
        if len(qpos):
            # distance (links) to nearest queued link downstream / upstream in travel order
            nxt = np.searchsorted(qpos, Lr, side="left")
            dd = np.where(nxt < len(qpos), qpos[np.minimum(nxt, len(qpos) - 1)] - Lr, 999)
            prv = np.searchsorted(qpos, Lr, side="right") - 1
            du = np.where(prv >= 0, Lr - qpos[np.maximum(prv, 0)], 999)
        else:
            dd = np.full(L, 999); du = np.full(L, 999)
        add("dist_q_down", dd[self.rank]); add("dist_q_up", du[self.rank])
        # clusters of queued links (allowing unknown gaps of 1)
        filled = np.nan_to_num(so, nan=0.0)
        cl_len = np.zeros(L); cl_head = np.full(L, 999.0); cl_tail = np.full(L, 999.0)
        i = 0
        while i < L:
            if filled[i] == 1:
                j = i
                while j + 1 < L and (filled[j + 1] == 1 or (np.isnan(so[j + 1]) and j + 2 < L and filled[j + 2] == 1)):
                    j += 1
                cl_len[i:j + 1] = j - i + 1
                cl_head[i:j + 1] = j - np.arange(i, j + 1)     # links to the downstream end
                cl_tail[i:j + 1] = np.arange(i, j + 1) - i     # links from the upstream end
                i = j + 1
            else:
                i += 1
        add("cl_len", cl_len[self.rank]); add("cl_to_head", cl_head[self.rank]); add("cl_from_tail", cl_tail[self.rank])
        add("n_q_T", np.full(L, np.nansum(so == 1)))
        # tail movement: upstream-most queued position at T vs 6 steps earlier (nearest cluster below)
        def tail_positions(row):
            s_ = q[row][self.order] & fin[row][self.order]
            return np.flatnonzero(s_)
        q_now = tail_positions(n - 1); q_before = tail_positions(max(n - 7, 0))
        add("nq_change6", np.full(L, len(q_now) - len(q_before)))
        # window of neighbour states (travel order) +-PATCH_J at T..T-PATCH_K
        vro_all = vr[:, self.order]
        for k in range(PATCH_K + 1):
            row = n - 1 - k
            r = vro_all[row]
            for j in range(-PATCH_J, PATCH_J + 1):
                idx = Lr + j
                ok = (idx >= 0) & (idx < L)
                val = np.where(ok, r[np.clip(idx, 0, L - 1)], np.nan)
                add(f"p_{k}_{j:+d}", val[self.rank])
        # neighbour loads at T (last obs) up / down
        lo_last = np.where(has, load[li, cols], np.nan)[self.order]
        for j in (-3, -1, 1, 3):
            idx = Lr + j
            ok = (idx >= 0) & (idx < L)
            add(f"load_n{j:+d}", np.where(ok, lo_last[np.clip(idx, 0, L - 1)], np.nan)[self.rank])
        # static
        add("vf", self.vfm); add("lanes", self.lanes); add("cap_lane", self.cap / self.lanes)
        add("cap_ratio_down", self.cap_ratio_down); add("lane_ratio_down", self.lane_ratio_down)
        add("length", self.length); add("link_pos", self.rank / max(L - 1, 1))
        add("head_rate", self.head_rate)
        for k in (0, 3, 6):
            add(f"prof_qu{k}", self.prof_qu[min(T_slot + k, 287)])
        add("slot", np.full(L, T_slot)); add("dow", np.full(L, dow))
        add("ongoing", np.full(L, 1.0 if condition == "queue_ongoing" else 0.0))
        add("panel", np.full(L, PANELS.index(self.panel)))
        return np.column_stack(feats), names


def select_windows_u(v, el, vcut, U, spacing=SPACING, max_persist=0.9, persist_cells="elig"):
    """Selector using the estimated underlying queue state U (N, L) for everything the
    organizer computes from the underlying state (horizon queue, persistence IoU), and
    the observed eligible queue for the visible-history condition."""
    N, L = v.shape
    Qv = queued_obs(v, vcut) & el
    E = el & np.isfinite(v)
    cumQ = np.concatenate([np.zeros((1, L), np.int32), np.cumsum(Qv, axis=0, dtype=np.int32)])
    cumE = np.concatenate([np.zeros((1, L), np.int32), np.cumsum(E, axis=0, dtype=np.int32)])
    anyU = U.any(axis=1)
    out = []
    last = {"queue_onset": -10**9, "queue_ongoing": -10**9}
    for i in range(HIST, N - 6):
        hq = anyU[i + 1:i + 7]
        if not hq.any():
            continue
        cov = (cumE[i] - cumE[i - HIST]).sum() / (HIST * L)
        if cov < 0.7:
            continue
        per_link = cumQ[i] - cumQ[i - HIST]
        ongoing = per_link.sum() >= 2 and per_link.max() >= 2
        cond = "queue_ongoing" if ongoing else "queue_onset"
        if i - last[cond] < spacing:
            continue
        if cond == "queue_onset":
            if hq[:5].any():
                continue
        else:
            fe = E[i + 1:i + 7] if persist_cells == "elig" else np.ones((6, L), bool)
            pa = U[i][None, :] & fe; tb = U[i + 1:i + 7] & fe
            u = (pa | tb).sum()
            iou = (pa & tb).sum() / u if u else 1.0
            if iou > max_persist:
                continue
        out.append((i, cond))
        last[cond] = i
    return out


def select_official(v, el, vcut, U, n_per=5, spacing=SPACING, max_persist=0.9, start=HIST, stop=None, hq_elig=True):
    """Two-pass emulation: onset windows first (earliest, spaced), then ongoing windows that are
    also spaced from every window already chosen (both conditions). Returns list of (i, cond).
    n_per=None keeps going to the end of the timeline."""
    N, L = v.shape
    stop = N - 6 if stop is None else stop
    Qv = queued_obs(v, vcut) & el
    E = el & np.isfinite(v)
    cumQ = np.concatenate([np.zeros((1, L), np.int32), np.cumsum(Qv, axis=0, dtype=np.int32)])
    cumE = np.concatenate([np.zeros((1, L), np.int32), np.cumsum(E, axis=0, dtype=np.int32)])
    anyU = (U & E).any(axis=1) if hq_elig else U.any(axis=1)
    chosen = []

    def ok_spacing(i):
        return all(abs(i - j) >= spacing for j, _ in chosen)

    for cond in ("queue_onset", "queue_ongoing"):
        cnt = 0
        for i in range(max(start, HIST), stop):
            if n_per is not None and cnt >= n_per:
                break
            hq = anyU[i + 1:i + 7]
            if not hq.any():
                continue
            if (cumE[i] - cumE[i - HIST]).sum() / (HIST * L) < 0.7:
                continue
            per_link = cumQ[i] - cumQ[i - HIST]
            ongoing = per_link.sum() >= 2 and per_link.max() >= 2
            if ongoing != (cond == "queue_ongoing"):
                continue
            if not ok_spacing(i):
                continue
            if cond == "queue_onset":
                if hq[:5].any():
                    continue
            else:
                fe = E[i + 1:i + 7]
                pa = U[i][None, :] & fe; tb = U[i + 1:i + 7] & fe
                u = (pa | tb).sum()
                if u and (pa & tb).sum() / u > max_persist:
                    continue
            chosen.append((i, cond)); cnt += 1
    return sorted(chosen)
