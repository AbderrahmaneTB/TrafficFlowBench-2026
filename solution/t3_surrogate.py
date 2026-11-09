"""Local Task 3 surrogate on train days, following score_task3.py.

The organizer boundary flux is projected onto the released observations, so for the
exact truth every scored transition balances: rhs = dN_true. A submission's residual is
then |dN_sub - dN_true|, non-zero only on transitions that touch a target cell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RELEASE, network


class PhysicsFrame:
    def __init__(self, panel: str, z: dict):
        self.panel = panel
        net = network(panel)
        self.links = [str(x) for x in z["links"]]
        L = len(self.links)
        topo = pd.read_csv(f"{RELEASE}/corridors/{panel}/network/lwr_mainline_topology.csv", dtype=str)
        topo_ids = set(topo.link_id.astype(str))
        sub = set(self.links)
        obs = {}
        on_att, off_att = set(), set()
        for r in topo.itertuples(index=False):
            inc = {v.strip() for v in str(r.incoming_link_ids).split(";") if v.strip()}
            out = {v.strip() for v in str(r.outgoing_link_ids).split(";") if v.strip()}
            obs[str(r.link_id)] = (inc & topo_ids) <= sub and (out & topo_ids) <= sub
            if isinstance(r.on_ramp_link_ids, str) and len(r.on_ramp_link_ids):
                on_att.add(str(r.link_id))
            if isinstance(r.off_ramp_link_ids, str) and len(r.off_ramp_link_ids):
                off_att.add(str(r.link_id))
        self.observable = np.array([obs.get(l, False) for l in self.links])
        # ramp validity per (day, slot, link): any valid ramp of that type mapped to the link
        amap = pd.read_csv(f"{RELEASE}/corridors/{panel}/network/ramp_attachment_map.csv", dtype=str)
        lidx = {l: i for i, l in enumerate(self.links)}
        rl = np.array([lidx.get(x, -1) for x in z["ramp_link"]])
        rtype = np.array([str(x).upper() for x in z["ramp_type"]])
        rvalid = z["relig"].astype(bool) & (z["rpct"] >= 75) & np.isfinite(z["rq"])
        D, T, _ = rvalid.shape
        on_v = np.zeros((D, T, L), bool); off_v = np.zeros((D, T, L), bool)
        for j in range(len(rl)):
            if rl[j] < 0:
                continue
            if rtype[j] in ("OR", "ON"):
                on_v[:, :, rl[j]] |= rvalid[:, :, j]
            elif rtype[j] in ("FR", "OFF"):
                off_v[:, :, rl[j]] |= rvalid[:, :, j]
        on_struct = np.array([l not in on_att for l in self.links])
        off_struct = np.array([l not in off_att for l in self.links])
        self.ramp_ok = (on_v | on_struct[None, None, :]) & (off_v | off_struct[None, None, :])
        self.length = net.length_km.to_numpy()
        self.lanes = net.lanes.to_numpy()
        self.vf = net.free_speed_kmh.to_numpy()
        cap = net.capacity_vph.to_numpy()
        self.kc = cap / np.maximum(self.vf, 1)
        kj = net.k_jam.to_numpy() if "k_jam" in net else np.full(L, np.nan)
        kj = np.where(np.isfinite(kj), kj, 100 * self.lanes)
        self.kj = np.maximum(kj, self.kc * 1.05)
        self.cap = cap

    def score(self, V, Q, elig, target, days):
        """V, Q: (D, T, L) submitted state over the chosen days (truth outside targets);
        elig: eligible cells; target: cells that were predicted; days: day indices in this regime."""
        Vd, Qd, E, Tg, RO = V[days], Q[days], elig[days], target[days], self.ramp_ok[days]
        D, T, L = Vd.shape
        k = Qd / np.maximum(Vd, 1.0)
        # FD term on every eligible cell
        ql = Qd / self.lanes; kl = k / self.lanes
        kc = self.kc / self.lanes; kj = self.kj / self.lanes; cl = self.cap / self.lanes
        qfd = np.where(kl <= kc, self.vf * kl, (cl / np.maximum(kj - kc, 1e-9)) * np.maximum(kj - kl, 0))
        m = E
        s_fd = max(0.0, 1 - np.abs(ql - qfd)[m].sum() / (np.abs(ql)[m].sum() + 1e-9))
        return s_fd, k * self.length

    def lwr(self, Nsub, Ntrue, elig, days):
        """Sum |dN_sub - dN_true| and sum |dN_true| over scored transitions of the chosen days.
        Consecutive days are joined across midnight like the official scorer's sort."""
        N1 = Nsub.reshape(-1, Nsub.shape[-1]); N0 = Ntrue.reshape(-1, Ntrue.shape[-1])
        E = elig[days].reshape(-1, elig.shape[-1])
        RO = self.ramp_ok[days].reshape(-1, elig.shape[-1])
        T = 288
        # a transition is valid if both cells are eligible and 5 minutes apart
        idx = np.arange(len(E))
        dayno = np.repeat(np.asarray(days), T)
        nxt_ok = np.r_[(dayno[1:] == dayno[:-1]) | ((dayno[1:] == dayno[:-1] + 1) & (idx[1:] % T == 0)), False]
        valid = E[:-1] & E[1:] & nxt_ok[:-1, None] & RO[:-1] & self.observable[None, :]
        d1 = N1[1:] - N1[:-1]; d0 = N0[1:] - N0[:-1]
        return float(np.abs(d1 - d0)[valid].sum()), float(np.abs(d0)[valid].sum())
