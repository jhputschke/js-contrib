#!/usr/bin/env python3
"""Parton- and hydro-level jet and wake observables from PairH5Writer productions.

Goes once through the pair files of one or more productions (``run_prod_jet.py``) and writes
everything the figures need into one HDF5 file. ``wake_observables.ipynb`` only reads that
file, so a new figure never needs another pass over the data.

What is computed
----------------
Per **shower** (one per parton PYTHIA hands to the energy loss):

* the initial parton (pid, E, pT, y, phi), its production vertex, the radial alignment
  ``cos_alpha`` = r_hat . n_hat relative to the fireball's centre (+1 heading out, -1 in)
  and the angle to the background's participant plane ``dphi_psi2`` in [0, pi/2];
* the energy bookkeeping from the shower graph: surviving (pstat 0, 1, 22), absorbed (-11),
  missing four-momentum (-13), holes (-17), and the net deposit absorbed + missing - holes,
  plus the surviving energy within dR < 0.4 of the initial direction;
* the same deposit from the droplet table, each droplet assigned to its shower exactly
  through the graph (see ``assign_droplets``), with the deposition times and the background
  temperature at each deposit;
* the medium along the parton's straight path, sampled from the BACKGROUND leg (the energy
  loss runs on MUSIC_1's medium, ``arr`` already contains the wake): path length above T_C,
  the line integrals of T^2 (collisional), T^3 and (tau - tau_in) T^3 (radiative, L^2-like),
  the flow factor gamma (1 - v.n) and flow-weighted T^2, T^3.

Per **droplet**: position, four-momentum, shower, whether it was injected (deposited while
the jet leg's hydro still ran), and the background e, T, v at the deposit. Also
``kernel_flux``, the sum of the droplet's CausalLiquefier kernel point-sampled at MUSIC's cell
centres: far from 1 for droplets at large |eta_s| or late tau (see ``KernelFlux``). It is the
C++ value (``source/flux``) when the file stores it, else the Python port. ``inj_factor`` is
the fraction of the droplet MUSIC_2 received: ``kernel_flux`` for files made before X-SCAPE
normalized the kernel, and 1 (0 for a droplet off the grid) for files with root attribute
``liquefier_normalize_on_hydro_grid = 1``. ``E_inj`` (droplet), ``E_inj_hydro`` (shower,
event) and ``evolution/D_hydro`` carry ``inj_factor``; the wake in ``arr`` follows them.

Per **event**: pTHat window and its cross section, the background's participant plane and
eccentricity, freeze-out times, and the energy totals.

Per event and tau frame (``evolution``): the four-momentum P^mu and entropy S of each leg
through the tau = const surface, and the cumulative four-momentum of the injected droplets.
With the ideal-fluid T^{mu nu} = (e + P) u^mu u^nu - P g^{mu nu} (P, s from the same EoS
table MUSIC used):

    P^mu = int tau dx dy deta [cosh(eta) T^{0 mu} - sinh(eta) T^{3 mu}]
    S    = int tau dx dy deta s gamma [cosh(eta) - sinh(eta) v_z]

The viscous stress pi^{mu nu} is not stored, and the output window (|x|, |y| < 10 fm,
|eta| < 5) is smaller than MUSIC's, so P_bg is not exactly conserved; P_jet - P_bg against
the droplets is the test of how much that matters for the wake.

Per event (``jetframe``): the wake of the leg that deposits most, binned about its source
position in (d_eta, d_phi), d_phi = 0 along its initial direction: the energy dP^0 and the
momentum along the jet dP.n per bin, at a frame while it still deposits and at the last
frame both legs are live. The maps sum to the Delta P inside the window.

Conventions: T in GeV, lengths and tau in fm (c = 1), line integrals in GeV^n fm. A
massless parton from z = 0 at t = 0 stays at eta_s = y and has moved tau transversely at
proper time tau, so the path is sampled at every output frame with tau >= TAU_START.

Usage::

    python wake_observables.py DIR_OR_PAIR_H5 [...] -o wake_observables.h5 [-j 8]
"""

import argparse
import glob
import os
import re
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import iv

try:  # Blosc, the default compression since h5_optim
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

FORMAT = "js-contrib/wake_observables"
VERSION = 1
T_C = 0.16            # GeV: Matter/Lbt <hydro_Tc>; no energy loss below it
TAU_START = 0.5       # fm: <Eloss><tStart>
E_MIN_TRACK = 0.01    # GeV: smaller droplets do not steer the source track
R_CONE = 0.4
SURVIVING = (0, 1, 22)
TOL = dict(rtol=1e-5, atol=1e-5)   # droplets are stored from float32
DETA_EDGES = np.linspace(-3.0, 3.0, 21)
DPHI_EDGES = np.linspace(0.0, 2 * np.pi, 25)


# ── EoS and grid ────────────────────────────────────────────────────────────────
class EoS:
    """MUSIC's hotQCD table (e, P, s, T), interpolated in log e."""

    def __init__(self, path):
        t = np.fromfile(path, dtype="<f8").reshape(-1, 4)
        t = t[np.argsort(t[:, 0])]
        self.e, self.P, self.s, self.T = t.T
        self.le = np.log(self.e)
        self.cs2 = np.gradient(self.P, self.e)

    def __call__(self, e):
        """(P, s, T) for energy density e [GeV/fm^3]; below the table, P ~ e, s ~ e^3/4."""
        e = np.asarray(e, dtype=np.float64)
        le = np.log(np.maximum(e, self.e[0]))
        P, s, T = (np.interp(le, self.le, c) for c in (self.P, self.s, self.T))
        low = e < self.e[0]
        if np.any(low):
            r = np.clip(e[low], 0.0, None) / self.e[0]
            P[low] *= r
            s[low] *= r ** 0.75
            T[low] *= r ** 0.25
        return P, s, T

    def cs(self, e):
        le = np.log(np.maximum(np.asarray(e, dtype=np.float64), self.e[0]))
        return np.sqrt(np.clip(np.interp(le, self.le, self.cs2), 1e-6, 1.0))


class Grid:
    def __init__(self, A):
        self.nx, self.ny, self.neta = int(A["nx"]), int(A["ny"]), int(A["neta"])
        self.dx, self.dy, self.deta, self.dtau = (float(A[k]) for k in ("dx", "dy", "deta", "dtau"))
        self.x0, self.y0, self.eta0, self.tau0 = (float(A[k]) for k in
                                                  ("x_min", "y_min", "eta_min", "tau_min"))
        self.x = self.x0 + self.dx * np.arange(self.nx)
        self.y = self.y0 + self.dy * np.arange(self.ny)
        self.eta = self.eta0 + self.deta * np.arange(self.neta)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        self.ch = np.cosh(self.eta)[None, None, :]
        self.sh = np.sinh(self.eta)[None, None, :]

    def tau(self, k):
        return self.tau0 + self.dtau * np.asarray(k)

    def index(self, x, y, eta):
        """Fractional array indices of points, for map_coordinates."""
        return ((np.asarray(x) - self.x0) / self.dx, (np.asarray(y) - self.y0) / self.dy,
                (np.asarray(eta) - self.eta0) / self.deta)


def find_eos(attrs, override):
    cands = [override, os.environ.get("MUSIC_EOS_TABLE", ""),
             os.path.join(str(attrs.get("prod_build", "")), "EOS", "hotQCD",
                          "hrg_hotqcd_eos_binary.dat")]
    path = next((c for c in cands if c and os.path.exists(c)), None)
    if path is None:
        sys.exit("hotQCD EoS table not found: pass --eos or set MUSIC_EOS_TABLE")
    if "hotqcd" not in str(attrs.get("eos_kind", "hotqcd")).lower():
        sys.exit(f"eos_kind is {attrs.get('eos_kind')!r}; only the hotQCD table is supported")
    return path


# ── hydro: fluxes through a tau surface ─────────────────────────────────────────
def flux_cells(frame, tau, eos, g):
    """Per cell: the contribution to P^mu (4, nx, ny, neta) and to S (nx, ny, neta)."""
    e = frame[0].astype(np.float64)
    vx, vy, vz = (frame[i].astype(np.float64) for i in (1, 2, 3))
    P, s, _ = eos(e)
    g2 = 1.0 / (1.0 - np.clip(vx * vx + vy * vy + vz * vz, 0.0, 1.0 - 1e-12))
    w = (e + P) * g2
    ch, sh = g.ch, g.sh
    dv = tau * g.dx * g.dy * g.deta
    p0 = dv * (ch * (w - P) - sh * w * vz)
    px = dv * w * vx * (ch - sh * vz)
    py = dv * w * vy * (ch - sh * vz)
    pz = dv * (ch * w * vz - sh * (w * vz * vz + P))
    S = dv * s * np.sqrt(g2) * (ch - sh * vz)
    return np.stack([p0, px, py, pz]), S


def flux_series(arr, eos, g):
    """(nt, 4) P^mu and (nt,) S for every frame of one leg (arr: (4, nx, ny, neta, nt))."""
    nt = arr.shape[-1]
    P, S = np.zeros((nt, 4)), np.zeros(nt)
    for k in range(nt):
        c, s = flux_cells(arr[..., k], g.tau(k), eos, g)
        P[k], S[k] = c.sum(axis=(1, 2, 3)), s.sum()
    return P, S


def participant_plane(e0, g):
    """(psi2, eps2, x_c, y_c) of the energy density at |eta| < 0.5."""
    m = np.abs(g.eta) < 0.5
    w = e0[:, :, m].sum(axis=2).astype(np.float64)
    xc, yc = np.average(g.X, weights=w), np.average(g.Y, weights=w)
    dx, dy = g.X - xc, g.Y - yc
    z = (w * (dx + 1j * dy) ** 2).sum()
    eps2 = abs(z) / (w * (dx * dx + dy * dy)).sum()
    return float((np.angle(z) + np.pi) / 2.0), float(eps2), float(xc), float(yc)


def sample(arr, g, x, y, eta, k):
    """Channels of arr (4, nx, ny, neta, nt) at points (x, y, eta) and frame indices k;
    NaN outside the output window."""
    ix, iy, ie = g.index(x, y, eta)
    coords = np.vstack([ix, iy, ie, np.asarray(k, dtype=np.float64)])
    out = np.stack([map_coordinates(arr[c], coords, order=1, mode="constant", cval=np.nan)
                    for c in range(arr.shape[0])])
    inside = ((ix >= 0) & (ix <= g.nx - 1) & (iy >= 0) & (iy <= g.ny - 1)
              & (ie >= 0) & (ie <= g.neta - 1))
    out[:, ~inside] = np.nan
    return out


# ── what MUSIC actually receives from a droplet ─────────────────────────────────
class KernelFlux:
    """Energy fraction of a droplet that MUSIC_2 receives: the CausalLiquefier kernel
    point-sampled at MUSIC's cell centres on the one step that deposits it.

    A port of CausalLiquefier::smearing_kernel (kernel_rho, kernel_j, the wave front of
    width width_delta) and of the deposit condition tau_n - dtau/2 <= tau_d + tau_delay <
    tau_n + dtau/2.  The source MUSIC gets is J^mu = (n . j) p^mu, so the four-momentum it
    receives is flux * p_droplet with flux = sum over cells of tau dx dy deta (n . j).  In the
    continuum flux = 1; on the grid it is not, and far from it when the kernel (a ball of
    radius c_diff (t - t_d), in lab z stretched by cosh(eta_d)) is resolved by a few cells:
    large |eta_d| or late tau, where one eta cell is tau * deta long."""

    def __init__(self, A):
        self.c = float(A["liquefier_c_diff"])
        self.gam = float(A["liquefier_gamma_relax"])
        self.w = float(A["liquefier_width_delta"])
        self.delay = float(A["liquefier_tau_delay"])
        self.dtau = float(A["liquefier_dtau"])
        self.tau0 = float(A["tau_min_MUSIC"])
        self.x = float(A["X_min_MUSIC"]) + float(A["dX_MUSIC"]) * np.arange(int(A["nX_MUSIC"]))
        self.y = float(A["Y_min_MUSIC"]) + float(A["dY_MUSIC"]) * np.arange(int(A["nY_MUSIC"]))
        self.eta = (float(A["eta_min_MUSIC"])
                    + float(A["deta_MUSIC"]) * np.arange(int(A["neta_MUSIC"])))
        self.cell = float(A["dX_MUSIC"]) * float(A["dY_MUSIC"]) * float(A["deta_MUSIC"])
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

    def on_grid(self, x, y, eta):
        """Whether the nearest MUSIC cell of each point exists (the normalized source
        deposits a droplet whose kernel misses every cell into that cell; off the grid it is
        lost)."""
        def inside(v, axis):
            i = np.rint((np.asarray(v) - axis[0]) / (axis[1] - axis[0]))
            return (i >= 0) & (i <= len(axis) - 1)
        return inside(x, self.x) & inside(y, self.y) & inside(eta, self.eta)

    def _kernel(self, t, r):
        c, gam = self.c, self.gam
        damp = np.exp(-gam * t) / (4 * np.pi)
        inside = r < c * t
        u = np.sqrt(np.clip(c * c * t * t - r * r, 1e-300, None))
        x = gam * u / c
        f = gam * gam / c
        rho = np.where(inside, f * (iv(1, x) / (c * u) + iv(2, x) * t / (u * u)), 0.0)
        jr = np.where(inside, f * iv(2, x) * r / (u * u), 0.0)
        rw = np.where(c * t <= self.w, c * t, self.w)
        front = (r >= c * t - rw) & inside
        xt = gam * t
        rd = np.where(front, (1 + xt + xt * xt / 2) / rw / np.clip(r, 1e-300, None) ** 2, 0.0)
        return damp * (rho + rd), damp * (jr + c * rd)

    def __call__(self, tau_d, x_d, y_d, eta_d):
        if not np.isfinite(eta_d):
            # a droplet at the hard vertex (t = z = 0) was stored with eta = 0/0 before
            # X-SCAPE set it to 0: every kernel value is then NaN in the C++, and MUSIC never
            # received it (the in-run C++ flux of such droplets is exactly 0)
            return 0.0
        tau = self.tau0 + np.round((tau_d + self.delay - self.tau0) / self.dtau) * self.dtau
        t_d, z_d = tau_d * np.cosh(eta_d), tau_d * np.sinh(eta_d)
        total = 0.0
        for eta in self.eta:                         # one eta slice at a time: small memory
            t, z = tau * np.cosh(eta), tau * np.sinh(eta)
            if t <= t_d:
                continue
            dt = t - t_d
            r = np.sqrt((self.X - x_d) ** 2 + (self.Y - y_d) ** 2 + (z - z_d) ** 2)
            near = r < self.c * dt
            if not near.any():
                continue
            rho, jr = self._kernel(dt, r[near])
            jz = (z - z_d) / np.clip(r[near], 1e-300, None) * jr
            total += (rho * np.cosh(eta) - jz * np.sinh(eta)).sum()
        return float(total * tau * self.cell)


# ── showers: graph bookkeeping and droplet assignment ───────────────────────────
def assign_droplets(Q, drops):
    """Shower of every droplet (-1 if none matches), exactly from the shower graph.

    LiquefierBase::add_hydro_sources makes one droplet per vertex, (parent) - (outgoing
    partons that are still in the shower), and one per parton it absorbs while LBT
    free-streams it (the whole parton, which is then a -11 leaf with no vertex of its own).
    Pass 1 matches the second kind by four-momentum, pass 2 the vertex droplets with those
    partons counted as outgoing."""
    src, tgt, st = Q[:, 1].astype(int), Q[:, 2].astype(int), Q[:, 4]
    p4 = Q[:, [8, 5, 6, 7]]
    hole = st == -17
    leaf = ~np.isin(tgt, src) & ~hole
    lab = np.full(len(drops), -1, dtype=np.int64)
    late = np.zeros(len(Q), bool)
    cand = np.flatnonzero(leaf & (st == -11))
    for i, d in enumerate(drops):
        m = [c for c in cand[np.all(np.isclose(p4[cand], d[4:8], **TOL), axis=1)] if not late[c]]
        if m:
            late[m[0]] = True
            lab[i] = int(Q[m[0], 0])

    stays = np.isin(st, SURVIVING) | late
    vd, vs = [], []
    for v in np.unique(src[~hole]):
        inn = np.flatnonzero((tgt == v) & ~hole)
        if not len(inn):
            continue                        # a shower's root
        out = (src == v) & ~hole & stays
        vd.append(p4[inn].sum(axis=0) - p4[out].sum(axis=0))
        vs.append(int(Q[inn[0], 0]))
    if vd:
        vd, vs, vused = np.array(vd), np.array(vs), np.zeros(len(vd), bool)
        for i, d in enumerate(drops):
            if lab[i] >= 0:
                continue
            m = np.flatnonzero(~vused & np.all(np.isclose(vd, d[4:8], **TOL), axis=1))
            if len(m):
                vused[m[0]] = True
                lab[i] = vs[m[0]]
    return lab


def rapidity(E, pz):
    return 0.5 * np.log(np.clip(E + pz, 1e-12, None) / np.clip(E - pz, 1e-12, None))


def shower_sums(Q, I):
    """Graph energies per shower, in the order of the initiator rows."""
    src, tgt, st = Q[:, 1].astype(int), Q[:, 2].astype(int), Q[:, 4]
    hole = st == -17
    leaf = ~np.isin(tgt, src) & ~hole
    surv = leaf & np.isin(st, SURVIVING)
    sgn = np.where(st == -1, -1.0, 1.0)
    rows = []
    for ini in I:
        s = Q[:, 0] == ini[0]
        E0, pz0 = ini[6], ini[5]
        y0, phi0 = rapidity(E0, pz0), np.arctan2(ini[4], ini[3])
        sv = s & (surv | (leaf & (st == -1)))
        p = (sgn[sv, None] * Q[sv][:, [8, 5, 6, 7]]).sum(axis=0)
        ys = rapidity(Q[sv, 8], Q[sv, 7])
        dphi = np.angle(np.exp(1j * (np.arctan2(Q[sv, 6], Q[sv, 5]) - phi0)))
        cone = np.hypot(ys - y0, dphi) < R_CONE
        E_hole = Q[s & hole, 8].sum()
        E_abs = Q[s & leaf & (st == -11), 8].sum()
        E_miss = Q[s & leaf & (st == -13), 8].sum()
        rows.append(dict(
            E_surv=p[0], pT_surv=float(np.hypot(p[1], p[2])),
            E_surv_R04=float((sgn[sv] * Q[sv, 8])[cone].sum()),
            n_surv=int(sv.sum()), E_absorbed=E_abs, E_miss=E_miss, E_holes=E_hole,
            E_dep_graph=E_abs + E_miss - E_hole, n_holes=int((s & hole).sum()),
            n_absorbed=int((s & leaf & (st == -11)).sum())))
    return rows


# ── one file ────────────────────────────────────────────────────────────────────
def process_file(job):
    fi, path, eos_path, max_events = job
    t0 = time.time()
    eos = EoS(eos_path)
    ev_rows, sh_rows, dr_rows, evo, maps = [], [], [], [], []
    with h5py.File(path, "r") as f:
        A = f.attrs
        g = Grid(A)
        NT = f["arr"].shape[-1]
        n_ev = len(f["ntau_freezeout"])
        if max_events:
            n_ev = min(n_ev, max_events)
        shared = "arr_bg_rows" in f
        bg_rows = f["arr_bg_rows"][:] if shared else np.arange(n_ev)
        bg_ds = f["arr_bg_store"] if shared else f["arr_bg"]
        diag = {k: f["diag"][k][:] for k in f["diag"]}
        sig = np.atleast_1d(A.get("pthat_bin_sigma_gen", [np.nan])).astype(float)
        tau_delay = float(A["liquefier_tau_delay"])
        e_fo = float(np.interp(float(A.get("T_fo", 0.15)), eos.T, eos.e))
        kflux = KernelFlux(A)
        normalized = int(A.get("liquefier_normalize_on_hydro_grid", -1)) == 1
        flux_ds = f["source/flux"] if "source/flux" in f else None
        tfo = f["tau_freezeout"][:]
        g_off, gP = f["shower/parton_offsets"][:], f["shower/partons"]
        i_off, gI = f["shower/initiator_offsets"][:], f["shower/initiators"]
        d_off, gD = f["source/offsets"][:], f["source/droplets"]
        bg = None
        for ev in range(n_ev):
            ntj, ntb = int(f["ntau_freezeout"][ev]), int(f["ntau_freezeout_bg"][ev])
            row = int(bg_rows[ev])
            if bg is None or bg["key"] != (row, ntb):
                arrB = bg_ds[row, :, :, :, :, :ntb]
                PB, SB = flux_series(arrB, eos, g)
                bg = dict(key=(row, ntb), arr=arrB, P=PB, S=SB,
                          pp=participant_plane(arrB[0, ..., 0], g),
                          e_last=float(arrB[0, ..., ntb - 1].max()))
            arrB = bg["arr"]
            arrJ = f["arr"][ev, :, :, :, :, :ntj]
            PJ, SJ = flux_series(arrJ, eos, g)
            e_last_jet = float(arrJ[0, ..., ntj - 1].max())
            psi2, eps2, xc, yc = bg["pp"]
            live = min(ntj, ntb)

            Q = gP[g_off[ev]:g_off[ev + 1]]
            I = gI[i_off[ev]:i_off[ev + 1]]
            D = gD[d_off[ev]:d_off[ev + 1]]
            lab = assign_droplets(Q, D) if len(D) else np.zeros(0, np.int64)
            eta_d0 = np.where(np.isfinite(D[:, 3]), D[:, 3], 0.0)   # tau_d = 0: eta is 0/0
            tdep = D[:, 0] + tau_delay
            inj = tdep <= tfo[ev]
            # the raw sampled kernel sum: the C++ value if stored, else the Python port
            phi = (flux_ds[d_off[ev]:d_off[ev + 1]] if flux_ds is not None
                   else np.full(len(D), -1.0))
            phi = np.array([p if p >= 0 else kflux(*d[:4]) for p, d in zip(phi, D)])
            # what MUSIC_2 received, as a fraction of the droplet
            fac = (np.where(kflux.on_grid(D[:, 1], D[:, 2], eta_d0), 1.0, 0.0) if normalized
                   else phi)
            winj = np.where(inj, fac, 0.0)

            # background at every droplet
            kd = np.floor((tdep - g.tau0) / g.dtau + 1e-9)
            ok = (kd >= 0) & (kd <= ntb - 1)
            bgd = np.full((4, len(D)), np.nan)
            if ok.any():
                bgd[:, ok] = sample(arrB, g, D[ok, 1], D[ok, 2], eta_d0[ok], kd[ok])
            e_d = bgd[0]
            T_d = np.where(np.isfinite(e_d), eos(np.nan_to_num(e_d))[2], np.nan)
            v_d = np.sqrt(np.sum(bgd[1:] ** 2, axis=0))
            for j in range(len(D)):
                dr_rows.append(dict(ev=ev, shower=int(lab[j]), tau_dep=tdep[j], x=D[j, 1],
                                    y=D[j, 2], eta=D[j, 3], E=D[j, 4], px=D[j, 5], py=D[j, 6],
                                    pz=D[j, 7], injected=bool(inj[j]), kernel_flux=phi[j],
                                    inj_factor=winj[j],
                                    E_inj=winj[j] * D[j, 4], e_bg=e_d[j],
                                    T_bg=T_d[j], v_bg=v_d[j], vx_bg=bgd[1, j],
                                    vy_bg=bgd[2, j], vz_bg=bgd[3, j]))

            # medium along each shower's path, from the background leg
            gs = shower_sums(Q, I)
            ks = np.arange(ntb)
            ks = ks[g.tau(ks) >= TAU_START - 1e-9]
            taus = g.tau(ks)
            leg = []
            for si, ini in enumerate(I):
                px, py, pz, E0 = ini[3:7]
                pT = float(np.hypot(px, py))
                y0 = float(rapidity(E0, pz))
                n3 = np.array([px, py, pz]) / np.linalg.norm([px, py, pz])
                nT = np.array([px, py]) / max(pT, 1e-12)
                x0, y0v = float(ini[7]), float(ini[8])
                xs, ys = x0 + taus * nT[0], y0v + taus * nT[1]
                fld = sample(arrB, g, xs, ys, np.full_like(taus, y0), ks)
                T = eos(fld[0])[2]
                vdotn = fld[1] * n3[0] + fld[2] * n3[1] + fld[3] * n3[2]
                gam = 1.0 / np.sqrt(np.clip(1.0 - np.sum(fld[1:] ** 2, axis=0), 1e-12, None))
                flow = gam * (1.0 - vdotn)
                hot = T >= T_C
                dt = g.dtau
                if hot.any():
                    t_in = taus[hot][0]
                    T3 = T[hot] ** 3
                    path = dict(L=dt * hot.sum(), tau_in=t_in, tau_exit=taus[hot][-1],
                                I_T2=dt * (T[hot] ** 2).sum(), I_T3=dt * T3.sum(),
                                I_T3L=dt * ((taus[hot] - t_in) * T3).sum(),
                                I_T2_flow=dt * (T[hot] ** 2 * flow[hot]).sum(),
                                I_T3_flow=dt * (T3 * flow[hot]).sum(),
                                flow_mean=float(np.average(flow[hot], weights=T3)),
                                vpar_mean=float(np.average(vdotn[hot], weights=T3)),
                                I_e=dt * fld[0][hot].sum(), T_max=float(T[hot].max()))
                else:
                    path = dict(L=0.0, tau_in=np.nan, tau_exit=np.nan, I_T2=0.0, I_T3=0.0,
                                I_T3L=0.0, I_T2_flow=0.0, I_T3_flow=0.0, flow_mean=np.nan,
                                vpar_mean=np.nan, I_e=0.0, T_max=float(T.max()) if len(T) else np.nan)
                path["T_start"] = float(T[0]) if len(T) else np.nan
                r = np.array([x0 - xc, y0v - yc])
                rn = np.linalg.norm(r)
                sid = int(ini[0])
                m = lab == sid
                wpos = np.clip(D[m, 4], 0, None)
                trk = m & inj & (np.abs(D[:, 4]) >= E_MIN_TRACK)
                rec = dict(ev=ev, shower=sid, pid=int(ini[1]), E0=float(E0), pT0=pT, y0=y0,
                           phi0=float(np.arctan2(py, px)), x_vtx=x0, y_vtx=y0v, r_vtx=float(rn),
                           cos_alpha=float(r @ nT / rn) if rn > 0.2 else np.nan,
                           dphi_psi2=float(abs((np.arctan2(py, px) - psi2 + np.pi / 2) % np.pi
                                               - np.pi / 2)),
                           E_dep_drop=float(D[m, 4].sum()),
                           E_dep_drop_pos=float(wpos.sum()),
                           E_dep_drop_neg=float(np.clip(D[m, 4], None, 0).sum()),
                           E_dep_injected=float(D[m & inj, 4].sum()),
                           E_inj_hydro=float((winj[m] * D[m, 4]).sum()),
                           n_drop=int(m.sum()),
                           tau_dep_mean=float(np.average(tdep[m], weights=wpos)) if wpos.sum() > 0 else np.nan,
                           tau_dep_first=float(tdep[trk].min()) if trk.any() else np.nan,
                           tau_dep_last=float(tdep[trk].max()) if trk.any() else np.nan,
                           T_dep_mean=(float(np.average(T_d[m][np.isfinite(T_d[m])],
                                                        weights=wpos[np.isfinite(T_d[m])]))
                                       if (wpos[np.isfinite(T_d[m])]).sum() > 0 else np.nan))
                rec.update(gs[si])
                rec.update(path)
                leg.append(rec)
                sh_rows.append(rec)

            # the leading-deposit leg's source track and its wake in the jet frame
            L = int(np.argmax([r_["E_dep_graph"] for r_ in leg])) if leg else -1
            mp = np.full((2, 2, len(DETA_EDGES) - 1, len(DPHI_EDGES) - 1), np.nan)
            mt = np.full(2, np.nan)
            apex_T = np.full(2, np.nan)
            apex_cs = np.full(2, np.nan)
            if L >= 0:
                sid = int(I[L, 0])
                trk = (lab == sid) & inj & (np.abs(D[:, 4]) >= E_MIN_TRACK)
                o = np.argsort(tdep[trk])
                tr = np.column_stack([tdep[trk][o], D[trk][o][:, 1:4]])
                phiL = float(np.arctan2(I[L, 4], I[L, 3]))
                nL = I[L, 3:6] / np.linalg.norm(I[L, 3:6])
                act = [k for k in range(live) if len(tr) and tr[0, 0] <= g.tau(k) <= tr[-1, 0]]
                frames = [act[int(0.8 * len(act))] if act else -1, live - 1]
                for j, k in enumerate(frames):
                    if k < 0 or not len(tr) or g.tau(k) < tr[0, 0]:
                        continue
                    apex = [float(np.interp(g.tau(k), tr[:, 0], tr[:, c])) for c in (1, 2, 3)]
                    cJ, _ = flux_cells(arrJ[..., k], g.tau(k), eos, g)
                    cB, _ = flux_cells(arrB[..., k], g.tau(k), eos, g)
                    dc = cJ - cB
                    dphi = np.mod(np.arctan2(g.Y - apex[1], g.X - apex[0])[:, :, None] - phiL,
                                  2 * np.pi)
                    dphi = np.broadcast_to(dphi, dc.shape[1:]).ravel()
                    deta = np.broadcast_to((g.eta - apex[2])[None, None, :], dc.shape[1:]).ravel()
                    wpar = np.tensordot(nL, dc[1:], axes=1).ravel()
                    for c, wt in enumerate((dc[0].ravel(), wpar)):
                        mp[j, c] = np.histogram2d(deta, dphi, bins=[DETA_EDGES, DPHI_EDGES],
                                                  weights=wt)[0]
                    mt[j] = g.tau(k)
                    eb = float(sample(arrB[:1], g, [apex[0]], [apex[1]], [apex[2]], [k])[0, 0])
                    apex_T[j], apex_cs[j] = float(eos(eb)[2]), float(eos.cs(eb))

            # cumulative injected droplet four-momentum at each frame
            Dcum, Dhyd = np.full((NT, 4), np.nan), np.full((NT, 4), np.nan)
            for k in range(NT):
                sel = inj & (tdep <= g.tau(k))
                Dcum[k] = D[sel][:, 4:8].sum(axis=0)
                Dhyd[k] = (winj[sel, None] * D[sel][:, 4:8]).sum(axis=0)
            pad = lambda a, n: np.concatenate([a, np.full((NT - n,) + a.shape[1:], np.nan)])
            evo.append(dict(P_jet=pad(PJ, ntj), P_bg=pad(bg["P"], ntb), S_jet=pad(SJ, ntj),
                            S_bg=pad(bg["S"], ntb), D_inj=Dcum, D_hydro=Dhyd))
            maps.append(dict(maps=mp, tau=mt, apex_T=apex_T, apex_cs=apex_cs))

            pb = int(diag["pthat_bin"][ev]) if "pthat_bin" in diag else 0
            ev_rows.append(dict(
                ev=ev, bg_row=row, pthat=float(diag.get("pthat", [np.nan] * (ev + 1))[ev]),
                pthat_bin=pb, sigma_bin=float(sig[pb]) if pb < len(sig) else np.nan,
                event_weight=float(diag.get("event_weight", np.ones(ev + 1))[ev]),
                n_showers=len(I), lead_leg=L, E_ini=float(I[:, 6].sum()),
                E_surv=float(sum(r_["E_surv"] for r_ in leg)),
                E_dep_graph=float(sum(r_["E_dep_graph"] for r_ in leg)),
                E_drop=float(D[:, 4].sum()), E_drop_injected=float(D[inj, 4].sum()),
                E_inj_hydro=float((winj * D[:, 4]).sum()),
                E_drop_unassigned=float(D[lab < 0, 4].sum()), n_drop=len(D),
                n_drop_unassigned=int((lab < 0).sum()),
                tau_fo_jet=float(tfo[ev]), tau_fo_bg=float(f["tau_freezeout_bg"][ev]),
                ntau_jet=ntj, ntau_bg=ntb, psi2=psi2, eps2=eps2, x_c=xc, y_c=yc,
                # patched with fix_lbt_double_counting.py, or made by an X-SCAPE with the fix
                # (every build that records the kernel normalization has it, X-SCAPE #154)
                double_count_fix=bool("lbt_double_count_fix" in A or
                                      int(A.get("liquefier_normalize_on_hydro_grid", -1)) >= 0),
                kernel_normalized=normalized,
                # MUSIC stops a leg once max e < e_fo; a leg whose last frame is still hot
                # never froze out (MUSIC ran to its maximum time, or broke): its evolution,
                # the wake and the freeze-out delay are not usable
                e_last_jet=e_last_jet, e_last_bg=bg["e_last"],
                jet_no_freezeout=bool(e_last_jet > 1.5 * e_fo),
                bg_no_freezeout=bool(bg["e_last"] > 1.5 * e_fo)))
    return fi, ev_rows, sh_rows, dr_rows, evo, maps, time.time() - t0, NT, g.tau0, g.dtau


# ── driver ──────────────────────────────────────────────────────────────────────
COLUMN_DOC = {
    "events": "one row per jet event; file = index into /files, ev = event in that file",
    "showers": "one row per shower; event = row in /events",
    "droplets": "one row per droplet; event = row in /events, shower_row = row in /showers "
                "(-1: not assigned)",
}


def is_pair_file(path):
    try:
        with h5py.File(path, "r") as f:
            return (f.attrs.get("format") == "xscape/hydro_evolution" and "arr_bg" in f
                    and "shower" in f)
    except OSError:
        return False


def pair_files(paths, exclude=()):
    """Pair files with a shower graph (directories are searched, other .h5 files skipped)."""
    out = []
    skip = {os.path.realpath(p) for p in exclude}
    for a in paths:
        cands = sorted(glob.glob(os.path.join(a, "*.h5"))) if os.path.isdir(a) else [a]
        out += [p for p in cands
                if os.path.realpath(p) not in skip
                and not re.search(r"_(particlize|hadrons_\w+)\.h5$", p) and is_pair_file(p)]
    return out


def write_table(g, rows, extra):
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {}
    cols.update(extra)
    for k, v in cols.items():
        g.create_dataset(k, data=v if v.dtype != object else v.astype("S"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="production directories or pair .h5 files")
    ap.add_argument("-o", "--out", default="wake_observables.h5")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="files processed in parallel")
    ap.add_argument("--eos", default="", help="hotQCD EoS table (default: from prod_build)")
    ap.add_argument("--max-events", type=int, default=0, help="per file, for tests")
    args = ap.parse_args(argv)

    files = pair_files(args.paths, exclude=[args.out])
    if not files:
        sys.exit("no pair files found")
    with h5py.File(files[0], "r") as f:
        eos_path = find_eos(f.attrs, args.eos)
    jobs = [(i, p, eos_path, args.max_events) for i, p in enumerate(files)]
    results = [None] * len(files)
    t0 = time.time()
    with Pool(max(1, min(args.jobs, len(files)))) as pool:
        for r in pool.imap_unordered(process_file, jobs):
            results[r[0]] = r
            print(f"[{sum(x is not None for x in results)}/{len(files)}] "
                  f"{os.path.basename(files[r[0]])}: {len(r[1])} events, {r[6]:.0f} s",
                  flush=True)

    NT = max(r[7] for r in results)
    if len({(round(r[8], 6), round(r[9], 6)) for r in results}) != 1:
        sys.exit("files have different tau grids; process them separately")
    ev_rows, sh_rows, dr_rows, evo, maps = [], [], [], [], []
    ev_file, sh_event, dr_event, dr_shrow = [], [], [], []
    for fi, er, sr, dr, eo, mp, *_ in results:
        base = len(ev_rows)
        sh_base = len(sh_rows)
        key = {}
        for r in sr:
            key[(r["ev"], r["shower"])] = len(sh_rows)
            sh_event.append(base + r["ev"])
            sh_rows.append(r)
        for r in er:
            # lead_leg counts the event's showers in initiator order, as sr lists them
            legs = [i for i in range(sh_base, len(sh_rows)) if sh_event[i] == base + r["ev"]]
            r["lead_leg_row"] = legs[r["lead_leg"]] if r["lead_leg"] >= 0 else -1
            ev_file.append(fi)
            ev_rows.append(r)
        for r in dr:
            dr_event.append(base + r["ev"])
            dr_shrow.append(key.get((r["ev"], r["shower"]), -1) if r["shower"] >= 0 else -1)
            dr_rows.append(r)
        for e_ in eo:
            evo.append({k: np.concatenate([v, np.full((NT - len(v),) + v.shape[1:], np.nan)])
                        for k, v in e_.items()})
        maps += mp

    with h5py.File(args.out, "w") as out:
        out.attrs.update(dict(
            format=FORMAT, version=VERSION, created=time.strftime("%Y-%m-%d %H:%M:%S"),
            eos_table=eos_path, T_C=T_C, tau_start=TAU_START, E_min_track=E_MIN_TRACK,
            R_cone=R_CONE, tau_min=results[0][8], dtau=results[0][9], ntau=NT,
            deta_edges=DETA_EDGES, dphi_edges=DPHI_EDGES,
            doc=__doc__))
        out.create_dataset("files", data=np.array(files, dtype="S"))
        write_table(out.create_group("events"), ev_rows,
                    {"file": np.array(ev_file, np.int64)})
        write_table(out.create_group("showers"), sh_rows,
                    {"event": np.array(sh_event, np.int64)})
        write_table(out.create_group("droplets"), dr_rows,
                    {"event": np.array(dr_event, np.int64),
                     "shower_row": np.array(dr_shrow, np.int64)})
        for k, doc in COLUMN_DOC.items():
            out[k].attrs["doc"] = doc
        ge = out.create_group("evolution")
        for k in evo[0]:
            ge.create_dataset(k, data=np.stack([e_[k] for e_ in evo]), compression="gzip")
        ge.attrs["doc"] = ("per event x tau frame: P_jet, P_bg (P^mu = E, px, py, pz, ideal "
                           "fluid, inside the output window), S_jet, S_bg (entropy), D_inj "
                           "(cumulative injected droplet four-momentum), D_hydro (the same "
                           "weighted with each droplet's kernel_flux: what MUSIC_2 received); "
                           "NaN after freeze-out")
        gm = out.create_group("jetframe")
        gm.create_dataset("maps", data=np.stack([m["maps"] for m in maps]), compression="gzip")
        for k in ("tau", "apex_T", "apex_cs"):
            gm.create_dataset(k, data=np.stack([m[k] for m in maps]))
        gm.attrs["doc"] = ("maps (event, frame, quantity, d_eta, d_phi): frame 0 = while the "
                           "leading-deposit leg still deposits (80% into its deposit window), "
                           "1 = last frame both legs are live; quantity 0 = dP^0, 1 = dP.n "
                           "(momentum along the leg's initial direction), GeV per bin; about "
                           "the leg's source position at that tau, d_phi = 0 ahead. tau, "
                           "apex_T, apex_cs: per frame, background T and c_s at the source")

    n_un = sum(r["n_drop_unassigned"] for r in ev_rows)
    print(f"\n{len(ev_rows)} events, {len(sh_rows)} showers, {len(dr_rows)} droplets "
          f"({n_un} not assigned to a shower) -> {args.out}  [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
