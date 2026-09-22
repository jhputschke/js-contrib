"""
mc_glauber_tilted.py -- 3D initial conditions from MC-Glauber participants with the
tilted-source ansatz of Bozek and Wyskiel (arXiv:1002.4999), plus the equations of
state needed to turn them into hydro input.

    rho(x, y, eta) = K * H(eta) * { (1 - alpha) [T+(x,y) f+(eta) + T-(x,y) f-(eta)]
                                    + alpha * Tcoll(x,y) }

    f+-(eta) = clip( (1 +- eta/eta_m) / 2, 0, 1 )
    H(eta)   = exp( -(|eta| - eta0)^2 / (2 sig_eta^2) )  for |eta| > eta0, else 1

T+ (T-) is the thickness of participants moving towards +eta (-eta); Tcoll is the
density of binary collisions.  `rho` is read either as the energy density
(quantity="energy") or, as in the paper, as the entropy density
(quantity="entropy"), which is then converted to e with the EoS.

The longitudinal structure is a PARAMETRIZATION: no longitudinal dynamics, no baryon
stopping, no initial flow.  `string_fluct=True` replaces the smooth f+- by one random
end point per participant (Broniowski and Bozek, arXiv:1509.04124); the event average
is unchanged, single events fluctuate in rapidity.

Equations of state (mu_B = 0, which is all the FV solver can use):

    IdealGasEoS          MUSIC EOS_to_use 0:  p = e/3, dof = 2(Nc^2-1) + 7/2 Nc Nf = 42.25
    TableEoS             any 1D table; readers for
        .from_music_binary   MUSIC EOS_to_use 9 / 91 (hotQCD): float64 rows (e, P, s, T)
        .from_music_check    MUSIC's own dump  check_EoS_<id>_PST.dat
    get_eos(0 | 9 | 91, path)      convenience constructor
    to_fv_eos(eos, fv)             matching ConformalEoS / TabulatedEoS for the FV solver

Stage 1 (this file): initial states on disk.  Stage 2 (separate, needs PyTorch): evolve them.

    python mc_glauber_tilted.py                      # self-check
    python mc_glauber_tilted.py check-eos FILE       # inspect a MUSIC EoS table
    python mc_glauber_tilted.py generate ic.h5 -n 200 --eos 91 --eos-path MUSIC/EOS/hotQCD \
           --target-T 0.36 --b 0 8 --string-fluct --gamma-k 2

    save_events(path, n, ...)     HDF5: e0 (N, X, Y, eta) float32 at tau0, per-event b / npart /
                                  ncoll / e_max / T_max / seed, the EoS itself, every parameter
    load_events(path, index)      -> (e0, meta)
    load_eos(path)                -> the stored EoS, bit-identical to the one used for writing
    regenerate_event(path, i)     -> event i recomputed from its stored seed

Hand-over to the FV solver in stage 2:

    e0, m = load_events("ic.h5", slice(0, 8));  eos = to_fv_eos(load_eos("ic.h5"), fv)
    g  = fv.Grid(m["nx"], m["ny"], m["neta"], m["dx"], m["dy"], m["deta"], device, dtype)
    q0 = fv.initial_state_from_energy(torch.as_tensor(e0, device=device), m["tau0"], eos)

numpy only (h5py is imported only by the functions that touch files).  Grid convention matches
structure_preserving_hydro_fv.Grid: cell centres, symmetric about zero, array layout
(X, Y, Z=eta).  Units: GeV, fm.
"""
from __future__ import annotations

import os

import numpy as np

HBARC = 0.19733          # GeV fm

# ====================================================================== EoS

class IdealGasEoS:
    """MUSIC `EOS_to_use 0`.  e = c T^4 with c = (pi^2/30) dof / hbarc^3, p = e/3.
    MUSIC uses Nc = 3, Nf = 2.5, i.e. dof = 42.25 (the FV ConformalEoS default is 47.5)."""

    name = "ideal"

    def __init__(self, dof=42.25):
        self.dof = float(dof)
        self.c = np.pi ** 2 / 30.0 * self.dof / HBARC ** 3            # GeV^-3 fm^-3

    def p(self, e):
        return np.asarray(e) / 3.0

    def cs2(self, e):
        return np.full_like(np.asarray(e, dtype=float), 1.0 / 3.0)

    def T(self, e):
        return (np.asarray(e) / self.c) ** 0.25

    def s(self, e):
        e = np.asarray(e)
        return 4.0 * e / (3.0 * self.T(e))

    def e_from_T(self, T):
        return self.c * np.asarray(T) ** 4

    def e_from_s(self, s):
        return (3.0 * np.asarray(s) / (4.0 * self.c ** 0.25)) ** (4.0 / 3.0)


class TableEoS:
    """1D table at mu_B = 0, resampled on a log-spaced e grid.

    p and T are interpolated linearly in log(e), exactly as the FV `TabulatedEoS` does,
    so both codes see the same p(e).  s = (e + p)/T (thermodynamic identity at mu_B = 0).

    Below the first table point the table is extended down to `e_floor` with
    p = (p0/e0) e and T = T0 (e/e0)^(1/4).  This is crude, but it keeps p < e in vacuum
    cells; without it an interpolator that holds the first value would return p > e.
    """

    def __init__(self, e, p, T, *, s_file=None, n_points=1500, e_floor=1e-12, name="table"):
        e, p, T = (np.asarray(a, dtype=float) for a in (e, p, T))
        keep = (e > 0) & (p > 0) & (T > 0)
        e, p, T = e[keep], p[keep], T[keep]
        order = np.argsort(e)
        e, p, T = e[order], p[order], T[order]
        if np.any(np.diff(e) <= 0):
            raise ValueError("energy density column must be strictly increasing")
        self.name = name
        self.e_raw_range = (e[0], e[-1])
        # thermodynamic consistency of the file's own entropy column, if it has one
        self.s_file_mismatch = None
        if s_file is not None:
            s_f = np.asarray(s_file, dtype=float)[keep][order]
            ok = s_f > 0
            self.s_file_mismatch = float(np.max(np.abs(s_f[ok] * T[ok] / (e[ok] + p[ok]) - 1.0)))

        eg = np.geomspace(e[0], e[-1], n_points)
        pg, Tg = np.interp(eg, e, p), np.interp(eg, e, T)
        lo = np.geomspace(e_floor, eg[0], 60, endpoint=False)
        self._finalise(np.concatenate([lo, eg]), np.concatenate([pg[0] / eg[0] * lo, pg]),
                       np.concatenate([Tg[0] * (lo / eg[0]) ** 0.25, Tg]), n_ext=len(lo))

    def _finalise(self, e_tab, p_tab, T_tab, n_ext):
        self.e_tab, self.p_tab, self.T_tab, self.n_ext = e_tab, p_tab, T_tab, int(n_ext)
        self.s_tab = (self.e_tab + self.p_tab) / self.T_tab
        self._loge = np.log(self.e_tab)
        self.cs2_tab = np.clip(np.gradient(self.p_tab, self.e_tab), 0.01, 1.0 / 3.0)   # MUSIC's clamp
        for col, label in ((self.s_tab, "entropy density"), (self.T_tab, "temperature")):
            if np.any(np.diff(col) <= 0):
                raise ValueError(f"{label} is not monotone in e; cannot invert the table")

    # ---- readers ---------------------------------------------------------------
    @classmethod
    def from_tables(cls, e_tab, p_tab, T_tab, *, n_ext=60, name="table", e_raw_range=None):
        """Rebuild from tables that were already processed (as stored by `save_events`).
        No resampling, so p(e) and T(e) are bit-identical to the original object."""
        self = cls.__new__(cls)
        self.name, self.s_file_mismatch = name, None
        e_tab = np.asarray(e_tab, dtype=float)
        self.e_raw_range = tuple(e_raw_range) if e_raw_range is not None else (e_tab[n_ext], e_tab[-1])
        self._finalise(e_tab, np.asarray(p_tab, dtype=float), np.asarray(T_tab, dtype=float), n_ext)
        return self

    @classmethod
    def from_music_binary(cls, path, **kw):
        """MUSIC hotQCD format (EOS 9 / 91): native float64, rows of (e, P, s, T) with
        e, P in GeV/fm^3, s in 1/fm^3, T in GeV (see MUSIC src/eos_hotQCD.cpp)."""
        tab = np.fromfile(path, dtype="<f8")
        if tab.size == 0 or tab.size % 4:
            raise ValueError(f"{path}: size is not a multiple of 4 doubles")
        tab = tab.reshape(-1, 4)
        kw.setdefault("name", os.path.basename(path))
        return cls(tab[:, 0], tab[:, 1], tab[:, 3], s_file=tab[:, 2], **kw)

    @classmethod
    def from_music_check(cls, path, **kw):
        """MUSIC's own dump `check_EoS_<id>_PST.dat`: columns e, P, s, T, cs^2.  This is
        the safest import, because it is whatever EoS the MUSIC run actually used."""
        tab = np.loadtxt(path, comments="#")
        kw.setdefault("name", os.path.basename(path))
        return cls(tab[:, 0], tab[:, 1], tab[:, 3], s_file=tab[:, 2], **kw)

    # ---- thermodynamics ----------------------------------------------------------
    def _interp(self, e, col):
        return np.interp(np.log(np.clip(e, self.e_tab[0], None)), self._loge, col)

    def p(self, e):
        return self._interp(e, self.p_tab)

    def T(self, e):
        return self._interp(e, self.T_tab)

    def cs2(self, e):
        return self._interp(e, self.cs2_tab)

    def s(self, e):
        e = np.asarray(e, dtype=float)
        return (e + self.p(e)) / self.T(e)

    def e_from_s(self, s):
        return np.exp(np.interp(np.log(np.clip(s, self.s_tab[0], None)), np.log(self.s_tab), self._loge))

    def e_from_T(self, T):
        return np.exp(np.interp(np.log(np.clip(T, self.T_tab[0], None)), np.log(self.T_tab), self._loge))

    def fv_tables(self):
        """(e_tab, p_tab, T_tab) for structure_preserving_hydro_fv.TabulatedEoS."""
        return self.e_tab, self.p_tab, self.T_tab

    def report(self):
        i = int(np.argmin(self.cs2_tab[self.n_ext:])) + self.n_ext
        lines = [f"EoS table '{self.name}':  e in [{self.e_raw_range[0]:.3e}, {self.e_raw_range[1]:.3e}] GeV/fm^3",
                 "  T(e = 0.2, 1, 10 GeV/fm^3) = " + ", ".join(f"{self.T(v):.4f}" for v in (0.2, 1.0, 10.0)) + " GeV",
                 f"  min cs^2 = {self.cs2_tab[i]:.3f} at T = {self.T_tab[i]:.3f} GeV;   cs^2(e = 100) = {self.cs2(100.0):.3f}",
                 f"  max p/e over the table = {np.max(self.p_tab / self.e_tab):.3f}  (must stay below 1)"]
        if self.s_file_mismatch is not None:
            lines.append(f"  file's s column vs (e+p)/T: max relative difference = {self.s_file_mismatch:.2e}")
        return "\n".join(lines)


_MUSIC_FILES = {9: "hrg_hotqcd_eos_binary.dat", 91: "hrg_hotqcd_eos_SMASH_binary.dat"}


def get_eos(which=0, path=None, **kw):
    """which = 0 | "ideal"                 -> IdealGasEoS(dof=42.25)      (MUSIC EOS 0)
       which = 9 | 91 | "hotqcd" | "hotqcd_smash" -> TableEoS from MUSIC's binary file.
       `path` is the file, or the directory that holds it (MUSIC's EOS/hotQCD)."""
    key = {"ideal": 0, "hotqcd": 9, "hotqcd_smash": 91}.get(which, which)
    if key == 0:
        return IdealGasEoS(**kw)
    if key in _MUSIC_FILES:
        if path is None:
            raise ValueError("hotQCD needs `path` (file or MUSIC's EOS/hotQCD directory); "
                             "fetch the table with MUSIC's EOS/download_hotQCD.sh")
        if os.path.isdir(path):
            path = os.path.join(path, _MUSIC_FILES[key])
        return TableEoS.from_music_binary(path, **kw)
    raise ValueError(f"unknown EoS '{which}'")


def to_fv_eos(eos, fv, device=None, dtype=None):
    """The same EoS as an object of the FV solver module `fv`.

    `device`/`dtype` are forwarded to TabulatedEoS so its tables are materialised once on the
    solver's device.  This is required on MPS, which has no float64 and therefore cannot hold
    the default tables at all."""
    if isinstance(eos, IdealGasEoS):
        return fv.ConformalEoS(dof=eos.dof)
    kw = {}
    if device is not None:
        kw["device"] = device
    if dtype is not None:
        kw["dtype"] = dtype
    return fv.TabulatedEoS(*eos.fv_tables(), **kw)


# ====================================================================== nuclei

NUCLEI = {            # A, Woods-Saxon radius R [fm], diffuseness a [fm]
    "Au": (197, 6.38, 0.535),
    "Pb": (208, 6.62, 0.546),
    "Cu": (63, 4.20, 0.596),
}
HULTHEN = (0.228, 1.18)      # deuteron: a, b [1/fm]


def sample_woods_saxon(A, R, a, rng, d_min=0.4):
    """Nucleon positions (A, 3).  Rejection sampling of r^2 rho(r); a nucleon closer
    than d_min to an accepted one is redrawn.  Re-centred on the centre of mass."""
    pos = np.empty((A, 3))
    r_max = R + 10.0 * a
    n = 0
    while n < A:
        r = r_max * rng.random()
        if rng.random() * r_max ** 2 > r ** 2 / (1.0 + np.exp((r - R) / a)):
            continue
        ct, ph = 2.0 * rng.random() - 1.0, 2.0 * np.pi * rng.random()
        st = np.sqrt(1.0 - ct ** 2)
        p = r * np.array([st * np.cos(ph), st * np.sin(ph), ct])
        if n and d_min > 0 and np.min(np.sum((pos[:n] - p) ** 2, axis=1)) < d_min ** 2:
            continue
        pos[n] = p
        n += 1
    return pos - pos.mean(axis=0)


def sample_deuteron(rng):
    """Proton and neutron at +-r/2, r from the Hulthen form P(r) ~ (e^-ar - e^-br)^2."""
    a, b = HULTHEN
    r_grid = np.linspace(1e-3, 20.0, 2000)
    p_max = np.max((np.exp(-a * r_grid) - np.exp(-b * r_grid)) ** 2)
    while True:
        r = 20.0 * rng.random()
        if rng.random() * p_max < (np.exp(-a * r) - np.exp(-b * r)) ** 2:
            break
    ct, ph = 2.0 * rng.random() - 1.0, 2.0 * np.pi * rng.random()
    st = np.sqrt(1.0 - ct ** 2)
    half = 0.5 * r * np.array([st * np.cos(ph), st * np.sin(ph), ct])
    return np.stack([half, -half])


def sample_nucleus(name, rng, d_min=0.4):
    if name == "d":
        return sample_deuteron(rng)
    if name == "p":
        return np.zeros((1, 3))
    A, R, a = NUCLEI[name]
    return sample_woods_saxon(A, R, a, rng, d_min)


# ====================================================================== collision

def glauber_event(proj, targ, b, sigma_nn_fm2, rng, d_min=0.4):
    """One event.  Projectile moves towards +eta and sits at x = +b/2.
    Black-disk criterion: collide if transverse distance^2 < sigma_nn / pi.
    Returns transverse positions of participants (+ and -) and of binary collisions."""
    A = sample_nucleus(proj, rng, d_min)[:, :2] + np.array([+0.5 * b, 0.0])
    B = sample_nucleus(targ, rng, d_min)[:, :2] + np.array([-0.5 * b, 0.0])
    d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)          # (nA, nB)
    hit = d2 < sigma_nn_fm2 / np.pi
    ia, ib = np.nonzero(hit)
    return {"plus": A[hit.any(axis=1)], "minus": B[hit.any(axis=0)],
            "coll": 0.5 * (A[ia] + B[ib]), "npart": int(hit.any(1).sum() + hit.any(0).sum()),
            "ncoll": int(hit.sum()), "b": float(b)}


# ====================================================================== profiles

def centres(n, d):
    return (np.arange(n) - 0.5 * (n - 1)) * d


def _gauss_1d(grid, x0, w):
    """(N, n) normalised Gaussians: sum over the grid times spacing ~ 1."""
    return np.exp(-(grid[None, :] - x0[:, None]) ** 2 / (2.0 * w ** 2)) / (np.sqrt(2.0 * np.pi) * w)


def H_envelope(eta, eta0, sig_eta):
    return np.exp(-np.clip(np.abs(eta) - eta0, 0.0, None) ** 2 / (2.0 * sig_eta ** 2))


def f_tilt(eta, eta_m, sign):
    return np.clip(0.5 * (1.0 + sign * eta / eta_m), 0.0, 1.0)


def tilted_profile(ev, x, y, eta, *, w=0.4, alpha=0.145, eta0=1.5, sig_eta=1.3, eta_m=3.36,
                   gamma_k=None, string_fluct=False, edge=0.3, rng=None):
    """The bracket of the ansatz times H(eta), for K = 1.  Shape (X, Y, Z), units fm^-2.

    gamma_k      : if set, each participant gets a Gamma(k, 1/k) weight (mean 1),
                   the multiplicity fluctuation used in TRENTo.
    string_fluct : one random end point per participant instead of the smooth f+-.
                   `edge` is the width [units of eta] of the smoothed step.
    """
    rng = np.random.default_rng() if rng is None else rng
    rho = np.zeros((x.size, y.size, eta.size))
    for key, sign in (("plus", +1.0), ("minus", -1.0)):
        p = ev[key]
        if len(p) == 0:
            continue
        wgt = rng.gamma(gamma_k, 1.0 / gamma_k, len(p)) if gamma_k else np.ones(len(p))
        gx, gy = _gauss_1d(x, p[:, 0], w), _gauss_1d(y, p[:, 1], w)
        if string_fluct:
            y_end = rng.uniform(-eta_m, eta_m, len(p))               # P(eta beyond end) = f+-(eta)
            h = 0.5 * (1.0 + np.tanh(sign * (eta[None, :] - y_end[:, None]) / edge))
        else:
            h = np.broadcast_to(f_tilt(eta, eta_m, sign), (len(p), eta.size))
        rho += (1.0 - alpha) * np.einsum("i,ix,iy,iz->xyz", wgt, gx, gy, h, optimize=True)
    if alpha > 0 and len(ev["coll"]):
        c = ev["coll"]
        Tc = np.einsum("ix,iy->xy", _gauss_1d(x, c[:, 0], w), _gauss_1d(y, c[:, 1], w))
        rho += alpha * Tc[:, :, None]
    return rho * H_envelope(eta, eta0, sig_eta)[None, None, :]


def energy_density_3d(ev, x, y, eta, *, K=1.0, quantity="energy", eos=None, e_floor=1e-6, **profile_kw):
    """e(x, y, eta) in GeV/fm^3, shape (X, Y, Z).

    quantity="energy"  : e = K * profile            (K in GeV/fm)
    quantity="entropy" : s = K * profile, e = eos.e_from_s(s)   (K in 1/fm; needs `eos`)
    """
    rho = K * tilted_profile(ev, x, y, eta, **profile_kw)
    if quantity == "energy":
        e = rho
    elif quantity == "entropy":
        if eos is None:
            raise ValueError('quantity="entropy" needs an EoS to convert s to e')
        e = np.where(rho > 0, eos.e_from_s(np.clip(rho, 1e-300, None)), 0.0)
    else:
        raise ValueError('quantity must be "energy" or "entropy"')
    return e + e_floor


def plot_initial_state(e, x, y, eta, *, title=None, out=None, threshold=0.02, show=None):
    """Quick-look visualisation of one initial state: a 3D scatter of the whole
    (x, y, eta) energy/entropy-density field, plus the x-y slice at eta = 0.

    e            : (X, Y, Z) density, as returned by energy_density_3d / make_event
    x, y, eta    : grid centres (fm, fm, rapidity), as returned by centres()
    threshold    : keep points with e > threshold * e.max() in the 3D scatter
    out          : if given, save the figure to this path instead of showing it
    show         : call plt.show() (default: only when `out` is not given and we are not in a
                   notebook, where the returned figure is displayed automatically anyway)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)

    fig = plt.figure(figsize=(12, 5.5))
    if title:
        fig.suptitle(title)

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    X, Y, ETA = np.meshgrid(x, y, eta, indexing="ij")
    mask = e > threshold * e.max()
    sc = ax3d.scatter(X[mask], Y[mask], ETA[mask], c=e[mask], cmap="inferno",
                       s=4, alpha=0.4, linewidths=0)
    ax3d.set_xlabel("x [fm]")
    ax3d.set_ylabel("y [fm]")
    ax3d.set_zlabel(r"$\eta$")
    ax3d.set_title("3D energy density")
    fig.colorbar(sc, ax=ax3d, shrink=0.6, pad=0.1, label=r"$e$ [GeV/fm$^3$]")

    ax2d = fig.add_subplot(1, 2, 2)
    ieta0 = int(np.argmin(np.abs(eta)))
    im = ax2d.pcolormesh(x, y, e[:, :, ieta0].T, cmap="inferno", shading="auto")
    ax2d.set_xlabel("x [fm]")
    ax2d.set_ylabel("y [fm]")
    ax2d.set_aspect("equal")
    ax2d.set_title(rf"x-y slice at $\eta$ = {eta[ieta0]:.2f}")
    fig.colorbar(im, ax=ax2d, label=r"$e$ [GeV/fm$^3$]")

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return fig
    if show is None:
        # In a notebook the returned figure is rendered by the display hook, so calling show()
        # as well draws it twice; under a non-interactive backend it merely warns.
        show = "inline" not in plt.get_backend().lower() and plt.isinteractive()
    if show:
        plt.show()
    return fig


def calibrate_K(target, kind="e", *, quantity="energy", eos=None, proj="Au", targ="Au", b=0.0,
                sigma_nn_mb=42.0, n_events=20, seed=0, **profile_kw):
    """K such that the event-averaged profile at x = y = 0, eta = 0 reaches the target.

    kind = "e" [GeV/fm^3], "T" [GeV] or "s" [1/fm^3]; the EoS converts the target to the
    quantity the ansatz describes.  Uses smooth, unweighted profiles at fixed b."""
    rng = np.random.default_rng(seed)
    if kind != ("e" if quantity == "energy" else "s") and eos is None:
        raise ValueError("this combination of `kind` and `quantity` needs an EoS")
    e_t = {"e": lambda v: v, "T": lambda v: eos.e_from_T(v), "s": lambda v: eos.e_from_s(v)}[kind](target)
    goal = float(e_t) if quantity == "energy" else float(target if kind == "s" else eos.s(e_t))
    o = np.zeros(1)
    kw = {k: v for k, v in profile_kw.items() if k not in ("gamma_k", "string_fluct", "edge")}
    centre = [tilted_profile(glauber_event(proj, targ, b, 0.1 * sigma_nn_mb, rng), o, o, o, **kw)[0, 0, 0]
              for _ in range(n_events)]
    return goal / float(np.mean(centre))


def make_event(proj="Au", targ="Au", *, b=None, b_max=None, sigma_nn_mb=42.0, grid=(64, 64, 32),
               spacing=(0.3125, 0.3125, 0.3125), seed=None, require_collision=True, eos=None,
               return_event=False, **kw):
    """Sample one event and return (e, meta), or (e, meta, ev) when return_event=True.
    b = None: P(b) ~ b on [0, b_max];  b = (lo, hi): P(b) ~ b on [lo, hi];  b = number: fixed.
    With an EoS, meta also carries the peak temperature.
    `ev` is the raw Glauber dict, whose "coll" (N_coll positions) allows jet production points
    to be sampled from the binary-collision density instead of uniformly."""
    rng = np.random.default_rng(seed)
    if b_max is None:
        radius = lambda n: NUCLEI[n][1] if n in NUCLEI else 2.0
        b_max = radius(proj) + radius(targ) + 3.0
    x, y, eta = (centres(n, d) for n, d in zip(grid, spacing))
    while True:
        if b is None:
            bb = b_max * np.sqrt(rng.random())
        elif np.ndim(b) == 1:                                   # (b_lo, b_hi), still P(b) ~ b
            bb = np.sqrt(b[0] ** 2 + rng.random() * (b[1] ** 2 - b[0] ** 2))
        else:
            bb = float(b)
        ev = glauber_event(proj, targ, bb, 0.1 * sigma_nn_mb, rng)      # 1 mb = 0.1 fm^2
        if ev["ncoll"] > 0 or not require_collision:
            break
    e = energy_density_3d(ev, x, y, eta, rng=rng, eos=eos, **kw)
    meta = {k: ev[k] for k in ("npart", "ncoll", "b")}
    meta["e_max"] = float(e.max())
    if eos is not None:
        meta["T_max"] = float(eos.T(e.max()))
        meta["eos"] = eos.name
    return (e, meta, ev) if return_event else (e, meta)


# ====================================================================== stage 1: events on disk

FORMAT = "mc_glauber_tilted/initial_state"
FORMAT_VERSION = 1
_PROFILE_DEFAULTS = dict(w=0.4, alpha=0.145, eta0=1.5, sig_eta=1.3, eta_m=3.36,
                         gamma_k=None, string_fluct=False, edge=0.3)


def _eos_id(eos):
    if isinstance(eos, IdealGasEoS):
        return 0
    return {v: k for k, v in _MUSIC_FILES.items()}.get(eos.name, -1)     # -1: some other table


def save_events(path, n_events, proj="Au", targ="Au", *, eos, K, quantity="energy", tau0=0.6, b=None,
                b_max=None, sigma_nn_mb=42.0, grid=(64, 64, 32), spacing=(0.3125, 0.3125, 0.3125),
                seed=0, e_floor=1e-6, overwrite=False, calibration=None, progress=True, **profile_kw):
    """Stage 1: write initial states to HDF5.  Nothing here needs PyTorch.

    Layout
        e0      (N, X, Y, eta) float32   energy density at tau0 [GeV/fm^3]
        b, npart, ncoll, e_max, T_max, seed   per event; `seed` regenerates the event exactly
        eos/    e_tab, p_tab, T_tab (table) or attribute dof (ideal gas)
        attrs   grid, tau0, system, K, quantity, every ansatz parameter, master seed

    The EoS is stored INSIDE the file.  Stage 2 rebuilds it with `load_eos`, so the s -> e
    conversion used here and the hydro evolution cannot end up with different equations of state.
    """
    import h5py

    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    unknown = set(profile_kw) - set(_PROFILE_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown profile parameters: {sorted(unknown)}")
    prof = {**_PROFILE_DEFAULTS, **profile_kw}
    seeds = np.random.SeedSequence(seed).generate_state(n_events, dtype=np.uint64)
    nx, ny, nz = grid
    x, y, eta = (centres(n, d) for n, d in zip(grid, spacing))

    with h5py.File(path, "w") as f:
        a = f.attrs
        a["format"], a["format_version"] = FORMAT, FORMAT_VERSION
        a["layout"] = "(N, X, Y, eta); cell centres, symmetric about zero"
        a["units"] = "e0: GeV/fm^3, lengths: fm, T: GeV"
        a["proj"], a["targ"], a["sigma_nn_mb"] = proj, targ, float(sigma_nn_mb)
        a["tau0"], a["quantity"], a["K"], a["e_floor"] = float(tau0), quantity, float(K), float(e_floor)
        a["nx"], a["ny"], a["neta"] = nx, ny, nz
        a["dx"], a["dy"], a["deta"] = (float(d) for d in spacing)
        a["x_min"], a["y_min"], a["eta_min"] = float(x[0]), float(y[0]), float(eta[0])
        a["master_seed"] = int(seed)
        a["b_mode"] = "min-bias" if b is None else ("range" if np.ndim(b) == 1 else "fixed")
        a["b_arg"] = np.atleast_1d(np.nan if b is None else b).astype(float)
        a["b_max"] = np.nan if b_max is None else float(b_max)
        for k, v in prof.items():
            a[f"profile_{k}"] = np.nan if v is None else v
        if calibration:
            for k, v in calibration.items():
                a[f"calibration_{k}"] = v
        g = f.create_group("eos")
        g.attrs["name"], g.attrs["music_eos_id"] = eos.name, _eos_id(eos)
        if isinstance(eos, IdealGasEoS):
            g.attrs["kind"], g.attrs["dof"] = "ideal", eos.dof
        else:
            g.attrs["kind"], g.attrs["n_ext"] = "table", eos.n_ext
            g.attrs["e_raw_range"] = np.asarray(eos.e_raw_range)
            for k in ("e_tab", "p_tab", "T_tab"):
                g.create_dataset(k, data=getattr(eos, k))

        e0 = f.create_dataset("e0", (n_events, nx, ny, nz), dtype="f4", chunks=(1, nx, ny, nz),
                              compression="gzip", compression_opts=4, shuffle=True)
        cols = {k: np.zeros(n_events) for k in ("b", "e_max", "T_max")}
        cols.update({k: np.zeros(n_events, dtype=np.int64) for k in ("npart", "ncoll")})
        for i in range(n_events):
            e, meta = make_event(proj, targ, b=b, b_max=b_max, sigma_nn_mb=sigma_nn_mb, grid=grid,
                                 spacing=spacing, seed=int(seeds[i]), eos=eos, K=K, quantity=quantity,
                                 e_floor=e_floor, **prof)
            e0[i] = e
            for k in cols:
                cols[k][i] = meta[k]
            if progress and (i + 1) % max(1, n_events // 10) == 0:
                print(f"  {i + 1}/{n_events} events", flush=True)
        for k, v in cols.items():
            f.create_dataset(k, data=v)
        f.create_dataset("seed", data=seeds)
    return path


def load_eos(path):
    """The EoS stored in a stage-1 file, rebuilt without resampling."""
    import h5py

    with h5py.File(path, "r") as f:
        g = f["eos"]
        if g.attrs["kind"] == "ideal":
            return IdealGasEoS(dof=float(g.attrs["dof"]))
        return TableEoS.from_tables(g["e_tab"][:], g["p_tab"][:], g["T_tab"][:], n_ext=int(g.attrs["n_ext"]),
                                    name=str(g.attrs["name"]), e_raw_range=g.attrs["e_raw_range"])


def load_events(path, index=slice(None)):
    """(e0, meta).  e0 has shape (n, X, Y, eta); meta holds the file attributes and the
    per-event columns for the selected events.  `index` is a slice or a sorted index list."""
    import h5py

    with h5py.File(path, "r") as f:
        if f.attrs.get("format", "") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        meta = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        for k in ("b", "npart", "ncoll", "e_max", "T_max", "seed"):
            meta[k] = f[k][index]
        return f["e0"][index], meta


def regenerate_event(path, i):
    """Recompute event i from the stored seed and parameters (float64, before the float32 cast)."""
    _, m = load_events(path, slice(i, i + 1))
    prof = {k: (None if isinstance(m[f"profile_{k}"], float) and np.isnan(m[f"profile_{k}"]) else m[f"profile_{k}"])
            for k in _PROFILE_DEFAULTS}
    prof["string_fluct"] = bool(prof["string_fluct"])
    b = {"min-bias": None, "fixed": float(m["b_arg"][0]), "range": tuple(m["b_arg"])}[m["b_mode"]]
    e, _ = make_event(m["proj"], m["targ"], b=b, b_max=None if np.isnan(m["b_max"]) else float(m["b_max"]),
                      sigma_nn_mb=float(m["sigma_nn_mb"]), grid=(int(m["nx"]), int(m["ny"]), int(m["neta"])),
                      spacing=(float(m["dx"]), float(m["dy"]), float(m["deta"])), seed=int(m["seed"][0]),
                      eos=load_eos(path), K=float(m["K"]), quantity=m["quantity"], e_floor=float(m["e_floor"]), **prof)
    return e


# ====================================================================== self-check

def _write_toy_music_table(path, n_rows=100000, e_max=600.0):
    """A thermodynamically consistent crossover EoS in MUSIC's hotQCD binary format.
    It is a STAND-IN for testing the reader, not the lattice EoS."""
    T = np.geomspace(0.01, 1.2, 40000)
    a = np.pi ** 2 / 90.0 * (3.0 + 0.5 * (47.5 - 3.0) * (1.0 + np.tanh((T - 0.17) / 0.03)))
    p = a * T ** 4 / HBARC ** 3
    s = np.gradient(p, T)
    e = T * s - p
    eg = np.linspace(e[0] * 1.001, e_max, n_rows)
    np.stack([eg, np.interp(eg, e, p), np.interp(eg, e, s), np.interp(eg, e, T)], axis=1).astype("<f8").tofile(path)


def _selftest():
    import tempfile

    rng = np.random.default_rng(1)
    x, y, eta = centres(64, 0.3125), centres(64, 0.3125), centres(32, 0.3125)
    dA = (x[1] - x[0]) * (y[1] - y[0])

    print("--- Glauber")
    n = [glauber_event("Au", "Au", 0.0, 4.2, rng) for _ in range(20)]
    print(f"Au+Au b=0   <Npart> = {np.mean([v['npart'] for v in n]):6.1f}   "
          f"<Ncoll> = {np.mean([v['ncoll'] for v in n]):7.1f}")
    n = []
    while len(n) < 400:
        v = glauber_event("d", "Au", 10.0 * np.sqrt(rng.random()), 4.2, rng)
        if v["ncoll"]:
            n.append(v)
    print(f"d+Au min-bias <Npart> = {np.mean([v['npart'] for v in n]):6.2f}   "
          f"<Ncoll> = {np.mean([v['ncoll'] for v in n]):6.2f}")

    ev = glauber_event("Au", "Au", 7.0, 4.2, rng)
    rho = tilted_profile(ev, x, y, eta, alpha=0.0, eta0=99.0, eta_m=1e9)      # f+- = 1/2, H = 1
    print(f"sum(profile) dA at one eta = {rho[:, :, 0].sum() * dA:7.2f}   vs   Npart/2 = {ev['npart'] / 2:7.2f}")

    e, meta = make_event("Au", "Au", b=7.0, seed=3)
    xbar = (e * x[:, None, None]).sum((0, 1)) / e.sum((0, 1))
    mid = 0.5 * (xbar[15] + xbar[16])
    print(f"Au+Au b=7: <x>(eta) - <x>(0) = {xbar[8] - mid:+.3f} fm at eta=-2.3, "
          f"{xbar[23] - mid:+.3f} fm at eta=+2.3   (tilt)")

    ev = glauber_event("Au", "Au", 0.0, 4.2, rng)
    smooth = tilted_profile(ev, x, y, eta, alpha=0.0, rng=rng).sum((0, 1))
    fl = np.mean([tilted_profile(ev, x, y, eta, alpha=0.0, string_fluct=True, rng=rng).sum((0, 1))
                  for _ in range(40)], axis=0)
    print(f"string_fluct: max |<fluct>/smooth - 1| over eta = {np.max(np.abs(fl / smooth - 1)):.3f}")

    print("--- EoS: ideal gas (MUSIC EOS 0)")
    ideal = get_eos(0)
    e_test = np.geomspace(1e-3, 300.0, 50)
    T_music = (90.0 / np.pi ** 2 * (e_test / HBARC / 3.0) / 42.25) ** 0.25 * HBARC      # formula in eos_idealgas.cpp
    print(f"T(e) vs MUSIC formula: max rel. diff = {np.max(np.abs(ideal.T(e_test) / T_music - 1)):.1e}")
    print(f"round trips: e_from_s {np.max(np.abs(ideal.e_from_s(ideal.s(e_test)) / e_test - 1)):.1e}, "
          f"e_from_T {np.max(np.abs(ideal.e_from_T(ideal.T(e_test)) / e_test - 1)):.1e}")

    print("--- EoS: table reader, tested on a toy file in MUSIC's hotQCD binary format")
    with tempfile.TemporaryDirectory() as tmp:
        _write_toy_music_table(os.path.join(tmp, _MUSIC_FILES[91]))
        tab = get_eos(91, tmp)                                     # directory, as for MUSIC's EOS/hotQCD
    print(tab.report())
    print(f"round trips: e_from_s {np.max(np.abs(tab.e_from_s(tab.s(e_test)) / e_test - 1)):.1e}, "
          f"e_from_T {np.max(np.abs(tab.e_from_T(tab.T(e_test)) / e_test - 1)):.1e}")
    print(f"vacuum: p/e at e = 1e-10 is {tab.p(1e-10) / 1e-10:.3f};  high-T limit p/e = {tab.p(500.0) / 500.0:.3f}")

    print("--- normalisation through the EoS (central Au+Au, target T = 0.36 GeV at the centre)")
    for eos in (ideal, tab):
        for quantity in ("energy", "entropy"):
            K = calibrate_K(0.36, "T", quantity=quantity, eos=eos, n_events=10)
            evs = [make_event("Au", "Au", b=0.0, seed=s, K=K, quantity=quantity, eos=eos) for s in range(12)]
            e_c = np.mean([e[31:33, 31:33, 15:17].mean() for e, _ in evs])        # event-averaged centre
            unit = "GeV/fm" if quantity == "energy" else "1/fm"
            print(f"{eos.name[:12]:12s} {quantity:8s} K = {K:7.3f} {unit:6s} centre: <e> = {e_c:5.1f} GeV/fm^3, "
                  f"T(<e>) = {eos.T(e_c):.3f} GeV | hottest spot: <T_max> = {np.mean([m['T_max'] for _, m in evs]):.3f} GeV")

    print("--- stage 1: write, reload, regenerate")
    with tempfile.TemporaryDirectory() as tmp:
        _write_toy_music_table(os.path.join(tmp, _MUSIC_FILES[91]), n_rows=20000)
        tab = get_eos(91, tmp)
        out = os.path.join(tmp, "ic.h5")
        K = calibrate_K(0.36, "T", quantity="entropy", eos=tab, n_events=10)
        save_events(out, 6, "Au", "Au", eos=tab, K=K, quantity="entropy", b=(0.0, 5.0), seed=11,
                    string_fluct=True, gamma_k=2.0, progress=False,
                    calibration={"target": 0.36, "kind": "T", "b": 0.0})
        e0, m = load_events(out)
        eos2 = load_eos(out)
        size = os.path.getsize(out) / 1e6
        print(f"file: {size:.2f} MB for {e0.shape[0]} events, e0 {e0.shape} {e0.dtype}; "
              f"b = {np.round(m['b'], 2)}, Npart = {m['npart']}")
        e_test = np.geomspace(1e-8, 300.0, 200)
        print(f"stored EoS vs original: max |dp| = {np.max(np.abs(eos2.p(e_test) - tab.p(e_test))):.1e}, "
              f"max |dT| = {np.max(np.abs(eos2.T(e_test) - tab.T(e_test))):.1e}   (music id {_eos_id(tab)})")
        diff = max(np.max(np.abs(regenerate_event(out, i).astype("f4") - e0[i])) for i in (0, 3, 5))
        print(f"regenerated from stored seeds: max |difference| = {diff:.1e}")
        sub, msub = load_events(out, [1, 4])
        print(f"partial read: {sub.shape}, seeds match = {bool(np.all(msub['seed'] == m['seed'][[1, 4]]))}")
        try:
            save_events(out, 1, eos=tab, K=K, progress=False)
        except FileExistsError:
            print("overwrite protection: ok")
        ideal_out = os.path.join(tmp, "ic_ideal.h5")
        save_events(ideal_out, 2, "d", "Au", eos=get_eos(0), K=5.0, seed=2, progress=False)
        print(f"ideal-gas file reloads as {type(load_eos(ideal_out)).__name__}(dof={load_eos(ideal_out).dof})")


def _cli(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="3D MC-Glauber initial states with a tilted source (stage 1).")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("selftest", help="run the built-in checks (default)")
    c = sub.add_parser("check-eos", help="inspect a MUSIC EoS table")
    c.add_argument("path")
    g = sub.add_parser("generate", help="write initial states to an HDF5 file")
    g.add_argument("out")
    g.add_argument("-n", "--n-events", type=int, default=100)
    g.add_argument("--proj", default="Au")
    g.add_argument("--targ", default="Au")
    g.add_argument("--sigma-nn", type=float, default=42.0, help="inelastic NN cross section [mb]")
    g.add_argument("--b", type=float, nargs="+", default=None, help="one value: fixed b; two values: range; none: min-bias")
    g.add_argument("--eos", default="0", help="0 | 9 | 91, or a path to MUSIC's check_EoS_<id>_PST.dat")
    g.add_argument("--eos-path", default=None, help="hotQCD binary file, or MUSIC's EOS/hotQCD directory")
    g.add_argument("--quantity", choices=("energy", "entropy"), default="entropy")
    norm = g.add_mutually_exclusive_group(required=True)
    norm.add_argument("--K", type=float, help="normalisation (GeV/fm for energy, 1/fm for entropy)")
    norm.add_argument("--target-T", type=float, help="event-averaged central T [GeV] at --calib-b")
    norm.add_argument("--target-e", type=float, help="same, energy density [GeV/fm^3]")
    norm.add_argument("--target-s", type=float, help="same, entropy density [1/fm^3]")
    g.add_argument("--calib-b", type=float, default=0.0)
    g.add_argument("--calib-events", type=int, default=None, help="default: 20, or 300 if a nucleus is d or p")
    g.add_argument("--tau0", type=float, default=0.6)
    g.add_argument("--grid", type=int, nargs=3, default=(64, 64, 32), metavar=("NX", "NY", "NETA"))
    g.add_argument("--spacing", type=float, nargs=3, default=(0.3125, 0.3125, 0.3125), metavar=("DX", "DY", "DETA"))
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--overwrite", action="store_true")
    for k, v in _PROFILE_DEFAULTS.items():
        if isinstance(v, bool):
            g.add_argument(f"--{k.replace('_', '-')}", action="store_true")
        else:
            g.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)

    p = sub.add_parser("plot", help="quick-look 3D + x-y-slice plot of one event")
    p.add_argument("--proj", default="Au")
    p.add_argument("--targ", default="Au")
    p.add_argument("--b", type=float, default=None, help="impact parameter [fm]; default: min-bias")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma-nn", type=float, default=42.0, help="inelastic NN cross section [mb]")
    p.add_argument("--grid", type=int, nargs=3, default=(64, 64, 32), metavar=("NX", "NY", "NETA"))
    p.add_argument("--spacing", type=float, nargs=3, default=(0.3125, 0.3125, 0.3125), metavar=("DX", "DY", "DETA"))
    p.add_argument("--K", type=float, default=1.0, help="normalisation (quantity='energy')")
    p.add_argument("--threshold", type=float, default=0.02, help="3D scatter cut, fraction of e_max")
    p.add_argument("--out", default=None, help="save the figure here instead of showing it")

    a = ap.parse_args(argv)

    if a.cmd in (None, "selftest"):
        return _selftest()
    if a.cmd == "check-eos":
        reader = TableEoS.from_music_check if a.path.endswith("PST.dat") else TableEoS.from_music_binary
        return print(reader(a.path).report())
    if a.cmd == "plot":
        e, meta = make_event(a.proj, a.targ, b=a.b, sigma_nn_mb=a.sigma_nn, grid=tuple(a.grid),
                             spacing=tuple(a.spacing), seed=a.seed, K=a.K)
        x, y, eta = (centres(n, d) for n, d in zip(a.grid, a.spacing))
        title = f"{a.proj}+{a.targ}  b={meta['b']:.2f} fm  Npart={meta['npart']}  Ncoll={meta['ncoll']}"
        return plot_initial_state(e, x, y, eta, title=title, out=a.out, threshold=a.threshold)

    eos = TableEoS.from_music_check(a.eos) if os.path.isfile(a.eos) else get_eos(int(a.eos), a.eos_path)
    prof = {k: getattr(a, k) for k in _PROFILE_DEFAULTS}
    b = None if a.b is None else (a.b[0] if len(a.b) == 1 else tuple(a.b[:2]))
    calibration = None
    K = a.K
    if K is None:
        kind, target = next((k, getattr(a, f"target_{k}")) for k in ("T", "e", "s") if getattr(a, f"target_{k}") is not None)
        n_cal = a.calib_events or (300 if {a.proj, a.targ} & {"d", "p"} else 20)
        K = calibrate_K(target, kind, quantity=a.quantity, eos=eos, proj=a.proj, targ=a.targ, b=a.calib_b,
                        sigma_nn_mb=a.sigma_nn, n_events=n_cal, seed=a.seed + 1,
                        **{k: v for k, v in prof.items() if k in ("w", "alpha", "eta0", "sig_eta", "eta_m")})
        calibration = {"target": target, "kind": kind, "b": a.calib_b, "n_events": n_cal}
        print(f"calibrated K = {K:.4f} for central {kind} = {target} at b = {a.calib_b} fm ({n_cal} events)")
    save_events(a.out, a.n_events, a.proj, a.targ, eos=eos, K=K, quantity=a.quantity, tau0=a.tau0, b=b,
                sigma_nn_mb=a.sigma_nn, grid=tuple(a.grid), spacing=tuple(a.spacing), seed=a.seed,
                overwrite=a.overwrite, calibration=calibration, **prof)
    _, m = load_events(a.out)
    print(f"wrote {a.out}: {len(m['b'])} events, <Npart> = {m['npart'].mean():.1f}, "
          f"<e_max> = {m['e_max'].mean():.1f} GeV/fm^3, <T_max> = {m['T_max'].mean():.3f} GeV, "
          f"{os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--check-eos":          # earlier spelling, kept working
        sys.argv[1] = "check-eos"
    _cli()
