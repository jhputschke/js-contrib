"""Visualisation for fast_data output files.

Everything here reads the HDF5 one frame at a time, so a multi-GiB file can be browsed
interactively without loading it.

That only works because of two things `EventBrowser` does on open, and it needs BOTH.  `arr` is
chunked one whole event per chunk -- 234 MB at the Au+Au geometry -- so a single-frame read has
to decompress the entire event unless the chunk is cached.

  1. Size the chunk cache to hold one chunk.  h5py's default is 1 MB, which cannot hold a 234 MB
     chunk, so nothing is ever cached.
  2. Hold the dataset handle open.  HDF5 stores the chunk cache on the dataset, so the natural
     `self._f["arr"][...]` -- which opens and drops a handle per call -- discards the cache on
     every read and makes step 1 useless.

Measured on a 3-event Au+Au file: 110 ms per frame with either one missing, 1.0 ms with both.
Drawing one wake history reads ~190 frames, so it went from 31 s to 0.55 s.

    from fast_data import viz
    ds = viz.EventBrowser("out/fastdata_AuAu200.h5")
    ds.plot_slice(event=0, itau=20, ieta=16)      # x-y heatmap + flow, at one (tau, eta)
    ds.plot_evolution(event=0)                    # the whole history at a glance
    ds.plot_initial_state(event=0)                # the 3D scatter of glauber.plot_initial_state

`workflow_fastdata/explore_fastdata.ipynb` wires these to sliders for event, eta and tau.

The channel convention is the FNO4d one -- (energy_density, vx, vy, vz) with CARTESIAN LAB
three-velocities -- so the arrows drawn on an x-y slice are lab-frame vx, vy and are directly
comparable between slices at different eta.  See README_FastData.md.
"""

from __future__ import annotations

import numpy as np

__all__ = ["EventBrowser", "DiffBrowser", "plot_slice", "plot_evolution",
           "plot_initial_state_3d", "flow_mask", "plot_mach"]

_E, _VX, _VY, _VZ = 0, 1, 2, 3
CHANNEL_LABELS = (r"$e$ [GeV/fm$^3$]", r"$v_x$", r"$v_y$", r"$v_z$")
SOURCE_LABELS = (r"$S^\tau$ [GeV/fm$^2$]", r"$S^x$ [GeV/fm$^2$]",
                 r"$S^y$ [GeV/fm$^2$]", r"$S^\eta$ [GeV/fm$^3$]")


class EventBrowser:
    """Lazy handle on a fast_data (or MUSIC-format) file, with the plotting helpers attached."""

    def __init__(self, path, *, max_cache_bytes=1 << 30):
        import h5py

        self.path = str(path)
        kw = {}
        with h5py.File(self.path, "r") as probe:              # size the chunk cache first
            ds = probe["arr"]
            if ds.chunks is not None:
                nb = int(np.prod(ds.chunks)) * ds.dtype.itemsize
                if nb <= max_cache_bytes:
                    # one chunk is one event; 1009 slots is prime, as HDF5 wants
                    kw = dict(rdcc_nbytes=int(nb * 1.15), rdcc_nslots=1009)
        self.cache_bytes = int(kw.get("rdcc_nbytes", 0))
        self._kw = kw
        self._open()
        a = dict(self._f.attrs)
        self.attrs = a
        self.nevents = int(self._f["arr"].shape[0])
        self.ntau = int(self._f["arr"].shape[-1])
        self.nx, self.ny, self.neta = (int(a["nx"]), int(a["ny"]), int(a["neta"]))
        self.x = float(a["x_min"]) + np.arange(self.nx) * float(a["dx"])
        self.y = float(a["y_min"]) + np.arange(self.ny) * float(a["dy"])
        self.eta = float(a["eta_min"]) + np.arange(self.neta) * float(a["deta"])
        self.tau = float(a["tau_min"]) + np.arange(self.ntau) * float(a["dtau"])
        self.ntau_fo = self._f["ntau_freezeout"][:]
        self.tau_fo = self._f["tau_freezeout"][:]
        self.has_source = bool(a.get("has_source", False)) and "source/S" in self._f

    # ------------------------------------------------------------------ the file handle
    def _open(self):
        import h5py

        self._file = h5py.File(self.path, "r", **self._kw)
        # Hold the dataset handles open.  HDF5 keeps the chunk cache on the dataset, so
        # `self._f["arr"][...]` -- which opens and drops a handle per call -- throws the cache
        # away every time and re-decompresses the whole 234 MB chunk: 110 ms a frame instead
        # of 1.6.  These two lines are the difference between browsing and waiting.
        self._arr_ds = self._file["arr"]
        self._S_ds = self._file["source/S"] if "source/S" in self._file else None

    def _ensure_open(self):
        """Reopen the file if it was closed under us.

        The notebook closes the browser in its last cell, but the sliders of the earlier cells
        stay live: the next one moved fires a read against a closed file, and h5py answers from
        inside the widget callback with

            OSError: Can't synchronously read data (identifier is not of specified type)

        -- an error about identifiers, raised in a cell that has nothing to do with closing,
        with no traceback line pointing at the `close()` that caused it.  Reopening costs one
        `H5Fopen` and is invisible; the alternative is that message.
        """
        if not self._file.id.valid:
            self._open()

    @property
    def _f(self):
        self._ensure_open()
        return self._file

    @property
    def _arr(self):
        self._ensure_open()
        return self._arr_ds

    @property
    def _S(self):
        self._ensure_open()
        return self._S_ds

    # ------------------------------------------------------------------ data access
    def frame(self, event, itau, channel=None):
        """One (4, nx, ny, neta) frame, or one channel of it.  Reads only that frame."""
        if channel is None:
            return self._arr[event, :, :, :, :, itau]
        return self._arr[event, channel, :, :, :, itau]

    def source_frame(self, event, itau, channel=0):
        if not self.has_source:
            return None
        return self._S[event, channel, :, :, :, itau]

    def initial_energy(self, event):
        return self.frame(event, 0, _E)

    def droplets(self, event):
        """The droplets of one event as (M, 8), or None."""
        if "source/droplets" not in self._f:
            return None
        off = self._f["source/offsets"][:]
        return self._f["source/droplets"][off[event]:off[event + 1]]

    def live(self, event):
        """The number of non-zero frames, i.e. ntau_freezeout - 1 (see README_FastData.md)."""
        return max(int(self.ntau_fo[event]) - 1, 0)

    def summary(self):
        """A per-event table of the diagnostics, as a dict of arrays."""
        out = {"event": np.arange(self.nevents),
               "ntau_freezeout": self.ntau_fo, "tau_freezeout": self.tau_fo}
        if "diag" in self._f:
            for k in self._f["diag"]:
                v = self._f[f"diag/{k}"][:]
                if v.shape == (self.nevents,):
                    out[k] = v
        return out

    def describe(self):
        a = self.attrs
        lines = [f"{self.path}",
                 f"  {self.nevents} events, grid {self.nx}x{self.ny}x{self.neta}, "
                 f"{self.ntau} tau frames",
                 f"  x, y in [{self.x[0]:.2f}, {self.x[-1]:.2f}] fm;  "
                 f"eta in [{self.eta[0]:.2f}, {self.eta[-1]:.2f}];  "
                 f"tau {self.tau[0]:.2f} .. {self.tau[-1]:.2f} fm",
                 f"  freeze-out frames: min {self.ntau_fo.min()}, median "
                 f"{int(np.median(self.ntau_fo))}, max {self.ntau_fo.max()}"]
        for k in ("generator", "eos_desc", "transport_mode", "source_model", "proj", "targ"):
            if k in a:
                lines.append(f"  {k}: {a[k]}")
        return "\n".join(lines)

    def close(self):
        """Release the handle.  A later read reopens it -- see `_ensure_open`."""
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------------ plots
    def plot_slice(self, event=0, itau=0, ieta=None, **kw):
        return plot_slice(self, event, itau, ieta, **kw)

    def plot_evolution(self, event=0, **kw):
        return plot_evolution(self, event, **kw)

    def plot_initial_state(self, event=0, **kw):
        return plot_initial_state_3d(self, event, **kw)


def flow_mask(e_slice, floor=1e-3):
    """Which cells hold enough medium for their flow velocity to mean anything.

    In a near-vacuum cell the Landau match is ill-conditioned and |v| runs up to the solver's
    cap, so an arrow there describes the reconstruction rather than the fluid -- and there are
    far more such cells than real ones, so they dominate a quiver plot.  Cells at or below
    `floor` times the slice's peak energy density are masked out.
    """
    e_slice = np.asarray(e_slice)
    peak = float(np.max(e_slice)) if e_slice.size else 0.0
    return e_slice > floor * max(peak, 1e-30)


def _log_norm(a, dynamic_range=1e4):
    """A LogNorm over the positive part of `a`, or None if there is none.

    Bounded below at `dynamic_range` under the peak: the raw minimum of a hydro field is the
    vacuum floor (1e-6 GeV/fm^3), and stretching the colour map over ten decades to reach it
    leaves the fireball itself in the top sliver of the bar.
    """
    from matplotlib.colors import LogNorm

    pos = np.asarray(a)[np.asarray(a) > 0]
    if not pos.size:
        return None
    hi = float(pos.max())
    return LogNorm(vmin=max(float(pos.min()), hi / dynamic_range), vmax=hi)


def _fig(figsize):
    import matplotlib.pyplot as plt
    return plt.subplots(1, 3, figsize=figsize, constrained_layout=True)


def plot_slice(br: EventBrowser, event=0, itau=0, ieta=None, *, channel=_E, quiver=True,
               n_arrows=18, quiver_floor=1e-3, log=False, dynamic_range=1e4, vmax=None,
               cmap="inferno", show_source=None, figsize=(15, 4.4)):
    """x-y heatmap at one (tau, eta), the x-eta slice at y=0, and the tau history.

    The middle panel is the longitudinal view, which is where the tilt of the Glauber initial
    state and any forward jet deposit actually show up; the right panel puts the chosen frame in
    the context of the whole evolution, with freeze-out and any jet deposits marked.

    `quiver_floor` hides the flow arrows in cells below that fraction of the frame's peak energy
    density -- without it the near-vacuum cells, where |v| runs to the solver's cap, dominate
    the picture.
    """
    import matplotlib.pyplot as plt

    if ieta is None:
        ieta = br.neta // 2
    itau = int(np.clip(itau, 0, br.ntau - 1))
    ieta = int(np.clip(ieta, 0, br.neta - 1))
    if show_source is None:
        show_source = False

    f = br.frame(event, itau)
    field = br.source_frame(event, itau, channel) if show_source else f[channel]
    lbl = (SOURCE_LABELS if show_source else CHANNEL_LABELS)[channel]

    fig, axes = _fig(figsize)
    live = br.live(event)
    dead = itau >= live

    # --- (1) transverse slice -------------------------------------------------------
    ax = axes[0]
    img = field[:, :, ieta].T
    norm = _log_norm(img, dynamic_range) if log else None
    im = ax.pcolormesh(br.x, br.y, img, cmap=cmap, shading="auto", norm=norm,
                       vmax=(None if norm is not None else vmax))
    fig.colorbar(im, ax=ax, label=lbl)
    if quiver and not show_source and channel == _E:
        s = max(br.nx // n_arrows, 1)
        X, Y = np.meshgrid(br.x[::s], br.y[::s], indexing="ij")
        u, v = f[_VX, ::s, ::s, ieta], f[_VY, ::s, ::s, ieta]
        keep = flow_mask(f[_E, :, :, ieta], quiver_floor)[::s, ::s]
        u = np.where(keep, u, np.nan)
        v = np.where(keep, v, np.nan)
        ax.quiver(X, Y, u, v, color="cyan", alpha=0.75, scale=8, width=0.004)
    ax.set_xlabel("x [fm]"); ax.set_ylabel("y [fm]"); ax.set_aspect("equal")
    ax.set_title(rf"$\tau$ = {br.tau[itau]:.2f} fm,  $\eta$ = {br.eta[ieta]:+.2f}"
                 + ("   (past freeze-out)" if dead else ""))

    # --- (2) longitudinal slice ------------------------------------------------------
    ax = axes[1]
    iy0 = br.ny // 2
    im = ax.pcolormesh(br.eta, br.x, field[:, iy0, :], cmap=cmap, shading="auto",
                       norm=_log_norm(field[:, iy0, :], dynamic_range) if log else None)
    fig.colorbar(im, ax=ax, label=lbl)
    ax.axvline(br.eta[ieta], color="cyan", lw=1.2, ls="--")
    ax.set_xlabel(r"$\eta$"); ax.set_ylabel("x [fm]")
    ax.set_title(rf"x-$\eta$ slice at y = {br.y[iy0]:.2f} fm")

    # --- (3) the frame in context ----------------------------------------------------
    ax = axes[2]
    prof = _tau_profile(br, event)
    ax.plot(br.tau[:len(prof)], prof, color="crimson", lw=1.6)
    ax.axvline(br.tau[itau], color="cyan", lw=1.2, ls="--", label="this frame")
    if live < br.ntau:
        ax.axvline(br.tau_fo[event], color="k", lw=1.0, ls=":",
                   label=rf"freeze-out ({br.tau_fo[event]:.2f} fm)")
    d = br.droplets(event)
    if d is not None and len(d):
        tdep = d[:, 0] + float(br.attrs.get("liquefier_tau_delay", 0.0))
        if len(tdep) > 8:                      # a trajectory: the window, not 32 hairlines
            ax.axvspan(tdep.min(), tdep.max(), color="tab:green", alpha=0.15,
                       label=f"jet depositing ({len(tdep)} droplets)")
        else:
            for k, t in enumerate(tdep):
                ax.axvline(t, color="tab:green", lw=1.0, alpha=0.8,
                           label="jet deposit" if k == 0 else None)
    ax.set_yscale("log"); ax.set_xlabel(r"$\tau$ [fm]")
    ax.set_ylabel(r"max $e$ [GeV/fm$^3$]")
    ax.set_title(f"event {event} history")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"{_title(br, event)}", fontsize=10)
    return fig


def _tau_profile(br, event, channel=_E):
    """max over cells of one channel, per frame, up to freeze-out."""
    live = max(br.live(event), 1)
    return np.array([np.abs(br.frame(event, k, channel)).max() for k in range(live)])


def _title(br, event):
    a = br.attrs
    sys_ = f"{a.get('proj', '?')}+{a.get('targ', '?')}" if "proj" in a else a.get("generator", "")
    d = br.summary()
    bits = [f"event {event}/{br.nevents - 1}", sys_]
    for k, fmt in (("b", "b = {:.2f} fm"), ("npart", "Npart = {:.0f}"), ("ncoll", "Ncoll = {:.0f}")):
        if k in d:
            bits.append(fmt.format(d[k][event]))
    if br.has_source:
        bits.append("with jet source")
    return "   ".join(b for b in bits if b)


def plot_evolution(br: EventBrowser, event=0, *, n_frames=6, ieta=None, cmap="inferno",
                   figsize=None, log=True, dynamic_range=1e3, per_frame=False):
    """A filmstrip of x-y slices from tau0 to freeze-out, plus the eta-tau history.

    Both panels are LOGARITHMIC by default, because a central Au+Au fireball cools by a factor
    of ~100 between tau0 and freeze-out (measured: e_max 31.6 -> 0.25 GeV/fm^3).  The obvious
    choice -- one linear scale pinned to the peak -- therefore renders every frame after the
    first two as black, and the eta-tau map as a bright sliver at early tau over an empty field.

    The scale spans `dynamic_range` below the peak, shared across the strip so the frames stay
    comparable.  `per_frame=True` rescales each panel to its own maximum instead: every frame is
    then maximally legible, but the cooling can no longer be read off the picture.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    if ieta is None:
        ieta = br.neta // 2
    live = max(br.live(event), 1)
    idx = np.unique(np.linspace(0, live - 1, n_frames).astype(int))
    figsize = figsize or (2.6 * len(idx), 5.8)

    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(cmap(0.0))          # LogNorm masks zeros; show them as the floor, not blank

    def norm_for(peak):
        hi = float(peak) if np.isfinite(peak) and peak > 0 else 1.0
        return LogNorm(vmin=hi / dynamic_range, vmax=hi) if log else Normalize(0.0, hi)

    slices = [br.frame(event, int(k), _E)[:, :, ieta].T for k in idx]
    shared = None if per_frame else norm_for(max(float(sl.max()) for sl in slices))

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, len(idx))
    strip, im = [], None
    for j, (k, sl) in enumerate(zip(idx, slices)):
        ax = fig.add_subplot(gs[0, j])
        im = ax.pcolormesh(br.x, br.y, sl, cmap=cmap, shading="auto",
                           norm=shared if shared is not None else norm_for(sl.max()))
        strip.append(ax)
        ax.set_title(rf"$\tau$={br.tau[k]:.2f}" + (rf"  (max {sl.max():.3g})" if per_frame else ""),
                     fontsize=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(rf"$e$ at $\eta$={br.eta[ieta]:+.1f}", fontsize=9)
    if shared is not None:
        fig.colorbar(im, ax=strip, label=r"$e$ [GeV/fm$^3$]", fraction=0.03, pad=0.01)

    # eta-tau map of the transverse-maximum energy density
    ax = fig.add_subplot(gs[1, :])
    m = np.array([br.frame(event, k, _E).max(axis=(0, 1)) for k in range(live)])
    im = ax.pcolormesh(br.tau[:live], br.eta, m.T, cmap=cmap, shading="auto",
                       norm=norm_for(m.max()))
    fig.colorbar(im, ax=ax, label=r"max$_{x,y}$ $e$ [GeV/fm$^3$]")
    ax.set_xlabel(r"$\tau$ [fm]"); ax.set_ylabel(r"$\eta$")
    for k in idx:
        ax.axvline(br.tau[k], color="w", lw=0.6, alpha=0.35)      # where the strip was sampled
    d = br.droplets(event)
    if d is not None and len(d):
        delay = float(br.attrs.get("liquefier_tau_delay", 0.0))
        ax.scatter(d[:, 0] + delay, d[:, 3], marker="x", s=45, c="cyan", label="jet deposits")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(_title(br, event) + ("   |   log scale" if log else "   |   linear scale"),
                 fontsize=10)
    return fig


def plot_initial_state_3d(br: EventBrowser, event=0, *, threshold=0.02, **kw):
    """The initial state of one stored event, drawn like glauber.plot_initial_state."""
    from . import glauber

    e = br.initial_energy(event).astype(np.float64)
    return glauber.plot_initial_state(e, br.x, br.y, br.eta,
                                      title=_title(br, event), threshold=threshold, **kw)


# =============================================================================== diff / Mach cone
#
# A jet's wake is a ~0.3 GeV/fm^3 ripple on a fireball that starts at 30 GeV/fm^3, so in raw `e`
# it is invisible -- which is the usual reason someone reports "I can't see the Mach cone".  The
# fix is not a better colour map, it is subtraction: the parton draw has its own RNG stream, so a
# run with `source.enabled: true` and a run with it false, at the same `run.seed`, share their
# initial conditions bit for bit and `e(jet) - e(no jet)` is the jet's contribution alone.

class DiffBrowser:
    """Two runs that differ only in the source, differenced frame by frame.

        d = DiffBrowser("out/fastdata_AuAu200_central_jet.h5",
                        "out/fastdata_AuAu200_central.h5")
        d.plot_mach(0, itau=90)

    Both files are read lazily, so this costs one frame of memory, not two files.
    """

    def __init__(self, jet, ref, *, check_ic=True):
        self.jet = jet if isinstance(jet, EventBrowser) else EventBrowser(jet)
        self.ref = ref if isinstance(ref, EventBrowser) else EventBrowser(ref)
        for k in ("nx", "ny", "neta", "dx", "dy", "deta", "tau_min", "dtau"):
            a, b = self.jet.attrs.get(k), self.ref.attrs.get(k)
            if a is not None and b is not None and not np.allclose(float(a), float(b)):
                raise ValueError(f"the two runs disagree on '{k}' ({a} vs {b}); a difference "
                                 f"between different grids is meaningless")
        self.nevents = min(self.jet.nevents, self.ref.nevents)
        self.ntau = min(self.jet.ntau, self.ref.ntau)
        for k in ("x", "y", "eta", "tau", "nx", "ny", "neta", "attrs"):
            setattr(self, k, getattr(self.jet, k))
        self.ic_matches = None
        if check_ic and self.nevents:
            self.ic_matches = bool(np.array_equal(self.jet.frame(0, 0), self.ref.frame(0, 0)))

    def diff(self, event, itau, channel=_E):
        """arr(jet) - arr(no jet) for one channel of one frame."""
        return (self.jet.frame(event, itau, channel).astype(np.float64)
                - self.ref.frame(event, itau, channel).astype(np.float64))

    def live(self, event):
        """Frames where BOTH runs are live -- the only range where the difference is a wake.

        Past the reference's freeze-out its frames are zeroed, so `jet - ref` degenerates into
        the jet run's own energy density.  A jet extends the lifetime (10.28 vs 9.88 fm on the
        shipped central pair), so that range is never empty, and it is not the whole file.
        """
        return min(self.jet.live(event), self.ref.live(event))

    # ------------------------------------------------------------------ the moving source
    def source_track(self, event):
        """The deposits as (M, 4) = (tau_dep, x, y, eta), in firing order.

        This is read from the droplet table, not reconstructed from the config, so it is right
        even for a randomly placed or multi-leg jet.
        """
        d = self.jet.droplets(event)
        if d is None or not len(d):
            return None
        delay = float(self.jet.attrs.get("liquefier_tau_delay", 0.0))
        t = np.column_stack([d[:, 0] + delay, d[:, 1], d[:, 2], d[:, 3]])
        return t[np.argsort(t[:, 0])]

    def source_at(self, event, tau):
        """Where the source was at proper time `tau`: (x, y, eta), or None.

        Linear in tau between deposits; clamped to the first/last deposit outside the range, so
        the marker parks at the end of the path rather than flying off.
        """
        t = self.source_track(event)
        if t is None:
            return None
        return tuple(np.interp(float(tau), t[:, 0], t[:, k]) for k in (1, 2, 3))

    def source_active(self, event, tau):
        """Is the jet still depositing at proper time `tau`?

        Once the last droplet has fired, the apex of the pattern is pinned to the terminal
        deposit and the front simply expands away from it, so the opening angle keeps growing
        and is no longer a Mach angle.  Measured on the shipped config: 42-45 deg while the
        source is live, drifting to 53 deg by freeze-out after it stops.
        """
        t = self.source_track(event)
        return bool(t is not None and t[0, 0] <= float(tau) <= t[-1, 0])

    def jet_direction(self, event):
        """The mean Cartesian direction of the deposited momentum, as a unit 3-vector."""
        d = self.jet.droplets(event)
        if d is None or not len(d):
            return None
        p = d[:, 5:8].sum(axis=0)
        n = float(np.linalg.norm(p))
        return (p / n) if n > 0 else None

    # ------------------------------------------------------------------ the Mach angle
    def sound_speed(self, e):
        """c_s at energy density `e`, from the EoS table stored in the jet file.

        Returns None if the file carries no table (a conformal run has c_s = 1/sqrt(3) exactly).
        """
        f = self.jet._f
        if "eos/e_tab" not in f:
            kind = str(self.jet.attrs.get("eos_kind", ""))
            return float(np.sqrt(1.0 / 3.0)) if kind.startswith("conformal") else None
        e_tab, p_tab = f["eos/e_tab"][:], f["eos/p_tab"][:]
        cs2 = np.gradient(p_tab, e_tab)
        return float(np.sqrt(np.clip(np.interp(float(e), e_tab, cs2), 1e-6, 1.0)))

    def mach_angle(self, event, itau, ieta=None):
        """The static-medium Mach half-angle at the source, in degrees: asin(c_s / v), v = 1.

        This is a LOWER BOUND on what you will measure.  The fireball is expanding, and the
        transverse flow at the wave front adds to c_s and opens the cone: measured 39-42 deg on
        the shipped central Au+Au config against 23 deg from this formula evaluated on the cold
        late-time medium.  Evaluating c_s where and when the front was actually launched
        (e ~ 1.5-3 GeV/fm^3, c_s ~ 0.46-0.49) and adding the transverse flow there (0.10-0.17)
        gives 35-41 deg, which is the honest comparison.
        """
        pos = self.source_at(event, self.tau[itau])
        if pos is None:
            return None
        if ieta is None:
            ieta = int(np.argmin(np.abs(self.eta - pos[2])))
        ix = int(np.argmin(np.abs(self.x - pos[0])))
        iy = int(np.argmin(np.abs(self.y - pos[1])))
        e_bg = float(self.ref.frame(event, itau, _E)[ix, iy, ieta])
        cs = self.sound_speed(e_bg)
        return None if cs is None else float(np.degrees(np.arcsin(min(cs, 1.0))))

    def blob_radius(self):
        """How far a single droplet has spread by the time it is deposited: c_diff*tau_delay."""
        a = self.attrs
        return float(a.get("liquefier_c_diff", 0.894)) * float(a.get("liquefier_tau_delay", 1.0))

    def front_angle(self, event, itau, ieta=None, *, y_range=(0.3, 5.0), rel=0.05):
        """The MEASURED half-angle of the compression front, in degrees, or None.

        For each row of constant y behind the source, the ridge is the cell of largest
        `Delta e`; a straight-line fit through those ridge points gives the angle to the jet
        axis.  Compare with `mach_angle`: on the shipped central Au+Au config this returns
        39-42 deg against a static-medium 23-24 deg, and the gap is the fireball's own
        transverse flow carrying the front outwards.
        """
        pos = self.source_at(event, self.tau[itau])
        n = self.jet_direction(event)
        if pos is None or n is None:
            return None
        if ieta is None:
            ieta = int(np.argmin(np.abs(self.eta - pos[2])))
        de = self.diff(event, itau)[:, :, ieta]
        if de.max() <= 0:
            return None
        behind = self.x < pos[0] - 0.3
        if not behind.any():
            return None
        rx, ry = [], []
        for iy, yv in enumerate(self.y):
            dy = yv - pos[1]
            if not (y_range[0] <= dy <= y_range[1]):
                continue
            col = de[behind, iy]
            if col.max() <= rel * de.max():
                continue
            rx.append(self.x[behind][int(col.argmax())]); ry.append(dy)
        if len(rx) < 4:
            return None
        k = float(np.polyfit(ry, rx, 1)[0])
        if k == 0:
            return 90.0
        return float(np.degrees(np.arctan(1.0 / abs(k))))

    def plot_mach(self, event=0, itau=None, ieta=None, *, vmax=None, cmap="RdBu_r",
                  guide=True, percentile=99.5, figsize=(15, 4.4)):
        return plot_mach(self, event, itau, ieta, vmax=vmax, cmap=cmap, guide=guide,
                         percentile=percentile, figsize=figsize)

    def close(self):
        self.jet.close(); self.ref.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return (f"DiffBrowser(jet={self.jet.path!r}, ref={self.ref.path!r}, "
                f"nevents={self.nevents}, ic_matches={self.ic_matches})")


def _diff_scale(d, de, ieta, pos, percentile):
    """A colour scale for the wake, not for the deposit.

    The cell the jet is currently depositing into runs an order of magnitude above the wake, so
    scaling to the maximum leaves the cone as a saturated blob.  Excluding a disc of twice the
    droplet's spread radius around the source puts the scale on the structure worth looking at.
    """
    sl = np.abs(de[:, :, ieta])
    if pos is not None:
        R = max(2.0 * d.blob_radius(), 1.0)
        X, Y = np.meshgrid(d.x, d.y, indexing="ij")
        far = (X - pos[0]) ** 2 + (Y - pos[1]) ** 2 > R ** 2
        if far.any() and sl[far].max() > 0:
            return float(sl[far].max())
    nz = sl[sl > 0]
    return float(np.percentile(nz, percentile)) if nz.size else 1.0


def plot_mach(d: DiffBrowser, event=0, itau=None, ieta=None, *, vmax=None, cmap="RdBu_r",
              guide=True, percentile=99.5, figsize=(15, 4.4)):
    """x-y and x-eta maps of `e(jet) - e(no jet)`, plus the wake's amplitude history.

    Red is compression (the Mach front), blue is the depletion the jet drags behind it -- the
    diffusion wake.  The star marks where the source is at this frame, read from the droplet
    table; the dashed lines are the static-medium Mach angle, which the expanding background
    will open up (see `DiffBrowser.mach_angle`).

    The colour scale is a percentile, not the maximum, because the cell the jet is depositing
    into is always an order of magnitude above the wake and would otherwise set the scale and
    leave the cone invisible.
    """
    import matplotlib.pyplot as plt

    live = d.live(event)
    if itau is None:
        itau = max(live - 1, 0)
    itau = int(np.clip(itau, 0, d.ntau - 1))
    pos = d.source_at(event, d.tau[itau])
    if ieta is None:
        ieta = int(np.argmin(np.abs(d.eta - pos[2]))) if pos is not None else d.neta // 2

    de = d.diff(event, itau)
    v = vmax or _diff_scale(d, de, ieta, pos, percentile)
    fig, axes = _fig(figsize)

    ax = axes[0]
    im = ax.pcolormesh(d.x, d.y, de[:, :, ieta].T, cmap=cmap, vmin=-v, vmax=v, shading="auto")
    fig.colorbar(im, ax=ax, label=r"$\Delta e$ [GeV/fm$^3$]")
    if pos is not None:
        ax.plot([pos[0]], [pos[1]], "k*", ms=13, mec="w", mew=0.8)
        n = d.jet_direction(event)
        if guide and n is not None:
            phi = np.arctan2(n[1], n[0])
            L = float(d.x[-1] - d.x[0])
            live_src = d.source_active(event, d.tau[itau])
            tag = "" if live_src else " (source stopped)"
            lines = [(d.mach_angle(event, itau, ieta), "k", (4, 3), 0.55,
                      r"static $\mu$ = {:.0f}$^\circ$ ($c_s$ alone)"),
                     (d.front_angle(event, itau, ieta), "limegreen", None, 0.9,
                      r"measured front = {:.0f}$^\circ$" + tag)]
            for ang, colour, dashes, alpha, fmt in lines:
                if ang is None:
                    continue
                for sgn in (+1, -1):
                    a = phi + np.pi - sgn * np.radians(ang)   # backward cone, both surfaces
                    ln, = ax.plot([pos[0], pos[0] + L * np.cos(a)],
                                  [pos[1], pos[1] + L * np.sin(a)],
                                  color=colour, lw=1.1, alpha=alpha,
                                  label=fmt.format(ang) if sgn > 0 else None)
                    if dashes:
                        ln.set_dashes(dashes)
            ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(d.x[0], d.x[-1]); ax.set_ylim(d.y[0], d.y[-1]); ax.set_aspect("equal")
    ax.set_xlabel("x [fm]"); ax.set_ylabel("y [fm]")
    ax.set_title(rf"$\Delta e$ at $\tau$={d.tau[itau]:.2f} fm, $\eta$={d.eta[ieta]:+.2f}")

    ax = axes[1]
    iy0 = int(np.argmin(np.abs(d.y - pos[1]))) if pos is not None else d.ny // 2
    im = ax.pcolormesh(d.eta, d.x, de[:, iy0, :], cmap=cmap, vmin=-v, vmax=v, shading="auto")
    fig.colorbar(im, ax=ax, label=r"$\Delta e$ [GeV/fm$^3$]")
    ax.axvline(d.eta[ieta], color="k", lw=1.0, ls="--", alpha=0.5)
    ax.set_xlabel(r"$\eta$"); ax.set_ylabel("x [fm]")
    ax.set_title(rf"x-$\eta$ slice at y = {d.y[iy0]:.2f} fm")

    ax = axes[2]
    amp = np.array([np.abs(d.diff(event, k)).max() for k in range(live)])
    ax.plot(d.tau[:live], amp, color="crimson", lw=1.6, label=r"max $|\Delta e|$")
    bg = np.array([float(d.ref.frame(event, k, _E).max()) for k in range(live)])
    ax.plot(d.tau[:live], bg, color="0.55", lw=1.2, ls="-", label=r"max $e$ (background)")
    ax.axvline(d.tau[itau], color="cyan", lw=1.2, ls="--", label="this frame")
    t = d.source_track(event)
    if t is not None:
        ax.plot(t[:, 0], np.full(len(t), amp.max() if len(amp) else 1.0), "|",
                color="tab:green", ms=8, label="deposits")
    ax.set_yscale("log"); ax.set_xlabel(r"$\tau$ [fm]"); ax.set_ylabel(r"[GeV/fm$^3$]")
    ax.set_title("wake amplitude vs background")
    ax.legend(fontsize=8)

    fig.suptitle(f"{_title(d.jet, event)}   |   jet minus no-jet", fontsize=10)
    return fig
