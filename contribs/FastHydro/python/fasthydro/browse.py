"""`PairBrowser` -- read a FastHydro pair out of ONE file.

FNO4d's `DiffBrowser` differences two runs that live in two files, because fast_data's
`generate.py` writes one evolution per file and a jet/no-jet pair means running it twice.
FastHydro runs both legs in a single event and writes them side by side (`arr` = jet,
`arr_bg` = background), so there is one file, one initial condition and no chance of pairing
the wrong two runs.

Rather than reimplement the wake machinery, this hands `DiffBrowser` two *views* of the same
file -- one bound to `arr`, one to `arr_bg` -- so `diff`, `source_track`, `source_at`,
`mach_angle`, `blob_radius` and the rest come through unchanged and behave exactly as they do
on an FNO4d pair.

    p = PairBrowser("out/fasthydro.h5")
    p.diff(0, k)            # e(jet) - e(background), one frame
    p.source_at(0, tau)     # where the jet was
    p.live(0)               # frames where BOTH legs are still live
"""

from __future__ import annotations

import numpy as np

__all__ = ["PairBrowser", "open_pair"]


def _views(path):
    from fast_data.viz import EventBrowser

    class _BackgroundView(EventBrowser):
        """An EventBrowser bound to `arr_bg` instead of `arr`."""

        def _open(self):
            super()._open()
            self._arr_ds = self._file["arr_bg"]
            # the background leg has its own freeze-out, and `live()` must use it
            if "ntau_freezeout_bg" in self._file:
                self.ntau_fo = self._file["ntau_freezeout_bg"][:]
                self.tau_fo = self._file["tau_freezeout_bg"][:]
            self._S_ds = None            # the source belongs to the jet leg only

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if "ntau_freezeout_bg" in self._f:
                self.ntau_fo = self._f["ntau_freezeout_bg"][:]
                self.tau_fo = self._f["tau_freezeout_bg"][:]
            self.has_source = False

    return EventBrowser(path), _BackgroundView(path)


def open_pair(path, *, check_ic=True):
    """-> `DiffBrowser` over the two legs of a single FastHydro file."""
    import h5py

    from fast_data.viz import DiffBrowser

    with h5py.File(str(path), "r") as f:
        if "arr_bg" not in f:
            raise KeyError(
                f"{path} has no 'arr_bg', so it is not a FastHydro pair -- it is a plain "
                f"fast_data evolution. For those use fast_data.viz.DiffBrowser(jet, nojet) "
                f"with the two files.")
    jet, bg = _views(path)
    return DiffBrowser(jet, bg, check_ic=check_ic)


class PairBrowser:
    """A `DiffBrowser` over one file, plus the things only a paired file can answer.

    Everything `DiffBrowser` exposes is forwarded, so notebooks written against FNO4d's
    two-file pairs work unchanged apart from the constructor.
    """

    def __init__(self, path, *, check_ic=True):
        self.path = str(path)
        self._d = open_pair(path, check_ic=check_ic)
        self.jet, self.bg = self._d.jet, self._d.ref

    def __getattr__(self, name):          # diff, source_at, mach_angle, tau, x, y, attrs, ...
        return getattr(self._d, name)

    def __repr__(self):
        a = self.attrs
        return (f"<PairBrowser {self.path}: {self.nevents} event(s), "
                f"{self.nx}x{self.ny}x{self.neta}x{self.ntau}, "
                f"source={a.get('source_mode', '?')}, vertex={a.get('hard_vertex', '?')}>")

    # ------------------------------------------------------------------ pair-only
    def ic_identical(self, event=0):
        """Both legs start from the same initial condition -- the premise of the difference.

        The writer already refuses to store a mismatched pair, so this re-checks the stored
        file rather than the run.
        """
        return bool(np.array_equal(self.jet.frame(event, 0), self.bg.frame(event, 0)))

    def first_difference(self, event=0):
        """Index of the first frame where the two legs differ at all.

        Before the first deposit they must be identical to the bit, since they are the same
        solver on the same IC. A nonzero difference earlier than the first droplet's
        deposition time means something other than the jet is leaking in.
        """
        for k in range(self.ntau):
            if np.abs(self.diff(event, k)).max() > 0:
                return k
        return None

    def deposited_energy(self, event=0):
        d = self.jet.droplets(event)
        return 0.0 if d is None or not len(d) else float(d[:, 4].sum())

    # ------------------------------------------------------------------ shower/
    @property
    def has_shower(self):
        return "shower/partons" in self.jet._f

    def showers(self, event=0):
        """-> (partons, vertices, initiators) for one event, or None.

        The three come back together on purpose: a parton's ``i_src``/``i_tgt`` are row
        indices into *this event's* vertex block, so slicing either one alone silently
        produces segments drawn from the wrong vertices.  Columns are
        `showers.PARTON_COLUMNS` / `VERTEX_COLUMNS` / `INITIATOR_COLUMNS`.
        """
        h = self.jet._f
        if "shower/partons" not in h:
            return None
        out = []
        for name, off in (("partons", "parton_offsets"), ("vertices", "vertex_offsets"),
                          ("initiators", "initiator_offsets")):
            o = h["shower/" + off][:]
            out.append(h["shower/" + name][o[event]:o[event + 1]])
        return tuple(out)

    def shower_segments(self, event=0, *, milne=False):
        """-> (start, end, splits) per parton: the finished segments.

        Cartesian ``(x, y, z, t)`` by default; ``milne=True`` converts to ``(tau, x, y, eta)``
        so the shower shares the hydro frames' clock and the droplets' coordinates.  `splits`
        marks the partons whose segment actually ends somewhere -- see `showers.segments`.
        """
        from .showers import segments, to_milne

        s = self.showers(event)
        if s is None:
            return None
        a, b, sp = segments(s[0], s[1])
        return (to_milne(a), to_milne(b), sp) if milne else (a, b, sp)

    def shower_at(self, event, t):
        """The shower as it stands at lab time `t` [fm/c]: (start, tip, alive) per parton.

        `tip` is the point to draw each parton out to, and `alive` marks the ones that have
        been produced by `t` -- the rest have not happened yet and do not belong on screen.

        Three behaviours, which is the whole reason this is not a one-liner:

        * a parton that **splits** is interpolated along its own segment and stops at the
          splitting point -- both ends are stored data, so this is exact;
        * one that was **absorbed** (`showers.ABSORBED`: dropped into the medium, or missed)
          stops at its production point -- its energy is in `source/droplets` from then on,
          and drawing it onward would double-count the jet;
        * anything else -- a final-state parton, or a hole -- is still travelling when the
          graph runs out, so it is carried at ``p/E`` to `t`.  That is the only case where
          anything is extrapolated.
        """
        from .showers import ABSORBED, velocities

        s = self.showers(event)
        if s is None:
            return None
        p = s[0]
        a, b, splits = self.shower_segments(event)
        t = float(t)
        t0 = a[:, 3]

        # splitting: interpolate on the segment, exact at both ends
        span = np.where(splits, b[:, 3] - t0, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.clip(np.where(span > 0, (t - t0) / np.where(span > 0, span, 1.0), 0.0),
                        0.0, 1.0)
        # `a + 1.0*(b-a)` is not bit-exactly `b`, and a finished parton should land on the
        # stored child point, not a rounding away from it.
        tip = np.where(f[:, None] >= 1.0, b, a + f[:, None] * (b - a))

        # not splitting, not absorbed: still in flight, so extrapolate along p/E
        flying = ~splits & ~np.isin(p[:, 4].astype(int), ABSORBED)
        dt = np.where(flying, np.maximum(t - t0, 0.0), 0.0)[:, None]
        tip = np.where(flying[:, None],
                       np.column_stack([a[:, :3] + velocities(p) * dt, t0 + dt[:, 0]]), tip)
        return a, tip, t0 <= t

    def parton_fates(self, event=0):
        """-> {fate name: count} over the event's partons; see `showers.FATES`.

        ``drop`` is the interesting one: those are the partons the liquefier absorbed into the
        medium, i.e. the ones that became the droplets in `source/droplets`.
        """
        import collections

        from .showers import fates

        s = self.showers(event)
        return None if s is None else dict(collections.Counter(fates(s[0])))

    def wake_energy(self, event, itau):
        """int |de| dV over the frame [GeV] -- the size of the disturbance, wake included."""
        de = self.diff(event, itau)
        a = self.attrs
        dV = (self.tau[itau] * float(a["dx"]) * float(a["dy"]) * float(a["deta"]))
        return float(np.abs(de).sum() * dV)

    def summary(self, event=0):
        first = self.first_difference(event)
        return {
            "events": self.nevents,
            "grid": (self.nx, self.ny, self.neta, self.ntau),
            "ic_identical": self.ic_identical(event),
            "first_differing_frame": first,
            "first_differing_tau": None if first is None else float(self.tau[first]),
            "n_droplets": 0 if self.jet.droplets(event) is None
                          else int(len(self.jet.droplets(event))),
            "E_deposited_GeV": self.deposited_energy(event),
            "tau_fo_jet": float(self.jet.tau_fo[event]),
            "tau_fo_bg": float(self.bg.tau_fo[event]),
            "live_frames_both": self.live(event),
            "blob_radius_fm": self.blob_radius(),
            "source_mode": str(self.attrs.get("source_mode", "?")),
            "hard_vertex": str(self.attrs.get("hard_vertex", "?")),
            "n_showers": None if not self.has_shower
                         else int(len(self.showers(event)[2])),
            "n_partons": None if not self.has_shower
                         else int(len(self.showers(event)[0])),
            "parton_fates": self.parton_fates(event),
        }

    def close(self):
        self.jet.close()
        self.bg.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
