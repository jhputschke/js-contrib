"""Hadrons from a particlized FastHydro leg: read, store compactly, and the kinematics.

`run_particlize.py` writes the framework's ``JetScapeWriterFinalStateHadronsAscii`` file: one
block per event, holding *all* of iSS's oversamples for that event with no separator between
them.  That text file is ~50 B per hadron -- ~100 MB per event at 1000 oversamples -- so
`ascii_to_npz` converts it once into::

    pid      (N,) int32
    p        (N, 4) float32   [E, px, py, pz]  GeV
    offsets  (n_events + 1,) int64            event k is rows offsets[k]:offsets[k+1]

and `load` reads it back as a `Hadrons` object.

Errors on oversample averages
-----------------------------
Oversamples cannot be told apart inside an event, so the variance of an oversample average of
``X = sum_i w_i`` is taken as compound Poisson, ``Var = sum_i w_i^2 / N_os^2``.  That is exact
for iSS's sampling (each species' multiplicity is Poisson, momenta are independent draws) and
it is what `hist` returns alongside every histogram.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CHARGED", "SPECIES", "Hadrons", "ascii_to_npz", "load", "read_ascii"]

#: |pid| of the charged hadrons that survive iSS's decays (plus leptons from them)
CHARGED = (211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13)

#: identified species, by signed pid
SPECIES = {
    "pi+": (211,), "pi-": (-211,), "K+": (321,), "K-": (-321,),
    "p": (2212,), "pbar": (-2212,),
    "pi": (211, -211), "K": (321, -321), "p+pbar": (2212, -2212),
}


def read_ascii(path):
    """-> (pid int32, p float32 (N,4), offsets int64) from a final-state hadron file."""
    pid, mom, offsets = [], [], [0]
    n = 0
    started = False
    with open(path) as f:
        for line in f:
            if line[0] == "#":
                if "Event" in line:
                    if started:
                        offsets.append(n)
                    started = True
                continue
            t = line.split()
            if len(t) < 7:
                continue
            pid.append(int(t[1]))
            mom.append((float(t[3]), float(t[4]), float(t[5]), float(t[6])))
            n += 1
    if started:
        offsets.append(n)
    return (np.asarray(pid, dtype=np.int32),
            np.asarray(mom, dtype=np.float32).reshape(-1, 4),
            np.asarray(offsets, dtype=np.int64))


def ascii_to_npz(src, dst):
    """Convert `src` (ASCII hadron file) to the compact `dst`.  -> number of events."""
    pid, p, off = read_ascii(src)
    np.savez(dst, pid=pid, p=p, offsets=off)
    return len(off) - 1


def load(path, n_oversample):
    return Hadrons.from_npz(path, n_oversample)


class Hadrons:
    """All hadrons of one leg; per-event views and oversample-averaged histograms."""

    def __init__(self, pid, p, offsets, n_oversample):
        self.pid = np.asarray(pid)
        self.p = np.asarray(p, dtype=np.float64)
        self.offsets = np.asarray(offsets)
        self.n_os = int(n_oversample)
        E, px, py, pz = self.p.T
        self.E, self.px, self.py, self.pz = E, px, py, pz
        self.pt = np.hypot(px, py)
        pabs = np.sqrt(self.pt ** 2 + pz ** 2)
        self.eta = 0.5 * np.log(np.clip(pabs + pz, 1e-300, None) /
                                np.clip(pabs - pz, 1e-300, None))
        self.y = 0.5 * np.log(np.clip(E + pz, 1e-300, None) / np.clip(E - pz, 1e-300, None))
        self.phi = np.arctan2(py, px)
        self.charged = np.isin(np.abs(self.pid), CHARGED)
        self.event = np.repeat(np.arange(len(self.offsets) - 1), np.diff(self.offsets))

    @classmethod
    def from_npz(cls, path, n_oversample):
        d = np.load(path)
        return cls(d["pid"], d["p"], d["offsets"], n_oversample)

    @property
    def n_events(self):
        return len(self.offsets) - 1

    def species(self, name):
        """Boolean mask for a name in SPECIES, 'charged', or 'all'."""
        if name == "all":
            return np.ones(len(self.pid), bool)
        if name == "charged":
            return self.charged
        return np.isin(self.pid, SPECIES[name])

    def hist(self, values, bins, mask=None, weights=None, events=None):
        """Oversample-averaged histogram, per event averaged over `events`.

        -> (h, err): the mean over the selected events of the per-oversample histogram, and its
        compound-Poisson error.  `values` may be a tuple for an N-d histogram.
        """
        m = np.ones(len(self.pid), bool) if mask is None else mask.copy()
        if events is not None:
            m &= np.isin(self.event, np.atleast_1d(events))
            n_ev = len(np.atleast_1d(events))
        else:
            n_ev = self.n_events
        w = np.ones(m.sum()) if weights is None else np.asarray(weights)[m]
        nd = isinstance(values, tuple)
        vals = tuple(v[m] for v in values) if nd else (values[m],)
        bins = bins if nd else (bins,)
        h = np.histogramdd(vals, bins=bins, weights=w)[0]
        v = np.histogramdd(vals, bins=bins, weights=w ** 2)[0]
        norm = self.n_os * n_ev
        return h / norm, np.sqrt(v) / norm

    def total(self, mask=None, weights=None, events=None):
        """Oversample-averaged sum over hadrons (default: count), and its error."""
        m = np.ones(len(self.pid), bool) if mask is None else mask.copy()
        if events is not None:
            m &= np.isin(self.event, np.atleast_1d(events))
            n_ev = len(np.atleast_1d(events))
        else:
            n_ev = self.n_events
        w = np.ones(m.sum()) if weights is None else np.asarray(weights)[m]
        norm = self.n_os * n_ev
        return w.sum() / norm, np.sqrt((w ** 2).sum()) / norm
