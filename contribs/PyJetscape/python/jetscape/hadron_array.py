"""
python/jetscape/hadron_array.py

HadronArray: one event's hadrons as one array per field, grouped into samples
(iSS oversamples; one sample for jet hadronization), for the Pythonic layer
(docs/Plans/PLAN_pyjetscape_pythonic.md, step 1).

It wraps the dict the bindings return (soft_hadrons_numpy, hadronization_hadrons_numpy,
hadronize_partons, soft_sample_cells) without copying: with soft_set_compact_output
iSS's per-field vectors are already numpy arrays, and a record array would have to
interleave them (memory constraint M1).  It stores no derived columns (M2): pt, eta,
phi and rapidity come from to_awkward() as `vector` Momentum4D records, computed when
used.  to_hadrons() converts to hadrons_h5.Hadrons when its hist/total are wanted.

    pid    (n,)    int32
    pstat  (n,)    int32
    p      (n, 4)  float32   E, px, py, pz
    x      (n, 4)  float32   t, x, y, z
    mass   (n,)    float32
    sample_offsets (n_samples + 1,) int64; sample s is rows sample_offsets[s]:[s+1]
"""

from __future__ import annotations

import numpy as np

__all__ = ["HadronArray"]


class HadronArray:
    """One event's hadrons, flat per field, with sample offsets.  numpy only."""

    __slots__ = ("pid", "pstat", "p", "x", "mass", "sample_offsets", "info")

    def __init__(self, pid, pstat, p, x, mass, sample_offsets=None, info=None):
        self.pid = np.asarray(pid)
        self.pstat = np.asarray(pstat)
        self.p = np.asarray(p).reshape(-1, 4)
        self.x = np.asarray(x).reshape(-1, 4)
        self.mass = np.asarray(mass)
        n = len(self.pid)
        if sample_offsets is None:
            sample_offsets = [0, n]
        self.sample_offsets = np.asarray(sample_offsets, dtype=np.int64)
        if self.sample_offsets[0] != 0 or self.sample_offsets[-1] != n:
            raise ValueError(f"sample_offsets must run from 0 to {n}, got "
                             f"{self.sample_offsets[0]}..{self.sample_offsets[-1]}")
        #: what came with the hadrons besides them (e.g. hadronize's seed, n_partons_out)
        self.info = dict(info or {})

    @classmethod
    def from_dict(cls, d) -> "HadronArray":
        """From a bindings hadron dict; the arrays are taken as they are (no copy).
        ``sample_counts`` (one per oversample) becomes ``sample_offsets``; without it the
        hadrons are one sample.  Other keys go to ``info``."""
        counts = d.get("sample_counts")
        offsets = None
        if counts is not None:
            offsets = np.zeros(len(counts) + 1, dtype=np.int64)
            np.cumsum(counts, out=offsets[1:])
        fields = ("pid", "pstat", "p", "x", "mass", "sample_counts")
        info = {k: v for k, v in d.items() if k not in fields}
        return cls(d["pid"], d["pstat"], d["p"], d["x"], d["mass"], offsets, info)

    def __len__(self) -> int:
        return len(self.pid)

    def __repr__(self) -> str:
        return f"HadronArray({len(self)} hadrons in {self.n_samples} samples)"

    @property
    def n_samples(self) -> int:
        return len(self.sample_offsets) - 1

    @property
    def sample_counts(self) -> np.ndarray:
        """Hadrons per sample, (n_samples,) int64."""
        return np.diff(self.sample_offsets)

    def to_hadrons(self, unit_offsets=None, attrs=None):
        """-> hadrons_h5.Hadrons (all samples of this event as one unit unless
        ``unit_offsets`` says otherwise), for its hist/total.  Hadrons computes its
        derived columns in float64: about 2.4x the memory of this array."""
        from .hadrons_h5 import Hadrons

        if unit_offsets is None:
            unit_offsets = [0, self.n_samples]
        return Hadrons(self.pid, self.pstat, self.p, self.x, self.sample_offsets,
                       unit_offsets, attrs=attrs)

    def to_awkward(self, position=False, charge=False):
        """samples x hadrons as an awkward array of Momentum4D records over these buffers
        (awkward_views.to_awkward)."""
        from .awkward_views import to_awkward

        return to_awkward(self, position=position, charge=charge)
