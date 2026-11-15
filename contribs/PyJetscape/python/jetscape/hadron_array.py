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
    x      (n, 4)  float32   t, x, y, z    (None: no positions, e.g. a file written without)
    mass   (n,)    float32                 (None: not known, e.g. read from a hadron file)
    sample_offsets (n_samples + 1,) int64; sample s is rows sample_offsets[s]:[s+1]

The same arrays go to a hadron file (to_dict() for HadronH5Writer.append_unit) and come
back from one (from_h5: the raw fields only, no derived columns, with the units' sample
offsets and their table in `info`).
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
        self.x = None if x is None else np.asarray(x).reshape(-1, 4)
        self.mass = None if mass is None else np.asarray(mass)
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
        return cls(d["pid"], d["pstat"], d["p"], d.get("x"), d.get("mass"), offsets, info)

    @classmethod
    def from_h5(cls, path, units=None) -> "HadronArray":
        """A hadron file (hadrons_h5 format: jet_frag, bulk_bg, ...) as one HadronArray: the
        stored fields as they are (float32, no derived columns), every sample of every unit
        in file order.  ``units``: an index array or slice of unit positions (not the
        recorded event numbers: ``HadronFile.unit_index`` maps those) to read only those
        units; only their rows are read from disk (after v1.5; before, the whole file).
        For events of a campaign by their number, with samples and a hadron selection, see
        ``HadronFileReader.hadron_array`` (new after v1.5).

        info: ``unit_offsets`` (samples of unit u: unit_offsets[u]:[u+1]), ``units`` (the
        file's per-unit table: event, seed, event_weight, ...), ``attrs`` (the file's
        attributes), ``path``.  The mass is not stored (None)."""
        import h5py

        from . import h5_compression  # noqa: F401  (the Blosc filter)
        from .hadrons_h5 import FORMAT, _read_units

        with h5py.File(path, "r") as f:
            if f.attrs.get("format") != FORMAT:
                raise ValueError(f"{path}: not a {FORMAT} file")
            attrs = dict(f.attrs)
            names = ["pid", "pstat", "p"] + (["x"] if "x" in f["hadrons"] else [])
            data, so, uo, table = _read_units(f, units, names)
        info = {"unit_offsets": uo, "units": table, "attrs": attrs, "path": str(path)}
        return cls(data["pid"], data["pstat"], data["p"], data.get("x"), None, so, info)

    def to_dict(self) -> dict:
        """The bindings' hadron dict (pid, pstat, p, x, mass, sample_counts), the arrays
        themselves (no copy): what HadronH5Writer.append_unit takes for one unit."""
        d = {"pid": self.pid, "pstat": self.pstat, "p": self.p,
             "sample_counts": self.sample_counts}
        if self.x is not None:
            d["x"] = self.x
        if self.mass is not None:
            d["mass"] = self.mass
        return d

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
        """-> hadrons_h5.Hadrons (the file's units after from_h5, else all samples as one
        unit, unless ``unit_offsets`` says otherwise), for its hist/total.  Hadrons computes its
        derived columns in float64: about 2.4x the memory of this array."""
        from .hadrons_h5 import Hadrons, _no_positions

        if unit_offsets is None:
            unit_offsets = self.info.get("unit_offsets", [0, self.n_samples])
        x = self.x if self.x is not None else _no_positions(len(self))
        return Hadrons(self.pid, self.pstat, self.p, x, self.sample_offsets, unit_offsets,
                       attrs=attrs if attrs is not None else self.info.get("attrs"),
                       units=self.info.get("units"))

    def to_awkward(self, position=False, charge=False, by_unit=False, charged=False,
                   massless=False, pion_mass=False):
        """samples x hadrons as an awkward array of Momentum4D records over these buffers
        (awkward_views.to_awkward).

        Args:
            position: add the nested ``position`` (t, x, y, z).
            charge: add ``charge`` [e], the PDG charge (needs ``particle``).
            by_unit: nest by ``info["unit_offsets"]``: units x samples x hadrons.  A unit is
                what the array was read by: a file's unit for :meth:`from_h5` (an event;
                for bulk_bg a background), an event for
                ``HadronFileReader.hadron_array``.  False: samples x hadrons, all samples
                in a row.
            charged: (``charged``, ``massless`` and ``pion_mass`` are new after v1.5) add a
                boolean ``charged`` field (pid in ``hadrons_h5.CHARGED``, as
                ``mask="charged"`` of HadronFileReader).
            massless: E recomputed as |p| in float64 (a new array): massless particles.
                px, py, pz stay float32 views, so vector's ``.mass`` shows their rounding
                (~1e-4 GeV); FastJet works in float64 and sees mass 0.
            pion_mass: E recomputed in float64 (a new array) with the pion mass
                (``awkward_views.M_PION``) for the charged particles (as ``charged``,
                leptons included) and no mass for the neutral ones: every track a pion,
                every neutral massless.  Not together with ``massless``.

        Only ``charge``, ``charged``, ``massless`` and ``pion_mass`` add arrays; the rest are views of
        this array's buffers."""
        from .awkward_views import to_awkward

        return to_awkward(self, position=position, charge=charge, by_unit=by_unit,
                          charged=charged, massless=massless, pion_mass=pion_mass)
