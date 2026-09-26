"""
python/jetscape/hadrons_h5.py

Hadrons in HDF5, with every sample kept apart: HadronH5Writer writes, Hadrons reads.

One file holds one kind of hadrons (``tag``), e.g. from example/prod_AuAu_0_10_jet/
hadronize.py::

    bulk_jet   iSS on the jet leg's surface     (bulk + wake)       unit = event
    bulk_bg    iSS on the background's surface  (bulk)              unit = background
    jet_frag   ColorlessHadronization on the final partons          unit = event

Layout::

    attrs   format, format_version, tag, n_samples (per unit, nominal), source (the
            particlize file) + its file_uuid, generator settings ...
    hadrons/pid            (N,)   int32
    hadrons/pstat          (N,)   int32
    hadrons/p              (N, 4) float32   [E, px, py, pz]  GeV
    hadrons/x              (N, 4) float32   [t, x, y, z]     fm   (zero for Pythia)
    hadrons/sample_offsets (S + 1,) int64   sample s is rows sample_offsets[s]:[s+1]
    hadrons/unit_offsets   (U + 1,) int64   unit u is samples unit_offsets[u]:[u+1]
    units/<key>            (U,)             unit (row in the source group), event (first
                                            event using it), seed, n_cells ...

Two levels of offsets because an oversample is a sample of the whole event: averages are
over samples, and the samples of one unit are not independent events.  Errors on sample
averages use the compound-Poisson estimate of ``fasthydro.hadrons`` (``Var = sum w^2 /
N_samples^2``), which is exact for iSS's sampling.

Writing is incremental (RaggedGroup): a crash loses nothing already appended.

Single events
-------------
Every sample is a complete event of that tag and can be taken alone::

    h = Hadrons.from_h5(path);   ev = h.sample_event(unit, k)    # in memory
    hf = HadronFile(path);       ev = hf.sample_event(unit, k)   # read from disk, lazily

``unit`` is the recorded unit (``units/unit``: the event for bulk_jet and jet_frag, the
background for bulk_bg); ``k`` counts that unit's samples from 0.  :class:`JetEvents` puts
the three tags of one production file together: ``jet_event(event, k)`` is oversample ``k``
of the jet leg's bulk plus a fragmentation of the same event's partons, and
``background_event(event, k)`` is oversample ``k`` of the background that event used.
Oversamples of one event share one fluid: they are independent Cooper-Frye samplings, not
independent collisions.
"""

from __future__ import annotations

import os

import numpy as np

from .fno_h5_writer import RaggedGroup
from .h5_compression import DEFAULT as DEFAULT_COMPRESSION

__all__ = ["FORMAT", "FORMAT_VERSION", "TAGS", "CHARGED", "SPECIES", "FIELDS", "ORIGIN",
           "HadronH5Writer", "Hadrons", "HadronFile", "JetEvents"]

#: per-hadron arrays of one event (a sample), as sample_event() returns them
FIELDS = ("pid", "pstat", "p", "x")

#: JetEvents.jet_event(): value of the ``origin`` array per source
ORIGIN = {"bulk": 0, "frag": 1}

FORMAT = "xscape/hadrons"
FORMAT_VERSION = 1

TAGS = ("bulk_jet", "bulk_bg", "jet_frag")

#: |pid| of the charged hadrons that survive iSS's decays (plus leptons from them)
CHARGED = (211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13)

#: identified species, by signed pid
SPECIES = {
    "pi+": (211,), "pi-": (-211,), "K+": (321,), "K-": (-321,),
    "p": (2212,), "pbar": (-2212,),
    "pi": (211, -211), "K": (321, -321), "p+pbar": (2212, -2212),
}


class HadronH5Writer:
    """Append units of samples of hadrons (see the module docstring)."""

    def __init__(self, path, *, tag, n_samples, attrs=None, compression=DEFAULT_COMPRESSION,
                 force=True):
        import h5py

        if tag not in TAGS:
            raise ValueError(f"tag must be one of {TAGS}, got {tag!r}")
        self.path = str(path)
        if os.path.exists(self.path) and not force:
            raise FileExistsError(f"{self.path} exists (pass force=True)")
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        self.f = h5py.File(self.path, "w")
        a = self.f.attrs
        a["format"], a["format_version"] = FORMAT, FORMAT_VERSION
        a["producer"] = "js-contrib/contribs/PyJetscape (jetscape.hadrons_h5)"
        a["tag"] = tag
        a["n_samples"] = int(n_samples)
        a["p_columns"] = ["E", "px", "py", "pz"]
        a["x_columns"] = ["t", "x", "y", "z"]
        a["units"] = "p in GeV; x in fm (t in fm/c)"
        a["nunits_written"] = 0
        a["complete"] = False
        for k, v in (attrs or {}).items():
            a[k] = v
        g = self.f.require_group("hadrons")
        self._h = RaggedGroup(g, "sample_offsets", {
            "pid": (np.int32, ()), "pstat": (np.int32, ()),
            "p": (np.float32, (4,)), "x": (np.float32, (4,))},
            compression=compression, chunk_rows=1 << 17, unit="free")
        self._unit_off = g.create_dataset("unit_offsets", data=np.zeros(1, dtype=np.int64),
                                          maxshape=(None,), chunks=(4096,))
        from .particlize_h5 import _ScalarTable
        self._units = _ScalarTable(self.f.require_group("units"))
        self._u = 0

    def append_unit(self, samples, **unit_scalars):
        """Append one unit.  ``samples`` is a list of hadron dicts (one per sample, keys
        pid, pstat, p, x), or one dict with ``sample_counts`` (as soft_hadrons_numpy
        returns).  Returns the unit index."""
        if isinstance(samples, dict):
            counts = np.asarray(samples.get("sample_counts", [len(samples["pid"])]),
                                dtype=np.int64)
            cuts = np.concatenate([[0], np.cumsum(counts)])
            samples = [{k: np.asarray(samples[k])[cuts[s]:cuts[s + 1]]
                        for k in ("pid", "pstat", "p", "x")} for s in range(len(counts))]
        for s in samples:
            self._h.append({k: s[k] for k in ("pid", "pstat", "p", "x")})
        n = self._unit_off.shape[0]
        self._unit_off.resize((n + 1,))
        self._unit_off[n] = self._h.units_written
        u = self._u
        self._units.append(u, dict(unit_scalars, n_samples=len(samples)))
        self._u += 1
        self.f.attrs["nunits_written"] = self._u
        self.f.flush()
        return u

    def close(self, complete=True):
        if self.f is None:
            return
        self.f.attrs["complete"] = bool(complete)
        self.f.close()
        self.f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close(complete=exc[0] is None)
        return False


class Hadrons:
    """All hadrons of one file, with per-unit and per-sample structure and kinematics.

        h = Hadrons.from_h5("hadrons_bulk_jet.h5")
        dndy, err = h.hist(h.y, np.linspace(-1, 1, 11), mask=h.species("charged"))

    ``hist`` / ``total`` return the average over the selected units of the per-sample
    quantity, i.e. sum of weights divided by the number of samples in those units, and
    its compound-Poisson error.
    """

    def __init__(self, pid, pstat, p, x, sample_offsets, unit_offsets, attrs=None,
                 units=None):
        self.attrs = dict(attrs or {})
        self.units = dict(units or {})
        self.pid = np.asarray(pid)
        self.pstat = np.asarray(pstat)
        self.p = np.asarray(p, dtype=np.float64)
        self.x = np.asarray(x)
        self.sample_offsets = np.asarray(sample_offsets, dtype=np.int64)
        self.unit_offsets = np.asarray(unit_offsets, dtype=np.int64)
        E, px, py, pz = self.p.T if len(self.p) else (np.zeros(0),) * 4
        self.E, self.px, self.py, self.pz = E, px, py, pz
        self.pt = np.hypot(px, py)
        pabs = np.sqrt(self.pt ** 2 + pz ** 2)
        self.eta = 0.5 * np.log(np.clip(pabs + pz, 1e-300, None) /
                                np.clip(pabs - pz, 1e-300, None))
        self.y = 0.5 * np.log(np.clip(E + pz, 1e-300, None) / np.clip(E - pz, 1e-300, None))
        self.phi = np.arctan2(py, px)
        self.charged = np.isin(np.abs(self.pid), CHARGED)
        n_samples = np.diff(self.sample_offsets)
        self.sample = np.repeat(np.arange(len(n_samples)), n_samples)
        per_unit = np.diff(self.unit_offsets)
        sample_unit = np.repeat(np.arange(len(per_unit)), per_unit)
        self.unit = sample_unit[self.sample] if len(self.sample) else np.zeros(0, int)
        self.samples_per_unit = per_unit

    @classmethod
    def from_h5(cls, path, units=None):
        """Read a file (``units``: an index array/slice to keep only those units)."""
        import h5py

        from . import h5_compression  # noqa: F401  (Blosc filter)

        with h5py.File(path, "r") as f:
            if f.attrs.get("format") != FORMAT:
                raise ValueError(f"{path}: not a {FORMAT} file")
            g = f["hadrons"]
            so = g["sample_offsets"][:]
            uo = g["unit_offsets"][:]
            attrs = dict(f.attrs)
            utab = {k: f["units"][k][:] for k in f["units"]} if "units" in f else {}
            if units is None:
                pid, pstat = g["pid"][:], g["pstat"][:]
                p, x = g["p"][:], g["x"][:]
            else:
                sel = np.arange(len(uo) - 1)[units]
                rows, s_new, u_new = [], [0], [0]
                for u in sel:
                    s0, s1 = int(uo[u]), int(uo[u + 1])
                    for s in range(s0, s1):
                        rows.append((int(so[s]), int(so[s + 1])))
                        s_new.append(s_new[-1] + int(so[s + 1] - so[s]))
                    u_new.append(u_new[-1] + (s1 - s0))
                idx = (np.concatenate([np.arange(a, b) for a, b in rows])
                       if rows else np.zeros(0, dtype=np.int64))
                pid, pstat = g["pid"][:][idx], g["pstat"][:][idx]
                p, x = g["p"][:][idx], g["x"][:][idx]
                so, uo = np.asarray(s_new), np.asarray(u_new)
                utab = {k: v[sel] for k, v in utab.items()}
        return cls(pid, pstat, p, x, so, uo, attrs=attrs, units=utab)

    @property
    def n_units(self):
        return len(self.unit_offsets) - 1

    def unit_index(self, unit):
        """Position of recorded unit ``unit`` (``units/unit``) among the loaded units."""
        return _unit_index(self.units.get("unit"), unit, self.n_units)

    def n_samples(self, unit):
        """Number of samples (oversamples / fragmentations) of recorded unit ``unit``."""
        return int(self.samples_per_unit[self.unit_index(unit)])

    def sample_event(self, unit, k):
        """Sample ``k`` of recorded unit ``unit`` as one event: dict of pid, pstat, p
        [E, px, py, pz], x [t, x, y, z]."""
        u = self.unit_index(unit)
        s = _sample_of(self.unit_offsets, u, k, unit)
        a, b = int(self.sample_offsets[s]), int(self.sample_offsets[s + 1])
        return {"pid": self.pid[a:b], "pstat": self.pstat[a:b],
                "p": self.p[a:b].astype(np.float32), "x": self.x[a:b]}

    def species(self, name):
        """Boolean mask for a name in SPECIES, 'charged', or 'all'."""
        if name == "all":
            return np.ones(len(self.pid), bool)
        if name == "charged":
            return self.charged
        return np.isin(self.pid, SPECIES[name])

    def _select(self, mask, units):
        m = np.ones(len(self.pid), bool) if mask is None else np.asarray(mask).copy()
        if units is None:
            n_samples = int(self.samples_per_unit.sum())
        else:
            u = np.atleast_1d(units)
            m &= np.isin(self.unit, u)
            n_samples = int(self.samples_per_unit[u].sum())
        return m, max(n_samples, 1)

    def hist(self, values, bins, mask=None, weights=None, units=None):
        """Sample-averaged histogram over ``units`` (default all), and its error.
        ``values`` may be a tuple for an N-d histogram."""
        m, norm = self._select(mask, units)
        w = np.ones(m.sum()) if weights is None else np.asarray(weights)[m]
        nd = isinstance(values, tuple)
        vals = tuple(v[m] for v in values) if nd else (values[m],)
        bins = bins if nd else (bins,)
        h = np.histogramdd(vals, bins=bins, weights=w)[0]
        v = np.histogramdd(vals, bins=bins, weights=w ** 2)[0]
        return h / norm, np.sqrt(v) / norm

    def total(self, mask=None, weights=None, units=None):
        """Sample-averaged sum over hadrons (default: count), and its error."""
        m, norm = self._select(mask, units)
        w = np.ones(m.sum()) if weights is None else np.asarray(weights)[m]
        return w.sum() / norm, np.sqrt((w ** 2).sum()) / norm


class HadronFile:
    """A hadron file opened for random access: single samples are read from disk, so a
    campaign-sized file is never loaded whole.

        with HadronFile("…_hadrons_bulk_jet.h5") as hf:
            ev = hf.sample_event(3, 17)       # oversample 17 of event 3
    """

    def __init__(self, path):
        import h5py

        from . import h5_compression  # noqa: F401  (Blosc filter)

        self.path = str(path)
        self.f = h5py.File(self.path, "r")
        if self.f.attrs.get("format") != FORMAT:
            raise ValueError(f"{self.path}: not a {FORMAT} file")
        self.attrs = dict(self.f.attrs)
        self.tag = str(self.attrs.get("tag", ""))
        g = self.f["hadrons"]
        self._g = g
        self.sample_offsets = g["sample_offsets"][:]
        self.unit_offsets = g["unit_offsets"][:]
        self.units = ({k: self.f["units"][k][:] for k in self.f["units"]}
                      if "units" in self.f else {})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None

    @property
    def n_units(self):
        return len(self.unit_offsets) - 1

    def unit_index(self, unit):
        return _unit_index(self.units.get("unit"), unit, self.n_units)

    def n_samples(self, unit):
        u = self.unit_index(unit)
        return int(self.unit_offsets[u + 1] - self.unit_offsets[u])

    def sample_event(self, unit, k):
        """Sample ``k`` of recorded unit ``unit``: dict of pid, pstat, p, x (see FIELDS)."""
        u = self.unit_index(unit)
        s = _sample_of(self.unit_offsets, u, k, unit)
        a, b = int(self.sample_offsets[s]), int(self.sample_offsets[s + 1])
        return {name: self._g[name][a:b] for name in FIELDS}


class JetEvents:
    """The hadron files of one production file, combined per event.

        with JetEvents.from_stem("out/AuAu_0_10_jet_seed0001") as je:
            ev = je.jet_event(0, 17)          # bulk_jet oversample 17 + a fragmentation
            bg = je.background_event(0, 17)   # the same event's background, oversample 17

    ``jet_event`` concatenates the bulk hadrons and the fragments and adds an ``origin``
    array (``ORIGIN``: 0 bulk, 1 jet fragment).  The fragmentation used for oversample
    ``k`` is ``frag_sample`` if given, else ``k mod n_frag`` -- with ``--n-frag`` equal to
    ``--oversample`` every oversample gets its own.  The background an event used comes from
    the particlize file (``events/bg_unit``), which is what makes ``--reuse`` work; without
    it only events that started a background can be looked up.
    """

    def __init__(self, bulk_jet=None, jet_frag=None, bulk_bg=None, particlize=None):
        self.bulk_jet = HadronFile(bulk_jet) if bulk_jet else None
        self.jet_frag = HadronFile(jet_frag) if jet_frag else None
        self.bulk_bg = HadronFile(bulk_bg) if bulk_bg else None
        self._bg_unit = None
        if particlize:
            from .particlize_h5 import ParticlizeFile
            with ParticlizeFile(particlize) as pf:
                if pf.has_events("bg_unit"):
                    self._bg_unit = pf.events("bg_unit")
        for name in ("bulk_jet", "jet_frag", "bulk_bg"):
            hf = getattr(self, name)
            if hf is not None and hf.tag != name:
                raise ValueError(f"{hf.path} holds tag {hf.tag!r}, not {name!r}")

    @classmethod
    def from_stem(cls, stem):
        """The files hadronize.py writes for ``<stem>_particlize.h5`` (missing ones are
        skipped)."""
        import os

        def have(p):
            return p if os.path.exists(p) else None

        return cls(have(f"{stem}_hadrons_bulk_jet.h5"), have(f"{stem}_hadrons_jet_frag.h5"),
                   have(f"{stem}_hadrons_bulk_bg.h5"), have(f"{stem}_particlize.h5"))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        for hf in (self.bulk_jet, self.jet_frag, self.bulk_bg):
            if hf is not None:
                hf.close()

    def _need(self, name):
        hf = getattr(self, name)
        if hf is None:
            raise ValueError(f"no {name} file")
        return hf

    def n_oversamples(self, event):
        """iSS oversamples of ``event``'s jet leg (0: its surface was empty)."""
        return self._need("bulk_jet").n_samples(event)

    def n_frag(self, event):
        return self._need("jet_frag").n_samples(event)

    def bg_unit(self, event):
        """The background unit ``event`` used."""
        if self._bg_unit is not None:
            return int(self._bg_unit[event])
        bg = self._need("bulk_bg")
        first = bg.units.get("event")
        hits = np.flatnonzero(np.asarray(first) == event) if first is not None else []
        if len(hits) != 1:
            raise ValueError(f"event {event} did not start a background; pass the "
                             "particlize file to look up reused backgrounds")
        return int(bg.units["unit"][hits[0]])

    def jet_event(self, event, k, frag_sample=None, fragments=True):
        """Oversample ``k`` of ``event``'s jet-leg bulk plus one fragmentation of its
        partons: dict of pid, pstat, p, x and origin (0 bulk, 1 fragment)."""
        bulk = self._need("bulk_jet").sample_event(event, k)
        parts = [(bulk, ORIGIN["bulk"])]
        if fragments and self.jet_frag is not None:
            n = self.jet_frag.n_samples(event)
            if n:
                j = (k % n) if frag_sample is None else int(frag_sample)
                parts.append((self.jet_frag.sample_event(event, j), ORIGIN["frag"]))
        return _concat(parts)

    def background_event(self, event, k):
        """Oversample ``k`` of the background ``event`` used: dict of pid, pstat, p, x."""
        return self._need("bulk_bg").sample_event(self.bg_unit(event), k)

    def iter_jet_events(self, event, frag_sample=None):
        """All oversamples of ``event`` as jet events (k = 0 .. n_oversamples - 1)."""
        for k in range(self.n_oversamples(event)):
            yield self.jet_event(event, k, frag_sample=frag_sample)


# ── helpers ─────────────────────────────────────────────────────────────────────
def _unit_index(recorded, unit, n_units):
    """Position of recorded unit id ``unit``; falls back to the position itself."""
    if recorded is None or len(recorded) == 0:
        if not 0 <= int(unit) < n_units:
            raise IndexError(f"unit {unit} out of range (0..{n_units - 1})")
        return int(unit)
    hits = np.flatnonzero(np.asarray(recorded) == int(unit))
    if len(hits) == 0:
        raise IndexError(f"no unit {unit} in this file (units: {list(recorded)[:10]}"
                         f"{' ...' if len(recorded) > 10 else ''})")
    return int(hits[0])


def _sample_of(unit_offsets, u, k, unit):
    n = int(unit_offsets[u + 1] - unit_offsets[u])
    if not 0 <= int(k) < n:
        raise IndexError(f"unit {unit} has {n} sample(s); asked for sample {k}"
                         + (" (an empty surface: no samples)" if n == 0 else ""))
    return int(unit_offsets[u]) + int(k)


def _concat(parts):
    out = {name: np.concatenate([np.asarray(ev[name]) for ev, _ in parts])
           for name in FIELDS}
    out["origin"] = np.concatenate([np.full(len(ev["pid"]), o, dtype=np.int8)
                                    for ev, o in parts])
    return out

