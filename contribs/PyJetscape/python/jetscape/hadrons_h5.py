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

``p`` and ``x`` are full float32 unless the writer was given ``keep_bits``: then they are
rounded to that many mantissa bits (``keep_mantissa_bits`` / ``max_rel_error`` on the
dataset, :func:`hadron_precision` reads them back); pid, pstat and the offsets are always
exact.  Rounding is for campaign storage: e.g. ``keep_bits={"p": 12, "x": 8}`` keeps 58% of
the bytes, with relative errors <= 1.2e-4 (p) and 2e-3 (x).  The inputs (particlize file +
``units/seed``) reproduce the exact hadrons at any time.

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

A whole campaign
----------------
:class:`HadronFileReader` reads the files of many production files (one per seed) as one
data set, the way the hydro files of a campaign are read side by side rather than merged:
events get a global index, ``jet_event``/``background_event`` work across seeds, and
``hist``/``total``/``jet_minus_background`` accumulate event by event over all files, with
the background re-evaluated per event (jet-relative observables) and correct errors when a
background is reused.
"""

from __future__ import annotations

import os

import numpy as np

from .fno_h5_writer import RaggedGroup
from .h5_compression import DEFAULT as DEFAULT_COMPRESSION
from .h5_compression import dataset_keep_bits, h5_filter_kwargs, round_mantissa, tag_dataset

__all__ = ["FORMAT", "FORMAT_VERSION", "TAGS", "CHARGED", "SPECIES", "FIELDS", "ORIGIN",
           "ROUNDABLE", "hadron_precision", "HadronH5Writer", "Hadrons", "HadronFile",
           "JetEvents", "HadronFileReader",
           "EventHadrons", "EventInfo"]

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


#: the float fields ``keep_bits`` may round; pid, pstat and the offsets stay exact
ROUNDABLE = ("p", "x")


def _keep_bits_map(keep_bits):
    """None, an int (p and x alike) or a mapping {"p": bits, "x": bits} -> {field: bits},
    with None for lossless (also for 23 bits, the full float32 mantissa)."""
    if keep_bits is None or isinstance(keep_bits, (int, np.integer)):
        m = dict.fromkeys(ROUNDABLE, keep_bits)
    else:
        m = dict(keep_bits)
        bad = set(m) - set(ROUNDABLE)
        if bad:
            raise ValueError(f"keep_bits: only {ROUNDABLE} can be rounded, got {sorted(bad)}")
    out = {}
    for k in ROUNDABLE:
        v = m.get(k)
        if v is not None:
            v = int(v)
            if not 1 <= v <= 23:
                raise ValueError(f"keep_bits[{k!r}] must be 1..23, got {v}")
        out[k] = None if v in (None, 23) else v
    return out


def hadron_precision(f):
    """``{"p": bits, "x": bits}`` a hadron file (path or open h5py.File) was written with;
    None means full float32."""
    import h5py

    if not isinstance(f, h5py.File):
        with h5py.File(f, "r") as h:
            return hadron_precision(h)
    return {k: dataset_keep_bits(f["hadrons"][k]) for k in ROUNDABLE}


class HadronH5Writer:
    """Append units of samples of hadrons (see the module docstring).

    ``keep_bits`` (default None: lossless) rounds ``p`` and ``x`` before they are written:
    an int for both, or a mapping such as ``{"p": 12, "x": 8}``.  The precision is recorded
    on each dataset (:func:`hadron_precision`)."""

    def __init__(self, path, *, tag, n_samples, attrs=None, compression=DEFAULT_COMPRESSION,
                 force=True, keep_bits=None):
        import h5py

        if tag not in TAGS:
            raise ValueError(f"tag must be one of {TAGS}, got {tag!r}")
        self.keep_bits = _keep_bits_map(keep_bits)
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
        if any(v is not None for v in self.keep_bits.values()):
            import warnings
            with warnings.catch_warnings():         # RaggedGroup already warned, if at all
                warnings.simplefilter("ignore")
                label = h5_filter_kwargs(compression)[1]
            for k, v in self.keep_bits.items():
                tag_dataset(g[k], label, v)
        self._unit_off = g.create_dataset("unit_offsets", data=np.zeros(1, dtype=np.int64),
                                          maxshape=(None,), chunks=(4096,))
        from .particlize_h5 import _ScalarTable
        self._units = _ScalarTable(self.f.require_group("units"))
        self._u = 0

    def append_unit(self, samples, **unit_scalars):
        """Append one unit.  ``samples`` is a list of hadron dicts (one per sample, keys
        pid, pstat, p, x), or one dict with ``sample_counts`` (as soft_hadrons_numpy
        returns).  Returns the unit index."""
        keys = ("pid", "pstat", "p", "x")
        if isinstance(samples, dict):
            counts = np.asarray(samples.get("sample_counts", [len(samples["pid"])]),
                                dtype=np.int64)
            rows = {k: samples[k] for k in keys}
        else:
            counts = [len(s["pid"]) for s in samples]
            rows = {k: np.concatenate([np.asarray(s[k]) for s in samples])
                    for k in keys} if samples else None
        if rows is not None:
            for k, bits in self.keep_bits.items():
                if bits is not None:
                    rows[k] = round_mantissa(rows[k], bits)
        # all samples in one write: one append per sample recompresses the partly
        # filled last chunk every time (most of hadronize.py's write time)
        self._h.append_many(rows, counts)
        n_samples = len(counts)
        n = self._unit_off.shape[0]
        self._unit_off.resize((n + 1,))
        self._unit_off[n] = self._h.units_written
        u = self._u
        self._units.append(u, dict(unit_scalars, n_samples=n_samples))
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


# ── a whole campaign ────────────────────────────────────────────────────────────
class EventHadrons:
    """All samples of one tag for one event: per-hadron arrays with kinematics.

    Attributes: ``pid``, ``pstat``, ``p`` [E, px, py, pz], ``x`` [t, x, y, z], ``E``,
    ``px``, ``py``, ``pz``, ``pt``, ``eta``, ``y``, ``phi``, ``charged``, ``sample`` (the
    sample each hadron belongs to, 0 .. n_samples-1) and ``n_samples``.
    """

    _ARRAYS = ("pid", "pstat", "p", "x", "E", "px", "py", "pz", "pt", "eta", "y", "phi",
               "charged")

    def __init__(self, h, u):
        s0, s1 = int(h.unit_offsets[u]), int(h.unit_offsets[u + 1])
        a, b = int(h.sample_offsets[s0]), int(h.sample_offsets[s1])
        for name in self._ARRAYS:
            setattr(self, name, getattr(h, name)[a:b])
        self.n_samples = s1 - s0
        self.sample = h.sample[a:b] - s0

    def __len__(self):
        return len(self.pid)

    def species(self, name):
        """Boolean mask for a name in SPECIES, 'charged', or 'all'."""
        if name == "all":
            return np.ones(len(self.pid), bool)
        if name == "charged":
            return self.charged
        return np.isin(self.pid, SPECIES[name])


class EventInfo:
    """What is known about one event of a campaign (passed to the callables of
    :meth:`HadronFileReader.hist`)."""

    def __init__(self, reader, g, i, local):
        self._reader = reader
        self.event = int(g)                  # global event index in the campaign
        self.file_index = int(i)
        self.local_event = int(local)        # event index inside its production file
        f = reader._files[i]
        self.stem = f["stem"]
        self.seed = f["seed"]
        self.bg_unit = int(f["bg_unit"][local]) if f["bg_unit"] is not None else None

    def initiators(self):
        """This event's shower-initiating partons from the pair file (``shower/``), as a
        (K, 11) array, columns ``shower, pid, pstat, px, py, pz, E, x, y, z, t``."""
        return self._reader._initiators(self.file_index, self.local_event)

    def __repr__(self):
        return (f"EventInfo(event={self.event}, stem={self.stem!r}, "
                f"local_event={self.local_event}, bg_unit={self.bg_unit})")


class HadronFileReader:
    """The hadron files of a whole campaign, read as one data set.

        reader = HadronFileReader("out_had")            # every *_particlize.h5 in there
        reader.n_events                                  # events of all seeds
        ev = reader.jet_event(123, 17)                   # global event 123, oversample 17
        dN, err = reader.hist("bulk_jet", "pt", np.linspace(0, 3, 31), mask="charged")
        dN, err = reader.jet_minus_background(dphi_jet, bins, mask=soft_charged)

    ``source`` is a directory, a glob pattern (of particlize or hadron files), or a list of
    stems / particlize paths.  Every production file needs its ``<stem>_particlize.h5``
    (event count, the event -> background map, the file uuid) and whichever
    ``<stem>_hadrons_<tag>.h5`` hadronize.py made.  With ``check_uuid`` a hadron file whose
    ``source_uuid`` is not its particlize file's ``file_uuid`` is refused: files renamed or
    mixed across runs cannot pair up silently.

    **Histograms** are event averages of per-sample means: for every event the samples of
    its unit are histogrammed and divided by that unit's number of samples, and these
    per-event means are averaged over the events -- every event weighs the same, even when
    the files were hadronized with different ``--oversample``.  Errors are compound-Poisson
    per event, added over events.  ``values``, ``mask`` and ``weights`` are either
    a name (an :class:`EventHadrons` attribute, or 'charged', or a species name for
    ``mask``) or a callable ``f(ev, info)`` returning one value per hadron, where ``ev`` is
    an :class:`EventHadrons` and ``info`` an :class:`EventInfo` -- so jet-relative
    observables use ``info.initiators()``.  ``values`` may return a tuple for an N-d
    histogram.  A background reused by several events is evaluated once per event (with
    that event's ``info``), and its hadrons, counted several times, enter the error as the
    correlated sum they are.  Events whose unit has no samples (an empty surface) are
    skipped.
    """

    def __init__(self, source, *, check_uuid=True):
        self._files = []
        for stem in _find_stems(source):
            self._files.append(self._open_stem(stem, check_uuid))
        if not self._files:
            raise FileNotFoundError(f"no *_particlize.h5 found for {source!r}")
        for tag in TAGS:
            precs = {tuple(f["precision"][tag].items()) for f in self._files
                     if tag in f["precision"]}
            if len(precs) > 1:
                import warnings
                seen = ", ".join(str(dict(p)) for p in sorted(precs, key=str))
                warnings.warn(f"HadronFileReader: the {tag} files were written with different "
                              f"precision ({seen}; None = full float32): keep one --keep-bits "
                              "setting per campaign", RuntimeWarning, stacklevel=2)
        n = np.array([f["nevents"] for f in self._files], dtype=np.int64)
        self._offsets = np.concatenate([[0], np.cumsum(n)])
        self._samples = {}              # (file index, tag) -> {unit id: n_samples}
        self._init_cache = {}
        self._je = (None, None)         # (file index, JetEvents) of the last lookup
        self._loaded = (None, None, None, None)   # (file, tag, unit positions, Hadrons)

    # ── discovery ───────────────────────────────────────────────────────────────
    @staticmethod
    def _open_stem(stem, check_uuid):
        import os

        import h5py

        from . import h5_compression  # noqa: F401
        from .particlize_h5 import ParticlizeFile

        with ParticlizeFile(f"{stem}_particlize.h5") as pf:
            uuid = str(pf.attrs.get("file_uuid", ""))
            info = {"stem": stem, "nevents": pf.nevents, "uuid": uuid,
                    "seed": pf.attrs.get("prod_seed"),
                    "pair_file": pf.attrs.get("pair_file"),
                    "bg_unit": (pf.events("bg_unit") if pf.has_events("bg_unit")
                                else None)}
        info["tags"], info["precision"] = {}, {}
        for tag in TAGS:
            path = f"{stem}_hadrons_{tag}.h5"
            if not os.path.exists(path):
                continue
            with h5py.File(path, "r") as f:
                ftag, src = str(f.attrs.get("tag", "")), str(f.attrs.get("source_uuid", ""))
                info["precision"][tag] = hadron_precision(f)
            if ftag != tag:
                raise ValueError(f"{path} holds tag {ftag!r}, not {tag!r}")
            if check_uuid and src != uuid:
                raise ValueError(f"{path} was made from particlize file uuid {src!r}, but "
                                 f"{stem}_particlize.h5 is {uuid!r}: not the same run "
                                 "(pass check_uuid=False to override)")
            info["tags"][tag] = path
        return info

    # ── bookkeeping ─────────────────────────────────────────────────────────────
    @property
    def n_events(self):
        return int(self._offsets[-1])

    @property
    def n_files(self):
        return len(self._files)

    @property
    def stems(self):
        return [f["stem"] for f in self._files]

    def precision(self, file_index):
        """``{tag: {"p": bits, "x": bits}}`` of one production file (None = full float32)."""
        return dict(self._files[file_index]["precision"])

    def tags(self, file_index=None):
        """Tags present in every file (or in one file)."""
        if file_index is not None:
            return tuple(t for t in TAGS if t in self._files[file_index]["tags"])
        return tuple(t for t in TAGS if all(t in f["tags"] for f in self._files))

    def locate(self, event):
        """Global event index -> (file index, event inside that production file)."""
        g = int(event)
        if not 0 <= g < self.n_events:
            raise IndexError(f"event {g} out of range (0..{self.n_events - 1})")
        i = int(np.searchsorted(self._offsets, g, side="right") - 1)
        return i, g - int(self._offsets[i])

    def global_event(self, file_index, local_event):
        return int(self._offsets[file_index]) + int(local_event)

    def event_info(self, event):
        i, local = self.locate(event)
        return EventInfo(self, event, i, local)

    def events(self, selection=None):
        """Global event indices: all (None), one int, a slice, or an iterable."""
        if selection is None:
            return np.arange(self.n_events)
        if isinstance(selection, slice):
            return np.arange(self.n_events)[selection]
        g = np.unique(np.atleast_1d(np.asarray(selection, dtype=np.int64)))
        if len(g) and (g[0] < 0 or g[-1] >= self.n_events):
            raise IndexError(f"events out of range (0..{self.n_events - 1})")
        return g

    def _path(self, i, tag):
        path = self._files[i]["tags"].get(tag)
        if path is None:
            raise ValueError(f"{self._files[i]['stem']}: no {tag} file")
        return path

    def _unit_of(self, i, tag, local):
        if tag != "bulk_bg":
            return int(local)
        bg = self._files[i]["bg_unit"]
        if bg is None:
            raise ValueError(f"{self._files[i]['stem']}_particlize.h5 has no events/bg_unit")
        return int(bg[local])

    def n_samples(self, tag, event):
        """Samples of ``tag`` for global ``event`` (bulk_bg: of the background it used)."""
        i, local = self.locate(event)
        return self._samples_of(i, tag).get(self._unit_of(i, tag, local), 0)

    def _samples_of(self, i, tag):
        key = (i, tag)
        if key not in self._samples:
            with HadronFile(self._path(i, tag)) as hf:
                ids = hf.units.get("unit", np.arange(hf.n_units))
                per = np.diff(hf.unit_offsets)
                self._samples[key] = {int(u): int(n) for u, n in zip(ids, per)}
        return self._samples[key]

    def _initiators(self, i, local):
        import os

        import h5py

        if i not in self._init_cache:
            f = self._files[i]
            pair = f["pair_file"]
            path = os.path.join(os.path.dirname(f["stem"]), str(pair)) if pair else None
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"pair file {path!r} of {f['stem']} not found: "
                                        "initiators() reads its shower/ group")
            with h5py.File(path, "r") as h:
                self._init_cache[i] = (h["shower/initiators"][:],
                                       h["shower/initiator_offsets"][:])
        data, off = self._init_cache[i]
        return data[int(off[local]):int(off[local + 1])]

    # ── single events ───────────────────────────────────────────────────────────
    def _jet_events(self, i):
        if self._je[0] != i:
            if self._je[1] is not None:
                self._je[1].close()
            self._je = (i, JetEvents.from_stem(self._files[i]["stem"]))
        return self._je[1]

    def jet_event(self, event, k, frag_sample=None, fragments=True):
        """Oversample ``k`` of global ``event``'s jet-leg bulk plus a fragmentation
        (see :meth:`JetEvents.jet_event`)."""
        i, local = self.locate(event)
        return self._jet_events(i).jet_event(local, k, frag_sample, fragments)

    def background_event(self, event, k):
        i, local = self.locate(event)
        return self._jet_events(i).background_event(local, k)

    def n_oversamples(self, event):
        return self.n_samples("bulk_jet", event)

    def n_frag(self, event):
        return self.n_samples("jet_frag", event)

    def iter_jet_events(self, event, frag_sample=None):
        for k in range(self.n_oversamples(event)):
            yield self.jet_event(event, k, frag_sample=frag_sample)

    # ── histograms ──────────────────────────────────────────────────────────────
    def hist(self, tag, values, bins, *, mask=None, weights=None, events=None):
        """Sample-averaged histogram of ``tag`` over ``events`` (default all) and its
        error.  See the class docstring for ``values``, ``mask`` and ``weights``."""
        return self._accumulate(tag, values, bins, mask, weights, self.events(events))

    def total(self, tag, *, mask=None, weights=None, events=None):
        """Sample-averaged sum over hadrons (default: count) and its error."""
        h, e = self._accumulate(tag, None, None, mask, weights, self.events(events))
        return float(h[0]), float(e[0])

    def jet_minus_background(self, values, bins, *, mask=None, weights=None, events=None,
                             fragments=False):
        """<bulk_jet> (+ <jet_frag>) - <bulk_bg> over the events that have both surfaces,
        each event against its own background.  -> (difference, error)."""
        g = self.events(events)
        keep = np.array([self.n_samples("bulk_jet", e) > 0
                         and self.n_samples("bulk_bg", e) > 0 for e in g], dtype=bool)
        g = g[keep]
        hj, ej = self._accumulate("bulk_jet", values, bins, mask, weights, g)
        hb, eb = self._accumulate("bulk_bg", values, bins, mask, weights, g)
        d, var = hj - hb, ej ** 2 + eb ** 2
        if fragments:
            hf, ef = self._accumulate("jet_frag", values, bins, mask, weights, g)
            d, var = d + hf, var + ef ** 2
        return d, np.sqrt(var)

    def _hadrons(self, i, tag, positions):
        f, t, pos, h = self._loaded
        if f == i and t == tag and pos is not None and set(positions) <= pos:
            return h
        h = Hadrons.from_h5(self._path(i, tag), units=np.array(sorted(positions)))
        self._loaded = (i, tag, set(positions), h)
        return h

    def _accumulate(self, tag, values, bins, mask, weights, g):
        edges = _edges(bins)
        shape = tuple(len(e) - 1 for e in edges) if edges is not None else (1,)
        nbins = int(np.prod(shape))
        total = np.zeros(nbins)
        sq = np.zeros(nbins)
        n_used = 0
        if len(g) == 0:
            return total.reshape(shape), sq.reshape(shape)
        files = np.searchsorted(self._offsets, g, side="right") - 1
        for i in np.unique(files):
            local = g[files == i] - int(self._offsets[i])
            counts = self._samples_of(i, tag)
            by_unit = {}
            for e in local:
                u = self._unit_of(i, tag, e)
                if counts.get(u, 0) > 0:
                    by_unit.setdefault(u, []).append(int(e))
            if not by_unit:
                continue
            with HadronFile(self._path(i, tag)) as hf:
                positions = [hf.unit_index(u) for u in by_unit]
            h = self._hadrons(i, tag, positions)
            for u, evs in by_unit.items():
                ev = EventHadrons(h, h.unit_index(u))
                keys, wsum = [], []
                for e in evs:
                    info = EventInfo(self, self.global_event(i, e), i, e)
                    idx, w, rows = _binned(ev, info, values, edges, shape, mask, weights)
                    w = w / ev.n_samples            # this event's per-sample mean
                    total += np.bincount(idx, weights=w, minlength=nbins)
                    n_used += 1
                    if len(evs) == 1:
                        sq += np.bincount(idx, weights=w * w, minlength=nbins)
                    else:
                        keys.append(rows.astype(np.int64) * nbins + idx)
                        wsum.append(w)
                if len(evs) > 1 and keys:
                    # the same hadrons, counted once per event: their weights add per
                    # (hadron, bin) before squaring
                    k, inv = np.unique(np.concatenate(keys), return_inverse=True)
                    wk = np.bincount(inv, weights=np.concatenate(wsum))
                    sq += np.bincount(k % nbins, weights=wk * wk, minlength=nbins)
        norm = max(n_used, 1)
        return (total / norm).reshape(shape), (np.sqrt(sq) / norm).reshape(shape)

    def close(self):
        if self._je[1] is not None:
            self._je[1].close()
        self._je = (None, None)
        self._loaded = (None, None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _find_stems(source):
    import glob
    import os

    suffixes = ("_particlize.h5",) + tuple(f"_hadrons_{t}.h5" for t in TAGS)
    if isinstance(source, (str, os.PathLike)):
        src = str(source)
        if os.path.isdir(src):
            paths = glob.glob(os.path.join(src, "*_particlize.h5"))
        elif any(c in src for c in "*?["):
            paths = glob.glob(src)
        else:
            paths = [src]
    else:
        paths = [str(p) for p in source]
    stems = set()
    for p in paths:
        for suf in suffixes:
            if p.endswith(suf):
                p = p[:-len(suf)]
                break
        stems.add(p)
    missing = [s for s in stems if not os.path.exists(f"{s}_particlize.h5")]
    if missing:
        raise FileNotFoundError(f"no particlize file for {sorted(missing)[:3]}")
    return sorted(stems)


def _edges(bins):
    if bins is None:
        return None
    if isinstance(bins, (list, tuple)) and len(bins) and np.ndim(bins[0]) == 1:
        return [np.asarray(b, dtype=float) for b in bins]
    return [np.asarray(bins, dtype=float)]


def _resolve(spec, ev, info):
    if callable(spec):
        return spec(ev, info)
    if spec in ("charged", "all") or spec in SPECIES:
        return ev.species(spec)
    return getattr(ev, spec)


def _select(ev, info, mask):
    return np.ones(len(ev), bool) if mask is None else np.asarray(_resolve(mask, ev, info),
                                                                 dtype=bool)


def _binned(ev, info, values, edges, shape, mask, weights):
    """-> (flat bin index, weight, row in ev) of the selected hadrons inside the bins."""
    m = _select(ev, info, mask)
    rows = np.flatnonzero(m)
    w = np.ones(len(rows)) if weights is None else np.asarray(
        _resolve(weights, ev, info), dtype=float)[m]
    if edges is None:
        return np.zeros(len(w), dtype=np.int64), w, rows
    v = _resolve(values, ev, info)
    v = v if isinstance(v, tuple) else (v,)
    if len(v) != len(edges):
        raise ValueError(f"{len(v)} value array(s) for {len(edges)} bin dimension(s)")
    inside = np.ones(len(w), bool)
    idx = []
    for vd, ed in zip(v, edges):
        vd = np.asarray(vd, dtype=float)[m]
        j = np.searchsorted(ed, vd, side="right") - 1
        j[vd == ed[-1]] = len(ed) - 2           # numpy convention: last edge inclusive
        inside &= (j >= 0) & (j < len(ed) - 1)
        idx.append(j)
    flat = np.ravel_multi_index(tuple(j[inside] for j in idx), shape)
    return flat.astype(np.int64), w[inside], rows[inside]

