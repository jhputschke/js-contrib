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
    initiators/data        (K, 11) float64  optional, bulk_jet and jet_frag only: the event's
    initiators/offsets     (U + 1,) int64   shower-initiating partons (INITIATOR_COLUMNS), unit
                                            u is rows offsets[u]:[u+1]; copied by hadronize.py
                                            from the pair file's shower/, so jet-relative
                                            analyses do not need the (large) pair file

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
from .showers import INITIATOR_COLUMNS

__all__ = ["FORMAT", "FORMAT_VERSION", "TAGS", "CHARGED", "SPECIES", "FIELDS", "ORIGIN",
           "ROUNDABLE", "INITIATOR_COLUMNS", "INITIATOR_TAGS", "hadron_precision",
           "pair_initiators", "add_initiators", "HadronH5Writer", "Hadrons", "HadronFile",
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

#: tags whose unit is an event, so they can carry the event's shower initiators
INITIATOR_TAGS = ("bulk_jet", "jet_frag")


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
    on each dataset (:func:`hadron_precision`).

    ``initiators=True`` (bulk_jet and jet_frag only) adds the ``initiators/`` group: every
    :meth:`append_unit` then takes that event's shower initiators, a (K, 11) array in
    ``INITIATOR_COLUMNS`` order (None: none)."""

    def __init__(self, path, *, tag, n_samples, attrs=None, compression=DEFAULT_COMPRESSION,
                 force=True, keep_bits=None, initiators=False):
        import h5py

        if tag not in TAGS:
            raise ValueError(f"tag must be one of {TAGS}, got {tag!r}")
        if initiators and tag not in INITIATOR_TAGS:
            raise ValueError(f"initiators: only {INITIATOR_TAGS} have one event per unit, "
                             f"not {tag!r}")
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
        self._ini = _initiator_group(self.f) if initiators else None
        self._u = 0

    def append_unit(self, samples, initiators=None, **unit_scalars):
        """Append one unit.  ``samples`` is a list of hadron dicts (one per sample, keys
        pid, pstat, p, x), or one dict with ``sample_counts`` (as soft_hadrons_numpy
        returns); ``initiators`` the event's (K, 11) initiator rows, for a writer made with
        ``initiators=True``.  Returns the unit index."""
        if initiators is not None and self._ini is None:
            raise ValueError("initiators given, but the writer was made without "
                             "initiators=True")
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
        if self._ini is not None:
            self._ini.append(_initiator_rows(initiators))
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
    def has_initiators(self):
        return "initiators" in self.f

    def initiators(self, unit):
        """Shower initiators of recorded unit ``unit`` (an event), (K, 11) in
        ``INITIATOR_COLUMNS`` order; None if the file has no ``initiators/``."""
        if not self.has_initiators:
            return None
        u = self.unit_index(unit)
        off = self.f["initiators/offsets"]
        return self.f["initiators/data"][int(off[u]):int(off[u + 1])]

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
def _initiator_group(f):
    g = f.require_group("initiators")
    g.attrs["columns"] = list(INITIATOR_COLUMNS)
    g.attrs["units"] = "p, E in GeV; x, y, z in fm; t in fm/c (Cartesian lab)"
    g.attrs["source"] = "pair file shower/initiators (jetscape.showers)"
    return RaggedGroup(g, "offsets", {"data": (np.float64, (len(INITIATOR_COLUMNS),))},
                       compression="gzip", chunk_rows=4096, unit="event")


def _initiator_rows(ini):
    rows = np.zeros((0, len(INITIATOR_COLUMNS))) if ini is None else np.asarray(
        ini, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != len(INITIATOR_COLUMNS):
        raise ValueError(f"initiators must be (K, {len(INITIATOR_COLUMNS)}), got {rows.shape}")
    return {"data": rows}


def pair_initiators(path, nevents=None):
    """``(data, offsets)`` of a pair file's ``shower/initiators``, or None when the file or
    the group is missing.  With ``nevents`` the pair file must hold that many events."""
    import h5py

    if not path or not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        if "shower/initiators" not in f or "shower/initiator_offsets" not in f:
            return None
        cols = f["shower"].attrs.get("initiator_columns")
        if cols is not None and tuple(str(c) for c in cols) != INITIATOR_COLUMNS:
            raise ValueError(f"{path}: initiator columns {list(cols)} are not "
                             f"{list(INITIATOR_COLUMNS)}")
        data, off = f["shower/initiators"][:], f["shower/initiator_offsets"][:]
    if nevents is not None and len(off) - 1 != int(nevents):
        raise ValueError(f"{path} has initiators for {len(off) - 1} event(s), the particlize "
                         f"file {nevents}: not the same run")
    return data, off


def add_initiators(path, data, offsets, *, force=False):
    """Add ``initiators/`` to an existing bulk_jet or jet_frag file: unit u gets the rows
    of event ``units/event[u]`` from ``(data, offsets)`` (as :func:`pair_initiators`
    returns).  An existing group is kept unless ``force``.  Returns False if it was kept."""
    import h5py

    from . import h5_compression  # noqa: F401  (Blosc filter)

    with h5py.File(path, "r+") as f:
        tag = str(f.attrs.get("tag", ""))
        if tag not in INITIATOR_TAGS:
            raise ValueError(f"{path}: initiators only go into {INITIATOR_TAGS}, not {tag!r}")
        if "initiators" in f:
            if not force:
                return False
            del f["initiators"]
        events = f["units/event"][:] if "units/event" in f else np.arange(
            len(f["hadrons/unit_offsets"]) - 1)
        g = _initiator_group(f)
        for e in events:
            if not 0 <= int(e) < len(offsets) - 1:
                raise IndexError(f"{path}: event {int(e)} has no initiators "
                                 f"({len(offsets) - 1} event(s))")
            g.append(_initiator_rows(data[int(offsets[e]):int(offsets[e + 1])]))
    return True


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
        # run_prod_jet.py --pthat-bins: this event's pTHat window and pTHat (else None)
        self.pthat_bin = (int(f["pthat_bin"][local]) if f["pthat_bin"] is not None
                          else None)
        self.pthat = float(f["pthat"][local]) if f["pthat"] is not None else None

    def initiators(self):
        """This event's shower-initiating partons, as a (K, 11) array, columns ``shower,
        pid, pstat, px, py, pz, E, x, y, z, t``: from the hadron files' ``initiators/``
        (hadronize.py copies them there), else from the pair file's ``shower/``."""
        return self._reader._initiators(self.file_index, self.local_event)

    def __repr__(self):
        return (f"EventInfo(event={self.event}, stem={self.stem!r}, "
                f"local_event={self.local_event}, bg_unit={self.bg_unit}"
                + (f", pthat_bin={self.pthat_bin}" if self.pthat_bin is not None else "")
                + ")")


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
    per event, added over events; :meth:`jet_minus_background` of correlated legs
    (hadronize.py --correlated) takes its error from the per-sample differences instead
    (``paired``).  ``values``, ``mask`` and ``weights`` are either
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
        dup = self.duplicate_backgrounds()
        if dup:
            import warnings
            n_id = sum(d["kind"] == "identical" for d in dup)
            ex = dup[0]
            where = ", ".join(f"{os.path.basename(self._files[self.locate(g)[0]]['stem'])} "
                              f"event {self.locate(g)[1]}" for g in ex["events"][:3])
            warnings.warn(
                f"HadronFileReader: {len(dup)} background(s) occur in more than one production "
                f"file ({n_id} identical, {len(dup) - n_id} probably the same collision on "
                f"another output grid), e.g. {where}. Campaigns over the same seeds share "
                "their collisions (the jets may differ): these events are not independent. "
                "Use disjoint seed ranges per campaign; duplicate_backgrounds() lists them.",
                RuntimeWarning, stacklevel=2)
        self._samples = {}              # (file index, tag) -> {unit id: n_samples}
        self._init_cache = {}
        self._je = (None, None)         # (file index, JetEvents) of the last lookup
        self._loaded = (None, None, None, None)   # (file, tag, unit positions, Hadrons)

    # ── discovery ───────────────────────────────────────────────────────────────
    @staticmethod
    def _open_stem(stem, check_uuid):
        import h5py

        from . import h5_compression  # noqa: F401
        from .particlize_h5 import ParticlizeFile

        with ParticlizeFile(f"{stem}_particlize.h5") as pf:
            uuid = str(pf.attrs.get("file_uuid", ""))
            info = {"stem": stem, "nevents": pf.nevents, "uuid": uuid,
                    "seed": pf.attrs.get("prod_seed"),
                    "pair_file": pf.attrs.get("pair_file"),
                    "bg_unit": (pf.events("bg_unit") if pf.has_events("bg_unit")
                                else None),
                    "bg_key": (pf.events("bg_key") if pf.has_events("bg_key") else None),
                    "n_cells_bg": (pf.events("n_cells_bg") if pf.has_events("n_cells_bg")
                                   else None),
                    "pthat_bin": (pf.events("pthat_bin") if pf.has_events("pthat_bin")
                                  else None),
                    "pthat": pf.events("pthat") if pf.has_events("pthat") else None}
            for key, attr in (("pthat_bins", "pthat_bins"),
                              ("sigma_gen", "pthat_bin_sigma_gen"),
                              ("sigma_err", "pthat_bin_sigma_err"),
                              ("n_accepted", "pthat_bin_n_accepted"),
                              ("n_tried", "pthat_bin_n_tried"),
                              ("n_kept", "pthat_bin_n_kept")):
                v = pf.attrs.get(attr)
                info[key] = None if v is None else np.asarray(v)
            info["parton_ymax"] = pf.attrs.get("parton_ymax")
        info["tags"], info["precision"], info["correlated"] = {}, {}, {}
        for tag in TAGS:
            path = f"{stem}_hadrons_{tag}.h5"
            if not os.path.exists(path):
                continue
            with h5py.File(path, "r") as f:
                ftag, src = str(f.attrs.get("tag", "")), str(f.attrs.get("source_uuid", ""))
                info["precision"][tag] = hadron_precision(f)
                info["correlated"][tag] = bool(f.attrs.get("correlated_sampling", False))
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

    def duplicate_backgrounds(self):
        """Backgrounds found in more than one production file, as a list of
        ``{"kind", "events"}``: ``events`` the global index of the first event using that
        background in each file.  ``kind`` "identical": the same background leg bit for bit
        (particlize ``events/bg_key``, a hash of it on the output grid); "probable": same
        production seed and the same number of freeze-out cells (``n_cells_bg``, which does
        not depend on the output grid) but another hash, as for the same collision written
        on another grid.  Both come from campaigns run over the same seeds."""
        ident, fprint = {}, {}
        for i, f in enumerate(self._files):
            if f["bg_unit"] is None:
                continue
            first = {}
            for local, u in enumerate(np.asarray(f["bg_unit"])):
                first.setdefault(int(u), local)
            for local in first.values():
                g = self.global_event(i, local)
                if f["bg_key"] is not None:
                    key = f["bg_key"][local]
                    key = key.decode() if isinstance(key, bytes) else str(key)
                    if key:
                        ident.setdefault(key, []).append((i, g))
                if f["n_cells_bg"] is not None and f["seed"] is not None:
                    n = int(f["n_cells_bg"][local])
                    if n > 0:
                        fprint.setdefault((int(f["seed"]), n), []).append((i, g))
        out, seen = [], set()
        for kind, table in (("identical", ident), ("probable", fprint)):
            for items in table.values():
                events = tuple(sorted(g for _, g in items))
                if len({i for i, _ in items}) < 2 or events in seen:
                    continue
                seen.add(events)
                out.append({"kind": kind, "events": list(events)})
        return out

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

    # ── pTHat windows (run_prod_jet.py --pthat-bins) ────────────────────────────
    @property
    def pthat_bins(self):
        """The campaign's pTHat windows, a (K, 2) array [GeV], or None if its files were
        made without --pthat-bins.  All files must have the same windows."""
        have = [f["pthat_bins"] for f in self._files if f["pthat_bins"] is not None]
        if not have:
            return None
        if len(have) != len(self._files) or any(
                b.shape != have[0].shape or not np.array_equal(b, have[0]) for b in have):
            raise ValueError("HadronFileReader: the files have different pTHat windows "
                             "(or some have none): read each set on its own")
        return have[0].reshape(-1, 2)

    def pthat_bin_events(self, k):
        """Global indices of the events in pTHat window ``k``, for ``events=``."""
        if self.pthat_bins is None:
            raise ValueError("no pTHat windows in this campaign (--pthat-bins)")
        out = []
        for i, f in enumerate(self._files):
            b = f["pthat_bin"]
            if b is None:
                raise ValueError(f"{f['stem']}_particlize.h5 has no events/pthat_bin")
            out.append(self._offsets[i] + np.flatnonzero(np.asarray(b, dtype=np.int64) == k))
        return np.concatenate(out).astype(np.int64)

    def pthat_bin_sigma(self, k):
        """Window ``k``'s cross section [mb] and its error over the whole campaign: the
        files' Pythia estimates, weighted by their accepted events.  To combine windows,
        weight each window's event average by its cross section."""
        if self.pthat_bins is None:
            raise ValueError("no pTHat windows in this campaign (--pthat-bins)")
        sig, err, n = [], [], []
        for f in self._files:
            if f["sigma_gen"] is None or f["n_accepted"] is None:
                raise ValueError(f"{f['stem']}_particlize.h5 has no cross sections (job "
                                 "not finished?)")
            sig.append(f["sigma_gen"][k])
            err.append(f["sigma_err"][k])
            n.append(f["n_accepted"][k])
        sig, err, n = (np.asarray(x, dtype=np.float64) for x in (sig, err, n))
        if n.sum() == 0:
            return float("nan"), float("nan")
        return (float(np.sum(n * sig) / n.sum()),
                float(np.sqrt(np.sum((n * err) ** 2)) / n.sum()))

    def pthat_bin_acceptance(self, k):
        """Window ``k``'s acceptance of the parton rapidity cut (run_prod_jet.py
        --parton-ymax) over the whole campaign: events kept / tried, and its binomial
        error.  (1.0, 0.0) without a cut.  The cross sections (:meth:`pthat_bin_sigma`)
        already include it (sigma_gen = Pythia's x kept/tried per file)."""
        if self.pthat_bins is None:
            raise ValueError("no pTHat windows in this campaign")
        tried = kept = 0
        for f in self._files:
            if f["n_tried"] is None:
                continue
            tried += int(f["n_tried"][k])
            kept += int(f["n_kept"][k])
        if tried == 0:
            return 1.0, 0.0
        acc = kept / tried
        return acc, float(np.sqrt(acc * (1.0 - acc) / tried))

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
        """From a hadron file's ``initiators/`` (bulk_jet, then jet_frag), else from the
        pair file's ``shower/``."""
        if i not in self._init_cache:
            self._init_cache[i] = self._load_initiators(i)
        data, off, pos = self._init_cache[i]
        u = int(local) if pos is None else pos.get(int(local))
        if u is None:
            raise IndexError(f"{self._files[i]['stem']}: event {local} is not in the hadron "
                             "files' initiators/")
        return data[int(off[u]):int(off[u + 1])]

    def _load_initiators(self, i):
        f = self._files[i]
        for tag in INITIATOR_TAGS:
            path = f["tags"].get(tag)
            if path is None:
                continue
            with HadronFile(path) as hf:
                if hf.has_initiators:
                    ids = hf.units.get("unit", np.arange(hf.n_units))
                    return (hf.f["initiators/data"][:], hf.f["initiators/offsets"][:],
                            {int(u): k for k, u in enumerate(ids)})
        pair = f["pair_file"]
        path = os.path.join(os.path.dirname(f["stem"]), str(pair)) if pair else None
        found = pair_initiators(path)
        if found is None:
            raise FileNotFoundError(
                f"{f['stem']}: no initiators/ in its hadron files and pair file {path!r} not "
                "found (initiators() reads either; hadronize.py --add-initiators adds them "
                "to existing hadron files)")
        return found[0], found[1], None

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

    @property
    def correlated(self):
        """True if every file's bulk_jet and bulk_bg were sampled correlated
        (hadronize.py --correlated)."""
        return all(f["correlated"].get(t, False) for f in self._files
                   for t in ("bulk_jet", "bulk_bg"))

    def jet_minus_background(self, values, bins, *, mask=None, weights=None, events=None,
                             fragments=False, paired=None):
        """<bulk_jet> (+ <jet_frag>) - <bulk_bg> over the events that have both surfaces,
        each event against its own background.  -> (difference, error).

        ``paired``: the error of the bulk difference from the spread of the per-sample
        differences, sample k of the jet leg minus sample k of its background (the events
        sharing a background summed per sample), instead of the two legs' errors added.
        With correlated legs (hadronize.py --correlated) only the paired error is right:
        the added one ignores that the legs' noise cancels and is ~3x too large.  None
        (default): paired if the files were sampled correlated (:attr:`correlated`).
        Paired needs the same number of samples, at least 2, on both legs of an event.
        """
        g = self.events(events)
        keep = np.array([self.n_samples("bulk_jet", e) > 0
                         and self.n_samples("bulk_bg", e) > 0 for e in g], dtype=bool)
        g = g[keep]
        if paired is None:
            paired = self.correlated
        if paired:
            d, err = self._paired_difference(values, bins, mask, weights, g)
            var = err ** 2
        else:
            hj, ej = self._accumulate("bulk_jet", values, bins, mask, weights, g)
            hb, eb = self._accumulate("bulk_bg", values, bins, mask, weights, g)
            d, var = hj - hb, ej ** 2 + eb ** 2
        if fragments:
            hf, ef = self._accumulate("jet_frag", values, bins, mask, weights, g)
            d, var = d + hf, var + ef ** 2
        return d, np.sqrt(var)

    def _paired_difference(self, values, bins, mask, weights, g):
        """<bulk_jet> - <bulk_bg> and its error from the per-sample differences.

        The events sharing a background are correlated with it (and with each other)
        sample by sample, so they are summed per sample first: S_k = sum over those events
        of (J_ek - B_k), with B evaluated with each event's info.  Different backgrounds
        are independent.  -> (mean over events, error)."""
        edges = _edges(bins)
        shape = tuple(len(e) - 1 for e in edges) if edges is not None else (1,)
        nbins = int(np.prod(shape))
        total = np.zeros(nbins)
        var = np.zeros(nbins)
        n_used = 0
        files = np.searchsorted(self._offsets, g, side="right") - 1
        for i in np.unique(files):
            local = [int(e) for e in g[files == i] - int(self._offsets[i])]
            groups = {}
            for e in local:
                groups.setdefault(self._unit_of(i, "bulk_bg", e), []).append(e)
            legs = {}
            for tag, units in (("bulk_jet", sorted({self._unit_of(i, "bulk_jet", e)
                                                    for e in local})),
                               ("bulk_bg", sorted(groups))):
                with HadronFile(self._path(i, tag)) as hf:
                    positions = [hf.unit_index(u) for u in units]
                legs[tag] = Hadrons.from_h5(self._path(i, tag),
                                            units=np.array(sorted(positions)))
            hj, hb = legs["bulk_jet"], legs["bulk_bg"]
            for u, evs in groups.items():
                evb = EventHadrons(hb, hb.unit_index(u))
                K = evb.n_samples
                if K < 2:
                    raise ValueError(f"{self._files[i]['stem']}: background {u} has {K} "
                                     "sample(s); paired errors need at least 2")
                S = np.zeros(K * nbins)
                for e in evs:
                    evj = EventHadrons(hj, hj.unit_index(self._unit_of(i, "bulk_jet", e)))
                    if evj.n_samples != K:
                        raise ValueError(
                            f"{self._files[i]['stem']}: event {e} has {evj.n_samples} jet "
                            f"but {K} background samples; paired needs equal counts "
                            "(or pass paired=False)")
                    info = EventInfo(self, self.global_event(i, e), i, e)
                    for ev, sign in ((evj, 1.0), (evb, -1.0)):
                        idx, w, rows = _binned(ev, info, values, edges, shape, mask,
                                               weights)
                        S += sign * np.bincount(ev.sample[rows] * nbins + idx, weights=w,
                                                minlength=K * nbins)
                    n_used += 1
                S = S.reshape(K, nbins)
                total += S.mean(0)
                var += S.var(0, ddof=1) / K
        norm = max(n_used, 1)
        return (total / norm).reshape(shape), (np.sqrt(var) / norm).reshape(shape)

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

