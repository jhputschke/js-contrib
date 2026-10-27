"""
python/jetscape/awkward_views.py

Nested views of the Pythonic layer's flat arrays (docs/Plans/PLAN_pyjetscape_pythonic.md,
step 1): to_awkward() and PDG selections.  awkward, vector and particle are imported
inside the functions, never when jetscape is imported.

    to_awkward(HadronArray)          samples x hadrons, Momentum4D records over the same
                                     buffers (no copy; p's columns are strided views)
    to_awkward(final partons or      showers x partons (ak.unflatten on the runs of
               initiator records)    `shower`; small tables, a copy)
    to_awkward(edge, node or hard    a flat record array
               parton records)

Momentum fields are E, px, py, pz, so `vector` gives pt, eta, phi, rapidity, mass when
asked (computed, not stored), and FastJet's ClusterSequence takes the result directly:
one call clusters every sample.  The position is a nested `position` field (a Vector4D
record t, x, y, z): `vector` would read flat t, x, y, z fields as the coordinates of the
Momentum4D record.  Node records (no momentum) are Vector4D records themselves.

    select(arr, "pi+"), select(arr, ["K+", "K-"]), select(arr, "charged")

returns a mask of arr's shape.  Species are PDG names as `particle` spells them
("pi+", "K-", "p~", "Lambda", "K(S)0") or classes: "charged", "neutral", "hadron",
"meson", "baryon", "strange".  Each distinct pid is looked up once.

"charged" is the PDG charge != 0, for every pid: resonances included.  hadrons_h5.CHARGED
(used by Hadrons.charged and the analyses) lists the stable charged hadrons and leptons
only; the two agree on decayed hadrons and differ on files with undecayed primaries
(iSS <keep_primaries>).
"""

from __future__ import annotations

import sys

import numpy as np

__all__ = ["to_awkward", "select", "pdg_charge", "SPECIES_CLASSES"]

_MOMENTUM = ("E", "px", "py", "pz")
_POSITION = ("t", "x", "y", "z")
_VECTOR_REGISTERED = False


def _modules():
    global _VECTOR_REGISTERED
    try:
        import awkward as ak
        import vector
    except ImportError as e:  # pragma: no cover
        raise ImportError("to_awkward needs the awkward and vector packages "
                          "(pip install awkward vector)") from e
    if not _VECTOR_REGISTERED:
        vector.register_awkward()
        _VECTOR_REGISTERED = True
    return ak


def _pdgid():
    try:
        from particle import pdgid
    except ImportError as e:
        raise ImportError("PDG lookups need the particle package (scikit-hep): "
                          "pip install particle") from e
    return pdgid


def _per_unique(pid, func, dtype):
    """func evaluated once per distinct pid, mapped back with numpy."""
    pid = np.asarray(pid)
    uniq, inverse = np.unique(pid, return_inverse=True)
    values = np.array([func(int(u)) for u in uniq], dtype=dtype)
    return values[inverse].reshape(pid.shape)


def pdg_charge(pid) -> np.ndarray:
    """PDG charge [e] per pid (float64; quarks fractional, unknown pids 0)."""
    pdgid = _pdgid()

    def charge(u):
        c = pdgid.charge(u)
        return 0.0 if c is None else float(c)
    return _per_unique(pid, charge, np.float64)


def to_awkward(obj, position=False, charge=False):
    """The nested awkward view of a HadronArray, a hadron dict or a record table.

    position: add t, x, y, z (HadronArray; records always carry theirs).
    charge:   add a `charge` field [e] from the PDG (needs `particle`).
    """
    ak = _modules()
    from .hadron_array import HadronArray

    if isinstance(obj, dict):
        obj = HadronArray.from_dict(obj)
    if isinstance(obj, HadronArray):
        return _hadrons(ak, obj, position, charge)
    rec = np.asarray(obj)
    if rec.dtype.names is None:
        raise TypeError("to_awkward takes a HadronArray, a hadron dict or a record array "
                        f"(an *_array() result), not {type(obj).__name__} of {rec.dtype}")
    return _records(ak, rec, charge)


def _hadrons(ak, h, position, charge):
    names, contents = [], []

    def add(name, a):
        names.append(name)
        contents.append(ak.contents.NumpyArray(a))

    for i, n in enumerate(_MOMENTUM):
        add(n, h.p[:, i])
    add("pid", h.pid)
    add("pstat", h.pstat)
    add("mass", h.mass)
    if position:
        names.append("position")
        contents.append(ak.contents.RecordArray(
            [ak.contents.NumpyArray(h.x[:, i]) for i in range(4)], list(_POSITION),
            length=len(h), parameters={"__record__": "Vector4D"}))
    if charge:
        add("charge", pdg_charge(h.pid))
    record = ak.contents.RecordArray(contents, names, length=len(h),
                                     parameters={"__record__": "Momentum4D"})
    layout = ak.contents.ListOffsetArray(ak.index.Index64(h.sample_offsets), record)
    return ak.Array(layout)


def _records(ak, rec, charge):
    fields = {n: rec[n] for n in rec.dtype.names}
    if charge and "pid" in fields:
        fields["charge"] = pdg_charge(rec["pid"])
    has_position = all(n in fields for n in _POSITION)
    if all(n in fields for n in _MOMENTUM):
        with_name = "Momentum4D"
        if has_position:
            fields["position"] = ak.zip({n: fields.pop(n) for n in _POSITION},
                                        with_name="Vector4D")
    else:
        with_name = "Vector4D" if has_position else None
    flat = ak.zip(fields, with_name=with_name, depth_limit=1)
    if "shower" not in fields:
        return flat
    shower = rec["shower"]
    if len(shower) == 0:
        return ak.unflatten(flat, np.zeros(0, dtype=np.int64))
    starts = np.flatnonzero(np.r_[True, shower[1:] != shower[:-1]])
    counts = np.diff(np.r_[starts, len(shower)])
    return ak.unflatten(flat, counts)


SPECIES_CLASSES = ("charged", "neutral", "hadron", "meson", "baryon", "strange")


def _class_test(pdgid, name):
    if name == "charged":
        return lambda u: (pdgid.charge(u) or 0) != 0
    if name == "neutral":
        return lambda u: (pdgid.charge(u) or 0) == 0
    if name == "hadron":
        return lambda u: bool(pdgid.is_hadron(u))
    if name == "meson":
        return lambda u: bool(pdgid.is_meson(u))
    if name == "baryon":
        return lambda u: bool(pdgid.is_baryon(u))
    if name == "strange":
        return lambda u: bool(pdgid.has_strange(u))
    return None


def _species_test(species):
    pdgid = _pdgid()
    from particle import Particle

    if isinstance(species, (str, int, np.integer)):
        species = [species]
    tests, pids = [], set()
    for s in species:
        if isinstance(s, (int, np.integer)):
            pids.add(int(s))
            continue
        t = _class_test(pdgid, s)
        if t is not None:
            tests.append(t)
            continue
        try:
            pids.add(int(Particle.from_name(s).pdgid))
        except Exception as e:
            raise ValueError(f"select: {s!r} is neither a PDG name `particle` knows nor one "
                             f"of {SPECIES_CLASSES}") from e
    return lambda u: u in pids or any(t(u) for t in tests)


def select(arr, species):
    """Mask of the hadrons (or partons) of `species` in arr: a HadronArray (flat mask),
    an awkward array with a `pid` field (a mask of its shape), a record array or a pid
    array.  species: a name or class, or a list of them (any of them selects)."""
    from .hadron_array import HadronArray

    test = _species_test(species)
    if isinstance(arr, HadronArray):
        return _per_unique(arr.pid, test, bool)
    ak = sys.modules.get("awkward")
    if ak is not None and isinstance(arr, ak.Array):
        _modules()

        def mask(layout, **kwargs):
            if layout.is_numpy:
                return ak.contents.NumpyArray(_per_unique(np.asarray(layout.data), test,
                                                          bool))
            return None
        return ak.transform(mask, arr["pid"])
    a = np.asarray(arr)
    return _per_unique(a["pid"] if a.dtype.names else a, test, bool)
