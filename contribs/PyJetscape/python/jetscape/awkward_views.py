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
one call clusters every sample.  A stored mass (HadronArray.mass, the records' `mass`
column: X-SCAPE's rest mass) becomes the field `rest_mass`: vector takes every temporal
name (E, mass, m, tau, ...) as a coordinate and, from 1.9 on, refuses a record with two.  The position is a nested `position` field (a Vector4D
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

__all__ = ["to_awkward", "select", "pdg_charge", "SPECIES_CLASSES", "M_PION"]

_MOMENTUM = ("E", "px", "py", "pz")
_POSITION = ("t", "x", "y", "z")
#: the field a stored mass gets in the views (vector computes .mass from E and p)
_REST_MASS = "rest_mass"
#: the charged pion mass [GeV] (PDG 2024), the mass to_awkward(pion_mass=True) gives tracks
M_PION = 0.13957039
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


def to_awkward(obj, position=False, charge=False, by_unit=False, charged=False,
               massless=False, pion_mass=False):
    """The nested awkward view of a HadronArray, a hadron dict or a record table.

    position: add the nested `position` (HadronArray; records always carry theirs).
    charge:   add a `charge` field [e] from the PDG (needs `particle`).
    by_unit:  a HadronArray with info["unit_offsets"] (HadronArray.from_h5: the file's
              units; HadronFileReader.hadron_array: the events): units x samples x hadrons.
              False: samples x hadrons, all samples in a row.
    (charged, massless and pion_mass are new after v1.5)
    charged:  add a boolean `charged` field: the pid is in hadrons_h5.CHARGED, the stable
              charged hadrons and leptons, as HadronFileReader's mask="charged" (no
              `particle` needed; `charge` above is the PDG charge, resonances included).
    massless: E recomputed as |p| (float64, a new array), as for massless particles; the
              stored E is not used.
    pion_mass: E recomputed as sqrt(p^2 + m^2) (float64, a new array) with m = M_PION
              for the charged particles (`charged` above, leptons included) and m = 0 for
              the neutral ones: every track as a pion, every neutral as massless, the usual
              assumption of a detector-level analysis.  Not together with massless.
    """
    if massless and pion_mass:
        raise ValueError("to_awkward: massless and pion_mass exclude each other")
    ak = _modules()
    from .hadron_array import HadronArray

    if isinstance(obj, dict):
        obj = HadronArray.from_dict(obj)
    if isinstance(obj, HadronArray):
        return _hadrons(ak, obj, position, charge, by_unit, charged, massless, pion_mass)
    rec = np.asarray(obj)
    if rec.dtype.names is None:
        raise TypeError("to_awkward takes a HadronArray, a hadron dict or a record array "
                        f"(an *_array() result), not {type(obj).__name__} of {rec.dtype}")
    return _records(ak, rec, charge, charged, massless, pion_mass)


def _charged(pid):
    from .hadrons_h5 import CHARGED

    return np.isin(np.abs(pid), CHARGED)


def _massless_e(px, py, pz):
    px, py, pz = (np.asarray(a, dtype=np.float64) for a in (px, py, pz))
    return np.sqrt(px * px + py * py + pz * pz)


def _pion_mass_e(px, py, pz, pid):
    """E with the pion mass for the charged particles (hadrons_h5.CHARGED), 0 otherwise."""
    m2 = np.where(_charged(pid), M_PION * M_PION, 0.0)
    px, py, pz = (np.asarray(a, dtype=np.float64) for a in (px, py, pz))
    return np.sqrt(px * px + py * py + pz * pz + m2)


def _hadrons(ak, h, position, charge, by_unit=False, charged=False, massless=False,
             pion_mass=False):
    names, contents = [], []

    def add(name, a):
        names.append(name)
        contents.append(ak.contents.NumpyArray(a))

    for i, n in enumerate(_MOMENTUM):
        if n == "E" and massless:
            add(n, _massless_e(h.p[:, 1], h.p[:, 2], h.p[:, 3]))
        elif n == "E" and pion_mass:
            add(n, _pion_mass_e(h.p[:, 1], h.p[:, 2], h.p[:, 3], h.pid))
        else:
            add(n, h.p[:, i])
    add("pid", h.pid)
    add("pstat", h.pstat)
    if h.mass is not None:
        add(_REST_MASS, h.mass)               # not "mass": a second temporal coordinate
    if position:
        if h.x is None:
            raise ValueError("to_awkward(position=True): these hadrons have no positions")
        names.append("position")
        contents.append(ak.contents.RecordArray(
            [ak.contents.NumpyArray(h.x[:, i]) for i in range(4)], list(_POSITION),
            length=len(h), parameters={"__record__": "Vector4D"}))
    if charge:
        add("charge", pdg_charge(h.pid))
    if charged:
        add("charged", _charged(h.pid))
    record = ak.contents.RecordArray(contents, names, length=len(h),
                                     parameters={"__record__": "Momentum4D"})
    layout = ak.contents.ListOffsetArray(ak.index.Index64(h.sample_offsets), record)
    if by_unit:
        unit_offsets = h.info.get("unit_offsets")
        if unit_offsets is None:
            raise ValueError("to_awkward(by_unit=True) needs info['unit_offsets'] "
                             "(a HadronArray read with HadronArray.from_h5)")
        layout = ak.contents.ListOffsetArray(
            ak.index.Index64(np.asarray(unit_offsets, dtype=np.int64)), layout)
    return ak.Array(layout)


def _records(ak, rec, charge, charged=False, massless=False, pion_mass=False):
    fields = {n: rec[n] for n in rec.dtype.names}
    if charge and "pid" in fields:
        fields["charge"] = pdg_charge(rec["pid"])
    if charged and "pid" in fields:
        fields["charged"] = _charged(rec["pid"])
    if massless and all(n in fields for n in _MOMENTUM):
        fields["E"] = _massless_e(rec["px"], rec["py"], rec["pz"])
    if pion_mass and "pid" in fields and all(n in fields for n in _MOMENTUM):
        fields["E"] = _pion_mass_e(rec["px"], rec["py"], rec["pz"], rec["pid"])
    has_position = all(n in fields for n in _POSITION)
    if "mass" in fields:                      # vector would read it as a coordinate
        fields = {(_REST_MASS if n == "mass" else n): v for n, v in fields.items()}
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
