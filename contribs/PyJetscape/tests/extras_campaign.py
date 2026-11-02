"""
tests/extras_campaign.py

Small synthetic Z+jet production files with extra hard scatterings (an ``extra_frag`` file,
docs/Plans/PLAN_multi_hard.md), for the reader, converter and analysis tests: the production
files of test_particlize_h5._seed_files (particlize, pair file, bulk_jet, bulk_bg, jet_frag)
with one pTHat window, a trigger Z per event, a recoiling gluon as the shower initiator (also
in bulk_jet's initiators/, as hadronize.py copies them) and, with ``extras``,
the extra_frag file (its hadrons, scatterings and initiators) and the particlize attribute
``extra_hard``.  ``minbias`` makes a minimum-bias job's files instead: no jet_frag, no Z,
``extra_no_trigger``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
sys.path.insert(0, str(HERE))

from test_particlize_h5 import PHI_JET, _seed_files  # noqa: E402

#: the extras' hadrons per event and fragmentation: N_HARD charged pions of p_T PT_HARD (a
#: jet of ~N_HARD * PT_HARD GeV at (ETA_X, phi_jet + pi / 2)) and N_SOFT kaons of p_T 1.5
N_HARD, PT_HARD, N_SOFT, ETA_X = 5, 9.0, 3, 0.5
#: the extras' scatterings: the trigger's row first (scatter 0), then two extras
SCAT = np.array([[0, 60.0, 241, 0.1, -0.2, 2, 120.0, 100.0],
                 [1, 20.0, 111, 1.5, 0.4, 2, 45.0, 38.0],
                 [2, 16.0, 113, -2.0, 1.0, 2, 35.0, 30.0]])


#: one pTHat window (the map steps' file records need one)
WINDOW = {"pthat_bins": np.array([[50.0, 70.0]]), "pthat_bin_sigma_gen": [1.0],
          "pthat_bin_sigma_err": [0.1], "pthat_bin_n_accepted": [2], "prod_campaign": "xtiny"}


def z_record(name, e):
    """A Z opposite the event's jet (CMS 2507.09307's and 1702.01060's selections pass)."""
    phi = PHI_JET[(name, e)] - np.pi
    return {"z_pt": 60.0, "z_y": 0.2, "z_phi": phi, "z_m": 91.0,
            "lep1_pid": 13, "lep1_pt": 40.0, "lep1_eta": 0.4, "lep1_phi": phi + 0.3,
            "lep2_pid": -13, "lep2_pt": 30.0, "lep2_eta": -0.1, "lep2_phi": phi - 0.4}


def _extra_hadrons(name, e, j):
    phi = PHI_JET[(name, e)] + np.pi / 2
    n = N_HARD + N_SOFT
    pt = np.r_[np.full(N_HARD, PT_HARD), np.full(N_SOFT, 1.5)]
    eta = np.r_[ETA_X + 0.02 * np.arange(N_HARD), -0.3 + 0.1 * np.arange(N_SOFT)]
    ph = np.r_[phi + 0.02 * np.arange(N_HARD), phi + 0.5 + 0.1 * np.arange(N_SOFT)] + 0.01 * j
    m = np.r_[np.full(N_HARD, 0.13957), np.full(N_SOFT, 0.49368)]
    p = np.zeros((n, 4), np.float32)
    p[:, 1], p[:, 2], p[:, 3] = pt * np.cos(ph), pt * np.sin(ph), pt * np.sinh(eta)
    p[:, 0] = np.sqrt(pt ** 2 * np.cosh(eta) ** 2 + m ** 2)
    return {"pid": np.r_[np.full(N_HARD, 211), np.full(N_SOFT, 321)].astype(np.int32),
            "pstat": np.zeros(n, np.int32), "p": p, "x": np.zeros((n, 4), np.float32)}


def add_extra_frag(stem, name, n_ev, n_frag=2, empty=()):
    """The extra_frag file of ``stem``: ``n_frag`` fragmentations per event (events in
    ``empty`` have no extras: empty samples, as the hadronizer writes them, and the trigger's
    scattering row only)."""
    import h5py

    from jetscape.hadrons_h5 import HadronH5Writer
    with h5py.File(f"{stem}_particlize.h5", "r") as f:
        uuid = str(f.attrs["file_uuid"])
    none = {"pid": np.zeros(0, np.int32), "pstat": np.zeros(0, np.int32),
            "p": np.zeros((0, 4), np.float32), "x": np.zeros((0, 4), np.float32)}
    with HadronH5Writer(f"{stem}_hadrons_extra_frag.h5", tag="extra_frag", n_samples=n_frag,
                        attrs={"source_uuid": uuid}, initiators=True) as w:
        for e in range(n_ev):
            if e in empty:
                w.append_unit([none] * n_frag, unit=e, event=e, seed=e,
                              initiators=np.zeros((0, 11)), initiator_scatter=[],
                              scatterings=SCAT[:1])
                continue
            ini = np.zeros((2, 11))
            ini[:, 0], ini[:, 1], ini[:, 3], ini[:, 6] = [3, 4], [21, 1], [20.0, 16.0], 30.0
            w.append_unit([_extra_hadrons(name, e, j) for j in range(n_frag)], unit=e, event=e,
                          seed=100 + e, initiators=ini, initiator_scatter=[1, 2],
                          scatterings=SCAT)


def zjet_files(directory, name="A", bg_ids=(0, 0), extras=True, minbias=False, **kw):
    """One synthetic production file in ``directory`` (see the module docstring) -> its
    stem."""
    import h5py

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    attrs = dict(WINDOW, **(kw.pop("attrs_extra", None) or {}))
    if extras or minbias:
        attrs["extra_hard"] = True
    if minbias:
        attrs["extra_no_trigger"] = True
    events = kw.pop("events_extra", None) or [{} for _ in bg_ids]
    events = [{"pthat_bin": 0, "pthat": 60.0, **ev} for ev in events]
    if not minbias:
        events = [{**z_record(name, e), **ev} for e, ev in enumerate(events)]
    stem = _seed_files(directory, name, bg_ids, events_extra=events,
                       attrs_extra=attrs or None, **kw)
    with h5py.File(f"{stem}.h5", "a") as f:            # the pair file: a recoiling gluon
        ini = f["shower/initiators"][:]
        ini[:, 1], ini[:, 3], ini[:, 4] = 21, 40.0 * ini[:, 3], 40.0 * ini[:, 4]
        ini[:, 6] = 40.0
        f["shower/initiators"][...] = ini
        off = f["shower/initiator_offsets"][:]
    if not minbias:                 # as hadronize.py: the trigger's initiators in bulk_jet
        from jetscape.hadrons_h5 import add_initiators
        add_initiators(f"{stem}_hadrons_bulk_jet.h5", ini, off)
    if extras or minbias:
        add_extra_frag(stem, name, len(bg_ids))
    if minbias:
        Path(f"{stem}_hadrons_jet_frag.h5").unlink()
    return stem
