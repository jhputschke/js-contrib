#!/usr/bin/env python3
"""
analysis/OO-RHIC/diag_strings.py -- the colour-string topology of what iColored hands to Pythia,
against standalone Pythia's parton level: why the X-SCAPE chain with Pythia's full soft MPI
makes ~3x STAR's underlying event (tuning page, "Full soft MPI").

    # the input partons of 600 events at pTHat 9-11 GeV (tune MPI and Pythia's default MPI):
    python ../../prod_OO_isr/run_prod_oo_isr.py --vacuum --user-xml design/xml/best.xml \\
        --events 600 --seed 42 --pthat-bins 9-11 --store-hadronization-input \\
        --pythia-line "PDF:pSet = 20" --pythia-line "SigmaProcess:alphaSvalue = 0.153" \\
        --pythia-line "MultipartonInteractions:alphaSvalue = 0.153" \\
        [--pythia-line "MultipartonInteractions:pTmin = 0.2"] --outdir PROD/diag/<name> ...
    python diag_strings.py PROD/diag/best PROD/diag/best_mpidef     # + Pythia, CR on and off

With dndeta.py's samples (3000 events at pTHat 9-11 GeV, all four MPI settings, all particles):
    python diag_strings.py PROD/diag/eta/9-11/best_nompi PROD/diag/eta/9-11/best \
        PROD/diag/eta/9-11/best_mpi4 PROD/diag/eta/9-11/best_mpidef

Per event, from the colour tags (col / acol, exact): the colour-singlet systems (open strings
q ... g ... qbar and closed gluon loops), their pieces (neighbouring partons along a string),
the pieces that cross y = 0, the partons beyond |y| > 3.5 (the beam remnants sit there), and the
Lund string-length measure lambda = sum over pieces of ln(1 + m_ij^2 / m0^2) (m0 = 1 GeV,
m_ij^2 = 2 p_i.p_j), to which the string fragmentation's multiplicity is roughly proportional.
The charged hadrons with pT > 0.2 GeV, |eta| < 1 per event check that.  Also the partons'
rapidity distribution (where the strings end).  The means and the
per-event charged multiplicities go to results/diag_strings.json (the notebook's section
"Where the full-MPI excess comes from").

Pythia: the same window, Pythia's default MPI, HardQCD, with the tune's alpha_s (0.153, also the
MPI's) and PDF (pSet 20) as in the X-SCAPE runs, and with Pythia's defaults (0.130, pSet 13); the final
partons before hadronization (beam remnants included), colour reconnection on and off; then the
same event hadronized (forceHadronLevel) for the charged multiplicity.  Pythia's beam remnants can
form junctions (n_junctions; ~60 % of the events with the tune's MPI coupling): chains() follows
colour lines only, so the junction's three legs count as open strings and the pieces at the
junction are missing from lambda; X-SCAPE's input has no junctions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

M0 = 1.0
Y_EDGES = np.arange(-8, 8.0001, 0.25)        # parton rapidity distribution (dndeta.py's bins)


def chains(pid, col, acol):
    """Colour-singlet systems as lists of parton indices (open strings and gluon loops)."""
    by_acol = {int(a): i for i, a in enumerate(acol) if a != 0}
    used = np.zeros(len(pid), dtype=bool)
    out = []
    for i in range(len(pid)):              # open strings start at a colour end (col, no acol)
        if col[i] != 0 and acol[i] == 0 and not used[i]:
            s, j = [], i
            while j is not None and not used[j]:
                used[j] = True
                s.append(j)
                j = by_acol.get(int(col[j])) if col[j] != 0 else None
            out.append((s, False))
    for i in range(len(pid)):              # what is left: closed loops (gluons)
        if not used[i] and (col[i] != 0 or acol[i] != 0):
            s, j = [], i
            while j is not None and not used[j]:
                used[j] = True
                s.append(j)
                j = by_acol.get(int(col[j])) if col[j] != 0 else None
            out.append((s, True))
    return out


def measures(pid, col, acol, p):
    """p: (n, 4) E, px, py, pz."""
    E, px, py, pz = p.T
    y = 0.5 * np.log(np.maximum(E + pz, 1e-9) / np.maximum(E - pz, 1e-9))
    lam, n_piece, n_cross = 0.0, 0, 0
    sys_ = chains(pid, col, acol)
    for s, closed in sys_:
        pairs = list(zip(s[:-1], s[1:])) + ([(s[-1], s[0])] if closed and len(s) > 1 else [])
        for i, j in pairs:
            m2 = 2 * (E[i] * E[j] - px[i] * px[j] - py[i] * py[j] - pz[i] * pz[j])
            lam += np.log1p(max(m2, 0.0) / M0 ** 2)
            n_piece += 1
            n_cross += (y[i] * y[j] < 0)
    return dict(n_partons=len(pid), n_systems=len(sys_), n_pieces=n_piece,
                n_cross=n_cross, n_beam=int(np.sum(np.abs(y) > 3.5)), lam=lam, n_junctions=0,
                yh=np.histogram(y, Y_EDGES)[0])


def xscape(directory):
    import hdf5plugin  # noqa: F401
    import h5py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import observables as O
    rows, nch = [], []
    for path in sorted(Path(directory).glob("*_hadrons_jet_frag.h5")):
        with h5py.File(path) as f:
            d = f["hadronization_input/data"][:]
            off = f["hadronization_input/offsets"][:]
        for k in range(len(off) - 1):
            q = d[off[k]:off[k + 1]]
            rows.append(measures(q[:, 0], q[:, 1], q[:, 2], q[:, 3:7]))
        fd = O.read_file(str(path))
        pt, _, eta, _ = O._kin(fd["p"])
        ch = np.isin(np.abs(fd["pid"]), O._charged_ids()) & (pt > 0.2) & (np.abs(eta) < 1)
        ev = np.repeat(np.arange(len(fd["counts"])), fd["counts"])
        nch.append(np.bincount(ev[ch], minlength=len(fd["counts"])))
    return rows, np.concatenate(nch)


TUNE_COUPLING = ["PDF:pSet = 20", "SigmaProcess:alphaSvalue = 0.153",
                 "MultipartonInteractions:alphaSvalue = 0.153"]


def pythia(cr, n=3000, window=(9, 11), coupling=TUNE_COUPLING):
    """coupling: the tune's PDF and alpha_s (as the X-SCAPE runs), or [] for Pythia's defaults."""
    try:
        import pythia8
    except ImportError:  # the analysis environment's PyPI wheel (pythia8mc)
        import pythia8mc as pythia8
    py = pythia8.Pythia("", False)
    for s in ["Beams:eCM = 200", "HardQCD:all = on", f"PhaseSpace:pTHatMin = {window[0]}",
              f"PhaseSpace:pTHatMax = {window[1]}", *coupling,
              f"ColourReconnection:reconnect = {'on' if cr else 'off'}", "Random:setSeed = on",
              "Random:seed = 4242", "Next:numberCount = 0", "111:mayDecay = off",
              "221:mayDecay = off", "HadronLevel:all = off"]:
        py.readString(s)
    py.init()
    rows, nch = [], []
    while len(rows) < n:
        if not py.next():
            continue
        q = [(x.id(), x.col(), x.acol(), x.e(), x.px(), x.py(), x.pz()) for x in py.event
             if x.isFinal() and (x.col() != 0 or x.acol() != 0)]
        q = np.asarray(q, dtype=float)
        r = measures(q[:, 0], q[:, 1], q[:, 2], q[:, 3:7])
        r["n_junctions"] = py.event.sizeJunction()
        # hadronize this same event (its junctions kept: a re-hadronization of the bare parton
        # list loses them, fails for ~half of the events and drops the busiest ones)
        if not py.forceHadronLevel(False):
            continue
        rows.append(r)
        nch.append(sum(1 for x in py.event if x.isFinal() and x.isCharged() and x.pT() > 0.2
                       and abs(x.eta()) < 1))
    return rows, np.asarray(nch)


KEYS = ("n_partons", "n_systems", "n_pieces", "n_cross", "n_beam", "lam", "n_junctions")


def summary(label, rows, nch):
    m = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
    print(f"{label:30s} " + "  ".join(f"{k} {m[k]:6.1f}" for k in KEYS)
          + f"  | Nch(|eta|<1, pT>0.2) {nch.mean():5.1f}  ({len(rows)} events)")
    yh = np.array([r["yh"] for r in rows]) / np.diff(Y_EDGES)
    return dict(label=label, n_events=len(rows), means=m,
                parton_y=yh.mean(0).tolist(), parton_y_sem=(yh.std(0) / np.sqrt(len(rows))).tolist(),
                sem={k: float(np.std([r[k] for r in rows]) / np.sqrt(len(rows))) for k in KEYS},
                nch_mean=float(nch.mean()), nch_sem=float(nch.std() / np.sqrt(len(nch))),
                nch=nch.astype(int).tolist())


def main(argv):
    import json
    out = {}
    for d in argv:
        rows, nch = xscape(d)
        out["xscape_" + Path(d).name] = summary("X-SCAPE " + Path(d).name, rows, nch)
    for coupling, tag, lab in ((TUNE_COUPLING, "tune", "tune PDF/alpha_s"),
                               ([], "default", "Pythia defaults")):
        for cr in (True, False):
            rows, nch = pythia(cr, coupling=coupling)
            out[f"pythia_{tag}_cr_{'on' if cr else 'off'}"] = summary(
                f"Pythia, {lab}, CR {'on' if cr else 'off'}", rows, nch)
    res = Path(__file__).resolve().parent / "results"
    res.mkdir(exist_ok=True)
    out["y_edges"] = Y_EDGES.tolist()
    (res / "diag_strings.json").write_text(json.dumps(out))
    print("wrote", res / "diag_strings.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
