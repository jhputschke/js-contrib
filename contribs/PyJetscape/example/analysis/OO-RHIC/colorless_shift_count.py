#!/usr/bin/env python3
"""
analysis/OO-RHIC/colorless_shift_count.py -- how often ColorlessHadronization's close-pair shift fires and how much
energy it adds (colorless_emul.py, icolored_diag.py).

For each event, both Colorless passes (positive partons: pstat 0, 1000, 1, 22; negative: holes, pstat -1) go through
colorless_emul.colorless_strings, which also adds the fake remnant ends, both with and without the shift.  Reported
per event and pass:
  shifts    the number of pairs closer than 4 Lambda_QCD (0.8 GeV) in 3-momentum, each giving one parton
            dpz = +-0.8 GeV
  E_in      the partons' energy (E from p and the Pythia mass m0, as Colorless hands them to Pythia)
  E_fake    the fake remnant ends' energy (sqrt(s)/6 = 33 GeV each)
  E_shift   the energy the shift adds (E after - E before, same masses)
  share     E_shift / E_out and E_fake / E_out, where E_out = E_in + E_fake + E_shift is what Pythia
            hadronizes (Pythia conserves it)

    python colorless_shift_count.py      # -> ~/prod/AuAu-RHIC_hadtests/diag/colorless_shift_count.json
Model results, not vetted.
"""
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import colorless_emul as CE
import icolored_diag as I

SAMPLES = {
    "vacuum pp_ref 23-30": (f"{I.HT}/from3090/pp_ref_w23_30", 4, None),
    "T1 Au+Au MATTER only 23-30": (f"{I.HT}/auau_matteronly_w23_30", 10, None),
    "B1 Au+Au 0-10% 23-30": (f"{I.HT}/from3090/auau_med_w23_30", 20, None),
    "B2 Au+Au 0-10% (Type 3) 23-30": (os.path.expanduser("~/prod/AuAu-RHIC_phys/auau_t3_w23_30"), 20, None),
    "O+O oo_med 23-30, top 10% Npart": (os.path.expanduser("~/prod/OO-RHIC_phys/oo_med_w23_30"), 20, 0.10),
    "O+O oo_med 23-30, all": (os.path.expanduser("~/prod/OO-RHIC_phys/oo_med_w23_30"), 20, None),
    "run C js_chain_t3 (all windows)": (os.path.expanduser("~/prod/AuAu-RHIC_jschain/js_chain_t3"), 10, None),
}
_M0 = None


def m0(pid):
    global _M0
    if _M0 is None:
        _M0 = {}
    if pid not in _M0:
        _M0[pid] = CE.pythia().particleData.m0(int(pid))
    return _M0[pid]


def energy(pid, p):
    return sum(np.sqrt((p[k, 1:] ** 2).sum() + m0(int(pid[k])) ** 2) for k in range(len(pid)))


def one(args):
    f, seed = args
    import h5py
    import hdf5plugin  # noqa: F401
    rng = np.random.default_rng(seed)
    with h5py.File(f) as h:
        n = int(h.attrs["nunits_written"])
        po, par = h["partons/offsets"][: n + 1], h["partons/data"][:]
        npart = h["units/glauber_npart"][:n] if "units/glauber_npart" in h else np.zeros(n)
    rows = []
    for u in range(n):
        q = par[po[u]:po[u + 1]]
        st = q[:, 2].astype(int)
        r = [float(npart[u])]
        for sel in (np.isin(st, (0, 1000, 1)), st == -1):
            x = q[sel]
            if len(x) == 0:
                r += [0, 0, 0, 0, 0]
                continue
            pid0, p0 = x[:, 1], x[:, [3, 4, 5, 6]]
            state = rng.bit_generator.state
            pid_n, _, _, p_n = CE.colorless_strings(pid0, p0, rng, shift=False)
            rng.bit_generator.state = state           # the same fake directions with the shift
            pid_s, _, _, p_s = CE.colorless_strings(pid0, p0, rng, shift=True)
            nfake = len(pid_n) - len(pid0)
            e_in = energy(pid_n[: len(pid0)], p_n[: len(pid0)])
            e_fake = energy(pid_n[len(pid0):], p_n[len(pid0):])
            e_shift = energy(pid_s, p_s) - energy(pid_n, p_n)
            # number of firings, with the C++ loop's own updates (outer parton read once, inner current)
            P = [np.array(v, float) for v in p_n]
            cnt = 0
            for i in range(len(P)):
                pi = P[i].copy()
                for j in range(i + 1, len(P)):
                    if np.sqrt(((pi[1:] - P[j][1:]) ** 2).sum()) < CE.SHIFT:
                        cnt += 1
                        if abs(pi[3]) >= abs(P[j][3]):
                            q_ = pi.copy(); q_[3] += (1.0 if pi[3] >= 0 else -1.0) * CE.SHIFT; P[i] = q_
                        else:
                            P[j] = P[j].copy(); P[j][3] += (1.0 if P[j][3] >= 0 else -1.0) * CE.SHIFT
            r += [cnt, e_in, e_fake, e_shift, nfake]
        rows.append(r)
    return rows


COLS = ["npart"] + [f"{s}_{k}" for s in ("pos", "neg") for k in ("shifts", "E_in", "E_fake", "E_shift", "n_fake")]


def main():
    res = {}
    with ProcessPoolExecutor(18) as ex:
        for name, (d, nf, top) in SAMPLES.items():
            fs = sorted(glob.glob(f"{d}/*_hadrons_jet_frag.h5"))[:nf]
            R = np.array([r for rr in ex.map(one, [(f, 7 + k) for k, f in enumerate(fs)]) for r in rr])
            if top:
                R = R[R[:, 0] >= np.quantile(R[:, 0], 1 - top)]
            m = dict(zip(COLS, R.mean(0)))
            out = {"events": len(R), "npart_mean": m["npart"]}
            for s in ("pos", "neg"):
                e_out = m[f"{s}_E_in"] + m[f"{s}_E_fake"] + m[f"{s}_E_shift"]
                out[s] = {"shifts_per_event": m[f"{s}_shifts"], "shifts_median": float(np.median(R[:, COLS.index(f"{s}_shifts")])),
                          "E_in": m[f"{s}_E_in"], "E_fake": m[f"{s}_E_fake"], "E_shift": m[f"{s}_E_shift"],
                          "fakes_per_event": m[f"{s}_n_fake"], "E_out": e_out,
                          "share_shift": m[f"{s}_E_shift"] / e_out if e_out else 0.0,
                          "share_fake": m[f"{s}_E_fake"] / e_out if e_out else 0.0}
            res[name] = out
            p, q = out["pos"], out["neg"]
            print(f"{name:34s} ev {len(R):5d} Npart {m['npart']:6.1f} | pos: shifts {p['shifts_per_event']:8.1f} (med {p['shifts_median']:.0f})"
                  f" E_in {p['E_in']:6.1f} fake {p['E_fake']:5.1f} shift {p['E_shift']:6.1f} -> {100 * p['share_shift']:4.1f}% shift,"
                  f" {100 * p['share_fake']:4.1f}% fake | holes: shifts {q['shifts_per_event']:8.1f} E_in {q['E_in']:5.1f}"
                  f" fake {q['E_fake']:5.1f} shift {q['E_shift']:6.1f} -> {100 * q['share_shift']:4.1f}% shift", flush=True)
    json.dump(res, open(f"{I.HT}/diag/colorless_shift_count.json", "w"), indent=1)


if __name__ == "__main__":
    main()
