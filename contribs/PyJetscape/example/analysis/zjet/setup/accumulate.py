#!/usr/bin/env python3
"""Histograms of the Z+jet test (README.md) for compare.py -> data/hists.npz.

Pb+Pb (data/pbpb), per hydro background (compare.py's jackknife):
  * (track pT class, deta, dphi, eta_lab) of the Z-correlated yield, wake
    (<bulk_jet> - <bulk_bg>, paired) and jet fragments, weights w (zjet_wake.event_weights:
    window cross section / events x N_coll), deta = eta - y_Z, dphi = phi - phi_Z;
  * (track pT class, deta) with weights w^2, for the mixed events' self pairs.
pp (data/vacuum): the same from jet_frag.  Tracks: charged, |eta| < 2.4, never the Z's
leptons; events: CMS 2507.09307's selection (zjet_wake.CMS_2507).

The reader's histograms are sums over the events of w_i x yield_i divided by the number of
events; here they are stored per background group with that group's event count."""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))       # jetcorr
sys.path.insert(0, os.path.dirname(HERE))                        # zjet_wake.py
try:
    import jetscape  # noqa: F401
except ImportError:                                              # the source tree
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "python"))
import zjet_wake as zw                                           # noqa: E402
from jetcorr import runs                                         # noqa: E402
from jetscape.hadrons_h5 import HadronFileReader                 # noqa: E402

PT = np.asarray(zw.PT_CLASSES)                    # 1-2, 2-4, 4-10 GeV
DETA = np.linspace(-4.8, 4.8, 25)                 # 0.4: |eta|, |y_Z| < 2.4
DPHI = np.linspace(-np.pi / 2, 3 * np.pi / 2, 21)  # CMS: pi / 10
ETA = np.linspace(-2.4, 2.4, 49)                  # 0.1, lab frame


def pt_deta_dphi_eta(ev, info):
    deta, dphi = zw.z_frame(ev, info)
    return ev.pt, deta, dphi, ev.eta


def pt_deta(ev, info):
    return ev.pt, zw.z_frame(ev, info)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=runs.data("zjet_setup"),
                    help="directory with pbpb/ and vacuum/ (default: runs/zjet_setup/data)")
    ap.add_argument("--out", default=None, help="default: <data>/hists.npz")
    a = ap.parse_args()
    out_path = a.out or os.path.join(a.data, "hists.npz")
    b4, b2 = (PT, DETA, DPHI, ETA), (PT, DETA)
    mask = zw.track_mask(2.4)
    out = {}
    t0 = time.time()

    r = HadronFileReader(os.path.join(a.data, "pbpb"))
    t = zw.event_table(r)
    sel = zw.select(t, **zw.CMS_2507)
    w = zw.event_weights(t, sel)
    # the hydro background of every event: (production file, background unit)
    infos = [r.event_info(g) for g in range(r.n_events)]
    bg = np.array([(i.file_index, -1 if i.bg_unit is None else i.bg_unit) for i in infos])
    _, group = np.unique(bg, axis=0, return_inverse=True)
    group = group.ravel()
    out.update(w=w, sel=sel, y_z=t["z_y"], n_coll=t["n_coll"], group=group)
    groups = np.unique(group[sel])
    for g in groups:
        ev = np.flatnonzero(sel & (group == g))
        kw = dict(mask=mask, events=ev, per_event=True)
        res = dict(
            wake=r.jet_minus_background(pt_deta_dphi_eta, b4, weights=zw.weight_fn(w), **kw),
            jet=r.hist("jet_frag", pt_deta_dphi_eta, b4, weights=zw.weight_fn(w), **kw),
            wake_w2=r.jet_minus_background(pt_deta, b2, weights=zw.weight_fn(w ** 2), **kw),
            jet_w2=r.hist("jet_frag", pt_deta, b2, weights=zw.weight_fn(w ** 2), **kw))
        for k, (h, e) in res.items():
            out[f"g{g}_{k}"], out[f"g{g}_{k}_err"] = h, e
        out[f"g{g}_n"] = len(ev)
        print(f"Pb+Pb background {g} (N_coll {int(t['n_coll'][ev[0]])}): {len(ev)} selected "
              f"events, {time.time() - t0:.0f} s", flush=True)
    out["groups"] = groups

    rv = HadronFileReader(os.path.join(a.data, "vacuum"))
    tv = zw.event_table(rv)
    sv = zw.select(tv, **zw.CMS_2507)
    wv = zw.event_weights(tv, sv)
    ev = np.flatnonzero(sv)
    kw = dict(mask=mask, events=ev, per_event=True)
    out.update(vac_w=wv, vac_sel=sv, vac_y_z=tv["z_y"])
    out["vac"], out["vac_err"] = rv.hist("jet_frag", pt_deta_dphi_eta, b4,
                                         weights=zw.weight_fn(wv), **kw)
    out["vac_w2"], out["vac_w2_err"] = rv.hist("jet_frag", pt_deta, b2,
                                               weights=zw.weight_fn(wv ** 2), **kw)
    print(f"pp: {len(ev)} selected events, {time.time() - t0:.0f} s", flush=True)
    np.savez(out_path, PT=PT, DETA=DETA, DPHI=DPHI, ETA=ETA, **out)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
