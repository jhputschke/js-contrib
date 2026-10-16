#!/usr/bin/env python3
"""The leading-parton cut |y| < 1.9: what it removes from CMS 2507.09307's Z sample, and the pp
Delta phi / Delta y against CMS with and without it (vacuum references)."""
import glob
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                        # zjet_wake.py
sys.path.insert(0, HERE)                                         # compare.py
try:
    import jetscape  # noqa: F401
except ImportError:                                              # the source tree
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
import numpy as np
import zjet_wake as zw
import compare as cmp
from jetscape.hadrons_h5 import HadronFileReader
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--with-cut", default=os.path.join(HERE, "data", "vacuum"), help="with --parton-ymax 1.9")
ap.add_argument("--without", default=os.path.join(HERE, "data", "vacuum_zh"), help="without it (make_data.sh vacuum_zh / vacuum_nocut)")
ap.add_argument("--hepdata", default=os.path.join(HERE, "data", "hepdata", "HEPData-ins2945894-v1-yaml"))
a = ap.parse_args()
CUT, NOCUT, HEP = a.with_cut, a.without, a.hepdata
def sample(path):
    r = HadronFileReader(path); t = zw.event_table(r)
    sel = zw.select(t, **zw.CMS_2507); w = zw.event_weights(t, sel, ncoll=False)
    sig = np.array([r.pthat_bin_sigma(i)[0] for i in range(len(r.pthat_bins))]) * 1e9
    nk = np.bincount(t["pthat_bin"], minlength=len(sig))
    xs = (sig[t["pthat_bin"]] / nk[t["pthat_bin"]] * sel).sum()       # pb passing 2507
    return r, t, sel, w, sig, xs
rc, tc, sc, wc, sigc, xsc = sample(CUT)
rn, tn, sn, wn, sign, xsn = sample(NOCUT)
print("sigma x generator acceptance per window [pb]  with cut:", np.round(sigc, 2).tolist())
print("                                           without cut:", np.round(sign, 2).tolist())
print(f"CMS 2507.09307 selection: {xsc:.2f} pb with the cut, {xsn:.2f} pb without -> the cut removes {100*(1-xsc/xsn):.1f} %")
# leading parton |y| in the no-cut sample, among the 2507-selected events
ylead = np.full(rn.n_events, np.nan)
for g in np.flatnonzero(sn):
    ini = rn.event_info(g).initiators()
    ini = np.asarray(ini, float)                       # INITIATOR_COLUMNS: shower, pid, pstat, px, py, pz, E, ...
    pid = np.abs(ini[:, 1]); keep = ~np.isin(pid, (11, 13, 22))
    if not keep.any(): continue
    px, py, pz, E = (ini[keep, c] for c in (3, 4, 5, 6))
    i = np.argmax(np.hypot(px, py))
    ylead[g] = 0.5 * np.log((E[i] + pz[i]) / (E[i] - pz[i]))
m = sn & np.isfinite(ylead)
fw = np.average(np.abs(ylead[m]) > 1.9, weights=wn[m])
print(f"no-cut sample, 2507-selected Z's: leading parton |y| > 1.9 in {100*fw:.1f} % (window-weighted)")
# pp Delta phi / Delta y against CMS, both samples
EDGES = dict(DETA=np.linspace(-4.8, 4.8, 25), DPHI=np.linspace(-np.pi/2, 3*np.pi/2, 21), ETA=np.linspace(-2.4, 2.4, 49))
PT = np.asarray(zw.PT_CLASSES)
def v4(ev, info):
    de, dp = zw.z_frame(ev, info); return ev.pt, de, dp, ev.eta
def v2(ev, info):
    return ev.pt, zw.z_frame(ev, info)[0]
model = cmp.Model(EDGES["DETA"], EDGES["DPHI"], EDGES["ETA"])
for name, r, t, sel, w in (("with cut", rc, tc, sc, wc), ("without", rn, tn, sn, wn)):
    ev = np.flatnonzero(sel); n = len(ev)
    kw = dict(mask=zw.track_mask(2.4), events=ev, per_event=True)
    h, e = r.hist("jet_frag", v4, (PT, EDGES["DETA"], EDGES["DPHI"], EDGES["ETA"]), weights=zw.weight_fn(w), **kw)
    h2, _ = r.hist("jet_frag", v2, (PT, EDGES["DETA"]), weights=zw.weight_fn(w ** 2), **kw)
    (p, ep), (y, ey) = model.observables(n * h, n * e, n * h2, t["z_y"][sel], w[sel])
    y4, ey4 = cmp.rebin_y(y, ey, 1)
    out = []
    for c, tag in enumerate(cmp.PT_TAGS):
        _, yq, sq, sy = cmp.cms(HEP, "phi", "pp", tag); tq = np.hypot(sq, sy)
        chi_p = 0.5 * np.sum((p[c] - yq) ** 2 / (ep[c] ** 2 + tq ** 2))
        _, yy, syy, sly = cmp.cms(HEP, "y", "pp", tag); ty = np.hypot(syy, sly)
        chi_y = 0.5 * np.sum((y4[c] - yy) ** 2 / (ey4[c] ** 2 + ty ** 2))
        out.append(f"{tag.replace('_<_p_t_<_', '-')} GeV: dphi {chi_p:5.1f}/10, dy {chi_y:5.1f}/10, dy(0) {y4[c][9]:+.3f} (CMS {yy[9]:+.3f})")
    print(f"pp {name:9s} ({n} Z's): " + " | ".join(out))
