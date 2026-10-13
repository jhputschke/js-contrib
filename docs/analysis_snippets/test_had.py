import sys, time
OUT = sys.argv[1]
STEM = sys.argv[2] if len(sys.argv) > 2 else "AuAu_0_10_jet_seed0001"
import h5py, hdf5plugin
import xml.etree.ElementTree as ET
with h5py.File(f"{OUT}/{STEM}.h5", "r") as f:
    job = ET.fromstring(f.attrs["prod_user_xml"])
    late = job.findtext(".//Hydro/MUSIC/late_delta_tau")
    print("late_delta_tau =", late.strip() if late else "absent: constant step")

import numpy as np
from jetscape.hadrons_h5 import HadronFileReader

def lead(info):
    """phi and pseudorapidity of the event's hardest shower-initiating parton."""
    ini = info.initiators()                            # (K, 11): shower, pid, pstat, px, py, pz, E, ...
    px, py, pz = ini[np.argmax(np.hypot(ini[:, 3], ini[:, 4])), 3:6]
    return np.arctan2(py, px), np.arcsinh(pz / np.hypot(px, py))

def dphi(ev, info):                                    # azimuth relative to the leading parton
    return np.mod(ev.phi - lead(info)[0] + np.pi / 2, 2 * np.pi) - np.pi / 2

def soft_charged(ev, info):
    return ev.charged & (ev.pt > 0.2) & (ev.pt < 4) & (np.abs(ev.eta) < 1)

t0 = time.time()
with HadronFileReader(OUT) as r:                       # every *_particlize.h5 in OUT
    print(r.n_files, "files,", r.n_events, "events, tags", r.tags())
    # per event: <sample> over the oversamples, then averaged over events
    pt, pt_err = r.hist("bulk_bg", "pt", np.linspace(0, 3, 31), mask="charged")
    n_mid, n_err = r.total("bulk_jet", mask=lambda ev, info: ev.charged & (np.abs(ev.eta) < 1))
    # the wake: <jet leg> - <its background>, each event against its own background
    bins = np.linspace(-np.pi / 2, 3 * np.pi / 2, 17)
    wake, wake_err = r.jet_minus_background(dphi, bins, mask=soft_charged)
    wake_f, _ = r.jet_minus_background(dphi, bins, mask=soft_charged, fragments=True)
    print("dN_ch/deta(|eta|<1)/2 =", n_mid / 2, "+-", n_err / 2)
    print("wake sum", wake.sum(), "+-", np.sqrt((wake_err ** 2).sum()), "with frag", wake_f.sum())
    ev = r.jet_event(0, 17)                            # dict: pid, pstat, p, x, origin
    bg = r.background_event(0, 17)
    info = r.event_info(0)
    print(info, len(ev["pid"]), len(bg["pid"]), ev["p"].shape, np.unique(ev["origin"]))
    print("precision", r.precision(0), "eta_max", r.eta_max(), "selection", r.selection())
    print(r.n_oversamples(0), r.n_frag(0), r.n_samples("bulk_bg", 0))
print("took", time.time() - t0)
