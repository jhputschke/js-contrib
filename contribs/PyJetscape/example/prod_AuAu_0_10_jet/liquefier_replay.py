#!/usr/bin/env python3
"""
liquefier_replay.py -- one event of a pair file again, with its recorded droplets replayed into
MUSIC_2: the energy the fluid gains against the energy the droplets deposited, with the
production's viscous hydro or with ideal hydro, so that only the hydro changes.

    python liquefier_replay.py PAIR.h5 --event 0 --out OUT/visc.npz
    python liquefier_replay.py PAIR.h5 --event 0 --ideal --out OUT/ideal.npz
    python liquefier_replay.py --compare OUT/visc.npz OUT/ideal.npz

The job's own user XML (the pair file's prod_user_xml) without <Hard> and <Eloss>: the same
3D Glauber event and both MUSIC legs from tau0; a CausalLiquefier (its <Liquefier> block) is
attached to MUSIC_2 by hand and given the event's droplets (source/droplets).  --ideal sets
eta/s and zeta/s to zero in both legs.  The liquefier's frame is whatever the X-SCAPE build
does (<CausalLiquefier><frame>, if it has one; --frame sets it).

The energy is that of the ideal part of T^{mu nu} through the tau surface, in the lab frame
(tutorial 06): E = sum tau dx dy deta [cosh(eta) T^00 - sinh(eta) T^03], with
T^00 = (e + P) g^2 - P and T^03 = (e + P) g^2 v_z.  For ideal hydro it is the whole energy.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EOS_NPZ = os.path.join(HERE, "..", "..", "..", "..", "tutorials", "pyjetscape", "_common",
                       "eos_hotqcd_91.npz")
SHOW = (1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0)


def pressure_of(npz=EOS_NPZ):
    eos = np.load(npz)

    def pressure(e):
        e = np.maximum(e, 1e-12)
        low = e * eos["P"][0] / eos["e"][0]
        return np.where(e < eos["e"][0], low, np.interp(np.log(e), np.log(eos["e"]), eos["P"]))
    return pressure


def energy(frame, tau, src, etas, pressure, win):
    """Lab energy [GeV] of one frame (nx, ny, neta, 4): all cells, and |eta_s| < win."""
    e, vx, vy, vz = (frame[..., c].astype(np.float64) for c in range(4))
    g2 = 1.0 / (1.0 - np.clip(vx * vx + vy * vy + vz * vz, 0, 1 - 1e-12))
    p = pressure(e)
    w = (e + p) * g2
    dens = np.cosh(etas) * (w - p) - np.sinh(etas) * w * vz
    per_eta = dens.sum(axis=(0, 1)) * tau * src.dx * src.dy * src.deta
    return per_eta.sum(), per_eta[np.abs(etas) < win].sum()


def user_xml(pair_xml, ideal, frame):
    root = ET.fromstring(pair_xml)
    for tag in ("Hard", "Eloss"):
        for el in root.findall(tag):
            root.remove(el)
    root.find("nEvents").text = "1"
    for el in root.iter("setReuseHydro"):
        el.text = "false"
    for hydro in root.findall("Hydro"):
        for el in hydro.findall("AddLiquefier"):
            el.text = "false"                        # attached by hand below
        if ideal:
            for music in hydro.findall("MUSIC"):
                for tag, val in (("shear_viscosity_eta_over_s", "0"),
                                 ("T_dependent_Shear_to_S_ratio", "0"),
                                 ("temperature_dependent_bulk_viscosity", "0")):
                    el = music.find(tag)
                    if el is None:
                        el = ET.SubElement(music, tag)
                    el.text = val
    if frame is not None:
        liq = root.find("Liquefier/CausalLiquefier")
        el = liq.find("frame")
        if el is None:
            el = ET.SubElement(liq, "frame")
        el.text = frame
    return ET.tostring(root, encoding="unicode")


def run(a):
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(a.pair, "r") as f:
        x = f.attrs["prod_user_xml"]
        x = x.decode() if isinstance(x, bytes) else x
        off = f["source/offsets"][:]
        drops = f["source/droplets"][off[a.event]:off[a.event + 1]].astype(np.float64)
        tau_delay = float(f.attrs["liquefier_tau_delay"])
        bg_ref = None
        if not a.ideal and "arr_bg" in f and f.attrs.get("nx") == f.attrs.get("nX_MUSIC"):
            bg_ref = (float(f.attrs["tau_min"]), float(f.attrs["dtau"]),
                      np.moveaxis(f["arr_bg"][a.event, ..., int(round((2.0 - f.attrs["tau_min"])
                                                                     / f.attrs["dtau"]))], 0, -1))
    print(f"event {a.event}: {len(drops)} droplets, {drops[:, 4].sum():.2f} GeV;"
          f" hydro {'ideal' if a.ideal else 'viscous (as produced)'}"
          + (f"; liquefier frame {a.frame}" if a.frame else ""))

    import jetscape as js
    from jetscape import pyjetscape_core as core
    from jetscape.bulk_sources import event_array

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    work = tempfile.mkdtemp(prefix="replay_", dir=os.path.dirname(out) or ".")
    xml_path = os.path.join(work, "replay.xml")
    with open(xml_path, "w") as fx:
        fx.write(user_xml(x, a.ideal, a.frame))
    pressure = pressure_of()
    with js.JobDir(work, xml_path) as job:
        jet_scape = js.JetScapePerEvent()
        jet_scape.SetXMLMainFileName(job.main_xml)
        jet_scape.SetXMLUserFileName(job.user_xml)
        jet_scape.Init()
        legs = {t.GetId(): t for t in jet_scape.GetTaskList()}
        bg, jet = legs["MUSIC_1"], legs["MUSIC_2"]
        for leg in (bg, jet):
            leg.set_dump_hydro_only(True)
            if hasattr(leg, "set_store_evolution"):
                leg.set_store_evolution(True)
            leg.set_skip_surface(True)
        liq = core.CausalLiquefier()                 # reads <Liquefier><CausalLiquefier>
        jet.add_a_liquefier(liq)
        jet_scape.ExecInit()
        liq.add_droplets_numpy(drops)
        jet_scape.ExecPerEvent()
        flux = np.asarray(liq.droplet_fluxes_numpy()) if hasattr(liq, "droplet_fluxes_numpy") else None
        curves = {}
        for name, leg in (("bg", bg), ("jet", jet)):
            arr, src, _ = event_array(leg, "native")
            etas = src.eta_min + src.deta * np.arange(arr.shape[3])
            taus = src.tau_min + src.dtau * np.arange(arr.shape[0])
            curves[name] = np.array([energy(arr[k], taus[k], src, etas, pressure, a.window)
                                     for k in range(arr.shape[0])])
            curves[name + "_tau"] = taus
            if name == "bg" and bg_ref is not None:
                k = int(round((2.0 - taus[0]) / src.dtau))
                mine = energy(arr[k], taus[k], src, etas, pressure, a.window)[0]
                ref = energy(bg_ref[2], taus[k], src, etas, pressure, a.window)[0]
                print(f"background at tau = {taus[k]:.2f}: {mine:.1f} GeV, the pair file's "
                      f"{ref:.1f} GeV ({'the same event' if abs(mine / ref - 1) < 1e-3 else 'NOT the same event'})")
            del arr
        jet_scape.ClearPerEvent()
        jet_scape.ExecFinish() if hasattr(jet_scape, "ExecFinish") else None
    n = min(len(curves["bg"]), len(curves["jet"]))
    tau = curves["bg_tau"][:n]
    dE = curves["jet"][:n] - curves["bg"][:n]
    np.savez(out, tau=tau, dE_full=dE[:, 0], dE_win=dE[:, 1], drops=drops,
             tau_delay=tau_delay,
             flux=flux if flux is not None else np.array([]), ideal=a.ideal, window=a.window,
             frame=a.frame or "")
    report(out)


def deposited(d, tau, tau_delay, window=None):
    t = d[:, 0] + tau_delay
    m = np.ones(len(d), bool) if window is None else np.abs(d[:, 3]) < window
    return np.array([d[m & (t <= x), 4].sum() for x in tau])


def report(*files):
    for fn in files:
        z = np.load(fn, allow_pickle=True)
        d, tau = z["drops"], z["tau"]
        full = deposited(d, tau, float(z["tau_delay"]))
        win = deposited(d, tau, float(z["tau_delay"]), float(z["window"]))
        label = ("ideal" if bool(z["ideal"]) else "viscous") + (f", frame {z['frame']}" if str(z["frame"]) else "")
        print(f"\n{os.path.basename(fn)} ({label}): fluid energy gained / droplet energy deposited")
        print(f"{'tau':>6} | {'full grid':>9} {'(GeV)':>14} | {'|eta_s|<' + str(float(z['window'])):>9} {'(GeV)':>14}")
        last = tau[-2] if len(tau) > 1 else tau[-1]   # the last frame both legs have whole
        for t in SHOW + (last,):
            k = int(np.argmin(np.abs(tau - t)))
            r1 = z["dE_full"][k] / full[k] if full[k] > 0 else np.nan
            r2 = z["dE_win"][k] / win[k] if win[k] > 0 else np.nan
            print(f"{tau[k]:6.2f} | {r1:9.3f} {z['dE_full'][k]:6.1f}/{full[k]:6.1f} | {r2:9.3f}"
                  f" {z['dE_win'][k]:6.1f}/{win[k]:6.1f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pair", nargs="?")
    p.add_argument("--event", type=int, default=0)
    p.add_argument("--ideal", action="store_true", help="eta/s = zeta/s = 0 in both legs")
    p.add_argument("--frame", choices=("lab", "fluid"), default=None,
                   help="<CausalLiquefier><frame> (an X-SCAPE build with the option)")
    p.add_argument("--window", type=float, default=1.0)
    p.add_argument("--out", default="replay.npz")
    p.add_argument("--compare", nargs="+", metavar="NPZ", help="print saved results")
    a = p.parse_args(argv)
    if a.compare:
        report(*a.compare)
        return 0
    if not a.pair:
        p.error("PAIR.h5 is needed")
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
