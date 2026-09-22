#!/usr/bin/env python
"""Run the two-stage FastHydro + Matter/LBT + liquefier pipeline.

    python run_two_stage.py --config ../config/fasthydro_twostage.yaml \
                            --user-xml ../config/jetscape_user_fasthydro.xml \
                            --main-xml ../../../../../config/jetscape_main.xml \
                            --events 1 --out out/pair.npz --dump-droplets out/run.droplets.npz

Run it from the X-SCAPE build tree: the framework resolves several paths relative to the
working directory.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import _bootstrap  # noqa: F401  (puts PyJetscape + FastHydro on sys.path)

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="fast_data YAML")
    ap.add_argument("--user-xml", required=True)
    ap.add_argument("--main-xml", default="../config/jetscape_main.xml")
    ap.add_argument("--events", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="output file. A .h5 name writes FNO4d's HDF5 schema (the dataset "
                         "format: arr = jet leg, source/S, arr_bg) and streams one event at "
                         "a time. A .npz name writes a single pair, for a quick look. "
                         "Defaults to run.out from the YAML.")
    ap.add_argument("--dump-droplets", default=None, help="npz for the replay path")
    ap.add_argument("--hard", default="PythiaGun",
                    help="PythiaGun uses the sampled hard-scattering vertex; PGun samples it "
                         "and then zeroes it (PGun.cc:117-120), so every shower starts at the "
                         "fireball centre. '' for no hard process.")
    ap.add_argument("--set", action="append", default=[], metavar="k.p=v",
                    help="dotted override of the YAML, repeatable")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    from fasthydro.config import load_config   # layers the fasthydro: block on fast_data's
    from fasthydro.liquefier_bridge import save_droplets_npz
    from fasthydro.pipeline import build_two_stage
    from jetscape.run_jetscape import run_manual

    cfg = load_config(a.config, a.set)
    nev = a.events if a.events is not None else int(cfg["run"]["nevents"])

    modules, parts = build_two_stage(cfg, user_xml=a.user_xml, main_xml=a.main_xml,
                                     hard=(a.hard or None),
                                     verbose=not a.quiet)

    # The module objects only ever hold the CURRENT event, so a multi-event run has to take
    # each pair as it is produced. For HDF5 that means streaming straight into the writer --
    # peak memory stays at one pair regardless of nevents; for npz we keep them.
    out = a.out or cfg["run"]["out"]
    as_h5 = bool(out) and str(out).endswith((".h5", ".hdf5"))
    pairs, writer, n_written = [], None, [0]
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        if as_h5:
            from fasthydro.h5_writer import PairedH5Writer
            writer = PairedH5Writer(out, cfg, nev)
        bridge_ref = parts["bridge"]
        _orig_clear = bridge_ref.Clear

        def _grab():
            bg_, jet_ = parts["hyd_bg"], parts["hyd_jet"]
            if bg_.arr is not None and jet_.arr is not None:
                if writer is not None:
                    writer.append(n_written[0], bg_, jet_, bridge_ref)
                    n_written[0] += 1
                else:
                    pairs.append((bg_.arr.copy(), jet_.arr.copy(),
                                  None if jet_.src is None else jet_.src.copy()))
            _orig_clear()

        bridge_ref.Clear = _grab

    run_manual(a.main_xml, a.user_xml, modules, n_events=nev)

    bg, jet, bridge = parts["hyd_bg"], parts["hyd_jet"], parts["bridge"]
    nd = len(bridge.droplets) if bridge.droplets is not None else 0
    print(f"\n=== summary ===")
    print(f"droplets (last event) : {nd}")
    if bg.arr is not None and jet.arr is not None:
        dE = np.abs(jet.arr[0] - bg.arr[0])
        print(f"max |e_jet - e_bg|    : {dE.max():.6g} GeV/fm^3")
        if nd == 0 and dE.max() != 0:
            print("  WARNING: no droplets but the legs differ -- the pair is not clean")
    oor = bg.get_out_of_range_count()
    print(f"out-of-grid queries   : {oor}")
    if oor:
        g = bg.g
        print(f"  Matter/LBT sampled the medium outside the stored grid "
              f"(|x|<={-g.x_min:.2f}, |y|<={-g.y_min:.2f}, |eta|<={-g.eta_min:.2f} fm, "
              f"tau {g.tau0:.2f}..{g.tau_max:.2f}); those queries saw vacuum (T=0).\n"
              f"  That is normal for partons that leave the fireball, but if the grid is too "
              f"small it silently removes quenching. Enlarge grid: in the YAML to check.")

    if writer is not None:
        # The last event's Clear() fires inside Finish(), so everything is already in.
        writer.close(complete=(n_written[0] == nev))
        print(f"wrote {out}  ({n_written[0]} event(s), FNO4d HDF5 schema: "
              f"arr = jet leg, source/S, arr_bg = background)")
    elif out:
        if not pairs and bg.arr is not None:          # single event, Clear never fired
            pairs.append((bg.arr, jet.arr, jet.src))
        if len(pairs) == 1:
            b, j, sc = pairs[0]
            np.savez_compressed(out, arr=b, arr_jet=j,
                                src=(sc if sc is not None else np.zeros(0, np.float32)),
                                tau=bg.g.tau_grid())
            print(f"wrote {out}")
        else:
            stem, ext = os.path.splitext(out)
            for i, (b, j, sc) in enumerate(pairs):
                np.savez_compressed(f"{stem}_ev{i}{ext or '.npz'}", arr=b, arr_jet=j,
                                    src=(sc if sc is not None else np.zeros(0, np.float32)),
                                    tau=bg.g.tau_grid())
            print(f"wrote {len(pairs)} files: {stem}_ev0{ext or '.npz'} .. "
                  f"{stem}_ev{len(pairs) - 1}{ext or '.npz'}")
            print("  (a .h5 name instead writes them all into one FNO4d-schema dataset, "
                  "streaming, at one pair of peak memory)")

    if a.dump_droplets:
        os.makedirs(os.path.dirname(os.path.abspath(a.dump_droplets)) or ".", exist_ok=True)
        cfg_sha = hashlib.sha256(repr(sorted(cfg.items())).encode()).hexdigest()
        save_droplets_npz(a.dump_droplets, bridge, cfg_sha256=cfg_sha,
                          ic=parts["ini"].e0,
                          ic_sha256=[jet.ic_sha256 or ""],
                          seeds=[int(cfg["run"]["seed"])],
                          arr_sha256=[hashlib.sha256(jet.arr.tobytes()).hexdigest()
                                      if jet.arr is not None else ""])
        print(f"wrote {a.dump_droplets}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
