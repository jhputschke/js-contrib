#!/usr/bin/env python3
"""Nch, [p_T], sigma([p_T]) and v_n{2} with an eta gap of a production, per composition (full
event, without extras, bulk_jet, bulk_bg) and as the paired difference to bulk_bg, in classes
of N_coll.  The definitions are in flow.py's docstring and README.md.

    python bulk_obs.py <dir | glob | files ...> [--ncoll-classes 0 10 30 50 100]
                       [--out results.json | results.npz] [--table events.npz]

Sources: a directory (every production file in it), a quoted glob of particlize or hadron
files, or several of either (as jetscape.hadrons_h5.HadronFileReader reads them; a
directory given among several sources contributes its ``*_hadrons_bulk_jet.h5``).  Works on
Pb+Pb, Au+Au, O+O and Ne+Ne files, with or without extra hard scatterings, and on
minimum-bias files (run_prod_jet.py --hard none: no jet_frag).

``--out`` writes the class results (.json: {class: {composition: {quantity: [value,
error]}}}, .npz: one [value, error] array per "class|composition/quantity"), ``--table`` the
per-event table (.npz; flow.event_table's columns with "." written as "__").
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flow  # noqa: E402
from jetscape.hadrons_h5 import HadronFileReader  # noqa: E402

ROWS = ("nch", "dnch_deta", "pt", "sigma_pt", "sigma_pt_dyn", "sigma_pt_one")
DROWS = ("dnch", "rdnch", "dpt")


def sources(args):
    """One source as given (HadronFileReader takes a directory or a glob), several expanded
    to files."""
    if len(args) == 1:
        return args[0]
    out = []
    for a in args:
        if os.path.isdir(a):
            out += sorted(glob.glob(os.path.join(a, "*_hadrons_bulk_jet.h5")))
        elif any(c in a for c in "*?["):
            out += sorted(glob.glob(a))
        else:
            out.append(a)
    if not out:
        raise SystemExit(f"no files found in {args}")
    return out


def parse_events(spec):
    """'0:100', '3,5,9', '0:10,20:30' -> a list of global event indices."""
    if spec is None:
        return None
    out = []
    for part in spec.split(","):
        if ":" in part:
            a, b = part.split(":")
            out += range(int(a), int(b))
        elif part:
            out.append(int(part))
    return out


def fmt(v, e, d=4):
    if not np.isfinite(v):
        return "nan".center(19)
    if not np.isfinite(e) or e == 0:
        return f"{v:.{d}g}".rjust(9) + " " * 10
    return f"{v:.{d}g}".rjust(9) + " +- " + f"{e:.2g}".ljust(6)


def print_class(label, res, harmonics, out=None):
    out = out or sys.stdout
    n = int(res["full/n_events"][0])
    nb = int(res["n_blocks"][0])
    print(f"\n== {label}: {n} events, {nb} jackknife blocks", file=out)
    heads = ["<Nch>", "dNch/deta", "<[pT]> GeV", "sigma([pT])", "sigma_dyn", "sigma_one"]
    heads += [f"v{h}{{2}}" for h in harmonics]
    print(" " * 16 + "".join(h.center(19) for h in heads), file=out)
    for c in flow.COMPOSITIONS:
        cells = [fmt(*res[f"{c}/{q}"]) for q in ROWS]
        cells += [fmt(*res[f"{c}/v{h}"]) for h in harmonics]
        print(f"{c:<16}" + "".join(cells), file=out)
    heads = ["dNch", "dNch/Nch_bg", "d<[pT]> GeV"]
    for h in harmonics:
        heads += [f"dc{h} = d<<2>>", f"dv{h}{{2}}"]
    print(" " * 16 + "".join(h.center(19) for h in heads), file=out)
    for d in flow.DELTAS:
        cells = [fmt(*res[f"{d}/{q}"]) for q in DROWS]
        for h in harmonics:
            cells += [fmt(*res[f"{d}/dc{h}"]), fmt(*res[f"{d}/dv{h}"])]
        print(f"{d:<16}" + "".join(cells), file=out)
    if f"{flow.DELTAS[0]}/kick{harmonics[0]}" not in res:
        return
    heads = []                           # the flow vector's kick (flow.py's docstring)
    for h in harmonics:
        heads += [f"|dV{h}|", f"|dV{h}|/v{h}", f"2<V.dV>{h}", f"r{h}"]
    print(" " * 16 + "".join(x.center(19) for x in heads), file=out)
    for d in flow.DELTAS:
        cells = []
        for h in harmonics:
            cells += [fmt(*res[f"{d}/{q}{h}"]) for q in ("kick", "rkick", "align", "r")]
        print(f"{d:<16}" + "".join(cells), file=out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="+", help="directory, quoted glob, or files")
    ap.add_argument("--eta-max", type=float, default=0.8, help="|eta| window of Nch, [pT]")
    ap.add_argument("--pt-min", type=float, default=0.2)
    ap.add_argument("--pt-max", type=float, default=3.0)
    ap.add_argument("--eta-gap", type=float, default=1.0,
                    help="subevents eta < -gap/2 and eta > gap/2")
    ap.add_argument("--flow-eta-max", type=float, default=None,
                    help="|eta| reach of the subevents (default: --eta-max)")
    ap.add_argument("--harmonics", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--events", default=None, help="global events, e.g. 0:100 or 3,5,9")
    ap.add_argument("--ncoll-min", type=float, default=None)
    ap.add_argument("--ncoll-max", type=float, default=None)
    ap.add_argument("--ncoll-classes", type=float, nargs="+", default=None,
                    help="percent edges in N_coll (largest first), e.g. 0 10 30 50 100")
    ap.add_argument("--max-blocks", type=int, default=100, help="jackknife blocks (at most)")
    ap.add_argument("--out", default=None, help="class results, .json or .npz")
    ap.add_argument("--table", default=None, help="per-event table, .npz")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    cuts = flow.Cuts(eta_max=a.eta_max, pt_min=a.pt_min, pt_max=a.pt_max, eta_gap=a.eta_gap,
                     flow_eta_max=a.flow_eta_max, harmonics=tuple(a.harmonics))
    t0 = time.time()
    with HadronFileReader(sources(a.source)) as reader:
        tags = reader.tags()
        extras = reader.has_extras()
        stems = reader.stems
        correlated = reader.correlated
        n_all = reader.n_events

        def progress(i, n):
            if not a.quiet and (i % 50 == 0 or i == n):
                print(f"\r  {i}/{n} events  {time.time() - t0:.0f} s", end="",
                      file=sys.stderr, flush=True)

        t = flow.event_table(reader, cuts, parse_events(a.events), progress=progress)
    if not a.quiet:
        print(file=sys.stderr)

    print(f"{len(stems)} production file(s), {n_all} events, tags {', '.join(tags)}")
    if "jet_frag" not in tags:
        print("  no jet_frag in (some) files (minimum bias): full = bulk_jet + extra_frag")
    if not extras:
        print("  no extra_frag: full = no_extras")
    elif "extra_frag" not in tags:
        print("  extra_frag in some files only")
    print(f"  bulk_jet / bulk_bg sampled correlated: {correlated}")
    print(f"  cuts: {json.dumps(cuts.as_dict(), separators=(',', ':'))[:200]}")
    if len(t["event"]):
        print(f"  oversamples per event: bulk_jet {np.unique(t['full.K']).tolist()}, "
              f"bulk_bg {np.unique(t['bulk_bg.K']).tolist()};  N_coll "
              f"{np.nanmin(t['n_coll']):.0f}-{np.nanmax(t['n_coll']):.0f};  extras per event "
              f"{t['n_extra'].mean():.2f}, extra_E {t['extra_E'].mean():.0f} GeV")

    nc = None if a.ncoll_min is None and a.ncoll_max is None else (a.ncoll_min, a.ncoll_max)
    base = flow.select(t, n_coll=nc)
    classes = {"all": base}
    if a.ncoll_classes:
        e = a.ncoll_classes
        for lo, hi in zip(e[:-1], e[1:]):
            classes[f"Ncoll {lo:g}-{hi:g} %"] = flow.select(t, n_coll_percent=(lo, hi),
                                                            mask=base)
    results = {}
    for label, m in classes.items():
        res = flow.class_averages(t, m, max_blocks=a.max_blocks)
        results[label] = res
        print_class(label, res, a.harmonics)

    if a.out:
        if a.out.endswith(".npz"):
            np.savez(a.out, **{f"{lab}|{k}": np.array(v) for lab, res in results.items()
                               for k, v in res.items()},
                     cuts=json.dumps(cuts.as_dict()), sources=np.array(stems))
        else:
            doc = {"cuts": cuts.as_dict(), "sources": stems, "tags": list(tags),
                   "extras": bool(extras), "correlated": bool(correlated),
                   "classes": {lab: flow.nested(res) for lab, res in results.items()}}
            with open(a.out, "w") as f:
                json.dump(doc, f, indent=1, default=float)
        print(f"\nwrote {a.out}")
    if a.table:
        np.savez(a.table, **flow.table_arrays(t), sources=np.array(stems))
        print(f"wrote {a.table}")
    return results


if __name__ == "__main__":
    main()
