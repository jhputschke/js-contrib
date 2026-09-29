#!/usr/bin/env python3
"""
example/prod_AuAu_0_10_jet/run_h5toROOT.py

Convert a hadronized campaign to ROOT: every production file's hadron files (hadronize.py /
run_hadronize.py) and the event information of its particlize file go into one ROOT file,
P at a time, and a campaign file adds what needs all files (the cross sections).  The
run_hadronize.py counterpart for the ROOT export (root_export/README.md).

    ./run_h5toROOT.py out_had -j 4                       # RNTuple, next to the inputs
    ./run_h5toROOT.py out_had -j 4 --out-dir out_root --no-x --eta-max 1 --charged
    ./run_h5toROOT.py out_had -j 4 --format ttree        # for ROOT < 6.34
    ./run_h5toROOT.py out_had --dry-run                  # the plan only

INPUTS are directories (their ``*_particlize.h5``), particlize or hadron files, or globs.

Per production file ``<stem>``: ``<stem>_hadrons.root`` with

    bulk_jet, bulk_bg, jet_frag   one entry per sample (oversample): event, unit, sample,
                                  bg_unit, pid[n], pstat[n], E/px/py/pz[n], t/x/y/z[n]
                                  (root_export/hadrons_to_root.py)
    events                        one entry per event: the particlize file's event columns
                                  (pthat_bin, pthat, event_weight, bg_unit, the partons'
                                  pT and y, droplets, ...), this file's cross section of the
                                  event's window (sigma_file_mb), the samples and seeds of
                                  its units, and its shower initiators (ini_*[n_ini])
    windows                       one entry per pTHat window: bounds, this file's cross
                                  section, acceptance and counts (--pthat-bins runs)
    provenance                    TObjString, JSON: every attribute of the particlize and
                                  hadron files (settings, seeds, XML, EoS, precision, cut,
                                  sampling) and the converter's settings

and ``<campaign>_campaign.root`` (--campaign-file) with

    windows                       per pTHat window over all converted files: sigma_mb and
                                  its error (HadronFileReader.pthat_bin_sigma), acceptance,
                                  n_events (with jet-leg hadrons; n_events_produced: all)
                                  and weight_mb = sigma_mb / n_events, the weight of one
                                  event of that window in a cross-section-weighted sum
                                  over windows
    files                         one entry per production file, in HadronFileReader's
                                  order: file_index, n_events, event_offset (global event =
                                  event_offset + event), prod_seed, n_backgrounds
    provenance                    TObjString, JSON: the files (stems, ROOT files, uuids), the
                                  campaign's |eta| cut, precision and sampling, the settings

- **Only complete inputs.**  A production file is converted once its particlize file and
  the hadron files of every --tags tag are complete; the others are reported and skipped.
- **Restarting.**  Outputs are written to ``.part`` and renamed when complete; existing
  outputs are kept (with a warning if they were made with other settings) unless --force.
  Re-running the same command finishes an interrupted pass.
- **Writer.**  --writer auto (default) uses ROOT (PyROOT) for the hadron ntuples when it
  imports, uproot otherwise; the small tables are always written by uproot.  See
  root_export/README.md for sizes and speeds.
- **Memory.**  ~80 B per hadron of the largest tag while it is converted: ~3.5 GB for a
  15-event file at 400 oversamples.
- **Statistics.**  Oversamples of one event, and the events of one background, are not
  independent: see root_export/README.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import multiprocessing
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "root_export"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

TAGS = ("bulk_jet", "bulk_bg", "jet_frag")
#: settings recorded in provenance; an existing output made with others gets a warning
SETTING_KEYS = ("tags", "format", "writer", "no_x", "eta_max", "charged", "bits_p", "bits_x",
                "compression")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="directories, particlize/hadron files or globs")
    p.add_argument("-j", "--jobs", type=int, default=1, help="processes at once (default 1)")
    p.add_argument("--out-dir", default=None, dest="out_dir",
                   help="output directory (default: next to each input)")
    p.add_argument("--tags", default=",".join(TAGS),
                   help=f"comma-separated subset of {', '.join(TAGS)} (default: all)")
    p.add_argument("--format", choices=("rntuple", "ttree"), default="rntuple",
                   help="rntuple (default; ROOT >= 6.34 to read) or ttree (any ROOT)")
    p.add_argument("--writer", choices=("auto", "root", "uproot"), default="auto",
                   help="hadron ntuples: auto (default: root if PyROOT imports), root, uproot")
    p.add_argument("--no-x", action="store_true", dest="no_x", help="leave out t, x, y, z")
    p.add_argument("--eta-max", type=float, default=None, dest="eta_max",
                   help="only hadrons with |eta| < ETA_MAX")
    p.add_argument("--charged", action="store_true", help="only charged hadrons")
    p.add_argument("--bits-p", type=int, default=0, dest="bits_p",
                   help="truncated E, px, py, pz (ROOT writer; 0: float, the default)")
    p.add_argument("--bits-x", type=int, default=0, dest="bits_x",
                   help="truncated t, x, y, z (ROOT writer; 0: float, the default)")
    p.add_argument("--compression", type=int, default=505,
                   help="algorithm*100 + level (default 505: ZSTD 5)")
    p.add_argument("--campaign-file", default=None, dest="campaign_file",
                   help="campaign summary file (default: <out-dir or the first input's "
                        "directory>/<prod_campaign>_campaign.root); 'none' to skip it")
    p.add_argument("--force", action="store_true", help="redo existing outputs")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="show what would be done and exit")
    a = p.parse_args(argv)
    a.tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    bad = set(a.tags) - set(TAGS)
    if bad:
        p.error(f"unknown tag(s) {sorted(bad)}")
    if a.jobs < 1:
        p.error("-j must be >= 1")
    if a.eta_max is not None and not a.eta_max > 0:
        p.error("--eta-max must be > 0")
    return a


# ── finding the inputs ─────────────────────────────────────────────────────────────
def find_stems(inputs):
    """-> sorted production-file stems (paths without ``_particlize.h5``)."""
    suffixes = ("_particlize.h5",) + tuple(f"_hadrons_{t}.h5" for t in TAGS)
    paths = set()
    for item in inputs:
        if os.path.isdir(item):
            paths.update(glob.glob(os.path.join(item, "*_particlize.h5")))
        elif any(c in item for c in "*?["):
            paths.update(glob.glob(item))
        elif os.path.exists(item):
            paths.add(item)
    stems = set()
    for p in paths:
        for suf in suffixes:
            if p.endswith(suf):
                stems.add(os.path.abspath(p[:-len(suf)]))
                break
    return sorted(s for s in stems if os.path.exists(f"{s}_particlize.h5"))


def _complete(path):
    import h5py
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as f:
            return bool(f.attrs.get("complete", False))
    except OSError:
        return False


def input_state(stem, tags):
    """'ready', or why not."""
    if not _complete(f"{stem}_particlize.h5"):
        return "particlize incomplete"
    missing = [t for t in tags if not _complete(f"{stem}_hadrons_{t}.h5")]
    if missing:
        return "hadrons missing or incomplete: " + ", ".join(missing)
    return "ready"


def out_path(stem, out_dir):
    return os.path.join(out_dir or os.path.dirname(stem), os.path.basename(stem) + "_hadrons.root")


# ── the small tables ───────────────────────────────────────────────────────────────
def _jsonable(v):
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()] if v.dtype == object else v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _attrs(path):
    import h5py
    with h5py.File(path, "r") as f:
        return {k: _jsonable(v) for k, v in f.attrs.items()}


def _bg_key_words(keys):
    """32-hex-digit background keys -> (hi, lo) uint64; 0, 0 where missing."""
    hi, lo = np.zeros(len(keys), np.uint64), np.zeros(len(keys), np.uint64)
    for i, k in enumerate(keys):
        k = k.decode() if isinstance(k, bytes) else str(k)
        if len(k) == 32:
            hi[i], lo[i] = int(k[:16], 16), int(k[16:], 16)
    return hi, lo


def event_table(stem, tags):
    """-> ({column: per-event array}, {initiator column: (values, counts)} or None)."""
    import h5py

    from jetscape import h5_compression  # noqa: F401
    from jetscape.hadrons_h5 import INITIATOR_COLUMNS

    with h5py.File(f"{stem}_particlize.h5", "r") as f:
        ev = f["events"]
        n = int(f.attrs.get("nevents_written", f.attrs.get("nevents")))
        cols = {}
        for k in ev:
            v = ev[k][:n]
            if k == "bg_key":
                cols["bg_key_hi"], cols["bg_key_lo"] = _bg_key_words(v)
            elif v.dtype.kind in "iub":
                cols[k] = v.astype(np.int32)
            elif v.dtype.kind == "f":
                cols[k] = v.astype(np.float64)
        sigma = f.attrs.get("pthat_bin_sigma_gen")
    if "event" not in cols:
        cols["event"] = np.arange(n, dtype=np.int32)
    if sigma is not None and "pthat_bin" in cols:
        k = cols["pthat_bin"]
        ok = (k >= 0) & (k < len(sigma))
        cols["sigma_file_mb"] = np.where(ok, np.asarray(sigma, float)[np.clip(k, 0, len(sigma) - 1)],
                                         np.nan)
    ini = None
    for t in TAGS:
        path = f"{stem}_hadrons_{t}.h5"
        if t not in tags or not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            units = {k: f["units"][k][:] for k in f["units"]}
            if ini is None and "initiators" in f:
                # unit = event for bulk_jet and jet_frag; events without a unit (not
                # hadronized, e.g. --events a:b) get no initiators
                data, off = f["initiators/data"][:], f["initiators/offsets"][:]
                by_event = {int(u): data[off[i]:off[i + 1]]
                            for i, u in enumerate(units["unit"])}
                empty = np.zeros((0, len(INITIATOR_COLUMNS)))
                rows = [by_event.get(int(e), empty) for e in cols["event"]]
                counts = np.array([len(r) for r in rows], np.int64)
                flat = np.concatenate(rows)
                ini = {c: (flat[:, j].astype(np.int32 if c in ("shower", "pid", "pstat")
                                             else np.float64), counts)
                       for j, c in enumerate(INITIATOR_COLUMNS)}
        name = {"bulk_jet": "jet", "bulk_bg": "bg", "jet_frag": "frag"}[t]
        key = cols["bg_unit"] if t == "bulk_bg" else cols["event"]
        n_s = dict(zip(units["unit"].tolist(), units["n_samples"].tolist()))
        seed = dict(zip(units["unit"].tolist(), units["seed"].tolist()))
        cols[f"n_samples_{name}"] = np.array([n_s.get(int(u), 0) for u in key], np.int32)
        cols[f"seed_{name}"] = np.array([seed.get(int(u), -1) for u in key], np.int64)
    if ini is None:
        ini = _pair_file_initiators(stem, n, INITIATOR_COLUMNS)
    return cols, ini


def _pair_file_initiators(stem, n, columns):
    """Initiators from the pair file next to the particlize file (hadron files made before
    hadronize.py copied them), as HadronFileReader finds them; None if it is not there."""
    from jetscape.hadrons_h5 import pair_initiators

    pair = _attrs(f"{stem}_particlize.h5").get("pair_file")
    if not pair:
        return None
    path = os.path.join(os.path.dirname(stem), os.path.basename(str(pair)))
    try:
        found = pair_initiators(path, nevents=n)
    except ValueError:
        return None
    if found is None:
        return None
    data, off = found
    counts = np.diff(np.asarray(off, np.int64))
    return {c: (np.asarray(data)[:, j].astype(np.int32 if c in ("shower", "pid", "pstat")
                                              else np.float64), counts)
            for j, c in enumerate(columns)}


def window_table(attrs):
    bins = attrs.get("pthat_bins")
    if bins is None:
        return None
    bins = np.asarray(bins, float).reshape(-1, 2)
    t = {"window": np.arange(len(bins), dtype=np.int32), "pthat_lo": bins[:, 0],
         "pthat_hi": bins[:, 1]}
    for col, attr, dt in (("sigma_mb", "pthat_bin_sigma_gen", np.float64),
                          ("sigma_err_mb", "pthat_bin_sigma_err", np.float64),
                          ("acceptance", "pthat_bin_acceptance", np.float64),
                          ("n_tried", "pthat_bin_n_tried", np.int64),
                          ("n_kept", "pthat_bin_n_kept", np.int64),
                          ("n_accepted", "pthat_bin_n_accepted", np.int64),
                          ("seed", "pthat_bin_seeds", np.int64)):
        if attrs.get(attr) is not None:
            t[col] = np.asarray(attrs[attr], dtype=dt)
    return t


def write_table(fo, name, cols, fmt, jagged=None, prefix="ini"):
    """A flat table (+ optional jagged columns sharing one count) with uproot."""
    import awkward as ak

    jag = {}
    if jagged:
        jag = {f"{prefix}_{c}": ak.unflatten(v, counts) for c, (v, counts) in jagged.items()}
    if fmt == "rntuple":
        fo.mkrntuple(name, ak.zip({**cols, **jag}, depth_limit=1))
        return
    types = {k: v.dtype for k, v in cols.items()}
    data = dict(cols)
    if jag:
        rec = ak.zip({k[len(prefix) + 1:]: v for k, v in jag.items()})
        types[prefix] = rec.type
        data[prefix] = rec
    tree = fo.mktree(name, types, counter_name=lambda counted: f"n_{counted}",
                     field_name=lambda outer, inner: f"{outer}_{inner}")
    tree.extend(data)


# ── one production file ────────────────────────────────────────────────────────────
def settings_of(a, writer):
    return {"tags": list(a.tags), "format": a.format, "writer": writer, "no_x": a.no_x,
            "eta_max": a.eta_max, "charged": a.charged, "bits_p": a.bits_p,
            "bits_x": a.bits_x, "compression": a.compression}


def convert_stem(stem, out, settings):
    """Write ``out`` for one production file. -> summary dict."""
    import uproot

    import hadrons_to_root as h2r

    t0 = time.time()
    tmp = out + ".part"
    s = settings
    fmt, writer = s["format"], s["writer"]
    part_attrs = _attrs(f"{stem}_particlize.h5")
    prov = {"producer": "js-contrib/contribs/PyJetscape example/prod_AuAu_0_10_jet/"
                        "run_h5toROOT.py",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "stem": os.path.basename(stem),
            "settings": s, "particlize": part_attrs,
            "hadrons": {t: _attrs(f"{stem}_hadrons_{t}.h5") for t in s["tags"]},
            "layout": "one entry per sample in bulk_jet, bulk_bg, jet_frag; jet_frag sample "
                      "k % n_samples_frag goes with bulk_jet sample k (JetEvents.jet_event); "
                      "for correlated files bulk_bg (unit=bg_unit, sample=k) pairs with "
                      "bulk_jet sample k"}
    ev, ini = event_table(stem, s["tags"])
    win = window_table(part_attrs)
    stats = {"stem": stem, "out": out, "n_events": int(len(ev["event"])), "tags": {}}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    loads = dict(with_x=not s["no_x"], eta_max=s["eta_max"], charged=s["charged"])
    level = s["compression"] % 100
    with uproot.recreate(tmp, compression=uproot.ZSTD(level)) as fo:
        write_table(fo, "events", ev, fmt, jagged=ini)
        if win is not None:
            write_table(fo, "windows", win, fmt)
        fo["provenance"] = json.dumps(prov, default=str)
        if writer == "uproot":
            for t in s["tags"]:
                cols = h2r.load_columns(f"{stem}_hadrons_{t}.h5", **loads,
                                        particlize=f"{stem}_particlize.h5")
                h2r.write_uproot_into(fo, cols, fmt=fmt)
                stats["tags"][t] = {"samples": len(cols["soff"]) - 1, "hadrons": len(cols["pid"])}
                del cols
    if writer == "root":
        for t in s["tags"]:
            cols = h2r.load_columns(f"{stem}_hadrons_{t}.h5", **loads,
                                        particlize=f"{stem}_particlize.h5")
            h2r.write_root(tmp, cols, fmt=fmt, bits_p=s["bits_p"], bits_x=s["bits_x"],
                           compression=s["compression"], update=True)
            stats["tags"][t] = {"samples": len(cols["soff"]) - 1, "hadrons": len(cols["pid"])}
            del cols
    os.replace(tmp, out)
    stats["seconds"] = time.time() - t0
    stats["bytes"] = os.path.getsize(out)
    return stats


def existing_settings(out):
    """The converter settings recorded in an existing output (None if unreadable)."""
    try:
        import uproot
        with uproot.open(out) as f:
            return json.loads(str(f["provenance"]))["settings"]
    except Exception:                                  # noqa: BLE001
        return None


# ── the campaign file ──────────────────────────────────────────────────────────────
def write_campaign(path, stems, outs, settings):
    """Campaign-level tables over the converted production files."""
    import uproot

    from jetscape.hadrons_h5 import HadronFileReader

    with HadronFileReader(stems) as r:
        files = {"file_index": np.arange(r.n_files, dtype=np.int32),
                 "n_events": np.array([f["nevents"] for f in r._files], np.int32),
                 "event_offset": np.asarray(r._offsets[:-1], np.int64),
                 "prod_seed": np.array([int(f["seed"]) if f["seed"] is not None else -1
                                        for f in r._files], np.int64),
                 "n_backgrounds": np.array([len(np.unique(f["bg_unit"]))
                                            if f["bg_unit"] is not None else 0
                                            for f in r._files], np.int32)}
        win = None
        if r.pthat_bins is not None:
            bins = np.asarray(r.pthat_bins, float).reshape(-1, 2)
            k = range(len(bins))
            sig = [r.pthat_bin_sigma(i) for i in k]
            acc = [r.pthat_bin_acceptance(i) for i in k]
            # the events that have hadrons, as jet_minus_background counts them (all, for
            # a completely hadronized campaign)
            has = "bulk_jet" in settings["tags"]
            n_all = np.array([len(r.pthat_bin_events(i)) for i in k], np.int64)
            n_ev = np.array([sum(1 for e in r.pthat_bin_events(i)
                                 if not has or r.n_samples("bulk_jet", e) > 0) for i in k],
                            np.int64)
            s_mb = np.array([x[0] for x in sig])
            win = {"window": np.arange(len(bins), dtype=np.int32), "pthat_lo": bins[:, 0],
                   "pthat_hi": bins[:, 1], "sigma_mb": s_mb,
                   "sigma_err_mb": np.array([x[1] for x in sig]),
                   "acceptance": np.array([x[0] for x in acc]),
                   "acceptance_err": np.array([x[1] for x in acc]),
                   "n_events": n_ev, "n_events_produced": n_all,
                   "weight_mb":np.where(n_ev > 0, s_mb / np.maximum(n_ev, 1), 0.0)}
        prov = {"producer": "run_h5toROOT.py", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "files":[{"file_index": i, "stem": os.path.basename(st),
                           "root_file": os.path.basename(outs[st]),
                           "uuid": r._files[i]["uuid"]} for i, st in enumerate(r.stems)],
                "eta_max": r.eta_max(), "correlated": bool(r.correlated),
                "precision": r.precision(0) if r.n_files else None,
                "settings": settings,
                "weights": "a cross-section-weighted sum over windows weighs each event of "
                           "window k with weight_mb[k] (mb per event); within one window "
                           "every event weighs the same. Oversamples are samples of their "
                           "event, not events."}
    prov["campaigns"] = sorted({_attrs(f"{st}_particlize.h5").get("prod_campaign", "")
                                for st in stems})
    fmt = settings["format"]
    tmp = path + ".part"
    with uproot.recreate(tmp, compression=uproot.ZSTD(settings["compression"] % 100)) as fo:
        write_table(fo, "files", files, fmt)
        if win is not None:
            write_table(fo, "windows", win, fmt)
        fo["provenance"] = json.dumps(prov, default=str)
    os.replace(tmp, path)
    return win


def campaign_path(a, stems):
    if a.campaign_file and a.campaign_file.lower() == "none":
        return None
    if a.campaign_file:
        return os.path.abspath(a.campaign_file)
    names = sorted({str(_attrs(f"{s}_particlize.h5").get("prod_campaign", "")) for s in stems})
    name = names[0] if len(names) == 1 and names[0] else "campaign"
    return os.path.join(a.out_dir or os.path.dirname(stems[0]), f"{name}_campaign.root")


# ── main ───────────────────────────────────────────────────────────────────────────
def main(argv=None):
    a = parse_args(argv)
    if a.out_dir:
        a.out_dir = os.path.abspath(a.out_dir)
    import hadrons_to_root as h2r

    writer = a.writer
    if writer == "auto":
        writer = "root" if h2r.have_pyroot() else "uproot"
        if writer == "uproot":
            print("run_h5toROOT.py: note -- PyROOT not available, the hadron ntuples are "
                  "written with uproot (larger files, see root_export/README.md)",
                  file=sys.stderr)
    if writer == "uproot" and (a.bits_p or a.bits_x):
        sys.exit("run_h5toROOT.py: --bits-p/--bits-x need the ROOT writer (PyROOT)")
    settings = settings_of(a, writer)

    stems = find_stems(a.inputs)
    if not stems:
        sys.exit(f"run_h5toROOT.py: no particlize files in {a.inputs}")
    plan, done, skipped = [], [], {}
    for st in stems:
        out = out_path(st, a.out_dir)
        state = input_state(st, a.tags)
        if state != "ready":
            skipped[st] = state
        elif os.path.exists(out) and not a.force:
            have = existing_settings(out)
            if have is not None and {k: have.get(k) for k in SETTING_KEYS} != \
                    {k: settings[k] for k in SETTING_KEYS}:
                print(f"run_h5toROOT.py: WARNING -- {os.path.basename(out)} exists with other "
                      f"settings ({have}); kept as it is (--force redoes it)", file=sys.stderr)
            done.append(st)
        else:
            plan.append(st)
    camp = campaign_path(a, stems)
    print(f"run_h5toROOT.py: {len(stems)} production file(s): {len(plan)} to convert, "
          f"{len(done)} already converted, {len(skipped)} not ready; {a.format} by {writer}, "
          f"tags {','.join(a.tags)}, -j {a.jobs}" + (f"; campaign file {camp}" if camp else ""),
          flush=True)
    for st, why in skipped.items():
        print(f"  not ready:  {os.path.basename(st)} ({why})")
    if a.dry_run:
        for st in plan:
            print(f"  convert:    {os.path.basename(st)} -> {out_path(st, a.out_dir)}")
        for st in done:
            print(f"  converted:  {out_path(st, a.out_dir)}")
        return 0

    failed = []
    t0 = time.time()

    def report(st, get):
        try:
            r = get()
        except Exception as err:                           # noqa: BLE001 - report, go on
            failed.append(st)
            print(f"  FAILED      {os.path.basename(st)}: {type(err).__name__}: {err}",
                  flush=True)
            return
        done.append(st)
        had = ", ".join(f"{t} {v['hadrons'] / 1e6:.1f} M" for t, v in r["tags"].items())
        print(f"  done        {os.path.basename(r['out'])}: {r['n_events']} events, "
              f"{had} hadrons, {r['bytes'] / 1e6:.0f} MB in {r['seconds']:.0f} s", flush=True)

    if plan and a.jobs == 1:
        for st in plan:
            report(st, lambda st=st: convert_stem(st, out_path(st, a.out_dir), settings))
    elif plan:
        ctx = multiprocessing.get_context("spawn")        # ROOT and fork do not mix
        with cf.ProcessPoolExecutor(max_workers=a.jobs, mp_context=ctx) as ex:
            futs = {ex.submit(convert_stem, st, out_path(st, a.out_dir), settings): st
                    for st in plan}
            for fut in cf.as_completed(futs):
                report(futs[fut], fut.result)
    for st in failed:
        part = out_path(st, a.out_dir) + ".part"
        if os.path.exists(part):
            os.remove(part)                                # our own partial output
    if camp and done:
        done = sorted(done)
        win = write_campaign(camp, done, {st: out_path(st, a.out_dir) for st in done},
                             settings)
        print(f"  campaign    {camp}: {len(done)} file(s)"
              + (", windows " + ", ".join(f"{lo:g}-{hi:g} GeV: sigma {s:.3g} mb, {n} events"
                                          for lo, hi, s, n in zip(win["pthat_lo"],
                                                                  win["pthat_hi"],
                                                                  win["sigma_mb"],
                                                                  win["n_events"]))
                 if win is not None else ""))
    print(f"run_h5toROOT.py: {len(done)} converted, {len(failed)} failed, {len(skipped)} not "
          f"ready, in {time.time() - t0:.0f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
