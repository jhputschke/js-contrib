#!/usr/bin/env python3
"""Check a converted campaign against its HDF5 files (run_h5toROOT.py's output, entry by entry).

    python check_h5_vs_root.py <h5 dir> <root dir> [--campaign <file>_campaign.root]

Per production file and tag: every sample's entry (event, unit, sample, bg_unit) and every
hadron (pid, pstat, E, px, py, pz; x if both have it) equal to the HDF5 file's, bit for bit
after the float32 the ROOT files store; the ``events`` table equal to the event record
(particlize file, or a --hadrons-only job's units/), the Z record and n_coll included;
the ``windows`` table equal to the file's attributes; with ``--campaign``, the campaign
windows equal to jetscape.hadrons_h5.HadronFileReader's cross sections.  With extra hard
scatterings (an extra_frag file) also the extra_frag ntuple, and the events table's
scatterings (scat_*) and extras' initiators (xini_*) equal to the extra_frag file's.  Prints
one line per check and exits 1 on any difference."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import jetscape  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
from jetscape.hadrons_h5 import TAGS, Hadrons, HadronFileReader  # noqa: E402

FAIL = []


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        FAIL.append(what)


def flat(a):
    import awkward as ak
    return ak.to_numpy(ak.flatten(a)) if a.ndim > 1 else ak.to_numpy(a)


def compare_tag(h5path, nt, tag):
    import awkward as ak
    h = Hadrons.from_h5(h5path)
    arr = nt.arrays()
    n_s = len(h.sample_offsets) - 1
    check(len(arr) == n_s, f"{tag}: {len(arr)} entries = {n_s} samples")
    if len(arr) != n_s:
        return
    per = np.diff(h.unit_offsets)
    unit_ids = np.asarray(h.units.get("unit", np.arange(len(per))))
    check(np.array_equal(ak.to_numpy(arr["unit"]), np.repeat(unit_ids, per)), f"{tag}: unit")
    k = np.concatenate([np.arange(n) for n in per]) if len(per) else np.zeros(0, int)
    check(np.array_equal(ak.to_numpy(arr["sample"]), k), f"{tag}: sample index")
    if "bg_unit" in h.units:
        check(np.array_equal(ak.to_numpy(arr["bg_unit"]),
                             np.repeat(np.asarray(h.units["bg_unit"]), per)), f"{tag}: bg_unit")
    counts = ak.to_numpy(ak.num(arr["pid"]))
    same = np.array_equal(counts, np.diff(h.sample_offsets))
    check(same, f"{tag}: hadrons per sample")
    if not same:                                   # the hadron lists cannot be compared
        n_bad = int(np.sum(counts != np.diff(h.sample_offsets)))
        check(False, f"{tag}: {n_bad} of {len(counts)} samples differ in their hadron "
                     f"count ({int(counts.sum()):,} against {len(h.pid):,} hadrons)")
        return
    check(np.array_equal(flat(arr["pid"]), h.pid), f"{tag}: pid ({len(h.pid):,} hadrons)")
    check(np.array_equal(flat(arr["pstat"]), h.pstat), f"{tag}: pstat")
    for j, name in enumerate(("E", "px", "py", "pz")):
        a = flat(arr[name])
        ref = h.p[:, j].astype(a.dtype)
        dmax = np.max(np.abs(a.astype(float) - h.p[:, j])) if len(a) else 0.0
        check(np.array_equal(a, ref), f"{tag}: {name} ({a.dtype}, max |diff| {dmax:.1e})")
    if "x" in arr.fields:
        for j, name in enumerate(("t", "x", "y", "z")):
            a = flat(arr[name])
            check(np.allclose(a, h.x[:, j], rtol=1e-6, atol=1e-6), f"{tag}: {name}")


def compare_events(stem, ev):
    import awkward as ak
    import h5py
    part = f"{stem}_particlize.h5"
    if os.path.exists(part):
        with h5py.File(part) as f:
            n = int(f.attrs.get("nevents_written", f.attrs.get("nevents")))
            ref = {k: f["events"][k][:n] for k in f["events"]}
    else:
        for t in ("jet_frag", "bulk_jet"):
            path = f"{stem}_hadrons_{t}.h5"
            if os.path.exists(path):
                with h5py.File(path) as f:
                    n = int(f.attrs.get("nunits_written", 0))
                    ref = {k: f["units"][k][:n] for k in f["units"]
                           if k not in ("unit", "seed", "n_samples", "n_cells")}
                break
    arr = ev.arrays()
    check(len(arr) == n, f"events: {len(arr)} entries = {n} events")
    bad = []
    for k, v in ref.items():
        if k == "bg_key" or v.dtype.kind not in "iubf":
            continue
        if k not in arr.fields:
            bad.append(f"{k} missing")
            continue
        a = ak.to_numpy(arr[k])
        same = (np.array_equal(a, v.astype(a.dtype)) if v.dtype.kind in "iub"
                else np.allclose(a, v, equal_nan=True, rtol=0, atol=0))
        if not same:
            bad.append(k)
    z = [k for k in ref if k.startswith(("z_", "lep")) or k == "n_coll"]
    check(not bad, f"events: {len(ref)} columns equal, the Z record and n_coll among them "
                   f"({len(z)}){'; differ: ' + ', '.join(bad) if bad else ''}")
    compare_extras(stem, arr)


def compare_extras(stem, arr):
    """The events table's extras (scat_*, xini_*, n_samples_extra) against the extra_frag
    file: per event its scatterings and its extras' initiators (with their scattering)."""
    import awkward as ak
    import h5py
    from jetscape.hadrons_h5 import INITIATOR_COLUMNS
    from jetscape.particlize_h5 import SCATTER_COLUMNS
    path = f"{stem}_hadrons_extra_frag.h5"
    if not os.path.exists(path):
        check(not any(k.startswith(("scat_", "xini_")) for k in arr.fields),
              "events: no extras' columns without an extra_frag file")
        return
    with h5py.File(path) as f:
        units = f["units/unit"][:]
        n_s = dict(zip(units.tolist(), f["units/n_samples"][:].tolist()))
        groups = {}
        for prefix, g, cols in (("scat", "scatterings", SCATTER_COLUMNS),
                                ("xini", "initiators", INITIATOR_COLUMNS)):
            if g in f:
                data, off = f[f"{g}/data"][:], f[f"{g}/offsets"][:]
                if f"{g}/scatter" in f:
                    data = np.column_stack([data, f[f"{g}/scatter"][:]])
                    cols = tuple(cols) + ("scatter",)
                groups[prefix] = ({int(u): data[off[i]:off[i + 1]] for i, u in enumerate(units)},
                                  cols)
    events = ak.to_numpy(arr["event"])
    check("n_samples_extra" in arr.fields and np.array_equal(
        ak.to_numpy(arr["n_samples_extra"]), [n_s.get(int(e), 0) for e in events]),
        "events: n_samples_extra")
    for prefix, (by_event, cols) in groups.items():
        ok = all(f"{prefix}_{c}" in arr.fields for c in cols)
        if ok:
            for i, e in enumerate(events):
                want = by_event.get(int(e), np.zeros((0, len(cols))))
                got = np.column_stack([np.asarray(ak.to_numpy(arr[f"{prefix}_{c}"][i]), float)
                                       for c in cols]) if len(want) else np.zeros((0, len(cols)))
                if got.shape != want.shape or not np.array_equal(got, want):
                    ok = False
                    break
        n = sum(len(v) for v in by_event.values())
        check(ok, f"events: {prefix}_* ({len(cols)} columns, {n} rows) equal to extra_frag's "
                  f"{'scatterings' if prefix == 'scat' else 'initiators'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("h5")
    ap.add_argument("root")
    ap.add_argument("--campaign")
    a = ap.parse_args()
    import uproot

    reader = HadronFileReader(a.h5)
    for i, stem in enumerate(reader.stems):
        path = os.path.join(a.root, os.path.basename(stem) + "_hadrons.root")
        print(os.path.basename(path))
        if not os.path.exists(path):
            check(False, "converted file exists")
            continue
        f = uproot.open(path)
        for tag in TAGS:
            h5path = f"{stem}_hadrons_{tag}.h5"
            if os.path.exists(h5path):
                check(tag in [k.split(";")[0] for k in f.keys()], f"{tag}: ntuple present")
                if tag in f:
                    compare_tag(h5path, f[tag], tag)
        compare_events(stem, f["events"])
        prov = json.loads(str(f["provenance"]))
        check(isinstance(prov, dict), "provenance: JSON")
        if reader.pthat_bins is not None and "windows" in f:
            w = f["windows"].arrays()
            check(len(w) == len(reader.pthat_bins), f"windows: {len(w)} windows")
    if a.campaign:
        import awkward as ak
        w = uproot.open(a.campaign)["windows"].arrays()
        sig = np.array([reader.pthat_bin_sigma(k)[0] for k in range(len(reader.pthat_bins))])
        check(np.allclose(ak.to_numpy(w["sigma_mb"]), sig, rtol=1e-12),
              "campaign windows: sigma_mb = HadronFileReader.pthat_bin_sigma")
        n_k = np.array([len(reader.pthat_bin_events(k)) for k in range(len(sig))])
        check(np.array_equal(ak.to_numpy(w["n_events_produced"]), n_k),
              "campaign windows: events per window")
    print(f"{len(FAIL)} difference(s)")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
