#!/usr/bin/env python3
"""Two same-seed jobs of the scan must share everything but the jet leg's fluid: the same
events, Z's, hydro backgrounds, partons and droplets.

    python check_pairing.py <base *_hadrons_jet_frag.h5> <variant *_hadrons_jet_frag.h5>

Compares the jet files' parton lists (partons/, initiators/, exact) and their per-event
records (units/, exact).  A liquefier setting acts after the energy loss, on MUSIC_2 alone, so
only the jet leg's MUSIC record may change (CHANGES), and the hadrons with their per-event
seeds (units/seed: the iSS and fragmentation seeds include each job's random file_uuid, the
same partons, other random numbers).  Prints what differs; exit 1 if anything outside CHANGES
does."""
import sys

import h5py
import numpy as np

#: per-event records allowed to change: the jet leg's MUSIC (MUSIC_2) record, and the event's
#: hadronization seed (hadronizer.unit_seed, keyed by the job's random file_uuid)
CHANGES = {"E_droplets_early", "E_droplets_late", "n_cells_jet", "ntau_jet_music",
           "jet_hit_boundary", "jet_hit_edge", "wall_s", "seed"}


def same(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return False
    if x.dtype.kind in "fc":
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    pa, pb = sys.argv[1:]
    bad, changed = [], []
    with h5py.File(pa, "r") as a, h5py.File(pb, "r") as b:
        for grp in ("partons", "initiators"):
            for k in sorted(set(a[grp]) | set(b[grp])):
                if k not in a[grp] or k not in b[grp] or not same(a[grp][k][()], b[grp][k][()]):
                    bad.append(f"{grp}/{k}")
        for k in sorted(set(a["units"]) | set(b["units"])):
            if k not in a["units"] or k not in b["units"]:
                bad.append(f"units/{k} (absent)")
                continue
            if not same(a["units"][k][()], b["units"][k][()]):
                (changed if k in CHANGES else bad).append(f"units/{k}")
    print(f"{pa.split('/')[-1]} / {pb.split('/')[-1]}: "
          + (f"allowed changes: {', '.join(changed)}; " if changed else "")
          + ("paired" if not bad else f"NOT PAIRED, differences: {bad}"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
