#!/usr/bin/env python3
"""Two same-seed jobs of the scan must share what the scanned setting cannot touch.

    python check_pairing.py <base *_hadrons_jet_frag.h5> <variant *_hadrons_jet_frag.h5> \\
        [--level jets | events]

Compares the jet files' parton lists (initiators/, partons/, exact) and their per-event
records (units/, exact).  The hadrons always differ: their iSS and fragmentation seeds include
each job's random file_uuid (units/seed).
  jets    (default) a setting that acts after the energy loss, on MUSIC_2 alone (CausalLiquefier's
          tau_delay, ...): the same events, Z's, hydro backgrounds, partons and droplets; only
          the jet leg's MUSIC record may change (MUSIC_2).
  events  a setting that acts during the energy loss (the liquefier's e_threshold, alpha_s, ...):
          the same events, Z's, hydro backgrounds and shower initiators (Pythia's partons); the
          final partons, the droplets and the jet leg's fluid may change (ELOSS).
Prints what differs; exit 1 if anything outside the level's allowed changes does."""
import argparse

import h5py
import numpy as np

#: per-event records the jet leg's MUSIC (MUSIC_2) writes, and the event's hadronization seed
#: (hadronizer.unit_seed, keyed by the job's random file_uuid)
MUSIC_2 = {"E_droplets_early", "E_droplets_late", "n_cells_jet", "ntau_jet_music",
           "jet_hit_boundary", "jet_hit_edge", "wall_s", "seed"}
#: and what the energy loss decides: the final partons and the droplets
ELOSS = MUSIC_2 | {"n_partons", "n_droplets", "E_droplets"}
LEVELS = {"jets": (MUSIC_2, ("initiators", "partons")), "events": (ELOSS, ("initiators",))}


def same(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return False
    if x.dtype.kind in "fc":
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--level", choices=tuple(LEVELS), default="jets")
    x = ap.parse_args()
    allowed, groups = LEVELS[x.level]
    bad, changed = [], []
    with h5py.File(x.a, "r") as a, h5py.File(x.b, "r") as b:
        for grp in ("initiators", "partons"):
            for k in sorted(set(a[grp]) | set(b[grp])):
                if k not in a[grp] or k not in b[grp] or not same(a[grp][k][()], b[grp][k][()]):
                    (bad if grp in groups else changed).append(f"{grp}/{k}")
        for k in sorted(set(a["units"]) | set(b["units"])):
            if k not in a["units"] or k not in b["units"]:
                bad.append(f"units/{k} (absent)")
                continue
            if not same(a["units"][k][()], b["units"][k][()]):
                (changed if k in allowed else bad).append(f"units/{k}")
    print(f"{x.a.split('/')[-1]} / {x.b.split('/')[-1]} ({x.level}): "
          + (f"allowed changes: {', '.join(changed)}; " if changed else "")
          + ("paired" if not bad else f"NOT PAIRED, differences: {bad}"))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
