#!/usr/bin/env python3
"""
analysis/OO-RHIC/collinear_split.py -- a collinear-safety test of the jet hadronizations (icolored_diag.py).

Copies a jet_frag file with partons/ and hadronization_input/ and splits every gluon with pT > --pt-min into
--n exactly collinear gluons of equal momentum, colour-connected as a vacuum g -> gg chain (the first keeps the
colour, the last the anticolour, new tags in between).  Applied identically to partons/ (Colorless input) and
to hadronization_input/ (iColored input, matched by momentum).  An infrared and collinear safe hadronization
gives the same hadrons (up to fluctuations) before and after.

    python collinear_split.py IN_hadrons_jet_frag.h5 OUT_hadrons_jet_frag.h5 [--pt-min 4] [--n 2]
Model results, not vetted.
"""
import argparse

import h5py
import hdf5plugin  # noqa: F401
import numpy as np


def split_rows(rows, i_pid, i_col, i_acol, i_p, i_e, pt_min, n, tag0, i_mass=None):
    """rows: (N, k).  Returns new rows and the next free tag; i_p = (px, py, pz) columns."""
    out, tag = [], tag0
    for r in rows:
        pt = np.hypot(r[i_p[0]], r[i_p[1]])
        if int(r[i_pid]) == 21 and pt > pt_min and r[i_col] != 0 and r[i_acol] != 0:
            tags = [int(r[i_col])] + [tag + k for k in range(n - 1)] + [int(r[i_acol])]
            tag += n - 1
            for k in range(n):
                q = r.copy()
                for c in i_p:
                    q[c] = r[c] / n
                q[i_e] = r[i_e] / n
                if i_mass is not None:
                    q[i_mass] = 0.0
                q[i_col], q[i_acol] = tags[k], tags[k + 1]
                out.append(q)
        else:
            out.append(r)
    return np.array(out), tag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--pt-min", type=float, default=4.0); ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    with h5py.File(a.src) as s, h5py.File(a.dst, "w") as d:
        for k, v in s.attrs.items():
            d.attrs[k] = v
        d.attrs["collinear_split"] = f"gluons pT > {a.pt_min} GeV split into {a.n} collinear gluons (collinear_split.py)"
        for g in s:
            if g not in ("partons", "hadronization_input"):
                s.copy(g, d)
        n = int(s.attrs["nunits_written"])
        po, P = s["partons/offsets"][:], s["partons/data"][:]
        io, I = s["hadronization_input/offsets"][:], s["hadronization_input/data"][:]
        newP, newI, opo, oio = [], [], [0], [0]
        for u in range(n):
            p = P[po[u]:po[u + 1]]
            inp = I[io[u]:io[u + 1]]
            tag0 = int(max(np.abs(p[:, 12:14]).max(initial=0), np.abs(inp[:, 1:3]).max(initial=0))) + 1000
            # partons/: shower pid pstat E px py pz t x y z mass col acol (holes are not split: pstat < 0)
            keep = p[:, 2] < 0
            sp, tag1 = split_rows(p[~keep], 1, 12, 13, (4, 5, 6), 3, a.pt_min, a.n, tag0, i_mass=11)
            # the same splitting in hadronization_input/ (pid col acol E px py pz m), same tags in the same order
            si, _ = split_rows(inp, 0, 1, 2, (4, 5, 6), 3, a.pt_min, a.n, tag0, i_mass=7)
            q = np.vstack([sp, p[keep]]) if keep.any() else sp
            newP.append(q); newI.append(si)
            opo.append(opo[-1] + len(q)); oio.append(oio[-1] + len(si))
        for grp, data, off, src in (("partons", newP, opo, "partons"), ("hadronization_input", newI, oio, "hadronization_input")):
            g = d.create_group(grp)
            for k, v in s[src].attrs.items():
                g.attrs[k] = v
            g.create_dataset("data", data=np.vstack(data))
            g.create_dataset("offsets", data=np.array(off, np.int64))
    print(f"{a.src} -> {a.dst}")


if __name__ == "__main__":
    main()
