#!/usr/bin/env python3
"""conservation.py HADRONS.dat [--sqrts E] [--A A --Z Z]

Totals of a final-state hadron file (JETSCAPE_FINAL_STATE v2) per event: N,
energy and momentum, net baryon number B, charge Q and strangeness S (from
the PDG codes), against the colliding nuclei (default Au+Au: A = 197,
Z = 79; E = A sqrt(s_NN) in the centre-of-mass frame).  For the dynamic
fluidization runs (SMASH next to a Cartesian MUSIC): B, Q, S must come out
exactly, E up to what the hydro leaves behind (the fluid below the switching
density, never particlized).
"""
import argparse
from collections import defaultdict


def quantum_numbers(pid):
    """(B, Q, S) from a PDG code (hadrons; leptons and photons give Q only)."""
    a = abs(pid)
    sign = 1 if pid > 0 else -1
    if a in (11, 13, 15):
        return 0, -sign, 0
    if a in (12, 14, 16, 22):
        return 0, 0, 0
    if a in (130, 310):        # K0_L, K0_S: mixtures, S averages to 0
        return 0, 0, 0
    if a >= 1000000000:        # nuclei: 10LZZZAAAI
        Z = (a // 10000) % 1000
        A = (a // 10) % 1000
        L = (a // 10000000) % 10
        return sign * A, sign * Z, -sign * L
    ch3 = {1: -1, 2: 2, 3: -1, 4: 2, 5: -1, 6: 2}   # quark charges in thirds
    nq1, nq2, nq3 = (a // 1000) % 10, (a // 100) % 10, (a // 10) % 10
    if nq1 == 0:                                    # meson
        if nq2 == nq3:
            return 0, 0, 0
        # odd (down-type) nq2: content nq3 anti-nq2; even: nq2 anti-nq3
        q, qbar = (nq3, nq2) if nq2 % 2 == 1 else (nq2, nq3)
        Q3 = ch3[q] - ch3[qbar]
        S = (-1 if q == 3 else 0) + (1 if qbar == 3 else 0)
        return 0, sign * (Q3 // 3), sign * S
    Q3 = ch3[nq1] + ch3[nq2] + ch3[nq3]             # baryon
    S = -sum(1 for q in (nq1, nq2, nq3) if q == 3)
    return sign, sign * (Q3 // 3), sign * S


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hadrons")
    ap.add_argument("--sqrts", type=float, default=7.7)
    ap.add_argument("--A", type=int, default=197)
    ap.add_argument("--Z", type=int, default=79)
    args = ap.parse_args()
    events = []
    cur = None
    with open(args.hadrons) as f:
        for line in f:
            if line.startswith("#"):
                if "Event" in line:
                    cur = defaultdict(float)
                    events.append(cur)
                continue
            if cur is None:
                continue
            w = line.split()
            pid = int(w[1])
            E, px, py, pz = map(float, w[3:7])
            B, Q, S = quantum_numbers(pid)
            cur["N"] += 1
            cur["E"] += E
            cur["px"] += px
            cur["py"] += py
            cur["pz"] += pz
            cur["B"] += B
            cur["Q"] += Q
            cur["S"] += S
    E0 = args.A * args.sqrts
    print("nuclei: B = %d, Q = %d, S = 0, E = %.1f GeV" % (2 * args.A, 2 * args.Z, E0))
    for i, ev in enumerate(events):
        print("event %d: N = %d, E = %.1f GeV (%.1f %%), p = (%.2f, %.2f, %.2f), "
              "B = %d, Q = %d, S = %d"
              % (i + 1, ev["N"], ev["E"], 100. * ev["E"] / E0, ev["px"], ev["py"],
                 ev["pz"], ev["B"], ev["Q"], ev["S"]))


if __name__ == "__main__":
    main()
