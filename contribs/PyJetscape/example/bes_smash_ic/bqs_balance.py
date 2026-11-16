#!/usr/bin/env python3
"""B, Q, S balance of a SMASHInitialCondition run, from its log and hadron file.

Usage: bqs_balance.py run.log [final_state_hadrons.dat]

From the log (timestamped or not):
  - the hadrons handed to the hydro as sources (MpiMusic: "Total net baryon
    number of sources = ...");
  - MUSIC's grid totals N_B, N_Q, N_S per step (first and last after the
    sources are in, and the largest deviation in between);
  - iSS's surface totals when the run printed them (vlevel >= 1).
  - with <charge_check> 1: B, Q, S through the surfaces handed to iSS and in
    its samples (mean over the samples), summed over the event.
From the hadron file: net B, Q, S of the final-state hadrons.

The hydro and the surface are deterministic for a given seed; the hadrons of
one sample fluctuate by roughly the square root of the number of carriers
(10-20 units for B and Q at sqrt(s_NN) = 19.6 GeV), so one sample tests
conservation only coarsely: use --samples (run_bes_smash_ic.sh) for that.
"""
import re
import sys


def quark_content_charges(pid):
    """(B, Q, S) of a hadron from its PDG code; None for non-hadrons."""
    a = abs(pid)
    sign = 1 if pid > 0 else -1
    if a in (11, 13, 15):
        return (0, -sign, 0)
    if a in (12, 14, 16, 22, 130, 310):  # K0L, K0S: strangeness mixed
        return (0, 0, 0)
    nq3 = (a // 10) % 10
    nq2 = (a // 100) % 10
    nq1 = (a // 1000) % 10
    qcharge = {1: -1, 2: 2, 3: -1, 4: 2, 5: -1, 6: 2}  # in units of e/3
    if nq1 == 0:  # meson q qbar, nq2 >= nq3
        if nq2 == 0:
            return None
        # quark nq2, antiquark nq3; the sign of the code tells which is which
        q = qcharge[nq2] - qcharge[nq3]
        s = (-1 if nq2 == 3 else 0) - (-1 if nq3 == 3 else 0)
        # for neutral codes (nq2 == nq3) everything is 0
        if nq2 == nq3:
            return (0, 0, 0)
        # PDG: the sign of a meson code follows the heavier quark's charge:
        # positive code means the up-type (or heavier) quark is a quark
        if nq2 % 2 == 0:  # up-type heavier: q(nq2) qbar(nq3)
            Q = q // 3
            S = s
        else:  # down-type heavier: qbar(nq2) q(nq3) for positive codes
            Q = -q // 3
            S = -s
        return (0, sign * Q, sign * S)
    q = qcharge[nq1] + qcharge[nq2] + qcharge[nq3]
    s = -sum(1 for n in (nq1, nq2, nq3) if n == 3)
    return (sign, sign * (q // 3), sign * s)


def parse_log(path):
    out = {"sources": None, "grid": [], "surface": []}
    num = r"([-+0-9.eE]+)"
    for line in open(path, errors="replace"):
        m = re.search(r"Total net baryon number of sources = " + num +
                      r", total net electric charge = " + num +
                      r", total net strangeness = " + num, line)
        if m:
            out["sources"] = tuple(float(x) for x in m.groups())
        m = re.search(r"net baryon number N_B = " + num, line)
        if m:
            out["grid"].append([float(m.group(1)), None, None])
        m = re.search(r"net electric charges N_Q = " + num +
                      r", net strangeness N_S = " + num, line)
        if m and out["grid"] and out["grid"][-1][1] is None:
            out["grid"][-1][1:] = [float(m.group(1)), float(m.group(2))]
        m = re.search(r"event so far: surface " + num + " " + num + " " + num
                      + ", hadrons " + num + " " + num + " " + num, line)
        if m:
            v = [float(x) for x in m.groups()]
            out["surface"] = [tuple(v[:3]), tuple(v[3:])]
    return out


def hadron_totals(path):
    tot = [0, 0, 0]
    n = 0
    unknown = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        try:
            pid = int(f[1])
        except (IndexError, ValueError):
            continue
        c = quark_content_charges(pid)
        if c is None:
            unknown[pid] = unknown.get(pid, 0) + 1
            continue
        n += 1
        for i in range(3):
            tot[i] += c[i]
    return n, tot, unknown


def main():
    log = parse_log(sys.argv[1])
    if log["sources"]:
        print("sources into the hydro:  B %8.2f  Q %8.2f  S %8.2f" %
              log["sources"])
    grid = [g for g in log["grid"] if g[1] is not None]
    filled = [g for g in grid if abs(g[0]) > 0 or abs(g[1]) > 0]
    if filled:
        first, last = filled[0], filled[-1]
        print("grid, first with sources: B %8.3f  Q %8.3f  S %10.3g" %
              tuple(first))
        print("grid, last step:          B %8.3f  Q %8.3f  S %10.3g" %
              tuple(last))
        if log["sources"]:
            ref = log["sources"]
            dev = [max(abs(g[i] - ref[i]) for g in filled[2:] or filled)
                   for i in range(3)]
            print("grid, max |total - sources| once all are in: "
                  "B %.3g  Q %.3g  S %.3g  (over %d steps)" %
                  (dev[0], dev[1], dev[2], len(filled[2:])))
    if log["surface"]:
        print("through the surfaces:     B %8.3f  Q %8.3f  S %10.3g"
              % log["surface"][0])
        print("iSS samples (mean):       B %8.3f  Q %8.3f  S %10.3g"
              % log["surface"][1])
    if len(sys.argv) > 2:
        n, tot, unknown = hadron_totals(sys.argv[2])
        print("final-state hadrons: %d   B %d  Q %d  S %d" %
              (n, tot[0], tot[1], tot[2]))
        if unknown:
            print("  codes not counted:", unknown)


if __name__ == "__main__":
    main()
