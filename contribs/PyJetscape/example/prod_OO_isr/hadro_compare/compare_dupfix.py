#!/usr/bin/env python3
"""
example/prod_OO_isr/hadro_compare/compare_dupfix.py

The duplicate-colour-tag fix on its own: run_hadro_compare.sh's before/ and after/ campaigns
(O+O 200 GeV, Matter only without recoils, iColoredHadronization).  The two share their
partons event by event; before/ hands Pythia the colour tags of the old code (every shower's
Matter inherited one counter, so showers repeated each other's tags), after/ the tags of one
counter per event (X-SCAPE lbt_colored_hadro).  The closures, Matter's recoil colours and LBT's
tags do not enter: without recoils and LBT there is no colour line to close.

What it checks and writes (<out>/dupfix.txt, <out>/*.png; default out figures_dupfix/):
- isolation: the Pythia input of every event is the same except col/acol; where before/ had
  no duplicate the tags are a one-to-one relabelling and the jet hadrons must be identical;
- the paired after/before ratios (all events, and the events with duplicates), as
  compare_hadro.py computes them;
- dose and response: the ratios in bins of duplicated tags per event, and per pTHat window.

Events: those present in both campaigns (the old code wrote 2 failed fragmentations out as
"hadrons"; the new code skips them).  Weights: Ncoll x sigma(window) / N(window).  Errors:
jackknife over the jobs, paired.

    python compare_dupfix.py [--data DIR] [--out DIR] [--jobs N]
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_hadro as ch

# compare_hadro's readers, accumulators and figures read these globals: two campaigns, one pair
ch.CAMPAIGNS = ("before", "after")
ch.PAIRS = (("after", "before"),)
h5py = ch.h5py

DUP = "#1baf7a"                     # the third categorical slot (compare_hadro.COLOR order)
DUP_BINS = ((1, 2), (3, 5), (6, 10), (11, 10 ** 9))


def _label(lo, hi):
    return f"{lo}-{hi}" if hi < 10 ** 9 else f">= {lo}"


# ------------------------------------------------------------------------------ isolation

def _inputs(path):
    """{event: (n, 8) Pythia input} of one job file (hadronization_input/)"""
    with h5py.File(path, "r") as f:
        d = f["hadronization_input/data"][:]
        off = f["hadronization_input/offsets"][:]
        ev = f["units/event"][:]
    return {int(e): d[off[k]:off[k + 1]] for k, e in enumerate(ev)}


def _relabelling(xb, xa):
    """True if after's tags are a one-to-one relabelling of before's (same colour topology)"""
    fwd, bwd = {}, {}
    for b, a in zip(np.r_[xb[:, 1], xb[:, 2]].astype(np.int64),
                    np.r_[xa[:, 1], xa[:, 2]].astype(np.int64)):
        if (b == 0) != (a == 0):
            return False
        if b == 0:
            continue
        if fwd.setdefault(b, a) != a or bwd.setdefault(a, b) != b:
            return False
    return True


def isolation(data, seeds, ev):
    """the Pythia inputs of before/ and after/, event by event"""
    out = collections.Counter()
    for s in seeds:
        paths = {c: glob.glob(os.path.join(data, c, f"*seed{s:04d}_hadrons_jet_frag.h5"))[0]
                 for c in ch.CAMPAIGNS}
        inb, ina = _inputs(paths["before"]), _inputs(paths["after"])
        for e in ev["after"][s]:
            xb, xa = inb[e], ina[e]
            dup = ev["before"][s][e]["dup"] > 0
            out["events"] += 1
            same = len(xb) == len(xa) and np.array_equal(xb[:, [0, 3, 4, 5, 6, 7]],
                                                         xa[:, [0, 3, 4, 5, 6, 7]])
            out["same_kin"] += same
            if same:
                out["relabel_dup" if dup else "relabel_clean"] += _relabelling(xb, xa)
            out["dup" if dup else "clean"] += 1
            out["partons"] += len(xa)
    return out


# ------------------------------------------------------------------------------ tables

KEYS = [("all particles", "n_all", "w", None), ("charged hadrons", "n_ch", "w", None),
        ("charged, |eta| < 1", "n_ch_mid", "w", None),
        ("<pT> charged, |eta| < 1 [GeV]", "sumpt_ch_mid", "n_ch_mid", None),
        ("energy of the jet hadrons [GeV]", "E_all", "w", None)]
for _n in ch.SPECIES:
    KEYS.append((f"{_n}, |y| < 1", "yield_mid/" + _n, "w", None))
    KEYS.append((f"<pT> {_n}, |y| < 1 [GeV]", "sumpt_mid/" + _n, "yield_mid/" + _n, None))
for _x, _y in ch.RATIOS:
    KEYS.append((f"{_x} / {_y}, |y| < 1", "yield_mid/" + _x, "w", "yield_mid/" + _y))
KEYS += [("ch. pT/pT,lead, DR < 0.2", "core_pt", "n_lead", None),
         ("ch. pT/pT,lead, 0.5 <= DR < 1", "wide_pt", "n_lead", None)]
# the observables of the dose-response and window tables
SHORT = [("charged, |eta| < 1", "n_ch_mid", "w"), ("<pT> charged, |eta| < 1", "sumpt_ch_mid",
         "n_ch_mid"), ("ch. pT/pT,lead, DR < 0.2", "core_pt", "n_lead")]


def table(blocks):
    """blocks: [(title, acc)]; rows: before, after, after/before of each block"""
    head = f"{'per event':34s}" + "".join(
        f"{'before':>11s}{'after':>11s}{'after/before (' + t + ')':>32s}" for t, _ in blocks)
    out = [head, "-" * len(head)]
    for label, key, norm, num2 in KEYS:
        row = f"{label:34s}"
        for _, acc in blocks:
            tb, _ = ch.value(acc["before"], key, norm, num2)
            ta, _ = ch.value(acc["after"], key, norm, num2)
            rt, er = ch.ratio(acc["after"], acc["before"], key, norm, num2)
            row += f"{float(tb):11.4g}{float(ta):11.4g}{float(rt):23.4f} ± {float(er):6.4f}"
        out.append(row)
    return out


def changed(seeds, ev):
    """{(seed, event): True if the jet hadrons differ (pid or four-momenta)}"""
    out = {}
    for s in seeds:
        for e in ev["after"][s]:
            a, b = ev["after"][s][e], ev["before"][s][e]
            out[(s, e)] = not (len(a["pid"]) == len(b["pid"])
                               and np.array_equal(a["pid"], b["pid"])
                               and np.allclose(a["p"], b["p"], rtol=0, atol=1e-5))
    return out


# ------------------------------------------------------------------------------ figures

def figure_pair(plt, acc, sub, key, bins, log, xlabel, ylabel, path, norm="w", logy=True,
                ylim=(0.8, 1.2), title=None):
    """before and after (all events), then after/before for all events and for the events
    with duplicate tags"""
    fig, axs = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=True,
                            gridspec_kw={"height_ratios": (2.2, 1.2)})
    x = ch._centres(bins, log)
    dx = np.diff(bins)
    for c in ch.CAMPAIGNS:
        t, _ = ch.per_event(acc[c], key, norm)
        axs[0].step(x, t / dx, where="mid", color=ch.COLOR[c], ls=ch.STYLE[c],
                    label=ch.LABEL[c])
    if logy:
        axs[0].set_yscale("log")
    axs[0].set_ylabel(ylabel)
    axs[0].legend(fontsize=7)
    if title:
        axs[0].set_title(title, fontsize=9)
    for a_, color, ls, label in ((acc, ch.COLOR["after"], "-", "after / before, all events"),
                                 (sub, DUP, "--", "after / before, events with duplicates")):
        rt, er = ch.ratio(a_["after"], a_["before"], key, norm)
        axs[1].fill_between(x, rt - er, rt + er, color=color, alpha=0.22, lw=0, step="mid")
        axs[1].step(x, rt, where="mid", color=color, ls=ls, label=label)
    axs[1].axhline(1.0, color="0.4", lw=0.8)
    axs[1].set_ylim(*ylim)
    axs[1].set_ylabel("ratio (paired)")
    axs[1].legend(fontsize=7)
    axs[1].set_xlabel(xlabel)
    if log:
        axs[1].set_xscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_dose(plt, dose, path):
    """after/before vs the number of duplicated tags per event, and the share of events whose
    hadrons changed"""
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": (1.6, 1)})
    labels = [d["label"] for d in dose]
    x = np.arange(len(dose))
    marks = (("charged, |eta| < 1", ch.COLOR["after"], "o", -0.12),
             ("<pT> charged, |eta| < 1", ch.COLOR["before"], "s", 0.0),
             ("ch. pT/pT,lead, DR < 0.2", DUP, "^", 0.12))
    for name, color, m, off in marks:
        r = np.array([d["ratio"][name][0] for d in dose])
        e = np.array([d["ratio"][name][1] for d in dose])
        axs[0].errorbar(x + off, r, yerr=e, color=color, marker=m, ms=6, lw=1.4, capsize=2,
                        label=name)
    axs[0].axhline(1.0, color="0.4", lw=0.8)
    axs[0].set_xticks(x, [f"{lab}\n({d['n']} ev.)" for lab, d in zip(labels, dose)])
    axs[0].set_xlabel("duplicated tags per event, before the fix")
    axs[0].set_ylabel("after / before (paired)")
    axs[0].legend(fontsize=7)
    frac = [100.0 * d["changed"] / d["n"] for d in dose]
    axs[1].bar(x, frac, width=0.6, color=ch.COLOR["after"])
    for xi, f_ in zip(x, frac):
        axs[1].text(xi, f_ + 1.5, f"{f_:.0f} %", ha="center", fontsize=7, color="0.2")
    axs[1].set_xticks(x, labels)
    axs[1].set_ylim(0, 110)
    axs[1].set_xlabel("duplicated tags per event, before the fix")
    axs[1].set_ylabel("events whose jet hadrons changed [%]")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------ main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=HERE, help="the directory with before/ and after/")
    p.add_argument("--out", default=os.path.join(HERE, "figures_dupfix"))
    p.add_argument("--jobs", type=int, default=None, help="only the first N common seeds")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    seeds, ev, leaks = ch.load(a.data, a.jobs)
    w = ch.weights(seeds, ev)
    acc = ch.accumulate(seeds, ev, w)
    dup = {(s, e): ev["before"][s][e]["dup"] for s in seeds for e in ev["before"][s]}
    sub = ch.accumulate(seeds, ev, w, lambda s, e: dup[(s, e)] > 0)
    chg = changed(seeds, ev)
    iso = isolation(a.data, seeds, ev)
    n_ev = len(dup)
    n_dup = sum(d > 0 for d in dup.values())

    lines = [(f"jobs {len(seeds)} (seeds {seeds[0]}-{seeds[-1]}), events {n_ev} common to "
              f"before/ and after/; weights Ncoll x sigma(window)/N(window); errors: jackknife "
              f"over the jobs (paired)"), "",
             ("events written, and those whose final particles include partons (a failed "
              "fragmentation written out):")]
    for c in ch.CAMPAIGNS:
        lines.append(f"  {c:7s} {leaks[c][0]} events, {leaks[c][1]} with partons")
    clean_changed = sum(chg[k] for k, d in dup.items() if d == 0)
    dup_changed = sum(chg[k] for k, d in dup.items() if d > 0)
    dE = max(abs(ev["after"][s][e]["p"][:, 0].sum() - ev["before"][s][e]["p"][:, 0].sum())
             for s, e in dup)
    lines += ["", "isolation (the Pythia input of each event, before vs after):",
              (f"  same partons (pid, four-momenta, mass, order): {iso['same_kin']} of "
               f"{iso['events']} events ({iso['partons']} partons)"),
              (f"  events without duplicates before: {iso['clean']}; tags a one-to-one "
               f"relabelling in {iso['relabel_clean']}; jet hadrons changed in {clean_changed}"),
              (f"  events with duplicates before: {iso['dup']}; a relabelling in "
               f"{iso['relabel_dup']}; jet hadrons changed in {dup_changed}"),
              f"  largest per-event difference of the jet-hadron energy: {dE:.2e} GeV",
              (f"  duplicated tags per event with duplicates: "
               f"{np.mean([d for d in dup.values() if d > 0]):.2f} (max {max(dup.values())})"),
              ""]
    lines += [f"all: the {n_ev} events; dup.: the {n_dup} with duplicate tags before"]
    lines += table([("all", acc), ("dup.", sub)])

    # dose and response
    dose = []
    clean = [k for k, d in dup.items() if d == 0]
    for lo, hi in DUP_BINS:
        keys = {k for k, d in dup.items() if lo <= d <= hi}
        acc_b = ch.accumulate(seeds, ev, w, lambda s, e, keys=keys: (s, e) in keys)
        d = {"label": _label(lo, hi), "n": len(keys), "changed": sum(chg[k] for k in keys),
             "ratio": {}}
        for name, key, norm in SHORT:
            d["ratio"][name] = ch.ratio(acc_b["after"], acc_b["before"], key, norm)
        dose.append(d)
    lines += ["", "dose and response: after/before by duplicated tags per event (before)",
              f"  {'duplicates':>11s}{'events':>8s}{'changed':>9s}" +
              "".join(f"{n:>32s}" for n, _, _ in SHORT)]
    lines.append(f"  {'0':>11s}{len(clean):8d}{clean_changed:9d}" +
                 "".join(f"{'identical':>32s}" for _ in SHORT))
    for d in dose:
        lines.append(f"  {d['label']:>11s}{d['n']:8d}{d['changed']:9d}" + "".join(
            f"{float(d['ratio'][n][0]):23.4f} ± {float(d['ratio'][n][1]):6.4f}"
            for n, _, _ in SHORT))

    # per pTHat window
    lines += ["", "per pTHat window: after/before, all events",
              f"  {'window':>8s}{'events':>8s}{'with dup.':>10s}{'dup./ev.':>9s}" +
              "".join(f"{n:>32s}" for n, _, _ in SHORT)]
    for b in sorted({ev["after"][s][e]["bin"] for s, e in dup}):
        keys = {(s, e) for s, e in dup if ev["after"][s][e]["bin"] == b}
        acc_b = ch.accumulate(seeds, ev, w, lambda s, e, keys=keys: (s, e) in keys)
        nd = [dup[k] for k in keys]
        row = (f"  {b:>8d}{len(keys):8d}{sum(x > 0 for x in nd):10d}"
               f"{np.mean(nd):9.2f}")
        for name, key, norm in SHORT:
            rt, er = ch.ratio(acc_b["after"], acc_b["before"], key, norm)
            row += f"{float(rt):23.4f} ± {float(er):6.4f}"
        lines.append(row)

    with open(os.path.join(a.out, "dupfix.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    plt = ch._plt()
    figure_pair(plt, acc, sub, "ch_pt", ch.PT_BINS, True, "p_T [GeV]",
                "dN_ch/dp_T per event [1/GeV]", os.path.join(a.out, "ch_pt.png"),
                ylim=(0.6, 1.4), title="charged hadrons, |eta| < 1")
    figure_pair(plt, acc, sub, "ch_eta", ch.ETA_BINS, False, "eta", "dN_ch/deta per event",
                os.path.join(a.out, "ch_eta.png"), logy=False, title="charged hadrons")
    figure_pair(plt, acc, sub, "nch_mid", ch.NCH_BINS, False, "N_ch (|eta| < 1)",
                "P(N_ch) per event", os.path.join(a.out, "nch.png"), ylim=(0.5, 1.5),
                title="charged multiplicity, |eta| < 1")
    figure_pair(plt, acc, sub, "shape", ch.DR_BINS, False, "Delta R to the leading parton",
                "sum p_T / p_T,lead per unit Delta R", os.path.join(a.out, "shape.png"),
                norm="n_lead", ylim=(0.6, 1.4),
                title="charged-hadron p_T around the leading hard parton")
    figure_pair(plt, acc, sub, "z", ch.Z_BINS, True, "z = p_T / p_T,lead (Delta R < 0.4)",
                "dN/dz per event", os.path.join(a.out, "z.png"), norm="n_lead",
                ylim=(0.6, 1.4), title="charged hadrons near the leading hard parton")
    ch.figure_identified(plt, acc, os.path.join(a.out, "identified_pt.png"))
    ch.figure_particle_ratios(plt, acc, os.path.join(a.out, "particle_ratios.png"))
    figure_dose(plt, dose, os.path.join(a.out, "dose.png"))
    print(f"compare_dupfix.py: figures and dupfix.txt in {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
