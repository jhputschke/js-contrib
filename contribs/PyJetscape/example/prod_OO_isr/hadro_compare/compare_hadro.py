#!/usr/bin/env python3
"""
example/prod_OO_isr/hadro_compare/compare_hadro.py

The jet hadrons of run_hadro_compare.sh's five campaigns (O+O 200 GeV, Matter only,
iColoredHadronization), compared in pairs that share their partons event by event:

    after / before         the colour fixes (one tag counter per event, the checks), no recoils
    thermal / reconnect    the closure of the colour lines that end in the medium, with recoils
    fake / reconnect
    reconnect / after      context only: Matter's recoils on vs off (the partons differ)

Events: those present in all five campaigns (same seed, same event of the job).  Weights:
Ncoll x sigma(pTHat window) / N(window), the same per event in every campaign.  Errors:
jackknife over the jobs (delete one job); the paired ratios keep the jobs together, so what
the two campaigns share cancels.

    python compare_hadro.py [--data DIR] [--out DIR] [--jobs N]

Writes <out>/*.png and <out>/hadro_compare.txt (default out: figures/, git-ignored).
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
import jetscape.h5_compression  # noqa: E402,F401  (the HDF5 filter plugin)
import h5py  # noqa: E402

CAMPAIGNS = ("before", "after", "reconnect", "thermal", "fake")
LABEL = {"before": "before (duplicate tags)", "after": "after (unique tags)",
         "reconnect": "recoils, reconnect", "thermal": "recoils, thermal",
         "fake": "recoils, fake"}
# categorical slots in fixed order (validated: dataviz validate_palette.js, light surface)
COLOR = {"before": "#2a78d6", "after": "#eb6834", "reconnect": "#1baf7a",
         "thermal": "#eda100", "fake": "#e87ba4"}
STYLE = {"before": "--", "after": "-", "reconnect": "-", "thermal": "--", "fake": ":"}
# (variant, reference): the hadronization comparisons, then the context
PAIRS = (("after", "before"), ("thermal", "reconnect"), ("fake", "reconnect"),
         ("reconnect", "after"))

CHARGED = (211, 321, 2212, 3222, 3112, 3312, 3334)     # |pid| of charged hadrons
SPECIES = {"pi+-": (211,), "K+-": (321,), "p pbar": (2212,), "K0S": (310,),
           "Lambda": (3122,), "Sigma+-": (3222, 3112), "Xi-": (3312,), "Omega-": (3334,),
           "gamma": (22,)}
ID_PLOT = ("pi+-", "K+-", "p pbar", "K0S", "Lambda")
RATIOS = (("K+-", "pi+-"), ("p pbar", "pi+-"), ("Lambda", "K0S"))

PT_BINS = np.geomspace(0.15, 40.0, 26)
PT_ID_BINS = np.geomspace(0.15, 20.0, 16)
ETA_BINS = np.linspace(-8.0, 8.0, 65)
NCH_BINS = np.arange(-0.5, 80.5, 2.0)
DR_BINS = np.linspace(0.0, 1.0, 21)
Z_BINS = np.geomspace(0.01, 1.5, 21)


# ------------------------------------------------------------------------------------ reading

def _events_of(path, campaign):
    """{event: record} of one job file."""
    out = {}
    with h5py.File(path, "r") as f:
        pid = f["hadrons/pid"][:]
        p = f["hadrons/p"][:].astype(np.float64)
        so, uo = f["hadrons/sample_offsets"][:], f["hadrons/unit_offsets"][:]
        u = {k: f["units/" + k][:] for k in f["units"]}
        ini = f["initiators/data"][:] if "initiators" in f else None
        ioff = f["initiators/offsets"][:] if ini is not None else None
        hin = f["hadronization_input/data"][:] if "hadronization_input" in f else None
        hoff = f["hadronization_input/offsets"][:] if hin is not None else None
        sigma = f.attrs["pthat_bin_sigma_gen"][:].astype(np.float64)
    for k in range(len(uo) - 1):
        a, b = so[uo[k]], so[uo[k + 1]]
        rec = {"pid": pid[a:b], "p": p[a:b], "bin": int(u["pthat_bin"][k]),
               "ncoll": float(u["glauber_ncoll"][k]), "sigma": sigma}
        if ini is not None:
            x = ini[ioff[k]:ioff[k + 1]]
            hard = x[x[:, 2] == 1000]            # PythiaIsrGun's outgoing partons
            if len(hard):
                j = np.argmax(np.hypot(hard[:, 3], hard[:, 4]))
                px, py, pz = hard[j, 3:6]
                pt = np.hypot(px, py)
                rec["lead"] = (pt, np.arcsinh(pz / pt), np.arctan2(py, px))
        if hin is not None:
            x = hin[hoff[k]:hoff[k + 1]]
            rec["E_in"] = float(x[:, 3].sum())
            c, ac = x[:, 1].astype(np.int64), x[:, 2].astype(np.int64)
            cc = collections.Counter(c[c != 0])
            aa = collections.Counter(ac[ac != 0])
            rec["dup"] = (sum(v - 1 for v in cc.values() if v > 1)
                          + sum(v - 1 for v in aa.values() if v > 1))
        if "colour_unpaired_col" in u:
            rec["open"] = int(u["colour_unpaired_col"][k] + u["colour_unpaired_acol"][k])
            rec["audit"] = {key[7:]: u[key][k] for key in u if key.startswith("colour_")}
        out[int(u["event"][k])] = rec
    return out


def load(data, n_jobs=None):
    """{campaign: {seed: {event: record}}} for the seeds complete in every campaign."""
    files = {}
    for c in CAMPAIGNS:
        files[c] = {}
        for path in glob.glob(os.path.join(data, c, "*_hadrons_jet_frag.h5")):
            m = re.search(r"seed(\d+)_hadrons", path)
            if not os.path.exists(path.replace("_hadrons_jet_frag.h5", ".json")):
                continue                       # the job is still running
            try:
                with h5py.File(path, "r") as f:
                    if not bool(f.attrs.get("complete", False)):
                        continue
            except OSError:                    # open (locked) by a running job
                continue
            files[c][int(m.group(1))] = path
    seeds = sorted(set.intersection(*(set(v) for v in files.values())))
    if n_jobs:
        seeds = seeds[:n_jobs]
    if not seeds:
        sys.exit(f"compare_hadro.py: no seed complete in all of {CAMPAIGNS} under {data}")
    ev = {c: {s: _events_of(files[c][s], c) for s in seeds} for c in CAMPAIGNS}
    # before the trimming: events per campaign, and those whose "hadrons" include partons
    # (a failed fragmentation written out: the old code did not check pythia.next())
    leaks = {}
    for c in CAMPAIGNS:
        n_all = n_leak = 0
        for s in seeds:
            for rec in ev[c][s].values():
                n_all += 1
                ap = np.abs(rec["pid"])
                n_leak += bool(np.any((ap <= 6) | (ap == 21)))
        leaks[c] = (n_all, n_leak)
    for s in seeds:                    # only the events every campaign has
        common = set.intersection(*(set(ev[c][s]) for c in CAMPAIGNS))
        for c in CAMPAIGNS:
            ev[c][s] = {e: ev[c][s][e] for e in sorted(common)}
    return seeds, ev, leaks


def weights(seeds, ev):
    """{(seed, event): w}: Ncoll x <sigma(window)> / N(window), from the after campaign."""
    ref = ev["after"]
    n_bin = collections.Counter(r["bin"] for s in seeds for r in ref[s].values())
    sig = collections.defaultdict(list)
    for s in seeds:
        for r in ref[s].values():
            sig[r["bin"]].append(r["sigma"][r["bin"]])
    sig = {b: float(np.mean(v)) for b, v in sig.items()}
    w = {(s, e): r["ncoll"] * sig[r["bin"]] / n_bin[r["bin"]]
         for s in seeds for e, r in ref[s].items()}
    norm = len(w) / sum(w.values())
    return {k: v * norm for k, v in w.items()}


# ------------------------------------------------------------------------------- per event

def kinematics(rec):
    p = rec["p"]
    E, px, py, pz = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    pt = np.hypot(px, py)
    pabs = np.sqrt(pt * pt + pz * pz)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = 0.5 * np.log((pabs + pz) / (pabs - pz))
        y = 0.5 * np.log((E + pz) / (E - pz))
    return E, pt, eta, y, np.arctan2(py, px)


def fill(rec, w, H):
    """Add one event (weight w) to the job's histograms H."""
    apid = np.abs(rec["pid"])
    E, pt, eta, y, phi = kinematics(rec)
    ch = np.isin(apid, CHARGED)
    mid = np.abs(eta) < 1.0
    ymid = np.abs(y) < 1.0
    H["w"] += w
    H["ch_pt"] += w * np.histogram(pt[ch & mid], PT_BINS)[0]
    H["ch_eta"] += w * np.histogram(eta[ch], ETA_BINS)[0]
    H["dEdeta"] += w * np.histogram(eta, ETA_BINS, weights=E)[0]
    H["nch_mid"] += w * np.histogram([np.sum(ch & mid)], NCH_BINS)[0]
    H["n_all"] += w * len(apid)
    H["n_ch"] += w * np.sum(ch)
    H["n_ch_mid"] += w * np.sum(ch & mid)
    H["E_all"] += w * E.sum()
    H["sumpt_ch_mid"] += w * pt[ch & mid].sum()
    for name, ids in SPECIES.items():
        sel = np.isin(apid, ids)
        H["yield/" + name] += w * np.sum(sel)
        H["yield_mid/" + name] += w * np.sum(sel & ymid)
        H["sumpt_mid/" + name] += w * pt[sel & ymid].sum()
        if name in ID_PLOT:
            H["pt/" + name] += w * np.histogram(pt[sel & ymid], PT_ID_BINS)[0]
    if "lead" in rec:
        lpt, leta, lphi = rec["lead"]
        dphi = np.angle(np.exp(1j * (phi - lphi)))
        dr = np.hypot(eta - leta, dphi)
        sel = ch & np.isfinite(dr)
        H["shape"] += w * np.histogram(dr[sel], DR_BINS, weights=pt[sel] / lpt)[0]
        jet = sel & (dr < 0.4)
        H["z"] += w * np.histogram(pt[jet] / lpt, Z_BINS)[0]
        H["n_lead"] += w


def new_hists():
    H = collections.defaultdict(float)
    for k, n in (("ch_pt", PT_BINS), ("ch_eta", ETA_BINS), ("dEdeta", ETA_BINS),
                 ("nch_mid", NCH_BINS), ("shape", DR_BINS), ("z", Z_BINS)):
        H[k] = np.zeros(len(n) - 1)
    for name in ID_PLOT:
        H["pt/" + name] = np.zeros(len(PT_ID_BINS) - 1)
    return H


def accumulate(seeds, ev, w, select=None):
    """{campaign: [H per job]}; select(seed, event) -> bool keeps a subset of the events"""
    out = {}
    for c in CAMPAIGNS:
        jobs = []
        for s in seeds:
            H = new_hists()
            for e, rec in ev[c][s].items():
                if select is None or select(s, e):
                    fill(rec, w[(s, e)], H)
            jobs.append(H)
        out[c] = jobs
    return out


def affected(ev):
    """the events a change can touch: duplicate tags before (after/before), open colour
    lines with recoils (the closures)"""
    return {"before": lambda s, e: ev["before"][s][e]["dup"] > 0,
            "reconnect": lambda s, e: ev["reconnect"][s][e]["open"] > 0}


# --------------------------------------------------------------------------------- statistics

def per_event(jobs, key, norm="w"):
    """the observable per weighted event (norm: 'w' or another counter), and its jackknife
    replicas"""
    num = np.array([np.asarray(H[key], dtype=float) for H in jobs])
    den = np.array([float(H[norm]) for H in jobs])
    with np.errstate(divide="ignore", invalid="ignore"):
        tot = num.sum(0) / den.sum()
        jk = np.array([(num.sum(0) - num[k]) / (den.sum() - den[k])
                       for k in range(len(jobs))])
    return tot, jk


def jk_error(jk):
    n = len(jk)
    with np.errstate(invalid="ignore"):
        return np.sqrt((n - 1) / n * np.sum((jk - jk.mean(0)) ** 2, axis=0))


def ratio(var, ref, key, norm="w", num2=None):
    """var/ref of an observable (or of num/num2 within each campaign), jackknife error"""
    def obs(jobs):
        t, j = per_event(jobs, key, norm)
        if num2 is not None:
            t2, j2 = per_event(jobs, num2, norm)
            with np.errstate(divide="ignore", invalid="ignore"):
                return t / t2, j / j2
        return t, j
    tv, jv = obs(var)
    tr, jr = obs(ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        return tv / tr, jk_error(jv / jr)


def value(jobs, key, norm="w", num2=None):
    t, j = per_event(jobs, key, norm)
    with np.errstate(divide="ignore", invalid="ignore"):
        if num2 is not None:
            t2, j2 = per_event(jobs, num2, norm)
            return t / t2, jk_error(j / j2)
        return t, jk_error(j)


# ------------------------------------------------------------------------------ event level

def event_changes(seeds, ev, w):
    """per pair: the fraction of events whose hadrons changed, split by the reason the
    hadronization could change (duplicate tags before / open colour lines with recoils), and
    the paired energy difference"""
    rows = {}
    for v, r in PAIRS[:3]:
        n = changed = 0
        split = collections.defaultdict(lambda: [0, 0])     # reason -> [events, changed]
        dE = []
        for s in seeds:
            for e in ev[r][s]:
                a, b = ev[v][s][e], ev[r][s][e]
                same = (len(a["pid"]) == len(b["pid"]) and np.array_equal(a["pid"], b["pid"])
                        and np.allclose(a["p"], b["p"], rtol=0, atol=1e-5))
                why = ((ev["before"][s][e]["dup"] > 0) if r == "before"
                       else (ev["reconnect"][s][e]["open"] > 0))
                n += 1
                changed += not same
                split[why][0] += 1
                split[why][1] += not same
                dE.append((a["p"][:, 0].sum() - b["p"][:, 0].sum(), w[(s, e)]))
        dE = np.array(dE)
        rows[(v, r)] = {"n": n, "changed": changed, "split": dict(split),
                        "dE_mean": float(np.average(dE[:, 0], weights=dE[:, 1])),
                        "dE_rms": float(np.sqrt(np.average(dE[:, 0] ** 2, weights=dE[:, 1])))}
    return rows


def audit_totals(seeds, ev, c):
    tot = collections.Counter()
    n = 0
    for s in seeds:
        for rec in ev[c][s].values():
            n += 1
            for k, val in rec.get("audit", {}).items():
                tot[k] += float(val)
            tot["events_open"] += rec.get("open", 0) > 0
    return n, tot


# ---------------------------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linewidth": 0.5, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False,
                         "lines.linewidth": 1.6})
    return plt


def _centres(b, log=False):
    return np.sqrt(b[1:] * b[:-1]) if log else 0.5 * (b[1:] + b[:-1])


def _ratio_panel(ax, acc, key, bins, log, pairs, norm="w", num2=None, ylim=None):
    x = _centres(bins, log)
    for v, r in pairs:
        rt, er = ratio(acc[v], acc[r], key, norm, num2)
        ax.fill_between(x, rt - er, rt + er, color=COLOR[v], alpha=0.25, lw=0, step="mid")
        ax.step(x, rt, where="mid", color=COLOR[v], ls=STYLE[v], label=f"{v} / {r}")
    ax.axhline(1.0, color="0.4", lw=0.8)
    ax.set_ylim(*(ylim or (0.8, 1.2)))
    ax.legend(fontsize=7, ncol=2)


def figure_observable(plt, acc, key, bins, log, xlabel, ylabel, path, norm="w", width=None,
                      logy=True, ylim=None, title=None):
    """the five campaigns, then the hadronization ratios, then the recoil context"""
    fig, axs = plt.subplots(3, 1, figsize=(6.2, 7.2), sharex=True,
                            gridspec_kw={"height_ratios": (2.2, 1, 1)})
    x = _centres(bins, log)
    dx = np.diff(bins) if width is None else width
    for c in CAMPAIGNS:
        t, _ = per_event(acc[c], key, norm)
        axs[0].step(x, t / dx, where="mid", color=COLOR[c], ls=STYLE[c], label=LABEL[c])
    if logy:
        axs[0].set_yscale("log")
    axs[0].set_ylabel(ylabel)
    axs[0].legend(fontsize=7)
    if title:
        axs[0].set_title(title, fontsize=9)
    _ratio_panel(axs[1], acc, key, bins, log, PAIRS[:3], norm, ylim=ylim)
    axs[1].set_ylabel("ratio (paired)")
    _ratio_panel(axs[2], acc, key, bins, log, PAIRS[3:], norm, ylim=(0.6, 1.4))
    axs[2].set_ylabel("recoils / none")
    axs[2].set_xlabel(xlabel)
    if log:
        axs[2].set_xscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_identified(plt, acc, path):
    fig, axs = plt.subplots(2, len(ID_PLOT), figsize=(13, 5.4), sharex=True,
                            gridspec_kw={"height_ratios": (2, 1)})
    x = _centres(PT_ID_BINS, True)
    dx = np.diff(PT_ID_BINS)
    for i, name in enumerate(ID_PLOT):
        for c in CAMPAIGNS:
            t, _ = per_event(acc[c], "pt/" + name)
            axs[0, i].step(x, t / dx, where="mid", color=COLOR[c], ls=STYLE[c], label=LABEL[c])
        axs[0, i].set_yscale("log")
        axs[0, i].set_title(f"{name}, |y| < 1", fontsize=9)
        _ratio_panel(axs[1, i], acc, "pt/" + name, PT_ID_BINS, True, PAIRS[:3],
                     ylim=(0.6, 1.4))
        axs[1, i].set_xscale("log")
        axs[1, i].set_xlabel("p_T [GeV]")
        if i:
            axs[1, i].get_legend().remove()
    axs[0, 0].set_ylabel("dN/dp_T per event [1/GeV]")
    axs[1, 0].set_ylabel("ratio (paired)")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_particle_ratios(plt, acc, path):
    fig, axs = plt.subplots(2, len(RATIOS), figsize=(11, 5.4), sharex=True,
                            gridspec_kw={"height_ratios": (2, 1)})
    x = _centres(PT_ID_BINS, True)
    for i, (a, b) in enumerate(RATIOS):
        for c in CAMPAIGNS:
            t, _ = value(acc[c], "pt/" + a, num2="pt/" + b)
            axs[0, i].step(x, t, where="mid", color=COLOR[c], ls=STYLE[c], label=LABEL[c])
        axs[0, i].set_title(f"{a} / {b} vs p_T, |y| < 1", fontsize=9)
        _ratio_panel(axs[1, i], acc, "pt/" + a, PT_ID_BINS, True, PAIRS[:3], num2="pt/" + b,
                     ylim=(0.6, 1.4))
        axs[1, i].set_xscale("log")
        axs[1, i].set_xlabel("p_T [GeV]")
        if i:
            axs[1, i].get_legend().remove()
    axs[1, 0].set_ylabel("double ratio (paired)")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_affected(plt, sub, path):
    """the paired ratios for the events a change can touch only"""
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    panels = (("ch_pt", PT_BINS, True, "p_T [GeV] (charged, |eta| < 1)"),
              ("ch_eta", ETA_BINS, False, "eta (charged)"),
              ("dEdeta", ETA_BINS, False, "eta (energy flow)"))
    for ax, (key, bins, log, xl) in zip(axs, panels):
        for ref, acc in sub.items():
            _ratio_panel(ax, acc, key, bins, log, [pr for pr in PAIRS[:3] if pr[1] == ref],
                         ylim=(0.6, 1.4))
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(xl)
    axs[0].set_ylabel("ratio, affected events (paired)")
    fig.suptitle("only the events with duplicate tags before (after/before) or with open "
                 "colour lines (closures)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=HERE, help="the directory with the five campaigns")
    p.add_argument("--out", default=os.path.join(HERE, "figures"))
    p.add_argument("--jobs", type=int, default=None, help="only the first N common seeds")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    seeds, ev, leaks = load(a.data, a.jobs)
    w = weights(seeds, ev)
    acc = accumulate(seeds, ev, w)
    n_ev = sum(len(ev["after"][s]) for s in seeds)

    lines = [f"jobs {len(seeds)} (seeds {seeds[0]}-{seeds[-1]}), events {n_ev} common to "
             f"all five campaigns; weights Ncoll x sigma(window)/N(window); errors: "
             f"jackknife over the jobs", "",
             "events written per campaign, and those whose final particles include partons "
             "(a failed fragmentation written out):"]
    for c in CAMPAIGNS:
        lines.append(f"  {c:10s} {leaks[c][0]} events, {leaks[c][1]} with partons")
    lines.append("")

    def table(acc, pairs, title):
        head = f"{title:34s}" + "".join(f"{c:>13s}" for c in CAMPAIGNS) + \
            "".join(f"{v + '/' + r:>24s}" for v, r in pairs)
        out = [head, "-" * len(head)]
        for label, key, norm, num2 in keys:
            row = f"{label:34s}"
            for c in CAMPAIGNS:
                t, _ = value(acc[c], key, norm, num2)
                row += f"{float(t):13.4g}"
            for v, r in pairs:
                rt, er = ratio(acc[v], acc[r], key, norm, num2)
                row += f"{float(rt):15.4f} ± {float(er):6.4f}"
            out.append(row)
        return out

    # per-event yields and <pT>
    keys = [("all particles", "n_all", "w", None), ("charged hadrons", "n_ch", "w", None),
            ("charged, |eta| < 1", "n_ch_mid", "w", None),
            ("<pT> charged, |eta| < 1", "sumpt_ch_mid", "n_ch_mid", None),
            ("energy of the jet hadrons [GeV]", "E_all", "w", None)]
    for name in SPECIES:
        keys.append((f"{name}, all", "yield/" + name, "w", None))
        keys.append((f"{name}, |y| < 1", "yield_mid/" + name, "w", None))
        keys.append((f"<pT> {name}, |y| < 1", "sumpt_mid/" + name, "yield_mid/" + name, None))
    for x_, y_ in RATIOS:
        keys.append((f"{x_} / {y_}, |y| < 1", "yield_mid/" + x_, "w", "yield_mid/" + y_))
    lines += table(acc, PAIRS, "per event, all events")
    sub = {}
    for ref, sel in affected(ev).items():
        n_sel = sum(sel(s, e) for s in seeds for e in ev[ref][s])
        sub[ref] = accumulate(seeds, ev, w, sel)
        pairs = [pr for pr in PAIRS[:3] if pr[1] == ref]
        why = "duplicate tags before" if ref == "before" else "open colour lines"
        lines += ["", f"only the events with {why} ({n_sel} of {n_ev}):"]
        lines += table(sub[ref], pairs, "per event, affected events")

    lines += ["", "events whose jet hadrons changed (pid and four-momenta), paired:"]
    for (v, r), d in event_changes(seeds, ev, w).items():
        why = "duplicate tags before" if r == "before" else "open colour lines"
        sp = "; ".join(f"{why if k else 'none'}: {n_} events, {c_} changed"
                       for k, (n_, c_) in sorted(d["split"].items()))
        lines.append(f"  {v}/{r}: {d['changed']} of {d['n']} ({100 * d['changed'] / d['n']:.1f} %)"
                     f" [{sp}]; energy of the jet hadrons {v} - {r}: mean "
                     f"{d['dE_mean']:+.3f} GeV, rms {d['dE_rms']:.3f} GeV per event")
    lines += ["", "colour audit (sums over the events):"]
    for c in ("after", "reconnect", "thermal", "fake"):
        n, tot = audit_totals(seeds, ev, c)
        keep = ("holes", "hole_energy", "unpaired_col", "unpaired_acol", "reconnected",
                "reconnected_cross", "partners", "partner_energy", "fakes", "fake_energy",
                "retries", "duplicate_tags", "events_open")
        lines.append(f"  {c:10s} {n} events: " +
                     ", ".join(f"{k} {tot[k]:.4g}" for k in keep))
    dups = [ev["before"][s][e]["dup"] for s in seeds for e in ev["before"][s]]
    lines.append(f"  before     {len(dups)} events: with duplicate tags "
                 f"{sum(d > 0 for d in dups)}, duplicates per such event "
                 f"{np.mean([d for d in dups if d > 0]) if any(dups) else 0:.2f}")
    with open(os.path.join(a.out, "hadro_compare.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    plt = _plt()
    figure_observable(plt, acc, "ch_pt", PT_BINS, True, "p_T [GeV]",
                      "dN_ch/dp_T per event [1/GeV]", os.path.join(a.out, "ch_pt.png"),
                      title="charged hadrons, |eta| < 1")
    figure_observable(plt, acc, "ch_eta", ETA_BINS, False, "eta", "dN_ch/deta per event",
                      os.path.join(a.out, "ch_eta.png"), logy=False,
                      title="charged hadrons")
    figure_observable(plt, acc, "dEdeta", ETA_BINS, False, "eta", "dE/deta per event [GeV]",
                      os.path.join(a.out, "dEdeta.png"), title="energy flow, all particles")
    figure_observable(plt, acc, "nch_mid", NCH_BINS, False, "N_ch (|eta| < 1)",
                      "P(N_ch) per event", os.path.join(a.out, "nch.png"),
                      title="charged multiplicity, |eta| < 1", ylim=(0.5, 1.5))
    figure_observable(plt, acc, "shape", DR_BINS, False, "Delta R to the leading parton",
                      "sum p_T / p_T,lead per unit Delta R", os.path.join(a.out, "shape.png"),
                      norm="n_lead", title="charged-hadron p_T around the leading hard parton")
    figure_observable(plt, acc, "z", Z_BINS, True, "z = p_T / p_T,lead (Delta R < 0.4)",
                      "dN/dz per event", os.path.join(a.out, "z.png"), norm="n_lead",
                      title="charged hadrons near the leading hard parton", ylim=(0.6, 1.4))
    figure_affected(plt, sub, os.path.join(a.out, "affected.png"))
    figure_identified(plt, acc, os.path.join(a.out, "identified_pt.png"))
    figure_particle_ratios(plt, acc, os.path.join(a.out, "particle_ratios.png"))
    print(f"compare_hadro.py: figures and hadro_compare.txt in {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
