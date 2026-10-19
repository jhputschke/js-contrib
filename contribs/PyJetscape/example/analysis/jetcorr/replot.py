#!/usr/bin/env python3
"""Redraw a figure from its figure data (figdata.py's <stem>.npz), without the analysis.

    python replot.py figures/zjet_jets_pp_test.npz         # -> zjet_jets_pp_test_replot.png
    python replot.py figures/qa_hadrons_test.npz --out my.png --style my.mplstyle --dpi 200
    python replot.py figures/zjet_jets_pp_test.npz --list             # the calls and their arrays

The calls are made again in their order on axes at the recorded positions, hist and hist2d as
stairs and pcolormesh of their stored contents.  To restyle: a matplotlib style file
(--style), or in Python

    from jetcorr import replot
    fig, axes = replot.draw("figures/zjet_jets_pp_test.npz", edit=my_edit)

with ``my_edit(call, args, kwargs) -> (args, kwargs)`` (or None to drop the call) changing any
call's arguments (colours, markers, labels, ...) before it is made; ``figdata.load`` gives the
manifest and the arrays themselves."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # example/analysis
from jetcorr import figdata  # noqa: E402

# keyword arguments of hist that stairs takes over
_STAIRS_KW = ("color", "label", "linestyle", "ls", "linewidth", "lw", "alpha", "zorder", "hatch",
              "edgecolor", "facecolor")


def draw(path, edit=None, figsize=None):
    """-> (fig, axes) redrawn from a figure-data file."""
    import matplotlib.pyplot as plt
    manifest, arrays = figdata.load(path)
    fig = plt.figure(figsize=figsize or manifest["figsize"])
    axes = []
    for a in manifest["axes"]:
        ax = fig.add_axes(a["position"])
        if not a["visible"]:
            ax.set_axis_off()
        axes.append(ax)
    made = {}
    for c, call in enumerate(manifest["calls"]):
        ax = axes[call["ax"]]
        m = call["method"]
        kw = {k: figdata.decode(v, arrays, ax) for k, v in call["kwargs"].items()}
        kw = {k: v for k, v in kw.items() if v is not None or call["kwargs"][k] is None}
        if m == "hist":
            r = call["result"]
            n = np.atleast_2d(np.asarray(figdata.decode(r["n"], arrays), float))
            bins = np.asarray(figdata.decode(r["bins"], arrays), float)
            fill = kw.get("histtype", "bar") in ("bar", "barstacked", "stepfilled")
            args, m = [n, bins], "stairs"
            kw = {k: v for k, v in kw.items() if k in _STAIRS_KW}
            kw["fill"] = fill
        elif m == "hist2d":
            r = call["result"]
            h = np.asarray(figdata.decode(r["h"], arrays), float)
            args = [figdata.decode(r["xedges"], arrays), figdata.decode(r["yedges"], arrays),
                    np.ma.masked_invalid(h.T)]
            m = "pcolormesh"
            kw = {k: v for k, v in kw.items() if k in ("cmap", "norm", "vmin", "vmax", "alpha")}
        else:
            args = [figdata.decode(x, arrays, ax) for x in call["args"]]
        if edit is not None:
            e = edit(call, args, kw)
            if e is None:
                continue
            args, kw = e
        if m == "stairs" and np.ndim(args[0]) == 2:          # a hist of several datasets
            labels = kw.pop("label", None)
            for j, row in enumerate(args[0]):
                k2 = dict(kw)
                if isinstance(labels, (list, tuple)) and j < len(labels):
                    k2["label"] = labels[j]
                elif j == 0 and labels is not None:
                    k2["label"] = labels
                made[c] = ax.stairs(row, args[1], **k2)
            continue
        made[c] = getattr(ax, m)(*args, **kw)
    for ax, a in zip(axes, manifest["axes"]):
        if ax.get_xscale() != a["xscale"]:              # (a recorded set_*scale keeps its
            ax.set_xscale(a["xscale"])                  # parameters, e.g. symlog's linthresh)
        if ax.get_yscale() != a["yscale"]:
            ax.set_yscale(a["yscale"])
        ax.set_xlim(a["xlim"])
        ax.set_ylim(a["ylim"])
    for cb in manifest["colorbars"]:
        if cb["call"] in made:
            fig.colorbar(made[cb["call"]], cax=fig.add_axes(cb["position"]), label=cb["label"])
    st = manifest["suptitle"]
    if st and st["text"]:
        fig.suptitle(st["text"], x=st["x"], y=st["y"], fontsize=st["fontsize"])
    return fig, axes


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz")
    ap.add_argument("--out", default=None, help="default: <stem>_replot.png")
    ap.add_argument("--style", default=None, help="a matplotlib style (name or .mplstyle file)")
    ap.add_argument("--dpi", type=float, default=None, help="default: the original's")
    ap.add_argument("--list", action="store_true", help="print the calls and arrays, no figure")
    a = ap.parse_args()
    if a.list:
        manifest, arrays = figdata.load(a.npz)
        print(f"{manifest['png']}: {len(manifest['axes'])} axes, {len(manifest['calls'])} calls")
        for c, call in enumerate(manifest["calls"]):
            lab = call["kwargs"].get("label")
            keys = sorted(k for k in arrays if k.startswith(f"c{c:03d}_"))
            print(f"  c{c:03d} ax{call['ax']:<2d} {call['method']:12s} "
                  f"{lab if isinstance(lab, str) else '':40s} "
                  + ", ".join(f"{k}{list(arrays[k].shape)}" for k in keys))
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if a.style:
        plt.style.use(a.style)
    manifest, _ = figdata.load(a.npz)
    fig, _ = draw(a.npz)
    out = a.out or os.path.splitext(a.npz)[0] + "_replot.png"
    fig.savefig(out, dpi=a.dpi or manifest["dpi"])
    print(out)


if __name__ == "__main__":
    main()
