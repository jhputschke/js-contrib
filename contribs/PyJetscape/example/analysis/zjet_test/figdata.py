"""Figure data: every figure's plotted arrays next to its PNG, to restyle without the analysis.

    fig, ax = figdata.subplots(2, 3, figsize=(16, 8))      # instead of plt.subplots
    ...                                                    # draw as usual
    figdata.savefig(fig, "figures/foo.png", dpi=100)       # instead of fig.savefig

writes figures/foo.png (unchanged: the drawing calls go through untouched) and

  foo.npz   every drawing call in order (``__manifest__``, JSON: method, axes, arguments,
            keyword arguments) with its arrays (``cNNN_<argument>``), and per axes its final
            title, labels, scales, limits and position;  ``replot.py foo.npz`` redraws it
  foo.root  the same data as ROOT objects, one directory per axes (``ax<i>``): TH1D (stairs,
            hist), TH2D (hist2d), TGraphAsymmErrors (errorbar, bands), TGraph (plot, bar,
            scatter); object titles "label;x label;y label"; the manifest as a TObjString.
            PyROOT if importable, else uproot (graphs then as small TTrees x, y, exl, exh, eyl,
            eyh)

hist and hist2d are stored as their results (contents and edges, after density and weights),
not as the input values.  JS_FIGDATA=0 in the environment writes the PNG only,
JS_FIGDATA_ROOT=0 skips the ROOT file."""
import json
import os
import re

import numpy as np

METHODS = ("plot", "errorbar", "stairs", "hist", "hist2d", "fill_between", "fill_betweenx",
           "axhline", "axvline", "axhspan", "axvspan", "bar", "barh", "scatter", "step", "text",
           "annotate", "imshow", "pcolormesh", "set", "set_title", "set_xlabel", "set_ylabel",
           "set_xscale", "set_yscale", "set_xticks", "set_yticks",
           "set_xticklabels", "set_yticklabels", "legend", "grid", "tick_params", "axis",
           "set_aspect", "invert_xaxis", "invert_yaxis")
# (set_xlim / set_ylim are left out: matplotlib's autoscaling calls them itself; the final
# limits are in the manifest's axes)
_INPUT_ONLY = {"hist": ("weights",), "hist2d": ("weights",)}


def subplots(*args, **kwargs):
    """plt.subplots with the recorder attached."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(*args, **kwargs)
    Recorder(fig)
    return fig, ax


def savefig(fig, path, **kwargs):
    """fig.savefig(path) and, if recorded, <stem>.npz and <stem>.root next to it."""
    fig.savefig(path, **kwargs)
    rec = getattr(fig, "_figdata", None)
    if rec is None or os.environ.get("JS_FIGDATA", "1") == "0":
        return
    stem = os.path.splitext(path)[0]
    manifest, arrays = rec.finish(os.path.basename(path), kwargs.get("dpi"))
    np.savez_compressed(stem + ".npz", __manifest__=np.array(json.dumps(manifest)), **arrays)
    if os.environ.get("JS_FIGDATA_ROOT", "1") != "0":
        write_root(stem + ".root", manifest, arrays)


class Recorder:
    """Wraps a figure's axes' drawing methods: each call is recorded, then made as before."""

    def __init__(self, fig):
        self.fig = fig
        self.calls, self.arrays, self.axes = [], {}, []
        self.mappables, self.colorbars = {}, []
        self._depth = 0
        fig._figdata = self
        for a in fig.axes:
            self._wrap(a)
        orig = fig.colorbar

        def colorbar(mappable, *a, **k):
            cb = orig(mappable, *a, **k)
            self.colorbars.append((self.mappables.get(id(mappable)), cb.ax, k.get("label")))
            return cb
        fig.colorbar = colorbar

    def _wrap(self, ax):
        self.axes.append(ax)
        i = len(self.axes) - 1
        for m in METHODS:
            if hasattr(ax, m):
                setattr(ax, m, self._recording(ax, i, m, getattr(ax, m)))
        for twin in ("twinx", "twiny"):
            orig = getattr(ax, twin)

            def make(*a, _orig=orig, **k):
                t = _orig(*a, **k)
                self._wrap(t)
                return t
            setattr(ax, twin, make)

    def _recording(self, ax, i, method, orig):
        def call(*args, **kwargs):
            if self._depth:                      # a call from inside matplotlib: not ours
                return orig(*args, **kwargs)
            self._depth += 1
            try:
                res = orig(*args, **kwargs)
            finally:
                self._depth -= 1
            self._record(ax, i, method, args, kwargs, res)
            return res
        return call

    def _record(self, ax, i, method, args, kwargs, res):
        c = len(self.calls)
        enc = Encoder(self.arrays, f"c{c:03d}", ax)
        entry = {"ax": i, "method": method}
        drop = _INPUT_ONLY.get(method, ())
        entry["kwargs"] = {k: enc(v, k) for k, v in kwargs.items() if k not in drop}
        if method == "hist":                     # the result, not the input values
            n, bins = res[0], res[1]
            entry["result"] = {"n": enc(np.asarray(n, float), "n"), "bins": enc(bins, "bins")}
            entry["args"] = []
        elif method == "hist2d":
            h, xe, ye, img = res
            entry["result"] = {"h": enc(h, "h"), "xedges": enc(xe, "xedges"),
                               "yedges": enc(ye, "yedges")}
            entry["args"] = []
            self.mappables[id(img)] = c
        else:
            entry["args"] = [enc(v, f"a{j}") for j, v in enumerate(args)]
            if method in ("pcolormesh", "imshow", "scatter"):
                self.mappables[id(res)] = c
        self.calls.append(entry)

    def finish(self, png, dpi):
        """-> (manifest, arrays) with every axes' final state."""
        fig = self.fig
        axes = []
        for ax in self.axes:
            axes.append({"title": ax.get_title(), "xlabel": ax.get_xlabel(),
                         "ylabel": ax.get_ylabel(), "xscale": ax.get_xscale(),
                         "yscale": ax.get_yscale(), "xlim": list(map(float, ax.get_xlim())),
                         "ylim": list(map(float, ax.get_ylim())),
                         "position": list(map(float, ax.get_position().bounds)),
                         "visible": bool(ax.get_visible() and ax.axison)})
        cbs = [{"call": c, "position": list(map(float, cax.get_position().bounds)),
                "label": lab} for c, cax, lab in self.colorbars]
        st = getattr(fig, "_suptitle", None)
        sup = {"text": st.get_text(), "x": float(st.get_position()[0]),
               "y": float(st.get_position()[1]), "fontsize": float(st.get_fontsize())} \
            if st is not None else None
        manifest = {"png": png, "dpi": dpi, "figsize": list(map(float, fig.get_size_inches())),
                    "suptitle": sup, "axes": axes, "colorbars": cbs, "calls": self.calls,
                    "format": "figdata 1"}
        return manifest, self.arrays


class Encoder:
    """Arguments -> JSON, arrays into ``arrays`` under '<prefix>_<name>'."""

    def __init__(self, arrays, prefix, ax):
        self.arrays, self.prefix, self.ax = arrays, prefix, ax

    def __call__(self, v, name):
        import matplotlib
        if v is None or isinstance(v, (bool, str)):
            return v
        if isinstance(v, (int, float, np.integer, np.floating)):
            x = float(v) if isinstance(v, (float, np.floating)) else int(v)
            return x if np.isfinite(x) else {"__float__": str(x)}
        if isinstance(v, matplotlib.colors.Normalize):
            return {"__norm__": type(v).__name__, "vmin": self(v.vmin, name + "_vmin"),
                    "vmax": self(v.vmax, name + "_vmax")}
        if isinstance(v, matplotlib.colors.Colormap):
            return {"__cmap__": v.name}
        if isinstance(v, matplotlib.transforms.Transform):
            for nm, t in (("axes", self.ax.transAxes), ("data", self.ax.transData),
                          ("figure", self.ax.figure.transFigure)):
                if v == t:
                    return {"__transform__": nm}
            return {"__repr__": repr(v)}
        if isinstance(v, np.ma.MaskedArray):
            v = v.astype(float).filled(np.nan)
        if isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
            return list(v)
        if isinstance(v, (list, tuple)) and any(np.ndim(x) > 0 for x in v) and \
                len({np.shape(x) for x in v}) > 1:
            return {"__list__": [self(x, f"{name}_{j}") for j, x in enumerate(v)]}
        if isinstance(v, (np.ndarray, list, tuple)):
            arr = np.asarray(v)
            if arr.dtype.kind in "biuf":
                key = f"{self.prefix}_{name}"
                self.arrays[key] = arr
                return {"__array__": key, "tuple": isinstance(v, tuple)}
            if arr.dtype.kind in "U":
                return arr.tolist()
            return {"__list__": [self(x, f"{name}_{j}") for j, x in enumerate(v)]}
        return {"__repr__": repr(v)}


def decode(v, arrays, ax=None):
    """The inverse of Encoder (for replot.py); transforms need the axes."""
    import matplotlib
    if isinstance(v, list):
        return v
    if not isinstance(v, dict):
        return v
    if "__array__" in v:
        a = np.asarray(arrays[v["__array__"]])
        return tuple(a.tolist()) if v.get("tuple") and a.ndim == 1 and a.size <= 4 else a
    if "__float__" in v:
        return float(v["__float__"])
    if "__list__" in v:
        return [decode(x, arrays, ax) for x in v["__list__"]]
    if "__norm__" in v:
        cls = getattr(matplotlib.colors, v["__norm__"], matplotlib.colors.Normalize)
        return cls(vmin=decode(v["vmin"], arrays), vmax=decode(v["vmax"], arrays))
    if "__cmap__" in v:
        return v["__cmap__"]
    if "__transform__" in v and ax is not None:
        return {"axes": ax.transAxes, "data": ax.transData,
                "figure": ax.figure.transFigure}[v["__transform__"]]
    return None                                      # a repr: not restorable


def load(path):
    """<stem>.npz -> (manifest, arrays)."""
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files if k != "__manifest__"}
        manifest = json.loads(str(f["__manifest__"]))
    return manifest, arrays


# ── ROOT

def _name(s, used):
    base = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")[:60] or "obj"
    n, k = base, 1
    while n in used:
        k += 1
        n = f"{base}_{k}"
    used.add(n)
    return n


def objects(manifest, arrays):
    """The figure's data as plain records per axes: (axes, kind, name, title, data)."""
    out = []
    for c, call in enumerate(manifest["calls"]):
        ax = manifest["axes"][call["ax"]]
        m, kw = call["method"], call["kwargs"]
        args = [decode(x, arrays) for x in call["args"]]
        label = kw.get("label")
        label = label if isinstance(label, str) else ""
        stem = f"c{c:03d}_{m}" + (f"_{label}" if label else "")
        title = f"{label or m};{ax['xlabel']};{ax['ylabel']}"

        def arr(x):
            return np.asarray(x, float).ravel()
        try:
            if m == "stairs" and args:
                vals = arr(args[0])
                edges = arr(args[1]) if len(args) > 1 else arr(decode(kw.get("edges"), arrays))
                base = kw.get("baseline")
                base = decode(base, arrays) if isinstance(base, dict) else base
                if base is not None and np.ndim(base) > 0:
                    x = 0.5 * (edges[1:] + edges[:-1])
                    lo, hi = arr(base), vals
                    mid = 0.5 * (lo + hi)
                    out.append((call["ax"], "band", stem, title,
                                {"x": x, "y": mid, "exl": x - edges[:-1], "exh": edges[1:] - x,
                                 "eyl": mid - np.minimum(lo, hi), "eyh": np.maximum(lo, hi) - mid}))
                else:
                    out.append((call["ax"], "th1", stem, title, {"values": vals, "edges": edges}))
            elif m == "hist":
                n = np.asarray(decode(call["result"]["n"], arrays), float)
                bins = arr(decode(call["result"]["bins"], arrays))
                for j, row in enumerate(np.atleast_2d(n)):
                    out.append((call["ax"], "th1", stem + (f"_{j}" if n.ndim > 1 else ""), title,
                                {"values": row, "edges": bins}))
            elif m == "hist2d":
                r = call["result"]
                out.append((call["ax"], "th2", stem, title,
                            {"values": np.asarray(decode(r["h"], arrays), float),
                             "xedges": arr(decode(r["xedges"], arrays)),
                             "yedges": arr(decode(r["yedges"], arrays))}))
            elif m == "errorbar" and len(args) >= 2:
                x, y = arr(args[0]), arr(args[1])
                yerr = args[2] if len(args) > 2 else decode(kw.get("yerr"), arrays)
                xerr = args[3] if len(args) > 3 else decode(kw.get("xerr"), arrays)

                def split(e):
                    if e is None:
                        return np.zeros_like(x), np.zeros_like(x)
                    e = np.asarray(e, float)
                    if e.ndim == 2:
                        return e[0], e[1]
                    e = np.broadcast_to(e, x.shape).astype(float)
                    return e, e
                eyl, eyh = split(yerr)
                exl, exh = split(xerr)
                out.append((call["ax"], "graph", stem, title,
                            {"x": x, "y": y, "exl": exl, "exh": exh, "eyl": eyl, "eyh": eyh}))
            elif m in ("plot", "scatter", "bar", "step") and args:
                if len(args) >= 2 and not isinstance(args[1], str):
                    y = arr(args[1])
                    cat = np.asarray(args[0]).dtype.kind not in "biuf"   # categories: 0, 1, ..
                    x = np.arange(len(y), dtype=float) if cat else arr(args[0])
                    if cat:
                        title = f"{label or m} ({', '.join(map(str, args[0]))});" \
                                f"{ax['xlabel']};{ax['ylabel']}"
                else:
                    y = arr(args[0])
                    x = np.arange(len(y), dtype=float)
                out.append((call["ax"], "graph", stem, title, {"x": x, "y": y}))
            elif m == "fill_between" and len(args) >= 2:
                x, y1 = arr(args[0]), arr(args[1])
                y2 = arr(args[2]) if len(args) > 2 else np.zeros_like(y1)
                y2 = np.broadcast_to(y2, y1.shape)
                mid = 0.5 * (y1 + y2)
                z = np.zeros_like(x)
                out.append((call["ax"], "band", stem, title,
                            {"x": x, "y": mid, "exl": z, "exh": z,
                             "eyl": mid - np.minimum(y1, y2), "eyh": np.maximum(y1, y2) - mid}))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def write_root(path, manifest, arrays):
    try:
        import ROOT
    except ImportError:
        return _write_uproot(path, manifest, arrays)
    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.AddDirectory(False)
    f = ROOT.TFile(path, "RECREATE")
    f.WriteObject(ROOT.TObjString(json.dumps(manifest)), "manifest")
    dirs, used = {}, {}
    for i, a in enumerate(manifest["axes"]):
        d = f.mkdir(f"ax{i}", a["title"] or f"axes {i}")
        dirs[i], used[i] = d, set()
    for i, kind, name, title, d in objects(manifest, arrays):
        n = _name(name, used[i])
        if kind == "th1":
            e = np.ascontiguousarray(d["edges"], float)
            h = ROOT.TH1D(n, title, len(e) - 1, e)
            for b, v in enumerate(d["values"]):
                h.SetBinContent(b + 1, float(v) if np.isfinite(v) else 0.0)
            obj = h
        elif kind == "th2":
            xe = np.ascontiguousarray(d["xedges"], float)
            ye = np.ascontiguousarray(d["yedges"], float)
            h = ROOT.TH2D(n, title, len(xe) - 1, xe, len(ye) - 1, ye)
            for bx in range(len(xe) - 1):
                for by in range(len(ye) - 1):
                    v = d["values"][bx, by]
                    h.SetBinContent(bx + 1, by + 1, float(v) if np.isfinite(v) else 0.0)
            obj = h
        else:
            x = np.ascontiguousarray(d["x"], float)
            y = np.ascontiguousarray(d["y"], float)
            if "eyl" in d:
                g = ROOT.TGraphAsymmErrors(len(x), x, y,
                                           *[np.ascontiguousarray(d[k], float)
                                             for k in ("exl", "exh", "eyl", "eyh")])
            else:
                g = ROOT.TGraph(len(x), x, y)
            g.SetNameTitle(n, title)
            obj = g
        dirs[i].WriteObject(obj, n)
    f.Close()


def _write_uproot(path, manifest, arrays):
    import uproot
    with uproot.recreate(path) as f:
        f["manifest"] = json.dumps(manifest)
        used = {i: set() for i in range(len(manifest["axes"]))}
        for i, kind, name, title, d in objects(manifest, arrays):
            n = f"ax{i}/{_name(name, used[i])}"
            if kind == "th1":
                f[n] = (np.nan_to_num(d["values"]), d["edges"])
            elif kind == "th2":
                f[n] = (np.nan_to_num(d["values"]), d["xedges"], d["yedges"])
            else:
                t = f.mktree(n, {k: np.float64 for k in d}, title=title)
                t.extend({k: np.asarray(v, float) for k, v in d.items()})
