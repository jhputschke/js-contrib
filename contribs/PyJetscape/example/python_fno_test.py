"""
examples/python_fno_test.py

Brick validation test replacing the C++ FnoHydro module with the Python
PyFNOHydro trampoline.

The pipeline (Mode B — manual task list) is:
    TrentoInitial → FreestreamMilne → PyFNOHydro
                → JetEnergyLossManager → JetEnergyLoss (Matter)
                → HadronizationManager → Hadronization (ColorlessHadronization)

Run from the JETSCAPE-FNO repository root:
    conda activate js_fno
    cd /path/to/JETSCAPE-FNO
    python examples/python_fno_test.py

Optional arguments:
    --model   Path to a TorchScript .pt file.
              Default: fno_hydro/model/traced_JS3.7_10k_3feat_fno_model_cpu_40_60_59bins.pt
    --main    Path to main XML.   Default: config/jetscape_main.xml
    --user    Path to user XML.   Default: config/jetscape_user_AA_dukeTune.xml
    --events  Number of events.   Default: 1
    --device  torch device.       Default: cpu
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Make sure the python package is importable ────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

# torch must be imported BEFORE pyjetscape_core (loaded via jetscape.__init__)
# to avoid a dual-OpenMP-runtime crash: ROOT (pulled in by pyjetscape_core)
# and PyTorch each ship libomp; whichever is initialised second segfaults.
import torch  # noqa: E402

from jetscape import create_module, fno_config_from_xml
from jetscape.fno_hydro import PyFNOHydro
from jetscape.run_jetscape import run_manual
from jetscape.utils import bulk_info_to_numpy, bulk_info_to_tensor

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_MODEL = os.path.join(
    _REPO_ROOT,
    "fno_hydro", "models",
    "traced_JS3.7_10k_3feat_fno_model_cpu_40_60_59bins.pt",
)
_DEFAULT_MAIN = os.path.join(_REPO_ROOT, "config", "jetscape_main.xml")
_DEFAULT_USER = os.path.join(_REPO_ROOT, "fno_hydro/config", "jetscape_user_root_bulk_test.xml")

# FNO grid configuration — must match the model the .pt file was trained on.
# These values correspond to traced_JS3.7_10k_3feat_fno_model_cpu_40_60_59bins.pt
FNO_CONFIG = dict(
    nx=60,
    ny=60,
    ntau=59,          # time steps predicted by the model
    neta=1,
    n_features=3,     # [energy_density*tau, temperature, vx]  (3-feature model)
    x_min=-15.0,      # fm
    y_min=-15.0,      # fm
    dtau=0.1,         # fm/c
    deta=0.0,
    T_freeze=0.136,   # GeV  (freeze-out temperature)
    tau_normalise=True,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",  default=_DEFAULT_MODEL,
                   help="Path to TorchScript .pt model file")
    p.add_argument("--main",   default=_DEFAULT_MAIN,
                   help="Path to main XML config")
    p.add_argument("--user",   default=_DEFAULT_USER,
                   help="Path to user XML config")
    p.add_argument("--events", type=int, default=1,
                   help="Number of collision events to simulate")
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "cuda", "mps"],
                   help="PyTorch device for FNO inference")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip matplotlib diagnostics (useful in headless CI)")
    p.add_argument(
        "--config-source",
        choices=["dict", "python-xml", "jetscape-xml"],
        default="dict",
        help=(
            "How to supply the FNO grid configuration:\n"
            "  dict         — hardcoded FNO_CONFIG dict (original approach)\n"
            "  python-xml   — parse user XML with Python stdlib (fno_config_from_xml)\n"
            "  jetscape-xml — let PyFNOHydro read <Hydro><FNO> via JetScapeXML\n"
            "                 C++ binding in InitializeHydro() [requires rebuild]"
        ),
    )
    return p.parse_args()


def build_pipeline(args: argparse.Namespace):
    """
    Construct all JETSCAPE module objects for Mode-B execution.

    Returns a list in pipeline order suitable for run_manual().

    Three config-source strategies are demonstrated (select with
    ``--config-source``):

    dict (default)
        Use the hardcoded ``FNO_CONFIG`` dict defined at the top of this
        script.  Fastest to get running; no XML parsing needed.

    python-xml
        Call ``fno_config_from_xml(args.user)`` to parse ``<Hydro><FNO>``
        from the user XML with Python's stdlib ``xml.etree.ElementTree``.
        Keeps the XML as the single source of truth without requiring a C++
        rebuild.

    jetscape-xml
        Pass ``config=None`` to ``PyFNOHydro``; the module reads all grid
        parameters from the loaded JETSCAPE XML inside ``InitializeHydro()``
        via ``self.get_xml_element_*`` — exactly like C++ ``FnoHydro``.
        Requires that ``bind_framework.cc`` has been rebuilt with the
        ``get_xml_element_*`` binding additions.
    """
    if not os.path.isfile(args.model):
        raise FileNotFoundError(
            f"FNO model file not found: {args.model}\n"
            "Download or train a model and pass --model <path>."
        )

    # ── Initial state ─────────────────────────────────────────────────────────
    ini = create_module("TrentoInitial")

    # ── Pre-equilibrium ───────────────────────────────────────────────────────
    preeq = create_module("NullPreDynamics")

    # ── FNO hydro (Python trampoline) — config-source dispatch ───────────────
    source = args.config_source

    if source == "dict":
        # ── Approach 1: hardcoded dict ────────────────────────────────────────
        # Original approach: every parameter lives in this script.
        cfg = {**FNO_CONFIG, "device": args.device}
        fno_hydro = PyFNOHydro(args.model, cfg)
        print(f"[config-source=dict] Using hardcoded FNO_CONFIG dict.")

    elif source == "python-xml":
        # ── Approach 2: parse user XML with Python stdlib ─────────────────────
        # fno_config_from_xml() reads <Hydro><FNO> tags using
        # xml.etree.ElementTree — no rebuild needed, works before js.Init().
        cfg = fno_config_from_xml(args.user, device=args.device)
        fno_hydro = PyFNOHydro(args.model, cfg)
        print(f"[config-source=python-xml] Config parsed from {args.user}:")
        for k, v in cfg.items():
            print(f"    {k} = {v}")

    elif source == "jetscape-xml":
        # ── Approach 3: JetScapeXML class via C++ binding ────────────────────
        # Pass config=None; PyFNOHydro.InitializeHydro() calls
        # self.get_xml_element_{int,double}(["Hydro", "FNO", ...]) — the same
        # C++ accessors used by FnoHydro.  Requires bind_framework.cc rebuild.
        fno_hydro = PyFNOHydro(args.model, device=args.device)
        print(
            f"[config-source=jetscape-xml] Config will be read from "
            f"<Hydro><FNO> in {args.user} via JetScapeXML binding at Init time."
        )

    else:
        raise ValueError(f"Unknown --config-source: {source!r}")

    # JetEnergyLossManager and HadronizationManager are framework-internal
    # classes not registered in the module factory; they are only created by
    # the C++ core when running fully from XML (Mode A).
    # For this bulk-medium validation test we run the three-stage chain only:
    #   InitialState → PreequilibriumDynamics → FluidDynamics

    # Pipeline order: IS → Preq → Hydro
    return [ini, preeq, fno_hydro]


def inspect_results(fno_hydro: PyFNOHydro, n_features: int, no_plots: bool) -> None:
    """Print bulk_info diagnostics and optionally plot energy density slices."""
    bulk = fno_hydro.get_bulk_info()

    ntau = bulk.ntau
    nx   = bulk.nx
    ny   = bulk.ny
    size = bulk.get_data_size()
    print(f"\n=== Bulk Info Grid ===")
    print(f"  ntau = {ntau}, nx = {nx}, ny = {ny}")
    print(f"  stored cells: {size}  (expected {ntau * nx * ny})")

    if size == 0:
        print("  [!] No evolution data — EvolveHydro() may not have run.")
        return

    # ── numpy view ────────────────────────────────────────────────────────────
    arr = bulk_info_to_numpy(bulk, n_features)
    print(f"\n=== numpy array shape: {arr.shape}  dtype: {arr.dtype} ===")
    print(f"  energy_density  min={arr[:,:,:,0].min():.4f}  "
          f"max={arr[:,:,:,0].max():.4f}  [GeV/fm³]")
    if n_features > 1:
        print(f"  temperature     min={arr[:,:,:,1].min():.4f}  "
              f"max={arr[:,:,:,1].max():.4f}  [GeV]")
    if n_features > 2:
        print(f"  vx              min={arr[:,:,:,2].min():.4f}  "
              f"max={arr[:,:,:,2].max():.4f}")

    # ── tensor view ───────────────────────────────────────────────────────────
    try:
        import torch
        t = bulk_info_to_tensor(bulk, n_features)
        print(f"\n=== torch.Tensor shape: {list(t.shape)}  dtype: {t.dtype} ===")
        print(f"  (batch=1, features={n_features}, nx={nx}, ny={ny}, ntau={ntau})")
    except ImportError:
        pass

    # ── freeze-out surface ────────────────────────────────────────────────────
    surface = fno_hydro.get_surface_cells()
    print(f"\n=== Freeze-out Surface ===")
    print(f"  Number of surface cells: {len(surface)}")

    # ── optional plots ────────────────────────────────────────────────────────
    if not no_plots:
        _plot_slices(arr, n_features)


def _plot_slices(arr: "np.ndarray", n_features: int) -> None:
    """Plot energy density slices at tau_0, tau_mid, tau_end."""
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n[!] matplotlib not found — skipping diagnostic plots.")
        return

    ntau, nx, ny, _ = arr.shape

    tau_steps = [0, ntau // 2, ntau - 1]
    labels    = [f"τ step {k}" for k in tau_steps]

    fig, axes = plt.subplots(1, len(tau_steps), figsize=(14, 4))
    for ax, k, lbl in zip(axes, tau_steps, labels):
        ed = arr[k, :, :, 0]
        im = ax.imshow(ed.T, origin="lower", cmap="inferno")
        fig.colorbar(im, ax=ax, label="e [GeV/fm³]")
        ax.set_title(lbl)
        ax.set_xlabel("x grid index")
        ax.set_ylabel("y grid index")

    fig.suptitle("PyFNOHydro — energy density slices")
    fig.tight_layout()

    out_path = "python_fno_test_bulk.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n  Saved bulk info plot → {out_path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  JETSCAPE-FNO Python interface test")
    print(f"  model         : {args.model}")
    print(f"  main          : {args.main}")
    print(f"  user          : {args.user}")
    print(f"  events        : {args.events}")
    print(f"  device        : {args.device}")
    print(f"  config-source : {args.config_source}")
    print("=" * 60)

    modules = build_pipeline(args)

    # Locate the PyFNOHydro instance for post-run inspection
    fno_hydro = next(m for m in modules if isinstance(m, PyFNOHydro))

    js = run_manual(args.main, args.user, modules, n_events=args.events)

    # n_features: read from config if already resolved, else fall back to
    # FNO_CONFIG (config=None case: _config is populated after js.Init())
    n_features = (
        fno_hydro._config["n_features"]
        if fno_hydro._config is not None
        else FNO_CONFIG["n_features"]
    )
    inspect_results(fno_hydro, n_features=n_features,
                    no_plots=args.no_plots)

    print("\nTest complete.")


if __name__ == "__main__":
    main()
