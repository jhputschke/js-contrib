"""
python/jetscape/fno_hydro.py

PyFNOHydro — Python FluidDynamics trampoline that runs a PyTorch FNO model
as a first-class JETSCAPE hydro module.

Usage (three model-loading approaches):

    # Approach 1: JIT-traced .pt  (compatible with C++ FnoHydro traced models)
    hydro = PyFNOHydro("models/traced.pt", config)

    # Approach 2: checkpoint + Python class
    from my_model import FNOModel
    net = FNOModel(...)
    hydro = PyFNOHydro((net, "checkpoints/epoch50.pt"), config)

    # Approach 3: live Python model (no serialisation)
    hydro = PyFNOHydro(net, config)

All three produce a PyFNOHydro instance that can be added to a JETSCAPE
pipeline with:

    js.Add(hydro)     # after Add(preeq) and before Add(jloss_mgr)

Config source options
---------------------
config dict (any approach above):
    Explicitly pass every grid parameter — the original approach.

fno_config_from_xml(user_xml_path):
    Parse the JETSCAPE user XML with Python's stdlib before constructing the
    pipeline.  No rebuild required.  Use when you want to inspect the config
    before run_manual() is called.

config=None (reads from JetScapeXML via C++ binding):
    Pass config=None and the module reads <Hydro><FNO> values from the already-
    loaded JETSCAPE XML inside InitializeHydro() — the most JETSCAPE-native
    approach, exactly like C++ FnoHydro.  Requires the shared library to be
    rebuilt after the binding addition in bind_framework.cc.
"""

from __future__ import annotations

import xml.etree.ElementTree as _ET
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from .pyjetscape_core import FluidDynamics
from .utils import rebin_preeq_to_fno_grid


def fno_config_from_xml(xml_path: str, *, device: str = "cpu") -> dict:
    """Build an FNO config dict by parsing a JETSCAPE user XML file.

    Reads ``<Hydro><FNO>`` tags using Python's stdlib
    ``xml.etree.ElementTree`` — no C++ framework or rebuild needed.
    Can be called before the pipeline is constructed.

    Parameters
    ----------
    xml_path : str
        Path to the JETSCAPE user XML file (the one passed as ``--user``).
    device : str
        PyTorch device string to embed in the returned dict (``'cpu'``,
        ``'cuda'``, ``'mps'``).

    Returns
    -------
    dict
        Config dict compatible with ``PyFNOHydro(model, config)``.

    Raises
    ------
    ValueError
        If ``<Hydro><FNO>`` block or a required tag is missing.
    """
    tree = _ET.parse(xml_path)
    root = tree.getroot()

    fno = root.find("Hydro/FNO")
    if fno is None:
        raise ValueError(
            f"No <Hydro><FNO> block found in {xml_path}. "
            "Make sure the user XML contains a <Hydro><FNO>…</FNO></Hydro> section."
        )

    def _int(tag, default=None):
        el = fno.find(tag)
        if el is None:
            if default is None:
                raise ValueError(f"<{tag}> not found in <Hydro><FNO> of {xml_path}")
            return default
        return int(el.text)

    def _float(tag, default=None):
        el = fno.find(tag)
        if el is None:
            if default is None:
                raise ValueError(f"<{tag}> not found in <Hydro><FNO> of {xml_path}")
            return default
        return float(el.text)

    n_features = _int("n_features")
    return dict(
        nx          = _int("nx"),
        ny          = _int("ny"),
        ntau        = _int("ntau"),
        neta        = _int("neta", default=1),
        n_features  = n_features,
        x_min       = _float("x_min"),
        y_min       = _float("y_min"),
        dtau        = _float("dtau"),
        deta        = _float("deta", default=0.0),
        T_freeze    = _float("freezeout_temperature"),
        tau_normalise = n_features == 3,
        device      = device,
    )


class PyFNOHydro(FluidDynamics):
    """
    JETSCAPE FluidDynamics module backed by a PyTorch FNO model.

    The model receives initial conditions from the pre-equilibrium module
    and writes hydrodynamic evolution history back into the C++ EvolutionHistory,
    including finding the constant-temperature freeze-out surface.

    Parameters
    ----------
    model : str | (nn.Module, str) | nn.Module
        * ``str``                 — path to a TorchScript ``.pt`` file (loaded
          with ``torch.jit.load``).  Compatible with traced models from
          C++ FnoHydro.
        * ``(nn.Module, str)``    — tuple of (model instance, checkpoint path).
          The checkpoint is loaded with ``torch.load`` and
          ``load_state_dict``.
        * ``nn.Module``           — live model used as-is; no loading step.
    config : dict | None
        When a **dict**, must contain:
          nx      : int      — FNO grid points in x
          ny      : int      — FNO grid points in y
          ntau    : int      — number of time steps predicted by the model
          neta    : int      — eta grid points (1 for boost-invariant)
          n_features : int   — number of model input/output features (3 or 4)
          x_min   : float    — minimum x [fm]  (e.g. -7.5)
          y_min   : float    — minimum y [fm]  (e.g. -7.5)
          dtau    : float    — tau step [fm/c]
          deta    : float    — eta step (0 for boost-invariant)
          T_freeze : float   — freeze-out temperature [GeV]
        Optional keys in dict:
          device  : str      — 'cpu', 'cuda', 'mps'  (default: 'cpu')
          tau_normalise : bool — whether model output is ed*tau (default True
                                 for n_features==3, False for n_features==4)

        When **None**, grid parameters are read from the loaded JETSCAPE XML
        (``<Hydro><FNO>`` block) inside ``InitializeHydro()``, i.e. after
        ``js.Init()`` has opened the XML files.  This is the most JETSCAPE-
        native approach and mirrors what C++ ``FnoHydro`` does.
        Requires that ``bind_framework.cc`` has been rebuilt with the
        ``get_xml_element_*`` binding additions.
    device : str
        PyTorch device string (``'cpu'``, ``'cuda'``, ``'mps'``).
        Only used when *config* is ``None``; when config is a dict, use
        ``config.get('device', 'cpu')`` instead.
    """

    def __init__(self, model, config=None, *, device: str = "cpu"):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyFNOHydro requires PyTorch.  "
                "Install it with: pip install torch"
            )
        super().__init__()

        # ── Load model ────────────────────────────────────────────────────────
        if isinstance(model, str):
            # Approach 1: JIT-traced .pt
            self._model = torch.jit.load(model)
            self._model.eval()
        elif isinstance(model, tuple) and len(model) == 2:
            # Approach 2: (nn.Module instance, checkpoint path)
            net, ckpt_path = model
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            net.load_state_dict(state)
            net.eval()
            self._model = net
        elif _TORCH_AVAILABLE and isinstance(model, nn.Module):
            # Approach 3: live Python model
            self._model = model
            self._model.eval()
        else:
            raise TypeError(
                f"Unsupported model argument type: {type(model)}. "
                "Expected: str path, (nn.Module, str) tuple, or nn.Module."
            )

        # config=None → read from JETSCAPE XML in InitializeHydro()
        self._config = config
        _dev = config.get("device", device) if config is not None else device
        self._device = torch.device(_dev)
        self._model  = self._model.to(self._device)

        self.SetId("PyFNOHydro")

        # Preserve bulk_info.data across ClearTasks() so Python can read it
        # after run_manual() returns.  EvolveHydro() calls clear_up_evolution_data()
        # itself at the start of each event, so multi-event runs are still correct.
        self.set_preserve_bulk_info(True)

    # ── JETSCAPE interface ─────────────────────────────────────────────────────

    def _read_config_from_xml(self) -> dict:
        """Read FNO grid config from the loaded JETSCAPE XML via C++ binding.

        Uses ``get_xml_element_{int,double,text}`` exposed on
        ``JetScapeModuleBase`` (requires bind_framework.cc rebuild).
        Called only inside ``InitializeHydro()``, after the XML files have
        been opened by the framework.
        """
        g = ["Hydro", "FNO"]
        n_features = self.get_xml_element_int(g + ["n_features"])
        return dict(
            nx           = self.get_xml_element_int(g + ["nx"]),
            ny           = self.get_xml_element_int(g + ["ny"]),
            ntau         = self.get_xml_element_int(g + ["ntau"]),
            neta         = self.get_xml_element_int(g + ["neta"], False) or 1,
            n_features   = n_features,
            x_min        = self.get_xml_element_double(g + ["x_min"]),
            y_min        = self.get_xml_element_double(g + ["y_min"]),
            dtau         = self.get_xml_element_double(g + ["dtau"]),
            deta         = self.get_xml_element_double(g + ["deta"], False),
            T_freeze     = self.get_xml_element_double(g + ["freezeout_temperature"]),
            tau_normalise = n_features == 3,
        )

    def InitializeHydro(self, params):
        """Read grid configuration and initialise hydro.

        Called by FluidDynamics::Init() after the XML is loaded.
        If *config* was ``None`` at construction, grid parameters are read
        from ``<Hydro><FNO>`` in the JETSCAPE XML via ``get_xml_element_*``.
        """
        if self._config is None:
            self._config = self._read_config_from_xml()
        cfg = self._config
        self._nx        = int(cfg["nx"])
        self._ny        = int(cfg["ny"])
        self._ntau      = int(cfg["ntau"])
        self._neta      = int(cfg.get("neta", 1))
        self._n_features = int(cfg["n_features"])
        self._x_min     = float(cfg["x_min"])
        self._y_min     = float(cfg["y_min"])
        self._dx        = -2.0 * self._x_min / self._nx
        self._dy        = -2.0 * self._y_min / self._ny
        self._dtau      = float(cfg["dtau"])
        self._deta      = float(cfg.get("deta", 0.0))
        self._T_freeze  = float(cfg["T_freeze"])
        # For n_features==3 the model predicts ed*tau; for ==4 it predicts ed.
        self._tau_norm  = bool(cfg.get("tau_normalise",
                                       self._n_features == 3))

    def EvolveHydro(self):
        """
        1. Fetch pre-equilibrium data as numpy (via get_preeq_pointer()).
        2. Rebin to FNO grid → input tensor (1, n_features, nx, ny, ntau).
        3. Run model (GIL is still held; heavy compute is in torch C++).
        4. Un-normalise output (ed*tau → ed for n_features==3).
        5. Set grid info on bulk_info.
        6. Store fluid cells from output numpy array.
        7. Mark hydro as FINISHED.
        8. Find freeze-out surface.
        """
        preeq = self.get_preeq_pointer()
        ini   = self.get_ini_pointer()

        if preeq is None:
            raise RuntimeError(
                "PyFNOHydro: pre-equilibrium module pointer is None. "
                "Make sure a PreequilibriumDynamics module was added before "
                "PyFNOHydro in the pipeline."
            )

        tau0 = preeq.GetPreequilibriumEndTime()

        # ── Step 1+2: build input tensor ──────────────────────────────────────
        input_np = rebin_preeq_to_fno_grid(
            preeq, ini,
            nx_fno=self._nx, ny_fno=self._ny,
            x_min_fno=self._x_min, y_min_fno=self._y_min,
            dx_fno=self._dx,   dy_fno=self._dy,
            n_features=self._n_features,
        )  # shape (n_features, nx, ny, 1)

        # Repeat along tau-axis to get (n_features, nx, ny, ntau)
        input_np = np.repeat(input_np, self._ntau, axis=3)

        if self._tau_norm and self._n_features == 3:
            # Multiply ed by tau_0 on the first step so the model receives the
            # same normalised input it was trained on:  ed * tau_0 → channel 0.
            input_np[0, :, :, :] *= tau0

        input_tensor = torch.from_numpy(input_np).to(self._device)
        # Add batch dim → (1, n_features, nx, ny, ntau)
        input_tensor = input_tensor.unsqueeze(0)

        # ── Step 3: run model ─────────────────────────────────────────────────
        with torch.no_grad():
            output = self._model(input_tensor)  # (1, n_features, nx, ny, ntau)

        # Prepend the initial time step (tau_0) so total ntau = model_ntau + 1
        output = torch.cat([input_tensor[:, :, :, :, :1], output], dim=4)
        # output shape: (1, n_features, nx, ny, ntau+1)
        ntau_total = output.shape[4]

        # ── Step 4: un-normalise if needed ─────────────────────────────────────
        if self._tau_norm and self._n_features == 3:
            # output[..., 0, k] = ed * tau(k)  → divide by tau(k)
            tau_vec = tau0 + self._dtau * torch.arange(
                ntau_total, dtype=output.dtype, device=self._device
            )  # shape (ntau_total,)
            # Broadcast over (1, 1, nx, ny, ntau_total)
            output[:, 0, :, :, :] = output[:, 0, :, :, :] / tau_vec

        # Convert to numpy (n_features, nx, ny, ntau_total), float32, C-contiguous
        out_np = output.squeeze(0).cpu().numpy().astype(np.float32)
        if not out_np.flags["C_CONTIGUOUS"]:
            out_np = np.ascontiguousarray(out_np)

        # ── Step 5: set grid metadata on bulk_info ────────────────────────────
        self.set_hydro_grid_info(
            tau_min  = float(tau0),
            dtau     = self._dtau,
            ntau     = ntau_total,
            x_min    = self._x_min,
            dx       = self._dx,
            nx       = self._nx,
            y_min    = self._y_min,
            dy       = self._dy,
            ny       = self._ny,
            eta_min  = 0.0,
            deta     = self._deta,
            neta     = self._neta,
            boost_inv = True,
            tau_eta_is_tz = False,
        )

        # ── Step 6: store fluid cells ─────────────────────────────────────────
        self.clear_up_evolution_data()
        self.store_fluid_cells_from_numpy(out_np)

        # ── Step 7: mark finished ─────────────────────────────────────────────
        self.set_hydro_status_finished()

        # ── Step 8: freeze-out surface ────────────────────────────────────────
        self.find_freezeout_surface(self._T_freeze)
