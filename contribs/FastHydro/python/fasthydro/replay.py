"""Re-run the jet leg from a droplet dump, with no X-SCAPE process.

Replay deliberately does **not** go through `FastHydro`, and so does not need
`pyjetscape_core` at all -- it drives `fast_data` directly.  That is the point: once the
droplets are on disk, changing the solver settings, the grid or the EoS costs one hydro run
instead of a full Matter+LBT shower, and it can be done on a machine with no X-SCAPE build.

Determinism.  On CPU/float64 a replay reproduces the original `arr_jet` **bit for bit**, and
`verify_replay` asserts exactly that.  Two properties make it legitimate rather than lucky:
the solver is deterministic, and the deposit is step-sequence independent -- with
``tau_eval_mode="dep"`` the kernel is evaluated at ``tau_dep`` itself, so the deposited shape
does not depend on how the solver chose its substeps (see fast_data/liquefier/source.py).
On GPU or MPS, reductions are not reproducible; compare within a tolerance instead.
"""

from __future__ import annotations

import numpy as np

from .cells import DEFAULT_FIELDS  # noqa: F401  (re-exported for symmetry)
from .grid import GridSpec

__all__ = ["replay_event", "replay_pair", "verify_replay"]


def replay_event(cfg, e0, droplets, params, *, device=None, dtype=None, source_kw=None):
    """-> (arr, src, diag), the same three things `FastHydro.EvolveHydro` produces.

    `e0` is the (nx, ny, neta) float64 initial energy density in GeV/fm^3 -- the very array
    the original run used, not a regenerated one, if reproducibility matters.
    """
    import torch

    from fast_data import evolve, fv
    from fast_data.eos import resolve_eos
    from fast_data.liquefier import CausalLiquefierSource

    g = GridSpec.from_cfg(cfg)
    dev = device or cfg["run"]["device"]
    dt = getattr(torch, dtype or cfg["run"]["dtype"])
    if str(dev) == "mps" and dt is torch.float64:
        raise ValueError("device 'mps' has no float64; pass dtype='float32'")

    _np_eos, eos = resolve_eos(cfg["eos"], device=dev, dtype=dt)
    grid = g.to_fv_grid(dev, dt)

    tr = cfg["transport"]
    transport = None
    if str(tr["mode"]).lower() in ("israel_stewart", "viscous", "is"):
        transport = fv.Transport(
            eta_over_s=tr["eta_over_s"], zeta_over_s=tr["zeta_over_s"],
            tau_pi_coeff=tr["tau_pi_coeff"], delta_pipi=tr["delta_pipi"],
            delta_PiPi=tr["delta_PiPi"], tau_min=tr["tau_min"],
            pi_rho_max=tr["pi_rho_max"], pi_e_min=tr["pi_e_min"],
            pi_advection=tr["pi_advection"],
            Pi_p_bounds=(tuple(tr["Pi_p_bounds"])
                         if tr["Pi_p_bounds"] is not None else None))

    e0 = np.ascontiguousarray(e0, dtype=np.float64)
    if e0.shape != (g.nx, g.ny, g.neta):
        raise ValueError(f"IC {e0.shape} != config grid {(g.nx, g.ny, g.neta)}")
    q = fv.initial_state_from_energy(torch.as_tensor(e0, device=grid.device,
                                                     dtype=grid.dtype)[None], g.tau0, eos)

    src = None
    if droplets is not None and len(droplets) > 0:
        s = dict(cfg["source"])
        s.update(source_kw or {})
        src = CausalLiquefierSource(
            droplets, params, mode=s["mode"], renorm=s["renorm"],
            tau_eval_mode=s["tau_eval_mode"], n_sub=s["n_sub"], n_sub_max=s["n_sub_max"],
            min_in_grid=s["min_in_grid"], on_out_of_grid=s["on_out_of_grid"],
            device=dev, dtype=dt)

    # see the note in hydro.py: pi/Pi must exist or strang_step runs ideal regardless
    pi = grid.zeros(1, 10) if transport is not None else None
    Pi = grid.zeros(1, 1) if transport is not None else None

    t, o = cfg["time"], cfg["output"]
    shape = (4, g.nx, g.ny, g.neta, g.ntau)
    arr = np.zeros(shape, np.float32)
    src_out = np.zeros(shape, np.float32) if src is not None else None
    diag = evolve.evolve_event(
        q, pi, Pi, g.tau_grid(), grid, eos, transport=transport, source=src,
        cfl=t["cfl"], hydro_dtau=t["hydro_dtau"], dtau_max=t["dtau_max"],
        out=arr, src_out=src_out, T_fo=o["T_fo"], freezeout=o["freezeout"],
        zero_tail=o["zero_after_freezeout"], stop_at_freezeout=o["stop_at_freezeout"])
    return arr, src_out, diag


def verify_replay(arr, reference_sha256=None, reference_arr=None, *, exact=True):
    """-> (ok, message). Bitwise on CPU/float64; tolerance-based otherwise."""
    import hashlib

    if reference_arr is not None:
        if exact:
            ok = np.array_equal(arr, reference_arr)
            d = float(np.abs(arr.astype(np.float64) - reference_arr.astype(np.float64)).max())
            return ok, ("bitwise identical to the reference" if ok
                        else f"differs from the reference, max |diff| = {d:.6g}")
        denom = max(1e-30, float(np.abs(reference_arr[0]).max()))
        rel = float(np.abs(arr[0] - reference_arr[0]).max()) / denom
        return rel < 1e-6, f"max relative difference in e = {rel:.3e}"

    if reference_sha256:
        got = hashlib.sha256(arr.tobytes()).hexdigest()
        ok = got == reference_sha256
        return ok, ("sha256 matches the dump" if ok
                    else f"sha256 {got[:16]} != recorded {reference_sha256[:16]}")
    return False, "nothing to compare against"


def replay_pair(path, cfg, e0, droplets, params, *, device=None, dtype=None,
                source_kw=None, meta=None, shower=None, overwrite=None):
    """Replay BOTH legs of a pair and write them as an FNO4d-schema file.

    This is what makes a controlled comparison possible. In a live run the shower responds to
    the medium it traverses, so two runs that differ in `transport.mode` also differ in the
    droplets Matter+LBT produce -- a real physical effect, but it means `visc - ideal` mixes
    the hydrodynamic response with a different jet. Replaying one fixed droplet set through
    both solvers separates them.

    Returns the path written.
    """
    import numpy as np

    from .grid import GridSpec
    from .h5_writer import PairedH5Writer

    g = GridSpec.from_cfg(cfg)
    bg_arr, _, bg_diag = replay_event(cfg, e0, None, params,
                                      device=device, dtype=dtype)
    jet_arr, jet_src, jet_diag = replay_event(cfg, e0, droplets, params, device=device,
                                              dtype=dtype, source_kw=source_kw)

    class _Leg:
        def __init__(self, arr, src, diag):
            self.arr, self.src, self.diag, self.g = arr, src, diag, g
            self.ic_sha256 = __import__("hashlib").sha256(
                np.ascontiguousarray(e0, dtype=np.float64).tobytes()).hexdigest()

    class _Bridge:
        def __init__(self):
            self.droplets = droplets
            self.params = params
            # The shower graph is not re-derived by a replay -- it is the ORIGINAL run's, and
            # that is the point: the same jet through a different solver. Carried through from
            # the npz so a replayed file stays animatable. None -> no shower/ group.
            self.shower = shower

    # None falls back to run.overwrite in the YAML, as a live run does; a driver with its
    # own --force/--overwrite flag passes it explicitly.
    with PairedH5Writer(path, cfg, 1, overwrite=overwrite) as w:
        w.append(0, _Leg(bg_arr, None, bg_diag), _Leg(jet_arr, jet_src, jet_diag), _Bridge())
        f = w._w.f
        f.attrs["provenance"] = "replayed droplets (fixed across legs)"
        for k, v in (meta or {}).items():
            f.attrs[k] = v
    return str(path)
