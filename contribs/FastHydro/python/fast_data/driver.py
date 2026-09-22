"""The generator: config in, an FNO4d-schema HDF5 file out.

    MC-Glauber initial state  ->  structure-preserving Milne FV hydro  (+ optional jet source)
                              ->  arr (e, vx, vy, vz)  +  source/S

One event at a time, streaming into the writer, so peak memory is two float32 frame buffers plus
the solver state regardless of how many events are asked for.
"""

from __future__ import annotations

import os
import time

import numpy as np

from . import convert, evolve, glauber, partons, writer
from .eos import eos_descriptor, resolve_eos, write_eos_group
from .liquefier import CausalLiquefierSource, LiquefierParams

__all__ = ["run", "build_grid", "make_initial_state"]

_DTYPES = {"float32": "float32", "float64": "float64"}


def build_grid(cfg, device=None, dtype=None):
    import torch

    from . import fv as _fv
    g = cfg["grid"]
    dev = torch.device(device or cfg["run"]["device"])
    dt = getattr(torch, dtype or cfg["run"]["dtype"])
    return _fv.Grid(int(g["nx"]), int(g["ny"]), int(g["neta"]),
                    float(g["dx"]), float(g["dy"]), float(g["deta"]), device=dev, dtype=dt)


def make_initial_state(cfg, np_eos, seed, grid, ic_cache=None, index=0):
    """-> (e (nx,ny,neta) float64 GeV/fm^3, meta dict).  Honours the IC cache if configured."""
    ini = cfg["initial_state"]
    g = cfg["grid"]
    shape = (int(g["nx"]), int(g["ny"]), int(g["neta"]))
    spacing = (float(g["dx"]), float(g["dy"]), float(g["deta"]))

    if ini["kind"] == "from_file" or ic_cache is not None:
        path = ini.get("file") or ic_cache
        if path and os.path.exists(path):
            e0, meta = glauber.load_events(path, slice(index, index + 1))
            if tuple(e0.shape[1:]) != shape:
                raise ValueError(f"IC file grid {tuple(e0.shape[1:])} != config grid {shape}")
            m = {k: (v[0] if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == 1 else v)
                 for k, v in meta.items() if k in ("b", "npart", "ncoll", "e_max", "T_max")}
            return e0[0].astype(np.float64), m

    if ini["kind"] == "smooth":
        from . import fv as _fv
        s = ini["smooth"]
        e = _fv.smooth_initial_energy(grid, s["e0"], s["R"], a2=s["a2"], psi2=s["psi2"],
                                      a3=s["a3"], psi3=s["psi3"], eta_flat=s["eta_flat"],
                                      sigma_eta=s["sigma_eta"])
        e = e[0].detach().cpu().numpy().astype(np.float64)
        return e, {"b": np.nan, "npart": 0, "ncoll": 0, "e_max": float(e.max())}

    prof = {k: v for k, v in ini["profile"].items()}
    e, meta = glauber.make_event(
        ini["proj"], ini["targ"], b=ini["b"], b_max=ini["b_max"],
        sigma_nn_mb=ini["sigma_nn_mb"], grid=shape, spacing=spacing, seed=int(seed),
        require_collision=ini["require_collision"], eos=np_eos,
        K=cfg["_K"], quantity=ini["quantity"], e_floor=ini["e_floor"], **prof)
    return e.astype(np.float64), meta


def _calibrate_K(cfg, np_eos, log=None):
    """Resolve the one normalisation the config gave into a K, once for the whole run."""
    ini = cfg["initial_state"]
    if ini.get("K") is not None:
        return float(ini["K"])
    if ini["kind"] != "mc_glauber":
        return 1.0
    kind, target = next((k, ini[k]) for k in ("target_T", "target_e", "target_s")
                        if ini.get(k) is not None)
    n = ini["calib_events"]
    if n is None:
        n = 300 if (ini["proj"] in ("d", "p") or ini["targ"] in ("d", "p")) else 20
    prof = dict(ini["profile"])
    if log:
        log(f"calibrating K for {kind}={target} over {n} events at b={ini['calib_b']} ...")
    K = glauber.calibrate_K(target, kind=kind.replace("target_", ""), quantity=ini["quantity"],
                            eos=np_eos, proj=ini["proj"], targ=ini["targ"], b=ini["calib_b"],
                            sigma_nn_mb=ini["sigma_nn_mb"], n_events=int(n),
                            seed=int(cfg["run"]["seed"]) + 1, **prof)
    if log:
        unit = "GeV/fm" if ini["quantity"] == "energy" else "1/fm"
        log(f"  K = {K:.4f} {unit}")
    return float(K)


def run(cfg, *, log=print, shard=None, resume=False, dry_run=False):
    """Generate the dataset described by `cfg`.  Returns the output path (None for a dry run)."""
    import torch

    from . import config as _cfgmod
    from . import fv as _fv

    run_cfg, out_cfg, src_cfg = cfg["run"], cfg["output"], cfg["source"]
    tau_grid = _cfgmod.derived_tau_grid(cfg)
    T = len(tau_grid)
    nevents = int(run_cfg["nevents"])

    grid = build_grid(cfg)
    if run_cfg["compile"] and not dry_run:
        _fv.enable_compiled_recovery(True)
        log("compile: primitive_recovery via torch.compile(dynamic=True) -- the first call pays "
            "~16 s, and output is NOT bit-identical to an eager run (1.5e-06 relative above "
            "T_fo, more in the dilute tail; see fv.enable_compiled_recovery)")
    np_eos, fv_eos = resolve_eos(cfg["eos"], device=grid.device, dtype=grid.dtype)
    log(f"eos: {eos_descriptor(np_eos)}")

    transport = None
    if cfg["transport"]["mode"] == "israel_stewart":
        t = cfg["transport"]
        transport = _fv.Transport(eta_over_s=t["eta_over_s"], zeta_over_s=t["zeta_over_s"],
                                  tau_pi_coeff=t["tau_pi_coeff"], delta_pipi=t["delta_pipi"],
                                  delta_PiPi=t["delta_PiPi"], tau_min=t["tau_min"],
                                  pi_rho_max=t["pi_rho_max"], pi_e_min=t["pi_e_min"],
                                  pi_advection=t["pi_advection"])
        log("NOTE: transport.mode=israel_stewart -- shear is validated against the Marrochio et al. "
            "Gubser solution (fv.test_gubser_viscous: e, u and pi converge together at orders "
            "1.3-1.6); the second-order couplings tau_pipi, phi_7, lambda_piPi, lambda_Pipi are "
            "omitted, and bulk (zeta_over_s > 0) is untested -- see README_FastData.md")

    params = LiquefierParams.from_config(src_cfg["params"]) if src_cfg["enabled"] else None
    ic_seeds, src_seeds = partons.seed_streams(run_cfg["seed"], nevents)

    lo, hi = 0, nevents
    if shard is not None:
        i, n = shard
        lo = (i * nevents) // n
        hi = ((i + 1) * nevents) // n
        log(f"shard {i}/{n}: events [{lo}, {hi})")

    cfg["_K"] = _calibrate_K(cfg, np_eos, log=log) if not dry_run else 1.0

    attrs = writer.grid_attrs(grid.nx, grid.ny, grid.neta, grid.dx, grid.dy, grid.deta,
                              tau_grid[0], cfg["time"]["record_dtau"], T)
    extra = _provenance_attrs(cfg, np_eos, params)
    extra.update(out_cfg.get("extra_attrs") or {})

    if dry_run:
        _print_dry_run(cfg, grid, tau_grid, nevents, log)
        return None

    out_path = run_cfg["out"]
    if shard is not None:
        stem, ext = os.path.splitext(out_path)
        out_path = f"{stem}.shard{shard[0]:02d}{ext}"

    n_out = hi - lo
    w = writer.FnoH5Writer(out_path, attrs, n_out, compression=out_cfg["compression"],
                           source_compression=out_cfg["source_compression"],
                           chunk_events=out_cfg["chunk_events"],
                           write_source=bool(src_cfg["enabled"] and out_cfg["write_source"]),
                           write_diagnostics=out_cfg["write_diagnostics"],
                           extra_attrs=extra, np_eos=np_eos if cfg["eos"]["store_table"] else None,
                           force=run_cfg["overwrite"], resume=resume)

    arr_buf = np.zeros((4, grid.nx, grid.ny, grid.neta, T), dtype=np.float32)
    src_buf = np.zeros_like(arr_buf) if w.write_source else None
    t_start = time.perf_counter()
    fo_seen, cached_tau_fo = [], None
    try:
        for j in range(w.start, n_out):
            ev = lo + j
            e0, meta = make_initial_state(cfg, np_eos, int(ic_seeds[ev]), grid,
                                          ic_cache=cfg["initial_state"].get("cache"), index=ev)
            e_t = torch.as_tensor(e0, dtype=grid.dtype, device=grid.device).unsqueeze(0)
            q = _fv.initial_state_from_energy(e_t, tau_grid[0], fv_eos)
            pi = grid.zeros(1, 10) if transport is not None else None
            Pi = grid.zeros(1, 1) if transport is not None else None

            source, drops = None, None
            if src_cfg["enabled"]:
                rng = np.random.default_rng(int(src_seeds[ev]))
                drops = partons.build_droplets(
                    src_cfg, rng, params, tau0=tau_grid[0], tau_fo=cached_tau_fo,
                    e_initial=e0, grid=grid, log=log)
                source = CausalLiquefierSource(
                    drops, params, mode=src_cfg["mode"], renorm=src_cfg["renorm"],
                    tau_eval_mode=src_cfg["tau_eval_mode"], n_sub=src_cfg["n_sub"],
                    n_sub_max=src_cfg["n_sub_max"], min_in_grid=src_cfg["min_in_grid"],
                    on_out_of_grid=src_cfg["on_out_of_grid"],
                    device=grid.device, dtype=grid.dtype)

            arr_buf[:] = 0.0
            if src_buf is not None:
                src_buf[:] = 0.0
            d = evolve.evolve_event(q, pi, Pi, tau_grid, grid, fv_eos, transport=transport,
                                    source=source, cfl=cfg["time"]["cfl"],
                                    hydro_dtau=cfg["time"]["hydro_dtau"],
                                    dtau_max=cfg["time"]["dtau_max"],
                                    out=arr_buf, src_out=src_buf, T_fo=out_cfg["T_fo"],
                                    freezeout=out_cfg["freezeout"],
                                    zero_tail=out_cfg["zero_after_freezeout"],
                                    stop_at_freezeout=out_cfg["stop_at_freezeout"])
            cached_tau_fo = d["tau_freezeout"]
            fo_seen.append(d["ntau_freezeout"])

            diag = {"b": meta.get("b", np.nan), "npart": meta.get("npart", 0),
                    "ncoll": meta.get("ncoll", 0), "e_max": meta.get("e_max", np.nan),
                    "T_max0": float(d["T_max"][0]), "n_steps": d["n_steps"],
                    "v_max": d["v_max"], "wall_s": d["wall_s"],
                    "frozen_out": d["frozen_out"], "seed": np.uint64(ic_seeds[ev]),
                    "source_seed": np.uint64(src_seeds[ev])}
            if source is not None:
                diag.update({f"src_{k}": v for k, v in source.report().items()
                             if np.isscalar(v)})

            w.append_event(j, arr_buf, d["ntau_freezeout"], d["tau_freezeout"],
                           S_ev=src_buf, P_cart=d["P_cart"],
                           droplets=(drops.data if drops is not None else None), diag=diag)

            if (j - w.start) % max(1, run_cfg["progress_every"]) == 0:
                _log_event(log, ev, meta, d, source)
    finally:
        w.close()

    el = time.perf_counter() - t_start
    log(f"wrote {out_path}  ({n_out - w.start} events, {el:.1f} s, "
        f"{el / max(n_out - w.start, 1):.2f} s/event)")
    if fo_seen:
        log(f"ntau_freezeout over the run: min {min(fo_seen)}, median "
            f"{int(np.median(fo_seen))}, max {max(fo_seen)} of {T} frames")
        n_never = sum(1 for n in fo_seen if n >= T)
        if n_never:
            log(f"  WARNING: {n_never}/{len(fo_seen)} events never reached T_fo="
                f"{out_cfg['T_fo']} GeV; nothing was zeroed for them. Raise time.tau_end.")
    return out_path


def _log_event(log, ev, meta, d, source):
    b = meta.get("b", float("nan"))
    line = (f"[{ev:4d}] b={b:5.2f} Npart={meta.get('npart', 0):4d} "
            f"Ncoll={meta.get('ncoll', 0):5d} e_max={meta.get('e_max', float('nan')):7.2f} "
            f"| steps={d['n_steps']:4d} tau_fo={d['tau_freezeout']:5.2f} "
            f"(ntau_fo={d['ntau_freezeout']}) v_max={d['v_max']:.3f} {d['wall_s']:5.1f}s")
    if d.get("frames_skipped"):
        line += f" [+{d['frames_skipped']} frames skipped past freeze-out]"
    if source is not None:
        r = source.report()
        line += (f" | {r['n_fired']}/{r['n_droplets']} droplets, "
                 f"E_dep={r['E_injected']:.2f} GeV, in_grid>={r['in_grid_min']:.3f}")
        fl = source.flag_summary()
        if fl:
            line += " [" + ",".join(f"{k}:{v}" for k, v in sorted(fl.items())) + "]"
    log(line)


def _provenance_attrs(cfg, np_eos, params):
    import json
    a = {
        "generator": "fast_data",
        "solver": "structure_preserving_hydro_fv",
        "eos_kind": cfg["eos"]["kind"],
        "eos_desc": eos_descriptor(np_eos),
        "transport_mode": cfg["transport"]["mode"],
        "master_seed": int(cfg["run"]["seed"]),
        "cfl": float(cfg["time"]["cfl"]),
        "T_fo": float(cfg["output"]["T_fo"]),
        "freezeout_rule": cfg["output"]["freezeout"],
        "solver_dtype": cfg["run"]["dtype"],
        "device": cfg["run"]["device"],
        "initial_state_kind": cfg["initial_state"]["kind"],
        "source_model": cfg["source"]["model"] if cfg["source"]["enabled"] else "none",
        "config_json": json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")},
                                  default=str, sort_keys=True),
    }
    if cfg["initial_state"]["kind"] == "mc_glauber":
        a.update(proj=cfg["initial_state"]["proj"], targ=cfg["initial_state"]["targ"],
                 sigma_nn_mb=float(cfg["initial_state"]["sigma_nn_mb"]),
                 quantity=cfg["initial_state"]["quantity"], K=float(cfg.get("_K", np.nan)))
    if params is not None:
        a.update(params.to_attrs())
        a.update(source_mode=cfg["source"]["mode"], source_renorm=cfg["source"]["renorm"],
                 source_tau_eval_mode=cfg["source"]["tau_eval_mode"])
    return a


def _print_dry_run(cfg, grid, tau_grid, nevents, log):
    g, t = cfg["grid"], cfg["time"]
    T = len(tau_grid)
    gx = float(grid.x[0]); ge = float(grid.eta[0])
    nbytes = nevents * 4 * grid.nx * grid.ny * grid.neta * T * 4
    log(f"grid        {grid.nx} x {grid.ny} x {grid.neta}   "
        f"x in [{gx:.3f}, {-gx:.3f}]  eta in [{ge:.3f}, {-ge:.3f}]")
    log(f"tau axis    {T} frames  tau = {tau_grid[0]:.3f} .. {tau_grid[-1]:.3f}  "
        f"step {t['record_dtau']}")
    step = t["hydro_dtau"] or grid.max_dtau(tau_grid[0], t["cfl"])
    log(f"solver      {'fixed' if t['hydro_dtau'] else 'adaptive CFL'} dtau ~ {step:.4f} fm "
        f"-> ~{int((tau_grid[-1] - tau_grid[0]) / step)} steps/event")
    log(f"arr         {nevents} x 4 x {grid.nx} x {grid.ny} x {grid.neta} x {T} float32 "
        f"= {nbytes / 2**30:.2f} GiB uncompressed")
    if cfg["source"]["enabled"] and cfg["output"]["write_source"]:
        log(f"source/S    same shape (mostly zeros; gzip+shuffle)")
    log(f"peak RAM    ~{2 * 4 * grid.nx * grid.ny * grid.neta * T * 4 / 2**30:.2f} GiB of frame "
        f"buffers + solver state")
    log("DRY RUN -- nothing written.")
