"""Soft particlization of one FastHydro leg with the framework's iSS (and SMASH).

FastHydro needs no surface code of its own.  It fills `bulk_info`, and when iSS finds that the
hydro handed over no surface, it asks the hydro to build one from that stored evolution
(`SoftParticlization::FindHydroHyperSurface` -> `FluidDynamics::FindSurfaceFromEvolution`, which
runs the Cornelius-based `SurfaceFinder`).  All of that is C++ in X-SCAPE core.

What this module does is make sure the evolution handed to it *can* give a meaningful surface,
and say which leg is sampled:

* ``<SoftParticlization><hydro_id>`` picks the leg -- ``FastHydro_jet`` or ``FastHydro_bg``.
  Only the soft-particlization signals follow it; Matter, LBT and the liquefier keep querying
  the background leg.  One run particlizes one leg.  The jet-induced signal is
  (jet run) - (background run) on the same initial condition, averaged over iSS oversamples.
* The surface must close inside the stored evolution.  A zeroed tail
  (``output.zero_after_freezeout``) would put a fake surface at the zeroing time, and a fireball
  that is still above T_sw at the last frame or on a transverse boundary would leave it open.
  The eta edges are open by construction (finite eta grid); analyse at mid-rapidity.
* The fluid's EoS has to be the hadron gas iSS samples, or Cooper-Frye does not conserve
  energy at the switch: ``eos.kind: hotqcd_smash`` with the SMASH particle list.

pi^{mu nu} and Pi are stored as zero (cells.py), so the Cooper-Frye is ideal whatever
``transport.mode`` says; the delta-f switches in ``<iSS>`` have nothing to act on.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

__all__ = ["LEGS", "closure_report", "config_problems", "read_xml", "write_iss_music_input"]

#: module ids of the two FastHydro legs, as build_two_stage names them
LEGS = ("FastHydro_bg", "FastHydro_jet")

#: jetscape_main.xml default for <SoftParticlization><T_sw>
DEFAULT_T_SW = 0.15


def read_xml(user_xml):
    """-> dict describing the particlization the user XML asks for, or None if it asks none.

    Keys: hydro_id (None = the first hydro), T_sw, surface_{dtau,dx,deta}, n_oversample,
    afterburner (bool).
    """
    root = ET.parse(user_xml).getroot() if isinstance(user_xml, str) else user_xml
    sp = root.find("SoftParticlization")
    if sp is None:
        return None

    def text(node, tag):
        n = node.find(tag) if node is not None else None
        t = (n.text or "").strip() if n is not None else ""
        return t or None

    hydro_id = text(sp, "hydro_id")
    return {
        "hydro_id": None if hydro_id in (None, "first") else hydro_id,
        "T_sw": float(text(sp, "T_sw") or DEFAULT_T_SW),
        "surface_dtau": float(text(sp, "surface_dtau") or 0.0),
        "surface_dx": float(text(sp, "surface_dx") or 0.0),
        "surface_deta": float(text(sp, "surface_deta") or 0.0),
        "n_oversample": int(text(sp.find("iSS"), "number_of_repeated_sampling") or 1),
        "iss_working_path": text(sp.find("iSS"), "iSS_working_path") or ".",
        "afterburner": root.find("Afterburner") is not None,
    }


#: What iSS reads from <iSS_working_path>/music_input (external_packages/iSS/src/readindata.cpp):
#: the EoS id picks the hadron list (91 = SMASH, no partial chemical equilibrium), and the
#: flags say which viscous/charge fields a surface cell carries.  FastHydro stores none.
_MUSIC_INPUT = """\
# Written by fasthydro.particlization for iSS; FastHydro does not use MUSIC.
EOS_to_use 91
Include_Bulk_Visc_Yes_1_No_0 0
Include_Rhob_Yes_1_No_0 0
turn_on_baryon_diffusion 0
freeze_surface_in_binary 1
output_vorticity 0
EndOfData
"""


def write_iss_music_input(working_path):
    """Create the ``music_input`` iSS insists on reading, in iSS's working directory.

    Without it iSS exits; and the wrapper's fallback symlinks <Hydro><MUSIC><MUSIC_input_file>,
    which for a FastHydro run points at nothing sensible.  An existing file is kept only if it
    says ``EOS_to_use 91`` -- anything else would sample the wrong hadron list.
    """
    import os
    import re

    os.makedirs(working_path, exist_ok=True)
    path = os.path.join(working_path, "music_input")
    if os.path.lexists(path):
        try:
            with open(path) as f:
                eos = re.search(r"^\s*EOS_to_use\s+(\d+)", f.read(), re.M)
        except OSError:
            eos = None
        if eos and eos.group(1) == "91":
            return path
        raise ValueError(f"{path} exists but does not say EOS_to_use 91, so iSS would sample "
                         f"the wrong hadron list for FastHydro's hotqcd_smash EoS. Point "
                         f"<SoftParticlization><iSS><iSS_working_path> at a fresh directory.")
    with open(path, "w") as f:
        f.write(_MUSIC_INPUT)
    return path


def config_problems(cfg, info):
    """Settings that would make the sampled surface wrong.  -> list of messages."""
    out, eos = cfg["output"], cfg["eos"]
    problems = []
    if out["zero_after_freezeout"] and out["freezeout"] != "never":
        problems.append(
            "  output.zero_after_freezeout is true: the zeroed frames after freeze-out would "
            "put a fake surface at the zeroing time. Set it to false.")
    if out["stop_at_freezeout"] and out["freezeout"] != "never":
        problems.append(
            "  output.stop_at_freezeout is true: the frames after the stop are zero, which "
            "fakes a surface there. Set it to false.")
    if eos["kind"] != "hotqcd_smash":
        problems.append(
            f"  eos.kind = {eos['kind']!r}: iSS samples the SMASH hadron gas, so the fluid "
            f"must end on the matching EoS or Cooper-Frye does not conserve energy at "
            f"T_sw = {info['T_sw']} GeV. Use eos.kind: hotqcd_smash.")
    return problems


def closure_report(arr, eos, T_sw):
    """Is the T = T_sw surface closed inside the stored evolution?

    ``arr`` is the solver buffer (4, nx, ny, neta, ntau) = [e, vx, vy, vz].  Returns the max
    temperature at the last frame, on the four transverse faces (any frame), and on the two
    eta faces (any frame).  The first two must be below T_sw; the eta faces are open by
    construction and only reported.
    """
    import torch

    e = arr[0]

    def T_of_max(block):
        e_max = float(np.max(block)) if block.size else 0.0
        t = eos.T(torch.as_tensor(max(e_max, 0.0), device=getattr(eos, "device", None),
                                  dtype=getattr(eos, "dtype", torch.float64)))
        return float(t)

    T_last = T_of_max(e[..., -1])
    T_side = max(T_of_max(e[0]), T_of_max(e[-1]), T_of_max(e[:, 0]), T_of_max(e[:, -1]))
    T_eta = max(T_of_max(e[:, :, 0]), T_of_max(e[:, :, -1]))
    return {
        "T_sw": float(T_sw),
        "T_max_last_frame": T_last,
        "T_max_transverse_boundary": T_side,
        "T_max_eta_boundary": T_eta,
        "closed": T_last < T_sw and T_side < T_sw,
    }
