"""Assemble the two-stage FastHydro + Matter/LBT + liquefier pipeline.

    FastGlauberInitialState -> HardProcess -> NullPreDynamics
      -> FastHydro("bg")                      background, no source
      -> JetEnergyLossManager[JetEnergyLoss(Matter, LBT)] + CausalLiquefier
      -> DropletBridge                        droplets -> the jet leg's source
      -> FastHydro("jet")                     same IC, with the source

This mirrors `config/jetscape_user_twostagehydro.xml` and
`examples/custom_examples/TwoStagesHydro.cc`, with FastHydro in place of MUSIC.

Things that are not free choices
--------------------------------
* **NullPreDynamics is mandatory.**  `FluidDynamics::Init()` lets only the ids "MUSIC" and
  "Brick" run without pre-equilibrium and `exit(-1)`s otherwise (`FluidDynamics.cc:117-124`).
* **Matter before LBT.**  Matter sets the virtuality LBT takes over at Q0, and
  ``<Eloss><mutex>ON</mutex>`` arbitrates the handover.
* **Only the FIRST FluidDynamics is the framework's hydro.**  `JetScape::SetPointers()` stops
  at the first one, so Matter, LBT *and* the liquefier all query the background leg through
  `GetHydroCellSignal`, and the jet leg is invisible to signals.  That is exactly the
  two-stage semantics, and it is why the background leg must come first.
* **`hyd2.add_a_liquefier(liq)` is bookkeeping, not physics.**  FastHydro does not pull source
  terms through `FluidDynamics::get_source_term` the way MUSIC does -- it uses fast_data's own
  `CausalLiquefierSource` over the same droplets, via `DropletBridge`.  Attaching the
  liquefier anyway is what lets `FastHydro.Clear()` find it and empty the droplet list between
  events (nothing else in the framework does that once `Clear()` is overridden).  There is no
  double counting.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from .grid import GridSpec

__all__ = ["build_two_stage", "check_xml_agrees_with_cfg"]


def check_xml_agrees_with_cfg(user_xml: str, cfg) -> None:
    """Fail loudly when the user XML and the fast_data YAML describe different geometry.

    The grid and tau0 are the only things stated in both places -- `<IS><grid_*>` because the
    framework reads it, and `<Preequilibrium><taus>` because NullPreDynamics does.  Everything
    else about the solver lives in the YAML alone, which is deliberate: a new XML tag would
    have to be added to `config/jetscape_main.xml` or `JetScape::Init()` exits (-1).
    """
    g = GridSpec.from_cfg(cfg)
    root = ET.parse(user_xml).getroot()

    def val(path, cast=float):
        node = root.find(path)
        return None if node is None or not (node.text or "").strip() else cast(node.text.strip())

    problems = []
    for tag, want in (("IS/grid_max_x", g.is_ranges()[0]),
                      ("IS/grid_max_y", g.is_ranges()[1]),
                      ("IS/grid_max_z", g.is_ranges()[2]),
                      ("IS/grid_step_x", g.dx), ("IS/grid_step_y", g.dy),
                      ("IS/grid_step_z", g.deta)):
        got = val(tag)
        if got is not None and abs(got - want) > 1e-9:
            problems.append(f"  <{tag.replace('/', '><')}> = {got} but the YAML grid needs {want}")

    taus = val("Preequilibrium/taus")
    if taus is not None and abs(taus - g.tau0) > 1e-9:
        problems.append(f"  <Preequilibrium><taus> = {taus} but time.tau0 = {g.tau0}")

    if root.find("SoftParticlization") is not None:
        problems.append("  <SoftParticlization> is present, but FastHydro computes no "
                        "Cooper-Frye surface: iSS would sample an empty surface and produce "
                        "zero soft hadrons. Remove the block.")

    auto = root.find("enableAutomaticTaskListDetermination")
    if auto is None or "false" not in (auto.text or "").lower():
        problems.append("  <enableAutomaticTaskListDetermination> must be false for a "
                        "hand-wired pipeline")

    if problems:
        raise ValueError(f"{user_xml} disagrees with the fast_data config:\n" + "\n".join(problems))


def build_two_stage(cfg, *, user_xml=None, main_xml=None, ic=None, hard="PGun",
                    verbose=True, store=None, keep_bg_arr=True):
    """-> (modules, parts) ready for `jetscape.run_jetscape.run_manual`.

    `parts` is a dict of the individual objects (ini, hyd_bg, hyd_jet, liq, bridge, ...) so a
    driver can read results out after the run.
    """
    from jetscape.pyjetscape_core import (CausalLiquefier, JetEnergyLoss,
                                          JetEnergyLossManager, create_module, load_xml)

    from .hydro import FastHydro
    from .initial_state import FastGlauberInitialState
    from .liquefier_bridge import DropletBridge

    if user_xml:
        check_xml_agrees_with_cfg(user_xml, cfg)

    # CausalLiquefier's 0-argument constructor reads <Liquefier><CausalLiquefier> from the XML
    # singleton, and we are building modules before JetScape exists to open it.  Without this
    # the liquefier warns "XML User file not found" and silently uses its defaults.
    if main_xml or user_xml:
        load_xml(main_xml or "", user_xml or "")

    ini = ic or FastGlauberInitialState(cfg, verbose=verbose)
    preeq = create_module("NullPreDynamics")
    hyd_bg = FastHydro(cfg, stage=1, module_id="FastHydro_bg", ic=ini,
                       store=store, keep_arr=keep_bg_arr, verbose=verbose)

    # The 0-argument constructor reads <Liquefier><CausalLiquefier> from the loaded XML.
    liq = CausalLiquefier()

    jloss = JetEnergyLoss()
    jloss.Add(create_module("Matter"))     # Matter first: it sets the virtuality
    jloss.Add(create_module("Lbt"))
    jloss.add_a_liquefier(liq)
    jmgr = JetEnergyLossManager()
    jmgr.Add(jloss)

    hyd_jet = FastHydro(cfg, stage=2, module_id="FastHydro_jet", ic=ini,
                        store=store, verbose=verbose)
    hyd_jet.add_a_liquefier(liq)           # bookkeeping; see the module docstring
    bridge = DropletBridge(liq, hyd_jet, cfg, verbose=verbose)

    # PGun samples the hard-scattering vertex and then overwrites it with zeros
    # (src/initialstate/PGun.cc:117-120), so every shower starts at the fireball centre
    # regardless of fasthydro.hard_vertex.mode. PythiaGun uses it (PythiaGun.cc:293-297).
    hv_mode = ((cfg.get("fasthydro") or {}).get("hard_vertex") or {}).get("mode", "ncoll")
    if hard == "PGun" and hv_mode != "centre":
        print(f"[build_two_stage] WARNING: hard=PGun ignores the sampled vertex "
              f"(PGun.cc:117-120 zeroes it), so fasthydro.hard_vertex.mode={hv_mode!r} will "
              f"have no effect and every shower will start at (0,0,0). Use hard='PythiaGun', "
              f"or set mode: centre to say so deliberately.", flush=True)

    modules = [ini]
    if hard:
        modules.append(create_module(hard))
    modules += [preeq, hyd_bg, jmgr, bridge, hyd_jet]

    parts = dict(ini=ini, preeq=preeq, hyd_bg=hyd_bg, liq=liq, jloss=jloss,
                 jmgr=jmgr, bridge=bridge, hyd_jet=hyd_jet)
    return modules, parts
