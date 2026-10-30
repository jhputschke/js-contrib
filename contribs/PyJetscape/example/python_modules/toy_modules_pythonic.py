"""
example/python_modules/toy_modules_pythonic.py

The toys of toy_modules.py written with the Pythonic layer
(docs/Plans/PLAN_pyjetscape_pythonic.md): the same physics, so run_by_hand_pythonic.py
gives the same SUMMARY with either set.

    @js.module(name, section)   registers the class; its xml_parameters become the main-XML
                                default and are read into self.<key> before InitTask
    return [...]                instead of appending to pOut / hOut
    js.make_parton/make_hadron  particles by keywords: momentum (E, px, py, pz), position
                                (t, x, y, z)
    p.momentum, p.position      FourVectors that unpack in those orders

ToyAfterburner and ToyWriter have nothing the layer changes; they are reused as they are.
Importing this file registers the modules (CustomModule...V2), as the decorator does.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import jetscape as js  # noqa: E402
from toy_modules import ToyAfterburner, ToyWriter  # noqa: E402,F401


@js.module("CustomModuleToyGunV2", section="Hard")
class ToyGunV2(js.pyjetscape_core.HardProcess):
    """ToyGun: two back-to-back gluons of E (GeV) along x at the initial state's hard
    vertex (the origin without one)."""

    xml_parameters = {"E": 40.0}

    def ExecuteTask(self):
        position = (0.0, 0.0, 0.0, 0.0)
        ini = js.JetScapeSignalManager.Instance().GetInitialStatePointer()
        if ini is not None and ini.GetNcoll() > 0:
            position = ini.sample_binary_collision_point()        # (t, x, y, z)
        for sign in (+1, -1):
            self.AddParton(js.make_parton(21, (self.E, sign * self.E, 0.0, 0.0), position))

    def GetSigmaGen(self):
        return 1.0


@js.module("CustomModuleToySplitV2", section="Eloss")
class ToySplitV2(js.pyjetscape_core.JetEnergyLoss):
    """ToySplit: a parton above e_min splits into two halves at t >= t_split, with
    +-1 GeV along y.  Returns the two partons (None: the parton goes on unchanged)."""

    xml_parameters = {"t_split": 1.0, "e_min": 30.0}

    def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut):
        p = pIn[0]
        if time >= self.t_split and p.e() > self.e_min:
            E, px, py, pz = p.momentum
            return [js.make_parton(p.pid(), (E / 2, px / 2, s * 1.0, pz / 2),
                                   position=p.position) for s in (+1, -1)]


@js.module("CustomModuleToyHadronizerV2")
class ToyHadronizerV2(js.pyjetscape_core.Hadronization):
    """ToyHadronizer: every final parton becomes a pi+ with its momentum and position."""

    def DoHadronization(self, shower, hOut, pOut):
        return [js.make_hadron(211, p.momentum, p.position, stat=1)
                for partons in shower for p in partons]


@js.module("CustomModuleToySamplerV2", section="SoftParticlization")
class ToySamplerV2(js.pyjetscape_core.SoftParticlization):
    """ToySampler: three soft pi0 per event (one oversample)."""

    def ExecuteTask(self):
        self.set_hadron_list([[js.make_hadron(111, (0.5, 0.3 * k, 0.0, 0.0))
                               for k in (-1, 0, 1)]])
