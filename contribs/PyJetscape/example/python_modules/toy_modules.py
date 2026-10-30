"""
example/python_modules/toy_modules.py

One small JETSCAPE module per stage, written in Python: templates for your own. Each
subclasses the C++ base class and overrides the method JETSCAPE calls, with the C++ name
and argument order (see the C++ module next to it for the full interface):

    ToyGun          HardProcess          ExecuteTask()             (C++: PGun)
    ToySplit        JetEnergyLoss        DoEnergyLoss(...)         (C++: Matter)
    ToyHadronizer   Hadronization        DoHadronization(...)      (C++: ColorlessHadronization)
    ToySampler      SoftParticlization   ExecuteTask()             (C++: iSS)
    ToyAfterburner  Afterburner          ExecuteTask(), GetCurrentHadronList()  (C++: SMASH)
    ToyWriter       JetScapeWriter       Write(obj), WriteEvent()  (C++: JetScapeWriterAscii)

register_toy_modules() puts them into JETSCAPE's module factory under CustomModule names,
so the XML can name them (run_from_xml.py); run_by_hand.py builds the same pipeline with
the objects directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from jetscape import pyjetscape_core as js


class ToyGun(js.HardProcess):
    """Two back-to-back gluons of E (GeV) along x at the hard vertex of the initial state
    (the origin without one); a pseudo cross section for the event header."""

    E = 40.0

    def InitTask(self):
        # parameters from the XML, as a C++ module reads them: the user XML's value, else
        # the default register_module() put into the main XML
        e = self.get_xml_element_double(["Hard", "CustomModuleToyGun", "E"], False)
        self.e = e if e > 0 else self.E

    def ExecuteTask(self):
        x = js.FourVector()
        ini = js.JetScapeSignalManager.Instance().GetInitialStatePointer()
        if ini is not None and ini.GetNcoll() > 0:
            t, vx, vy, vz = ini.sample_binary_collision_point()
            x = js.FourVector(vx, vy, vz, t)
        for sign in (+1, -1):
            self.AddParton(js.Parton(0, 21, 0, js.FourVector(sign * self.e, 0.0, 0.0, self.e),
                                     x))

    def GetSigmaGen(self):
        return 1.0


class ToySplit(js.JetEnergyLoss):
    """Splits every parton above 30 GeV into two halves once it is at t >= 1 fm/c,
    with +-1 GeV of relative momentum along y. Not physics: it shows the interface."""

    def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut):
        p = pIn[0]                  # the C++ parton of this step (change it in place)
        if time >= 1.0 and p.e() > 30.0:
            for s in (+1, -1):
                pOut.append(js.Parton(0, p.pid(), 0,
                                      js.FourVector(0.5 * p.px(), s * 1.0, 0.5 * p.pz(),
                                                    0.5 * p.e()), p.x_in()))
        # pOut left empty: the parton goes on unchanged (as a C++ module)


class ToyHadronizer(js.Hadronization):
    """Every final parton becomes a pi+ with its momentum and position."""

    def DoHadronization(self, shower, hOut, pOut):
        for partons in shower:
            for p in partons:
                hOut.append(js.Hadron(0, 211, 1, p.p_in(), p.x_in()))


class ToySampler(js.SoftParticlization):
    """Three soft pi0 per event (one oversample), in place of sampling the surface
    (self.GetHydroHyperSurface() gives it when there is a hydro)."""

    def ExecuteTask(self):
        soft = [js.Hadron(0, 111, 0, js.FourVector(0.3 * k, 0.0, 0.0, 0.5), js.FourVector())
                for k in (-1, 0, 1)]
        self.set_hadron_list([soft])


class ToyAfterburner(js.Afterburner):
    """Takes the soft and the jet hadrons and hands them on unchanged."""

    def ExecuteTask(self):
        self.hadrons = [h for sample in self.GatherAfterburnerHadrons() for h in sample]

    def GetCurrentHadronList(self):
        return self.hadrons


class ToyWriter(js.JetScapeWriter):
    """Counts what the tasks write per event (every Parton, Vertex, PartonShower, Hadron,
    str) and keeps the event header's cross section."""

    def __init__(self):
        super().__init__()
        self.events = []
        self.current = {}

    def GetStatus(self):
        return True

    def Write(self, obj):
        kind = type(obj).__name__
        self.current[kind] = self.current.get(kind, 0) + 1

    def WriteEvent(self):
        self.current["sigma_gen"] = self.GetHeader().GetSigmaGen()
        self.events.append(self.current)
        self.current = {}


# name: (class, main-XML section of its default entry, the defaults of its parameters)
TOY_MODULES = {
    "CustomModuleToyGun": (ToyGun, "Hard", "<E>40</E>"),
    "CustomModuleToySplit": (ToySplit, "Eloss", ""),
    "CustomModuleToyHadronizer": (ToyHadronizer, "", ""),   # <JetHadronization><name>
    "CustomModuleToySampler": (ToySampler, "SoftParticlization", ""),
    "CustomModuleToyAfterburner": (ToyAfterburner, "Afterburner", ""),
}


def register_toy_modules():
    for name, (cls, section, defaults) in TOY_MODULES.items():
        js.register_module(name, cls, section=section, defaults=defaults)
