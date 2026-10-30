"""
tests/equivalence_driver.py -- one run for test_bindings_equivalence.py (validation V3 of
docs/Plans/PLAN_pyjetscape_complete.md): a 100 GeV gluon (PGun), Matter in vacuum,
ColorlessHadronization, the C++ final-state hadron writer, and a Python writer in the
same format. MODE swaps one C++ module for a Python one:

    cpp       PGun and Matter in C++ (the reference)
    pygun     PyPGun: PGun ported to Python
    pymatter  PyMatter: a Python energy-loss module that hands every step to a C++
              Matter of its own
    pymatter_unaligned
              the same without giving the inner Matter its task number: its random
              engine differs, so the hadrons must differ (the control of the test)

The physics draws its random numbers from one engine per task, seeded by the task's
number, so every mode makes the same tasks in the same order: PyMatter gives its inner
Matter its own task number (next_task_number / set_next_task_number), as a C++ run has one
task there. The hadron files of the three modes must then be the same, byte for byte.

    python equivalence_driver.py MODE MAIN_XML OUTDIR [EVENTS]
"""

import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from jetscape import pyjetscape_core as js


class PyPGun(js.HardProcess):
    """C++ PGun (src/initialstate/PGun.cc) in Python: one parton of fixed pT at y = 0,
    phi = 0, at the origin, with PGun's colour tags. Light partons only (PGun takes the
    mass from Pythia's particle data; 0 here)."""

    def InitTask(self):
        self.pT = self.get_xml_element_double(["Hard", "CustomModulePyPGun", "pT"])
        self.parID = int(self.get_xml_element_double(["Hard", "CustomModulePyPGun", "parID"]))

    def ExecuteTask(self):
        pT, mass, rapidity, phi = self.pT, 0.0, 0.0, 0.0
        pz = math.sqrt(pT * pT + mass * mass) * math.sinh(rapidity)
        e = math.sqrt(pT * pT + mass * mass) * math.cosh(rapidity)
        p_abs = math.sqrt(pT * pT + pz * pz)
        eta = 0.5 * math.log((p_abs + pz) / (p_abs - pz))
        p = js.Parton(0, self.parID, 0, pT, eta, phi, e, [0.0, 0.0, 0.0, 0.0])
        p.set_color(100 if self.parID > 0 else 0)
        p.set_anti_color(0 if (self.parID > 0 or self.parID == 21) else 101)
        p.set_max_color(102)
        self.AddParton(p)


class PyMatter(js.JetEnergyLoss):
    """Every step handed to a C++ Matter of its own; the per-shower copy (Clone) copies
    that Matter too."""

    def __init__(self):
        n = js.next_task_number()
        super().__init__()                         # this task: number n ...
        js.set_next_task_number(n)
        self.matter = js.create_module("Matter")   # ... and the inner Matter gets n too

    def InitTask(self):
        self.matter.Init()                         # Matter reads <Eloss><Matter>

    def Clone(self):
        n = js.next_task_number()
        inner = self.matter.Clone()                # as a C++ run's per-shower Matter
        js.set_next_task_number(n)
        copy = js.JetEnergyLoss.Clone(self)        # the default copy (C++ copy constructor)
        copy.matter = inner
        return copy

    calls = 0

    def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut):
        PyMatter.calls += 1
        self.matter.DoEnergyLoss(deltaT, time, Q2, pIn, pOut)


class PyMatterUnaligned(PyMatter):
    """PyMatter whose inner Matter takes the next task number, as any new task would:
    a task more than a C++ run, so other random engines than the reference's."""

    def __init__(self):
        js.JetEnergyLoss.__init__(self)
        self.matter = js.create_module("Matter")

    def Clone(self):
        inner = self.matter.Clone()
        copy = js.JetEnergyLoss.Clone(self)
        copy.matter = inner
        return copy


class FinalHadronsWriter(js.JetScapeWriter):
    """The particle lines of JetScapeWriterFinalStateHadronsAscii, in Python."""

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.hadrons = []
        self.out = open(path, "w")  # noqa: SIM115 (open for the run, closed in FinishTask)

    def GetStatus(self):
        return True

    def Write(self, obj):
        if isinstance(obj, js.Hadron):
            self.hadrons.append(obj)

    def WriteEvent(self):
        for i, h in enumerate(self.hadrons):
            # C++ ostream defaults: %g, 6 significant digits
            self.out.write(f"{i} {h.pid()} {h.pstat()} {h.e():g} {h.px():g} {h.py():g} "
                           f"{h.pz():g}\n")
        self.hadrons = []

    def FinishTask(self):
        self.out.close()


USER_XML = """<jetscape>
  <nEvents>{events}</nEvents>
  <vlevel>0</vlevel>
  <Random><seed>7</seed></Random>
  <JetScapeWriterFinalStateHadronsAscii> on </JetScapeWriterFinalStateHadronsAscii>
  <outputFilename>run</outputFilename>
  <Hard>{hard}</Hard>
  <Eloss>
    <maxT>20</maxT>
    {eloss}
  </Eloss>
  <JetHadronization><name>colorless</name></JetHadronization>
</jetscape>
"""


def main():
    mode, main_xml, outdir = sys.argv[1:4]
    events = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    hard = "<PGun><name>PGun</name><pT>100</pT><parID>21</parID></PGun>"
    eloss = "<Matter> </Matter>"
    if mode == "pygun":
        js.register_module("CustomModulePyPGun", PyPGun, section="Hard",
                           defaults="<pT>100</pT><parID>21</parID>")
        hard = "<CustomModulePyPGun> </CustomModulePyPGun>"
    elif mode in ("pymatter", "pymatter_unaligned"):
        cls = PyMatter if mode == "pymatter" else PyMatterUnaligned
        js.register_module("CustomModulePyMatter", cls, section="Eloss")
        eloss = "<CustomModulePyMatter> </CustomModulePyMatter>"
    elif mode != "cpp":
        sys.exit(f"unknown mode {mode}")
    with open("user.xml", "w") as f:
        f.write(USER_XML.format(events=events, hard=hard, eloss=eloss))

    driver = js.JetScape()
    driver.SetXMLMainFileName(main_xml)
    driver.SetXMLUserFileName("user.xml")
    driver.Add(FinalHadronsWriter("python_hadrons.dat"))   # in every mode: same tasks
    driver.Init()
    t0 = time.perf_counter()
    driver.Exec()
    wall = time.perf_counter() - t0
    driver.Finish()
    print(f"WALL {wall:.4f}")
    print(f"CALLS {PyMatter.calls}")


if __name__ == "__main__":
    main()
