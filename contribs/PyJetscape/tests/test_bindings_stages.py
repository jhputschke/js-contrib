"""
A Python module for every stage (docs/Plans/PLAN_pyjetscape_complete.md, phase 1): the
trampolines called through C++, and one run where the gun, the energy loss, the jet
hadronization, the soft particlization, the afterburner and a writer are all Python.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))
# a pyjetscape_core that is there but does not load is an error, not a skip
core = pytest.importorskip("jetscape.pyjetscape_core", exc_type=ModuleNotFoundError)

XSCAPE = Path(__file__).resolve().parents[5]
MAIN_XML = XSCAPE / "config" / "jetscape_main.xml"


def parton(e=40.0, sign=1):
    return core.Parton(0, 21, 0, core.FourVector(sign * e, 0.0, 0.0, e), core.FourVector())


def pion(p):
    return core.Hadron(0, 211, 1, p.p_in(), p.x_in())


# ------------------------------------------------------------------- through C++

def test_hard_process():
    class Gun(core.HardProcess):
        def ExecuteTask(self):
            self.AddParton(parton(sign=+1))
            self.AddParton(parton(sign=-1))

        def GetSigmaGen(self):
            return 2.5

    g = Gun()
    core.HardProcess.ExecuteTask(g)
    assert g.GetNHardPartons() == 2 and [p.px() for p in g.GetPartonList()] == [40.0, -40.0]
    assert core.HardProcess.GetSigmaGen(g) == 2.5           # the C++ virtual reaches it


def test_hadronization():
    class Hadro(core.Hadronization):
        def DoHadronization(self, shower, hOut, pOut):
            for partons in shower:
                for p in partons:
                    (hOut if p.e() > 10 else pOut).append(pion(p) if p.e() > 10 else p)

    shower = [[parton(40.0), parton(5.0)], [parton(20.0, -1)]]
    h_out, p_out = [], []
    core.Hadronization.DoHadronization(Hadro(), shower, h_out, p_out)
    assert [h.pid() for h in h_out] == [211, 211] and len(p_out) == 1
    assert p_out[0].e() == 5.0


def test_preequilibrium():
    class Preeq(core.PreequilibriumDynamics):
        def EvolvePreequilibrium(self):
            self.set_e_numpy(np.arange(4.0))
            self.set_bulk_Pi_numpy(np.full(4, 0.5))

        def GetPreequilibriumEvodtau(self):
            return 0.05

    p = Preeq()
    core.PreequilibriumDynamics.EvolvePreequilibrium(p)
    assert p.get_e_numpy().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert p.get_bulk_Pi_numpy().tolist() == [0.5] * 4
    # Jetscape::real is a float
    assert core.PreequilibriumDynamics.GetPreequilibriumEvodtau(p) == pytest.approx(0.05)


def test_soft_particlization_hadron_list():
    sp = core.SoftParticlization()
    sp.set_hadron_list([[pion(parton(1.0))] * 3, [pion(parton(2.0))]])
    assert [len(s) for s in sp.get_hadron_list()] == [3, 1]


def test_afterburner():
    class AB(core.Afterburner):
        def GetCurrentHadronList(self):
            return [pion(parton(3.0)), pion(parton(4.0))]

    hs = core.Afterburner.GetCurrentHadronList(AB())
    assert [h.e() for h in hs] == [3.0, 4.0]


def test_writer():
    class W(core.JetScapeWriter):
        def __init__(self):
            super().__init__()
            self.got = []

        def GetStatus(self):
            return True

        def Write(self, obj):
            self.got.append(type(obj).__name__)

    w = W()
    for obj in (parton(), pion(parton()), "text"):
        core.JetScapeWriter.Write(w, obj)
    assert w.got == ["Parton", "Hadron", "str"]
    assert core.JetScapeWriter.GetStatus(w)
    w.GetHeader().SetSigmaGen(3.0)
    assert w.GetHeader().GetSigmaGen() == 3.0


def test_liquefier():
    class Liq(core.LiquefierBase):
        def smearing_kernel(self, tau, x, y, eta, droplet):
            return [tau, x, y, eta]

    liq = Liq()
    d = core.Droplet([0.5, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0])
    assert core.LiquefierBase.smearing_kernel(liq, 1.0, 2.0, 3.0, 0.1, d) == pytest.approx(
        [1.0, 2.0, 3.0, 0.1])
    jl = core.JetEnergyLoss()
    jl.add_a_liquefier(liq)
    assert jl.get_liquefier() is liq


# ------------------------------------------------------------------- one run, all Python

RUN = textwrap.dedent('''
    import json, sys
    sys.path.insert(0, {python_dir!r})
    import jetscape.pyjetscape_core as core

    def pion(p, stat=1):
        return core.Hadron(0, 211, stat, p.p_in(), p.x_in())

    class Gun(core.HardProcess):
        def ExecuteTask(self):
            for sign in (+1, -1):
                self.AddParton(core.Parton(0, 21, 0, core.FourVector(sign * 40.0, 0.0, 0.0, 40.0),
                                           core.FourVector()))
        def GetSigmaGen(self):
            return 2.5

    class Split(core.JetEnergyLoss):
        def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut):
            p = pIn[0]
            if time >= 1.0 and p.e() > 30.0:
                for s in (+1, -1):
                    pOut.append(core.Parton(0, p.pid(), 0, core.FourVector(
                        0.5 * p.px(), s * 1.0, 0.5 * p.pz(), 0.5 * p.e()), p.x_in()))

    class Hadro(core.Hadronization):
        def DoHadronization(self, shower, hOut, pOut):
            for partons in shower:
                for p in partons:
                    hOut.append(pion(p))

    class Soft(core.SoftParticlization):
        def ExecuteTask(self):
            self.set_hadron_list([[pion(core.Parton(0, 21, 0, core.FourVector(0.3, 0.0, 0.0, 0.5),
                                                    core.FourVector()), stat=0)] * 3])

    class AB(core.Afterburner):
        def ExecuteTask(self):
            self.out = [h for sample in self.GatherAfterburnerHadrons() for h in sample]
        def GetCurrentHadronList(self):
            return self.out

    class Writer(core.JetScapeWriter):
        def __init__(self):
            super().__init__()
            self.events = []
            self.cur = {{}}
        def GetStatus(self):
            return True
        def Write(self, obj):
            k = type(obj).__name__
            self.cur[k] = self.cur.get(k, 0) + 1
        def WriteEvent(self):
            self.cur["sigma"] = self.GetHeader().GetSigmaGen()
            self.events.append(self.cur)
            self.cur = {{}}

    for name, cls in (("CustomModulePyGun", Gun), ("CustomModulePySplit", Split),
                      ("CustomModulePyHadro", Hadro), ("CustomModulePySoft", Soft),
                      ("CustomModulePyAB", AB)):
        core.register_module(name, cls)
    js = core.JetScapePerEvent()
    js.SetXMLMainFileName({main!r})
    js.SetXMLUserFileName({user!r})
    writer = Writer()
    js.Add(writer)
    js.Init()
    out = {{"tasks": [type(t).__name__ for t in js.GetTaskList()], "afterburner": []}}
    ab = [t for t in js.GetTaskList() if isinstance(t, core.Afterburner)][0]
    for ev in range(2):
        js.ExecPerEvent()
        out["afterburner"].append(sorted(round(h.e(), 3) for h in ab.GetCurrentHadronList()))
        js.ClearPerEvent()
    js.Finish()
    out["writer"] = writer.events
    print("RESULT" + json.dumps(out))
''')


@pytest.mark.skipif(not MAIN_XML.exists(), reason="no X-SCAPE config next to js-contrib")
def test_every_stage_in_python(tmp_path):
    user = tmp_path / "user.xml"
    user.write_text(textwrap.dedent("""\
        <jetscape>
          <nEvents>2</nEvents>
          <vlevel>0</vlevel>
          <Random><seed>1</seed></Random>
          <Hard><CustomModulePyGun> </CustomModulePyGun></Hard>
          <Eloss>
            <deltaT>0.1</deltaT>
            <maxT>2</maxT>
            <CustomModulePySplit> </CustomModulePySplit>
          </Eloss>
          <JetHadronization><name>CustomModulePyHadro</name></JetHadronization>
          <SoftParticlization><CustomModulePySoft> </CustomModulePySoft></SoftParticlization>
          <Afterburner>
            <CustomModulePyAB> </CustomModulePyAB>
            <include_fragmentation_hadrons>1</include_fragmentation_hadrons>
            <output_only_final_state_hadrons>0</output_only_final_state_hadrons>
          </Afterburner>
        </jetscape>
        """))
    script = tmp_path / "run.py"
    script.write_text(RUN.format(python_dir=str(PYTHON_DIR), main=str(MAIN_XML),
                                 user=str(user)))
    res = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, check=False,
                         text=True, timeout=300)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    line = next(x for x in res.stdout.splitlines() if x.startswith("RESULT"))
    out = json.loads(line[len("RESULT"):])
    assert out["tasks"] == ["Writer", "Gun", "JetEnergyLossManager", "HadronizationManager",
                            "Soft", "AB"]
    for hadrons in out["afterburner"]:
        # 3 soft hadrons (E 0.5) + 4 jet hadrons (E 20) per event
        assert hadrons == pytest.approx([0.5] * 3 + [20.0] * 4)
    assert len(out["writer"]) == 2
    for ev in out["writer"]:
        assert ev["sigma"] == 2.5                         # the gun's header, via C++
        assert ev["Hadron"] >= 4 and ev["Parton"] >= 4
