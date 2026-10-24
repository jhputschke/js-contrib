"""
Energy-loss modules written in Python (docs/Plans/PLAN_pyjetscape_complete.md, phase 1):
the trampoline's lists, Clone(), and a module the XML makes and the manager copies.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))
# a pyjetscape_core that is there but does not load is an error, not a skip
core = pytest.importorskip("jetscape.pyjetscape_core", exc_type=ModuleNotFoundError)

XSCAPE = Path(__file__).resolve().parents[5]
MAIN_XML = XSCAPE / "config" / "jetscape_main.xml"


def parton(e=50.0, pid=21):
    return core.Parton(0, pid, 0, core.FourVector(e, 0.0, 0.0, e), core.FourVector())


class Split(core.JetEnergyLoss):
    """splits a parton above 30 GeV into two halves (+-1 GeV in px), once it is at
    time >= 1 fm/c; marks the parton it saw with pstat 5"""

    def __init__(self):
        super().__init__()
        self.SetId("PySplit")
        self.seen = []

    def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut):
        p = pIn[0]
        self.seen.append(time)
        p.set_stat(5)
        if time >= 1.0 and p.e() > 30.0:
            for s in (+1, -1):
                pOut.append(core.Parton(0, p.pid(), 0,
                                        core.FourVector(0.5 * p.px() + s, 0.5 * p.py(),
                                                        0.5 * p.pz(), 0.5 * p.e()),
                                        p.x_in()))


def test_the_trampoline_passes_lists_and_writes_back():
    m = Split()
    pin, pout = [parton()], []
    core.JetEnergyLoss.DoEnergyLoss(m, 0.1, 0.5, 0.0, pin, pout)   # through C++
    assert pout == [] and pin[0].pstat() == 5                     # changed in place
    core.JetEnergyLoss.DoEnergyLoss(m, 0.1, 1.0, 0.0, pin, pout)
    assert [round(q.px(), 6) for q in pout] == [26.0, 24.0]
    assert sum(q.e() for q in pout) == pytest.approx(50.0)
    assert m.seen == [0.5, 1.0]


def test_clone_copies_settings_and_attributes():
    m = Split()
    m.SetDeltaT(0.2)
    m.extra = [1, 2]
    c = m.Clone()
    assert type(c) is Split and c is not m
    assert c.GetId() == "PySplit" and c.GetDeltaT() == 0.2
    assert c.extra is m.extra                                     # shallow, as C++
    assert c.GetMyTaskNumber() != m.GetMyTaskNumber()             # as a C++ clone


def test_clone_can_be_overridden():
    class Deep(Split):
        def Clone(self):
            c = Deep()
            c.seen = list(self.seen)
            return c

    m = Deep()
    m.seen.append(9.0)
    c = m.Clone()
    assert type(c) is Deep and c.seen == [9.0] and c.seen is not m.seen


def test_a_python_module_needs_no_init_task():
    Split().InitTask()                     # no container check (no sub-modules)


def test_a_cpp_module_is_a_jet_energy_loss():
    matter = core.create_module("Matter")
    assert isinstance(matter, core.JetEnergyLoss)
    assert callable(matter.DoEnergyLoss)


RUN = textwrap.dedent('''
    import json, sys
    sys.path.insert(0, {python_dir!r})
    import jetscape.pyjetscape_core as core

    class Split(core.JetEnergyLoss):
        def __init__(self):
            super().__init__()
            self.SetId("PySplit")
        def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut):
            p = pIn[0]
            if time >= 1.0 and p.e() > 30.0:
                for s in (+1, -1):
                    pOut.append(core.Parton(0, p.pid(), 0,
                        core.FourVector(0.5 * p.px() + s, 0.5 * p.py(), 0.5 * p.pz(),
                                        0.5 * p.e()), p.x_in()))
        def InitTask(self):
            self.hydro = self.GetHydroCell(0.5, 0.0, 0.0, 0.0)

    core.register_module("CustomModulePySplit", Split)
    js = core.JetScapePerEvent()
    js.SetXMLMainFileName({main!r})
    js.SetXMLUserFileName({user!r})
    js.Init()
    sm = core.JetScapeSignalManager.Instance()
    out = []
    for ev in range(2):
        js.ExecPerEvent()
        mgr = sm.GetJetEnergyLossManagerPointer()
        modules = [t for jl in mgr.GetTaskList() for t in jl.GetTaskList()]
        out.append({{"partons": mgr.final_partons_numpy().tolist(),
                     "modules": [type(t).__name__ for t in modules]}})
        js.ClearPerEvent()
    js.Finish()
    print("RESULT" + json.dumps(out))
''')


@pytest.mark.skipif(not MAIN_XML.exists(), reason="no X-SCAPE config next to js-contrib")
def test_a_python_module_from_the_xml(tmp_path):
    user = tmp_path / "user.xml"
    user.write_text(textwrap.dedent("""\
        <jetscape>
          <nEvents>2</nEvents>
          <vlevel>0</vlevel>
          <Random><seed>1</seed></Random>
          <Hard><PGun><name>PGun</name><pT>50</pT><parID>21</parID></PGun></Hard>
          <Eloss>
            <deltaT>0.1</deltaT>
            <maxT>2</maxT>
            <CustomModulePySplit> </CustomModulePySplit>
          </Eloss>
        </jetscape>
        """))
    script = tmp_path / "run.py"
    script.write_text(RUN.format(python_dir=str(PYTHON_DIR), main=str(MAIN_XML),
                                 user=str(user)))
    res = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, check=False,
                         text=True, timeout=300)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    line = next(x for x in res.stdout.splitlines() if x.startswith("RESULT"))
    events = json.loads(line[len("RESULT"):])
    assert len(events) == 2
    for ev in events:
        assert ev["modules"] == ["Split"]          # the manager's per-shower copy
        e = [row[3] for row in ev["partons"]]      # FINAL_PARTON_COLUMNS: E
        assert e == pytest.approx([25.0, 25.0])
