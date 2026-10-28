"""
The Pythonic layer in a JETSCAPE run (docs/Plans/PLAN_pyjetscape_pythonic.md, steps 2, 4
and 5): every new accessor against the function it wraps, hadronize() on records against
hadronize_partons() on the old array, a module declared with @module / xml_parameters /
a returned list against the same module written the old way, self.rng, and the driver as
a context manager.  Each run is a subprocess: the XML and the seed are per process.
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
pytestmark = pytest.mark.skipif(not MAIN_XML.exists(),
                                reason="no X-SCAPE config next to js-contrib")

USER_XML = """<jetscape>
  <nEvents>3</nEvents>
  <vlevel>0</vlevel>
  <Random><seed>11</seed></Random>
  <Hard><PGun><name>PGun</name><pT>60</pT><parID>21</parID></PGun></Hard>
  <Eloss><maxT>20</maxT>{eloss}</Eloss>
  <JetHadronization><name>colorless</name></JetHadronization>
</jetscape>
"""


def _run(tmp_path, script, eloss="<Matter> </Matter>"):
    (tmp_path / "user.xml").write_text(USER_XML.format(eloss=eloss))
    path = tmp_path / "run.py"
    path.write_text(textwrap.dedent(script).format(python_dir=str(PYTHON_DIR),
                                                   main=str(MAIN_XML)))
    res = subprocess.run([sys.executable, str(path)], cwd=tmp_path, capture_output=True,
                         check=False, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    line = next(x for x in res.stdout.splitlines() if x.startswith("RESULT"))
    return json.loads(line[len("RESULT"):])


ACCESSORS = """
import json, sys
import numpy as np
sys.path.insert(0, {python_dir!r})
import jetscape as js
from jetscape import pyjetscape_core as core
from jetscape.records import (final_parton_records, initiator_records,
                              hard_parton_records)
from jetscape.showers import initiators_from_manager

def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

def same_dict(d, h):
    return all(same(d[k], getattr(h, k)) for k in ("pid", "pstat", "p", "x", "mass"))

checks, stored = [], []
with js.per_event({main!r}, "user.xml") as driver:
    sm = core.JetScapeSignalManager.Instance()
    hard = next(t for t in driver.GetTaskList() if isinstance(t, core.HardProcess))
    for i in driver.events():
        mgr = driver.energy_loss_manager
        old_mgr = sm.GetJetEnergyLossManagerPointer()
        fp = mgr.final_parton_array()
        stored.append(fp)
        had = driver.hadronization_manager
        checks.append({{
            "event": i,
            "manager": mgr is not None and old_mgr is not None,
            "final_partons": same(fp, final_parton_records(old_mgr.final_partons_numpy())),
            "initiators": same(mgr.initiator_array(),
                               initiator_records(initiators_from_manager(old_mgr))),
            "showers": len(mgr.showers) == old_mgr.number_of_showers() > 0,
            "hard": same(hard.parton_array(),
                         hard_parton_records(core.hard_partons_numpy(hard))),
            "hard_info": hard.info == core.hard_process_info(hard),
            "jet_hadrons": same_dict(core.hadronization_hadrons_numpy(had),
                                     had.hadron_array()),
            "n_hadrons": len(had.hadron_array()) > 0,
            "seed": had.last_random_seed == core.jet_hadronization_last_random_seed(had),
        }})
    # hadronize the stored partons again: records through the layer, rows the old way
    colorless = core.create_module("ColorlessHadronization")
    colorless.Init()
    for fp in stored:
        new = colorless.hadronize(fp, seed=1234)
        old = core.hadronize_partons(colorless, js.records.final_parton_rows(fp), 1234)
        checks.append({{"hadronize": same_dict(old, new) and new.info["seed"] == old["seed"],
                        "n": len(new)}})
    n_events = driver.GetNumberOfEvents()
print("RESULT" + json.dumps({{"checks": checks, "n_events": n_events}}))
"""


def test_the_accessors_equal_the_functions_they_wrap(tmp_path):
    out = _run(tmp_path, ACCESSORS)
    assert out["n_events"] == 3
    per_event = [c for c in out["checks"] if "event" in c]
    assert [c["event"] for c in per_event] == [0, 1, 2]
    for c in out["checks"]:
        bad = [k for k, v in c.items() if v is False]
        assert not bad, (c, bad)
    assert sum(c["n"] for c in out["checks"] if "hadronize" in c) > 0


MODULES = """
import json, sys
import numpy as np
sys.path.insert(0, {python_dir!r})
import jetscape as js
from jetscape import pyjetscape_core as core

STYLE = sys.argv[1]

if STYLE == "old":
    class Split(core.JetEnergyLoss):
        def InitTask(self):
            self.t_split = self.get_xml_element_double(
                ["Eloss", "CustomModulePySplitOld", "t_split"], False) or 1.0
            self.e_min = self.get_xml_element_double(
                ["Eloss", "CustomModulePySplitOld", "e_min"], False) or 30.0

        def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut):
            p = pIn[0]
            if time >= self.t_split and p.e() > self.e_min:
                for s in (+1, -1):
                    pOut.append(core.Parton(0, p.pid(), 0, core.FourVector(
                        0.5 * p.px(), s * 1.0, 0.5 * p.pz(), 0.5 * p.e()), p.x_in()))
    core.register_module("CustomModulePySplitOld", Split, section="Eloss",
                         defaults="<t_split>1.0</t_split><e_min>30</e_min>")
else:
    @js.module("CustomModulePySplitNew", section="Eloss")
    class Split(core.JetEnergyLoss):
        xml_parameters = {{"t_split": 1.0, "e_min": 30.0}}
        seen = []

        def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut):
            p = pIn[0]
            Split.seen.append((self.t_split, self.e_min))
            if time >= self.t_split and p.e() > self.e_min:
                E, px, py, pz = p.momentum
                return [js.make_parton(p.pid(), (E / 2, px / 2, s * 1.0, pz / 2),
                                       position=p.position) for s in (+1, -1)]

partons = []
with js.per_event({main!r}, "user.xml") as driver:
    for _ in driver.events():
        partons.append(driver.energy_loss_manager.final_parton_array().tolist())
result = {{"partons": partons}}
if STYLE == "new":
    result["params"] = sorted(set(Split.seen))
    # self.rng: one stream per task; a Clone gets its own, the module keeps its own
    m = Split()
    g = m.rng
    c = m.Clone()
    result["rng"] = {{"cached": m.rng is g, "clone_differs": c.rng is not g,
                      "draws_differ": bool(g.integers(1 << 30) != c.rng.integers(1 << 30)),
                      "first": int(Split().rng.integers(1 << 30))}}
print("RESULT" + json.dumps(result))
"""


def test_a_module_in_the_new_style_equals_the_old_one(tmp_path):
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    eloss = "<{n}><e_min>25</e_min></{n}>"
    script = MODULES.replace('sys.argv[1]', '"{style}"')
    old = _run(old_dir, script.replace("{style}", "old"),
               eloss=eloss.format(n="CustomModulePySplitOld"))
    new = _run(new_dir, script.replace("{style}", "new"),
               eloss=eloss.format(n="CustomModulePySplitNew"))
    assert new["params"] == [[1.0, 25.0]]             # user XML value, else the default
    assert new["partons"] == old["partons"]
    assert any(len(ev) > 1 for ev in new["partons"])  # the module did split
    rng = new["rng"]
    assert rng["cached"] and rng["clone_differs"] and rng["draws_differ"]
    again = _run(new_dir, script.replace("{style}", "new"),
                 eloss=eloss.format(n="CustomModulePySplitNew"))
    assert again["rng"]["first"] == rng["first"]     # reproducible with the seed


@pytest.mark.skipif(not getattr(core, "ICOLORED_HAS_COLOUR_CLOSURE", False),
                    reason="X-SCAPE without iColoredHadronization's colour closure")
def test_the_colour_methods_wrap_the_colour_functions(tmp_path):
    script = """
    import json, sys
    sys.path.insert(0, {python_dir!r})
    import jetscape as js
    from jetscape import pyjetscape_core as core
    js.per_event({main!r}, "user.xml").Init()
    ic = core.create_module("iColoredHadronization")
    ic.Init()
    fp = js.records.final_parton_records([[0, 21, 0, 50, 0, 0, 50, 0, 0, 0, 0, 0, 101, 102],
                                          [0, 1, 0, 30, 0, 0, -30, 0, 0, 0, 0, 0, 102, 0],
                                          [0, -1, 0, 30, 0, 30, 0, 0, 0, 0, 0, 0, 0, 101]])
    h = ic.hadronize(fp, seed=5)
    audit = ic.colour_audit()
    same_audit = audit == core.jet_hadronization_colour_audit(ic)
    old = core.hadronize_partons(ic, js.records.final_parton_rows(fp), 5)
    same = all(old[k].tobytes() == getattr(h, k).tobytes()
               for k in ("pid", "pstat", "p", "x", "mass"))
    ic.set_colour_closure("thermal")
    print("RESULT" + json.dumps({{
        "audit": same_audit, "same_hadrons": same, "seed": h.info["seed"] == old["seed"],
        "totals": ic.colour_totals() == core.jet_hadronization_colour_totals(ic),
        "n": len(h)}}))
    """
    out = _run(tmp_path, script)
    assert out["audit"] and out["totals"] and out["n"] > 0
    assert out["same_hadrons"] and out["seed"]      # records == the old array, one seed


def test_run_by_hand_with_the_context_manager_and_new_style_toys(tmp_path):
    """Step 5: run_by_hand.py's SUMMARY from js.per_event(...) + driver.events(), with the
    old toys and with the toys of toy_modules_pythonic.py."""
    example = PYTHON_DIR.parent / "example" / "python_modules"

    def summary(script, *args):
        res = subprocess.run([sys.executable, str(example / script), "--outdir",
                              str(tmp_path / "_".join((script,) + args)), *args],
                             capture_output=True, text=True, check=False, timeout=600)
        assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
        return next(x for x in res.stdout.splitlines() if x.startswith("SUMMARY"))

    reference = summary("run_by_hand.py")
    assert summary("run_by_hand_pythonic.py", "--style", "old") == reference
    assert summary("run_by_hand_pythonic.py", "--style", "new") == reference


def test_events_skips_an_event_whose_module_throws(tmp_path):
    script = """
    import json, sys
    sys.path.insert(0, {python_dir!r})
    import jetscape as js
    from jetscape import pyjetscape_core as core

    @js.module("CustomModulePyThrows", section="Eloss")
    class Throws(core.JetEnergyLoss):
        def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut):
            if self.GetCurrentEvent() == 1:
                raise RuntimeError("event 1 fails")

    skipped, done = [], []
    with js.per_event({main!r}, "user.xml") as driver:
        for i in driver.events(on_error=lambda i, exc: skipped.append((i, str(exc)))):
            done.append(i)
    print("RESULT" + json.dumps({{"done": done, "skipped": skipped}}))
    """
    out = _run(tmp_path, script, eloss="<CustomModulePyThrows> </CustomModulePyThrows>")
    assert out["done"] == [0, 2]
    assert [s[0] for s in out["skipped"]] == [1] and "event 1 fails" in out["skipped"][0][1]
