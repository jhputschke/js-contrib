"""
The foundations of the complete binding (docs/Plans/PLAN_pyjetscape_complete.md, phase 0):
particles made and changed in Python, Python modules in the module factory and owned by
C++, the task random engines.
"""

import gc
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
# a pyjetscape_core that is there but does not load is an error, not a skip
core = pytest.importorskip("jetscape.pyjetscape_core", exc_type=ModuleNotFoundError)

XSCAPE = Path(__file__).resolve().parents[5]
MAIN_XML = XSCAPE / "config" / "jetscape_main.xml"


# ------------------------------------------------------------------------------ particles

def test_fourvector_component_order_is_x_y_z_t():
    v = core.FourVector(1.0, 2.0, 3.0, 4.0)
    assert (v.x(), v.y(), v.z(), v.t()) == (1.0, 2.0, 3.0, 4.0)
    assert [v.comp(i) for i in range(4)] == [4.0, 1.0, 2.0, 3.0]     # t first, as C++
    w = v + core.FourVector(1.0, 1.0, 1.0, 1.0)
    assert w.to_list() == [2.0, 3.0, 4.0, 5.0] and v.to_list() == [1.0, 2.0, 3.0, 4.0]


def test_a_parton_made_in_python():
    p = core.Parton(0, 21, 0, core.FourVector(3.0, 4.0, 0.0, 10.0),
                    core.FourVector(0.5, 0.0, 0.0, 1.5))
    assert p.pid() == 21 and p.pstat() == 0 and p.plabel() == 0
    assert (p.px(), p.py(), p.pz(), p.e()) == (3.0, 4.0, 0.0, 10.0)
    assert p.pt() == pytest.approx(5.0)
    # position accessors are the position, as before; the virtuality is virtuality()
    assert (p.x(), p.y(), p.z(), p.t()) == (0.5, 0.0, 0.0, 1.5)
    assert p.time() == 1.5
    assert p.virtuality() == pytest.approx(10.0 ** 2 - 5.0 ** 2)


def test_parton_setters():
    p = core.Parton(1, 2, 0, core.FourVector(0.0, 0.0, 20.0, 20.0), core.FourVector())
    p.set_stat(1000)
    p.set_label(-3)
    p.set_id(-2)
    p.set_form_time(0.7)
    p.set_color(101)
    p.set_anti_color(0)
    p.set_max_color(4000)
    p.set_x(core.FourVector(1.0, 2.0, 3.0, 4.0))
    p.reset_momentum(1.0, 2.0, 3.0, 9.0)
    assert (p.pstat(), p.plabel(), p.pid()) == (1000, -3, -2)
    assert p.form_time() == 0.7
    assert (p.color(), p.anti_color(), p.max_color()) == (101, 0, 4000)
    assert p.x_in().to_list() == [1.0, 2.0, 3.0, 4.0]
    assert p.p_in().to_list() == [1.0, 2.0, 3.0, 9.0]
    p.reset_p(0.0, 0.0, 5.0)                              # keeps E
    assert p.p_in().to_list() == [0.0, 0.0, 5.0, 9.0]


def test_a_parton_copy_is_independent():
    p = core.Parton(0, 1, 0, 10.0, 0.5, 1.0, 12.0)        # (pt, eta, phi, e)
    q = p.copy()
    q.set_stat(7)
    assert p.pstat() == 0 and q.pstat() == 7
    assert q.pt() == pytest.approx(10.0) and q.eta() == pytest.approx(0.5)


def test_a_hadron_made_in_python():
    h = core.Hadron(0, 211, 0, core.FourVector(0.3, 0.0, 0.0, 0.33),
                    core.FourVector(), 0.13957, 1, 0, 0)
    assert h.pid() == 211 and h.charge() == 1 and h.baryon_number() == 0
    assert h.restmass() == pytest.approx(0.13957)
    h.set_charge(-1)
    assert h.charge() == -1
    assert isinstance(h, core.JetScapeParticleBase)


# ------------------------------------------------------------------------------ factory

class Counter(core.JetScapeModuleBase):
    """counts its ExecuteTask calls"""

    def __init__(self):
        super().__init__()
        self.n = 0

    def ExecuteTask(self):
        self.n += 1


@pytest.fixture
def counter_registered():
    core.register_module("CustomModuleCounter", Counter)
    yield "CustomModuleCounter"
    core.unregister_module("CustomModuleCounter")


def test_register_and_create(counter_registered):
    assert counter_registered in core.registered_modules(python_only=True)
    assert "Matter" in core.registered_modules()
    mod = core.create_module(counter_registered)
    assert type(mod) is Counter and mod.n == 0


def test_a_python_module_owned_by_cpp_survives(counter_registered):
    js = core.JetScape()
    js.Add(core.create_module(counter_registered))    # no Python reference kept
    gc.collect()
    mod = js.GetTaskAt(0)
    assert type(mod) is Counter
    mod.Exec()                                       # JetScapeTask::Exec -> ExecuteTask
    mod.Exec()
    assert js.GetTaskAt(0).n == 2


def test_a_cpp_module_name_is_refused():
    with pytest.raises(ValueError):
        core.register_module("Matter", Counter)


def test_unregister():
    core.register_module("CustomModuleGone", Counter)
    assert core.unregister_module("CustomModuleGone")
    assert "CustomModuleGone" not in core.registered_modules()
    assert not core.unregister_module("CustomModuleGone")
    with pytest.raises(ValueError):
        core.create_module("CustomModuleGone")


# ------------------------------------------------------------------------------ random

@pytest.mark.skipif(not MAIN_XML.exists(), reason="no X-SCAPE config next to js-contrib")
def test_task_generators_are_reproducible(tmp_path):
    user = tmp_path / "user.xml"
    user.write_text("<jetscape><Random><seed>12345</seed></Random></jetscape>")
    core.load_xml(str(MAIN_XML), str(user))
    assert core.random_seed() == 12345
    if not core.HAS_NEXT_TASK_NUMBER:
        pytest.skip("this X-SCAPE has no JetScapeTaskSupport::GetNextTaskNumber")
    n = core.next_task_number()
    a, b = Counter(), Counter()
    assert (a.GetMyTaskNumber(), b.GetMyTaskNumber()) == (n, n + 1)
    draws_a = [a.GetMt19937Generator()() for _ in range(3)]
    assert draws_a != [b.GetMt19937Generator()() for _ in range(3)]   # one engine per task
    core.set_next_task_number(n)
    c = Counter()                                       # the same task number as a
    gc_ = c.GetMt19937Generator()
    assert [gc_() for _ in range(3)] == draws_a
    assert 0.0 <= gc_.uniform() < 1.0
