"""
Validation V3 of docs/Plans/PLAN_pyjetscape_complete.md: Python modules reproduce the C++
ones exactly. equivalence_driver.py runs a 100 GeV gluon (PGun), Matter in vacuum and
ColorlessHadronization with one C++ module swapped for a Python one; with the same seed
the final-state hadron files must be the same, byte for byte:

    pygun      PGun ported to Python
    pymatter   a Python energy-loss module handing every step to a C++ Matter of its own
    (cpp)      the Python writer next to the C++ one: the same particle lines

and a Python Matter whose inner Matter takes a task number of its own must differ (the
seeds follow the task numbers), so the comparison can fail.
"""

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
# a pyjetscape_core that is there but does not load is an error, not a skip
core = pytest.importorskip("jetscape.pyjetscape_core", exc_type=ModuleNotFoundError)

XSCAPE = HERE.parents[4]
MAIN_XML = XSCAPE / "config" / "jetscape_main.xml"
DRIVER = HERE / "equivalence_driver.py"
EVENTS = 20

pytestmark = pytest.mark.skipif(not MAIN_XML.exists(),
                                reason="no X-SCAPE config next to js-contrib")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    base = tmp_path_factory.mktemp("equivalence")
    for mode in ("cpp", "pygun", "pymatter", "pymatter_unaligned"):
        d = base / mode
        res = subprocess.run([sys.executable, str(DRIVER), mode, str(MAIN_XML), str(d),
                              str(EVENTS)], capture_output=True, text=True, timeout=600,
                             check=False)
        assert res.returncode == 0, mode + "\n" + res.stdout[-3000:] + res.stderr[-3000:]
        out[mode] = d
    return out


def hadron_file(d):
    return (d / "run_final_state_hadrons.dat").read_bytes()


def test_the_reference_has_hadrons(runs):
    lines = hadron_file(runs["cpp"]).decode().splitlines()
    assert sum(x.startswith("#\tEvent") for x in lines) == EVENTS
    assert len(lines) > 10 * EVENTS                       # showers, not single partons


@pytest.mark.parametrize("mode", ["pygun", "pymatter"])
def test_a_python_module_gives_the_same_hadrons(runs, mode):
    assert hadron_file(runs[mode]) == hadron_file(runs["cpp"])


def test_the_python_writer_writes_the_same_lines(runs):
    cpp = [x for x in hadron_file(runs["cpp"]).decode().splitlines() if not x.startswith("#")]
    py = (runs["cpp"] / "python_hadrons.dat").read_text().splitlines()
    assert py == cpp


def test_other_seeds_give_other_hadrons(runs):
    assert hadron_file(runs["pymatter_unaligned"]) != hadron_file(runs["cpp"])
