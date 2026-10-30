"""
Pipelines built in Python and the objects they make (docs/Plans/PLAN_pyjetscape_complete.md,
phases 2 and 3): shower navigation, the ISR chain, shower generators, the C++ writers and
the ASCII reader, and the example/python_modules scripts (the same all-Python pipeline from
the XML and by hand).
"""

import json
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
EXAMPLE = HERE.parent / "example" / "python_modules"


def parton(px, py, pz, e):
    return core.Parton(0, 21, 0, core.FourVector(px, py, pz, e), core.FourVector())


def test_a_shower_built_and_walked_in_python():
    ps = core.PartonShower()
    v0, v1, v2, v3 = (ps.new_vertex(core.Vertex(0.0, 0.0, 0.0, t)) for t in (0.0, 1.0, 2.0, 2.0))
    mother = parton(40.0, 0.0, 0.0, 40.0)                     # transverse (z is in pT)
    d1, d2 = parton(30.0, 1.0, 0.0, 30.0), parton(10.0, -1.0, 0.0, 10.0)
    ps.new_parton(v0, v1, mother)
    ps.new_parton(v1, v2, d1)
    ps.new_parton(v1, v3, d2)
    assert ps.number_of_partons() == 3 and ps.number_of_vertices() == 4
    # the same objects (in GTL's edge order, not the insertion order)
    assert set(map(id, ps.children(mother))) == {id(d1), id(d2)}
    assert ps.parents(d1) == [mother]
    assert ps.children(d1) == [] and ps.parents(mother) == []
    assert set(map(id, ps.final_partons())) == {id(d1), id(d2)}
    assert ps.edge_of(d1)[1:] == (v1, v2)
    assert ps.end_vertex(mother).x_in().t() == 1.0
    assert ps.vertex_id(ps.production_vertex(d2)) == v1
    assert ps.is_one_to_two(v1) and ps.is_end_vertex(v2)
    assert ps.split_z(v1) == pytest.approx(0.25, abs=0.01)   # softer pT / mother pT
    rows = ps.to_numpy()                                      # the same ids
    assert sorted(map(tuple, rows[:, :2].astype(int).tolist())) == [(v0, v1), (v1, v2), (v1, v3)]


def test_the_isr_chain_by_hand():
    isr, jet, imatter = core.IsrManager(), core.IsrJet(), core.iMATTER()
    jet.SetDeltaT(-0.1)
    jet.SetMaxT(-50.0)
    imatter.SetMaxT(-50.0)
    jet.AddPartonShowerGenerator(core.IsrShowerPSG())
    jet.Add(imatter)
    isr.Add(jet)
    gun = core.create_module("PythiaIsrGun")
    gun.Add(isr)
    gun.Add(core.create_module("MCGlauberGenString"))
    assert isinstance(gun, core.HardProcess)
    assert type(gun.GetTaskAt(0)).__name__ == "IsrManager"
    assert isinstance(isr.GetTaskAt(0).GetTaskAt(0), core.iMATTER)
    assert isinstance(jet.GetPartonShowerGenerator(), core.IsrShowerPSG)


def test_a_python_shower_generator():
    class Gen(core.PartonShowerGenerator):
        def __init__(self):
            super().__init__()
            self.calls = []

        def DoShower(self, jloss):
            self.calls.append(jloss.GetId())

    gen = Gen()
    jl = core.JetEnergyLoss()
    jl.SetId("one")
    jl.AddPartonShowerGenerator(gen)
    core.PartonShowerGenerator.DoShower(jl.GetPartonShowerGenerator(), jl)   # through C++
    assert gen.calls == ["one"]


def test_cpp_writers_from_python():
    w = core.create_module("JetScapeWriterFinalStateHadronsAscii")
    assert isinstance(w, core.JetScapeWriter)
    core.set_writer_output_file(w, "x.dat")
    assert w.GetOutputFileName() == "x.dat"


def _run_example(script, tmp_path):
    res = subprocess.run([sys.executable, str(EXAMPLE / script), "--outdir", str(tmp_path),
                          "--main-xml", str(MAIN_XML)],
                         capture_output=True, text=True, timeout=300, check=False)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    line = next(x for x in res.stdout.splitlines() if x.startswith("SUMMARY "))
    return json.loads(line[len("SUMMARY "):])


@pytest.mark.skipif(not MAIN_XML.exists(), reason="no X-SCAPE config next to js-contrib")
@pytest.mark.parametrize("script, gun_e", [("run_from_xml.py", 50.0), ("run_by_hand.py", 40.0)])
def test_the_example_pipelines(script, gun_e, tmp_path):
    """The XML sets <E>50 for the toy gun; by hand it takes the default register_module()
    put into the main XML, 40."""
    out = _run_example(script, tmp_path)
    assert out["afterburner"] == [7, 7, 7]           # 3 soft + 4 jet hadrons per event
    for ev in out["writer"]:
        assert (ev["PartonShower"], ev["Hadron"], ev["sigma_gen"]) == (2, 4, 1.0)
    # the C++ ASCII writer's file, read back: 2 showers, 4 hadrons of E/2 each
    assert out["ascii"] == [[2, 4, pytest.approx(2 * gun_e), 1.0]] * 3
