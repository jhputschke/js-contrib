"""The Python side of FastHydro's soft particlization (fasthydro/particlization.py).

The surface and the sampling are C++ (SurfaceFinder, iSS) and are tested there -- PyJetscape's
tests/test_surface_finder.py for the surface.  What is tested here is the glue that decides
whether a run can give a meaningful surface at all: the config checks, the XML reading, the
music_input iSS insists on, and the closure report.  None of it needs an X-SCAPE build.
"""
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG = ROOT / "config"


def _cfg(name):
    from fasthydro.config import load_config
    return load_config(CFG / name)


def test_shipped_pair_passes():
    from fasthydro.pipeline import check_xml_agrees_with_cfg
    check_xml_agrees_with_cfg(str(CFG / "jetscape_user_fasthydro_particlize.xml"),
                              _cfg("fasthydro_particlize.yaml"))


def test_parton_level_pair_still_passes():
    """No <SoftParticlization> block: nothing changes for the existing configs."""
    from fasthydro.pipeline import check_xml_agrees_with_cfg
    check_xml_agrees_with_cfg(str(CFG / "jetscape_user_fasthydro_wake.xml"),
                              _cfg("fasthydro_wake.yaml"))


@pytest.mark.parametrize("mutate,expect", [
    (lambda c: c["output"].__setitem__("zero_after_freezeout", True), "zero_after_freezeout"),
    (lambda c: c["output"].__setitem__("stop_at_freezeout", True), "stop_at_freezeout"),
    (lambda c: c["eos"].__setitem__("kind", "conformal"), "hotqcd"),
    # the SMASH list without SMASH: iSS would not decay the resonances
    (lambda c: c["eos"].__setitem__("kind", "hotqcd_smash"), "Afterburner"),
])
def test_settings_that_fake_a_surface_are_refused(mutate, expect):
    from fasthydro.pipeline import check_xml_agrees_with_cfg
    cfg = _cfg("fasthydro_particlize.yaml")
    mutate(cfg)
    with pytest.raises(ValueError, match=expect):
        check_xml_agrees_with_cfg(str(CFG / "jetscape_user_fasthydro_particlize.xml"), cfg)


def test_smash_list_needs_smash_and_vice_versa():
    from fasthydro.particlization import config_problems
    cfg = _cfg("fasthydro_particlize.yaml")
    assert config_problems(cfg, {"T_sw": 0.15, "afterburner": False}) == []
    assert config_problems(cfg, {"T_sw": 0.15, "afterburner": True})       # UrQMD list into SMASH
    cfg["eos"]["kind"] = "hotqcd_smash"
    assert config_problems(cfg, {"T_sw": 0.15, "afterburner": True}) == []
    assert config_problems(cfg, {"T_sw": 0.15, "afterburner": False})


def test_freezeout_never_allows_the_tail_flags():
    """With freezeout: never nothing is zeroed or stopped, whatever the flags say."""
    from fasthydro.particlization import config_problems
    cfg = _cfg("fasthydro_particlize.yaml")
    cfg["output"].update(freezeout="never", zero_after_freezeout=True, stop_at_freezeout=True)
    assert config_problems(cfg, {"T_sw": 0.15}) == []


def test_unknown_hydro_id_is_refused(tmp_path):
    from fasthydro.pipeline import check_xml_agrees_with_cfg
    xml = (CFG / "jetscape_user_fasthydro_particlize.xml").read_text()
    bad = tmp_path / "bad.xml"
    bad.write_text(xml.replace("<hydro_id>FastHydro_jet</hydro_id>",
                               "<hydro_id>MUSIC</hydro_id>"))
    with pytest.raises(ValueError, match="hydro_id"):
        check_xml_agrees_with_cfg(str(bad), _cfg("fasthydro_particlize.yaml"))


def test_read_xml():
    from fasthydro.particlization import read_xml
    assert read_xml(str(CFG / "jetscape_user_fasthydro_wake.xml")) is None
    s = read_xml(str(CFG / "jetscape_user_fasthydro_particlize.xml"))
    assert s["hydro_id"] == "FastHydro_jet"
    assert s["T_sw"] == pytest.approx(0.15)
    assert (s["surface_dtau"], s["surface_dx"], s["surface_deta"]) == (0.2, 0.625, 0.625)
    assert s["n_oversample"] == 20
    assert s["iss_working_path"] == "./fasthydro_iss"
    assert s["afterburner"] is False


def test_music_input_is_written_and_checked(tmp_path):
    from fasthydro.particlization import ISS_EOS, write_iss_music_input
    assert ISS_EOS == {"hotqcd": 9, "hotqcd_smash": 91}
    p = write_iss_music_input(str(tmp_path / "iss"), 9)
    text = pathlib.Path(p).read_text()
    assert "EOS_to_use 9\n" in text and "Include_Bulk_Visc_Yes_1_No_0 0" in text
    # ours: rewritten for another EoS rather than refused
    write_iss_music_input(str(tmp_path / "iss"), 91)
    assert "EOS_to_use 91" in pathlib.Path(p).read_text()

    other = tmp_path / "other"
    other.mkdir()
    (other / "music_input").write_text("EOS_to_use 91\n")           # someone else's MUSIC run
    assert write_iss_music_input(str(other), 91)                     # same EoS: kept as is
    with pytest.raises(ValueError, match="EOS_to_use 9"):
        write_iss_music_input(str(other), 9)


class _EoS:
    """T(e) = (e / 10)^(1/4): monotonic, which is all closure_report relies on."""

    def T(self, e):
        return (e / 10.0) ** 0.25


@pytest.mark.parametrize("hot,closed", [
    (None, True),
    ("last_frame", False),
    ("x_boundary", False),
    ("eta_boundary", True),          # open in eta by construction; reported, not fatal
])
def test_closure_report(hot, closed):
    pytest.importorskip("torch")
    from fasthydro.particlization import closure_report

    T_sw = 0.15
    e_cold, e_hot = 10.0 * 0.10 ** 4, 10.0 * 0.20 ** 4               # T = 0.10, 0.20 GeV
    arr = np.zeros((4, 9, 9, 7, 12), dtype=np.float32)
    arr[0] = e_cold
    arr[0, 3:6, 3:6, 2:5, :6] = e_hot                              # a fireball that cools
    if hot == "last_frame":
        arr[0, 4, 4, 3, -1] = e_hot
    elif hot == "x_boundary":
        arr[0, 0, 4, 3, 2] = e_hot
    elif hot == "eta_boundary":
        arr[0, 4, 4, 0, 2] = e_hot
    r = closure_report(arr, _EoS(), T_sw)
    assert r["closed"] is closed
    assert r["T_max_eta_boundary"] == pytest.approx(0.20 if hot == "eta_boundary" else 0.10,
                                                    rel=1e-5)
