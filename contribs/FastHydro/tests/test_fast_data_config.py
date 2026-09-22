"""Gates for the YAML schema, its validation and the dotted overrides.

Stdlib-only and the fastest file here.  A data-generation run is expensive, so a typo that
silently does nothing is worse than a crash: every key is checked and every override path must
already exist.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from fast_data import config as C            # noqa: E402
from fast_data import partons                # noqa: E402

yaml = pytest.importorskip("yaml")


def _write(tmp_path, d, name="c.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(d))
    return str(p)


def test_defaults_fill_in_and_tau_grid_is_derived(tmp_path):
    cfg = C.load_config(_write(tmp_path, {"run": {"nevents": 3}}))
    assert cfg["run"]["nevents"] == 3
    assert cfg["grid"]["nx"] == C.DEFAULTS["grid"]["nx"]
    taus = C.derived_tau_grid(cfg)
    assert len(taus) == cfg["time"]["choose_ntau"]
    assert taus[0] == cfg["time"]["tau0"]
    assert cfg["time"]["tau_end"] == pytest.approx(taus[-1])


def test_unknown_keys_are_rejected_by_name(tmp_path):
    with pytest.raises(C.ConfigError, match="nevent"):
        C.load_config(_write(tmp_path, {"run": {"nevent": 3}}))
    with pytest.raises(C.ConfigError, match="grod"):
        C.load_config(_write(tmp_path, {"grod": {"nx": 3}}))


def test_overrides_must_name_an_existing_path(tmp_path):
    p = _write(tmp_path, {})
    cfg = C.load_config(p, ["run.nevents=7", "grid.nx=16"])
    assert cfg["run"]["nevents"] == 7 and cfg["grid"]["nx"] == 16
    with pytest.raises(C.ConfigError, match="does not exist"):
        C.load_config(p, ["run.nevent=7"])
    with pytest.raises(C.ConfigError, match="does not exist"):
        C.load_config(p, ["nope.deep.path=1"])


def test_overrides_parse_yaml_values(tmp_path):
    cfg = C.load_config(_write(tmp_path, {}), ["time.dtau_max=0.05", "initial_state.b=[2.0, 6.0]",
                                               "run.log=false"])
    assert cfg["time"]["dtau_max"] == 0.05
    assert cfg["initial_state"]["b"] == [2.0, 6.0]
    assert cfg["run"]["log"] is False


def test_exactly_one_normalisation_is_required(tmp_path):
    with pytest.raises(C.ConfigError, match="exactly one"):
        C.load_config(_write(tmp_path, {"initial_state": {"K": 1.0}}))   # K *and* the default T
    cfg = C.load_config(_write(tmp_path, {"initial_state": {"K": 1.0, "target_T": None}}))
    assert cfg["initial_state"]["K"] == 1.0
    with pytest.raises(C.ConfigError, match="exactly one"):
        C.load_config(_write(tmp_path, {"initial_state": {"target_T": None}}))


def test_tau_end_and_choose_ntau_must_agree(tmp_path):
    cfg = C.load_config(_write(tmp_path, {"time": {"tau0": 0.6, "record_dtau": 0.1,
                                                   "tau_end": 1.5, "choose_ntau": None}}))
    assert cfg["time"]["choose_ntau"] == 10
    with pytest.raises(C.ConfigError, match="implies"):
        C.load_config(_write(tmp_path, {"time": {"tau0": 0.6, "record_dtau": 0.1,
                                                 "tau_end": 1.5, "choose_ntau": 99}}))


def test_mps_requires_float32(tmp_path):
    with pytest.raises(C.ConfigError, match="float32"):
        C.load_config(_write(tmp_path, {"run": {"device": "mps", "dtype": "float64"}}))
    C.load_config(_write(tmp_path, {"run": {"device": "mps", "dtype": "float32"}}))


def test_xscape_mode_requires_the_fixed_step_the_cpp_assumes(tmp_path):
    """conservative mode imposes nothing (the deposit is step-size independent); xscape snaps to
    the hydro node grid and so needs the C++'s fixed step."""
    base = {"source": {"enabled": True, "mode": "xscape",
                       "partons": {"x": 0, "y": 0, "eta": 0, "tau": 0.6,
                                   "E": 1, "px": 1, "py": 0, "pz": 0}}}
    with pytest.raises(C.ConfigError, match="hydro_dtau"):
        C.load_config(_write(tmp_path, base))
    bad = dict(base, time={"hydro_dtau": 0.03})
    with pytest.raises(C.ConfigError, match="=="):
        C.load_config(_write(tmp_path, bad))
    # record_dtau must be an integer multiple of the fixed step
    bad2 = dict(base, time={"hydro_dtau": 0.02, "record_dtau": 0.03})
    with pytest.raises(C.ConfigError, match="multiple"):
        C.load_config(_write(tmp_path, bad2))
    # and the step must respect the CFL bound, which is tightest at tau0 because the eta
    # light speed is 1/tau
    bad3 = dict(base, time={"hydro_dtau": 0.02, "tau0": 0.02}, grid={"deta": 0.3})
    with pytest.raises(C.ConfigError, match="CFL"):
        C.load_config(_write(tmp_path, bad3))
    # the valid combination passes
    C.load_config(_write(tmp_path, dict(base, time={"hydro_dtau": 0.02, "record_dtau": 0.1})))


def test_source_enabled_without_partons_is_an_error(tmp_path):
    with pytest.raises(C.ConfigError, match="partons"):
        C.load_config(_write(tmp_path, {"source": {"enabled": True}}))


def test_batching_with_a_source_is_refused(tmp_path):
    with pytest.raises(C.ConfigError, match="batch_size"):
        C.load_config(_write(tmp_path, {
            "run": {"batch_size": 4},
            "source": {"enabled": True, "partons": {"x": 0, "y": 0, "eta": 0, "tau": 0.6,
                                                    "E": 1, "px": 1, "py": 0, "pz": 0}}}))


def test_seed_streams_are_independent():
    """Editing the source block must not perturb the initial conditions, so a with-source and a
    without-source dataset stay matched event for event."""
    ic_a, src_a = partons.seed_streams(12345, 10)
    ic_b, src_b = partons.seed_streams(12345, 10)
    assert (ic_a == ic_b).all() and (src_a == src_b).all()
    assert not (ic_a == src_a).any()
    ic_c, _ = partons.seed_streams(999, 10)
    assert not (ic_a == ic_c).any()


def test_auto_tau_window_reports_an_empty_window(tmp_path):
    from fast_data.liquefier import LiquefierParams
    p = LiquefierParams(tau_delay=2.0)
    assert partons.auto_tau_window(0.58, 4.0, p) is not None
    # d+Au can freeze out at 1.98 fm.  With tau_delay = 2.0 every deposit inside that fireball
    # would need a parton produced at negative proper time, so there is no window at all.
    assert partons.auto_tau_window(0.58, 1.98, p) is None
    # a shorter delay makes the same event usable, which is why the jet config ships tau_delay=1
    assert partons.auto_tau_window(0.58, 1.98, LiquefierParams(tau_delay=1.0)) is not None
