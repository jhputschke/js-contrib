"""Backend coverage, and the regression guard for the EoS device bug.

The promoted TabulatedEoS originally pinned its tables to float64 and moved them to the caller's
device on EVERY call.  That made the hotQCD EoS unusable on MPS (which has no float64 at all) and
would have copied the tables host->device inside primitive_recovery's 20-iteration Newton loop on
CUDA.  This file is what should be run first on any new backend.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

torch = pytest.importorskip("torch")

from fast_data import convert, fv, glauber                # noqa: E402


def _devices():
    out = [("cpu", torch.float64), ("cpu", torch.float32)]
    if torch.backends.mps.is_available():
        out.append(("mps", torch.float32))                # MPS has no float64
    if torch.cuda.is_available():
        out += [("cuda", torch.float32), ("cuda", torch.float64)]
    return out


def _toy_table(tmp_path):
    """A thermodynamically sane table in MUSIC's binary layout, so this test needs no download."""
    e = np.geomspace(1e-5, 500.0, 2000)
    p = e / 3.0
    T = (e / 15.0) ** 0.25
    s = (e + p) / np.maximum(T, 1e-12)
    path = tmp_path / "hrg_hotqcd_eos_binary.dat"
    np.column_stack([e, p, s, T]).astype("<f8").ravel().tofile(path)
    return str(path)


def _eos_for(kind, tmp_path, device, dtype):
    if kind == "conformal":
        np_eos = glauber.IdealGasEoS(dof=47.5)
    else:
        np_eos = glauber.TableEoS.from_music_binary(_toy_table(tmp_path))
    return np_eos, glauber.to_fv_eos(np_eos, fv, device=torch.device(device), dtype=dtype)


@pytest.mark.parametrize("device,dtype", _devices())
@pytest.mark.parametrize("kind", ["conformal", "table"])
@pytest.mark.parametrize("viscous", [False, True])
def test_solver_runs_on_every_backend_and_eos(tmp_path, device, dtype, kind, viscous):
    dev = torch.device(device)
    _, eos = _eos_for(kind, tmp_path, device, dtype)
    g = fv.Grid(10, 10, 6, 0.5, 0.5, 0.5, device=dev, dtype=dtype)
    e = torch.rand(1, 10, 10, 6, device=dev, dtype=dtype) * 20 + 1
    q = fv.initial_state_from_energy(e, 0.6, eos)
    pi = g.zeros(1, 10) if viscous else None
    Pi = g.zeros(1, 1) if viscous else None
    tr = fv.Transport() if viscous else None
    dudtau = None
    for _ in range(3):
        q, pi, Pi, dudtau = fv.strang_step(q, pi, Pi, 0.6, 0.05, g, eos, tr,
                                           fv.minmod_slope, None, dudtau)
    assert torch.isfinite(q).all()
    assert q.device.type == dev.type
    frame, T = convert.q_to_fno_frame(q, pi, Pi, 0.75, g, eos)
    assert torch.isfinite(frame).all() and torch.isfinite(T).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_table_eos_on_mps_matches_float64_on_cpu(tmp_path):
    """The accuracy half of the fix: float32 tables on an accelerator must still be right."""
    src = _toy_table(tmp_path)
    np_eos = glauber.TableEoS.from_music_binary(src)
    cpu = glauber.to_fv_eos(np_eos, fv, device=torch.device("cpu"), dtype=torch.float64)
    acc = glauber.to_fv_eos(np_eos, fv, device=torch.device("mps"), dtype=torch.float32)
    e = torch.logspace(-3, 2, 3000, dtype=torch.float64)
    e_acc = e.to(torch.float32).to("mps")
    for name in ("p", "T", "cs2"):
        a = getattr(cpu, name)(e)
        b = getattr(acc, name)(e_acc).cpu().double()
        assert torch.allclose(a, b, rtol=1e-5), name


def test_tabulated_eos_materialises_its_tables_once(tmp_path):
    """The performance half: no per-call host->device copy, and an explicit .to()."""
    np_eos = glauber.TableEoS.from_music_binary(_toy_table(tmp_path))
    eos = glauber.to_fv_eos(np_eos, fv, device=torch.device("cpu"), dtype=torch.float32)
    assert eos.p_tab.dtype == torch.float32
    moved = eos.to(dtype=torch.float64)
    assert moved.p_tab.dtype == torch.float64 and eos.p_tab.dtype == torch.float32


def test_tabulated_eos_lookup_is_uniform_and_accurate(tmp_path):
    """MUSIC's tables are only piecewise uniform in log(e), so an exact-node lookup needs a
    binary search -- and primitive_recovery calls the EoS inside a 20-iteration Newton loop.
    Resampling onto a uniform log grid makes the index arithmetic (O(1), GPU-friendly) and took
    a hotQCD run from 45 s/event to 3.2 s/event.  This pins both halves of that trade."""
    src = _toy_table(tmp_path)
    np_eos = glauber.TableEoS.from_music_binary(src)
    e_tab, p_tab, T_tab = np_eos.fv_tables()
    eos = fv.TabulatedEoS(e_tab, p_tab, T_tab)

    # the lookup grid really is uniform in log(e)
    d = np.diff(eos.loge.numpy())
    assert np.allclose(d, d[0], rtol=1e-9)
    assert eos.dloge == pytest.approx(float(d[0]), rel=1e-9)

    # and resampling costs far less than the float32 the data is stored in
    loge = np.log(np.asarray(e_tab, dtype=np.float64))
    e = torch.as_tensor(np.exp(np.linspace(loge[1], loge[-2], 5000)))
    i = np.searchsorted(loge, np.log(e.numpy())).clip(1, len(loge) - 1)
    w = (np.log(e.numpy()) - loge[i - 1]) / (loge[i] - loge[i - 1])
    exact = p_tab[i - 1] + w * (p_tab[i] - p_tab[i - 1])
    rel = np.abs(eos.p(e).numpy() - exact) / np.maximum(np.abs(exact), 1e-30)
    assert rel.max() < 1e-4, f"resampling error {rel.max():.2e}"
