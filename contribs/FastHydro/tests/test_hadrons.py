"""fasthydro.hadrons: the hadron-file reader, the compact npz, and the oversample averages."""
import numpy as np
import pytest

from fasthydro.hadrons import Hadrons, ascii_to_npz, load, read_ascii

HEADER = "#\tJETSCAPE_FINAL_STATE\tv2\t|\tN\tpid\tstatus\tE\tPx\tPy\tPz\n"


def _write(path, events):
    with open(path, "w") as f:
        f.write(HEADER)
        for k, rows in enumerate(events):
            f.write(f"#\tEvent\t{k + 1}\tweight\t1\tEPangle\t0\tN_hadrons\t{len(rows)}\n")
            for i, (pid, E, px, py, pz) in enumerate(rows):
                f.write(f"{i} {pid} 11 {E} {px} {py} {pz}\n")


def test_ascii_roundtrip(tmp_path):
    ev = [[(211, 1.0, 0.5, 0.0, 0.1), (-321, 2.0, 0.0, 1.0, -0.2)],
          [(2212, 1.5, 0.3, 0.3, 0.0)],
          []]
    txt, npz = tmp_path / "h.dat", tmp_path / "h.npz"
    _write(txt, ev)
    pid, p, off = read_ascii(txt)
    assert off.tolist() == [0, 2, 3, 3]
    assert pid.tolist() == [211, -321, 2212]
    assert np.allclose(p[1], [2.0, 0.0, 1.0, -0.2])
    assert ascii_to_npz(txt, npz) == 3
    h = load(npz, n_oversample=2)
    assert h.n_events == 3 and h.event.tolist() == [0, 0, 1]
    assert h.species("charged").sum() == 3
    assert h.phi[1] == pytest.approx(np.pi / 2)


def test_oversample_average_and_poisson_error():
    """hist/total divide by N_os x N_events and carry sqrt(sum w^2) / norm."""
    rng = np.random.default_rng(1)
    n_os, n_ev, lam = 50, 3, 40.0
    n = rng.poisson(lam, size=n_ev * n_os)
    pid = np.full(n.sum(), 211)
    pt = rng.exponential(0.4, size=n.sum()) + 0.05
    p = np.column_stack([np.hypot(pt, 0.14), pt, np.zeros_like(pt), np.zeros_like(pt)])
    per_event = n.reshape(n_ev, n_os).sum(axis=1)
    off = np.concatenate([[0], np.cumsum(per_event)])
    h = Hadrons(pid, p, off, n_oversample=n_os)

    N, eN = h.total()
    assert N == pytest.approx(n.sum() / (n_os * n_ev))
    assert eN == pytest.approx(np.sqrt(n.sum()) / (n_os * n_ev))
    assert abs(N - lam) < 4 * eN                      # the Poisson mean, within errors

    edges = np.linspace(0, 30, 7)                     # wide enough to hold every p_T
    hh, eh = h.hist(h.pt, edges)
    assert hh.shape == (6,) and hh.sum() == pytest.approx(N, rel=1e-12)
    S, eS = h.total(weights=h.pt)
    assert S == pytest.approx(pt.sum() / (n_os * n_ev))
    assert eS == pytest.approx(np.sqrt((pt ** 2).sum()) / (n_os * n_ev))

    one, _ = h.total(events=[1])
    assert one == pytest.approx(per_event[1] / n_os)
    h2, _ = h.hist((h.pt, h.phi), (edges, np.linspace(-np.pi, np.pi, 5)))
    assert h2.shape == (6, 4) and h2.sum() == pytest.approx(N, rel=1e-12)
