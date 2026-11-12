"""utils/h5_compare.py: what counts as the same, what is reported, the exit codes."""
import importlib.util
import os
import shutil

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "utils", "h5_compare.py"))


@pytest.fixture(scope="module")
def h5c():
    spec = importlib.util.spec_from_file_location("h5_compare", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path, *, arr=None, wall=1.0, uuid="aaaa", extra=None, label="x", keys=("k0", "k1")):
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as h:
        h.create_dataset("arr", data=rng.normal(size=(6, 4, 5)).astype("f4") if arr is None
                         else arr)
        h.create_dataset("diag/wall_s", data=np.full(6, wall))
        h.create_dataset("diag/n", data=np.arange(6))
        h.create_dataset("keys", data=np.array(keys, dtype=object),
                         dtype=h5py.string_dtype())
        h.attrs["nevents"] = 6
        h.attrs["file_uuid"] = uuid
        h.attrs["prod_wall_s_total"] = wall * 6
        h["diag"].attrs["label"] = label
        if extra is not None:
            h.create_dataset("extra", data=extra)
    return path


def _run(h5c, capsys, *args):
    code = h5c.main([str(a) for a in args])
    return code, capsys.readouterr().out


def test_copy_is_the_same(h5c, tmp_path, capsys):
    a = _write(tmp_path / "a.h5")
    b = shutil.copy(a, tmp_path / "b.h5")
    code, out = _run(h5c, capsys, a, b)
    assert code == 0 and "the same" in out and "4 identical, 0 differ" in out


def test_stamps_differ_unless_ignored(h5c, tmp_path, capsys):
    a = _write(tmp_path / "a.h5")
    b = _write(tmp_path / "b.h5", wall=2.0, uuid="bbbb")
    code, out = _run(h5c, capsys, a, b)
    assert code == 1
    assert "diag/wall_s" in out and "6 of 6 values differ" in out
    assert "@file_uuid" in out and "aaaa | bbbb" in out and "@prod_wall_s_total" in out
    code, out = _run(h5c, capsys, a, b, "--ignore-stamps")
    assert code == 0 and "ignored:    3" in out
    code, out = _run(h5c, capsys, a, b, "--ignore", "*wall_s*", "--ignore", "@file_uuid", "-v")
    assert code == 0 and "diag/wall_s, @file_uuid, @prod_wall_s_total" in out


def test_values_shape_missing(h5c, tmp_path, capsys):
    arr = np.arange(120, dtype="f4").reshape(6, 4, 5)
    arr2 = arr.copy()
    arr2[4, 1, 2] += 0.5
    a = _write(tmp_path / "a.h5", arr=arr, extra=np.zeros(3))
    b = _write(tmp_path / "b.h5", arr=arr2, label="y", keys=("k0", "other"))
    code, out = _run(h5c, capsys, a, b, "--block-mb", "1")
    assert code == 1
    assert "1 of 120 values differ, max |a-b| 0.5" in out and "first at [4, 1, 2]" in out
    assert "extra" in out and "only in A" in out
    assert "keys" in out and "1 of 2 values differ" in out
    assert "diag@label" in out and "x | y" in out
    c = _write(tmp_path / "c.h5", arr=arr[:5])
    code, out = _run(h5c, capsys, a, c, "--no-attrs")
    assert code == 1 and "shape (6, 4, 5) | (5, 4, 5)" in out and "attributes" not in out


def test_blocks_find_a_late_difference(h5c, tmp_path, capsys):
    arr = np.zeros((40, 300, 300), dtype="f4")          # 14 MB: 14 blocks of 1 MB
    arr2 = arr.copy()
    arr2[37, 299, 0] = 1.0
    a = _write(tmp_path / "a.h5", arr=arr)
    b = _write(tmp_path / "b.h5", arr=arr2)
    code, out = _run(h5c, capsys, a, b, "--block-mb", "1")
    assert code == 1 and "first at [37, 299, 0]" in out


def test_tolerance_nan_and_signed_zero(h5c, tmp_path, capsys):
    arr = np.linspace(1, 2, 120).reshape(6, 4, 5)
    arr[0, 0, 0] = np.nan
    a = _write(tmp_path / "a.h5", arr=arr)
    b = _write(tmp_path / "b.h5", arr=arr * (1 + 1e-9))
    assert _run(h5c, capsys, a, b)[0] == 1
    code, out = _run(h5c, capsys, a, b, "--rtol", "1e-6")
    assert code == 0 and "within tolerance" in out
    code, out = _run(h5c, capsys, a, b, "--rtol", "1e-12")
    assert code == 1 and "outside the tolerance" in out
    zero = _write(tmp_path / "z0.h5", arr=np.zeros((6, 4, 5)))
    minus = _write(tmp_path / "z1.h5", arr=-np.zeros((6, 4, 5)))
    code, out = _run(h5c, capsys, zero, minus)
    assert code == 1 and "bit patterns differ, values equal" in out
    ints = _write(tmp_path / "i0.h5", arr=np.arange(120).reshape(6, 4, 5))
    ints2 = _write(tmp_path / "i1.h5", arr=np.arange(120).reshape(6, 4, 5) + 1)
    assert _run(h5c, capsys, ints, ints2, "--rtol", "10")[0] == 1      # integers: always exact


def test_directories_and_exit_codes(h5c, tmp_path, capsys):
    da, db = tmp_path / "A", tmp_path / "B"
    (da / "sub").mkdir(parents=True)
    (db / "sub").mkdir(parents=True)
    _write(da / "one.h5")
    _write(db / "one.h5")
    _write(da / "sub" / "two.h5")
    _write(db / "sub" / "two.h5", wall=3.0)
    _write(da / "three.h5")
    code, out = _run(h5c, capsys, da, db, "-q")
    assert code == 1
    assert "one.h5: the same" in out and "two.h5: DIFFERENT" in out
    assert "three.h5: only in A" in out and "2 pairs compared: 1 the same, 1 different" in out
    assert _run(h5c, capsys, da / "one.h5", db)[0] == 0            # file against directory
    assert h5c.main([str(da / "one.h5"), str(tmp_path / "nothing.h5")]) == 2
    (tmp_path / "bad.h5").write_text("not hdf5")
    assert h5c.main([str(da / "one.h5"), str(tmp_path / "bad.h5")]) == 2
