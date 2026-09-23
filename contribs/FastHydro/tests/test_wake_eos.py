"""How make_wake_data.py finds (or fetches) the hotQCD table.

The script used to fall back on a copy inside an FNO4d checkout, which made a js-contrib
script depend on an unrelated repo being present. It now looks for a copy the machine already
has and otherwise downloads one. These cover everything except the network path, which is
exercised separately rather than in the suite.
"""

import os
import pathlib
import sys

import pytest

EXAMPLE = pathlib.Path(__file__).resolve().parent.parent / "example"
sys.path.insert(0, str(EXAMPLE))

mwd = pytest.importorskip("make_wake_data")

RECORD = 32


def _fake_table(path, rows=1000):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * (rows * RECORD))
    return path


def test_a_whole_table_is_accepted(tmp_path):
    t = _fake_table(tmp_path / "t.dat")
    assert mwd._looks_like_a_table(t)


@pytest.mark.parametrize("nbytes", [0, 1, RECORD - 1, RECORD + 7])
def test_a_partial_table_is_rejected(tmp_path, nbytes):
    """A truncated download loads as a short table and quietly changes the physics, so the
    size has to be a whole number of 4 x float64 records."""
    p = tmp_path / "t.dat"
    p.write_bytes(b"\x00" * nbytes)
    assert not mwd._looks_like_a_table(p)


def test_missing_file_is_rejected(tmp_path):
    assert not mwd._looks_like_a_table(tmp_path / "nope.dat")


def test_an_existing_table_is_reused_not_refetched(tmp_path):
    """No network: the file is already where the config expects it."""
    dest = tmp_path / mwd.EOS_SUBDIR
    _fake_table(dest / mwd.EOS_FILE)
    got = mwd.ensure_eos(str(tmp_path), allow_download=False, verbose=False)
    assert got == str(dest)


def test_xscape_own_copy_is_picked_up(tmp_path):
    """X-SCAPE ships the table under EOS/hotQCD; use it rather than downloading again."""
    _fake_table(tmp_path / "EOS" / "hotQCD" / mwd.EOS_FILE)
    got = mwd.ensure_eos(str(tmp_path), allow_download=False, verbose=False)
    assert got == str(tmp_path / mwd.EOS_SUBDIR)
    assert os.path.exists(os.path.join(got, mwd.EOS_FILE))


def test_explicit_eos_dir_wins(tmp_path):
    elsewhere = tmp_path / "somewhere"
    _fake_table(elsewhere / mwd.EOS_FILE)
    got = mwd.ensure_eos(str(tmp_path), eos_dir=str(elsewhere),
                         allow_download=False, verbose=False)
    assert os.path.exists(os.path.join(got, mwd.EOS_FILE))


def test_a_truncated_local_table_is_discarded(tmp_path, capsys):
    """Half a table in the destination must not be used just because it is there."""
    dest = tmp_path / mwd.EOS_SUBDIR
    dest.mkdir(parents=True)
    (dest / mwd.EOS_FILE).write_bytes(b"\x00" * (RECORD + 3))
    got = mwd.ensure_eos(str(tmp_path), allow_download=False, verbose=False)
    assert got is None                                   # discarded, and no network allowed
    assert not (dest / mwd.EOS_FILE).exists()
    assert "truncated" in capsys.readouterr().out


def test_no_download_says_where_it_looked(tmp_path, capsys):
    assert mwd.ensure_eos(str(tmp_path), allow_download=False, verbose=False) is None
    out = capsys.readouterr().out
    assert "no-download" in out and "download_hotqcd" in out


def test_no_fno4d_dependency():
    """The point of the change: a js-contrib script must not need an FNO4d checkout."""
    src = (EXAMPLE / "make_wake_data.py").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    body = code.split('"""', 2)[-1]                      # drop the module docstring
    assert "FNO4d" not in body, "make_wake_data.py still reaches into an FNO4d checkout"


# --------------------------------------------------------------------------- file locking
def test_an_open_output_file_is_detected_before_anything_is_truncated(tmp_path):
    """HDF5 truncates BEFORE it takes its lock, so a run started while a notebook holds the
    output destroys the old file and only then fails, with a traceback that never mentions the
    notebook. `_is_locked` is what turns that into a preflight refusal."""
    import fcntl

    p = tmp_path / "held.h5"
    p.write_bytes(b"not empty")
    assert not mwd._is_locked(str(p))

    fd = os.open(str(p), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert mwd._is_locked(str(p)), "a held file must be reported as locked"
    finally:
        os.close(fd)
    assert not mwd._is_locked(str(p)), "the lock must clear when the holder goes away"
    assert p.read_bytes() == b"not empty", "the probe must never truncate what it is probing"


def test_a_missing_file_is_not_locked(tmp_path):
    assert not mwd._is_locked(str(tmp_path / "nope.h5"))


def test_holders_never_raises(tmp_path):
    p = tmp_path / "x.h5"
    p.write_bytes(b"")
    assert isinstance(mwd._holders([str(p)]), str)
