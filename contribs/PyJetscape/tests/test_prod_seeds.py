"""
tests/test_prod_seeds.py

Seeds and file names of the production drivers (example/prod_AuAu_0_10/run_prod.py, shared by
run_prod_jet.py) and of run_jobs.sh: --seed 0 draws a unique seed from OS entropy through the
seed registry, --campaign names the files.  Nothing here needs the compiled extension.

    pytest tests/test_prod_seeds.py -q
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "example"


def _run_prod():
    spec = importlib.util.spec_from_file_location("_run_prod_t", EXAMPLE / "prod_AuAu_0_10" /
                                                  "run_prod.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rp = _run_prod()


def _args(tmp_path, **kw):
    a = dict(seed=0, campaign=None, index=1, out=None, seed_registry=None,
             outdir=str(tmp_path / "out"))
    a.update(kw)
    return SimpleNamespace(**a)


def _registry(path):
    return [ln.split("\t") for ln in Path(path).read_text().splitlines()
            if not ln.startswith("#")]


def test_output_names():
    a = SimpleNamespace(seed=7, campaign=None, index=1, out=None)
    assert rp.output_name(a, "AuAu_0_10") == "AuAu_0_10_seed0007.h5"
    a = SimpleNamespace(seed=7, campaign="pth50", index=12, out=None)
    assert rp.output_name(a, "AuAu_0_10_jet") == "AuAu_0_10_jet_pth50_0012.h5"
    a = SimpleNamespace(seed=0, campaign=None, index=3, out=None)
    name = rp.output_name(a, "AuAu_0_10")          # seed 0 without a campaign: the start time
    assert a.campaign and len(a.campaign) == 15 and name == f"AuAu_0_10_{a.campaign}_0003.h5"
    a = SimpleNamespace(seed=0, campaign=None, index=1, out="mine.h5")
    assert rp.output_name(a, "AuAu_0_10") == "mine.h5" and a.campaign   # still recorded


def test_seed_range_and_campaign_names_are_checked():
    ok = SimpleNamespace(seed=rp.SEED_MAX, campaign="a.b_c-1", index=1)
    assert rp.seed_args_problems(ok) == []
    assert rp.seed_args_problems(SimpleNamespace(seed=0, campaign=None, index=1)) == []
    for bad in (dict(seed=rp.SEED_MAX + 1), dict(seed=2026092622), dict(seed=-1),
                dict(campaign="a/b"), dict(campaign="-x"), dict(index=0)):
        a = SimpleNamespace(**dict(dict(seed=1, campaign=None, index=1), **bad))
        assert rp.seed_args_problems(a), bad


def test_seed_zero_draws_from_entropy_and_avoids_the_registry(tmp_path, monkeypatch):
    import secrets

    reg = tmp_path / "seeds_used.tsv"
    a = _args(tmp_path, seed=5, campaign="x")
    rp.resolve_seed(a, str(tmp_path / "out" / "f5.h5"))
    assert a.seed == 5 and a.seed_source == "explicit"
    draws = iter([4, 4, 99])                            # randbelow + 1: 5 (taken), 5, 100
    monkeypatch.setattr(secrets, "randbelow", lambda n: next(draws))
    b = _args(tmp_path, campaign="x", index=2)
    rp.resolve_seed(b, str(tmp_path / "out" / "x_0002.h5"))
    assert b.seed == 100 and b.seed_source == "os_entropy"
    rows = _registry(reg)
    assert [(int(r[0]), r[1], r[2]) for r in rows] == [(5, "explicit", "x"),
                                                       (100, "os_entropy", "x")]
    assert rp.seed_provenance(b) == {"prod_seed": 100, "prod_seed_source": "os_entropy",
                                     "prod_campaign": "x", "prod_index": 2}


def test_explicit_seed_reuse_is_reported_not_refused(tmp_path, capsys):
    rp.resolve_seed(_args(tmp_path, seed=3), str(tmp_path / "out" / "a.h5"))
    rp.resolve_seed(_args(tmp_path, seed=3), str(tmp_path / "out" / "a.h5"))   # a restart
    assert capsys.readouterr().err == "" and len(_registry(tmp_path / "seeds_used.tsv")) == 1
    other = _args(tmp_path, seed=3, outdir=str(tmp_path / "out_b"))
    rp.resolve_seed(other, str(tmp_path / "out_b" / "a.h5"))
    assert "seed 3 was used before" in capsys.readouterr().err and other.seed == 3
    assert len(_registry(tmp_path / "seeds_used.tsv")) == 2


def test_dry_run_and_no_registry_record_nothing(tmp_path):
    a = _args(tmp_path)
    rp.resolve_seed(a, str(tmp_path / "out" / "x.h5"), record=False)
    reg = tmp_path / "seeds_used.tsv"
    assert 1 <= a.seed <= rp.SEED_MAX and (not reg.exists() or _registry(reg) == [])
    b = _args(tmp_path, seed_registry="none")
    rp.resolve_seed(b, str(tmp_path / "out" / "y.h5"))
    assert 1 <= b.seed <= rp.SEED_MAX and b.seed_source == "os_entropy"


def _draw(args):
    reg, i = args
    mod = _run_prod()
    a = SimpleNamespace(seed=0, campaign="c", index=i, out=None, seed_registry=reg,
                        outdir=os.path.dirname(reg))
    mod.resolve_seed(a, os.path.join(os.path.dirname(reg), f"c_{i:04d}.h5"))
    return a.seed


def test_jobs_starting_together_get_different_seeds(tmp_path):
    reg = str(tmp_path / "seeds_used.tsv")
    with mp.get_context("spawn").Pool(8) as pool:
        seeds = pool.map(_draw, [(reg, i) for i in range(1, 33)])
    rows = _registry(reg)
    assert len(set(seeds)) == 32 and len(rows) == 32         # none lost under the lock
    assert sorted(int(r[0]) for r in rows) == sorted(seeds)


# ─────────────────────────────────────────────────────────────── run_jobs.sh with a stub
STUB = r'''#!/usr/bin/env python3
"""Stands in for run_prod.py: records its arguments and writes a complete .json."""
import argparse, json, os, sys
p = argparse.ArgumentParser()
p.add_argument("--events", type=int); p.add_argument("--seed", type=int)
p.add_argument("--campaign"); p.add_argument("--index", type=int, default=1)
p.add_argument("--outdir"); p.add_argument("--fail", action="store_true")
a, extra = p.parse_known_args()
name = (f"AuAu_0_10_{a.campaign}_{a.index:04d}" if a.campaign else f"AuAu_0_10_seed{a.seed:04d}")
with open(os.path.join(a.outdir, "calls.txt"), "a") as fh:
    fh.write(json.dumps({"name": name, "seed": a.seed, "extra": extra}) + "\n")
if a.fail and a.index == 2:
    sys.exit(1)
json.dump({"events_written": a.events}, open(os.path.join(a.outdir, name + ".json"), "w"))
'''


def _run_jobs(tmp_path, *args):
    stub = tmp_path / "stub_prod.py"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PROD_SCRIPT=str(stub), PYTHIA8DATA="unused",
               PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])
    return subprocess.run(["/bin/bash", str(EXAMPLE / "prod_AuAu_0_10" / "run_jobs.sh"), *args],
                          capture_output=True, text=True, env=env)


def _calls(out):
    p = Path(out) / "calls.txt"
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_run_jobs_campaign_names_resume_and_mismatch(tmp_path):
    out = tmp_path / "out"
    r = _run_jobs(tmp_path, "--campaign", "mb_a", "3", "5", "0", str(out), "--grid", "g.yaml")
    assert r.returncode == 0, r.stderr
    calls = _calls(out)
    assert [c["name"] for c in calls] == [f"AuAu_0_10_mb_a_000{i}" for i in (1, 2, 3)]
    assert all(c["seed"] == 0 and c["extra"] == ["--grid", "g.yaml"] for c in calls)
    assert (out / "run_jobs.campaign").read_text().strip() == "mb_a"
    assert "campaign mb_a, jobs 1..3, seeds from OS entropy" in \
        (out / "run_jobs.finished").read_text()
    assert "job 0002: ok" in r.stdout
    # a re-run without --campaign resumes it: everything complete, nothing started
    r = _run_jobs(tmp_path, "4", "5", "0", str(out))
    assert r.returncode == 0 and [c["name"] for c in _calls(out)][3:] == ["AuAu_0_10_mb_a_0004"]
    assert r.stdout.count("already complete") == 3
    # another campaign name in the same OUTDIR is refused
    r = _run_jobs(tmp_path, "--campaign", "other", "1", "5", "0", str(out))
    assert r.returncode == 2 and "holds campaign 'mb_a'" in r.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_run_jobs_default_campaign_explicit_seeds_and_failures(tmp_path):
    out = tmp_path / "out_t"
    r = _run_jobs(tmp_path, "2", "5", "0", str(out))
    assert r.returncode == 0, r.stderr
    camp = (out / "run_jobs.campaign").read_text().strip()
    assert len(camp) == 15 and camp[8] == "-"                    # YYYYMMDD-HHMMSS
    assert [c["name"] for c in _calls(out)] == [f"AuAu_0_10_{camp}_0001",
                                                f"AuAu_0_10_{camp}_0002"]
    out = tmp_path / "out_s"                                     # explicit seeds: as before
    r = _run_jobs(tmp_path, "2", "5", "11", str(out))
    assert r.returncode == 0 and not (out / "run_jobs.campaign").exists()
    assert [(c["name"], c["seed"]) for c in _calls(out)] == [("AuAu_0_10_seed0011", 11),
                                                            ("AuAu_0_10_seed0012", 12)]
    assert "seeds 11..12" in (out / "run_jobs.finished").read_text()
    out = tmp_path / "out_f"                                     # a failed job is reported
    r = _run_jobs(tmp_path, "--campaign", "f", "3", "5", "0", str(out), "--fail")
    assert r.returncode == 1 and "job 0002: FAILED" in r.stdout
    assert "failed: job 0002" in (out / "run_jobs.finished").read_text()
