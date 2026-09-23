"""The vendored fast_data tree must be byte-identical to what UPSTREAM.json records.

Hard rule (VENDORING.md): no local patches to python/fast_data/.  Every adaptation belongs in
python/fasthydro/.  This test is what makes that rule enforceable rather than aspirational --
without it, a "quick fix" to a vendored file would survive until the next re-sync silently
reverted it.
"""

import hashlib
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent / "python" / "fast_data"
MANIFEST = ROOT / "UPSTREAM.json"


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.exists():
        pytest.skip(f"{MANIFEST} missing; run tools/sync_fast_data.sh")
    return json.loads(MANIFEST.read_text())


def test_manifest_records_provenance(manifest):
    up = manifest["upstream"]
    assert up["repo"] and up["path"]
    assert len(up["commit"]) == 40, "upstream commit must be a full sha"
    assert manifest["files"], "manifest lists no files"


def test_no_local_patches_declared(manifest):
    assert manifest["local_patches"] == [], (
        "UPSTREAM.json declares local patches to the vendored tree: "
        f"{manifest['local_patches']}. See VENDORING.md -- adaptations go in fasthydro/.")


def test_every_file_matches_its_checksum(manifest):
    drifted, missing = [], []
    for rel, rec in manifest["files"].items():
        f = ROOT / rel
        if not f.exists():
            missing.append(rel)
            continue
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != rec["sha256"]:
            drifted.append(rel)
    assert not missing, f"vendored files missing: {missing}"
    assert not drifted, (
        f"vendored files edited locally: {drifted}. "
        "Make the change upstream in FNO4d and re-run tools/sync_fast_data.sh, "
        "or put the adaptation in python/fasthydro/ instead (VENDORING.md).")


def test_no_extra_files_snuck_in(manifest):
    on_disk = {
        str(f.relative_to(ROOT))
        for f in ROOT.rglob("*")
        if f.is_file() and f.name != "UPSTREAM.json" and "__pycache__" not in f.parts
    }
    extra = on_disk - set(manifest["files"])
    assert not extra, f"files in the vendored tree that the manifest does not know: {extra}"
