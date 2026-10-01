#!/usr/bin/env python3
"""
utils/remote_transfer/js_gcs.py

Upload and download production directories and files to / from Google Cloud Storage, with a
service-account key.  Standalone: on its first run the script makes its own virtual
environment (``~/.cache/js_gcs/venv``, plain ``python3 -m venv`` + pip) and from then on
runs itself in it, whatever environment it is started from.  It needs transfer_core.py next
to it.  js_osdf.py is the same for Pelican/OSDF.

    ./js_gcs.py upload /data/AuAu_c1 --what root          # -> gs://test_fno/AuAu_c1/
    ./js_gcs.py upload /data/AuAu_a /data/AuAu_b --what pair,h5 --prefix productions -j 8
    ./js_gcs.py upload /data/out --what root --as AuAu_v1   # -> gs://test_fno/AuAu_v1/
    ./js_gcs.py upload /data/AuAu_c1/AuAu_c1_0003_hadrons.root /data/AuAu_c1/*_0004*.h5
                                                       # files -> gs://test_fno/AuAu_c1/<file>
    ./js_gcs.py download AuAu_c1 --what root --to /scratch    # -> /scratch/AuAu_c1/
    ./js_gcs.py download AuAu_c1/AuAu_c1_0003_hadrons.root 'AuAu_c1/*_0004_*' --to here
                                                       # files -> here/<file>
    ./js_gcs.py ls                                     # the bucket's top level
    ./js_gcs.py ls AuAu_c1 --what h5                   # files, sizes, kinds
    ./js_gcs.py rm -r AuAu_c1 --dry-run                # what would be removed; then without
    ./js_gcs.py rm AuAu_c1/AuAu_c1_0003_hadrons.root 'AuAu_c1/*_0004_*'   # files, patterns
    ./js_gcs.py setup --reinstall | --remove           # remake / delete the environment

What (``--what``, a comma list; default all) -- by file name, as run_prod_jet.py,
hadronize.py and run_h5toROOT.py name them:

    pair    <stem>.h5, the hydro pair file, with its <stem>.json, .xml, .log
    h5      <stem>_particlize.h5, <stem>_hadrons_{bulk_jet,bulk_bg,jet_frag}.h5,
            <stem>_hadronize.log
    root    <stem>_hadrons.root, <campaign>_campaign.root
    all     the three, plus every other file of the directory (run_jobs.campaign,
            wake_observables.h5, ...)

Uploads go to gs://BUCKET/[PREFIX/]<directory name>/<file>: for a directory, its files of
the --what kinds (not its subdirectories); for a file named on the command line, that file
whatever its kind, to where its directory's upload puts it (--flat: [PREFIX/]<file>).
--as NAME uploads to [PREFIX/]NAME/<file> instead (sources from one local directory).
Downloads of a directory go to TO/<its name>/..., of a file or of a pattern (* ? [..],
quoted; * also matches /) to TO/<file>.  A file already there with the same size and CRC32C
is skipped (--size-only: the size alone; --force: transfer anyway), so re-running a command
finishes an interrupted transfer.  HDF5 files whose ``complete`` attribute is False (a job
still writing) are not uploaded, unless --include-incomplete; ``*.part`` files never.
Every transfer is checked against the CRC32C; downloads go to ``<file>.part`` first and are
renamed when complete.

The key: --key-file, else $JS_GCS_KEY_FILE, else $GOOGLE_APPLICATION_CREDENTIALS, else
fnotest-wayne-gcs.json in the current directory, next to this script, or in
~/.config/js_gcs/.  The bucket: --bucket, else $JS_GCS_BUCKET, else gs://test_fno.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transfer_core as core  # noqa: E402

ENV = core.Env("js_gcs", ("google-cloud-storage>=2.14", "google-crc32c>=1.5", "h5py>=3.0"),
               "JS_GCS_ENV")
DEFAULT_BUCKET = "test_fno"
KEY_NAME = "fnotest-wayne-gcs.json"


def find_key(arg):
    """The service-account key file, or exit with where it was looked for."""
    here = Path(__file__).resolve().parent
    cands = [arg, os.environ.get("JS_GCS_KEY_FILE"),
             os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
             Path.cwd() / KEY_NAME, here / KEY_NAME,
             Path("~/.config/js_gcs").expanduser() / KEY_NAME]
    for c in cands[:3]:
        if c:                                       # given explicitly: must exist
            if not Path(c).expanduser().is_file():
                sys.exit(f"js_gcs: key file {c} not found")
            return Path(c).expanduser()
    for c in cands[3:]:
        if c.is_file():
            return c
    where = ", ".join(dict.fromkeys(str(c.parent) for c in cands[3:]))
    sys.exit(f"js_gcs: no key file: pass --key-file, set JS_GCS_KEY_FILE, or put {KEY_NAME} "
             f"in one of {where}")


def bucket_name(text):
    return text.strip().removeprefix("gs://").strip("/")


class GcsStore:
    """One bucket (transfer_core's store interface), a client per thread."""

    native_checksums = True
    manifest = False

    def __init__(self, bucket, key):
        self.bucket_name, self.key = bucket, str(key)
        self.label = f"gs://{bucket}"
        self._local = threading.local()

    def bucket(self):
        if not hasattr(self._local, "bucket"):
            from google.cloud import storage
            client = storage.Client.from_service_account_json(self.key)
            self._local.bucket = client.bucket(self.bucket_name)
        return self._local.bucket

    def relative(self, text):
        t = text.strip()
        if t.startswith("gs://"):
            b, _, t = t[5:].partition("/")
            if b != self.bucket_name:
                core.die(f"{text} is not in bucket {self.label} (use --bucket)")
        return t.strip("/")

    def list(self, prefix):
        p = prefix.strip("/")
        it = self.bucket().client.list_blobs(self.bucket_name, prefix=f"{p}/" if p else None)
        return {b.name: (int(b.size), b.crc32c) for b in it if not b.name.endswith("/")}

    def stat(self, name):
        b = self.bucket().get_blob(name)
        return None if b is None else (int(b.size), b.crc32c)

    def top_level(self, prefix):
        p = prefix.strip("/")
        it = self.bucket().client.list_blobs(self.bucket_name, prefix=f"{p}/" if p else None,
                                             delimiter="/")
        files = {b.name: (int(b.size), b.crc32c) for b in it}
        return sorted(it.prefixes), files

    def upload(self, path, name):
        from google.cloud.storage.retry import DEFAULT_RETRY
        blob = self.bucket().blob(name, chunk_size=64 * 1024 * 1024)
        # retried although not idempotent in general: we write the same content again
        blob.upload_from_filename(str(path), checksum="crc32c", timeout=600,
                                  retry=DEFAULT_RETRY)

    def download(self, name, path):
        self.bucket().blob(name).download_to_filename(str(path), checksum="crc32c",
                                                      timeout=600)

    def read_bytes(self, name):
        b = self.bucket().get_blob(name)
        return None if b is None else b.download_as_bytes()

    def delete(self, name):
        from google.api_core.exceptions import NotFound
        try:
            self.bucket().blob(name).delete(timeout=60)
        except NotFound:
            raise FileNotFoundError(f"{self.label}/{name}") from None

    def remove_dir(self, name):
        """GCS has no directories: only the placeholder object 'name/' some tools make
        (FileNotFoundError without one)."""
        self.delete(name + "/")


def add_args(p):
    p.add_argument("--key-file", dest="key_file", default=None,
                   help=f"service-account JSON key (default: see above; {KEY_NAME})")
    p.add_argument("--bucket", default=os.environ.get("JS_GCS_BUCKET", DEFAULT_BUCKET),
                   type=bucket_name, help="bucket (default gs://test_fno, or $JS_GCS_BUCKET)")


def parse_args(argv):
    return core.build_parser(__doc__, add_args, "gs://BUCKET").parse_args(argv)


def main(argv=None):
    return core.main("js_gcs", core.build_parser(__doc__, add_args, "gs://BUCKET"), ENV,
                     lambda a: GcsStore(a.bucket, find_key(a.key_file)),
                     ("google.cloud.storage", "google_crc32c"), argv)


if __name__ == "__main__":
    core.enter_env(ENV, __file__, sys.argv[1:], "JS_GCS_NO_ENV")
    sys.exit(main())
