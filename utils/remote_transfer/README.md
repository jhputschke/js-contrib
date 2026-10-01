# remote_transfer — productions to and from Google Cloud Storage and Pelican/OSDF

Two command-line tools upload production directories and files and download them again.
They pick the files by kind: pair files, the HDF5 hadron files, the ROOT files, or all of
them.

| script | where | credentials |
|---|---|---|
| [`js_gcs.py`](js_gcs.py) | a Google Cloud Storage bucket, default `gs://test_fno` | a service-account JSON key |
| [`js_osdf.py`](js_osdf.py) | a Pelican namespace on the OSDF, default `osdf:///fno4hic` (pelicanfs) | a bearer token for writing; reads of a public namespace need none |

Both have the same commands and options and follow the same rules. The code they share is
in [`transfer_core.py`](transfer_core.py); keep the three files together. Each script makes
its own Python environment on its first run and from then on runs itself in it, so neither
needs a conda env, venv, uv or pipx. A `python3` (≥ 3.9, with `venv`) is all.

```bash
cd utils/remote_transfer
./js_gcs.py ls                                                   # the bucket (gs://test_fno)
./js_gcs.py upload /data/AuAu_0_10_pth10-40_eta06_c1 --what root --dry-run
./js_gcs.py upload /data/AuAu_0_10_pth10-40_eta06_c1 --what root     # -> gs://test_fno/AuAu_0_10_pth10-40_eta06_c1/
./js_gcs.py upload /data/AuAu_a /data/AuAu_b --what pair,h5 --prefix productions -j 8
./js_gcs.py upload /data/out --what root --as AuAu_pth10-40_v1      # -> gs://test_fno/AuAu_pth10-40_v1/
./js_gcs.py ls AuAu_0_10_pth10-40_eta06_c1 --what h5             # files, sizes, kinds
./js_gcs.py download AuAu_0_10_pth10-40_eta06_c1 --what root --to /scratch
                                                       # -> /scratch/AuAu_0_10_pth10-40_eta06_c1/
./js_gcs.py upload /data/AuAu_c1/AuAu_c1_0003_hadrons.root /data/AuAu_c1/*_0004_*.h5
                                                       # single files (§ Single files)
./js_gcs.py download AuAu_c1/AuAu_c1_0003_hadrons.root 'AuAu_c1/*_0004_*' --to here

./js_osdf.py upload /data/AuAu_0_10_pth10-40_eta06_c1 --what root    # -> osdf:///fno4hic/AuAu_0_10_pth10-40_eta06_c1/
./js_osdf.py download AuAu_0_10_pth10-40_eta06_c1 --what root --to /scratch
./js_osdf.py ls                                                  # the namespace (osdf:///fno4hic)
```

## What: `--what`

The kinds go by file name, as `run_prod_jet.py`, `hadronize.py` and `run_h5toROOT.py` name
the files. `--what` takes one kind or a comma list (`pair,root`); the default is `all`.

| `--what` | files |
|---|---|
| `pair` | `<stem>.h5`, the hydro pair file, with the run's `<stem>.json`, `.xml`, `.log` |
| `h5` | `<stem>_particlize.h5`, `<stem>_hadrons_{bulk_jet,bulk_bg,jet_frag}.h5`, `<stem>_hadronize.log` |
| `root` | `<stem>_hadrons.root`, `<campaign>_campaign.root` |
| `all` | the three, plus every other file of the directory (`run_jobs.campaign`, `wake_observables.h5`, …) |

- **`h5` brings the particlize files along.** `HadronFileReader` needs them next to the
  hadron files.
- **`root` is complete on its own.** `HadronFileReader.h` and the macros of
  `analysis_root` need nothing else.
- **Pair files are recognized by their neighbours.** A `<stem>.h5` counts as a pair file
  only with its `<stem>.json`, `<stem>.xml` or `<stem>_particlize.h5` next to it. That
  tells it apart from other HDF5 files in the directory, such as `wake_observables.h5`.
  Files matched by a pattern are judged among all files of their directory.

`--dry-run` shows what would be transferred, with the sizes per kind. On
`AuAu_0_10_pth10-40_eta06_gridnorm` that is pair 80 files 50.6 GB, h5 100 files 51.4 GB,
and other 4 files 35 MB.

## Where

- **Uploads** of a directory go to `ROOT/[PREFIX/]<directory name>/<file>`. `ROOT` is
  `gs://BUCKET` or `osdf:///NAMESPACE`. Only the files directly in the directory are sent,
  not its subdirectories.
- **Another folder name:** `--as NAME` uploads to `ROOT/[PREFIX/]NAME/<file>` instead of
  using the local directory's name. Use it to publish `out/` as `AuAu_pth10-40_v1`, or to
  add single files of a directory to a folder uploaded that way. `NAME` may contain `/`.
  It names one folder, so all sources must come from one local directory. It doesn't go
  with `--flat`.
- **Downloads** of a remote directory go to `TO/<its name>/`.
- **Remote names** can be written `NAME` or `ROOT/NAME`; for OSDF also `/NAMESPACE/NAME`
  or `pelican://FEDERATION/NAMESPACE/NAME`.
- **The bucket** is `--bucket`, else `$JS_GCS_BUCKET`, else `gs://test_fno`.
- **The namespace** is `--namespace`, else `$JS_OSDF_NAMESPACE`, else `/fno4hic`. The
  federation is `--federation`, default `osg-htc.org` (the OSDF); another federation shows
  as `pelican://FEDERATION/NAMESPACE`.

## Single files

`upload` and `download` take any mix of directories and files, several at once.

- **Uploading a file:** a file named on the command line goes up whatever its kind
  (`--what` applies to directories only). It goes where its directory's upload puts it:
  `ROOT/[PREFIX/]<directory name>/<file>`. So `upload /data/AuAu_c1/AuAu_c1_0007.h5` adds
  one file to `ROOT/AuAu_c1/`. `--flat` puts it at `ROOT/[PREFIX/]<file>` instead. Shell
  globs work (`/data/AuAu_c1/*_0007*`).
- **Downloading a file:** a remote file goes to `TO/<file>`, like `cp`.
- **Downloading a pattern:** `*`, `?` and `[..]` select files by pattern, also to
  `TO/<file>`. `*` also matches `/`, and `--what` filters the matches. Quote patterns so
  the shell leaves them alone: `'AuAu_c1/*_0004_*'`. `ls` takes patterns too.
- **Same rules as directories:** files already there are skipped, incomplete HDF5 files
  and `*.part` are left out.
- **Overlaps:** a file asked for twice, e.g. through its directory and a pattern, goes to
  both places. Two remote files that would land on the same local file are refused;
  download them with different `--to`.

## Safe to re-run

- **Files already there are skipped:** the same size and the same CRC32C as the remote file
  (or the local one, for downloads). An interrupted command finishes when run again.
  - `--size-only`: compare sizes only, without computing the local CRC32C (that takes about
    a second per few GB).
  - `--force`: transfer anyway.
- **Unfinished files are not uploaded.** HDF5 files whose `complete` attribute is False
  (a job still writing) are left out and listed, unless `--include-incomplete`. `*.part`
  files never go up.
- **Downloads** are written to `<file>.part` and renamed only when their size (and CRC32C,
  where known) is right. A failed download leaves nothing behind.
- `-j` sets how many files are transferred at once (default 4).
- **Errors** of the connection or the server end the command with one line, including the
  root cause, e.g. `SSLCertVerificationError: … certificate has expired`.

**Checksums differ between the two stores:**
- **GCS** keeps a CRC32C of every object, and the client checks every upload and download
  against it. Large files go up as resumable uploads in 64 MB chunks, with retries.
- **Pelican** reports sizes but no checksums. So after each upload `js_osdf.py` checks the
  size the origin reports, and records the file's size and CRC32C in a manifest in the
  remote directory, `.js_transfer_crc32c.json`. Later uploads add to it.
  - The skip check and the check after a download use the manifest's CRC32C.
  - Files the manifest doesn't know, put there by other tools, are compared by size.
  - The manifest is never downloaded and `ls` doesn't show it.
  - Two `js_osdf.py` uploads into the same remote directory at the same time may lose each
    other's manifest entries. The files themselves are fine; the lost ones are then
    compared by size.
- **Reading from the origin:** listings, sizes and downloads come from the origin itself
  (pelicanfs `direct_reads`). An OSDF cache could still hold an older copy of a file that
  was uploaded again.

## Credentials

**GCS: a service-account JSON key,** looked for in this order:
1. `--key-file PATH`;
2. `$JS_GCS_KEY_FILE`;
3. `$GOOGLE_APPLICATION_CREDENTIALS`;
4. `fnotest-wayne-gcs.json` in the current directory, next to `js_gcs.py`, or in
   `~/.config/js_gcs/`.

The service account needs read access to the bucket for `ls` and `download`, and write
access for `upload`.

**OSDF: a bearer token** (a SciToken / WLCG token). `upload` needs one with write access to
the namespace. `ls` and `download` need one only for a protected namespace. Looked for in
this order:
1. `--token-file PATH`;
2. `$JS_OSDF_TOKEN_FILE`, then `$BEARER_TOKEN_FILE`;
3. `$BEARER_TOKEN` (the token itself);
4. `osdf.token` in the current directory, next to `js_osdf.py`, or in `~/.config/js_osdf/`.

With none of these, pelicanfs looks for one itself, e.g. in the WLCG default place
(`$XDG_RUNTIME_DIR/bt_u$UID`, `/tmp/bt_u$UID`). With the `pelican` CLI on `PATH` it can also
get one through the namespace's OAuth flow. Tokens expire, typically after hours, so a long
upload needs one that lasts.

**Keep credentials out of git.** `.gitignore` ignores `fnotest-wayne-gcs.json`, and every
`*.json` and `*token*` in this directory. The safest place is `~/.config/js_gcs/` or
`~/.config/js_osdf/` (`chmod 600`).

## Their environments

- **Where:** `~/.cache/js_gcs/venv` and `~/.cache/js_osdf/venv` (or `$JS_GCS_ENV`,
  `$JS_OSDF_ENV`).
- **What:**
  - js_gcs: `google-cloud-storage`, `google-crc32c` (C implementation) and `h5py`, for the
    `complete` check;
  - js_osdf: `pelicanfs`, `google-crc32c` and `h5py`.
- **How they are made:** with the `python3` that first runs the script, in a few seconds.
  Later runs use them right away.
- `./js_gcs.py setup` shows the environment (likewise `./js_osdf.py setup`), and
  `setup --reinstall` makes it again.
- `setup --remove` deletes it, and `~/.cache/js_gcs` (`js_osdf`) with it if that is then
  empty. It deletes only a venv the script made: one with its `pyvenv.cfg` and the
  script's stamp file. The next run makes it again.
- An environment is remade by itself when the requirements in the script change, or when
  the Python it was made from is gone.
- `JS_GCS_NO_ENV=1` / `JS_OSDF_NO_ENV=1` run the script in the current environment
  instead, if that one has the packages. One example is the `-gcs` production images
  ([`BuildContainerProd.md`](../BuildContainerProd.md)): they have pelicanfs,
  `google-cloud-storage`, `google-crc32c` and h5py, so both scripts run there as they are.
  The other images lack `google-crc32c`, so there the scripts make their environments.

## Tested

- **Offline:** `pytest utils/remote_transfer/test_transfer.py` (25 tests). Every command
  runs against two stores: an in-memory GCS bucket, and `OsdfStore` on fsspec's in-memory
  file system in place of pelicanfs. The tests cover:
  - directories, single files (`--flat`) and patterns;
  - skipping, incomplete HDF5 files, `.part` files, overlaps and local name clashes;
  - the manifest: written, added to, used for skipping, a corrupted download caught, an
    unreadable manifest;
  - the key and token lookup, remote names, and that the CRC32C matches GCS's.
- **On `gs://test_fno` (GB10, 2026-10-01):**
  - the environment was made from `/usr/bin/python3` (3.12) outside any conda env;
  - `ls` listed the bucket;
  - `upload --dry-run` classified `AuAu_0_10_pth10-40_eta06_gridnorm` as above;
  - `ls` and `download --dry-run` with a pattern and with one object, and `upload --dry-run`
    of single files, picked the expected files.
- **On the OSDF (2026-10-01), with pelicanfs 1.4.1:**
  - on the public `/ospool/uc-shared/public/OSG-Staff`: `ls`, `ls` with a pattern, and
    downloads of a directory, a pattern and one file. The files are right, and a second
    run skipped them.
  - **`/fno4hic` could not be reached.** Its origin,
    `wayne-origin.nationalresearchplatform.org:8090`, presents a TLS certificate that
    expired on 2026-09-21, so every request fails before authentication. The `pelican` CLI
    fails the same way.
- **Not tested yet:** real uploads and downloads on either store (only dry runs on
  `gs://test_fno`), and uploads to `/fno4hic`.
