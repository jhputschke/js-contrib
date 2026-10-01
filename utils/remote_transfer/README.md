# gcs_transfer — productions to and from Google Cloud Storage

`js_gcs.py` uploads production directories to a GCS bucket and downloads them again, with
a service-account key. It picks the files by kind: pair files, the HDF5 hadron files, the
ROOT files, or all of them. It is a single script. On its first run it makes its own Python
environment and from then on runs itself in it, so it needs no conda env, venv, uv or pipx.
A `python3` (≥ 3.9, with `venv`) is all.

```bash
cd utils/gcs_transfer
./js_gcs.py ls                                                   # the bucket (gs://test_fno)
./js_gcs.py upload /data/AuAu_0_10_pth10-40_eta06_c1 --what root --dry-run
./js_gcs.py upload /data/AuAu_0_10_pth10-40_eta06_c1 --what root     # -> gs://test_fno/AuAu_0_10_pth10-40_eta06_c1/
./js_gcs.py upload /data/AuAu_a /data/AuAu_b --what pair,h5 --prefix productions -j 8
./js_gcs.py ls AuAu_0_10_pth10-40_eta06_c1 --what h5             # files, sizes, kinds
./js_gcs.py download AuAu_0_10_pth10-40_eta06_c1 --what root --to /scratch
                                                       # -> /scratch/AuAu_0_10_pth10-40_eta06_c1/
./js_gcs.py upload /data/AuAu_c1/AuAu_c1_0003_hadrons.root /data/AuAu_c1/*_0004_*.h5
                                                       # single files (§ Single files)
./js_gcs.py download AuAu_c1/AuAu_c1_0003_hadrons.root 'AuAu_c1/*_0004_*' --to here
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

`--dry-run` shows what would be transferred, with the sizes per kind. On
`AuAu_0_10_pth10-40_eta06_gridnorm` that is pair 80 files 50.6 GB, h5 100 files 51.4 GB,
and other 4 files 35 MB.

## Where

- **Uploads** of a directory go to `gs://BUCKET/[PREFIX/]<directory name>/<file>`. Only the
  files directly in the directory are sent, not its subdirectories.
- **Downloads** of a directory in the bucket go to `TO/<its name>/`.
- **Names in the bucket** can be written `NAME` or `gs://BUCKET/NAME`.
- **The bucket** is `--bucket`, else `$JS_GCS_BUCKET`, else `gs://test_fno`.

## Single files

`upload` and `download` take any mix of directories and files, several at once.

- **Uploading a file:** a file named on the command line goes up whatever its kind
  (`--what` applies to directories only). It goes where its directory's upload puts it:
  `gs://BUCKET/[PREFIX/]<directory name>/<file>`. So
  `upload /data/AuAu_c1/AuAu_c1_0007.h5` adds one file to `gs://test_fno/AuAu_c1/`.
  `--flat` puts it at `gs://BUCKET/[PREFIX/]<file>` instead. Shell globs work
  (`/data/AuAu_c1/*_0007*`).
- **Downloading a file:** an object name goes to `TO/<file>`, like `cp`.
- **Downloading a pattern:** `*`, `?` and `[..]` select files by pattern, also to
  `TO/<file>`. `*` also matches `/`, and `--what` filters the matches. Quote patterns so
  the shell leaves them alone: `'AuAu_c1/*_0004_*'`. `ls` takes patterns too.
- **Same rules as directories:** files already there are skipped, incomplete HDF5 files
  and `*.part` are left out. Two objects that would land on the same local file are
  refused; download them with different `--to`.

## Safe to re-run

- **Files already there are skipped:** same size and the same CRC32C as the object in the
  bucket (or on disk, for downloads). An interrupted command finishes when run again.
  - `--size-only`: compare sizes only, without computing the local CRC32C (that takes about
    a second per few GB).
  - `--force`: transfer anyway.
- **Unfinished files are not uploaded.** HDF5 files whose `complete` attribute is False
  (a job still writing) are left out and listed, unless `--include-incomplete`. `*.part`
  files never go up.
- **Every transfer is checked** against its CRC32C. Downloads are written to
  `<file>.part` and renamed when complete.
- `-j` sets how many files are transferred at once (default 4). Large files go up as
  resumable uploads in 64 MB chunks, with retries.

## The key

A service-account JSON key, looked for in this order:
1. `--key-file PATH`;
2. `$JS_GCS_KEY_FILE`;
3. `$GOOGLE_APPLICATION_CREDENTIALS`;
4. `fnotest-wayne-gcs.json` in the current directory, next to `js_gcs.py`, or in
   `~/.config/js_gcs/`.

The service account needs read access to the bucket for `ls` and `download`, and write
access for `upload`.

**Keep the key out of git.** `.gitignore` ignores `fnotest-wayne-gcs.json` and every
`*.json` in this directory. The safest place is `~/.config/js_gcs/` (`chmod 600`).

## Its environment

- **Where:** `~/.cache/js_gcs/venv` (or `$JS_GCS_ENV`).
- **What:** `google-cloud-storage`, `google-crc32c` (C implementation) and `h5py`, for the
  `complete` check.
- **How it is made:** with the `python3` that first runs the script, in a few seconds.
  Later runs use it right away.
- `./js_gcs.py setup` shows it, and `./js_gcs.py setup --reinstall` makes it again.
- `./js_gcs.py setup --remove` deletes it, and `~/.cache/js_gcs` with it if that is then
  empty. It deletes only a venv the script made: one with its `pyvenv.cfg` and the
  script's stamp file. The next run makes it again. The
  environment is remade by itself when the requirements in the script change, or when the
  Python it was made from is gone.
- `JS_GCS_NO_ENV=1` runs the script in the current environment instead, if that one has
  the packages. One example is the `-gcs` production images
  ([`BuildContainerProd.md`](../BuildContainerProd.md)), which have
  `google-cloud-storage` and h5py built in.

## Tested

- **Offline:** `pytest utils/gcs_transfer/test_js_gcs.py` checks file kinds, `--what`, the
  key lookup, and upload and download against an in-memory bucket. That includes
  directories, single files (`--flat`), patterns, skipping, incomplete HDF5 files, `.part`
  files, local name clashes, and that the CRC32C matches GCS's.
- **On `gs://test_fno` (GB10, 2026-10-01):**
  - the environment was made from `/usr/bin/python3` (3.12) outside any conda env;
  - `ls` listed the bucket;
  - `upload --dry-run` classified `AuAu_0_10_pth10-40_eta06_gridnorm` as above;
  - `ls` and `download --dry-run` with a pattern and with one object, and `upload --dry-run`
    of single files, picked the expected files.
