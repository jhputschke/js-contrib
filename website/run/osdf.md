# Data on the OSDF

Productions are shared on the **OSDF**, the Open Science Data Federation of the OSG: a
national network of data origins and caches, reached through the Pelican platform. Data sit
on an *origin* under a *namespace*; clients ask a *director* for the closest cache and read
from it with plain HTTP. For us that means: one place where a production made on a cluster,
a workstation or a Mac can go, from which anyone can download it, or read just the part of a
file they need without downloading it, from any machine and without an account. js-contrib's
namespace is **`osdf:///fno4hic`**. Reading it is public; writing needs a login.

!!! info "What is there"
    Today the namespace holds example and test productions. In the future, besides the example
    files, training data sets (for the ML/FNO surrogates) and physics simulation productions
    will be made available there.

The test data named in this documentation:

| path | content |
|---|---|
| `test/AuAu_0_10_jet_pth10-40_eta06_c1/` | the ROOT files of an Au+Au jet campaign (pT̂ 10–40 GeV, 68 files, 2.1 GB), with its campaign file |
| `test/h5_read_test/` | a small Au+Au hadron file, to try reading in place |
| `test/OO_test/` | the hadron files of an O+O `oo_workflow` run, read in place by the notebook `pythonic/oo_analysis.ipynb` |

## The transfer tool: `js_osdf.py`

[`utils/remote_transfer/js_osdf.py`](https://github.com/jhputschke/js-contrib/blob/main/utils/remote_transfer/js_osdf.py)
lists, downloads, uploads and removes whole productions, single files or patterns, and picks
the files by kind. It runs from a js-contrib checkout and makes its own Python environment on
its first run (in `~/.cache/js_osdf/venv`; a `python3` ≥ 3.9 with `venv` and access to PyPI,
a few seconds), so it needs no conda env or venv of yours:

```bash
git clone https://github.com/jhputschke/js-contrib.git
cd js-contrib/utils/remote_transfer
./js_osdf.py ls                     # the namespace: one folder per production
```

Keep it next to `transfer_core.py`, which it imports. Every command takes remote names as
`NAME`, `osdf:///fno4hic/NAME` or `/fno4hic/NAME`.

## Browsing

```bash
./js_osdf.py ls                                  # the top level
./js_osdf.py ls test                             # a folder
./js_osdf.py ls test -r                          # every file below it
./js_osdf.py ls AuAu_c1 --what h5                # a production's files of one kind, with sizes
./js_osdf.py du                                  # space per top-level folder, and the total
./js_osdf.py du AuAu_c1 -d 2 --sort size         # below a folder, two levels, largest first
```

`AuAu_c1` stands for a production folder; `ls` shows what is there.

## Downloading

```bash
./js_osdf.py download AuAu_c1 --what h5 --to /scratch            # -> /scratch/AuAu_c1/
./js_osdf.py download AuAu_c1 --what root --to /scratch          # only the ROOT files
./js_osdf.py download 'AuAu_c1/*_000[1-5]_*' --to /scratch/few   # a pattern (quoted): a few jobs
./js_osdf.py download AuAu_c1 --what h5 --to /scratch --dry-run  # what would come, and how much
```

The kinds go by file name:

| `--what` | files | for |
|---|---|---|
| `h5` | `<stem>_particlize.h5`, `<stem>_hadrons_*.h5`, the hadronization logs | the hadron analyses: `HadronFileReader` needs the particlize files next to the hadron files, and `h5` brings them along |
| `root` | `<stem>_hadrons.root`, `<campaign>_campaign.root` | ROOT analyses alone: complete on their own |
| `pair` | `<stem>.h5`, the hydro (~0.17–0.29 GB per event), with its `.json`, `.xml`, `.log` | the hydro, the wake observables, FNO training |
| `all` (default) | everything in the folder | |

- **Resumable and checked.** Files already there with the same size and CRC32C are skipped, so
  an interrupted download finishes when the command runs again. Each file is written as
  `<file>.part` and renamed only when its size and checksum are right.
- **Where they go:** a folder to `--to/<folder>/`, files and patterns to `--to/<file>`.
- `-j N` transfers N files at once (default 4). Measured: ~69 MB/s down from a workstation.
- `--what` takes a comma list (`h5,root`).

## Reading files in place

In the [analysis environment](../install/analysis-env.md) (pelicanfs is installed by
default) and in the production container, `osdf://` paths can be read without downloading:
pelicanfs finds a cache for the path and reads by byte ranges, so only the parts asked for
travel. Both examples read public test files and run as they are:

```python
import uproot
c = uproot.open("osdf:///fno4hic/test/AuAu_0_10_jet_pth10-40_eta06_c1/pth10-40_eta06_c1_campaign.root")
sigma = c["windows"].arrays(["sigma_mb"], library="np")["sigma_mb"]   # the pT-hat windows' cross sections

import fsspec, h5py, hdf5plugin                   # hdf5plugin: the Blosc filter of the HDF5 files
fs = fsspec.filesystem("osdf")
with fs.open("/fno4hic/test/h5_read_test/AuAu_0_10_jet_st1_t000_0001_hadrons_bulk_jet.h5", "rb",
             cache_type="blockcache", block_size=2**20) as fo, h5py.File(fo, "r") as f:
    so = f["hadrons/sample_offsets"][:]
    p = f["hadrons/p"][so[3]:so[4]]               # one oversample's momenta (E, px, py, pz)
```

- **Pass `cache_type="blockcache", block_size=2**20`** for HDF5: fsspec's default read-ahead
  re-fetches overlapping ranges when HDF5 jumps back to its metadata. With 1 MB blocks, one
  oversample of a 100 MB hadron file moved 6.3 MB instead of 31.5 MB.
- **Time:** the first open in a process takes ~1–1.5 s (pelicanfs finds a cache); after that,
  one oversample ~0.1–0.2 s.
- **When to download instead:** for whole productions and for anything read more than once
  (in place, every pass goes over the network again). `HadronFileReader`, `ParticlizeFile`,
  `run_h5toROOT.py` and the `analysis/` scripts take local paths only.
- Inside the production container the `pelican` command line does the same for single files:
  `pelican object get osdf:///fno4hic/… /work/…`.

What the files hold and how to read them: [What a production writes](../analysis/files.md),
[Hadrons](../analysis/hadrons.md), [ROOT and C++](../analysis/root.md).

## Uploading

Writing to `/fno4hic` needs write access to the namespace: a browser login, or a bearer token.
**Whatever is uploaded can be read by everyone.**

```bash
./js_osdf.py login              # once: prints a link; log in and approve in any browser
./js_osdf.py status             # who, the scopes, the time left of the token and the login
./js_osdf.py upload /data/AuAu_c1 --what h5 --dry-run       # what would go
./js_osdf.py upload /data/AuAu_c1 --what h5                 # -> osdf:///fno4hic/AuAu_c1/
./js_osdf.py upload /data/out --what root --as AuAu_pth10-40_v1   # another folder name
./js_osdf.py rm -r AuAu_c1 --dry-run                         # what would be removed
```

- **The login** lasts 15 days (renewals don't extend it): log in again before a long run
  would pass that date. It is kept in `~/.config/js_osdf/web/` and used by every later command.
  `logout` revokes it. A bearer token works too (`--token-file`, `$BEARER_TOKEN_FILE`, …).
- **Uploads** of a directory go to `osdf:///fno4hic/[PREFIX/]<directory name>/` (`--as NAME`
  for another name, `--prefix` for a parent folder). Only the files directly in the directory
  are sent, not its subdirectories: upload `gpu0/`, `gpu1/` or `t000/`, … one by one (with
  `--as CAMPAIGN/gpu0`) or use `upload_follow.py` below.
- **Unfinished files are left out:** HDF5 files whose `complete` attribute is false (a job
  still writing) and `*.part` files.
- **Checked:** Pelican reports sizes but no checksums, so after each upload `js_osdf.py`
  checks the size at the origin and records size and CRC32C in a manifest in the remote
  folder; downloads are checked against it. Don't run two uploads into the same folder at
  once.
- **`rm`** asks first (`--yes` doesn't; `--dry-run` lists all). Removed files can't be
  brought back.

Measured: ~80 MB/s up from a workstation, 4 or 8 files at once alike.

### Uploading while a production runs

A campaign can write faster than a local disk holds. [`utils/upload_follow.py`](https://github.com/jhputschke/js-contrib/blob/main/utils/upload_follow.py)
uploads each job as soon as its `<stem>.json` says complete and, if asked, deletes the local
copies after a verified upload, so the local disk no longer limits the campaign:

```bash
utils/upload_follow.py /data/AuAu_a AuAu_a                      # one pass, upload only
utils/upload_follow.py /data/AuAu_a AuAu_a --follow --delete pair   # while the campaign runs
touch /data/AuAu_a/.upload_final       # when the campaign is done: a last pass, then it exits
```

| option | |
|---|---|
| `--delete none` (default) / `pair` / `all` | after a verified upload, delete nothing / `<stem>.h5` (the hydro) / also `<stem>_particlize.h5` |
| `--keep-free SIZE` | with `pair`/`all`: delete only while the disk has less than SIZE free, the oldest uploads first |
| `--verify-every N` | before deleting, download every Nth file and compare it byte by byte (default 20; 0: never) |
| `--follow` | repeat every `--interval` s (default 60) until `OUTBASE/.upload_final` appears or `--follow-pid` ends |

- **Verified** means: the origin holds a file of the local size, the manifest's CRC32C matches
  the local file, and every Nth file was downloaded and compared. The `.json`, `.log` and `.xml`
  files are never deleted (`run_jobs.sh` needs the `.json` to skip finished jobs).
- **It runs on the host**, next to Docker or on a node that sees a SLURM campaign's
  directories, from a js-contrib checkout; any directory of `run_jobs.sh` output directories
  works (`OUTBASE/<sub>/` → `REMOTE/<sub>/`). Its state survives restarts
  (`OUTBASE/upload_state.json`, log in `OUTBASE/upload.log`).
- **With `launch_2gpu.sh`**, `--upload REMOTE --delete pair` starts it with the campaign
  ([Clusters](clusters.md#several-gpus-in-one-machine)). `launch_2gpu.sh --upload` refuses to
  start without a login.
- Hadronization output is uploaded afterwards: `js_osdf.py upload OUTBASE/gpu0 --as REMOTE/gpu0 --what h5,root`.

**What to move where.** The particlize files are self-contained (surfaces, partons, shower
initiators), so hadronization on another cluster and the analysis need only `--what h5`; the
large pair files (`--what pair`) only go where the hydro is used, e.g. FNO training.

## Google Cloud Storage

[`js_gcs.py`](https://github.com/jhputschke/js-contrib/blob/main/utils/remote_transfer/js_gcs.py),
next to `js_osdf.py`, has the same commands and options for a Google Cloud Storage bucket
(default `gs://test_fno`). The bucket is not public: reading and writing need a
service-account JSON key (`--key-file`, `$JS_GCS_KEY_FILE` or
`$GOOGLE_APPLICATION_CREDENTIALS`). The analysis environment reads `gs://` paths with
`--with-gcs`.

```bash
./js_gcs.py upload /data/AuAu_a --what pair -j 8     # -> gs://test_fno/AuAu_a/
./js_gcs.py download AuAu_a --what root --to /scratch
```

All options, credentials and the tests:
[`utils/remote_transfer/README.md`](https://github.com/jhputschke/js-contrib/blob/main/utils/remote_transfer/README.md).

## Next steps

- [Analysis environment](../install/analysis-env.md): the venv that reads these files, with
  pelicanfs.
- [Notebooks](../analysis/notebooks.md): `oo_analysis.ipynb` streams the O+O test files from
  the OSDF.
