# launch_2gpu.sh: production campaigns on a two-GPU workstation

`launch_2gpu.sh` runs `prod_AuAu_0_10_jet` from the production image (default
`jhputschke/xscape-prod:cu124`) on both GPUs of one machine with Docker, and can upload the
finished jobs to the OSDF while it runs ([`upload_follow.py`](#uploading-to-the-osdf-fno4hic)).
It was written for, and all numbers below were measured on, a workstation with **2 × RTX 3090
(24 GB), 24 cores, 125 GB RAM** and a second disk at `/data` (2026-10-05; the benchmark notes
themselves are not in the repo). On other machines adjust `JOBS_PER_GPU`, `OMP_THREADS`,
`WORKDIR` and OUTBASE. The best measured setup there:

- **one container per GPU** (two MPS daemons in one container hang the second GPU's jobs);
- in each, `run_jobs.sh -j 8 --mps`: **8 jobs per GPU**, kernels shared through CUDA MPS;
- `OMP_NUM_THREADS=4`, `OMP_WAIT_POLICY=passive` per job.

> **Image.** The default `cu124` is `cu124-20261003-cbc7263` until the production images are
> rebuilt ([`docs/README_2stage.md`](../docs/README_2stage.md) §1): it has the memory fixes but
> writes the earlier 65 × 65 × 33 output grid and has no edge flag. For a campaign, pin a
> dated tag (`IMAGE=jhputschke/xscape-prod:cu124-20261003-cbc7263`), and for the new
> 64 × 64 × 32 grid pass `--grid` with the YAML from a checkout under `WORKDIR`
> (e.g. `--grid /work/grid_fno.yaml`). The memory and disk numbers below are for the earlier grid.

## Campaign default (2 × RTX 3090)

Run it from `utils/` of a js-contrib checkout: it finds `upload_follow.py` and
`remote_transfer/` next to itself.

```bash
cd js-contrib/utils
nohup ./launch_2gpu.sh 64 45 0 /data/camp_pth3 --campaign pth3 \
    --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 \
    --write-particlize both > /data/camp_pth3.log 2>&1 &
```

| | |
|---|---|
| What it does | 3 pT̂ windows (10-20, 20-30, 30-40 GeV) × 5 jets each = **one background per 15 jet events**; only events whose leading parton has \|y\| < 0.6; writes the pair file and the `_particlize.h5` for `hadronize.py` |
| Throughput | **~810 events/h** (450 without the bins, i.e. a new background every event) |
| Host memory | **~97 GB** peak of 125 GB (same as without the bins) |
| GPU memory | ~6.5 GB per GPU of 24 GB |
| Disk | **~0.25 GB per event** (pair 162 MB + particlize 92 MB) |
| This example | 64 jobs × 45 events = 2,880 events, 4 rounds of 16 jobs: **~3.5 h, ~730 GB** |

Choices behind it:
- **EVENTS must be a multiple of 15** (windows × jets per bin). 45 (3 backgrounds per job) spreads
  each job's ~10 s startup over more events than 15 does.
- **NJOBS a multiple of 16** keeps all 16 slots busy in the last round.
- **`/data`** (second disk, 1.6 TB free) rather than `~/prod_test` (/home, ~780 GB free), on
  the machine above.
- **`nohup … &`**: the script waits for both GPUs, so detach it from the terminal for long runs.

## Usage

```
./launch_2gpu.sh [upload options] NJOBS EVENTS_PER_JOB [FIRST_SEED] [OUTBASE] [run_prod_jet.py / run_jobs.sh args...]
```

The options come before NJOBS: `--min-free SIZE` (the [disk guard](#disk-guard-min-free)) and the
upload options (`--upload`, `--delete`, `--keep-free`, `--verify-every`; see
[Uploading to the OSDF](#uploading-to-the-osdf-fno4hic)).

| Argument | Meaning |
|---|---|
| `NJOBS` | total number of jobs (one seed and one `.h5` per job); GPU0 gets the first half (rounded up), GPU1 the rest |
| `EVENTS_PER_JOB` | events per job (`--events`); a multiple of the reuse factor with `--pthat-bins`/`--reuse` |
| `FIRST_SEED` | `0` (default): every job draws a unique seed, recorded in `OUTBASE/seeds_used.tsv` (shared by both GPUs, file-locked). `S > 0`: seeds S … S+NJOBS−1 (GPU0 the first half), files named by seed; for reproducible comparisons |
| `OUTBASE` | output root, default `/work/out`. `/work/...` = `~/prod_test/...`; any other host directory (e.g. `/data/campA`) is created and mounted into the containers at the same path. System paths (`/opt`, `/usr`, `/tmp`, …) are refused |
| further args | passed to every job: `run_prod_jet.py` options (`--pthat-bins`, `--write-particlize`, `--reuse`, `--grid`, …) and `run_jobs.sh`'s `--campaign NAME` |

Output: `OUTBASE/gpu0/` and `OUTBASE/gpu1/`, each with per job `<stem>.h5`, `<stem>_particlize.h5`
(with `--write-particlize`), `<stem>.json`, `<stem>.log`, `<stem>.xml`, and `run_jobs.finished` at the end.

Environment overrides:

| Variable | Default | |
|---|---|---|
| `JOBS_PER_GPU` | 8 | jobs at once per GPU (6: ~5–8 % slower, ~70 GB without particlize; 10: no faster, too much memory with particlize) |
| `OMP_THREADS` | 4 | OpenMP threads per job (3–6 equally good; 2–3 at 6–8 jobs/GPU is slower) |
| `IMAGE` | `jhputschke/xscape-prod:cu124` | |
| `WORKDIR` | `$HOME/prod_test` | mounted as `/work` |

## More examples

```bash
./launch_2gpu.sh 64 25                                   # 64 jobs x 25 events -> ~/prod_test/out
./launch_2gpu.sh 64 25 0 /data/campA --write-particlize both
./launch_2gpu.sh 16 3 1 /work/test                       # seeds 1..16, a quick check
JOBS_PER_GPU=6 ./launch_2gpu.sh 48 30 0 /data/campC      # fewer jobs at once, less memory
```

Measured rates (16 jobs at once): plain ~520–535 events/h, with `--write-particlize both` ~450,
campaign default above ~810.

## Running a campaign

- **Watch:** `docker logs -f xscape_gpu0_<pid>` (the names are printed at the start), `tail` the
  per-job `OUTBASE/gpuN/<stem>.log`, `nvidia-smi`, `free -g`.
- **The `event … done in` lines appear late, in batches**: the Python driver's output is
  block-buffered in the log file, MUSIC's is not. A job whose log shows only MUSIC lines is not stuck.
- **Stop:** Ctrl-C (or `kill` the script) stops both containers and the uploader (which saves its
  state). If the script itself was killed hard, `docker stop $(docker ps -q --filter name=xscape_gpu)`;
  the uploader then notices that the script is gone, does its last pass and exits.
- **Resume:** run the same command again; jobs whose `.json` says complete are skipped, and a
  campaign (`FIRST_SEED 0`) keeps its name from `OUTBASE/gpuN/run_jobs.campaign`.
- **File names with `FIRST_SEED 0`:** `AuAu_0_10_jet_<campaign>_<NNNN>.h5`, numbered from 0001 in
  *each* GPU directory, so `gpu0/` and `gpu1/` hold files of the same names (different seeds).
  Keep the two directories, or rename when merging.
- **One campaign per OUTBASE**; a different grid YAML or settings needs a new OUTBASE (the
  skip test only looks at seed and event count).
- **Memory:** 16 jobs with particlize peak at ~97 GB; don't run other large jobs at the same time.

## Disk guard (`--min-free`)

`run_jobs.sh` itself doesn't watch the disk: on a full disk every running job fails, and it
goes on starting jobs that compute for minutes and then fail too. So `launch_2gpu.sh` checks
the free space of OUTBASE's disk every minute and **stops both containers when it falls below
`--min-free`** (default **50G**; `0` turns the guard off). It also refuses to start below it.

- The log says `LOW DISK: … stopping the campaign`, the script exits 1, and the reason is in
  `OUTBASE/.stopped_low_disk` (removed at the next start).
- The jobs running at that moment are lost (their files lack the `complete` mark, their `.json`
  is missing). Free space, then **run the same command again**: finished jobs are skipped, the
  stopped ones run again (with new seeds for `FIRST_SEED 0`).
- The uploader (`--upload`) still does its last pass, so its deletions can free space.
- Sizes are powers of 1000 (`50G` = 50·10⁹ bytes), as in the messages; `df -h` shows powers of 1024.
- Tested (2026-10-05): refusing to start, a bad size, the guard stopping a running campaign
  (a 6 GB file pushed the disk below the limit; stopped within 1 s, containers removed), and the
  resume after it.

For the 5k test campaign on `/data` it should never fire (worst case ~1.28 TB of 1.6 TB);
it guards against other writers to the disk and against long upload outages in campaigns that
rely on `--delete` to fit.

## Uploading to the OSDF (fno4hic)

With `--upload REMOTE`, `launch_2gpu.sh` starts `upload_follow.py` on the host next to the
containers. It uploads every finished job to `osdf:///fno4hic/REMOTE/gpu0|gpu1/` while the
campaign runs, and can delete the uploaded HDF5 files locally, so a campaign is no longer
limited by the local disk.

**Once per 15 days:** `./remote_transfer/js_osdf.py login` (prints a link; approve in any
browser). `launch_2gpu.sh --upload` refuses to start without a login. `./remote_transfer/js_osdf.py
status` shows how long it lasts. The 20-min tokens are renewed by themselves, but the login
itself ends 15 days after `login` (it no longer slides): **log in again before a campaign that
would run past that date**.

**Issuer bug (2026-10-05):** the `/fno4hic` issuer (OA4MP at the Wayne origin) crashes
(HTTP 500 `server_error`, "Null pointer") when a renewal uses a refresh token that was itself
handed out by a renewal, while the login's own refresh token renews any number of times. Older
`js_osdf.py` versions switched to the new token at every renewal, so their login broke after
~40 min (one renewal) and reported this as "expired". js-contrib's `js_osdf.py` works around it
since `123ac76` (PR #43): renewals keep the login's refresh token (the issuer's new one is kept
as a fallback), a failed renewal prints the issuer's error, and a running process re-reads the
login file after a failed renewal, so a `login` in another terminal reaches a running uploader.
The copies inside the published images predate it; the uploader runs on the host from the
checkout, so this matters only for `js_osdf.py` run inside a container. Still worth reporting
to the origin's admins.

| Option | Default | |
|---|---|---|
| `--upload REMOTE` | – | remote folder in `/fno4hic`; `OUTBASE/gpuN` → `REMOTE/gpuN`, `OUTBASE/seeds_used.tsv` → `REMOTE/seeds_used.tsv` |
| `--delete none` | ✓ | upload only, keep every local file |
| `--delete pair` | | delete `<stem>.h5` (the hydro history, 162 MB/event) after a verified upload; keep `<stem>_particlize.h5` for `hadronize.py` |
| `--delete all` | | delete `<stem>.h5` and `<stem>_particlize.h5` |
| `--keep-free SIZE` | – | with `pair`/`all`: delete only while OUTBASE's disk has less than SIZE free (e.g. `300G`), the oldest uploads first; local copies stay as long as there is room |
| `--verify-every N` | 20 | before deleting, download every Nth file and compare it byte by byte; `0`: never |

```bash
# the campaign default, uploaded as it runs; pair files deleted, particlize files kept
nohup ./launch_2gpu.sh --upload AuAu_pth3_c1 --delete pair 64 45 0 /data/c1 --campaign c1 \
    --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 \
    --write-particlize both > /data/c1.log 2>&1 &
# keep everything locally while there is room, delete the oldest uploads below 300 GB free
./launch_2gpu.sh --upload AuAu_pth3_c2 --delete all --keep-free 300G 128 45 0 /data/c2 ...
```

**A ~5k-event test production** to decide how to hadronize: 112 jobs × 45 = 5,040 events,
7 rounds of 16, **~6.5 h**. It needs ~1.28 TB with every file kept, which fits `/data` (1.6 TB),
so `--delete none` is possible; `--delete pair` keeps only the particlize files (~0.46 TB),
enough to hadronize locally later; either way all files are on the OSDF.

```bash
nohup ./launch_2gpu.sh --upload AuAu_pth3_test5k --delete pair 112 45 0 /data/test5k \
    --campaign test5k --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 \
    --write-particlize both > /data/test5k.log 2>&1 &
```

**What is uploaded, and when:** a job once its `<stem>.json` says complete (the test
`run_jobs.sh` uses) and its files have not changed for 30 s: `<stem>.h5`, `<stem>_particlize.h5`,
`<stem>.json`, `.log`, `.xml`. A pass runs every 60 s; after the containers end, a last pass
also uploads `run_jobs.campaign`, `run_jobs.finished` and `seeds_used.tsv`. The upload is
remote_transfer's own (the stored size is checked, the CRC32C goes into the remote manifest).
Hadronization output is not uploaded by it: `./remote_transfer/js_osdf.py upload OUTBASE/gpuN
--as REMOTE/gpuN --what h5,root` later.

**What "verified" means before a file is deleted**, re-checked right before deleting:
- the origin holds a file of the local size;
- the remote manifest lists it with the CRC32C computed from the local file at upload;
- the local file still has that CRC32C;
- every Nth file (`--verify-every`) is also downloaded and compared byte by byte.

A file that fails a check is not deleted but uploaded again in the same pass. A failed
byte-by-byte comparison stops all deletions for the rest of the run. The `.json`, `.log`
and `.xml` files are never deleted: `run_jobs.sh` needs the `.json` to skip finished jobs
on a re-run.

**Failures:** an upload that fails (network, origin, expired login) is retried on the next
pass; nothing is deleted. The files pile up locally as without uploading, and the log warns
below 100 GB free (`--warn-free`).

**Measured** (2026-10-05): ~80 MB/s from the workstation (4 or 8 files at once alike); the
campaign default produces ~57 MB/s, so the upload keeps up; downloads ~69 MB/s,
byte-identical.

**Log and state:** `OUTBASE/upload.log` (one line per pass: uploaded, rate, deleted, free
space) and `OUTBASE/upload_state.json` (what was uploaded, verified, deleted). Re-running the
same `launch_2gpu.sh` command continues both.

**On its own**, e.g. for a production made without `--upload`, or to retry failures:

```bash
./upload_follow.py /data/c1 AuAu_pth3_c1                         # one pass, upload only
./upload_follow.py /data/c1 AuAu_pth3_c1 --delete pair           # ... and delete pair files
./upload_follow.py /data/c1 AuAu_pth3_c1 --follow                # until /data/c1/.upload_final
```

**Getting files back:** `./remote_transfer/js_osdf.py download AuAu_pth3_c1/gpu0 --to /data/back
--what h5` (checked against the manifest's CRC32C), from any machine.

Notes:
- **`/fno4hic` is public:** anyone can list and download what is uploaded.
- **Don't run a second upload into the same remote folder at the same time** (e.g. the
  `js_osdf.py upload` CLI): two writers can lose each other's manifest entries, and files
  without one are then not deleted (they fail the check) but uploaded again.
- **One REMOTE per campaign:** `gpu0/` and `gpu1/` keep the local layout, as their file names
  repeat (see above).
- `upload_follow.py` uses `remote_transfer/` next to it (or `JS_TRANSFER_DIR`) as a library
  and runs in `js_osdf.py`'s environment (`~/.cache/js_osdf/venv`, made on first use;
  `JS_OSDF_NO_ENV=1` uses the current Python instead).
- It is not tied to `launch_2gpu.sh`: any directory of `run_jobs.sh` output directories works
  (`OUTBASE/<sub>/` → `REMOTE/<sub>/`), e.g. a SLURM campaign's `t000/`, `t001/`, … (run it on a
  node that sees the files, with `--follow` while the campaign runs; without `--follow-pid` it
  stops only at `touch OUTBASE/.upload_final`).
