# Analysing the productions: environments, HDF5 and ROOT, FastJet

This guide is for the analysis of the files the productions write: the hydro evolutions, the
freeze-out surfaces and partons, the hadrons, and the ROOT files made from them. It covers the
environments, how to read each file in Python, jets with FastJet, the conversion to ROOT, and
the analysis in ROOT, again with FastJet. Running the productions is in
[`README_2stage.md`](README_2stage.md).

```
 production (GPU, container)          stage 2 (CPU, container)             analysis (this guide)
 ───────────────────────────          ────────────────────────             ─────────────────────
 <stem>.h5             hydro pair  ─────────────────────────────────────→  h5py, FNO4d, wake_observables.py
 <stem>_particlize.h5  surfaces,   → hadronize.py → <stem>_hadrons_       → HadronFileReader, notebooks,
                       partons                      {bulk_jet,bulk_bg,      FastJet (Python)
                                                     jet_frag}.h5
                                                          │
                                                          └→ run_h5toROOT.py → <stem>_hadrons.root,
                                                                                <campaign>_campaign.root
                                                                                → uproot, RDataFrame,
                                                                                  HadronFileReader.h,
                                                                                  FastJet (C++)
```

None of the analysis needs X-SCAPE, a GPU or the container. A Python venv reads every HDF5
file and writes the ROOT files. ROOT itself is needed only for the C++ side (§2). The O+O
production (one hydro per event, hadrons made in the job) has its own guide,
[`README_OO.md`](README_OO.md).

**Contents**
1. [What a campaign gives you](#1-what-a-campaign-gives-you), and [getting the files from the OSDF](#getting-the-files-the-osdf-and-gcs)
2. [Environments](#2-environments), with [threads and memory](#threads-and-memory)
3. [Reading the hydro](#3-reading-the-hydro)
4. [Reading the particlize files](#4-reading-the-particlize-files)
5. [Reading the hadrons](#5-reading-the-hadrons)
6. [Jets with FastJet in Python](#6-jets-with-fastjet-in-python)
7. [Notebooks and scripts](#7-notebooks-and-scripts)
8. [ROOT: converting and analysing](#8-root-converting-and-analysing)
9. [Jets with FastJet in ROOT and C++](#9-jets-with-fastjet-in-root-and-c)
10. [Troubleshooting](#10-troubleshooting)

The code blocks marked *(tested)* were run on the seed-1 files in `prod_AuAu_0_10_jet/out`
on 2026-10-13, in `js_fno` (h5py 3, uproot 5.7, scikit-hep fastjet 3.5.1, FastJet 3.5.1,
ROOT 6.36). The scripts behind them, and `run_doc_blocks.py` to re-run them, are in
[`analysis_snippets/`](analysis_snippets/README.md).

---

## 1. What a campaign gives you

### The files

| file | made by | holds | size per event | read with |
|---|---|---|---|---|
| `<stem>.h5`, the **pair file** | `prod_AuAu_0_10_jet` | both hydro legs on the output grid: `arr` (jet leg), `arr_bg` (background); `source/droplets`, `shower/`, `diag/` | ~170 MB with 15 jets per background, 285 MB with one | h5py, FNO4d's loader, `wake_observables.py` (§3) |
| `<stem>_particlize.h5` | `prod_AuAu_0_10_jet --write-particlize`; `prod_OO_isr` (default) | the freeze-out surfaces (32 columns per cell), the final partons, the shower initiators, `events/` | ~90 MB (Au+Au, 15 jets per background); ~10 MB (O+O) | `ParticlizeFile` (§4) |
| `<stem>_hadrons_bulk_jet.h5` | `hadronize.py` | iSS on the jet leg's surface: bulk + wake, N oversamples per event | ~0.2 MB per oversample at full precision; 58 % of that with `--keep-bits-p 12 --keep-bits-x 8` | `HadronFileReader` (§5) |
| `<stem>_hadrons_bulk_bg.h5` | `hadronize.py`; O+O in the job with `--bulk-hadrons` | iSS on the background's surface, once per background | as above | `HadronFileReader`; O+O: `Hadrons`, `HadronFile` |
| `<stem>_hadrons_jet_frag.h5` | `hadronize.py` (Colorless); O+O in the job (iColored) | the fragmented surviving partons, M fragmentations per event | small | as above |
| `<stem>_hadrons.root`, `<campaign>_campaign.root` | `run_h5toROOT.py` | the three hadron tags as RNTuple or TTree, one entry per oversample; `events`, `windows`, `provenance`; the campaign's cross sections | 0.9 × the HDF5 hadron files (RNTuple) | uproot, RDataFrame, `HadronFileReader.h` (§8) |
| `AuAu_0_10_*.h5` | `prod_AuAu_0_10` (hydro only) | `arr` | ~140 MB | h5py, `read_fast_h5_bulk`, FNO4d |
| `OO_isr_*.h5` | `prod_OO_isr` | MUSIC's evolution on `grid_oo.yaml`; `diag/` with the Glauber and ISR bookkeeping | ~17 MB | h5py |
| `<stem>.json`, `.xml`, `.log` | every job | the job summary (seed, events, droplets, edge hits), the exact job XML, the log | | `json`, an editor |

The Au+Au sizes are for the campaign setting (`--pthat-bins 10-20,20-30,30-40
--jets-per-bin 5 --parton-ymax 0.6`): pair + particlize file 0.243 GB per event, 0.097 GB
with `--particlize-only` ([`REPORT_SUMMARY_CLUSTER.md`](REPORT_SUMMARY_CLUSTER.md)).

### Which files an analysis needs

| analysis | files |
|---|---|
| hadron spectra, the wake at hadron level, jets | the particlize file and the three hadron files of every production file; or the ROOT files alone |
| the wake in the fluid, the energy balance, the deposit (`wake_observables.py`, `jet_wake.ipynb`) | the pair files (and the particlize files) |
| FNO training | the pair files, or the hydro-only files |
| more oversamples, other iSS settings | the particlize files, and `hadronize.py` in the container on a CPU node ([`README_2stage.md`](README_2stage.md) §3) |

The particlize files are self-contained (format version 2, from 2026-10-01): a hadron
analysis never needs the large pair files. To fetch only what an analysis needs, see the
next section.

### Getting the files: the OSDF (and GCS)

Productions are shared on the OSDF in the Pelican namespace **`osdf:///fno4hic`**. **Reading
it is public:** listing and downloading need no account, no login and no token (writing
does). [`utils/remote_transfer/js_osdf.py`](../utils/remote_transfer/README.md) downloads
whole campaigns, single files or patterns, picked by kind. It runs from a js-contrib checkout
and makes its own Python environment on the first run (a `python3` ≥ 3.9 with `venv` and
access to PyPI; a few seconds):

```bash
cd js-contrib/utils/remote_transfer
./js_osdf.py ls                                        # the namespace: one folder per campaign
./js_osdf.py ls AuAu_c1 --what h5                      # a campaign's files, sizes and kinds
./js_osdf.py du -d 1                                   # space per folder
./js_osdf.py download AuAu_c1 --what h5 --to /scratch  # particlize + hadron files -> /scratch/AuAu_c1/
./js_osdf.py download AuAu_c1 --what root --to /scratch          # only the ROOT files
./js_osdf.py download 'AuAu_c1/*_000[1-5]_*' --to /scratch/few   # a pattern (quoted): a few jobs
./js_osdf.py download AuAu_c1 --what h5 --to /scratch --dry-run  # what would come, and how much
```

| `--what` | files | for |
|---|---|---|
| `h5` | `<stem>_particlize.h5`, `<stem>_hadrons_*.h5`, the hadronization logs | `HadronFileReader`, the notebooks, `run_h5toROOT.py` (it brings the particlize files the reader needs) |
| `root` | `<stem>_hadrons.root`, `<campaign>_campaign.root` | ROOT analyses alone: `HadronFileReader.h`, the macros |
| `pair` | `<stem>.h5` (the hydro, ~0.17–0.29 GB per event) with its `.json`, `.xml`, `.log` | the hydro, `wake_observables.py`, FNO training |
| `all` (default) | everything in the folder | |

- **Resumable and checked:** files already there with the same size and CRC32C are skipped,
  so an interrupted download finishes when the command runs again; every file is checked
  after the download, and a failed one leaves nothing behind. `-j N` downloads N files at once.
- **Names:** `AuAu_c1` above stands for a campaign folder; `ls` shows what is there.
  Downloads of a folder go to `--to/<folder>/`, files and patterns to `--to/<file>`.
- **Measured:** ~69 MB/s down from a workstation (2026-10-05). On 2026-10-13 the namespace
  held `test/AuAu_0_10_jet_pth10-40_eta06_c1/` (68 ROOT files, 2.1 GB); `ls` and a download
  of its campaign file ran without a login.
- **Reading in place,** without downloading, in the analysis venv (pelicanfs is installed
  by default, §2). pelicanfs finds an OSDF cache (or the origin) for the path and hands back fsspec's
  HTTP file, which reads by byte ranges: only the parts asked for travel. uproot takes the URL;
  h5py takes the file object, for every file kind here (pair, particlize, hadron files, Blosc
  included, with `hdf5plugin` imported):

  ```python
  import uproot
  c = uproot.open("osdf:///fno4hic/test/AuAu_0_10_jet_pth10-40_eta06_c1/pth10-40_eta06_c1_campaign.root")
  sigma = c["windows"].arrays(["sigma_mb"], library="np")["sigma_mb"]   # 1.7 s, public

  import fsspec, h5py, hdf5plugin
  fs = fsspec.filesystem("osdf")
  with fs.open("/fno4hic/test/h5_read_test/AuAu_0_10_jet_st1_t000_0001_hadrons_bulk_jet.h5", "rb",
               cache_type="blockcache", block_size=2**20) as fo, h5py.File(fo, "r") as f:
      so = f["hadrons/sample_offsets"][:]
      p = f["hadrons/p"][so[3]:so[4]]                    # one oversample: 9,765 hadrons
  ```

  Both read public test files, so the two examples run as they are.

  - **Pass `cache_type="blockcache", block_size=2**20`.** fsspec's default (read-ahead, 5 MB)
    re-fetches overlapping ranges when HDF5 jumps back to its metadata. Measured with the
    seed-1 files served over HTTP ranges (2026-10-13):

    | read | default cache | blockcache 1 MB | blockcache 256 kB |
    |---|---|---|---|
    | one oversample of a 100 MB hadron file | 7 requests, 31.5 MB | 6, 6.3 MB | 11, 2.9 MB |
    | one τ frame of a 245 MB pair file | 7 requests, 32.6 MB | 6, 6.3 MB | 9, 2.4 MB |

    Each request costs the round trip to the cache (tens of ms), so 1 MB blocks are the usual
    balance.
  - **Tested (2026-10-13):**
    - **On the OSDF:** a 2.8 MB Au+Au hadron file uploaded to `/fno4hic/test/h5_read_test/`
      and read back with h5py over pelicanfs, without credentials. Every dataset and attribute
      was identical to the local file, with both cache settings.
    - **Time:** the first open in a process takes ~1.1–1.5 s (pelicanfs finds a cache), after
      that one oversample ~0.1–0.2 s. A file that small fits in the default 5 MB read-ahead, so
      there the cache setting changes nothing; it matters for the data moved from large files
      (the table above, measured on the pair and hadron files over a local range server).
    - uproot on the OSDF; pelicanfs random reads (1,000 bytes from the middle of a 30 MB file:
      0.7 s).
  - **When to download instead:** for whole campaigns, and for anything read more than once.
    In place, every pass goes over the network again. `HadronFileReader`, `ParticlizeFile`,
    `HadronFile`, `run_h5toROOT.py` and the `analysis/` scripts also take local paths only.
- **Google Cloud Storage** (`gs://test_fno`) works the same way with `js_gcs.py`, but the
  bucket is not public: it needs a service-account key
  ([credentials](../utils/remote_transfer/README.md#credentials)).
- **Inside the production container** the `pelican` command line does the same for single
  files (`pelican object get osdf:///fno4hic/… /work/…`).

### Production settings an analysis must know

None of these shows in the data arrays. Check them before combining files:

| setting | why it matters | where it is recorded |
|---|---|---|
| **MUSIC's late time step** (D1; on by default from the `cu124-final_optim` image, 2026-10-13): Δτ 0.02 → 0.05 fm/c from τ = 3 fm/c | The stored frames and the freeze-out checks stay at 0.1 fm/c, so the files look alike. But the medium differs from the constant step at 1e-3 (the GPU-vs-CPU difference is 2e-3), the cell-level wake δe by ~5 % at scales below ~1 fm (0.8 % after 2 fm smoothing), and the O+O freeze-out surface by +0.2 %. Hadron observables agree within ±2 % ([`REPORT_FINAL_OPTIM.md`](REPORT_FINAL_OPTIM.md)). **Don't mix late-step and constant-step files** in one training set or one comparison | `<late_delta_tau>` in the job XML (attribute `prod_user_xml`, file `<stem>.xml`); absent or 0: constant step |
| the image or build | MUSIC on the GPU and on the CPU differ at 2e-3; images with and without C4 (the solver stall break) at float rounding. One image per campaign | not in the files: the campaign's notes (image digest, see [`BuildContainerProd.md`](../utils/BuildContainerProd.md)); `prod_build`, `prod_host` |
| the output grid | 64 × 64 × 32 (±12.2 fm, from 2026-10-06) or 65 × 65 × 33 (±10 fm, earlier): never in one training set | `prod_grid_yaml`; `nx`, `dx`, … in the pair file |
| hadron precision and selection | `--keep-bits-*`, `--eta-max`, `--charged`, `--no-x`: one setting per campaign | `HadronFileReader.precision(i)`, `.eta_max()`, `.selection()`; the reader warns on a mix |
| correlated sampling | `--correlated` pairs sample k of a jet leg with sample k of its background; the errors must be paired | `HadronFileReader.correlated` |
| pT̂ windows, parton rapidity cut, reuse | each window has its own cross section; events sharing a background are correlated | `pthat_bins`, `pthat_bin_sigma_gen`, `parton_ymax`, `prod_reuse`; `events/bg_unit` |
| the liquefier normalisation (X-SCAPE #155) and the LBT double counting (#154), both 2026-09-28 | files made before need a rerun (the jet leg) or `fix_lbt_double_counting.py` (the partons) | pair file: `liquefier_normalize_on_hydro_grid`; `lbt_double_count_fix` |

The settings of every production file of a campaign *(tested)*:

```python
import glob, h5py, hdf5plugin, yaml              # hdf5plugin registers the Blosc filter
import xml.etree.ElementTree as ET

for path in sorted(glob.glob("/path/to/campaign/**/*_particlize.h5", recursive=True)):
    with h5py.File(path, "r") as f:
        A = f.attrs
        late = ET.fromstring(A["prod_user_xml"]).findtext(".//Hydro/MUSIC/late_delta_tau")
        grid = yaml.safe_load(A["prod_grid_yaml"])["grid"]
        print(path.rsplit("/", 1)[-1], "| seed", A["prod_seed"],
              "| late_delta_tau", late.strip() if late else "absent (constant step)",
              "| grid x", grid["x"], "| reuse", A.get("prod_reuse"))
```

The pair files carry the same attributes (`prod_user_xml`, `prod_grid_yaml`, ...).

---

## 2. Environments

| environment | for | has | doesn't have |
|---|---|---|---|
| **A. the analysis venv** (`utils/analysis_env`, recommended) | every HDF5 file, `run_h5toROOT.py`, the ROOT files through uproot, the notebooks, FastJet in Python, the visualization | h5py + hdf5plugin, uproot, awkward, scipy, matplotlib, `fastjet`, `vector`, Jupyter, PyJetscape (editable) | X-SCAPE, ROOT |
| **B. conda with ROOT + the venv** | A, plus PyROOT, the `root` prompt, the macros of `analysis_root`, RDataFrame | A + ROOT ≥ 6.34 | X-SCAPE |
| **C. `js_fno`** (the X-SCAPE build env) | everything here, plus the productions themselves | ROOT 6.36, scikit-hep `fastjet`, FastJet C++ (`fastjet-cxx`, `fastjet-contrib`) | |
| **D. the production container** | re-hadronizing (`hadronize.py`) on CPU nodes | X-SCAPE, iSS, the readers | Jupyter, ROOT |

### A. The analysis venv

```bash
git clone https://github.com/jhputschke/js-contrib.git && cd js-contrib
./utils/analysis_env/setup_analysis_env.sh            # venv in ~/.venvs/js_analysis (~1.6 GB)
source ~/.venvs/js_analysis/bin/activate
python utils/analysis_env/check_env.py /path/to/campaign   # opens every file in the directory
```

| option | |
|---|---|
| `--uv` | install with uv, much faster |
| `--no-viz` | without pyvista/vtk/ffmpeg (~0.6 GB less) |
| `--kernel NAME` | register a Jupyter kernel for the venv (for VS Code, another JupyterLab) |
| `--no-pelican` | without pelicanfs, which reads files in place over `osdf://` (fsspec) |
| `--with-gcs` | read files in place over `gs://` (fsspec) |
| `--system-site-packages` | see the packages of the Python the venv is made from, e.g. PyROOT (B) |
| `--with-fasthydro` | FastHydro's readers (`fasthydro.browse.open_pair`); not needed for the production files |

The script also downloads MUSIC's hotQCD EoS table and sets `MUSIC_EOS_TABLE` (e → T, §3).
All options and the tests: [`utils/analysis_env/README.md`](../utils/analysis_env/README.md).

### B. ROOT next to it

conda gives ROOT and its Python; the venv is made from that Python:

```bash
conda create -n root -c conda-forge "root>=6.34"           # RNTuple needs ROOT >= 6.34
conda activate root
./utils/analysis_env/setup_analysis_env.sh ~/.venvs/js_analysis_root --system-site-packages
```

Then `source ~/.venvs/js_analysis_root/bin/activate` has PyROOT. For the `root` prompt and
compiled macros, `conda activate root` (without it ACLiC doesn't find the system headers).
The one-env alternative is `conda env create -f utils/analysis_env/environment.yml` (not
tested yet). Details: [*With ROOT*](../utils/analysis_env/README.md#with-root).

### FastJet

| | Python | C++ (ROOT macros, programs) |
|---|---|---|
| package | scikit-hep `fastjet` from PyPI (in `requirements.txt`): the FastJet library in the wheel, the classic SWIG interface and the array (awkward) interface | conda-forge `fastjet-cxx` (FastJet, `fastjet-config`) and `fastjet-contrib` (fjcontrib, one library `libfastjetcontribfragile`); or a system or source install |
| install | in the venv already | `conda install -c conda-forge fastjet-cxx fastjet-contrib` in the ROOT env |
| fjcontrib | only four routines, through the array interface: SoftDrop grooming, Lund declusterings, energy correlators, N-subjettiness | all of it (`#include <fastjet/contrib/SoftDrop.hh>`, `ConstituentSubtractor.hh`, ...; link `-lfastjetcontribfragile`) |

Both were checked in `js_fno` (FastJet 3.5.1, fastjet-contrib 1.101): the C++ build of §9,
and a SoftDrop + ConstituentSubtractor test program.

### C. `js_fno`

It has everything, but `import jetscape` there loads the compiled `pyjetscape_core`, which
needs `PYTHIA8DATA`: `conda activate js_fno` sets it. Without it the import aborts with
`Assertion 'InternalHelperPythia.particleData.isParticle(id)' failed`. The venv (A) has no
core and doesn't need it.

### Threads and memory

The analysis tools are mostly single-threaded Python. Parallelise over files with processes,
and keep processes × threads at or below the cores:

| tool | parallel with | memory | measured |
|---|---|---|---|
| `HadronFileReader` | nothing (one process, one production file at a time); split a campaign by `events=` over processes for more | the units it reads: a background of 2000 oversamples is ~14 M hadrons, ~1.5 GB with the derived kinematics (estimate) | 4 events, 11 M hadrons: 2.7 s |
| `wake_observables.py -j N` | N processes over files; reads event k+1 while computing k | +~0.5 GB per process for the read-ahead | 300 events in ~1 min at `-j 10` |
| `wake_hadrons.py -j N` | N processes | | 300 events in ~3.5 min at `-j 10` |
| `run_h5toROOT.py -j N` | N processes | ~80 B per hadron of the largest tag: ~3.5 GB for a 15-event file at 400 oversamples (13.5 GB for a broken one) | 20 files in 2 min 40 s at `-j 4` |
| `hadron_distributions.C` (RDataFrame) | its `threads` argument | | 285 events in 21 s with 8 threads (`HadronFileReader`: ~6 min for the same histograms) |
| Blosc decompression (h5py) | `BLOSC_NTHREADS` | | helps little on the τ-sliced hydro chunks; leave it unset with `-j` or with a PyTorch `DataLoader` |
| numpy (OpenBLAS) | `OPENBLAS_NUM_THREADS` | | set it to 1 with `-j` (the productions do), or the processes oversubscribe the cores |
| `hadronize.py` (container or X-SCAPE build) | `-j` processes in `run_hadronize.py`, `OMP_NUM_THREADS` per process (iSS's yield loop) | ~1.4 GB per process up to ~1000 oversamples, 1.7 GB at 2000 | both legs at 500 oversamples, one core: 10.6 s per event, 8.8 s with `--correlated` (iSS Part C) |
| FastJet, classic interface | nothing; a Python loop over the particles | small | 20 oversamples of ~7,000 hadrons (the charged ones at \|η\| < 1.5 clustered), anti-kT with areas and ρ: 0.4 s |
| FastJet, array interface | one C++ call clusters a whole batch of samples | a batch with explicit ghosts is large: cluster in chunks (`CHUNK` in the notebook) | 100 samples: 0.2 s |

---

## 3. Reading the hydro

### The layout

| dataset | shape | |
|---|---|---|
| `arr` | `(nevents, 4, nx, ny, neta, ntau)` float32 | the **jet leg** (MUSIC_2) of a pair file, or the event of a hydro-only or O+O file. Channels `energy_density` [GeV/fm³], `vx`, `vy`, `vz` (Cartesian lab velocities) |
| `arr_bg` | as `arr` | the **background** (MUSIC_1) of a pair file. With a reused background it is a virtual dataset over `arr_bg_store`: read `arr_bg`, never the store |
| `ntau_freezeout`, `ntau_freezeout_bg` | `(nevents,)` | frames written per leg; every frame from there on is 0 |
| `tau_freezeout`, `tau_freezeout_bg` | `(nevents,)` | when each leg's hydro ended [fm/c] |
| `source/droplets`, `source/offsets` | `(M, 8)`, `(nevents + 1,)` | the droplets MUSIC_2 received: `tau, x, y, eta, E, px, py, pz`; event i is rows `offsets[i]:offsets[i+1]` |
| `shower/` | | the shower graphs (partons, vertices, initiators), read with `jetscape.showers` |
| `diag/` | `(nevents,)` each | `E_droplets`, `n_droplets`, `bg_id`, `frames_identical`, `{bg,jet}_hit_edge`, `{bg,jet}_hit_boundary`, `tau0_music`, `eloss_tstart` (where the jets' medium interaction started), `wall_s`, with pT̂ windows `pthat_bin`, `pthat`, `event_weight` |

- **The grid** is in the attributes: `x_min`, `dx`, `nx` (and `y_*`, `eta_*`), `tau_min`,
  `dtau`. Production default (`grid_fno.yaml`): 64 × 64 × 32 cells, x, y −12.2 … 12.2 fm at
  0.3875 fm, η_s −4.84 … 4.84 at 0.3125, τ = 0.5 + k · 0.1 fm/c. O+O (`grid_oo.yaml`):
  64 × 64 × 32 at 0.3125 fm, ±9.84 fm. MUSIC's own grid is in the `*_MUSIC` attributes.
- **The τ axis is the same with the late step:** frames every 0.1 fm/c either way (MUSIC
  stores every 5th step of 0.02 before τ = 3 fm/c, every 2nd step of 0.05 after).
- **One chunk per event and τ frame** (`(1, 4, nx, ny, neta, 1)`, Blosc-zstd). Read frame by
  frame (`f["arr"][i, ..., k]`, ~2 MB): a cell's whole history reads every chunk.
- **The two legs freeze out at different times:** the deposit reheats the jet leg (by 1–2 fm/c
  in 0–10 % Au+Au). After the background's freeze-out, `arr − arr_bg` is the jet leg's whole
  medium, not a wake. Compare only frames below both `ntau_freezeout` values.
- **Hydro-only files** (`prod_AuAu_0_10`) have `arr` only; **O+O files** have `arr`, `diag/`
  with `glauber_*`, `isr_sub_*`, `has_medium`, and `hydro_row` links the other files to it.

### Reading a pair *(tested)*

```python
import os
import h5py, hdf5plugin, numpy as np            # hdf5plugin: without it Blosc data won't read

with h5py.File("AuAu_0_10_jet_c1_0001.h5", "r") as f:
    A = f.attrs
    x   = A["x_min"]   + A["dx"]   * np.arange(A["nx"])      # cell centres [fm]
    y   = A["y_min"]   + A["dy"]   * np.arange(A["ny"])
    eta = A["eta_min"] + A["deta"] * np.arange(A["neta"])
    tau = A["tau_min"] + A["dtau"] * np.arange(f["arr"].shape[-1])   # 0.5, 0.6, ... fm/c

    i, k = 0, 40                                   # event 0, frame tau = 4.5 fm/c
    jet = f["arr"][i, :, :, :, :, k]               # (4, nx, ny, neta): e, vx, vy, vz
    bg  = f["arr_bg"][i, :, :, :, :, k]            # the same collision without the jet
    de  = jet[0] - bg[0]                           # the wake in e [GeV/fm^3]
    live = min(f["ntau_freezeout"][i], f["ntau_freezeout_bg"][i])   # frames where both legs live

    o = f["source/offsets"]
    droplets = f["source/droplets"][o[i]:o[i + 1]]   # tau, x, y, eta, E, px, py, pz
    E_dep = droplets[:, 4].sum()                   # = diag/E_droplets[i]

    # e -> T with MUSIC's hotQCD table (columns e, P, s, T); the venv sets MUSIC_EOS_TABLE
    tab = np.fromfile(os.environ["MUSIC_EOS_TABLE"], dtype="<f8").reshape(-1, 4)
    tab = tab[np.argsort(tab[:, 0])]
    T = np.interp(jet[0], tab[:, 0], tab[:, 3])    # GeV; T_fo = 0.15 at e = 0.234
```

### Readers and tools for the hydro

| | what |
|---|---|
| FNO4d `loc_libs.read_3d_hdf5.read_3d_data_hdf5(files, h5_key="arr"/"arr_bg", unique_bg=...)` | a lazy `MultiH5Array` over many files, as the FNO training reads them; `unique_bg` drops the repeated backgrounds of a reuse campaign (from `diag/bg_id`) |
| `jetscape.fast_h5_bulk.read_fast_h5_bulk(path)` | hydro-only files: each event as `(ntau, nx, ny, neta, 4)`, trimmed at its freeze-out |
| `python ../../python/jetscape/repad_h5.py FILES` | brings the files of a campaign to one τ length, in place (a metadata resize) |
| `fasthydro.browse.open_pair` (`--with-fasthydro`) | FastHydro's `PairBrowser`: `p.diff(event, frame)`, `p.summary(event)` |
| [`jet_wake.ipynb`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/jet_wake.ipynb) | one pair from e to hadrons, step by step (`WAKE_H5=<pair file>`) |
| [`analysis/wake_observables.py`](../contribs/PyJetscape/example/analysis/README.md#6-wake-observables-script-then-notebook) | one pass over a campaign's pair files: P^μ and S of both legs through every τ, the jet-frame wake, the showers' path lengths → `wake_observables.h5` |
| [`check_output.ipynb`](../contribs/PyJetscape/example/prod_AuAu_0_10/check_output.ipynb), [`oo_isr_overview.ipynb`](../contribs/PyJetscape/example/prod_OO_isr/oo_isr_overview.ipynb) | hydro-only and O+O files |
| [`contribs/Visualization`](../contribs/Visualization/README.md) | `wake_pyvista.py`, `hydro_jet_particles_pyvista.py`: 3D views and movies of a pair file |

The shower graph in `shower/` is read with `jetscape.showers` (`fates`, `segments`,
`to_milne`); `analysis/jet_edep_balance_check.py` shows its use for the energy balance.

---

## 4. Reading the particlize files

The input to hadronization, and the event table of a hadron analysis *(tested)*:

```python
from jetscape.particlize_h5 import ParticlizeFile, SURFACE_COLUMNS, FINAL_PARTON_COLUMNS

with ParticlizeFile("AuAu_0_10_jet_c1_0001_particlize.h5") as pf:
    cells    = pf.surface("jet", 0)          # (N, 32) float32: MUSIC_2's surface, event 0
    cells_bg = pf.surface("bg", 0)           # the background event 0 used (events/bg_unit)
    partons  = pf.partons(0)                 # (K, 14): shower, pid, pstat, E, px, py, pz, t, x, y, z, mass, col, acol
    ini      = pf.initiators(0)              # (J, 11): shower, pid, pstat, px, py, pz, E, x, y, z, t
    n_cells  = pf.events("n_cells_jet")      # the per-event table: bg_unit, droplets, pthat_bin, ...
    T_fo     = cells[:, SURFACE_COLUMNS.index("T")]
```

- **Surface columns** (`SURFACE_COLUMNS`): `tau, x, y, eta, ds0..ds3, u0..u3, e, T, P, nB, nQ,
  nS, muB, muQ, muS`, the 10 components of π^{μν}, `Pi`: exactly what iSS reads, lossless.
- **Partons** carry the liquefier's status: 0 shower, 1 recoil, −1 hole, −11/−13/−17 went into
  the droplets (never hadronized), 22 photon.
- **`events/`**: `bg_id`, `bg_unit`, `bg_key`, `n_cells_jet`, `n_cells_bg`, `n_partons`,
  droplet energies, MUSIC's τ0, the boundary flags, and with windows `pthat_bin`, `pthat`.
  A background with far more or fewer cells than its jet leg did not freeze out (§5, *bad
  runs*).

---

## 5. Reading the hadrons

### The layout

Each tag is one file (`jetscape.hadrons_h5`, format `xscape/hadrons`):

| | |
|---|---|
| `hadrons/pid`, `pstat` | int32, one row per hadron |
| `hadrons/p` | float32 `(n, 4)`: E, px, py, pz [GeV] |
| `hadrons/x` | float32 `(n, 4)`: t, x, y, z [fm]; absent with `--no-x` |
| `hadrons/sample_offsets` | where each sample (oversample or fragmentation) starts |
| `hadrons/unit_offsets` | where each unit's samples start; a unit is an event (`bulk_jet`, `jet_frag`) or a background (`bulk_bg`) |
| `units/` | per unit: `unit`, `event`, `n_samples`, `seed`, `n_cells`; O+O: the event's whole record (`pthat`, `glauber_*`, `isr_sub_*`, `hydro_row`, `has_medium`) |
| `initiators/` | `bulk_jet`, `jet_frag`: each event's shower initiators |

### What an event is, and the statistics

- **A jet event** is oversample k of `bulk_jet` plus fragmentation k mod `n_frag` of
  `jet_frag`; its background is sample k of `bulk_bg` unit `events/bg_unit`.
- **Oversamples are not independent collisions.** The N oversamples of an event sample one
  fluid. Average each event over its samples first, then over events, and take errors per
  event (`HadronFileReader` does both).
- **Events that share a background** (`--reuse`, `--pthat-bins`) are correlated through it.
  Per-window results and jet − background are fine; spectra stitched over windows have
  correlated errors.
- **`--correlated` files:** sample k of the jet leg and sample k of its background share
  most hadrons. jet − background is then ~8× less noisy at \|η\| < 1, but only the paired
  error (the spread of the per-sample differences) is right. `jet_minus_background` picks it
  automatically.

### `HadronFileReader`: a whole campaign *(tested)*

```python
import numpy as np
from jetscape.hadrons_h5 import HadronFileReader

def lead(info):
    """phi and pseudorapidity of the event's hardest shower-initiating parton."""
    ini = info.initiators()                     # (K, 11): shower, pid, pstat, px, py, pz, E, ...
    px, py, pz = ini[np.argmax(np.hypot(ini[:, 3], ini[:, 4])), 3:6]
    return np.arctan2(py, px), np.arcsinh(pz / np.hypot(px, py))

def dphi(ev, info):                             # azimuth relative to the leading parton
    return np.mod(ev.phi - lead(info)[0] + np.pi / 2, 2 * np.pi) - np.pi / 2

def soft_charged(ev, info):
    return ev.charged & (ev.pt > 0.2) & (ev.pt < 4) & (np.abs(ev.eta) < 1)

with HadronFileReader("/path/to/campaign") as r:          # every *_particlize.h5 below it
    print(r.n_files, "files,", r.n_events, "events, tags", r.tags())

    # names of EventHadrons attributes: per-event sample means, averaged over events
    pt, pt_err = r.hist("bulk_bg", "pt", np.linspace(0, 3, 31), mask="charged")
    n_mid, n_err = r.total("bulk_jet", mask=lambda ev, info: ev.charged & (np.abs(ev.eta) < 1))

    # the wake: <jet leg> - <its own background>, event by event
    bins = np.linspace(-np.pi / 2, 3 * np.pi / 2, 17)
    wake, wake_err = r.jet_minus_background(dphi, bins, mask=soft_charged)
    full, _ = r.jet_minus_background(dphi, bins, mask=soft_charged, fragments=True)  # + jet_frag

    # single events, numbered over all files
    ev = r.jet_event(0, 17)                     # dict: pid, pstat, p, x, origin (0 bulk, 1 fragment)
    bg = r.background_event(0, 17)
    info = r.event_info(0)                      # stem, local_event, bg_unit, seed, pthat_bin, initiators()
```

- **`values`, `mask`, `weights`** are names of `EventHadrons` attributes (`pid`, `pstat`,
  `p`, `x`, `E`, `px`, `py`, `pz`, `pt`, `eta`, `y`, `phi`, `charged`, `sample`), `"charged"`
  or a species name, or callables `f(ev, info)`. A tuple gives an N-d histogram.
- **Names are faster:** a shared background is then binned once and counted for each of its
  events. With a callable that doesn't use `info`, pass `per_event=False` for the same.
- **`events=`** restricts any `hist`, `total` or `jet_minus_background`, e.g. to a window.
- **Checks on open:** a hadron file from another run (`source_uuid`) is refused; mixed
  precisions, η cuts or selections, and backgrounds that appear in two files (campaigns
  over the same seeds) give warnings (`r.duplicate_backgrounds()`).

**pT̂ windows and cross sections** (`--pthat-bins` campaigns):

```python
with HadronFileReader("/path/to/campaign") as r:
    r.pthat_bins                                   # (K, 2): the windows
    edges = np.linspace(0, 40, 41)
    spec, var = np.zeros(40), np.zeros(40)
    for k in range(len(r.pthat_bins)):
        h, e = r.hist("jet_frag", "pt", edges, mask="charged", events=r.pthat_bin_events(k))
        sigma, _ = r.pthat_bin_sigma(k)            # mb, after the parton-y cut
        spec += sigma * h                          # each window: per-event mean x its sigma
        var += (sigma * e) ** 2                    # jet_frag: windows independent
```

`pthat_bin_sigma` combines the files' Pythia estimates weighted by their accepted events;
`pthat_bin_acceptance(k)` gives the parton-y cut's acceptance.

### Lower-level readers

```python
from jetscape.hadrons_h5 import HadronFile, JetEvents, Hadrons

with HadronFile("<stem>_hadrons_bulk_jet.h5") as hf:       # random access, nothing loaded
    one = hf.sample_event(0, 17)                            # unit (event) 0, sample 17: pid, pstat, p, x
with JetEvents.from_stem("<stem>") as je:                   # the three files + particlize
    ev = je.jet_event(0, 17)                                # bulk_jet sample 17 + a fragmentation

h = Hadrons.from_h5("OO_isr_c1_0001_hadrons_bulk_bg.h5")    # a whole file in memory
dn, err = h.hist(h.eta, np.linspace(-1, 1, 5), mask=h.charged)   # per sample
```

**O+O hadrons made in the job** (`--bulk-hadrons`) have no `source_uuid`, so
`HadronFileReader` refuses them. Read them with `Hadrons` or `HadronFile`: unit i of
`_hadrons_bulk_bg.h5` and unit i of `_hadrons_jet_frag.h5` are the same event, and the
units carry the event record (`units/pthat`, `glauber_npart`, `hydro_row`, ...). The O+O
guide, [`README_OO.md`](README_OO.md), has a campaign loader, the weights and recipes.

### Bad runs

Two rare failures show up in the data, not in the logs. Check for them before an analysis:

- **A leg that never froze out** (a MUSIC4GPU stall, seen in one job of the gridnorm
  campaign): its background has ~3× the freeze-out cells and ~11× the hadrons of its jet
  leg. `wake_hadrons.py`, `hadron_distributions.C` and `HadronFileReader.h` drop events
  whose `n_cells_bg / n_cells_jet` is off by more than 20 %.
- **A job whose jets lost almost no energy** (once in ~250 campaign jobs, 2026-10-12, not
  reproducible): `droplets_total / events_written` in its `.json` far below the campaign's.
  [`README_2stage.md`](README_2stage.md) §2 has the check.

---

## 6. Jets with FastJet in Python

### Conventions of the notebooks

- **Particles:** \|η\| < 1.5, pT > 0.15 GeV, charged by default, clustered as **massless**
  (E = \|p\|) so that rapidity = pseudorapidity and the ghosts cover the acceptance exactly.
- **Jets:** anti-kT, R = 0.4, \|η_jet\| < 1.5 − R; active areas with explicit ghosts (ghost
  area 0.005).
- **ρ:** the median of pT/A of the kT jets (R = 0.4) in \|η\| < 1.5 − R without the two
  hardest, with `set_use_area_4vector(False)` (the scalar area that pT − ρA uses; FastJet 3
  otherwise divides by the 4-vector area's transverse part, ~1 GeV larger ρ).
- **Three events per oversample k:**

  | event | hadrons | tests |
  |---|---|---|
  | bkg | `bulk_bg` sample k | ρ, random cones, combinatorial jets |
  | bkg + jet | `bulk_bg` sample k + `jet_frag` sample k mod n_frag | embedding: the jet on a background it didn't touch, ⟨δpT⟩ = 0 |
  | bkg + wake + jet | `bulk_jet` sample k + the same `jet_frag` sample (`r.jet_event(e, k)`) | the full event, with the medium response |

- **`user_index`** tags each particle with its row; offsetting the `jet_frag` rows tells a
  jet's fragment content exactly.

### The classic interface *(tested)*

```python
import numpy as np
import fastjet as fj
from jetscape.hadrons_h5 import CHARGED, HadronFileReader

ETA_MAX, R = 1.5, 0.4
JET_DEF  = fj.JetDefinition(fj.antikt_algorithm, R)
KT_DEF   = fj.JetDefinition(fj.kt_algorithm, R)
AREA_DEF = fj.AreaDefinition(fj.active_area_explicit_ghosts, fj.GhostedAreaSpec(ETA_MAX, 1, 0.005))
RHO_SEL  = (~fj.SelectorNHardest(2)) * fj.SelectorAbsRapMax(ETA_MAX - R)
JET_SEL  = fj.SelectorAbsRapMax(ETA_MAX - R) * fj.SelectorPtMin(10.0)
CHARGED_PID = np.array(CHARGED)

def pseudojets(ev, charged=True, pt_min=0.15):
    """Massless PseudoJets of one sample; user_index = the hadron's row in the sample."""
    E, px, py, pz = np.asarray(ev["p"], dtype=np.float64).T
    pt = np.hypot(px, py)
    eta = np.arcsinh(pz / np.maximum(pt, 1e-12))
    keep = (np.abs(eta) < ETA_MAX) & (pt > pt_min)
    if charged:
        keep &= np.isin(np.abs(ev["pid"]), CHARGED_PID)
    out = []
    for i in np.flatnonzero(keep):
        pj = fj.PseudoJet(px[i], py[i], pz[i], np.hypot(pt[i], pz[i]))
        pj.set_user_index(int(i))
        out.append(pj)
    return out

r = HadronFileReader("/path/to/campaign")
pt_sub, weight = [], []
for e in r.events():
    n = r.n_oversamples(e)
    for k in range(n):
        parts = pseudojets(r.jet_event(e, k))               # bkg + wake + jet
        cs = fj.ClusterSequenceArea(parts, JET_DEF, AREA_DEF)
        bge = fj.JetMedianBackgroundEstimator(RHO_SEL, KT_DEF, AREA_DEF)
        bge.set_use_area_4vector(False)
        bge.set_particles(parts)
        rho = bge.rho()
        for jet in fj.sorted_by_pt(JET_SEL(cs.inclusive_jets())):
            pt_sub.append(jet.pt() - rho * jet.area())
            weight.append(1.0 / (n * r.n_events))           # per event: mean over its samples
spectrum, edges = np.histogram(pt_sub, np.arange(-20, 104, 4.0), weights=weight)
```

### The array interface (awkward) *(tested)*

One `ClusterSequence` call clusters all samples of a unit in C++, with no Python loop
(continues the block above: `np`, `CHARGED_PID`, `r`):

```python
import awkward as ak, vector, fastjet as fj
from jetscape.hadrons_h5 import HadronFile
vector.register_awkward()

def samples(path, unit, n=None):
    """The samples of one unit as samples * var * Momentum4D (massless)."""
    with HadronFile(path) as hf:
        u = hf.unit_index(unit)
        s0, s1 = int(hf.unit_offsets[u]), int(hf.unit_offsets[u + 1])
        s1 = s1 if n is None else min(s1, s0 + n)
        so = hf.sample_offsets[s0:s1 + 1]
        p = hf.f["hadrons/p"][so[0]:so[-1]].astype(np.float64)
        pid = hf.f["hadrons/pid"][so[0]:so[-1]]
    rec = ak.zip({"px": p[:, 1], "py": p[:, 2], "pz": p[:, 3],
                  "E": np.linalg.norm(p[:, 1:], axis=1),
                  "charged": np.isin(np.abs(pid), CHARGED_PID)}, with_name="Momentum4D")
    return ak.unflatten(rec, np.diff(so))

info = r.event_info(0)
ev = samples(f"{info.stem}_hadrons_jet_frag.h5", info.local_event)      # the jet hadrons alone
ev = ak.to_packed(ev[(abs(ev.eta) < 1.5) & (ev.pt > 0.15) & ev.charged])
cs = fj.ClusterSequence(ev, fj.JetDefinition(fj.antikt_algorithm, 0.4))  # all samples at once
jets = cs.inclusive_jets(min_pt=10.0)                                     # samples * jets
cons = cs.constituent_index(min_pt=10.0)                                  # indices into ev
```

- **Supported:** inclusive and exclusive jets, constituents and their indices
  (`constituent_index` stands in for `user_index`), and fjcontrib's SoftDrop, Lund
  declusterings, energy correlators and N-subjettiness.
- **Not supported:** areas, background estimators, `Selector`s, `user_index`. A raw jet in
  the Au+Au bulk carries ρπR² of background, 25–30 GeV for charged hadrons in a 0–10 %
  event, so the subtraction needs the classic interface, or explicit ghosts added as
  particles, which `jet_fastjet_awkward.ipynb` §6 does.

### The notebooks

| notebook | what |
|---|---|
| [`jet_fastjet.ipynb`](../contribs/PyJetscape/example/analysis/tutorials/jet_fastjet.ipynb) | classic interface: the jet spectrum and fragmentation function of the jet hadrons alone; is the area right (tiling, πR², ghost size); ρ with and without the two hardest kT jets, random cones; the spectrum with and without ρA; embedding δpT for bkg + jet and bkg + wake + jet |
| [`jet_fastjet_awkward.ipynb`](../contribs/PyJetscape/example/analysis/tutorials/jet_fastjet_awkward.ipynb) | the same with the array interface, plus exclusive jets, SoftDrop, Lund planes, energy correlators, N-subjettiness, and areas with hand-made ghosts |

Both take `HADRON_DIR`, `MAX_EVENTS`, `MAX_OVERSAMPLES` and `PARTICLES` (`charged` or `full`)
from the environment:

```bash
HADRON_DIR=/path/to/campaign MAX_EVENTS=200 MAX_OVERSAMPLES=100 \
  jupyter nbconvert --to notebook --execute contribs/PyJetscape/example/analysis/tutorials/jet_fastjet.ipynb
```

The ghosts are placed with random jitter at every clustering (the Python bindings cannot set
FastJet's ghost seed), so areas and ρ change slightly from run to run; `jet_fastjet.ipynb` §4
measures by how much.

---

## 7. Notebooks and scripts

Everything here runs in the analysis venv. Paths are under
[`contribs/PyJetscape/example`](../contribs/PyJetscape/example/README.md).

| file | reads | what | settings |
|---|---|---|---|
| [`prod_AuAu_0_10_jet/jet_wake.ipynb`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/jet_wake.ipynb) | a pair file + its hadron files | one event from the energy density, the deposit and the freeze-out to hadrons and the energy balance | `WAKE_H5`, `MUSIC_EOS_TABLE` |
| [`analysis/wake_observables.py`](../contribs/PyJetscape/example/analysis/README.md#6-wake-observables-script-then-notebook) + [`.ipynb`](../contribs/PyJetscape/example/analysis/wake_observables.ipynb) | a campaign's pair files | P^μ, S of both legs vs τ, the jet-frame wake, showers and path lengths → `wake_observables.h5`; the figures from it | `-j`, `-o`; notebook `WAKE_OBS` |
| [`analysis/wake_hadrons.py`](../contribs/PyJetscape/example/analysis/README.md#7-the-wake-at-hadron-level-script-then-notebook) + [`.ipynb`](../contribs/PyJetscape/example/analysis/wake_hadrons.ipynb) | the hadron files | the wake at hadron level per event: jet frame, ΔR profiles, identified spectra, totals; batch errors for correlated files → `wake_hadrons.h5` | `-j`, `--batches`; notebook `WAKE_HAD`, `WAKE_OBS` |
| [`analysis/tutorials/hadron_distributions.ipynb`](../contribs/PyJetscape/example/analysis/tutorials/hadron_distributions.ipynb) | the hadron files | η, φ, pT of background, background + deposition, fragments and the wake | `HADRON_DIR`, `MAX_EVENTS`, `PAIRED` |
| [`analysis/tutorials/jet_fastjet.ipynb`](../contribs/PyJetscape/example/analysis/tutorials/jet_fastjet.ipynb), [`jet_fastjet_awkward.ipynb`](../contribs/PyJetscape/example/analysis/tutorials/jet_fastjet_awkward.ipynb) | the hadron files | jets (§6) | `HADRON_DIR`, `MAX_EVENTS`, `MAX_OVERSAMPLES`, `PARTICLES` |
| [`analysis/jet_edep_balance_check.py`](../contribs/PyJetscape/example/analysis/README.md#4-the-two-scripts) | pair files | the energy balance per event; finds doubly counted partons | |
| [`analysis/fix_lbt_double_counting.py`](../contribs/PyJetscape/example/analysis/README.md#4-the-two-scripts) | pair + particlize files | repairs productions made before X-SCAPE #154 (writes, reversibly) | `--apply`, `--revert` |
| [`analysis_root/read_hadrons.ipynb`](../contribs/PyJetscape/example/analysis_root/read_hadrons.ipynb) | the ROOT files | `read_hadrons.C` cell by cell, in ROOT's C++ kernel (§8) | `HADRON_DIR`, `HADRON_OUT`, `MAX_SAMPLES` |
| [`prod_AuAu_0_10/check_output.ipynb`](../contribs/PyJetscape/example/prod_AuAu_0_10/check_output.ipynb) | hydro-only files | sanity scan, e and flow vs τ, freeze-out times | `PROD_H5_GLOB` |
| [`prod_OO_isr/oo_isr_overview.ipynb`](../contribs/PyJetscape/example/prod_OO_isr/oo_isr_overview.ipynb) | one O+O job | the medium at three τ with the dijet; η, φ, pT, Δφ of bulk and jet hadrons | `OO_STEM`, `XSCAPE_BUILD` (EoS) |
| [`prod_OO_isr/hard_sub_compare/hard_sub_compare.ipynb`](../contribs/PyJetscape/example/prod_OO_isr/hard_sub_compare/README.md) | two O+O campaigns | with and without the hard-momentum subtraction | `HSC_DIR` |

Run a notebook headless with `jupyter nbconvert --to notebook --execute --inplace NB.ipynb`,
after setting its variables. The notebooks name the kernel they were saved with; in the venv
choose `Python 3 (ipykernel)`, or the kernel registered with `--kernel`.

---

## 8. ROOT: converting and analysing

### Converting a campaign: `run_h5toROOT.py`

[`run_h5toROOT.py`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/run_h5toROOT.py) turns
every hadronized production file into one ROOT file. It runs in the venv: with uproot, or
with ROOT when PyROOT imports.

```bash
cd contribs/PyJetscape/example/prod_AuAu_0_10_jet
python run_h5toROOT.py /path/to/campaign -j 4                        # RNTuple, next to the inputs
python run_h5toROOT.py /path/to/campaign -j 4 --format ttree         # TTree, for ROOT < 6.34
python run_h5toROOT.py /path/to/campaign -j 4 --out-dir root --no-x --charged --eta-max 1
python run_h5toROOT.py /path/to/campaign --dry-run                   # the plan only
```

| option | |
|---|---|
| `-j N` | files in parallel; ~80 B per hadron of the largest tag each |
| `--format {rntuple,ttree}` | RNTuple (default: 0.9× the HDF5 size, ~1.8× faster to read than a TTree) needs ROOT ≥ 6.34 to read |
| `--writer {auto,root,uproot}` | `auto`: ROOT if PyROOT imports (byte-split floats, 16–24 % smaller), else uproot |
| `--out-dir DIR` | where the ROOT files go (default: next to the inputs) |
| `--no-x`, `--charged`, `--eta-max X` | leave out positions (40 % of the bytes), neutral hadrons, \|η\| ≥ X |
| `--bits-p B`, `--bits-x B` | truncated floats (needs the ROOT writer); not useful on hadrons already rounded with `--keep-bits` |
| `--tags` | a subset of `bulk_jet,bulk_bg,jet_frag` |
| `--campaign-file NAME` / `none` | the campaign file's name, or none |
| `--force` | redo files already converted (else kept, with a warning if the settings differ) |

A file is converted once its particlize file and every tag's hadron file are complete;
outputs are written to `.part` and renamed, so rerunning the command finishes an
interrupted pass. Only the hadrons and the event information go to ROOT; the hydro stays in
HDF5 (read it with h5py, also from PyROOT).

**Per production file, `<stem>_hadrons.root`:**

| object | one entry per | content |
|---|---|---|
| `bulk_jet`, `bulk_bg`, `jet_frag` | oversample | `event`, `unit`, `sample`, `bg_unit`, `n`, `pid[n]`, `pstat[n]`, `E/px/py/pz[n]` (GeV), `t/x/y/z[n]` (fm, not with `--no-x`) |
| `events` | event | the particlize file's `events/`, `sigma_file_mb`, `n_samples_*`, `seed_*`, the initiators `ini_*[...]` |
| `windows` | pT̂ window | `pthat_lo/hi`, this file's `sigma_mb`, `sigma_err_mb`, `acceptance`, counts |
| `provenance` | — | JSON: every setting, seed, XML and EoS behind the file |

**`<campaign>_campaign.root`:** `windows` with the campaign's `sigma_mb`, `sigma_err_mb`,
`acceptance`, `n_events` and **`weight_mb = sigma_mb / n_events`**; `files` with
`event_offset` per file (global event = `event_offset[file] + event`, as in
`HadronFileReader`); `provenance`.

Layout, format study and the pairing of correlated samples:
[`root_export/README.md`](../contribs/PyJetscape/example/prod_AuAu_0_10_jet/root_export/README.md).

### Reading without ROOT: uproot *(tested)*

```python
import uproot, awkward as ak, numpy as np

f = uproot.open("AuAu_0_10_jet_c1_0001_hadrons.root")
jet = f["bulk_jet"].arrays(["event", "sample", "pid", "px", "py", "pz"], entry_stop=100)
pt = np.hypot(jet.px, jet.py)                     # jagged: samples * hadrons
ev = f["events"].arrays()                         # awkward: the ini_* columns are jagged
ini_pt = np.hypot(ev.ini_px, ev.ini_py)
c = uproot.open("pth10-40_c1_campaign.root")
if "windows" in c:                                # pT-hat-window campaigns only
    w = c["windows"].arrays(["sigma_mb", "weight_mb"])
```

Read `events` as awkward (the default): `library="np"` fails on the jagged initiator columns.
Files are also read in place over `osdf://` in the venv (pelicanfs, by default) and over
`gs://` with its `--with-gcs`.

### RDataFrame

A cross-section-weighted spectrum of the jet leg, every event the mean over its oversamples:

```cpp
ROOT::RDataFrame w("windows", "pth10-40_c1_campaign.root");
auto wt = *w.Take<double>("weight_mb");             // per window
ROOT::RDataFrame ev("events", "RUN_0001_hadrons.root");
auto nsamp = *ev.Take<int>("n_samples_jet");        // per event
auto bin   = *ev.Take<int>("pthat_bin");
ROOT::RDataFrame jet("bulk_jet", "RUN_0001_hadrons.root");
auto h = jet.Define("w", [&](int e) { return wt[bin[e]] / nsamp[e]; }, {"event"})
            .Define("pt", "sqrt(px*px + py*py)")
            .Histo1D({"pt", "", 100, 0, 5}, "pt", "w");
```

TTree and RNTuple read the same way in RDataFrame. `ROOT::EnableImplicitMT(n)` runs it on n
threads.

### C++: `HadronFileReader.h`

[`analysis_root/HadronFileReader.h`](../contribs/PyJetscape/example/analysis_root/README.md)
is header-only. It finds an event's entries in the three ntuples and returns their hadrons:

| call | the hadrons of |
|---|---|
| `r.bkg(e, k)` | the background (MUSIC_1), sample k of the background event e used |
| `r.bkg_dep(e, k)` | the jet leg (MUSIC_2): background + deposition |
| `r.bkg_dep_frag(e, k)` | the whole jet event: `bkg_dep(e, k)` + fragmentation k mod `n_frag(e)` |
| `r.frag(e, j)` | fragmentation j of the surviving partons |

```cpp
#include "HadronFileReader.h"
using namespace hadrons_root;

HadronFileReader r("/path/to/campaign");    // every *_hadrons.root, + its *_campaign.root
r.set_charged_eta(1.0);                     // optional filter while reading
for (long e : r.select_events()) {          // hadronized events, flagged ones left out
  const EventInfo &info = r.info(e);        // pthat_bin, bg_unit, initiators, leading(), weight_mb
  const int n = r.n_oversamples(e);
  for (int k = 0; k < n; ++k)
    for (const Hadron &h : r.bkg_dep_frag(e, k)) { /* h.pt(), h.eta(), h.phi(), h.p4(), h.origin */ }
}
```

- **Normalisation:** a hadron of `bkg_dep` or `bkg_dep_frag` weighs w_e / `n_oversamples(e)`,
  of `bkg` w_e / `n_bg_samples(e)`, of `frag` w_e / `n_frag(e)`, with w_e = 1/N per event or
  `sigma_mb(k) / N_k` for the cross section.
- **Errors** per event, not per sample; for correlated files pair `bkg_dep(e, k)` with
  `bkg(e, k)`.
- **The macros:** [`read_hadrons.C`](../contribs/PyJetscape/example/analysis_root/read_hadrons.C)
  (pT spectra and Δφ per source and for the wake; also as the notebook `read_hadrons.ipynb`)
  and [`hadron_distributions.C`](../contribs/PyJetscape/example/analysis_root/hadron_distributions.C)
  (η, φ, pT of every source with RDataFrame, multithreaded):

```bash
cd contribs/PyJetscape/example/analysis_root
root -l -b -q 'read_hadrons.C+("/path/to/campaign")'                       # per event
root -l -b -q 'read_hadrons.C+("/path/to/campaign", -1, true)'             # dsigma/dX [mb]
root -l -b -q 'hadron_distributions.C+("/path/to/campaign", -1, false, true, 1.0, "out.root", 8)'
```

- **PyROOT:** `ROOT.gInterpreter.Declare('#include "/path/HadronFileReader.h"')`, then
  `ROOT.hadrons_root.HadronFileReader(...)`.

Everything about the reader, cross sections and checks:
[`analysis_root/README.md`](../contribs/PyJetscape/example/analysis_root/README.md).

---

## 9. Jets with FastJet in ROOT and C++

With FastJet's C++ library next to ROOT (§2, *FastJet*), `HadronFileReader.h` hands the
hadrons straight to FastJet. This macro clusters the whole jet event of every oversample with
the conventions of §6 and fills the raw and ρA-subtracted spectrum *(tested)*:

```cpp
// jets_fastjet.C: anti-kT jets with rho*A subtraction on the ROOT files of run_h5toROOT.py
#include "HadronFileReader.h"

#include <fastjet/ClusterSequenceArea.hh>
#include <fastjet/Selector.hh>
#include <fastjet/tools/JetMedianBackgroundEstimator.hh>

#include <TFile.h>
#include <TH1D.h>

void jets_fastjet(const char *dir = "out_root", double eta_max = 1.5, double R = 0.4,
                  int max_samples = 20) {
  using namespace hadrons_root;
  HadronFileReader r(dir);
  r.set_charged_eta(eta_max);                    // charged hadrons at |eta| < eta_max only

  const fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, R);
  const fastjet::JetDefinition kt_def(fastjet::kt_algorithm, R);
  const fastjet::AreaDefinition area_def(fastjet::active_area_explicit_ghosts,
                                         fastjet::GhostedAreaSpec(eta_max, 1, 0.005));
  const fastjet::Selector rho_sel =
      (!fastjet::SelectorNHardest(2)) * fastjet::SelectorAbsRapMax(eta_max - R);
  const fastjet::Selector jet_sel =
      fastjet::SelectorAbsRapMax(eta_max - R) * fastjet::SelectorPtMin(5.0);

  TH1D h_raw("h_raw", ";p_{T,jet} (GeV);dN_{jet}/dp_{T} per event", 60, -20, 100);
  TH1D h_sub("h_sub", ";p_{T,jet} - #rho A (GeV);dN_{jet}/dp_{T} per event", 60, -20, 100);

  const std::vector<long> events = r.select_events();      // hadronized, flagged ones out
  for (long e : events) {
    const int n = std::min(r.n_oversamples(e), max_samples);
    const double w = 1.0 / (double(events.size()) * n);    // per event: mean over its samples
    for (int k = 0; k < n; ++k) {
      std::vector<fastjet::PseudoJet> parts;
      for (const Hadron &h : r.bkg_dep_frag(e, k))           // bulk (bkg + wake) + fragments
        if (h.pt() > 0.15) parts.emplace_back(h.px, h.py, h.pz, h.p());   // massless
      fastjet::ClusterSequenceArea cs(parts, jet_def, area_def);
      fastjet::JetMedianBackgroundEstimator bge(rho_sel, kt_def, area_def);
      bge.set_use_area_4vector(false);                       // rho = median of pT / A
      bge.set_particles(parts);
      const double rho = bge.rho();
      for (const fastjet::PseudoJet &jet : jet_sel(cs.inclusive_jets())) {
        h_raw.Fill(jet.pt(), w);
        h_sub.Fill(jet.pt() - rho * jet.area(), w);
      }
    }
  }
  h_raw.Scale(1.0, "width");
  h_sub.Scale(1.0, "width");
  TFile out("jets_fastjet.root", "RECREATE");
  h_raw.Write();
  h_sub.Write();
}
```

**Running it.** With ACLiC, the include paths and FastJet's libraries are given before the
macro is compiled (conda: FastJet's headers are in `$CONDA_PREFIX/include`):

```bash
conda activate root                         # the env with ROOT and fastjet-cxx
AR=/path/to/js-contrib/contribs/PyJetscape/example/analysis_root
root -l -b -q \
  -e "gSystem->AddIncludePath(\"-I$AR -I$CONDA_PREFIX/include\"); gSystem->Load(\"libfastjet\"); gSystem->Load(\"libfastjettools\")" \
  'jets_fastjet.C+("/path/to/campaign")'
```

As a program (a `main` that calls `jets_fastjet(argv[1])`):

```bash
g++ -O2 -std=c++17 -I$AR jets_main.cxx $(root-config --cflags --libs) \
    $(fastjet-config --cxxflags) $(fastjet-config --libs) -lROOTNTuple -o jets
```

`fastjet-config` takes one query at a time (`--cxxflags` and `--libs` separately). For
fjcontrib, add `-lfastjetcontribfragile` (conda-forge `fastjet-contrib`) and e.g.
`#include <fastjet/contrib/SoftDrop.hh>` or `ConstituentSubtractor.hh`; in a macro,
`gSystem->Load("libfastjetcontribfragile")`.

- **Cross sections:** replace `w` with `info.weight_mb / n` (or `r.sigma_mb(k) / N_k / n`
  for a subset of events, as `read_hadrons.C` does).
- **Embedding:** cluster `r.bkg(e, k)` plus `r.frag(e, k % r.n_frag(e))` for the jet on a
  background it didn't touch; the fragments have `Hadron::origin == kFrag`, which can go into
  the PseudoJets' `user_index`.
- **Measured:** one event, 20 oversamples of ~7,000 hadrons: 21 s with ACLiC compilation on
  the GB10.

---

## 10. Troubleshooting

| symptom | cause and fix |
|---|---|
| `OSError: Can't synchronously read data (can't open directory (…/hdf5/plugin) …)` (older HDF5: `Can't read data …`) | `import hdf5plugin` before reading (the Blosc filter). For the HDF5 command-line tools: `export HDF5_PLUGIN_PATH=$(python -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")` |
| `Assertion 'InternalHelperPythia.particleData.isParticle(id)' failed` on `import jetscape` | in `js_fno` without `conda activate` (no `PYTHIA8DATA`). In the venv this can't happen |
| `arr_bg` of an event reads all zeros | read from `arr_bg_store`, or only the `arr_bg` view was copied to another file: read `arr_bg`, copy files whole |
| `HadronFileReader`: `… was made from particlize file uuid …: not the same run` | a renamed or mixed file; O+O in-job hadrons (no uuid): read them with `Hadrons` / `HadronFile` (§5) |
| warnings about mixed precision, η cuts, charged-only or positions | files hadronized with different options; results are right only for what all files hold |
| `ValueError: cannot convert <RecordArray …> into np.ndarray` (uproot) | `events` read with `library="np"`; read it as awkward |
| `UnboundLocalError: … 'low'` (uproot) | a TTree with truncated floats written by ROOT ≥ 6.38: write those with ROOT < 6.38, or without `--bits-*` |
| ROOT can't open an RNTuple | ROOT < 6.34: convert with `--format ttree` |
| `'assert.h' file not found` (ACLiC) | `conda activate` the ROOT env before compiling |
| `fastjet/ClusterSequenceArea.hh: No such file` (ACLiC) | `gSystem->AddIncludePath("-I$CONDA_PREFIX/include")`, or install `fastjet-cxx` |
| `Error: You can only make one query at a time` | `fastjet-config --cxxflags --libs`: call it twice |
| ROOT's C++ notebook kernel dies at start | `pip install metakernel` in the env; deactivate other venvs first |
| no freeze-out contours, `wake_*.py` ask for `--eos` | `MUSIC_EOS_TABLE` unset: the venv sets it, else point it at `EOS/hotQCD/hrg_hotqcd_eos_binary.dat` of an X-SCAPE build |
| files with different `arr` shapes | two grids (64 × 64 × 32 and 65 × 65 × 33): keep them apart |
| a campaign's events differ systematically from another's at the 1e-3 level | late-step vs constant-step files, or different images: check `<late_delta_tau>` (§1) |
