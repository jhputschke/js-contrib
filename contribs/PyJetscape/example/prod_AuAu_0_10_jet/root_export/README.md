# Hadrons for ROOT: format study and converter prototype

`hadronize.py` writes the hadrons as HDF5 (`jetscape.hadrons_h5`), the natural format next to
the hydro files used for ML training. Most analyses in the field are written in ROOT. This
directory holds a converter prototype, `hadrons_to_root.py`, that turns a hadron file into a
ROOT file. It also holds the benchmark, `bench_formats.py`, that compared the candidate ROOT
layouts on a production file.

| file | what |
|---|---|
| [`../run_h5toROOT.py`](../run_h5toROOT.py) | **a whole campaign**: every production file → one ROOT file with all its hadrons and event information, plus a campaign file with the cross sections (below) |
| `hadrons_to_root.py` | one `<stem>_hadrons_<tag>.h5` → a TTree or RNTuple, one entry per oversample; optional `--eta-max`, `--charged`, `--no-x`, truncated floats |
| `root_writers.h` | its C++ writers (TTree, RNTuple), compiled by PyROOT at the first call |
| `bench_formats.py` | converts one file with every writer and measures size, write and read time |
| `read_bench.C` | the C++ side of the read timing (TTree loop, RNTupleReader, RDataFrame) |

Tests: `tests/test_hadrons_to_root.py`. The PyROOT writers are skipped without ROOT.

## Layout

One tree (TTree) or ntuple (RNTuple) per file, named after the tag (`bulk_jet`, `bulk_bg`,
`jet_frag`). **One entry is one oversample**, which is a complete sample of the event:

| field | type | |
|---|---|---|
| `event` | int | the event: the first event using the unit (`units/event`) |
| `unit` | int | the unit (`units/unit`): the event for `bulk_jet` and `jet_frag`, the background for `bulk_bg` |
| `sample` | int | the oversample k within the unit, 0 … N−1 |
| `bg_unit` | int | the event's background unit, from the particlize file next to the hadron file (−1 if it is not there; `bulk_bg`: the unit itself) |
| `n` | int | hadrons in this sample (TTree counter) |
| `pid`, `pstat` | int[n] | |
| `E`, `px`, `py`, `pz` | float[n] | GeV |
| `t`, `x`, `y`, `z` | float[n] | fm/c, fm; left out with `--no-x` |

Why this layout:
- **One entry per event doesn't work.** At 400 oversamples an entry would hold ~2.8 M
  hadrons.
- **One entry per hadron loses the event.** Every analysis would first have to rebuild it.
- **One entry per oversample** reads like an MC event.

**Oversamples are not independent events.** They share one fluid, and in a `--reuse` or
`--pthat-bins` campaign the events of a background share it too. Treating every entry as an
independent event underestimates the errors. Take errors per `event` (independent sampling)
or per (`bg_unit`, `sample`) pair (`hadronize.py --correlated`), as
`HadronFileReader.jet_minus_background` does.

**Pairing jet and background.** For `--correlated` files, sample k of an event belongs to
sample k of its background:

```cpp
bg->BuildIndex("unit", "sample");
jet->GetEntry(i);
bg->GetEntryWithIndex(jet_bg_unit, jet_sample);   // the paired background sample
```

## A whole campaign: `run_h5toROOT.py`

```bash
./run_h5toROOT.py out_had -j 4                       # RNTuple, next to the inputs
./run_h5toROOT.py out_had -j 4 --out-dir out_root --no-x --eta-max 1 --charged
./run_h5toROOT.py out_had -j 4 --format ttree        # for ROOT < 6.34
./run_h5toROOT.py out_had --dry-run                  # the plan only
```

Inputs are directories, particlize or hadron files, or globs, as for `run_hadronize.py`. A
production file is converted once its particlize file and the hadron files of every `--tags`
tag are complete. Outputs go to `.part` and are renamed when complete. Existing ones are
kept (with a warning if they were made with other settings) unless `--force`, so re-running
the same command finishes an interrupted pass. `-j` converts in parallel, at ~80 B per
hadron of the largest tag, ~3.5 GB for a 15-event file at 400 oversamples. The options of
`hadrons_to_root.py` (`--format`, `--writer`, `--no-x`, `--eta-max`, `--charged`,
`--bits-*`, `--compression`) apply to every file.

**Per production file, `<stem>_hadrons.root`:**

| object | one entry per | content |
|---|---|---|
| `bulk_jet`, `bulk_bg`, `jet_frag` | sample | the hadron ntuples (layout above) |
| `events` | event | every column of the particlize file's `events/`: `pthat_bin`, `pthat`, `event_weight`, `bg_unit`, `bg_id`, the leading and subleading parton pT and y, `n_showers`, `n_partons`, droplets and their energy, MUSIC frames, surface cells, boundary flags; `bg_key_hi`/`bg_key_lo` (the background's hash, two uint64). Then `sigma_file_mb`, this file's cross section of the event's window; `n_samples_jet/bg/frag` and `seed_jet/bg/frag` (its units' samples and seeds); and the shower initiators `ini_shower, ini_pid, ini_pstat, ini_px, ini_py, ini_pz, ini_E, ini_x, ini_y, ini_z, ini_t` |
| `windows` | pT̂ window | `pthat_lo`, `pthat_hi`, this file's `sigma_mb`, `sigma_err_mb`, `acceptance`, `n_tried`, `n_kept`, `n_accepted`, `seed` |
| `provenance` | — | TObjString, JSON: every attribute of the particlize file and the three hadron files. That covers the production settings, seeds, XML, `music_input`, the pT̂ bookkeeping, and the hadronization settings, precision, `eta_max` and sampling (`correlated_sampling`). It also records the converter's settings and how the samples pair up |

**`<campaign>_campaign.root`** (`--campaign-file`; `none` skips it) covers all converted
files:

| object | one entry per | content |
|---|---|---|
| `windows` | pT̂ window | `sigma_mb`, `sigma_err_mb` (`HadronFileReader.pthat_bin_sigma`: the files' estimates weighted by their accepted events), `acceptance`, `acceptance_err`, `n_events` (events with jet-leg hadrons), `n_events_produced`, and **`weight_mb = sigma_mb / n_events`** |
| `files` | production file | `file_index`, `n_events`, `event_offset`, `prod_seed`, `n_backgrounds`, in `HadronFileReader`'s order |
| `provenance` | — | JSON: stems, ROOT files and uuids by `file_index`; the campaign's `eta_max`, precision and sampling; the settings |

**Cross-section weights.** Within one window every event weighs the same. To combine
windows, weigh each event of window k with `weight_mb[k]` from the campaign file. Its
oversamples share that weight: divide by `n_samples_jet` when filling per sample. Per-file
`sigma_file_mb` is only that file's estimate. Use the campaign value for results. The
global event index of `HadronFileReader` is `event_offset[file_index] + event`.

```cpp
// weight per window from the campaign file, then a weighted jet-leg pT spectrum
ROOT::RDataFrame w("windows", "pth10-40_campaign.root");
auto wt = *w.Take<double>("weight_mb");
ROOT::RDataFrame ev("events", "RUN_0001_hadrons.root");
auto nsamp = *ev.Take<int>("n_samples_jet");       // by event
auto bin   = *ev.Take<int>("pthat_bin");
ROOT::RDataFrame jet("bulk_jet", "RUN_0001_hadrons.root");
auto h = jet.Define("w", [&](int e) { return wt[bin[e]] / nsamp[e]; }, {"event"})
            .Define("pt", "sqrt(px*px + py*py)")
            .Histo1D({"pt", "", 100, 0, 5}, "pt", "w");
```

Measured on `pth10-40_eta06_gridnorm` (20 files, 23 GB of HDF5, 400 oversamples, 12/8
bits, GB10, 2026-09-29), with the defaults (RNTuple, ROOT writer, positions kept):
- **Time and size.** 2 min 40 s with `-j 4`, 21–33 s per file. The result is 20.6 GB of
  ROOT files (0.90 of the HDF5), 0.76–1.1 GB per file.
- **Memory.** Peak 13.5 GB, for the one file with 176 M background hadrons (below); a
  typical file needs ~4 GB.
- **Checked against the HDF5 files.** For every file and every tag: entries = samples.
  `bulk_jet` and `bulk_bg` give the same charged-hadron count and ΣpT at \|η\| < 1 as the
  HDF5. `events` matches the particlize file, with initiators for every event and the
  provenance uuid. The campaign's `sigma_mb` and `acceptance` equal `HadronFileReader`'s,
  and `event_offset` equals its global index.
- **The weighting example above runs** on these files.
- **A leg that never froze out shows up here too.** File 0002's background ran to MUSIC's
  300-frame limit (`events.ntau_bg` = 300, against ~100 elsewhere). Its surface has 3.5 M
  cells, and iSS samples ~88 000 hadrons per sample from it instead of ~7000. The
  converter copies what is there. `events.ntau_bg`, `ntau_jet` and the `*_hit_boundary`
  flags let an analysis find and drop such events, as `wake_observables.ipynb` does.

## Converter usage

```bash
conda activate js_fno                                  # ROOT 6.36, uproot 5.7
python hadrons_to_root.py RUN_hadrons_bulk_jet.h5 -o RUN_bulk_jet.root           # RNTuple
python hadrons_to_root.py RUN_hadrons_bulk_jet.h5 -o RUN_bulk_jet.root --format ttree
                                                       # for ROOT < 6.34
python hadrons_to_root.py RUN_hadrons_bulk_jet.h5 -o small.root \
       --no-x --eta-max 1 --charged                    # a small analysis export
python hadrons_to_root.py RUN_hadrons_bulk_jet.h5 -o RUN_bulk_jet.root --writer uproot
                                                       # without ROOT
python bench_formats.py RUN_hadrons_bulk_jet.h5 --out-dir /scratch/root_bench     # the study
```

**Who needs ROOT.**

| | ROOT needed? |
|---|---|
| writing with `--writer root` (the smallest, fastest files; truncated floats) | yes, PyROOT |
| writing with `--writer uproot` (TTree or RNTuple, float32) | no: uproot + awkward |
| reading in Python (uproot) | no: uproot 5.7 reads every variant here, except a TTree with truncated floats written with ROOT ≥ 6.38 (below) |
| reading in C++ / RDataFrame | yes (≥ 6.34 for RNTuple) |

`--writer auto`, the default, uses ROOT when PyROOT imports, and uproot otherwise, with a
note. The files are readable by the same tools either way. uproot's RNTuple is 16–24%
larger and reads ~1.2× slower in ROOT (measured below): uproot stores the floats as plain
`Real32` columns, while ROOT byte-splits them (`SplitReal32`) before compressing. That is
the same trick as Blosc's shuffle, and it is worth 25% on `px` alone (85 against 107 MB).
ROOT with PyROOT needs no compilation: `conda install -c conda-forge root`.

**ROOT 6.38 and truncated-float TTrees.** ROOT 6.38 titles a truncated-float leaf with the
whole leaf list (`x[n]/f[0,0,12]`, against `f[0,0,12]` up to 6.36). uproot (checked: 5.7.3
and 5.7.6) then stops with `UnboundLocalError: ... 'low'`. ROOT reads these files as
before, and RNTuple files with truncated floats are not affected. For uproot readers,
write such TTrees with ROOT < 6.38 or without `--bits-p`/`--bits-x`; the `js_fno` install
scripts cap ROOT below 6.38 for this reason.

Reading in ROOT:

```cpp
ROOT::RDataFrame df("bulk_jet", "RUN_bulk_jet.root");  // TTree or RNTuple alike
auto h = df.Define("pt", "sqrt(px*px + py*py)").Histo1D({"pt", "", 100, 0, 5}, "pt");
```

## Which format: measured

`bench_formats.py` on a production file (GB10, 2026-09-29):
`pth10-40_eta06_gridnorm`, file 0001, `bulk_jet`. It holds 15 events × 400 oversamples =
6000 entries and 41.9 M hadrons, hadronized with `--keep-bits-p 12 --keep-bits-x 8`, so the
input is already rounded. All ROOT files use ZSTD level 5 (`505`).

Each read computes the number and ΣpT of charged hadrons at \|η\| < 1 over all entries,
reading only `pid`, `px`, `py`, `pz`, on one thread. Every variant gives the HDF5 result
exactly: 6 656 339 hadrons, ΣpT = 3.798194 × 10⁶ GeV. The files were read right after
writing, from the page cache, so the read times measure decompression and
deserialization, not the disk.

| variant | positions | size [MB] | vs HDF5 | bytes / hadron | write [s] | C++ loop [s] | RDataFrame [s] | uproot [s] |
|---|---|---|---|---|---|---|---|---|
| HDF5 (input, Blosc-zstd) | yes | 662 | 1.00 | 15.8 | — | — | — | h5py 2.5 |
| TTree, uproot | yes | 707 | 1.07 | 16.9 | 14.3 | 1.57 | 1.75 | 2.11 |
| TTree, ROOT, float | yes | 680 | 1.03 | 16.2 | 26.7 | 1.59 | 1.83 | 2.28 |
| TTree, ROOT, Float16_t 12/8 | yes | 676 | 1.02 | 16.1 | 16.5 | 1.71 | 1.87 | 1.98 |
| **RNTuple, ROOT, float** | yes | **592** | **0.89** | 14.1 | 14.1 | **0.88** | **1.09** | 1.80 |
| RNTuple, uproot, float | yes | 687 | 1.04 | 16.4 | 13.3 | 1.07 | 1.27 | 1.99 |
| RNTuple, Real32Trunc 12/8 | yes | 759 | 1.15 | 18.1 | 5.2 | 1.22 | 1.45 | 3.33 |
| TTree, uproot | no | 447 | 0.67 | 10.7 | 7.3 | 1.60 | 1.75 | 2.08 |
| TTree, ROOT, float | no | 427 | 0.64 | 10.2 | 15.8 | 1.61 | 1.84 | 2.13 |
| TTree, ROOT, Float16_t 12 | no | 423 | 0.64 | 10.1 | 6.0 | 1.69 | 1.86 | 1.83 |
| **RNTuple, ROOT, float** | no | **350** | **0.53** | 8.3 | 5.4 | **0.89** | **1.12** | 1.69 |
| RNTuple, uproot, float | no | 433 | 0.65 | 10.3 | 7.1 | 1.07 | 1.29 | 2.01 |
| RNTuple, Real32Trunc 12 | no | 440 | 0.66 | 10.5 | 2.7 | 1.20 | 1.41 | 3.26 |

Write times include the Python side (h5 columns → ROOT). The uproot writes also include
building the awkward arrays.

What this shows:
- **RNTuple with plain floats is the best format**: 11% smaller than the HDF5 file and ~15%
  smaller than any TTree, with the fastest reads (0.9 s against 1.6 s for the TTree loop,
  1.1 s against 1.8 s in RDataFrame). It stores each column in pages and byte-splits the
  floats, as Blosc's shuffle does. The mantissa bits that `--keep-bits` zeroed then compress
  away.
- **Without ROOT, uproot's RNTuple is the best choice.** It is 16–24% larger than ROOT's
  (plain `Real32` columns, see *Who needs ROOT*), about TTree size. In ROOT it still reads
  faster than any TTree: 1.07 s in the C++ loop, against 1.6 s. It is written in 500-sample
  clusters. One cluster for the whole file gave the same size but 1.4 s reads.
- **TTree is about as large as the HDF5 file.** Its baskets are compressed without byte
  splitting, so the zeroed bits help less. Writer and precision barely matter: `Float16_t`
  saves < 1% on data that is already rounded.
- **Truncated floats don't help on rounded input.** `Real32Trunc` bit-packs 21 (17) bits per
  value. That destroys the byte alignment the compressor exploits, so the file is 28%
  larger than plain floats. They would matter only for full-precision input, which this
  study doesn't cover. There, rounding in the converter and storing floats is likely still
  the better choice.
- **The positions are 40% of the bytes** in every format. Leave them out unless an analysis
  needs them (`--no-x`).
- **All variants are readable everywhere we tried**: C++ TTree/RNTuple loops, RDataFrame,
  and uproot 5.7 (including RNTuple and the truncated types).

For c1, 67 files, the HDF5 estimate is ~77 GB at 400 oversamples. The same ratios would give
~69 GB as RNTuple with positions and ~41 GB without. `--eta-max`, `--charged` and fewer
oversamples multiply on top of that.

**Recommendation.** Use RNTuple, plain float32, from rounded (`--keep-bits`) hadron files,
without positions by default. These are the converter's defaults. Write with ROOT where
PyROOT is available; without it, `--writer uproot` writes an RNTuple that is 16–24% larger
and just as readable (the `auto` fallback).

The one limit is compatibility: RNTuple needs **ROOT ≥ 6.34**, the first release with the
stable on-disk format (the version here is 6.36). For users stuck on older ROOT, a TTree
(`--format ttree`) costs ~20% more disk and ~1.8× the read time, and nothing else.

Not measured yet:
- cold-cache (disk) reads;
- multithreaded RDataFrame;
- other compression settings (LZMA, ZSTD levels);
- `bulk_bg` and `jet_frag` (same structure, so the same ratios are expected);
- full-precision input.
