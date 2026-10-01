# analysis_root — the hadrons of a campaign in ROOT

C++ analysis of the ROOT files that
[`../prod_AuAu_0_10_jet/run_h5toROOT.py`](../prod_AuAu_0_10_jet/run_h5toROOT.py) writes from a
hadronized campaign (layout in
[`../prod_AuAu_0_10_jet/root_export/README.md`](../prod_AuAu_0_10_jet/root_export/README.md)).
The Python analyses of the HDF5 files are in [`../analysis`](../analysis/README.md).

| file | what it is for | writes? |
|---|---|---|
| [`HadronFileReader.h`](HadronFileReader.h) | header-only reader: the hadrons of an event's oversample as `std::vector<Hadron>`, for background, background + deposition, the whole jet event and the fragments (§1) | no |
| [`read_hadrons.C`](read_hadrons.C) | the reader by example: pT spectra and Δφ to the leading parton, per source and for the wake (§2) | a new `.root` |
| [`hadron_distributions.C`](hadron_distributions.C) | RDataFrame macro: η, φ, pT of background, background + deposition, jet fragments and the wake (§3) | a new `.root` + `.pdf`/`.png` |

Everything here needs ROOT itself, **≥ 6.34** for the RNTuple files that `run_h5toROOT.py`
writes by default (TTree files, `--format ttree`, read with any recent ROOT). Set up as in
[`utils/analysis_env`, *With ROOT*](../../../../utils/analysis_env/README.md#with-root). With
the conda ROOT, `conda activate` before compiling a macro: without it ACLiC doesn't find the
system headers (`'assert.h' file not found`).

## 1. `HadronFileReader.h`

The C++ counterpart of `jetscape.hadrons_h5.HadronFileReader`, for the ROOT files. One
entry of a hadron ntuple is one oversample; the reader finds the entries of an event in all
three ntuples and returns their hadrons:

| call | ntuple | the hadrons of |
|---|---|---|
| `r.bkg(e, k)` | `bulk_bg` | the background (MUSIC_1): sample k of the background event e used |
| `r.bkg_dep(e, k)` | `bulk_jet` | the jet leg (MUSIC_2): background + deposition, sample k |
| `r.bkg_dep_frag(e, k)` | `bulk_jet` + `jet_frag` | the whole jet event: `bkg_dep(e, k)` plus fragmentation k mod `n_frag(e)` |
| `r.frag(e, j)` | `jet_frag` | fragmentation j of the surviving partons |

`bkg_dep_frag(e, k, j)` takes fragmentation j instead. The fragments have
`Hadron::origin == kFrag` and the bulk hadrons `kBulk`, as `origin` in `JetEvents.jet_event`.
`jet_event` and `background_event` are the same calls under the Python reader's names.

```cpp
#include "HadronFileReader.h"
using namespace hadrons_root;

HadronFileReader r("out_root");            // every *_hadrons.root there, + its *_campaign.root
r.set_charged_eta(1.0);                    // optional: charged hadrons at |eta| < 1 only
for (long e : r.select_events()) {         // hadronized events, flagged ones left out
  const EventInfo &info = r.info(e);       // pthat_bin, bg_unit, initiators, weight_mb ...
  for (int k = 0; k < r.n_oversamples(e); ++k) {
    Hadrons bkg  = r.bkg(e, k);
    Hadrons dep  = r.bkg_dep(e, k);
    Hadrons full = r.bkg_dep_frag(e, k);
    for (const Hadron &h : full) { /* h.pt(), h.eta(), h.phi(), h.p4(), h.origin ... */ }
  }
}
```

**The reader.**
- `HadronFileReader(source, positions = false, campaign = "")`. `source` is a directory or
  one `*_hadrons.root` file. `campaign` is the campaign file for the cross sections. By
  default the reader takes the `*_campaign.root` next to the files, if there is one;
  `"none"` skips it. A `std::vector<std::string>` of files works too. The campaign file is
  optional (*Cross sections*, below).
- `positions`: read t, x, y, z as well. That costs time, and files written with `--no-x`
  have none; without them they are 0.
- Events are numbered globally over the files, sorted by name: `event_offset(file) +
  events.event`. This is `HadronFileReader`'s numbering for the same production files.
- `n_events()`, `n_files()`, `file(i)`; `n_oversamples(e)`, `n_bg_samples(e)`, `n_frag(e)`.
- `select_events(window = -1, keep_flagged = false)`: the hadronized events
  (`n_oversamples > 0`), of one pT̂ window or all. Events whose legs are not alike are left
  out (`EventInfo::flagged`, below).
- `n_windows()`, `pthat_lo(k)`, `pthat_hi(k)`, `sigma_mb(k)`, `sigma_err_mb(k)`,
  `n_window_events(k)`, `weight_mb(k)`, `sigma_source()`: the pT̂ windows and their cross
  sections (*Cross sections*, below).
- `set_filter(f)` keeps only the hadrons `f(const Hadron&)` accepts, while reading, so the
  vectors stay small. `set_charged_eta(eta_max)` is the common case (`eta_max <= 0`: no η
  cut). `clear_filter()` turns it off.
- Bad input throws: a missing file or ntuple, an event out of range, a sample that isn't
  there (`std::runtime_error`, `std::out_of_range`).

**`Hadron`**: `pid`, `pstat`, `E`, `px`, `py`, `pz` (GeV), `t`, `x`, `y`, `z` (fm), `origin`;
`pt()`, `p()`, `eta()`, `rapidity()`, `phi()`, `mass()`, `charged()` and `p4()`, a
`ROOT::Math::PxPyPzEVector`. η is `jetscape.hadrons_h5`'s definition (±690 along the
beam), `charged()` its `CHARGED` list. For FastJet: `fastjet::PseudoJet(h.px, h.py, h.pz,
h.E)`.

**`EventInfo`** (`r.info(e)`) holds the event's row of the `events` table, plus the
event's weight:
- `event`, `file_index`, `local_event`, `bg_unit`, `pthat_bin`, `pthat`;
- `n_samples_jet/bg/frag` and `n_cells_jet/bg`;
- `sigma_file_mb` (this file's estimate) and `weight_mb` (`weight_mb(pthat_bin)`);
- `initiators`, the shower-initiating partons (`Initiator`: `shower, pid, pstat, px, py,
  pz, E, x, y, z, t`, with `pt()`, `eta()`, `phi()`), and `leading()`, the one with the
  largest pT;
- `flagged()`: the background has more than 20% more or fewer freeze-out cells than the jet
  leg, so one surface is not a freeze-out surface (as in `wake_hadrons.py`).

**Cross sections** (`--pthat-bins` campaigns). The cross section of window k is the
combination of all production files' Pythia estimates, weighted by their accepted events:
σ = Σ nᵢσᵢ / Σ nᵢ. This is `HadronFileReader.pthat_bin_sigma`.
- **Where σ comes from.** With a `*_campaign.root`, `sigma_mb(k)` is read from it.
  Without one, the reader computes the same combination from the `windows` tables of the
  files it reads (`combine_windows`), with one line of output saying so. `sigma_source()`
  tells which one was used.
- **Without either.** If a file has no cross sections (a job that didn't finish) or other
  windows than the rest, `n_windows()` is 0 and a line says why. A campaign file whose
  windows differ from the files' throws: it belongs to another campaign.
- **Weights.** `weight_mb(k)` is σ_k over `n_window_events(k)`, the hadronized events of
  window k in the files read, flagged ones included. For all files of a campaign this is
  the campaign file's `weight_mb`. For a subset it is right for that subset; σ still comes
  from the campaign file when there is one, and from that subset when there isn't.
- **Flagged events.** To leave them out without changing the cross section, use
  `sigma_mb(k) / N_k` with N_k the selected events of window k, as `read_hadrons.C` does.
- **The building blocks** are free functions. `read_windows(path)` reads one file's
  `windows` table into a `Windows` struct, and `combine_windows(files)` combines several,
  as above. `hadron_distributions.C` uses them as well.

**Normalization and errors.** Oversamples share one fluid: they are samplings of one
event, not independent events. Average each event over its own samples: weigh a hadron of
`bkg` with w_e / `n_bg_samples(e)`, of `bkg_dep` and `bkg_dep_frag` with w_e /
`n_oversamples(e)`, of `frag` with w_e / `n_frag(e)`. Use w_e = 1/N per event, or
σ_k / N_k in mb. A background shared by several events then counts once for each of them.
Take errors per event, not per sample. For `--correlated` files, sample k of an event and
sample k of its background are a pair, so `bkg_dep(e, k)` − `bkg(e, k)` cancels most of the
background's fluctuations. More in the
[ROOT export README](../prod_AuAu_0_10_jet/root_export/README.md), *Oversamples are not
independent events*.

**Compiling.**

```bash
cd example/analysis_root
root -l -b -q 'read_hadrons.C+("DIR")'                 # ACLiC, a macro next to the header
root -l -b -e 'gSystem->AddIncludePath("-I/path/to/analysis_root")' -q 'my_macro.C+'
g++ -O2 -std=c++17 -I/path/to/analysis_root my.cxx $(root-config --cflags --libs) \
    -lROOTNTuple -lROOTDataFrame -o my                  # a program
```

In PyROOT, `ROOT.gInterpreter.Declare('#include "/path/HadronFileReader.h"')` and then
`ROOT.hadrons_root.HadronFileReader(...)`. A filter has to be set from C++: cppyy can't
pass a Python callable as the `Filter`.

**Speed.** On `prod_AuAu_0_10_jet/out` (1 event, 500 oversamples, RNTuple), the whole jet
event of every oversample takes 0.24 s, compiled with g++: 3.5 M hadrons read, 0.58 M of
them charged at \|η\| < 1. Every sample is read from disk once per call. Keep the vector if it is
needed twice.

**Checked.** `tests/test_analysis_root_reader.py` covers RNTuple and TTree, the uproot and
ROOT writers, truncated floats (`--bits-p 12`) and `--no-x`. On every event, oversample and
source it gets the same hadrons as `HadronFileReader` on the HDF5 files (pid, pstat, E, p,
x, origin). It also checks the event table, the initiators, a background shared by two
events and the cross sections. With the campaign file, and with the campaign file deleted,
`sigma_mb`, `sigma_err_mb` and `weight_mb` equal the campaign file's and
`pthat_bin_sigma`; for one file they are that file's own; another campaign's file is
refused. `hadron_distributions.C` and `read_hadrons.C` give the same `xsec` histograms with
the campaign file and without it. On `prod_AuAu_0_10_jet/out`, `read_hadrons.C` and
`hadron_distributions.C` fill the same pT histograms, bin for bin.

## 2. `read_hadrons.C`: the reader by example

```bash
root -l -b -q 'read_hadrons.C+("DIR")'                    # per event, all windows
root -l -b -q 'read_hadrons.C+("DIR", 2)'                 # pT-hat window 2 only
root -l -b -q 'read_hadrons.C+("DIR", -1, true)'          # cross-section weighted [mb]
root -l -b -q 'read_hadrons.C+("DIR", -1, false, 1.0, "out.root", 20)'
         # |eta| < 1, output file, only the first 20 oversamples of each event
```

Arguments: `dir, window = -1, xsec = false, eta_cut = 1.0, out = "read_hadrons.root",
max_samples = -1`. For charged hadrons at \|η\| < `eta_cut` it fills, per source (`bkg`,
`bkgdep`, `full`, `frag`) and for `wake` = `bkgdep` − `bkg`:
- `h_pt_<source>`: the pT spectrum, 60 log bins from 0.1 to 100 GeV;
- `h_dphi_<source>`: φ relative to the event's leading initiator, in [−π/2, 3π/2).

Both are divided by the bin width. They are per event, or dσ/dX in mb with `xsec`. The
pT spectra equal `hadron_distributions.C`'s `h_ptlog_*`, bin for bin. `full` differs only in
its errors: here each fragmentation is filled once per jet-leg sample it is paired with,
whereas `hadron_distributions.C` adds `bkgdep` and `frag` in quadrature.

## 3. Hadron distributions in RDataFrame: `hadron_distributions.C`

`hadron_distributions.C` histograms η, φ and pT for each source of a jet event. It reads
the ntuples with RDataFrame rather than through the reader, which makes it fast and
multithreaded:

| name | ROOT ntuple | what |
|---|---|---|
| `bkg` | `bulk_bg` | iSS on the background's surface (MUSIC_1) |
| `bkgdep` | `bulk_jet` | iSS on the jet leg's surface (MUSIC_2): background + deposition |
| `frag` | `jet_frag` | ColorlessHadronization of the surviving partons |
| `wake` | | `bkgdep` − `bkg` |
| `full` | | `bkgdep` + `frag`, the whole jet event |

```bash
cd example/analysis_root
root -l -b -q 'hadron_distributions.C+("DIR")'                 # per event, all windows
root -l -b -q 'hadron_distributions.C+("DIR", 2)'              # pT-hat window 2 only
root -l -b -q 'hadron_distributions.C+("DIR", -1, true)'       # cross-section weighted [mb]
root -l -b -q 'hadron_distributions.C+("DIR", -1, false, false, 1.0, "out.root", 8)'
         # all hadrons (not only charged), |eta| < 1 for pT and phi, output file, 8 threads
```

Arguments: `dir, window = -1, xsec = false, charged = true, eta_cut = 1.0, out =
"hadron_distributions.root", threads = 1, keep_flagged = false`. η is filled for all pT;
pT and φ for \|η\| < `eta_cut`. The output holds `h_{eta,phi,pt,ptlog}_{bkg,bkgdep,frag,wake,full}`,
divided by the bin width, plus one page of plots (`.pdf`, `.png`). The rows of that page are
background vs background + deposition, fragments, and wake.

**Normalization is `HadronFileReader`'s.**
- **Per event.** Each event is the mean over its oversamples, so a hadron of event e
  weighs w_e / n_samples. A background shared by several events counts once for each of
  them.
- **Per event** (default): w_e = 1/N, so the histograms are dN/dX per event.
- **`xsec`**: w_e = σ_k / N_k, the window's cross section over its selected events. σ_k
  comes from the campaign file; without one, it is combined from the files' `windows`
  tables, as the reader does it, so the sum over windows is dσ/dX in mb. N_k counts only the events used,
  so dropping flagged events does not change the cross section.
- **Errors** are the compound-Poisson errors of independent sampling. For `--correlated`
  hadron files the wake's paired error is much smaller: use
  `HadronFileReader.jet_minus_background` for it.

**Bad runs** are dropped as in `wake_hadrons.py`: an event whose background has more than
20% more or fewer freeze-out cells than its jet leg (`events.n_cells_bg / n_cells_jet`) is
flagged and left out, unless `keep_flagged`. In `AuAu_0_10_pth10-40_eta06_gridnorm` those
are the 15 events of job 0002.

**Checked** on `AuAu_0_10_pth10-40_eta06_gridnorm`, 285 events, charged hadrons at
\|η\| < 1: every pT bin of `bkg`, `bkgdep`, `frag` and `wake`, and its error, equals
`HadronFileReader.hist` / `jet_minus_background` on the HDF5 files to 10⁻¹². Per event:
1308 (bkg), 1317 (bkg + deposition), 7.0 (fragments), 9.0 (wake) charged hadrons. The
campaign takes 21 s with 8 threads, against ~6 min for the same histograms with
`HadronFileReader` on the HDF5 files.
