# ROOT and C++

Most analyses in the field are written in ROOT. This page shows how to turn a hadronized
campaign into ROOT files, read them with or without ROOT (uproot, RDataFrame), loop over events
in C++ with the header-only `HadronFileReader.h`, cluster jets with FastJet's C++ library, and run
the C++ map steps of the Z+jet analyses. The ROOT files hold the hadrons and the event
information only; the hydro stays in HDF5 (read it with h5py, also from PyROOT).

## 1. Set up

| need | environment |
|---|---|
| converting (`run_h5toROOT.py`), reading with uproot | the analysis venv; it writes with uproot, or with ROOT when PyROOT imports |
| RDataFrame, the macros, `HadronFileReader.h`, PyROOT | ROOT **≥ 6.34** (RNTuple) next to the venv |
| FastJet in C++ | conda-forge `fastjet-cxx` (FastJet, `fastjet-config`) and `fastjet-contrib` (fjcontrib) in the ROOT env |

conda gives ROOT and its Python, and the venv is made from that Python:

```bash
conda create -n root -c conda-forge "root>=6.34"           # RNTuple needs ROOT >= 6.34
conda activate root
conda install -c conda-forge fastjet-cxx fastjet-contrib   # only for FastJet in C++
./utils/analysis_env/setup_analysis_env.sh ~/.venvs/js_analysis_root --system-site-packages
```

Then `source ~/.venvs/js_analysis_root/bin/activate` has PyROOT. For the `root` prompt and
compiled macros, `conda activate root` first: without it ACLiC does not find the system headers
(`'assert.h' file not found`). Details: [Analysis environment](../install/analysis-env.md).

## 2. Convert a campaign: `run_h5toROOT.py`

[`run_h5toROOT.py`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/prod_AuAu_0_10_jet/run_h5toROOT.py)
turns every hadronized production file into one ROOT file, plus one campaign file:

```bash
cd contribs/PyJetscape/example/prod_AuAu_0_10_jet
python run_h5toROOT.py /path/to/campaign -j 4                        # RNTuple, next to the inputs
python run_h5toROOT.py /path/to/campaign -j 4 --format ttree         # TTree, for ROOT < 6.34
python run_h5toROOT.py /path/to/campaign -j 4 --out-dir root --no-x --charged --eta-max 1
python run_h5toROOT.py /path/to/campaign --dry-run                   # the plan only
python run_h5toROOT.py /data/c00/gpu0 /data/c00/gpu1 -j 8 --out-dir /data/root/c00   # several directories into one
```

| option | |
|---|---|
| `-j N` | files in parallel; ~80 B per hadron of the largest tag each (~3.5 GB for a 15-event file at 400 oversamples) |
| `--format {rntuple,ttree}` | RNTuple (default): 0.9 × the HDF5 size, ~1.8 times faster to read than a TTree; needs ROOT ≥ 6.34 to read |
| `--writer {auto,root,uproot}` | `auto`: ROOT if PyROOT imports (16–24 % smaller files), else uproot |
| `--out-dir DIR` | where the ROOT files go (default: next to the inputs) |
| `--no-x`, `--charged`, `--eta-max X` | leave out positions (40 % of the bytes), neutral hadrons, \(|\eta| \ge X\) |
| `--tags` | a subset of `bulk_jet,bulk_bg,jet_frag` |
| `--campaign-file NAME` / `none` | the campaign file's name, or none |
| `--force` | redo files already converted |

A file is converted once its particlize file and every tag's hadron file are complete (hadrons-only
jobs: their hadron files). Outputs are written to `.part` and renamed, so rerunning finishes an
interrupted pass and skips what is done. The 20-file test campaign took 2 min 40 s at `-j 4`.

**Check a conversion** against the HDF5 files, every hadron bit for bit, the event record, the
windows and the cross sections (exit 1 on a difference):

```bash
python root_export/check_h5_vs_root.py /path/to/campaign /path/to/root --campaign /path/to/root/c1_campaign.root
```

### What is in the files

**Per production file, `<stem>_hadrons.root`:**

| object | one entry per | content |
|---|---|---|
| `bulk_jet`, `bulk_bg`, `jet_frag` | oversample | `event`, `unit`, `sample`, `bg_unit`, `n`, `pid[n]`, `pstat[n]`, `E/px/py/pz[n]` (GeV), `t/x/y/z[n]` (fm; not with `--no-x`) |
| `events` | event | the particlize file's `events/`, `sigma_file_mb`, `n_samples_*`, `seed_*`, the initiators `ini_*[...]` |
| `windows` | pT̂ window | `pthat_lo/hi`, this file's `sigma_mb`, `sigma_err_mb`, `acceptance`, counts |
| `provenance` | – | JSON: every setting, seed, XML and EoS behind the file |

**`<campaign>_campaign.root`:** `windows` with the campaign's `sigma_mb`, `sigma_err_mb`,
`acceptance`, `n_events` and **`weight_mb = sigma_mb / n_events`**; `files` with `event_offset`
per file (global event = `event_offset[file] + event`, the numbering of the Python
`HadronFileReader`); `provenance`.

**Why one entry per oversample:** one entry per event would hold ~2.8 M hadrons at 400
oversamples, and one entry per hadron loses the event. One entry per oversample reads like an MC
event. Remember that the oversamples of an event are samplings of one fluid, not independent
events ([What a production writes](files.md#what-an-event-is)).

**One file at a time.** `root_export/hadrons_to_root.py <stem>_hadrons_<tag>.h5 -o out.root`
converts a single hadron file (the same `--format`, `--writer`, `--no-x`, `--eta-max`,
`--charged` options); the O+O files are converted that way ([O+O](oo.md#13-root)). The format
study behind the defaults:
[`root_export/README.md`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/prod_AuAu_0_10_jet/root_export/README.md).

## 3. Reading without ROOT: uproot

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

- Read `events` as awkward (the default): `library="np"` fails on the jagged initiator columns.
- `np.isin` does not take jagged arrays; flatten and unflatten for a species mask:

```python
CHARGED = [211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13]        # jetscape.hadrons_h5.CHARGED
b = f["bulk_jet"].arrays(["event", "pid", "px", "py", "pz"])
eta = np.arcsinh(b.pz / np.hypot(b.px, b.py))
charged = ak.unflatten(np.isin(np.abs(ak.flatten(b.pid)), CHARGED), ak.num(b.pid))
n_mid = ak.sum(charged & (abs(eta) < 1), axis=1)  # charged at |eta| < 1, per oversample
```

- In the venv, uproot also reads files in place over `osdf://`, e.g.
  `uproot.open("osdf:///fno4hic/<campaign>/<file>.root")` ([Data on the OSDF](../run/osdf.md)).

## 4. In ROOT

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

TTree and RNTuple read the same way. `ROOT::EnableImplicitMT(n)` runs it on n threads.

### `HadronFileReader.h`

[`analysis_root/HadronFileReader.h`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis_root/HadronFileReader.h)
is the header-only C++ counterpart of the Python reader. It finds an event's entries in the three
ntuples and returns their hadrons as `std::vector<Hadron>`:

| call | the hadrons of |
|---|---|
| `r.bkg(e, k)` | the background (MUSIC_1): sample k of the background event e used |
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

- **Source:** a directory (without its subdirectories) or one `*_hadrons.root` file, or a
  `std::vector<std::string>` of files. For multi-GPU output, convert all subdirectories into one
  directory with `--out-dir`.
- **`EventInfo`:** `pthat_bin`, `pthat`, `bg_unit`, sample and cell counts, `weight_mb`, the
  `initiators` and `leading()`, `n_coll`, the trigger Z (`z`), and `flagged()`: the background
  has more than 20 % more or fewer freeze-out cells than the jet leg, so one surface is not a
  freeze-out surface. `select_events()` leaves those out.
- **Normalisation:** a hadron of `bkg_dep` or `bkg_dep_frag` weighs w_e / `n_oversamples(e)`, of
  `bkg` w_e / `n_bg_samples(e)`, of `frag` w_e / `n_frag(e)`, with w_e = 1/N per event or
  σ_k / N_k for the cross section. Errors per event, not per sample; for `--correlated` files pair
  `bkg_dep(e, k)` with `bkg(e, k)`.
- **Cross sections:** `sigma_mb(k)`, `weight_mb(k)` come from the campaign file, or are combined
  from the files' `windows` tables when there is none.
- **PyROOT:** `ROOT.gInterpreter.Declare('#include "/path/HadronFileReader.h"')`, then
  `ROOT.hadrons_root.HadronFileReader(...)`.
- **Speed:** the whole jet event of every oversample of one event with 500 oversamples (3.5 M
  hadrons) takes 0.24 s compiled. It gives the same hadrons as the Python reader on the HDF5
  files, event by event.

### The macros

```bash
cd contribs/PyJetscape/example/analysis_root
root -l -b -q 'read_hadrons.C+("/path/to/campaign")'                       # per event
root -l -b -q 'read_hadrons.C+("/path/to/campaign", -1, true)'             # dsigma/dX [mb]
root -l -b -q 'hadron_distributions.C+("/path/to/campaign", -1, false, true, 1.0, "out.root", 8)'
```

| macro | what |
|---|---|
| `read_hadrons.C` | the reader by example: pT spectra and Δφ to the leading initiator, per source (`bkg`, `bkgdep`, `full`, `frag`) and for the wake (`bkgdep − bkg`). Arguments: `dir, window = -1, xsec = false, eta_cut = 1.0, out, max_samples = -1` |
| `read_hadrons.ipynb` | the same, one step per cell, in ROOT's C++ notebook kernel (`HADRON_DIR`, `HADRON_OUT`, `MAX_SAMPLES`); needs `pip install metakernel` in the env |
| `hadron_distributions.C` | η, φ, pT of every source and of the wake, with RDataFrame, multithreaded. Arguments: `dir, window = -1, xsec = false, charged = true, eta_cut = 1.0, out, threads = 1, keep_flagged = false` |

`hadron_distributions.C` reproduces `HadronFileReader.hist` / `jet_minus_background` on the HDF5
files bin for bin: a 285-event campaign takes 21 s with 8 threads, against ~6 min in Python. Its
errors are those of independent sampling; for the paired error of correlated files use the
Python `jet_minus_background`. Full reference:
[`analysis_root/README.md`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis_root/README.md).

## 5. FastJet in C++

With FastJet's C++ library next to ROOT, `HadronFileReader.h` hands the hadrons straight to
FastJet. This macro clusters the whole jet event of every oversample with the conventions of
[Jets with FastJet](jets.md#2-conventions) and fills the raw and ρA-subtracted spectrum:

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

The file is in
[`docs/analysis_snippets/`](https://github.com/jhputschke/js-contrib/tree/main/docs/analysis_snippets),
with `jets_main.cxx`, a `main` that calls it.

=== "ACLiC"

    ```bash
    conda activate root                         # the env with ROOT and fastjet-cxx
    AR=/path/to/js-contrib/contribs/PyJetscape/example/analysis_root
    root -l -b -q \
      -e "gSystem->AddIncludePath(\"-I$AR -I$CONDA_PREFIX/include\"); gSystem->Load(\"libfastjet\"); gSystem->Load(\"libfastjettools\")" \
      'jets_fastjet.C+("/path/to/campaign")'
    ```

=== "g++"

    ```bash
    cd docs/analysis_snippets                   # jets_main.cxx includes jets_fastjet.C; AR as in the ACLiC tab
    g++ -O2 -std=c++17 -I$AR jets_main.cxx $(root-config --cflags --libs) \
        $(fastjet-config --cxxflags) $(fastjet-config --libs) -lROOTNTuple -o jets
    ./jets /path/to/campaign
    ```

- `fastjet-config` takes one query at a time: `--cxxflags` and `--libs` separately.
- **fjcontrib:** add `-lfastjetcontribfragile` (conda-forge `fastjet-contrib`) and include e.g.
  `<fastjet/contrib/SoftDrop.hh>` or `ConstituentSubtractor.hh`; in a macro,
  `gSystem->Load("libfastjetcontribfragile")`.
- **Cross sections:** replace `w` with `info.weight_mb / n` (or `r.sigma_mb(k) / N_k / n` for a
  subset of events, as `read_hadrons.C` does).
- **Embedding:** cluster `r.bkg(e, k)` plus `r.frag(e, k % r.n_frag(e))` for the jet on a
  background it did not touch; the fragments have `Hadron::origin == kFrag`, which can go into the
  PseudoJets' `user_index`.
- One event with 20 oversamples of ~7,000 hadrons took 21 s, ACLiC compilation included, on the
  GB10.

## 6. The Z+jet analyses in C++

[`analysis_root/zjet/`](https://github.com/jhputschke/js-contrib/tree/main/contribs/PyJetscape/example/analysis_root/zjet)
holds the C++ map steps of the Z+jet analyses ([Jets with FastJet](jets.md#7-worked-examples-the-zjet-campaign-analyses)).
Each has a Python twin and agrees with it value for value; the final stage is the same Python
for both:

| program | map step | output per production file |
|---|---|---|
| `zqa` | the QA: a table per Z event and per-source hadron histograms | `<stem>_qa.root`, drawn by `qa.py plot` |
| `zhadron_pools` | the Z–hadron pool (CMS 2507.09307) | `<stem>_pool.root`, read by `mix.py --pbpb-pools` |
| `zjets` | every oversample clustered with FastJet, ρA-subtracted, with embedding and random cones (CMS 1702.01060) | `<stem>_jets.root`, read by `zjet_jets.py` |

```bash
# DATA (the campaign), PP (the pp reference) and ANA (the outputs) as on the FastJet page
cd contribs/PyJetscape/example/analysis_root/zjet && make     # root-config and fastjet-config on PATH
P=../../prod_AuAu_0_10_jet
python $P/run_h5toROOT.py $DATA/gpu0 $DATA/gpu1 -j 8 --out-dir $ANA/root/pbpb \
    --campaign-file $ANA/root/pbpb/zjet_c00_campaign.root
python $P/run_h5toROOT.py $PP --out-dir $ANA/root/pp --campaign-file $ANA/root/pp/zjet_pp_campaign.root
./zjets $ANA/root/pbpb $ANA/jets_cxx/pbpb --files 0:13        # in slices, in parallel
./zjets $ANA/root/pp $ANA/jets_cxx/pp
cd ../../analysis/zjet
python zjet_jets.py --pbpb $ANA/jets_cxx/pbpb --pp $ANA/jets_cxx/pp --outdir $ANA/figures --tag _c00_cxx
python check_cxx.py jets $ANA/jets_py/pbpb $ANA/jets_cxx/pbpb | tail -1      # against the Python map step
```

`zjets` took 0.12 s per Pb+Pb oversample on the test set, at ~0.35 GB per process. `make FASTJET_PY=<the
fastjet wheel's package directory>` builds against the pip `fastjet` wheel instead of a system
FastJet, so both routes use the same FastJet. Options and details:
[`analysis_root/zjet/README.md`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis_root/zjet/README.md).

## 7. Troubleshooting

| symptom | cause and fix |
|---|---|
| ROOT can't open an RNTuple | ROOT < 6.34: convert with `--format ttree` |
| `ValueError: cannot convert <RecordArray …> into np.ndarray` (uproot) | `events` read with `library="np"`; read it as awkward |
| `UnboundLocalError: … 'low'` (uproot) | a TTree with truncated floats written by ROOT ≥ 6.38: write those with ROOT < 6.38, or without `--bits-*` |
| `'assert.h' file not found` (ACLiC) | `conda activate` the ROOT env before compiling |
| `fastjet/ClusterSequenceArea.hh: No such file` (ACLiC) | `gSystem->AddIncludePath("-I$CONDA_PREFIX/include")`, or install `fastjet-cxx` |
| `Error: You can only make one query at a time` | `fastjet-config --cxxflags --libs`: call it twice |
| ROOT's C++ notebook kernel dies at start | `pip install metakernel` in the env; deactivate other venvs first |
| `run_h5toROOT.py`: `KeyError: 'seed'` | O+O hadron files made in the job: convert them as in [O+O](oo.md#13-root) |

## Next steps

- The Python side of the same analyses: [Hadrons](hadrons.md), [Jets with FastJet](jets.md).
- Notebooks, including `read_hadrons.ipynb`: [Notebooks](notebooks.md).
