# Z+jet set-up: the 96-event test and the campaign set-up checks

A small end-to-end test of the Z+jet production chain (PythiaGun's Z trigger → 3dMCGlauber
→ MUSIC → Matter/LBT → liquefier → `--hadrons-only` iSS, correlated sampling), compared with
CMS's Z–hadron Δφ and Δy correlations in Pb+Pb 5.02 TeV 0–30 % and pp
([arXiv:2507.09307](https://arxiv.org/abs/2507.09307), PLB 2025, 140120;
HEPData [ins2945894](https://www.hepdata.net/record/ins2945894)). Made 2026-10-15 on the
GB10 with X-SCAPE `zjet` 38cd11cf and js-contrib `zjet` 3f85274 (the Pb+Pb jobs ran on
its then-uncommitted state; the vacuum on 417d991). Nothing in the model is tuned to these
data.

The campaign workflow (QA, map steps, mixing, jets) is in [`../README.md`](../README.md); this
folder keeps how the campaigns were set up. Data and figures live in
`../../runs/zjet_setup/` (`data/` and `figures/` below; `make_data.sh` writes there).

| file | |
|---|---|
| `make_data.sh` | the commands that made `data/` (Pb+Pb, pp, HEPData, movie) |
| `accumulate.py` | hadron files → `data/hists.npz` (~15 s) |
| `centrality_check.py` | minimum-bias 3dMCGlauber with the XML's settings: the model's σ_inel and how well the b cut matches activity classes (→ `data/glauber_mb/summary.txt`, ~8 min) |
| `compare.py` | `data/hists.npz` + HEPData → `figures/` |
| `figures/zjet_vs_cms2507_dphi.png`, `…_dy.png`, `summary.txt`, `RESULTS.md` | the result |
| `figures/MIXING.md` (`.html`) | the event-mixing document: modes and settings, the small-sample checks and comparisons, the plan for the campaigns (git-ignored with the figures) |
| `data/pbpb/` | 2 jobs × 48 Pb+Pb 0–30 % Z+jet events, hadrons only (833 MB) |
| `data/vacuum/` | 6,000 pp (vacuum) Z+jet events (63 MB) |
| `data/hepdata/` | CMS's HEPData record (YAML, 31 tables) |
| `data/movie/` | one Z+jet event with its hydro evolution, its hadrons and two movies (1.7 GB) |
| `data/logs/` | the production and render logs |


**The campaign set-up checks** (2026-10-16; `make_data.sh checks`, the scripts read `data/`):

| file | |
|---|---|
| `parton_cut_check.py` | what the leading-parton cut \|y\| < 1.9 removes from CMS 2507.09307's Z sample, and the pp Δφ / Δy against CMS with and without it (`data/vacuum` against `data/vacuum_zh`) |
| `window_layout.py` | effective Z's of 2507.09307, 2103.04377 and ATLAS for 5- and 6-window layouts from 20 to 200 GeV, by reweighting a vacuum sample with its events' p̂T (`data/vacuum_nocut`) |
| `zjet_pp_jets.py` | pp Z+jet against CMS 1702.01060: anti-k_T R = 0.3 jets from all particles except neutrinos and the Z's leptons, smeared with CMS's jet resolution (pp, and Pb+Pb 0–30 % for the "smeared pp" tables), Δφ_jZ, x_jZ, ⟨x_jZ⟩, R_jZ (`data/vacuum_all` → `figures/zjet_pp_vs_cms1702.png`, `zjet_pp_summary.txt`; ~2 min); its own jet finding on the HDF5, an independent cross-check of `zjet_jets.py`'s pp figure |
| `data/vacuum_nocut/`, `vacuum_zh/` | 6,000 vacuum events each, no parton cut, charged, \|η\| < 4: the test's windows / the campaign's (20–30, 30–35, 35–45, 45–55, 55–70, 70–200) |
| `data/vacuum_all/` | 12,000 vacuum events, the campaign's windows and storage with all particles (\|η\| < 2.6, no positions): the pp reference of the Z+jet analysis (174 MB) |
| `data/hepdata_zjet/` | CMS 1702.01060's HEPData record (ins1512112: Δφ_jZ, x_jZ, ⟨x_jZ⟩, R_jZ, pp and Pb+Pb) |
| `data/pbpb_campaign/`, `pbpb_campaign_charged/` | one 24-event Pb+Pb job (seed 101, one background, N_coll 443) with the campaign's arguments, all particles / `--charged`: 2.7 / 1.6 MB per event, ~14 s per event and 10.6 GB alone on the GB10; the seed-101 reference of `utils/zjet/zjet_3090_test.sh report` |

**The ROOT path** (2026-10-16; [`../../../analysis_root/zjet`](../../../analysis_root/zjet/README.md)):
`data/root/<sample>/` the ROOT files of `run_h5toROOT.py`, `data/mix_cxx/` and
`data/jets_cxx/` the C++ map steps' outputs, `data/jets_py/` the Python Z+jet map step's
(`*_s4`: 4 oversamples, for the jet-by-jet check).

`runs/` is git-ignored (`../../.gitignore`): the hadron files are 1–525 MB each,
the movie's pair and particlize files 930 and 770 MB; the figures are not public yet.

## Making the figures

Every PNG below comes with `<figure>.npz` and `<figure>.root`: what it draws, to restyle it
(`python ../../jetcorr/replot.py <figure>.npz --style my.mplstyle`, or `replot.draw(..., edit=…)`)
or to use its numbers elsewhere without rerunning (`../../jetcorr/figdata.py`'s header has the layout;
`JS_FIGDATA=0` writes the PNGs only, `JS_FIGDATA_ROOT=0` skips the ROOT files; ROOT via
PyROOT, else uproot with the graphs as small TTrees).

The checks (2026-10-16), once `make_data.sh checks` (or the single targets) made their data:

```bash
python parton_cut_check.py     # fraction removed by the cut, pp vs CMS 2507.09307 with / without
python window_layout.py        # the window layouts' effective Z's
python zjet_pp_jets.py         # -> figures/zjet_pp_vs_cms1702.png, figures/zjet_pp_summary.txt
```

The test:

```bash
./make_data.sh                 # or: ./make_data.sh pbpb | vacuum | hepdata | movie
python accumulate.py           # -> data/hists.npz
python compare.py              # -> figures/*.png, figures/summary.txt
```

`make_data.sh` needs X-SCAPE built with the PyJetscape bindings from the `zjet` branches
(the Z trigger and `--hadrons-only`) and a GPU; Pb+Pb runs two jobs in parallel, 15–18
min and 14–16 GB each on the GB10; the vacuum ~4 min on the CPU; the movie event 3 min
and 17 GB, its renders 1.5 min (pyvista, imageio). The Pb+Pb jobs are also the Pb+Pb side of
[docs/BENCHMARK_PbPb_vs_AuAu.md](../../../../../../docs/BENCHMARK_PbPb_vs_AuAu.md) (2.1× an Au+Au
job's time and memory, mostly MUSIC's larger box). With the files in `data/`, only the
last two steps are needed (numpy, PyYAML, matplotlib and `jetscape`; `accumulate.py` falls
back to `../../../../python` for `jetscape`).

The same data run all of `../zjet_wake.ipynb` (maps, profiles, p_T^Z dependence, the
hydro wake of the movie event and a movie frame), from `example/analysis/zjet/`:

```bash
D=../runs/zjet_setup/data
ZJET_HADRONS=$D/pbpb ZJET_VACUUM=$D/vacuum ZJET_PAIR=$D/movie/AuAu_0_10_jet_seed0201.h5 \
  ZJET_PAIR_EVENT=1 ZJET_MOVIE=$D/movie/zjet_wake_event1_frame.png jupyter lab zjet_wake.ipynb
```

## The data

- **Pb+Pb** (seeds 101, 102): `PbPb_0_30_ZJet.xml` (b < 8.57 fm, Z + parton, Z → ee/μμ,
  EOS 91), both leptons p_T > 20 GeV, |η| < 2.5, hardest parton |y| < 1.9. Per job 2 hydro
  backgrounds × 6 p̂T windows (20–30 … 120–200 GeV) × 4 jets; 4 backgrounds in all,
  (the campaigns' settings until 2026-10-16; since then no parton cut and the windows 20–30,
  30–35, 35–45, 45–55, 55–70, 70–200, `utils/zjet/README.md`; this test is not remade)
  N_coll 443, 487, 772, 835. 40 iSS samples of the jet and the background legs, sampled
  correlated, 10 fragmentations; charged hadrons, |η| < 4. `bulk_jet` holds one full
  Pb+Pb event (bulk + wake, ~5,550 hadrons per sample) per Z event, `bulk_bg` one per
  background: 48 against 2 units per job, hence 24× the size (the wake itself is ~+30
  hadrons per sample). These files keep the freeze-out positions (`x`) at full precision;
  campaigns use `--no-x` and fewer momentum bits (`utils/zjet/README.md`).
- **pp** (seed 901): `run_prod_vacuum.py` with the same XML's PythiaGun and Matter in vacuum,
  the same windows and cuts, 1,000 events per window, 10 fragmentations.
- **Movie** (seed 201): 2 events on one background with the full output
  (`--write-particlize both`): the pair file `AuAu_0_10_jet_seed0201.h5` (the hydro
  evolution and the shower) and the particlize file, p̂T 80–120 GeV, Z and hardest parton
  with |y| < 1; their hadrons from `hadronize.py` (4 samples, 2 fragmentations,
  correlated). Event 1 (Z → μμ, p_T^Z 109.6 GeV, y_Z −0.49) rendered with
  `contribs/Visualization` (Visualization README, "Z+jet events"):
  `zjet_wake_event1.mp4` (`wake_pyvista.py`: the wake in the energy density, the shower,
  the Z's muons and an arrow along the Z) and `zjet_particles_event1.mp4`
  (`hydro_jet_particles_pyvista.py`: the particlized hadrons), with one frame of each as
  PNG. The event ran on js-contrib 417d991; the Z overlay of the renders is in 46cea78.
- **CMS**: 40 < p_T^Z < 350 GeV, |y_Z| < 2.4, Z → μμ, ~2,000 Zs in Pb+Pb; tracks
  p_T > 1 GeV, |η| < 2.4; 0–30 / 30–50 / 50–90 % and pp, three track p_T classes. Only
  0–30 % and pp are used here.

## The observable

CMS measures Δ⟨N_ch⟩ = S − B per Z (eq. 2 of the paper): S the same-event Z–track pairs,
B mixed events (the Z of one event with the tracks of another Z event), merged over ±Δφ
(±Δy), scaled by 0.5 and reflected. The model does the same with:

- **Events**: CMS's selection (`zjet_wake.CMS_2507`: 40 < p_T^Z < 350 GeV, |y_Z| < 2.4,
  60 < m < 120 GeV, leptons p_T > 20 GeV, |η| < 2.4) → 63 Pb+Pb Zs, 4,024 pp Zs. Weights
  σ_window / N_window × N_coll (`zjet_wake.event_weights`): 34 effective Pb+Pb events.
- **Tracks**: charged, |η| < 2.4, without the Z's leptons.
- **S**: the Z-correlated yield, wake (⟨bulk_jet⟩ − ⟨bulk_bg⟩, each event against its own
  background) + jet fragments. The uncorrelated bulk cancels in S − B and is left out of both.
- **Δφ** (all Δy): B is flat in Δφ, so S − B is S minus its Δφ average (CMS's data also
  average to zero).
- **Δy** (|Δφ| < π/2, Δy = y_Z − η_ch): B is half (the Z side) of event i's y_Z folded with
  event j's lab-frame η distribution, over the pairs i ≠ j; pp is the check of this.
- **Errors**: Pb+Pb per-event statistics ⊕ a jackknife over the 4 hydro backgrounds (events
  sharing one are correlated; the jackknife is 0.8–2.3 × the per-event error); pp per-event
  statistics. χ² over the independent (unreflected) bins with CMS stat ⊕ syst, bin-to-bin
  correlations ignored.

## Result

In `figures/RESULTS.md` (the figures, a table and the findings), with all numbers in
`figures/summary.txt`; both are git-ignored with the figures until the result is public.

## Caveats

- **The parton cut |y| < 1.9 of this test** removes 18–19 % of CMS 2507.09307's Z's (those with
  the recoil at |y| > 1.9; vacuum references, 2026-10-16). Without it the pp reference matches CMS
  better: Δφ χ² 35 → 20 and Δy 17 → 12 over the three p_T classes (30 independent bins each),
  most at 4–10 GeV (Δφ 20.6/10 → 7.1/10). So the too-high pp jet-side peak of this test comes
  mostly from the cut; whether it also narrows the Pb+Pb Δy dip is not checked (the campaigns run
  without it).

- 4 hydro events: the geometry and its fluctuations are poorly sampled, and a 4-group
  jackknife is itself a rough error estimate.
- **Every hydro event has its event plane near φ = 0** (3dMCGlauber's impact parameter along x),
  in this test and in the campaigns: the bulk's φ, summed over backgrounds, is not flat. Harmless
  for the means; `mix.py` and `zjet_jets.py` (B_mixed, and `zjet_jets_qa.py` with it) rotate the
  backgrounds for the mixed modes' noise by default (above; `--no-rotate` as produced).
- Centrality from b < 8.57 fm with N_coll weights; CMS from the HF energy. `centrality_check.py`
  (200,000 minimum-bias events, 2026-10-16): this model's own σ_inel is 7.89 b (3dMCGlauber, σ_NN 67.6 mb), not the 7.7 b behind b_max, so b < 8.57 fm is its 0–29.3 % (exactly 0–30 %: 8.68 fm); classes defined by event activity (the string count, which is 3dMCGlauber's own centrality cut, N_part, or a two-component N_part/N_coll mix) agree with the b class for 96–97 % of their events, with the same ⟨N_coll⟩ (1,030) and N_coll-weighted ⟨b⟩ (4.9 fm). So the b cut is the model's 0–29.3 %
  and nearly the same events as an activity class; how HF energy orders the events is not
  modelled.
- Pythia 8 Z + parton at leading order; CMS's reference generator is MadGraph5_aMC@NLO.
- Hadron level, no detector; CMS's data are corrected. CMS uses Z → μμ only (no effect at
  this level).
- Δy bins of 0.8 (CMS Pb+Pb) are means of two 0.4 bins; the mixed-event error is
  approximate (the η distribution's errors at the mean y_Z).
