# The liquefier's frame: lab, bjorken and fluid

X-SCAPE's CausalLiquefier spreads each droplet (what a jet gives up to the medium) into MUSIC's
fluid by a causal diffusion, deposited on the slice tau_drop + tau_delay. The diffusion is
isotropic in one frame, and `<CausalLiquefier><frame>` chooses which:

| `<frame>` | the diffusion is solved in | |
|---|---|---|
| `lab` | the collision's frame, as in the liquefier's paper (Tachibana, Shen, Majumder, arXiv:2001.08321, for a medium with small flow) | X-SCAPE before 2026-11 |
| `bjorken` | the frame of the Bjorken flow at the droplet's eta_s (a longitudinal boost by eta_d) | the default |
| `fluid` | the rest frame of the background fluid where the droplet was made, transverse flow included | |

`run_prod_jet.py --liquefier-frame lab|bjorken|fluid` sets it; the pair file records it
(`liquefier_frame`; files without it were made in the lab frame). This directory tests the three
on the same events.

| file | |
|---|---|
| `make_data.sh` | the test data in the three frames, hadronized: tutorial 06's Au+Au event with 25 jets, or (`SYSTEM=zjet`) LHC Z+jet events with the Z+jet campaign's settings |
| `frame_summary.py` | the data reduced to `summary.npz` (hydro, droplets, hadrons) |
| `compare_frames.ipynb` | the comparison at RHIC: energy conservation, where the energy goes, wake observables |
| `compare_frames_zjet.ipynb` | the same for the LHC Z+jet events |
| `liquefier_kernel.py` | the kernel in Python, all three frames (it reproduces the C++ to 1e-6) |
| `liquefier_replay.py` | one event's MUSIC legs rerun with its recorded droplets (`--frame`, `--ideal`) |

## The fluid frame

The background fluid is MUSIC_1's, the medium Matter and LBT see: when X-SCAPE makes a droplet
(`LiquefierBase::add_hydro_sources`) it looks up the cell at the droplet's position through the
liquefier's hydro signal and stores the lab velocity and energy density with it
(`Droplet::get_vfluid`, `get_efluid`; in the pair file `source/fluid`, columns `vx, vy, vz, e`).
The kernel (`CausalLiquefier::smearing_kernel`) then

1. boosts by eta_d, exactly as `bjorken` does (the droplet at z = 0, a cell at
   (tau cosh(eta - eta_d), tau sinh(eta - eta_d))),
2. boosts the cell-droplet separation by the fluid's velocity **relative to the Bjorken flow**
   at the droplet, v (mostly transverse; `kernel_frame_velocity`), into the fluid's rest frame,
3. evaluates the kernel's density and radial current at that age and distance,
4. boosts the current back by -v and takes its flux through the tau slice.

Splitting the boost this way keeps forward droplets accurate (the lab speed there is close to 1)
and leaves `lab` and `bjorken` bit-identical to the builds before the option. The deposit stays on
the slice tau_d + tau_delay (the decision of 2026-11-07; tau_delay in the fluid's own time would
move the slice). Without a usable fluid, `e < <fluid_e_min>` (0.1 GeV/fm^3: made before the hydro
starts at tau0, or outside the fireball), a droplet takes the Bjorken frame; the relative speed is
capped at `<fluid_v_max>` (0.99). The kernel stays a conserved current: on a fine grid its flux
through the slice is 1 within 0.7 % for transverse flow up to 0.9 (`tests/test_liquefier_frame.py`),
and MUSIC normalizes every droplet exactly on its grid as before.

FastHydro (`fast_data`) has no fluid frame yet: its deposit assumes a kernel bounded along eta_s,
which the boosted, tilted cloud is not. `fasthydro.liquefier_bridge` refuses `fluid` rather than
mapping it to another frame.

## The data

```bash
OUT=~/JetScape/liquefier_frame_test ./make_data.sh       # ~15 min on an M3 Max, 20 GB
jupyter lab compare_frames.ipynb                           # LFT_DATA=... if OUT differs
SYSTEM=zjet ./make_data.sh                                 # ~45 min, 63 GB (~/JetScape/liquefier_frame_test_zjet)
jupyter lab compare_frames_zjet.ipynb
```

0-10% Au+Au at 200 GeV, seed 101, one background reused for 25 jets (PythiaGun pTHat 30-40 GeV,
hard partons at |y| < 0.6), both freeze-out surfaces; hadronized with correlated sampling (200
samples) and 20 fragmentations per event. These are tutorial 06's settings (`tutorials/v2/
AuAu_pair_mini`, made in the lab frame): its 5 events are the first 5 here (droplets and
background identical). The three runs differ only in the deposit: background, jets and droplets
are bit-identical (`frame_summary.py` checks it).

## Results: Au+Au at RHIC (25 events)

From `compare_frames.ipynb`. 578 droplets, 1118 GeV (|E|). Fluid speeds relative to the Bjorken
flow average 0.28, at most 0.75, so none is capped. 52 droplets (68 GeV) take the Bjorken frame,
47 of them made before tau0.

| | lab | bjorken | fluid |
|---|---|---|---|
| droplet energy into cells MUSIC cannot invert (estimate) | 3.5 % | 0.4 % | 1.6 % |
| fluid gained / deposited, whole grid, tau = 2 / 3 / 4 fm/c | 0.90 / 0.95 / 0.97 | 1.00 / 1.00 / 0.99 | 1.00 / 1.00 / 1.00 |
| events keeping < 0.9 at tau = 3 fm/c (worst) | 4 (0.54) | 0 (0.91) | 0 (0.92) |
| droplet energy with sampled flux < 0.5 or > 1.5 | 21 % | 3 % | 5 % |
| deposit centroid, \|eta_s - eta_d\| (energy-weighted mean) | 0.16 (up to 0.9) | 0.01 | 0.02 |
| shift along the transverse flow | 0 | 0 | ~ v x 1.1 fm (0.8 fm at v = 0.7) |
| MUSIC cells per deposit (median) | 45 | 52 | 50 |
| wake energy at \|eta_s\| < 1, tau = 8 fm/c, sum of 25 events | 693 GeV | 671 GeV | 666 GeV |
| per event, against bjorken | +0.86 +- 0.32 GeV | | -0.19 +- 0.11 GeV |
| charged-hadron wake at \|eta\| < 1: N per event | 11.89 +- 0.60 | 11.85 +- 0.62 | 11.67 +- 0.58 |
| sum pT per event | 10.08 +- 0.35 GeV | 10.29 +- 0.37 GeV | 9.71 +- 0.34 GeV |
| sum pT at \|eta\| < 2.4 per event | 12.89 +- 0.34 GeV | 13.53 +- 0.37 GeV | 12.14 +- 0.35 GeV |

- **The lab frame:**
  - It loses energy early: event 0's 12.9 GeV droplet at eta_s = 2.4 lands entirely in
    uninvertible cells.
  - It pulls deposits toward midrapidity, inflating the midrapidity wake by 3 %.
- **The Bjorken frame** keeps the energy and puts every deposit around its own eta_s.
- **The fluid frame** agrees with the Bjorken frame along eta_s and in the energy bookkeeping.
  - It carries late deposits outward with the radial flow and contracts them along it.
  - The midrapidity wake maps differ from the Bjorken frame's by 18-30 % (L2, median over
    events), as dipoles along the flow.
  - The uninvertible share rises to 1.6 %: late, contracted deposits in dilute fluid.
- **At hadron level the three frames agree within errors** at |eta| < 1. The fluid frame's
  summed pT at |eta| < 2.4 is 1.4 +- 0.5 GeV lower (2.7 sigma) in this one-background sample.
  - The errors are each frame's own: every run hadronized its own samples, the background's
    included. So the error of a difference is the two in quadrature.
  - An earlier version took the error from the spread of the event-by-event differences.
    That misses the offset the background's samples give all 25 events, and made the
    |eta| < 1 summed-pT difference look like 3.4 sigma.

## Results: Z+jet in Pb+Pb at the LHC (48 events)

From `compare_frames_zjet.ipynb`: the Z+jet campaign's settings (config/PbPb_0_30_ZJet.xml,
0-30 % Pb+Pb at 5.02 TeV, pTHat 20-200 GeV in six windows, leptons pT > 20 GeV at |eta| < 2.5,
no parton rapidity cut), seed 102, two backgrounds. 2149 droplets carry 10 340 GeV (|E|), and
they are made forward: 86 % of the energy at |eta_d| > 1, 58 % at |eta_d| > 2, 25 % at
|eta_d| > 3. 16 % is made before the hydro starts and takes the Bjorken frame in the fluid run.

| | lab | bjorken | fluid |
|---|---|---|---|
| droplets missing every MUSIC cell centre (put into one cell) | 178, 22 % of the energy | 1 | 2 |
| droplet energy into cells MUSIC cannot invert (estimate) | 6.5 % | 0.1 % | 0.5 % |
| fluid gained / deposited, whole grid, tau = 2 / 4 / 8 fm/c | 1.00 / 0.99 / 1.01 | 1.00 / 0.99 / 1.00 | 1.00 / 0.99 / 0.99 |
| deposit centroid \|eta_s - eta_d\| at \|eta_d\| > 2 | 0.89 | 0.00 | 0.06 |
| MUSIC cells per deposit at \|eta_d\| > 2 (median) | 6 | 98 | 96 |
| wake energy at \|eta_s\| < 1, tau = 2 fm/c, sum of 48 events (made there: 209 GeV) | 503 GeV | 196 GeV | 200 GeV |
| the same at tau = 8 fm/c | 1193 GeV | 1001 GeV | 1002 GeV |
| per event, against bjorken, tau = 8 | +4.0 +- 0.5 GeV (+19 %) | | +0.02 +- 0.09 GeV |
| charged-hadron wake, N per event, \|eta\| < 1 | 7.4 +- 1.5 | 9.7 +- 1.2 | 7.2 +- 1.4 |
| sum pT per event, \|eta\| < 2.4 | 21.8 +- 1.3 GeV | 19.8 +- 1.1 GeV | 18.8 +- 1.3 GeV |

- **At the LHC the lab frame's error is not a tail.**
  - It moves forward energy toward midrapidity: by 0.9 units for droplets at |eta_d| > 2.
    The midrapidity wake is 2.6 times too large at tau = 2 fm/c and 19 % too large at
    8 fm/c.
  - 22 % of the energy misses MUSIC's grid and goes into single cells.
  - The whole grid still balances, but only as a net. The estimate puts 163 GeV of positive
    and 361 GeV of negative droplets into uninvertible cells, which MUSIC discards; a lost
    negative deposit raises the fluid's energy, so the two nearly cancel. The dense forward
    fluid also keeps most of the single-cell deposits.
  - So the main error is where the energy goes. At RHIC, by contrast, the lab frame's loss is
    a real net deficit: 10 % at tau = 2 fm/c, mostly one 12.9 GeV forward droplet in thin
    fluid.
- **The Bjorken and fluid frames agree** in energy, along eta_s, and in the midrapidity wake
  (2 %).
  - Forward, the fluid lags the Bjorken flow a little, so the fluid frame centres deposits
    slightly inward (0.05 units).
  - Transversely it shifts them with the flow, as at RHIC.
- **At hadron level this sample is too small to separate the frames** (40 samples, two
  backgrounds: 12-20 % errors per frame). The lab frame's +2.0 +- 1.7 GeV in summed pT at
  |eta| < 2.4 has the sign of the forward energy it moves in.
- **Z+jet campaigns made before 2026-11 used the lab frame.** Their midrapidity wake is partly
  forward energy moved in. Rerunning the campaign's hadron-level observables (CMS 2507.09307)
  in the Bjorken frame needs `analysis/zjet/wake_scan` with the frame as its setting.

## Replaying an event

`liquefier_replay.py` reruns an event's two MUSIC legs with its recorded droplets, so that only
the hydro (`--ideal`) or the liquefier (`--frame`) changes; with `--frame fluid` an older pair
file's droplets take the fluid interpolated from `arr_bg`. Event 0 of a native-grid job (pTHat
30-40 GeV, seed 101), the fluid's gain over the deposited energy, full grid:

| tau [fm/c] | 2 | 3 | 4 | 8 | end |
|---|---|---|---|---|---|
| lab, viscous | 0.25 | 0.57 | 0.62 | 0.70 | 0.73 |
| lab, ideal | 0.25 | 0.56 | 0.59 | | |
| bjorken, viscous | 1.14 | 1.06 | 1.02 | 1.00 | 0.98 |
| bjorken, ideal | 1.00 | 1.03 | 1.00 | | |

(Ideal hydro pushes matter off the grid's edges late, so its full-grid count stops there.) The
lab frame loses three quarters of a 12.9 GeV droplet made at eta_s = 2.4: squeezed into four
cells, it leaves them with |T^{tau i}| > T^{tau tau}, and MUSIC reverts such cells, discarding the
deposit. The job's other four events keep 99-107 % over the full grid in both frames. The late
excess of a few percent is the shear stress, which this count (the ideal part of T^{mu nu})
leaves out.

```bash
python liquefier_kernel.py shapes               # one droplet at eta_s = 0, 1, 1.8, 2.4, all frames
python liquefier_kernel.py check PAIR.h5        # the C++ normalization's flux, reproduced
python liquefier_kernel.py window PAIR.h5       # energy made in / deposited in |eta_s| < 1
python liquefier_replay.py PAIR.h5 --event 0 [--ideal] [--frame lab|bjorken|fluid] --out r.npz
```
