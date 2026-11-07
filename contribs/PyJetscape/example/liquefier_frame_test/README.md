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
| `make_data.sh` | the test data: tutorial 06's event with 25 jets, in the three frames, hadronized |
| `frame_summary.py` | the data reduced to `summary.npz` (hydro, droplets, hadrons) |
| `compare_frames.ipynb` | the comparison: energy conservation, where the energy goes, wake observables |
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
```

0-10% Au+Au at 200 GeV, seed 101, one background reused for 25 jets (PythiaGun pTHat 30-40 GeV,
hard partons at |y| < 0.6), both freeze-out surfaces; hadronized with correlated sampling (200
samples) and 20 fragmentations per event. These are tutorial 06's settings (`tutorials/v2/
AuAu_pair_mini`, made in the lab frame): its 5 events are the first 5 here (droplets and
background identical). The three runs differ only in the deposit: background, jets and droplets
are bit-identical (`frame_summary.py` checks it).

## Results (25 events)

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
| charged-hadron wake at \|eta\| < 1: N per event | 11.89 | 11.85 | 11.67 |
| sum pT per event | 10.08 GeV | 10.29 GeV | 9.71 GeV |
| sum pT, against bjorken (event-by-event differences) | -0.21 +- 0.24 GeV | | -0.58 +- 0.17 GeV |

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
  - At hadron level the wake's multiplicity is the same, but its summed pT is 6 % lower and
    its spectrum slightly softer. That is a 3.4 sigma hint in this sample, with one
    background and 25 jets.

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
