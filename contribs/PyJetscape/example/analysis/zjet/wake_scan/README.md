# wake_scan: a paired scan of one liquefier setting

How the Z-side Δy of Pb+Pb − pp (CMS 2507.09307) responds to a liquefier setting, first
CausalLiquefier's `tau_delay` (0.5, 1.0 = the campaign's, 2.0 fm). What is measured is
[`../wake_decomp.py`](../wake_decomp.py)'s split of that Δy into the fragments' and the wake's
parts, each into its Z-side yield and its level (the mean subtraction of CMS's observable), for
every value and for every value against the baseline. Outputs in `../../runs/wake_scan/`
(git-ignored: the results are not public).

## Why tau_delay, and why paired

CausalLiquefier deposits each droplet as a causal-diffusion profile of size ~c·`tau_delay`
(c² = `d_diff` / `time_relax`, c ≈ 0.89 with the campaign's 0.08 / 0.1 fm: ~0.9 fm at
`tau_delay` 1.0), after the delay `tau_delay`. A smaller value gives a more compact, earlier
source: a stronger local perturbation of the fluid, whose hadrons respond more at higher p_T
(Cooper–Frye ∝ p_T / T). It changes only how MUSIC_2 receives the droplets: Matter and LBT run
on MUSIC_1, so the jets, fragments and droplets are the same for every value, and the
fragments' pieces of the split stay put. (`e_threshold` and α_s change the energy loss too, and
with it the fragments' level, which works against a Z-side depletion at 2–4 GeV; they are later
candidates.)

Every value runs the same jobs with the same seeds: the GB10 mini campaign's four
campaign-format jobs (seeds 102–105, 48 events each, two hydro backgrounds each, the campaign's
arguments: 40 oversamples, 40 fragmentations, correlated sampling). Same seed: the same
Pythia events, Z's, hydro backgrounds (MUSIC_1), energy loss and droplets; only the jet leg's
fluid (MUSIC_2) differs. A value minus the baseline is therefore formed event by event, with a
jackknife over the same blocks of hydro backgrounds: the error of the difference. The hadrons'
iSS and fragmentation seeds include each job's random `file_uuid`, so the particlization's
random numbers are not shared between values: the same partons and fluids, other samples (the
fragments' pieces of the differences show this noise floor; inside each value the wake keeps
its correlated sampling, `bulk_jet` and `bulk_bg` with common seeds).

`make_data.sh check` verifies the pairing seed by seed: the job XMLs differ in the scanned
setting alone (`check_settings.py`), and the jet files have identical events, partons,
initiators and per-event records except the jet leg's MUSIC record (`check_pairing.py`:
`E_droplets_early`/`late`, `n_cells_jet`, `ntau_jet_music`, `jet_hit_*`).

## Files

| file | |
|---|---|
| `make_xml.py` | a variant of the campaign XML (`../../../config/PbPb_0_30_ZJet.xml`) with one leaf changed in the text (comments and layout kept); the leaf's tag must occur once (`tau_delay`, `e_threshold`; not `alphas`) |
| `check_settings.py` | two XMLs differ in the allowed setting(s) alone: the variant XMLs against the campaign's, the job XMLs value against baseline |
| `check_pairing.py` | two same-seed jet files share events, partons and droplets |
| `make_data.sh` | `xml`, `prod` (then `check`), `check`, `pools`, `compare`, or `all` |
| `compare_scan.py` | every value's pieces and the paired differences → `wake_scan_<setting>.png`, `wake_scan_<setting>.txt` |

```bash
cd $EX/analysis/zjet/wake_scan
./make_data.sh            # or: xml | prod | check | pools | compare
# another setting or values (a tag that occurs once in the XML):
PARAM=Liquefier/e_threshold VALUES="2.0 3.0 4.0" BASE=2.0 ./make_data.sh
# OUT, SEEDS ("102 103 104 105"), EVENTS (48), PAR (2 jobs at a time), PP_POOLS
```

The pp (Pb+Pb − pp; it cancels in the differences): the campaign's pp pools of `../pp_comp`
(`../pp_comp/make_data.sh prod pools`; js-contrib branch `pp_comp` until it is merged), or any
version-4 pp pools of the campaign's settings via `PP_POOLS`.

On the GB10: a job 16–21 min with two at a time, so ~40 min per value, ~2 h for three; ~210 MB
per job; the pools and the comparison minutes. A rerun of `prod` skips the jobs whose
`<stem>.json` exists.

Size: ~100 selected Z's (CMS 2507.09307's cuts) on 8 hydro backgrounds. On the mini campaign
the wake's pieces at 2–4 GeV (first Δy bin) carry ±0.05–0.06 each; the paired differences carry
less (the same fluids, jets and droplets): enough for shifts of ~0.1, the size that would move
the model's 2–4 GeV Z side towards CMS. If a difference's error comes out above ~0.05, more
seeds (`SEEDS`) reduce it as 1/√(jobs).

Outputs (`../../runs/wake_scan/`): `data/xml/` the variant XMLs; `data/<setting>_<value>/` the
hadron files, job XMLs and JSONs; `data/mix/<setting>_<value>/pbpb/` the pools; `data/logs/`;
`figures/`.
