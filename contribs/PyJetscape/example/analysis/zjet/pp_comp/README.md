# pp_comp: the campaign's pp reference against JETSCAPE's PP19 tune

The Z+jet campaigns' pp reference (`run_prod_vacuum.py` with `../../../config/PbPb_0_30_ZJet.xml`,
[utils/zjet/README.md](../../../../../../utils/zjet/README.md) "pp reference") against JETSCAPE's
pp tune PP19 ([arXiv:1910.05481](https://arxiv.org/abs/1910.05481), Table I and §II.E), both in
CMS's pp Z–hadron (2507.09307) and Z+jet (1702.01060) observables and against each other.
Outputs in `../../runs/pp_comp/` (git-ignored: the results are not public).

## The two samples

PP19 and the campaign's pp differ in **Matter's Q0 alone**:

| | PP19 (Table I) | campaign | `pp19` here |
|---|---|---|---|
| Pythia | ISR on, MPI on, FSR off | the same | the same |
| partons to Matter | status 62, p_T > 2 GeV | `softMomentumCutoff` 2.0 | the same |
| hard processes | hard QCD, electroweak off | the Z + parton processes | the campaign's (a Z+jet reference) |
| Matter Q_ini | 0.5 p_T (`vir_factor` 0.25) | the main XML's 0.25 | the same |
| Λ_QCD | 0.2 GeV | the main XML's 0.2 | the same |
| **Matter Q0** | **1 GeV** | **2 GeV** (the Matter → LBT handover of the Pb+Pb runs) | **1.0** |
| hadronization | Colored and Colorless (the JETSCAPE repo's PP19 file: Colorless) | Colorless | Colorless, `../../../config/hadronize.xml` |
| decays | cτ < 1 cm | `tau0Max` 10 mm | the same |

Colored hadronization is left out (2026-10-19 decision): the tune does not require it.

`ZJet_pp_PP19.xml` is the campaign's PythiaGun and Matter blocks with `<Q0> 1.0`.
`check_settings.py inputs` checks that against the campaign XML, `check_settings.py jobs` that
the two job XMLs `run_prod_vacuum.py` wrote differ in `<Eloss><Matter><Q0>` and nothing else.

Both samples have the campaign's arguments and statistics (`zjet_pp` on the RTX box): 12,000
events, 2,000 per p̂T window (20–30, 30–35, 35–45, 45–55, 55–70, 70–200 GeV), leptons p_T > 20 GeV
and |η| < 2.5, 40 Colorless fragmentations per event, **seed 900001** for both. The trigger
and Pythia do not see Q0, so **the two samples have the same Pythia events and Z's**; only the
Matter showers and the hadronization differ. PP19 − campaign is therefore formed event by event,
with a jackknife over the same blocks of events (the error of the difference, much smaller than
either sample's).

## Files

| file | |
|---|---|
| `ZJet_pp_PP19.xml` | the PP19 sample's user XML (`run_prod_vacuum.py --user-xml`) |
| `check_settings.py` | the two samples differ in Q0 alone: `inputs` (the user XMLs), `jobs` (the job XMLs) |
| `make_data.sh` | everything, in steps: `prod` (both productions, then `check`), `map`, `pools`, `compare`, or `all` |
| `compare_pp.py` | the comparison → `pp_comp_zhadron_dphi.png`, `pp_comp_zhadron_dy.png`, `pp_comp_zjet.png`, `pp_comp_summary.txt` |

```bash
cd $EX/analysis/zjet/pp_comp
./make_data.sh            # or: prod | check | map | pools | compare
# OUT=<dir> EVENTS=<n> NFRAG=<n> SEED=<n> ./make_data.sh ...   (defaults: ../../runs/pp_comp, 12000, 40, 900001)
```

On the GB10 (CPU only): each production ~10 min (0.04–0.05 s per event with 40
fragmentations, the two in parallel), the map step a few minutes per sample, the pools and the
comparison minutes. Needs X-SCAPE's `build_gpu` with the PyJetscape bindings and the HEPData
records (`../get_hepdata.sh`).

Outputs (`../../runs/pp_comp/`): `data/<sample>/` the hadron files and the job XML and JSON;
`data/jets/<sample>/` the Z+jet map step (R = 0.3); `data/mix/<sample>/vacuum/` the Z–hadron
pools; `data/logs/`; `figures/`.

## The comparison (`compare_pp.py`)

- **Z–hadron, CMS 2507.09307 pp:** `mix.py`'s pp result for each sample, the factorized
  S − B ([`../MIXING_MODES.md`](../MIXING_MODES.md) §4: B pooled from all other Z events, flat in
  Δφ, Δy without the own event). Δφ and Δy (Z side) in the three track p_T classes against
  CMS's pp tables (stat ⊕ syst), model − CMS, and PP19 − campaign paired. χ² over the
  independent half of the reflected bins (`mix.chi2`). The summary also has the charged tracks
  per Z per p_T class.
- **Z+jet, CMS 1702.01060 pp:** `zjet_jets.py`'s pp observables (Δφ_jZ, x_jZ, ⟨x_jZ⟩, R_jZ;
  R = 0.3, |η_jet| < 1.6, p_T,jet > 30 GeV after the smearing) smeared to PbPb's jet resolution
  against CMS's smeared-pp tables 1, 3, 5, 7, smeared to pp's against table 9 (x_jZ), model / CMS,
  and PP19 − campaign paired, unsmeared (the smearing's random numbers would add their own
  noise).
