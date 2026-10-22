# hadro_compare — iColored hadronization in O+O: before vs after the colour fixes, and the closures

How the jet hadrons of the O+O ISR chain (Matter only, iColoredHadronization) change with
the colour fixes of X-SCAPE `lbt_colored_hadro` (one colour-tag counter per event, the input
checks), and between the closures of the colour lines that end in the medium
(`<colour_closure>`: `reconnect`, the default, `thermal`, `fake`). See
[`docs/Plans/PLAN_lbt_colour_tags.md`](../../../../../docs/Plans/PLAN_lbt_colour_tags.md) and
X-SCAPE `README_ColourFlow.md` for what the fixes and the closures do.

| file | purpose |
|---|---|
| `run_hadro_compare.sh` | runs the five campaigns (resumable) |
| `compare_hadro.py` | the comparison → `figures/*.png`, `figures/hadro_compare.txt` |
| `SUMMARY.md` | the results (not in git) |
| `before/`, `after/`, `reconnect/`, `thermal/`, `fake/`, `xml/`, `figures/` | the campaigns' files, the generated XML, the figures (not in git, see `.gitignore`) |

| campaign | code | Matter | closure | compared with |
|---|---|---|---|---|
| `before` | before the fixes: X-SCAPE `contrib` 41bdf8c5 + js-contrib `main` 03d14fa (`BASELINE`) | no recoils | — (none needed) | `after` |
| `after` | `lbt_colored_hadro`, `--matter-only` | no recoils | — (none needed) | `before` |
| `reconnect` | `lbt_colored_hadro` | recoils (`recoil_on` 1) | `reconnect` | `thermal`, `fake` |
| `thermal` | `lbt_colored_hadro` | recoils | `thermal` | `reconnect` |
| `fake` | `lbt_colored_hadro` | recoils | `fake` | `reconnect` |

- **No LBT.** Matter without recoils opens no colour line, so the closures differ only with
  Matter's recoils (their holes): hence `recoil_on` 1 in the last three (the O+O template
  without its `<Lbt>` block, written to `xml/`).
- **Paired.** The same seeds in every campaign: the same Pythia hard events, the same
  geometry and the same Pythia seed for the jet hadronization of each event. `before` and
  `after` have the same partons (bit-identical; only the colour tags differ), and so do
  `reconnect`, `thermal` and `fake`. `reconnect` vs `after` (recoils on vs off) is not a
  hadronization change; it is shown as context.
- **Settings** as `hard_sub_compare/`: O+O 200 GeV, b < 5.6 fm (0–80 %), pTHat windows 10–20,
  20–30, 30–40 GeV, both hard partons at |y| < 0.6 (before the ISR), jet hadrons only (no
  iSS), with the Pythia input of each event (`--store-hadronization-input`). 40 jobs × 75
  events, seeds 101–140: 3000 events per campaign.

```bash
conda activate js_fno
BASELINE=/path/to/X-SCAPE-baseline ./run_hadro_compare.sh > run_hadro_compare.log 2>&1 &
python compare_hadro.py                  # figures/ and figures/hadro_compare.txt
```

`BASELINE` is an X-SCAPE checkout at `contrib` 41bdf8c5 with js-contrib at `main` 03d14fa
inside, built (default `~/JetScape/X-SCAPE-regbase`, build `build_gpu_base`; the script
refuses a tree that already has the fixes). `NJOBS`, `EVENTS`, `SEED0`, `P`, `BLOCK`,
`BASE_BUILD`, `OUTROOT` and `OMP_NUM_THREADS` (environment) change the defaults. The
campaigns alternate in blocks of `BLOCK` (8) jobs, so a stopped run leaves equal statistics
in all five; `compare_hadro.py` uses the seeds complete in all five (`--jobs N`: the first N).

**What `compare_hadro.py` compares** (per weighted event, Ncoll × σ(window)/N(window); errors:
jackknife over the jobs, the paired ratios keep the jobs together):
- charged-hadron pT spectrum (|η| < 1), dN_ch/dη, the energy flow dE/dη (all particles), the
  charged multiplicity distribution;
- identified hadrons: π±, K±, p p̄, K⁰S, Λ (pT spectra at |y| < 1), and the yields and ⟨pT⟩ of
  those and of Σ±, Ξ⁻, Ω⁻ and photons; K/π, p/π, Λ/K⁰S;
- around the leading hard parton (from the shower initiators): the charged-pT profile in ΔR
  and z = pT/pT,lead within ΔR < 0.4;
- event by event: the fraction of events whose jet hadrons changed, split by whether the
  change could touch them (duplicate tags before / open colour lines), and the paired energy
  difference;
- the same ratios for the affected events only (`affected.png`, the second and third tables);
- the colour audits of the campaigns (holes, joins, partners, soft end partons, retries) and
  the duplicate tags of `before` (from its stored Pythia input).

Pythia's decays as in the job (iColored: decays on, cτ > 10 mm stable: K⁰S, Λ, Σ±, Ξ⁻, Ω⁻ are
final).
