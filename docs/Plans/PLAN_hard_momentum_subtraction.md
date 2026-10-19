# PLAN: energy–momentum conservation between the hard process and the strings (`Subtract_hard_momentum` 1) for the Z+jet Pb+Pb production

2026-10-18. Why: on the first two campaigns (`zjet_c00`, `zjet_c01`) the model's Z-side depletion
in Δy is about half as deep and narrower than CMS 2507.09307's at 1–2 GeV and absent at 2–4 GeV
(`analysis/runs/zjet_setup/figures/ANALYSIS_c00.md` §6, `ANALYSIS_c00_deta.md`). The first candidate
reason is that the Pb+Pb production does not take the Z + parton's momentum out of the two
nucleons' strings (`<Subtract_hard_momentum>0`): in reality those strings carry less, so the bulk
is depleted at rapidities tied to y_Z and y_jet, in every p_T class, without any wake. This plan
makes a small production that switches that on, with the model's twin kept exact, so that the
effect can be measured directly.

Written against the code of the campaign image `jhputschke/xscape-prod:cu124-zjet`
(X-SCAPE `38cd11cf`, 3dMCGlauber `b99d680`, js-contrib `f242174`). Items marked **verify** were
inferred from reading the code and have not been run.

## 1. How the switch works today, and why it needs the ISR chain

- `Subtract_hard_momentum` is a 3dMCGlauber parameter. The subtraction is done in
  `Glauber::Pick_and_subtract_hard_parton_momentum()` (`Glauber.cpp:426`) when strings are
  generated with `wound_nucleons = true` (`EventGenerator::generate_strings`): it finds the binary
  collision of the hard process by its position, marks its two nucleons as hard-collided, picks
  in each a valence quark with energy above the hard parton's (resampling the valence x's with
  x ≥ x_hard if none has), subtracts the hard parton's four-momentum from that quark
  (`get_a_valence_quark_sub_mom`), requires the two quarks within 0.5 fm transversally, erases
  the quark from the nucleon's string-making list, and marks the nucleons "subtracted", whose
  beam remnants are then **not** made into remnant strings (zeroed at `Glauber.cpp:1164`,
  `:1218`) but exported to the jet hadronization. With the flag 0 the hard momenta are zeroed
  before the subtraction, so nothing is subtracted.
- The hard momenta reach the Glauber through `InitialState::OutputHardPartonMomentum` (per side,
  `MCGlauberWrapper.cc:329`). Its only caller is `ISRRotation.cc:202`: the incoming (negative
  status) partons of the initial-state shower after iMATTER evolved them back to the beam.
  `PythiaGun` never calls it (in `iMATTER.cc:657` the call is commented out).
- The strings must be generated after the hard process: `JetScape.cc:343` enforces
  `generateOnlyPositions 1` with `<Hard><ISR>`; `MCGlauberGenStringWrapper::ExecuteTask` then
  fetches the hard momenta and position and calls `GenerateStrings(true, event_id)`. In the
  Pb+Pb production (`generateOnlyPositions 0`) the strings are made in
  `MCGlauberWrapper::ExecuteTask` with `wound_nucleons = false` before Pythia runs, so the
  subtraction code is never reached and **the flag alone does nothing**.
- The hard pair's remnants go to iColored, which hadronizes them with the jet partons through
  the ISR colour flow. That is the O+O production (`prod_OO_isr`,
  `config/OO_MCGlauber_MUSIC_isr.xml`, `docs/README_OO.md`), the framework of Soudi, Zhao,
  Majumder, PRC 112 (2025) 014905.

What stands in the way of using that chain for Z+jet Pb+Pb as is: `PythiaIsrGun` has no Z
trigger, no lepton pass-through and no `minPartons` (only the shared p̂T windows); iMATTER and
ISRRotation were built and used for HardQCD 2→2, an electroweak vertex with a colourless Z is
untested; with ISR the strings depend on the hard process, so there is no background reuse
(`README_OO.md`: no `setReuseHydro`), i.e. ~2× the cost per event; the two-leg, hadrons-only
driver is built around Colorless, the ISR driver around iColored. Hence the LO emulation below
first; the ISR/iMATTER version is step 7.

## 2. Design of the LO emulation

Keep the present flow for the background leg: the Glauber generates the strings in
`MCGlauberWrapper::ExecuteTask` (unsubtracted), MUSIC_1 runs on them once per background and is
reused for 24 jet events. Add, after PythiaGun has generated the Z+jet event, a second string
generation for the jet leg only: PythiaGun reports the momentum it took from each nucleon,
`MCGlauberGenString` regenerates the strings with `wound_nucleons = true`, and the existing
`Pick_and_subtract_hard_parton_momentum` removes that momentum from the hard pair. MUSIC_2 then
starts from τ₀ on the subtracted strings plus the droplets (no checkpoint restart).
`bulk_jet − bulk_bg` (the analysis's step 1) then contains the wake **and** the conservation
effect; with `--no-deposit` it contains the conservation effect alone.

Two things must hold for the twin to stay exact: the second string generation must reproduce
the first everywhere except at the hard pair (§3.3), and the hard pair's remnants and the
subtracted quark must stay in the bulk, since nothing takes them here (§3.2).

Expected size, as an order of magnitude only: at p_T^Z ≈ 50 GeV each nucleon gives up
x ≈ 0.02–0.05 (50–130 GeV; more when ISR and MPI are counted, §3.1), a few per cent of two
nucleons' strings out of ~300 participants whose strings make ~5 charged hadrons per unit η
each, localized at the subtracted quarks' rapidities: a depletion of order 0.1 per unit Δy at
1–2 GeV correlated with y_Z, against the 0.36 the model misses; little at 2–4 GeV, where the
bulk's spectrum ends.

## 3. C++ changes

Branches: X-SCAPE `zjet-hardsub` from `38cd11cf`; a 3dMCGlauber fork, branch `hardsub-noisr`
from `b99d680`, with X-SCAPE's `external_packages/3dMCGlauber` fetch pointed at it; js-contrib
`zjet-hardsub` from `zjet`.

### 3.1 `src/initialstate/PythiaGun.cc`: report the vertex and the momentum taken from each nucleon

In `ExecuteTask`, right after the vertex is sampled (`ini->SampleABinaryCollisionPoint(t, x, y, z)`,
line ~659):

```cpp
// tell the Glauber which binary collision the hard process sits at
// (MCGlauberGenString / GetMomandPos_* look the collision up by this position)
ini->OutputHardCollisionPosition(t, x, y, z);
```

After `py.next()` succeeded and before the status-62 hand-over loop:

```cpp
if (ini->GetSubtractHardMomentum() == 1) {
  // The momentum Pythia removed from each proton: the beam minus its remnants
  // (status 63). This is hard process + ISR + MPI, i.e. everything that is
  // handed over as status-62 partons, which is what the strings must not
  // carry a second time.
  const double P_A = 0.5 * eCM;                       // per nucleon, symmetric beams
  Pythia8::Vec4 taken_p(0., 0.,  P_A, P_A);           // projectile, +z
  Pythia8::Vec4 taken_t(0., 0., -P_A, P_A);           // target, -z
  for (int i = 0; i < py.event.size(); ++i) {
    const auto &q = py.event[i];
    if (q.status() != 63) continue;                   // beam remnants
    if (q.pz() >= 0.) taken_p -= q.p(); else taken_t -= q.p();
  }
  ini->ClearHardPartonMomentum();
  ini->OutputHardPartonMomentum(taken_p.e(), taken_p.px(), taken_p.py(), taken_p.pz(), +1, P_A);
  ini->OutputHardPartonMomentum(taken_t.e(), taken_t.px(), taken_t.py(), taken_t.pz(), -1, P_A);
  JSINFO << "PythiaGun: momentum taken from the nucleons, proj E " << taken_p.e()
         << " targ E " << taken_t.e() << " GeV (subtracted from the strings)";
}
```

`OutputHardPartonMomentum` already throws above 0.95 P_A. **Verify** that `ini` in PythiaGun is
the `InitialState` pointer with these virtuals (the ISR gun uses the same interface).
Alternative definition, for the hard system only: sum the incoming-shower partons per side
(status −41/−42, and −31 for MPI); the two agree up to the remnant bookkeeping. Pick one and
record it in the file attributes.

### 3.2 3dMCGlauber `src/Glauber.cpp`: keep the hard pair's remnants and the subtracted quark in the bulk

In the ISR chain the subtracted quark is erased and the hard nucleons' remnants are zeroed
because iColored hadronizes them. Here nothing takes them, so they must stay as strings. A new
parameter, default 1 so that the ISR path is unchanged (`Parameters.cpp/.h`):

```cpp
bool Parameters::remnants_to_jet_hadronization() const {
    return get_param_int("Remnants_to_jet_hadronization", 1) == 1;
}
```

In `Pick_and_subtract_hard_parton_momentum` (around lines 510 and 540): skip
`proj_collided->erase_one_quark()` and `targ_collided->erase_one_quark()` when the flag is 0, so
the quark, reduced by the hard momentum, still makes its string. **Verify** what
`get_a_valence_quark_sub_mom(E)` does to the quark's four-momentum (`Nucleon.cpp`): the intent
is "quark momentum minus the reported momentum", not only "select a quark with enough energy".

In `generate_strings` (the two remnant loops, lines ~1164 and ~1218), make the zeroing
conditional:

```cpp
if (iproj->is_hard_collided() && iproj->nucleon_is_subtracted()) {
    for (unsigned int ip = 0; ip < p_i.size(); ip++) {
        Mom_remnant_proj_.push_back(p_i[ip]);
        if (parameter_list.subtract_hard_momentum() &&
            parameter_list.remnants_to_jet_hadronization()) {
            p_i[ip] = 0.0;
        }
    }
}
```

and the same for the target. Expose the parameter in `MCGlauberWrapper.cc` next to
`Subtract_hard_momentum` (line ~123, `mc_gen_->set_parameter(...)`) and in the XML.

### 3.3 Reproducibility of the second string generation (the twin)

`EventGenerator::generate_strings(wound, event_id)` draws random numbers (string production
mode 1 picks collisions at random, string counts fluctuate), and `Pick_and_subtract` may
resample valence quarks. For MUSIC_2's strings to equal MUSIC_1's everywhere except at the hard
pair, the second call must consume the same random stream.

**Verify first**, standalone: `generate_pre_events()` + `generate_strings(false, id)`, then
`generate_strings(true, id)` with the subtraction off, and compare the two lists. If they differ,
save the generator state after `generate_pre_events()` and restore it before each
`generate_strings` (`RandomUtil::Random` holds the engine; add `save_state()` / `restore_state()`),
and give `Pick_and_subtract`'s resampling its own generator so that its draws do not shift the
stream. With the subtraction on, the lists must then differ only in the hard pair's entries.

Fallback that needs no Glauber change: do not regenerate; take the unsubtracted list and rescale
the strings attached to the hard pair's valence quarks so that their energy drops by the
reported momenta. Approximate, but an exact twin by construction, and enough for the size
estimate.

### 3.4 `src/framework/JetScape.cc`: the string regeneration without ISR

`DetermineTaskList` adds `MCGlauberGenString` and enforces `generateOnlyPositions 1` only under
`<Hard><ISR>`. Add a branch for `<IS><MCGlauber><Subtract_hard_momentum>1` without `<ISR>`:
do **not** require `generateOnlyPositions 1` (MUSIC_1 needs the strings from `ExecuteTask`);
append an `MCGlauberGenStringWrapper` task right after the hard module (its `ExecuteTask`
already calls `GetHardPartonPosAndMomentumProj/Targ()` and `GenerateStrings(true, event_id_)`);
replace the "energy counted twice" warning by an info line naming this mode.

**Verify** that MUSIC_2 takes its strings from the `InitialState` module at its own start (so
that the regenerated list is the one it sees) and that MUSIC_1 is not re-run in the same
event; and that PyJetscape's bindings expose `MCGlauberGenStringWrapper` if the driver builds
the task list itself rather than from the XML.

## 4. XML and driver (js-contrib)

A copy of the Z+jet XML, `config/PbPb_0_30_ZJet_hardsub.xml`:

```xml
<IS><MCGlauber>
  ...
  <generateOnlyPositions>0</generateOnlyPositions>   <!-- strings for MUSIC_1 as now -->
  <Subtract_hard_momentum>1</Subtract_hard_momentum>  <!-- regenerated, subtracted strings for MUSIC_2 -->
  <Remnants_to_jet_hadronization>0</Remnants_to_jet_hadronization>  <!-- no iColored: remnants stay in the bulk -->
</MCGlauber></IS>
```

`prod_AuAu_0_10_jet/run_prod_jet.py`:

1. **MUSIC_2 from τ₀, not from the checkpoint.** Find the A8 restart switch (grep `restart` in
   `run_prod_jet.py` and in music4gpu's X-SCAPE branch: the checkpoint at step 28,
   `MUSIC_RESTART_VERIFY`) and disable it when `Subtract_hard_momentum` is 1: the legs are no
   longer identical before the first droplet.
2. Record in the job's `.json` and the HDF5 attributes: `subtract_hard_momentum`,
   `remnants_to_jet_hadronization`, the taken momenta per event (parse the new `PythiaGun:` log
   line or add them to the event record), and the strings' total energy per leg.
3. Keep `--reuse 24` for the twin; the cost per event rises by about one MUSIC run (MUSIC_2 from
   0.42 instead of 0.96 fm/c and no checkpoint): roughly 1.5× the present time per event.

## 5. Build and tests on the GB10 (dev container)

`utils/BuildContainerDev.md` (X-SCAPE from source in `build_gpu`) with the three branches.
Then, in `prod_AuAu_0_10_jet`:

```bash
# T1: regression, flag 0 must be bit-identical to the old code
python run_prod_jet.py --events 4 --seed 1 --user-xml ../config/PbPb_0_30_ZJet.xml \
    --grid ../config/grid_PbPb.yaml --pthat-bins 20-30,30-35,35-45,45-55,55-70,70-200 --jets-per-bin 1 \
    --lepton-ptmin 20 --lepton-etamax 2.5 --hadrons-only --oversample 4 --n-frag 4 --correlated \
    --eta-max 2.6 --no-x --outdir out_ref --seed-registry none
#    the same command on the image's code; the three hadron files identical (h5diff, or
#    root_export/check_h5_vs_root.py after conversion)

# T2: the subtraction fires
python run_prod_jet.py ... --user-xml ../config/PbPb_0_30_ZJet_hardsub.xml --outdir out_sub --seed-registry none
grep "momentum taken from the nucleons" out_sub/*.log        # E per side ~50-300 GeV, no 0.95 P_A throw

# T3: the twin is untouched, the jet leg is not
#    bulk_bg of out_sub == bulk_bg of out_ref (same seeds, same strings for MUSIC_1): bit-identical;
#    bulk_jet differs; E(strings) before - after the regeneration == the reported momenta

# T4: the strings differ only at the hard pair (3.3)
#    dump both string lists and diff: all entries equal except the hard pair's

# T5: the conservation effect alone
python run_prod_jet.py ... --user-xml ../config/PbPb_0_30_ZJet_hardsub.xml --no-deposit --outdir out_sub_nodep
#    bulk_jet - bulk_bg is the energy-momentum effect without any wake: look at it in the Z frame
#    (zjet_wake / mix.py step 1 on out_sub_nodep)

# T6: energy bookkeeping as in analysis/jet_edep_balance_check.py, plus per event
#    E(strings, MUSIC_2) = E(strings, MUSIC_1) - E_taken(proj) - E_taken(targ)
```

T1 and T3 protect the campaigns' provenance; T5 is the physics check of the implementation.

## 6. The physics test (GB10, a few hours)

Three runs with the campaign's arguments (`--oversample 40 --n-frag 40 --correlated`, all
particles, `|η| < 2.6`, no parton cut), ~400 events each (17 backgrounds × 24; the GB10 did 323
events/h for this set-up, expect ~1.5× slower with MUSIC_2 from τ₀):

| run | deposition | subtraction | gives |
|---|---|---|---|
| A | on | off | the reference (c00's physics, new seeds) |
| B | on | on | wake + conservation |
| C | off (`--no-deposit`) | on | conservation alone |

Analysis as for c00 (`analysis/zjet/qa.py`, `mix.py` step 1; `ANA_Zhadron.md`): B − A and
C in Δy on the Z side at 1–2 and 2–4 GeV, and in Δφ. Step-1 errors with 400 events are ~0.05
per unit Δy at 1–2 GeV, enough to see the expected ~0.1. Note that with the subtraction the
"Z-correlated yield" of step 1 contains a term that CMS's mixed-event S − B also contains but
that the model's mixing analysis (steps 2–4 against step 1) will no longer close on: the
closure then measures the effect, which is how data see it.

## 7. Afterwards

- **Image**, once T1–T6 pass:
  `gh workflow run docker-prod.yml --ref zjet-hardsub -f xscape_ref=zjet-hardsub -f js_contrib_ref=zjet-hardsub -f variants=cu124 -f release_tag=zjet-hardsub -f moving_tag=false`
  → `jhputschke/xscape-prod:cu124-zjet-hardsub`, pinned by digest in the launcher; never mixed
  with the campaign image `2ca82bc4…` (c00/c01 must stay combinable). Add the 3dMCGlauber fork's
  URL to `utils/BuildContainerProd.md`.
- **The ISR/iMATTER version** (what the O+O framework does), if the LO emulation moves Δy:
  Z trigger, lepton pass-through and `minPartons` in `PythiaIsrGun` (as in `PythiaGun` on the
  `zjet` branch); iMATTER / ISRRotation with the electroweak vertex, shown in vacuum first;
  iColored behind LBT + liquefier; a Pb+Pb two-leg driver without background reuse (~2× the
  cost). The subtracted momentum is then the one the nucleons actually gave to the hard
  scattering including its ISR, which the LO emulation approximates by Pythia's beam minus
  remnants.
- A related production switch worth taking along: a random reaction-plane angle per
  background (all backgrounds have their plane along x; `ANALYSIS_c00.md` §7a), which would
  make the analyses' rotation unnecessary.
