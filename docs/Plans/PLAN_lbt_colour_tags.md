<!-- Design note, written 2026-10-15; checked against the code and updated 2026-10-21 (section
     "Check of 2026-10-21", corrections marked "(2026-10-21)", phases reordered). Line numbers refer
     to X-SCAPE `contrib` at 41bdf8c5. Branches: `lbt_colored_hadro` in X-SCAPE (from `contrib`)
     and js-contrib (from `main`). -->

# Plan: colour tags for LBT partons, so iColoredHadronization can hadronize Matter+LBT showers

## Status

- **Phase 1 done (2026-10-21)**, X-SCAPE and js-contrib `lbt_colored_hadro`:
  `HardProcess::NextColorTag` (C1) used by Matter's splits; iColoredHadronization's pstat
  policy, counting pass, `<colour_duplicates>` and checked `pythia.next()` (C2); the audit
  (C3) as `core.jet_hadronization_colour_audit` and the jet file's `units/colour_*`.
  Validated: the two 75-event O+O jobs have no duplicate and no unpaired tag (65–66 events
  with duplicates each before), their partons' pid and four-momenta are bit-identical to the
  run before the change; the Au+Au colorless regression job (`run_prod_jet.py --events 2
  --seed 1 --reuse 2 --write-particlize both`) is identical except the partons' col/acol
  and the 1-ulp noise of `source/flux`, and its Colorless jet hadrons (`hadronize.py --tags
  jet_frag`, fixed seed) are identical; PyJetscape tests pass (`test_icolored_colour.py`
  new).
- **Phases 2–5 done (2026-10-21)**, same branches:
  - Phase 2: Matter's recoil block keeps `Matter::flavor`'s flow (recoil, hole, the
    continuing jet parton), its two new tags reserved from the event counter, the tags
    swapped with the momenta.
  - Phase 3: iColoredHadronization collects the partons into a list (`ShowerParton`,
    `HadronizePartons`, bound through `core.hadronize_partons`), then the closure of
    `<colour_closure>` (B2 within a shower → B1 partners → B2 across showers → B3, each
    counted; `reconnect_global`, `thermal`, `fake`, `none`), and one retry with B3 when Pythia
    fails after a closure. `core.jet_hadronization_set_colour_closure` sets it by hand.
    The beam remnants always go to Pythia: they carry negative statuses, and a first version
    that applied the pstat policy to them dropped them all (caught by the bit-identity check).
  - Phase 4: LBT's colour arrays, the tags passed through every step, the flow per channel
    (A1's table, channel 2 as corrected), the splitting rule for every radiated gluon, the
    tags swapped wherever LBT swaps slots (identical flavours, the leading gluon of a gluon
    jet).
  - Phase 5: `run_prod_oo_isr.py` takes `<Eloss><Lbt>` when the build has the closure
    (`ICOLORED_HAS_COLOUR_CLOSURE`); the audit columns grew accordingly.
- Validation (GB10, two 75-event O+O jobs per setting, seeds 103 and 129):
  - Matter only, recoils off: jet files identical to phase 1 (no open line, nothing changed).
  - Matter with `recoil_on` 1: 77 holes; `reconnect` closes every line with 25 joins and 50
    partners, `thermal` with partners alone, `reconnect_global` with more joins; never a
    soft fallback, a cross-shower join, a retry or a failure.
  - Matter + LBT: 632 holes in 150 events (up to 19 in one); `reconnect`: 234 joins and 286
    partners, `thermal`: 568 partners; nothing else needed, no failure.
  - In every setting: every tag paired exactly once in what went to Pythia, no parton among
    the hadrons, hadron energy = Pythia input energy.
  - `thermal` inserts fewer partners than there are holes when one parton scatters twice and
    the second hole's line runs into the first through it: the two holes are connected inside
    the medium, and only the net open end needs a partner (with the right triplet charge).
  - Au+Au colorless regression (which has `recoil_on` 1 and LBT): identical except the
    partons' col/acol (untagged final quarks and gluons 11 % → 0 % in that job) and the 1-ulp
    noise of `source/flux`; its Colorless jet hadrons identical. PyJetscape tests: 279 passed.
- **2026-10-21, later:** Matter + LBT (Matter `recoil_on` 1, `<mutex>` on, as in the Au+Au
  and Pb+Pb jet XMLs) with iColored and `<colour_closure>reconnect` is the default of the O+O
  XML (`OO_MCGlauber_MUSIC_isr.xml`); `run_prod_oo_isr.py --matter-only` gives the Matter-only
  setting of before; the files carry an `eloss` attribute. X-SCAPE: `README_ColourFlow.md`,
  `config/jetscape_user_OO_ISR_MatterLBT_iColored.xml` for runJetscape, the run's colour
  summary in the log (`FinishTask`).
- **2026-10-23:** the cause of the duplicate tags re-traced from the stored inputs: Matter's
  colour counter carried over from event to event in the persistent energy-loss task and was
  copied into every shower's Matter, not the MPI block overlap first read from the code (see
  "Where the duplicates came from"). A pp Z+jet run (PythiaGun + Matter) had the same bug.
- Open: physics comparisons (iColored versus colorless on the same partons, the closure
  variants), the jet-file notebook, A2, negative hadrons for the holes, the Au+Au liquefier
  case (lines that leave with liquefied partons are closed by B2 across showers / B3).

## Goal

The O+O ISR job (`contribs/PyJetscape/example/prod_OO_isr/run_prod_oo_isr.py`) hadronizes the
whole hard nucleon pair (the dijet showered by Matter, the ISR emissions, Pythia's MPI partons
and the beam remnants) in one Pythia event with `iColoredHadronization`, which follows the
colour flow of the shower. The job is Matter-only and refuses `<Eloss><Lbt>`
(`run_prod_oo_isr.py:331-334`): LBT's partons carry no colour tag, and `iColoredHadronization`
throws on any unpaired tag. The same gap makes `ColoredHadronization` unusable in the Au+Au
two-stage job ([`PLAN_particlize_h5.md`](PLAN_particlize_h5.md), "Colored hadronization: not
supported": 75–91 % of the final partons untagged, 2–5 unpaired tags per event).

Goal: a Matter + LBT chain (and Matter with `recoil_on` 1) whose final partons carry a
consistent colour flow, and an `iColoredHadronization` that closes the colour lines that
scatterings off the medium open, so that the O+O job can run with LBT and still hadronize
the jets in the job.

These are two separate problems:
1. **Tags.** LBT, and Matter's recoil block, must assign tags (section A).
2. **Closure.** Every scattering off a thermal parton leaves one colour line that ends in the
   medium. At hadronization that line has to be closed by a modelling choice (section B).

Section C covers the bookkeeping both need, then phases, validation and risks.

## Check of 2026-10-21: the Matter-only O+O chain already breaks colour

Measured before any change, with the build of `contrib` 41bdf8c5:

- **Duplicate colour tags in 131 of 150 events.** Two 75-event O+O jobs (seeds 103 and 129,
  the `hard_sub_compare` settings, `--store-hadronization-input`): in 66 and 65 events a tag
  occurs more than once as a colour or as an anticolour among the partons handed to Pythia (up
  to 42 extra occurrences in one event). The tag counts still balance, so iColored's check
  passes, and Pythia strings the partons without a message. The largest tag grows from ~4,000
  in the first event to ~130,000 in the last.
- **Silent fragmentation failures.** In `prod_OO_isr/hard_sub_compare` (an older build), 100 of
  6031 events (1.7 %), in 2 of 80 jobs (`sub/` seeds 0103 from event 46 and 0129 from event 3,
  each to the end of the job), Pythia reported "ColConfig::insert: junction topology
  inconsistent; too few junction legs" and "hadronLevel failed; giving up". `pythia.next()` is
  not checked (`iColoredHadronization.cc:296`), so the input partons were written out as jet
  hadrons. The 150 events above had no failure.

### Where the duplicates came from (2026-10-23)

Traced from the stored iColored inputs of the two jobs and from a pp run. The first reading of
this note (iMATTER's blocks L × 1000 reused by the guns' blocks 1000 × (np + 1) once there are
MPI scatterings) is not what happened: none of the duplicated tags is an iMATTER tag, and 26 of
the 30 events without an MPI scattering have duplicates too.

- **Matter's counter lives in the Matter object.** `MaxColor` (`Matter.h:103`) is set to 101
  in `InitTask` and never reset. At every call Matter syncs it with the parton's `max_color`,
  the larger wins (`Matter.cc:277-285`): a smaller parton value is raised to the counter, a
  larger one becomes the counter. The assignment at a parton's first visit (`:604`) comes after
  the sync, so it never lowers the counter. Each split takes `++MaxColor` (`:1176`).
- **One Matter per shower, copied from a persistent one.** `JetEnergyLossManager` keeps task 0
  (the configured `JetEnergyLoss` and its Matter) for the whole run and makes tasks 1 … N−1 for
  each event as copies of task 0 (`JetEnergyLossManager.cc:110-145`; the copy clones Matter,
  counter included, `JetEnergyLossModule.h:41`). `ClearTask` erases the copies after the event
  (`:52-54`). So task 0's counter carries over from event to event and only grows, and every
  copy starts the event with it.
- **The blocks only separate the showers below that counter.** The guns give
  shower-initiating parton np the base 1000 × (np + 1) (`PythiaGun.cc:689`,
  `PythiaIsrGun.cc:610`). A shower numbers its tags from max(base, inherited counter) + 1. Once
  task 0's counter C is above the bases, every shower of the event numbers from C + 1: two
  showers that both split make the same tags C + 1, C + 2, … Each shower still pairs its own
  tags, so the counts balance, iColored's check passed, and Pythia joined each end to whichever
  matching end it found first, across showers.
- **What pushed the counter above the blocks:**
  - O+O: iMATTER gives each ISR sibling the base `Hard->GetMax_Color() + NISRShower × 1000`
    (`iMATTER.cc:581-583`), and `HardProcess::NISRShower` is never reset (`HardProcess.cc:80-88`),
    so the bases grow over the job (the largest tag from ~4,000 to ~130,000). When task 0
    held such a sibling, its counter took the sibling's base; from the next event on, the
    outgoing partons of every scattering (hard and MPI) numbered from the same value. The
    event's own siblings had still larger bases and kept separate blocks.
  - Any chain, through Matter's photon emission: `max_color` in `DoEnergyLoss` (`Matter.cc:245`)
    is an uninitialized local. A q → q γ split does not assign it (`:1171`) but writes it into
    both daughters (`:1486`, `:1585`), and the next call adopts the garbage as the counter
    (larger wins).
  - Without either, task 0's own splits make its counter creep through its block into the next
    shower's block after ~1000 tags (inferred from the code; in the pp run below the photon came
    first).
- **Measured:**
  - O+O, the two 75-event jobs: duplicates in 131 of 150 events. In every one the duplicated
    tags are a contiguous run starting one above a tag of an earlier event or a 1000-block base,
    i.e. above the inherited counter. Example: seed 103, event 7, three showers each carry 37001
    and 37002.
  - pp, the Z+jet vacuum campaign of `pp_comp` (`analysis/runs/pp_comp/data/campaign`,
    PythiaGun + Matter, no iMATTER, 12,000 events in one job, partons stored with shower, col
    and acol): tags of different showers coincide in 9,918 events (83 %), never within one
    shower. Events 0–124 are clean: shower 0 (task 0) creeps from 1001 to 1140 while the others
    start at 2001, 3001, … In event 125 shower 0 radiates a 127.5 GeV photon, the quark's next
    splits carry 3,642,272,109 and 3,642,272,110, and from event 126 on every shower numbers
    from there.
- **Who read the tags.** iColored: no repair, the wrong strings were made silently.
  `ColoredHadronization` relabels the extra occurrences pairwise from the end of its list
  (`ColoredHadronization.cc:252-307`): the tags become consistent, but paired by list order,
  not by shower. HybridHadronization reads the tags too (not checked). ColorlessHadronization
  does not read them: the Colorless productions (Au+Au, Pb+Pb, Z+jet, pp) are not affected.
- **What fixed it** is C1 below: one counter per event in the hard process, shared by all the
  copies and starting above every block; Matter takes its split tags from it. The member counter
  and the uninitialized `max_color` are still in Matter (`Matter.cc:258` on the branch), but no
  tag is made from them while there is a hard process. The pp case is fixed by the same change
  but was not re-run.

So C1 (one allocator) and C2 (pstat policy, a checked `pythia.next()`) are needed for O+O as it
runs today. They are now the first phase.

## What the code says

### LBT emits colourless partons

- `src/jet/LBT.cc` never calls `set_color` or `set_anti_color`. Its per-parton arrays hold the
  flavour only (`KATT1[]` positive, `KATT10[]` negative, `LBT.h:270-271`), kinematics,
  positions, the category `CAT[]` (2 = recoil) and the interaction time `Tint_lrf[]`.
- One parton per call: `DoEnergyLoss` loops over `pIn`, calls `jetClean()` (`:358`) and loads
  `pIn[i]` into slot j = 1 of the positive arrays (`:375-426`) or of the negative arrays
  (`:427-`, a hole that free-streams). The colour fields of `pIn[i]` are read nowhere.
- **(2026-10-21) Every LBT step re-creates the parton.** `flag_update` is set for every step in
  the medium with T > T_c and Q < Q0 (`:1165`), free streaming included, and the output loop
  then makes a new `Parton` from the arrays. So a parton loses its tags at its first LBT step,
  whether it scatters or not. Passing the input tags through slot 1 is most of A1; the
  scattering rules below matter only on the steps with a scattering.
- Output: `Parton(0, KATT1[j], out_stat, tempP, tempX)` for positives (`:578`; out_stat 0 jet,
  1 recoil) and `Parton(0, KATT10[j], -1, ...)` for holes (`:603`, `:639`). The `Parton`
  constructors set all five colour fields to 0 (`src/framework/JetScapeParticles.cc:262-266`).
  Only an `LBTUserInfo` (the lrf time) is attached.
- Slots: j = 1 is the parton that continues `pIn[i]`. An elastic scattering appends the recoil
  at `np0` and its hole at the same index of the negative arrays (`:1306-1375`); a radiated
  gluon takes the next `np0` (`:1440-1441`, `CAT` inherited at `:1479`, `:1542`). Positives
  j = 1..np and holes j = 2..np are index-aligned, so recoil/hole pairs stay recoverable.
- **(2026-10-21) At most one scattering per call.** `Nscatter = 1` is fixed (`:1303`), and
  `LBT0` processes only the slots present at its start (`np = np0` after the loop, `:1661`),
  i.e. slot 1. A call makes at most one elastic scattering plus its radiated gluons.
- Channels (`LBT::flavor`, `:2044-2209`): 1 gg→gg, 2 gg→qq̄, 3 gq→gq, 4 qq'→qq' (the thermal
  parton may be an antiquark of another flavour: then q q̄' → q q̄'), 5 qq→qq, 6 qq̄→q'q̄',
  7 qq̄→qq̄, 8 qq̄→gg, 11 Qq→Qq, 12 Qg→Qg, 13 qg→qg. For identical flavours the harder of
  jet-out and recoil becomes the jet (`:1338-1344`): momenta and flavours are swapped.
- Radiation: after the elastic step, `collHQ23` and then `radiationHQ` in a loop emit the
  gluons from the jet parton (slot i) for light and heavy partons alike (`:1420-1555`).
- Cuts: `cutOut` is 1e-6 GeV (`LBT.h:76`), so partons are not dropped in practice; `Ecmcut`
  (2 GeV) only free-streams. Holes (pstat −1) free-stream in later calls (`:427`) and end up
  as final partons of the shower.

### Matter computes recoil colours and throws them away

- The recoil block (`src/jet/Matter.cc:880-993`) calls
  `Matter::flavor(CT, pid0, pid2, pid3, max_color, col0, acol0, col2, acol2, col3, acol3)`
  (`:899`, defined at `:4375-4636`), which returns a large-Nc colour flow for jet-out (0),
  recoil (2) and hole (3). The result is then overwritten with 0 at `:963-988` ("assume
  colorless for recoiled and back-reaction parton, for the convenience of color string
  fragmentation in Pythia"); the jet parton keeps its old tags (`:989-993`, commented).
- `Matter::flavor` covers channels 1, 3, 4, 5, 7, 11, 12 and 13. Its convention for the hole:
  the tags of the **incoming** thermal parton, as Pythia stores beam partons. A tag on the
  hole's col pairs with the same tag on an outgoing col (it flows through), not with an
  acol. Read with col and acol swapped, the hole closes exactly the lines that jet-out and
  recoil leave open.
- Splittings (`:1167-1219`): one new tag n per split. q→q g: quark (n, 0), gluon (old col,
  n). g→g g: (old col, n), (n, old acol). g→q q̄: (old col, 0), (0, old acol). A photon leaves
  the parent's tags alone.
- Counter: a private `MaxColor`, set to 101 in `InitTask` (`:173`). Per parton it is synced
  with `pIn[0].max_color()` (`:277-284`, the larger wins), and set to the parton's base when
  Matter first sees a parton (`:604`); every split does `++MaxColor` (`:1176`) and writes the
  new value back to the daughters' `max_color`. (2026-10-23: the sync runs first, so `:604`
  never lowers the counter: each Matter object's counter only grows, and the per-event copies
  inherit task 0's. That is the duplicate-tag bug, see "Where the duplicates came from".)
- **(2026-10-21) Phase 2 is more than uncommenting.** The block must also (a) give the
  continuing jet parton `el_color0`/`el_anti_color0`, (b) write the advanced `el_max_color`
  back (today `flavor` advances a local copy only, so the next split would reuse the tags),
  (c) swap the tags in the identical-flavour swap (`:913-921`), and (d) the commented lines
  use a `max_color` that is not in scope there.

### Tag allocation with ISR

- `PythiaIsrGun` keeps Pythia's tags on the hard partons and gives parton np the block base
  1000 × (np + 1) as `max_color` (`src/initialstate/PythiaIsrGun.cc:610-616`;
  `max_colorPerShower` 1000 in `src/framework/HardProcess.h:381`). `PythiaGun` does the same
  (`PythiaGun.cc:689`). `iMATTER` allocates the ISR tags upward from −label × 1000
  (`src/jet/iMATTER.cc:381`, `:578-600`; it uses `MAX_COLOR + 1` when it picks the flavours,
  `:1039-1133`, and commits it after the split, `:578`). `HardProcess` already has
  `GetMax_Color` / `SetMax_Color` (`HardProcess.h:277-283`).
- (2026-10-23) With MPI scatterings iMATTER's blocks for labels −3, −4, … are the guns' blocks
  of the first scattering's outgoing partons (3000, 4000), but Matter leaves those blocks as
  soon as its inherited counter is above them, so the overlap was not seen in the 150 events;
  the duplicates came from Matter's counter (see "Where the duplicates came from").

### iColoredHadronization

- `src/hadronization/iColoredHadronization.cc` appends every final parton of every shower,
  holes included (the pstat filter at `:147` is commented out), as Pythia status 23 with the
  parton's tags, then the beam remnants (`:181-215`; it requires exactly two remnants per
  hard scattering, MPI scatterings included).
- It pairs the tags event-wide and throws `Unpaired colors sent to Pythia` (`:257`). The
  fake-parton closure of `ColoredHadronization` (`ColoredHadronization.cc:191-245`: one soft
  quark, antiquark or gluon per shower along the beam) is still in the file but unreachable
  after the throw. (2026-10-21) The pairing loop (`:237-251`) keeps reading `cols[icol]` after
  erasing the last element; C2 replaces it with a counting pass.
- No duplicate-tag repair (`ColoredHadronization.cc:252-307` has one), and `pythia.next()` is
  not checked (`:296`, and `:311` in `HadronizeInput`). An unchecked failure copies the input
  partons out as "hadrons" (Au+Au diagnostic, and the O+O events above).

### How the other modules treat holes

- `ColorlessHadronization` takes pstat 0, 22 and, with `take_recoil`, 1 as positives and
  hadronizes pstat −1 in a separate pass (`ColorlessHadronization.cc:220-240`).
- `HybridHadronization` sorts pstat < 0 into `neg_ptns`, hadronizes them separately, gives
  those hadrons a negative status and label −1, and rescales them so that positive minus
  negative conserves the shower energy (`HybridHadronization.cc:689-725`, `:1242-1270`). It
  samples thermal partons from the hypersurface to complete its strings.
- Framework: LBT's pstat −1 is an ordinary edge of the parton graph. Only the liquefier's
  statuses (−11 droplet, −13 miss, −17 negative; `src/framework/JetEnergyLoss.cc:124-126`)
  are treated specially. The O+O job has no liquefier. The Au+Au job's liquefier takes
  absorbed partons and holes out of the parton list, and their tags with them.
- In the Matter-only O+O job every final parton has pstat 1000 (PythiaIsrGun's mark, kept by
  Matter's daughters); recoils (1) and holes (−1) appear only with `recoil_on` or LBT.

### The physics constraint

In the large-Nc flow of a t-channel exchange the jet parton and the thermal parton trade a
colour line. In the final-state string system the jet-out and the recoil carry one tag each
that is matched only by the hole, and the hole is not a string parton. So every scattering off
the medium leaves exactly one colour line ending in the medium (the s-channel and
annihilation channels 2, 6 and 8 included, see the table in A1). Correct tags in LBT make the
number of open lines per event equal to the number of holes (plus whatever the liquefier
removed); they do not make it zero. The closure is a modelling choice. After a few soft
scatterings the colour of a jet parton is decohered anyway, so a reconnection with the medium
is defensible physics, not only a repair.

What the open ends are depends on the hole (2026-10-21):
- a **gluon hole** leaves one open colour and one open anticolour;
- a **quark hole** leaves one extra open colour (a triplet was taken out of the medium);
- an **antiquark hole** leaves one extra open anticolour.

## A. Where the tags come from

### A1. Tags inside LBT (recommended)

- Arrays `COL[]`, `ACOL[]` (positive) and `COL0[]`, `ACOL0[]` (negative) next to `KATT1` and
  `KATT10`, cleared in `jetClean()`. New tags come from the event-wide allocator (C1).
- Input (`:375`, `:427`): copy `pIn[i].color()` and `anti_color()` into slot 1.
- Elastic block (`:1312`): a colour-aware `flavor()`. Port `Matter::flavor`'s rules for
  channels 1, 3, 4, 5, 7, 11, 12 and 13, add channels 2, 6 and 8 (table below). Write the
  jet-out tags into slot i, the recoil's into `np0`, the hole's into `COL0[np0]`/`ACOL0[np0]`.
  Apply the identical-flavour swap (`:1338`) to the tags as well as to momenta and flavours.
- Radiation block (`:1440`, and the multi-gluon loop `:1495-1555`): Matter's splitting rule,
  one new tag per gluon, emitted from slot i: q→q g gives the quark (n, 0) and the gluon
  (old col, n); g→g g gives (old col, n) and (n, old acol); antiquarks mirrored.
- Output (`:578`, `:603`, `:639`): `set_color`, `set_anti_color`, `set_max_color` (and the
  min colours, as Matter does) on each new `Parton`.
- No random number is drawn for the tags, so every existing LBT run keeps its kinematics. The
  Au+Au colorless chain is the regression test.
- Size: about 250 lines in `LBT.cc` and `LBT.h`.

Elastic rules. Jet-in (c, a), new tags n and m, hole tags in Matter's incoming convention;
antiquark cases are the mirror image (col ↔ acol).

| CT | process | jet-out | recoil | hole (incoming) | source |
|---|---|---|---|---|---|
| 1 | g g → g g | (c, n) | (n, m) | (a, m) | `Matter::flavor` CT 1 |
| 3 | g q → g q | (c, n) | (n, 0) | (a, 0) | CT 3 |
| 13, 12 | q g → q g, Q g → Q g | (n, 0) | (m, n) | (m, c) | CT 13, 12 |
| 4, 5, 11 | q q' → q q', q q → q q, Q q → Q q | (n, 0) | (c, 0) | (n, 0) | CT 4, 5, 11 |
| 4, 7 | q q̄ → q q̄ | (n, 0) | (0, n) | (0, c) | CT 4, 7 |
| 8 | q q̄ → g g | (c, n) | (n, m) | (0, m) | new |
| 6 | q q̄ → q' q̄' | (c, 0) | (0, n) | (0, n) | new |
| 2 | g g → q q̄ | (c, 0) | (0, n) | (a, n) | new (2026-10-21, see below) |

(2026-10-21) Channel 2 was first written as quark (c, 0), antiquark (0, a), hole (n, n): that
makes the thermal gluon a colour singlet, which is not a large-Nc flow. In the flow above the
thermal gluon's colour annihilates the jet gluon's anticolour (t-channel quark exchange), and
the antiquark takes the thermal anticolour, so channel 2 opens one line like every other
channel and needs no "medium-internal" case. When the jet-out is the antiquark (`LBT::flavor`
draws the sign of `KATT2`), jet-out and recoil swap their tags.

### A2. Reconstruct the tags from the shower graph (cross-check only)

The `PartonShower` graph holds the parent edge (tagged) and LBT's children (untagged), whose
pstat tells their role (0 continuation, 1 recoil, −1 hole) and whose four flavours identify
the channel. A helper walking the graph before hadronization could apply the same rules
without touching LBT. It only works in the job (the stored parton files keep the final partons
and the initiators, not the edges). Since a call makes at most one scattering (above), it is
not ambiguous. Useful as an independent check of A1 on the same events, not as the main route.

## B. Closing the lines the medium opens

Needed with either A idea, and also the moment Matter's recoils are switched on with their
real colours. After A1 the open ends are known exactly: each hole with incoming tags
(h_c, h_a) leaves an outgoing col h_c without acol partner and an outgoing acol h_a without
col partner (zero tags excepted).

- **B1. Re-insert the medium partner.** Replace each hole by a positive parton of the
  conjugate flavour (an antiquark for a quark hole, a gluon for a gluon hole) with the hole's
  tags swapped, (col, acol) = (h_a, h_c), and the hole's own thermal momentum (it is the
  thermal parton that was kicked). The tags then pair automatically. It adds about one thermal
  energy per scattering to the event, the same approximation Hybrid makes with its sampled
  thermal partons; count it.
- **B2. Colour reconnection among the open ends (recommended default).** Drop the holes,
  collect the unpaired cols and acols, pair them by minimal string length (Pythia's λ
  measure, the log of the invariant masses) and relabel. Pair **within a shower first** (the
  partons of one initiator), and close what is left in that shower with B1 partners; never
  pair across showers by default, so that the two jets of a dijet are not strung together.
  A global variant (pairing over the whole event) is kept as a systematic. The counts differ
  in general (quark holes open a col, antiquark holes an acol, gluon holes one of each), so
  B1 is needed for the leftovers anyway. Adds no particles, and also absorbs every other way a
  line can break (liquefied partons, missing partners after a numerical reject, partons
  dropped by a cut).
- **B3. Fake end partons** as in `ColoredHadronization`: one soft quark or antiquark per open
  tag along the beam. Trivial, but it puts the medium partners at beam rapidity. Keep as a
  fallback for an event whose fragmentation fails, not as a default for O+O.
- **B4. The liquefier.** Deposit holes and soft recoils into the hydro as in the Au+Au two-leg
  job. In O+O that means a second MUSIC leg per event, and it still needs B2 for the tags that
  leave with the deposited partons. Only worth it if the wake matters more than the colour
  flow. Out of scope.

The holes' energy is a separate decision, independent of the closure: hadronize the negatives
in their own string system and give the hadrons a negative status as Hybrid does, deposit
them (B4), or drop them and accept a positive bias of about one thermal energy per
scattering. Start with "drop" and record the dropped energy per event.

### What gets built (2026-10-21)

B2 and B1 are both implemented; B1 is the step that closes what B2 cannot pair, and it can
also be chosen alone:

| `<colour_closure>` | what it does | role |
|---|---|---|
| `reconnect` | B2 within each shower, B1 partners for the leftovers | default |
| `reconnect_global` | B2 over the whole event, then B1 | systematic: string drag between the jets |
| `thermal` | B1 for every hole, no reconnection | systematic: the literal colour flow |
| `fake` | B3 | counted fallback for an event that fails to fragment |
| `none` | throw on unpaired tags (today) | |

For a gluon hole B2 joins its two open ends directly, which is B1 without the inserted gluon.

### Why B2 is the default and not B1

- **Energy accounting.** Per scattering the exact bookkeeping is
  jet-out + recoil − hole = jet-in. With B2 and negative hadrons that identity holds. B1 adds
  the partner on top, so the event is over by one thermal energy per scattering whatever is
  done with the holes, and B1 with negative hadrons is still over by that amount. With LBT's
  elastic rate that is tens of holes per event. The bulk hadrons from iSS already contain
  that medium, so B1 counts it twice.
- **Kinematics of the partner.** B1's partner carries the thermal momentum at the scattering
  point, early in the evolution, but Pythia fragments it as a final-state parton at
  freeze-out. A string from a hard parton to a soft parton pointing elsewhere in the fluid is
  long and yields soft hadrons that belong to neither the jet nor the bulk. B2 strings
  final-state partons to final-state partons, which is what the string picture assumes.
- **Coverage.** The liquefier removes holes and absorbed partons with their tags, and a
  failed or rejected parton leaves a dangling tag; B1 has nothing to insert in those cases, B2
  pairs whatever is left. This matters for the Au+Au job, where the liquefier is on.
- **Decoherence.** After several soft scatterings the colour of a jet parton is randomised
  relative to its original partner. Reconnecting the open ends by minimal string length is
  the model Pythia uses for multiparton systems, and it is cheap to vary.
- **Where B1 is the better physics.** It follows the colour flow literally: the medium partner
  exists, at the point of the scattering. Hybrid builds on the same idea, but it places its
  thermal partons on the hypersurface and uses recombination, so its kinematics are
  consistent; B1 as written is Hybrid's idea without Hybrid's placement. And B2 applied
  globally can reconnect the two jets of a dijet, which strings the event together and pulls
  hadrons to mid-rapidity (string drag); the within-shower rule above is there to prevent
  that.
- **Consequence.** B2 within a shower, B1 for the leftovers, and the spread between
  B2-within-shower, B2-global and B1 as the systematic. If that spread is large, the closure is
  a real model choice, and B1 then deserves a proper placement on the freeze-out surface,
  which is Hybrid territory.

## C. Bookkeeping

- **C1. One tag allocator per event.** (2026-10-21, design) `HardProcess::NextColorTag()`
  hands out tags from a per-event counter, reset in `HardProcess::ClearTask`, starting above
  1000 × (number of hard partons + 1): above Pythia's own tags, every `PythiaGun` /
  `PythiaIsrGun` block and every `iMATTER` block (labels up to half the hard partons). Matter
  (splits and recoils) and LBT draw their new tags from it; without a hard process (a module
  run on its own) they keep their present counter. `iMATTER` keeps its per-shower blocks,
  which are distinct within an event. The block bases are then no longer used to make tags.
  No random number is drawn, so the kinematics of every chain stay as they are; only tag
  values change (Colorless hadrons unchanged; the stored col/acol columns change).
- **C2. pstat policy in `iColoredHadronization`.** Positives (0, 1, 1000-marked, 22) to
  Pythia; holes (−1) to the closure (B) and the energy policy; liquefier statuses (−11, −13,
  −17) skipped. Pair the tags with a counting pass (also counts duplicates). Check the return
  value of `pythia.next()` and fail the event loudly (the job already isolates a throwing event
  with `--skip-failed`). Duplicates: counted, and the event fails by default; the
  `ColoredHadronization`-style relabelling is available as a counted option, not as a silent
  repair.
- **C3. Audit counters**, written per event with the jet hadrons (`units/colour_*` columns of
  the jet hadron file): partons, holes, liquefier partons skipped, unpaired cols and acols
  before closure, duplicate tags, reconnections, inserted partners, fragmentation failures,
  dropped hole energy. The expectation after A1 + B is "unpaired before = open ends of the
  holes, unpaired after = 0, duplicates = 0, failures = 0".
- XML sketch:

  ```xml
  <JetHadronization>
    <name>iColored</name>
    <colour_closure>reconnect</colour_closure>  <!-- reconnect | reconnect_global | thermal | fake | none (throw, today) -->
    <negative_partons>drop</negative_partons>   <!-- drop | hadronize -->
    <colour_duplicates>fail</colour_duplicates> <!-- fail | relabel -->
  </JetHadronization>
  ```

  LBT's tags need no switch: they change no kinematics and no random number.

## Phases (reordered 2026-10-21)

1. **C2 + C1 + C3 on the Matter-only O+O chain.** pstat policy, counting pass, checked
   `pythia.next()`, audit counters (with the Python binding and the jet hadron file's
   `units/colour_*`), and the event-wide allocator for Matter. Validate: duplicates 0 and
   failures 0 on the two 75-event O+O jobs above; their parton kinematics bit-identical (only
   the tags change); the Au+Au colorless chain bit-identical except the stored col/acol.
2. **Matter recoils with colour** (`recoil_on` 1, the recoil block of `Matter.cc:880-993`
   with (a)–(d) above). Produces coloured recoils and holes through code that exists, so B
   can be built and validated before any LBT change.
3. **Closure in `iColoredHadronization`** (B2 with the B1 leftovers, `thermal`, `fake`,
   `reconnect_global`, the XML switch), as a helper usable by `ColoredHadronization` too.
   Validate on the Matter recoil events: zero unpaired tags after closure, 100 %
   `pythia.next()` success.
4. **Tags inside LBT** (A1). Validate: Au+Au colorless chain bit-identical (hadron files and
   parton kinematics, not the col/acol columns); O+O with `<Lbt>` passes the audit.
5. **Job**: replace the `<Lbt>` refusal in `run_prod_oo_isr.py` by the audit, add the audit to
   `oo_isr_overview.ipynb`.
6. Optional: A2 as a cross-check on stored in-job graphs; negative hadrons for the holes.

## Validation

- Tag audit (C3) at the expected values on every event of a full seed, O+O with LBT and Au+Au
  with Matter + LBT (through `hadronize.py` once the parton files carry tags).
- Bit-identity: the kinematics of O+O and of the Au+Au colorless production unchanged after
  phases 1 and 4 (the tags draw no random numbers; the allocator must not either). The
  particlize and hadrons-only files store the final partons' col/acol: compare everything
  except those columns.
- Energy balance per event: hadrons = positives + inserted partners, and, with negative
  hadrons, hadrons − negative hadrons = initiators + (thermal energy of the inserted
  partners). With B2 and no partners the second term is zero.
- Physics: jet pT spectra, fragmentation functions and jet shapes for Matter + LBT with
  iColored versus colorless on the same partons, and Matter-only iColored versus colorless as
  the control (vacuum-like showers, where the two should agree). Closure variants
  (reconnect within a shower, reconnect global, thermal, fake) as a systematic; the
  within-shower versus global difference measures the string drag between the jets.
- Platforms: GB10 for the builds, the bit checks and the full seeds.

## Risks

- Pythia may fail to fragment strings that are tag-consistent but kinematically awkward (a
  thermal partner far from its string). B2 minimises the string length; a failing event falls
  back to B3 and is counted.
- B2 changes which partons share a string. Within a shower that is the intended model; across
  showers it would string the dijet together (string drag), hence the within-shower default.
  The jet-shape comparison between the within-shower and the global variant is the check.
- Each scattering rewrites the jet's tags, so the colour-aware `flavor()` runs before the
  radiation block of the same step.
- The multi-gluon loop and `radiationHQ` need the same care as the first gluon.
- The energy added by B1 partners is bounded by about one thermal energy per scattering, but
  it is not subtracted from the bulk hadrons; the audit records it.
- Matter's recoils re-coloured (phase 2) and the allocator (phase 1) change the Au+Au colorless
  production only if the colorless module reads the tags; it does not, so that chain keeps its
  hadrons. Check it.
- The O+O hadrons made with iColored change in phase 1 (no duplicates any more): runs before
  and after are not combinable.

## Related work and references

The pieces of the scheme have counterparts in other generators; the combination used here
(hole-based closure: join the open ends within a shower first, then the hole's own medium
partner, then counted fallbacks) we did not find elsewhere. References checked against
INSPIRE/arXiv on 2026-10-21.

**Colour flow of scatterings off the medium.** [BMW12a] give the first colour-differential
energy-loss calculation (large Nc, first order in opacity): in more than half of the
medium-modified branchings colour flows from the leading parton directly to the medium
("target") component, which makes colour-singlet clusters of parametrically larger
invariant mass (∝ √(p⁺ E_th)) and softens the hadrons; [BMW12b] works it out. That is the
line our holes carry, and our `thermal` closure is their "colour flows to the target" case.
The kinematics we give colour to are Matter's recoils (`Matter::flavor`) and LBT's recoils and
negative partons [HLWZ15].

| closure | closest counterpart | reference |
|---|---|---|
| `reconnect` (the line stays in the jet, no medium parton added) | JEWEL: every recoil becomes a gluon inserted into the jet's string next to the parton it scattered with (its "colour neighbour"); the thermal momenta are subtracted afterwards | [KZ17], [MWZ18] |
| B2, the shortest string first | the colour-reconnection models of Pythia (string length λ) and Herwig (cluster masses) | [AGS86], [SK94], [CS15], [GRS12] |
| `reconnect_global` | Pythia/Angantyr global colour reconnection in heavy-ion events | [BGLS18], [BGLS24] |
| `thermal` (a medium partner closes the line) | JETSCAPE Hybrid Hadronization: unpaired lines completed with thermal partons sampled on the hydro hypersurface (or beam partons), with recombination; POWLANG: heavy quarks string to thermal light partons of the freeze-out surface | [HFK16], [JS25], [BDMNP22] |
| `fake` (soft partons along the beam) | JETSCAPE ColoredHadronization's fake beam partons; Hybrid Hadronization's zero-momentum antiquark for a lone jet | [JF19], [JS20], [JS25] |
| holes not hadronized, energy recorded | LBT's negative partons, Hybrid Hadronization's negative hadrons, JEWEL's thermal-momentum subtraction | [HLWZ15], [JS25], [MWZ18] |

What differs here: our `thermal` partner carries the hole's own momentum (the thermal parton
where it scattered), not a parton sampled on the freeze-out surface as in [JS25] and
[BDMNP22]; our `reconnect` keeps the recoil's flavour and the large-Nc flow, where [KZ17]
turns every recoil into a gluon. The surrounding chain is X-SCAPE's ISR framework with
iColoredHadronization [SZM25].

- [AGS86] B. Andersson, G. Gustafson, B. Söderberg, "A probability measure on parton and
  string states", Nucl. Phys. B 264 (1986) 29.
- [BDMNP22] A. Beraudo, A. De Pace, M. Monteno, M. Nardi, F. Prino, "In-medium hadronization of
  heavy quarks and its effect on charmed meson and baryon distributions in heavy-ion
  collisions", Eur. Phys. J. C 82 (2022) 607, arXiv:2202.08732.
- [BGLS18] C. Bierlich, G. Gustafson, L. Lönnblad, H. Shah, "The Angantyr model for heavy-ion
  collisions in PYTHIA8", JHEP 10 (2018) 134, arXiv:1806.10820.
- [BGLS24] C. Bierlich, G. Gustafson, L. Lönnblad, H. Shah, "The dynamic hadronization of charm
  quarks in heavy-ion collisions", Eur. Phys. J. C 84 (2024) 231, arXiv:2309.12452.
- [BMW12a] A. Beraudo, J. G. Milhano, U. A. Wiedemann, "Medium-induced color flow softens
  hadronization", Phys. Rev. C 85 (2012) 031901, arXiv:1109.5025.
- [BMW12b] A. Beraudo, J. G. Milhano, U. A. Wiedemann, "The contribution of medium-modified
  color flow to jet quenching", JHEP 07 (2012) 144, arXiv:1204.4342.
- [CS15] J. R. Christiansen, P. Z. Skands, "String formation beyond leading colour", JHEP 08
  (2015) 003, arXiv:1505.01681.
- [GRS12] S. Gieseke, C. Röhr, A. Siódmok, "Colour reconnections in Herwig++", Eur. Phys. J. C
  72 (2012) 2225, arXiv:1206.0041.
- [HFK16] K. C. Han, R. J. Fries, C. M. Ko, "Jet fragmentation via recombination of parton
  showers", Phys. Rev. C 93 (2016) 045207, arXiv:1601.00708.
- [HLWZ15] Y. He, T. Luo, X.-N. Wang, Y. Zhu, "Linear Boltzmann transport for jet propagation
  in the quark-gluon plasma: Elastic processes and medium recoil", Phys. Rev. C 91 (2015)
  054908, Erratum Phys. Rev. C 97 (2018) 019902, arXiv:1503.03313.
- [JF19] J. H. Putschke, K. Kauder, E. Khalaj et al., "The JETSCAPE framework",
  arXiv:1903.07706 (preprint).
- [JS20] A. Kumar, Y. Tachibana, D. Pablos, C. Sirimanna, R. J. Fries et al. (JETSCAPE),
  "JETSCAPE framework: p+p results", Phys. Rev. C 102 (2020) 054906, arXiv:1910.05481.
- [JS25] A. Sengupta, R. J. Fries, M. Kordell II, B. Kim et al. (JETSCAPE), "Hybrid
  Hadronization -- A Study of In-Medium Hadronization of Jets", arXiv:2501.16482 (preprint).
- [KZ17] R. Kunnawalkam Elayavalli, K. C. Zapp, "Medium response in JEWEL and its impact on jet
  shape observables in heavy ion collisions", JHEP 07 (2017) 141, arXiv:1707.01539.
- [MWZ18] G. Milhano, U. A. Wiedemann, K. C. Zapp, "Sensitivity of jet substructure to
  jet-induced medium response", Phys. Lett. B 779 (2018) 409, arXiv:1707.04142.
- [SK94] T. Sjöstrand, V. A. Khoze, "On colour rearrangement in hadronic W+W− events", Z. Phys.
  C 62 (1994) 281, arXiv:hep-ph/9310242.
- [SZM25] I. Soudi, W. Zhao, A. Majumder, C. Shen, J. H. Putschke, B. Boudreaux et al.
  (JETSCAPE), "Soft-hard framework with exact four-momentum conservation for small systems",
  Phys. Rev. C 112 (2025) 014905, arXiv:2407.17443.

## Out of scope

- Hybrid hadronization for O+O, Martini, a treatment beyond large-Nc colour flow.
- The wake and the liquefier in O+O (B4), beyond noting what it would need.
- Changing LBT's kinematics or its random-number stream.
- `HardProcess::NISRShower` growing over the job (it only moves the ISR siblings' bases).
