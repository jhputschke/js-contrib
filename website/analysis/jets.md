# Jets with FastJet

This page shows how to find jets in the production hadrons with FastJet from Python: the
conventions the notebooks and analyses share, the classic (SWIG) interface with jet areas and
ρA background subtraction, embedding a jet into a background it never touched, charged and full
jets, and the array (awkward) interface. It ends with the Z+jet, Z+hadron and dijet Δη analyses
as worked examples of a complete campaign analysis. Use it once you can read hadrons
([Hadrons](hadrons.md)).

**What you need.** scikit-hep `fastjet` from PyPI, which is in the analysis venv: it ships the
FastJet library itself and offers both interfaces. For FastJet in C++ (ROOT macros, fjcontrib),
see [ROOT and C++](root.md#5-fastjet-in-c).

## 1. Why the model's events are good for jets

Every Au+Au and Pb+Pb jet event comes with its **twin without the jet**: the background leg,
the same fluid without the jet's energy (see [What a production writes](files.md#what-an-event-is)).
Per oversample k there are three events to cluster:

| event | hadrons | tests |
|---|---|---|
| **bkg** | `bulk_bg` sample k | ρ, random cones, combinatorial (fake) jets |
| **bkg + jet** | `bulk_bg` sample k + `jet_frag` sample k mod n_frag | embedding: the jet on a background it did not touch, ⟨δpT⟩ = 0 |
| **bkg + wake + jet** | `bulk_jet` sample k + the same `jet_frag` sample (`r.jet_event(e, k)`) | the full event, with the medium response |

Comparing the last two isolates what the medium response does to a jet; comparing the
reconstructed jets with the fragments alone measures the subtraction.

## 2. Conventions

The notebooks and the snippets below use one set of choices; change them deliberately:

- **Particles:** \(|\eta| < 1.5\), pT > 0.15 GeV, charged by default, clustered as **massless**
  (E = \|p\|), so that rapidity equals pseudorapidity and the ghosts cover the acceptance exactly.
- **Jets:** anti-kT, R = 0.4, \(|\eta_\text{jet}| < 1.5 - R\); active areas with explicit ghosts
  (ghost area 0.005).
- **ρ:** the median of pT/A of the kT jets (R = 0.4) in \(|\eta| < 1.5 - R\), without the two
  hardest, with `set_use_area_4vector(False)`: the scalar area that pT − ρA uses. Without it
  FastJet 3 divides by the 4-vector area's transverse part, which gives a ρ about 1 GeV larger.
- **`user_index`** tags each particle with its row in the sample; offsetting the fragment rows
  tells a jet's fragment content exactly.
- **Weights:** every event counts once, so each oversample of an event weighs 1/(n · N_events)
  with n its oversamples; multiply by σ_k/N_k for cross sections ([Hadrons](hadrons.md#5-pt-windows-and-cross-sections)).

!!! note "Ghost positions are random"
    The Python bindings cannot set FastJet's ghost seed, so the ghosts get new random jitter
    at every clustering: areas and ρ change slightly from run to run. `jet_fastjet.ipynb` §4
    measures by how much.

## 3. The classic interface, with ρA subtraction

The whole jet event of every oversample, clustered with areas, ρ from the kT jets, and the
subtracted spectrum per event:

```python
import numpy as np
import fastjet as fj
from jetscape.hadrons_h5 import CHARGED, HadronFileReader

ETA_MAX, R = 1.5, 0.4
JET_DEF  = fj.JetDefinition(fj.antikt_algorithm, R)
KT_DEF   = fj.JetDefinition(fj.kt_algorithm, R)
AREA_DEF = fj.AreaDefinition(fj.active_area_explicit_ghosts, fj.GhostedAreaSpec(ETA_MAX, 1, 0.005))
RHO_SEL  = (~fj.SelectorNHardest(2)) * fj.SelectorAbsRapMax(ETA_MAX - R)
JET_SEL  = fj.SelectorAbsRapMax(ETA_MAX - R) * fj.SelectorPtMin(10.0)
CHARGED_PID = np.array(CHARGED)

def pseudojets(ev, charged=True, pt_min=0.15):
    """Massless PseudoJets of one sample; user_index = the hadron's row in the sample."""
    E, px, py, pz = np.asarray(ev["p"], dtype=np.float64).T
    pt = np.hypot(px, py)
    eta = np.arcsinh(pz / np.maximum(pt, 1e-12))
    keep = (np.abs(eta) < ETA_MAX) & (pt > pt_min)
    if charged:
        keep &= np.isin(np.abs(ev["pid"]), CHARGED_PID)
    out = []
    for i in np.flatnonzero(keep):
        pj = fj.PseudoJet(px[i], py[i], pz[i], np.hypot(pt[i], pz[i]))
        pj.set_user_index(int(i))
        out.append(pj)
    return out

r = HadronFileReader("/path/to/campaign")
pt_sub, weight = [], []
for e in r.events():
    n = r.n_oversamples(e)
    for k in range(n):
        parts = pseudojets(r.jet_event(e, k))               # bkg + wake + jet
        cs = fj.ClusterSequenceArea(parts, JET_DEF, AREA_DEF)
        bge = fj.JetMedianBackgroundEstimator(RHO_SEL, KT_DEF, AREA_DEF)
        bge.set_use_area_4vector(False)
        bge.set_particles(parts)
        rho = bge.rho()
        for jet in fj.sorted_by_pt(JET_SEL(cs.inclusive_jets())):
            pt_sub.append(jet.pt() - rho * jet.area())
            weight.append(1.0 / (n * r.n_events))           # per event: mean over its samples
spectrum, edges = np.histogram(pt_sub, np.arange(-20, 104, 4.0), weights=weight)
```

A raw R = 0.4 jet in a 0–10 % Au+Au event carries ρπR² of background, 25–30 GeV for charged
hadrons, so the subtraction is not optional. Cost: 20 oversamples of ~7,000 hadrons (the charged
ones at \(|\eta| < 1.5\) clustered), anti-kT with areas and ρ, take 0.4 s.

**Charged or full jets.** `pseudojets(ev, charged=False)` keeps neutral hadrons too: full jets
need every hadron, which hadronizing with `--charged` would have dropped (check
`r.selection()`). The notebooks switch with `PARTICLES=charged` or `PARTICLES=full`.

## 4. Embedding: a jet on a background it did not touch

Cluster the background sample plus the event's fragments, and keep track of which constituents
are fragments through `user_index` (continuing the block above):

```python
def embedded(r, e, k):
    """bulk_bg sample k of event e's background + the event's fragments (no wake)."""
    bg = r.background_event(e, k)
    full = r.jet_event(e, k)
    frag = full["origin"] == 1                              # 0 bulk, 1 fragment
    return {"pid": np.concatenate([bg["pid"], full["pid"][frag]]),
            "p": np.concatenate([bg["p"], full["p"][frag]])}, len(bg["pid"])

ev, n_bg = embedded(r, 0, 0)
parts = pseudojets(ev)
cs = fj.ClusterSequenceArea(parts, JET_DEF, AREA_DEF)
for jet in fj.sorted_by_pt(JET_SEL(cs.inclusive_jets())):
    frag_pt = sum(c.pt() for c in jet.constituents() if c.user_index() >= n_bg)
```

The reference is the jet of the fragments alone: match the embedded jet to it, and
pT,emb − pT,ref is the subtraction's offset and resolution (δpT). Repeating the same with
`r.jet_event(e, k)`, the full event with the wake, shows what the medium response adds inside
the cone. Random cones in `r.background_event(e, k)` give the background's fluctuations.

## 5. The array interface (awkward)

One `ClusterSequence` call clusters all samples of a unit in C++, with no Python loop
(continuing the first block: `np`, `CHARGED_PID`, `r`):

```python
import awkward as ak, vector, fastjet as fj
from jetscape.hadrons_h5 import HadronFile
vector.register_awkward()

def samples(path, unit, n=None):
    """The samples of one unit as samples * var * Momentum4D (massless)."""
    with HadronFile(path) as hf:
        u = hf.unit_index(unit)
        s0, s1 = int(hf.unit_offsets[u]), int(hf.unit_offsets[u + 1])
        s1 = s1 if n is None else min(s1, s0 + n)
        so = hf.sample_offsets[s0:s1 + 1]
        p = hf.f["hadrons/p"][so[0]:so[-1]].astype(np.float64)
        pid = hf.f["hadrons/pid"][so[0]:so[-1]]
    rec = ak.zip({"px": p[:, 1], "py": p[:, 2], "pz": p[:, 3],
                  "E": np.linalg.norm(p[:, 1:], axis=1),
                  "charged": np.isin(np.abs(pid), CHARGED_PID)}, with_name="Momentum4D")
    return ak.unflatten(rec, np.diff(so))

info = r.event_info(0)
ev = samples(f"{info.stem}_hadrons_jet_frag.h5", info.local_event)      # the jet hadrons alone
ev = ak.to_packed(ev[(abs(ev.eta) < 1.5) & (ev.pt > 0.15) & ev.charged])
cs = fj.ClusterSequence(ev, fj.JetDefinition(fj.antikt_algorithm, 0.4))  # all samples at once
jets = cs.inclusive_jets(min_pt=10.0)                                     # samples * jets
cons = cs.constituent_index(min_pt=10.0)                                  # indices into ev
```

`jetscape.HadronArray.from_h5(path).to_awkward(by_unit=True)` gives the same jagged arrays
without writing `samples()` yourself ([Hadrons](hadrons.md#the-pythonic-hadronarray)).

| | supported |
|---|---|
| yes | inclusive and exclusive jets, constituents and their indices (`constituent_index` stands in for `user_index`), fjcontrib's SoftDrop, Lund declusterings, energy correlators and N-subjettiness |
| no | areas, background estimators, `Selector`s, `user_index` |

So the array interface is ideal for the jet hadrons alone, for pp, and for substructure; in the
heavy-ion background, use the classic interface for the subtraction, or add explicit ghosts as
particles by hand, which `jet_fastjet_awkward.ipynb` §6 does. A batch with explicit ghosts is
large: cluster it in chunks. 100 samples cluster in 0.2 s.

## 6. The notebooks

| notebook | what |
|---|---|
| [`jet_fastjet.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/tutorials/jet_fastjet.ipynb) | classic interface: the spectrum and fragmentation function of the jet hadrons alone; is the area right (tiling, πR², ghost size); ρ with and without the two hardest kT jets, random cones; the spectrum with and without ρA; embedding δpT for bkg + jet and bkg + wake + jet |
| [`jet_fastjet_awkward.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/tutorials/jet_fastjet_awkward.ipynb) | the same with the array interface, plus exclusive jets, SoftDrop, Lund planes, energy correlators, N-subjettiness, and areas with hand-made ghosts |

Both read `HADRON_DIR`, `MAX_EVENTS`, `MAX_OVERSAMPLES` and `PARTICLES` (`charged` or `full`)
from the environment:

```bash
HADRON_DIR=/path/to/campaign MAX_EVENTS=200 MAX_OVERSAMPLES=100 \
  jupyter nbconvert --to notebook --execute contribs/PyJetscape/example/analysis/tutorials/jet_fastjet.ipynb
```

## 7. Worked examples: the Z+jet campaign analyses

The Pb+Pb Z+jet campaigns (0–30 % Pb+Pb at 5.02 TeV, hadrons-only jobs with the Z in the event
record, a pp reference from `run_prod_vacuum.py`) are analysed against three CMS measurements.
Each is written up step by step, from the downloaded files to the figures, in Python on the HDF5
files (route A) and with C++ map steps on the ROOT files (route B, see
[ROOT and C++](root.md#6-the-zjet-analyses-in-c)). They share a design worth copying for any
campaign analysis:

- **A map step, then a final stage.** The map step reads every production file once and writes a
  small per-file summary (a "pool" of per-event histograms, or a tree of jets); the final stage
  computes everything else from those summaries in seconds to minutes. A new figure never needs
  another pass over the hadrons.
- **A ladder of steps** from the model-only ideal (using the event's twin without the jet) to an
  emulation of the experiment's procedure. The difference between two steps is what the added
  ingredient costs.
- **Jackknife errors over blocks of hydro backgrounds:** events that share a background are
  correlated, so the blocks are backgrounds, not events.
- **Weights:** the window cross section over its events, times N_coll in Pb+Pb.

!!! tip "Reading multi-GPU output"
    A campaign run on two GPUs sits in `gpu0/` and `gpu1/`. The readers do not search
    subdirectories: give Python a quoted glob, `"$DATA/gpu*/*_hadrons_jet_frag.h5"`, and
    convert both directories into one ROOT directory for C++.

### Z+jet: CMS 1702.01060

Δφ_jZ, x_jZ = pT,jet/pT,Z, ⟨x_jZ⟩ and R_jZ (jet partners per Z) with anti-kT R = 0.3 jets,
pT > 30 GeV, \(|\eta| < 1.6\). S are the Z–jet pairs of the Z's own event; B, the fake jets from
background fluctuations, is subtracted:

| step | jets of the Z event (S) | subtracted (B) |
|---|---|---|
| 1 | the fragments alone, smeared with CMS's resolution: the reference, a model quantity | none |
| 2 | the whole event, ρA-subtracted | the jets of the Z's **own** background (`bulk_bg`, sample k against sample k) |
| 3 | the whole event, ρA-subtracted | the jets of 40 `bulk_bg` samples of the 20 other backgrounds closest in N_coll: CMS's minimum-bias mixing |

Steps 2 and 3 are the model's version of the measured observable. Step 2 repeated on the
embedded event (the same fragments in `bulk_bg` sample k, no wake) gives "what the wake does to
the jets" with very small errors, since both share everything else. In short (route A):

```bash
DATA=/data/zjet_c00     # the campaign: gpu0/, gpu1/ with <stem>_hadrons_*.h5
PP=/data/zjet_pp        # the pp reference of run_prod_vacuum.py
ANA=/data/ana_c00       # the outputs (private)
cd contribs/PyJetscape/example/analysis/zjet
./get_hepdata.sh                                                    # CMS's HEPData, once
python zjet_jets_map.py "$DATA/gpu*/*_hadrons_jet_frag.h5" $ANA/jets_py/pbpb --files 0:13   # map, in slices
python zjet_jets_map.py "$PP/*_hadrons_jet_frag.h5" $ANA/jets_py/pp
python zjet_jets_qa.py --pbpb $ANA/jets_py/pbpb --pp $ANA/jets_py/pp --outdir $ANA/figures --tag _c00
python zjet_jets.py --pbpb $ANA/jets_py/pbpb --pp $ANA/jets_py/pp --outdir $ANA/figures --tag _c00 --blocks 100
```

The map step clusters every oversample (anti-kT, explicit ghosts on a fixed grid, ρ per jet from
the kT jets in a strip \(|y - y_\text{jet}| < 1\)), embeds the fragments into `bulk_bg`, throws
random cones and clusters every background sample. On the 9,600-event campaign (2 × RTX 3090 box) it took ~0.37 s
per Pb+Pb oversample in one process. Read the jet QA first (subtraction, areas, response,
geometry, fakes), then pp against CMS's pp, then steps 1 to 3. Full walk-through:
[`ANA_Zjet.md`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/zjet/ANA_Zjet.md).

### Z–hadron correlations: CMS 2507.09307

Charged hadrons around the Z in Δφ and Δy, per Z: S (the Z with the tracks of its own event)
minus B (the Z with the tracks of other Z events, 10 per Z in CMS). What remains is the
Z-correlated yield: the jet's fragments at Δφ ≈ π and the medium response, including the wake's
depletion near the Z. Here the model's twin makes the exact answer available:

| step | mode | S − B |
|---|---|---|
| 1 | factorized | (`bulk_jet` − `bulk_bg`) + fragments, sample by sample: the reference, compared with data directly |
| 2 | all | the full event against the pooled density of all events: the **closure test** of the mixing method on the model |
| 3 | cms10 | 10 partner events from 10 other backgrounds: CMS's procedure |
| 4 | strict | one Z per background, one oversample: an experiment's statistics, to compare the errors |

```bash
cd contribs/PyJetscape/example/analysis/zjet          # DATA, PP, ANA as in the Z+jet block
python qa.py map "$DATA/gpu*/*_hadrons_jet_frag.h5" $ANA/qa_py/pbpb --jobs 8      # QA, both analyses
python qa.py map "$PP/*_hadrons_jet_frag.h5" $ANA/qa_py/pp
python qa.py plot --pbpb $ANA/qa_py/pbpb --pp $ANA/qa_py/pp --outdir $ANA/figures --tag _c00
python mix.py --pbpb "$DATA/gpu*/*_hadrons_jet_frag.h5" --vacuum "$PP/*_hadrons_jet_frag.h5" \
    --pools $ANA/pools --hists '' --jobs 8 --outdir $ANA/figures --tag _c00   # all four steps
```

One `mix.py` run writes the pools (once) and computes all four steps, the checks and the
jackknife. Two details matter in the model: every hydro background has its event plane near
φ = 0 (the impact parameter is always along x), so `mix.py` rotates each background by its own
random angle (an exact symmetry; `--no-rotate` keeps them as produced); and steps 3 and 4 never
take a partner from the Z's own background, which shares its fluid. Full walk-through:
[`ANA_Zhadron.md`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/zjet/ANA_Zhadron.md);
every mode written out:
[`MIXING_MODES.md`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/zjet/MIXING_MODES.md);
every cut of the paper and its setting in production and analysis:
[`README_zjet_2507.09307.md`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/zjet/README_zjet_2507.09307.md).

### The η-separated dijet wake: CMS HIN-25-012

CMS compared dijet–hadron correlations for dijets with a large rapidity gap (R_asym) and a small
one (R_sym) to see the diffusion wake. The same analysis runs on the Z+jet campaigns with the Z
as jet1 (\(|y_Z| < 1\), pT,Z > 60 GeV) and the leading R = 0.4 jet above 30 GeV as jet2. It
builds on the Z–hadron pools and the jets at R = 0.4, and its final stage is
[`dijet/deta.py`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/dijet/deta.py):

```bash
RUNS=contribs/PyJetscape/example/analysis/runs    # data/ and figures/ per campaign (git-ignored)
OPT="--pools $RUNS/zjet_c00/data/mix/pbpb --vacuum-pools $RUNS/zjet_pp/data/mix/vacuum \
   --jets $RUNS/zjet_c00/data/jets_r04 --pp-jets $RUNS/zjet_pp/data/jets_r04 --run zjet_c00"
python contribs/PyJetscape/example/analysis/dijet/deta.py $OPT --tag _c00
```

It reports the model's truth (wake + fragments), the wake alone, the experiment-like full event,
and closure contents that must come out flat, next to CMS's HEPData. With the Z as jet1 the
comparison with CMS is a shape reference, not like for like. Full walk-through, including the
map steps at R = 0.4:
[`ANA_Deta.md`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/dijet/ANA_Deta.md).

!!! warning "Results not public yet"
    The Z+jet and dijet results are not public: write their outputs (`--outdir`, `runs/`)
    outside the repository or in its git-ignored `runs/` directory.

## Next steps

- FastJet in C++ on the ROOT files, and the C++ map steps: [ROOT and C++](root.md).
- The wake behind these jets in the fluid: [Hydro and hadrons together](hydro-hadrons.md).
- Jets in O+O, where there is no background twin: [O+O](oo.md#11-jets).
