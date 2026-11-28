#!/usr/bin/env python3
"""
analysis/OO-RHIC/hadronization/make_colorless_shift_nb.py -- writes colorless_shift_study.ipynb (source without outputs):
Colorless's close-pair shift (CPS) and the Au+Au 0-10 % R_AA of the JS chain (JS T3-20k, JS T3-78k) and the
XS-ISR chain (XS-ISR T2, XS-ISR T3, XS-ISR T3 (reuse)), each hadronized with CL+CPS, CL and iColored.

    python make_colorless_shift_nb.py
    jupyter nbconvert --to notebook --execute colorless_shift_study.ipynb --output-dir ../results --output colorless_shift_study_executed.ipynb

The notebook reads the GB10's production and re-hadronization directories (ROOT below; override with
$CSS_ROOT) and runs auau_raa.py per sample and hadronization (results/auau_raa_<tag>.json).  Model results,
not vetted.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# Colorless's close-pair shift and the Au+Au 0–10 % R_AA

*Model results, not vetted.* GB10 study of 2026-11-26/27 (JS T3-78k and XS-ISR T3 (reuse), the re-hadronizations); background in
`auau_jet_raa.ipynb` (Figs. 6–8) and the investigation note
"O+O Quenching: Tracing the Missing Energy Loss to iColored Hadronization".

## Motivation

- **O+O under-quenches.** Our O+O chain (3dMCGlauber strings + 3D MUSIC, PythiaIsrGun with iMATTER ISR, MATTER + LBT
  with the AA22 *Type 2* energy loss, iColored hadronization with the real beam remnants) gives less suppression
  than sPHENIX's O+O jets.
- **The Au+Au test.** The same chain in central Au+Au (0–10 %, b < 4.7 fm) — XS-ISR T2 (Type 2) and XS-ISR T3 (Type 3, the
  published default of arXiv:2204.01163) — gives a hadron and charged-jet R_AA about **2× the STAR/PHENIX data**,
  while **JS T3-20k** (JETSCAPE's own chain — PythiaGun, MATTER + LBT Type 3, **Colorless** hadronization — on our
  medium) reproduces the data and JETSCAPE's Fig. 14.
- **The difference is the hadronization.** Energy to the medium and the parton-level loss are the same in all of
  them; on the same partons iColored gives ~2× the high-p_T hadrons of Colorless in the medium, and the two agree
  in vacuum.
- **The diagnosis** (GB10, `icolored_diag.py`, `colorless_emul.py`, `colorless_shift_count.py`): most of the
  factor 2 is a step inside Colorless that does not conserve energy — the *close-pair shift* below. This notebook
  tests that with dedicated high-statistics runs and all hadronizations of the same partons.
""")

md(r"""
## Names used here

**Chains.**
- **JS:** JETSCAPE's chain. PythiaGun, no ISR, MATTER + LBT with recoils and holes, Colorless hadronization in
  the job (`run_prod_jet_single.py`).
- **XS-ISR:** the X-SCAPE ISR chain. PythiaIsrGun with iMATTER ISR and real beam remnants, MATTER + LBT, iColored
  hadronization in the job (`run_prod_oo_isr.py`).

Both run on the same central Au+Au medium: 3dMCGlauber strings + 3D MUSIC, RHIC tune, b < 4.7 fm.

**Energy loss.**
- **T2:** Type 2, α_s 0.25.
- **T3:** Type 3, the published default of arXiv:2204.01163: α_s 0.3, q̂A 10, q̂B 100, LBT α_s 0.3.

| sample | earlier name | chain, energy loss, hydro | events |
|---|---|---|---|
| **JS T3-20k** | run C | JS, T3, hydro reused 20× | 20k |
| **JS T3-78k** | run 1 | JS, T3, hydro reused 20×; independent seeds; parts (a), (b), (c) | 78k |
| **XS-ISR T2** | B1 | XS-ISR, T2, one hydro per event | 20k |
| **XS-ISR T3** | B2 | XS-ISR, T3, one hydro per event | 20k |
| **XS-ISR T3 (reuse)** | run 1′ | XS-ISR, T3, hydro reused 20× (local X-SCAPE opt-in `<allowReuseWithISR>`); parts (a), (b), (c) | 78k |
| **PP19** | PP19 | the JS chain's p+p reference (vacuum MATTER) | 160k + 128k (5–9 GeV) |
| **XS-ISR pp** | pp_ref | the XS-ISR chain's p+p reference | 70k + 20k (7–9 GeV) |

**Hadronizations.** All are re-hadronizations of the stored partons with 10 fragmentations per event, except
"(job)". Each Au+Au sample is compared with its p+p reference hadronized the same way.

| code | hadronization |
|---|---|
| **CL+CPS** | Colorless with the close-pair shift (CPS): the production code |
| **CL** | Colorless without the CPS (`<close_pair_shift>0`, local X-SCAPE change) |
| **iCol-R** | iColored with the real beam remnants (XS-ISR samples only) |
| **iCol-B/rec** | iColored with √s/6 beam ends, the holes' open colour lines reconnected (JS samples: no stored remnants) |
| **iCol-B/fake** | the same, the holes' open lines closed with 1 GeV fake partons along the beam |
| **(job)** | the hadrons made in the production job, one fragmentation per event: CL+CPS for JS, iCol-R for XS-ISR |

Internal tags and file names keep the earlier names (`css_run1_*`, `run1p`, `B2`, `pp_ref`, ...).
""")

md(r"""
## Why: the close-pair shift (CPS)

`ColorlessHadronization::DoHadronization` (X-SCAPE `src/hadronization/ColorlessHadronization.cc` ≈ l. 424–486;
JETSCAPE since commit b70ebb58, 2018-06-07, "Separation of partons in colorless hadronization: partons that are
too collinear are separated by 4*Lambda_QCD"; unchanged in JETSCAPE v3.0–3.7, so in the code of
arXiv:2204.01163):

```c++
// Making collinear partons not collinear
// Could have dangerous effects, yet to be tested....
for (ipart ...) for (j = ipart+1 ...)
  if (|p_ipart - p_j| < 4*Lambda_QCD)            // 0.8 GeV
     the parton with the larger |p_z|: p_z += sign(p_z) * 4*Lambda_QCD, E recomputed
```

- It acts on **every pair** of the event's partons (not only neighbours on a string), cumulatively, and always
  outward along the beam: it **adds energy** (≈ 0.8 GeV × |p_z|/|p| per firing) and drags the strings forward.
- The number of close pairs grows roughly with the **square of the number of soft partons**. In vacuum it hardly
  fires; with LBT's ≈ 100 thermal recoils (and ≈ 85 holes, hadronized in a second pass) per central Au+Au event it
  fires hundreds of times.
- **LBT's recoils, not MATTER's:** the XS-ISR T2 job with LBT removed but MATTER's own recoils on (`recoil_on 1`) has
  ≈ 6 recoils per event and the CPS changes nothing measurable (cell below).
- Without it, Colorless shows no Pythia failures (Colorless gives partons their Pythia masses, its strings end on
  33 GeV fakes or real quarks): the CPS protects against nothing in the present code.
- arXiv:2204.01163 Sec. II D describes JETSCAPE's Colorless hadronization (strings from the partons, fake
  remnants); the CPS is part of the code used there.
""")

code(r"""
import json, os, glob, subprocess, sys, textwrap
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "..")
import _paths  # noqa: F401  (analysis/OO-RHIC and its test directories on sys.path)
AUAU_RAA = str(_paths.ROOT / "auau_calibration" / "auau_raa.py")
ROOT = Path(os.environ.get("CSS_ROOT", "/home/putschke/prod"))
RH = ROOT / "AuAu-RHIC_runs/rehad"          # the re-hadronizations: RH/<sample>/<kind>/<dir>/
F3 = ROOT / "AuAu-RHIC_hadtests/from3090"   # XS-ISR T2/T3 (B1, B2), XS-ISR pp (pp_ref), PP19 as produced
RUNS = ROOT / "AuAu-RHIC_runs"              # runs 1 and 1' as produced
RUNC = ROOT / "AuAu-RHIC_jschain/js_chain_t3"
G = "*_hadrons_jet_frag.h5"
# display names (internal tags and files keep the earlier names: run1, runC, run1p, B1, B2, pp_ref)
NAME = {"run1": "JS T3-78k", "runC": "JS T3-20k", "run1p": "XS-ISR T3 (reuse)", "B2": "XS-ISR T3", "B1": "XS-ISR T2",
        "run1a": "JS T3-78k (a)"}
KIND = {"cl_shift": "CL+CPS", "cl_noshift": "CL", "icol_remn": "iCol-R", "icol_beam_reco": "iCol-B/rec",
        "icol_beam_fake": "iCol-B/fake", "injob": "(job)"}
KIND_L = {"cl_shift": "CL+CPS: Colorless with the close-pair shift", "cl_noshift": "CL: Colorless without the CPS",
          "icol_remn": "iCol-R: iColored, real remnants", "icol_beam_reco": "iCol-B/rec: iColored, √s/6 beam ends, holes reconnect",
          "icol_beam_fake": "iCol-B/fake: iColored, √s/6 beam ends, holes fake",
          "injob": "(job): as produced in the job, n_frag 1"}
# tag: (label, Au+Au glob, p+p glob)
S = {}
for k in ("cl_shift", "cl_noshift", "icol_beam_reco", "icol_beam_fake"):
    S[f"run1_{k}"] = (f"{NAME['run1']}, {KIND[k]}", f"{RH}/run1/{k}/*/{G}", f"{RH}/PP19/{k}/*/{G}")
S["run1_injob"] = (f"{NAME['run1']}, CL+CPS (job)", f"{RUNS}/run1/*/{G}", f"{F3}/pp19_icdecays*/{G}")
S["runC_injob"] = (f"{NAME['runC']}, CL+CPS (job)", f"{RUNC}/{G}", f"{F3}/pp19_icdecays*/{G}")
for smp, src in (("run1p", f"{RUNS}/run1p/*/{G}"), ("B2", f"{F3}/auau_t3_w*/{G}"), ("B1", f"{F3}/auau_med_w*/{G}")):
    S[f"{smp}_injob"] = (f"{NAME[smp]}, iCol-R (job)", src, f"{F3}/pp_ref_w*/{G}")
    for k in ("icol_remn", "cl_shift", "cl_noshift"):
        S[f"{smp}_{k}"] = (f"{NAME[smp]}, {KIND[k]}", f"{RH}/{smp}/{k}/*/{G}", f"{RH}/pp_ref/{k}/*/{G}")
# extra arms for the comparison with the earlier test (b) (CL, n_frag 1, AuAu-RHIC_hadtests/noshift)
NS = ROOT / "AuAu-RHIC_hadtests/noshift"
S["runC_cl_noshift"] = (f"{NAME['runC']}, CL", f"{RH}/runC/cl_noshift/*/{G}", f"{RH}/PP19/cl_noshift/*/{G}")
S["runC_cl_noshift_nf1"] = (f"{NAME['runC']}, CL (n_frag 1, test b)", f"{NS}/js_chain_t3/{G}", f"{NS}/pp19_icdecays/{G}")
S["B1_cl_noshift_nf1"] = (f"{NAME['B1']}, CL (n_frag 1, test b)", f"{NS}/auau_med_w*/{G}", f"{NS}/pp_ref_w*/{G}")
S["run1a_cl_noshift"] = (f"{NAME['run1a']} only (20k), CL", f"{RH}/run1/cl_noshift/a/{G}", f"{RH}/PP19/cl_noshift/*/{G}")
# the closure study: XS-ISR T3 (reuse) with the job's real remnants rebuilt before the closure (rehadronize_icolored.py
# --beam-ends stored), closure varied; p+p: XS-ISR pp iCol-R (vacuum has no holes, so no closure acts there)
CLOSURES = {"reco": "reconnect (default)", "fake": "fake", "thermal": "thermal", "rglobal": "reconnect_global"}
for cl, cn in CLOSURES.items():
    KIND[f"icol_remn_{cl}"] = f"iCol-R/{cl}"
    KIND_L[f"icol_remn_{cl}"] = f"iCol-R/{cl}: iColored, real remnants, closure {cn}"
    S[f"run1p_icol_remn_{cl}"] = (f"{NAME['run1p']}, iCol-R/{cl}", f"{RH}/run1p/icol_remn_{cl}/*/{G}", f"{RH}/pp_ref/icol_remn/*/{G}")
RES = {}
for tag, (lab, ga, gp) in S.items():
    fn = Path(f"../results/auau_raa_css_{tag}.json")
    if not glob.glob(ga) or not glob.glob(gp):
        print(f"{tag}: no files ({ga})"); continue
    if not fn.exists():
        r = subprocess.run([sys.executable, AUAU_RAA, "--family", f"css_{tag}", "-j", "12", "--auau-glob", ga, "--pp-glob", gp],
                           capture_output=True, text=True)
        if r.returncode:
            print(tag, r.stderr[-800:]); continue
    RES[tag] = json.loads(fn.read_text())
    print(f"{tag:22s} {RES[tag]['auau_events']:6d} Au+Au events ({RES[tag]['auau_files']} files), p+p {RES[tag]['pp_events_in_cut']} in the cut")
STAR = json.loads(Path("../refs/star_auau200_chjet/star_chjet_raa.json").read_text())["centrality"]
PHX = json.loads(Path("../refs/phenix_auau200_pi0/phenix_pi0_raa_0_10.json").read_text())["points"]
SH = json.loads(Path("../refs/star_auau200_hadrons/star_hadron_raa.json").read_text())
JS14 = json.loads(Path("../refs/jetscape_2204_fig14/js_fig14.json").read_text())["panels"]
def js_step(ax_, key, **kw):
    q = JS14[key]; e = [q[0]["lo"]] + [r["hi"] for r in q]; v = [r["value"] for r in q]
    ax_.stairs(v, e, baseline=None, color="C4", lw=1.8, **kw)
def model(tag, key):
    d = RES[tag]["raa"][key]; e = np.array(d["edges"])
    return 0.5 * (e[1:] + e[:-1]), 0.5 * np.diff(e), np.array(d["raa"], float), np.array(d["err"], float)
def integ(tag, key, lo, hi, which="err"):
    # which = "err": grouped per shared medium (reused hydro); "err_event": per event (if auau_raa.py wrote it)
    d = RES[tag]["raa"][key]; e = np.array(d["edges"]); a = np.array(d["auau"]); p = np.array(d["pp"])
    er = np.array(d.get(which, d["err"]))
    s = (e[:-1] >= lo - 1e-9) & (e[1:] <= hi + 1e-9)
    return a[s].sum() / p[s].sum(), np.sqrt(((er * p)[s] ** 2).sum()) / p[s].sum()
def band_mean(pts, lo, hi):
    v = [q["value"] for q in pts if q["lo"] >= lo - 1e-9 and q["hi"] <= hi + 1e-9]
    return float(np.mean(v)) if v else float("nan")
""")

md(r"""
### How often the CPS fires, and whose recoils it needs

Per event, the CPS's firings and the energy it adds in Colorless's positive pass (partons with pstat 0/1000/1;
the hole pass in brackets), from `colorless_shift_count.py` (the X-SCAPE loop replayed, validated against
X-SCAPE's Colorless). The second block: the same seeds (926000–926002, p̂_T 23–30 GeV, 150 events each) with
MATTER without recoils (T1), MATTER with its own recoils but no LBT, and MATTER + LBT (XS-ISR T2).
""")

code(r"""
import colorless_shift_count as SC
def shift_stats(files):
    rows = [r for k, f in enumerate(files) for r in SC.one((f, 7 + k))]
    R = np.array(rows); c = dict(zip(SC.COLS, R.mean(0)))
    eo = c["pos_E_in"] + c["pos_E_fake"] + c["pos_E_shift"]
    return len(R), c, 100 * c["pos_E_shift"] / eo
SAMP_SHIFT = [("PP19 (vacuum, JS p+p)", sorted(glob.glob(f"{F3}/pp19_icdecays/{G}"))[:1]),
              ("XS-ISR pp (vacuum)", sorted(glob.glob(f"{F3}/pp_ref_w23_30/{G}"))[:4]),
              (NAME["run1"], sorted(glob.glob(f"{RUNS}/run1/a/{G}"))[:10]),
              (NAME["run1p"], sorted(glob.glob(f"{RUNS}/run1p/a/{G}"))[:10]),
              (NAME["B2"], sorted(glob.glob(f"{F3}/auau_t3_w23_30/{G}"))[:10]),
              (NAME["B1"], sorted(glob.glob(f"{F3}/auau_med_w23_30/{G}"))[:10])]
print(f"{'sample':34s} {'events':>7s} {'firings':>9s} {'(holes)':>8s} {'E_partons':>10s} {'E_shift':>8s} {'share':>6s}")
SHIFT = {}
for lab, fs in SAMP_SHIFT:
    if not fs:
        continue
    n, c, share = shift_stats(fs); SHIFT[lab] = (c, share)
    print(f"{lab:34s} {n:7d} {c['pos_shifts']:9.1f} {c['neg_shifts']:8.1f} {c['pos_E_in']:10.1f} {c['pos_E_shift']:8.1f} {share:5.1f}%")
MR = ROOT / "AuAu-RHIC_hadtests/matter_recoil"
print("\nsame seeds, p̂_T 23-30 GeV:")
for lab, d in (("MATTER, recoil_on 0, no LBT (T1)", ROOT / "AuAu-RHIC_hadtests/auau_matteronly_w23_30"),
               ("MATTER, recoil_on 1, no LBT", MR / "w23_30"), ("MATTER + LBT (XS-ISR T2)", F3 / "auau_med_w23_30")):
    fs = [str(d / f"OO_isr_seed{s}_hadrons_jet_frag.h5") for s in (926000, 926001, 926002)]
    if all(Path(f).exists() for f in fs):
        n, c, share = shift_stats(fs)
        import h5py, hdf5plugin
        rec = sum(int((h5py.File(f)["partons/data"][:, 2] == 1).sum()) for f in fs) / n
        print(f"  {lab:34s} recoils/event {rec:6.1f}  firings {c['pos_shifts']:7.1f}  share {share:5.1f}%")
""")

md(r"""
## The runs

| sample | chain | energy loss | events | hydro | p+p reference |
|---|---|---|---|---|---|
| **JS T3-20k** | JS: PythiaGun (PP19 hard), MATTER + LBT, recoils + holes | Type 3 (α_s 0.3, q̂A 10, q̂B 100; LBT α_s 0.3) | 20k (2k × 10 windows) | reused 20× | PP19 (`pp19_icdecays`, 160k, 16k per window; + `pp19_icdecays_ext`, 64k each in 5–7 and 7–9) |
| **JS T3-78k** | the same as JS T3-20k | Type 3 | 78k: (a) 2k × 10 windows, (b) 6k × 7–40 GeV (reuse 21), (c) 8k each in 5–7 and 7–9 → 10k in 5–7, 16k in 7–9, 8k in 9–40, 2k in 40–80 | reused 20× (b: 21×) | PP19 |
| **XS-ISR T2** | XS-ISR: PythiaIsrGun + iMATTER ISR, real remnants, MATTER + LBT | Type 2 (α_s 0.25) | 20k (2k × 10) | one per event | XS-ISR pp (`pp_ref`, 70k; 7–9 GeV extended 5k → 25k on the RTX box) |
| **XS-ISR T3** | XS-ISR | Type 3 | 20k (2k × 10) | one per event | XS-ISR pp |
| **XS-ISR T3 (reuse)** | XS-ISR, with hydro reuse (`--reuse 20`, local X-SCAPE opt-in `<allowReuseWithISR>`) | Type 3 | 78k: (a) 2k × 10 windows, (b) 6k × 7–40 GeV, (c) 8k each in 5–7 and 7–9 → 10k in 5–7, 16k in 7–9, 8k in 9–40, 2k in 40–80 | reused 20× | XS-ISR pp |

All: central Au+Au 200 GeV, b < 4.7 fm, the RHIC-tune 3dMCGlauber strings + 3D MUSIC, hard partons |y| < 1
("any"). Every Au+Au sample is compared to **its own p+p reference hadronized the same way**:

- the JS chain (JS T3-78k): CL+CPS (production build), CL
  (`<close_pair_shift>0`, local build), iColored with √s/6 beam ends and the holes' lines closed by
  *reconnect* or *fake* (`rehadronize_icolored.py --beam-ends sqrts6`; these files have no remnants);
- the XS-ISR chain (XS-ISR T2, XS-ISR T3, XS-ISR T3 (reuse)): iCol-R (`rehadronize_jets.py --new-seeds 1`), CL+CPS and CL.

Re-hadronizations: `--n-frag 10` (ten fragmentations of the same partons per event, averaged per event) — the
in-job hadrons (`injob`) have one. The statistical errors group the events that share a reused medium
(`auau_raa.py`: `units/hydro_event`, else unit // reuse). The samples are **kept separate** (no combining of XS-ISR T3
with XS-ISR T3 (reuse) or of JS T3-20k with JS T3-78k).
""")

code(r"""
print(f"{'tag':22s} {'label':68s} {'Au+Au ev':>9s} {'files':>6s} {'p+p in cut':>11s}")
for tag, r in RES.items():
    print(f"{tag:22s} {S[tag][0]:68s} {r['auau_events']:9d} {r['auau_files']:6d} {r['pp_events_in_cut']:11d}")
""")

md(r"""
### Energy balance

Per event, the energy of the hadrons each hadronization makes (positive hadrons, all η; Colorless's hole hadrons
apart) against the partons it was handed (pstat ≥ 0; iColored's real remnants and √s/6 beam ends are extra input
by construction). Colorless *with* the CPS makes energy; without it, hadrons = partons + its fake ends.
""")

code(r"""
import h5py, hdf5plugin
def energy(fs, nmax=8):
    eh = en = ep = n = 0.0
    for f in fs[:nmax]:
        with h5py.File(f) as h:
            k = int(h.attrs["nunits_written"]); so = h["hadrons/sample_offsets"][:]; uo = h["hadrons/unit_offsets"][: k + 1]
            p = h["hadrons/p"][: so[uo[-1]]].astype(float); st = h["hadrons/pstat"][: so[uo[-1]]]
            nsamp = uo[-1]
            par = h["partons/data"][:] if "partons/data" in h else np.zeros((0, 14))
        eh += p[st >= 0, 0].sum() / max(nsamp / k, 1); en += p[st == -1, 0].sum() / max(nsamp / k, 1)
        ep += par[par[:, 2] >= 0, 3].sum(); n += k
    return eh / n, en / n, ep / n
print(f"{'tag':22s} {'E_partons':>10s} {'E_hadrons':>10s} {'E_hole_hadrons':>15s}  (GeV per event, first 8 files)")
for tag in RES:
    fs = sorted(glob.glob(S[tag][1]))
    a, b, c = energy(fs)
    print(f"{tag:22s} {c:10.1f} {a:10.1f} {b:15.1f}")
""")

md(r"""
## The spectra behind the R_AA: p+p references and Au+Au, hadronized each way

Every R_AA below divides a Au+Au spectrum by a p+p reference hadronized the same way. Both are shown here first.
- **PP19** (the JS chain's p+p: JETSCAPE p+p, vacuum MATTER, no ISR) is the reference of JS T3-78k and JS T3-20k.
- **XS-ISR pp** (`pp_ref`: the XS-ISR chain's p+p) is the reference of XS-ISR T3 (reuse), XS-ISR T2 and XS-ISR T3.
- **Au+Au:** JS T3-78k (the JS chain) and XS-ISR T3 (reuse) (the XS-ISR chain), 0–10 %, one hard scattering per event, in the
  same normalization as p+p (σ_window / N_window per event). So spectrum(Au+Au) / spectrum(p+p) is R_AA directly.
  Charged jets are shown without STAR's leading-track cut; holes are subtracted as in the R_AA.

Each figure has three rows:
- **Top:** the cross section per p_T bin in each observable's acceptance.
- **Middle:** the ratio to one scheme, which is itself not drawn there. PP19 and JS T3-78k: CL+CPS,
  JETSCAPE's choice. XS-ISR pp and XS-ISR T3 (reuse): iCol-R. The Au+Au ratio axis runs from 0 to 5,
  because the schemes differ by up to about 5×.
- **Bottom:** n_eff per bin, the effective number of equally weighted events, (Σw)² / Σw². Below about 20
  (dashed line), a few heavily weighted events from the low p̂_T windows dominate the bin and its error is
  unreliable.

The ratio errors combine the two samples' errors as if independent, using n_eff. All schemes fragment the same
partons, so the true ratio errors are smaller.

In vacuum there are no holes, so iColored's holes' closure (reconnect / fake) cannot matter: those two p+p curves
must coincide. The CPS fires 0.4–0.7 times per p+p event, so CL+CPS and CL should nearly coincide
in p+p too.

In Au+Au, Colorless subtracts the holes' hadrons (pstat −1) at hadron level, so a bin can come out negative; such
bins are left out of the top row.
""")

code(r"""
SPANELS = [("pion", "(π⁺+π⁻)/2 |η|<0.35"), ("had", "h± |η|<1"), ("chjet_R0.2", "charged jets R=0.2 |η|<0.8"),
           ("chjet_R0.4", "charged jets R=0.4 |η|<0.6"), ("fjet_R0.4", "full jets R=0.4 |η|<0.5")]
SSTY = {"cl_shift": ("C3", "o"), "cl_noshift": ("C2", "s"), "icol_remn": ("C0", "D"), "icol_beam_reco": ("C1", "^"),
        "icol_beam_fake": ("C6", "v"), "injob": ("0.4", "x")}
SNAME_K = {"run1": {"injob": "(job): CL+CPS as produced, n_frag 1"},
           "run1p": {"injob": "(job): iCol-R as produced, n_frag 1"}}
def spec(tag, key, side):
    d = RES[tag]["raa"][key]; e = np.array(d["edges"]); y = np.array(d[side], float)
    n = np.array(d.get(f"neff_{side}", np.zeros(len(y))), float)
    ey = np.divide(np.abs(y), np.sqrt(n), out=np.zeros_like(y), where=n > 0)
    return e, y / np.diff(e), ey / np.diff(e), n
def spectra_fig(title, smp, kinds, side):
    kw = dict(sharex="col", gridspec_kw={"height_ratios": [2, 1, 1]})
    fig_h, ax_h = plt.subplots(3, 2, figsize=(11, 8.6), **kw)     # hadrons: pi, h+-
    fig_j, ax_j = plt.subplots(3, 3, figsize=(16, 8.6), **kw)     # jets: charged R=0.2, R=0.4, full R=0.4
    ax = np.concatenate([ax_h, ax_j], axis=1)
    base = f"{smp}_{kinds[0]}"; bname = KIND[kinds[0]]
    for k in kinds:
        tag = f"{smp}_{k}"
        if tag not in RES or base not in RES:
            continue
        c, mk = SSTY[k]; lab = SNAME_K[smp].get(k, KIND_L[k])
        for j, (key, _) in enumerate(SPANELS):
            e, y, ey, n = spec(tag, key, side); _, yb, eyb, _ = spec(base, key, side)
            x, xe = 0.5 * (e[1:] + e[:-1]), 0.5 * np.diff(e)
            ok = y > 0
            ax[0, j].errorbar(x[ok], y[ok], ey[ok], xe[ok], fmt=mk, ms=4, color=c, label=lab)
            r = np.divide(y, yb, out=np.full_like(y, np.nan), where=yb > 0)
            er = np.abs(r) * np.sqrt(np.divide(ey, np.abs(y), out=np.zeros_like(y), where=y != 0) ** 2
                                     + np.divide(eyb, np.abs(yb), out=np.zeros_like(yb), where=yb != 0) ** 2)
            if k != kinds[0]:                       # the baseline itself is not drawn in the ratio row
                ax[1, j].errorbar(x, r, er, xe, fmt=mk, ms=4, color=c)
            ax[2, j].plot(x[n > 0], n[n > 0], mk + "-", ms=4, lw=0.8, color=c)
    for j, (key, t) in enumerate(SPANELS):
        ax[0, j].set_yscale("log"); ax[0, j].set_title(t, fontsize=10)
        ax[1, j].axhline(1, color="k", lw=0.8)
        ax[1, j].set_ylim(*((0, 5) if side == "auau" else (0.5, 1.5)))   # ratios: linear scale (Au+Au: schemes up to ~5x apart)
        ax[2, j].set_yscale("log"); ax[2, j].axhline(20, color="k", lw=0.8, ls="--"); ax[2, j].set_xlabel(r"$p_T$ [GeV]")
    for c0 in (0, 2):                                         # the first column of each figure
        ax[0, c0].set_ylabel(r"d$\sigma$/d$p_T$ in the acceptance [mb/GeV]")
        ax[1, c0].set_ylabel(f"ratio to {bname}", fontsize=8)
        ax[2, c0].set_ylabel("n_eff per bin", fontsize=9)
        ax[2, c0].plot([], [], "k--", lw=0.8, label="n_eff = 20: errors unreliable below"); ax[2, c0].legend(fontsize=7, loc="upper left")
        ax[0, c0].legend(fontsize=7, loc="lower left", title="hadronization (same markers below)", title_fontsize=7)
    num, rest = title.split(". ", 1)
    for f_, part in ((fig_h, "hadrons"), (fig_j, "jets")):
        f_.suptitle(f"{num} ({part}). {rest} — hadronized each way (model, not vetted)", fontsize=11); f_.tight_layout()
    plt.show()
    print(title.split(".")[0] + f": {side} ratio to {bname}, integrated")
    for k in kinds[1:]:
        tag = f"{smp}_{k}"
        if tag in RES and base in RES:
            v = []
            for key, lo, hi in (("pion", 8, 20), ("had", 10, 25), ("chjet_R0.2", 16, 30), ("chjet_R0.4", 16, 30), ("fjet_R0.4", 15, 50)):
                d, db = RES[tag]["raa"][key], RES[base]["raa"][key]; e = np.array(d["edges"])
                s = (e[:-1] >= lo - 1e-9) & (e[1:] <= hi + 1e-9)
                v.append(f"{key} {lo}-{hi} {np.sum(np.array(d[side])[s]) / np.sum(np.array(db[side])[s]):.3f}")
            print(f"  {KIND[k] if k != 'injob' else SNAME_K[smp][k]:36s} " + "  ".join(v))
spectra_fig("Figure 1a. PP19 p+p, the reference of JS T3-78k and JS T3-20k", "run1",
            ("cl_shift", "cl_noshift", "icol_beam_reco", "icol_beam_fake", "injob"), "pp")
spectra_fig("Figure 1b. XS-ISR pp, the reference of XS-ISR T3 (reuse), XS-ISR T3 and XS-ISR T2", "run1p",
            ("icol_remn", "cl_shift", "cl_noshift", "injob"), "pp")
""")

code(r"""
spectra_fig("Figure 1c. JS T3-78k, Au+Au 0–10 %", "run1",
            ("cl_shift", "cl_noshift", "icol_beam_reco", "icol_beam_fake", "injob"), "auau")
spectra_fig("Figure 1d. XS-ISR T3 (reuse), Au+Au 0–10 %", "run1p",
            ("icol_remn", "cl_shift", "cl_noshift", "injob"), "auau")
""")

md(r"""
## Results: R_AA of the JS chain (JS T3-78k) and of the XS-ISR chain (XS-ISR T3 (reuse))

Each panel: data (PHENIX π⁰ 0–10 %, STAR π± 0–12 %, STAR (h⁺+h⁻)/2 0–5 % / 10–20 %, STAR charged jets 0–10 %),
JETSCAPE's published Type 3 curve (arXiv:2204.01163 Fig. 14, purple steps), and the model with each
hadronization against its own p+p reference. Charged jets: STAR's definition (p_T,lead > 5 GeV on the Au+Au jets),
holes + partners subtracted.
""")

code(r"""
STY = {"cl_shift": ("C3", "o", ":"), "cl_noshift": ("C2", "s", "-"), "icol_remn": ("C0", "D", "-"),
       "icol_beam_reco": ("C1", "^", "-"), "icol_beam_fake": ("C6", "v", "--"), "injob": ("0.4", "x", ":")}
def data_panels(ax):
    for q in PHX:
        ax[0].add_patch(plt.Rectangle((q["lo"], q["value"] - q["syst_B"]), q["hi"] - q["lo"], 2 * q["syst_B"], color="0.6", alpha=0.35, lw=0))
    ax[0].errorbar([q["pt"] for q in PHX], [q["value"] for q in PHX], [q["stat"] for q in PHX], fmt="o", ms=4, color="k", label="PHENIX π⁰ 0-10 % (stat.)")
    ax[0].add_patch(plt.Rectangle((0, 0), 0, 0, color="0.6", alpha=0.35, lw=0, label="PHENIX π⁰ syst. (type B)"))
    for q in SH["pion"]["points"]:
        if q["hi"] > 4.5:
            ax[0].errorbar(0.5 * (q["lo"] + q["hi"]), q["value"], q["stat"], fmt="*", ms=9, color="navy")
    ax[0].plot([], [], "*", ms=9, color="navy", label="STAR π± 0-12 %")
    js_step(ax[0], "pion", label="JETSCAPE Fig. 14 (Type 3)")
    for c_, mfc in (("0-5%", "navy"), ("10-20%", "white")):
        pts = [q for q in SH["hadron"]["centrality"][c_] if q["hi"] > 4.5]
        ax[1].errorbar([0.5 * (q["lo"] + q["hi"]) for q in pts], [q["value"] for q in pts], [q["stat_syst"] for q in pts],
                       fmt="D", ms=5, color="navy", mfc=mfc, label=f"STAR h± {c_}")
    for j, R in ((2, 0.2), (3, 0.4)):
        for q in STAR["0-10"][f"R{R}"]:
            ax[j].plot(0.5 * (q["lo"] + q["hi"]), q["value"], "*", ms=10, color="navy", mfc="navy" if q["kind"] == "unbiased" else "white")
        ax[j].plot([], [], "*", ms=10, color="navy", label="STAR 0-10 %, unbiased")
        ax[j].plot([], [], "*", ms=10, color="navy", mfc="white", label="STAR 0-10 %, biased (lead. track, 2006.00582)")
        js_step(ax[j], f"chjet_R{R}", label="JETSCAPE Fig. 14 (Type 3)")
def raa_figure(smp, kinds, title):
    fig_h, ax_h = plt.subplots(1, 2, figsize=(12, 4.8))      # hadrons: pi, h+-
    fig_j, ax_j = plt.subplots(1, 3, figsize=(17, 4.8))      # jets: charged R=0.2, R=0.4, full R=0.4
    ax = [*ax_h, *ax_j]
    data_panels(ax)
    for k in kinds:
        tag = f"{smp}_{k}"
        if tag not in RES:
            continue
        c, mk, ls = STY[k]; lab = KIND_L[k] if k != "injob" else SNAME_K[smp]["injob"]
        for j, key in ((0, "pion"), (1, "had_star"), (2, "chjet_R0.2_star"), (3, "chjet_R0.4_star"), (4, "fjet_R0.4")):
            x, xe, r, e = model(tag, key)
            ax[j].errorbar(x, r, e, xe, fmt=mk + ls, ms=4, color=c, label=lab)
    for a_, t, xl in zip(ax, ("pions |η|<0.35 (model)", "(h⁺+h⁻)/2 |η|<0.5", "charged jets R=0.2 (STAR def.)",
                              "charged jets R=0.4 (STAR def.)", "full jets R=0.4 |η|<0.5 (model only)"),
                         ((4.5, 20.5), (4.5, 15.5), (7, 32), (7, 32), (9, 50))):
        a_.set_title(t, fontsize=10); a_.set_xlim(*xl); a_.set_ylim(0, 1.6); a_.axhline(1, color="k", lw=0.8)
        a_.set_xlabel(r"$p_T$ [GeV]")
    ax[0].set_ylabel(r"$R_{AA}$"); ax[2].set_ylabel(r"$R_{AA}$")
    ax[0].legend(fontsize=7, loc="upper left"); ax[3].legend(fontsize=7, loc="upper left")
    ax[1].legend(handles=[h for h in ax[1].get_legend_handles_labels()[0][:2]], fontsize=7, loc="upper right")   # STAR h± 0-5 % (filled) / 10-20 % (open)
    ax[2].legend(handles=[h for h in ax[2].get_legend_handles_labels()[0][:3]], fontsize=7, loc="upper left")    # STAR jets filled / open, JETSCAPE
    ax[4].legend(fontsize=7, loc="upper right")                                                                  # the model curves
    num, rest = title.split(". ", 1)
    for f_, part in ((fig_h, "hadrons"), (fig_j, "jets")):
        f_.suptitle(f"{num} ({part}). {rest}", fontsize=11); f_.tight_layout()
    plt.show()
raa_figure("run1", ("cl_shift", "cl_noshift", "icol_beam_reco", "icol_beam_fake"),
           "Figure 2a. JS T3-78k: the same partons hadronized four ways, each against PP19 hadronized the same way (model, not vetted)")
raa_figure("run1p", ("icol_remn", "cl_shift", "cl_noshift", "injob"),
           "Figure 2b. XS-ISR T3 (reuse): iCol-R, CL+CPS, CL and (job), each against XS-ISR pp hadronized the same way (model, not vetted)")
""")

md(r"""
### The two chains hadronized the same way: XS-ISR T3 (reuse) vs JS T3-78k

The same energy loss (T3), the same medium and the same hydro reuse; only the hard process and initial state
differ (PythiaGun vs PythiaIsrGun + iMATTER ISR with real remnants), and each R_AA is against its own chain's p+p
reference hadronized the same way.
- Both chains are hadronized with CL+CPS and with CL. If the chains agree there, the differences between the R_AA
  figures above come from the hadronization alone.
- iCol-R, iColored with the real remnants, exists only for the XS-ISR chain: the JS chain stores no remnants. It is
  shown for reference, without a ratio.

Top: R_AA. Bottom: XS-ISR T3 (reuse) / JS T3-78k for the same hadronization.
""")

code(r"""
CMP = [("cl_shift", "C3", "o"), ("cl_noshift", "C2", "s")]
def cmp_fig(keys, titles, xlims, fig_title):
    fig, ax = plt.subplots(2, len(keys), figsize=(6 * len(keys), 6.4), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    for j, key in enumerate(keys):
        for k, c, mk in CMP:
            tj, tx = f"run1_{k}", f"run1p_{k}"
            if tj not in RES or tx not in RES:
                continue
            xj, xej, rj, ej = model(tj, key); xx, xex, rx, ex = model(tx, key)
            ax[0, j].errorbar(xj, rj, ej, xej, fmt=mk + "-", ms=5, color=c, label=f"JS T3-78k, {KIND_L[k]}")
            ax[0, j].errorbar(xx, rx, ex, xex, fmt=mk + "--", ms=5, color=c, mfc="white", label=f"XS-ISR T3 (reuse), {KIND_L[k]}")
            r = np.divide(rx, rj, out=np.full_like(rx, np.nan), where=rj > 0)
            er = np.abs(r) * np.hypot(np.divide(ex, rx, out=np.zeros_like(rx), where=rx != 0), np.divide(ej, rj, out=np.zeros_like(rj), where=rj != 0))
            ax[1, j].errorbar(xx, r, er, xex, fmt=mk, ms=5, color=c, mfc="white", label=f"XS-ISR / JS, {KIND[k]}")
        if "run1p_icol_remn" in RES:
            x, xe, r, e = model("run1p_icol_remn", key)
            ax[0, j].errorbar(x, r, e, xe, fmt="D--", ms=5, color="C0", mfc="white", label=f"XS-ISR T3 (reuse), {KIND_L['icol_remn']} (XS-ISR only)")
        ax[0, j].set_title(titles[j], fontsize=10); ax[0, j].set_ylim(0, 1.6); ax[0, j].axhline(1, color="k", lw=0.8)
        ax[1, j].set_ylim(0, 2); ax[1, j].axhline(1, color="k", lw=0.8); ax[1, j].set_xlim(*xlims[j]); ax[1, j].set_xlabel(r"$p_T$ [GeV]")
    ax[0, 0].set_ylabel(r"$R_{AA}$"); ax[1, 0].set_ylabel("XS-ISR T3 (reuse) / JS T3-78k", fontsize=8)
    ax[0, 0].legend(fontsize=7, loc="upper right"); ax[1, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(fig_title, fontsize=11); fig.tight_layout(); plt.show()
cmp_fig(("pion", "had_star"), ("pions |η|<0.35", "(h⁺+h⁻)/2 |η|<0.5"), ((4.5, 20.5), (4.5, 15.5)),
        "Figure 2c (hadrons). XS-ISR T3 (reuse) vs JS T3-78k, both hadronized with CL+CPS and with CL (model, not vetted)")
cmp_fig(("chjet_R0.2_star", "chjet_R0.4_star", "fjet_R0.4"), ("charged jets R=0.2 (STAR def.)", "charged jets R=0.4 (STAR def.)",
        "full jets R=0.4 |η|<0.5"), ((7, 32), (7, 32), (9, 50)),
        "Figure 2c (jets). XS-ISR T3 (reuse) vs JS T3-78k, both hadronized with CL+CPS and with CL (model, not vetted)")
print("integrated, XS-ISR T3 (reuse) / JS T3-78k:")
for k, _, _ in CMP:
    v = []
    for key, lo, hi, lab in (("pion", 8, 20, "π 8-20"), ("had_star", 8, 15, "h± 8-15"), ("had", 10, 25, "h± |η|<1 10-25"),
                             ("chjet_R0.2_star", 16, 30, "jet R0.2 16-30"), ("chjet_R0.4_star", 16, 30, "jet R0.4 16-30"),
                             ("fjet_R0.4", 15, 50, "full jet 15-50")):
        a_, b_ = integ(f"run1p_{k}", key, lo, hi), integ(f"run1_{k}", key, lo, hi)
        v.append(f"{lab} {a_[0] / b_[0]:.2f}±{np.hypot(a_[1] / a_[0], b_[1] / b_[0]) * a_[0] / b_[0]:.2f}")
    print(f"  {KIND[k]:7s} " + "  ".join(v))
""")

md(r"""
**Reading** (model, not vetted; integrated ratios printed above).
- **CL+CPS: the two chains agree.** XS-ISR T3 (reuse) / JS T3-78k is 0.93–1.07 for every observable, within
  1–1.5 σ of 1. The parton level agrees too: holes 82 vs 79 per event, recoils 97 vs 94 (5 files each). So the
  medium, the energy loss and the hard process give the same suppression in both chains once the hadronization is
  the same. Within statistics, the differences between Figs. 2a and 2b come from the hadronization.
- **CL: the chains differ for hadrons.** The ratio is 0.72–0.84 for hadrons, 0.84 for jets R0.2, and 0.93–0.97
  for jets R0.4 and full jets.
  - Without the CPS, Colorless is sensitive to how the partons are made: ISR or not, and how its fake string ends
    sit relative to the partons.
  - The JS chain has the upward feeding from the low p̂_T windows (see "Contribution from thinly sampled
    windows"); the XS-ISR chain's 5–7 GeV window puts no hadron above 8 GeV.
  - So CL is no chain-independent reference either, unlike CL+CPS, where the CPS's energy injection dominates.
- **iCol-R** is available only in the XS-ISR chain, because the JS chain stores no remnants. That is the one
  hadronization the comparison cannot test across chains. Its closure systematic is studied in Fig. 4.
""")

md(r"""
### All samples, integrated

R_AA integrated over the comparison ranges (model statistical errors with the reuse grouping), with the data's
mean over the same range.
""")

code(r"""
OBS = [("pion", 8, 20, "π 8-20"), ("had_star", 8, 15, "h± 8-15"), ("had", 10, 25, "h± |η|<1 10-25"),
       ("chjet_R0.2_star", 16, 30, "jet R0.2 16-30"), ("chjet_R0.4_star", 16, 30, "jet R0.4 16-30"), ("fjet_R0.4", 15, 50, "full jet 15-50")]
print(f"{'':62s}" + "".join(f"{o[3]:>17s}" for o in OBS))
print(f"{'data (PHENIX pi0 / STAR h 0-5% / - / STAR jets)':62s}" + f"{band_mean(PHX, 8, 20):17.2f}"
      + f"{band_mean([q for q in SH['hadron']['centrality']['0-5%']], 8, 15):17.2f}{'':>17s}"
      + f"{band_mean(STAR['0-10']['R0.2'], 16, 30):17.2f}{band_mean(STAR['0-10']['R0.4'], 16, 30):17.2f}")
TAB = {}
for tag in RES:
    vals = [integ(tag, k, lo, hi) for k, lo, hi, _ in OBS]; TAB[tag] = vals
    print(f"{S[tag][0][:62]:62s}" + "".join(f"{'%.2f ± %.2f' % v:>17s}" for v in vals))
REUSED = [t for t in RES if t.startswith(("run1", "runC"))]
if any("err_event" in RES[t]["raa"]["had"] for t in REUSED):
    print("\nsamples with hydro reuse: error per shared medium (grouped) / per event, integrated ranges")
    for tag in REUSED:
        if "err_event" not in RES[tag]["raa"]["had"]:
            continue
        g = [integ(tag, k, lo, hi)[1] for k, lo, hi, _ in OBS]; ev = [integ(tag, k, lo, hi, "err_event")[1] for k, lo, hi, _ in OBS]
        print(f"{S[tag][0][:62]:62s}" + "".join(f"{'%.3f/%.3f=%.2f' % (a_, b_, a_ / b_) if b_ > 0 else '-':>17s}" for a_, b_ in zip(g, ev)))

# effective number of events n_eff = (sum w)^2 / sum w^2 (auau_raa.py, per bin); over a range from the bins' yields,
# (sum y)^2 / sum y_b^2 / n_eff,b (bins taken as independent), and the smallest bin's n_eff in the range
def neff_int(tag, key, lo, hi, side):
    d = RES[tag]["raa"][key]; e = np.array(d["edges"]); y = np.array(d[side], float); n = np.array(d.get(f"neff_{side}", []), float)
    if not len(n):
        return np.nan, np.nan
    s = (e[:-1] >= lo - 1e-9) & (e[1:] <= hi + 1e-9) & (n > 0)
    return y[s].sum() ** 2 / (y[s] ** 2 / n[s]).sum(), n[s].min()
NEFF_OBS = [("had", 10, 25, "h± 10-25"), ("pion", 8, 20, "π 8-20")]
print("\neffective events n_eff, integrated range (smallest bin in brackets); Au+Au | p+p")
print(f"{'':62s}" + "".join(f"{o[3] + ' AA':>20s}{o[3] + ' pp':>20s}" for o in NEFF_OBS))
NEFF = {}
for tag in RES:
    v = [neff_int(tag, k, lo, hi, side) for k, lo, hi, _ in NEFF_OBS for side in ("auau", "pp")]; NEFF[tag] = v
    print(f"{S[tag][0][:62]:62s}" + "".join(f"{'%.0f (%.0f)' % x:>20s}" for x in v))
low = sorted({(t, o[3]) for t in NEFF for o, x in zip([o for o in NEFF_OBS for _ in (0, 1)], NEFF[t]) if x[1] < 20})
print("bins with n_eff < 20 (unreliable errors) in:", ", ".join(f"{t} {o}" for t, o in low) if low else "none")
""")

md(r"""
### n_eff before and after the low-p̂_T extensions

The hadron and π statistics are limited by the low p̂_T windows, where few events carry large weights. Three
extensions add events there:
- **JS T3-78k (c):** 8k events each in 5–7 and 7–9 GeV. JS T3-78k had 2k and 8k.
- **PP19 extension:** +64k events each in 5–7 and 7–9 GeV. PP19 had 16k per window. Same job XML, seeds
  998001–998016, `pp19_icdecays_ext`.
- **XS-ISR T3 (reuse) (c):** 8k events each in 5–7 and 7–9 GeV, the same job settings (seeds 969001–969080).
- **XS-ISR pp extension:** +20k events in 7–9 GeV, 25k instead of 5k. Run on the RTX box with the same container and
  launcher as phase 3, seeds 881010–881049.

All three are re-hadronized the same four or three ways as the samples they extend. The table compares n_eff
(integrated, with the smallest bin in brackets) before and after, for every sample whose Au+Au or p+p changed.
The "before" files are the R_AA results kept in `results/neff_before/`; that directory is machine-local, and the
cell is skipped without it.
""")

code(r"""
NB_DIR = Path("../results/neff_before")
def neff_of(R, key, lo, hi, side):
    d = R["raa"][key]; e = np.array(d["edges"]); y = np.array(d[side], float); n = np.array(d.get(f"neff_{side}", []), float)
    if not len(n):
        return np.nan, np.nan
    s = (e[:-1] >= lo - 1e-9) & (e[1:] <= hi + 1e-9) & (n > 0)
    return y[s].sum() ** 2 / (y[s] ** 2 / n[s]).sum(), n[s].min()
if NB_DIR.exists():
    print("n_eff before -> after: integrated (smallest bin)")
    print(f"{'':52s}" + "".join(f"{o[3] + ' ' + sd:>26s}" for o in NEFF_OBS for sd in ("AA", "pp")))
    for tag in RES:
        f = NB_DIR / f"auau_raa_css_{tag}.json"
        if not f.exists():
            continue
        old = json.loads(f.read_text())
        cells = []
        for k, lo, hi, _ in NEFF_OBS:
            for side in ("auau", "pp"):
                a_, b_ = neff_of(old, k, lo, hi, side), neff_of(RES[tag], k, lo, hi, side)
                cells.append(f"{a_[0]:.0f} ({a_[1]:.0f}) -> {b_[0]:.0f} ({b_[1]:.0f})")
        print(f"{S[tag][0][:52]:52s}" + "".join(f"{c:>26s}" for c in cells))
    print("\nintegrated R_AA before -> after (same ranges)")
    for tag in [t for t in RES if t != "_old"]:
        f = NB_DIR / f"auau_raa_css_{tag}.json"
        if not f.exists():
            continue
        old = json.loads(f.read_text()); RES["_old"] = old
        v = [f"{integ('_old', k, lo, hi)[0]:.2f}±{integ('_old', k, lo, hi)[1]:.2f} -> {integ(tag, k, lo, hi)[0]:.2f}±{integ(tag, k, lo, hi)[1]:.2f}"
             for k, lo, hi, _ in NEFF_OBS]
        print(f"{S[tag][0][:52]:52s}" + "".join(f"{x:>28s}" for x in v))
    RES.pop("_old", None)
else:
    print("../results/neff_before not found: before/after comparison skipped")
""")

md(r"""
### Contribution from thinly sampled windows (< 5 p+p entries)

For completeness, every sample is analysed a second time with `auau_raa.py --min-entries 5`. A p̂_T window then
enters a bin only where the p+p reference has at least 5 events with entries there. The same windows are used in
Au+Au and p+p, and only the p+p entries are counted: counting the suppressed Au+Au entries would bias R_AA upward.

The table shows R_AA (all windows, the default) and the change when the thinly sampled windows are left out.
- **What the change means.** The events left out are real yield. The change measures how much the thinly
  sampled windows contribute; it is not an uncertainty of the method. It must not be subtracted, nor added to the
  errors as a systematic.
- **Where it is large,** the windows concerned need more events to pin their contribution down.
- **The main case is CL h± 10–25 in JS T3-78k:** Colorless without the CPS feeds 10–12 GeV hadrons up from
  p̂_T 5–7 GeV, at about 8× the p+p rate. There are 4 events on each side, so that window's R_AA of about 7 is
  uncertain to roughly ±70 %. This is a model feature of CL, not a stitching problem.
""")

code(r"""
MIN = 5
RES_MIN = {}
for tag, (lab, ga, gp) in S.items():
    if tag not in RES:
        continue
    fn = Path(f"../results/auau_raa_css_{tag}_min{MIN}.json")
    if not fn.exists():
        r = subprocess.run([sys.executable, AUAU_RAA, "--family", f"css_{tag}", "-j", "12", "--auau-glob", ga,
                            "--pp-glob", gp, "--min-entries", str(MIN)], capture_output=True, text=True)
        if r.returncode:
            print(tag, r.stderr[-400:]); continue
    RES_MIN[tag] = json.loads(fn.read_text())
def integ_min(tag, key, lo, hi):
    RES["_min"] = RES_MIN[tag]
    try:
        return integ("_min", key, lo, hi)
    finally:
        RES.pop("_min", None)
def dropped(tag, key, lo, hi):
    d = RES_MIN[tag]["raa"][key]; e = np.array(d["edges"]); s = (e[:-1] >= lo - 1e-9) & (e[1:] <= hi + 1e-9)
    sa = np.array(d.get("dropped_share_auau", np.zeros(len(e) - 1)), float); a = np.array(RES[tag]["raa"][key]["auau"], float)
    return float((sa * a)[s].sum() / a[s].sum()) if a[s].sum() else 0.0
print(f"R_AA, all windows -> without the thinly sampled windows (< {MIN} p+p entries); [Au+Au yield share left out]")
print(f"{'':52s}" + "".join(f"{o[3]:>30s}" for o in OBS))
for tag in [t for t in RES if t in RES_MIN]:
    cells = []
    for k, lo, hi, _ in OBS:
        a_, b_ = integ(tag, k, lo, hi), integ_min(tag, k, lo, hi)
        cells.append(f"{a_[0]:.2f} -> {b_[0]:.2f} ({b_[0] - a_[0]:+.2f}) [{100 * dropped(tag, k, lo, hi):.0f}%]")
    print(f"{S[tag][0][:52]:52s}" + "".join(f"{c:>30s}" for c in cells))
""")

md(r"""
**n_eff.** Per bin, n_eff = (Σw)² / Σw² is the number of equally weighted events that would give the same
statistical precision (`auau_raa.py` ≥ 7a345bb). Where a bin has n_eff below ~20, a few heavily weighted events
dominate it, and its error is unreliable, typically too small. This happens in the low p̂_T windows' tails, the
case behind the JS T3-20k vs JS T3-78k gap. The integrated n_eff above treats the bins as independent; the number in
brackets is the smallest bin's n_eff.
""")

md(r"""
### CL (no CPS): R_AA above 1, and the earlier window h± 10–25

Without the CPS, JS T3-78k gives π 8–20 1.10 and h± 8–15 1.14. The earlier test (b) on JS T3-20k's partons gave
h± (|η| < 1) 10–25 = 0.69. The table below puts all arms of JS T3-78k, JS T3-20k and XS-ISR T2 in that earlier window, and
splits it into a low end (h± 4–10, π 8–10) and a high end (h± 10–15, 15–25, π 10–20).

JS T3-78k (b) and (c) alone do not cover all windows, so only JS T3-78k (a) is shown as a 20k sample of the same size as JS T3-20k.
`auau_raa.py` matches windows by their edges, so the combined JS T3-78k is complete.
""")

code(r"""
ARMS = ["runC_injob", "runC_cl_noshift_nf1", "runC_cl_noshift",
        "run1_injob", "run1_cl_shift", "run1_cl_noshift", "run1a_cl_noshift", "run1_icol_beam_reco", "run1_icol_beam_fake",
        "B1_injob", "B1_icol_remn", "B1_cl_shift", "B1_cl_noshift", "B1_cl_noshift_nf1"]
WIN = [("had", 10, 25, "h± 10-25"), ("had", 4, 10, "h± 4-10"), ("had", 10, 15, "h± 10-15"), ("had", 15, 25, "h± 15-25"),
       ("pion", 8, 10, "π 8-10"), ("pion", 10, 20, "π 10-20")]
print(f"{'':58s} {'events':>7s}" + "".join(f"{w[3]:>14s}" for w in WIN))
for t in ARMS:
    if t not in RES:
        continue
    print(f"{S[t][0][:58]:58s} {RES[t]['auau_events']:7d}" + "".join(f"{'%.2f ± %.2f' % integ(t, k, lo, hi):>14s}" for k, lo, hi, _ in WIN))
print("\nper bin, h± |η| < 1, CL (n_frag 10): JS T3-20k vs JS T3-78k, difference in σ")
a_, b_ = RES["runC_cl_noshift"]["raa"]["had"], RES["run1_cl_noshift"]["raa"]["had"]
for i, (lo, hi) in enumerate(zip(a_["edges"][:-1], a_["edges"][1:])):
    if lo >= 5:
        d = (b_["raa"][i] - a_["raa"][i]) / np.hypot(a_["err"][i], b_["err"][i])
        print(f"  {lo:5.1f}-{hi:5.1f}  JS T3-20k {a_['raa'][i]:.2f} ± {a_['err'][i]:.2f}   JS T3-78k {b_['raa'][i]:.2f} ± {b_['err'][i]:.2f}   {d:+.1f} σ")
""")

md(r"""
**Finding: R_AA > 1 without the CPS comes from the low end** (model, not vetted).
- Without the CPS, R_AA is above 1 below about 10 GeV in every sample and every sample size: h± 5–8 GeV
  1.3–1.4 and π 5–8.5 GeV 1.1–1.9, in JS T3-20k and JS T3-78k alike, and h± 4–10 1.1–1.2 in XS-ISR T2 too. We attribute them to
  recoil and medium-response hadrons (the open "low-p_T R_AA ≈ 1 in JS T3-20k" item). With the CPS, the same bins are at
  about 0.2–0.3.
- The π 8–20 (1.10) and h± 8–15 (1.14) windows of JS T3-78k start inside this region.
- Above 12 GeV, every sample without the CPS is below 1: h± per bin JS T3-20k 0.69–0.86 and JS T3-78k 0.71–0.89.
  JS T3-78k's 10–12 GeV bin is 1.11 ± 0.09, fed upward from p̂_T 5–7 GeV (see "Contribution from thinly sampled
  windows"). π 10–20 is 0.74 ± 0.07 (JS T3-20k) and 0.96 ± 0.06 (JS T3-78k). So this is not a problem of the new
  samples.
- **JS T3-20k vs JS T3-78k in the earlier window h± 10–25:** 0.69 ± 0.09 (n_frag 1) and 0.79 ± 0.05 (n_frag 10) against
  1.02 ± 0.06 (JS T3-78k).
  - n_frag is not the cause.
  - The remaining difference sits mostly in one bin (10–12 GeV, 3.0 σ). The other bins are within 1.3 σ, and
    JS T3-78k (a), 20k like JS T3-20k, gives 0.96 ± 0.08.
  - With the CPS the two samples agree (0.21 vs 0.24), and so do jets and π without it. The per-window diagnostics below
    trace the h± 10–25 gap to the low p̂_T windows (5–7, 7–9 and 9–11 GeV).
""")

md(r"""
### JS T3-20k vs JS T3-78k without the CPS: per-window diagnostics

The h± (|η| < 1) 10–25 gap without the CPS is 0.79 ± 0.05 (JS T3-20k) vs 1.02 ± 0.06 (JS T3-78k), both with n_frag 10 and
the same p+p reference. `css_gap_diag.py` checks it from the stored files only:
- **(0) Stitching.** `auau_raa.py` has no minimum-entries rule; every window enters every bin with σ_k/N_k. The
  script lists, per p̂_T window, the events contributing to each h± bin from 8 to 25 GeV and the window's weighted
  share of the bin's yield, for Au+Au and p+p. It also recomputes R_AA with oo_analysis_rhic's rule (a window
  enters a bin only with at least 10 entries, in Au+Au and in p+p alike).
- **(2) Parton level per window,** from the produced files: the leading final jet parton, its ratio to the hard
  parton, the jet partons' energy, and the recoils and holes per event.
- **(3) Which windows carry the gap.** With the same p+p reference, window k contributes
  (AA_k(JS T3-78k) − AA_k(JS T3-20k)) / pp to the gap, and the contributions add up to it.

Both arms are shown: without the CPS, and with it. For the matched arm with the CPS, JS T3-20k was
re-hadronized with the CPS at n_frag 10 (100 files, all ok).

**(1) Settings.**
- JS T3-20k's and JS T3-78k (a)'s job XMLs are identical apart from the seed. JS T3-78k (b) differs only in nEvents (294),
  nReuseHydro (21) and pTHatBins (the 7 windows 7–40).
- The job summaries agree in every setting: b_max 4.7, parton |y| < 1 "any", the hadronization XML and the grid.
- Same driver: commit bcac3d4 (10:22) predates JS T3-20k (10:35). No `.so` in the build changed after 26 Nov 10:00.
- p+p references:
  - n_frag 10 arms: PP19 re-hadronized the same way (rehad/PP19/<kind>, 8 files, 16k events per window);
  - test (b): noshift/pp19_icdecays;
  - in-job arms: from3090/pp19_icdecays plus its 5–7 / 7–9 GeV extension pp19_icdecays_ext (+64k each).
""")

code(r"""
GF = Path("../results/css_gap_diag.json")
if not GF.exists():
    subprocess.run([sys.executable, "css_gap_diag.py", "-j", "12"], check=True, capture_output=True)
GD = json.loads(GF.read_text())
for arm in ("noshift", "shift"):
    o = GD[arm]; W = o["windows"]
    print(f"===== {'CL (no CPS)' if arm == 'noshift' else 'CL+CPS'}")
    for k, lab in (("runC", "JS T3-20k"), ("run1", "JS T3-78k"), ("pp", "PP19")):
        print(f"{lab + ' events':>27s}" + "".join(f"{x:8d}" for x in o["events"][k]))
    print(f"{'window':>27s}" + "".join(f"{w:>8s}" for w in W))
    for b in ("10-12", "12-15"):
        for k in ("runC", "run1", "pp"):
            print(f"{b + ' ' + {'runC': 'JS T3-20k', 'run1': 'JS T3-78k', 'pp': 'PP19'}[k]:>19s} entries" + "".join(f"{x:8d}" for x in o[f'entries_{k}'][b]))
            print(f"{'':>19s} share %" + "".join(f"{100 * x:8.1f}" for x in o[f'share_{k}'][b]))
    for k in ("runC", "run1"):
        x = o["raa"][k]["10-25"]
        print(f"{NAME[k]:9s} h± 10-25: all windows {x['all'][0]:.3f} ± {x['all'][1]:.3f};  >= {GD['min_entries']} entries "
              f"{x['min_entries'][0]:.3f} ± {x['min_entries'][1]:.3f} (drops {100 * x['dropped_share_auau']:.1f} % of the Au+Au yield)")
    g = o["gap"]["10-25"]
    print(f"gap h± 10-25 = {g['total']:+.3f}; per window: " + "  ".join(
        f"{w} {v:+.3f}±{e:.3f}" for w, v, e in zip(W, g["per_window"], g["per_window_err"]) if abs(v) > 0.0005))
    print()
sp = GD["split_noshift"]
print("CL, low windows (runC = JS T3-20k, run1a/run1b = JS T3-78k parts (a)/(b)): events with a 10-12 GeV hadron (rate) and the window's own R_AA(h± 10-25)")
for i, w in enumerate(sp["windows"][:5]):
    print(f"  {w:>6s}  " + "  ".join(
        f"{k} {sp[k]['entries_10_12'][i]:4d}/{sp[k]['events'][i]:5d} ({100 * sp[k]['entries_10_12'][i] / max(sp[k]['events'][i], 1):.2f}%)"
        + (f" R {sp[k]['raa_10_25'][i][0]:.2f}±{sp[k]['raa_10_25'][i][1]:.2f}" if k != "pp" and sp[k]["events"][i] and sp[k]['raa_10_25'][i][1] == sp[k]['raa_10_25'][i][1] else "")
        for k in ("runC", "run1a", "run1b", "pp")))
print("\nparton level, JS T3-78k / JS T3-20k per window (produced files)")
PC = ("lead_pt_final", "lead_final_over_hard", "E_jet_partons", "n_recoils", "E_recoils", "n_holes", "E_holes")
print(f"{'window':>8s}" + "".join(f"{c[:16]:>18s}" for c in PC))
for w, a_ in GD["partons"]["runC"].items():
    b_ = GD["partons"]["run1"].get(w)
    if b_:
        print(f"{w:>8s}" + "".join(f"{'%.3f ± %.3f' % (b_[c][0] / a_[c][0], np.hypot(a_[c][1] / a_[c][0], b_[c][1] / b_[c][0]) * b_[c][0] / a_[c][0]):>18s}" for c in PC))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
x = np.arange(len(GD["noshift"]["windows"]))
for arm, off, mk, lab in (("noshift", -0.12, "o", "CL (no CPS)"), ("shift", 0.12, "s", "CL+CPS")):
    g = GD[arm]["gap"]["10-25"]
    ax[0].errorbar(x + off, g["per_window"], g["per_window_err"], fmt=mk, ms=6, label=f"{lab} (gap {g['total']:+.3f})")
ax[0].axhline(0, color="k", lw=0.8); ax[0].set_xticks(x, GD["noshift"]["windows"], rotation=40, fontsize=8)
ax[0].set_xlabel("p̂_T window [GeV]"); ax[0].set_ylabel("contribution to R_AA(JS T3-78k) − R_AA(JS T3-20k)")
ax[0].set_title("h± |η|<1 10–25 GeV: which windows carry the gap", fontsize=10); ax[0].legend(fontsize=8)
for k, off, mk, lab in (("runC", -0.15, "o", "JS T3-20k"), ("run1a", 0.0, "s", "JS T3-78k part (a), 20k"), ("run1b", 0.15, "^", "JS T3-78k part (b), 42k")):
    v = np.array([r if r[1] == r[1] else [np.nan, np.nan] for r in sp[k]["raa_10_25"]], float)
    ax[1].errorbar(np.arange(len(v)) + off, v[:, 0], v[:, 1], fmt=mk, ms=6, label=lab)
ax[1].axhline(1, color="k", lw=0.8); ax[1].set_xticks(np.arange(len(sp["windows"])), sp["windows"], rotation=40, fontsize=8)
ax[1].set_xlabel("p̂_T window [GeV]"); ax[1].set_ylabel("R_AA of the window, h± 10–25")
ax[1].set_title("CL (no CPS): each window's own R_AA", fontsize=10); ax[1].legend(fontsize=8); ax[1].set_ylim(0, 3)
fig.suptitle("Figure 3a. JS T3-20k vs JS T3-78k per p̂_T window, CL+CPS and CL (model, not vetted)", fontsize=11)
fig.tight_layout(); plt.show()
""")

md(r"""
**Findings: the gap comes from the low p̂_T windows** (model, not vetted).
- **(1) Settings** are identical: XML, summaries, driver and build. The p+p reference is the same file set.
- **(2) Parton level.** Per window, JS T3-78k / JS T3-20k agrees within 1–3 % and within errors for all seven quantities.
  No 15–20 % difference anywhere, so this is not a settings difference.
- **(3) Where the gap sits.** Without the CPS it is +0.228: the 5–7 window carries +0.085 ± 0.049, 7–9
  +0.040 ± 0.038 and 9–11 +0.099 ± 0.026; every window from 11 GeV up contributes |Δ| ≤ 0.01. With the CPS the
  gap is +0.018, and no window stands out.
- **The low windows feed upward.** Without the CPS, events with p̂_T 7–11 GeV make 10–25 GeV hadrons more often
  in Au+Au than in p+p: the window's own R_AA is 1.6–1.7 at 7–11 GeV in JS T3-78k. That is a tail, ≤ 2 % of the
  events, and in JS T3-78k the 5–7, 7–9 and 9–11 windows together hold 48 % of the 10–12 GeV bin's yield.
- **The 5–7 GeV window** (JS T3-78k (c): 10k events instead of 2k). Four of its 10k events put a 10–12 GeV hadron
  into CL, against 4 of 80k in PP19, so the window's own R_AA is about 7, uncertain to about ±70 %. JS T3-20k, with
  2k events in 5–7, has none. This raised JS T3-78k's CL h± 10–25 from 0.94 to 1.02.
- **JS T3-20k is the outlier in that tail.** Its 9–11 window has 19 of 2000 events with a 10–12 GeV hadron, against
  40/2000 in JS T3-78k (a) and 107/6006 in JS T3-78k (b). That is a 2.8 σ local deviation (Poisson p ≈ 0.001 for ≤ 19 at
  JS T3-78k's rate). JS T3-78k (a) and (b) agree with each other: R_AA of that window 1.62 ± 0.29 and 1.62 ± 0.20, vs
  0.79 ± 0.17 in JS T3-20k. We picked the largest deviation out of about 50 window × bin × arm cells, so this is
  consistent with a fluctuation in a rarely populated tail; JS T3-78k has 4× the events there.
- **(0) Minimum-entries rule (≥ 10 in Au+Au and p+p).** Without the CPS: JS T3-20k 0.794 → 0.774 and JS T3-78k
  1.022 → 0.942 (dropping 9.8 % of its Au+Au yield, mostly the 5–7 window). The gap shrinks to 0.17 but stays: the
  deciding JS T3-20k window, 9–11 GeV, has 19 entries, above the threshold.
  - With the CPS, the rule raises R_AA by 0.02–0.03 (JS T3-20k 0.219 → 0.248, dropping 17 % of the Au+Au yield).
    The windows it drops are those where Au+Au has few entries *because* of the suppression, so a rule counting
    Au+Au entries biases R_AA upward.
  - A minimum-entries rule should count the p+p entries only, or report the change in R_AA it causes.
- **Consequence.** Without the CPS, the hadron R_AA at 10–25 GeV relies on the tail of low p̂_T windows, which
  2000 events per window sample poorly. The windows below the hadron p_T need at least ~10k events each, or the
  quoted error should include the window-to-window spread. JS T3-78k (78k) is the better estimate. JS T3-20k and JS T3-78k stay
  separate samples.
""")

md(r"""
### Cross-checks: JS T3-20k vs JS T3-78k, XS-ISR T3 vs XS-ISR T3 (reuse)

JS T3-78k repeats JS T3-20k with about 4× the events (same settings, other seeds); XS-ISR T3 (reuse) repeats XS-ISR T3 with the hydro reused 20×
instead of one hydro per event. Both pairs should agree within statistics (the samples stay separate).
""")

code(r"""
PAIRS = [("runC_injob", "run1_injob", "JS T3-20k vs JS T3-78k: CL+CPS (job)"),
         ("B2_injob", "run1p_injob", "XS-ISR T3 vs XS-ISR T3 (reuse): iCol-R (job)"),
         ("B2_icol_remn", "run1p_icol_remn", "XS-ISR T3 vs XS-ISR T3 (reuse): iCol-R"),
         ("B2_cl_noshift", "run1p_cl_noshift", "XS-ISR T3 vs XS-ISR T3 (reuse): CL")]
fig, ax = plt.subplots(1, len(PAIRS), figsize=(6 * len(PAIRS), 4.2), sharey=True)
for a_, (t1, t2, title) in zip(ax, PAIRS):
    if t1 not in RES or t2 not in RES:
        a_.set_title(title + " (missing)", fontsize=9); continue
    x = np.arange(len(OBS))
    for t, off, mk in ((t1, -0.12, "o"), (t2, 0.12, "s")):
        v = np.array([integ(t, k, lo, hi) for k, lo, hi, _ in OBS])
        a_.errorbar(x + off, v[:, 0], v[:, 1], fmt=mk, ms=6, label=textwrap.fill(S[t][0], 46))
    a_.set_xticks(x, [o[3] for o in OBS], rotation=40, fontsize=8); a_.set_title(title, fontsize=9); a_.legend(fontsize=7)
    a_.set_ylim(0, 1.2)
ax[0].set_ylabel("integrated $R_{AA}$")
fig.suptitle("Figure 3b. Cross-checks: the same settings, independent samples (model, not vetted)", fontsize=11)
fig.tight_layout(); plt.show()
print("ratios: second sample / first (JS T3-78k / JS T3-20k, XS-ISR T3 (reuse) / XS-ISR T3), integrated ranges")
for t1, t2, title in PAIRS:
    if t1 in RES and t2 in RES:
        print(title + ": " + "  ".join(
            f"{o[3]} {a[0] / b[0]:.3f}±{np.hypot(a[1] / a[0], b[1] / b[0]) * a[0] / b[0]:.3f}"
            for o, a, b in zip(OBS, (integ(t2, k, lo, hi) for k, lo, hi, _ in OBS), (integ(t1, k, lo, hi) for k, lo, hi, _ in OBS))))
""")

md(r"""
### The CPS's effect, sample by sample, and the holes' closure

Three panels:
- **Left:** integrated R_AA without / with the CPS, Colorless, each against its own p+p reference hadronized the
  same way.
- **Middle:** the holes' closure in the JS chain. iColored on JS T3-78k's partons with √s/6 beam ends, the holes'
  colour lines closed by *fake* relative to *reconnect* (the production default).
- **Right:** the holes' closure with the real remnants, XS-ISR T3 (reuse), each closure relative to reconnect.
  - Method: the stored Pythia input is recorded after iColored's closure. `rehadronize_icolored.py --beam-ends
    stored` therefore rebuilds the input from before the closure, from the stored partons plus the job's real
    remnants, and runs iColored with each closure.
  - Validation: the rebuilt path with reconnect against the job-input path (iCol-R).
  - The p+p reference is XS-ISR pp iCol-R for all of them: vacuum has no holes, so no closure acts there.
""")

code(r"""
fig, ax = plt.subplots(1, 3, figsize=(20, 4.3))
x = np.arange(len(OBS))
SNAME = NAME
def ratio_pts(a, b):
    v = np.array([[*integ(a, k, lo, hi), *integ(b, k, lo, hi)] for k, lo, hi, _ in OBS])
    r = v[:, 0] / v[:, 2]
    return r, r * np.hypot(v[:, 1] / v[:, 0], v[:, 3] / v[:, 2])
for i, smp in enumerate(("run1", "run1p", "B2", "B1")):
    a, b = f"{smp}_cl_noshift", f"{smp}_cl_shift"
    if a in RES and b in RES:
        v = np.array([[*integ(a, k, lo, hi), *integ(b, k, lo, hi)] for k, lo, hi, _ in OBS])
        r = v[:, 0] / v[:, 2]; e = r * np.hypot(v[:, 1] / v[:, 0], v[:, 3] / v[:, 2])
        ax[0].errorbar(x + 0.08 * (i - 1.5), r, e, fmt="os^D"[i], ms=6, label=SNAME[smp])
if "run1_icol_beam_fake" in RES and "run1_icol_beam_reco" in RES:
    r, e = ratio_pts("run1_icol_beam_fake", "run1_icol_beam_reco")
    ax[1].errorbar(x, r, e, fmt="v", ms=6, color="C6", label="JS T3-78k: iCol-B/fake / iCol-B/rec")
for i, (t, lab, mk, c) in enumerate((("run1p_icol_remn_fake", "fake / reconnect", "v", "C6"),
                                     ("run1p_icol_remn_thermal", "thermal / reconnect", "s", "C2"),
                                     ("run1p_icol_remn_rglobal", "reconnect_global / reconnect", "^", "C4"),
                                     ("run1p_icol_remn", "job-input path / rebuilt path (both reconnect): validation", "D", "C0"))):
    if t in RES and "run1p_icol_remn_reco" in RES:
        r, e = ratio_pts(t, "run1p_icol_remn_reco")
        ax[2].errorbar(x + 0.08 * (i - 1.5), r, e, fmt=mk, ms=6, color=c, label=f"XS-ISR T3 (reuse) iCol-R: {lab}")
for a_, t in zip(ax, ("R_AA(CL) / R_AA(CL+CPS)", "holes' closure, JS (√s/6 beam ends): variant / reconnect",
                      "holes' closure, XS-ISR (real remnants): variant / reconnect")):
    a_.set_xticks(x, [o[3] for o in OBS], rotation=40, fontsize=8); a_.axhline(1, color="k", lw=0.8); a_.set_title(t, fontsize=10)
    a_.legend(fontsize=8)
fig.suptitle("Figure 4. The CPS's effect on R_AA, and the holes' closure systematic (model, not vetted)", fontsize=11)
fig.tight_layout(); plt.show()
""")

md(r"""
**Reading: the closure with the real remnants** (model, not vetted; XS-ISR T3 (reuse), integrated ranges as in the
table; ratios to the rebuilt reconnect).
- **Validation.** The job-input path (iCol-R) over the rebuilt reconnect path is 0.96–1.01 for h± 10–25 and the
  jets, and 0.94–0.98 for π 8–20 and h± 8–15.
  The rebuilt path also places exactly the job's partner partons (9.0 per event). So the rebuilt input is the
  job's input before the closure.
- **fake / reconnect: 0.64–0.84** (h± 10–25 0.74, jets 0.81 / 0.70, full jets 0.64, π 0.84), close to the JS chain's
  iCol-B/fake / iCol-B/rec (0.63–0.70). The real remnants do not reduce the sensitivity: the fake ends act on the
  ≈ 73 open lines the holes leave, in the jet region, not at the beam.
- **reconnect_global / reconnect: 0.83–0.93.** It joins more open ends (34 vs 32 per event) and inserts fewer
  partners (4 vs 9).
- **thermal / reconnect: 0.81–1.27.** It rises from π 8–20 (0.81) and h± 10–25 (0.92) to full jets (1.27). Thermal ends every open
  line on a medium partner (56 partners, 71 GeV per event), which adds that energy to the jets' surroundings.
- **The hadron ranges π 8–20 and h± 8–15** are usable since XS-ISR T3 (reuse) (c), with n_eff ≥ 90 except for
  fake. Relative to reconnect: fake 0.84 / 0.78, thermal 0.81 / 0.86, reconnect_global 0.87 / 0.93. The fake
  closure's π stays thin (n_eff 9).
- **Bottom line.** For iColored in central Au+Au the holes' closure is a 10–35 % effect on R_AA, with or without
  the real remnants. Reconnect, the default, sits in the middle of the variants.
""")

md(r"""
### Why iColored with the holes' lines closed by fakes comes out near CL+CPS (jets)

In JS T3-78k, iColored with holes fake (π 0.52, jets R0.4 0.32) lies much closer to CL+CPS (0.27,
0.23) than iColored with reconnect does (0.70, 0.46). The two suppressions have different origins.

**What the closure does.** When LBT scatters a jet parton, the recoil's colour partner is the hole, which is not
hadronized. That leaves about 72 open colour lines per event in JS T3-78k.
- **Reconnect** pairs them with nearby partons: ~32 reconnections plus ~8 added partners (~10 GeV). The strings
  stay short and low-mass inside the jet.
- **Fake** ends each line on a 1 GeV massless parton along the beam axis (~72 GeV per event). The jet parton's
  string is stretched in rapidity, and its energy goes to more, softer hadrons.

The holes are the same in both (subtracted at parton level), and so is the p+p reference: vacuum has no open
lines, as Fig. 1a shows.

The cell below checks this on the same partons:
- the σ-weighted Au+Au and p+p yields, fake / reconnect, from the R_AA files;
- per p̂_T window, the h± count above 8 and 12 GeV (|η| < 1) and the energy flow, from JS T3-78k (a)'s 100 files.
""")

code(r"""
from concurrent.futures import ProcessPoolExecutor
def _closure_one(f):
    import h5py, hdf5plugin  # noqa: F401
    import icolored_diag as I
    ch = I.charged_ids(); out = {}
    for k in ("icol_beam_reco", "icol_beam_fake"):
        with h5py.File(f.replace("icol_beam_reco", k)) as h:
            n = int(h.attrs["nunits_written"]); so = h["hadrons/sample_offsets"][:]; uo = h["hadrons/unit_offsets"][: n + 1]
            bins = [f"{a:g}-{b:g}" for a, b in np.asarray(h.attrs["pthat_bins"]).reshape(-1, 2)]; win = h["units/pthat_bin"][:n]
            ns = int(uo[-1]); p = h["hadrons/p"][: so[ns]].astype(float); pid = h["hadrons/pid"][: so[ns]]
        unit = np.repeat(np.arange(n), np.diff(uo))[np.repeat(np.arange(ns), np.diff(so[: ns + 1]))]
        pt = np.hypot(p[:, 1], p[:, 2]); eta = np.arcsinh(p[:, 3] / np.maximum(pt, 1e-9))
        m = np.isin(np.abs(pid), ch) & (np.abs(eta) < 1)
        r = {"samples": ns, "hadrons": len(pid), "E_fwd": p[np.abs(eta) > 2, 0].sum(), "E_tot": p[:, 0].sum(),
             "sumpt_mid2": pt[(np.abs(eta) < 1) & (pt > 2)].sum()}
        for lab, cut in (("pt8", pt > 8), ("pt12", pt > 12)):
            for w, b in enumerate(bins):
                r[(lab, b)] = int((m & cut & (win[unit] == w)).sum())
        out[k] = r
    return out
CF = sorted(glob.glob(f"{RH}/run1/icol_beam_reco/a/{G}"))
with ProcessPoolExecutor(12) as ex:
    CR = list(ex.map(_closure_one, CF))
tot = {k: {} for k in ("icol_beam_reco", "icol_beam_fake")}
for r in CR:
    for k, d in r.items():
        for q, v in d.items():
            tot[k][q] = tot[k].get(q, 0) + v
a_, b_ = tot["icol_beam_reco"], tot["icol_beam_fake"]
print(f"JS T3-78k part (a), {len(CF)} files, {a_['samples']} fragmentations; per fragmentation, iCol-B/rec -> iCol-B/fake:")
for q, lab in (("hadrons", "hadrons"), ("E_tot", "hadron energy [GeV]"), ("E_fwd", "energy at |η| > 2 [GeV]"),
               ("sumpt_mid2", "Σp_T |η|<1, p_T > 2 [GeV]")):
    print(f"  {lab:28s} {a_[q] / a_['samples']:8.1f} -> {b_[q] / b_['samples']:8.1f}   (x{b_[q] / a_[q]:.2f})")
print("\nσ-weighted yields from the R_AA files (fake / reconnect): Au+Au | p+p | R_AA")
for key, lo, hi in (("pion", 8, 20), ("had", 10, 25), ("chjet_R0.4_star", 16, 30), ("fjet_R0.4", 15, 50)):
    da, db = RES["run1_icol_beam_reco"]["raa"][key], RES["run1_icol_beam_fake"]["raa"][key]; e = np.array(da["edges"])
    s = (e[:-1] >= lo - 1e-9) & (e[1:] <= hi + 1e-9)
    A1, A2 = np.sum(np.array(da["auau"])[s]), np.sum(np.array(db["auau"])[s]); P1, P2 = np.sum(np.array(da["pp"])[s]), np.sum(np.array(db["pp"])[s])
    print(f"  {key:16s} {lo}-{hi}: {A2 / A1:.3f} | {P2 / P1:.3f} | {(A2 / P2) / (A1 / P1):.3f}")
print("\nper p̂_T window: h± |η|<1 above 8 / 12 GeV, fake / reconnect (windows with >= 20 reconnect counts)")
WINS = sorted({q[1] for q in a_ if isinstance(q, tuple)}, key=lambda s: float(s.split("-")[0]))
for lab in ("pt8", "pt12"):
    for w in WINS:
        x, y = a_[(lab, w)], b_[(lab, w)]
        if x >= 20:
            print(f"  p_T > {lab[2:]:2s} GeV  window {w:6s}  {x:6d} -> {y:6d}   {y / x:.2f} ± {np.sqrt(1 / x + 1 / max(y, 1)) * y / x:.2f}")
""")

md(r"""
**Reading** (model, not vetted).
- The p+p reference is the same for both closures (ratio 1.000). The whole R_AA difference is in Au+Au: the
  σ-weighted high-p_T yield drops by 30–37 %.
- Why p+p does not see it: the closure acts only on the colour lines that removing the holes leaves open, and holes
  exist only where LBT scatters in the medium. In PP19 (vacuum MATTER) the logs show 0 open lines per event, and no
  fakes or reconnections, so both closures give the same hadrons (Fig. 1a: the two curves coincide). The √s/6 beam
  ends close the shower's own open lines in p+p and Au+Au alike, so they cancel in R_AA. The stretching is a pure
  medium effect, entirely in the numerator.
- The fake closure makes about 50 % more hadrons (169 → 249 per fragmentation) and puts more energy forward:
  41 % more at |η| > 2. Mid-rapidity Σp_T above 2 GeV drops by 11 %.
- The loss is concentrated where one hadron carries a large share of the parton's p_T. With fake, h± above 8 GeV
  from the p̂_T 9–23 GeV windows drop by 14–22 % (and by about 70 % from 7–9 GeV, few counts), but those from the
  23–80 GeV windows drop by only 3–5 %.
- Because the spectrum falls steeply, the σ-weighted yield at 8–20 GeV comes mostly from those low windows. So the
  closure changes R_AA by a factor 0.69–0.70 for hadrons and 0.63–0.70 for jets.

The mechanisms differ:
- **Colorless's shift** adds energy to the partons and to the holes. The hole hadrons (about 310 instead of
  107 GeV per event) are then subtracted, which removes far more than the jet lost: a factor 4–5 for hadrons.
- **iColored with fake ends** adds no mid-rapidity energy and hadronizes no holes. It only softens the jet's
  fragmentation: a factor 1.3–1.4.

That the jet R_AA of the two come out close is a coincidence. The holes' closure is a genuine systematic with
and without the real remnants: with them (XS-ISR T3 (reuse), Fig. 4 right), fake / reconnect is 0.64–0.84, about
as large as for the JS chain's √s/6 beam ends.
""")

md(r"""
## Parton-level reference: which hadronization keeps the parton-level suppression

**Motivation.** The hadronizations disagree by a factor 2–4 in the jet R_AA (Figs. 2a–2c). The parton level
needs no hadronization and is therefore the reference. Without it we cannot tell which hadronizer adds
suppression and which removes it. In particular: does the remaining CL vs iCol-R gap for jets (XS-ISR T3 (reuse):
0.65 / 0.68 vs 0.45 / 0.43) come from iColored's strings pulling energy out of the cone, or from Colorless putting
energy back?

**Method** (`css_parton_jets.py`; the parton-level jets as in `quench_diag.py` A1).
- **Parton-level jets** come from the stored final partons. Anti-k_T, R = 0.2 and 0.4, |η_jet| < 1 − R. Two
  definitions:
  - *parton, no response:* the shower and ISR partons only (pstat 0, 1000, 22);
  - *parton level, recoils − holes:* the same plus LBT's recoils (pstat 1), with the holes (pstat −1) subtracted
    inside the cone using the full four-momentum. This is JETSCAPE's balance and the parton-level reference.
- **Hadron-level jets** use the same definition: visible hadrons with p_T > 0.2 GeV and |η| < 1, holes subtracted
  as in the R_AA (Colorless: its pstat −1 hadrons; iColored: holes plus partner copies). They use sample 0 of the
  ten fragmentations, i.e. one fragmentation per event. These are full jets, so they compare directly with the
  parton level.
- **Weights:** the same windows, weights and stitching as `auau_raa.py`, and each chain's own p+p reference
  (PP19, XS-ISR pp).
- **Per parton jet** (parton level with recoils − holes, p_T > 8 GeV): r = p_T(matched hadron jet) / p_T(parton
  jet), where the matched hadron jet is the hardest within ΔR < R (r = 0 if none). From that:
  - the weighted mean ⟨r⟩ per parton-jet p_T bin, in Au+Au and p+p;
  - the double ratio ⟨r⟩_AA / ⟨r⟩_pp.

  A double ratio of 1 means the hadronization treats Au+Au jets like p+p jets, so it keeps the parton-level
  suppression. Above 1, it adds energy to the medium-modified jets; below 1, it removes energy.
- **Charged-like parton jets** are not made: a parton's charge is not defined cleanly.
""")

code(r"""
PJ_F = Path("../results/css_parton_jets.json")
if not PJ_F.exists():
    subprocess.run([sys.executable, "css_parton_jets.py", "-j", "12"], check=True, capture_output=True)
PJ = json.loads(PJ_F.read_text()); PE = np.array(PJ["edges"]); PX = 0.5 * (PE[1:] + PE[:-1]); PXE = 0.5 * np.diff(PE)
PJ_CH = {"JS": ("JS T3-78k vs PP19", ("cl_shift", "cl_noshift", "icol_beam_reco", "icol_beam_fake")),
         "XS": ("XS-ISR T3 (reuse) vs XS-ISR pp", ("cl_shift", "cl_noshift", "icol_remn"))}
PJ_STY = {"cl_shift": ("C3", "o"), "cl_noshift": ("C2", "s"), "icol_remn": ("C0", "D"), "icol_beam_reco": ("C1", "^"),
          "icol_beam_fake": ("C6", "v")}
def pj_int(d, lo, hi):
    a, p = np.array(d["auau"]), np.array(d["pp"]); s = (PE[:-1] >= lo) & (PE[1:] <= hi)
    return a[s].sum() / p[s].sum()
print("integrated jet R_AA, full jets (parton level / hadron level, the same definition):")
for smp, (lab, kinds) in PJ_CH.items():
    o = PJ[smp]
    for R in PJ["R"]:
        for lo, hi in ((10, 16), (16, 30)):
            print(f"  {lab:32s} R={R} {lo}-{hi} GeV: parton (recoils - holes) {pj_int(o[f'resp_R{R}'], lo, hi):.2f}, "
                  f"no response {pj_int(o[f'par_R{R}'], lo, hi):.2f} | " +
                  "  ".join(f"{KIND[k]} {pj_int(o[k][f'had_R{R}'], lo, hi):.2f}" for k in kinds))
print("\ndouble ratio <r>_AA / <r>_pp per parton-jet pT bin " + " ".join(f"{a:g}-{b:g}" for a, b in zip(PE[:-1], PE[1:])))
for smp, (lab, kinds) in PJ_CH.items():
    for R in PJ["R"]:
        for k in kinds:
            d = PJ[smp][k][f"r_R{R}"]
            print(f"  {lab[:18]:18s} R={R} {KIND[k]:11s} " + " ".join(f"{x:.2f}" for x in d["double"])
                  + f"   (unmatched in Au+Au, lowest bins: {', '.join(f'{u:.2f}' for u in d['auau_unmatched'][:3])})")

for fig_no, (what, ylab, ylim) in enumerate((("raa", "jet R_AA (full jets)", (0, 1.6)),
                                            ("double", "⟨r⟩_AA / ⟨r⟩_pp", (0.6, 2.0)))):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.4), sharex=True)
    for i, (smp, (lab, kinds)) in enumerate(PJ_CH.items()):
        o = PJ[smp]
        for j, R in enumerate(PJ["R"]):
            a_ = ax[i, j]
            if what == "raa":
                d = o[f"resp_R{R}"]
                a_.errorbar(PX, d["raa"], d["err"], PXE, fmt="k*-", ms=8, lw=1.5, label="parton level, recoils − holes (reference)")
                a_.plot(PX, o[f"par_R{R}"]["raa"], "--", color="0.5", lw=1.2, label="parton level, no medium response")
                for k in kinds:
                    c, mk = PJ_STY[k]; h = o[k][f"had_R{R}"]
                    a_.errorbar(PX, h["raa"], h["err"], PXE, fmt=mk, ms=5, color=c, mfc="white", label=f"hadron level, {KIND_L[k]}")
            else:
                for k in kinds:
                    c, mk = PJ_STY[k]; d = o[k][f"r_R{R}"]
                    a_.errorbar(PX, d["double"], d["double_err"], PXE, fmt=mk + "-", ms=5, color=c, label=KIND_L[k])
                a_.axhline(1, color="k", lw=0.8)
            a_.set_ylim(*ylim); a_.set_title(f"{lab}, R = {R}", fontsize=10)
            if i == 1:
                a_.set_xlabel("jet p_T [GeV]" if what == "raa" else "parton-level jet p_T [GeV]")
            if j == 0:
                a_.set_ylabel(ylab)
                a_.legend(fontsize=7, loc="upper right")
    fig.suptitle(("Figure 5a. Jet R_AA: parton level vs hadron level, the same jet definition" if what == "raa" else
                  "Figure 5b. Double ratio ⟨r⟩_AA / ⟨r⟩_pp, r = p_T(hadron jet) / p_T(parton jet)") + " (model, not vetted)", fontsize=11)
    fig.tight_layout(); plt.show()
""")

md(r"""
**Reading** (model, not vetted; full jets, integrated over 16–30 GeV).
- **The parton level alone is already close to the data.** With recoils − holes the parton-level jet R_AA is
  0.26–0.27 (R = 0.2) and 0.31–0.32 (R = 0.4), the same in both chains. Without the medium response it is
  0.24–0.25. For comparison, STAR's charged jets give 0.31 / 0.30 (0–10 %, 16–30 GeV). That is a different
  observable (charged, with a leading-track cut), so this is only indicative.
- **The hadronizations move the jet R_AA away from the parton level.**
  - CL+CPS: 0.24–0.25, slightly below the parton level. Its double ratio is 0.74–0.94 at R = 0.4 and 0.90–1.07
    at R = 0.2: the CPS takes energy out of the Au+Au jets.
  - iColored: iCol-R 0.45–0.46 and iCol-B/rec 0.46–0.48, about 1.5× the parton level; iCol-B/fake 0.32–0.37.
    The double ratio is 1.00–1.06 (R = 0.4) and 1.00–1.12 (R = 0.2) above 16 GeV, rising to 1.1–1.5 at 8–12 GeV.
  - CL: 0.65–0.76, 2–3× the parton level. Its double ratio is 1.04–1.38 at R = 0.4 and up to 1.9 at R = 0.2 and
    low p_T.
- **Why a few per cent matter.** The jet spectrum falls steeply (roughly p_T^-6). A 5–10 % shift of the Au+Au jet
  p_T relative to p+p therefore moves R_AA by 30–70 %, and the large double ratios at low parton-jet p_T migrate
  into the higher bins.
- **The CL vs iCol-R gap.** Neither hadronizer pulls energy out of the cone relative to p+p: iCol-R's double
  ratio is ≥ 1. The gap comes from **Colorless putting energy back into the Au+Au jets.** CL's double ratio is
  larger than iCol-R's at every p_T, by 0.04–0.4. Its in-cone hadron p_T is 0.96 of the parton level in Au+Au
  against 0.79 in p+p (next section). iColored is the closer of the two to keeping the parton-level relation: it
  adds 0–6 % at R = 0.4 above 16 GeV. The low-p_T excess comes from the medium response, i.e. the recoils'
  strings.
- **Caveat for iCol-B.** Its partner copies are not stored, so only the holes are subtracted, as in its R_AA.
  That makes iCol-B's double ratio an upper bound.
""")

md(r"""
## Colorless's hole subtraction: energy balance, and full vs charged jets

**Motivation.** Colorless hadronizes the holes in a second pass and subtracts those hadrons (pstat −1). That is
only right if the subtracted energy matches the holes' energy. The CPS and Colorless's fake string ends
(√s/6 = 33 GeV each) add energy to both passes. An open item was also why CL's full jets (R = 0.4) reach R_AA ≈ 0.9
at 10–15 GeV while the charged jets in STAR's definition stay at 0.5–0.65.

**Method** (`css_parton_jets.py`, sample 0, σ-weighted per event):
- Σ E(positive hadrons) − Σ E(pstat −1 hadrons) against Σ E(partons, pstat ≥ 0 with the recoils) − Σ E(holes);
- inclusive (all η), and inside ΔR < 0.4 of the parton-level jets (recoils − holes, p_T > 10 GeV, |η| < 0.6), as
  visible p_T and charged p_T;
- CL and CL+CPS, with iColored for comparison.
""")

code(r"""
print("inclusive energy per event [GeV] (σ-weighted, sample 0): hadrons + - hadrons - | partons - holes")
for smp, (lab, kinds) in PJ_CH.items():
    for k in kinds:
        b = PJ[smp][k]["balance"]["auau"]
        print(f"  {lab[:18]:18s} {KIND[k]:11s} E+ {b['E_pos']:7.1f}  E- {b['E_neg']:6.1f}  E+ - E- {b['E_pos'] - b['E_neg']:7.1f}"
              f"  | partons {b['E_partons']:6.1f} - holes {b['E_holes']:5.1f} = {b['E_partons'] - b['E_holes']:6.1f}")
print("\nin the parton-jet cone (dR < 0.4, parton jets pT > 10, |eta| < 0.6), per jet: visible pT (hadrons + - -) / parton pT (partons - holes)"
      "\n and the charged fraction of the hadron pT, Au+Au | p+p")
for smp, (lab, kinds) in PJ_CH.items():
    for k in kinds:
        q = PJ[smp][k]["balance"]
        out = []
        for side in ("auau_cone", "pp_cone"):
            c = q[side]; vis = c["vis_pos_pt"] - c["vis_neg_pt"]; par = c["par_pt"] - c["hole_pt"]
            out.append(f"{vis / par:.3f} (charged frac {(c['ch_pos_pt'] - c['ch_neg_pt']) / max(vis, 1e-9):.3f})")
        ra = (q["auau_cone"]["vis_pos_pt"] - q["auau_cone"]["vis_neg_pt"]) / (q["auau_cone"]["par_pt"] - q["auau_cone"]["hole_pt"])
        rp = (q["pp_cone"]["vis_pos_pt"] - q["pp_cone"]["vis_neg_pt"]) / (q["pp_cone"]["par_pt"] - q["pp_cone"]["hole_pt"])
        print(f"  {lab[:18]:18s} {KIND[k]:11s} AA {out[0]}  | pp {out[1]}  | AA/pp {ra / rp:.3f}")
print("\nCL, full vs charged jets R = 0.4 (R_AA per bin): full jets, charged jets without and with STAR's leading-track cut")
for t in ("run1_cl_noshift", "run1p_cl_noshift", "run1_cl_shift", "run1p_icol_remn"):
    if t not in RES:
        continue
    r = RES[t]["raa"]
    print(f"  {S[t][0]}")
    for key, lab in (("fjet_R0.4", "full jets |η|<0.5"), ("chjet_R0.4", "charged, no lead cut"), ("chjet_R0.4_star", "charged, STAR def. (lead > 5 GeV)")):
        e = r[key]["edges"]
        print(f"    {lab:32s} " + "  ".join(f"{a:g}-{b:g}: {x:.2f}" for a, b, x in zip(e[:-1], e[1:], r[key]["raa"])))
""")

md(r"""
**Reading** (model, not vetted).
- **Inclusive balance.**
  - CL: Σ E⁺ − Σ E⁻ is 44.3 GeV per event against partons − holes 45.2 GeV (JS T3-78k). The XS-ISR numbers are
    analogous. The fake ends add about the same energy to both passes (18–19 GeV), so they cancel: **CL's hole
    subtraction is balanced inclusively.**
  - CL+CPS: E⁺ − E⁻ is 138.6 GeV against 45.2 GeV. The CPS adds about 200 GeV to the positive pass and 106 GeV to
    the hole pass, so the net excess is about 94 GeV per event.
- **In the jet cone.**
  - CL+CPS: the hadrons keep 0.685 of the parton-level p_T in Au+Au against 0.79 in p+p (AA/pp 0.87). The CPS adds
    energy overall, but outside the cone: its p_z kicks push hadrons forward and stretch the strings. Inside the
    cone, more p_T is removed than the jet lost.
  - CL: 0.96 in Au+Au against 0.79 in p+p (AA/pp 1.21). Colorless gives the Au+Au jets 21 % more of their
    parton-level p_T than the p+p jets. This is the "putting back" seen in the double ratio.
  - iColored: 0.95 against 0.78 before the partner copies are subtracted. The R_AA also subtracts the holes and
    partners at parton level, which removes about 1 GeV per cone.
- **The charged fraction** of the in-cone hadron p_T is the same in Au+Au and p+p (0.59 for Colorless; 0.575 vs
  0.59 for iColored). There is no neutral/charged imbalance from the hole subtraction.
- **Full vs charged jets for CL.** The low charged-jet values come from STAR's leading-track cut (a 5 GeV track,
  on the Au+Au jets only), not from the subtraction:
  - charged jets *without* the cut give 1.18 / 1.09 / 0.90 at 8–14 GeV, even above the full jets (0.92 at
    10–15 GeV);
  - at equal jet energy, a charged jet's p_T is only about 0.6 of the full jet's.

  At low p_T, CL's Au+Au jets are largely soft medium-response hadrons that fail the leading-track cut. That
  closes the open "full jets 0.9" item.
""")

md(r"""
## Conclusions

*(filled in from the executed results; see `results/colorless_shift_study_executed.ipynb`)*

All numbers are model results from these runs, not vetted. Au+Au 0–10 % at 200 GeV. Integrated ranges: π 8–20,
h± 8–15 and jets 16–30 GeV, as in the table above. JS T3-78k and XS-ISR T3 (reuse) have 78k events each; JS T3-20k, XS-ISR T2 and XS-ISR T3 20k each.

**What is established**

1. **The CPS dominates Colorless in central Au+Au with LBT.**
   - Firings per event: about 800–990, against 0.4–0.7 in vacuum and 2–7 with MATTER alone (with or without
     recoils).
   - 62–67 % of the energy Colorless hands Pythia is added by the CPS. This holds in both passes, partons and
     holes.
   - LBT's recoils and holes, dense in momentum space, set it off; MATTER's ~6 recoils per event do not.
2. **The CPS sets the RHIC R_AA.** R_AA(CL) / R_AA(CL+CPS) is 4–5 for π and h± and 2.5–3.3 for
   jets (Fig. 4). The factor is the same in all four samples, i.e. both chains and both Type 2 and Type 3 settings.
   - With the CPS, every sample is near the data: π 0.14–0.22 (PHENIX 0.25), jets R0.2 0.22–0.28 (STAR 0.31) and
     R0.4 0.17–0.26 (STAR 0.30). JS T3-78k and JS T3-20k also reproduce JETSCAPE's Fig. 14.
   - Without the CPS: hadrons 0.7–1.15, jets 0.56–0.78. Values above 1 come from the low end of the hadron
     windows (below ~10 GeV, see the section on R_AA above 1). Above 12 GeV all samples are below 1; JS T3-78k's
     10–12 GeV bin is fed upward from p̂_T 5–7 GeV.
3. **iColored has no such shift, and it lies in between.**
   - JS T3-78k with √s/6 beam ends: π 0.78 / 0.54 and jets R0.4 0.46 / 0.32 (holes reconnect / fake).
   - XS-ISR T3 and XS-ISR T3 (reuse) with real remnants (reconnect): π 0.71–0.73, jets 0.43–0.49.

   So with these settings, neither hadronization without the CPS reaches the measured suppression. In these
   runs, the agreement of the published RHIC setup with data is carried to a large part by the CPS.
4. **The chains reproduce themselves (Fig. 3b).**
   - JS T3-78k vs JS T3-20k (the JS chain, Colorless in the job): ratios are consistent with 1 within 1–1.5 σ, e.g. jets
     R0.2 0.96 ± 0.15 and full jets 1.04 ± 0.11.
   - XS-ISR T3 (reuse) vs XS-ISR T3 (the XS-ISR chain, hydro reused 20× vs one hydro per event): ratios are
     - 0.91–1.09 ± 0.06 for jets and 0.87–1.04 for hadrons with iColored (n_frag 10);
     - 0.94–1.10 with CL.
   - So hydro reuse with ISR (`allowReuseWithISR`, local X-SCAPE change) reproduces the hydro-per-event result
     within statistics. Jets R0.2 at 0.91 ± 0.06 is the largest deviation, 1.5 σ.
5. **Errors with reuse.** The grouped error (per shared medium) is 1.00–1.02 times the per-event error for
   hadrons, and 1.02–1.12 times for jets. Events sharing a medium are nearly independent for these high-pT
   observables. For jets, the reused medium correlates the events a little, so 20 events per hydro are worth
   about 80 % of independent ones.
6. **The JS and XS-ISR chains differ only where CL is used.**
   - With CL: JS T3-78k vs XS-ISR T3 (reuse) is hadrons 1.10–1.14 vs 0.91–0.93 and jets 0.73–0.78 vs
     0.65–0.68 (both T3). With CL+CPS the two chains agree: XS-ISR / JS is 0.93–1.07 (Fig. 2c).
   - With iColored, the jets agree: 0.49 / 0.46 vs 0.45 / 0.43.
   - CL is sensitive to how the partons were made (no ISR, PythiaGun, PP19 reference),
     while iColored is not (cf. XS-ISR T2 vs XS-ISR T3).
   - Within the JS chain, JS T3-20k and JS T3-78k without the CPS differ in h± 10–25 (0.79 ± 0.05 vs
     1.02 ± 0.06, n_frag 10). Settings and parton level are identical. The gap comes from the low p̂_T windows:
     Colorless's upward feeding from 5–7 GeV, now sampled with 10k events, and a low fluctuation in JS T3-20k's
     9–11 GeV window (2.8 σ local).
7. **Use the n_frag 10 hadron rows for iColored.** With one fragmentation per event (in the job), single events
   from low p̂_T windows with large weights make spikes. For example, XS-ISR T3 (reuse) π is 0.95 ± 0.32 in the job
   against 0.73 ± 0.05 with n_frag 10.
8. **Statistics in the low p̂_T windows.** JS T3-78k (c) and XS-ISR T3 (reuse) (c) (+8k each in 5–7 and 7–9 GeV),
   and the p+p extensions (PP19 +64k each in 5–7 and 7–9; XS-ISR pp 7–9 at 25k), raise n_eff where it was
   thinnest.
   - JS T3-78k CL+CPS π 8–20: Au+Au 13 → 137, p+p 448 → 1993. Its R_AA moves from 0.27 ± 0.08 (a single-event
     artifact) to 0.22 ± 0.02.
   - XS-ISR T3 (reuse) iCol-R π 8–20: Au+Au n_eff 30 → 283 (smallest bin 5 → 38); R_AA 0.88 ± 0.16 → 0.73 ± 0.05.
     Its h± 8–15 is 0.69 → 0.64 ± 0.03.
   - Thinly sampled windows (< 5 p+p entries) now contribute visibly only in JS T3-78k CL h± 10–25 (−0.08; the
     upward feeding) and in the one-fragmentation (job) and fake-closure π of XS-ISR T3 (reuse). For its iCol-R
     (reconnect) the change is −0.03 to −0.05.
9. **The holes' closure with the real remnants** (XS-ISR T3 (reuse), Fig. 4 right). Relative to reconnect (the
   default), over all six ranges: fake 0.64–0.84, reconnect_global 0.83–0.93, thermal 0.81–1.27. The rebuilt
   input reproduces the job within 0–6 %. For iColored in central Au+Au the closure is a 10–35 % effect,
   with real remnants or √s/6 beam ends alike.
10. **The parton-level reference** (Figs. 5a, 5b).
    - The jet R_AA at parton level (recoils − holes, 16–30 GeV) is 0.26–0.27 (R = 0.2) and 0.31–0.32 (R = 0.4) in
      both chains.
    - At hadron level, the same jet definition gives:
      - CL+CPS 0.24–0.25: its double ratio ⟨r⟩_AA / ⟨r⟩_pp is 0.74–0.94 at R = 0.4, so it removes energy;
      - iColored 0.45–0.48: double ratio ≈ 1.00–1.06 above 16 GeV, i.e. it nearly keeps the parton-level
        relation; it adds energy below that from the medium response;
      - CL 0.65–0.76: double ratio 1.04–1.38, so Colorless puts energy back into the Au+Au jets.
    - The CL vs iCol-R gap comes from Colorless putting energy back, not from iColored pulling energy out of the
      cone.
    - Because the jet spectrum is steep, 5–10 % shifts in jet p_T move R_AA by 30–70 %.
11. **Colorless's hole subtraction.**
    - CL is balanced inclusively: Σ E⁺ − Σ E⁻ = 44.3 GeV per event against partons − holes = 45.2 GeV (JS
      T3-78k), and 61.5 against 62.0 GeV (XS-ISR T3 (reuse)); the fake ends cancel between the two passes. The CPS adds about 94 GeV per event net.
    - In the jet cone, CL keeps 0.96 of the parton p_T in Au+Au against 0.79 in p+p (AA/pp 1.21); CL+CPS keeps
      0.685 against 0.79 (0.87).
    - The charged fraction is the same in Au+Au and p+p (0.59).
    - CL's full jets (0.9 at 10–15 GeV) against its low STAR-definition charged jets: STAR's leading-track cut on
      the Au+Au jets, plus the 0.6 charged-to-full p_T scale. Charged jets without the cut are as high as the
      full jets.

**Open items**

- **The holes' closure.** With iColored, closing the holes' colour lines differently changes R_AA by 10–35 %,
  with or without real remnants (Fig. 4). This is iColored's dominant hadronization systematic in central Au+Au,
  and it needs a physics choice: thermal (colour to the medium) vs reconnect.
- **Charged-like parton-level jets** were not built: a parton's charge is not defined cleanly. The parton-level
  reference is for full jets.
- **LHC.** The size of the CPS at 5.02 TeV is not measured here. With more LBT recoils it is expected to be at
  least as large, and the LHC calibrations used the same Colorless.
- **Q_sw (arXiv:2408.08247).** The firings scale with what LBT produces. A Q_sw scan with and without the CPS
  would show whether the Q_sw preference depends on it.

**Consequences for the calibration and for O+O**

- A calibration made with Colorless and the CPS absorbs the CPS into the jet-medium parameters. With iColored,
  or CL, the same parameters give 2–3 times less jet suppression at RHIC. A consistent
  recalibration needs stronger coupling.
- Our energy-loss parameters come from that calibration: T3 is αs 0.3, qhatA 10, qhatB 100 (JETSCAPE, Colorless
  with the CPS). With them, the XS-ISR chain and iColored under-quench central Au+Au. XS-ISR T3 (reuse) gives π 0.73 and jets
  0.45 / 0.43, against 0.25 and 0.31 / 0.30 in data.
- **Indicative:** at parton level, with recoils − holes, T3 already gives full-jet R_AA 0.31–0.32 (R = 0.4,
  16–30 GeV), close to STAR's charged jets. The hadronization then moves R_AA by −20 % (CL+CPS) to +50 %
  (iColored). A recalibration with iColored therefore has to fix the coupling and the hadronization systematics
  together: the holes' closure and the medium response.
- The CPS depends on system size. In central O+O (top 10 % Npart, p̂_T 23–30) it fires 25 times per event
  (9.9 % of the energy), against about 800–1000 times (62–67 %) in central Au+Au.
  - So parameters fitted to Au+Au with the CPS predict too little suppression in O+O: the CPS suppresses Au+Au
    a lot and O+O barely.
  - Our O+O analysis (oo_analysis_rhic.ipynb) uses iColored, so its hadronization is not affected. Its energy-loss
    parameters are: they are the T3 values from the CL+CPS calibration, so our O+O predictions are
    likely under-quenched too.
  - A recalibration of the energy-loss parameters with iColored (or CL) on Au+Au data is
    needed before O+O predictions. Any comparison with O+O predictions built on the Colorless calibration must take
    the CPS into account.
""")

nb = {"cells": [{**c, "source": [l + "\n" for l in c["source"].split("\n")][:-1] + [c["source"].split("\n")[-1]]}
                for c in CELLS],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(HERE / "colorless_shift_study.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("->", HERE / "colorless_shift_study.ipynb")
