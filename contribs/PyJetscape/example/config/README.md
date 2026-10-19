# config — the productions' inputs

One user XML per setup, the hadronization settings they share, and the output-grid YAMLs. The
drivers in `../prod_*` take their defaults from here (`run_prod.CONFIG`); pass another file with
`--user-xml`, `--hadronize-xml` or `--grid`. In the image this folder is
`/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/config/`.

| file | setup | driver |
|---|---|---|
| `AuAu_MCGlauber_MUSIC_0_10_fast.xml` | Au+Au 200 GeV 0–10 %, hydro only | `../prod_AuAu_0_10/run_prod.py` |
| `AuAu_MCGlauber_MUSIC_0_10_jet.xml` | Au+Au 200 GeV 0–10 % with a jet (PythiaGun, Matter + LBT, liquefier, two MUSIC legs) | `../prod_AuAu_0_10_jet/run_prod_jet.py`, `run_prod_vacuum.py` (its pp reference) |
| `PbPb_0_10_jet.xml` | Pb+Pb 5.02 TeV 0–10 % with a jet: the Au+Au jet XML through [README_2stage §9](../../../../docs/README_2stage.md)'s recipe | `run_prod_jet.py` |
| `PbPb_0_30_ZJet.xml` | Pb+Pb 5.02 TeV 0–30 % Z+jet (the CMS class): `PbPb_0_10_jet.xml` with `b_max` 8.57 fm and a Z+jet `<PythiaGun>` (`<trigger>Z`; [PLAN_ZJET_CMS10x.md](../../../../docs/Plans/PLAN_ZJET_CMS10x.md)) | `run_prod_jet.py`, `run_prod_vacuum.py`, `utils/zjet/` |
| `OO_MCGlauber_MUSIC_isr.xml` | O+O with ISR (PythiaIsrGun + iMATTER, iColored) | `../prod_OO_isr/run_prod_oo_isr.py` |
| `hadronize.xml` | iSS and Colorless settings of every system; `eCMforHadronization` is set to the job's collision energy (`<PythiaGun><eCM>`) by every driver | `hadronize.py`, `run_prod_jet.py --hadrons-only / --validate-inline`, `run_prod_vacuum.py`; O+O: its iSS block |
| `grid_fno.yaml` | output grid, Au+Au (default of `run_prod.py`, `run_prod_jet.py`) | |
| `grid_x10_eta2p5.yaml` | `grid_fno.yaml` cut to \|η_s\| ≤ 2.5 | |
| `grid_PbPb.yaml` | output grid, Pb+Pb (pass it with `--grid`) | |
| `grid_oo.yaml` | output grid, O+O (default of `run_prod_oo_isr.py`) | |

Each job writes the XML it ran next to its output (`<stem>.xml`) and stores it in its files
(`prod_user_xml`), so a file always records what it was made with.
