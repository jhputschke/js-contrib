# analysis_snippets — the scripts behind the "(tested)" code of the analysis guides

The code blocks marked *(tested)* in [`README_analysis.md`](../README_analysis.md) and
[`README_OO.md`](../README_OO.md) were checked with the scripts here, on 2026-10-13 in
`js_fno` (h5py 3, uproot 5.7, scikit-hep fastjet 3.5.1, FastJet 3.5.1 with fastjet-contrib
1.101, ROOT 6.36). They are references to start an analysis from and the way to re-check the
guides after a change, not production tools.

| file | what | data it ran on |
|---|---|---|
| [`run_doc_blocks.py`](run_doc_blocks.py) | runs the ```` ```python ```` blocks of a guide in one namespace, with the guide's placeholder paths replaced (`--sub OLD=NEW`), all of them or only the *(tested)* ones (`--tested`); `--list` shows them | both guides (below) |
| [`test_prov.py`](test_prov.py) | the production settings of every file of a campaign (late time step, grid, reuse) | `prod_AuAu_0_10_jet/out` |
| [`test_hydro.py`](test_hydro.py) | a pair file: grid axes, the wake Δe, the droplets, e → T with MUSIC's table; a particlize file | the seed-1 pair and particlize files |
| [`test_had.py`](test_had.py) | `HadronFileReader`: `hist`, `total`, `jet_minus_background` relative to the leading initiator, single events, precision and selection | the seed-1 hadron files |
| [`test_fj.py`](test_fj.py) | FastJet in Python: classic interface with areas and ρA, and the array (awkward) interface | the seed-1 hadron files |
| [`jets_fastjet.C`](jets_fastjet.C), [`jets_main.cxx`](jets_main.cxx) | FastJet in C++ on the ROOT files through `HadronFileReader.h`: anti-kT jets, raw and ρA-subtracted spectra (README_analysis §9) | `*_hadrons.root` of `run_h5toROOT.py` |
| [`contrib_test.cxx`](contrib_test.cxx) | that fjcontrib links (SoftDrop, ConstituentSubtractor) from conda-forge's `fastjet-contrib` | none |
| [`test_oo.py`](test_oo.py) | O+O: the campaign loader, cross-section and Ncoll weights, the bookkeeping check, dN_ch/dη, v₂{2} from different oversamples, jets, the initiators | `prod_OO_isr/hard_sub_compare/sub` |
| [`test_oo_root.py`](test_oo_root.py) | O+O to ROOT: `hadrons_to_root.py` per tag and the `events` table (writes next to the inputs: use a copy) | a copy of `sub/` |
| [`oo_dndeta.C`](oo_dndeta.C) | RDataFrame on those files: the Ncoll-weighted bulk dN_ch/dη, each event the mean over its oversamples (22.54 at η = 0.1 for `OO_isr_seed0101`, the same as from the HDF5 file) | the output of `test_oo_root.py` |

## Running them

From the js-contrib root, with PyJetscape importable (the analysis venv, or `js_fno` after
`conda activate js_fno`):

```bash
O=contribs/PyJetscape/example/prod_AuAu_0_10_jet/out
python docs/analysis_snippets/test_prov.py $O
MUSIC_EOS_TABLE=/path/to/EOS/hotQCD/hrg_hotqcd_eos_binary.dat python docs/analysis_snippets/test_hydro.py $O
python docs/analysis_snippets/test_had.py $O                      # an optional 2nd argument: the stem
python docs/analysis_snippets/test_fj.py $O
python docs/analysis_snippets/test_oo.py contribs/PyJetscape/example/prod_OO_isr/hard_sub_compare/sub
cp -r contribs/PyJetscape/example/prod_OO_isr/hard_sub_compare/sub /scratch/oo_copy
python docs/analysis_snippets/test_oo_root.py /scratch/oo_copy \
    contribs/PyJetscape/example/prod_AuAu_0_10_jet/root_export/hadrons_to_root.py

# C++ (conda activate an env with ROOT and fastjet-cxx / fastjet-contrib first)
AR=contribs/PyJetscape/example/analysis_root
cd docs/analysis_snippets
root -l -b -q -e "gSystem->AddIncludePath(\"-I../../$AR -I$CONDA_PREFIX/include\"); gSystem->Load(\"libfastjet\"); gSystem->Load(\"libfastjettools\")" \
  'jets_fastjet.C+("../../contribs/PyJetscape/example/prod_AuAu_0_10_jet/out")'
g++ -O2 -std=c++17 -I../../$AR jets_main.cxx $(root-config --cflags --libs) \
    $(fastjet-config --cxxflags) $(fastjet-config --libs) -lROOTNTuple -o jets
g++ -std=c++17 contrib_test.cxx $(fastjet-config --cxxflags) $(fastjet-config --libs) -lfastjetcontribfragile -o contrib_test
root -l -b -q 'oo_dndeta.C+("/scratch/oo_copy/OO_isr_seed0101")'
```

**The guides' blocks** (the way to re-check them after editing a guide):

```bash
python docs/analysis_snippets/run_doc_blocks.py docs/README_OO.md --tested \
    --sub /path/to/OO_c1=/scratch/oo_copy
O=contribs/PyJetscape/example/prod_AuAu_0_10_jet/out
MUSIC_EOS_TABLE=… python docs/analysis_snippets/run_doc_blocks.py docs/README_analysis.md --tested \
    --sub /path/to/campaign=$O --sub AuAu_0_10_jet_c1_0001=$O/AuAu_0_10_jet_seed0001 \
    --sub pth10-40_c1_campaign.root=$O/campaign_campaign.root
```

On 2026-10-13 both ran through: 7 tested blocks of each guide. The seed-1 files have no
pT̂ windows, so the window blocks are not marked tested; `test_oo.py` covers windows on the
O+O campaign. The scripts take the seed-1 file names (`AuAu_0_10_jet_seed0001`) by default;
ACLiC writes its `.so`, `.d` and `.pcm` files next to the macros (not in git).
