// jets_fastjet.C: anti-kT jets with rho*A subtraction on the ROOT files of run_h5toROOT.py
#include "HadronFileReader.h"

#include <fastjet/ClusterSequenceArea.hh>
#include <fastjet/Selector.hh>
#include <fastjet/tools/JetMedianBackgroundEstimator.hh>

#include <TFile.h>
#include <TH1D.h>

void jets_fastjet(const char *dir = "out_root", double eta_max = 1.5, double R = 0.4,
                  int max_samples = 20) {
  using namespace hadrons_root;
  HadronFileReader r(dir);
  r.set_charged_eta(eta_max);                    // charged hadrons at |eta| < eta_max only

  const fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, R);
  const fastjet::JetDefinition kt_def(fastjet::kt_algorithm, R);
  const fastjet::AreaDefinition area_def(fastjet::active_area_explicit_ghosts,
                                         fastjet::GhostedAreaSpec(eta_max, 1, 0.005));
  const fastjet::Selector rho_sel =
      (!fastjet::SelectorNHardest(2)) * fastjet::SelectorAbsRapMax(eta_max - R);
  const fastjet::Selector jet_sel =
      fastjet::SelectorAbsRapMax(eta_max - R) * fastjet::SelectorPtMin(5.0);

  TH1D h_raw("h_raw", ";p_{T,jet} (GeV);dN_{jet}/dp_{T} per event", 60, -20, 100);
  TH1D h_sub("h_sub", ";p_{T,jet} - #rho A (GeV);dN_{jet}/dp_{T} per event", 60, -20, 100);

  const std::vector<long> events = r.select_events();      // hadronized, flagged ones out
  for (long e : events) {
    const int n = std::min(r.n_oversamples(e), max_samples);
    const double w = 1.0 / (double(events.size()) * n);    // per event: mean over its samples
    for (int k = 0; k < n; ++k) {
      std::vector<fastjet::PseudoJet> parts;
      for (const Hadron &h : r.bkg_dep_frag(e, k))           // bulk (bkg + wake) + fragments
        if (h.pt() > 0.15) parts.emplace_back(h.px, h.py, h.pz, h.p());   // massless
      fastjet::ClusterSequenceArea cs(parts, jet_def, area_def);
      fastjet::JetMedianBackgroundEstimator bge(rho_sel, kt_def, area_def);
      bge.set_use_area_4vector(false);                       // rho = median of pT / A
      bge.set_particles(parts);
      const double rho = bge.rho();
      for (const fastjet::PseudoJet &jet : jet_sel(cs.inclusive_jets())) {
        h_raw.Fill(jet.pt(), w);
        h_sub.Fill(jet.pt() - rho * jet.area(), w);
      }
    }
  }
  h_raw.Scale(1.0, "width");
  h_sub.Scale(1.0, "width");
  TFile out("jets_fastjet.root", "RECREATE");
  h_raw.Write();
  h_sub.Write();
  printf("%zu events, %.1f jets per event above 5 GeV (raw pT)\n", events.size(),
         h_raw.Integral("width"));
}
