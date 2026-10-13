// oo_dndeta.C: charged dN/deta of the O+O bulk, each event the mean over its oversamples,
// events weighted with Ncoll (the hard-process rate); files of hadrons_to_root.py + events
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <TFile.h>
#include <algorithm>
#include <cmath>

using ROOT::RVecD;
using ROOT::RVecF;
using ROOT::RVecI;

void oo_dndeta(const char *stem, const char *out = "oo_dndeta.root") {
  ROOT::RDataFrame ev("events", std::string(stem) + "_events.root");
  auto unit = *ev.Take<Long64_t>("unit");
  auto ncoll = *ev.Take<Long64_t>("glauber_ncoll");
  auto nsamp = *ev.Take<Long64_t>("n_samples_bg");
  std::vector<double> w(1 + *std::max_element(unit.begin(), unit.end()), 0.0);
  double wsum = 0;
  for (size_t i = 0; i < unit.size(); ++i)
    if (nsamp[i] > 0) { w[unit[i]] = ncoll[i] / double(nsamp[i]); wsum += ncoll[i]; }

  const std::vector<int> charged{211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13};
  ROOT::RDataFrame bulk("bulk_bg", std::string(stem) + "_bulk_bg.root");
  auto h = bulk
    .Define("eta_ch", [&](const RVecI &pid, const RVecF &px, const RVecF &py, const RVecF &pz) {
        RVecD eta;
        for (size_t i = 0; i < pid.size(); ++i)
          if (std::find(charged.begin(), charged.end(), std::abs(pid[i])) != charged.end())
            eta.push_back(std::asinh(pz[i] / std::hypot(px[i], py[i])));
        return eta;
      }, {"pid", "px", "py", "pz"})
    .Define("w", [&](const RVecD &eta, int u) { return RVecD(eta.size(), w[u] / wsum); },
            {"eta_ch", "unit"})
    .Histo1D({"dndeta", ";#eta;dN_{ch}/d#eta per event", 48, -6, 6}, "eta_ch", "w");
  h->Scale(1.0, "width");
  printf("dN_ch/deta at eta = 0: %.2f\n", h->GetBinContent(h->FindBin(0.1)));
  TFile f(out, "RECREATE");
  h->Write();
}
