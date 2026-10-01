// example/analysis_root/hadron_distributions.C
//
// eta, phi and pT distributions of the hadrons of a campaign converted to ROOT
// (prod_AuAu_0_10_jet/run_h5toROOT.py), for the three sources of a jet event:
//
//   bkg      bulk_bg    iSS on the background's surface (MUSIC_1, no jet)
//   bkgdep   bulk_jet   iSS on the jet leg's surface (MUSIC_2): background + deposition
//   frag     jet_frag   ColorlessHadronization of the surviving partons
//
// and two sums of them: wake = bkgdep - bkg (the deposited energy at hadron level) and
// full = bkgdep + frag (the whole jet event).
//
//   root -l -b -q 'hadron_distributions.C+("DIR")'                 // per event, all windows
//   root -l -b -q 'hadron_distributions.C+("DIR", 2)'              // pTHat window 2 only
//   root -l -b -q 'hadron_distributions.C+("DIR", -1, true)'       // cross-section weighted
//   root -l -b -q 'hadron_distributions.C+("DIR", -1, false, false, 1.0, "out.root", 8)'
//
// Arguments: the directory with the <stem>_hadrons.root files (and <campaign>_campaign.root
// for the cross sections); the pTHat window (-1: all); xsec: weigh each event with its
// window's weight_mb from the campaign file (distributions in mb) instead of the same
// weight for every event (distributions per event); charged: charged hadrons only (else
// all); eta_cut: pT and phi are filled for |eta| < eta_cut, eta for all pT; the output
// file (histograms, plus a .pdf and .png of the same name); threads for RDataFrame (1: none).
//
// Normalization, as HadronFileReader does it: every event is the mean over its
// oversamples, so a hadron of event e weighs w_e / n_samples(unit); a background shared by
// several events counts once for each event using it (the sum of their w_e).  Per event:
// w_e = 1 / N (N selected events); xsec: w_e = sigma_k / N_k, the window's cross section
// (campaign file) over its selected events, so the sum over windows is d(sigma)/dX.  N_k
// counts only the events used, so leaving flagged events out does not change sigma.
// The errors are Sumw2 of these weights: the compound-Poisson errors of HadronFileReader
// (independent sampling).  wake adds the two legs' errors, which is right for
// independent sampling only: for hadronize.py --correlated files the paired error is much
// smaller (HadronFileReader.jet_minus_background).
//
// Events whose legs are not alike are left out, as wake_hadrons.py does: background over
// jet-leg freeze-out cells outside [1/1.2, 1.2] means one surface is not a freeze-out
// surface (a leg that never froze out, as the background of gridnorm job 0002).  Pass
// keep_flagged = true to keep them.
//
// frag includes what ColorlessHadronization's beam remnants make (E = sqrt(s)/6 each,
// ../analysis/README.md section 7): the remnants sit at |eta| > 5, but their strings put
// fragments at all eta, so no eta cut removes them.

#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TMath.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TSystemDirectory.h>
#include <TSystemFile.h>

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace hd {

using ROOT::RVecD;
using ROOT::RVecF;
using ROOT::RVecI;

inline bool is_charged(int pid) {
  switch (std::abs(pid)) {
    case 211: case 321: case 2212: case 3222: case 3112: case 3312: case 3334: case 11:
    case 13:
      return true;
    default:
      return false;
  }
}

// the histograms of one source
struct Hists {
  std::unique_ptr<TH1D> eta, phi, pt, ptlog;
};

std::vector<double> log_edges(int n, double lo, double hi) {
  std::vector<double> e(n + 1);
  for (int i = 0; i <= n; ++i) e[i] = lo * std::pow(hi / lo, double(i) / n);
  return e;
}

Hists make(const std::string &src, const char *what) {
  static const auto le = log_edges(60, 0.1, 100.0);
  Hists h;
  h.eta.reset(new TH1D(("h_eta_" + src).c_str(), (std::string(what) + ";#eta").c_str(),
                       200, -10, 10));
  h.phi.reset(new TH1D(("h_phi_" + src).c_str(), (std::string(what) + ";#varphi").c_str(),
                       72, -TMath::Pi(), TMath::Pi()));
  h.pt.reset(new TH1D(("h_pt_" + src).c_str(), (std::string(what) + ";p_{T} [GeV]").c_str(),
                      100, 0, 5));
  h.ptlog.reset(new TH1D(("h_ptlog_" + src).c_str(),
                         (std::string(what) + ";p_{T} [GeV]").c_str(), 60, le.data()));
  for (TH1D *x : {h.eta.get(), h.phi.get(), h.pt.get(), h.ptlog.get()}) {
    x->SetDirectory(nullptr);
    x->Sumw2();
  }
  return h;
}

std::vector<std::string> list_files(const std::string &dir, const std::string &suffix) {
  std::vector<std::string> out;
  TSystemDirectory d(dir.c_str(), dir.c_str());
  std::unique_ptr<TList> files(d.GetListOfFiles());
  if (!files) return out;
  for (TObject *o : *files) {
    const std::string name = o->GetName();
    if (name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
      out.push_back(dir + "/" + name);
  }
  std::sort(out.begin(), out.end());
  return out;
}

// the per-entry selection and weights of one tag: fills h with the hadrons of every entry,
// each weighted with wt[key] (key: the entry's event or unit)
void fill(const std::string &file, const char *tag, const char *key,
          const std::map<int, double> &wt, bool charged, double eta_cut, Hists &h) {
  ROOT::RDataFrame df(tag, file);
  auto sel = df.Filter([&wt](int k) { return wt.count(k) > 0; }, {key})
                 .Define("w", [&wt](int k) { return wt.at(k); }, {key})
                 .Define("keep", [charged](const RVecI &pid) {
                   RVecI m(pid.size());
                   for (size_t i = 0; i < pid.size(); ++i) m[i] = !charged || is_charged(pid[i]);
                   return m;
                 }, {"pid"})
                 .Define("pt_all", [](const RVecF &px, const RVecF &py) {
                   RVecD v(px.size());
                   for (size_t i = 0; i < px.size(); ++i) v[i] = std::hypot(double(px[i]), double(py[i]));
                   return v;
                 }, {"px", "py"})
                 .Define("eta_all", [](const RVecD &pt, const RVecF &pz) {
                   RVecD v(pt.size());
                   for (size_t i = 0; i < pt.size(); ++i) v[i] = std::asinh(double(pz[i]) / pt[i]);
                   return v;
                 }, {"pt_all", "pz"})
                 .Define("phi_all", [](const RVecF &px, const RVecF &py) {
                   RVecD v(px.size());
                   for (size_t i = 0; i < px.size(); ++i) v[i] = std::atan2(double(py[i]), double(px[i]));
                   return v;
                 }, {"px", "py"})
                 // eta: every selected hadron (pT > 0); pT and phi: at |eta| < eta_cut
                 .Define("all", [](const RVecI &keep, const RVecD &pt) {
                   RVecI m(pt.size());
                   for (size_t i = 0; i < pt.size(); ++i) m[i] = keep[i] && pt[i] > 0;
                   return m;
                 }, {"keep", "pt_all"})
                 .Define("mid", [eta_cut](const RVecI &all, const RVecD &eta) {
                   RVecI m(eta.size());
                   for (size_t i = 0; i < eta.size(); ++i) m[i] = all[i] && std::fabs(eta[i]) < eta_cut;
                   return m;
                 }, {"all", "eta_all"})
                 .Define("eta_v", [](const RVecD &x, const RVecI &m) { return RVecD(x[m]); },
                         {"eta_all", "all"})
                 .Define("pt_v", [](const RVecD &x, const RVecI &m) { return RVecD(x[m]); },
                         {"pt_all", "mid"})
                 .Define("phi_v", [](const RVecD &x, const RVecI &m) { return RVecD(x[m]); },
                         {"phi_all", "mid"})
                 .Define("w_eta", [](const RVecD &v, double w) { return RVecD(v.size(), w); },
                         {"eta_v", "w"})
                 .Define("w_mid", [](const RVecD &v, double w) { return RVecD(v.size(), w); },
                         {"pt_v", "w"});
  auto he = sel.Histo1D<RVecD, RVecD>(*h.eta, "eta_v", "w_eta");
  auto hp = sel.Histo1D<RVecD, RVecD>(*h.phi, "phi_v", "w_mid");
  auto ht = sel.Histo1D<RVecD, RVecD>(*h.pt, "pt_v", "w_mid");
  auto hl = sel.Histo1D<RVecD, RVecD>(*h.ptlog, "pt_v", "w_mid");
  h.eta->Add(he.GetPtr());
  h.phi->Add(hp.GetPtr());
  h.pt->Add(ht.GetPtr());
  h.ptlog->Add(hl.GetPtr());
}

}  // namespace hd

void hadron_distributions(const char *dir, int window = -1, bool xsec = false,
                          bool charged = true, double eta_cut = 1.0,
                          const char *out = "hadron_distributions.root", int threads = 1,
                          bool keep_flagged = false) {
  using namespace hd;
  if (threads > 1) ROOT::EnableImplicitMT(threads);
  if (gSystem->AccessPathName(dir)) {                 // (sic) true if it does NOT exist
    std::printf("hadron_distributions: ERROR -- directory %s does not exist\n", dir);
    return;
  }
  const auto files = list_files(dir, "_hadrons.root");
  if (files.empty()) {
    std::printf("hadron_distributions: ERROR -- no *_hadrons.root in %s (convert the "
                "campaign with prod_AuAu_0_10_jet/run_h5toROOT.py first)\n", dir);
    return;
  }
  std::vector<double> sigma_mb;
  if (xsec) {
    const auto camp = list_files(dir, "_campaign.root");
    if (camp.empty()) {
      std::printf("hadron_distributions: xsec needs the *_campaign.root of run_h5toROOT.py "
                  "in %s\n", dir);
      return;
    }
    std::unique_ptr<TFile> cf(TFile::Open(camp[0].c_str()));
    if (!cf || !cf->GetKey("windows")) {
      std::printf("hadron_distributions: ERROR -- %s has no pTHat windows (not a "
                  "--pthat-bins campaign): no cross sections for xsec\n", camp[0].c_str());
      return;
    }
    cf.reset();
    ROOT::RDataFrame w("windows", camp[0]);
    sigma_mb = *w.Take<double>("sigma_mb");
    std::printf("hadron_distributions: cross sections from %s:", camp[0].c_str());
    for (double x : sigma_mb) std::printf(" %.4g", x);
    std::printf(" mb\n");
  }
  const char *what = charged ? "charged hadrons" : "hadrons";
  Hists bkg = make("bkg", what), dep = make("bkgdep", what), frag = make("frag", what);

  // pass 1: the selected events of every file, and how many per window
  struct Ev { int e, k, bg, ns_jet, ns_bg, ns_frag; };
  std::vector<std::vector<Ev>> sel(files.size());
  std::map<int, long> n_window;
  long n_events = 0, n_flagged = 0;
  for (size_t f = 0; f < files.size(); ++f) {
    ROOT::RDataFrame ev0("events", files[f]);
    // runs without --pthat-bins have no windows: all their events are "window 0"
    const bool has_bins = ev0.HasColumn("pthat_bin");
    if (!has_bins && (window >= 0 || xsec)) {
      std::printf("hadron_distributions: ERROR -- %s has no pTHat windows (not a "
                  "--pthat-bins run): drop the window and xsec arguments\n", files[f].c_str());
      return;
    }
    auto ev = has_bins ? ev0.Define("bin_", "pthat_bin") : ev0.Define("bin_", "0");
    auto event = ev.Take<int>("event");
    auto bg_unit = ev.Take<int>("bg_unit");
    auto bin = ev.Take<int>("bin_");
    auto ns_jet = ev.Take<int>("n_samples_jet");
    auto ns_bg = ev.Take<int>("n_samples_bg");
    auto ns_frag = ev.Take<int>("n_samples_frag");
    auto cells_jet = ev.Take<int>("n_cells_jet");
    auto cells_bg = ev.Take<int>("n_cells_bg");
    for (size_t i = 0; i < event->size(); ++i) {
      const int k = (*bin)[i];
      if (window >= 0 && k != window) continue;
      if ((*ns_jet)[i] <= 0) continue;                 // not hadronized
      const double ratio = (*cells_jet)[i] > 0 ? double((*cells_bg)[i]) / (*cells_jet)[i] : 0;
      if (!keep_flagged && (ratio < 1 / 1.2 || ratio > 1.2)) {
        ++n_flagged;
        continue;
      }
      sel[f].push_back({(*event)[i], k, (*bg_unit)[i], (*ns_jet)[i], (*ns_bg)[i],
                        (*ns_frag)[i]});
      ++n_window[k];
      ++n_events;
    }
  }
  // the weight of one event: per event 1 / N; xsec sigma_k / N_k, N_k the events of window k
  // that are used (the campaign's weight_mb counts the flagged ones too)
  auto event_weight = [&](int k) {
    if (!xsec) return 1.0 / n_events;
    if (k < 0 || k >= int(sigma_mb.size())) return 0.0;
    return sigma_mb[k] / n_window[k];
  };

  // pass 2: fill
  for (size_t f = 0; f < files.size(); ++f) {
    std::map<int, double> w_jet, w_frag, w_bg;       // hadron weight per event / bg unit
    std::map<int, int> nbg;
    for (const Ev &x : sel[f]) {
      const double we = event_weight(x.k);
      w_jet[x.e] = we / x.ns_jet;
      if (x.ns_frag > 0) w_frag[x.e] = we / x.ns_frag;
      w_bg[x.bg] += we;                              // a shared background: once per event
      nbg[x.bg] = x.ns_bg;
    }
    for (auto &kv : w_bg) kv.second /= std::max(nbg[kv.first], 1);
    if (w_jet.empty()) continue;
    fill(files[f], "bulk_jet", "event", w_jet, charged, eta_cut, dep);
    fill(files[f], "bulk_bg", "unit", w_bg, charged, eta_cut, bkg);
    if (!w_frag.empty()) fill(files[f], "jet_frag", "event", w_frag, charged, eta_cut, frag);
    std::printf("  %s: %zu event(s)\n", gSystem->BaseName(files[f].c_str()), w_jet.size());
  }
  if (n_events == 0) {
    std::printf("hadron_distributions: no events selected\n");
    return;
  }

  // per bin width (the event weights already give per event, or mb)
  const double norm = 1.0;
  const char *ylab = xsec ? "d#sigma/dX [mb]" : "dN/dX per event";
  for (Hists *h : {&bkg, &dep, &frag})
    for (TH1D *x : {h->eta.get(), h->phi.get(), h->pt.get(), h->ptlog.get()}) {
      x->Scale(norm, "width");
      x->GetYaxis()->SetTitle(ylab);
    }
  Hists wake = make("wake", what), full = make("full", what);
  const TH1D *B[4] = {bkg.eta.get(), bkg.phi.get(), bkg.pt.get(), bkg.ptlog.get()};
  const TH1D *D[4] = {dep.eta.get(), dep.phi.get(), dep.pt.get(), dep.ptlog.get()};
  const TH1D *F[4] = {frag.eta.get(), frag.phi.get(), frag.pt.get(), frag.ptlog.get()};
  TH1D *W[4] = {wake.eta.get(), wake.phi.get(), wake.pt.get(), wake.ptlog.get()};
  TH1D *T[4] = {full.eta.get(), full.phi.get(), full.pt.get(), full.ptlog.get()};
  for (int i = 0; i < 4; ++i) {
    W[i]->Add(D[i], B[i], 1, -1);
    T[i]->Add(D[i], F[i], 1, 1);
    W[i]->GetYaxis()->SetTitle(ylab);
    T[i]->GetYaxis()->SetTitle(ylab);
  }

  // summary: yields at |eta| < eta_cut per event (or in mb)
  auto yield = [&](const TH1D *h) { return h->Integral("width"); };
  std::printf("hadron_distributions: %ld event(s)%s, %ld flagged and left out; %s, |eta| < %g:\n",
              n_events, window >= 0 ? Form(" in window %d", window) : "", n_flagged, what,
              eta_cut);
  std::printf("  %-7s N = %10.4g   (%s)\n", "bkg", yield(B[2]), xsec ? "mb" : "per event");
  std::printf("  %-7s N = %10.4g\n", "bkgdep", yield(D[2]));
  std::printf("  %-7s N = %10.4g\n", "frag", yield(F[2]));
  std::printf("  %-7s N = %10.4g\n", "wake", yield(W[2]));

  // output: histograms + one page of plots
  std::unique_ptr<TFile> fo(TFile::Open(out, "RECREATE"));
  for (Hists *h : {&bkg, &dep, &frag, &wake, &full})
    for (TH1D *x : {h->eta.get(), h->phi.get(), h->pt.get(), h->ptlog.get()}) x->Write();
  fo->Close();

  // one page: columns eta, phi, pT; rows bkg and bkg + deposition, jet frag, wake
  gStyle->SetOptStat(0);
  TCanvas c("c", "hadron distributions", 1500, 1300);
  c.Divide(3, 3);
  const TH1D *rows[3][2][3] = {{{B[0], B[1], B[3]}, {D[0], D[1], D[3]}},
                               {{F[0], F[1], F[3]}, {nullptr, nullptr, nullptr}},
                               {{W[0], W[1], W[3]}, {nullptr, nullptr, nullptr}}};
  const int col[3][2] = {{kBlue + 1, kGreen + 2}, {kRed + 1, 0}, {kBlack, 0}};
  const char *lab[3][2] = {{"bkg (bulk_bg)", "bkg + deposition (bulk_jet)"},
                           {"jet frag (jet_frag)", nullptr},
                           {"wake = bkg + deposition - bkg", nullptr}};
  for (int r = 0; r < 3; ++r)
    for (int p = 0; p < 3; ++p) {
      c.cd(3 * r + p + 1);
      gPad->SetLeftMargin(0.13);
      if (p == 2) {
        gPad->SetLogx();
        if (r < 2) gPad->SetLogy();
      }
      if (r == 1 && p == 0) gPad->SetLogy();        // frag: beam remnants at |eta| > 5
      const int n = rows[r][1][p] ? 2 : 1;
      double ymax = 0, ymin = 1e300;
      for (int s = 0; s < n; ++s) {
        ymax = std::max(ymax, rows[r][s][p]->GetMaximum());
        ymin = std::min(ymin, rows[r][s][p]->GetMinimum(0));
      }
      TLegend *leg = new TLegend(0.16, 0.80, 0.70, 0.89);
      leg->SetBorderSize(0);
      leg->SetFillStyle(0);
      leg->SetTextSize(0.035);
      for (int s = 0; s < n; ++s) {
        auto *h = static_cast<TH1D *>(rows[r][s][p]->Clone());
        h->SetLineColor(col[r][s]);
        h->SetLineWidth(2);
        h->SetTitle(p == 0 ? what : Form("%s, |#eta| < %g", what, eta_cut));
        const bool logy = gPad->GetLogy();
        if (r == 2) {
          h->SetMarkerStyle(20);
          h->SetMarkerSize(0.5);
          h->SetMaximum(1.4 * ymax);
        } else if (logy) {
          h->SetMinimum(0.5 * ymin);
          h->SetMaximum(30 * ymax);
        } else {
          h->SetMinimum(0);
          h->SetMaximum(1.3 * ymax);
        }
        h->Draw(r == 2 ? (s ? "e same" : "e") : (s ? "hist same" : "hist"));
        leg->AddEntry(h, lab[r][s], r == 2 ? "lep" : "l");
      }
      leg->Draw();
    }
  std::string pdf = out;
  if (pdf.size() > 5 && pdf.substr(pdf.size() - 5) == ".root") pdf = pdf.substr(0, pdf.size() - 5);
  c.SaveAs((pdf + ".png").c_str());
  pdf += ".pdf";
  c.SaveAs(pdf.c_str());
  std::printf("hadron_distributions: histograms -> %s, plots -> %s (and .png)\n", out,
              pdf.c_str());
}
