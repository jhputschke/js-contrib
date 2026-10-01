// example/analysis_root/read_hadrons.C
//
// HadronFileReader.h by example: a loop over the events and oversamples of a campaign
// converted to ROOT (prod_AuAu_0_10_jet/run_h5toROOT.py), with the hadrons of each
// oversample as vectors:
//
//   bkg            r.bkg(e, k)            bulk_bg    background (MUSIC_1)
//   bkgdep         r.bkg_dep(e, k)        bulk_jet   background + deposition (MUSIC_2)
//   full           r.bkg_dep_frag(e, k)   bulk_jet + jet_frag: the whole jet event
//   frag           r.frag(e, j)           jet_frag   fragments of the surviving partons
//
// It fills, for charged hadrons at |eta| < eta_cut, the pT spectrum of each and the
// azimuth relative to the event's leading initiator, dphi = phi - phi_lead in
// [-pi/2, 3pi/2), and writes them with the wake = bkgdep - bkg.  The pT spectra and the
// yields it prints are those of hadron_distributions.C on the same files.
//
//   root -l -b -q 'read_hadrons.C+("DIR")'                    // per event, all windows
//   root -l -b -q 'read_hadrons.C+("DIR", 2)'                 // pTHat window 2 only
//   root -l -b -q 'read_hadrons.C+("DIR", -1, true)'          // cross-section weighted
//   root -l -b -q 'read_hadrons.C+("DIR", -1, false, 1.0, "out.root", 20)'
//
// Arguments: the directory with the <stem>_hadrons.root files (and its *_campaign.root, if
// any: HadronFileReader takes the cross sections from there, else from the files); the
// pTHat window (-1: all); xsec: weigh each event with sigma_k / N_k of its
// window (N_k its selected events), else 1 / N; eta_cut; the output file; max_samples > 0:
// only the first max_samples oversamples of each event (faster, noisier).  Flagged events
// are left out (EventInfo::flagged, as hadron_distributions.C does).
//
// Normalization: every event is the mean over the oversamples it uses, so a hadron of a
// sample weighs w_e / n_used.  Each source is averaged over its own samples: bkg_dep and
// bkg_dep_frag over the jet leg's, bkg over the background's (a reused background
// counts once for every event using it), frag over the fragmentations.

#include "HadronFileReader.h"

#include <TFile.h>
#include <TH1D.h>
#include <TMath.h>
#include <TStopwatch.h>

#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace rh {

std::vector<double> log_edges(int n, double lo, double hi) {
  std::vector<double> e(n + 1);
  for (int i = 0; i <= n; ++i) e[i] = lo * std::pow(hi / lo, double(i) / n);
  return e;
}

// the histograms of one source
struct Hists {
  TH1D *pt, *dphi;
  Hists(const std::string &src, const std::string &what) {
    static const auto le = log_edges(60, 0.1, 100.0);
    pt = new TH1D(("h_pt_" + src).c_str(), (what + ";p_{T} [GeV]").c_str(), 60, le.data());
    dphi = new TH1D(("h_dphi_" + src).c_str(),
                    (what + ";#Delta#varphi to the leading initiator").c_str(), 72,
                    -TMath::PiOver2(), 1.5 * TMath::Pi());
    for (TH1D *h : {pt, dphi}) {
      h->SetDirectory(nullptr);
      h->Sumw2();
    }
  }
  // the hadrons of one sample, each with weight w
  void fill(const hadrons_root::Hadrons &hs, double phi_lead, double w) {
    for (const auto &h : hs) {
      pt->Fill(h.pt(), w);
      double d = h.phi() - phi_lead;
      while (d < -TMath::PiOver2()) d += TMath::TwoPi();
      while (d >= 1.5 * TMath::Pi()) d -= TMath::TwoPi();
      dphi->Fill(d, w);
    }
  }
};

}  // namespace rh

void read_hadrons(const char *dir, int window = -1, bool xsec = false, double eta_cut = 1.0,
                  const char *out = "read_hadrons.root", int max_samples = -1) {
  using namespace rh;
  TStopwatch sw;
  hadrons_root::HadronFileReader r(dir);
  r.set_charged_eta(eta_cut);                          // charged, |eta| < eta_cut only
  const auto events = r.select_events(window);
  if (events.empty()) {
    std::printf("read_hadrons: no events selected\n");
    return;
  }
  if (xsec && r.n_windows() == 0) {
    std::printf("read_hadrons: xsec needs the cross sections of a --pthat-bins campaign\n");
    return;
  }
  std::map<int, long> n_window;                        // selected events per window
  for (long e : events) ++n_window[r.info(e).pthat_bin];
  auto event_weight = [&](const hadrons_root::EventInfo &i) {
    if (!xsec) return 1.0 / events.size();
    return r.sigma_mb(i.pthat_bin) / n_window[i.pthat_bin];
  };
  auto used = [max_samples](int n) { return max_samples > 0 ? std::min(n, max_samples) : n; };

  const std::string what = Form("charged hadrons, |#eta| < %g", eta_cut);
  Hists bkg("bkg", what), dep("bkgdep", what), full("full", what), frag("frag", what);
  for (long e : events) {
    const auto &info = r.info(e);
    const auto *lead = info.leading();
    const double phi_lead = lead ? lead->phi() : 0.0;
    const double we = event_weight(info);
    const int nj = used(r.n_oversamples(e)), nb = used(r.n_bg_samples(e)),
              nf = used(r.n_frag(e));
    for (int k = 0; k < nj; ++k) {
      dep.fill(r.bkg_dep(e, k), phi_lead, we / nj);
      full.fill(r.bkg_dep_frag(e, k), phi_lead, we / nj);
    }
    for (int k = 0; k < nb; ++k) bkg.fill(r.bkg(e, k), phi_lead, we / nb);
    for (int j = 0; j < nf; ++j) frag.fill(r.frag(e, j), phi_lead, we / nf);
  }

  const char *ylab = xsec ? "d#sigma/dX [mb]" : "dN/dX per event";
  for (Hists *h : {&bkg, &dep, &full, &frag})
    for (TH1D *x : {h->pt, h->dphi}) {
      x->Scale(1.0, "width");
      x->GetYaxis()->SetTitle(ylab);
    }
  Hists wake("wake", what + ", bkg + deposition - bkg");
  wake.pt->Add(dep.pt, bkg.pt, 1, -1);
  wake.dphi->Add(dep.dphi, bkg.dphi, 1, -1);
  for (TH1D *x : {wake.pt, wake.dphi}) x->GetYaxis()->SetTitle(ylab);

  std::printf("read_hadrons: %zu event(s)%s, %s:\n", events.size(),
              window >= 0 ? Form(" in window %d", window) : "", what.c_str());
  for (Hists *h : {&bkg, &dep, &frag, &full, &wake})
    std::printf("  %-7s N = %10.4g   (%s)\n", h->pt->GetName() + 5,
                h->pt->Integral("width"), xsec ? "mb" : "per event");
  std::unique_ptr<TFile> fo(TFile::Open(out, "RECREATE"));
  for (Hists *h : {&bkg, &dep, &full, &frag, &wake})
    for (TH1D *x : {h->pt, h->dphi}) x->Write();
  fo->Close();
  std::printf("read_hadrons: histograms -> %s (%.1f s)\n", out, sw.RealTime());
}
