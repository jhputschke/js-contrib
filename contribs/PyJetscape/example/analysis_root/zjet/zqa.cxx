// example/analysis_root/zjet/zqa.cxx
//
// The QA map step in C++ (the twin of example/analysis/zjet/qa.py map): per production
// file, from run_h5toROOT.py's ROOT files, <out>/<stem>_qa.root with
//
//   qa_events   per event with a Z: event, file, bg, pthat_bin, n_coll, w_s (window cross
//               section / the window's events in the directory), n_samples, the Z and its
//               leptons, the total charged multiplicity at |eta| < 0.5 and < 2.4 (all p_T) of
//               the full event (bulk_jet + jet_frag: mean and variance over the oversamples),
//               of the background and of the fragments, the Z's leptons found among the
//               fragments (mean, minimum over the fragmentations; 2 expected)
//   qa_hist     one entry: per source (bulk_jet, bulk_bg, jet_frag; vacuum: jet_frag)
//               h_<source>_<pt|eta|phi>_<charged|all>_<n|sc> and s_<source>_<n|sc> (species):
//               the per-event mean over the oversamples, summed unweighted (n) and with
//               w_s x N_coll (sc); the Z's leptons removed from the fragments
//
// Extra hard scatterings (files with extra_frag): the full event's multiplicity is bulk_jet +
// jet_frag + extra_frag, extra_frag gets its own histograms (h_extra_frag_*, s_extra_frag_*)
// and qa_events nch05_extra, nch24_extra (the mean over its fragmentations) and n_extra (the
// extras' number); --extras 0 leaves them out (the outputs of the files without extras).
//
//   ./zqa <root dir> <out dir> [--files i:j] [--extras 0]
//
// `python qa.py plot --pbpb <out dir>` draws them (as the Python map step's .npz);
// `python check_cxx.py qa <python dir> <C++ dir>` compares the two map steps.

#include "ZJetCommon.h"

#include <TFile.h>
#include <TTree.h>

#include <chrono>
#include <cstdio>
#include <map>

using namespace zjet;

namespace {

// the bins and the species of qa.py
std::vector<double> concat(std::initializer_list<std::vector<double>> parts) {
  std::vector<double> out;
  for (const auto &p : parts) out.insert(out.end(), p.begin(), p.end());
  return out;
}
std::vector<double> drop_last(std::vector<double> v) {
  v.pop_back();
  return v;
}
const std::vector<double> &edges(const std::string &var) {
  static const std::vector<double> pt =
      concat({drop_last(linspace(0, 2, 21)), drop_last(linspace(2, 5, 13)),
              drop_last(linspace(5, 10, 11)), drop_last(linspace(10, 30, 11)),
              std::vector<double>{30, 40, 60, 100}});
  static const std::vector<double> eta = linspace(-2.6, 2.6, 53);
  static const std::vector<double> phi = linspace(-M_PI, M_PI, 37);
  return var == "pt" ? pt : var == "eta" ? eta : phi;
}
const std::vector<int> SPECIES = {211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13, 22,
                                  111, 130, 310, 311, 2112, 3122, 3212, 3322, 221};
int species_index(int pid) {
  const int a = std::abs(pid);
  for (size_t i = 0; i < SPECIES.size(); ++i)
    if (a == SPECIES[i]) return int(i);
  return int(SPECIES.size());
}
const char *VARS[3] = {"pt", "eta", "phi"};
const char *CHARGES[2] = {"charged", "all"};

using Hists = std::map<std::string, std::vector<double>>;   // (var_charge or "species") -> counts

double value(const Hadron &h, int var) {
  return var == 0 ? h.pt() : var == 1 ? h.eta() : h.phi();
}

// the summed counts of one source's samples of one event, per-event mean (/ n_samples)
Hists source_hists(const std::vector<Hadrons> &samples, const ZRecord *z) {
  Hists out;
  for (int c = 0; c < 2; ++c)
    for (int v = 0; v < 3; ++v)
      out[std::string(VARS[v]) + "_" + CHARGES[c]].assign(edges(VARS[v]).size() - 1, 0.0);
  out["species"].assign(SPECIES.size() + 1, 0.0);
  for (const Hadrons &hs : samples)
    for (const Hadron &h : hs) {
      if (z && z->is_lepton(h)) continue;
      for (int c = 0; c < 2; ++c) {
        if (c == 0 && !h.charged()) continue;
        for (int v = 0; v < 3; ++v) {
          const int j = bin_index(value(h, v), edges(VARS[v]));
          if (j >= 0) out[std::string(VARS[v]) + "_" + CHARGES[c]][j] += 1.0;
        }
      }
      out["species"][species_index(h.pid)] += 1.0;
    }
  const double n = double(samples.size());
  for (auto &p : out)
    for (auto &x : p.second) x /= n;
  return out;
}

// charged particles at |eta| < eta_max per sample
std::vector<double> nch(const std::vector<Hadrons> &samples, const ZRecord *z, double eta_max) {
  std::vector<double> out;
  for (const Hadrons &hs : samples) {
    double n = 0;
    for (const Hadron &h : hs)
      if (h.charged() && std::fabs(h.eta()) < eta_max && !(z && z->is_lepton(h))) n += 1;
    out.push_back(n);
  }
  return out;
}
double mean(const std::vector<double> &v) {
  double s = 0;
  for (double x : v) s += x;
  return v.empty() ? 0 : s / v.size();
}
double var1(const std::vector<double> &v) {        // numpy var(ddof=1)
  if (v.size() < 2) return 0;
  const double m = mean(v);
  double s = 0;
  for (double x : v) s += (x - m) * (x - m);
  return s / (v.size() - 1);
}

int run(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <root dir> <out dir> [--files i:j] [--extras 0]\n",
                 argv[0]);
    return 2;
  }
  const std::string in = argv[1], out = argv[2];
  const auto o = options(argc, argv, 3);
  const bool extras = !o.count("--extras") || std::stoi(o.at("--extras")) != 0;
  gSystem->mkdir(out.c_str(), true);
  HadronFileReader r(in);
  r.set_extras(extras);
  const long n = r.n_events();
  std::vector<long> nk(std::max(r.n_windows(), 1), 0);
  for (long e = 0; e < n; ++e)
    if (r.info(e).pthat_bin >= 0) ++nk[r.info(e).pthat_bin];
  const auto bgid = background_ids(r);
  const auto range = file_range(o, r.n_files());

  for (int f = range.first; f < range.second; ++f) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::string name = stem_of(r.file(f));
    bool vacuum = false;
    for (long e = r.event_offset(f); e < n && r.info(e).file_index == f; ++e)
      vacuum = r.info(e).vacuum;
    std::vector<std::string> sources = vacuum ? std::vector<std::string>{"jet_frag"}
                                              : std::vector<std::string>{"bulk_jet", "bulk_bg",
                                                                         "jet_frag"};
    const bool xtra = extras && !vacuum && r.has_extras(f);   // extra hard scatterings
    if (xtra) sources.push_back("extra_frag");
    std::map<std::string, std::vector<double>> H;
    for (const auto &s : sources) {
      for (int v = 0; v < 3; ++v)
        for (int c = 0; c < 2; ++c)
          for (const char *w : {"n", "sc"})
            H["h_" + s + "_" + VARS[v] + "_" + CHARGES[c] + "_" + w].assign(
                edges(VARS[v]).size() - 1, 0.0);
      for (const char *w : {"n", "sc"}) H["s_" + s + "_" + w].assign(SPECIES.size() + 1, 0.0);
    }
    auto add = [&](const std::string &src, const Hists &h, double w_sc) {
      for (const auto &p : h) {
        const std::string base = p.first == "species" ? "s_" + src : "h_" + src + "_" + p.first;
        auto &hn = H[base + "_n"];
        auto &hs = H[base + "_sc"];
        for (size_t i = 0; i < p.second.size(); ++i) {
          hn[i] += p.second[i];
          hs[i] += w_sc * p.second[i];
        }
      }
    };
    // the per-event table
    const std::string path = out + "/" + name + "_qa.root";
    TFile fo((path + ".part").c_str(), "RECREATE");
    TTree te("qa_events", "QA per event");
    te.SetDirectory(&fo);
    Long64_t c_event = 0, c_bg = 0;
    int c_file = f, c_bin = -1, c_ncoll = -1, c_ns = 0, c_pid1 = 0, c_pid2 = 0;
    double d[21];
    const char *dn[21] = {"w_s", "z_pt", "z_y", "z_phi", "z_m", "lep1_pt", "lep1_eta",
                          "lep1_phi", "lep2_pt", "lep2_eta", "lep2_phi", "nch05_full",
                          "nch24_full", "nch24_full_var", "nch05_bg", "nch24_bg", "nch05_frag",
                          "nch24_frag", "nzlep", "nzlep_min", "spare"};
    te.Branch("event", &c_event);
    te.Branch("file", &c_file);
    te.Branch("bg", &c_bg);
    te.Branch("pthat_bin", &c_bin);
    te.Branch("n_coll", &c_ncoll);
    te.Branch("n_samples", &c_ns);
    te.Branch("lep1_pid", &c_pid1);
    te.Branch("lep2_pid", &c_pid2);
    for (int i = 0; i < 20; ++i) te.Branch(dn[i], &d[i]);
    double x05m = 0, x24m = 0;                     // with extras: their multiplicity, number
    int c_nextra = 0;
    if (xtra) {
      te.Branch("nch05_extra", &x05m);
      te.Branch("nch24_extra", &x24m);
      te.Branch("n_extra", &c_nextra);
    }
    struct BgQA { Hists h; double n05, n24; };
    std::map<int, BgQA> bg_cache;
    long n_ev = 0;
    for (long e = r.event_offset(f); e < n && r.info(e).file_index == f; ++e) {
      const EventInfo &info = r.info(e);
      const ZRecord &z = info.z;
      if (!z.valid) continue;
      const int nf = info.n_samples_frag;
      if (nf == 0) continue;
      std::vector<Hadrons> F;
      for (int j = 0; j < nf; ++j) F.push_back(r.frag(e, j));
      std::vector<Hadrons> J, X;
      if (!info.vacuum) {
        if (info.n_samples_jet == 0) continue;
        for (int k = 0; k < info.n_samples_jet; ++k) J.push_back(r.bkg_dep(e, k));
        if (xtra)
          for (int j = 0; j < info.n_samples_extra; ++j) X.push_back(r.extra(e, j));
        if (!bg_cache.count(info.bg_unit)) {
          std::vector<Hadrons> B;
          for (int k = 0; k < info.n_samples_bg; ++k) B.push_back(r.bkg(e, k));
          bg_cache[info.bg_unit] = {source_hists(B, nullptr), mean(nch(B, nullptr, 0.5)),
                                    mean(nch(B, nullptr, 2.4))};
        }
      }
      const double w_s = info.pthat_bin >= 0 && r.n_windows() > 0
                             ? r.sigma_mb(info.pthat_bin) / std::max(nk[info.pthat_bin], 1L)
                             : 1.0;
      const double w_sc = w_s * (info.n_coll >= 0 ? info.n_coll : 1.0);
      // the order of qa.py: the fragments, then bulk_jet, then the background
      add("jet_frag", source_hists(F, &z), w_sc);
      if (!info.vacuum) add("bulk_jet", source_hists(J, nullptr), w_sc);
      if (!X.empty()) add("extra_frag", source_hists(X, nullptr), w_sc);
      std::vector<double> zl;
      for (const Hadrons &hs : F) {
        double k = 0;
        for (const Hadron &h : hs) k += z.is_lepton(h);
        zl.push_back(k);
      }
      const auto f05 = nch(F, &z, 0.5), f24 = nch(F, &z, 2.4);
      std::vector<double> full05, full24;
      double b05 = 0, b24 = 0;
      if (!info.vacuum) {
        const BgQA &b = bg_cache[info.bg_unit];
        add("bulk_bg", b.h, w_sc);
        b05 = b.n05;
        b24 = b.n24;
        const auto j05 = nch(J, nullptr, 0.5), j24 = nch(J, nullptr, 2.4);
        for (size_t k = 0; k < J.size(); ++k) {
          full05.push_back(j05[k] + f05[k % nf]);
          full24.push_back(j24[k] + f24[k % nf]);
        }
        x05m = x24m = 0;
        if (!X.empty()) {                          // the extras belong to the event
          const auto x05 = nch(X, nullptr, 0.5), x24 = nch(X, nullptr, 2.4);
          for (size_t k = 0; k < J.size(); ++k) {
            full05[k] += x05[k % X.size()];
            full24[k] += x24[k % X.size()];
          }
          x05m = mean(x05);
          x24m = mean(x24);
        }
        c_nextra = 0;
        for (const auto &sc : info.scatterings) c_nextra += sc.scatter > 0;
      } else {
        full05 = f05;
        full24 = f24;
      }
      c_event = e; c_bg = bgid[e]; c_bin = info.pthat_bin; c_ncoll = info.n_coll;
      c_ns = int(full24.size()); c_pid1 = z.lep_pid[0]; c_pid2 = z.lep_pid[1];
      const double vals[20] = {w_s, z.pt, z.y, z.phi, z.m, z.lep_pt[0], z.lep_eta[0],
                               z.lep_phi[0], z.lep_pt[1], z.lep_eta[1], z.lep_phi[1],
                               mean(full05), mean(full24), var1(full24), b05, b24, mean(f05),
                               mean(f24), mean(zl), *std::min_element(zl.begin(), zl.end())};
      for (int i = 0; i < 20; ++i) d[i] = vals[i];
      te.Fill();
      ++n_ev;
    }
    // the histograms: one entry
    TTree th("qa_hist", "QA histograms");
    th.SetDirectory(&fo);
    for (auto &p : H)
      th.Branch(p.first.c_str(), p.second.data(),
                Form("%s[%zu]/D", p.first.c_str(), p.second.size()));
    th.Fill();
    fo.cd();
    te.Write();
    th.Write();
    fo.Close();
    gSystem->Rename((path + ".part").c_str(), path.c_str());
    const double dt =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s: %ld events, %.1f s -> %s\n", name.c_str(), n_ev, dt, path.c_str());
  }
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &x) {
    std::fprintf(stderr, "zqa: %s\n", x.what());
    return 1;
  }
}
