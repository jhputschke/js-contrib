// example/analysis_root/zjet/zjets.cxx
//
// The map step of the Z+jet analysis (CMS 1702.01060) in C++ with fastjet: per production
// file the jets of every oversample of the Z events, from run_h5toROOT.py's ROOT files, into
// <out>/<stem>_jets.root.  The final stage (selection, the optional CMS smearing, HEPData,
// chi^2, jackknife, figures) is example/analysis/zjet_test/zjet_jets.py; zjet_jets_map.py
// is the same map step in Python (check_cxx.py jets compares the two).
//
//   ./zjets <root dir> <out dir> [--seed 1] [--files i:j] [--samples N] [--cones 4]
//           [--ptmin 5]
//
// Events: a Z with 60 < m < 120 GeV and p_T^Z > 30 GeV (the final stage applies the paper's
// cuts).  Samples: k = 0 .. min(n, --samples) - 1, n the oversamples (vacuum: the
// fragmentations).  Particles of sample k: bulk_jet k + jet_frag (k mod n_frag) (vacuum:
// jet_frag k), all final particles but neutrinos and the Z's two leptons.
//
// Jets: anti-k_T R = 0.3, E scheme, p_T > --ptmin, |eta| < 2.  Pb+Pb: active areas with
// explicit ghosts on a fixed grid (|y| < 2.6, area 0.01, no scatter: the same jets as the
// Python map step) and rho(jet) the median p_T / area of the k_T R = 0.4 jets with |y| < 2.2
// in a strip |y - y_jet| < 1; p_T = p_T,raw - rho A.  Vacuum: no subtraction.
// Embedding (Pb+Pb): the fragments of sample k alone (reference jets, p_T > 10 GeV,
// |eta| < 2) and embedded in bulk_bg sample k (no wake), clustered and subtracted as above;
// each reference jet with the embedded jet nearest in Delta R (NaN beyond 0.15).
// Random cones (Pb+Pb): --cones cones per sample in bulk_bg k at |eta| < 1.6, positions
// from zjet_wake.key_uniform; delta p_T = sum p_T within Delta R < 0.3 - rho pi 0.3^2.
// Background jets (Pb+Pb, tree bg_jets): every hydro background's bulk_bg samples k = 0 ..
// min(n, --samples) - 1 clustered and subtracted as above, the model's minimum-bias events
// without any hard process: the final stage's B (jets not from the Z's interaction), from the
// Z's own background or mixed from others of similar N_coll.

#include "ZJetCommon.h"

#include <TFile.h>
#include <TTree.h>

#include <fastjet/AreaDefinition.hh>
#include <fastjet/ClusterSequence.hh>
#include <fastjet/ClusterSequenceArea.hh>
#include <fastjet/Selector.hh>
#include <fastjet/tools/JetMedianBackgroundEstimator.hh>

#include <chrono>
#include <cstdio>
#include <limits>

using namespace zjet;
namespace fj = fastjet;

namespace {

constexpr double R_JET = 0.3, ETA_JET = 2.0, PT_REF = 10.0, DR_MATCH = 0.15;
constexpr double ETA_RC = 1.6;
constexpr std::uint64_t RC_SALT = 0x5243000000000000ULL;     // "RC": away from k_one's salt 0
const double NaN = std::numeric_limits<double>::quiet_NaN();

bool keep(const Hadron &h, const ZRecord &z) {
  const int a = std::abs(h.pid);
  return a != 12 && a != 14 && a != 16 && !z.is_lepton(h);
}

void to_pseudojets(const Hadrons &hs, const ZRecord &z, std::vector<fj::PseudoJet> &out) {
  for (const Hadron &h : hs)
    if (keep(h, z)) out.emplace_back(double(h.px), double(h.py), double(h.pz), double(h.E));
}

struct Clusterer {
  fj::JetDefinition akt{fj::antikt_algorithm, R_JET}, kt{fj::kt_algorithm, 0.4};
  fj::AreaDefinition area{fj::active_area_explicit_ghosts,
                          fj::GhostedAreaSpec(2.6, 1, 0.01, 0.0, 0.0)};
  fj::Selector rho_sel = fj::SelectorAbsRapMax(2.2) * fj::SelectorStrip(1.0);
};

// the jets of a particle list: raw p_T, subtracted p_T, eta, phi, area, rho
struct Jet {
  double pt_raw, pt, eta, phi, area, rho;
};

std::vector<Jet> jets_of(const std::vector<fj::PseudoJet> &parts, bool subtract,
                         double ptmin, Clusterer &c) {
  std::vector<Jet> out;
  if (!subtract) {
    fj::ClusterSequence cs(parts, c.akt);
    for (const auto &j : fj::sorted_by_pt(cs.inclusive_jets(ptmin)))
      if (std::fabs(j.eta()) < ETA_JET) out.push_back({j.pt(), j.pt(), j.eta(), j.phi_std(), 0, 0});
    return out;
  }
  fj::JetMedianBackgroundEstimator bge(c.rho_sel, c.kt, c.area);
  bge.set_particles(parts);
  fj::ClusterSequenceArea cs(parts, c.akt, c.area);
  for (const auto &j : fj::sorted_by_pt(cs.inclusive_jets(1.0))) {
    if (!(std::fabs(j.eta()) < ETA_JET)) continue;
    const double rho = bge.rho(j), a = j.area(), pt = j.pt() - rho * a;
    if (pt > ptmin) out.push_back({j.pt(), pt, j.eta(), j.phi_std(), a, rho});
  }
  return out;
}

int run(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <root dir> <out dir> [--seed 1] [--files i:j] "
                         "[--samples N] [--cones 4] [--ptmin 5]\n", argv[0]);
    return 2;
  }
  const std::string in = argv[1], out = argv[2];
  const auto o = options(argc, argv, 3);
  const long seed = o.count("--seed") ? std::stol(o.at("--seed")) : 1;
  const int max_samples = o.count("--samples") ? std::stoi(o.at("--samples")) : 1 << 30;
  const int n_cones = o.count("--cones") ? std::stoi(o.at("--cones")) : 4;
  const double ptmin = o.count("--ptmin") ? std::stod(o.at("--ptmin")) : 5.0;
  gSystem->mkdir(out.c_str(), true);
  fj::ClusterSequence::print_banner();

  HadronFileReader r(in);
  const long n = r.n_events();
  // sigma_k / N_k over every event of the window (zjet_wake.event_weights before N_coll)
  std::vector<long> nk(std::max(r.n_windows(), 1), 0);
  for (long e = 0; e < n; ++e)
    if (r.info(e).pthat_bin >= 0) ++nk[r.info(e).pthat_bin];
  const auto bgid = background_ids(r);
  const auto range = file_range(o, r.n_files());
  Clusterer c;

  for (int f = range.first; f < range.second; ++f) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::string name = stem_of(r.file(f));
    const std::string path = out + "/" + name + "_jets.root";
    TFile fo((path + ".part").c_str(), "RECREATE");
    // per event
    Long64_t ev = 0, bg = 0;
    int local = 0, file = f, pthat_bin = -1, n_coll = -1, n_used = 0;
    double weight_mb = NaN, zpt, zy, zphi, zm, lpt[2], leta[2], lphi[2];
    int lpid[2];
    TTree te("jet_events", "Z events of the Z+jet map step");
    te.SetDirectory(&fo);
    te.Branch("event", &ev);
    te.Branch("local_event", &local);
    te.Branch("file", &file);
    te.Branch("bg", &bg);
    te.Branch("pthat_bin", &pthat_bin);
    te.Branch("weight_mb", &weight_mb);
    te.Branch("n_coll", &n_coll);
    te.Branch("n_samples", &n_used);
    te.Branch("z_pt", &zpt); te.Branch("z_y", &zy); te.Branch("z_phi", &zphi);
    te.Branch("z_m", &zm);
    for (int l = 0; l < 2; ++l) {
      const std::string p = "lep" + std::to_string(l + 1) + "_";
      te.Branch((p + "pid").c_str(), &lpid[l]);
      te.Branch((p + "pt").c_str(), &lpt[l]);
      te.Branch((p + "eta").c_str(), &leta[l]);
      te.Branch((p + "phi").c_str(), &lphi[l]);
    }
    // per (event, sample)
    int sample = 0, frag_sample = 0;
    std::vector<double> j_raw, j_pt, j_eta, j_phi, j_area, j_rho;
    std::vector<double> e_rpt, e_reta, e_rphi, e_pt, e_dr;
    std::vector<double> c_eta, c_phi, c_sum, c_rho, c_dpt;
    TTree tj("jets", "jets per oversample");
    tj.SetDirectory(&fo);
    tj.Branch("event", &ev);
    tj.Branch("sample", &sample);
    tj.Branch("frag_sample", &frag_sample);
    for (auto &p : std::vector<std::pair<const char *, std::vector<double> *>>{
             {"jet_pt_raw", &j_raw}, {"jet_pt", &j_pt}, {"jet_eta", &j_eta},
             {"jet_phi", &j_phi}, {"jet_area", &j_area}, {"jet_rho", &j_rho},
             {"emb_ref_pt", &e_rpt}, {"emb_ref_eta", &e_reta}, {"emb_ref_phi", &e_rphi},
             {"emb_pt", &e_pt}, {"emb_dr", &e_dr}, {"rc_eta", &c_eta}, {"rc_phi", &c_phi},
             {"rc_sum", &c_sum}, {"rc_rho", &c_rho}, {"rc_dpt", &c_dpt}})
      tj.Branch(p.first, p.second);

    long n_ev = 0, n_jets = 0;
    std::vector<long> bg_first;                    // the first event of every background
    std::map<int, bool> bg_seen;
    for (long e = r.event_offset(f); e < n && r.info(e).file_index == f; ++e) {
      const EventInfo &i0 = r.info(e);
      if (!i0.vacuum && i0.bg_unit >= 0 && !bg_seen[i0.bg_unit]) {
        bg_seen[i0.bg_unit] = true;
        bg_first.push_back(e);
      }
    }
    for (long e = r.event_offset(f); e < n && r.info(e).file_index == f; ++e) {
      const EventInfo &info = r.info(e);
      const ZRecord &z = info.z;
      if (!z.valid || !(z.m > 60 && z.m < 120 && z.pt > 30)) continue;
      const int nf = info.n_samples_frag;
      const int ns = info.vacuum ? nf : info.n_samples_jet;
      if (ns <= 0 || nf <= 0) continue;
      ev = e; local = info.local_event; bg = bgid[e]; pthat_bin = info.pthat_bin;
      weight_mb = pthat_bin >= 0 && r.n_windows() > 0 ? r.sigma_mb(pthat_bin) / nk[pthat_bin] : NaN;
      n_coll = info.n_coll;
      n_used = std::min(ns, max_samples);
      zpt = z.pt; zy = z.y; zphi = z.phi; zm = z.m;
      for (int l = 0; l < 2; ++l) {
        lpid[l] = z.lep_pid[l];
        lpt[l] = z.lep_pt[l];
        leta[l] = z.lep_eta[l];
        lphi[l] = z.lep_phi[l];
      }
      te.Fill();
      ++n_ev;
      const std::uint64_t key = event_key(seed, name, info.local_event);
      for (int k = 0; k < n_used; ++k) {
        sample = k;
        frag_sample = info.vacuum ? k : k % nf;
        for (auto *v : {&j_raw, &j_pt, &j_eta, &j_phi, &j_area, &j_rho, &e_rpt, &e_reta,
                        &e_rphi, &e_pt, &e_dr, &c_eta, &c_phi, &c_sum, &c_rho, &c_dpt})
          v->clear();
        std::vector<fj::PseudoJet> parts;
        to_pseudojets(info.vacuum ? r.frag(e, k) : r.bkg_dep_frag(e, k), z, parts);
        for (const Jet &j : jets_of(parts, !info.vacuum, ptmin, c)) {
          j_raw.push_back(j.pt_raw); j_pt.push_back(j.pt); j_eta.push_back(j.eta);
          j_phi.push_back(j.phi); j_area.push_back(j.area); j_rho.push_back(j.rho);
        }
        n_jets += long(j_pt.size());
        if (!info.vacuum) {
          // embedding: the fragments alone, and in the background without the wake
          std::vector<fj::PseudoJet> frag, emb;
          to_pseudojets(r.frag(e, frag_sample), z, frag);
          const Hadrons bkg = r.bkg(e, k);
          to_pseudojets(bkg, z, emb);
          emb.insert(emb.end(), frag.begin(), frag.end());
          const auto ref = jets_of(frag, false, PT_REF, c);
          const auto ej = jets_of(emb, true, 1.0, c);
          for (const Jet &rj : ref) {
            double best = NaN, dr_best = std::numeric_limits<double>::infinity();
            for (const Jet &x : ej) {
              const double dr = std::hypot(x.eta - rj.eta, wrap_dphi(x.phi - rj.phi, -M_PI));
              if (dr < dr_best) { dr_best = dr; best = x.pt; }
            }
            e_rpt.push_back(rj.pt); e_reta.push_back(rj.eta); e_rphi.push_back(rj.phi);
            e_pt.push_back(dr_best < DR_MATCH ? best : NaN);
            e_dr.push_back(std::isfinite(dr_best) ? dr_best : NaN);
          }
          // random cones in the background
          std::vector<fj::PseudoJet> bp;
          to_pseudojets(bkg, z, bp);
          fj::JetMedianBackgroundEstimator bge(c.rho_sel, c.kt, c.area);
          bge.set_particles(bp);
          for (int q = 0; q < n_cones; ++q) {
            const std::uint64_t s = RC_SALT | (std::uint64_t(k) << 16) | std::uint64_t(q);
            const double eta = -ETA_RC + 2 * ETA_RC * key_uniform(key, 2 * s);
            const double phi = -M_PI + 2 * M_PI * key_uniform(key, 2 * s + 1);
            fj::PseudoJet axis;
            axis.reset_PtYPhiM(1.0, eta, phi, 0.0);
            double sum = 0;
            for (const auto &p : bp)
              if (p.delta_R(axis) < R_JET) sum += p.pt();
            const double rho = bge.rho(axis);
            c_eta.push_back(eta); c_phi.push_back(axis.phi_std()); c_sum.push_back(sum);
            c_rho.push_back(rho); c_dpt.push_back(sum - rho * M_PI * R_JET * R_JET);
          }
        }
        tj.Fill();
      }
    }
    // the background jets: per hydro background and bulk_bg sample
    Long64_t b_bg = 0;
    int b_unit = 0, b_sample = 0, b_ncoll = -1;
    std::vector<double> b_raw, b_pt, b_eta, b_phi, b_area, b_rho;
    TTree tbj("bg_jets", "jets of the backgrounds' bulk_bg samples");
    tbj.SetDirectory(&fo);
    tbj.Branch("bg", &b_bg);
    tbj.Branch("bg_unit", &b_unit);
    tbj.Branch("sample", &b_sample);
    tbj.Branch("n_coll", &b_ncoll);
    for (auto &p : std::vector<std::pair<const char *, std::vector<double> *>>{
             {"jet_pt_raw", &b_raw}, {"jet_pt", &b_pt}, {"jet_eta", &b_eta},
             {"jet_phi", &b_phi}, {"jet_area", &b_area}, {"jet_rho", &b_rho}})
      tbj.Branch(p.first, p.second);
    const ZRecord no_z;                            // a background has no Z leptons
    long n_bgj = 0;
    for (long e : bg_first) {
      const EventInfo &info = r.info(e);
      b_bg = bgid[e]; b_unit = info.bg_unit; b_ncoll = info.n_coll;
      for (int k = 0; k < std::min(info.n_samples_bg, max_samples); ++k) {
        b_sample = k;
        for (auto *v : {&b_raw, &b_pt, &b_eta, &b_phi, &b_area, &b_rho}) v->clear();
        std::vector<fj::PseudoJet> bp;
        to_pseudojets(r.bkg(e, k), no_z, bp);
        for (const Jet &j : jets_of(bp, true, ptmin, c)) {
          b_raw.push_back(j.pt_raw); b_pt.push_back(j.pt); b_eta.push_back(j.eta);
          b_phi.push_back(j.phi); b_area.push_back(j.area); b_rho.push_back(j.rho);
        }
        n_bgj += long(b_pt.size());
        tbj.Fill();
      }
    }
    fo.cd();
    te.Write();
    tj.Write();
    if (!bg_first.empty()) tbj.Write();
    fo.Close();
    gSystem->Rename((path + ".part").c_str(), path.c_str());
    const double dt =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s: %ld Z events, %ld jets, %zu backgrounds (%ld jets), %.1f s -> %s\n",
                name.c_str(), n_ev, n_jets, bg_first.size(), n_bgj, dt, path.c_str());
  }
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &x) {
    std::fprintf(stderr, "zjets: %s\n", x.what());
    return 1;
  }
}
