// example/analysis_root/zjet/zjets.cxx
//
// The map step of the Z+jet analysis (CMS 1702.01060) in C++ with fastjet: per production
// file the jets of every oversample of the Z events, from run_h5toROOT.py's ROOT files, into
// <out>/<stem>_jets.root.  The final stage (selection, the optional CMS smearing, HEPData,
// chi^2, jackknife, figures) is example/analysis/zjet_test/zjet_jets.py; zjet_jets_map.py
// is the same map step in Python (check_cxx.py jets compares the two).
//
//   ./zjets <root dir> <out dir> [--seed 1] [--files i:j] [--samples N] [--cones 4]
//           [--ptmin 5] [--qa-samples 1]
//
// Events: a Z with 60 < m < 120 GeV and p_T^Z > 30 GeV (the final stage applies the paper's
// cuts).  Samples: k = 0 .. min(n, --samples) - 1, n the oversamples (vacuum: the
// fragmentations).  Particles of sample k: bulk_jet k + jet_frag (k mod n_frag) (vacuum:
// jet_frag k), all final particles but neutrinos and the Z's two leptons.
//
// Jets: anti-k_T R = 0.3, E scheme, p_T,raw > 1 GeV, p_T > --ptmin, |eta| < 2.  Pb+Pb: active
// areas with explicit ghosts on a fixed grid (|y| < 2.6, area 0.01, no scatter: the same jets
// as the Python map step) and rho(jet) the median p_T / area of the k_T R = 0.4 jets with
// |y| < 2.2 in a strip |y - y_jet| < 1; p_T = p_T,raw - rho A.  Vacuum: no subtraction.  Per
// jet also its mass, the number of its (non-ghost) constituents and the leading one's p_T.
// Embedding (Pb+Pb): the fragments of sample k alone (reference jets, p_T > 10 GeV,
// |eta| < 2) and embedded in bulk_bg sample k (no wake), clustered and subtracted as above;
// each reference jet with the embedded jet nearest in Delta R (NaN beyond 0.15).
// Random cones (Pb+Pb): --cones cones per sample in bulk_bg k at |eta| < 1.6, positions
// from zjet_wake.key_uniform; delta p_T = sum p_T within Delta R < 0.3 - rho pi 0.3^2.
// Background jets (Pb+Pb, tree bg_jets): every hydro background's bulk_bg samples k = 0 ..
// min(n, --samples) - 1 clustered and subtracted as above, the model's minimum-bias events
// without any hard process: the final stage's B (jets not from the Z's interaction), from the
// Z's own background or mixed from others of similar N_coll.
// The jet QA (trees qa_jets, qa_bg_jets): the first --qa-samples samples of every Z event and
// of every background with all their jets (p_T,raw > 1 GeV, no cut on p_T = p_T,raw - rho A),
// for the subtraction's negative tail; zjet_jets_qa.py makes the figures.
// Exact shortcuts, the outputs unchanged (as zjet_jets_map.py): rho(jet) once per strip
// content (StripRho), no rho for jets that cannot pass the p_T cut, the background's particles
// and estimator kept per hydro background and sample.

#include "ZJetCommon.h"

#include <TFile.h>
#include <TTree.h>

#include <fastjet/AreaDefinition.hh>
#include <fastjet/ClusterSequence.hh>
#include <fastjet/ClusterSequenceArea.hh>
#include <fastjet/Selector.hh>
#include <fastjet/tools/JetMedianBackgroundEstimator.hh>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>

using namespace zjet;
namespace fj = fastjet;

namespace {

constexpr double R_JET = 0.3, ETA_JET = 2.0, PT_REF = 10.0, DR_MATCH = 0.15;
constexpr double ETA_RC = 1.6;
constexpr double STRIP = 1.0;                  // the rho strip's half width (Clusterer::rho_sel)
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

// rho(jet) of JetMedianBackgroundEstimator(SelectorAbsRapMax(2.2) * SelectorStrip(1.0)),
// computed once per strip content.  With explicit ghosts the median depends only on which k_T
// jets lie in the strip |y - y_jet| < 1, and that set changes only where y_jet +- 1 crosses a
// k_T jet's rapidity: each distinct content (the number of k_T jets below y_jet - 1 and below
// y_jet + 1) is computed once.  A k_T jet within 1e-9 of a strip edge (where the selector's
// comparison decides) is never cached.  The same values as the estimator's, bit for bit
// (zjet_jets_map.py's StripRho).
struct StripRho {
  fj::ClusterSequenceArea cs;                  // the estimator's jets refer to it
  fj::JetMedianBackgroundEstimator bge;
  std::vector<double> ys;
  std::map<std::pair<long, long>, double> cache;
  StripRho(const std::vector<fj::PseudoJet> &parts, const Clusterer &c)
      : cs(parts, c.kt, c.area), bge(c.rho_sel) {
    const auto jets = cs.inclusive_jets();
    bge.set_jets(jets);
    for (const auto &j : jets) ys.push_back(j.rap());
    std::sort(ys.begin(), ys.end());
  }
  double rho(const fj::PseudoJet &jet) {
    const double y = jet.rap();
    long key[2];
    for (int s = 0; s < 2; ++s) {
      const double edge = y + (s ? STRIP : -STRIP);
      const long i = std::lower_bound(ys.begin(), ys.end(), edge) - ys.begin();
      if ((i < long(ys.size()) && std::fabs(ys[i] - edge) < 1e-9) ||
          (i > 0 && std::fabs(ys[i - 1] - edge) < 1e-9))
        return bge.rho(jet);
      key[s] = i;
    }
    const auto k = std::make_pair(key[0], key[1]);
    const auto it = cache.find(k);
    if (it != cache.end()) return it->second;
    return cache[k] = bge.rho(jet);
  }
};

// the jets of a particle list: raw p_T, subtracted p_T, eta, phi, area, rho, mass, the number
// of real (non-ghost) constituents, the leading one's p_T
struct Jet {
  double pt_raw, pt, eta, phi, area, rho, m, nconst, lead;
};

Jet make_jet(const fj::PseudoJet &j, double pt, double area, double rho, bool ghosts) {
  int n = 0;
  double lead = 0;
  for (const auto &q : j.constituents())
    if (!ghosts || !q.is_pure_ghost()) {
      ++n;
      lead = std::max(lead, q.pt());
    }
  return {j.pt(), pt, j.eta(), j.phi_std(), area, rho, j.m(), double(n), lead};
}

// every jet with p_T,raw > 1 GeV and |eta| < 2; ptmin cuts on p_T (subtracted in Pb+Pb), -inf
// keeps them all
std::vector<Jet> jets_of(const std::vector<fj::PseudoJet> &parts, bool subtract,
                         double ptmin, Clusterer &c) {
  std::vector<Jet> out;
  if (!subtract) {
    fj::ClusterSequence cs(parts, c.akt);
    for (const auto &j : fj::sorted_by_pt(cs.inclusive_jets(1.0)))
      if (std::fabs(j.eta()) < ETA_JET && j.pt() >= ptmin)
        out.push_back(make_jet(j, j.pt(), 0, 0, false));
    return out;
  }
  StripRho bge(parts, c);
  fj::ClusterSequenceArea cs(parts, c.akt, c.area);
  for (const auto &j : fj::sorted_by_pt(cs.inclusive_jets(1.0))) {
    if (!(std::fabs(j.eta()) < ETA_JET)) continue;
    if (j.pt() <= ptmin) continue;              // rho >= 0: p_T <= p_T,raw cannot pass
    const double rho = bge.rho(j), a = j.area(), pt = j.pt() - rho * a;
    if (pt > ptmin) out.push_back(make_jet(j, pt, a, rho, true));
  }
  return out;
}

// the jet columns of a tree: jet_pt_raw, jet_pt, ... (one vector each, refilled per row)
struct JetColumns {
  std::vector<double> v[9];
  static constexpr const char *names[9] = {"jet_pt_raw", "jet_pt", "jet_eta", "jet_phi",
                                           "jet_area", "jet_rho", "jet_m", "jet_nconst",
                                           "jet_lead_pt"};
  void branch(TTree &t) {
    for (int i = 0; i < 9; ++i) t.Branch(names[i], &v[i]);
  }
  void clear() {
    for (auto &x : v) x.clear();
  }
  void push(const Jet &j) {
    const double x[9] = {j.pt_raw, j.pt, j.eta, j.phi, j.area, j.rho, j.m, j.nconst, j.lead};
    for (int i = 0; i < 9; ++i) v[i].push_back(x[i]);
  }
  size_t size() const { return v[0].size(); }
};

int run(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <root dir> <out dir> [--seed 1] [--files i:j] "
                         "[--samples N] [--cones 4] [--ptmin 5] [--qa-samples 1]\n", argv[0]);
    return 2;
  }
  const std::string in = argv[1], out = argv[2];
  const auto o = options(argc, argv, 3);
  const long seed = o.count("--seed") ? std::stol(o.at("--seed")) : 1;
  const int max_samples = o.count("--samples") ? std::stoi(o.at("--samples")) : 1 << 30;
  const int n_cones = o.count("--cones") ? std::stoi(o.at("--cones")) : 4;
  const double ptmin = o.count("--ptmin") ? std::stod(o.at("--ptmin")) : 5.0;
  const int qa_samples = o.count("--qa-samples") ? std::stoi(o.at("--qa-samples")) : 1;
  const double ALL = -std::numeric_limits<double>::infinity();
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
    JetColumns jc, qc;
    std::vector<double> e_rpt, e_reta, e_rphi, e_pt, e_dr;
    std::vector<double> c_eta, c_phi, c_sum, c_rho, c_dpt;
    TTree tj("jets", "jets per oversample");
    tj.SetDirectory(&fo);
    tj.Branch("event", &ev);
    tj.Branch("sample", &sample);
    tj.Branch("frag_sample", &frag_sample);
    jc.branch(tj);
    for (auto &p : std::vector<std::pair<const char *, std::vector<double> *>>{
             {"emb_ref_pt", &e_rpt}, {"emb_ref_eta", &e_reta}, {"emb_ref_phi", &e_rphi},
             {"emb_pt", &e_pt}, {"emb_dr", &e_dr}, {"rc_eta", &c_eta}, {"rc_phi", &c_phi},
             {"rc_sum", &c_sum}, {"rc_rho", &c_rho}, {"rc_dpt", &c_dpt}})
      tj.Branch(p.first, p.second);
    // the jet QA: every jet of the first qa_samples samples
    TTree tq("qa_jets", "all jets of the first --qa-samples samples (no p_T cut)");
    tq.SetDirectory(&fo);
    tq.Branch("event", &ev);
    tq.Branch("sample", &sample);
    qc.branch(tq);

    long n_ev = 0, n_jets = 0;
    const ZRecord no_z;                            // a background has no Z leptons
    // bulk_bg sample k's particles and their StripRho, shared by the background's events (one
    // background at a time) unless one of the event's Z leptons is among them
    std::map<std::pair<int, int>,
             std::pair<std::vector<fj::PseudoJet>, std::unique_ptr<StripRho>>> bg_cache;
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
        jc.clear();
        qc.clear();
        for (auto *v : {&e_rpt, &e_reta, &e_rphi, &e_pt, &e_dr, &c_eta, &c_phi, &c_sum, &c_rho,
                        &c_dpt})
          v->clear();
        std::vector<fj::PseudoJet> parts;
        to_pseudojets(info.vacuum ? r.frag(e, k) : r.bkg_dep_frag(e, k), z, parts);
        for (const Jet &j : jets_of(parts, !info.vacuum, k < qa_samples ? ALL : ptmin, c)) {
          if (info.vacuum ? j.pt >= ptmin : j.pt > ptmin) jc.push(j);
          if (k < qa_samples) qc.push(j);
        }
        if (k < qa_samples) tq.Fill();
        n_jets += long(jc.size());
        if (!info.vacuum) {
          // embedding: the fragments alone, and in the background without the wake
          std::vector<fj::PseudoJet> frag, emb;
          to_pseudojets(r.frag(e, frag_sample), z, frag);
          const Hadrons bkg = r.bkg(e, k);
          to_pseudojets(bkg, z, emb);
          const std::size_t n_kept = emb.size();
          std::size_t n_all = 0;
          for (const Hadron &h : bkg) n_all += keep(h, no_z);
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
          // random cones in the background: its particles (the event's Z leptons removed)
          // and their rho, shared by the background's events if no Z lepton is among them
          const std::vector<fj::PseudoJet> *bpp;
          StripRho *bge;
          std::vector<fj::PseudoJet> own_bp;
          std::unique_ptr<StripRho> own_bge;
          if (n_kept == n_all) {
            if (!bg_cache.empty() && bg_cache.begin()->first.first != info.bg_unit)
              bg_cache.clear();
            auto it = bg_cache.find({info.bg_unit, k});
            if (it == bg_cache.end()) {
              std::vector<fj::PseudoJet> v(emb.begin(), emb.begin() + long(n_kept));
              auto rho = std::make_unique<StripRho>(v, c);
              it = bg_cache.emplace(std::make_pair(info.bg_unit, k),
                                    std::make_pair(std::move(v), std::move(rho))).first;
            }
            bpp = &it->second.first;
            bge = it->second.second.get();
          } else {
            own_bp.assign(emb.begin(), emb.begin() + long(n_kept));
            own_bge = std::make_unique<StripRho>(own_bp, c);
            bpp = &own_bp;
            bge = own_bge.get();
          }
          for (int q = 0; q < n_cones; ++q) {
            const std::uint64_t s = RC_SALT | (std::uint64_t(k) << 16) | std::uint64_t(q);
            const double eta = -ETA_RC + 2 * ETA_RC * key_uniform(key, 2 * s);
            const double phi = -M_PI + 2 * M_PI * key_uniform(key, 2 * s + 1);
            fj::PseudoJet axis;
            axis.reset_PtYPhiM(1.0, eta, phi, 0.0);
            double sum = 0;
            for (const auto &p : *bpp)
              if (p.delta_R(axis) < R_JET) sum += p.pt();
            const double rho = bge->rho(axis);
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
    JetColumns bc, bqc;
    TTree tbj("bg_jets", "jets of the backgrounds' bulk_bg samples");
    TTree tbq("qa_bg_jets", "all jets of the backgrounds' first --qa-samples bulk_bg samples");
    for (TTree *t : {&tbj, &tbq}) {
      t->SetDirectory(&fo);
      t->Branch("bg", &b_bg);
      t->Branch("bg_unit", &b_unit);
      t->Branch("sample", &b_sample);
      t->Branch("n_coll", &b_ncoll);
    }
    bc.branch(tbj);
    bqc.branch(tbq);
    long n_bgj = 0;
    for (long e : bg_first) {
      const EventInfo &info = r.info(e);
      b_bg = bgid[e]; b_unit = info.bg_unit; b_ncoll = info.n_coll;
      for (int k = 0; k < std::min(info.n_samples_bg, max_samples); ++k) {
        b_sample = k;
        bc.clear();
        bqc.clear();
        std::vector<fj::PseudoJet> bp;
        to_pseudojets(r.bkg(e, k), no_z, bp);
        for (const Jet &j : jets_of(bp, true, k < qa_samples ? ALL : ptmin, c)) {
          if (j.pt > ptmin) bc.push(j);
          if (k < qa_samples) bqc.push(j);
        }
        n_bgj += long(bc.size());
        tbj.Fill();
        if (k < qa_samples) tbq.Fill();
      }
    }
    fo.cd();
    te.Write();
    tj.Write();
    tq.Write();
    if (!bg_first.empty()) {
      tbj.Write();
      tbq.Write();
    }
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
