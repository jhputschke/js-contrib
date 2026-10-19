// example/analysis_root/zjet/zjets.cxx
//
// The map step of the Z+jet analysis (CMS 1702.01060) in C++ with fastjet: per production
// file the jets of every oversample of the Z events, from run_h5toROOT.py's ROOT files, into
// <out>/<stem>_jets.root.  The final stage (selection, the optional CMS smearing, HEPData,
// chi^2, jackknife, figures) is example/analysis/zjet/zjet_jets.py; zjet_jets_map.py
// is the same map step in Python (check_cxx.py jets compares the two).
//
//   ./zjets <root dir> <out dir> [--R 0.3] [--seed 1] [--files i:j] [--samples N]
//           [--emb-samples N] [--cones 4] [--ptmin 5] [--qa-samples 1]
//
// Events: a Z with 60 < m < 120 GeV and p_T^Z > 30 GeV (the final stage applies the paper's
// cuts).  Samples: k = 0 .. min(n, --samples) - 1, n the oversamples (vacuum: the
// fragmentations).  Particles of sample k: bulk_jet k + jet_frag (k mod n_frag) (vacuum:
// jet_frag k), all final particles but neutrinos and the Z's two leptons.
//
// Every output stands alone: its file_info tree (the production file's name, events, hydro
// backgrounds and pTHat windows, ../common/AnaCommon.h's FileRecord) lets the loader
// (zjet_jets.load_jets) rebuild the ids and weights of any set of outputs, from one or
// several campaigns; the stored event, file, bg and weight_mb are those of the input
// directory.  run_info: the settings (R, ghost area, --ptmin, --cones, --seed, ...).
//
// Jets: anti-k_T R = --R (0.3), E scheme, p_T,raw > 1 GeV, p_T > --ptmin, |eta| < 2.  Pb+Pb: active
// areas with explicit ghosts on a fixed grid (|y| < 2.6, area 0.01, no scatter: the same jets
// as the Python map step) and rho(jet) the median p_T / area of the k_T R = 0.4 jets with
// |y| < 2.2 in a strip |y - y_jet| < 1; p_T = p_T,raw - rho A.  Vacuum: no subtraction.  Per
// jet also its mass, the number of its (non-ghost) constituents and the leading one's p_T.
// Embedding (Pb+Pb): the fragments of sample k alone (reference jets, p_T > 10 GeV,
// |eta| < 2) and embedded in bulk_bg sample k (no wake), clustered and subtracted as above;
// each reference jet with the embedded jet nearest in Delta R (NaN beyond R / 2).  The
// embedded event's jets above --ptmin are also stored (nw_*): the Z event without its wake,
// for the final stage's "no wake" variant.  --emb-samples N: the embedded event (emb_*, nw_*,
// zcone_nw_*) only for the first N oversamples, about half the time of the others; the jets,
// the Z cone and the random cones of every oversample are unchanged (default: all).
// Random cones (Pb+Pb): --cones cones per sample in bulk_bg k at |eta| < 1.6, positions
// from zjet_wake.key_uniform; delta p_T = sum p_T within Delta R < R - rho pi R^2.
// The Z cone (zcone_*): the same at (y_Z, phi_Z) in the full event (rho its own) and in the
// embedded event without the wake (zcone_nw_*; vacuum: rho 0, no nw): jet1's p_T variant of
// the dijet proxy (docs/Plans/PLAN_dijet_deta_wake.md A2).
// Background jets (Pb+Pb, tree bg_jets): every hydro background's bulk_bg samples k = 0 ..
// min(n, --samples) - 1 clustered and subtracted as above, the model's minimum-bias events
// without any hard process: the final stage's B (jets not from the Z's interaction), from the
// Z's own background or mixed from others of similar N_coll.
// The jet QA (trees qa_jets, qa_bg_jets): the first --qa-samples samples of every Z event and
// of every background with all their jets (p_T,raw > 1 GeV, no cut on p_T = p_T,raw - rho A),
// for the subtraction's negative tail; zjet_jets_qa.py makes the figures.
// The clustering is ../common/JetMap.h's.  Exact shortcuts, the outputs unchanged (as
// zjet_jets_map.py): rho(jet) once per strip content (StripRho), no rho for jets that cannot
// pass the p_T cut, the background's particles and estimator kept per hydro background and
// sample.

#include "ZJetCommon.h"
#include "../common/JetMap.h"

#include <TFile.h>
#include <TTree.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>

using namespace zjet;
namespace fj = fastjet;

namespace {

constexpr double PT_REF = 10.0;
constexpr double ETA_RC = 1.6;
constexpr long long MAP_VERSION = 2;          // 2: file_info, run_info, bg_unit, zcone_*, --R
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

int run(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <root dir> <out dir> [--R 0.3] [--seed 1] [--files i:j] "
                         "[--samples N] [--emb-samples N] [--cones 4] [--ptmin 5] "
                         "[--qa-samples 1]\n", argv[0]);
    return 2;
  }
  const std::string in = argv[1], out = argv[2];
  const auto o = options(argc, argv, 3);
  const long seed = o.count("--seed") ? std::stol(o.at("--seed")) : 1;
  const int max_samples = o.count("--samples") ? std::stoi(o.at("--samples")) : 1 << 30;
  const int n_cones = o.count("--cones") ? std::stoi(o.at("--cones")) : 4;
  const double ptmin = o.count("--ptmin") ? std::stod(o.at("--ptmin")) : 5.0;
  const int qa_samples = o.count("--qa-samples") ? std::stoi(o.at("--qa-samples")) : 1;
  const double R = o.count("--R") ? std::stod(o.at("--R")) : R_JET;
  const int emb_samples = o.count("--emb-samples") ? std::stoi(o.at("--emb-samples")) : 1 << 30;
  const double dr_match = R / 2;
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
  Clusterer c(R);

  for (int f = range.first; f < range.second; ++f) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::string name = stem_of(r.file(f));
    const std::string path = out + "/" + name + "_jets.root";
    TFile fo((path + ".part").c_str(), "RECREATE");
    // per event
    Long64_t ev = 0, bg = 0;
    int local = 0, file = f, bg_unit = -1, pthat_bin = -1, n_coll = -1, n_used = 0;
    double weight_mb = NaN, zpt, zy, zphi, zm, lpt[2], leta[2], lphi[2];
    int lpid[2];
    TTree te("jet_events", "Z events of the Z+jet map step");
    te.SetDirectory(&fo);
    te.Branch("event", &ev);
    te.Branch("local_event", &local);
    te.Branch("file", &file);
    te.Branch("bg", &bg);
    te.Branch("bg_unit", &bg_unit);
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
    JetColumns jc, nwc, qc;                        // nwc: the embedded event's jets (no wake)
    std::vector<double> e_rpt, e_reta, e_rphi, e_pt, e_dr;
    std::vector<double> c_eta, c_phi, c_sum, c_rho, c_dpt;
    double zc[6];                                  // zcone_{sum,rho,dpt}, zcone_nw_{...}
    TTree tj("jets", "jets per oversample");
    tj.SetDirectory(&fo);
    tj.Branch("event", &ev);
    tj.Branch("sample", &sample);
    tj.Branch("frag_sample", &frag_sample);
    jc.branch(tj);
    nwc.branch(tj, "nw");
    for (auto &p : std::vector<std::pair<const char *, std::vector<double> *>>{
             {"emb_ref_pt", &e_rpt}, {"emb_ref_eta", &e_reta}, {"emb_ref_phi", &e_rphi},
             {"emb_pt", &e_pt}, {"emb_dr", &e_dr}, {"rc_eta", &c_eta}, {"rc_phi", &c_phi},
             {"rc_sum", &c_sum}, {"rc_rho", &c_rho}, {"rc_dpt", &c_dpt}})
      tj.Branch(p.first, p.second);
    const char *zc_names[6] = {"zcone_sum", "zcone_rho", "zcone_dpt", "zcone_nw_sum",
                               "zcone_nw_rho", "zcone_nw_dpt"};
    for (int i = 0; i < 6; ++i) tj.Branch(zc_names[i], &zc[i]);
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
      bg_unit = info.vacuum ? info.local_event : info.bg_unit;
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
        nwc.clear();
        qc.clear();
        for (auto *v : {&e_rpt, &e_reta, &e_rphi, &e_pt, &e_dr, &c_eta, &c_phi, &c_sum, &c_rho,
                        &c_dpt})
          v->clear();
        std::vector<fj::PseudoJet> parts;
        to_pseudojets(info.vacuum ? r.frag(e, k) : r.bkg_dep_frag(e, k), z, parts);
        std::unique_ptr<StripRho> pb;
        if (!info.vacuum) pb = std::make_unique<StripRho>(parts, c);
        const Cone zcone = cone(parts, pb.get(), z.y, z.phi, R);
        zc[0] = zcone.sum; zc[1] = zcone.rho; zc[2] = zcone.dpt;
        zc[3] = zc[4] = zc[5] = NaN;
        for (const Jet &j : jets_of(parts, !info.vacuum, k < qa_samples ? ALL : ptmin, c,
                                    pb.get())) {
          if (info.vacuum ? j.pt >= ptmin : j.pt > ptmin) jc.push(j);
          if (k < qa_samples) qc.push(j);
        }
        if (k < qa_samples) tq.Fill();
        n_jets += long(jc.size());
        if (!info.vacuum) {
          // embedding: the fragments alone, and in the background without the wake
          std::vector<fj::PseudoJet> frag, emb;
          const Hadrons bkg = r.bkg(e, k);
          to_pseudojets(bkg, z, emb);
          const std::size_t n_kept = emb.size();
          std::size_t n_all = 0;
          for (const Hadron &h : bkg) n_all += keep(h, no_z);
          if (k < emb_samples) {                  // the embedded event (--emb-samples)
            to_pseudojets(r.frag(e, frag_sample), z, frag);
            emb.insert(emb.end(), frag.begin(), frag.end());
            const auto ref = jets_of(frag, false, PT_REF, c);
            StripRho nwb(emb, c);
            const auto ej = jets_of(emb, true, 1.0, c, &nwb);
            const Cone znw = cone(emb, &nwb, z.y, z.phi, R);
            zc[3] = znw.sum; zc[4] = znw.rho; zc[5] = znw.dpt;
            for (const Jet &x : ej)               // the same event without the wake
              if (x.pt > ptmin) nwc.push(x);
            for (const Jet &rj : ref) {
              double best = NaN, dr_best = std::numeric_limits<double>::infinity();
              for (const Jet &x : ej) {
                const double dr = std::hypot(x.eta - rj.eta, wrap_dphi(x.phi - rj.phi, -M_PI));
                if (dr < dr_best) { dr_best = dr; best = x.pt; }
              }
              e_rpt.push_back(rj.pt); e_reta.push_back(rj.eta); e_rphi.push_back(rj.phi);
              e_pt.push_back(dr_best < dr_match ? best : NaN);
              e_dr.push_back(std::isfinite(dr_best) ? dr_best : NaN);
            }
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
            const Cone rc = cone(*bpp, bge, eta, phi, R);
            c_eta.push_back(eta); c_phi.push_back(rc.axis.phi_std()); c_sum.push_back(rc.sum);
            c_rho.push_back(rc.rho); c_dpt.push_back(rc.dpt);
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
    write_file_info(fo, file_record(r, f));
    write_run_info(fo, {{"R", R}, {"ghost_area", GHOST_AREA}, {"ptmin", ptmin}},
                   {{"cones", n_cones}, {"seed", seed}, {"qa_samples", qa_samples},
                    {"samples", max_samples}, {"emb_samples", emb_samples},
                    {"version", MAP_VERSION}});
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
