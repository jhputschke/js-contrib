// example/analysis_root/zjet/zhadron_pools.cxx
//
// The map step of the Z-hadron analysis (CMS 2507.09307) in C++: per production file the
// pool of zjet_mixing.file_pool, from run_h5toROOT.py's ROOT files, into
// <out>/<stem>_pool.root.  The final stage is example/analysis/zjet_test/mix.py
// (--pbpb-pools / --vacuum-pools), which reads these with zjet_mixing.read_root_pool.
//
//   ./zhadron_pools <root dir> <out dir> [--seed 1] [--eta-max 2.4] [--files i:j]
//
// Per selected event (CMS 2507.09307's Z selection, zjet_wake.CMS_2507): S in its Z frame,
// the per-sample mean of the full event (bulk_jet + jet_frag) and one oversample k_one
// (zjet_wake.key_index(event_key(seed, stem, event), n)), that oversample's tracks; per
// hydro background the sums of w S, w^2 S and w rho (lab frame) for the full event and the
// Z-correlated yield (bulk_jet - bulk_bg + jet_frag).  Vacuum files: the fragments, one
// content.  The weights (window cross section / events x N_coll) need every file, so the
// whole directory is read even with --files.

#include "ZJetCommon.h"

#include <TFile.h>
#include <TTree.h>

#include <chrono>
#include <cstdio>
#include <map>

using namespace zjet;

namespace {

struct BgSums {
  double W = 0, W2 = 0;
  std::vector<double> SW_full = std::vector<double>(N_Z), T_full = std::vector<double>(N_Z),
                      SW_zc = std::vector<double>(N_Z), T_zc = std::vector<double>(N_Z),
                      R_full = std::vector<double>(N_LAB), R_zc = std::vector<double>(N_LAB);
};

struct BgCache {
  int K = 0;
  std::vector<Track> tracks;
  std::vector<double> R;           // lab-frame counts summed over the samples
};

int run(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <root dir> <out dir> [--seed 1] [--eta-max 2.4] "
                         "[--files i:j]\n", argv[0]);
    return 2;
  }
  const std::string in = argv[1], out = argv[2];
  const auto o = options(argc, argv, 3);
  const long seed = o.count("--seed") ? std::stol(o.at("--seed")) : 1;
  const double eta_max = o.count("--eta-max") ? std::stod(o.at("--eta-max")) : 2.4;
  gSystem->mkdir(out.c_str(), true);

  HadronFileReader r(in);
  const long n = r.n_events();
  std::vector<char> sel(n, 0);
  for (long e = 0; e < n; ++e) sel[e] = select(r.info(e).z, CMS_2507);
  const auto w = event_weights(r, sel);
  const auto bgid = background_ids(r);
  const auto range = file_range(o, r.n_files());

  for (int f = range.first; f < range.second; ++f) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::string name = stem_of(r.file(f));
    // the per-event output
    Long64_t ev_event = 0, ev_bg = 0;
    double ev_w = 0, ev_y = 0, ev_phi = 0;
    int ev_k = 0, ev_ns = 0;
    std::vector<float> S_full(N_Z), S_one(N_Z);
    std::vector<int> trk_pt;
    std::vector<float> trk_eta, trk_phi;
    const std::string path = out + "/" + name + "_pool.root";
    TFile fo((path + ".part").c_str(), "RECREATE");
    TTree te("pool_events", "zjet_mixing pool: per selected event");
    te.SetDirectory(&fo);   // the reader opens other files: keep the trees here
    te.Branch("event", &ev_event);
    te.Branch("bg", &ev_bg);
    te.Branch("w", &ev_w);
    te.Branch("y", &ev_y);
    te.Branch("phi", &ev_phi);
    te.Branch("k_one", &ev_k);
    te.Branch("n_samples", &ev_ns);
    te.Branch("S_full", S_full.data(), Form("S_full[%d]/F", N_Z));
    te.Branch("S_one", S_one.data(), Form("S_one[%d]/F", N_Z));
    te.Branch("trk_pt", &trk_pt);
    te.Branch("trk_eta", &trk_eta);
    te.Branch("trk_phi", &trk_phi);

    std::map<long, BgSums> sums;
    std::map<int, BgCache> bg_cache;
    bool vacuum_file = false;
    long n_ev = 0;
    for (long e = r.event_offset(f); e < n && r.info(e).file_index == f; ++e) {
      if (!sel[e]) continue;
      const EventInfo &info = r.info(e);
      const ZRecord &z = info.z;
      vacuum_file = info.vacuum;
      const int nf = info.n_samples_frag;
      std::vector<Track> tf, tj;
      for (int j = 0; j < nf; ++j) add_tracks(r.frag(e, j), z, eta_max, j, tf);
      int K = 0;
      BgCache *bc = nullptr;
      if (!info.vacuum) {
        K = info.n_samples_jet;
        auto it = bg_cache.find(info.bg_unit);
        if (it == bg_cache.end()) {
          BgCache c;
          c.K = info.n_samples_bg;
          for (int k = 0; k < c.K; ++k) add_tracks(r.bkg(e, k), z, eta_max, k, c.tracks);
          c.R.assign(N_LAB, 0.0);
          lab_hist(c.tracks, c.R);
          it = bg_cache.emplace(info.bg_unit, std::move(c)).first;
        }
        bc = &it->second;
        if (K == 0 || nf == 0 || bc->K != K) {
          std::fprintf(stderr, "%s event %d: %d jet / %d background samples, %d "
                               "fragmentations: left out\n", name.c_str(), info.local_event, K,
                       bc->K, nf);
          continue;
        }
        for (int k = 0; k < K; ++k) add_tracks(r.bkg_dep(e, k), z, eta_max, k, tj);
      } else if (nf == 0) {
        continue;
      }
      const int ns = info.vacuum ? nf : K;
      const int k_one = int(key_index(event_key(seed, name, info.local_event), ns));
      // S and rho, the per-sample means; the order of the operations as in file_pool
      std::vector<double> SF(N_Z, 0), RF(N_LAB, 0), Sf(N_Z), Szc(N_Z), Rf(N_LAB), Rzc(N_LAB);
      z_hist(tf, z.y, z.phi, SF);
      lab_hist(tf, RF);
      for (auto &x : SF) x /= nf;
      for (auto &x : RF) x /= nf;
      std::vector<Track> one;
      if (!info.vacuum) {
        std::vector<double> SJ(N_Z, 0), RJ(N_LAB, 0), SB(N_Z, 0);
        z_hist(tj, z.y, z.phi, SJ);
        lab_hist(tj, RJ);
        z_hist(bc->tracks, z.y, z.phi, SB);
        for (int i = 0; i < N_Z; ++i) {
          const double sj = SJ[i] / K, sb = SB[i] / K;
          Sf[i] = sj + SF[i];
          Szc[i] = sj - sb + SF[i];
        }
        for (int i = 0; i < N_LAB; ++i) {
          const double rj = RJ[i] / K, rb = bc->R[i] / K;
          Rf[i] = rj + RF[i];
          Rzc[i] = rj - rb + RF[i];
        }
        for (const Track &x : tj) if (x.sample == k_one) one.push_back(x);
        for (const Track &x : tf) if (x.sample == k_one % nf) one.push_back(x);
      } else {
        Sf = SF; Szc = SF; Rf = RF; Rzc = RF;
        for (const Track &x : tf) if (x.sample == k_one) one.push_back(x);
      }
      std::vector<double> So(N_Z, 0);
      z_hist(one, z.y, z.phi, So);
      // the event's entry
      ev_event = e; ev_bg = bgid[e]; ev_w = w[e]; ev_y = z.y; ev_phi = z.phi;
      ev_k = k_one; ev_ns = ns;
      for (int i = 0; i < N_Z; ++i) { S_full[i] = float(Sf[i]); S_one[i] = float(So[i]); }
      trk_pt.clear(); trk_eta.clear(); trk_phi.clear();
      for (const Track &x : one) {
        trk_pt.push_back(x.ipt);
        trk_eta.push_back(float(x.eta));
        trk_phi.push_back(float(x.phi));
      }
      te.Fill();
      ++n_ev;
      // the background's sums
      BgSums &s = sums[bgid[e]];
      const double we = w[e];
      s.W += we;
      s.W2 += we * we;
      for (int i = 0; i < N_Z; ++i) {
        s.SW_full[i] += we * Sf[i];
        s.T_full[i] += we * we * Sf[i];
        s.SW_zc[i] += we * Szc[i];
        s.T_zc[i] += we * we * Szc[i];
      }
      for (int i = 0; i < N_LAB; ++i) {
        s.R_full[i] += we * Rf[i];
        s.R_zc[i] += we * Rzc[i];
      }
    }
    // per background
    Long64_t bg_id = 0;
    double W = 0, W2 = 0;
    std::vector<double> SWf(N_Z), Tf(N_Z), SWz(N_Z), Tz(N_Z), Rfb(N_LAB), Rzb(N_LAB);
    TTree tb("pool_bg", "zjet_mixing pool: per hydro background");
    tb.SetDirectory(&fo);
    tb.Branch("bg_id", &bg_id);
    tb.Branch("W", &W);
    tb.Branch("W2", &W2);
    tb.Branch("SW_full", SWf.data(), Form("SW_full[%d]/D", N_Z));
    tb.Branch("T_full", Tf.data(), Form("T_full[%d]/D", N_Z));
    tb.Branch("R_full", Rfb.data(), Form("R_full[%d]/D", N_LAB));
    if (!vacuum_file) {
      tb.Branch("SW_zc", SWz.data(), Form("SW_zc[%d]/D", N_Z));
      tb.Branch("T_zc", Tz.data(), Form("T_zc[%d]/D", N_Z));
      tb.Branch("R_zc", Rzb.data(), Form("R_zc[%d]/D", N_LAB));
    }
    for (auto &p : sums) {
      bg_id = p.first; W = p.second.W; W2 = p.second.W2;
      // into the branches' buffers (the vectors keep their storage)
      std::copy(p.second.SW_full.begin(), p.second.SW_full.end(), SWf.begin());
      std::copy(p.second.T_full.begin(), p.second.T_full.end(), Tf.begin());
      std::copy(p.second.R_full.begin(), p.second.R_full.end(), Rfb.begin());
      std::copy(p.second.SW_zc.begin(), p.second.SW_zc.end(), SWz.begin());
      std::copy(p.second.T_zc.begin(), p.second.T_zc.end(), Tz.begin());
      std::copy(p.second.R_zc.begin(), p.second.R_zc.end(), Rzb.begin());
      tb.Fill();
    }
    fo.cd();
    te.Write();
    tb.Write();
    fo.Close();
    gSystem->Rename((path + ".part").c_str(), path.c_str());
    const double dt =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s: %ld events, %zu backgrounds, %.1f s -> %s\n", name.c_str(), n_ev,
                sums.size(), dt, path.c_str());
  }
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &x) {
    std::fprintf(stderr, "zhadron_pools: %s\n", x.what());
    return 1;
  }
}
