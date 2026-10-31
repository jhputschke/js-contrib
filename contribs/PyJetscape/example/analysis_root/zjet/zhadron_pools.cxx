// example/analysis_root/zjet/zhadron_pools.cxx
//
// The map step of the Z-hadron analysis (CMS 2507.09307) in C++: per production file the
// pool of zjet_mixing.file_pool, from run_h5toROOT.py's ROOT files, into
// <out>/<stem>_pool.root.  The final stage is example/analysis/zjet/mix.py
// (--pbpb-pools / --vacuum-pools), which reads these with zjet_mixing.read_root_pool.
//
//   ./zhadron_pools <root dir> <out dir> [--seed 1] [--eta-max 2.4] [--files i:j] [--extras 0]
//
// Per selected event (CMS 2507.09307's Z selection, zjet_wake.CMS_2507): S in its Z frame,
// the per-sample mean of the full event (bulk_jet + jet_frag) and one oversample k_one
// (zjet_wake.key_index(event_key(seed, stem, event), n)), that oversample's tracks; the dijet
// proxy's contents on the fine Delta y grid per side of the Z (Y_full, Y_one, Y_zc, Y_frag)
// and in lab eta per side (L_full, L_zc, L_frag); its keys (local_event, pthat_bin, n_coll,
// bg_unit).  Per hydro background and pTHat window, unweighted (tree pool_bgw): the number
// of selected events, N_coll, the sums of S (full event and Z-correlated yield, bulk_jet -
// bulk_bg + jet_frag) and of the full event's lab density; per background its bulk_bg lab
// density per sample (pool_bgr); the production file's record (file_info); the grid's edges
// (pool_grid, zjet_mixing.GRID: check_grid refuses pools binned otherwise).  Pool version 4:
// zjet_mixing.assemble rebuilds the weights and the weighted sums of any set of pools, from
// several campaigns.  Vacuum files: the fragments, one content.  The stored w (window cross
// section / events x N_coll) is that of the input directory, for the record.
// Extra hard scatterings (files with extra_frag, docs/Plans/PLAN_multi_hard.md): the full
// event (S_full, S_one and its tracks, Y_full, Y_one, L_full, R_full) is bulk_jet + jet_frag +
// extra_frag, the Z-correlated contents stay the trigger's (bulk_jet - bulk_bg + jet_frag), so
// B, the bulk alone, is bulk_bg + extra_frag; pool_bgw also has R_extra (the extras' lab
// density, which zjet_mixing.assemble takes out of R_zc).  --extras 0 leaves them out: the
// pools of the files without extras.

#include "ZJetCommon.h"

#include <TFile.h>
#include <TTree.h>

#include <chrono>
#include <cstdio>
#include <map>

using namespace zjet;

namespace {

struct BgSums {                   // per (background, window), unweighted
  long n = 0;
  double ncoll = 0;
  std::vector<double> S_full = std::vector<double>(N_Z), S_zc = std::vector<double>(N_Z),
                      R_full = std::vector<double>(N_LAB), R_extra = std::vector<double>(N_LAB);
};

struct BgCache {
  int K = 0;
  std::vector<Track> tracks;
  std::vector<double> R;           // lab-frame counts summed over the samples
};

// the one-entry tree pool_grid: the edges of zjet_mixing.GRID (read_root_pool's grid_*)
void write_grid(TFile &fo) {
  std::vector<std::pair<const char *, std::vector<double>>> g = {
      {"pt", PT()},   {"dy", DETA()}, {"dphi", DPHI()},
      {"eta", ETA()}, {"phi", PHI()}, {"dyf", DY_FINE()}};
  TTree t("pool_grid", "zjet_mixing pool: the grid's edges");
  t.SetDirectory(&fo);
  for (auto &x : g) t.Branch(x.first, x.second.data(), Form("%s[%zu]/D", x.first, x.second.size()));
  t.Fill();
  fo.cd();
  t.Write();
}

int run(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <root dir> <out dir> [--seed 1] [--eta-max 2.4] "
                         "[--files i:j] [--extras 0]\n", argv[0]);
    return 2;
  }
  const std::string in = argv[1], out = argv[2];
  const auto o = options(argc, argv, 3);
  const long seed = o.count("--seed") ? std::stol(o.at("--seed")) : 1;
  const double eta_max = o.count("--eta-max") ? std::stod(o.at("--eta-max")) : 2.4;
  const bool extras = !o.count("--extras") || std::stoi(o.at("--extras")) != 0;
  gSystem->mkdir(out.c_str(), true);

  HadronFileReader r(in);
  r.set_extras(extras);
  const long n = r.n_events();
  std::vector<char> sel(n, 0);
  for (long e = 0; e < n; ++e) sel[e] = select(r.info(e).z, CMS_2507);
  const auto w = event_weights(r, sel);
  const auto bgid = background_ids(r);
  const auto range = file_range(o, r.n_files());

  for (int f = range.first; f < range.second; ++f) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::string name = stem_of(r.file(f));
    const bool xtra = extras && r.has_extras(f);        // extra hard scatterings
    // the per-event output
    Long64_t ev_event = 0, ev_bg = 0;
    double ev_w = 0, ev_y = 0, ev_phi = 0;
    int ev_k = 0, ev_ns = 0;
    std::vector<float> S_full(N_Z), S_one(N_Z), Y_full(N_Y), Y_one(N_Y), Y_zc(N_Y), Y_frag(N_Y),
        L_full(N_LS), L_zc(N_LS), L_frag(N_LS);
    int ev_local = 0, ev_bin = -1, ev_unit = -1;
    double ev_ncoll = 0;
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
    te.Branch("local_event", &ev_local);
    te.Branch("pthat_bin", &ev_bin);
    te.Branch("n_coll", &ev_ncoll);
    te.Branch("bg_unit", &ev_unit);
    te.Branch("trk_pt", &trk_pt);
    te.Branch("trk_eta", &trk_eta);
    te.Branch("trk_phi", &trk_phi);

    std::map<std::pair<long, int>, BgSums> sums;     // (unit, window)
    std::map<long, std::vector<double>> bg_rb;       // unit -> bulk_bg lab density per sample
    std::map<int, BgCache> bg_cache;
    bool vacuum_file = false, branched = false;
    long n_ev = 0;
    for (long e = r.event_offset(f); e < n && r.info(e).file_index == f; ++e) {
      if (!sel[e]) continue;
      const EventInfo &info = r.info(e);
      const ZRecord &z = info.z;
      vacuum_file = info.vacuum;
      if (!branched) {               // the contents a vacuum file has (one) or Pb+Pb's
        branched = true;
        te.Branch("Y_full", Y_full.data(), Form("Y_full[%d]/F", N_Y));
        te.Branch("Y_one", Y_one.data(), Form("Y_one[%d]/F", N_Y));
        te.Branch("L_full", L_full.data(), Form("L_full[%d]/F", N_LS));
        if (!info.vacuum) {
          te.Branch("Y_zc", Y_zc.data(), Form("Y_zc[%d]/F", N_Y));
          te.Branch("Y_frag", Y_frag.data(), Form("Y_frag[%d]/F", N_Y));
          te.Branch("L_zc", L_zc.data(), Form("L_zc[%d]/F", N_LS));
          te.Branch("L_frag", L_frag.data(), Form("L_frag[%d]/F", N_LS));
        }
      }
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
      const int nx = xtra && !info.vacuum ? r.n_extra_frag(e) : 0;   // the extras' fragments
      std::vector<Track> tx;
      for (int j = 0; j < nx; ++j) add_tracks(r.extra(e, j), z, eta_max, j, tx);
      const int ns = info.vacuum ? nf : K;
      const int k_one = int(key_index(event_key(seed, name, info.local_event), ns));
      // S and rho, the per-sample means; the order of the operations as in file_pool
      std::vector<double> SF(N_Z, 0), RF(N_LAB, 0), Sf(N_Z), Szc(N_Z), Rf(N_LAB), Rzc(N_LAB);
      std::vector<double> YF(N_Y, 0), LF(N_LS, 0), Yf(N_Y), Yz(N_Y), Lf(N_LS), Lz(N_LS);
      z_hist(tf, z.y, z.phi, SF);
      lab_hist(tf, RF);
      side_hist(tf, z.y, z.phi, YF);
      lab_side_hist(tf, z.phi, LF);
      for (auto &x : SF) x /= nf;
      for (auto &x : RF) x /= nf;
      for (auto &x : YF) x /= nf;
      for (auto &x : LF) x /= nf;
      std::vector<Track> one;
      std::vector<double> RX(N_LAB, 0);
      if (!info.vacuum) {
        // the extras: in the full event, not in the Z-correlated yield
        std::vector<double> SX(N_Z, 0), YX(N_Y, 0), LX(N_LS, 0);
        if (nx > 0) {
          z_hist(tx, z.y, z.phi, SX);
          lab_hist(tx, RX);
          side_hist(tx, z.y, z.phi, YX);
          lab_side_hist(tx, z.phi, LX);
          for (auto &x : SX) x /= nx;
          for (auto &x : RX) x /= nx;
          for (auto &x : YX) x /= nx;
          for (auto &x : LX) x /= nx;
        }
        std::vector<double> SJ(N_Z, 0), RJ(N_LAB, 0), SB(N_Z, 0);
        z_hist(tj, z.y, z.phi, SJ);
        lab_hist(tj, RJ);
        z_hist(bc->tracks, z.y, z.phi, SB);
        for (int i = 0; i < N_Z; ++i) {
          const double sj = SJ[i] / K, sb = SB[i] / K;
          Sf[i] = nx > 0 ? sj + SF[i] + SX[i] : sj + SF[i];
          Szc[i] = sj - sb + SF[i];
        }
        for (int i = 0; i < N_LAB; ++i) {
          const double rj = RJ[i] / K, rb = bc->R[i] / K;
          Rf[i] = nx > 0 ? rj + RF[i] + RX[i] : rj + RF[i];
          Rzc[i] = rj - rb + RF[i];
        }
        std::vector<double> YJ(N_Y, 0), YB(N_Y, 0), LJ(N_LS, 0), LB(N_LS, 0);
        side_hist(tj, z.y, z.phi, YJ);
        side_hist(bc->tracks, z.y, z.phi, YB);
        lab_side_hist(tj, z.phi, LJ);
        lab_side_hist(bc->tracks, z.phi, LB);
        for (int i = 0; i < N_Y; ++i) {
          const double yj = YJ[i] / K, yb = YB[i] / K;
          Yf[i] = nx > 0 ? yj + YF[i] + YX[i] : yj + YF[i];
          Yz[i] = yj - yb + YF[i];
        }
        for (int i = 0; i < N_LS; ++i) {
          const double lj = LJ[i] / K, lb = LB[i] / K;
          Lf[i] = nx > 0 ? lj + LF[i] + LX[i] : lj + LF[i];
          Lz[i] = lj - lb + LF[i];
        }
        for (const Track &x : tj) if (x.sample == k_one) one.push_back(x);
        for (const Track &x : tf) if (x.sample == k_one % nf) one.push_back(x);
        if (nx > 0)
          for (const Track &x : tx) if (x.sample == k_one % nx) one.push_back(x);
      } else {
        Sf = SF; Szc = SF; Rf = RF; Rzc = RF;
        Yf = YF; Yz = YF; Lf = LF; Lz = LF;
        for (const Track &x : tf) if (x.sample == k_one) one.push_back(x);
      }
      std::vector<double> So(N_Z, 0), Yo(N_Y, 0);
      z_hist(one, z.y, z.phi, So);
      side_hist(one, z.y, z.phi, Yo);
      // the event's entry
      ev_event = e; ev_bg = bgid[e]; ev_w = w[e]; ev_y = z.y; ev_phi = z.phi;
      ev_k = k_one; ev_ns = ns;
      ev_local = info.local_event; ev_bin = info.pthat_bin;
      ev_ncoll = info.n_coll >= 0 ? double(info.n_coll) : std::nan("");
      const long unit = info.vacuum ? info.local_event : info.bg_unit;
      ev_unit = int(unit);
      for (int i = 0; i < N_Z; ++i) { S_full[i] = float(Sf[i]); S_one[i] = float(So[i]); }
      for (int i = 0; i < N_Y; ++i) {
        Y_full[i] = float(Yf[i]); Y_one[i] = float(Yo[i]);
        Y_zc[i] = float(Yz[i]); Y_frag[i] = float(YF[i]);
      }
      for (int i = 0; i < N_LS; ++i) {
        L_full[i] = float(Lf[i]); L_zc[i] = float(Lz[i]); L_frag[i] = float(LF[i]);
      }
      trk_pt.clear(); trk_eta.clear(); trk_phi.clear();
      for (const Track &x : one) {
        trk_pt.push_back(x.ipt);
        trk_eta.push_back(float(x.eta));
        trk_phi.push_back(float(x.phi));
      }
      te.Fill();
      ++n_ev;
      // the background's unweighted sums in the event's window
      BgSums &s = sums[{unit, info.pthat_bin}];
      if (s.n == 0) s.ncoll = ev_ncoll;
      ++s.n;
      for (int i = 0; i < N_Z; ++i) {
        s.S_full[i] += Sf[i];
        s.S_zc[i] += Szc[i];
      }
      for (int i = 0; i < N_LAB; ++i) s.R_full[i] += Rf[i];
      if (xtra)
        for (int i = 0; i < N_LAB; ++i) s.R_extra[i] += RX[i];
      if (!info.vacuum && !bg_rb.count(unit)) {
        std::vector<double> rb(N_LAB);
        for (int i = 0; i < N_LAB; ++i) rb[i] = bc->R[i] / K;
        bg_rb[unit] = rb;
      }
    }
    // per background and window, and per background
    Long64_t b_unit = 0, b_n = 0;
    int b_k = 0;
    double b_ncoll = 0;
    std::vector<double> Sfb(N_Z), Szb(N_Z), Rfb(N_LAB), RBb(N_LAB), Rxb(N_LAB);
    TTree tb("pool_bgw", "zjet_mixing pool: per hydro background and pTHat window, unweighted");
    tb.SetDirectory(&fo);
    tb.Branch("unit", &b_unit);
    tb.Branch("k", &b_k);
    tb.Branch("n", &b_n);
    tb.Branch("ncoll", &b_ncoll);
    tb.Branch("S_full", Sfb.data(), Form("S_full[%d]/D", N_Z));
    if (!vacuum_file) tb.Branch("S_zc", Szb.data(), Form("S_zc[%d]/D", N_Z));
    tb.Branch("R_full", Rfb.data(), Form("R_full[%d]/D", N_LAB));
    if (xtra) tb.Branch("R_extra", Rxb.data(), Form("R_extra[%d]/D", N_LAB));
    for (auto &p : sums) {
      b_unit = p.first.first; b_k = p.first.second;
      b_n = p.second.n; b_ncoll = p.second.ncoll;
      // into the branches' buffers (the vectors keep their storage)
      std::copy(p.second.S_full.begin(), p.second.S_full.end(), Sfb.begin());
      std::copy(p.second.S_zc.begin(), p.second.S_zc.end(), Szb.begin());
      std::copy(p.second.R_full.begin(), p.second.R_full.end(), Rfb.begin());
      std::copy(p.second.R_extra.begin(), p.second.R_extra.end(), Rxb.begin());
      tb.Fill();
    }
    TTree tr("pool_bgr", "zjet_mixing pool: per hydro background, bulk_bg's lab density");
    tr.SetDirectory(&fo);
    tr.Branch("unit", &b_unit);
    tr.Branch("RB", RBb.data(), Form("RB[%d]/D", N_LAB));
    for (auto &p : bg_rb) {
      b_unit = p.first;
      std::copy(p.second.begin(), p.second.end(), RBb.begin());
      tr.Fill();
    }
    fo.cd();
    te.Write();
    tb.Write();
    if (!vacuum_file) tr.Write();
    write_file_info(fo, file_record(r, f));
    write_grid(fo);
    fo.Close();
    gSystem->Rename((path + ".part").c_str(), path.c_str());
    const double dt =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s: %ld events, %zu (background, window) sums, %.1f s -> %s\n", name.c_str(),
                n_ev, sums.size(), dt, path.c_str());
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
