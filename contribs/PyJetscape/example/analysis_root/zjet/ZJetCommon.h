// example/analysis_root/zjet/ZJetCommon.h
//
// What the Z+jet analyses share, the C++ mirror of example/analysis/zjet/zjet_wake.py and
// zjet_mixing.py: the Z selection and event weights, the track mask, the Z frame, the
// binning (numpy's conventions, so that the C++ and Python map steps bin every track alike),
// and the portable random choices (zjet_wake.event_key / key_index / key_uniform).
//
// The map steps here (zhadron_pools.cxx, zjets.cxx) write ROOT files; the final stage
// (HEPData, chi^2, jackknife, figures) is the Python of example/analysis/zjet, shared
// with the Python map steps (README.md).

#pragma once

#include "../HadronFileReader.h"

#include <TSystem.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace zjet {

using hadrons_root::EventInfo;
using hadrons_root::Hadron;
using hadrons_root::Hadrons;
using hadrons_root::HadronFileReader;
using hadrons_root::ZRecord;

// ── binning: numpy.linspace edges, numpy.searchsorted(side="right") - 1, the last edge
// inclusive (zjet_mixing._index, the reader's histograms)
inline std::vector<double> linspace(double a, double b, int n) {
  std::vector<double> v(n);
  const double step = (b - a) / (n - 1);
  for (int i = 0; i < n; ++i) v[i] = i * step + a;
  v[n - 1] = b;
  return v;
}

inline int bin_index(double v, const std::vector<double> &e) {
  if (v == e.back()) return int(e.size()) - 2;
  const int j = int(std::upper_bound(e.begin(), e.end(), v) - e.begin()) - 1;
  return (j < 0 || j > int(e.size()) - 2) ? -1 : j;
}

// zjet_mixing's grid: track p_T classes, the Z frame (Delta y 0.4, Delta phi pi/10), the
// lab frame (eta 0.1, phi 6 degrees)
inline const std::vector<double> &PT() {
  static const std::vector<double> v = {1, 2, 4, 10};
  return v;
}
inline const std::vector<double> &DETA() {
  static const auto v = linspace(-4.8, 4.8, 25);
  return v;
}
inline const std::vector<double> &DPHI() {
  static const auto v = linspace(-M_PI / 2, 3 * M_PI / 2, 21);
  return v;
}
inline const std::vector<double> &ETA() {
  static const auto v = linspace(-2.4, 2.4, 49);
  return v;
}
inline const std::vector<double> &PHI() {
  static const auto v = linspace(-M_PI, M_PI, 61);
  return v;
}
constexpr int N_PT = 3, N_DY = 24, N_DPHI = 20, N_ETA = 48, N_PHI = 60;
constexpr int N_Z = N_PT * N_DY * N_DPHI, N_LAB = N_PT * N_ETA * N_PHI;

// angles into [lo, lo + 2 pi), numpy.mod's convention (zjet_wake.wrap_dphi)
inline double wrap_dphi(double x, double lo = -M_PI / 2) {
  double r = std::fmod(x - lo, 2 * M_PI);
  if (r != 0 && r < 0) r += 2 * M_PI;
  return r + lo;
}

// ── portable random choices: the same numbers as zjet_wake's
inline std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
inline std::uint64_t fnv1a64(const std::string &s) {
  std::uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char c : s) h = (h ^ c) * 0x100000001B3ULL;
  return h;
}
// (seed, the production file's name, the event in the file) -> a 64-bit key
inline std::uint64_t event_key(std::int64_t seed, const std::string &name, std::int64_t event) {
  return splitmix64(splitmix64(fnv1a64(name) ^ std::uint64_t(seed)) ^ std::uint64_t(event));
}
inline long key_index(std::uint64_t key, long n, std::uint64_t salt = 0) {
  return long(splitmix64(key ^ salt) % std::uint64_t(n));
}
inline double key_uniform(std::uint64_t key, std::uint64_t salt = 0) {
  return double(splitmix64(key ^ salt) >> 11) * 0x1.0p-53;
}

// the production file's name (its stem) of a *_hadrons.root
inline std::string stem_of(const std::string &path) {
  std::string b = gSystem->BaseName(path.c_str());
  const std::string suf = "_hadrons.root";
  if (b.size() > suf.size() && b.compare(b.size() - suf.size(), suf.size(), suf) == 0)
    b.resize(b.size() - suf.size());
  return b;
}

// ── the Z selection (zjet_wake.select; strict inequalities as there)
struct ZCuts {
  double pt_lo, pt_hi, ymax, m_lo, m_hi, lep_pt, lep_eta;   // ymax < 0: no |y_Z| cut
};
// CMS 2507.09307 (zjet_wake.CMS_2507)
constexpr ZCuts CMS_2507 = {40.0, 350.0, 2.4, 60.0, 120.0, 20.0, 2.4};

inline bool select(const ZRecord &z, const ZCuts &c) {
  if (!z.valid || !(z.pt > c.pt_lo && z.pt < c.pt_hi && z.m > c.m_lo && z.m < c.m_hi))
    return false;
  if (c.ymax >= 0 && !(std::fabs(z.y) < c.ymax)) return false;
  for (int i = 0; i < 2; ++i)
    if (!(z.lep_pt[i] > c.lep_pt && std::fabs(z.lep_eta[i]) < c.lep_eta)) return false;
  return true;
}

// Per-event weights sigma_k / N_k (x n_coll), normalized to mean 1 over the selected
// events, 0 outside them (zjet_wake.event_weights): N_k counts every event of window k.
inline std::vector<double> event_weights(const HadronFileReader &r, const std::vector<char> &sel,
                                         bool ncoll = true) {
  const long n = r.n_events();
  std::vector<double> w(n, 1.0);
  bool binned = false;
  for (long e = 0; e < n; ++e) binned |= r.info(e).pthat_bin >= 0;
  if (binned && r.n_windows() > 0) {
    std::vector<long> nk(r.n_windows(), 0);
    for (long e = 0; e < n; ++e)
      if (r.info(e).pthat_bin >= 0) ++nk[r.info(e).pthat_bin];
    for (long e = 0; e < n; ++e) {
      const int k = r.info(e).pthat_bin;
      w[e] = k >= 0 ? r.sigma_mb(k) / double(nk[k]) : std::nan("");
    }
  }
  bool all_ncoll = ncoll;
  for (long e = 0; e < n; ++e)
    if (sel[e] && r.info(e).n_coll < 0) all_ncoll = false;
  double sum = 0;
  long m = 0;
  for (long e = 0; e < n; ++e) {
    if (all_ncoll) w[e] *= r.info(e).n_coll;
    if (!sel[e]) w[e] = 0.0;
    else { sum += w[e]; ++m; }
  }
  if (m)
    for (long e = 0; e < n; ++e) w[e] /= sum / m;
  return w;
}

// the hydro background of every event as a dense id over the campaign: (file, bg_unit)
// pairs in sorted order, as mix.py's numpy.unique; vacuum events are their own
inline std::vector<long> background_ids(const HadronFileReader &r) {
  std::map<std::pair<int, int>, long> ids;
  std::vector<std::pair<int, int>> key(r.n_events());
  for (long e = 0; e < r.n_events(); ++e) {
    const EventInfo &i = r.info(e);
    key[e] = {i.file_index, i.vacuum ? i.local_event : i.bg_unit};
    ids[key[e]] = 0;
  }
  long k = 0;
  for (auto &p : ids) p.second = k++;
  std::vector<long> out(r.n_events());
  for (long e = 0; e < r.n_events(); ++e) out[e] = ids[key[e]];
  return out;
}

// ── tracks (zjet_wake.track_mask): charged, |eta| < eta_max, not the Z's leptons, inside
// the p_T classes
struct Track {
  int ipt;
  double eta, phi;
  int sample;
};

inline void add_tracks(const Hadrons &hs, const ZRecord &z, double eta_max, int sample,
                       std::vector<Track> &out) {
  for (const Hadron &h : hs) {
    if (!h.charged()) continue;
    const double eta = h.eta();
    if (!(std::fabs(eta) < eta_max) || z.is_lepton(h)) continue;
    const int ipt = bin_index(h.pt(), PT());
    if (ipt < 0) continue;
    out.push_back({ipt, eta, h.phi(), sample});
  }
}

// counts in the Z frame (N_Z, zjet_mixing.z_hist) and in the lab frame (N_LAB, lab_hist)
inline void z_hist(const std::vector<Track> &t, double y_z, double phi_z, std::vector<double> &h,
                   int sample = -1) {
  for (const Track &x : t) {
    if (sample >= 0 && x.sample != sample) continue;
    const int idy = bin_index(x.eta - y_z, DETA());
    const int idp = bin_index(wrap_dphi(x.phi - phi_z), DPHI());
    if (idy < 0 || idp < 0) continue;
    h[(x.ipt * N_DY + idy) * N_DPHI + idp] += 1.0;
  }
}
inline void lab_hist(const std::vector<Track> &t, std::vector<double> &h) {
  for (const Track &x : t) {
    const int ie = bin_index(x.eta, ETA()), ip = bin_index(x.phi, PHI());
    if (ie < 0 || ip < 0) continue;
    h[(x.ipt * N_ETA + ie) * N_PHI + ip] += 1.0;
  }
}

// ── command line: --name value pairs after the positional arguments
inline std::map<std::string, std::string> options(int argc, char **argv, int first) {
  std::map<std::string, std::string> o;
  for (int i = first; i + 1 < argc; i += 2) o[argv[i]] = argv[i + 1];
  return o;
}
inline std::pair<int, int> file_range(const std::map<std::string, std::string> &o, int n) {
  auto it = o.find("--files");
  if (it == o.end()) return {0, n};
  const auto c = it->second.find(':');
  const int a = std::stoi(it->second.substr(0, c));
  const int b = c == std::string::npos ? a + 1 : std::stoi(it->second.substr(c + 1));
  return {std::max(0, a), std::min(n, b)};
}

}  // namespace zjet
