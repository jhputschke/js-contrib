// example/analysis_root/common/AnaCommon.h
//
// What the C++ map steps of every hadron-correlation analysis share (zjet/, later dijet/), the
// C++ mirror of example/analysis/jetcorr: the binning (numpy's conventions, so that the C++
// and Python map steps bin every track alike), angles, the portable random choices
// (jetcorr.keys), the event weights and background ids (jetcorr.weights), and the command
// line.  Nothing here knows the trigger.

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

namespace jetcorr {

using hadrons_root::EventInfo;
using hadrons_root::HadronFileReader;

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

// angles into [lo, lo + 2 pi), numpy.mod's convention (jetcorr.keys.wrap_dphi)
inline double wrap_dphi(double x, double lo = -M_PI / 2) {
  double r = std::fmod(x - lo, 2 * M_PI);
  if (r != 0 && r < 0) r += 2 * M_PI;
  return r + lo;
}

// ── portable random choices: the same numbers as jetcorr.keys'
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

// Per-event weights sigma_k / N_k (x n_coll), normalized to mean 1 over the selected
// events, 0 outside them (jetcorr.weights.event_weights): N_k counts every event of window k.
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

}  // namespace jetcorr
