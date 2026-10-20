// example/analysis_root/common/AnaCommon.h
//
// What the C++ map steps of every hadron-correlation analysis share (zjet/, later dijet/), the
// C++ mirror of example/analysis/jetcorr: the binning (numpy's conventions, so that the C++
// and Python map steps bin every track alike), angles, the portable random choices
// (jetcorr.keys), the event weights and background ids (jetcorr.weights), the outputs'
// bookkeeping records (jetcorr.mapio: file_info, run_info), and the command line.  Nothing
// here knows the trigger.

#pragma once

#include "../HadronFileReader.h"

#include <TFile.h>
#include <TSystem.h>
#include <TTree.h>

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
using hadrons_root::Windows;

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

// ── the bookkeeping records (jetcorr.weights.file_record, jetcorr.mapio): every output of
// a map step carries its production file's record, so that the event and background ids and
// the weights of any set of outputs, from several campaigns, follow at load time
struct FileRecord {
  std::string name;                          // the production file's stem
  long long n_events = 0;                    // all its events
  int vacuum = 0;
  std::vector<long long> units;              // its hydro backgrounds (vacuum: its events)
  std::vector<double> pthat_lo, pthat_hi, sigma_gen, sigma_err, n_accepted;   // per window
  std::vector<long long> n_stored;           // its events per window, before any cut
};

inline FileRecord file_record(const HadronFileReader &r, int f) {
  FileRecord rec;
  rec.name = stem_of(r.file(f));
  std::vector<long long> units;
  std::vector<long long> nk(std::max(r.n_windows(), 0), 0);
  for (long e = r.event_offset(f); e < r.n_events() && r.info(e).file_index == f; ++e) {
    const EventInfo &i = r.info(e);
    ++rec.n_events;
    rec.vacuum = i.vacuum;
    units.push_back(i.vacuum ? i.local_event : i.bg_unit);
    if (i.pthat_bin >= 0 && i.pthat_bin < long(nk.size())) ++nk[i.pthat_bin];
  }
  std::sort(units.begin(), units.end());
  units.erase(std::unique(units.begin(), units.end()), units.end());
  rec.units = units;
  if (const auto w = hadrons_root::read_windows(r.file(f))) {
    rec.pthat_lo = w->pthat_lo;
    rec.pthat_hi = w->pthat_hi;
    rec.sigma_gen = w->sigma_mb;
    rec.sigma_err = w->sigma_err_mb;
    rec.n_accepted = w->n_accepted;
    rec.n_stored.assign(w->size(), 0);
    for (size_t k = 0; k < w->size() && k < nk.size(); ++k) rec.n_stored[k] = nk[k];
  }
  return rec;
}

// the one-entry tree file_info (jetcorr.mapio.file_info_tree's names) into the current file
inline void write_file_info(TFile &fo, FileRecord rec) {
  TTree t("file_info", "the production file's record (jetcorr.mapio)");
  t.SetDirectory(&fo);
  t.Branch("name", &rec.name);
  t.Branch("n_events", &rec.n_events);
  t.Branch("vacuum", &rec.vacuum);
  t.Branch("units", &rec.units);
  t.Branch("win_pthat_lo", &rec.pthat_lo);
  t.Branch("win_pthat_hi", &rec.pthat_hi);
  t.Branch("win_sigma_gen", &rec.sigma_gen);
  t.Branch("win_sigma_err", &rec.sigma_err);
  t.Branch("win_n_accepted", &rec.n_accepted);
  t.Branch("win_n_stored", &rec.n_stored);
  t.Fill();
  fo.cd();
  t.Write();
}

// the one-entry tree run_info: the map step's settings (doubles and integers)
inline void write_run_info(TFile &fo, std::vector<std::pair<std::string, double>> dbl,
                           std::vector<std::pair<std::string, long long>> ints) {
  TTree t("run_info", "the map step's settings (jetcorr.mapio)");
  t.SetDirectory(&fo);
  for (auto &p : dbl) t.Branch(p.first.c_str(), &p.second);
  for (auto &p : ints) t.Branch(p.first.c_str(), &p.second);
  t.Fill();
  fo.cd();
  t.Write();
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
