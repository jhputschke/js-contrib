// example/analysis_root/HadronFileReader.h
//
// The ROOT files of a hadronized campaign (prod_AuAu_0_10_jet/run_h5toROOT.py), read event
// by event as vectors of hadrons: the ROOT counterpart of jetscape.hadrons_h5's
// HadronFileReader.  Header only; RNTuple or TTree files alike (ROOT >= 6.34 for RNTuple).
//
//   #include "HadronFileReader.h"
//   hadrons_root::HadronFileReader r("out_root");   // every *_hadrons.root in there
//   for (long e : r.select_events()) {               // hadronized, not flagged
//     const auto &info = r.info(e);
//     for (int k = 0; k < r.n_oversamples(e); ++k) {
//       auto bkg  = r.bkg(e, k);                     // bulk_bg:  background (MUSIC_1)
//       auto dep  = r.bkg_dep(e, k);                 // bulk_jet: background + deposition
//       auto full = r.bkg_dep_frag(e, k);            // bulk_jet + jet_frag (+ extra_frag):
//                                                    // the jet event
//       for (const auto &h : full) if (h.charged() && std::fabs(h.eta()) < 1) ...
//     }
//   }
//
// The sources of one oversample k of event e:
//
//   bkg(e, k)            bulk_bg    iSS on the background's surface; sample k of the
//                                   background e used (shared by the events reusing it)
//   bkg_dep(e, k)        bulk_jet   iSS on the jet leg's surface: background + deposition
//   frag(e, j)           jet_frag   ColorlessHadronization of the surviving partons (with
//                                   extra hard scatterings: the trigger's showers only)
//   extra(e, j)          extra_frag ColorlessHadronization of the showers of the event's
//                                   other hard scatterings (PythiaGun <extraHard>,
//                                   docs/Plans/PLAN_multi_hard.md); files without them have
//                                   none
//   bkg_dep_frag(e, k)   bulk_jet sample k + jet_frag sample k mod n_frag (or frag_sample)
//                        + extra_frag sample k mod n_extra_frag (the same frag_sample),
//                        Hadron::origin 0 (bulk), 1 (fragment) or 2 (extras' fragment), as
//                        JetEvents.jet_event
//   bkg_extra(e, k)      bulk_bg sample k + extra_frag sample k mod n_extra_frag: the event
//                        without the trigger's jet and the wakes (B, the bulk alone, of the
//                        jet analyses); bkg(e, k) without extras
//
// The extras belong to the event, not to the trigger: bkg_dep_frag and bkg_extra include
// them whenever the file has them (EventInfo::has_extras); set_extras(false), or the
// argument extras = false, leaves them out (the event as without extra hard scatterings).
// Files without extra_frag read exactly as before.  A minimum-bias file (run_prod_jet.py
// --hard none) has no jet_frag: its full event is bulk_jet + extra_frag.
//
// Events are numbered globally over the files (sorted by name), as HadronFileReader numbers
// them when it reads the same production files: event_offset + events.event.  Oversamples
// of one event share one fluid: they are not independent events.  Weigh each event's
// samples with 1 / n_samples to get the per-event mean (hadron_distributions.C,
// README.md): bkg uses n_bg_samples(e), bkg_dep n_oversamples(e), frag n_frag(e), extra
// n_extra_frag(e).
//
// Cross sections of a --pthat-bins campaign come from the *_campaign.root next to the
// files; without one, from the files' own windows tables, combined as the campaign file
// does it (combine_windows).  weight_mb(k) = sigma_mb(k) / the hadronized events of
// window k in the files read, so it is right for a subset of the files too.
//
// set_filter keeps only the hadrons a predicate accepts (e.g. charged at |eta| < 1), which
// keeps the vectors small.  Positions (t, x, y, z) are read only with positions = true,
// and only from files that have them (run_h5toROOT.py --no-x leaves them out); else 0.

#pragma once

#include <TFile.h>
#include <TKey.h>
#include <TSystem.h>
#include <TSystemDirectory.h>
#include <TSystemFile.h>
#include <TTree.h>

#include <Math/Vector4D.h>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleView.hxx>
#include <ROOT/RVec.hxx>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace hadrons_root {

// Hadron::origin, as jetscape.hadrons_h5.ORIGIN
constexpr int kBulk = 0;
constexpr int kFrag = 1;
constexpr int kExtra = 2;                  // the fragments of the extra hard scatterings

// |pid| of the charged hadrons that survive iSS's decays (plus leptons), hadrons_h5.CHARGED
inline bool is_charged(int pid) {
  switch (std::abs(pid)) {
    case 211: case 321: case 2212: case 3222: case 3112: case 3312: case 3334: case 11:
    case 13:
      return true;
    default:
      return false;
  }
}

// 0.5 log(a / b) with both clipped at 1e-300, as hadrons_h5 does it (eta ~ +-690 along the
// beam, 0 for p = 0)
inline double half_log_ratio(double a, double b) {
  return 0.5 * std::log(std::max(a, 1e-300) / std::max(b, 1e-300));
}

struct Hadron {
  int pid = 0, pstat = 0;
  float E = 0, px = 0, py = 0, pz = 0;     // GeV
  float t = 0, x = 0, y = 0, z = 0;        // fm/c, fm (0 without positions)
  int origin = kBulk;                      // kBulk, kFrag or kExtra

  double pt() const { return std::hypot(double(px), double(py)); }
  double p() const { return std::sqrt(double(px) * px + double(py) * py + double(pz) * pz); }
  double eta() const { return half_log_ratio(p() + pz, p() - pz); }
  double rapidity() const { return half_log_ratio(double(E) + pz, double(E) - pz); }
  double phi() const { return std::atan2(double(py), double(px)); }
  double mass() const {
    const double m2 = double(E) * E - p() * p();
    return m2 > 0 ? std::sqrt(m2) : 0.0;
  }
  bool charged() const { return is_charged(pid); }
  ROOT::Math::PxPyPzEVector p4() const { return {px, py, pz, E}; }
};

using Hadrons = std::vector<Hadron>;

// which hadrons to keep (HadronFileReader::set_filter); empty: all
using Filter = std::function<bool(const Hadron &)>;

// a shower-initiating parton (events.ini_*, hadrons_h5.INITIATOR_COLUMNS; the extras':
// events.xini_*)
struct Initiator {
  int shower = 0, pid = 0, pstat = 0;
  double px = 0, py = 0, pz = 0, E = 0, x = 0, y = 0, z = 0, t = 0;
  int scatter = 0;                         // its hard scattering: 0 the trigger, 1, 2, ...

  double pt() const { return std::hypot(px, py); }
  double eta() const {
    const double p = std::sqrt(px * px + py * py + pz * pz);
    return half_log_ratio(p + pz, p - pz);
  }
  double rapidity() const { return half_log_ratio(E + pz, E - pz); }
  double phi() const { return std::atan2(py, px); }
};

// PythiaGun's trigger Z (events.z_*, lep{1,2}_*; jetscape.hadrons_h5.Z_COLUMNS): the lepton
// pair and its two decay leptons, hardest first.  valid is false without a Z.
struct ZRecord {
  bool valid = false;
  double pt = 0, y = 0, phi = 0, m = 0;
  int lep_pid[2] = {0, 0};
  double lep_pt[2] = {0, 0}, lep_eta[2] = {0, 0}, lep_phi[2] = {0, 0};

  // the momentum of lepton i (massless, from pt, eta, phi), as zjet_wake.lepton_p3
  std::array<double, 3> lep_p3(int i) const {
    return {lep_pt[i] * std::cos(lep_phi[i]), lep_pt[i] * std::sin(lep_phi[i]),
            lep_pt[i] * std::sinh(lep_eta[i])};
  }
  // h is one of the Z's two leptons: the same pid and momentum within rtol of |p_lep|
  // (zjet_wake.not_z_lepton); hadron-decay leptons are not
  bool is_lepton(const Hadron &h, double rtol = 1e-3) const {
    if (!valid) return false;
    for (int i = 0; i < 2; ++i) {
      if (h.pid != lep_pid[i]) continue;
      const auto p = lep_p3(i);
      const double pl = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      const double d = std::sqrt(std::pow(double(h.px) - p[0], 2) +
                                 std::pow(double(h.py) - p[1], 2) +
                                 std::pow(double(h.pz) - p[2], 2));
      if (d < rtol * std::max(pl, 1.0)) return true;
    }
    return false;
  }
};

// one hard scattering of an event with extra hard scatterings (events.scat_*,
// particlize_h5.SCATTER_COLUMNS): the trigger's first (scatter 0; none in a minimum-bias
// file), then the extras 1, 2, ...
struct Scattering {
  int scatter = 0, code = 0, n_partons = 0;    // code: Pythia's process code
  double pthat = 0, x = 0, y = 0, E = 0, pt = 0;   // GeV; x, y [fm]: its binary collision;
                                                   // E, pt: summed over its partons
};

// what is known about one event (the file's events table, plus the campaign weight)
struct EventInfo {
  long event = -1;              // global index over the files
  int file_index = -1;
  int local_event = -1;         // events.event: the event inside its production file
  int bg_unit = -1;             // the background it used (-1: unknown)
  int pthat_bin = -1;           // its pTHat window (-1: not a --pthat-bins run)
  double pthat = std::numeric_limits<double>::quiet_NaN();
  int n_samples_jet = 0, n_samples_bg = 0, n_samples_frag = 0;
  int n_samples_extra = 0;      // extra_frag's fragmentations (0: none, or no extras)
  int n_cells_jet = -1, n_cells_bg = -1;
  double sigma_file_mb = std::numeric_limits<double>::quiet_NaN();   // this file's estimate
  double weight_mb = std::numeric_limits<double>::quiet_NaN();       // weight_mb(pthat_bin)
  int n_coll = -1;              // the collision's binary collisions (-1: not recorded)
  ZRecord z;                    // the trigger Z (z.valid false: none)
  bool vacuum = false;          // a file without jet leg (run_prod_vacuum.py: jet_frag only)
  std::vector<Initiator> initiators;
  // extra hard scatterings (PythiaGun <extraHard>): has_extras if the file has extra_frag;
  // events.n_extra, extra_mu, extra_E, extra_pt (-1 / NaN if not recorded); the
  // scatterings and the extras' shower initiators (empty without)
  bool has_extras = false;
  int n_extra = -1;
  double extra_mu = std::numeric_limits<double>::quiet_NaN();
  double extra_E = std::numeric_limits<double>::quiet_NaN();
  double extra_pt = std::numeric_limits<double>::quiet_NaN();
  std::vector<Scattering> scatterings;
  std::vector<Initiator> extra_initiators;

  // hadronized: the jet leg has samples (vacuum files: the fragments)
  bool hadronized() const { return vacuum ? n_samples_frag > 0 : n_samples_jet > 0; }

  // the legs are not alike: background over jet-leg freeze-out cells outside
  // [1/1.2, 1.2], one surface is not a freeze-out surface (wake_hadrons.py)
  bool flagged() const {
    if (n_cells_jet < 0 || n_cells_bg < 0) return false;          // not recorded
    const double r = n_cells_jet > 0 ? double(n_cells_bg) / n_cells_jet : 0.0;
    return r < 1 / 1.2 || r > 1.2;
  }
  // the initiator with the largest pT (nullptr if there are none)
  const Initiator *leading() const {
    const Initiator *best = nullptr;
    for (const auto &i : initiators)
      if (!best || i.pt() > best->pt()) best = &i;
    return best;
  }
};

namespace detail {

inline bool ends_with(const std::string &s, const std::string &suffix) {
  return s.size() > suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline std::vector<std::string> list_files(const std::string &dir, const std::string &suffix) {
  std::vector<std::string> out;
  TSystemDirectory d(dir.c_str(), dir.c_str());
  std::unique_ptr<TList> files(d.GetListOfFiles());
  if (!files) return out;
  for (TObject *o : *files) {
    const std::string name = o->GetName();
    if (ends_with(name, suffix)) out.push_back(dir + "/" + name);
  }
  std::sort(out.begin(), out.end());
  return out;
}

inline std::string class_of(const std::string &path, const std::string &name) {
  std::unique_ptr<TFile> f(TFile::Open(path.c_str()));
  if (!f || f->IsZombie()) throw std::runtime_error("HadronFileReader: cannot open " + path);
  TKey *k = f->GetKey(name.c_str());
  return k ? k->GetClassName() : "";
}

// one tag of one file: the hadrons of an entry (= one sample), RNTuple or TTree
class TagSource {
 public:
  virtual ~TagSource() = default;
  virtual long n_entries() = 0;
  virtual void keys(std::vector<int> &unit, std::vector<int> &sample) = 0;
  virtual void append(long entry, int origin, const Filter &keep, Hadrons &out) = 0;
};

class RNTupleSource : public TagSource {
 public:
  RNTupleSource(const std::string &path, const std::string &tag, bool positions)
      : r_(ROOT::RNTupleReader::Open(tag, path)) {
    has_x_ = positions && r_->GetDescriptor().FindFieldId("t") != ROOT::kInvalidDescriptorId;
    pid_.emplace(r_->GetView<std::vector<int>>("pid"));
    pstat_.emplace(r_->GetView<std::vector<int>>("pstat"));
    const char *pn[4] = {"E", "px", "py", "pz"}, *xn[4] = {"t", "x", "y", "z"};
    for (int i = 0; i < 4; ++i) p_[i].emplace(r_->GetView<std::vector<float>>(pn[i]));
    if (has_x_)
      for (int i = 0; i < 4; ++i) x_[i].emplace(r_->GetView<std::vector<float>>(xn[i]));
  }
  long n_entries() override { return long(r_->GetNEntries()); }
  void keys(std::vector<int> &unit, std::vector<int> &sample) override {
    auto vu = r_->GetView<int>("unit");
    auto vs = r_->GetView<int>("sample");
    for (auto e : r_->GetEntryRange()) {
      unit.push_back(vu(e));
      sample.push_back(vs(e));
    }
  }
  void append(long entry, int origin, const Filter &keep, Hadrons &out) override {
    const auto &pid = (*pid_)(entry);
    const auto &pstat = (*pstat_)(entry);
    const auto &E = (*p_[0])(entry), &px = (*p_[1])(entry), &py = (*p_[2])(entry),
               &pz = (*p_[3])(entry);
    const std::vector<float> *x[4] = {nullptr, nullptr, nullptr, nullptr};
    if (has_x_)
      for (int i = 0; i < 4; ++i) x[i] = &(*x_[i])(entry);
    out.reserve(out.size() + pid.size());
    for (size_t i = 0; i < pid.size(); ++i) {
      Hadron h;
      h.pid = pid[i]; h.pstat = pstat[i];
      h.E = E[i]; h.px = px[i]; h.py = py[i]; h.pz = pz[i];
      if (has_x_) { h.t = (*x[0])[i]; h.x = (*x[1])[i]; h.y = (*x[2])[i]; h.z = (*x[3])[i]; }
      h.origin = origin;
      if (!keep || keep(h)) out.push_back(h);
    }
  }

 private:
  std::unique_ptr<ROOT::RNTupleReader> r_;
  bool has_x_ = false;
  std::optional<ROOT::RNTupleView<std::vector<int>>> pid_, pstat_;
  std::optional<ROOT::RNTupleView<std::vector<float>>> p_[4], x_[4];
};

class TTreeSource : public TagSource {
 public:
  TTreeSource(const std::string &path, const std::string &tag, bool positions)
      : f_(TFile::Open(path.c_str())) {
    if (!f_ || f_->IsZombie())
      throw std::runtime_error("HadronFileReader: cannot open " + path);
    t_ = f_->Get<TTree>(tag.c_str());
    if (!t_)
      throw std::runtime_error("HadronFileReader: no TTree " + tag + " in " + path);
    has_x_ = positions && t_->GetBranch("t");
    const long nmax = std::max(1L, long(t_->GetMaximum("n")) + 1);
    pid_.resize(nmax); pstat_.resize(nmax);
    for (auto &v : p_) v.resize(nmax);
    if (has_x_) for (auto &v : x_) v.resize(nmax);
    t_->SetBranchStatus("*", false);
    const char *pn[4] = {"E", "px", "py", "pz"}, *xn[4] = {"t", "x", "y", "z"};
    std::vector<const char *> on = {"n", "pid", "pstat", "E", "px", "py", "pz"};
    if (has_x_) on.insert(on.end(), {"t", "x", "y", "z"});
    for (const char *b : on) t_->SetBranchStatus(b, true);
    t_->SetBranchAddress("n", &n_);
    t_->SetBranchAddress("pid", pid_.data());
    t_->SetBranchAddress("pstat", pstat_.data());
    for (int i = 0; i < 4; ++i) t_->SetBranchAddress(pn[i], p_[i].data());
    if (has_x_)
      for (int i = 0; i < 4; ++i) t_->SetBranchAddress(xn[i], x_[i].data());
  }
  long n_entries() override { return long(t_->GetEntries()); }
  void keys(std::vector<int> &unit, std::vector<int> &sample) override {
    int u = 0, s = 0;
    t_->SetBranchStatus("unit", true);                 // only for the index
    t_->SetBranchStatus("sample", true);
    TBranch *bu = t_->GetBranch("unit"), *bs = t_->GetBranch("sample");
    t_->SetBranchAddress("unit", &u);
    t_->SetBranchAddress("sample", &s);
    for (long e = 0; e < n_entries(); ++e) {
      bu->GetEntry(e);
      bs->GetEntry(e);
      unit.push_back(u);
      sample.push_back(s);
    }
    t_->ResetBranchAddress(bu);
    t_->ResetBranchAddress(bs);
    t_->SetBranchStatus("unit", false);
    t_->SetBranchStatus("sample", false);
  }
  void append(long entry, int origin, const Filter &keep, Hadrons &out) override {
    t_->GetEntry(entry);
    out.reserve(out.size() + n_);
    for (int i = 0; i < n_; ++i) {
      Hadron h;
      h.pid = pid_[i]; h.pstat = pstat_[i];
      h.E = p_[0][i]; h.px = p_[1][i]; h.py = p_[2][i]; h.pz = p_[3][i];
      if (has_x_) { h.t = x_[0][i]; h.x = x_[1][i]; h.y = x_[2][i]; h.z = x_[3][i]; }
      h.origin = origin;
      if (!keep || keep(h)) out.push_back(h);
    }
  }

 private:
  std::unique_ptr<TFile> f_;
  TTree *t_ = nullptr;
  bool has_x_ = false;
  int n_ = 0;
  std::vector<int> pid_, pstat_;
  std::vector<float> p_[4], x_[4];
};

// a tag of one file, opened on first use: its source and the (unit, sample) -> entry index
struct Tag {
  std::unique_ptr<TagSource> src;
  std::unordered_map<std::int64_t, long> entry;
  static std::int64_t key(int unit, int sample) {
    return (std::int64_t(unit) << 32) | std::uint32_t(sample);
  }
};

}  // namespace detail

// the pTHat windows of a windows table (a production file's, or the campaign file's)
struct Windows {
  std::vector<double> pthat_lo, pthat_hi, sigma_mb, sigma_err_mb;
  std::vector<double> n_accepted;            // production files only (empty otherwise)
  size_t size() const { return pthat_lo.size(); }
  bool same_bins(const Windows &o) const {
    return pthat_lo == o.pthat_lo && pthat_hi == o.pthat_hi;
  }
};

namespace detail {

// a numeric column as doubles, whatever its type (uproot writes int64 as Long64_t in a
// TTree, std::int64_t in an RNTuple)
inline std::vector<double> take_double(ROOT::RDataFrame &df, const std::string &c) {
  const std::string t = df.GetColumnType(c);
  auto conv = [](const auto &v) { return std::vector<double>(v.begin(), v.end()); };
  if (t == "double" || t == "Double_t") return *df.Take<double>(c);
  if (t == "float" || t == "Float_t") return conv(*df.Take<float>(c));
  if (t == "int" || t == "Int_t" || t == "std::int32_t") return conv(*df.Take<int>(c));
  if (t == "Long64_t" || t == "long long") return conv(*df.Take<Long64_t>(c));
  if (t == "long" || t == "Long_t" || t == "std::int64_t") return conv(*df.Take<long>(c));
  throw std::runtime_error("HadronFileReader: column " + c + " has type " + t);
}

}  // namespace detail

// the windows table of a file; nullopt if it has none (not a --pthat-bins run).  Columns
// it lacks (cross sections of a job that did not finish) stay empty.
inline std::optional<Windows> read_windows(const std::string &path) {
  {
    std::unique_ptr<TFile> f(TFile::Open(path.c_str()));
    if (!f || f->IsZombie())
      throw std::runtime_error("HadronFileReader: cannot open " + path);
    if (!f->GetKey("windows")) return std::nullopt;
  }
  ROOT::RDataFrame df("windows", path);
  Windows w;
  w.pthat_lo = detail::take_double(df, "pthat_lo");
  w.pthat_hi = detail::take_double(df, "pthat_hi");
  if (df.HasColumn("sigma_mb")) w.sigma_mb = detail::take_double(df, "sigma_mb");
  if (df.HasColumn("sigma_err_mb")) w.sigma_err_mb = detail::take_double(df, "sigma_err_mb");
  if (df.HasColumn("n_accepted")) w.n_accepted = detail::take_double(df, "n_accepted");
  return w;
}

// The campaign's cross section per pTHat window from the production files' own windows
// tables, as run_h5toROOT.py's campaign file (HadronFileReader.pthat_bin_sigma) does it:
// the files' estimates weighted by their accepted events, sigma = sum n_k sigma_k / sum n_k,
// error sqrt(sum (n_k err_k)^2) / sum n_k.  nullopt, with the reason in *why, if a file
// has no windows, other windows than the first, or no cross sections.
inline std::optional<Windows> combine_windows(const std::vector<std::string> &files,
                                              std::string *why = nullptr) {
  auto fail = [why](const std::string &m) -> std::optional<Windows> {
    if (why) *why = m;
    return std::nullopt;
  };
  std::optional<Windows> out;
  std::vector<double> num, err2;
  for (const auto &path : files) {
    const auto w = read_windows(path);
    if (!w) return fail(path + " has no pTHat windows");
    const size_t n = w->size();
    if (w->sigma_mb.size() != n || w->sigma_err_mb.size() != n || w->n_accepted.size() != n)
      return fail(path + " has no cross sections (job not finished?)");
    if (!out) {
      out = *w;
      out->n_accepted.assign(n, 0.0);
      num.assign(n, 0.0);
      err2.assign(n, 0.0);
    } else if (!out->same_bins(*w)) {
      return fail(path + " has other pTHat windows than " + files[0]);
    }
    for (size_t k = 0; k < n; ++k) {
      out->n_accepted[k] += w->n_accepted[k];
      num[k] += w->n_accepted[k] * w->sigma_mb[k];
      err2[k] += std::pow(w->n_accepted[k] * w->sigma_err_mb[k], 2);
    }
  }
  if (!out) return fail("no files");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (size_t k = 0; k < out->size(); ++k) {
    const double n = out->n_accepted[k];
    out->sigma_mb[k] = n > 0 ? num[k] / n : nan;
    out->sigma_err_mb[k] = n > 0 ? std::sqrt(err2[k]) / n : nan;
  }
  return out;
}

class HadronFileReader {
 public:
  // source: a directory (its *_hadrons.root, and its *_campaign.root if any) or one
  // *_hadrons.root file.  positions: also read t, x, y, z.  campaign: the campaign file
  // (default: the *_campaign.root next to the files, if there is one; "none": don't use
  // one).  Without a campaign file the cross sections come from the files themselves.
  explicit HadronFileReader(const std::string &source, bool positions = false,
                            const std::string &campaign = "")
      : positions_(positions) {
    std::string dir = source;
    if (gSystem->AccessPathName(source.c_str()))      // (sic) true if it does NOT exist
      throw std::runtime_error("HadronFileReader: " + source + " does not exist");
    if (detail::ends_with(source, ".root")) {
      files_.push_back(source);
      dir = gSystem->GetDirName(source.c_str()).Data();
    } else {
      files_ = detail::list_files(source, "_hadrons.root");
    }
    if (files_.empty())
      throw std::runtime_error("HadronFileReader: no *_hadrons.root in " + source +
                               " (convert the campaign with prod_AuAu_0_10_jet/"
                               "run_h5toROOT.py first)");
    init(campaign.empty() ? find_campaign(dir) : campaign);
  }

  // the given *_hadrons.root files, in this order
  explicit HadronFileReader(const std::vector<std::string> &files, bool positions = false,
                            const std::string &campaign = "none")
      : positions_(positions), files_(files) {
    if (files_.empty()) throw std::runtime_error("HadronFileReader: no files");
    init(campaign.empty() ? "none" : campaign);
  }

  HadronFileReader(const HadronFileReader &) = delete;
  HadronFileReader &operator=(const HadronFileReader &) = delete;

  // ── bookkeeping ───────────────────────────────────────────────────────────────
  long n_events() const { return long(events_.size()); }
  int n_files() const { return int(files_.size()); }
  const std::string &file(int i) const { return files_.at(i); }
  long event_offset(int i) const { return offsets_.at(i); }
  const EventInfo &info(long event) const { return events_.at(check(event)); }
  const std::string &campaign_file() const { return campaign_; }

  int n_oversamples(long event) const { return info(event).n_samples_jet; }
  int n_bg_samples(long event) const { return info(event).n_samples_bg; }
  int n_frag(long event) const { return info(event).n_samples_frag; }
  int n_extra_frag(long event) const { return info(event).n_samples_extra; }

  // extra hard scatterings: whether file i (-1: any file) has extra_frag
  bool has_extras(int file = -1) const {
    for (const auto &e : events_)
      if ((file < 0 || e.file_index == file) && e.has_extras) return true;
    return false;
  }
  // the extras' fragments in the full event and in bkg_extra (default), or not
  // (set_extras(false): every event as without extra hard scatterings)
  void set_extras(bool on) { extras_ = on; }
  bool extras() const { return extras_; }
  // the event's full event and B include extras: its file has them and set_extras is on
  bool with_extras(long event) const { return extras_ && info(event).has_extras; }

  // pTHat windows (0 for a run without them, or without cross sections) and their cross
  // sections [mb]: from the campaign file, else combined from the files' windows tables
  int n_windows() const { return int(windows_.size()); }
  double pthat_lo(int window) const { return windows_.pthat_lo.at(window); }
  double pthat_hi(int window) const { return windows_.pthat_hi.at(window); }
  double sigma_mb(int window) const { return windows_.sigma_mb.at(window); }
  double sigma_err_mb(int window) const { return windows_.sigma_err_mb.at(window); }
  // hadronized events of the window in the files read (flagged ones too)
  long n_window_events(int window) const { return n_window_.at(window); }
  // sigma_mb / n_window_events: mb per event (0 for an empty window)
  double weight_mb(int window) const { return weight_mb_.at(window); }
  // where the cross sections come from: the campaign file, the files' windows tables, or
  // "" (none)
  const std::string &sigma_source() const { return sigma_source_; }

  // hadronized events (n_oversamples > 0) of a window (-1: all), without the flagged ones
  // unless keep_flagged
  std::vector<long> select_events(int window = -1, bool keep_flagged = false) const {
    std::vector<long> out;
    for (const auto &e : events_) {
      if (window >= 0 && e.pthat_bin != window) continue;
      if (!e.hadronized()) continue;
      if (!keep_flagged && e.flagged()) continue;
      out.push_back(e.event);
    }
    return out;
  }

  // keep only the hadrons keep(h) accepts (an empty function: all)
  void set_filter(Filter keep) { keep_ = std::move(keep); }
  void clear_filter() { keep_ = nullptr; }
  // charged hadrons at |eta| < eta_max only (eta_max <= 0: no eta cut)
  void set_charged_eta(double eta_max) {
    set_filter([eta_max](const Hadron &h) {
      return h.charged() && (eta_max <= 0 || std::fabs(h.eta()) < eta_max);
    });
  }

  // ── the hadrons of one oversample ─────────────────────────────────────────────
  // bulk_bg: sample k of the background event used
  Hadrons bkg(long event, int k) {
    const EventInfo &e = info(event);
    if (e.bg_unit < 0)
      throw std::runtime_error("HadronFileReader: event " + std::to_string(event) +
                               " has no bg_unit");
    Hadrons out;
    append(e.file_index, kTagBg, e.bg_unit, k, kBulk, out);
    return out;
  }
  // bulk_jet: sample k of event's jet leg, background + deposition
  Hadrons bkg_dep(long event, int k) {
    const EventInfo &e = info(event);
    Hadrons out;
    append(e.file_index, kTagJet, e.local_event, k, kBulk, out);
    return out;
  }
  // jet_frag: fragmentation j of event's surviving partons
  Hadrons frag(long event, int j) {
    const EventInfo &e = info(event);
    Hadrons out;
    append(e.file_index, kTagFrag, e.local_event, j, kFrag, out);
    return out;
  }
  // extra_frag: fragmentation j of the showers of event's extra hard scatterings (origin
  // kExtra); whatever set_extras says
  Hadrons extra(long event, int j) {
    const EventInfo &e = info(event);
    if (!e.has_extras)
      throw std::runtime_error("HadronFileReader: " + files_[e.file_index] +
                               " has no extra_frag (no extra hard scatterings)");
    Hadrons out;
    append(e.file_index, kTagExtra, e.local_event, j, kExtra, out);
    return out;
  }
  // the whole jet event: bulk_jet sample k plus fragmentation k mod n_frag (frag_sample
  // >= 0: that one) plus, with extras (with_extras, and extras), the extras' fragmentation
  // k mod n_extra_frag (frag_sample: the same index); origin tells bulk (kBulk), fragments
  // (kFrag) and the extras' fragments (kExtra) apart.  No fragments if the event has none.
  // Vacuum files (no jet leg): fragmentation k mod n_frag (or frag_sample) alone.
  Hadrons bkg_dep_frag(long event, int k, int frag_sample = -1, bool extras = true) {
    const EventInfo &e = info(event);
    Hadrons out;
    if (!e.vacuum) append(e.file_index, kTagJet, e.local_event, k, kBulk, out);
    if (e.n_samples_frag > 0) {
      const int j = frag_sample >= 0 ? frag_sample : k % e.n_samples_frag;
      append(e.file_index, kTagFrag, e.local_event, j, kFrag, out);
    }
    if (extras) append_extra(e, event, k, frag_sample, out);
    return out;
  }
  // B, the bulk alone, of the jet analyses: bulk_bg sample k of the background event used
  // plus, with extras (with_extras), the event's extras' fragmentation k mod n_extra_frag
  // (extra_sample >= 0: that one).  The extras are drawn per jet event, so B is per event,
  // not per background.  Without extras: bkg(event, k).
  Hadrons bkg_extra(long event, int k, int extra_sample = -1) {
    Hadrons out = bkg(event, k);
    append_extra(info(event), event, k, extra_sample, out);
    return out;
  }

  // the names of jetscape.hadrons_h5.HadronFileReader
  Hadrons background_event(long event, int k) { return bkg(event, k); }
  Hadrons jet_event(long event, int k, int frag_sample = -1, bool extras = true) {
    return bkg_dep_frag(event, k, frag_sample, extras);
  }

 private:
  enum { kTagJet = 0, kTagBg = 1, kTagFrag = 2, kTagExtra = 3, kNTags = 4 };
  static const char *tag_name(int t) {
    static const char *names[kNTags] = {"bulk_jet", "bulk_bg", "jet_frag", "extra_frag"};
    return names[t];
  }

  // the extras' fragmentation k mod n_extra_frag (j >= 0: j) of event, if with_extras
  void append_extra(const EventInfo &e, long event, int k, int j, Hadrons &out) {
    if (!with_extras(event) || e.n_samples_extra <= 0) return;
    append(e.file_index, kTagExtra, e.local_event, j >= 0 ? j : k % e.n_samples_extra,
           kExtra, out);
  }

  size_t check(long event) const {
    if (event < 0 || event >= n_events())
      throw std::out_of_range("HadronFileReader: event " + std::to_string(event) +
                              " out of range (0.." + std::to_string(n_events() - 1) + ")");
    return size_t(event);
  }

  static std::string find_campaign(const std::string &dir) {
    const auto camp = detail::list_files(dir, "_campaign.root");
    if (camp.size() > 1)
      std::printf("HadronFileReader: %zu campaign files in %s, using %s\n", camp.size(),
                  dir.c_str(), camp[0].c_str());
    return camp.empty() ? "none" : camp[0];
  }

  void init(const std::string &campaign) {
    tags_.resize(files_.size());
    for (size_t f = 0; f < files_.size(); ++f) {
      offsets_.push_back(n_events());
      read_events(int(f));
    }
    bool binned = false;
    for (const auto &e : events_) binned |= e.pthat_bin >= 0;
    if (!binned) return;                               // not a --pthat-bins run
    std::string why;
    const auto from_files = combine_windows(files_, &why);
    std::optional<Windows> camp;
    if (campaign != "none") {
      campaign_ = campaign;
      camp = read_windows(campaign);
      if (camp && from_files && !camp->same_bins(*from_files))
        throw std::runtime_error("HadronFileReader: " + campaign + " has other pTHat "
                                 "windows than the files: not their campaign file");
    }
    if (camp && camp->sigma_mb.size() == camp->size()) {
      windows_ = *camp;
      sigma_source_ = "campaign file " + campaign;
    } else if (from_files) {
      windows_ = *from_files;
      sigma_source_ = "the windows tables of the " + std::to_string(n_files()) + " file(s)";
      std::printf("HadronFileReader: no campaign file, cross sections from %s\n",
                  sigma_source_.c_str());
    } else {
      std::printf("HadronFileReader: no cross sections: no campaign file, and %s\n",
                  why.c_str());
      return;
    }
    n_window_.assign(windows_.size(), 0);
    for (const auto &e : events_)
      if (e.pthat_bin >= 0 && e.pthat_bin < n_windows() && e.hadronized())
        ++n_window_[e.pthat_bin];
    weight_mb_.resize(windows_.size());
    for (size_t k = 0; k < windows_.size(); ++k)
      weight_mb_[k] = n_window_[k] > 0 ? windows_.sigma_mb[k] / n_window_[k] : 0.0;
    for (auto &e : events_)
      if (e.pthat_bin >= 0 && e.pthat_bin < n_windows()) e.weight_mb = weight_mb_[e.pthat_bin];
  }

  // the events table of file f -> events_
  void read_events(int f) {
    const std::string &path = files_[f];
    ROOT::RDataFrame df("events", path);
    using IntCol = ROOT::RDF::RResultPtr<std::vector<int>>;
    auto ints = [&](const char *c) -> std::optional<IntCol> {
      if (!df.HasColumn(c)) return std::nullopt;
      return df.Take<int>(c);
    };
    auto event = df.Take<int>("event");
    auto bg_unit = ints("bg_unit"), bin = ints("pthat_bin"), nsj = ints("n_samples_jet"),
         nsb = ints("n_samples_bg"), nsf = ints("n_samples_frag"), ncj = ints("n_cells_jet"),
         ncb = ints("n_cells_bg"), nsx = ints("n_samples_extra"), n_extra = ints("n_extra");
    // the file's extra hard scatterings: its extra_frag ntuple (run_h5toROOT.py adds
    // n_samples_extra with it); per event the extras' record, as doubles
    const bool has_extras = df.HasColumn("n_samples_extra");
    std::map<std::string, std::vector<double>> xc;
    for (const char *c : {"extra_mu", "extra_E", "extra_pt"})
      if (df.HasColumn(c)) xc[c] = detail::take_double(df, c);
    std::optional<ROOT::RDF::RResultPtr<std::vector<double>>> pthat, sigma;
    if (df.HasColumn("pthat")) pthat = df.Take<double>("pthat");
    if (df.HasColumn("sigma_file_mb")) sigma = df.Take<double>("sigma_file_mb");
    const bool vacuum = !df.HasColumn("n_samples_jet");
    auto n_coll = ints("n_coll");
    // the Z record (hadrons_h5.Z_COLUMNS), as doubles whatever their type
    const bool has_z = df.HasColumn("z_pt");
    std::map<std::string, std::vector<double>> zc;
    if (has_z)
      for (const char *c : {"z_pt", "z_y", "z_phi", "z_m", "lep1_pid", "lep1_pt", "lep1_eta",
                            "lep1_phi", "lep2_pid", "lep2_pt", "lep2_eta", "lep2_phi"})
        zc[c] = detail::take_double(df, c);
    const bool has_ini = df.HasColumn("ini_px");
    std::optional<ROOT::RDF::RResultPtr<std::vector<ROOT::RVecI>>> ii[3];
    std::optional<ROOT::RDF::RResultPtr<std::vector<ROOT::RVecD>>> id[8];
    static const char *ini_i[3] = {"ini_shower", "ini_pid", "ini_pstat"};
    static const char *ini_d[8] = {"ini_px", "ini_py", "ini_pz", "ini_E",
                                   "ini_x",  "ini_y",  "ini_z",  "ini_t"};
    if (has_ini) {
      for (int c = 0; c < 3; ++c) ii[c] = df.Take<ROOT::RVecI>(ini_i[c]);
      for (int c = 0; c < 8; ++c) id[c] = df.Take<ROOT::RVecD>(ini_d[c]);
    }
    // the scatterings (scat_*) and the extras' initiators (xini_*), jagged
    const bool has_scat = df.HasColumn("scat_pthat"), has_xini = df.HasColumn("xini_px");
    std::optional<ROOT::RDF::RResultPtr<std::vector<ROOT::RVecI>>> si[3], xi[4];
    std::optional<ROOT::RDF::RResultPtr<std::vector<ROOT::RVecD>>> sd[5], xd[8];
    static const char *scat_i[3] = {"scat_scatter", "scat_code", "scat_n_partons"};
    static const char *scat_d[5] = {"scat_pthat", "scat_x", "scat_y", "scat_E", "scat_pt"};
    static const char *xini_i[4] = {"xini_shower", "xini_pid", "xini_pstat", "xini_scatter"};
    static const char *xini_d[8] = {"xini_px", "xini_py", "xini_pz", "xini_E",
                                    "xini_x",  "xini_y",  "xini_z",  "xini_t"};
    if (has_scat) {
      for (int c = 0; c < 3; ++c) si[c] = df.Take<ROOT::RVecI>(scat_i[c]);
      for (int c = 0; c < 5; ++c) sd[c] = df.Take<ROOT::RVecD>(scat_d[c]);
    }
    if (has_xini) {
      for (int c = 0; c < 4; ++c) xi[c] = df.Take<ROOT::RVecI>(xini_i[c]);
      for (int c = 0; c < 8; ++c) xd[c] = df.Take<ROOT::RVecD>(xini_d[c]);
    }
    auto get = [](std::optional<IntCol> &c, size_t i, int dflt) {
      return c ? (**c)[i] : dflt;
    };
    for (size_t i = 0; i < event->size(); ++i) {
      EventInfo e;
      e.event = n_events();
      e.file_index = f;
      e.local_event = (*event)[i];
      e.bg_unit = get(bg_unit, i, -1);
      e.pthat_bin = get(bin, i, -1);
      if (pthat) e.pthat = (**pthat)[i];
      if (sigma) e.sigma_file_mb = (**sigma)[i];
      e.n_samples_jet = get(nsj, i, 0);
      e.n_samples_bg = get(nsb, i, 0);
      e.n_samples_frag = get(nsf, i, 0);
      e.n_cells_jet = get(ncj, i, -1);
      e.n_cells_bg = get(ncb, i, -1);
      e.n_coll = get(n_coll, i, -1);
      e.vacuum = vacuum;
      e.has_extras = has_extras;
      e.n_samples_extra = get(nsx, i, 0);
      e.n_extra = get(n_extra, i, -1);
      if (xc.count("extra_mu")) e.extra_mu = xc["extra_mu"][i];
      if (xc.count("extra_E")) e.extra_E = xc["extra_E"][i];
      if (xc.count("extra_pt")) e.extra_pt = xc["extra_pt"][i];
      if (has_scat) {
        const size_t n = (**si[0])[i].size();
        e.scatterings.resize(n);
        for (size_t j = 0; j < n; ++j) {
          Scattering &s = e.scatterings[j];
          s.scatter = (**si[0])[i][j]; s.code = (**si[1])[i][j]; s.n_partons = (**si[2])[i][j];
          s.pthat = (**sd[0])[i][j]; s.x = (**sd[1])[i][j]; s.y = (**sd[2])[i][j];
          s.E = (**sd[3])[i][j];     s.pt = (**sd[4])[i][j];
        }
      }
      if (has_xini) {
        const size_t n = (**xi[0])[i].size();
        e.extra_initiators.resize(n);
        for (size_t j = 0; j < n; ++j) {
          Initiator &p = e.extra_initiators[j];
          p.shower = (**xi[0])[i][j]; p.pid = (**xi[1])[i][j]; p.pstat = (**xi[2])[i][j];
          p.scatter = (**xi[3])[i][j];
          p.px = (**xd[0])[i][j]; p.py = (**xd[1])[i][j]; p.pz = (**xd[2])[i][j];
          p.E = (**xd[3])[i][j];  p.x = (**xd[4])[i][j];  p.y = (**xd[5])[i][j];
          p.z = (**xd[6])[i][j];  p.t = (**xd[7])[i][j];
        }
      }
      if (has_z && std::isfinite(zc["z_pt"][i])) {
        ZRecord &z = e.z;
        z.valid = true;
        z.pt = zc["z_pt"][i]; z.y = zc["z_y"][i]; z.phi = zc["z_phi"][i]; z.m = zc["z_m"][i];
        for (int l = 0; l < 2; ++l) {
          const std::string p = "lep" + std::to_string(l + 1) + "_";
          z.lep_pid[l] = int(zc[p + "pid"][i]);
          z.lep_pt[l] = zc[p + "pt"][i]; z.lep_eta[l] = zc[p + "eta"][i];
          z.lep_phi[l] = zc[p + "phi"][i];
        }
      }
      if (has_ini) {
        const size_t n = (**ii[0])[i].size();
        e.initiators.resize(n);
        for (size_t j = 0; j < n; ++j) {
          Initiator &p = e.initiators[j];
          p.shower = (**ii[0])[i][j]; p.pid = (**ii[1])[i][j]; p.pstat = (**ii[2])[i][j];
          p.px = (**id[0])[i][j]; p.py = (**id[1])[i][j]; p.pz = (**id[2])[i][j];
          p.E = (**id[3])[i][j];  p.x = (**id[4])[i][j];  p.y = (**id[5])[i][j];
          p.z = (**id[6])[i][j];  p.t = (**id[7])[i][j];
        }
      }
      events_.push_back(std::move(e));
    }
  }

  detail::Tag &tag(int f, int t) {
    detail::Tag &g = tags_[f][t];
    if (g.src) return g;
    const std::string &path = files_[f];
    const std::string cls = detail::class_of(path, tag_name(t));
    if (cls == "ROOT::RNTuple")
      g.src = std::make_unique<detail::RNTupleSource>(path, tag_name(t), positions_);
    else if (cls == "TTree")
      g.src = std::make_unique<detail::TTreeSource>(path, tag_name(t), positions_);
    else
      throw std::runtime_error(std::string("HadronFileReader: no ") + tag_name(t) + " in " +
                               path);
    std::vector<int> unit, sample;
    g.src->keys(unit, sample);
    for (size_t e = 0; e < unit.size(); ++e)
      g.entry[detail::Tag::key(unit[e], sample[e])] = long(e);
    return g;
  }

  void append(int f, int t, int unit, int k, int origin, Hadrons &out) {
    detail::Tag &g = tag(f, t);
    const auto it = g.entry.find(detail::Tag::key(unit, k));
    if (it == g.entry.end())
      throw std::out_of_range(std::string("HadronFileReader: ") + files_[f] + ": no " +
                              tag_name(t) + " sample " + std::to_string(k) + " of unit " +
                              std::to_string(unit));
    g.src->append(it->second, origin, keep_, out);
  }

  bool positions_ = false;
  std::vector<std::string> files_;
  std::string campaign_;
  std::vector<long> offsets_;
  std::vector<EventInfo> events_;
  Windows windows_;
  std::string sigma_source_;
  std::vector<long> n_window_;
  std::vector<double> weight_mb_;
  std::vector<std::array<detail::Tag, kNTags>> tags_;
  Filter keep_;
  bool extras_ = true;
};

}  // namespace hadrons_root
