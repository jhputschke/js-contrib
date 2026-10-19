// example/analysis_root/common/JetMap.h
//
// Jets on hadron lists with fastjet, the C++ twin of example/analysis/jetcorr/jetmap.py:
// anti-k_T (R = 0.3 by default, Clusterer(R)), E scheme, p_T,raw > 1 GeV, |eta| < 2.  Pb+Pb:
// active areas with explicit ghosts on a fixed grid (|y| < 2.6, area 0.01, no scatter: the
// same jets as the Python map step) and rho(jet) the median p_T / area of the k_T R = 0.4
// jets with |y| < 2.2 in a strip |y - y_jet| < 1; p_T = p_T,raw - rho A.  Vacuum: no
// subtraction.  Per jet also its mass, the number of its (non-ghost) constituents and the
// leading one's p_T, and with Clusterer(R, true) rho_m of its strip (the median of the k_T
// jets' (m_T - p_T) / area, for mass subtraction; off by default: a quarter of the time).
// Nothing here knows the trigger: the caller chooses the particles.

#pragma once

#include <TTree.h>

#include <fastjet/AreaDefinition.hh>
#include <fastjet/ClusterSequence.hh>
#include <fastjet/ClusterSequenceArea.hh>
#include <fastjet/Selector.hh>
#include <fastjet/tools/JetMedianBackgroundEstimator.hh>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace jetcorr {

namespace fj = fastjet;

constexpr double R_JET = 0.3, ETA_JET = 2.0;   // R_JET: the default anti-k_T radius
constexpr double GHOST_AREA = 0.01;            // the fixed ghost grid's cell (PLAN D8: kept)
constexpr double STRIP = 1.0;                  // the rho strip's half width (Clusterer::rho_sel)

// anti-k_T jets of radius R (the rho estimate's k_T R = 0.4 does not change with it);
// rho_m: also every jet's rho_m
struct Clusterer {
  double R;
  bool rho_m;
  fj::JetDefinition akt, kt{fj::kt_algorithm, 0.4};
  fj::AreaDefinition area{fj::active_area_explicit_ghosts,
                          fj::GhostedAreaSpec(2.6, 1, GHOST_AREA, 0.0, 0.0)};
  fj::Selector rho_sel = fj::SelectorAbsRapMax(2.2) * fj::SelectorStrip(1.0);
  explicit Clusterer(double r = R_JET, bool with_rho_m = false)
      : R(r), rho_m(with_rho_m), akt(fj::antikt_algorithm, r) {}
};

// rho(jet) of JetMedianBackgroundEstimator(SelectorAbsRapMax(2.2) * SelectorStrip(1.0)),
// computed once per strip content.  With explicit ghosts the median depends only on which k_T
// jets lie in the strip |y - y_jet| < 1, and that set changes only where y_jet +- 1 crosses a
// k_T jet's rapidity: each distinct content (the number of k_T jets below y_jet - 1 and below
// y_jet + 1) is computed once.  A k_T jet within 1e-9 of a strip edge (where the selector's
// comparison decides) is never cached.  The same values as the estimator's, bit for bit
// (jetcorr/jetmap.py's StripRho).
struct StripRho {
  fj::ClusterSequenceArea cs;                  // the estimator's jets refer to it
  fj::JetMedianBackgroundEstimator bge;
  std::vector<double> ys;
  std::map<std::pair<long, long>, double> cache, cache_m;
  StripRho(const std::vector<fj::PseudoJet> &parts, const Clusterer &c)
      : cs(parts, c.kt, c.area), bge(c.rho_sel) {
    const auto jets = cs.inclusive_jets();
    // rho_m only on request: FastJet computes it by default, from every k_T jet's
    // constituents (a quarter of the map step's time); rho is the median of p_T / area
    // either way, the same with and without
    bge.set_compute_rho_m(c.rho_m);
    bge.set_jets(jets);
    for (const auto &j : jets) ys.push_back(j.rap());
    std::sort(ys.begin(), ys.end());
  }
  // the jet's strip content; false at a strip edge (never cached)
  bool key_of(const fj::PseudoJet &jet, std::pair<long, long> &k) const {
    const double y = jet.rap();
    long key[2];
    for (int s = 0; s < 2; ++s) {
      const double edge = y + (s ? STRIP : -STRIP);
      const long i = std::lower_bound(ys.begin(), ys.end(), edge) - ys.begin();
      if ((i < long(ys.size()) && std::fabs(ys[i] - edge) < 1e-9) ||
          (i > 0 && std::fabs(ys[i - 1] - edge) < 1e-9))
        return false;
      key[s] = i;
    }
    k = std::make_pair(key[0], key[1]);
    return true;
  }
  double rho(const fj::PseudoJet &jet) {
    std::pair<long, long> k;
    if (!key_of(jet, k)) return bge.rho(jet);
    const auto it = cache.find(k);
    if (it != cache.end()) return it->second;
    return cache[k] = bge.rho(jet);
  }
  // rho_m of the jet's strip (needs a Clusterer with rho_m)
  double rho_m(const fj::PseudoJet &jet) {
    std::pair<long, long> k;
    if (!key_of(jet, k)) return bge.rho_m(jet);
    const auto it = cache_m.find(k);
    if (it != cache_m.end()) return it->second;
    return cache_m[k] = bge.rho_m(jet);
  }
};

// the jets of a particle list: raw p_T, subtracted p_T, eta, phi, area, rho, mass, the number
// of real (non-ghost) constituents, the leading one's p_T
struct Jet {
  double pt_raw, pt, eta, phi, area, rho, m, nconst, lead, rho_m;
};

Jet make_jet(const fj::PseudoJet &j, double pt, double area, double rho, bool ghosts,
             double rho_m = 0) {
  int n = 0;
  double lead = 0;
  for (const auto &q : j.constituents())
    if (!ghosts || !q.is_pure_ghost()) {
      ++n;
      lead = std::max(lead, q.pt());
    }
  return {j.pt(), pt, j.eta(), j.phi_std(), area, rho, j.m(), double(n), lead, rho_m};
}

// every jet with p_T,raw > 1 GeV and |eta| < 2; ptmin cuts on p_T (subtracted in Pb+Pb), -inf
// keeps them all; bge: the particles' StripRho if the caller has it, else made here
std::vector<Jet> jets_of(const std::vector<fj::PseudoJet> &parts, bool subtract,
                         double ptmin, Clusterer &c, StripRho *bge = nullptr) {
  std::vector<Jet> out;
  if (!subtract) {
    fj::ClusterSequence cs(parts, c.akt);
    for (const auto &j : fj::sorted_by_pt(cs.inclusive_jets(1.0)))
      if (std::fabs(j.eta()) < ETA_JET && j.pt() >= ptmin)
        out.push_back(make_jet(j, j.pt(), 0, 0, false));
    return out;
  }
  std::unique_ptr<StripRho> own;
  if (!bge) {
    own = std::make_unique<StripRho>(parts, c);
    bge = own.get();
  }
  fj::ClusterSequenceArea cs(parts, c.akt, c.area);
  for (const auto &j : fj::sorted_by_pt(cs.inclusive_jets(1.0))) {
    if (!(std::fabs(j.eta()) < ETA_JET)) continue;
    if (j.pt() <= ptmin) continue;              // rho >= 0: p_T <= p_T,raw cannot pass
    const double rho = bge->rho(j), a = j.area(), pt = j.pt() - rho * a;
    if (pt > ptmin) out.push_back(make_jet(j, pt, a, rho, true, c.rho_m ? bge->rho_m(j) : 0));
  }
  return out;
}

// a cone of radius R around (eta, phi) (taken as a rapidity): the p_T sum of the particles
// within Delta R < R, rho there (bge's; nullptr: 0) and sum - rho pi R^2 (the random cones,
// the Z cone; jetcorr.jetmap.cone)
struct Cone {
  fj::PseudoJet axis;
  double sum, rho, dpt;
};
inline Cone cone(const std::vector<fj::PseudoJet> &parts, StripRho *bge, double eta, double phi,
                 double R) {
  fj::PseudoJet axis;
  axis.reset_PtYPhiM(1.0, eta, phi, 0.0);
  double sum = 0;
  for (const auto &p : parts)
    if (p.delta_R(axis) < R) sum += p.pt();
  const double rho = bge ? bge->rho(axis) : 0.0;
  return {axis, sum, rho, sum - rho * M_PI * R * R};
}

// the jet columns of a tree: <prefix>_pt_raw, <prefix>_pt, ... (one vector each, refilled per
// row; prefix "jet", or "nw" for the embedded event's jets); <prefix>_rho_m with rho_m
struct JetColumns {
  std::vector<double> v[10];
  int n;
  static constexpr const char *names[10] = {"pt_raw", "pt", "eta", "phi", "area", "rho", "m",
                                            "nconst", "lead_pt", "rho_m"};
  explicit JetColumns(bool rho_m = false) : n(rho_m ? 10 : 9) {}
  void branch(TTree &t, const std::string &prefix = "jet") {
    for (int i = 0; i < n; ++i) t.Branch((prefix + "_" + names[i]).c_str(), &v[i]);
  }
  void clear() {
    for (auto &x : v) x.clear();
  }
  void push(const Jet &j) {
    const double x[10] = {j.pt_raw, j.pt, j.eta, j.phi, j.area, j.rho, j.m, j.nconst, j.lead,
                          j.rho_m};
    for (int i = 0; i < n; ++i) v[i].push_back(x[i]);
  }
  size_t size() const { return v[0].size(); }
};

}  // namespace jetcorr
