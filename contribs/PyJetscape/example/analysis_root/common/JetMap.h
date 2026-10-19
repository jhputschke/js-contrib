// example/analysis_root/common/JetMap.h
//
// Jets on hadron lists with fastjet, the C++ twin of example/analysis/jetcorr/jetmap.py:
// anti-k_T R = 0.3, E scheme, p_T,raw > 1 GeV, |eta| < 2.  Pb+Pb: active areas with explicit
// ghosts on a fixed grid (|y| < 2.6, area 0.01, no scatter: the same jets as the Python map
// step) and rho(jet) the median p_T / area of the k_T R = 0.4 jets with |y| < 2.2 in a strip
// |y - y_jet| < 1; p_T = p_T,raw - rho A.  Vacuum: no subtraction.  Per jet also its mass,
// the number of its (non-ghost) constituents and the leading one's p_T.  Nothing here knows
// the trigger: the caller chooses the particles.

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
#include <string>
#include <utility>
#include <vector>

namespace jetcorr {

namespace fj = fastjet;

constexpr double R_JET = 0.3, ETA_JET = 2.0;
constexpr double STRIP = 1.0;                  // the rho strip's half width (Clusterer::rho_sel)

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
// (jetcorr/jetmap.py's StripRho).
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

// the jet columns of a tree: <prefix>_pt_raw, <prefix>_pt, ... (one vector each, refilled per
// row; prefix "jet", or "nw" for the embedded event's jets)
struct JetColumns {
  std::vector<double> v[9];
  static constexpr const char *names[9] = {"pt_raw", "pt", "eta", "phi", "area", "rho", "m",
                                           "nconst", "lead_pt"};
  void branch(TTree &t, const std::string &prefix = "jet") {
    for (int i = 0; i < 9; ++i) t.Branch((prefix + "_" + names[i]).c_str(), &v[i]);
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

}  // namespace jetcorr
