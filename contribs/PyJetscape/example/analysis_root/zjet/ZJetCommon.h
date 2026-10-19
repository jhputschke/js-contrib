// example/analysis_root/zjet/ZJetCommon.h
//
// What the Z+jet analyses share, the C++ mirror of example/analysis/zjet/zjet_wake.py and
// zjet_mixing.py: the Z selection, the track mask, the Z frame and its grid.  The binning,
// the event weights and the portable random choices are ../common/AnaCommon.h's (the mirror of
// example/analysis/jetcorr), visible here through namespace zjet.
//
// The map steps here (zhadron_pools.cxx, zjets.cxx) write ROOT files; the final stage
// (HEPData, chi^2, jackknife, figures) is the Python of example/analysis/zjet, shared
// with the Python map steps (README.md).

#pragma once

#include "../common/AnaCommon.h"

#include <cmath>
#include <vector>

namespace zjet {

using namespace jetcorr;      // binning, keys, weights, command line (common/AnaCommon.h)

using hadrons_root::EventInfo;
using hadrons_root::Hadron;
using hadrons_root::Hadrons;
using hadrons_root::HadronFileReader;
using hadrons_root::ZRecord;

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

}  // namespace zjet
