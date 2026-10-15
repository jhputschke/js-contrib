/*******************************************************************************
 * surface_rows.h
 *
 * Surface cells as table rows (SURFACE_CELL_COLUMNS), shared by the bindings that
 * hand surfaces to Python and back (bind_fluid_dynamics.cc) and to iSS directly
 * (bind_hadronization.cc), so both convert a stored row the same way.
 ******************************************************************************/
#pragma once

#include "SurfaceCellInfo.h"

namespace {

using Jetscape::SurfaceCellInfo;

// The fields iSS reads from a SurfaceCellInfo, in the order
// iSpectraSamplerWrapper::getSurfCellVector copies them into its FO_surf.
// SurfaceCellInfo is Jetscape::real = float, so float32 rows are lossless.
constexpr int kSurfaceNCols = 32;
constexpr const char *kSurfaceColumns[kSurfaceNCols] = {
    "tau",  "x",    "y",    "eta",  "ds0",  "ds1",  "ds2",  "ds3",
    "u0",   "u1",   "u2",   "u3",   "e",    "T",    "P",    "nB",
    "nQ",   "nS",   "muB",  "muQ",  "muS",  "pi00", "pi01", "pi02",
    "pi03", "pi11", "pi12", "pi13", "pi22", "pi23", "pi33", "Pi"};

inline void surface_cell_to_row(const SurfaceCellInfo &c, float *r) {
  r[0] = c.tau;
  r[1] = c.x;
  r[2] = c.y;
  r[3] = c.eta;
  for (int j = 0; j < 4; ++j) r[4 + j] = c.d3sigma_mu[j];
  for (int j = 0; j < 4; ++j) r[8 + j] = c.umu[j];
  r[12] = c.energy_density;
  r[13] = c.temperature;
  r[14] = c.pressure;
  r[15] = c.baryon_density;
  r[16] = c.electric_charge_density;
  r[17] = c.strangeness_density;
  r[18] = c.mu_B;
  r[19] = c.mu_Q;
  r[20] = c.mu_S;
  for (int j = 0; j < 10; ++j) r[21 + j] = c.pi[j];
  r[31] = c.bulk_Pi;
}

inline void surface_cell_from_row(const float *r, SurfaceCellInfo &c) {
  c.tau = r[0];
  c.x = r[1];
  c.y = r[2];
  c.eta = r[3];
  for (int j = 0; j < 4; ++j) c.d3sigma_mu[j] = r[4 + j];
  for (int j = 0; j < 4; ++j) c.umu[j] = r[8 + j];
  c.energy_density = r[12];
  c.temperature = r[13];
  c.pressure = r[14];
  c.baryon_density = r[15];
  c.electric_charge_density = r[16];
  c.strangeness_density = r[17];
  c.mu_B = r[18];
  c.mu_Q = r[19];
  c.mu_S = r[20];
  for (int j = 0; j < 10; ++j) c.pi[j] = r[21 + j];
  c.bulk_Pi = r[31];
}

}  // namespace
