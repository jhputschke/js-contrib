/*******************************************************************************
 * Copyright (c) The JETSCAPE Collaboration, 2018
 *
 * Modular, task-based framework for simulating all aspects of heavy-ion collisions
 *
 * For the list of contributors see AUTHORS.
 *
 * Report issues at https://github.com/JETSCAPE/JETSCAPE/issues
 *
 * or via email to bugs.jetscape@gmail.com
 *
 * Distributed under the GNU General Public License 3.0 (GPLv3 or later).
 * See COPYING for details.
 ******************************************************************************/

#ifndef FNOROOIN_H
#define FNOROOIN_H

#include <memory>

#include "FluidDynamics.h"
//#include "hydro_source_base.h"
#include "LiquefierBase.h"
//#include "data_struct.h" /in iSS ..
#include "JetScapeConstants.h"
#include "MakeUniqueHelper.h"
#include "eos.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <Riostream.h>
#include "TFile.h"
#include "TString.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"

using namespace Jetscape;

class FnoRooIn : public FluidDynamics {
private:

  Jetscape::real freezeout_temperature; //!< [GeV]
  //int doCooperFrye;                     //!< flag to run Cooper-Frye freeze-out
                                        //!< for soft particles
  //bool has_source_terms;
  //std::shared_ptr<HydroSourceJETSCAPE> hydro_source_terms_ptr;

  Jetscape::real x_min_fno, dx_fno;
  Jetscape::real y_min_fno, dy_fno;
  Jetscape::real deta_fno;
  Jetscape::real dtau_fno;
  //Jetscape::real z_min_fno, dz_fno;
  int nx_fno, ny_fno; // nz_fno
  int ntau_fno;
  int neta_fno;

  int n_features;
  bool fullHydroIn;

  TFile *f;
  TTree *t;

  //std::vector<std::vector<std::vector<std::vector<double>>>> *m_xyt;
  std::vector<std::vector<std::vector<std::vector<float>>>> *m_xyt;

  torch::jit::script::Module module;
  torch::Tensor output;
  torch::Device device;

  std::unique_ptr<EOS> fnoEOS;
  float GetTemperatureFromEos(float ed);

  // Allows the registration of the module so that it is available to be
  // used by the Jetscape framework.
  static RegisterJetScapeModule<FnoRooIn> reg;

public:

  FnoRooIn();
  ~FnoRooIn();

  void InitializeHydro(Parameter parameter_list);

  void EvolveHydro();
  void GetHydroInfo(Jetscape::real t, Jetscape::real x, Jetscape::real y,
                    Jetscape::real z, std::unique_ptr<FluidCellInfo> &fluid_cell_info_ptr);

  void GetHydroInfo_JETSCAPE(Jetscape::real t, Jetscape::real x, Jetscape::real y,
                    Jetscape::real z, std::unique_ptr<FluidCellInfo> &fluid_cell_info_ptr);

  void SetHydroGridInfo();
  void PassHydroEvolutionHistoryToFrameworkFromRoot();
  void PassHydroEvolutionHistoryToFramework();
  //void PassHydroSurfaceToFramework();

  //void add_a_liquefier(std::shared_ptr<LiquefierBase> new_liqueifier) {
    //liquefier_ptr = new_liqueifier;
    //hydro_source_terms_ptr->add_a_liquefier(liquefier_ptr.lock());
    //}

  //void GetHyperSurface(Jetscape::real T_cut,
  //                    SurfaceCellInfo *surface_list_ptr){};
  //void collect_freeze_out_surface();
};

#endif // FNOROOIN_H
