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

#include <stdio.h>
#include <sys/stat.h>
#include <MakeUniqueHelper.h>

#include <string>
#include <sstream>
#include <vector>
#include <memory>

#include "JetScapeLogger.h"
//#include "surfaceCell.h"
#include "FnoHydro.h"
#include "util.h"

#include <Riostream.h>
#include "TRandom.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TF1.h"
#include "TMath.h"
#include "TFile.h"
#include "TString.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"

using namespace Jetscape;

// Register the module with the base class
RegisterJetScapeModule<FnoHydro> FnoHydro::reg("FnoHydro");

//****************************************************************************************

// Get from actual grid ...
// Think about resizing etc maybe use same interpoolation as for fluid cells !???
// put in class and read in from preq module ...
/* *
const int nx =150;
const int ny =150;

// https://github.com/JETSCAPE/JETSCAPE/pull/254/files
// Rotation etc fix ... make sure not to repeat here !!!
// x vs y axis ...
int GetPreqCellIndex(int id_x, int id_y)
{
    id_x = std::min(nx - 1, std::max(0, id_x));
    id_y = std::min(ny - 1, std::max(0, id_y));
    return (id_x * ny + id_y);
}

void GetCellIndicesFromGlobalPreqIndex(int global_index, int& id_x, int& id_y)
{
    id_x = global_index / ny;
    id_y = global_index % ny;
}
*/

//****************************************************************************************

FnoHydro::FnoHydro() : device({}) {
  hydro_status = NOT_START;
  freezeout_temperature = 0.0;
  //doCooperFrye = 0;

  x_min_preq = dx_preq = y_min_preq = dy_preq = 0.0;
  nx_preq = ny_preq = 0;

  x_min_fno = dx_fno = y_min_fno = dy_fno = 0.0;
  nx_fno = ny_fno = 60;
  ntau_fno = 50;
  n_features = 4;

  neta_fno = 1;
  deta_fno = 0.0;

  //device = torch::Device({});
  //has_source_terms = false;
  SetId("FnoHydro");
  //hydro_source_terms_ptr =
  //    std::shared_ptr<HydroSourceJETSCAPE>(new HydroSourceJETSCAPE());
}

FnoHydro::~FnoHydro() {}

void FnoHydro::InitializeHydro(Parameter parameter_list) {
  JSINFO << "Initialize FnoHydro ...";
  VERBOSE(8);

  //*************************************************************************************************
  //REMARK: Somehow no real improvment when using more threads for tensor and model operations !????
  //*************************************************************************************************

  // For testing now, overwrites the env settings ...
  torch::set_num_threads(1);
  JSINFO << "Number of threads (libtorch OMP): " << torch::get_num_threads();
  JSINFO << "Default device: " << device; // << std::endl;
  JSINFO << BOLDCYAN << "IMPORTANT: edensity*Tau normalization hardcoded for now!!!";

  //REMARK: Same issue as with tracing, not everything in FNO model gets moved to the proper other device ...
  /*
    if (torch::mps::is_available()) {
        JSINFO<< "MPS device available :-). Use it ...";// << std::endl;
        device=torch::Device(torch::kMPS);
    }
    else if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        JSINFO << "CUDA device available :-). Use it ...";// << std::endl;
    }
    else {
        JSINFO << "MPS or CUDA device(s) not available. Use CPU ..."; // << std::endl;
        device = torch::Device(torch::kCPU);
    }
  */
  try {

    string input_model_file = GetXMLElementText({"Hydro", "FNO", "model_file"});
    JSINFO<<"Loading the traced Pytorch model : "<<input_model_file.c_str();//<<endl;
    module = torch::jit::load(input_model_file.c_str()); //, device);
    //module.to(device);

    /// ------------------------------------------------------------------
  }
  catch (const c10::Error& e) {
    JSWARN << "error loading the model :-( ...\n";
    exit(-1);
  }

  JSINFO << "--> traced Pytorch model loaded";
  /// ------------------------------------------------------------------

  // ******************************************************************************************************************************************
  //REMARK: Super-resolution does not seem to work, related to similar error in the embedding layer, might be hardcoded when serialzed !!!????
  // ---> try to follow up !!!! Check if it works in Python ....
  // ******************************************************************************************************************************************

  x_min_fno = GetXMLElementDouble({"Hydro", "FNO", "x_min"});
  y_min_fno = GetXMLElementDouble({"Hydro", "FNO", "y_min"});
  //z_min_fno = GetBinMin({"Hydro", "FNO", "z_min"});

  nx_fno = GetXMLElementInt({"Hydro", "FNO", "nx"});
  ny_fno = GetXMLElementInt({"Hydro", "FNO", "ny"});
  //nz_fno = GetXMLElementInt({"Hydro", "FNO", "nz"});

  dx_fno = -2*x_min_fno/(double) nx_fno;
  dy_fno = -2*y_min_fno/(double) ny_fno;
  //dz_fno = -2*z_min_fno/(double) nz_fno;

  ntau_fno = GetXMLElementInt({"Hydro", "FNO", "ntau"});
  neta_fno = GetXMLElementInt({"Hydro", "FNO", "neta"});

  deta_fno = GetXMLElementDouble({"Hydro", "FNO", "deta"});
  dtau_fno = GetXMLElementDouble({"Hydro", "FNO", "dtau"});

  n_features = GetXMLElementInt({"Hydro", "FNO", "n_features"});

  JSINFO<<"# of FNO training features = "<<n_features;

  //cout<<x_min_fno<<" "<<y_min_fno<<" "<<nx_fno<<" "<<ny_fno<<" "<<endl;
  //cout<<dx_fno<<" "<<dy_fno<<" "<<endl;
  //cout<<ntau_fno<<" "<<dtau_fno<<" "<<endl;

  freezeout_temperature = GetXMLElementDouble({"Hydro", "MUSIC", "freezeout_temperature"});

  int EOS_id_MUSIC = GetXMLElementInt({"Hydro", "FNOROOIN", "EOS_id_MUSIC"});
  JSINFO<<"Use EOS (Music id) = "<<EOS_id_MUSIC;
  fnoEOS=make_unique<EOS>(EOS_id_MUSIC);

  /*
  if (freezeout_temperature > 0.05) {
    music_hydro_ptr->set_parameter("T_freeze", freezeout_temperature);
  } else {
    JSWARN << "The input freeze-out temperature is too low! T_frez = "
           << freezeout_temperature << " GeV!";
    exit(1);
  }

  music_hydro_ptr->add_hydro_source_terms(hydro_source_terms_ptr);
  */
}

void FnoHydro::EvolveHydro() {
  VERBOSE(8);
  JSINFO << "Initialize density profiles in FnoHydro ...";

  auto start = std::chrono::high_resolution_clock::now();

  if (pre_eq_ptr == nullptr) {
    JSWARN << "Missing the pre-equilibrium module ...";
    exit(1);
  }

  dx_preq = ini->GetXStep();
  dy_preq = ini->GetZStep();
  x_min_preq = - ini->GetXMax();
  y_min_preq = - ini->GetYMax();
  double z_max = ini->GetZMax();
  int nz = ini->GetZSize();
  double tau0 = pre_eq_ptr->GetPreequilibriumEndTime();

  JSINFO << "hydro initial time set by PreEq module tau0 = " << tau0 << " fm/c";
  //JSINFO << "initial density profile dx = " << dx_preq << " fm";

  SetPreEqGridInfo();

  //cout<<dx_preq<<" "<<dy_preq<<" "<<z_max<<" "<<nz<<" "<<endl;
  //cout<<x_min_preq<<" "<<y_min_preq<<" "<<endl;
  //cout<<nx_preq<<" "<<ny_preq<<" "<<endl; //nz_preq<<" "<<endl;

  // last dimension, number of timesteps for predictions fix to only 1 for now !!!!
  torch::Tensor fno_input_tensor = torch::zeros({n_features, nx_fno, ny_fno, 1});

  string tensorShapeInput = "Input Tesnor shape from Preq : ";
  c10::IntArrayRef shape = fno_input_tensor.sizes();
  //cout<< "Tensor shape: ";
    for (int i = 0; i < shape.size(); ++i) {
      //std::cout << shape[i] << " ";
      tensorShapeInput += std::to_string(shape[i]) ; tensorShapeInput += " ";
    }
  //std::cout<< tensorShapeInput.c_str() << std::endl;
  JSINFO << tensorShapeInput.c_str();

  //REBIN attempt ...
  //kinda takes time maybe some of the torch operations quicker ... think aboutit !!!
  // OMP fore the loops ...
  // try with getting torch tensor and dimension from flat input vector ... !!!!
  for (int i=0;i<nx_fno;i++)
    for (int j=0;j<ny_fno;j++) {

        double x_In = x_min_fno + i*dx_fno;
        double y_In = y_min_fno + j*dy_fno;

        int preq_glob_index = GetPreqCellIndex(GetPreqIdX(x_In),GetPreqIdY(y_In));
        double ed = pre_eq_ptr->e_[preq_glob_index];

        //h2dIS_rebin->Fill(i,j,ed);

        //for (int k=0;k<50;k++) {
        int k=0;
        if (n_features == 4 ) {

            double T = GetTemperatureFromEos(ed);

            fno_input_tensor[0][i][j][k] = ed;
            fno_input_tensor[1][i][j][k] = T;
            // only for null preq module ... extend here at some point ... when a real dynamic evolution is used and how to get the first time-step ....
            fno_input_tensor[2][i][j][k] = 0;
            fno_input_tensor[3][i][j][k] = 0;
        }
        else if (n_features == 3) {
            fno_input_tensor[0][i][j][k] = ed;
            // only for null preq module ... extend here at some point ... when a real dynamic evolution is used and how to get the first time-step ....
            fno_input_tensor[1][i][j][k] = 0;
            fno_input_tensor[2][i][j][k] = 0;
        }
        else {JSWARN<<" Not enough FNO features edensity, vx,vy ... to be used further in JETSCAPE !"; exit(-1);}

        //}
    }

  fno_input_tensor = fno_input_tensor.repeat({1, 1, 1, ntau_fno});

  clear_up_evolution_data();
  PassPreEqEvolutionHistoryToFramework();

  SetHydroGridInfo();

  hydro_status = INITIALIZED;

  if (hydro_status == INITIALIZED) {

    JSINFO << "Running FNO model hydro time evolution prediction ...";

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(fno_input_tensor.unsqueeze(0));
    // Execute the model and turn its output into a tensor.
    output = module.forward(inputs).toTensor();

    torch::Tensor first_ntau_steps = fno_input_tensor.slice(3, 0, 1).unsqueeze(0);

    // This creates a tensor with shape [n_features, nx_fno, ny_fno, ntau_fno]
    //std::cout << "Original tensor shape: " << fno_input_tensor.sizes() << std::endl;
    //std::cout << "Selected first " << ntau_fno << " steps shape: " << first_ntau_steps.sizes() << std::endl;

    output = torch::cat({first_ntau_steps, output}, 4);

    shape = output.sizes();
    tensorShapeInput = "Output Tesnor shape from FNO : ";
    //c10::IntArrayRef shape = fno_input_tensor.sizes();
    //cout<< "Tensor shape: ";
    for (int i = 0; i < shape.size(); ++i) {
        //std::cout << shape[i] << " ";
        tensorShapeInput += std::to_string(shape[i]) ; tensorShapeInput += " ";
      }
    //std::cout<< tensorShapeInput.c_str() << std::endl;
    JSINFO << tensorShapeInput.c_str();

    bulk_info.ntau = bulk_info.ntau+1;

    for(auto t : inputs)
      t.toTensor().reset();
    inputs.clear();

    hydro_status = FINISHED;
  }

  auto end = std::chrono::high_resolution_clock::now();
  // Calculate duration
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

  //Grifd info for bulk history ...
  //cout<<bulk_info.Tau0()<<" "<<bulk_info.TauMax()<<" "<<bulk_info.ntau<<" "<<bulk_info.dtau<<endl;
  //cout<<bulk_info.XMin()<<" "<<bulk_info.XMax()<<" "<<bulk_info.nx<<" "<<bulk_info.dx<<endl;
  //cout<<bulk_info.YMin()<<" "<<bulk_info.YMax()<<" "<<bulk_info.ny<<" "<<bulk_info.dy<<endl;

  PassHydroEvolutionHistoryToFramework();

  JSINFO << "number of fluid cells received by the JETSCAPE: "
             << bulk_info.data.size();


  clearSurfaceCellVector();
  FindAConstantTemperatureSurface(freezeout_temperature, surfaceCellVector_);

  output.reset();


}

void FnoHydro::SetPreEqGridInfo() {

  bulk_info.tau_min = pre_eq_ptr->GetPreequilibriumStartTime();
  //bulk_info.dtau = pre_eq_ptr->GetPreequilibriumEvodtau();
  dx_preq = ini->GetXStep();
  dy_preq = ini->GetYStep();
  //dz_preq = ini->GetZStep();

  x_min_preq = - ini->GetXMax();
  y_min_preq = - ini->GetYMax();
  //z_max_preq = ini->GetZMax();

  nx_preq = ini->GetXSize();
  ny_preq = ini->GetYSize();
  //nz_preq = ini->GetZSize();

  bulk_info.x_min = x_min_preq;
  bulk_info.y_min = y_min_preq;

  JSINFO << "Use preEq evo: tau_0 = " << bulk_info.tau_min
         << " fm/c, and Xmin = Ymin = "<<bulk_info.x_min<< " fm, for FNO hydro (grid) time evolution prediction ...";
}


void FnoHydro::SetHydroGridInfo() {

  //REMARK: HArdcoded for new make read in from xml or some other info wrt FNO model (w/ and w/o super-resolution) !!!
  //Or get from output tensor dimensions ... !!!!

  bulk_info.neta = neta_fno; //boost invvariant ...
  bulk_info.nx = nx_fno;
  bulk_info.ny =ny_fno;
  //bulk_info.x_min = -music_hydro_ptr->get_hydro_x_max();
  bulk_info.dx =dx_fno;
  //bulk_info.y_min = -music_hydro_ptr->get_hydro_x_max();
  bulk_info.dy = dy_fno;
  bulk_info.eta_min = 0;
  bulk_info.deta = deta_fno;

  bulk_info.dtau=dtau_fno;
  bulk_info.ntau=ntau_fno;

  bulk_info.boost_invariant = true;
  /*
  if (flag_preEq_output_evo_to_memory == 0) {
    bulk_info.tau_min = music_hydro_ptr->get_hydro_tau0();
    bulk_info.dtau = music_hydro_ptr->get_hydro_dtau();
    bulk_info.ntau = music_hydro_ptr->get_ntau();
  } else {
    bulk_info.ntau = music_hydro_ptr->get_ntau() + pre_eq_ptr->get_ntau();
  }
  */
}

void FnoHydro::PassPreEqEvolutionHistoryToFramework() {
  JSINFO << "Passing preEq evolution information to JETSCAPE ... ";
  auto number_of_cells = pre_eq_ptr->get_number_of_fluid_cells();
  JSINFO << "total number of preEq fluid cells: " << number_of_cells;

  //SetPreEqGridInfo();

  for (int i = 0; i < number_of_cells; i++) {
    std::unique_ptr<FluidCellInfo> fluid_cell_info_ptr(new FluidCellInfo);
    pre_eq_ptr->get_fluid_cell_with_index(i, fluid_cell_info_ptr);
    StoreHydroEvolutionHistory(fluid_cell_info_ptr);
  }
  pre_eq_ptr->clear_evolution_data();
}

void FnoHydro::PassHydroEvolutionHistoryToFramework() {
  JSINFO << "Passing hydro evolution information to JETSCAPE ... ";

  int number_of_cells = output.numel()/(double) n_features; //music_hydro_ptr->get_number_of_fluid_cells();
  JSINFO << "total number of FNO prediction hydro fluid cells: " << number_of_cells;

  //Works too after permute .. and quicker than loops ... !!!!
  torch::Tensor flattened_tensor = torch::squeeze(output, 0);
  string tensorShapeInput = "Squeezed Tesnor shape from FNO : ";
  c10::IntArrayRef shape = flattened_tensor.sizes();
  //cout<< "Tensor shape: ";
  for (int i = 0; i < shape.size(); ++i) {
      //std::cout << shape[i] << " ";
      tensorShapeInput += std::to_string(shape[i]) ; tensorShapeInput += " ";
    }
  //std::cout<< tensorShapeInput.c_str() << std::endl;
  JSINFO << tensorShapeInput.c_str() <<" with bulkInfo nTau = "<<bulk_info.ntau;

  //Tremendous speed up !!!
  auto accessor = flattened_tensor.accessor<float, 4>();

  for (int k=0;k<bulk_info.ntau;k++)
    for (int i=0;i<bulk_info.nx;i++)
        for (int j=0;j<bulk_info.ny;j++)
        {
            std::unique_ptr<FluidCellInfo> fluid_cell_info_ptr(new FluidCellInfo);

            if (n_features == 4 ) {
                fluid_cell_info_ptr->energy_density = accessor[0][i][j][k];
                fluid_cell_info_ptr->temperature = accessor[1][i][j][k];
                fluid_cell_info_ptr->vx = accessor[2][i][j][k];
                fluid_cell_info_ptr->vy = accessor[3][i][j][k];
            }
            else if (n_features == 3) {

                //=============================================================
                //IMPORTANT: ednsity * Tau normalization hardcoded for now!!!!
                //=============================================================
                float eNormInverse = accessor[0][i][j][k]/(bulk_info.Tau0()+bulk_info.dtau*k);
                float mTemperature = GetTemperatureFromEos(eNormInverse);

                fluid_cell_info_ptr->energy_density = eNormInverse;
                fluid_cell_info_ptr->temperature = mTemperature; //GetTemperatureFromEos(eNormInverse);
                fluid_cell_info_ptr->vx = accessor[1][i][j][k];
                fluid_cell_info_ptr->vy = accessor[2][i][j][k];

                // Check if this cut alrerady here made the problem with the overall mult problem with semi central using central FNO ...
                /*
                if (mTemperature>freezeout_temperature)
                {
                    fluid_cell_info_ptr->energy_density = eNormInverse;
                    fluid_cell_info_ptr->temperature = mTemperature; //GetTemperatureFromEos(eNormInverse);
                    fluid_cell_info_ptr->vx = accessor[1][i][j][k];
                    fluid_cell_info_ptr->vy = accessor[2][i][j][k];
                }
                else
                {
                    fluid_cell_info_ptr->energy_density = 0;
                    fluid_cell_info_ptr->temperature = 0; //GetTemperatureFromEos(eNormInverse);
                    fluid_cell_info_ptr->vx = 0;
                    fluid_cell_info_ptr->vy = 0;
                }
                */
            }
            else {JSWARN<<" Not enough FNO features edensity, vx,vy ... to be used further in JETSCAPE !"; exit(-1);}

            fluid_cell_info_ptr->vz = 0.0 ;//fluidCell_ptr->vz;
            fluid_cell_info_ptr->entropy_density = 0.0; //fluidCell_ptr->sd;
            fluid_cell_info_ptr->pressure = 0.0; //fluidCell_ptr->pressure;
            fluid_cell_info_ptr->mu_B = 0.0;
            fluid_cell_info_ptr->mu_C = 0.0;
            fluid_cell_info_ptr->mu_S = 0.0;
            fluid_cell_info_ptr->qgp_fraction = 0.0;
            for (int i = 0; i < 4; i++) {
              for (int j = 0; j < 4; j++) {
                fluid_cell_info_ptr->pi[i][j] = 0.0;
              }
            }
            fluid_cell_info_ptr->bulk_Pi = 0.0;
            bulk_info.data.push_back(*fluid_cell_info_ptr);
          }

  flattened_tensor.reset();
}

/*
// *****************************************************************************************
// Remark: Make sure that the tensor operation flatten is doing the same as the global index
// of the bulk history --> check with filling via loops !!! and ouput histogram !!!
// ==> looks like this falattening is not the same as the global index !!!
// Check with moving time axis to front and then flatten !!!
// *****************************************************************************************

void FnoHydro::PassHydroEvolutionHistoryToFramework() {
  JSINFO << "Passing hydro evolution information to JETSCAPE ... ";

  auto start = std::chrono::high_resolution_clock::now();

  int number_of_cells = output.numel()/(double) n_features;  //music_hydro_ptr->get_number_of_fluid_cells();
  JSINFO << "total number of FNO prediction hydro fluid cells: " << number_of_cells;

  //Works too after permute .. and quicker than loops ... !!!!
  torch::Tensor flattened_tensor = torch::squeeze(output, 0);
  flattened_tensor = flattened_tensor.permute({0, 3, 1, 2});
  flattened_tensor = flattened_tensor.reshape({flattened_tensor.size(0), -1});

  string tensorShapeInput = "Flattened Tesnor shape from FNO : ";
  c10::IntArrayRef shape = flattened_tensor.sizes();
  //cout<< "Tensor shape: ";
  for (int i = 0; i < shape.size(); ++i) {
      //std::cout << shape[i] << " ";
      tensorShapeInput += std::to_string(shape[i]) ; tensorShapeInput += " ";
    }
  //std::cout<< tensorShapeInput.c_str() << std::endl;
  JSINFO << tensorShapeInput.c_str();

  auto end = std::chrono::high_resolution_clock::now();
  // Calculate duration
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

  start = std::chrono::high_resolution_clock::now();

  //Tremendous speed up !!!
  auto accessor = flattened_tensor.accessor<float, 2>();

  for (int i = 0; i < number_of_cells; i++) {
    std::unique_ptr<FluidCellInfo> fluid_cell_info_ptr(new FluidCellInfo);
    //music_hydro_ptr->get_fluid_cell_with_index(i, fluidCell_ptr);

    if (n_features == 4 ) {
        fluid_cell_info_ptr->energy_density = accessor[0][i]; // flattened_tensor[0][i].item<double>();
        fluid_cell_info_ptr->temperature = accessor[1][i]; //flattened_tensor[1][i].item<double>();
        fluid_cell_info_ptr->vx = accessor[2][i]; //flattened_tensor[2][i].item<double>();
        fluid_cell_info_ptr->vy = accessor[3][i]; //flattened_tensor[3][i].item<double>();
    }
    else if (n_features == 3) {
        fluid_cell_info_ptr->energy_density = accessor[0][i]; // flattened_tensor[0][i].item<double>();
        fluid_cell_info_ptr->temperature = GetTemperatureFromEos(accessor[0][i]); //flattened_tensor[1][i].item<double>();
        fluid_cell_info_ptr->vx = accessor[1][i]; //flattened_tensor[2][i].item<double>();
        fluid_cell_info_ptr->vy = accessor[2][i];
    }
    else {JSWARN<<" Not enough FNO features edensity, vx,vy ... to be used further in JETSCAPE !"; exit(-1);}

    fluid_cell_info_ptr->vz = 0.0 ;//fluidCell_ptr->vz;
    fluid_cell_info_ptr->entropy_density = 0.0; //fluidCell_ptr->sd;
    fluid_cell_info_ptr->pressure = 0.0; //fluidCell_ptr->pressure;
    fluid_cell_info_ptr->mu_B = 0.0;
    fluid_cell_info_ptr->mu_C = 0.0;
    fluid_cell_info_ptr->mu_S = 0.0;
    fluid_cell_info_ptr->qgp_fraction = 0.0;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        fluid_cell_info_ptr->pi[i][j] = 0.0;
      }
    }
    fluid_cell_info_ptr->bulk_Pi = 0.0;
    //StoreHydroEvolutionHistory(fluid_cell_info_ptr);
    bulk_info.data.push_back(*fluid_cell_info_ptr);

  }

  flattened_tensor.reset();

  end = std::chrono::high_resolution_clock::now();
  // Calculate duration
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken loop to copy to framework: " << duration.count() << " milliseconds" << std::endl;
}
*/

float FnoHydro::GetTemperatureFromEos(float ed) {
    return fnoEOS->get_temperature((float) ed/Util::hbarc, 0)*Util::hbarc;
}

void FnoHydro::GetHydroInfo(
    Jetscape::real t, Jetscape::real x, Jetscape::real y, Jetscape::real z,
    std::unique_ptr<FluidCellInfo> &fluid_cell_info_ptr) {
    GetHydroInfo_JETSCAPE(t, x, y, z, fluid_cell_info_ptr);
  //GetHydroInfo_MUSIC(t, x, y, z, fluid_cell_info_ptr);
}

void FnoHydro::GetHydroInfo_JETSCAPE(
    Jetscape::real t, Jetscape::real x, Jetscape::real y, Jetscape::real z,
    std::unique_ptr<FluidCellInfo> &fluid_cell_info_ptr) {
  auto temp = bulk_info.get_tz(t, x, y, z);
  fluid_cell_info_ptr = std::unique_ptr<FluidCellInfo>(new FluidCellInfo(temp));
}

int FnoHydro::GetPreqCellIndex(int id_x, int id_y) const
{
    id_x = std::min(nx_preq - 1, std::max(0, id_x));
    id_y = std::min(ny_preq - 1, std::max(0, id_y));
    return (id_x * ny_preq + id_y);
}

void FnoHydro::GetCellIndicesFromGlobalPreqIndex(int global_index, int& id_x, int& id_y) const
{
    id_x = global_index / ny_preq; //check, not important since mostly symmetric ... !
    id_y = global_index % ny_preq;
}
