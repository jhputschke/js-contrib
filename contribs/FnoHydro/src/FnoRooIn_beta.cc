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
#include "JetScapeSignalManager.h"
#include "JetEnergyLoss.h"
#include "JetScape.h"
//#include "surfaceCell.h"
#include "FnoRooIn.h"
#include "util.h"

#include <Riostream.h>
#include "TRandom.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TF1.h"
#include "TMath.h"

using namespace Jetscape;

// Register the module with the base class
RegisterJetScapeModule<FnoRooIn> FnoRooIn::reg("FnoRooIn");

//****************************************************************************************

void save_tensor_legacy_pickle(const torch::Tensor& tensor, const std::string& filename) {
    std::cout << "\n=== Saving tensor with legacy pickle format ===" << std::endl;
    try {
        // For LibTorch, we need to use the older serialization format
        // This is equivalent to _use_new_zipfile_serialization=False in Python

        // Convert tensor to CPU for compatibility
        torch::Tensor cpu_tensor = tensor.to(torch::kCPU);

        // Use torch::pickle_save for legacy format
        std::vector<char> buffer = torch::pickle_save(cpu_tensor);

        // Write to file
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }

        file.write(buffer.data(), buffer.size());
        file.close();

        std::cout << "Tensor saved with legacy pickle format to: " << filename << std::endl;
        std::cout << "File size: " << buffer.size() << " bytes" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error saving tensor with legacy format: " << e.what() << std::endl;
    }
}
//****************************************************************************************

FnoRooIn::FnoRooIn() : device({}) {
  hydro_status = NOT_START;
  freezeout_temperature = 0.0;
  //doCooperFrye = 0;

  x_min_fno = dx_fno = y_min_fno = dy_fno = 0.0;
  nx_fno = ny_fno = 60;
  ntau_fno = 50;
  n_features = 4;

  neta_fno = 1;
  deta_fno = 0.0;

  fullHydroIn = false;
  //device = torch::Device({});
  //has_source_terms = false;
  SetId("FnoRooIn");
  //hydro_source_terms_ptr =
  //    std::shared_ptr<HydroSourceJETSCAPE>(new HydroSourceJETSCAPE());
}

FnoRooIn::~FnoRooIn() {}

void FnoRooIn::InitializeHydro(Parameter parameter_list) {
  JSINFO << "Initialize FnoRooIn ...";
  VERBOSE(8);

  //*************************************************************************************************
  //REMARK: Somehow no real improvment when using more threads for tensor and model operations !????
  //*************************************************************************************************

  string input_root_file = GetXMLElementText({"Hydro", "FNOROOIN", "root_file"});

  JSINFO<<"Loading ROOT as input/full hydro file : "<<input_root_file.c_str();

  f=new TFile(input_root_file.c_str(),"READ");
  t=(TTree*)f->Get("t");

  m_xyt = nullptr;
  t->SetBranchAddress("user_res",&m_xyt);

  fullHydroIn = GetXMLElementInt({"Hydro", "FNOROOIN", "fullHydroIn"});

  if (fullHydroIn) {
    JSINFO << "Full hydro input is used ...";
  } else {
    JSINFO << "FNO prediction beyond first time step is used ...";

    // For testing now, overwrites the env settings ...
    torch::set_num_threads(1);
    JSINFO << "Number of threads (libtorch OMP): " << torch::get_num_threads();
    //device = torch::Device(torch::kMPS);
    JSINFO << "Default device: " << device; // << std::endl;

    try {
    // ************************************************************************************
    // Try improve speed : https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    // Use oneDNN Graph with TorchScript for inference ...
    // --> No real improvement ==> Follow up on this !???
    // ************************************************************************************
    //
    string input_model_file = GetXMLElementText({"Hydro", "FNOROOIN", "model_file"});
    JSINFO<<"Loading the traced Pytorch model : "<<input_model_file.c_str();//<<endl;
    module = torch::jit::load(input_model_file.c_str()); //, device);
    //module = torch::jit::no_grad(module); //freez    //
    module.to(device);
    }
    catch (const c10::Error& e) {
    JSWARN << "error loading the model :-( ...\n";
    exit(-1);
    }

    JSINFO << "--> traced Pytorch model loaded";
  }
  // ******************************************************************************************************************************************
  //REMARK: Super-resolution does not seem to work, related to similar error in the embedding layer, might be hardcoded when serialzed !!!????
  // ---> try to follow up !!!! Check if it works in Python ....
  // ******************************************************************************************************************************************

  x_min_fno = GetXMLElementDouble({"Hydro", "FNOROOIN", "x_min"});
  y_min_fno = GetXMLElementDouble({"Hydro", "FNOROOIN", "y_min"});
  //z_min_fno = GetBinMin({"Hydro", "FNO", "z_min"});

  nx_fno = GetXMLElementInt({"Hydro", "FNOROOIN", "nx"});
  ny_fno = GetXMLElementInt({"Hydro", "FNOROOIN", "ny"});
  //nz_fno = GetXMLElementInt({"Hydro", "FNO", "nz"});

  dx_fno = -2*x_min_fno/(double) nx_fno;
  dy_fno = -2*y_min_fno/(double) ny_fno;
  //dz_fno = -2*z_min_fno/(double) nz_fno;

  ntau_fno = GetXMLElementInt({"Hydro", "FNOROOIN", "ntau"});
  neta_fno = GetXMLElementInt({"Hydro", "FNOROOIN", "neta"});

  deta_fno = GetXMLElementDouble({"Hydro", "FNOROOIN", "deta"});
  dtau_fno = GetXMLElementDouble({"Hydro", "FNOROOIN", "dtau"});

  n_features = GetXMLElementInt({"Hydro", "FNOROOIN", "n_features"});

  JSINFO<<"# of FNO training features = "<<n_features;

  freezeout_temperature = GetXMLElementDouble({"Hydro", "MUSIC", "freezeout_temperature"});

  int EOS_id_MUSIC = GetXMLElementInt({"Hydro", "FNOROOIN", "EOS_id_MUSIC"});
  JSINFO<<"Use EOS (Music id) = "<<EOS_id_MUSIC;
  fnoEOS=make_unique<EOS>(EOS_id_MUSIC);

  //cout<<x_min_fno<<" "<<y_min_fno<<" "<<nx_fno<<" "<<ny_fno<<" "<<endl;
  //cout<<dx_fno<<" "<<dy_fno<<" "<<endl;
  //cout<<ntau_fno<<" "<<dtau_fno<<" "<<endl;
}

void FnoRooIn::EvolveHydro() {
  VERBOSE(8);
  JSINFO << "Initialize density profiles in FnoHydro ...";

  auto start = std::chrono::high_resolution_clock::now();

  //**************************************************************
  //REMARK: Read in somehow from XML ...
  //**************************************************************
  bulk_info.tau_min = 0.5; //pre_eq_ptr->GetPreequilibriumStartTime();
  JSINFO << "Use preEq evo: tau_0 = " << bulk_info.tau_min<< " fm/c.";

  //TH2D *h2dIS_root = new TH2D("h2dIS_root", "", 60, 0, 60, 60, 0, 60);
  //*********************************
  //WARNING: Memory leak ... !!!????
  //*********************************

  t->GetEntry(GetCurrentEvent());

  //cout<<GetCurrentEvent()<<endl;
  //cout<<t->GetEntries()<<endl;
  //cout<<m_xyt<<endl;
  clear_up_evolution_data();

  if (!fullHydroIn) {
    torch::Tensor fno_input_tensor = torch::zeros({n_features, nx_fno, ny_fno, 1});

    string tensorShapeInput = "Input Tensor shape from Root file tauId = 0 : ";
    c10::IntArrayRef shape = fno_input_tensor.sizes();
    //cout<< "Tensor shape: ";
        for (int i = 0; i < shape.size(); ++i) {
        //std::cout << shape[i] << " ";
        tensorShapeInput += std::to_string(shape[i]) ; tensorShapeInput += " ";
        }
    //std::cout<< tensorShapeInput.c_str() << std::endl;
    JSINFO << tensorShapeInput.c_str();

    int ntau_fno_loop = 1;
    //if (fullHydroIn) {ntau_fno_loop = ntau_fno;}
    //cout<<ntau_fno_loop<<endl;
    cout<<bulk_info.Tau0()<<" "<<bulk_info.TauMax()<<" "<<bulk_info.ntau<<" "<<bulk_info.dtau<<endl;

    for (int i=0;i<nx_fno;i++)
        for (int j=0;j<ny_fno;j++) {
            for (int k=0;k<ntau_fno_loop;k++)
            {
                //cout<<k<<endl;
                //h2dIS_root->Fill(i,j,(*m_xyt)[i][j][k][0]);
                if (n_features == 4 ) {
                    fno_input_tensor[0][i][j][k] = (*m_xyt)[i][j][k][0];
                    fno_input_tensor[1][i][j][k] = (*m_xyt)[i][j][k][1];
                    // only for null preq module ... extend here at some point ... when a real dynamic evolution is used and how to get the first time-step ....
                    fno_input_tensor[2][i][j][k] = 0;
                    fno_input_tensor[3][i][j][k] = 0;
                }
                else if (n_features == 3) {
                    // if (i==30 and j==30) {
                    //     cout<< "IS : " << (*m_xyt)[i][j][k][0] << " "<< (*m_xyt)[i][j][k][0]*bulk_info.Tau0() << " " <<GetTemperatureFromEos( (*m_xyt)[i][j][k][0])<<endl;
                    // }
                    fno_input_tensor[0][i][j][k] = (*m_xyt)[i][j][k][0]*bulk_info.Tau0();
                    // only for null preq module ... extend here at some point ... when a real dynamic evolution is used and how to get the first time-step ....
                    fno_input_tensor[1][i][j][k] = 0;
                    fno_input_tensor[2][i][j][k] = 0;
                }
                else {JSWARN<<" Not enough FNO features edensity, vx,vy ... to be used further in JETSCAPE !"; exit(-1);}
            }
        }

    fno_input_tensor = fno_input_tensor.repeat({1, 1, 1, ntau_fno});

    tensorShapeInput = "Input Tensor shape for FNO model with " + std::to_string(ntau_fno) + " copies of IS : ";
    shape = fno_input_tensor.sizes();
    //cout<< "Tensor shape: ";
        for (int i = 0; i < shape.size(); ++i) {
        //std::cout << shape[i] << " ";
        tensorShapeInput += std::to_string(shape[i]) ; tensorShapeInput += " ";
        }
    //std::cout<< tensorShapeInput.c_str() << std::endl;
    JSINFO << tensorShapeInput.c_str();

    //save_tensor_legacy_pickle(fno_input_tensor.clone(), "fno_input_tensor.pt");

    hydro_status = INITIALIZED;

    //if (fullHydroIn) hydro_status = FINISHED;

    if (hydro_status == INITIALIZED) {

        JSINFO << "Running FNO model hydro time evolution prediction ...";

        std::vector<torch::jit::IValue> inputs;
        //inputs.push_back(fno_input_tensor.unsqueeze(0).to(device)); //fno_input_tensor_unsqueeze); // .to(at::kMPS));
        inputs.push_back(fno_input_tensor.unsqueeze(0));
        // Execute the model and turn its output into a tensor.
        //torch::NoGradGuard no_grad;
        output = module.forward(inputs).toTensor();

        torch::Tensor first_ntau_steps = fno_input_tensor.slice(3, 0, 1).unsqueeze(0);

        // This creates a tensor with shape [n_features, nx_fno, ny_fno, ntau_fno]
        //std::cout << "Original tensor shape: " << fno_input_tensor.sizes() << std::endl;
        //std::cout << "Selected first " << ntau_fno << " steps shape: " << first_ntau_steps.sizes() << std::endl;

        output = torch::cat({first_ntau_steps, output}, 4);

        shape = output.sizes();
        tensorShapeInput = "Output Tensor shape from FNO with IS at Tau0 : ";
        //c10::IntArrayRef shape = fno_input_tensor.sizes();
        //cout<< "Tensor shape: ";
        for (int i = 0; i < shape.size(); ++i) {
            //std::cout << shape[i] << " ";
            tensorShapeInput += std::to_string(shape[i]) ; tensorShapeInput += " ";
        }
        JSINFO << tensorShapeInput.c_str();

        for(auto t : inputs)
            t.toTensor().reset();
        inputs.clear();

        //save_tensor_legacy_pickle(output, "fno_output_tensor.pt");

        fno_input_tensor.reset(); //maybe redundant ...

        hydro_status = FINISHED;
    }
    // Maybe move to Init ... !??? Think about ...
  }

  auto end = std::chrono::high_resolution_clock::now();
  // Calculate duration
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

  SetHydroGridInfo();

  //Grifd info for bulk history ...
  //  inline Jetscape::real TauMax() const { return (tau_min + (ntau - 1) * dtau); } !??? Why -1 !???
  //cout<<bulk_info.Tau0()<<" "<<bulk_info.TauMax()<<" "<<bulk_info.ntau<<" "<<bulk_info.dtau<<endl;
  //cout<<bulk_info.XMin()<<" "<<bulk_info.XMax()<<" "<<bulk_info.nx<<" "<<bulk_info.dx<<endl;
  //cout<<bulk_info.YMin()<<" "<<bulk_info.YMax()<<" "<<bulk_info.ny<<" "<<bulk_info.dy<<endl;

  start = std::chrono::high_resolution_clock::now();

  if (fullHydroIn) {
   //bulk_info.ntau = (*m_xyt)[0][0].size();
    cout<<bulk_info.ntau<<endl;
    PassHydroEvolutionHistoryToFrameworkFromRoot();
  } else {
    PassHydroEvolutionHistoryToFramework();
  }

  end = std::chrono::high_resolution_clock::now();
  // Calculate duration
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

  JSINFO << "number of fluid cells received by the JETSCAPE: "
             << bulk_info.data.size();

  // DEBUG QA ...
  // ///*
  // TH2D *h2dIS_rebin_torch_pred_bulkhist = new TH2D("h2dIS_rebin_torch_pred_bulkhist", "", 60, 0, 60, 60, 0, 60);

  // for (int i=0;i<nx_fno;i++)
  //   for (int j=0;j<ny_fno;j++)
  //   {
  //       h2dIS_rebin_torch_pred_bulkhist->Fill(i,j,bulk_info.data[bulk_info.CellIndex(45,i,j,0)].energy_density);
  //   }
  // //*/
  // TCanvas *c3 = new TCanvas("c3", "Canvas", 800, 600);
  // h2dIS_rebin_torch_pred_bulkhist->Draw("colz");
  // //h2dIS_root->Draw("colz");
  // c3->SaveAs("h2dIS_rebin_root_bulkhist.gif");

  output.reset();
  m_xyt->clear();

  // shape = output.sizes();
  // tensorShapeInput = "Output Tesnor shape from FNO : ";
  // //c10::IntArrayRef shape = fno_input_tensor.sizes();
  // //cout<< "Tensor shape: ";
  // for (int i = 0; i < shape.size(); ++i) {
  //     //std::cout << shape[i] << " ";
  //     tensorShapeInput += std::to_string(shape[i]) ; tensorShapeInput += " ";
  //   }
  // //std::cout<< tensorShapeInput.c_str() << std::endl;
  // JSINFO << tensorShapeInput.c_str();

  /*
  if (flag_surface_in_memory == 1) {
    clearSurfaceCellVector();
    PassHydroSurfaceToFramework();
  } else {
    collect_freeze_out_surface();
  }

  if (hydro_status == FINISHED && doCooperFrye == 1) {
    music_hydro_ptr->run_Cooper_Frye();
  }
  */

  // ---------------------------------------------
  // Fix seed for JetEnergyLoss to allow event by event FNO assessment...
  // Seems to be working with PGun !!! (Check with Pythia if needed ...)!!! and Matter, follow up with Lbt too!!!!

  auto jLossManager = JetScapeSignalManager::Instance()->GetJetEnergyLossManagerPointer();
  //cout<<jLossManage->
  for (auto it : jLossManager.lock()->GetTaskList())
  {
    //JSINFO << it->GetId();
    //taskMap.emplace(it->GetId(), it);
    for (auto it2 : it->GetTaskList())
    {
    //JSINFO  << " " << it2->GetId() ;
     //taskMap.emplace(it2->GetId(), it2);
     auto mRan =  std::dynamic_pointer_cast<JetEnergyLoss>(it2)->GetMt19937Generator();
     //cout<<GetCurrentEvent()+1<<endl;
     mRan->seed(GetCurrentEvent()+1);
    }
  }

  // ---------------------------------------------

}

void FnoRooIn::SetHydroGridInfo() {

  //REMARK: HArdcoded for new make read in from xml or some other info wrt FNO model (w/ and w/o super-resolution) !!!
  //Or get from output tensor dimensions ... !!!!

  bulk_info.neta = neta_fno; //boost invvariant ...
  bulk_info.nx = nx_fno;
  bulk_info.ny =ny_fno;
  bulk_info.x_min = x_min_fno;
  bulk_info.dx =dx_fno;
  bulk_info.y_min = y_min_fno;
  bulk_info.dy = dy_fno;
  bulk_info.eta_min = 0;
  bulk_info.deta = deta_fno;

  bulk_info.dtau=dtau_fno;
  bulk_info.ntau=ntau_fno;

  bulk_info.boost_invariant = true;
}

// *****************************************************************************************
// Remark: Make sure that the tensor operation flatten is doing the same as the global index
// of the bulk history --> check with filling via loops !!! and ouput histogram !!!
// ==> looks like this falattening is not the same as the global index !!!
// Check with moving time axis to front and then flatten !!!
// With accesors from PyTorch flattend or loops same performance !!!!
// *****************************************************************************************

void FnoRooIn::PassHydroEvolutionHistoryToFramework() {
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
  JSINFO << tensorShapeInput.c_str();

  //Tremendous speed up !!!
  auto accessor = flattened_tensor.accessor<float, 4>();

  for (int k=0;k<bulk_info.ntau+1;k++)
    for (int i=0;i<bulk_info.nx;i++)
        for (int j=0;j<bulk_info.ny;j++)
        {
            std::unique_ptr<FluidCellInfo> fluid_cell_info_ptr(new FluidCellInfo);
            if (n_features == 4 ) {
                fluid_cell_info_ptr->energy_density = accessor[0][i][j][k]; // flattened_tensor[0][i].item<double>();
                fluid_cell_info_ptr->temperature = accessor[1][i][j][k]; //flattened_tensor[1][i].item<double>();
                fluid_cell_info_ptr->vx = accessor[2][i][j][k]; //flattened_tensor[2][i].item<double>();
                fluid_cell_info_ptr->vy = accessor[3][i][j][k]; //flattened_tensor[3][i].item<double>();
            }
            else if (n_features == 3) {
                float eNormInverse = accessor[0][i][j][k]/(bulk_info.Tau0()+bulk_info.dtau*k);

                //DEBUG:
                //if (i==30 and j==30) {
                //    cout<< eNormInverse << " "<< accessor[0][i][j][k] << " "<<bulk_info.Tau0()+bulk_info.dtau*k<< " "<<k<<" "<<GetTemperatureFromEos(eNormInverse)<<endl;
                //    cout<<  (*m_xyt)[i][j][k][0] << " "<<bulk_info.Tau0()+bulk_info.dtau*k<< " "<<k<<" "<<GetTemperatureFromEos( (*m_xyt)[i][j][k][0])<<endl;
                //}

                fluid_cell_info_ptr->energy_density = eNormInverse; // flattened_tensor[0][i].item<double>();
                fluid_cell_info_ptr->temperature = GetTemperatureFromEos(eNormInverse); //flattened_tensor[1][i].item<double>();
                fluid_cell_info_ptr->vx = accessor[1][i][j][k]; //flattened_tensor[2][i].item<double>();
                fluid_cell_info_ptr->vy = accessor[2][i][j][k];

                //DEBUG:
                // if (i==30 and j==30) {
                //      cout<<  std::setprecision(1) << bulk_info.Tau0()+bulk_info.dtau*k << " " << std::setprecision(5) << fluid_cell_info_ptr->energy_density << " "<<fluid_cell_info_ptr->temperature <<endl;
                //      cout<<  std::setprecision(1) << bulk_info.Tau0()+bulk_info.dtau*k << " " << std::setprecision(5) <<(*m_xyt)[i][j][k][0] << GetTemperatureFromEos( (float) (*m_xyt)[i][j][k][0]) <<endl;
                // }
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
}

/* // Flattened ...
void FnoRooIn::PassHydroEvolutionHistoryToFramework() {
  JSINFO << "Passing hydro evolution information to JETSCAPE ... ";

  int number_of_cells = output.numel()/(double) n_features; //music_hydro_ptr->get_number_of_fluid_cells();
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
        fluid_cell_info_ptr->energy_density = accessor[0][i]; ///(bulk_info.Tau0())+bulk_info.dtau*; // flattened_tensor[0][i].item<double>();
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
}
*/

// ROOT and st vector significantly faster than torch vector !??? Follow up ...
void FnoRooIn::PassHydroEvolutionHistoryToFrameworkFromRoot()
{
    JSINFO << "Passing hydro evolution information to JETSCAPE from ROOT file ... ";

    for (int k=0;k<bulk_info.ntau+1;k++)
      for (int i=0;i<bulk_info.nx;i++)
          for (int j=0;j<bulk_info.ny;j++)
          {
              std::unique_ptr<FluidCellInfo> fluid_cell_info_ptr(new FluidCellInfo);
              //music_hydro_ptr->get_fluid_cell_with_index(i, fluidCell_ptr);

              //cout<<i<<" "<<j<<" "<<k<<endl;
              //cout<<output[0][i][j][k].item<double>()<<endl;
              if (n_features == 4 ) {
                fluid_cell_info_ptr->energy_density = (*m_xyt)[i][j][k][0];
                fluid_cell_info_ptr->temperature = (*m_xyt)[i][j][k][1];
                fluid_cell_info_ptr->vx =(*m_xyt)[i][j][k][2];
                fluid_cell_info_ptr->vy =(*m_xyt)[i][j][k][3];
              }
              else if (n_features == 3) {
                  fluid_cell_info_ptr->energy_density = (*m_xyt)[i][j][k][0];
                  fluid_cell_info_ptr->temperature = GetTemperatureFromEos((float) (*m_xyt)[i][j][k][0]);
                  fluid_cell_info_ptr->vx =(*m_xyt)[i][j][k][1];
                  fluid_cell_info_ptr->vy =(*m_xyt)[i][j][k][2];
                  //DEBUG:
                  //if (i==30 and j==30)
                  //     cout<< bulk_info.Tau0()+bulk_info.dtau*k << " " <<fluid_cell_info_ptr->energy_density << " "<<fluid_cell_info_ptr->temperature <<endl;
                  //    cout<<  (*m_xyt)[i][j][k][0] << " "<<bulk_info.Tau0()+bulk_info.dtau*k<< " "<<k<<" "<<GetTemperatureFromEos( (*m_xyt)[i][j][k][0])<<endl;
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

              //DEBUG: Check EOS from Music to get temperatyre from edensity and compare to stored temperature from ROOT file ...
              // See check in bulkWriterFull, exact results!!??? Follow up !!!!!!

              // if ((*m_xyt)[i][j][k][0] > 0.1 && k<1) {
              //   //With correct units !!!! edinst/hbarc and temp from EOS *habrc!!!!
              //   //cout<<(*m_xyt)[i][j][k][0]<<" "<<(*m_xyt)[i][j][k][1]<<" via EOS = "<<fnoEOS->get_temperature((*m_xyt)[i][j][k][0]/Util::hbarc, 0)*Util::hbarc<<endl;
              //   cout<<(*m_xyt)[i][j][k][0]<<" "<<(*m_xyt)[i][j][k][1]<<" via EOS = "<<GetTemperatureFromEos((*m_xyt)[i][j][k][0])<<endl;
              //   //cout<<Util::hbarc<<endl;
              //   //cout<<(*m_xyt)[i][j][k][1]/fnoEOS->get_temperature((*m_xyt)[i][j][k][0], 0)<<endl;
              // }

          }
}

float FnoRooIn::GetTemperatureFromEos(float ed) {
    return fnoEOS->get_temperature((float) ed/Util::hbarc, 0)*Util::hbarc;
}

/*
void FnoHydro::PassHydroSurfaceToFramework() {
    JSINFO << "Passing hydro surface cells to JETSCAPE ... ";
    auto number_of_cells = music_hydro_ptr->get_number_of_surface_cells();
    JSINFO << "total number of fluid cells: " << number_of_cells;
    SurfaceCell surfaceCell_i;
    for (int i = 0; i < number_of_cells; i++) {
        SurfaceCellInfo surface_cell_info;
        music_hydro_ptr->get_surface_cell_with_index(i, surfaceCell_i);
        surface_cell_info.tau = surfaceCell_i.xmu[0];
        surface_cell_info.x = surfaceCell_i.xmu[1];
        surface_cell_info.y = surfaceCell_i.xmu[2];
        surface_cell_info.eta = surfaceCell_i.xmu[3];
        double u[4];
        for (int j = 0; j < 4; j++) {
            surface_cell_info.d3sigma_mu[j] = surfaceCell_i.d3sigma_mu[j];
            surface_cell_info.umu[j] = surfaceCell_i.umu[j];
        }
        surface_cell_info.energy_density = surfaceCell_i.energy_density;
        surface_cell_info.temperature = surfaceCell_i.temperature;
        surface_cell_info.pressure = surfaceCell_i.pressure;
        surface_cell_info.baryon_density = surfaceCell_i.rho_b;
        surface_cell_info.mu_B = surfaceCell_i.mu_B;
        surface_cell_info.mu_Q = surfaceCell_i.mu_Q;
        surface_cell_info.mu_S = surfaceCell_i.mu_S;
        for (int j = 0; j < 10; j++) {
            surface_cell_info.pi[j] = surfaceCell_i.shear_pi[j];
        }
        surface_cell_info.bulk_Pi = surfaceCell_i.bulk_Pi;
        StoreSurfaceCell(surface_cell_info);
    }
}
*/

void FnoRooIn::GetHydroInfo(
    Jetscape::real t, Jetscape::real x, Jetscape::real y, Jetscape::real z,
    std::unique_ptr<FluidCellInfo> &fluid_cell_info_ptr) {
    GetHydroInfo_JETSCAPE(t, x, y, z, fluid_cell_info_ptr);
  //GetHydroInfo_MUSIC(t, x, y, z, fluid_cell_info_ptr);
}

void FnoRooIn::GetHydroInfo_JETSCAPE(
    Jetscape::real t, Jetscape::real x, Jetscape::real y, Jetscape::real z,
    std::unique_ptr<FluidCellInfo> &fluid_cell_info_ptr) {
  auto temp = bulk_info.get_tz(t, x, y, z);
  fluid_cell_info_ptr = std::unique_ptr<FluidCellInfo>(new FluidCellInfo(temp));
}
