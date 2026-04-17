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

//REMARK: FOr now maybe not yet, just take the value from finer grid ... !!!!!
//void GetPreqInterpolation(Jetscape::real x, Jetscape::real y) {
  //int id_x = GetIdX(x);
  //int id_y = GetIdY(y);

  /*
  auto c000 = GetFluidCell(id_tau, id_x, id_y, id_eta);
  auto c001 = GetFluidCell(id_tau, id_x, id_y, id_eta + 1);
  auto c010 = GetFluidCell(id_tau, id_x, id_y + 1, id_eta);
  auto c011 = GetFluidCell(id_tau, id_x, id_y + 1, id_eta + 1);
  auto c100 = GetFluidCell(id_tau, id_x + 1, id_y, id_eta);
  auto c101 = GetFluidCell(id_tau, id_x + 1, id_y, id_eta + 1);
  auto c110 = GetFluidCell(id_tau, id_x + 1, id_y + 1, id_eta);
  auto c111 = GetFluidCell(id_tau, id_x + 1, id_y + 1, id_eta + 1);
  real x0 = XCoord(id_x);
  real x1 = XCoord(id_x + 1);
  real y0 = YCoord(id_y);
  real y1 = YCoord(id_y + 1);
  */

  //return TrilinearInt(x0, x1, y0, y1, eta0, eta1, c000, c001, c010, c011, c100,
  //                    c101, c110, c111, x, y, eta);
//}

const double mNc = 3;
const double mNf = 3;

double get_temperature_ideal_EOS(double eps, double rhob = 0) {
    return pow(
        90.0 / M_PI / M_PI * (eps / 3.0)
            / (2 * (mNc * mNc - 1) + 7. / 2 * mNc * mNf),
        .25);
}

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

  /// ------------------------------------------------------------------
  // Dummy test here ...
  //
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    //c10::Device device(c10::DeviceType::CPU);
    //REMARK: Libtorch has its on OMP library so currently we cannot use it with MUSCIC at the same time !!!
    //setenv OMP_NUM_THREADS 1 for testing, single core ...

    // ************************************************************************************
    // Try improve speed : https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    // Use oneDNN Graph with TorchScript for inference ...
    // --> No real improvement ==> Follow up on this !???
    // ************************************************************************************
    //
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

  //JSINFO << "hydro initial time set by PreEq module tau0 = " << tau0 << " fm/c";
  //JSINFO << "initial density profile dx = " << dx_preq << " fm";

  SetPreEqGridInfo();

  cout<<dx_preq<<" "<<dy_preq<<" "<<z_max<<" "<<nz<<" "<<endl;
  cout<<x_min_preq<<" "<<y_min_preq<<" "<<endl;
  cout<<nx_preq<<" "<<ny_preq<<" "<<endl; //nz_preq<<" "<<endl;

  //JSINFO << "number of fluid cells received by the JETSCAPE: "
  //          << bulk_info.data.size();
  /*
  has_source_terms = false;
  if (hydro_source_terms_ptr->get_number_of_sources() > 0) {
    has_source_terms = true;
  }
  JSINFO << "number of source terms: "
         << hydro_source_terms_ptr->get_number_of_sources()
         << ", total E = " << hydro_source_terms_ptr->get_total_E_of_sources()
         << " GeV.";
   */

   // *************************************************************************
   // REMARK: How to get form the pre-eq the 4 features e-density, temp, ux, uy
   // see music.cpp and init.cpp .... but maybe a short cut!???
   // *************************************************************************
   //
   // https://github.com/JETSCAPE/JETSCAPE/pull/254/files
   // Rotation etc fix ... make sure not to repeat here !!!
   // See also Dave email: wrt to training data:  [E, T, vy, vx] as opposed to [E, T, vx, vy] try to confirn ASAP!!!!
   // *************************************************************************

  //DEBUG
  //for(int i=0;i<10000;i++) cout<<pre_eq_ptr->e_[i]<<" "; // this is the energy density according to Chun ...
  //cout<<endl;

  // *************************************************************************
  // DEBUG QA ...

  // TH2D *h2dIS = new TH2D("h2dIS", "", 150, 0, 150, 150, 0, 150);
  // //TH2D *h2dIS_rebin = new TH2D("h2dIS_rebin", "", 60, -15, 15, 60, -15, 15);
  // TH2D *h2dIS_rebin = new TH2D("h2dIS_rebin", "", 60, 0, 60, 60, 0, 60);
  // TH2D *h2dIS_rebin_torch = new TH2D("h2dIS_rebin_torch", "", 60, 0, 60, 60, 0, 60);
  // TH2D *h2dIS_rebin_torch_pred = new TH2D("h2dIS_rebin_torch_pred", "", 60, 0, 60, 60, 0, 60);
  // TH2D *h2dIS_rebin_torch_pred_bulkhist = new TH2D("h2dIS_rebin_torch_pred_bulkhist", "", 60, 0, 60, 60, 0, 60);

  // TH2D *h2dIS_T = new TH2D("h2dIS_T", "", 150, 0, 150, 150, 0, 150);
  // TH2D *h2dIS_2 = new TH2D("h2dIS_2", "", 150, -15, 15, 150, 15, 15);
  //
  //cout<<h2dIS_T->GetBin(75,75)<<endl;
  //h2dIS->SetBinContent(75, 75, 12.);

  // *********************************************************************************************
  //REMARK: Issue with ideal EOS and temperature values > 2x higher then via bulk root writer !!!!
  // *********************************************************************************************

  // for (int i=0;i<150;i++)
  //   for (int j=0;j<150;j++) {
  //     double ed = pre_eq_ptr->e_[GetPreqCellIndex(i,j)];
  //     double T = get_temperature_ideal_EOS(ed);

  //     h2dIS->SetBinContent(i,j,ed);
  //     h2dIS_T->SetBinContent(i,j,T);
  //   }

  // *******************************************************************
  // Grid for FNO (explore superresolution at a later stage ...)
  // Alternative, define Trento grid accordingingly!!!
  // Certainly not ideal ...
  // *******************************************************************

  //double m_dX = dx_fno;
  //double m_xMin = x_min_fno;

  //auto tensor fno_input_tensor = torch.zeros({1, 4, 60, 60, 50});
  //torch::Tensor fno_input_tensor = torch::zeros({4, 60, 60, 50});
  //torch::Tensor fno_input_tensor = torch::zeros({4, 60, 60, 1});

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
        double T = GetTemperatureFromEos((ed);

        //h2dIS_rebin->Fill(i,j,ed);

        //for (int k=0;k<50;k++) {
        int k=0;
        if (n_features == 4 ) {
            fno_input_tensor[0][i][j][k] = ed;
            fno_input_tensor[1][i][j][k] = T;
            // only for null preq module ... extend here at some point ... when a real dynamic evolution is used and how to get the first time-step ....
            fno_input_tensor[2][i][j][k] = 0;
            fno_input_tensor[3][i][j][k] = 0;
        }
        else if (n_features == 3) {
            fno_input_tensor[0][i][j][k] = ed;
            fno_input_tensor[1][i][j][k] = 0;
            fno_input_tensor[2][i][j][k] = 0;
        }
        else {JSWARN<<" Not enough FNO features edensity, vx,vy ... to be used further in JETSCAPE !"; exit(-1);}

        //}
    }

  fno_input_tensor = fno_input_tensor.repeat({1, 1, 1, ntau_fno});

 //DEBUG ..
  // for (int i=0;i<60;i++)
  //   for (int j=0;j<60;j++)
  //   {
  //       h2dIS_rebin_torch->Fill(i,j,fno_input_tensor[0][i][j][49].item<double>());
  //   }

  // TCanvas *c1 = new TCanvas("c1", "Canvas", 800, 600);
  // //h2dIS_T->SetOptStat(0);
  // //h2dIS_rebin->Draw("colz");
  // h2dIS_rebin_torch->Draw("colz");
  // c1->SaveAs("h2dIS_rebin_torch_id.gif");

  // *************************************************************************

  clear_up_evolution_data();
  PassPreEqEvolutionHistoryToFramework();

  hydro_status = INITIALIZED;

  if (hydro_status == INITIALIZED) {

    JSINFO << "Running FNO model hydro time evolution prediction ...";

    // FIll bulk grid info from/defined by FNO (as said super resolution later)!!!
    // TBD ...

    //cout<<M_PI<<endl;

    //********************************************************
    //WARNING: definitely a memeory leak here ... !!!!????
    //********************************************************

    /// ------------------------------------------------------------------
    // Dummy test here ...
    // Create a vector of inputs.

    // *************************************************************************
    // REMARK: How to duplicate the first entry, like in nump!????
    // *************************************************************************

    // tensor.unsqueeze(0): Adds a new dimension at the beginning (position 0), changing the shape from [4] to [1, 4].
    // // Repeat the tensor twice along dimension 1 and once along dimension 0
    // auto repeated_tensor_2 = tensor.repeat({1, 2});
    // std::cout << "Repeated tensor (1x2):\n" << repeated_tensor_2 << std::endl;
    // x_initial to a PyTorch tensor.
    // x_initial.repeat(1, 1, 1, self.time_steps - 1): Repeats the tensor along the time dimension (axis 3 in NumPy corresponds to the last dimension in PyTorch).

    //torch::Tensor fno_input_tensor_unsqueeze = fno_input_tensor.unsqueeze(0);
    //torch::Tensor fno_input_tensor_unsqueeze_duplicate = fno_input_tensor_unsqueeze.repeat({1, 49, 49, 49});
    std::vector<torch::jit::IValue> inputs;
    //inputs.push_back(fno_input_tensor.unsqueeze(0).to(device)); //fno_input_tensor_unsqueeze); // .to(at::kMPS));
    inputs.push_back(fno_input_tensor.unsqueeze(0));
    // Execute the model and turn its output into a tensor.
    output = module.forward(inputs).toTensor();
    //***********************************************************************************
    //REMARK: Check if ultimately the 0 time step is missing, if so attach from Preqq ...
    //***********************************************************************************

    //torch::Tensor output = module.forward(fno_input_tensor_unsqueeze);
    //torch::Tensor output = module.forward(fno_input_tensor.unsqueeze(0)).toTensor(); //Why not direclty a tensor!???

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

    for(auto t : inputs)
      t.toTensor().reset();
    inputs.clear();
    //cout<<inputs.size()<<endl;

    /// ------------------------------------------------------------------

    // for (int i=0;i<60;i++)
    //   for (int j=0;j<60;j++)
    //   {
    //       h2dIS_rebin_torch_pred->Fill(i,j,output[0][0][i][j][40].item<double>());
    //   }

    // TCanvas *c2 = new TCanvas("c2", "Canvas", 800, 600);
    // //h2dIS_T->SetOptStat(0);
    // //h2dIS_rebin->Draw("colz");
    // h2dIS_rebin_torch_pred->Draw("colz");
    // c2->SaveAs("h2dIS_rebin_torch_id_pred.gif");

    hydro_status = FINISHED;
  }

  auto end = std::chrono::high_resolution_clock::now();
  // Calculate duration
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

  // Maybe move to Init ... !??? Think about ...
  SetHydroGridInfo();

  //Grifd info for bulk history ...
  cout<<bulk_info.Tau0()<<" "<<bulk_info.TauMax()<<" "<<bulk_info.ntau<<" "<<bulk_info.dtau<<endl;
  cout<<bulk_info.XMin()<<" "<<bulk_info.XMax()<<" "<<bulk_info.nx<<" "<<bulk_info.dx<<endl;
  cout<<bulk_info.YMin()<<" "<<bulk_info.YMax()<<" "<<bulk_info.ny<<" "<<bulk_info.dy<<endl;

  PassHydroEvolutionHistoryToFramework();

  JSINFO << "number of fluid cells received by the JETSCAPE: "
             << bulk_info.data.size();

  // DEBUG QA ...
  // TH2D *h2dIS_rebin_torch_pred_bulkhist = new TH2D("h2dIS_rebin_torch_pred_bulkhist", "", 60, 0, 60, 60, 0, 60);

  // for (int i=0;i<nx_fno;i++)
  //   for (int j=0;j<ny_fno;j++)
  //   {
  //       h2dIS_rebin_torch_pred_bulkhist->Fill(i,j,bulk_info.data[bulk_info.CellIndex(40,i,j,0)].energy_density);
  //   }

  // TCanvas *c3 = new TCanvas("c3", "Canvas", 800, 600);
  // h2dIS_rebin_torch_pred_bulkhist->Draw("colz");
  // c3->SaveAs("h2dIS_rebin_torch_pred_bulkhist_fromTensor.gif");

  output.reset();

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

// *****************************************************************************************
// Remark: Make sure that the tensor operation flatten is doing the same as the global index
// of the bulk history --> check with filling via loops !!! and ouput histogram !!!
// ==> looks like this falattening is not the same as the global index !!!
// Check with moving time axis to front and then flatten !!!
// *****************************************************************************************

void FnoHydro::PassHydroEvolutionHistoryToFramework() {
  JSINFO << "Passing hydro evolution information to JETSCAPE ... ";

  auto start = std::chrono::high_resolution_clock::now();

  int number_of_cells = output.numel()/4.; //music_hydro_ptr->get_number_of_fluid_cells();
  JSINFO << "total number of FNO prediction hydro fluid cells: " << number_of_cells;

  //REMRAK: This works and results in correct bulk history filling ... !!!!
  /*
  output = torch::squeeze(output, 0);

  for (int k=0;k<bulk_info.ntau;k++)
    for (int i=0;i<bulk_info.nx;i++)
        for (int j=0;j<bulk_info.ny;j++)
        {
            std::unique_ptr<FluidCellInfo> fluid_cell_info_ptr(new FluidCellInfo);
            //music_hydro_ptr->get_fluid_cell_with_index(i, fluidCell_ptr);

            //cout<<i<<" "<<j<<" "<<k<<endl;
            //cout<<output[0][i][j][k].item<double>()<<endl;

            fluid_cell_info_ptr->energy_density = output[0][i][j][k].item<double>();
            fluid_cell_info_ptr->entropy_density = 0.0; //fluidCell_ptr->sd;
            fluid_cell_info_ptr->temperature = output[1][i][j][k].item<double>();
            fluid_cell_info_ptr->pressure = 0.0; //fluidCell_ptr->pressure;
            fluid_cell_info_ptr->vx = output[2][i][j][k].item<double>();
            fluid_cell_info_ptr->vy = output[3][i][j][k].item<double>();
            fluid_cell_info_ptr->vz = 0.0 ;//fluidCell_ptr->vz;
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
   */

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

double FnoHydro::GetTemperatureFromEos(double ed) {
    return fnoEOS->get_temperature(ed/Util::hbarc, 0)*Util::hbarc;
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
