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
#include "SurfaceCellInfo.h"
#include "FnoRooIn.h"
#include "util.h"

#include <Riostream.h>
#include "TRandom.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TMath.h"

using namespace Jetscape;

// Register the module with the base class
RegisterJetScapeModule<FnoRooIn> FnoRooIn::reg("FnoRooIn");

//****************************************************************************************
//DEBUG!!!!
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
  bulkHadroFull = false;
  QAoutput = false;
  doReuseHydro = false;
  nReuseHydro = 0;
  //device = torch::Device({});
  //has_source_terms = false;
  SetId("FnoRooIn");
  //hydro_source_terms_ptr =
  //    std::shared_ptr<HydroSourceJETSCAPE>(new HydroSourceJETSCAPE());
}

FnoRooIn::~FnoRooIn() {}

void FnoRooIn::FinishTask()
{
    if (fOut) {
    fOut->Write();
    fOut->Close();
    }
}

void FnoRooIn::InitializeHydro(Parameter parameter_list) {
  JSINFO << "Initialize FnoRooIn ...";
  VERBOSE(8);

  string input_root_file = GetXMLElementText({"Hydro", "FNOROOIN", "root_file"});

  JSINFO<<"Loading ROOT as input/full hydro file : "<<input_root_file.c_str();

  f=new TFile(input_root_file.c_str(),"READ");
  t=(TTree*)f->Get("t");

  //Make more general/debug for now ...
  //fOut=new TFile("fno_predictions.root","RECREATE");

  m_xyt = nullptr;
  m_foSurf = nullptr;
  m_foEdT = nullptr;

  t->SetBranchAddress("user_res",&m_xyt);
  t->SetBranchAddress("foSurf",&m_foSurf);
  t->SetBranchAddress("foEdT",&m_foEdT);

  fullHydroIn = GetXMLElementInt({"Hydro", "FNOROOIN", "fullHydroIn"});
  bulkHadroFull = GetXMLElementInt({"Hydro", "FNOROOIN", "bulkHadroFull"});
  QAoutput = GetXMLElementInt({"Hydro", "FNOROOIN", "QAoutput"});

  if (fullHydroIn) {
    JSINFO << "Full hydro input is used ...";
    JSINFO << "Bulk Hadronization full on/off = "<<bulkHadroFull;
  } else {
    JSINFO << "Bulk Hadronization full on/off = "<<bulkHadroFull;
    JSINFO << "FNO prediction beyond first time step is used ...";
    JSINFO << BOLDCYAN << "IMPORTANT: edensity*Tau normalization hardcoded for now!!!";

    // For testing now, overwrites the env settings ...
    torch::set_num_threads(1);
    JSINFO << "Number of threads (libtorch OMP): " << torch::get_num_threads();
    //device = torch::Device(torch::kMPS);
    JSINFO << "Default device: " << device << " (for now only CPU supported!)";// << std::endl;

    try {
    // ************************************************************************************
    // Try improve speed : https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    // Use oneDNN Graph with TorchScript for inference ...
    // --> No real improvement ==> Follow up on this !???
    // ************************************************************************************
    //
    string input_model_file = GetXMLElementText({"Hydro", "FNOROOIN", "model_file"});
    JSINFO<<"Loading the traced Pytorch model : "<<input_model_file.c_str();
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
  // ***********************************************************************************************************
  // REMARK: Super-resolution requiures serilazed FNO trained model with the proper tracing tensor dimensions!!!
  // ***********************************************************************************************************

  tau0 = GetXMLElementDouble({"Hydro", "FNOROOIN", "tau0"});
  JSINFO << "Use preEq evo: tau_0 (but from XML) = " << tau0;

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

  freezeout_temperature = GetXMLElementDouble({"Hydro", "FNOROOIN", "freezeout_temperature"});
  JSINFO << "Freezeout temperature = " << freezeout_temperature << " GeV";

  int EOS_id_MUSIC = GetXMLElementInt({"Hydro", "FNOROOIN", "EOS_id_MUSIC"});
  JSINFO<<"Use EOS (Music id) = "<<EOS_id_MUSIC;
  fnoEOS=make_unique<EOS>(EOS_id_MUSIC);

  int nRoot = t->GetEntries();
  int nEv = GetXMLElementInt({"nEvents"});

  std::string reuseHydro = GetXMLElementText({"setReuseHydro"});
  if ((int)reuseHydro.find("true") >= 0) doReuseHydro = true;
  if (doReuseHydro) {
     nReuseHydro = GetXMLElementInt({"nReuseHydro"});

     JSINFO<< "Asked for total number of events = "<<nEv;
     JSINFO << "with do re-use hydro event with nReuseHydro = "<<nReuseHydro;
     JSINFO << " --> number of effective events needed = "<<(int) (nEv/(double) nReuseHydro);
     JSINFO << " --> adjust and check if compatible with number of events in ROOT file = "<<nRoot;

     int nEvEff = (double) nEv / (double) nReuseHydro;
     if (nEvEff>nRoot) {
         JSWARN<<"Not enough events in ROOT file, reduce nReuseHydro!!!";
         exit(-1);
        }

     JSINFO << " --> Done. Fine. Moving on ...";

  }
  else
  {
      if (nEv>nRoot) {
           JSWARN<<"Not enough events in ROOT file = "<<nRoot<<" but asked for total number of events = "<<nEv<<" ... reduce number of events!!!";
           exit(-1);
      }
  }

}

void FnoRooIn::EvolveHydro() {
  VERBOSE(8);
  JSINFO << "Initialize density profiles in FnoHydro ...";

  bulk_info.tau_min = tau0; //pre_eq_ptr->GetPreequilibriumStartTime();
  JSINFO << "Use preEq evo: tau_0 (from XML) = " << bulk_info.tau_min<< " fm/c.";

  // ---------------------------------------------

  clear_up_evolution_data();

  // Quick fix to get the test data part of the 10k training + test sample ... TBD ...
  //t->GetEntry(9999-GetCurrentEvent());
  if (!doReuseHydro) {
    t->GetEntry(GetCurrentEvent());
  }
  else {
     int getNumberOfRootEvent =  GetCurrentEvent()/(double) nReuseHydro;
     //DEBUG: cout<<GetCurrentEvent()<<" "<<nReuseHydro<<" "<<getNumberOfRootEvent<<endl;
     t->GetEntry(getNumberOfRootEvent);
  }

  // ---------------------------------------------

  SetHydroGridInfo();

  bool useEvent = true;

  if (bulkHadroFull)
    useEvent = CheckEventForFullHadro();

  // ---------------------------------------------

  if (useEvent)
  {
    //auto start = std::chrono::high_resolution_clock::now();
    //DEBUG:
    //cout<<bulk_info.ntau<<" "<<(*m_xyt)[0][0].size()<<endl;

    // ---------------------------------------------

    if (fullHydroIn)
    {
        // ---------------------------------------------------------
        //REMARK: Maybe do this directly in the SetHydroGridInfo!!!!
        // ---------------------------------------------------------
        bulk_info.ntau = bulk_info.ntau+1;

        PassHydroEvolutionHistoryToFrameworkFromRoot();
    }
    else
    {
        GetFnoPrediction();
        PassHydroEvolutionHistoryToFramework();
    }

    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

    JSINFO << "number of fluid cells received by the JETSCAPE: "
                << bulk_info.data.size();

    if (QAoutput) {
        //Save3dHist();
        Print2dHist();
    }

    // ---------------------------------------------

    if (bulkHadroFull) {
    clearSurfaceCellVector();
    FindAConstantTemperatureSurface(freezeout_temperature, surfaceCellVector_);
    //PassHydroSurfaceToFrameworkFromRoot();
    }

    // ---------------------------------------------
  }

 // ---------------------------------------------

  output.reset();
  m_xyt->clear();

  m_foSurf->clear();
  m_foEdT->clear();

  // ---------------------------------------------

  //REMARK: Maybe not needed at all ... TBD ...
  //SetElossSeedsToCurrentEventNumber();

}

// ---------------------------------------------

bool FnoRooIn::CheckEventForFullHadro()
{
    bool useEvent = true;

    auto softHadro = JetScapeSignalManager::Instance()->GetSoftParticlizationPointer();
    if (!softHadro.lock()) {JSWARN<<"Asked for bulk hadronization, but no SoftParticlization module found!"; exit(-1);}

    softHadro.lock()->SetActive(true);

    int m_root_ntau =  (*m_xyt)[0][0].size() ;
    //DEBUG:
   // cout<<m_root_ntau<<endl;
    //cout<<m_foSurf->size()<<endl;
    //
    if (m_root_ntau > bulk_info.ntau)
    {
        useEvent = false;
        JSINFO<<CYAN<<"Skip this event for bulk hadronization since FNO prediction timestep < the Hydro from file length:  "<<bulk_info.ntau<< " < "<<m_root_ntau;
        softHadro.lock()->SetActive(false);
    }

    return useEvent;
}

// ---------------------------------------------
//
void FnoRooIn:: GetFnoPrediction()
{
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

    for (int i=0;i<nx_fno;i++)
        for (int j=0;j<ny_fno;j++) {
            for (int k=0;k<ntau_fno_loop;k++)
            {
                if (n_features == 4 ) {
                    fno_input_tensor[0][i][j][k] = (*m_xyt)[i][j][k][0];
                    fno_input_tensor[1][i][j][k] = (*m_xyt)[i][j][k][1];
                    // only for null preq module trained so far so vx=vy=0!
                    fno_input_tensor[2][i][j][k] = (*m_xyt)[i][j][k][2];
                    fno_input_tensor[3][i][j][k] = (*m_xyt)[i][j][k][3];
                }
                else if (n_features == 3) {
                    //=============================================================
                    //IMPORTANT: ednsity * Tau normalization hardcoded for now!!!!
                    //=============================================================
                    fno_input_tensor[0][i][j][k] = (*m_xyt)[i][j][k][0]*bulk_info.Tau0();
                    // only for null preq module trained so far so vx=vy=0!
                    fno_input_tensor[1][i][j][k] = (*m_xyt)[i][j][k][1];
                    fno_input_tensor[2][i][j][k] = (*m_xyt)[i][j][k][2];
                }
                else {JSWARN<<" Not enough FNO features edensity, vx,vy ... to be used further in JETSCAPE !"; exit(-1);}
            }
        }

    fno_input_tensor = fno_input_tensor.repeat({1, 1, 1, ntau_fno});

    tensorShapeInput = "Input Tensor shape for FNO model with " + std::to_string(ntau_fno) + " copies of IS : ";
    shape = fno_input_tensor.sizes();
        for (int i = 0; i < shape.size(); ++i) {
        tensorShapeInput += std::to_string(shape[i]) ; tensorShapeInput += " ";
        }
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
        for (int i = 0; i < shape.size(); ++i) {
            tensorShapeInput += std::to_string(shape[i]) ; tensorShapeInput += " ";
        }
        JSINFO << tensorShapeInput.c_str();

        // ---------------------------------------------------------
        //REMARK: Maybe do this directly in the SetHydroGridInfo!!!!
        // ---------------------------------------------------------
        bulk_info.ntau = bulk_info.ntau+1;

        for(auto t : inputs)
            t.toTensor().reset();
        inputs.clear();

        //save_tensor_legacy_pickle(output, "fno_output_tensor.pt");

        fno_input_tensor.reset(); //maybe redundant ...

        hydro_status = FINISHED;
    }
}

// ---------------------------------------------
// Fix seed for JetEnergyLoss to allow event by event FNO assessment...
// Seems to be working with PGun !!! (Check with Pythia if needed ...)!!! and Matter, follow up with Lbt too!!!!
void FnoRooIn::SetElossSeedsToCurrentEventNumber()
{
    auto jLossManager = JetScapeSignalManager::Instance()->GetJetEnergyLossManagerPointer();
    if (jLossManager.lock())
    {
        for (auto it : jLossManager.lock()->GetTaskList())
        {
            //JSINFO << it->GetId();
            for (auto it2 : it->GetTaskList())
            {
                //JSINFO  << " " << it2->GetId() << " seed to -> "<< GetCurrentEvent()+1;
                auto mRan =  std::dynamic_pointer_cast<JetEnergyLoss>(it2)->GetMt19937Generator();
                mRan->seed(GetCurrentEvent()+1);
            }
        }
    }

}

// --------------------------------------------------------
// --------------------------------------------------------
//DEBUG only for now ... needs more work to generalize ...
// only if bulkHadroFull = True! Fix later !!!
//
void FnoRooIn::Fill3dHist(TH3F* h3d, TH3F* h3dOrg)
{
    for (int k=0; k<(*m_xyt)[0][0].size();k++) {
        for (int i=0; i<(bulk_info.nx); i++){
            for (int j=-0; j<(bulk_info.ny); j++){

            int nIndex = bulk_info.CellIndex(k,i,j,0); // last eta here index 0 since boost invariant ...
            auto mCell = bulk_info.data.at(nIndex);

            h3d->SetBinContent(i,j,k,(float) (mCell.energy_density));
            h3dOrg->SetBinContent(i,j,k,(float) (*m_xyt)[i][j][k][0]);
            }
        }
    }
}

void FnoRooIn::Save3dHist()
{
    TH3F *hFno=new TH3F("hFno","hFno",60,0,60,60,0,60,50,0,50);
    TH3F *hHydro=new TH3F("hHydro","hHydro",60,0,60,60,0,60,50,0,50);

    Fill3dHist(hFno, hHydro);

    string currentEv = std::to_string(GetCurrentEvent());
    string sFno = "hFno_"; sFno += currentEv;
    string sHyd = "hHydro_"; sHyd += currentEv;

    fOut->cd();
    hFno->Write(sFno.c_str());
    hHydro->Write(sHyd.c_str());

    delete hFno; delete hHydro;
}

void FnoRooIn::Fill2dHist(TH2F* h2d, TH2F* h2dOrg, int ntau)
{
    for (int i=0; i<(bulk_info.nx); i++){
        for (int j=-0; j<(bulk_info.ny); j++){

        int nIndex = bulk_info.CellIndex(ntau,i,j,0); // last eta here index 0 since boost invariant ...
        auto mCell = bulk_info.data.at(nIndex);

        h2d->SetBinContent(i,j,(float) (mCell.energy_density));
        h2dOrg->SetBinContent(i,j,(float) (*m_xyt)[i][j][ntau][0]);

        }
    }
}

void FnoRooIn::Print2dHist()
{
    TH2F *hIn=new TH2F("hIn","hIn",60,0,60,60,0,60);
    TH2F *hMax=new TH2F("hMax","hMax",60,0,60,60,0,60);
    TH2F *hInOrg=new TH2F("hInOrg","hIn Hydro",60,0,60,60,0,60);
    TH2F *hMaxOrg=new TH2F("hMaxOrg","hMax Hydro",60,0,60,60,0,60);

    Fill2dHist(hIn,hInOrg,10);
    int maxNtau = bulk_info.ntau-1;
    if (bulkHadroFull)
        maxNtau = (*m_xyt)[0][0].size()-1;


    //DEBUG:
    cout<<bulk_info.ntau<<" "<<maxNtau<<" "<<(*m_xyt)[0][0].size()<<endl;

    Fill2dHist(hMax,hMaxOrg,maxNtau);

    TCanvas *c1 = new TCanvas("c1", "Canvas", 1000, 800);
    c1->Divide(3,2);
    c1->cd(1);
    hIn->SetStats(0);
    hIn->DrawCopy("colz");
    c1->cd(2);
    hInOrg->SetStats(0);
    hInOrg->Draw("colz");
    c1->cd(3);
    gPad->SetLeftMargin(0.0);
    gPad->SetRightMargin(0.25);
    hIn->Add(hInOrg,-1);
    hIn->SetTitle("FNO-Hydro nTau=10");
    hIn->Draw("colz");
    c1->cd(4);
    hMax->SetStats(0);
    hMax->DrawCopy("colz");
    c1->cd(5);
    hMaxOrg->SetStats(0);
    hMaxOrg->Draw("colz");
    c1->cd(6);
    gPad->SetLeftMargin(0.0);
    gPad->SetRightMargin(0.25);
    hMax->SetTitle("FNO-Hydro");
    hMax->Add(hMaxOrg,-1);
    hMax->Draw("colz");

    string currentEv = std::to_string(GetCurrentEvent());
    string gifOut = "h2d_ednsity_"; gifOut += currentEv; gifOut += ".gif";
    c1->SaveAs(gifOut.c_str());

    delete hInOrg; delete hMaxOrg;
    delete hIn; delete hMax; delete c1;


}

// --------------------------------------------------------
// --------------------------------------------------------

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

void FnoRooIn::PassHydroEvolutionHistoryToFrameworkFromRoot()
{
    JSINFO << "Passing hydro evolution information to JETSCAPE from ROOT file : bulkInfo nTau = "<<bulk_info.ntau<<" and ROOT file nTau(Max) = "<<(*m_xyt)[0][0].size();

    //===========================================================================
    // REMARK: +1 or not depending on max ntau or ntau from FNO < max ... fix!!!
    //===========================================================================

    int ntau_root = (*m_xyt)[0][0].size();
    int m_ntau = bulk_info.ntau;
    if (bulk_info.ntau>=ntau_root) {
        m_ntau = ntau_root;
        bulk_info.ntau = m_ntau;
    }

    //DEBUG:
    //cout<<bulk_info.ntau<<" "<<m_ntau<<" "<<(*m_xyt)[0][0].size()<<endl;

    //REMARK: Shortcut for using the full stored hydro information ...
    //        (write seperate module for just reading in ROOT file to decouple FNO validation)
    //bulk_info.ntau = (*m_xyt)[0][0].size();
    //m_ntau = bulk_info.ntau;

    for (int k=0;k<m_ntau;k++)
      for (int i=0;i<bulk_info.nx;i++)
          for (int j=0;j<bulk_info.ny;j++)
          {
              std::unique_ptr<FluidCellInfo> fluid_cell_info_ptr(new FluidCellInfo);

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
}

float FnoRooIn::GetTemperatureFromEos(float ed) {
    return fnoEOS->get_temperature((float) ed/Util::hbarc, 0)*Util::hbarc;
}

void FnoRooIn::PassHydroSurfaceToFrameworkFromRoot() {
    JSINFO << "Passing hydro surface cells to JETSCAPE from ROOT file ... ";
    auto number_of_cells = m_foSurf->size();
    JSINFO << "total number of fluid cells: " << number_of_cells;

    for (int i = 0; i < number_of_cells; i++) {
        SurfaceCellInfo surface_cell_info;

        surface_cell_info.tau = (*m_foSurf)[i][0];
        surface_cell_info.x =  (*m_foSurf)[i][1];
        surface_cell_info.y =  (*m_foSurf)[i][2];
        surface_cell_info.eta = 0;
        surface_cell_info.d3sigma_mu[0] = (*m_foSurf)[i][3];
        surface_cell_info.d3sigma_mu[1] = (*m_foSurf)[i][4];
        surface_cell_info.d3sigma_mu[2] = (*m_foSurf)[i][5];
        surface_cell_info.d3sigma_mu[3] = 0;
        surface_cell_info.umu[0] =  (*m_foSurf)[i][6];
        surface_cell_info.umu[1] =  (*m_foSurf)[i][7];
        surface_cell_info.umu[2] =  (*m_foSurf)[i][8];
        surface_cell_info.umu[3] =  0;

        surface_cell_info.energy_density = (*m_foEdT)[0];
        surface_cell_info.temperature = (*m_foEdT)[1];

        surface_cell_info.pressure = 0;
        surface_cell_info.baryon_density = 0;
        surface_cell_info.mu_B = 0;
        surface_cell_info.mu_Q =0;
        surface_cell_info.mu_S = 0;
        for (int j = 0; j < 10; j++) {
            surface_cell_info.pi[j] = 0;
        }
        surface_cell_info.bulk_Pi = 0;
        StoreSurfaceCell(surface_cell_info);
    }
}

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
