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
// ------------------------------------------------------------
// JetScape Framework Brick Test Program with Pythia IS
// (use either shared library (need to add paths; see setup.csh)
// (or create static library and link in)
// -------------------------------------------------------------

//********************************************************************************************************************
//REMARK:  cmake .. -DUSE_ROOT=ON -DUSE_MUSIC=ON -DUSE_ISS=ON
//Download in external:  get_iSS.sh get_music.sh (Iss just for creating full hadrons for later use too ...)
//********************************************************************************************************************

#include <iostream>
#include <time.h>
#include <string>

// JetScape Framework includes ...
#include "JetScape.h"
#include "JetEnergyLoss.h"
#include "JetEnergyLossManager.h"
#include "JetScapeWriterStream.h"
#include "JetScapeSignalManager.h"
#ifdef USE_HEPMC
#include "JetScapeWriterHepMC.h"
//#include "JetScapeWriterRootHepMC.h"
#endif


// User modules derived from jetscape framework clasess
#include "TrentoInitial.h"
#include "AdSCFT.h"
#include "Matter.h"
#include "LBT.h"
#include "Martini.h"
#include "Brick.h"
#include "GubserHydro.h"
#include "MusicWrapper.h"
#include "PythiaGun.h"
#include "iSpectraSamplerWrapper.h"
#include "TrentoInitial.h"
#include "NullPreDynamics.h"
#include "PGun.h"
#include "HadronizationManager.h"
#include "Hadronization.h"
#include "ColoredHadronization.h"
#include "ColorlessHadronization.h"
//#include "HydroFromFile.h"

#include <chrono>
#include <thread>

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

using namespace std;
using namespace Jetscape;

// Forward declaration
void Show();

class DemoBulk : public JetScapeModuleBase
{
  public:

  DemoBulk() : JetScapeModuleBase() {SetId("DemoBulk"); oName = ""; initBranch=false;}
  DemoBulk(string m_oName) : JetScapeModuleBase() {SetId("DemoBulk"); oName = m_oName; initBranch=false;}
  virtual ~DemoBulk() {f->cd();f->ls(); t->Print(); f->Write();f->Close();}

  void Init() {

      f=new TFile(oName.c_str(),"RECREATE");
      //hPt = new TH1D("hPt","",100,0,100);
      t=new TTree("t","Tree");
  }

  void InitBranch(int nX=20, int nY=20, int nT=10)
  {
      cout<<"InitBranch : "<<nX<<" "<<nY<<" "<<nT<<endl;
      initBranch = true;
      m_xyt=std::vector<std::vector<std::vector<double>>>(nX, std::vector<std::vector<double>>(nY, std::vector<double>(nT)));

      t->Branch("sim",&m_xyt);
      t->Branch("sim2",&m_xyt2);
  }
  void Exec() {

    // quick and dirty test to get hydro bulk info
    auto hydro = JetScapeSignalManager::Instance()->GetHydroPointer();
    //if (hydro!=0)
    auto bInfo = hydro.lock()->get_bulk_info();
    //if (bInfo!=0)

    // ***************************************
    // REMARK: total number of cells is changing from event to event!!!! MAnily due to different tau_max etc ...
    // Either try to fix this here or in principle akward and uproot can handle this, but requiures some padding !???
    // Definitely something to think about ... !!!!!
    // ***************************************

    cout<<bInfo.get_data_size()<<" "<<bInfo.is_boost_invariant()<<endl;
    cout<<"Grid Info : "<<bInfo.ntau<<" "<<bInfo.nx<<" "<<bInfo.ny<<" "<<bInfo.neta<<endl;
    /*
    auto dataInfo = bInfo.data_info;
    cout<<dataInfo.size()<<endl;
    for (auto s : dataInfo) cout<<s<<" ";
    cout<<endl;
    cout<<bInfo.data_vector.size()<<endl;
    */
    cout<<bInfo.tau_min<<" "<<bInfo.Tau0()<<" "<<bInfo.XMin()<<" "<<bInfo.YMin()<<" "<<bInfo.EtaMin()<<endl;
    cout<<bInfo.TauMax()<<" "<<bInfo.XMax()<<" "<<bInfo.YMax()<<" "<<bInfo.EtaMax()<<endl;
    cout<<bInfo.dtau<<" "<<bInfo.dx<<" "<<bInfo.dy<<" "<<bInfo.deta<<endl;
    cout<<bInfo.GetIdTau(0.6)<<" "<<bInfo.GetIdTau(10.)<<" "<<bInfo.GetIdTau(100.)<<endl;
    auto fCell = bInfo.GetAtTimeStep(bInfo.GetIdTau(0.6),0,0,0);
    cout<<fCell.temperature<<endl;
    fCell = bInfo.GetAtTimeStep(bInfo.GetIdTau(5.),0,0,0);
    cout<<fCell.temperature<<endl;

    // ****************************************
    //FluidCellInfo get(Jetscape::real tau, Jetscape::real x, Jetscape::real y, Jetscape::real etas) const;
    //--> could be used for arbirtrary storage for FNO ...
    // Nonethless, try to get the full hydro gridd saved in vector of fCells to 3d etc format too !!!!
    // ****************************************
    /*
    for (int i=0;i<bInfo.ntau;i++)
    {
        int nIndex = bInfo.CellIndex(i,75,75,0);
        auto mCell = bInfo.data.at(nIndex);
        //cout<<mCell<<endl;
        //cout<<nIndex<<" "<<bInfo.XCoord(0)<<" "<<mCell.temperature<<endl;
        //cout<<bInfo.GetFluidCell(31,0,0,0).temperature<<endl;
        cout<<nIndex<<" "<<bInfo.TauCoord(i)<<" "<<mCell.temperature<<endl; //energy_density;
    }
    */

    // -------------------------------------
    // Dummy test ...
    //

    int nX = bInfo.ny;
    int nY = bInfo.nx;
    int nT = bInfo.ntau; // quick fix for now ...

    if (!initBranch)
        InitBranch(nX,nY,250);

    //works with arbitrary size ... but maybe later wrt to FNO etc ... start maaybe with fixed tau bins ...
    m_xyt2=std::vector<std::vector<std::vector<double>>>(nX, std::vector<std::vector<double>>(nY, std::vector<double>(nT)));

    // add different resolution, directly, use the interpolation etc of bulk_info ... or try to do it in oython later !???

    for (int k=0; k<(nT); k++){
        for (int i=0; i<(nX); i++){
            for (int j=-0; j<(nY); j++){

                int nIndex = bInfo.CellIndex(k,i,j,0);
                auto mCell = bInfo.data.at(nIndex);

                if (k<250)
                    m_xyt[i][j][k] = mCell.energy_density; //temperature; //energy_density;
                //more complicated, push_back of 1d to 2d ... or: since only 3 dimension is variable ...
                m_xyt2[i][j][k] = (mCell.energy_density); //temperature; //energy_density
                //DEBUG: Not working, check indicies above!!!!
                //if (mCell.energy_density>0)
                //  cout<<i<<" "<<j<<" "<<k<<" "<<m_xyt2[i][j][k]<<endl;
            }
        }
    }

    // -------------------------------------
    t->Fill();
    //t->Print();
  }
  //Bummer, finish actually not yet recursively implemented !!!
  //void Finish() {cout<<"Finish called ..."<<endl; f->cd();hPt->Write("hPt");t->Write("t");f->ls();
  //    f->Write();f->Close();
  //}

  private:

  string oName;
  TFile *f;
  //TH1D *hPt;
  TTree *t;

  bool initBranch;

  std::vector<std::vector<std::vector<double>>> m_xyt;
  std::vector<std::vector<std::vector<double>>> m_xyt2;

};

// -------------------------------------

int main(int argc, char** argv)
{
    clock_t t; t = clock();
    time_t start, end; time(&start);

    Show();

    JetScapeLogger::Instance()->SetDebug(false);
    JetScapeLogger::Instance()->SetRemark(false);
    //SetVerboseLevel (9 a lot of additional debug output ...)
    //If you want to suppress it: use SetVerboseLevle(0) or max  SetVerboseLevle(9) or 10
    JetScapeLogger::Instance()->SetVerboseLevel(0);

    // -------------------------------------

    auto jetscape = make_shared<JetScape>();
    jetscape->SetXMLMainFileName("../config/jetscape_main.xml");
    jetscape->SetXMLUserFileName("../fno_hydro/config/jetscape_user_root_bulk_test.xml");
    //jetscape->SetReuseHydro (false);
    //jetscape->SetNReuseHydro (0);

    // Initial conditions and hydro
    auto trento = make_shared<TrentoInitial>();
    auto null_predynamics = make_shared<NullPreDynamics> ();
    auto pGun= make_shared<PGun> ();
    auto hydro = make_shared<MpiMusic> ();
    jetscape->Add(trento);
    jetscape->Add(null_predynamics);
    //jetscape->Add(pGun);
    jetscape->Add(hydro);

    // surface sampler
    auto iSS = make_shared<iSpectraSamplerWrapper> ();
    //jetscape->Add(iSS);


    auto jlossmanager = make_shared<JetEnergyLossManager> ();
    auto jloss = make_shared<JetEnergyLoss> ();
    auto matter = make_shared<Matter> ();

    jloss->Add(matter);
    jlossmanager->Add(jloss);
    //jetscape->Add(jlossmanager);

    // -------------------------------------

    auto demoBulk =  make_shared<DemoBulk>("demo_bulk_test.root");
    jetscape->Add(demoBulk);

    auto writer= make_shared<JetScapeWriterAscii> ("test_out.dat");
    jetscape->Add(writer);

    // -------------------------------------

    jetscape->Init();
    jetscape->Exec();
    jetscape->Finish();

    // -------------------------------------

    INFO_NICE<<"Finished!";
    cout<<endl;

    t = clock() - t;
    time(&end);
    printf ("CPU time: %f seconds.\n",((float)t)/CLOCKS_PER_SEC);
    printf ("Real time: %f seconds.\n",difftime(end,start));
    cout<<endl;

}

// -------------------------------------

void Show()
{
  INFO_NICE<<"-----------------------------------------";
  INFO_NICE<<"| Bulk ROOT Test JetScape Framework ... |";
  INFO_NICE<<"-----------------------------------------";
  INFO_NICE;
}
