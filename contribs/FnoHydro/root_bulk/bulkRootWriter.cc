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

// Forward declarations
// / -------------------------------------

void Show();

// -------------------------------------

class BulkRootWriter : public JetScapeModuleBase
{
  public:

    BulkRootWriter() : JetScapeModuleBase() {SetId("BulkRootWriter"); oName = ""; initBranch=false; nFeatures=3;}
    BulkRootWriter(string m_oName) : JetScapeModuleBase() {SetId("BulkRootWriter"); oName = m_oName; initBranch=false; nFeatures=3;}
    BulkRootWriter(string m_oName, int m_nFeatures) : JetScapeModuleBase() {SetId("BulkRootWriter"); oName = m_oName; initBranch=false; nFeatures=m_nFeatures;}
    virtual ~BulkRootWriter();

    void Init();
    void InitBranch(int nX=100, int nY=100, int nT=100);
    void Exec();

  private:

    string oName;

    TFile *f;
    TTree *t;

    bool initBranch;
    int nFeatures;

    // REMARK: Check with float precision ...
    //std::vector<std::vector<std::vector<double>>> m_xyt;
    //std::vector<std::vector<std::vector<double>>> m_xyt2;
    //std::vector<std::vector<std::vector<std::vector<double>>>> m_xyt;
    //std::vector<std::vector<std::vector<std::vector<double>>>> m_xyt2;

    std::vector<std::vector<std::vector<std::vector<float>>>> m_xyt;
    std::vector<std::vector<std::vector<std::vector<float>>>> m_xyt2;
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
    //auto pGun= make_shared<PGun> ();
    auto hydro = make_shared<MpiMusic> ();

    jetscape->Add(trento);
    jetscape->Add(null_predynamics);
    //jetscape->Add(pGun);
    jetscape->Add(hydro);

    // surface sampler
    //auto iSS = make_shared<iSpectraSamplerWrapper> ();
    //jetscape->Add(iSS);

    auto jlossmanager = make_shared<JetEnergyLossManager> ();
    auto jloss = make_shared<JetEnergyLoss> ();
    auto matter = make_shared<Matter> ();

    jloss->Add(matter);
    jlossmanager->Add(jloss);
    //jetscape->Add(jlossmanager);

    // -------------------------------------

    //auto bulkWriter =  make_shared<BulkRootWriter>("bulk_root_writer_JS_3.7_AuAu_0_10_250ev.root");
    auto bulkWriter =  make_shared<BulkRootWriter>("bulk_root_writer_test.root");
    jetscape->Add(bulkWriter);

    //auto writer= make_shared<JetScapeWriterAscii> ("test_out.dat");
    //jetscape->Add(writer);

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
  INFO_NICE<<"------------------------------------------";
  INFO_NICE<<"| Bulk ROOT Writer JetScape Framework ... |";
  INFO_NICE<<"------------------------------------------";
  INFO_NICE;
}

// -------------------------------------

void BulkRootWriter::Init() {

    f=new TFile(oName.c_str(),"RECREATE");
    t=new TTree("t","Tree");
}

BulkRootWriter::~BulkRootWriter() {
    //cout<<"Finish called ..."<<endl;
    f->cd();f->ls(); t->Print(); f->Write();f->Close();
}

// Can be done in Init too ...
void BulkRootWriter::InitBranch(int nX, int nY, int nT)
{
    JSINFO<<"InitBranch Custom Resolution (x,y,tau)        : "<<nX<<" "<<nY<<" "<<nT;
    JSINFO<<"# of features:"<<nFeatures;
    initBranch = true;

    //m_xyt=std::vector<std::vector<std::vector<std::vector<double>>>> (nX, std::vector<std::vector<std::vector<double>>>(nY, std::vector<std::vector<double>>(nT)));
    m_xyt = std::vector<std::vector<std::vector<std::vector<float>>>>(
            nX,
            std::vector<std::vector<std::vector<float>>>(
                nY,
                std::vector<std::vector<float>>(
                    nT,
                    std::vector<float>(nFeatures)
                )
            )
        );
    //REMARK: Maybe better two trees than two branches, could be easier to load only one tree via uproot than a single branch !????
    t->Branch("user_res",&m_xyt);
    t->Branch("hydro_res",&m_xyt2);
}

// -------------------------------------

void BulkRootWriter::Exec() {

  // quick and dirty test to get hydro bulk info
  auto hydro = JetScapeSignalManager::Instance()->GetHydroPointer();
  if (hydro.lock()) {
    auto bInfo = hydro.lock()->get_bulk_info();

    // REMARK: Think about how to best store the hydro grid and other grid info in tree etc ...
    /*
    cout<<bInfo.get_data_size()<<" "<<bInfo.is_boost_invariant()<<endl;
    cout<<"Grid Info : "<<bInfo.ntau<<" "<<bInfo.nx<<" "<<bInfo.ny<<" "<<bInfo.neta<<endl;
    cout<<bInfo.tau_min<<" "<<bInfo.Tau0()<<" "<<bInfo.XMin()<<" "<<bInfo.YMin()<<" "<<bInfo.EtaMin()<<endl;
    cout<<bInfo.TauMax()<<" "<<bInfo.XMax()<<" "<<bInfo.YMax()<<" "<<bInfo.EtaMax()<<endl;
    cout<<bInfo.dtau<<" "<<bInfo.dx<<" "<<bInfo.dy<<" "<<bInfo.deta<<endl;
    cout<<bInfo.GetIdTau(0.6)<<" "<<bInfo.GetIdTau(10.)<<" "<<bInfo.GetIdTau(100.)<<endl;
    auto fCell = bInfo.GetAtTimeStep(bInfo.GetIdTau(0.6),0,0,0);
    cout<<fCell.temperature<<endl;
    fCell = bInfo.GetAtTimeStep(bInfo.GetIdTau(5.),0,0,0);
    cout<<fCell.temperature<<endl;
    */

    // REMARK: Maybe more effcient ...

    int nX = bInfo.ny;
    int nY = bInfo.nx;
    int nT = bInfo.ntau;

    double tau0 = bInfo.Tau0();

    double xMin = bInfo.XMin();
    double xMax = bInfo.XMax(); //same for y axis ...
    double tauMax = bInfo.TauMax();

    //DBEUG:
    cout<<nT<<" "<<tauMax<<endl;

    double dX = bInfo.dx;
    double dTau = bInfo.dtau;

    double m_TauMax = 5.0; //in fermi ..
    double m_dTau = 0.1; //in fermi ...
    int m_nT = m_TauMax/m_dTau;

    double m_dX = 0.5;
    int m_nX = (xMax-xMin+dX)/m_dX;
    int m_nY = m_nX;

    if (!initBranch) {
        JSINFO<<"InitBrach: tau0 = "<<tau0;
        JSINFO<<"InitBranch Hydro Sim Resolution (x,y) all tau : "<<bInfo.nx<<" "<<bInfo.ny;

        InitBranch(m_nX,m_nY,m_nT);
    }

    // -------------------------------------
    // Full Hydro evolution ...
    // REMARK: Figure out dimnensions to save other cell info ... channels in FNO !!!

    //m_xyt2=std::vector<std::vector<std::vector<std::vector<float>>>>(nX, std::vector<std::vector<float>>(nY, std::vector<float>(nT)));
    /*
    m_xyt2 = std::vector<std::vector<std::vector<std::vector<float>>>>(
            nX,
            std::vector<std::vector<std::vector<float>>>(
                nY,
                std::vector<std::vector<float>>(
                    nT,
                    std::vector<float>(nFeatures)
                )
            )
        );
    */

    /*
    for (int k=0; k<(nT); k++){
        for (int i=0; i<(nX); i++){
            for (int j=-0; j<(nY); j++){

                int nIndex = bInfo.CellIndex(k,i,j,0); // last eta here index 0 since boost invariant ...
                auto mCell = bInfo.data.at(nIndex);

                m_xyt2[i][j][k][0] = (float) (mCell.energy_density); //temperature; //energy_density
                //m_xyt2[i][j][k][1] = (float) (mCell.temperature);
                m_xyt2[i][j][k][1] = (float) (mCell.vx);
                m_xyt2[i][j][k][2] = (float) (mCell.vy);
                // 2+1D vz = 0  ...
                //m_xyt2[i][j][k][4] = (mCell.vz);
                //cout<<mCell.vx<<" "<<mCell.vy<<" "<<mCell.vz<<endl;
                //if ((m_xyt2)[i][j][k][0] > 0.1 && k<1) {
                //  cout<<(m_xyt2)[i][j][k][0]<<" "<<(m_xyt2)[i][j][k][1]<<endl;//" via EOS = "<<fnoEOS->get_temperature((*m_xyt)[i][j][k][0], 0)<<endl;
                // }
            }
        }
    }
   */

    // -------------------------------------
    //
    // -------------------------------------
    // User Hydro evolution resolution ...

    for (int k=0; k<(m_nT); k++){
        for (int i=0; i<(m_nX); i++){
            for (int j=-0; j<(m_nY); j++){

                //int nIndex = bInfo.CellIndex(k,i,j,0); // last eta here index 0 since boost invariant ...
                double tau_In = tau0 + k*m_dTau;
                double x_In = xMin + i*m_dX;
                double y_In = xMin + j*m_dX;

                auto mCell = bInfo.get(tau_In, x_In, y_In, 0);

                m_xyt[i][j][k][0] = (float) (mCell.energy_density);
                //m_xyt[i][j][k][1] = (float) (mCell.temperature); //temperature; //energy_density
                m_xyt[i][j][k][1] = (float) (mCell.vx);
                m_xyt[i][j][k][2] = (float) (mCell.vy);
                //m_xyt[i][j][k][1] = (mCell.vz);
                 // 2+1D vz = 0  ...
                 //if ((m_xyt)[i][j][k][0] > 0.1 && k<1) {
                   //cout<<(m_xyt)[i][j][k][0]<<" "<<(m_xyt)[i][j][k][1]<<endl;//" via EOS = "<<fnoEOS->get_temperature((*m_xyt)[i][j][k][0], 0)<<endl;
                   // }
            }
        }
    }

    // -------------------------------------

    t->Fill();
    //t->Print();
    //REMARK: Check for memory leaks --> does not look like there is one !!!!
    m_xyt2.clear(); //maybe better resize it above ...
  }
  else {JSWARN<<"No hydro pointer available ...";}
}
