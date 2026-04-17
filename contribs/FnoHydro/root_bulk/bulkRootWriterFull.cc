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

    //std::unique_ptr<EOS> fnoEOS;

    string oName;

    TFile *f;
    TTree *t;

    bool initBranch;
    int nFeatures;

    // REMARK: Check with float precision ... seems fine wrt to check Python notebook ... but to be verified !!!!
    //
    //std::vector<std::vector<std::vector<double>>> m_xyt;
    //std::vector<std::vector<std::vector<double>>> m_xyt2;

    //std::vector<std::vector<std::vector<std::vector<double>>>> m_xyt;
    //std::vector<std::vector<std::vector<std::vector<double>>>> m_xyt2;

    std::vector<std::vector<std::vector<std::vector<float>>>> m_xyt;
    std::vector<std::vector<std::vector<std::vector<float>>>> m_xyt2;
    std::vector<std::vector<float>> m_foSurf;
    std::vector<float> m_foEdT;

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

    auto bulkWriter =  make_shared<BulkRootWriter>("bulk_root_writer_full_test.root");
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

    JSINFO<<"BulkRootWriter: Output file = "<<oName;
    JSINFO<<"BulkRootWriter: # of features = "<<nFeatures;
}

BulkRootWriter::~BulkRootWriter() {
    //cout<<"Finish called ..."<<endl;
    f->cd();f->ls(); t->Print(); f->Write();f->Close();
}

// Can be done in Init too ...
void BulkRootWriter::InitBranch(int nX, int nY, int nT)
{
    JSINFO<<"InitBranch Custom Resolution (x,y) all tau    : "<<nX<<" "<<nY;//<<" "<<nT;
    JSINFO<<"# of features = "<<nFeatures;
    initBranch = true;

    //REMARK: Maybe better two trees than two branches, could be easier to load only one tree via uproot than a single branch !????
    t->Branch("user_res",&m_xyt);
    t->Branch("hydro_res",&m_xyt2);
    t->Branch("foSurf",&m_foSurf);
    t->Branch("foEdT",&m_foEdT);
}

// -------------------------------------

void BulkRootWriter::Exec() {

  // quick and dirty test to get hydro bulk info
  auto hydro = JetScapeSignalManager::Instance()->GetHydroPointer();
  if (hydro.lock()) {
    auto bInfo = hydro.lock()->get_bulk_info();

    // REMARK: Maybe more effcient ...

    int nX = bInfo.ny;
    int nY = bInfo.nx;
    int nT = bInfo.ntau;

    double tau0 = bInfo.Tau0();

    double xMin = bInfo.XMin();
    double xMax = bInfo.XMax(); //same for y axis ...
    double tauMax = bInfo.TauMax();

    double dX = bInfo.dx;
    double dTau = bInfo.dtau;

    //double m_TauMax = 5.0; //in fermi ..
    double m_dTau = 0.1; //in fermi ...
    int m_nT = (tauMax-tau0)/m_dTau+1;

    //DBEUG:
    //cout<<nT<<" "<<tauMax<<" "<<m_nT<<" "<<m_nT*m_dTau+tau0<<endl;

    double m_dX = 0.5;
    int m_nX = (xMax-xMin+dX)/m_dX;
    int m_nY = m_nX;

    if (!initBranch) {

        JSINFO<<"InitBrach: tau0 = "<<tau0;
        JSINFO<<"InitBranch Hydro Sim Resolution (x,y) all tau : "<<bInfo.nx<<" "<<bInfo.ny;

        InitBranch(m_nX,m_nY,m_nT);

        //quick and dirty ...
        //fnoEOS=std::make_unique<EOS>(91);
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

    // -------------------------------------
    // User Hydro evolution resolution full time evolution ...

    m_xyt = std::vector<std::vector<std::vector<std::vector<float>>>>(
            m_nX,
            std::vector<std::vector<std::vector<float>>>(
                m_nY,
                std::vector<std::vector<float>>(
                    m_nT,
                    std::vector<float>(nFeatures)
                )
            )
        );

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

                // **********************************************************************************************************************************
                // Temp and EOS debug ... ==> identical results (some differences seen after wrting / reading ROOT file !??? precision issues !!???)
                // **********************************************************************************************************************************
                // float tempEOS = fnoEOS->get_temperature((float) (mCell.energy_density)/Util::hbarc, 0)*Util::hbarc;
                // if (i==30 and j==30)
                //     cout<<tau_In<< " "<<mCell.energy_density << " "<<mCell.temperature<<" "<<tempEOS<<endl;

                //----------------------------------
                //m_xyt[i][j][k][1] = (mCell.vz);
                 // 2+1D vz = 0  ...
                 //if ((m_xyt)[i][j][k][0] > 0.1 && k<1) {
                   //cout<<(m_xyt)[i][j][k][0]<<" "<<(m_xyt)[i][j][k][1]<<endl;//" via EOS = "<<fnoEOS->get_temperature((*m_xyt)[i][j][k][0], 0)<<endl;
                   // }
                 //----------------------------------
            }
        }
    }

    // -------------------------------------
    // Freezeout surface vector/grid to be stored ...
    // REMARK: Figure out what format and if it useable for FNO !??? ...
    //
    // ==> minimum info for further usage in case of ideal hydro for now ... Follow up !!!!
    //

    std::vector<SurfaceCellInfo> m_surfaceCellVector;
    hydro.lock()->getSurfaceCellVector(m_surfaceCellVector);

    int nSurfCells = m_surfaceCellVector.size();

    //cout<<nSurfCells<<endl;

    // save ed and T per event as backup ...
    // hardcoded for now: tau, x, y, d3sigma_mu[0], d3sigma_mu[1], d3sigma_mu[2], umu[0], umu[1], umu[2],
    m_foSurf = std::vector<std::vector<float>>(
        nSurfCells,
        std::vector<float>(9)
    );

    m_foEdT = std::vector<float>(2);
    m_foEdT[0] = m_surfaceCellVector[0].energy_density;
    m_foEdT[1] = m_surfaceCellVector[0].temperature;

    for (int i=0; i<nSurfCells; i++){
        m_foSurf[i][0] = (float) m_surfaceCellVector[i].tau;
        m_foSurf[i][1] = (float) m_surfaceCellVector[i].x;
        m_foSurf[i][2] = (float) m_surfaceCellVector[i].y;
        m_foSurf[i][3] = (float) m_surfaceCellVector[i].d3sigma_mu[0];
        m_foSurf[i][4] = (float) m_surfaceCellVector[i].d3sigma_mu[1];
        m_foSurf[i][5] = (float) m_surfaceCellVector[i].d3sigma_mu[2];
        m_foSurf[i][6] = (float) m_surfaceCellVector[i].umu[0];
        m_foSurf[i][7] = (float) m_surfaceCellVector[i].umu[1];
        m_foSurf[i][8] = (float) m_surfaceCellVector[i].umu[2];
    }

    //DEBUG: ...
    // REMARK: for ideal 2+1 hydro: pi[] = 0 , eta = 0, cell.d3sigma_mu[3] = 0 and cell.umu[3] = 0
    // Also: energy_density and temperature given by freezout temp is the same for all cells!!!
    //
    /*
    for (auto& cell : m_surfaceCellVector) {


        // Process each cell in the surface vector
        //if (cell.eta>0) cout<<"AAAhhhh ...."<<endl;
        cout<<cell.tau<<" "<<cell.x<<" "<<cell.y<<" "<<cell.eta<<" "<<cell.energy_density<<" "<<cell.temperature<<" "<<cell.pressure<<" "<<cell.entropy_density<<" "<<cell.baryon_density<<endl;
        cout<<" "<<cell.d3sigma_mu[0]<<" "<<cell.d3sigma_mu[1]<<" "<<cell.d3sigma_mu[2]<<" "<<cell.d3sigma_mu[3]<<" "<<endl;
        cout<<cell.umu[0]<<" "<<cell.umu[1]<<" "<<cell.umu[2]<<" "<<cell.umu[3]<<" "<<endl;
        cout<<cell.mu_B<<" "<<cell.mu_Q<<" "<<cell.mu_S<<" "<<cell.bulk_Pi<<" "<<endl;
        cout<<cell.pi[0]<<" "<<cell.pi[1]<<" "<<cell.pi[2]<<" "<<cell.pi[3]<<" "<<endl;
    }
    //*/
    // -------------------------------------

    t->Fill();
    //t->Print();

    //REMARK: Check for memory leaks --> does not look like there is one !!!!
    m_xyt2.clear(); //maybe better resize it above ...
    m_xyt.clear(); //maybe better resize it above ...
    m_foSurf.clear();
    m_foEdT.clear();

  }
  else {JSWARN<<"No hydro pointer available ...";}
}
