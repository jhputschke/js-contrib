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
//REMARK:  cmake .. -DUSE_ROOT=ON -DUSE_MUSIC=ON -DUSE_ISS=ON (wihh muscis and libtotch QMP confflict!!!
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
#include "iSpectraSamplerWrapper.h"
#include "NullPreDynamics.h"
#include "FnoRooIn.h"

#include <chrono>
#include <thread>

using namespace std;
using namespace Jetscape;

// Forward declarations
// / -------------------------------------

void Show();

// -------------------------------------

int main(int argc, char** argv)
{
    clock_t t; t = clock();
    time_t start, end; time(&start);

    JetScapeLogger::Instance()->SetDebug(false);
    JetScapeLogger::Instance()->SetRemark(false);
    JetScapeLogger::Instance()->SetVerboseLevel(0);

    string fName="test_bulk_out.dat.gz";
    bool roofileIn = false;
    if (argc > 1)
      roofileIn = atoi(argv[1]);
      if (argc > 2)
        fName = argv[2];

    if (roofileIn)
        cout<<" ==> Use ROOT file for full hydro evolution ..."<<endl;

    // -------------------------------------
    Show();

    auto jetscape = make_shared<JetScape>();
    jetscape->SetXMLMainFileName("../config/jetscape_main.xml");
    if (roofileIn)
        jetscape->SetXMLUserFileName("../fno_hydro/config/fno_bulk_test_hydro.xml");
    else
        jetscape->SetXMLUserFileName("../fno_hydro/config/fno_bulk_test.xml");

    auto hydro = make_shared<FnoRooIn> ();
    jetscape->Add(hydro);

    // surface sampler
    auto iSS = make_shared<iSpectraSamplerWrapper> ();
    jetscape->Add(iSS);

    auto writer= make_shared<JetScapeWriterAsciiGZ>(fName);
    jetscape->Add(writer);


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
  INFO_NICE<<"----------------------------------------";
  INFO_NICE<<"| FNO Bulk Test JetScape Framework ... |";
  INFO_NICE<<"----------------------------------------";
  INFO_NICE;
}

// -------------------------------------
