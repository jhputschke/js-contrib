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
// Reader test (focus on graph)

#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <thread>

#include "gzstream.h"
#include "PartonShower.h"
#include "JetScapeLogger.h"
#include "JetScapeReader.h"
#include "JetScapeBanner.h"
#include "fjcore.hh"

#include <GTL/dfs.h>

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
//using namespace fjcore;

using namespace Jetscape;

// -------------------------------------

// Forward declaration
ostream & operator<<(ostream & ostr, const fjcore::PseudoJet & jet);

// ----------------------

int main(int argc, char** argv)
{
  JetScapeLogger::Instance()->SetDebug(false);
  JetScapeLogger::Instance()->SetRemark(false);
  //SetVerboseLevel (9 a lot of additional debug output ...)
  //If you want to suppress it: use SetVerboseLevle(0) or max  SetVerboseLevle(9) or 10
  JetScapeLogger::Instance()->SetVerboseLevel(0);

  TString fNameOut = "test_ana.root";
  string fNameIn = "test_out_fno.dat";
  string fNameIn2 = "test_out_hydro.dat";

  if (argc > 1)
    fNameIn = argv[1];
    if (argc > 2)
      fNameIn2 = argv[2];
      if (argc > 3)
        fNameOut = argv[3];


  cout<<endl;

  TFile* file = new TFile(fNameOut, "RECREATE");
  TH1D* hPt = new TH1D("hPt_FNO", "Pt", 60, 0, 60); hPt->Sumw2();
  TH1D* hM = new TH1D("hM_FNO", "PM", 40, 0, 10); hM->Sumw2();
  TH1D* hPtHydro = new TH1D("hPt_Hydro", "Pt", 60, 0, 60); hPtHydro->Sumw2();
  TH1D* hMHydro = new TH1D("hM_hydro", "PM", 40, 0, 10); hMHydro->Sumw2();
  TH1D* hdPt = new TH1D("hdPt","",40,-5,5);hdPt->Sumw2();
  TH1D* hdM = new TH1D("hdM","",16,-2,2);hdM->Sumw2();

  //Do some dummy jetfinding ...
  fjcore::JetDefinition jet_def(fjcore::antikt_algorithm, 0.4);

  vector<shared_ptr<PartonShower>> mShowers;
  vector<shared_ptr<PartonShower>> mShowers2;

  // Hide Template (see class declarations in reader/JetScapeReader.h) ...
  auto reader=make_shared<JetScapeReaderAscii>(fNameIn);
  auto reader2=make_shared<JetScapeReaderAscii>(fNameIn2);
  //auto reader=make_shared<JetScapeReaderAsciiGZ>("test_out.dat.gz");

  while (!reader->Finished())
    {
      reader->Next();
      reader2->Next();

      //if (reader->GetCurrentEvent()>10) break;

      cout<<"Analyze current event = "<<reader->GetCurrentEvent()<<" "<<reader2->GetCurrentEvent()<<endl;
      mShowers=reader->GetPartonShowers();
      mShowers2=reader2->GetPartonShowers();

      cout<< mShowers[0]->GetFinalPartonsForFastJet().size()<< " "<<mShowers2[0]->GetFinalPartonsForFastJet().size()<<endl;

      fjcore::ClusterSequence cs(mShowers[0]->GetFinalPartonsForFastJet(), jet_def);
      fjcore::ClusterSequence cs2(mShowers2[0]->GetFinalPartonsForFastJet(), jet_def);

      vector<fjcore::PseudoJet> jets = fjcore::sorted_by_pt(cs.inclusive_jets(2));
      vector<fjcore::PseudoJet> jets2 = fjcore::sorted_by_pt(cs2.inclusive_jets(2));

	  for (int k=0;k<jets.size();k++) {
		if (k>0) break;
	    cout<<"Anti-kT jet "<<k<<" : "<<jets[k]<<endl;
		cout<<"Anti-kT jet "<<k<<" : "<<jets2[k]<<endl;
		//hdPt->Fill(jets[k].pt()-jets2[k].pt());
		}

	  cout<<endl;
	  cout<<"Shower initiating parton : "<<*(mShowers[0]->GetPartonAt(0))<<endl;
	  cout<<"Shower initiating parton : "<<*(mShowers2[0]->GetPartonAt(0))<<endl;
	  cout<<endl;

    //cout << " Found " << finals << " final state partons." << endl;

      auto hadrons = reader->GetHadrons();
      cout<<"Number of hadrons is: " << hadrons.size() << endl;

      auto hadrons2 = reader2->GetHadrons();
      cout<<"Number of hadrons is: " << hadrons2.size() << endl;

      fjcore::ClusterSequence hcs(reader->GetHadronsForFastJet(), jet_def);
      vector<fjcore::PseudoJet> hjets = fjcore::sorted_by_pt(hcs.inclusive_jets(2));

      fjcore::ClusterSequence hcs2(reader2->GetHadronsForFastJet(), jet_def);
      vector<fjcore::PseudoJet> hjets2 = fjcore::sorted_by_pt(hcs2.inclusive_jets(2));
      //cout<<"AT HADRONIC LEVEL " << endl;
      cout<<"Number of Jets is : "<<hjets.size()<<endl;
      for (int k=0;k<hjets.size();k++) {
          if (k>0) break;
          cout<<"Anti-kT jet "<<k<<" : "<<hjets[k]<<endl;
          cout<<"Anti-kT jet "<<k<<" : "<<hjets2[k]<<endl;
          hPt->Fill(hjets[k].pt());
          hM->Fill(hjets[k].m());
          hPtHydro->Fill(hjets2[k].pt());
          hMHydro->Fill(hjets2[k].m());
          hdPt->Fill(hjets[k].pt()-hjets2[k].pt());
      }

    }

    reader->Close();
    file->Write();
    file->Close();
}

//----------------------------------------------------------------------
/// overloaded jet info output

ostream & operator<<(ostream & ostr, const fjcore::PseudoJet & jet) {
  if (jet == 0) {
    ostr << " 0 ";
  } else {
    ostr << " pt = " << jet.pt()
         << " m = " << jet.m()
         << " y = " << jet.rap()
         << " phi = " << jet.phi();
  }
  return ostr;
}


//----------------------------------------------------------------------
