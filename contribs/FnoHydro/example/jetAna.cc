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

  TString fNameOut = "test_jet_ana.root";
  string fNameIn = "test_jet_out.dat";

  if (argc > 1)
    fNameIn = argv[1];
      if (argc > 2)
        fNameOut = argv[2];


  cout<<endl;

  TFile* file = new TFile(fNameOut, "RECREATE");
  TH1D* hPt = new TH1D("hPt", "Jet Pt (hadronic)", 60, 0, 60); hPt->Sumw2();
  TH1D* hPtP = new TH1D("hPtP", "Jet Pt (partonic)", 60, 0, 60); hPtP->Sumw2();
  TH1D* hM = new TH1D("hM", "Jet M (hadronic)", 64, 0, 16); hM->Sumw2();
  TH1D* hz = new TH1D("hz","FF (hadronic)",20,0,1); hz->Sumw2();
  TH1D* hzP = new TH1D("hzP","FF (partonic)",20,0,1); hzP->Sumw2();
  TH1D* hzIP = new TH1D("hzIP","FF (hadronic) with pT shower IP",20,0,1); hzIP->Sumw2();
  TH1D* hzPIP = new TH1D("hzPIP","FF (partonic) with pT shower IP",20,0,1); hzPIP->Sumw2();
  TH1D* hEta = new TH1D("hEta","Eta Jet (hadronic)",40,-2,2); hEta->Sumw2();
  TH1D *hPhi = new TH1D("hPhi","Phi Jet (hadronic)",45,0,2*TMath::Pi());  hPhi->Sumw2();

  fjcore::JetDefinition jet_def(fjcore::antikt_algorithm, 0.7);

  vector<shared_ptr<PartonShower>> mShowers;

  auto reader=make_shared<JetScapeReaderAsciiGZ>(fNameIn);
  //auto reader=make_shared<JetScapeReaderAscii>(fNameIn);

  int nHadroJets = 0;
  int nParJets = 0;

  while (!reader->Finished())
    {
      reader->Next();

      //if (reader->GetCurrentEvent()>10) break;

      cout<<"Analyze current event = "<<reader->GetCurrentEvent()<<endl;
      mShowers=reader->GetPartonShowers();

      cout<<"Number of shower initiating partons = "<<mShowers.size()<<endl;

      cout<<"Shower initiating parton : "<<*(mShowers[0]->GetPartonAt(0))<<endl;
      cout<<"Number of Partons is: "<<mShowers[0]->GetFinalPartonsForFastJet().size()<< endl;

      fjcore::ClusterSequence cs(mShowers[0]->GetFinalPartonsForFastJet(), jet_def);
      vector<fjcore::PseudoJet> jets = fjcore::sorted_by_pt(cs.inclusive_jets(2));

      cout<<"Number of Partonic Jets is : "<<jets.size()<<endl;
	  for (int k=0;k<jets.size();k++) {
			if (k>0) break;
	        cout<<"Anti-kT jet "<<k<<" : "<<jets[k]<<endl;

			hPtP->Fill(jets[k].pt());

			// ----
			auto cons = jets[k].constituents();
            //cout<<cons.size()<<endl;
            for (auto c : cons) {
              hzP->Fill(c.pt()/jets[k].pt());
              hzPIP->Fill(c.pt()/mShowers[0]->GetPartonAt(0)->pt());
            }

            nParJets++;
		}

      auto hadrons = reader->GetHadrons();
      cout<<"Number of hadrons is: " << hadrons.size() << endl;

      fjcore::ClusterSequence hcs(reader->GetHadronsForFastJet(), jet_def);
      vector<fjcore::PseudoJet> hjets = fjcore::sorted_by_pt(hcs.inclusive_jets(2));

      cout<<"Number of Hadronic Jets is : "<<hjets.size()<<endl;
      for (int k=0;k<hjets.size();k++) {
          if (k>0) break;
          cout<<"Anti-kT jet "<<k<<" : "<<hjets[k]<<endl;

          hPt->Fill(hjets[k].pt());
          hM->Fill(hjets[k].m());
          hEta->Fill(hjets[k].eta());
          hPhi->Fill(hjets[k].phi());

          // ----
          auto cons = hjets[k].constituents();
          //cout<<cons.size()<<endl;
          for (auto c : cons) {
            hz->Fill(c.pt()/hjets[k].pt());
            hzIP->Fill(c.pt()/mShowers[0]->GetPartonAt(0)->pt());
          }

          nHadroJets++;
      }

    }

    hz->Scale(1/(double) nHadroJets); hzIP->Scale(1/(double) nHadroJets);
    hzP->Scale(1/(double) nParJets); hzPIP->Scale(1/(double) nParJets);

    cout<<endl;
    cout<<"# Hadronic jets = "<<nHadroJets<<endl;
    cout<<"# Partonic jets = "<<nParJets<<endl;
    cout<<endl;

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
