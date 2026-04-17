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
#include "JetScapeReaderFinalStateHadrons.h"
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

  string fNameIn = "test_out.dat";
  TString fNameOut = "test_bulk_ana.root";

  if (argc > 1)
    fNameIn = argv[1];
    if (argc > 2)
      fNameOut = argv[2];

  cout<<endl;

  TFile* file = new TFile(fNameOut, "RECREATE");
  TH1D* hPt = new TH1D("hPt", "Pt", 50, 0, 5); hPt->Sumw2();
  TH1D* hPtMid = new TH1D("hPtMid", "PtMid", 50, 0, 5); hPtMid->Sumw2();
  //TH1D *hPhi = new TH1D("hPhi","Phi",180,0,2*TMath::Pi());
  TH1D *hPhi = new TH1D("hPhi","Phi",45,-TMath::Pi(),TMath::Pi());  hPhi->Sumw2();
  TH1D *hPhiMid = new TH1D("hPhiMid","PhiMid",45,-TMath::Pi(),TMath::Pi());  hPhiMid->Sumw2();
  TH1D *hEta = new TH1D("hEta","Eta",22,-1.1,1.1); hEta->Sumw2();
  TH1D *hEtaFull = new TH1D("hEtaFull","Eta",40,-5,5); hEtaFull->Sumw2();
  TH1D *hMult = new TH1D("hMult","Mult |eta|<1",200,0,200);

  auto reader=make_shared<JetScapeReaderAsciiGZ>(fNameIn);
  //auto reader=make_shared<JetScapeReaderAscii>(fNameIn);

  int nFullEvents = 0;

  while (!reader->Finished())
    {
      reader->Next();
      int evMult = 0;

      cout<<"Analyze current event = "<<reader->GetCurrentEvent()<<endl;
      auto hadrons = reader->GetHadrons();
      cout<<"Number of hadrons is: " << hadrons.size() << endl;
      if (hadrons.size()>0)
        nFullEvents++;

      for (auto h : hadrons)
      {
        //cout<<h<<endl;
        hEtaFull->Fill(h->eta());
        if (TMath::Abs(h->eta())<1) {
            hPhiMid->Fill(h->phi_std());
            hPtMid->Fill(h->pt());
            evMult++;
        }

        if (TMath::Abs(h->eta())<10) {
            hEta->Fill(h->eta());
            hPt->Fill(h->pt());
            //if (h->pt()>0.2)
            hPhi->Fill(h->phi_std());
        }
      }
      hMult->Fill(evMult);
    }

    cout<<"Number of full hadronization events = "<<nFullEvents<<endl;

    hPhi->Scale(1/(double) nFullEvents * 1/hPhi->GetBinWidth(1));
    hPhiMid->Scale(1/(double) nFullEvents * 1/hPhi->GetBinWidth(1));
    hEta->Scale(1/(double) nFullEvents * 1/hEta->GetBinWidth(1));
    hPt->Scale(1/(double) nFullEvents * 1/hPt->GetBinWidth(1));
    hPtMid->Scale(1/(double) nFullEvents * 1/hPtMid->GetBinWidth(1));

    //hPhi->Scale(1/(double) hPhi->GetEntries() * 1/hPhi->GetBinWidth(1));
    //hEta->Scale(1/(double) hEta->GetEntries() * 1/hEta->GetBinWidth(1));
    //hPt->Scale(1/(double) hPt->GetEntries() * 1/hPt->GetBinWidth(1));


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
