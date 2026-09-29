// example/prod_AuAu_0_10_jet/root_export/root_writers.h
//
// C++ writers for hadrons_to_root.py: one hadron file (one tag) -> a TTree or an RNTuple,
// one entry per sample (oversample), from flat column arrays prepared in Python.  Declared
// with ROOT.gInterpreter.Declare, called through cppyy with numpy buffers.
//
// Entry layout (both formats):
//   event, unit, sample, bg_unit     int     the event (first event using the unit), the
//                                            unit (units/unit), the sample k within it, the
//                                            event's background unit (-1 if unknown)
//   n                                int     hadrons in this sample (TTree counter)
//   pid, pstat                       int[n]
//   E, px, py, pz                    float[n]  GeV
//   t, x, y, z                       float[n]  fm (only with with_x)
//
// Precision: bits_p / bits_x > 0 store p / x with that many mantissa bits (TTree:
// Float16_t "[0,0,bits]", RNTuple: Real32Trunc with 1 + 8 + bits bits); <= 0 full float32.
// update != 0 adds the tree / ntuple to an existing file (run_h5toROOT.py: the small tables
// are written first, by uproot), else the file is recreated.
// Returns the wall time of the ROOT part in seconds.

#pragma once

#include <TFile.h>
#include <TStopwatch.h>
#include <TTree.h>

#include <ROOT/RField.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace hadrons_root {

struct Columns {
  long nsamp = 0;
  const long *soff = nullptr;                          // sample s: rows soff[s] .. soff[s+1]
  const int *event = nullptr, *unit = nullptr, *ksample = nullptr, *bgunit = nullptr;
  const int *pid = nullptr, *pstat = nullptr;
  const float *p[4] = {nullptr, nullptr, nullptr, nullptr};   // E, px, py, pz
  const float *x[4] = {nullptr, nullptr, nullptr, nullptr};   // t, x, y, z
  bool with_x = false;
};

inline Columns make_columns(long nsamp, const long *soff, const int *event, const int *unit,
                            const int *ksample, const int *bgunit, const int *pid,
                            const int *pstat, const float *E, const float *px,
                            const float *py, const float *pz, const float *t, const float *x,
                            const float *y, const float *z, int with_x) {
  Columns c;
  c.nsamp = nsamp; c.soff = soff; c.event = event; c.unit = unit; c.ksample = ksample;
  c.bgunit = bgunit; c.pid = pid; c.pstat = pstat;
  c.p[0] = E; c.p[1] = px; c.p[2] = py; c.p[3] = pz;
  c.x[0] = t; c.x[1] = x; c.x[2] = y; c.x[3] = z;
  c.with_x = with_x != 0;
  return c;
}

static const char *P_NAMES[4] = {"E", "px", "py", "pz"};
static const char *X_NAMES[4] = {"t", "x", "y", "z"};

inline double write_ttree(const char *path, const char *name, int compression, int bits_p,
                          int bits_x, const Columns &c, int update = 0) {
  TStopwatch sw;
  std::unique_ptr<TFile> f(TFile::Open(path, update ? "UPDATE" : "RECREATE", "", compression));
  f->SetCompressionSettings(compression);
  TTree *tree = new TTree(name, name);
  long nmax = 0;
  for (long s = 0; s < c.nsamp; ++s) nmax = std::max(nmax, c.soff[s + 1] - c.soff[s]);
  int n = 0, event = 0, unit = 0, ksample = 0, bgunit = -1;
  std::vector<int> pid(std::max(nmax, 1L)), pstat(std::max(nmax, 1L));
  std::vector<std::vector<float>> pb(4, std::vector<float>(std::max(nmax, 1L)));
  std::vector<std::vector<float>> xb(4, std::vector<float>(std::max(nmax, 1L)));
  tree->Branch("event", &event, "event/I");
  tree->Branch("unit", &unit, "unit/I");
  tree->Branch("sample", &ksample, "sample/I");
  tree->Branch("bg_unit", &bgunit, "bg_unit/I");
  tree->Branch("n", &n, "n/I");
  tree->Branch("pid", pid.data(), "pid[n]/I");
  tree->Branch("pstat", pstat.data(), "pstat[n]/I");
  auto leaf = [](const char *nm, int bits) {
    return bits > 0 ? std::string(nm) + "[n]/f[0,0," + std::to_string(bits) + "]"
                    : std::string(nm) + "[n]/F";
  };
  for (int k = 0; k < 4; ++k) tree->Branch(P_NAMES[k], pb[k].data(), leaf(P_NAMES[k], bits_p).c_str());
  if (c.with_x)
    for (int k = 0; k < 4; ++k) tree->Branch(X_NAMES[k], xb[k].data(), leaf(X_NAMES[k], bits_x).c_str());
  for (long s = 0; s < c.nsamp; ++s) {
    const long a = c.soff[s], b = c.soff[s + 1];
    n = static_cast<int>(b - a);
    event = c.event[s]; unit = c.unit[s]; ksample = c.ksample[s]; bgunit = c.bgunit[s];
    std::copy(c.pid + a, c.pid + b, pid.begin());
    std::copy(c.pstat + a, c.pstat + b, pstat.begin());
    for (int k = 0; k < 4; ++k) std::copy(c.p[k] + a, c.p[k] + b, pb[k].begin());
    if (c.with_x)
      for (int k = 0; k < 4; ++k) std::copy(c.x[k] + a, c.x[k] + b, xb[k].begin());
    tree->Fill();
  }
  tree->Write();
  f->Close();
  return sw.RealTime();
}

inline void add_float_vector(ROOT::RNTupleModel &model, const char *name, int bits) {
  auto item = std::make_unique<ROOT::RField<float>>("_0");
  if (bits > 0) item->SetTruncated(1 + 8 + bits);      // sign + exponent + mantissa
  model.AddField(std::make_unique<ROOT::RVectorField>(name, std::move(item)));
}

inline double write_rntuple(const char *path, const char *name, int compression, int bits_p,
                            int bits_x, const Columns &c, int update = 0) {
  TStopwatch sw;
  auto model = ROOT::RNTupleModel::Create();
  auto event = model->MakeField<int>("event");
  auto unit = model->MakeField<int>("unit");
  auto ksample = model->MakeField<int>("sample");
  auto bgunit = model->MakeField<int>("bg_unit");
  auto pid = model->MakeField<std::vector<int>>("pid");
  auto pstat = model->MakeField<std::vector<int>>("pstat");
  for (int k = 0; k < 4; ++k) add_float_vector(*model, P_NAMES[k], bits_p);
  if (c.with_x)
    for (int k = 0; k < 4; ++k) add_float_vector(*model, X_NAMES[k], bits_x);
  ROOT::RNTupleWriteOptions opts;
  opts.SetCompression(compression);
  std::unique_ptr<TFile> file;
  std::unique_ptr<ROOT::RNTupleWriter> writer;
  if (update) {
    file.reset(TFile::Open(path, "UPDATE"));
    writer = ROOT::RNTupleWriter::Append(std::move(model), name, *file, opts);
  } else {
    writer = ROOT::RNTupleWriter::Recreate(std::move(model), name, path, opts);
  }
  auto &entry = writer->GetModel().GetDefaultEntry();
  std::shared_ptr<std::vector<float>> pv[4], xv[4];
  for (int k = 0; k < 4; ++k) pv[k] = entry.GetPtr<std::vector<float>>(P_NAMES[k]);
  if (c.with_x)
    for (int k = 0; k < 4; ++k) xv[k] = entry.GetPtr<std::vector<float>>(X_NAMES[k]);
  for (long s = 0; s < c.nsamp; ++s) {
    const long a = c.soff[s], b = c.soff[s + 1];
    *event = c.event[s]; *unit = c.unit[s]; *ksample = c.ksample[s]; *bgunit = c.bgunit[s];
    pid->assign(c.pid + a, c.pid + b);
    pstat->assign(c.pstat + a, c.pstat + b);
    for (int k = 0; k < 4; ++k) pv[k]->assign(c.p[k] + a, c.p[k] + b);
    if (c.with_x)
      for (int k = 0; k < 4; ++k) xv[k]->assign(c.x[k] + a, c.x[k] + b);
    writer->Fill();
  }
  writer.reset();                                      // writes the footer
  if (file) file->Close();
  return sw.RealTime();
}

}  // namespace hadrons_root
