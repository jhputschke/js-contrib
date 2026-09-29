// example/prod_AuAu_0_10_jet/root_export/read_bench.C
//
// Read timing for bench_formats.py: the same observable -- the number and the summed pT of
// charged hadrons at |eta| < 1 -- over every entry of a hadrons_to_root.py file, reading
// only pid, px, py, pz, three ways:
//
//   root -l -b -q 'read_bench.C+("f.root", "bulk_jet", "ttree")'    TTree, SetBranchAddress
//   root -l -b -q 'read_bench.C+("f.root", "bulk_jet", "rntuple")'  RNTupleReader views
//   root -l -b -q 'read_bench.C+("f.root", "bulk_jet", "rdf")'      RDataFrame (either)
//
// Prints one line: RESULT <mode> <entries> <n selected> <sum pT> <seconds>.  Single thread.

#include <TFile.h>
#include <TStopwatch.h>
#include <TTree.h>

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RVec.hxx>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

inline bool charged(int pid) {
  switch (std::abs(pid)) {
    case 211: case 321: case 2212: case 3222: case 3112: case 3312: case 3334: case 11:
    case 13:
      return true;
    default:
      return false;
  }
}

// adds the selected hadrons of one entry to n and spt
template <class I, class F>
inline void select(long len, const I &pid, const F &px, const F &py, const F &pz,
                   long long &n, double &spt) {
  for (long i = 0; i < len; ++i) {
    if (!charged(pid[i])) continue;
    // in double, as numpy does it in bench_formats.py
    const double pt = std::hypot(static_cast<double>(px[i]), static_cast<double>(py[i]));
    const double eta = std::asinh(static_cast<double>(pz[i]) / pt);
    if (std::fabs(eta) < 1.0) {
      ++n;
      spt += pt;
    }
  }
}

}  // namespace

void read_bench(const char *path, const char *name, const char *mode) {
  TStopwatch sw;
  long long entries = 0, n = 0;
  double spt = 0;
  const std::string m = mode;
  if (m == "ttree") {
    std::unique_ptr<TFile> f(TFile::Open(path));
    auto *t = f->Get<TTree>(name);
    const long nmax = static_cast<long>(t->GetMaximum("n")) + 1;
    int len = 0;
    std::vector<int> pid(nmax);
    std::vector<float> px(nmax), py(nmax), pz(nmax);
    t->SetBranchStatus("*", false);
    for (const char *b : {"n", "pid", "px", "py", "pz"}) t->SetBranchStatus(b, true);
    t->SetBranchAddress("n", &len);
    t->SetBranchAddress("pid", pid.data());
    t->SetBranchAddress("px", px.data());
    t->SetBranchAddress("py", py.data());
    t->SetBranchAddress("pz", pz.data());
    entries = t->GetEntries();
    for (long long e = 0; e < entries; ++e) {
      t->GetEntry(e);
      select(len, pid, px, py, pz, n, spt);
    }
  } else if (m == "rntuple") {
    auto r = ROOT::RNTupleReader::Open(name, path);
    auto vpid = r->GetView<std::vector<int>>("pid");
    auto vpx = r->GetView<std::vector<float>>("px");
    auto vpy = r->GetView<std::vector<float>>("py");
    auto vpz = r->GetView<std::vector<float>>("pz");
    entries = r->GetNEntries();
    for (auto e : r->GetEntryRange()) {
      const auto &pid = vpid(e);
      select(static_cast<long>(pid.size()), pid, vpx(e), vpy(e), vpz(e), n, spt);
    }
  } else if (m == "rdf") {
    ROOT::RDataFrame df(name, path);
    using ROOT::RVecF;
    using ROOT::RVecI;
    auto d = df.Define("sel", [](const RVecI &pid, const RVecF &px, const RVecF &py,
                                 const RVecF &pz) {
                 long long k = 0;
                 double s = 0;
                 select(static_cast<long>(pid.size()), pid, px, py, pz, k, s);
                 return ROOT::RVecD{static_cast<double>(k), s};
               }, {"pid", "px", "py", "pz"})
                 .Define("nsel", "sel[0]")
                 .Define("ptsel", "sel[1]");
    auto cn = d.Sum<double>("nsel");
    auto cs = d.Sum<double>("ptsel");
    auto ce = d.Count();
    n = static_cast<long long>(*cn);
    spt = *cs;
    entries = static_cast<long long>(*ce);
  } else {
    std::fprintf(stderr, "read_bench: unknown mode %s\n", mode);
    return;
  }
  std::printf("RESULT %s %lld %lld %.10e %.3f\n", mode, entries, n, spt, sw.RealTime());
}
