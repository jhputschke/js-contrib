/*******************************************************************************
 * bind_hadronization.cc
 *
 * Hadrons as numpy, and re-hadronizing stored inputs:
 *
 *   soft_hadrons_numpy(task)            SoftParticlization (iSS) output, all
 *                                       oversamples, with per-sample counts
 *   soft_set_next_random_seed(task, s)  one-shot iSS seed (exact re-sampling)
 *   soft_last_random_seed(task)         the seed the last event used
 *   soft_set_number_of_samples(task, n) oversamples per event from the next event on
 *   soft_set_compact_output(task, on)   hand iSS's hadrons over as flat arrays, not one
 *                                       Hadron object each (~8x less memory)
 *   hadronization_hadrons_numpy(task)   a HadronizationManager's / Hadronization's
 *                                       output hadrons (jet hadronization in a job)
 *   hadronize_partons(module, partons, seed=None)
 *                                       run a jet hadronization module
 *                                       (ColorlessHadronization, ColoredHadronization)
 *                                       on stored final partons (FINAL_PARTON_COLUMNS)
 *   jet_hadronization_last_random_seed(module)
 *   soft_primaries_numpy(task)          iSS <keep_primaries> N: the undecayed hadrons of
 *                                       the first N oversamples (None if not kept)
 *   hadronization_input_numpy(task)     iColoredHadronization <store_input> 1: the parton
 *                                       list the last event handed to Pythia
 *   hadronize_input(module, input, seed=None)
 *                                       hadronize such a stored list again
 *                                       (iColoredHadronization)
 *   soft_sample_cells(task, cells, seed, n_samples)
 *                                       iSS on given surface cells (SURFACE_CELL_COLUMNS)
 *                                       outside the event loop -> soft_hadrons_numpy's
 *                                       dict; the primaries then via soft_primaries_numpy
 *
 * Capabilities: ISS_DECAYS_SMASH_LIST, SOFT_HAS_PRIMARIES, ICOLORED_HAS_INPUT_RECORD,
 * SOFT_HAS_SAMPLE_CELLS (False when this X-SCAPE predates them), PYTHIA_VERSION (e.g.
 * 8312).
 *
 * All are free functions taking a task: create_module() and GetTaskList() hand out
 * JetScapeTask / JetScapeModuleBase pointers whose dynamic types (iSpectraSamplerWrapper,
 * ColorlessHadronization ...) are not registered with pybind11, so methods on bound
 * classes would never be reached.  The dynamic_pointer_cast happens here instead.
 *
 * Hadron arrays: pid int32 (N,), pstat int32 (N,), p float32 (N, 4) [E, px, py, pz]
 * GeV, x float32 (N, 4) [t, x, y, z] fm, mass float32 (N,) GeV.
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "JetScapeTask.h"
#include "JetScapeParticles.h"
#include "SoftParticlization.h"
#include "surface_rows.h"
#include "Hadronization.h"
#include "HadronizationManager.h"
#include "ColorlessHadronization.h"
#if __has_include("iColoredHadronization.h")
#include "iColoredHadronization.h"
#endif

namespace py = pybind11;
using namespace Jetscape;

namespace {

// FINAL_PARTON_COLUMNS (bind_jet.cc): shower pid pstat E px py pz t x y z mass col acol
constexpr int kColShower = 0, kColPid = 1, kColPstat = 2, kColE = 3, kColPx = 4,
              kColPy = 5, kColPz = 6, kColT = 7, kColX = 8, kColY = 9, kColZ = 10,
              kColCol = 12, kColAcol = 13, kFinalPartonNCols = 14;

py::dict hadrons_to_dict(const std::vector<std::shared_ptr<Hadron>> &h) {
  const py::ssize_t n = static_cast<py::ssize_t>(h.size());
  py::array_t<int> pid(n), pstat(n);
  py::array_t<float> p({n, (py::ssize_t)4}), x({n, (py::ssize_t)4}), mass(n);
  auto a_pid = pid.mutable_unchecked<1>();
  auto a_st = pstat.mutable_unchecked<1>();
  auto a_p = p.mutable_unchecked<2>();
  auto a_x = x.mutable_unchecked<2>();
  auto a_m = mass.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i) {
    const auto &hi = h[i];
    a_pid(i) = hi->pid();
    a_st(i) = hi->pstat();
    a_p(i, 0) = hi->e();
    a_p(i, 1) = hi->px();
    a_p(i, 2) = hi->py();
    a_p(i, 3) = hi->pz();
    a_x(i, 0) = hi->x_in().t();
    a_x(i, 1) = hi->x_in().x();
    a_x(i, 2) = hi->x_in().y();
    a_x(i, 3) = hi->x_in().z();
    a_m(i) = hi->restmass();
  }
  py::dict d;
  d["pid"] = pid;
  d["pstat"] = pstat;
  d["p"] = p;
  d["x"] = x;
  d["mass"] = mass;
  return d;
}

// A numpy array that takes over the vector's memory (no copy); numpy frees it.
template <typename T>
py::array_t<T> vector_to_numpy(std::vector<T> &&v, std::vector<py::ssize_t> shape) {
  if (v.empty())
    return py::array_t<T>(shape);
  auto *owner = new std::vector<T>(std::move(v));
  py::capsule free_when_done(
      owner, [](void *o) { delete static_cast<std::vector<T> *>(o); });
  return py::array_t<T>(shape, owner->data(), free_when_done);
}

std::shared_ptr<SoftParticlization> as_soft(const std::shared_ptr<JetScapeTask> &t) {
  auto s = std::dynamic_pointer_cast<SoftParticlization>(t);
  if (!s)
    throw std::invalid_argument("not a SoftParticlization module (e.g. iSS): " +
                                (t ? t->GetId() : std::string("None")));
  return s;
}

std::shared_ptr<Hadronization> as_hadronization(const std::shared_ptr<JetScapeTask> &t) {
  auto h = std::dynamic_pointer_cast<Hadronization>(t);
  if (!h)
    throw std::invalid_argument(
        "not a jet hadronization module (ColorlessHadronization, ...): " +
        (t ? t->GetId() : std::string("None")));
  return h;
}

// A module of type T: `module` itself, else one of its sub-tasks or theirs (a
// Hadronization parent or the HadronizationManager).
template <typename T>
std::shared_ptr<T> find_module(const std::shared_ptr<JetScapeTask> &module) {
  if (auto t = std::dynamic_pointer_cast<T>(module))
    return t;
  for (auto &sub : module->GetTaskList()) {
    if (auto t = std::dynamic_pointer_cast<T>(sub))
      return t;
    for (auto &subsub : sub->GetTaskList())
      if (auto t = std::dynamic_pointer_cast<T>(subsub))
        return t;
  }
  return nullptr;
}

py::dict hadron_arrays_to_dict(SoftParticlization::HadronArrays &a) {
  const py::ssize_t n = static_cast<py::ssize_t>(a.pid.size());
  py::dict d;
  d["pid"] = vector_to_numpy(std::move(a.pid), {n});
  d["pstat"] = vector_to_numpy(std::move(a.pstat), {n});
  d["p"] = vector_to_numpy(std::move(a.p), {n, 4});
  d["x"] = vector_to_numpy(std::move(a.x), {n, 4});
  d["mass"] = vector_to_numpy(std::move(a.mass), {n});
  const py::ssize_t ns = static_cast<py::ssize_t>(a.sample_counts.size());
  d["sample_counts"] = vector_to_numpy(std::move(a.sample_counts), {ns});
  return d;
}

}  // namespace

void bind_hadronization(py::module_ &m) {
#ifdef ISS_DECAYS_SMASH_LIST
  m.attr("ISS_DECAYS_SMASH_LIST") = true;
#else
  m.attr("ISS_DECAYS_SMASH_LIST") = false;
#endif
#ifdef SOFTPARTICLIZATION_HAS_PRIMARIES
  m.attr("SOFT_HAS_PRIMARIES") = true;
#else
  m.attr("SOFT_HAS_PRIMARIES") = false;
#endif
#ifdef SOFTPARTICLIZATION_HAS_SAMPLE_CELLS
  m.attr("SOFT_HAS_SAMPLE_CELLS") = true;
#else
  m.attr("SOFT_HAS_SAMPLE_CELLS") = false;
#endif
#ifdef ICOLORED_HAS_INPUT_RECORD
  m.attr("ICOLORED_HAS_INPUT_RECORD") = true;
#else
  m.attr("ICOLORED_HAS_INPUT_RECORD") = false;
#endif
#ifdef PYTHIA_VERSION_INTEGER
  m.attr("PYTHIA_VERSION") = static_cast<int>(PYTHIA_VERSION_INTEGER);
#else
  m.attr("PYTHIA_VERSION") = 0;
#endif

  m.def(
      "soft_hadrons_numpy",
      [](std::shared_ptr<JetScapeTask> task) {
        auto soft = as_soft(task);
        if (soft->HasCompactHadrons()) {      // soft_set_compact_output
          SoftParticlization::HadronArrays a;
          if (!soft->TakeCompactHadrons(a))
            throw std::runtime_error(
                "soft_hadrons_numpy: this event's hadrons were already handed over "
                "(with soft_set_compact_output, call it once per event)");
          return hadron_arrays_to_dict(a);
        }
        std::vector<std::shared_ptr<Hadron>> flat;
        const auto &lists = soft->Hadron_list_;
        py::array_t<long long> counts((py::ssize_t)lists.size());
        auto a_c = counts.mutable_unchecked<1>();
        for (size_t s = 0; s < lists.size(); ++s) {
          a_c(s) = static_cast<long long>(lists[s].size());
          flat.insert(flat.end(), lists[s].begin(), lists[s].end());
        }
        py::dict d = hadrons_to_dict(flat);
        d["sample_counts"] = counts;
        return d;
      },
      "All hadrons a SoftParticlization module (iSS) produced this event: dict of pid, "
      "pstat, p [E,px,py,pz], x [t,x,y,z], mass, and sample_counts (one entry per "
      "oversample; the rows are the samples concatenated in order). Call between "
      "ExecPerEvent() and ClearPerEvent().",
      py::arg("task"));

  m.def(
      "soft_primaries_numpy",
      [](std::shared_ptr<JetScapeTask> task) -> py::object {
#ifdef SOFTPARTICLIZATION_HAS_PRIMARIES
        auto soft = as_soft(task);
        SoftParticlization::HadronArrays a;
        if (!soft->TakeCompactPrimaries(a))
          return py::none();
        return hadron_arrays_to_dict(a);
#else
        throw std::runtime_error("soft_primaries_numpy: this X-SCAPE has no primaries "
                                 "(SOFT_HAS_PRIMARIES is False)");
#endif
      },
      "The undecayed (primary) hadrons of the first N oversamples this event, as sampled "
      "before any decay (iSS <keep_primaries> N, with soft_set_compact_output): the "
      "soft_hadrons_numpy dict, sample_counts with N entries. None if none were kept. "
      "Takes them over without copying: call it once per event, after "
      "soft_hadrons_numpy.",
      py::arg("task"));

  m.def(
      "soft_sample_cells",
      [](std::shared_ptr<JetScapeTask> task,
         py::array_t<float, py::array::c_style | py::array::forcecast> cells, long seed,
         int n_samples) {
#ifdef SOFTPARTICLIZATION_HAS_SAMPLE_CELLS
        auto soft = as_soft(task);
        if (cells.ndim() != 2 || cells.shape(1) != kSurfaceNCols)
          throw std::invalid_argument(
              "soft_sample_cells: need an (N, 32) array, columns SURFACE_CELL_COLUMNS");
        auto a = cells.unchecked<2>();
        std::vector<SurfaceCellInfo> v(static_cast<size_t>(a.shape(0)));
        for (py::ssize_t i = 0; i < a.shape(0); ++i)
          surface_cell_from_row(&a(i, 0), v[i]);    // fields iSS ignores stay zero
        bool ok;
        {
          py::gil_scoped_release release;
          ok = soft->SampleCells(v, seed, n_samples);
        }
        if (!ok)
          throw std::invalid_argument(
              "soft_sample_cells: this module cannot sample given cells (or n_samples < 1, "
              "or it is not initialised yet)");
        SoftParticlization::HadronArrays h;
        soft->TakeCompactHadrons(h);
        return hadron_arrays_to_dict(h);
#else
        throw std::runtime_error("soft_sample_cells: this X-SCAPE has no SampleCells "
                                 "(SOFT_HAS_SAMPLE_CELLS is False)");
        return py::dict();
#endif
      },
      "Sample the given surface cells now, outside the framework's event loop: an (N, 32) "
      "float32 array (SURFACE_CELL_COLUMNS, e.g. a particlize file's surface unit), this "
      "seed and n_samples oversamples. Returns soft_hadrons_numpy's dict (one entry of "
      "sample_counts per oversample; none for no cells); with iSS <keep_primaries> the "
      "undecayed hadrons follow from soft_primaries_numpy. Equal to replaying the cells "
      "through the event loop with soft_set_next_random_seed. The module must be "
      "initialised (Init); it need not be in a task list.",
      py::arg("task"), py::arg("cells"), py::arg("seed"), py::arg("n_samples"));

  m.def(
      "soft_set_next_random_seed",
      [](std::shared_ptr<JetScapeTask> task, long seed) {
        as_soft(task)->SetNextRandomSeed(seed);
      },
      "Use `seed` for the next event's sampling (one-shot) instead of a draw from the "
      "module's generator.",
      py::arg("task"), py::arg("seed"));

  m.def(
      "soft_set_number_of_samples",
      [](std::shared_ptr<JetScapeTask> task, int n) {
        if (!as_soft(task)->SetNumberOfSamples(n))
          throw std::invalid_argument(
              "soft_set_number_of_samples: this module has no such setting (or n < 1, "
              "or it is not initialised yet)");
      },
      "Samples (oversamples) per event from the next event on (iSS: "
      "number_of_repeated_sampling). Call after Init().",
      py::arg("task"), py::arg("n"));

  m.def(
      "soft_set_compact_output",
      [](std::shared_ptr<JetScapeTask> task, bool on) {
        if (!as_soft(task)->SetCompactHadronOutput(on))
          throw std::invalid_argument(
              "soft_set_compact_output: this module cannot hand its hadrons over as "
              "arrays");
      },
      "From the next event on, hand the hadrons over as flat arrays instead of one "
      "Hadron object each: the same soft_hadrons_numpy result with ~8x less memory. "
      "soft_hadrons_numpy then takes the arrays over without copying, so call it once per "
      "event. The framework's Hadron_list_ stays empty (no bulk hadrons for writers or "
      "afterburners). iSS: ExecuteTask only.",
      py::arg("task"), py::arg("on") = true);

  m.def(
      "soft_last_random_seed",
      [](std::shared_ptr<JetScapeTask> task) { return as_soft(task)->GetLastRandomSeed(); },
      "The seed the SoftParticlization module used for the last event.", py::arg("task"));

  m.def(
      "hadronization_hadrons_numpy",
      [](std::shared_ptr<JetScapeTask> task) {
        std::vector<std::shared_ptr<Hadron>> all;
        auto collect = [&all](const std::shared_ptr<JetScapeTask> &t) {
          auto h = std::dynamic_pointer_cast<Hadronization>(t);
          if (h) {
            auto v = h->GetHadrons();
            all.insert(all.end(), v.begin(), v.end());
          }
        };
        if (std::dynamic_pointer_cast<HadronizationManager>(task)) {
          for (auto &t : task->GetTaskList())
            collect(t);
        } else {
          as_hadronization(task);
          collect(task);
        }
        return hadrons_to_dict(all);
      },
      "Output hadrons of a HadronizationManager (all its Hadronization tasks) or of one "
      "Hadronization task, as a dict of arrays. Call between ExecPerEvent() and "
      "ClearPerEvent().",
      py::arg("task"));

  m.def(
      "hadronize_partons",
      [](std::shared_ptr<JetScapeTask> module,
         py::array_t<double, py::array::c_style | py::array::forcecast> partons,
         py::object seed) {
        auto had = as_hadronization(module);
        if (partons.ndim() != 2 || partons.shape(1) < kFinalPartonNCols)
          throw std::invalid_argument(
              "hadronize_partons: need an (N, 14) array, columns FINAL_PARTON_COLUMNS");
        auto a = partons.unchecked<2>();

        // Group by shower index, keeping the stored order within and across showers.
        std::map<long, std::vector<std::shared_ptr<Parton>>> by_shower;
        for (py::ssize_t i = 0; i < a.shape(0); ++i) {
          FourVector p(a(i, kColPx), a(i, kColPy), a(i, kColPz), a(i, kColE));
          FourVector x(a(i, kColX), a(i, kColY), a(i, kColZ), a(i, kColT));
          auto parton = std::make_shared<Parton>(0, (int)a(i, kColPid),
                                                 (int)a(i, kColPstat), p, x);
          parton->set_color((unsigned int)a(i, kColCol));
          parton->set_anti_color((unsigned int)a(i, kColAcol));
          by_shower[(long)a(i, kColShower)].push_back(parton);
        }
        std::vector<std::vector<std::shared_ptr<Parton>>> shower;
        for (auto &kv : by_shower)
          shower.push_back(kv.second);

        auto colorless = std::dynamic_pointer_cast<ColorlessHadronization>(module);
        if (!seed.is_none()) {
          if (!colorless)
            throw std::invalid_argument(
                "hadronize_partons: seed is only supported for ColorlessHadronization");
          colorless->SetNextRandomSeed(seed.cast<unsigned int>());
        }
        std::vector<std::shared_ptr<Hadron>> hOut;
        std::vector<std::shared_ptr<Parton>> pOut;
        had->DoHadronization(shower, hOut, pOut);
        py::dict d = hadrons_to_dict(hOut);
        d["n_partons_out"] = static_cast<long long>(pOut.size());
        d["seed"] = colorless ? (long long)colorless->GetLastRandomSeed() : 0LL;
        return d;
      },
      "Hadronize stored final partons (an (N, 14) array, FINAL_PARTON_COLUMNS) with a jet "
      "hadronization module, e.g. create_module('ColorlessHadronization') after a JetScape "
      "Init() that loaded the XML and after module.Init(). With `seed`, the "
      "ColorlessHadronization generators are reseeded with it first (exact replay of an "
      "event hadronized with <reseed_per_event> 1). Returns the hadron dict plus "
      "n_partons_out and the seed used.",
      py::arg("module"), py::arg("partons"), py::arg("seed") = py::none());

  m.def(
      "jet_hadronization_last_random_seed",
      [](std::shared_ptr<JetScapeTask> module) {
        if (auto c = find_module<ColorlessHadronization>(module))
          return c->GetLastRandomSeed();
#ifdef ICOLORED_HAS_INPUT_RECORD
        if (auto c = find_module<iColoredHadronization>(module))
          return c->GetLastRandomSeed();
#endif
        throw std::invalid_argument(
            "no ColorlessHadronization (or iColoredHadronization) found in " +
            module->GetId());
      },
      "The seed ColorlessHadronization (or iColoredHadronization) used for the last event "
      "(0 if it was not reseeded). Accepts the module itself, its Hadronization parent or "
      "the HadronizationManager.",
      py::arg("module"));

#ifdef ICOLORED_HAS_INPUT_RECORD
  m.def(
      "hadronization_input_numpy",
      [](std::shared_ptr<JetScapeTask> module) {
        auto c = find_module<iColoredHadronization>(module);
        if (!c)
          throw std::invalid_argument("no iColoredHadronization found in " +
                                      module->GetId());
        const auto &in = c->GetLastInput();
        const py::ssize_t n = static_cast<py::ssize_t>(in.size());
        py::array_t<int> pid(n), col(n), acol(n);
        py::array_t<double> p({n, (py::ssize_t)4}), mass(n);
        auto a_id = pid.mutable_unchecked<1>();
        auto a_c = col.mutable_unchecked<1>();
        auto a_a = acol.mutable_unchecked<1>();
        auto a_p = p.mutable_unchecked<2>();
        auto a_m = mass.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < n; ++i) {
          a_id(i) = in[i].id;
          a_c(i) = in[i].col;
          a_a(i) = in[i].acol;
          a_p(i, 0) = in[i].e;
          a_p(i, 1) = in[i].px;
          a_p(i, 2) = in[i].py;
          a_p(i, 3) = in[i].pz;
          a_m(i) = in[i].m;
        }
        py::dict d;
        d["pid"] = pid;
        d["col"] = col;
        d["acol"] = acol;
        d["p"] = p;
        d["mass"] = mass;
        return d;
      },
      "iColoredHadronization with <store_input> 1: the parton list the last event handed "
      "to Pythia (final partons, beam remnants, the colour-balancing parton), as a dict of "
      "pid, col, acol (int32), p (N, 4) float64 [E, px, py, pz] and mass float64. Empty "
      "without <store_input>. Accepts the module, its parent or the HadronizationManager.",
      py::arg("module"));

  m.def(
      "hadronize_input",
      [](std::shared_ptr<JetScapeTask> module, py::array_t<int> pid, py::array_t<int> col,
         py::array_t<int> acol,
         py::array_t<double, py::array::c_style | py::array::forcecast> p,
         py::array_t<double, py::array::c_style | py::array::forcecast> mass,
         py::object seed) {
        auto c = find_module<iColoredHadronization>(module);
        if (!c)
          throw std::invalid_argument("hadronize_input: no iColoredHadronization in " +
                                      module->GetId());
        const py::ssize_t n = pid.shape(0);
        if (col.shape(0) != n || acol.shape(0) != n || mass.shape(0) != n ||
            p.ndim() != 2 || p.shape(0) != n || p.shape(1) != 4)
          throw std::invalid_argument(
              "hadronize_input: pid, col, acol, mass (N,) and p (N, 4) [E, px, py, pz]");
        auto a_id = pid.unchecked<1>();
        auto a_c = col.unchecked<1>();
        auto a_a = acol.unchecked<1>();
        auto a_p = p.unchecked<2>();
        auto a_m = mass.unchecked<1>();
        std::vector<iColoredHadronization::InputParton> in(n);
        for (py::ssize_t i = 0; i < n; ++i)
          in[i] = {a_id(i), a_c(i), a_a(i), a_p(i, 1), a_p(i, 2), a_p(i, 3), a_p(i, 0),
                   a_m(i)};
        if (!seed.is_none())
          c->SetNextRandomSeed(seed.cast<unsigned int>());
        std::vector<std::shared_ptr<Hadron>> hOut;
        c->HadronizeInput(in, hOut);
        py::dict d = hadrons_to_dict(hOut);
        d["seed"] = (long long)c->GetLastRandomSeed();
        return d;
      },
      "Hadronize a parton list stored with hadronization_input_numpy again with "
      "iColoredHadronization (create_module('iColoredHadronization') after a JetScape "
      "Init() that loaded the job's XML, then module.Init()). With the stored `seed` and "
      "the same Pythia version and <JetHadronization> settings the hadrons are those of "
      "the job. Returns the hadron dict plus the seed used.",
      py::arg("module"), py::arg("pid"), py::arg("col"), py::arg("acol"), py::arg("p"),
      py::arg("mass"), py::arg("seed") = py::none());
#endif
}
