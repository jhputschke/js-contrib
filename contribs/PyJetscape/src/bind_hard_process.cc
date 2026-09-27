/*******************************************************************************
 * bind_hard_process.cc
 *
 * The hard process of the current event, and PythiaGun's pTHat windows:
 *
 *   hard_process_info(task)   pthat, sigma_gen, sigma_err, event_weight of the
 *                             current event (any HardProcess: PythiaGun, PGun, ...)
 *   pythia_gun_bins(task)     PythiaGun's pTHat windows (<pTHatBins>): the bins,
 *                             the current event's window, and per window the
 *                             Pythia seed, sigma_gen, sigma_err, n_accepted
 *   PYTHIA_GUN_HAS_PTHAT_BINS True if X-SCAPE's PythiaGun has <pTHatBins>. An older
 *                             PythiaGun ignores the element and runs pTHatMin-pTHatMax
 *                             only, so a driver that writes it must check this.
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "HardProcess.h"
#include "JetScapeTask.h"
#include "PythiaGun.h"

namespace py = pybind11;
using namespace Jetscape;

namespace {

std::shared_ptr<HardProcess> as_hard(const std::shared_ptr<JetScapeTask> &t) {
  auto h = std::dynamic_pointer_cast<HardProcess>(t);
  if (!h)
    throw std::invalid_argument("not a HardProcess module (PythiaGun, PGun, ...): " +
                                (t ? t->GetId() : std::string("None")));
  return h;
}

}  // namespace

void bind_hard_process(py::module_ &m) {
#ifdef PYTHIAGUN_HAS_PTHAT_BINS
  m.attr("PYTHIA_GUN_HAS_PTHAT_BINS") = true;
#else
  m.attr("PYTHIA_GUN_HAS_PTHAT_BINS") = false;
#endif

  m.def(
      "hard_process_info",
      [](std::shared_ptr<JetScapeTask> task) {
        auto h = as_hard(task);
        py::dict d;
        d["pthat"] = h->GetPtHat();
        d["sigma_gen"] = h->GetSigmaGen();
        d["sigma_err"] = h->GetSigmaErr();
        d["event_weight"] = h->GetEventWeight();
        return d;
      },
      R"pbdoc(
        The current event's hard process: ``pthat`` [GeV], ``sigma_gen`` and
        ``sigma_err`` (the generator's running cross-section estimate, mb; for
        PythiaGun with several pTHat windows, the current event's window) and
        ``event_weight``.  Call between ExecPerEvent() and ClearPerEvent().
      )pbdoc",
      py::arg("task"));

  m.def(
      "pythia_gun_bins",
      [](std::shared_ptr<JetScapeTask> task) {
        auto gun = std::dynamic_pointer_cast<PythiaGun>(task);
        if (!gun)
          throw std::invalid_argument("not a PythiaGun: " +
                                      (task ? task->GetId() : std::string("None")));
#ifdef PYTHIAGUN_HAS_PTHAT_BINS
        const int n = gun->GetNPtHatBins();
        py::list bins, seeds, sigma, err, acc;
        for (int k = 0; k < n; ++k) {
          bins.append(py::make_tuple(gun->GetPtHatBinMin(k), gun->GetPtHatBinMax(k)));
          seeds.append(gun->GetPythiaSeed(k));
          sigma.append(gun->GetSigmaGen(k));
          err.append(gun->GetSigmaErr(k));
          acc.append(gun->GetNAccepted(k));
        }
        py::dict d;
        d["n_bins"] = n;
        d["active"] = gun->GetActivePtHatBin();
        d["bins"] = bins;
        d["seeds"] = seeds;
        d["sigma_gen"] = sigma;
        d["sigma_err"] = err;
        d["n_accepted"] = acc;
        return d;
#else
        throw std::runtime_error(
            "pythia_gun_bins: X-SCAPE's PythiaGun has no <pTHatBins> (branch "
            "N_ptHat_per_hydro or later needed)");
#endif
      },
      R"pbdoc(
        PythiaGun's pTHat windows (<Hard><PythiaGun><pTHatBins>; one window,
        pTHatMin-pTHatMax, without it): ``n_bins``, ``bins`` [(min, max)],
        ``active`` (the current event's window: event i uses i mod n_bins), and
        per window ``seeds`` (Pythia's Random:seed), ``sigma_gen``, ``sigma_err``
        [mb] and ``n_accepted`` so far.  After Init(): the windows and seeds; after
        the last event: the final cross sections.
      )pbdoc",
      py::arg("task"));
}
