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
 *   hard_partons_numpy(task)  the partons the hard process handed to the framework
 *                             this event: pid, px, py, pz, E, pT, y (by pT, hardest
 *                             first) -- for PythiaGun what <partonYMax> cuts on
 *   PYTHIA_GUN_HAS_PARTON_Y_CUT True if it has <partonYMax> / <partonYMode>; then
 *                             pythia_gun_bins also gives the cut, per window the events
 *                             tried / kept, the acceptance and Pythia's raw sigma.
 ******************************************************************************/

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "HardProcess.h"
#include "JetScapeParticles.h"
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
#ifdef PYTHIAGUN_HAS_PARTON_Y_CUT
  m.attr("PYTHIA_GUN_HAS_PARTON_Y_CUT") = true;
#else
  m.attr("PYTHIA_GUN_HAS_PARTON_Y_CUT") = false;
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
#ifdef PYTHIAGUN_HAS_PARTON_Y_CUT
        py::list tried, kept, accept, raw, raw_err;
        for (int k = 0; k < n; ++k) {
          tried.append(gun->GetNYTried(k));
          kept.append(gun->GetNYKept(k));
          accept.append(gun->GetYAcceptance(k));
          raw.append(gun->GetSigmaGenRaw(k));
          raw_err.append(gun->GetSigmaErrRaw(k));
        }
        d["y_max"] = gun->GetPartonYMax();       // 0: no cut
        d["y_mode"] = gun->GetPartonYMode();
        d["n_tried"] = tried;                    // events that reached the cut
        d["n_kept"] = kept;                      // and passed it
        d["acceptance"] = accept;                // kept / tried (1 without a cut)
        d["sigma_gen_raw"] = raw;                // Pythia's, incl. the rejected events
        d["sigma_err_raw"] = raw_err;
#endif
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
        the last event: the final cross sections.  With PYTHIA_GUN_HAS_PARTON_Y_CUT
        also ``y_max`` (0 = no cut), ``y_mode`` and per window ``n_tried`` /
        ``n_kept`` (events reaching / passing the cut), ``acceptance`` and
        ``sigma_gen_raw`` / ``sigma_err_raw`` (Pythia's own, which counts the rejected
        events; ``sigma_gen`` is it times the acceptance).
      )pbdoc",
      py::arg("task"));

  m.def(
      "hard_partons_numpy",
      [](std::shared_ptr<JetScapeTask> task) {
        auto h = as_hard(task);
        std::vector<std::array<double, 7>> rows;
        for (int i = 0; i < h->GetNHardPartons(); ++i) {
          auto p = h->GetPartonAt(i);
          const auto q = p->p_in();
          const double pt = std::hypot(q.x(), q.y());
          const double y = 0.5 * std::log((q.t() + q.z()) / (q.t() - q.z()));
          rows.push_back({double(p->pid()), q.x(), q.y(), q.z(), q.t(), pt, y});
        }
        std::stable_sort(rows.begin(), rows.end(),
                         [](const auto &a, const auto &b) { return a[5] > b[5]; });
        py::array_t<double> out({py::ssize_t(rows.size()), py::ssize_t(7)});
        auto r = out.mutable_unchecked<2>();
        for (size_t i = 0; i < rows.size(); ++i)
          for (int j = 0; j < 7; ++j)
            r(i, j) = rows[i][j];
        return out;
      },
      R"pbdoc(
        The partons the hard process handed to the framework this event (for
        PythiaGun: status 62 after ISR/MPI, or the final partons with FSR_on),
        as an (n, 7) array pid, px, py, pz, E, pT, y, hardest first.  For
        PythiaGun this is what <partonYMax> cuts on.  Call between
        ExecPerEvent() and ClearPerEvent().
      )pbdoc",
      py::arg("task"));
}
