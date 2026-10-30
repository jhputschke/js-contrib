"""Jets on hadron lists with FastJet, the Python twin of analysis_root/common/JetMap.h: anti-k_T
(R = 0.3 by default, ``Clusterer(fj, R)``), E scheme; Pb+Pb with explicit ghosts on a fixed
grid (|y| < 2.6, area 0.01) and
rho(jet) from the k_T R = 0.4 median in a strip |y - y_jet| < 1 (|y| < 2.2), p_T = p_T,raw -
rho A; per jet also its mass, the number of its (non-ghost) constituents and the leading
one's p_T, and with ``Clusterer(..., rho_m=True)`` rho_m of its strip (the median of the k_T
jets' (m_T - p_T) / area, for mass subtraction; off by default: it costs a quarter of the
time).  Nothing here knows the trigger: the caller chooses the particles."""
import math

import numpy as np

R_JET, ETA_JET = 0.3, 2.0     # R_JET: the default anti-k_T radius (Clusterer's R)
GHOST_AREA = 0.01            # the fixed ghost grid's cell (kept at 0.01, PLAN_dijet_deta_wake D8)
ALL = -math.inf      # Clusterer.jets' ptmin that keeps every jet
JET_NAMES = ("pt_raw", "pt", "eta", "phi", "area", "rho", "m", "nconst", "lead_pt")
JET_NAMES_RHO_M = JET_NAMES + ("rho_m",)     # with Clusterer(..., rho_m=True)
STRIP = 1.0          # the rho strip's half width in rapidity (Clusterer.rho_sel)


class StripRho:
    """rho(jet) of JetMedianBackgroundEstimator(SelectorAbsRapMax(2.2) * SelectorStrip(1.0)),
    computed once per strip content.  With explicit ghosts the median depends only on which
    k_T jets lie in the strip |y - y_jet| < 1, and that set changes only where y_jet +- 1
    crosses a k_T jet's rapidity: the jets in between share one value, so each distinct
    content (the number of k_T jets below y_jet - 1 and below y_jet + 1) is computed once.
    A k_T jet within 1e-9 of a strip edge (where the selector's comparison decides) is never
    cached.  The same values as the estimator's, bit for bit."""

    def __init__(self, fj, cs, sel, rho_m=False):
        self.cs = cs                             # the estimator's jets refer to it
        jets = cs.inclusive_jets()
        self.b = fj.JetMedianBackgroundEstimator(sel)
        # rho_m only on request: FastJet computes it by default, from every k_T jet's
        # constituents (a quarter of the map step's time); rho is the median of p_T / area
        # either way, the same with and without
        self.b.set_compute_rho_m(bool(rho_m))
        self.b.set_jets(jets)
        self.ys = np.sort(np.array([j.rap() for j in jets]))
        self.cache, self.cache_m = {}, {}

    def _key(self, jet):
        """The strip content of a jet, or None at a strip edge (never cached)."""
        y, ys = jet.rap(), self.ys
        key = []
        for edge in (y - STRIP, y + STRIP):
            i = int(np.searchsorted(ys, edge))
            if (i < len(ys) and abs(ys[i] - edge) < 1e-9) or (i and abs(ys[i - 1] - edge) < 1e-9):
                return None
            key.append(i)
        return tuple(key)

    def rho(self, jet):
        key = self._key(jet)
        if key is None:
            return self.b.rho(jet)
        if key not in self.cache:
            self.cache[key] = self.b.rho(jet)
        return self.cache[key]

    def rho_m(self, jet):
        """rho_m of the jet's strip (needs StripRho(..., rho_m=True))."""
        key = self._key(jet)
        if key is None:
            return self.b.rho_m(jet)
        if key not in self.cache_m:
            self.cache_m[key] = self.b.rho_m(jet)
        return self.cache_m[key]


def rap_phi(ev, keep):
    """Rapidity and phi of the kept particles, as FastJet computes them (for preselecting the
    random cones' particles; the cone sum itself uses PseudoJet.delta_R)."""
    p = np.asarray(ev["p"], np.float64)[keep]
    E, px, py, pz = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    kt2 = px * px + py * py
    m2 = np.maximum(E * E - kt2 - pz * pz, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = 0.5 * np.log((kt2 + m2) / (E + np.abs(pz)) ** 2)
    y = np.where(pz > 0, -y, y)
    return np.nan_to_num(y, nan=0.0, posinf=1e5, neginf=-1e5), np.arctan2(py, px)


class Clusterer:
    """anti-k_T jets of radius ``R`` (the rho estimate's k_T R = 0.4 does not change with it)."""

    def __init__(self, fj, R=R_JET, rho_m=False):
        self.fj = fj
        self.R = float(R)
        self.rho_m = bool(rho_m)
        self.names = JET_NAMES_RHO_M if self.rho_m else JET_NAMES
        self.akt = fj.JetDefinition(fj.antikt_algorithm, self.R)
        self.kt = fj.JetDefinition(fj.kt_algorithm, 0.4)
        self.area = fj.AreaDefinition(fj.active_area_explicit_ghosts,
                                      fj.GhostedAreaSpec(2.6, 1, GHOST_AREA, 0.0, 0.0))
        self.rho_sel = fj.SelectorAbsRapMax(2.2) * fj.SelectorStrip(1.0)

    def bge(self, parts):
        """The k_T jets' rho estimator of a particle list (StripRho: rho(jet) as
        JetMedianBackgroundEstimator's, computed once per strip content)."""
        return StripRho(self.fj, self.fj.ClusterSequenceArea(parts, self.kt, self.area),
                        self.rho_sel, self.rho_m)

    def _jet(self, j, pt, a, rho, ghosts, rho_m=0.0):
        n, lead = 0, 0.0
        for q in j.constituents():
            if not ghosts or not q.is_pure_ghost():
                n += 1
                lead = max(lead, q.pt())
        out = (j.pt(), pt, j.eta(), j.phi_std(), a, rho, j.m(), float(n), lead)
        return out + (rho_m,) if self.rho_m else out

    def jets(self, parts, subtract, ptmin, bge=None):
        """-> list of (pt_raw, pt, eta, phi, area, rho, m, nconst, lead_pt[, rho_m]) (the
        fields of ``names``), as zjets.cxx's jets_of: p_T,raw > 1 GeV, |eta| < 2, p_T >
        ptmin (ALL: no cut).  ``bge``: the particles' StripRho if the caller has it
        (``bge(parts)``), else made here."""
        fj = self.fj
        out = []
        if not subtract:
            cs = fj.ClusterSequence(parts, self.akt)
            for j in fj.sorted_by_pt(cs.inclusive_jets(1.0)):
                if abs(j.eta()) < ETA_JET and j.pt() >= ptmin:
                    out.append(self._jet(j, j.pt(), 0.0, 0.0, False))
            return out
        b = self.bge(parts) if bge is None else bge
        cs = fj.ClusterSequenceArea(parts, self.akt, self.area)
        for j in fj.sorted_by_pt(cs.inclusive_jets(1.0)):
            if not abs(j.eta()) < ETA_JET:
                continue
            if j.pt() <= ptmin:                  # rho >= 0: p_T <= p_T,raw cannot pass
                continue
            rho, a = b.rho(j), j.area()
            pt = j.pt() - rho * a
            if pt > ptmin:
                out.append(self._jet(j, pt, a, rho, True, b.rho_m(j) if self.rho_m else 0.0))
        return out


def cone(fj, parts, y, phi, rho_est, eta, phi0, R):
    """A cone of radius R around (eta, phi0) (taken as a rapidity): the p_T sum of ``parts``
    (their rapidities ``y`` and phis ``phi``, from rap_phi) within Delta R < R, rho there
    (``rho_est``'s, a StripRho; None: 0) and sum - rho pi R^2 -> (axis, sum, rho, delta p_T).
    The particles are preselected in numpy (0.01 margin), then FastJet's delta_R in their
    order: the same particles and the same sum as over all of them (the random cones, the Z
    cone; zjets.cxx's cone_sum)."""
    axis = fj.PseudoJet()
    axis.reset_PtYPhiM(1.0, eta, phi0, 0.0)
    dphi = np.abs(phi - axis.phi())
    dphi = np.minimum(dphi, 2 * np.pi - dphi)
    near = np.flatnonzero((y - axis.rap()) ** 2 + dphi ** 2 < (R + 0.01) ** 2)
    tot = sum(parts[i].pt() for i in near if parts[i].delta_R(axis) < R)
    rho = rho_est.rho(axis) if rho_est is not None else 0.0
    return axis, tot, rho, tot - rho * math.pi * R ** 2
