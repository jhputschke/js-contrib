"""CMS 2507.09307's Z-hadron observables (HEPData ins2945894) from summed histograms: the
HEPData reader (:func:`cms`), the model's Delta phi and Delta y in CMS's convention
(:class:`Model`, the "factorized" result of mix.py) and the Delta y rebinning.

CMS's observable: Delta<N_ch> = S - B, B the mixed-event yield (the Z of one event with the
tracks of another Z event), merged over +/- Delta phi (+/- Delta y) and scaled by 0.5 (the
mean of the +/- bins), per unit angle.  The model's S is its Z-correlated yield, wake
(<bulk_jet> - <bulk_bg>, paired) + jet fragments; the uncorrelated bulk cancels in S - B
and is left out of both.
* Delta phi (all Delta y): B is flat in Delta phi -> S minus its Delta phi average.
* Delta y (|Delta phi| < pi/2): B = half (the Z side) of the y_Z of event i folded with
  the lab-frame eta distribution of event j != i.
Used by mix.py and test/compare.py (the first test), test/parton_cut_check.py."""
import os

import numpy as np
import yaml

PT_TAGS = ["1_<_p_t_<_2", "2_<_p_t_<_4", "4_<_p_t_<_10"]
PT_LABELS = ["1 < p_T^ch < 2 GeV", "2 < p_T^ch < 4 GeV", "4 < p_T^ch < 10 GeV"]


def cms(hep, var, kind, pt_tag):
    """-> bin centres, value, stat, syst of a HEPData table; var 'phi' / 'y', kind
    'pbpb' (0-30 %) / 'pp'."""
    v = "\\delta\\phi_{ch,z}" if var == "phi" else "\\delta_y_{ch,z}"
    k = "in_pbpb_0-30%" if kind == "pbpb" else "spectra_in_pp"
    with open(os.path.join(hep, f"${v}$_{k}_for_${pt_tag}$_gev.yaml")) as f:
        d = yaml.safe_load(f)
    x = d["independent_variables"][0]["values"]
    vals = d["dependent_variables"][0]["values"]
    centre = np.array([0.5 * (b["low"] + b["high"]) for b in x])
    value = np.array([float(b["value"]) for b in vals])
    stat = np.array([float(b["errors"][0]["symerror"]) for b in vals])
    syst = np.array([float(b["errors"][1]["symerror"]) for b in vals])
    return centre, value, stat, syst


class Model:
    """CMS's observables from the summed histograms of a set of events."""

    def __init__(self, deta, dphi, eta):
        self.deta, self.dphi, self.eta = deta, dphi, eta
        self.cdeta = 0.5 * (deta[1:] + deta[:-1])
        self.cphi = 0.5 * (dphi[1:] + dphi[:-1])
        self.ceta = 0.5 * (eta[1:] + eta[:-1])
        # the bin at -dphi (dphi in [-pi/2, 3pi/2))
        neg = (-self.cphi + np.pi / 2) % (2 * np.pi) - np.pi / 2
        self.mirror = np.array([int(np.argmin(np.abs(x - self.cphi))) for x in neg])
        self.zside = np.abs(self.cphi) < np.pi / 2

    def observables(self, H, E, T2, yz, wz):
        """H, E: sum_i w_i x (pT, deta, dphi, eta) yields and their errors; T2: sum_i
        w_i^2 x (pT, deta) yields (all dphi); yz, wz: the events' y_Z and weights.
        -> dphi (value, error) per radian, (pT, 20); dy (value, error) per unit, (pT, 24)."""
        W, W2 = wz.sum(), (wz ** 2).sum()
        # Delta phi
        hp = H.sum(axis=(1, 3)) / W
        ep = np.sqrt((E ** 2).sum(axis=(1, 3))) / W
        hp = 0.5 * (hp + hp[:, self.mirror])
        ep = 0.5 * np.hypot(ep, ep[:, self.mirror])
        width = np.diff(self.dphi)
        dphi = ((hp - hp.mean(axis=1, keepdims=True)) / width, ep / width)
        # Delta y, Z side: same event - half the mixed events
        s = H[:, :, self.zside, :].sum(axis=(2, 3)) / W
        es = np.sqrt((E[:, :, self.zside, :] ** 2).sum(axis=(2, 3))) / W
        rho = H.sum(axis=(1, 2)) / W                       # (pT, eta_lab) per Z
        erho = np.sqrt((E ** 2).sum(axis=(1, 2))) / W
        nb = len(self.cdeta)
        conv = np.zeros_like(s)
        for y, wi in zip(yz, wz):                          # all pairs (i, j)
            j = np.clip(np.searchsorted(self.deta, self.ceta - y) - 1, 0, nb - 1)
            for p in range(len(rho)):
                conv[p] += wi / W * np.bincount(j, rho[p], nb)
        mixed = (W ** 2 * conv - T2) / (W ** 2 - W2)       # pairs i != j
        # its error: rho's errors, shifted by the mean y_Z
        j = np.clip(np.searchsorted(self.deta, self.ceta - np.average(yz, weights=wz)) - 1,
                    0, nb - 1)
        emixed = np.sqrt(np.array([np.bincount(j, e ** 2, nb) for e in erho]))
        dy = s - 0.5 * mixed
        edy = np.hypot(es, 0.5 * emixed)
        dy = 0.5 * (dy + dy[:, ::-1])
        edy = 0.5 * np.hypot(edy, edy[:, ::-1])
        bw = np.diff(self.deta)
        return dphi, (dy / bw, edy / bw)


def rebin_y(h, e, k):
    """|Delta y| < 4 (bins 2..21 of 0.4) into bins of 0.4 k: their mean."""
    h, e = h[..., 2:22], e[..., 2:22]
    shp = h.shape[:-1] + (h.shape[-1] // k, k)
    return h.reshape(shp).mean(-1), np.sqrt((e.reshape(shp) ** 2).sum(-1)) / k
