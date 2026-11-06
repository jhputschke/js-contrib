"""
contribs/Visualization/full_event_pyvista.py

One heavy-ion event with a jet, followed from before the collision to the hadrons: one
production file of PyJetscape's `example/prod_AuAu_0_10_jet`.

    t < 0              two Lorentz-contracted nuclei approach along the beam (z)
    t = 0              they collide: participants light up and fade out, the spectators
                       fly on; the hard scattering fires at the event's hard vertex
    t >= t_vertex      the jet partons run out of the hard vertex, coloured by jet leg
    t >= tau_0         the medium forms: MUSIC_2, the jet leg `arr` (background + wake)
    t >= t_emit        each bulk hadron appears where iSS emitted it and free-streams
    t >= --frag-time   the partons are replaced by their fragmentation hadrons

It extends `hydro_jet_particles_pyvista.py` (whose readers it uses) by the initial state in
front and by the colour coding of the hadrons.

The nuclei are SYNTHETIC
------------------------
The production files hold no nucleon positions (3dMCGlauber keeps them to itself), so the
two nuclei are sampled here: Woods-Saxon nucleons with a minimum distance, at an impact
parameter `b` drawn from the file's centrality class (`<cenMin>`, `<cenMax>` of its user XML;
b in [sqrt(cmin), sqrt(cmax)] * (2R + d_NN), uniform in b^2 -- a geometric estimate), and
participants decided by the black-disk criterion d_perp < sqrt(sigma_NN / pi).  This is
an illustration of the collision geometry, NOT the event's nucleons.  Two things tie it to
the event:
  * `b` points along the second-order participant plane of the medium's first frame
    (`--no-align` turns it off), so the almond of the synthetic participants lies the
    way the medium does;
  * the participants' transverse centroid is moved onto the medium's.
`--impact-b`, `--glauber-seed` fix the sampling; the run prints what it drew.

Lorentz contraction: the nuclei move at the beam velocity of the event's sqrt(s_NN) (from
the user XML, default 200 GeV) and are flattened by its gamma (~107 at RHIC): 0.12 fm thin
discs.  `--gamma-vis G` draws them thicker (flattened by G only); the speed is unchanged.

After the collision the participants fade out over `--participant-fade` fm/c (default:
until tau_0, at least 1 fm/c) -- in 3D MC-Glauber they become the strings the medium is
made of -- while the spectators keep flying along the beam until they leave the box.

Colours
-------
`--color-by species` (default)
    bulk hadrons by species: pions, kaons, (anti)protons, resonances, other (neutrons,
    hyperons, photons, leptons, ...).  The files hold FINAL-state hadrons (iSS and Pythia
    decayed the resonances), so "resonances" -- any hadron that is not long-lived -- only
    holds what iSS leaves undecayed: eta and Sigma0 (electromagnetic decays).
    Partons and jet hadrons by JET LEG: every shower has the colour of its initiator
    (`--leg-colors`), and each jet hadron that of the initiator nearest to it in
    (eta, phi).  That is an assignment, not an ancestry: Pythia's strings mix the partons
    and the jet_frag file has no hadron->parton link.  `--leg-dr-max R` leaves hadrons
    further than R from both legs white.  `--jet-hadron-color species` colours the jet
    hadrons by species instead (still drawn larger).
`--color-by pt`
    every hadron, bulk and jet, and every parton on ONE logarithmic pT scale
    (`--jet-cmap`, from `--pt-scale-min` GeV up to the hardest parton or hadron); jet
    hadrons are told apart by their larger points (`--frag-size`).

Usage
-----
    conda activate fno_pyvista_env            # or the analysis venv (utils/analysis_env)
    cd external_packages/js-contrib/contribs/Visualization

    P=../PyJetscape/example/prod_AuAu_0_10_jet/out_viz/AuAu_0_10_jet_seed0003.h5
    python full_event_pyvista.py --file $P --movie full_event.mp4
    python full_event_pyvista.py --file $P --color-by pt --movie full_event_pt.mp4
    python full_event_pyvista.py --file $P --gamma-vis 8 --impact-b 3 --interactive

The data are the three files `run_prod_jet.py` + `hadronize.py --tags bulk_jet,jet_frag`
leave next to each other (see `hydro_jet_particles_pyvista.py`).
"""

from __future__ import annotations

import os
import re
import secrets
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import hydro_jet_particles_pyvista as hjpp                            # noqa: E402
import wake_pyvista as wp                                             # noqa: E402
import hydro_pyvista as hp                                            # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

#: Woods-Saxon (A, R [fm], a [fm]); the 3dMCGlauber values for Au, Pb, Cu; O is a rough WS
NUCLEI = {"Au": (197, 6.38, 0.535), "Pb": (208, 6.62, 0.546),
          "Cu": (63, 4.20641, 0.5977), "O": (16, 2.608, 0.513)}
M_NUCLEON = 0.93827

#: (label, colour) of the nucleons: projectile (+z) and target (-z)
NUCLEON_STYLE = {
    "proj_part": ("{} +z: participants", "#ff8c42"),
    "proj_spec": ("{} +z: spectators", "#8a5a2b"),
    "targ_part": ("{} -z: participants", "#a597ff"),
    "targ_spec": ("{} -z: spectators", "#4f4887"),
}

#: species classes: (key, legend label, colour)
SPECIES = (("pion", "pions (pi\u00b1, pi0)", "#4f9dff"),
           ("kaon", "kaons (K\u00b1, K0)", "#3ddc84"),
           ("proton", "(anti)protons", "#ff5a5a"),
           ("resonance", "resonances", "#ffb13d"),
           ("other", "other", "#b8b8c8"))
PIONS = (211, 111)
KAONS = (321, 311, 310, 130)
PROTONS = (2212,)
#: hadrons that reach a detector (or decay weakly); any other hadron counts as a resonance
LONG_LIVED_HADRONS = (111, 211, 130, 310, 311, 321, 2112, 2212, 3122, 3112, 3222, 3312,
                      3322, 3334)

#: default jet-leg colours (leading leg first); white = not assigned (--leg-dr-max)
LEG_COLORS = "#ffd400,#ff4fd8,#7cfc00,#ff6b3d"
UNASSIGNED = "#ffffff"
QUARK_NAMES = {1: "d", 2: "u", 3: "s", 4: "c", 5: "b", 6: "t", 21: "g"}


# ──────────────────────────────────────────────────────────────────────────────
# Hadron classes
# ──────────────────────────────────────────────────────────────────────────────

def species_class(pid):
    """-> int array, the index into SPECIES of each pid."""
    a = np.abs(np.asarray(pid, dtype=np.int64))
    out = np.full(a.shape, 4, dtype=np.int64)                        # other
    hadron = a >= 100
    out[hadron & ~np.isin(a, LONG_LIVED_HADRONS)] = 3                # resonance
    out[np.isin(a, PROTONS)] = 2
    out[np.isin(a, KAONS)] = 1
    out[np.isin(a, PIONS)] = 0
    return out


def eta_phi(p):
    """(E, px, py, pz) rows -> (pseudorapidity, azimuth)."""
    p = np.asarray(p, float)
    pabs = np.linalg.norm(p[:, 1:4], axis=1)
    eta = 0.5 * np.log(np.maximum(pabs + p[:, 3], 1e-12) / np.maximum(pabs - p[:, 3], 1e-12))
    return eta, np.arctan2(p[:, 2], p[:, 1])


def assign_legs(p_had, p_ini, dr_max=None):
    """-> index of the initiator nearest to each hadron in (eta, phi); -1 beyond dr_max.

    `p_had`, `p_ini`: (E, px, py, pz) rows.
    """
    if not len(p_had):
        return np.zeros(0, dtype=np.int64)
    if not len(p_ini):
        return np.full(len(p_had), -1, dtype=np.int64)
    eh, ph = eta_phi(p_had)
    ei, pi_ = eta_phi(p_ini)
    dphi = np.angle(np.exp(1j * (ph[:, None] - pi_[None, :])))
    dr2 = (eh[:, None] - ei[None, :]) ** 2 + dphi ** 2
    leg = np.argmin(dr2, axis=1)
    if dr_max is not None:
        leg = np.where(dr2[np.arange(len(leg)), leg] <= dr_max ** 2, leg, -1)
    return leg


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Glauber nuclei
# ──────────────────────────────────────────────────────────────────────────────

def sample_nucleus(A, R, a, rng, d_min=0.4):
    """A Woods-Saxon nucleus at rest, (A, 3) [fm], centre of mass at the origin.

    Radii by rejection from r^2 * WS(r), directions isotropic; a nucleon closer than
    d_min to an accepted one is redrawn (the hard core 3dMCGlauber uses too).
    """
    r_max = R + 10.0 * a
    pos = np.empty((0, 3))
    while len(pos) < A:
        n = 4 * (A - len(pos)) + 16
        r = r_max * rng.random(n) ** (1.0 / 3.0)                      # uniform in volume
        r = r[rng.random(n) < 1.0 / (1.0 + np.exp((r - R) / a))]
        cth = 2.0 * rng.random(len(r)) - 1.0
        phi = 2.0 * np.pi * rng.random(len(r))
        sth = np.sqrt(1.0 - cth ** 2)
        cand = np.column_stack([r * sth * np.cos(phi), r * sth * np.sin(phi), r * cth])
        for c in cand:
            if len(pos) == A:
                break
            if d_min > 0 and len(pos) and np.min(np.sum((pos - c) ** 2, axis=1)) < d_min ** 2:
                continue
            pos = np.vstack([pos, c])
    return pos - pos.mean(axis=0)


def collide(proj, targ, b, sigma_nn_mb, phi_b=0.0):
    """Place two nuclei at impact parameter b and find the participants.

    The projectile is shifted by +b/2 along (cos phi_b, sin phi_b), the target by -b/2.
    A nucleon is a participant when some nucleon of the other nucleus passes within
    d = sqrt(sigma_NN / pi) of it in the transverse plane (1 mb = 0.1 fm^2).
    -> (proj, targ, wounded_proj, wounded_targ), positions as rest-frame (x, y, z).
    """
    u = np.array([np.cos(phi_b), np.sin(phi_b), 0.0])
    proj = np.asarray(proj, float) + 0.5 * b * u
    targ = np.asarray(targ, float) - 0.5 * b * u
    d2 = 0.1 * sigma_nn_mb / np.pi
    dxy = proj[:, None, :2] - targ[None, :, :2]
    hit = np.sum(dxy ** 2, axis=2) < d2
    return proj, targ, hit.any(axis=1), hit.any(axis=0)


def sigma_nn(sqrt_s):
    """Inelastic NN cross section [mb], 25 + 0.146 ln(s)^2 (41 mb at 200 GeV, 67 at 5.02 TeV)."""
    s = float(sqrt_s) ** 2
    return float(25.0 + 0.146 * np.log(s) ** 2) if sqrt_s > 10 else 30.0


def b_range(cen_min, cen_max, R, sigma_mb):
    """Geometric impact-parameter range [fm] of a centrality class (in %)."""
    b_all = 2.0 * R + np.sqrt(0.1 * sigma_mb / np.pi)
    return np.sqrt(cen_min / 100.0) * b_all, np.sqrt(cen_max / 100.0) * b_all


def medium_geometry(arr, meta, eta_cut=1.0):
    """-> (x_c, y_c, psi_2) of the first frame's energy density at |eta_s| < eta_cut.

    psi_2 is the participant-plane angle, eps_2 exp(2i psi_2) = -<r^2 exp(2i phi)>/<r^2>:
    the direction of the almond's SHORT axis, i.e. of the impact parameter.
    """
    e = np.asarray(arr[0, ..., hp.FEAT_E], float)                    # (nx, ny[, neta])
    if e.ndim == 3:
        etas = meta["eta_min"] + meta["deta"] * np.arange(e.shape[2])
        sel = np.abs(etas) < eta_cut
        e = e[:, :, sel if sel.any() else slice(None)].sum(axis=2)
    xs = meta["x_min"] + meta["dx"] * np.arange(e.shape[0])
    ys = meta["y_min"] + meta["dy"] * np.arange(e.shape[1])
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    w = e.sum()
    if w <= 0:
        return 0.0, 0.0, 0.0
    xc, yc = (e * X).sum() / w, (e * Y).sum() / w
    dx, dy = X - xc, Y - yc
    r2 = dx ** 2 + dy ** 2
    phi = np.arctan2(dy, dx)
    c = -(e * r2 * np.cos(2 * phi)).sum()
    s = -(e * r2 * np.sin(2 * phi)).sum()
    return float(xc), float(yc), float(0.5 * np.arctan2(s, c))


class Nuclei:
    """The two nuclei in the lab: positions at lab time t, by nucleon class."""

    def __init__(self, proj, targ, w_proj, w_targ, gamma, gamma_vis, t_fade):
        self.rest = {"proj": proj, "targ": targ}
        self.wounded = {"proj": w_proj, "targ": w_targ}
        self.beta = np.sqrt(max(0.0, 1.0 - 1.0 / gamma ** 2))
        self.flat = 1.0 / gamma_vis
        self.t_fade = float(t_fade)

    @property
    def n_part(self):
        return int(self.wounded["proj"].sum() + self.wounded["targ"].sum())

    def at(self, t):
        """-> {class key: (positions (n, 3), opacity)}; opacity 0 means not drawn."""
        out = {}
        for side, sign in (("proj", 1.0), ("targ", -1.0)):
            p = self.rest[side].copy()
            p[:, 2] = p[:, 2] * self.flat + sign * self.beta * t
            w = self.wounded[side]
            fade = 1.0 if t <= 0 else max(0.0, 1.0 - t / self.t_fade)
            out[f"{side}_part"] = (p[w], fade)
            out[f"{side}_spec"] = (p[~w], 1.0)
        return out


def nucleus_glyph(radius, flat):
    """A nucleon: a sphere flattened along the beam by `flat` (= 1/gamma_vis)."""
    import pyvista as pv
    s = pv.Sphere(radius=radius, theta_resolution=14, phi_resolution=10)
    s.points[:, 2] *= flat
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Reading
# ──────────────────────────────────────────────────────────────────────────────

def load_initiators(path, event=0):
    """The pair file's shower initiators (K, 11) without a Z's leptons, or an empty array."""
    import h5py
    with h5py.File(str(path), "r") as f:
        if "shower/initiators" not in f:
            return np.zeros((0, 11))
        o = f["shower/initiator_offsets"][:]
        ini = np.asarray(f["shower/initiators"][o[event]:o[event + 1]], float)
    return ini[~np.isin(np.abs(ini[:, 1]).astype(int), wp.LEPTON_PIDS)]


def collision_setup(attrs):
    """-> (nucleus, sqrt_s, cen_min, cen_max) from the file's user XML, with defaults."""
    xml = wp._attr_str(attrs, "prod_user_xml", "")

    def num(tag, default):
        m = re.search(rf"<{tag}>\s*([-+0-9.eE]+)\s*</{tag}>", xml)
        return float(m.group(1)) if m else default

    m = re.search(r"<projectile>\s*(\w+)\s*</projectile>", xml)
    nucleus = m.group(1) if m and m.group(1) in NUCLEI else "Au"
    sqrt_s = num("sqrts", num("eCM", 200.0))
    return nucleus, sqrt_s, num("cenMin", 0.0), num("cenMax", 10.0)


# ──────────────────────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────────────────────

class Tracks(hjpp.FreeStreaming):
    """Straight hadron tracks with a species class and a jet leg per hadron."""

    def __init__(self, p, x0, t0, t_on, pid, leg=None):
        p = np.asarray(p, float)
        super().__init__(p, x0, t0, t_on, np.hypot(p[:, 1], p[:, 2]))
        self.cls = species_class(pid)
        self.leg = (np.asarray(leg, dtype=np.int64) if leg is not None
                    else np.full(len(p), -1, dtype=np.int64))

    def state(self, t):
        """-> (on mask, positions of the hadrons that are on)."""
        on = t >= self.t_on
        return on, self.x0[on] + self.v[on] * (t - self.t0[on])[:, None]


def parton_tips(seg, t):
    """Parton tips at lab time t (as hydro_jet_pyvista.make_jet_overlay draws them)."""
    t0, t1 = seg["t0"], seg["t1"]
    span = np.where(t1 > t0, t1 - t0, 1.0)
    frac = np.clip((t - t0) / span, 0.0, 1.0)
    tips = seg["starts"] + (seg["ends"] - seg["starts"]) * frac[:, None]
    stream = seg["is_leaf"] & (t > t1)
    if stream.any():
        tips[stream] = seg["ends"][stream] + seg["vel"][stream] * (t - t1[stream])[:, None]
    return tips


def _add_partons(plotter, name, starts, tips, dirs, radius, color=None, pT=None,
                 cmap=None, clim=None):
    """Parton arrows (tube + cone head); a fixed colour, or pT on a log scale."""
    import pyvista as pv
    plotter.remove_actor(name, reset_camera=False)
    plotter.remove_actor(name + "_heads", reset_camera=False)
    m = len(starts)
    if m == 0:
        return
    pts = np.empty((2 * m, 3))
    pts[0::2], pts[1::2] = starts, tips
    lines = np.column_stack([np.full(m, 2), np.arange(0, 2 * m, 2),
                             np.arange(1, 2 * m, 2)]).astype(np.int64)
    poly = pv.PolyData(pts, lines=lines.ravel())
    heads = pv.PolyData(np.asarray(tips, float))
    heads["dir"] = np.asarray(dirs, float)
    if pT is not None:
        v = np.maximum(np.asarray(pT, float), clim[0])
        poly["pT"] = np.repeat(v, 2)
        heads["pT"] = v
    shafts = poly.tube(radius=radius, n_sides=8)
    glyphs = heads.glyph(orient="dir", scale=False, factor=1.0,
                         geom=pv.Cone(radius=radius * 2.4, height=radius * 5.0,
                                      resolution=14))
    kw = (dict(color=color) if pT is None else
          dict(scalars="pT", cmap=cmap, clim=clim, log_scale=True))
    common = dict(reset_camera=False, show_scalar_bar=False)
    plotter.add_mesh(shafts, name=name, **kw, **common)
    plotter.add_mesh(glyphs, name=name + "_heads", **kw, **common)


def _add_points(plotter, name, pos, size, color=None, pT=None, cmap=None, clim=None,
                opacity=1.0):
    import pyvista as pv
    if not len(pos):
        plotter.remove_actor(name, reset_camera=False)
        return
    cloud = pv.PolyData(np.ascontiguousarray(pos, dtype=float))
    if pT is None:
        kw = dict(color=color)
    else:
        cloud["pT"] = np.maximum(np.asarray(pT, float), clim[0])
        kw = dict(scalars="pT", cmap=cmap, clim=clim, log_scale=True)
    plotter.add_mesh(cloud, name=name, point_size=size, render_points_as_spheres=True,
                     reset_camera=False, show_scalar_bar=False, opacity=opacity, **kw)


def _add_pt_bar(plotter, clim, cmap):
    import pyvista as pv
    proxy = pv.PolyData(np.zeros((2, 3)))
    proxy["pT"] = np.asarray(clim, float)
    plotter.add_mesh(proxy, scalars="pT", cmap=cmap, clim=clim, log_scale=True,
                     opacity=0.0, name="pt_bar", reset_camera=False, show_scalar_bar=True,
                     scalar_bar_args=dict(title="pT  [GeV]",
                                          color="white", title_font_size=15,
                                          label_font_size=12, n_labels=5, fmt="%.1f",
                                          vertical=True, position_x=0.045,
                                          position_y=0.30, width=0.04, height=0.5))


def _set_legend(plotter, entries):
    try:
        plotter.remove_legend()
    except Exception:          # no legend yet
        pass
    if entries:
        h = 0.024 * len(entries) + 0.01
        leg = plotter.add_legend(labels=[list(e) for e in entries], bcolor=hp.PANEL_GREY,
                                 border=False, size=(0.26, h), loc="upper right",
                                 face="circle", background_opacity=0.6)
        # 'upper right' leaves the long jet labels running off the window: pin the box
        # (normalized viewport, lower-left corner + size) clear of the edge and of the
        # energy-density bar's title below it
        leg.SetPosition(0.71, 0.985 - h)
        leg.SetPosition2(0.27, h)


def make_overlay(*, nuclei, glyph, seg, legs, bulk, frag, args, info):
    """overlay(plotter, t) for hydro_pyvista.render_event.

    `legs`: list of (label, colour) per initiator; `seg["leg"]` is each parton's leg.
    `info`: dict with t_vertex, vertex, t_off (partons off), tau0, tau_end, box, anchor,
    pt_clim, header, z_box, nucleus.
    """
    pt_mode = args.color_by == "pt"
    clim = info["pt_clim"]
    state = {"legend": None, "bar": False, "anchor": False}
    n_sample = (np.bincount(bulk.cls, minlength=len(SPECIES)) if bulk is not None
                else np.zeros(len(SPECIES), dtype=int))

    def leg_color(i):
        return legs[i][1] if 0 <= i < len(legs) else UNASSIGNED

    def legend_entries(nuc_drawn, shown):
        """The nucleon classes on screen, then (once any bulk hadron is) the species with
        `shown / in the sample` -- most of a sample is emitted after the last frame or far
        along the beam, so the totals alone overstate what is drawn."""
        out = [(NUCLEON_STYLE[k][0].format(info["nucleus"]), NUCLEON_STYLE[k][1])
               for k in ("proj_part", "proj_spec", "targ_part", "targ_spec")
               if k in nuc_drawn]
        if shown.sum() and not pt_mode:
            out += [(f"{lab}  {n} / {n_tot}", col) for (_, lab, col), n, n_tot
                    in zip(SPECIES, shown, n_sample)]
        if not pt_mode and not args.jet_color:
            out += list(legs)
        return out

    def overlay(plotter, t):
        import pyvista as pv
        if not state["anchor"]:
            lo, hi = info["anchor"]
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                for z in (lo[2], hi[2])], float)
            plotter.add_mesh(pv.PolyData(corners), opacity=0.0, name="cam_anchor",
                             reset_camera=False, show_scalar_bar=False)
            state["anchor"] = True
        if pt_mode and not state["bar"]:
            _add_pt_bar(plotter, clim, args.jet_cmap)
            state["bar"] = True

        # ── nuclei ───────────────────────────────────────────────────────────
        nuc_drawn = []
        for key, (pos, alpha) in (nuclei.at(t).items() if nuclei is not None else ()):
            if len(pos) and key.endswith("_spec"):
                pos = pos[np.abs(pos[:, 2]) < info["z_box"]]       # gone once out of the box
            name = "nuc_" + key
            if alpha <= 0.0 or not len(pos):
                plotter.remove_actor(name, reset_camera=False)
                continue
            nuc_drawn.append(key)
            g = pv.PolyData(np.ascontiguousarray(pos)).glyph(geom=glyph, scale=False,
                                                            orient=False)
            plotter.add_mesh(g, name=name, color=NUCLEON_STYLE[key][1], opacity=alpha,
                             smooth_shading=True, reset_camera=False,
                             show_scalar_bar=False)

        # ── hard scattering flash ────────────────────────────────────────────
        dt = t - info["t_vertex"]
        if 0.0 <= dt < args.flash_time:
            s = pv.Sphere(radius=0.9 * (1.0 - dt / args.flash_time) + 0.2,
                          center=tuple(info["vertex"]))
            plotter.add_mesh(s, name="hard_flash", color="white",
                             opacity=0.9 * (1.0 - dt / args.flash_time),
                             reset_camera=False, show_scalar_bar=False)
        else:
            plotter.remove_actor("hard_flash", reset_camera=False)

        # ── partons ──────────────────────────────────────────────────────────
        n_legs_drawn = max(len(legs), 1)
        if seg is not None and t < info["t_off"]:
            tips = parton_tips(seg, t)
            draw = (t >= seg["t0"]) & (np.linalg.norm(tips - seg["starts"], axis=1) > 1e-4)
            if pt_mode:
                d = draw
                _add_partons(plotter, "partons", seg["starts"][d], tips[d], seg["dirs"][d],
                             args.jet_radius, pT=seg["pT"][d], cmap=args.jet_cmap, clim=clim)
            else:
                for i in range(-1, n_legs_drawn):
                    d = draw & (seg["leg"] == i)
                    col = args.jet_color or leg_color(i)
                    _add_partons(plotter, f"partons_{i}", seg["starts"][d], tips[d],
                                 seg["dirs"][d], args.jet_radius, color=col)
        else:
            for i in range(-1, n_legs_drawn):
                _add_partons(plotter, f"partons_{i}", [], [], [], args.jet_radius)
            _add_partons(plotter, "partons", [], [], [], args.jet_radius)

        # ── bulk hadrons ─────────────────────────────────────────────────────
        shown = np.zeros(len(SPECIES), dtype=int)
        n_bulk = 0
        if bulk is not None:
            on, pos = bulk.state(t)
            n_bulk = len(pos)
            cls = bulk.cls[on]
            shown = np.bincount(cls, minlength=len(SPECIES))
            if pt_mode:
                _add_points(plotter, "bulk", pos, args.bulk_size, pT=bulk.pT[on],
                            cmap=args.jet_cmap, clim=clim, opacity=args.bulk_opacity)
            else:
                for i, (k, _, col) in enumerate(SPECIES):
                    m = cls == i
                    _add_points(plotter, "bulk_" + k, pos[m], args.bulk_size, color=col,
                                opacity=args.bulk_opacity)

        # ── jet hadrons ──────────────────────────────────────────────────────
        n_frag = 0
        if frag is not None:
            on, pos = frag.state(t)
            n_frag = len(pos)
            if pt_mode:
                _add_points(plotter, "frag", pos, args.frag_size, pT=frag.pT[on],
                            cmap=args.jet_cmap, clim=clim)
            elif args.jet_hadron_color == "species":
                for i, (k, _, col) in enumerate(SPECIES):
                    m = frag.cls[on] == i
                    _add_points(plotter, "frag_" + k, pos[m], args.frag_size, color=col)
            else:
                for i in range(-1, n_legs_drawn):
                    m = frag.leg[on] == i
                    _add_points(plotter, f"frag_leg{i}", pos[m], args.frag_size,
                                color=leg_color(i))

        # ── legend, read-outs ────────────────────────────────────────────────
        entries = legend_entries(nuc_drawn, shown)
        if entries != state["legend"]:            # the counts change every frame
            _set_legend(plotter, entries)
            state["legend"] = entries

        if t < 0:
            stage = "approach"
        elif t < info["tau0"]:
            stage = "collision, hard scattering, pre-equilibrium"
        elif t <= info["tau_end"]:
            stage = "QGP: MUSIC_2 (medium + wake), freeze-out"
        else:
            stage = "hadrons free-streaming"
        # under hydro_pyvista's "event / t =" label (the window is 1008 x 800 pixels)
        plotter.add_text(stage, name="stage", position=(12, 690), font_size=11,
                         color="#ffe9a8", shadow=True)

        lines = [info["header"]]
        if n_bulk:
            lines.append(f"bulk hadrons shown {n_bulk} / {int(n_sample.sum())}: " + ", ".join(
                f"{k} {n}" for (k, _, _), n in zip(SPECIES, shown) if n))
        if frag is not None:
            lines.append(f"jet hadrons shown {n_frag} / {len(frag)}"
                         + ("  (large points)" if pt_mode else ""))
        plotter.add_text("\n".join(lines), name="hadron_note", position="lower_left",
                         font_size=10, color="#c8d6e0", shadow=False)
        axes_actor = getattr(plotter.renderer, "cube_axes_actor", None)
        if axes_actor is not None:
            axes_actor.update_bounds(info["box"])

    return overlay


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = hjpp.build_parser()
    p.description = __doc__
    g = p.add_argument_group("initial state (synthetic Glauber nuclei)")
    g.add_argument("--nucleus", choices=sorted(NUCLEI), default=None,
                   help="the colliding nuclei (default: <projectile> of the file's user "
                        "XML, else Au)")
    g.add_argument("--sqrt-s", type=float, default=None, dest="sqrt_s",
                   help="sqrt(s_NN) in GeV for the beam velocity, gamma and sigma_NN "
                        "(default: from the user XML, else 200)")
    g.add_argument("--impact-b", type=float, default=None, dest="impact_b",
                   help="impact parameter in fm (default: drawn from the file's "
                        "centrality class)")
    g.add_argument("--glauber-seed", type=int, default=None, dest="glauber_seed",
                   help="seed of the synthetic nuclei (default: fresh; the run prints it)")
    g.add_argument("--gamma-vis", type=float, default=None, dest="gamma_vis",
                   help="Lorentz factor the nuclei are DRAWN flattened by (default: the "
                        "true one, ~107 at 200 GeV; e.g. 8 for visibly thick nuclei)")
    g.add_argument("--nucleon-radius", type=float, default=0.55, dest="nucleon_radius",
                   help="drawn nucleon radius in fm (default 0.55)")
    g.add_argument("--participant-fade", type=float, default=None, dest="participant_fade",
                   help="fm/c over which the participants fade out after t = 0 "
                        "(default max(tau_0, 1))")
    g.add_argument("--no-align", action="store_true", dest="no_align",
                   help="do not turn b along the medium's participant plane")
    g.add_argument("--no-nuclei", action="store_true", dest="no_nuclei",
                   help="no initial state (start at --t-min 0 like "
                        "hydro_jet_particles_pyvista.py)")
    g.add_argument("--flash-time", type=float, default=0.8, dest="flash_time",
                   help="how long the hard-scattering flash is shown, fm/c (default 0.8)")
    c = p.add_argument_group("colours")
    c.add_argument("--color-by", choices=("species", "pt"), default="species",
                   dest="color_by",
                   help="'species' (default): bulk hadrons by species, partons and jet "
                        "hadrons by jet leg; 'pt': all hadrons and partons on one log pT "
                        "scale (--jet-cmap)")
    c.add_argument("--jet-hadron-color", choices=("leg", "species"), default="leg",
                   dest="jet_hadron_color",
                   help="with --color-by species: jet hadrons by jet leg (default) or "
                        "by species")
    c.add_argument("--leg-colors", default=LEG_COLORS, dest="leg_colors",
                   help=f"comma-separated jet-leg colours, leading leg first "
                        f"(default {LEG_COLORS})")
    c.add_argument("--leg-dr-max", type=float, default=None, dest="leg_dr_max",
                   help="jet hadrons further than this from every initiator in "
                        "(eta, phi) are drawn white (default: all assigned)")
    c.add_argument("--pt-scale-min", type=float, default=0.1, dest="pt_scale_min",
                   help="lower end of the log pT scale in GeV (default 0.1)")
    # thicker parton arrows than hydro_jet_pyvista's 0.04 fm: they cross a lit medium here
    p.set_defaults(movie="full_event.mp4", t_min=None, nt=90, jet_radius=0.1)
    return p


def _leg_label(ini_row, rank):
    pid = int(ini_row[1])
    name = QUARK_NAMES.get(abs(pid), str(pid))
    if pid < 0 and abs(pid) != 21:
        name += "bar"
    pT = float(np.hypot(ini_row[3], ini_row[4]))
    return f"jet {rank + 1}: {name}, pT {pT:.1f} GeV"


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not (args.movie or args.vtk_dir or args.interactive):
        args.movie = "full_event.mp4"
    if not os.path.isabs(args.outdir):
        args.outdir = os.path.join(_THIS_DIR, args.outdir)

    print(f"=== {args.file}  event {args.event} ===")
    panels, meta, attrs = wp.load_pair(args.file, args.event, ("jet",),
                                       eos_table=args.eos_table)
    arr = panels["jet"]
    tau0 = meta["tau_min"]
    tau_max = tau0 + (meta["ntau"] - 1) * meta["dtau"]
    print(f"  MUSIC_2 (arr, medium + wake): grid {meta['nx']}x{meta['ny']}x{meta['neta']}"
          f"x{meta['ntau']}, tau {tau0:.2f}..{tau_max:.2f} fm/c; T from {meta['eos']}")

    # ── jet legs ──────────────────────────────────────────────────────────────
    ini = load_initiators(args.file, args.event)
    order = np.argsort(-np.hypot(ini[:, 3], ini[:, 4])) if len(ini) else np.zeros(0, int)
    colors = [c.strip() for c in args.leg_colors.split(",") if c.strip()] or [UNASSIGNED]
    legs = [(_leg_label(ini[i], r), colors[r % len(colors)]) for r, i in enumerate(order)]
    shower_to_leg = {int(ini[i, 0]): r for r, i in enumerate(order)}
    p_ini = ini[order][:, [6, 3, 4, 5]] if len(ini) else np.zeros((0, 4))   # E, px, py, pz
    for lab, col in legs:
        print(f"  {lab}  ({col})")

    seg = None if args.no_jet else wp.load_shower(args.file, args.event,
                                                  args.jet_min_energy)
    seg_all = seg if seg is not None else wp.load_shower(args.file, args.event, 0.0)
    if seg is not None:
        seg["leg"] = np.array([shower_to_leg.get(int(s), -1) for s in seg["shower"]])
        print(f"  shower: {len(seg['starts'])} partons, max pT {seg['pT'].max():.1f} GeV")

    had = hjpp.load_hadrons(args.file, args.event, args.hadron_stem, args.sample,
                            args.frag_sample, args.rng_seed)
    t_shower = (float(max(seg_all["t0"].max(), seg_all["t1"].max()))
                if seg_all is not None else None)
    t_frag = args.frag_time if args.frag_time is not None else (t_shower or tau_max)
    if args.t_max is None:
        args.t_max = max(t_frag, tau_max) + args.stream_time
    z_box = args.z_max if args.z_max is not None else 0.8 * tau_max
    if args.t_min is None:
        args.t_min = 0.0 if args.no_nuclei else -0.9 * z_box
    args.z_max = max(z_box, -args.t_min + 1.0) if not args.no_nuclei else z_box

    # ── hadrons ───────────────────────────────────────────────────────────────
    bulk = frag = None
    if had["bulk"] is not None and not args.no_bulk:
        h = had["bulk"]
        keep = hjpp.select(h, args.hadron_eta_max, args.hadron_pt_min)
        x = np.asarray(h["x"], float)[keep]
        bulk = Tracks(np.asarray(h["p"], float)[keep], x[:, 1:], x[:, 0], x[:, 0],
                      np.asarray(h["pid"])[keep])
        counts = ", ".join(f"{lab} {int((bulk.cls == i).sum())}"
                           for i, (_, lab, _) in enumerate(SPECIES))
        print(f"  bulk_jet sample {had['k']} of {had['n_k']}: {len(bulk)} hadrons ({counts})")
    if had["frag"] is not None and not args.no_frag:
        h = had["frag"]
        keep = hjpp.select(h, args.hadron_eta_max, args.hadron_pt_min)
        p = np.asarray(h["p"], float)[keep]
        pid = np.asarray(h["pid"])[keep]
        n = len(p)
        t_on = (np.full(n, float(t_frag)) if args.formation_tau0 is None else
                hjpp.formation_times(pid, p, had["t_vertex"], args.formation_tau0, t_frag))
        leg = assign_legs(p, p_ini, args.leg_dr_max)
        frag = Tracks(p, np.tile(had["vertex"], (n, 1)), np.full(n, had["t_vertex"]),
                      t_on, pid, leg)
        per_leg = ", ".join(f"jet {i + 1}: {int((leg == i).sum())}" for i in range(len(legs)))
        print(f"  jet_frag sample {had['j']} of {had['n_j']}: {n} hadrons ({per_leg}"
              + (f", unassigned {int((leg < 0).sum())}" if (leg < 0).any() else "")
              + f"), shown from t = {t_frag:.2f} fm/c")
    if args.sample is None or args.frag_sample is None:
        print(f"  (random sample choice; reproduce with --sample {had['k']} "
              f"--frag-sample {had['j']})")

    # ── synthetic nuclei ──────────────────────────────────────────────────────
    nucleus, sqrt_s, cmin, cmax = collision_setup(attrs)
    nucleus = args.nucleus or nucleus
    sqrt_s = args.sqrt_s or sqrt_s
    A, R, a = NUCLEI[nucleus]
    sig = sigma_nn(sqrt_s)
    gamma = sqrt_s / (2.0 * M_NUCLEON)
    nuclei = glyph = None
    b = 0.0
    if not args.no_nuclei:
        seed = args.glauber_seed if args.glauber_seed is not None else secrets.randbelow(2 ** 31)
        rng = np.random.default_rng(seed)
        b_lo, b_hi = b_range(cmin, cmax, R, sig)
        b = (args.impact_b if args.impact_b is not None else
             float(np.sqrt(b_lo ** 2 + (b_hi ** 2 - b_lo ** 2) * rng.random())))
        xc, yc, psi2 = medium_geometry(arr, meta)
        phi_b = 0.0 if args.no_align else psi2
        proj, targ, wp_, wt_ = collide(sample_nucleus(A, R, a, rng),
                                       sample_nucleus(A, R, a, rng), b, sig, phi_b)
        part = np.vstack([proj[wp_], targ[wt_]])
        if len(part):
            shift = np.array([xc, yc, 0.0]) - np.r_[part[:, :2].mean(axis=0), 0.0]
            proj, targ = proj + shift, targ + shift
        gamma_vis = args.gamma_vis or gamma
        t_fade = args.participant_fade or max(tau0, 1.0)
        nuclei = Nuclei(proj, targ, wp_, wt_, gamma, gamma_vis, t_fade)
        glyph = nucleus_glyph(args.nucleon_radius, 1.0 / gamma_vis)
        print(f"  synthetic {nucleus}+{nucleus} at sqrt(s_NN) = {sqrt_s:g} GeV "
              f"(gamma {gamma:.1f}, drawn {gamma_vis:.1f}), sigma_NN {sig:.1f} mb: "
              f"b = {b:.2f} fm (class {cmin:g}-{cmax:g}% -> b in [{b_lo:.1f}, {b_hi:.1f}]), "
              f"N_part = {nuclei.n_part}, b along "
              + ("x (--no-align)" if args.no_align else
                 f"the medium's psi_2 = {np.degrees(psi2):.0f} deg")
              + f"; --glauber-seed {seed}")

    # ── scene ─────────────────────────────────────────────────────────────────
    xy_max = args.xy_max if args.xy_max is not None else hp.medium_xy_max(arr, meta)
    if nuclei is not None:
        xy_max = max(xy_max, R + 0.5 * b + 1.0)
        args.xy_max = xy_max
    box = hp._scene_bounds(hp.cartesian_axes(meta, args, args.t_max, xy_max))
    lo, hi = np.array(box[0::2], float), np.array(box[1::2], float)
    t_anchor = min(t_shower if t_shower is not None else tau_max, args.t_max)
    if seg is not None:
        tips = parton_tips(seg, t_anchor)
        pts = np.vstack([seg["starts"], tips])
        lo, hi = np.minimum(lo, pts.min(axis=0)), np.maximum(hi, pts.max(axis=0))

    pt_max = 1.0
    for arr_pt in ((seg["pT"] if seg is not None else None),
                   (bulk.pT if bulk is not None else None),
                   (frag.pT if frag is not None else None)):
        if arr_pt is not None and len(arr_pt):
            pt_max = max(pt_max, float(arr_pt.max()))
    t_off = hjpp.parton_off_time(
        args.parton_off if args.parton_off != "auto" else
        ("formed" if args.formation_tau0 is not None else "frag"), t_frag, frag)

    header = (f"{nucleus}+{nucleus}  sqrt(s_NN) = {sqrt_s:g} GeV,  "
              f"{wp._attr_str(attrs, 'prod_hard', 'jet')}")
    if nuclei is not None:
        header += f"\nsynthetic nuclei: b = {b:.1f} fm, N_part = {nuclei.n_part}"
    header += (f"\nbulk sample {had['k']}/{had['n_k']}, fragmentation "
               f"{had['j']}/{had['n_j']};  partons -> jet hadrons at t = {t_frag:.1f} fm/c")
    info = dict(t_vertex=float(had["t_vertex"]), vertex=np.asarray(had["vertex"], float),
                t_off=t_off, tau0=tau0, tau_end=tau_max, box=box, anchor=(lo, hi),
                pt_clim=(args.pt_scale_min, pt_max), header=header, z_box=args.z_max,
                nucleus=nucleus)
    overlay = make_overlay(nuclei=nuclei, glyph=glyph, seg=seg, legs=legs, bulk=bulk,
                           frag=frag, args=args, info=info)
    hp.render_event(args.event, arr, meta, args, overlay=overlay)
    print(f"Done. Outputs in {os.path.abspath(args.outdir)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
