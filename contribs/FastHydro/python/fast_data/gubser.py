#!/usr/bin/env python3
"""
Viscous (Israel-Stewart) Gubser-flow benchmark data generator for FNO
training / validation.

Physics
-------
Implements the semi-analytic solution of conformal Israel-Stewart
hydrodynamics under Gubser symmetry from

    H. Marrochio, J. Noronha, G. S. Denicol, M. Luzum, S. Jeon, C. Gale,
    "Solutions of Conformal Israel-Stewart Relativistic Viscous Fluid
    Dynamics", arXiv:1307.6130 [Phys. Rev. C 91, 014903 (2015)].

This is the same solution used as the standard regression test of MUSIC
(the paper's Sec. IV), i.e. the generated fields follow the *same sign and
index conventions* as the MUSIC Gubser test.

Strategy (paper Sec. II-III):
  1. Weyl-rescale Milne/hyperbolic coordinates x~ = (tau~, r~, phi, eta)
     (tau~ = q*tau, r~ = q*r) to dS3 x R with coordinates (rho, theta):

         sinh(rho) = -(1 - tau~^2 + r~^2) / (2 tau~)          [Eq. (8)]
         tan(theta) =  2 r~ / (1 + tau~^2 - r~^2)

  2. In de Sitter coordinates the flow is static and all fields depend on
     rho only.  The IS equations reduce to two coupled ODEs [Eqs. (12),(13)]:

         (1/That) dThat/drho + (2/3) tanh(rho) = (1/3) pibar tanh(rho)

         tauR_hat [ dpibar/drho + (4/3) pibar^2 tanh(rho) ] + pibar
                  = (4/3) (eta/s) tanh(rho) / That

     with pibar := pihat^eta_eta / (That * shat),
     tauR_hat = c * (eta/s) / That  (relaxation time, c = 5 default).

  3. Map back with the Weyl/coordinate dictionary [Eqs. (9)-(11)]:

         u_mu(tau~, r~)    = tau~ * (d xhat^nu / d x~^mu) uhat_nu
         T(tau, r)         = That(rho) / tau
         pi_munu(tau~,r~)  = (1/tau~^2) (dxhat^a/dx~^mu)(dxhat^b/dx~^nu) pihat_ab

     The only independent de Sitter shear component is
     pihat^eta_eta (= pibar * That * shat); orthogonality gives
     pihat^{rho mu} = 0 and tracelessness + SO(3) symmetry give
     pihat^theta_theta = pihat^phi_phi = -pihat^eta_eta / 2.

Closed-form cross-checks built into --selftest:
  * ideal limit (eta/s -> 0):  That = That0 / cosh^(2/3)(rho),
    i.e. T_ideal(tau,r) of Eq. (15);
  * pi^mu_mu = 0 and u_mu pi^{mu nu} = 0 on the Milne grid (machine prec.);
  * |pibar(rho -> -inf)| -> sqrt(1/c)  (paper Sec. III asymptotics);
  * analytic Jacobians vs. finite differences;
  * HDF5 round-trip against the schema of read_3d_hdf5.read_3d_data_hdf5.

Output schema (identical to the project's training files)
----------------------------------------------------------
HDF5:
  attrs   : nFeatures, nx, ny, neta, x_min, y_min, eta_min,
            dx, dy, deta, tau_min, tau_min_MUSIC, dtau,
            choose_ntau, nevents (+ feature_names, units as extras)
  datasets: arr             (nevents, nFeatures, nx, ny, neta, ntau) float32
            ntau_freezeout  (nevents,)
            tau_freezeout   (nevents,)
            params/*        per-event Gubser parameters (extra, ignored by reader)

ROOT (optional, requires PyROOT; uproot cannot write TParameter<double>):
  TParameter<double> objects for every scalar key above, plus tree 't' with
  branches user_res (flattened (ntau, nx, ny, neta, nFeatures) per event),
  ntau_freezeout, tau_freezeout -- byte-compatible with read_3d_root.py.

Units
-----
Internally everything is solved in dimensionless q-units.  --units fm gives
T in fm^-1 and e, pi in fm^-4 (multiply by hbar*c = 0.19733 GeV fm for GeV);
--units GeV gives T in GeV and e, pi in GeV/fm^3.

Channels
--------
Selected with --channels (comma list).  Available:
  e        energy density  (conformal EOS  e = a_e T^4,  e = 3p)
  T        temperature
  ux, uy   transverse 4-velocity components u^x, u^y  (u^eta = 0 exactly)
  ueta     u^eta (identically zero; kept for layout compatibility)
  vx, vy   3-velocity v^i = u^i / u^tau
  utau     u^tau
  pixx, piyy, pixy, pitautau, pitaux, pitauy
           contravariant Milne shear components pi^{mu nu}
  tau2_pietaeta
           tau^2 * pi^{eta eta}  (= pi^eta_eta; the quantity plotted in
           Fig. 2 of arXiv:1307.6130; same units as the other pi channels)
Default: "e,ux,uy,ueta"  (nFeatures = 4, matching the current pipeline).

Example
-------
  python gubser_benchmark.py --nevents 64 --out gubser_64ev.h5
  python gubser_benchmark.py --selftest
  python gubser_benchmark.py --plot fig2_check.png       # reproduce Fig. 2
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

HBARC = 0.19733  # GeV fm

# ----------------------------------------------------------------------------
# 1. de Sitter ODEs  [arXiv:1307.6130, Eqs. (12)-(13)]
# ----------------------------------------------------------------------------

def solve_de_sitter(That0=1.2, pibar0=0.0, eta_s=0.2, c_relax=5.0,
                    rho_min=-30.0, rho_max=30.0, rho_ic=0.0,
                    n_dense=8001, rtol=1e-10, atol=1e-12):
    """Solve the Israel-Stewart Gubser ODEs in de Sitter time rho.

    State y = (ln That, pibar); integrating ln That keeps That > 0 exactly.

    Returns (That(rho), pibar(rho)) as cubic-spline callables on
    [rho_min, rho_max].
    """
    if eta_s < 1e-8 and pibar0 == 0.0:
        # Exact ideal limit [Eq. (15)]: pibar = 0, That = That0 * (cosh rho_ic / cosh rho)^(2/3).
        # Integrating the IS system here is hopelessly stiff (tau_R -> 0), so use the closed form.
        def That_fn(rho):
            return That0 * (np.cosh(rho_ic) / np.cosh(np.asarray(rho, dtype=float))) ** (2.0 / 3.0)

        def pibar_fn(rho):
            return np.zeros_like(np.asarray(rho, dtype=float))

        return That_fn, pibar_fn

    def rhs(rho, y):
        lnThat, pibar = y
        That = np.exp(lnThat)
        th = np.tanh(rho)
        dlnT = (-2.0 / 3.0 + pibar / 3.0) * th                    # Eq. (12)
        dpib = ((4.0 / (3.0 * c_relax)) * th                      # Eq. (13)
                - (4.0 / 3.0) * pibar**2 * th
                - pibar * That / (c_relax * eta_s))
        return (dlnT, dpib)

    y0 = (np.log(That0), pibar0)
    grids, vals = [], []
    for lo, hi in ((rho_ic, rho_min), (rho_ic, rho_max)):
        if lo == hi:
            continue
        t_eval = np.linspace(lo, hi, n_dense)
        sol = solve_ivp(rhs, (lo, hi), y0, t_eval=t_eval,
                        method='RK45', rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"de Sitter ODE integration failed: {sol.message}")
        grids.append(sol.t)
        vals.append(sol.y)

    rho = np.concatenate([g[::-1] if g[1] < g[0] else g for g in grids])
    y = np.concatenate([v[:, ::-1] if g[1] < g[0] else v
                        for g, v in zip(grids, vals)], axis=1)
    order = np.argsort(rho)
    rho, y = rho[order], y[:, order]
    rho, uniq = np.unique(rho, return_index=True)
    y = y[:, uniq]

    lnT_spl = CubicSpline(rho, y[0])
    pib_spl = CubicSpline(rho, y[1])
    return (lambda r: np.exp(lnT_spl(r))), (lambda r: pib_spl(r))


# ----------------------------------------------------------------------------
# 2. de Sitter -> Milne dictionary  [arXiv:1307.6130, Eqs. (5), (8)-(11)]
# ----------------------------------------------------------------------------

def gubser_fields(tau, x, y, q, That_fn, pibar_fn, a_e):
    """Evaluate all Milne fields on broadcastable arrays tau, x, y (fm).

    Returns dict of arrays in *fm units* (T in fm^-1; e, pi in fm^-4).
    All shear components are contravariant Milne components, except
    'tau2_pietaeta' = tau^2 pi^{eta eta} = pi^eta_eta.
    """
    tt = q * tau                                 # tau~
    r = np.sqrt(x * x + y * y)
    rt = q * r                                   # r~

    S = 1.0 + tt**2 + rt**2
    D = S**2 - 4.0 * tt**2 * rt**2               # = (1+tau~^2-r~^2)^2 + 4 r~^2
    sqrtD = np.sqrt(D)

    sinh_rho = -(1.0 - tt**2 + rt**2) / (2.0 * tt)          # Eq. (8)
    rho = np.arcsinh(sinh_rho)
    cosh_rho = sqrtD / (2.0 * tt)                # closed form of cosh(rho)

    That = That_fn(rho)
    pibar = pibar_fn(rho)

    # --- temperature / energy density  [Eq. (10)] ---
    T = That / tau                               # fm^-1   (T~ = That/tau~, T = q T~)
    e = a_e * T**4                               # fm^-4, conformal e = 3p

    # --- flow  [Eq. (5) / Eq. (9)] ---
    cosh_k = S / sqrtD
    sinh_k = 2.0 * tt * rt / sqrtD
    with np.errstate(invalid='ignore', divide='ignore'):
        cphi = np.where(r > 0, x / np.where(r > 0, r, 1.0), 1.0)
        sphi = np.where(r > 0, y / np.where(r > 0, r, 1.0), 0.0)
    ux = sinh_k * cphi
    uy = sinh_k * sphi
    utau = cosh_k

    # --- shear  [Eq. (11)] ---
    # pihat^eta_eta = pibar * That * shat,  shat = (4/3) a_e That^3
    pihat = (4.0 / 3.0) * a_e * pibar * That**4              # dimensionless
    # covariant dS components: pihat_{theta theta} = -cosh^2(rho)/2 * pihat,
    # pihat_{phi phi} = sin^2(theta) * pihat_{theta theta}, pihat_{eta eta} = pihat,
    # pihat_{rho mu} = 0.
    # Analytic Jacobians of (rho, theta)(tau~, r~):
    drho_dt = S / (2.0 * tt**2 * cosh_rho)
    drho_dr = -rt / (tt * cosh_rho)
    dth_dt = -4.0 * tt * rt / D
    dth_dr = 2.0 * S / D
    # sin(theta) = 2 r~ / sqrt(D)  (theta in (0, pi) => sin >= 0)

    pth = -0.5 * pihat * cosh_rho**2             # pihat_{theta theta}
    inv_tt2 = 1.0 / tt**2

    # covariant hyperbolic components [Eq. (11)]; pihat_{rho a} = 0 drops rho terms
    p_tt_lo = inv_tt2 * dth_dt**2 * pth
    p_tr_lo = inv_tt2 * dth_dt * dth_dr * pth
    p_rr_lo = inv_tt2 * dth_dr**2 * pth
    # r~^2 * pi~^{phi phi} written without dividing by r (regular at r = 0):
    # pi~_{phi phi} = sin^2(th) pth / tt^2 ;  r~^2 pi~^{phi phi} = pi~_{phi phi}/r~^2
    r2_pff_up = -2.0 * cosh_rho**2 * pihat / (D * tt**2)
    # mixed eta component: pi~^eta_eta = pihat / tau~^4
    p_ee_mix = pihat / tt**4

    # raise indices with g~ = diag(-1, 1, 1, tau~^2)
    p_tt_up = p_tt_lo                # (g^tt)^2 = 1
    p_tr_up = -p_tr_lo
    p_rr_up = p_rr_lo

    # polar -> Cartesian (contravariant), pi^{r phi} = 0
    pixx = cphi**2 * p_rr_up + sphi**2 * r2_pff_up
    piyy = sphi**2 * p_rr_up + cphi**2 * r2_pff_up
    pixy = cphi * sphi * (p_rr_up - r2_pff_up)
    pitaux = cphi * p_tr_up
    pitauy = sphi * p_tr_up

    # physical normalisation: dimensionless q-unit components -> fm units.
    # (tau, x, y) block carries q^4; tau^2 pi^{eta eta} = pi^eta_eta also q^4.
    q4 = q**4
    out = {
        'T': T, 'e': e,
        'utau': utau, 'ux': ux, 'uy': uy, 'ueta': np.zeros_like(ux),
        'vx': ux / utau, 'vy': uy / utau,
        'pitautau': q4 * p_tt_up, 'pitaux': q4 * pitaux, 'pitauy': q4 * pitauy,
        'pixx': q4 * pixx, 'piyy': q4 * piyy, 'pixy': q4 * pixy,
        'tau2_pietaeta': q4 * p_ee_mix,
    }
    out['_rho'] = rho        # for tests
    return out


# ----------------------------------------------------------------------------
# 3. Grid + event assembly
# ----------------------------------------------------------------------------

DEFAULT_GRID = dict(nx=64, ny=64, neta=8, ntau=40,
                    dx=0.2, dy=0.2, deta=0.5, dtau=0.1,
                    x_min=-6.3, y_min=-6.3, eta_min=-1.75, tau_min=1.0)

DEFAULT_CHANNELS = ('e', 'ux', 'uy', 'ueta')

# QGP ideal-gas conformal EOS  e = a_e T^4  (Nc = 3, Nf = 2.5)
A_E_DEFAULT = (np.pi**2 / 30.0) * (2 * (3**2 - 1) + 3.5 * 3 * 2.5)   # ~ 13.898


def make_event(params, grid, channels, units='fm', T_fo_GeV=0.150,
               pad_after_freezeout=False):
    """One Gubser 'event' -> (arr (nF, nx, ny, neta, ntau), ntau_fo, tau_fo)."""
    g = grid
    tau = g['tau_min'] + g['dtau'] * np.arange(g['ntau'])
    xv = g['x_min'] + g['dx'] * np.arange(g['nx'])
    yv = g['y_min'] + g['dy'] * np.arange(g['ny'])

    That_fn, pibar_fn = solve_de_sitter(
        That0=params['That0'], pibar0=params['pibar0'],
        eta_s=params['eta_s'], c_relax=params['c_relax'])

    X = xv[:, None, None]
    Y = yv[None, :, None]
    TAU = tau[None, None, :]
    f = gubser_fields(TAU, X, Y, params['q'], That_fn, pibar_fn,
                      a_e=params.get('a_e', A_E_DEFAULT))

    scale = HBARC if units == 'GeV' else 1.0     # fm^-1 -> GeV, fm^-4 -> GeV/fm^3
    unit_of = {'T': scale, 'e': scale,
               'pitautau': scale, 'pitaux': scale, 'pitauy': scale,
               'pixx': scale, 'piyy': scale, 'pixy': scale,
               'tau2_pietaeta': scale}

    nF = len(channels)
    arr = np.empty((nF, g['nx'], g['ny'], g['neta'], g['ntau']), dtype=np.float32)
    for i, ch in enumerate(channels):
        field = f[ch] * unit_of.get(ch, 1.0)          # (nx, ny, ntau)
        arr[i] = np.broadcast_to(field[:, :, None, :],     # boost invariant
                                 (g['nx'], g['ny'], g['neta'], g['ntau']))

    # freeze-out metadata from the central temperature (fm^-1 internally)
    T_fo = T_fo_GeV / HBARC
    ix, iy = np.argmin(np.abs(xv)), np.argmin(np.abs(yv))
    T_center = f['T'][ix, iy, :]
    above = np.nonzero(T_center >= T_fo)[0]
    ntau_fo = int(above[-1] + 1) if above.size else 1
    tau_fo = float(tau[ntau_fo - 1])

    if pad_after_freezeout:                       # mimic MUSIC-file zero padding
        arr[..., ntau_fo:] = 0.0

    return arr, ntau_fo, tau_fo


def sample_params(rng, n,
                  q_range=(0.8, 1.5), That0_range=(1.0, 1.6),
                  eta_s_range=(0.08, 0.30), c_relax=5.0, pibar0=0.0):
    """Per-event parameter family: the operator-learning 'distribution'."""
    return [dict(q=float(rng.uniform(*q_range)),
                 That0=float(rng.uniform(*That0_range)),
                 eta_s=float(rng.uniform(*eta_s_range)),
                 c_relax=float(c_relax), pibar0=float(pibar0))
            for _ in range(n)]


# ----------------------------------------------------------------------------
# 4. Writers (schema-compatible with read_3d_hdf5.py / read_3d_root.py)
# ----------------------------------------------------------------------------

def write_hdf5(path, events, grid, channels, units, param_list):
    import h5py
    nev = len(events)
    nF = len(channels)
    g = grid
    with h5py.File(path, 'w') as hf:
        for k in ('nx', 'ny', 'neta', 'x_min', 'y_min', 'eta_min',
                  'dx', 'dy', 'deta', 'tau_min', 'dtau'):
            hf.attrs[k] = float(g[k])
        hf.attrs['nFeatures'] = float(nF)
        hf.attrs['tau_min_MUSIC'] = float(g.get('tau_min_MUSIC', g['tau_min']))
        hf.attrs['choose_ntau'] = int(g['ntau'])
        hf.attrs['nevents'] = int(nev)
        hf.attrs['feature_names'] = list(channels)        # extra (ignored by reader)
        hf.attrs['units'] = units
        hf.attrs['source'] = ('viscous Gubser semi-analytic benchmark, '
                              'arXiv:1307.6130')

        hf.create_dataset('ntau_freezeout',
                          data=np.array([e[1] for e in events], dtype=np.int32))
        hf.create_dataset('tau_freezeout',
                          data=np.array([e[2] for e in events], dtype=np.float64))

        ds = hf.create_dataset(
            'arr', shape=(nev, nF, g['nx'], g['ny'], g['neta'], g['ntau']),
            dtype=np.float32,
            chunks=(1, nF, g['nx'], g['ny'], g['neta'], g['ntau']),
            compression='lzf')
        for i, (arr, _, _) in enumerate(events):
            ds[i] = arr

        pg = hf.create_group('params')                    # extra provenance
        for key in param_list[0]:
            pg.create_dataset(key, data=np.array([p[key] for p in param_list]))
    print(f"Wrote {nev} events -> {path}")


def write_root(path, events, grid, channels):
    """Exact read_3d_root.py schema; needs PyROOT (uproot cannot write
    TParameter<double>).  Falls back with a message if ROOT is missing."""
    try:
        import ROOT
        from array import array
    except ImportError:
        print("PyROOT not available -- skipping ROOT output "
              "(HDF5 file already matches the training pipeline; "
              "read_3d_hdf5.convert_root_to_hdf5 is then unnecessary).")
        return
    g = grid
    f = ROOT.TFile(path, 'RECREATE')
    scalars = dict(nFeatures=len(channels), nx=g['nx'], ny=g['ny'],
                   neta=g['neta'], x_min=g['x_min'], y_min=g['y_min'],
                   eta_min=g['eta_min'], dx=g['dx'], dy=g['dy'],
                   deta=g['deta'], tau_min=g['tau_min'],
                   tau_min_MUSIC=g.get('tau_min_MUSIC', g['tau_min']),
                   dtau=g['dtau'])
    for k, v in scalars.items():
        ROOT.TParameter('double')(k, float(v)).Write(k)

    t = ROOT.TTree('t', 'gubser benchmark')
    vec = ROOT.std.vector('float')()
    ntf = array('i', [0])
    tf = array('d', [0.0])
    t.Branch('user_res', vec)
    t.Branch('ntau_freezeout', ntf, 'ntau_freezeout/I')
    t.Branch('tau_freezeout', tf, 'tau_freezeout/D')
    for arr, n_fo, tau_fo in events:
        # reader reshapes flat -> (ntau, nx, ny, neta, nFeatures)
        flat = np.ascontiguousarray(arr.transpose(4, 1, 2, 3, 0)).ravel()
        vec.clear()
        vec.reserve(len(flat))
        for v in flat:
            vec.push_back(float(v))
        ntf[0], tf[0] = int(n_fo), float(tau_fo)
        t.Fill()
    t.Write()
    f.Close()
    print(f"Wrote ROOT file -> {path}")


# ----------------------------------------------------------------------------
# 5. Self-tests: analytic ground truth, tensor identities, schema round-trip
# ----------------------------------------------------------------------------

def selftest(verbose=True, tmpdir=None):
    rng = np.random.default_rng(0)
    ok = True

    def check(name, val, tol):
        nonlocal ok
        status = 'PASS' if val < tol else 'FAIL'
        ok &= (val < tol)
        if verbose:
            print(f"  [{status}] {name:55s} {val:.3e} (tol {tol:.0e})")

    # (a) Jacobians of (rho, theta)(tau~, r~) vs finite differences
    q = 1.13
    tau = np.array([1.7]); x = np.array([2.1]); y = np.array([-0.9])
    eps = 1e-6
    def rho_theta(tau, x, y):
        tt, rt = q * tau, q * np.sqrt(x * x + y * y)
        rho = np.arcsinh(-(1 - tt**2 + rt**2) / (2 * tt))
        th = np.arctan2(2 * rt, 1 + tt**2 - rt**2)
        return rho, th
    tt, rt = q * tau, q * np.sqrt(x**2 + y**2)
    S = 1 + tt**2 + rt**2; D = S**2 - 4 * tt**2 * rt**2
    cosh_rho = np.sqrt(D) / (2 * tt)
    ana = dict(drho_dt=S / (2 * tt**2 * cosh_rho), drho_dr=-rt / (tt * cosh_rho),
               dth_dt=-4 * tt * rt / D, dth_dr=2 * S / D)
    r0, t0 = rho_theta(tau, x, y)
    r1, t1 = rho_theta(tau + eps / q, x, y)                      # d/dtau~ = (1/q) d/dtau
    rr = np.sqrt(x**2 + y**2)
    r2, t2 = rho_theta(tau, x * (1 + eps / (q * rr)), y * (1 + eps / (q * rr)))
    check('Jacobian d(rho)/d(tau~) (FD)', abs((r1 - r0) / eps - ana['drho_dt'])[0], 1e-4)
    check('Jacobian d(theta)/d(tau~) (FD)', abs((t1 - t0) / eps - ana['dth_dt'])[0], 1e-4)
    check('Jacobian d(rho)/d(r~) (FD)', abs((r2 - r0) / eps - ana['drho_dr'])[0], 1e-4)
    check('Jacobian d(theta)/d(r~) (FD)', abs((t2 - t0) / eps - ana['dth_dr'])[0], 1e-4)

    # (b) ideal limit vs closed form, Eq. (15):  T = That0 / (tau cosh^{2/3} rho)
    That_fn, pibar_fn = solve_de_sitter(That0=1.2, eta_s=1e-12)
    tau = np.linspace(1.0, 4.9, 25)[None, None, :]
    xg = np.linspace(-6.3, 6.3, 21)[:, None, None]
    yg = np.linspace(-6.3, 6.3, 21)[None, :, None]
    f = gubser_fields(tau, xg, yg, q=1.0, That_fn=That_fn, pibar_fn=pibar_fn,
                      a_e=A_E_DEFAULT)
    T_exact = 1.2 / (tau * np.cosh(f['_rho'])**(2.0 / 3.0))
    check('ideal limit: max rel. error vs Eq. (15)',
          float(np.max(np.abs(f['T'] - T_exact) / T_exact)), 1e-6)

    # (c) viscous solution: tensor identities on the Milne grid
    That_fn, pibar_fn = solve_de_sitter(That0=1.2, eta_s=0.2, c_relax=5.0)
    f = gubser_fields(tau, xg, yg, q=1.0, That_fn=That_fn, pibar_fn=pibar_fn,
                      a_e=A_E_DEFAULT)
    norm = np.abs(f['pitautau']) + np.abs(f['pixx']) + np.abs(f['piyy']) \
        + np.abs(f['tau2_pietaeta']) + 1e-300
    trace = -f['pitautau'] + f['pixx'] + f['piyy'] + f['tau2_pietaeta']
    check('tracelessness  max |pi^mu_mu| / ||pi||',
          float(np.max(np.abs(trace) / norm)), 1e-9)
    # transversality u_mu pi^{mu nu} = 0 (Cartesian Milne, g = diag(-1,1,1,tau^2))
    u_t, u_x, u_y = -f['utau'], f['ux'], f['uy']
    res_t = u_t * f['pitautau'] + u_x * f['pitaux'] + u_y * f['pitauy']
    res_x = u_t * f['pitaux'] + u_x * f['pixx'] + u_y * f['pixy']
    res_y = u_t * f['pitauy'] + u_x * f['pixy'] + u_y * f['piyy']
    for nm, r in (('tau', res_t), ('x', res_x), ('y', res_y)):
        check(f'transversality max |u_mu pi^(mu {nm})| / ||pi||',
              float(np.max(np.abs(r) / norm)), 1e-9)

    # (d) IS asymptotics: |pibar| -> sqrt(1/c)  (paper Sec. III)
    pb = abs(float(pibar_fn(-28.0)))
    check('|pibar(rho -> -inf)| vs sqrt(1/c)', abs(pb - np.sqrt(1 / 5.0)), 5e-2)

    # (e) HDF5 schema round-trip against read_3d_hdf5.read_3d_data_hdf5
    import os, tempfile
    grid = dict(DEFAULT_GRID, nx=16, ny=16, neta=4, ntau=10)
    params = sample_params(rng, 2)
    events = [make_event(p, grid, DEFAULT_CHANNELS) for p in params]
    tmpdir_owned = tmpdir is None
    tmpdir = tempfile.mkdtemp(prefix='gubser_selftest_') if tmpdir_owned else str(tmpdir)
    h5path = os.path.join(tmpdir, '_gubser_selftest.h5')
    write_hdf5(h5path, events, grid, DEFAULT_CHANNELS, 'fm', params)
    try:
        # loc_libs is this file's grandparent; the reader lives there.
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from read_3d_hdf5 import read_3d_data_hdf5
        dat = read_3d_data_hdf5(h5path)
        check('reader round-trip max |arr diff|',
              float(np.max(np.abs(dat['arr'][0] - events[0][0]))), 1e-12)
        assert dat['arr'].shape == (2, 4, 16, 16, 4, 10)
    except ImportError:
        print("  [SKIP] project reader not importable here; raw h5py check instead")
        import h5py
        with h5py.File(h5path, 'r') as hf:
            check('h5py round-trip max |arr diff|',
                  float(np.max(np.abs(hf['arr'][0] - events[0][0]))), 1e-12)
    finally:
        if tmpdir_owned:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    print('selftest:', 'ALL PASS' if ok else 'FAILURES PRESENT')
    return ok


def plot_fig2(out_png):
    """Reproduce Fig. 2 of arXiv:1307.6130: T(r) and tau^2 pi^{eta eta}(r)
    at tau = 1.2, 1.5, 2.0 fm for eta/s = 0.2, c = 5, That0 = 1.2, q = 1/fm."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    That_fn, pibar_fn = solve_de_sitter(That0=1.2, eta_s=0.2, c_relax=5.0)
    r = np.linspace(0, 5, 400)
    styles = {1.2: ('k', '-'), 1.5: ('b', '--'), 2.0: ('r', '-.')}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for tau, (col, ls) in styles.items():
        # GeV units for direct overlay on the paper: T[fm^-1]*hbarc -> GeV,
        # tau^2 pi^{eta eta}[fm^-4]*hbarc -> GeV/fm^3.
        f = gubser_fields(np.array([tau]), r, np.zeros_like(r), 1.0,
                          That_fn, pibar_fn, A_E_DEFAULT)
        ax[0].plot(r, f['T'] * HBARC, col + ls, label=f'$\\tau = {tau}$ fm')
        ax[1].plot(r, f['tau2_pietaeta'] * HBARC, col + ls,
                   label=f'$\\tau = {tau}$ fm')
    ax[0].set_xlabel('radius (fm)'); ax[0].set_ylabel('T (GeV)')
    ax[0].set_xlim(0, 5); ax[0].set_ylim(0.04, 0.20)
    ax[1].set_xlabel('radius (fm)')
    ax[1].set_ylabel(r'$\tau^2 \pi^{\xi\xi}$ (GeV/fm$^3$)')
    ax[1].set_xlim(0, 5); ax[1].set_ylim(0.0, 0.21)
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.suptitle('Israel-Stewart Gubser flow (this generator, GeV units) '
                 'vs. arXiv:1307.6130 Fig. 2:  '
                 r'$\eta/s=0.2$, $c=5$, $\hat T_0=1.2$, $q=1$ fm$^{-1}$')
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"Wrote {out_png}")


# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='gubser_benchmark.h5')
    p.add_argument('--root-out', default=None,
                   help='also write a ROOT file (requires PyROOT)')
    p.add_argument('--nevents', type=int, default=64)
    p.add_argument('--channels', default=','.join(DEFAULT_CHANNELS))
    p.add_argument('--units', choices=('fm', 'GeV'), default='fm')
    p.add_argument('--seed', type=int, default=20260609)
    p.add_argument('--nx', type=int, default=DEFAULT_GRID['nx'])
    p.add_argument('--ny', type=int, default=DEFAULT_GRID['ny'])
    p.add_argument('--neta', type=int, default=DEFAULT_GRID['neta'])
    p.add_argument('--ntau', type=int, default=DEFAULT_GRID['ntau'])
    p.add_argument('--dx', type=float, default=DEFAULT_GRID['dx'])
    p.add_argument('--dtau', type=float, default=DEFAULT_GRID['dtau'])
    p.add_argument('--tau-min', type=float, default=DEFAULT_GRID['tau_min'])
    p.add_argument('--q-range', type=float, nargs=2, default=(0.8, 1.5))
    p.add_argument('--That0-range', type=float, nargs=2, default=(1.0, 1.6))
    p.add_argument('--eta-s-range', type=float, nargs=2, default=(0.08, 0.30))
    p.add_argument('--c-relax', type=float, default=5.0)
    p.add_argument('--pad-after-freezeout', action='store_true',
                   help='zero-fill tau steps past freeze-out (mimics MUSIC files)')
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--plot', metavar='PNG',
                   help='reproduce Fig. 2 of arXiv:1307.6130 and exit')
    args = p.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.plot:
        plot_fig2(args.plot)
        return

    grid = dict(DEFAULT_GRID)
    grid.update(nx=args.nx, ny=args.ny, neta=args.neta, ntau=args.ntau,
                dx=args.dx, dy=args.dx, dtau=args.dtau, tau_min=args.tau_min)
    grid['x_min'] = -0.5 * (grid['nx'] - 1) * grid['dx']     # symmetric grid
    grid['y_min'] = -0.5 * (grid['ny'] - 1) * grid['dy']
    grid['eta_min'] = -0.5 * (grid['neta'] - 1) * grid['deta']

    channels = tuple(c.strip() for c in args.channels.split(','))
    rng = np.random.default_rng(args.seed)
    params = sample_params(rng, args.nevents, q_range=tuple(args.q_range),
                           That0_range=tuple(args.That0_range),
                           eta_s_range=tuple(args.eta_s_range),
                           c_relax=args.c_relax)
    events = []
    for i, prm in enumerate(params):
        events.append(make_event(prm, grid, channels, units=args.units,
                                 pad_after_freezeout=args.pad_after_freezeout))
        if (i + 1) % 16 == 0 or i == args.nevents - 1:
            print(f"  generated {i + 1}/{args.nevents} events")
    write_hdf5(args.out, events, grid, channels, args.units, params)
    if args.root_out:
        write_root(args.root_out, events, grid, channels)


if __name__ == '__main__':
    main()
