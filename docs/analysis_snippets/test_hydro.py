import os, sys
OUT = sys.argv[1]
STEM = sys.argv[2] if len(sys.argv) > 2 else "AuAu_0_10_jet_seed0001"
if not os.environ.get("MUSIC_EOS_TABLE"):
    sys.exit("set MUSIC_EOS_TABLE to MUSIC's hrg_hotqcd_eos_binary.dat (the analysis venv does)")

# --- snippet: hydro pair ---
import h5py, hdf5plugin, numpy as np            # hdf5plugin: the Blosc filter

with h5py.File(f"{OUT}/{STEM}.h5", "r") as f:
    A = f.attrs
    x   = A["x_min"]   + A["dx"]   * np.arange(A["nx"])      # cell centres [fm]
    y   = A["y_min"]   + A["dy"]   * np.arange(A["ny"])
    eta = A["eta_min"] + A["deta"] * np.arange(A["neta"])
    tau = A["tau_min"] + A["dtau"] * np.arange(f["arr"].shape[-1])   # 0.5, 0.6, ... fm/c

    i, k = 0, 40                                   # event 0, frame tau = 4.5 fm/c
    jet = f["arr"][i, :, :, :, :, k]               # (4, nx, ny, neta): e, vx, vy, vz
    bg  = f["arr_bg"][i, :, :, :, :, k]            # the same event without the jet
    de  = jet[0] - bg[0]                           # the wake in e [GeV/fm^3]
    n_jet, n_bg = f["ntau_freezeout"][i], f["ntau_freezeout_bg"][i]   # frames written
    print("de max", de.max(), "frames", n_jet, n_bg, "tau", tau[k])

    droplets = f["source/droplets"][f["source/offsets"][i]:f["source/offsets"][i + 1]]
    E_dep = droplets[:, 4].sum()                   # columns tau, x, y, eta, E, px, py, pz
    print("E_dep", E_dep, f["diag/E_droplets"][i])

    # e -> T with MUSIC's table (columns e, P, s, T)
    tab = np.fromfile(os.environ["MUSIC_EOS_TABLE"], dtype="<f8").reshape(-1, 4)
    tab = tab[np.argsort(tab[:, 0])]
    T = np.interp(jet[0], tab[:, 0], tab[:, 3])    # GeV
    print("T max", T.max(), "T at e_fo", np.interp(0.2342, tab[:, 0], tab[:, 3]))

    import re
    m = re.search(r"<late_delta_tau>\s*([\d.]+)", A["prod_user_xml"])
    print("late_delta_tau:", m.group(1) if m else "absent (constant step)")
    print(A["prod_seed"], A["prod_grid_yaml"].splitlines()[0])

# --- particlize ---
from jetscape.particlize_h5 import ParticlizeFile, SURFACE_COLUMNS, FINAL_PARTON_COLUMNS

with ParticlizeFile(f"{OUT}/{STEM}_particlize.h5") as pf:
    cells = pf.surface("jet", 0)                   # (N, 32) float32, SURFACE_COLUMNS
    cells_bg = pf.surface("bg", 0)                 # the background event 0 used
    partons = pf.partons(0)                        # (K, 14), FINAL_PARTON_COLUMNS
    ini = pf.initiators(0)                         # (J, 11): shower, pid, pstat, px, py, pz, E, x, y, z, t
    n_jet, n_bg = pf.events("n_cells_jet"), pf.events("n_cells_bg")
    print(cells.shape, cells_bg.shape, partons.shape, None if ini is None else ini.shape, n_jet, n_bg)
    T_col = SURFACE_COLUMNS.index("T")
    print("T on surface", cells[:, T_col].mean())
