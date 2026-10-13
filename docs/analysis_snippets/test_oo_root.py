import sys
CAMPAIGN, CONV = sys.argv[1], sys.argv[2]
import glob, subprocess, h5py, hdf5plugin, numpy as np, uproot

for jf in sorted(glob.glob(f"{CAMPAIGN}/**/*_hadrons_jet_frag.h5", recursive=True)):
    stem = jf[: -len("_hadrons_jet_frag.h5")]
    for tag in ("bulk_bg", "jet_frag"):                     # one ntuple per tag, one entry per sample
        subprocess.run([sys.executable, CONV, f"{stem}_hadrons_{tag}.h5",
                        "-o", f"{stem}_{tag}.root", "--no-x"], check=True)
    with h5py.File(jf, "r") as f:                           # the event record (jet file)
        ev = {k: f["units"][k][:] for k in f["units"]}
        win = {k: np.atleast_1d(f.attrs[k]) for k in ("pthat_bin_sigma_gen", "pthat_bin_n_accepted")
               if k in f.attrs}
    with h5py.File(f"{stem}_hadrons_bulk_bg.h5", "r") as f:  # unit i = event i in both files
        ev["n_samples_bg"] = f["units/n_samples"][:]
    ev["n_samples_jet"] = ev.pop("n_samples")
    with uproot.recreate(f"{stem}_events.root") as out:
        out["events"] = {k: v.astype(np.float64 if v.dtype.kind == "f" else np.int64)
                         for k, v in ev.items()}
        if win:
            out["windows"] = {k: v.astype(np.float64) for k, v in win.items()}
print("ok")
