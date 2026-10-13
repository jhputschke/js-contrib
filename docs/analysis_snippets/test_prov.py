import sys
CAMPAIGN = sys.argv[1]
import glob, h5py, hdf5plugin, yaml               # hdf5plugin: the Blosc filter
import xml.etree.ElementTree as ET

for path in sorted(glob.glob(f"{CAMPAIGN}/**/*_particlize.h5", recursive=True)):
    with h5py.File(path, "r") as f:
        A = f.attrs
        late = ET.fromstring(A["prod_user_xml"]).findtext(".//Hydro/MUSIC/late_delta_tau")
        grid = yaml.safe_load(A["prod_grid_yaml"])["grid"]
        print(path.rsplit("/", 1)[-1], "| seed", A["prod_seed"],
              "| late_delta_tau", late.strip() if late else "absent (constant step)",
              "| grid x", grid["x"], "| reuse", A.get("prod_reuse"),
              "| windows", A.get("pthat_bins", "none").tolist() if "pthat_bins" in A else "none")
