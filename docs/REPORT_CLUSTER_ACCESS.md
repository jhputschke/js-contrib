# Choosing NSF ACCESS GPU resources for the two-stage production

Which ACCESS GPU resource to ask for, how many credits a campaign needs, and how to run on it.
The workload is stage 1 of `prod_AuAu_0_10_jet`: the Au+Au two-stage campaign with
`cu124-final_optim` (C4, D1, A12). Stage 2 (hadronization) is CPU-only and is covered by the
CPU and storage request.

Written 2026-10-13. **None of the ACCESS machines has been run.** Every rate below is an
estimate scaled from the measured 2 × RTX 3090 box ([BENCHMARK_2x3090.md](BENCHMARK_2x3090.md))
and the GB10 ([BENCHMARK_GB10.md](BENCHMARK_GB10.md)), with the method of
[REPORT_SUMMARY_CLUSTER.md](REPORT_SUMMARY_CLUSTER.md). It supersedes the per-node numbers in
[Plans/Final_Optims.md](Plans/Final_Optims.md) §6, which predate D1 and C4.

**Summary**
- **The job is limited by the CPU, not the GPU:** an Au+Au event needs about 2 GPU-seconds
  (RTX 3090) and about 35 host core-seconds. The cores and memory that come with each GPU
  decide the rate more than the GPU model does.
- **ACCESS prices GPUs about by speed:** on well-sized nodes every option gives
  **15–25 events per ACCESS credit**. An H100 costs twice as much per hour as an A100 and
  delivers about twice the events, provided it has at least about 24 cores.
- **First choice: NCSA DeltaAI (GH200).** It has 72 Grace cores per H100, the largest H100 pool
  (608 GPUs) and the best events per credit (about 19–22). The arm64 image and the GB10
  settings carry over.
- **Second choice: NCSA Delta A40.** Same site, same `/work/hdd` filesystem, the x86 image of the
  3090 box, and the A40 is the 3090's chip (GA102), so the measured numbers transfer most
  directly. About 1,000–1,400 events/h per A40, limited by 16 cores and 62.5 GB per GPU; 15–21
  events per credit at the A100 rate, ~30–40 if the (unstated) A40 discount halves it (§6).
  Purdue Anvil AI (H100, x86, 24 cores per GPU) is the alternative to both.
- **GPU credits are small, storage is not:** 1 million Au+Au events need about 50,000–55,000
  credits of GPU time, well within an Explore allocation (400,000 credits). The files take
  100–240 TB. They go to our OSDF space, so on site only a scratch buffer is needed: ask for
  20–50 TB of DeltaAI `/work/hdd` (the default is 1 TB; up to 100 TB on request), §8.

---

## 1. The workload per event

Measured on the 2 × RTX 3090 box (Threadripper 3960X, Zen 2) with `cu124-final_optim`, campaign
setting (10 jobs per GPU, `--write-particlize both`, `--pthat-bins 10-20,20-30,30-40
--jets-per-bin 5`), [BENCHMARK_2x3090.md](BENCHMARK_2x3090.md):

| per Au+Au event | value |
|---|---|
| GPU time (RTX 3090) | 2.09 GPU-s |
| host CPU time (Zen 2 under load) | 35 core-s |
| wall time in a loaded job | ~25 s |
| host memory per job (plan) | ~7 GB with the pair file, ~5 GB with `--particlize-only` |
| GPU memory per job | ~1 GB |
| disk | 0.243 GB (pair + particlize), 0.097 GB (`--particlize-only`) |
| measured rate | 1,080 events/h per GPU, host-bound (GPUs 63 % busy) |

**Where the CPU time goes** ([Plans/Final_Optims.md](Plans/Final_Optims.md) §4, job alone):

| item | core-s/event |
|---|---|
| freeze-out search, both legs | 8–12 |
| pair-file compression | ~6 (none with `--particlize-only`) |
| droplet deposition | 4–5 |
| writer: frame read, resample | ~3 |
| launches, syncs, copies, per-event init | 3–5 |

The CPU needs about 17× more time than the GPU, so the cores that come with a GPU decide
its rate.

---

## 2. Method

Per GPU, the expected rate is the smallest of three limits:

- **GPU limit** = 0.9 × 3600 / (GPU-s per event). GPU-s are scaled from the 3090 with a
  two-term fit to the 3090 and the GB10: about 78 % of the time scales with compute, 22 % with
  memory bandwidth ([REPORT_SUMMARY_CLUSTER.md](REPORT_SUMMARY_CLUSTER.md)).
- **CPU limit:** about 85 events/h per Zen 2-class physical core, already including 15 %
  headroom. Newer cores (Zen 3/4, Sapphire Rapids, Neoverse V2) count 1.2–1.5×. The GB10's
  X925 cores need about half the core-seconds of a Zen 2 core
  ([Plans/REPORT_CPU_GPU.md](Plans/REPORT_CPU_GPU.md)).
- **Memory limit:** the jobs that fit in the usable host memory per GPU (Slurm gives 10–15 %
  less than installed), at ~7 GB per job (pair file) or ~5 GB (`--particlize-only`). By
  Little's law, a GPU needs about events/h × (s/event in a loaded job) / 3600 jobs to stay busy.

**GPU time per event relative to the RTX 3090:**

| GPU | compute (SMs × clock) | memory bandwidth | time vs 3090 | GPU-s/event | GPU limit, events/h |
|---|---|---|---|---|---|
| RTX 3090 | 82 × 1.7 GHz | 936 GB/s | 1 (measured) | 2.09 | ~1,550 |
| A10 | 72 × 1.5–1.7 GHz (150 W) | 600 GB/s | 1.2–1.4 | 2.5–2.9 | 1,100–1,300 |
| A40 | 84 × ~1.6 GHz | 696 GB/s | 1.0–1.2 | 2.1–2.5 | 1,300–1,550 |
| A100 40 GB | 108 × 1.4 GHz (64 FP32 lanes/SM) | 1.6 TB/s | 0.9–1.4 | 1.9–2.9 | 1,100–1,700 |
| V100 | 80 × 1.5 GHz (64 lanes/SM) | 0.9 TB/s | 1.5–2.0 | 3.1–4.2 | 770–1,040 |
| H100 / GH200 | 132 × ~1.9 GHz | 3.3–4 TB/s | 0.47–0.55 | 1.0–1.15 | 2,800–3,200 |

**Cores needed to keep one GPU busy:** an A40 or A100 needs about 16–20 Zen 2-class cores, an
H100 about 30–40. With fewer cores the GPU waits.

---

## 3. ACCESS exchange rates

From the [ACCESS exchange calculator](https://allocations.access-ci.org/exchange_calculator)
(rates taken from the page source on 2026-10-13; 1 credit ≈ 1 core-hour):

| resource | credits per unit | unit |
|---|---|---|
| NCSA DeltaAI (GH200) | 133.3 | GPU-hour |
| Purdue Anvil AI (H100) | 133.3 | GPU-hour |
| Texas Tech REPACSS GPU (H100 NVL) | 100 | GPU-hour |
| Purdue Anvil GPU (A100) | 67.3 | GPU-hour |
| NCSA Delta GPU (A100 base rate) | 66.7 | GPU-hour |
| NCAR Derecho-GPU (A100) | 66.7 | GPU-hour |
| GATech Nexus (B200, RTX Blackwell; listed as "coming soon") | 66.7 | GPU-hour |
| Indiana Jetstream2 GPU | 1 | SU (`g3.xl` = 1 A100 = 64 SU/h; H100 `g5.xl` = 128 SU/h) |
| Prototype National Research Platform | 60 | GPU-hour |
| SDSC Expanse GPU (V100) | 53.8 | GPU-hour |
| PSC Bridges-2 GPU | 53.8 | GPU-hour |
| NCSA Delta CPU, Anvil CPU, Expanse CPU, Bridges-2 RM | 1 | core-hour |
| Open Storage Network | 3,000 | TB |
| NCSA Granite tape | 1,000 | TB |

**Delta details** ([Delta job accounting](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/job_accounting.html)):
- **Base unit:** one Delta GPU unit is one A100 GPU-hour on the quad-A100 nodes.
- **A40 discount:** A40s are charged less, by a factor the page doesn't give and the calculator
  doesn't show. Ask Delta support.
- **H200:** H200s cost 3× an A100.
- **Charging basis:** a job is charged for its cores or its share of memory, whichever is larger,
  so asking for more than 62.5 GB per GPU costs extra.

---

## 4. Estimated throughput and cost per resource

Per GPU, Au+Au campaign setting. "Share per GPU" is the node's cores and memory divided by its
GPUs.

| resource | share per GPU | limit | events/h per GPU | **events per credit** | notes |
|---|---|---|---|---|---|
| **NCSA DeltaAI** (GH200, 96 GB) | 72 Grace cores, 120 GB (~105 usable) | GPU (memory sets the jobs) | 2,500–3,000 (`--particlize-only`), 2,000–2,500 (pair) | **19–22** | arm64 image. 608 GPUs. First option (§5). |
| Purdue Anvil AI (H100 80 GB SXM) | 24 Xeon 8468 cores, 256 GB | cores | 2,200–2,700 | 16–20 | x86, `cu124` unchanged. 84 GPUs; cores and memory from a third-party listing. |
| NCAR Derecho-GPU (A100 40 GB) | 16 Milan cores, 128 GB | GPU / cores | 1,100–1,600 | 16–24 | Check whether NCAR's ACCESS share is open to nuclear physics. |
| Purdue Anvil GPU (A100) | 32 Milan cores (unverified) | GPU | 1,100–1,700 | 16–25 | 64 GPUs; ACCESS steers new requests to Anvil AI. |
| Jetstream2 `g3.xl` (A100 40 GB) | 32 vCPU, 120 GB | GPU | 1,100–1,500 | 17–23 | Cloud VM with Docker: `run_jobs.sh` and MPS as on a workstation. |
| NCSA Delta (A40 / A100 40 GB) | 16 Milan cores, 62.5 GB | memory, then cores | 1,000–1,400 (`--particlize-only`), 800–1,100 (pair) | 15–21 at the A100 rate; **~30–40 if A40s cost half** | x86, `cu124`. 400 A40 + 400 A100. Second option (§6). |
| Texas Tech REPACSS (H100 NVL) | 16 Xeon 6448Y cores, 128 GB | cores | 1,500–1,800 | 15–18 | Too few cores for an H100. |
| SDSC Expanse (V100 32 GB) | 10 Xeon 6248 cores, 96 GB | GPU / cores | 700–850 | 13–16 | |
| PSC Bridges-2 (8 × V100) | 5 cores | cores | ~400 | ~7 | |
| SDSC Cosmos (MI300A), TACC Stampede3 (Intel GPUs) | – | – | not usable | – | MUSIC4GPU has CUDA and Metal back ends only. |

**Are H100s overkill?**
- **Same cost per event:** where the node gives at least ~24 cores per H100, its higher rate is
  matched by its higher charge (Anvil AI, DeltaAI), and it needs half as many GPUs and jobs.
- **Wasted with too few cores:** with 16 cores (REPACSS) the H100 runs at the rate of an A100.
- **A100 vs A40:** an A100 adds nothing over an A40 for this job.
- **If the CPU work shrinks**, the H100 gains: `--particlize-only` (−6 core-s), the freeze-out
  search on the GPU (B2, −8–12) and droplet deposition on the GPU (B1′, −4–5) would bring 35
  core-s to about 15–18. An A40 would then be busy with ~10 cores, and an H100 with 24 cores
  would reach ~2,500–3,000 events/h against the A40's ~1,400.

---

## 5. First option: NCSA DeltaAI

**Why:**
- **Most cores per GPU:** each GH200 superchip has an H100 (96 GB HBM3) and a 72-core Grace
  CPU with 120 GB LPDDR5X; four per node.
- **Best events per credit** (19–22), and the largest H100 pool on ACCESS (608 GPUs).
- **Same kind of machine as the GB10:** ARM cores with coherent CPU–GPU memory, so the GB10
  settings (`MUSIC_CUDA_SYNC=block`, MPS, passive OpenMP) and the arm64 image carry over.
- **Spare cores for stage 2:** stage 1 needs only ~20–25 of the 72 cores. The charge covers the
  whole superchip, so the remaining ~45 cores can run hadronization (iSS) on the same
  allocation at no extra credit cost. Host memory is the limit: fewer stage-1 jobs, or stage 2
  in separate DeltaAI jobs.

**Settings per GH200** (one Slurm task per GPU, each with its own MPS daemon: the
one-container-per-GPU rule of the 3090 box):

```bash
# Slurm, per GPU:  --gpus=1  --cpus-per-task=72  --mem=100G
W=<your directory on /work/hdd>          # bound to /work in the container
PROD=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
mkdir -p $W/lbt_cache
apptainer exec --nv --pwd "$PROD" --bind "$W:/work" <arm64 image> \
  env OMP_NUM_THREADS=4 OMP_WAIT_POLICY=passive MUSIC_CUDA_SYNC=block \
      LBT_TABLE_CACHE=/work/lbt_cache/lbt_tables_v1_arm64.bin \
  ./run_jobs.sh -j 12 --mps --stagger 12 --campaign <name> NJOBS 45 0 /work/<name> \
      --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 \
      --write-particlize both --particlize-only
```

- **Jobs per GPU:** `-j 12–16` with `--particlize-only` (5 GB per job); `-j 10–12` with the pair
  file. Memory, not cores, sets this.
- **Long jobs:** 45 events per job. The stagger ramp costs ~10 % in 15-event jobs.
- **`--pwd` and `--bind`:** Apptainer starts in your home, not in the production folder, and
  only bound directories are writable; OUTDIR and the LBT cache are paths inside the container.
- **The LBT cache:** `LBT_TABLE_CACHE` lets all jobs of a node share one read-only copy of
  LBT's 0.66 GB of tables (−0.6 GB per job, already in the 5–7 GB per job). The first job
  writes the file; make it once with a 1-event job before the first array, so the tasks don't
  all write it at once. Setup, check and versions: [README_2stage.md](README_2stage.md) §2,
  [*The LBT table cache with Apptainer and SLURM*](README_2stage.md#the-lbt-table-cache-with-apptainer-and-slurm).

**The image:**
- **Use the published arm64 images of 2026-10-14:** `cu130` = `cu130-20261013-9e84450`
  (`sha256:76c02874…`, R580+ driver) or `cu126` = `cu126-20261013-9e84450`
  (`sha256:2e870cac…`, R560+). Both are built from the merged branches (X-SCAPE `9e84450`,
  MUSIC4GPU `46dad43`, iSS `98deaff`), with Matter's qhat guard and the iSS leak fix, and both
  include sm_90. Pick by DeltaAI's driver (`nvidia-smi`); full digests in
  [README_2stage.md](README_2stage.md) §1.
- **A locally built GB10 image** must include sm_90 as well: the GB10 is sm_121, the GH200
  sm_90, and an image built only for the GB10 can't run MUSIC4GPU's kernels on DeltaAI. Check
  with `cuobjdump --list-elf` on `libmusic`, or run a 1-event job.
- **Not run yet:** neither the 2026-10-14 images nor anything on a GH200. Au+Au output of these
  images is bit-identical to `cu124-final_optim`, the image the 3090 numbers come from.

---

## 6. Second option: NCSA Delta A40

**Why:**
- **Closest to the measured machine:** the A40 is the RTX 3090's chip (GA102, sm_86) with 84 SMs
  and 48 GB, slightly lower clock and memory bandwidth (696 vs 936 GB/s). The x86 `cu124` image of
  the 3090 box runs unchanged. Of all candidates, its estimate needs the least extrapolation.
- **Same site as DeltaAI:** one NCSA allocation can exchange credits on both; Delta and DeltaAI
  share `/work/hdd`, so output and the upload setup are common (§8). Keep one LBT cache file
  per architecture (§8).
- **Large pool:** 100 nodes × 4 A40 (and 100 × 4 A100, which give the same rate for this job at
  the full A100 charge).
- **Possibly the cheapest per event:** Delta charges A40s less than its A100 base rate (66.7
  credits per GPU-hour), by a factor neither the accounting page nor the calculator gives. At half
  the rate an A40 gives ~30–40 events per credit, against 19–22 on DeltaAI. **Ask Delta support
  for the factor before choosing between the two.**

**The node and its limits:**
- **Node:** 4 × A40 (48 GB GDDR6), one 64-core AMD EPYC 7763 (Milan, Zen 3, 2.55 GHz), 256 GB
  DDR4. A GPU's share, and one GPU's charge: **16 cores and 62.5 GB** (decimal GB).
- **Charging:** cores or memory fraction, whichever is larger. More than 62.5 GB per GPU is
  charged as more GPUs, so the jobs have to fit in one share.
- **GPU limit:** 1,300–1,550 events/h (§2).
- **CPU limit:** 16 Milan cores × ~85 × 1.15–1.5 ≈ 1,560–2,040 events/h.
- **Memory limit (the one that binds):** ~60 GB fits 10–11 jobs with `--particlize-only` (~5 GB
  each) or 8 with the pair file (~7 GB). By Little's law (25–35 s per event in a loaded job):
  **~1,000–1,400 events/h** with `--particlize-only`, ~800–1,100 with the pair file. GPU ~65–90 %
  busy.
- **Per node:** 4,000–5,600 events/h with `--particlize-only`.
- **No spare cores:** unlike DeltaAI, stage 1 uses all 16. Stage 2 goes to Delta CPU (1 credit
  per core-hour, same filesystem).

**Settings per A40** (one Slurm task per GPU, each with its own MPS daemon):

```bash
# Slurm, per GPU:  --partition=gpuA40x4  --account=<project>-delta-gpu
#                  --gpus-per-node=1  --cpus-per-task=16  --mem=58G   (58 GiB = 62.3 GB: one GPU's share)
W=<your directory on /work/hdd>          # bound to /work in the container
PROD=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
mkdir -p $W/lbt_cache
apptainer exec --nv --pwd "$PROD" --bind "$W:/work" xscape-prod_cu124-20261013-9e84450.sif \
  env OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive MUSIC_CUDA_SYNC=block \
      LBT_TABLE_CACHE=/work/lbt_cache/lbt_tables_v1_amd64.bin \
  ./run_jobs.sh -j 10 --mps --stagger 12 --campaign <name> NJOBS 45 0 /work/<name> \
      --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 \
      --write-particlize both --particlize-only
```

- **Jobs per GPU:** `-j 10` with `--particlize-only`, `-j 8` with the pair file. Memory, not
  cores, sets this; watch the job's peak with `sacct -o MaxRSS` in the benchmark and lower `-j`
  before Slurm's OOM killer does it.
- **Threads:** 2 per job, 20 threads on 16 cores (the 3090 box ran 10 jobs × 3 threads per GPU on
  12 cores, 24 hardware threads).
- **MPS** needs the GPU in the `Default` compute mode (`nvidia-smi -q | grep "Compute Mode"`).
  If it is `Exclusive_Process`, drop `--mps` and lose the 5–7 % it gives.
- **Long jobs:** 45 events per job, as on DeltaAI.
- **`--pwd`, `--bind` and the LBT cache:** as on DeltaAI (§5), with an `amd64` cache file.

**The image:**
- **`cu124` = `cu124-20261013-9e84450`**, `sha256:2cf57f7556dcfff0289c1c46b9b188b58a3ce8026b2733cc4a17f90bca761b58`
  (amd64, sm_70–90 + PTX, R550+ driver), from the merged branches with Matter's qhat guard and
  the iSS leak fix. Pull once into a `.sif`:
  ```bash
  apptainer pull xscape-prod_cu124-20261013-9e84450.sif \
      docker://jhputschke/xscape-prod@sha256:2cf57f7556dcfff0289c1c46b9b188b58a3ce8026b2733cc4a17f90bca761b58
  ```
- **Not run yet** on an A40; Au+Au output is bit-identical to `cu124-final_optim`, the image the
  3090 numbers come from.

---

## 7. Sizing the request

| item | 1 million Au+Au events |
|---|---|
| DeltaAI GPU time | ~350–400 GH200-hours ≈ **45,000–55,000 credits** |
| Delta A40 time | ~700–1,000 A40-hours ≈ 47,000–67,000 credits at the A100 rate; less by the A40 discount |
| permanent storage, `--particlize-only` | ~100 TB, on our OSDF space (no ACCESS credits; OSN would cost ~300,000 credits, Granite tape ~100,000) |
| permanent storage, pair + particlize | ~240 TB, on OSDF |
| scratch buffer on site | 20–50 TB of `/work/hdd` (shared by DeltaAI and Delta), free on request (§8) |
| stage 2 (hadronization) | CPU core-hours at 1 credit each, Delta CPU or Anvil CPU (same filesystem as the GPU partitions); or the spare Grace cores (§5) |

- **Explore** (400,000 credits) covers the benchmark and a first campaign of several million
  events of GPU time. With the files on OSDF, no ACCESS storage credits are needed.

**Before the main request:** use part of an Explore allocation for one benchmark per candidate:
one GH200 (`-j 12` vs `-j 16`) and one Delta A40 (`-j 8` vs `-j 10`), 15 events per job,
`--particlize-only`. That replaces these estimates with measurements, fixes the jobs per GPU,
and, with Delta's A40 factor, decides between the two.

---

## 8. Scratch space and the upload to OSDF

The files are kept on our OSDF space (`osdf:///fno4hic/...`), so the cluster's disk only has
to hold them until they are uploaded. Purge rules then don't matter.

**Default storage on the candidate sites:**

| site | scratch / work area | project space | purged | node-local disk |
|---|---|---|---|---|
| **NCSA DeltaAI** | `/work/hdd` **1 TB** default, up to **100 TB** by support request; `/work/nvme` 1 TB, up to 100 TB | `/projects` 500 GB, up to 25 TB (large requests may carry a fee) | no | `/tmp` ~3.9 TB, wiped after each job |
| **NCSA Delta** | `/work/hdd` 1 TB, up to 100 TB (the same filesystem as DeltaAI) | 500 GB, up to 25 TB | no | – |
| **Purdue Anvil** (CPU, GPU, AI) | `/anvil/scratch` **100 TB per user**, 1 M files | 5 TB per allocation | scratch: files older than 30 days | – |

Elsewhere the pattern is similar: tens of GB of home, 0.5–5 TB of project space, and 10–100+ TB
of scratch, often purged after 30–90 days.

**How fast production fills it** (rates of §5 and §6):

| output | per GH200 | one DeltaAI node (4 GH200) | per A40 | one Delta A40 node (4 A40) | default 1 TB full after (one node: DeltaAI / Delta) |
|---|---|---|---|---|---|
| `--particlize-only` (0.097 GB/event) | 0.25–0.3 TB/h | ~1.1 TB/h | 0.1–0.14 TB/h | ~0.5 TB/h | **~1 h / ~2 h** |
| pair + particlize (0.243 GB/event) | 0.5–0.6 TB/h | ~2.2 TB/h | 0.2–0.27 TB/h | ~0.9 TB/h | ~30 min / ~1 h |

**What to do:**
- **Ask for 20–50 TB of `/work/hdd`** with the DeltaAI allocation (support request; free up to
  100 TB according to the NCSA docs). Delta sees the same filesystem, so one request covers
  both. On Anvil the 100 TB per user is enough as it is.
- **Upload while producing:** `upload_follow.py --delete pair|all` with `--keep-free`, and the
  disk guard `--min-free` on, as on the 3090 box ([README_2stage.md](README_2stage.md) §5,
  [utils/remote_transfer](../utils/remote_transfer/README.md)).
- **The upload rate decides the buffer:** the 3090 box reached ~290 GB/h to OSDF; one DeltaAI
  node writes ~1.1 TB/h with `--particlize-only`. If the upload is slower than production, the
  buffer has to absorb the difference for the whole campaign: e.g. (1.1 − 0.3) TB/h × 10 h ≈
  8 TB per node. Then either more scratch, more parallel upload streams, or fewer nodes at once.
- **Check how DeltaAI and Delta reach OSDF:** whether compute nodes have outbound network access, or the
  upload has to run on a login or data-transfer node (the usual case), with `upload_follow.py`
  following the output directory on the shared `/work/hdd`. Measure the rate in the first
  benchmark.
- **The LBT table cache** goes on `/work/hdd` next to the output (`$W/lbt_cache/`), one
  0.66 GB file for all nodes of a campaign; each node reads it once into its page cache.
  **One file per architecture:** `lbt_tables_v1_arm64.bin` for DeltaAI, `…_amd64.bin` for
  Delta. Both are made from the same tables and are probably identical, but that has not
  been checked, and the file's header checks only the layout and the table files' sizes: a
  file from the other image would be used without a warning. Make each once before the first array, and check the job logs for
  `LBT: tables mapped read-only` ([README_2stage.md](README_2stage.md#the-lbt-table-cache-with-apptainer-and-slurm)).
  DeltaAI's node-local `/tmp` works too (`LBT_TABLE_CACHE` under a bound `/tmp`), at 6–11 s
  more for the first job of every Slurm job, since `/tmp` is wiped at the end.
- **Node-local `/tmp`** (3.9 TB, fast) is wiped when the job ends. Use it only if the job waits
  for its uploads to finish before it exits.

---

## 9. For comparison: commercial cloud

Same method; useful for pricing against ACCESS or for a quick campaign outside an allocation.

**Lambda 8 × A100 40 GB SXM4, 124 vCPU, 1,800 GiB RAM, 6 TiB SSD:**
- **Rate:** about **5,000–7,500 events/h** for the node (650–950 per GPU), limited by its
  ~62 physical cores. The GPUs will be about 40–60 % busy.
- **Settings:** `-j 8` per GPU (64 jobs), `OMP_NUM_THREADS=2`, MPS, `MUSIC_CUDA_SYNC=block`,
  `--lbt-cache`, `--stagger 12`; ~450 GB host memory.
- **Disk:** at ~6,000 events/h the pair file fills 6 TiB in ~4.5 h, `--particlize-only` in
  ~11 h. 1.5 TB holds ~6,200 events with the pair file, ~15,500 with `--particlize-only`.
- **Launcher:** `launch_2gpu.sh` drives GPUs 0 and 1 only. For 8 GPUs run one container per
  GPU by hand (they share the seed registry `OUTDIR/seeds_used.tsv`), or extend the launcher to a
  list of GPUs.

**One A10 (24 GB, 150 W):**
- **Rate:** about **1,100–1,300 events/h** with at least 14–16 cores per GPU; ~680 with 8
  cores (e.g. AWS g5.4xlarge), ~1,000 with 12.
- **Settings:** `-j 10`, `OMP_NUM_THREADS=2–3`; ~80 GB host memory (pair file).

---

## Sources

- [ACCESS exchange calculator](https://allocations.access-ci.org/exchange_calculator)
- [ACCESS active resources](https://operations.access-ci.org/resources/access-allocated)
- [DeltaAI system architecture](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/architecture.html),
  [DeltaAI hardware](https://delta.ncsa.illinois.edu/deltaai-hardware-and-network/),
  [DeltaAI allocations](https://delta.ncsa.illinois.edu/deltaai-allocations/)
- [Delta hardware](https://delta.ncsa.illinois.edu/hardware_and_network/),
  [Delta job accounting](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/job_accounting.html),
  [Delta allocations](https://delta.ncsa.illinois.edu/delta-allocations/)
- [DeltaAI data management](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/data-mgmt.html),
  [Delta data management](https://docs.ncsa.illinois.edu/systems/delta/en/rh9/user_guide/data_mgmt.html)
- [Anvil file systems](https://www.rcac.purdue.edu/knowledge/anvil/storage/filesystems),
  [Anvil policies](https://docs.rcac.purdue.edu/userguides/anvil/policies/)
- [Purdue Anvil AI (ACCESS)](https://operations.access-ci.org/node/872),
  [Anvil AI announcement](https://rcac.purdue.edu/news/7054)
- [Derecho GPU (ACCESS)](https://support.access-ci.org/documentation/resources/derecho-gpu),
  [NCAR Derecho](https://cisl.ucar.edu/capabilities/derecho)
- [Jetstream2 instance flavors](https://docs.jetstream-cloud.org/general/instance-flavors/),
  [ACCESS credits and Jetstream2](https://docs.jetstream-cloud.org/general/access/)
- [REPACSS GPU (ACCESS)](https://support.access-ci.org/documentation/resources/repacss-gpu)
- [GATech Nexus (ACCESS)](https://operations.access-ci.org/node/978)
- In this repository: [BENCHMARK_2x3090.md](BENCHMARK_2x3090.md),
  [BENCHMARK_GB10.md](BENCHMARK_GB10.md), [REPORT_SUMMARY_CLUSTER.md](REPORT_SUMMARY_CLUSTER.md),
  [Plans/Final_Optims.md](Plans/Final_Optims.md), [Plans/REPORT_CPU_GPU.md](Plans/REPORT_CPU_GPU.md),
  [README_2stage.md](README_2stage.md)
