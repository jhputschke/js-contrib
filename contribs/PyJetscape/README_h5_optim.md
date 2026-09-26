# HDF5 compression for hydro evolutions

This covers what the writers in js-contrib (PyJetscape, FastHydro) and FNO4d now use to
compress the `arr` / `arr_bg` evolution datasets, why, and how to change it. The same
file is in FNO4d as `README_h5_optim.md`.

## Summary

- **The default is now Blosc + zstd (level 3) with byte shuffle**, replacing lzf. It is
  lossless. Files are **1.4–1.6× smaller** than with lzf; a one-event AuAu 0–10% MUSIC
  pair file goes from 400 MB to 285 MB. Job wall time is unchanged.
  - With one thread, reads of MUSIC's tau-sliced chunks are about 15% slower than lzf.
  - With whole-event chunks and Blosc threads, reads are twice as fast as lzf.
  - `blosc-lz4` trades about 7% of the size gain for faster writes.
- **Optional lossy mode: `keep_bits=N`** rounds each float32 value to N mantissa bits
  before compressing. With `N=12`, the maximum relative error per value is 1.2e-4, and
  files are **2.5–3.2× smaller than with lzf** (the same pair file: 159 MB). Exact zeros
  stay exactly zero.
- **BITSHUFFLE does not help lossless data here.** On full-precision MUSIC and FastHydro
  output it compresses worse than plain byte shuffle. It helps only after rounding, and
  then only with LZ4.
- **Reading a new file needs the `hdf5plugin` package** in the reading environment.
  Importing `jetscape`, `fast_data`, `fasthydro` or FNO4d's `read_3d_hdf5` loads it
  automatically. Old lzf files still read as before.
- **Same picture on an Apple M3 Max:** the same ratios, with zstd writing ~10 % slower
  and reading ~20 % slower than lzf there, and the same bit-exactness. See
  [Apple M3 Max](#apple-m3-max).

## What was measured

All numbers are from the GB10 (20 cores) with 1 Blosc thread unless noted;
[Apple M3 Max](#apple-m3-max) repeats the MUSIC measurements on a Mac. The
evolution data was written the way the writers write it: tau-sliced chunks one tau frame
at a time, and whole-event chunks one event at a time. Reads came from the page cache,
so they measure decompression plus HDF5's copy into the output array, not disk speed.
Every lossless filter was checked to round-trip bit for bit. To reproduce:
[`utils/h5_compression_bench.py`](../../utils/h5_compression_bench.py).

### Lossless filters

Ratio is raw size divided by compressed size; write and read speeds are in MB/s of
uncompressed data. Each cell reads ratio / write speed / read speed.

| Filter | MUSIC `arr`, AuAu 0–10% (2 ev, 486 MB, chunks `(1,4,65,65,33,1)`) | MUSIC pair `arr_bg` (1 ev, 122 MB, chunks `(1,4,65,65,17,1)`) | FastHydro `arr` (2 ev, 647 MB, chunks `(1,4,65,65,33,145)`) |
|---|---|---|---|
| lzf (old default) | 1.11 / 410 / 720 | 1.25 / 490 / 950 | 1.25 / 410 / 920 |
| gzip 4 + shuffle | **1.60** / 80 / 320 | **1.77** / 90 / 360 | **2.02** / 84 / 290 |
| Blosc lz4 + shuffle | 1.47 / 860 / 620 | 1.63 / 1130 / 850 | 1.75 / 670 / 840 |
| Blosc lz4 + bitshuffle | 1.37 / 670 / 620 | 1.55 / 790 / 870 | 1.47 / 525 / 800 |
| Blosc zstd 3 + shuffle (**new default**) | 1.57 / 370 / 590 | 1.73 / 400 / 820 | 2.00 / 210 / 870 |
| Blosc zstd 3 + bitshuffle | 1.41 / 290 / 560 | 1.60 / 310 / 830 | 1.75 / 160 / 705 |

Tested and not adopted:

- Standalone Bitshuffle+LZ4 filter: 1.32×, read 308 MB/s (MUSIC `arr`).
- Blosc2 lz4 + bitshuffle: 1.20×, read 226 MB/s (MUSIC `arr`).
- Blosc lz4hc + bitshuffle: 1.43×, write 90 MB/s (MUSIC `arr`).

**Why lzf does so badly.** A float32 value is sign, exponent and mantissa bytes
interleaved. The low mantissa bytes of a hydro field are close to random, and a
byte-oriented codec like lzf sees no repetition in them. Byte shuffle regroups the
array so that all first bytes come first, then all second bytes, and so on. The
exponent bytes of neighbouring cells are nearly identical, so those groups compress
well.

**Why bitshuffle loses.** Bitshuffle regroups bit by bit instead of byte by byte. That
only helps when individual bit planes are predictable. In full-precision data the lower
~16 mantissa bits are noise, so splitting them into bit planes adds overhead and finds
nothing new.

### Threads

Blosc can decompress with several threads (`BLOSC_NTHREADS`). Leaving it unset gives
the same speed as 1 thread.

| FastHydro `arr`, whole-event chunks | Write, 1 thread | Write, 8 threads | Read, 1 thread | Read, 8 threads |
|---|---|---|---|---|
| lzf | 440 | 430 | 970 | 920 |
| Blosc zstd 3 + shuffle | 215 | **800** | 820 | **1800** |
| Blosc lz4 + shuffle | 730 | 1230 | 1060 | **1800** |

With whole-event chunks (fast_data, FastHydro), decompression is what limits reads, so
threads help. Uncompressed, the same read runs at 17 GB/s. With MUSIC's tau-sliced
chunks, 8 threads change little. There, every codec reads at roughly 600–950 MB/s,
because HDF5 must scatter each `(…, 1)` chunk into a tau-last array, element by
element. That copy, not the codec, is the limit (an uncompressed read of the same data
was even slower, 11 MB/s). The codec cannot fix this; a different chunk layout could
(see *Open points*).

### Lossy: rounding the mantissa (`keep_bits`)

A float32 value has 23 explicit mantissa bits. Keeping N of them, with round-to-nearest
and ties to even, bounds the relative error per value at 2^-(N+1). The dropped bits
become zeros, and zeros compress well.

| keep_bits | Max relative error per value |
|---|---|
| 10 | 4.9e-4 |
| 12 | 1.2e-4 |
| 14 | 3.1e-5 |
| 16 | 7.6e-6 |
| 23 | 0 (no rounding) |

Ratios with 12 bits (ratio / write / read, MB/s):

| Filter after `keep_bits=12` | MUSIC `arr` | pair `arr_bg` | FastHydro `arr` |
|---|---|---|---|
| Blosc lz4 + shuffle | 2.25 / 885 / 610 | 2.43 / 1140 / 850 | 2.88 / 810 / 890 |
| Blosc lz4 + bitshuffle | 2.45 / 680 / 615 | 2.71 / 795 / 850 | 2.90 / 640 / 890 |
| Blosc zstd 3 + shuffle | **2.84** / 310 / 510 | **3.07** / 335 / 675 | **4.01** / 290 / 765 |
| Blosc zstd 3 + bitshuffle | 2.56 / 385 / 565 | — | 3.68 / 270 / 770 |

After rounding, bitshuffle does help LZ4, but zstd with byte shuffle is still better.
So with `keep_bits` set, a Blosc spec switches to bitshuffle automatically only for the
LZ4 family (`blosc-lz4`, `blosc-lz4hc`, `blosc-blosclz`); zstd keeps byte shuffle.

**Effect on the jet wake.** `arr - arr_bg` is a difference of two large numbers, so it
is the quantity most exposed to rounding. Both legs are rounded the same way, so the
difference stays exactly zero wherever the legs agree. Measured on a MUSIC pair event
with CausalLiquefier deposition (energy density up to 70 GeV/fm³, wake |Δe| up to
2.4 GeV/fm³):

| keep_bits | Wake Δe, relative L2 error | Δe in cells above 1% of max | Wake Δvx, relative L2 error | Total Δe |
|---|---|---|---|---|
| 10 | 2.4e-3 | 1.3e-3 | 5.8e-4 | unchanged |
| 12 | 6.8e-4 | 3.1e-4 | 1.6e-4 | unchanged |
| 14 | 2.0e-4 | 7.7e-5 | 4.2e-5 | unchanged |
| 16 | 6.1e-5 | 1.9e-5 | 1.1e-5 | unchanged |

Rounding created no spurious non-zero wake cells. For FNO training, 12 bits is far below
the model's own error. For precision wake studies with small |Δe|/e, use 14–16 bits, or
leave `keep_bits` unset for bit-exact files.

### Real production runs

One AuAu 0–10% jet event (`run_prod_jet.py --events 1 --seed 1`, PythiaGun, deposition
on), run once per setting. The output grid is 65×65×33 with 106 tau frames, and each
leg is 236 MB raw.

| Setting | `arr` (jet leg) | `arr_bg` | File | Job wall time |
|---|---|---|---|---|
| `--compression lzf` | 212 MB (1.11×) | 188 MB (1.26×) | 400 MB | 38.9 s |
| default (`blosc-zstd`) | 151 MB (1.57×) | 134 MB (1.77×) | 285 MB | 39.6 s |
| `--keep-bits 12` | 84 MB (2.82×) | 75 MB (3.16×) | 159 MB | 39.5 s |

Verified on these three files:

- The lzf and Blosc files hold bit-identical data.
- The `--keep-bits 12` file equals `round_mantissa(lossless, 12)` exactly, with maximum
  relative error 1.2e-4 and every zero preserved.
- Reading one event through `MultiH5Array` took 690 / 600 / 525 MB/s respectively.
- Writing takes about 1 s of the ~39 s job, so the filter does not change the job time.

### Apple M3 Max

Measured 2026-09-26 on an Apple M3 Max: 16 cores, 64 GB, music4gpu on Metal, env
`fno_env_mlx` with h5py 3.15.1, HDF5 1.14.6 and hdf5plugin 7.1.0. `BLOSC_NTHREADS` was
unset, i.e. 1 thread. Threads and FastHydro output were not measured on the Mac.

**Filters**, from `utils/h5_compression_bench.py` on a real pair file, MUSIC `arr`,
AuAu 0–10%: 2 events, 468 MB raw, 105 tau frames, chunks `(1,4,65,65,33,1)`. Cells
read ratio / write / read, in MB/s of uncompressed data:

| Filter | Lossless | After `keep_bits=12` |
|---|---|---|
| lzf (old default) | 1.11 / 326 / 562 | — |
| gzip 4 + shuffle | **1.59** / 71 / 275 | — |
| Blosc lz4 + shuffle | 1.46 / 594 / 478 | 2.23 / 605 / 471 |
| Blosc lz4 + bitshuffle | 1.36 / 598 / 478 | 2.42 / 600 / 441 |
| Blosc lz4 level 9 + shuffle | 1.48 / 549 / 468 | — |
| Blosc zstd 3 + shuffle (**default**) | 1.56 / 287 / 453 | **2.81** / 255 / 398 |
| Blosc zstd 3 + bitshuffle | 1.40 / 262 / 437 | 2.54 / 327 / 416 |

- **The ratios match the GB10's** to within 0.03, although the event is a different
  one (seed 1 does not give the same collision on both machines). They are set by the
  data, not the machine.
- **Speeds are lower**, but the ranking is the same. Relative to lzf, zstd writes 12 %
  slower (GB10: 10 %) and reads 19 % slower (GB10: 18 %).
- **Bitshuffle** loses on lossless data here too, and after rounding helps only LZ4.
- **Uncompressed**, the tau-sliced read ran at 9 MB/s: the same copy-bound layout effect
  as on the GB10 (see [Threads](#threads)).

**Production runs.** `run_prod_jet.py --events 2 --seed 1`, PythiaGun, deposition on,
one run per setting. The output grid is 65×65×33, with 101/100 and 105/103 jet/background
tau frames:

| Setting | File | Per-event wall time |
|---|---|---|
| `--compression lzf` | 840 MB | 25.5 / 27.7 s |
| default (`blosc-zstd`) | 597 MB (−29 %) | 27.0 / 28.8 s |
| `--keep-bits 12` | 330 MB (−61 %) | 26.0 / 28.9 s |

- **Data:** the lzf and Blosc files hold bit-identical data (all 30 datasets). Two more
  Blosc runs, with and without the LBT-tables link, were identical as well.
- **`--keep-bits 12`:** maximum relative error 1.22055e-4 (bound 2^-13 = 1.22070e-4) in
  both `arr` and `arr_bg`, every zero preserved. `arr` goes from 468 MB raw to 166 MB
  (2.8×). Both legs are tagged `compression = "blosc-zstd:3+shuffle"` and
  `keep_mantissa_bits = 12`, and `h5_inspect.py` shows both.
- **Wall time:** the four Blosc runs averaged ~1 s per event more than the single lzf
  run. That is within run-to-run scatter: by the speeds above, the filter itself costs
  ~0.2 s per event pair (~470 MB of evolution).
- **Tests:** PyJetscape `test_pair_h5.py` and `test_h5_bulk.py` pass (59), and FastHydro
  `test_h5_output.py` and `test_fast_data_writer.py` pass (13; 2 skipped because FNO4d's
  loaders are not vendored).
- **Reading:** both example notebooks import `hdf5plugin`. The single-leg writer
  (`run_prod.py`, `H5BulkWriter`) also writes readable Blosc files on the Mac.

## Using it

### Compression specs

Every writer takes the same `compression` argument (the `--compression` command-line
option, or `output.compression` in fast_data/FastHydro YAML):

| Spec | Meaning |
|---|---|
| `blosc-zstd` | Blosc + zstd level 3, byte shuffle. **Default.** |
| `blosc-lz4` | Blosc + LZ4 level 5, byte shuffle. Fastest write, about 7% larger files. |
| `blosc-<codec>[:level][+shuffle\|+bitshuffle\|+noshuffle]` | Any Blosc codec: `blosclz`, `lz4`, `lz4hc`, `snappy`, `zlib`, `zstd`. |
| `lzf` | The old default. |
| `gzip[:level]` | Deflate, level 4 by default, with byte shuffle. |
| `none` / `None` | Uncompressed. |
| a mapping | Passed to `h5py.create_dataset` unchanged, e.g. `dict(hdf5plugin.Blosc(...))`. |

If `hdf5plugin` is not installed, a Blosc spec falls back to lzf with a `RuntimeWarning`,
so a run never fails over the optional package.

Each evolution dataset records what it was written with:

- `compression`, e.g. `"blosc-zstd:3+shuffle"`. The label is itself a valid spec.
- `keep_mantissa_bits` and `max_rel_error`, only for rounded data. If
  `keep_mantissa_bits` is absent, the data is bit-exact.

`utils/h5_inspect.py` shows both.

### Where to set it

| Writer | Setting |
|---|---|
| `jetscape.FnoH5Writer` | `FnoH5Writer(path, attrs, compression="blosc-zstd", keep_bits=None)`. `add_evolution` datasets (e.g. `arr_bg`) inherit both. |
| `jetscape.pair_h5.PairH5Writer` | `compression=`, `keep_bits=`. Both legs get the same filter and rounding. |
| `jetscape.fast_h5_bulk.H5BulkWriter` | `compression=`, `keep_bits=` |
| `prod_AuAu_0_10/run_prod.py`, `prod_AuAu_0_10_jet/run_prod_jet.py` | `--compression SPEC`, `--keep-bits N`, validated before the first event |
| FastHydro `PairedH5Writer` / fast_data `FnoH5Writer` | YAML `output.compression`, `output.keep_bits`. `arr_bg` gets `arr`'s filter and rounding. The shipped `config/*.yaml` now use `blosc-zstd`. |
| FNO4d `root2hdf5/root_to_hdf5.py`, `loc_libs/downsample.py` | `--compression SPEC`, `--keep-bits N` |
| FNO4d `convert_root_to_hdf5`, `write_downsampled_h5` | `compression=`, `keep_bits=` |

Ragged tables (`source/droplets`, `shower/*`) and FastHydro's `source/S` keep
`gzip:4+shuffle`, now also through the same spec parser.

```bash
python run_prod_jet.py --events 10 --seed 1                     # blosc-zstd, bit-exact
python run_prod_jet.py --events 10 --seed 1 --keep-bits 12      # ~2x smaller again
python run_prod_jet.py --events 10 --seed 1 --compression lzf   # the old files
```

### Reading

- **Install `hdf5plugin`** wherever the files are read. It is now a dependency of
  PyJetscape's `hdf5` extra, of FastHydro, and of FNO4d (`loc_libs`,
  `requirements-base.txt`).
- The filters must be registered before the first read. These imports do it:
  - `jetscape`
  - `fast_data`, `fasthydro`
  - FNO4d `read_3d_hdf5` (all training reads go through it), `downsample`,
    `repad_h5`, `h5_inspect`, `root2hdf5/inspect_file.py`
  - `Visualization/wake_pyvista.py`
  - the notebooks: `prod_AuAu_0_10/check_output.ipynb` and `prod_AuAu_0_10_jet/jet_wake.ipynb`
    import `hdf5plugin` in their first cell. FastHydro's `notebooks/jet_wake.ipynb` reads
    through `PairBrowser`, which imports `fast_data`. The two `hadron_*` notebooks read
    no HDF5.

  Your own scripts that call `h5py.File` directly need `import hdf5plugin` first.
  Without it, h5py fails with *"required filter 'blosc' is not registered"*. On macOS
  (h5py 3.15, HDF5 1.14.6) the same failure reads *"Can't synchronously read data
  (can't open directory (/usr/local/hdf5/lib/plugin) …"*, and only when the data is
  read, not when the file is opened.
- **Command-line HDF5 tools** (`h5dump`, `h5ls -v`) need the plugin path:

  ```bash
  export HDF5_PLUGIN_PATH=$(python -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
  ```

  (lzf needed a plugin for these tools too.)
- **Threads:** with whole-event chunks, `BLOSC_NTHREADS=4..8` roughly doubles read
  speed. In a PyTorch `DataLoader` with several workers, leave it unset (1 thread), so
  that workers × threads do not oversubscribe the cores.
- Files written with different filters can be mixed in one `MultiH5Array`. The filter is
  per dataset, and HDF5 decodes it transparently.
- `repad_h5` is unchanged; resizing needs no decompression. Its standalone copy is now
  three files: `fno_h5_writer.py`, `repad_h5.py` and `h5_compression.py`.

## Code

- `h5_compression.py` holds the spec parser, `round_mantissa`, `tag_dataset` and the
  `hdf5plugin` registration. It needs only numpy and h5py (`hdf5plugin` optional).
  - **Canonical copy:** FNO4d `loc_libs/fast_data/h5_compression.py`.
  - **FastHydro:** `python/fast_data/`, re-vendored with `tools/sync_fast_data.sh`.
  - **PyJetscape:** `python/jetscape/h5_compression.py`, a verbatim copy.

  Change it in FNO4d, then copy it to the other two.
- Tests:
  - FNO4d `tests/test_h5_compression.py`
  - PyJetscape `tests/test_pair_h5.py` (every evolution gets one filter; `keep_bits`
    rounds both legs identically)
  - FastHydro `tests/test_h5_output.py` (`arr_bg` gets `arr`'s filter and rounding)

## Open points

- **Chunk layout.** Tau-sliced chunks (`chunk_tau=1`) let PyJetscape write one frame at
  a time without read-modify-write. The cost is that whole-event reads are copy-bound at
  about 600–950 MB/s. Larger `chunk_tau` (e.g. 8–16, buffered in the writer) or
  whole-event chunks would let Blosc threads speed up reads too. Worth measuring against
  real `DataLoader` throughput before changing.
- **Chunk cache.** A MUSIC tau-slice chunk is 2.2 MB, larger than h5py's default 1 MB
  chunk cache. Whole-event reads are unaffected, but readers that take sub-slices across
  tau should open with a larger `rdcc_nbytes`.
- **Existing files** are not converted. A small `h5_recompress` tool (copy every dataset
  and re-filter the evolutions) would be simple if the campaign's lzf files need to
  shrink.
