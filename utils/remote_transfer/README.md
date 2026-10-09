# remote_transfer — productions to and from Google Cloud Storage and Pelican/OSDF

Two command-line tools upload production directories and files, download them again, and
remove them.
They pick the files by kind: pair files, the HDF5 hadron files, the ROOT files, or all of
them.

| script | where | credentials |
|---|---|---|
| [`js_gcs.py`](js_gcs.py) | a Google Cloud Storage bucket, default `gs://test_fno` | a service-account JSON key, for reading too |
| [`js_osdf.py`](js_osdf.py) | a Pelican namespace on the OSDF, default `osdf:///fno4hic` (pelicanfs) | for writing (`upload`, `rm`) a browser login (`login`) or a bearer token; reading (`ls`, `download`) `/fno4hic` needs none: it is public |

Both have the same commands and options and follow the same rules. The code they share is
in [`transfer_core.py`](transfer_core.py); keep the three files together. Each script makes
its own Python environment on its first run and from then on runs itself in it, so neither
needs a conda env, venv, uv or pipx. A `python3` (≥ 3.9, with `venv`) is all.

```bash
cd utils/remote_transfer
./js_gcs.py ls                                                   # the bucket (gs://test_fno)
./js_gcs.py upload /data/AuAu_0_10_pth10-40_eta06_c1 --what root --dry-run
./js_gcs.py upload /data/AuAu_0_10_pth10-40_eta06_c1 --what root     # -> gs://test_fno/AuAu_0_10_pth10-40_eta06_c1/
./js_gcs.py upload /data/AuAu_a /data/AuAu_b --what pair,h5 --prefix productions -j 8
./js_gcs.py upload /data/out --what root --as AuAu_pth10-40_v1      # -> gs://test_fno/AuAu_pth10-40_v1/
./js_gcs.py ls AuAu_0_10_pth10-40_eta06_c1 --what h5             # files, sizes, kinds
./js_gcs.py download AuAu_0_10_pth10-40_eta06_c1 --what root --to /scratch
                                                       # -> /scratch/AuAu_0_10_pth10-40_eta06_c1/
./js_gcs.py upload /data/AuAu_c1/AuAu_c1_0003_hadrons.root /data/AuAu_c1/*_0004_*.h5
                                                       # single files (§ Single files)
./js_gcs.py download AuAu_c1/AuAu_c1_0003_hadrons.root 'AuAu_c1/*_0004_*' --to here

./js_osdf.py login                                               # once: log in in a browser (§ Credentials)
./js_osdf.py upload /data/AuAu_0_10_pth10-40_eta06_c1 --what root    # -> osdf:///fno4hic/AuAu_0_10_pth10-40_eta06_c1/
./js_osdf.py download AuAu_0_10_pth10-40_eta06_c1 --what root --to /scratch
./js_osdf.py ls                                                  # the namespace (osdf:///fno4hic)
./js_osdf.py du                                                  # space used per folder, and in all (§ Space used)
./js_osdf.py rm -r AuAu_0_10_pth10-40_eta06_c1 --dry-run          # what would be removed (§ Removing)
./js_osdf.py rm AuAu_c1/AuAu_c1_0003_hadrons.root 'AuAu_c1/*_0004_*'   # files and patterns; asks first
```

## What: `--what`

The kinds go by file name, as `run_prod_jet.py`, `hadronize.py` and `run_h5toROOT.py` name
the files. `--what` takes one kind or a comma list (`pair,root`); the default is `all`.

| `--what` | files |
|---|---|
| `pair` | `<stem>.h5`, the hydro pair file, with the run's `<stem>.json`, `.xml`, `.log` |
| `h5` | `<stem>_particlize.h5`, `<stem>_hadrons_{bulk_jet,bulk_bg,jet_frag}.h5`, `<stem>_hadronize.log` |
| `root` | `<stem>_hadrons.root`, `<campaign>_campaign.root` |
| `all` | the three, plus every other file of the directory (`run_jobs.campaign`, `wake_observables.h5`, …) |

- **`h5` brings the particlize files along.** `HadronFileReader` needs them next to the
  hadron files.
- **`root` is complete on its own.** `HadronFileReader.h` and the macros of
  `analysis_root` need nothing else.
- **Pair files are recognized by their neighbours.** A `<stem>.h5` counts as a pair file
  only with its `<stem>.json`, `<stem>.xml` or `<stem>_particlize.h5` next to it. That
  tells it apart from other HDF5 files in the directory, such as `wake_observables.h5`.
  Files matched by a pattern are judged among all files of their directory.

`--dry-run` shows what would be transferred, with the sizes per kind. On
`AuAu_0_10_pth10-40_eta06_gridnorm` that is pair 80 files 50.6 GB, h5 100 files 51.4 GB,
and other 4 files 35 MB.

## Where

- **Uploads** of a directory go to `ROOT/[PREFIX/]<directory name>/<file>`. `ROOT` is
  `gs://BUCKET` or `osdf:///NAMESPACE`. Only the files directly in the directory are sent,
  not its subdirectories.
- **Another folder name:** `--as NAME` uploads to `ROOT/[PREFIX/]NAME/<file>` instead of
  using the local directory's name. Use it to publish `out/` as `AuAu_pth10-40_v1`, or to
  add single files of a directory to a folder uploaded that way. `NAME` may contain `/`.
  It names one folder, so all sources must come from one local directory. It doesn't go
  with `--flat`.
- **Downloads** of a remote directory go to `TO/<its name>/`.
- **Remote names** can be written `NAME` or `ROOT/NAME`; for OSDF also `/NAMESPACE/NAME`
  or `pelican://FEDERATION/NAMESPACE/NAME`.
- **The bucket** is `--bucket`, else `$JS_GCS_BUCKET`, else `gs://test_fno`.
- **The namespace** is `--namespace`, else `$JS_OSDF_NAMESPACE`, else `/fno4hic`. The
  federation is `--federation`, default `osg-htc.org` (the OSDF); another federation shows
  as `pelican://FEDERATION/NAMESPACE`.

## Single files

`upload` and `download` take any mix of directories and files, several at once.

- **Uploading a file:** a file named on the command line goes up whatever its kind
  (`--what` applies to directories only). It goes where its directory's upload puts it:
  `ROOT/[PREFIX/]<directory name>/<file>`. So `upload /data/AuAu_c1/AuAu_c1_0007.h5` adds
  one file to `ROOT/AuAu_c1/`. `--flat` puts it at `ROOT/[PREFIX/]<file>` instead. Shell
  globs work (`/data/AuAu_c1/*_0007*`).
- **Downloading a file:** a remote file goes to `TO/<file>`, like `cp`.
- **Downloading a pattern:** `*`, `?` and `[..]` select files by pattern, also to
  `TO/<file>`. `*` also matches `/`, and `--what` filters the matches. Quote patterns so
  the shell leaves them alone: `'AuAu_c1/*_0004_*'`. `ls` takes patterns too.
- **Same rules as directories:** files already there are skipped, incomplete HDF5 files
  and `*.part` are left out.
- **Overlaps:** a file asked for twice, e.g. through its directory and a pattern, goes to
  both places. Two remote files that would land on the same local file are refused;
  download them with different `--to`.

## Space used: `du`

```bash
./js_osdf.py du                                    # per top-level folder, and the total
./js_osdf.py du AuAu_c1 -d 2 --sort size           # below a folder, two levels, largest first
./js_osdf.py du -d 0 --what root                   # only the total of the ROOT files
./js_osdf.py du 'AuAu_*'                           # a pattern (quoted)
```

On `/fno4hic` (2026-10-09):

```
     2.1 GB       70  osdf:///fno4hic/test/
     2.1 GB       70  osdf:///fno4hic/  total
js_osdf: manifest 1 file(s) 10.3 kB, other 1 file(s) 76 B, root 68 file(s) 2.1 GB
```

- **A line per directory** down to `-d`/`--depth` below the prefix (default 1; `0`: only
  the total), with its size and number of files. Each counts everything below it, as `du`
  does. Files directly in the prefix get a line of their own, then comes the total, then
  the sizes per kind.
- **Everything stored counts**, the manifests too (`ls` doesn't show them). `--what` counts
  only those kinds, without the manifests.
- **The prefix** can be a directory, a file or a pattern (directories then count from the
  pattern's directory). Empty directories hold nothing and don't show.
- **One listing:** like `ls -r`, `du` lists the files once (about a second for `/fno4hic`).

## Removing: `rm`

```bash
./js_osdf.py rm AuAu_c1/AuAu_c1_0003_hadrons.root     # one file
./js_osdf.py rm 'AuAu_c1/*_0004_*'                     # a pattern (quoted)
./js_osdf.py rm -r AuAu_c1 --dry-run                   # a directory: what would go
./js_osdf.py rm -r AuAu_c1                             # ... then all of it
./js_osdf.py rm -r AuAu_c1 --what h5                   # only a kind; the directory stays
```

- **It asks first.** `rm` lists what it will remove (the first 20 files; `--dry-run` lists
  all and removes nothing) and removes only after `y`. `--yes` doesn't ask. Without a
  terminal (a script, a batch job) it refuses unless `--yes`. Removed files can't be
  brought back.
- **Targets** are files, patterns (`*`, `?`, `[..]`, as for `download`) and, with `-r`,
  directories. A directory without `-r` is refused, and so is the whole bucket or
  namespace.
- **A directory with `-r`** goes with everything below it: the files, the manifests and
  the directories, also empty ones. With `--what` only the files of those kinds go, and
  the directories stay.
- **The manifest** of a directory that stays loses the entries of the removed files; it is
  removed when it is then empty.
- `-j` removes that many files at once (default 4). A file already gone counts as removed.
- **OSDF:** `rm` needs `storage.modify` on the namespace (the browser login has it). The
  removal is an HTTP DELETE at the origin; pelicanfs has none. **GCS:** the service account
  needs delete rights on the bucket.

## Safe to re-run

- **Files already there are skipped:** the same size and the same CRC32C as the remote file
  (or the local one, for downloads). An interrupted command finishes when run again.
  - `--size-only`: compare sizes only, without computing the local CRC32C (that takes about
    a second per few GB).
  - `--force`: transfer anyway.
- **Unfinished files are not uploaded.** HDF5 files whose `complete` attribute is False
  (a job still writing) are left out and listed, unless `--include-incomplete`. `*.part`
  files never go up.
- **Downloads** are written to `<file>.part` and renamed only when their size (and CRC32C,
  where known) is right. A failed download leaves nothing behind.
- `-j` sets how many files are transferred at once (default 4).
- **Errors** of the connection or the server end the command with one line, including the
  root cause, e.g. `SSLCertVerificationError: … certificate has expired`.

**Checksums differ between the two stores:**
- **GCS** keeps a CRC32C of every object, and the client checks every upload and download
  against it. Large files go up as resumable uploads in 64 MB chunks, with retries.
- **Pelican** reports sizes but no checksums. So after each upload `js_osdf.py` checks the
  size the origin reports, and records the file's size and CRC32C in a manifest in the
  remote directory, `.js_transfer_crc32c.json`. Later uploads add to it.
  - The skip check and the check after a download use the manifest's CRC32C.
  - Files the manifest doesn't know, put there by other tools, are compared by size.
  - The manifest is never downloaded and `ls` doesn't show it.
  - Two `js_osdf.py` uploads into the same remote directory at the same time may lose each
    other's manifest entries. The files themselves are fine; the lost ones are then
    compared by size.
- **Reading from the origin:** listings, sizes and downloads come from the origin itself
  (pelicanfs `direct_reads`). An OSDF cache could still hold an older copy of a file that
  was uploaded again.

## Uploading while a production runs: `upload_follow.py`

[`../upload_follow.py`](../upload_follow.py) builds on `js_osdf.py`: it uploads each job of a
production directory as soon as its `<stem>.json` says complete, every `--interval` seconds
with `--follow` (until `OUTBASE/.upload_final` appears or the process `--follow-pid` ends), and
can delete the uploaded pair and/or particlize files
locally (`--delete pair|all`, optionally only below `--keep-free SIZE`). A file is deleted only
after a re-check right before: the origin's size, the manifest's CRC32C against the local file,
and every Nth file (`--verify-every`) downloaded and compared byte by byte. Its state survives
restarts (`OUTBASE/upload_state.json`).

```bash
utils/upload_follow.py /data/c1 AuAu_c1                          # one pass, upload only
utils/upload_follow.py /data/c1 AuAu_c1 --follow --delete pair   # while the campaign runs,
touch /data/c1/.upload_final                                     # until this: a last pass, exit
```

`launch_2gpu.sh --upload` starts it next to a campaign. All options and the checks:
[`../README_launch.md`](../README_launch.md#uploading-to-the-osdf-fno4hic).

## Credentials

**GCS: a service-account JSON key,** looked for in this order:
1. `--key-file PATH`;
2. `$JS_GCS_KEY_FILE`;
3. `$GOOGLE_APPLICATION_CREDENTIALS`;
4. `fnotest-wayne-gcs.json` in the current directory, next to `js_gcs.py`, or in
   `~/.config/js_gcs/`.

The service account needs read access to the bucket for `ls` and `download`, and write
access for `upload`. Without the key nothing can be read: the bucket isn't public.

**OSDF: a browser login or a bearer token, for writing only.** `upload` and `rm` need write
access to the namespace. `ls` and `download` need credentials only for a protected
namespace.

**Reading `/fno4hic` is public**, as opposed to `gs://test_fno`: anyone can list and
download its files, with no login, no token and no account; the director reports
`require-token=false` for it. Writing without credentials is refused (HTTP 403). So
`./js_osdf.py ls` and `download` work on any machine as they are, and whatever is uploaded
to `/fno4hic` can be read by everyone.

**The browser login** needs no token file:

```bash
./js_osdf.py login              # prints a link; log in and approve in any browser, on any machine
./js_osdf.py upload /data/AuAu_c1 --what root     # later commands use the login by themselves
./js_osdf.py upload /data/AuAu_c1 --web           # or: log in when needed, in the same command
./js_osdf.py status             # who, the scopes, the time left of the token and the login
./js_osdf.py logout             # revoke the login at the issuer and forget it
```

- **How:** the OAuth2 device flow of the namespace's own token issuer, which the director
  names for the namespace (for `/fno4hic` the Wayne origin's issuer). The login asks for
  `storage.read`, `storage.create` and `storage.modify` on the namespace; `modify` is what
  lets `--force` and the manifest overwrite files. It is the login the `pelican` CLI does,
  without the CLI.
- **Kept:** in `~/.config/js_osdf/web/<federation>_<namespace>.json` (mode 600), with the
  refresh token. The token (20 min on `/fno4hic`) is renewed by itself before every
  request, also in the middle of a long upload. When the issuer no longer renews it,
  `login` again: a command already running picks the new login up from the file at its
  next renewal; one that fails before reports its files FAILED (re-run it). A new command
  says so, with the issuer's answer, and goes on without a token.
- **`--web` is not for one command only.** It keeps the login as `login` does, and every
  later command uses it, with or without `--web` (for one command only, `logout` after
  it). `--web` differs from a command without it in two ways:
  - it logs in when there is no login or it can no longer be renewed (it prints the link
    and waits); without `--web` the command says so and goes on without credentials, so
    an upload then fails;
  - it uses the browser login even if there is a bearer token.
- **How long:** until the login's refresh token expires (15 days on `/fno4hic`), or
  `logout`. **Log in again before a long run would pass that date:** renewals don't extend
  it, see the next point.
- **Renewals keep the login's refresh token.** The `/fno4hic` issuer hands out a new
  refresh token with every renewal, but crashes when that one is used (HTTP 500
  `server_error`, "Null pointer"; OA4MP at the Wayne origin, found 2026-10-05), while the
  login's own refresh token renews any number of times. Switching to the new one, as
  js_osdf.py did before, ended every login after ~40 min (one renewal). So the login's is
  kept, and the issuer's latest only noted (`refresh_token_rotated` in the login file): it
  is used when the kept one is refused, e.g. once the issuer is fixed and invalidates old
  refresh tokens, and then kept instead. A failed renewal prints the issuer's answer.
- **`status`** shows the login without asking the issuer: the user, the issuer, the
  scopes, the time left of the token and of the refresh token, and a bearer token that
  would come first. It exits 1 when there is nothing to write with.

  ```text
  js_osdf: browser login for /fno4hic:
    user     Joern Putschke
    issuer   https://wayne-origin.nationalresearchplatform.org:8455
    scopes   storage.modify:/ storage.create:/ storage.read:/
    token    valid until 2026-10-01 20:58 (20 min left)
    refresh  valid until 2026-10-16 20:38 (15.0 days); renewals don't extend it: login again before then
    file     ~/.config/js_osdf/web/osg-htc.org_fno4hic.json
  ```
- **Which credential:** without `--web` a bearer token found as below comes first, then the
  browser login. `--web` uses the browser login even if there is a token; it doesn't go
  with `--token-file`.
- Options like `--web` and `--token-file` go before or after the command.

**A bearer token** (a SciToken / WLCG token) is looked for in this order:
1. `--token-file PATH`;
2. `$JS_OSDF_TOKEN_FILE`, then `$BEARER_TOKEN_FILE`;
3. `$BEARER_TOKEN` (the token itself);
4. `osdf.token` in the current directory, next to `js_osdf.py`, or in `~/.config/js_osdf/`.

With none of these and no browser login, pelicanfs looks for one itself, e.g. in the WLCG default place
(`$XDG_RUNTIME_DIR/bt_u$UID`, `/tmp/bt_u$UID`). With the `pelican` CLI on `PATH` it can also
get one through the namespace's OAuth flow. Tokens expire, typically after hours, so a long
upload needs one that lasts.

**Keep credentials out of git.** `.gitignore` ignores `fnotest-wayne-gcs.json`, and every
`*.json` and `*token*` in this directory. The safest place is `~/.config/js_gcs/` or
`~/.config/js_osdf/` (`chmod 600`).

## Their environments

- **Where:** `~/.cache/js_gcs/venv` and `~/.cache/js_osdf/venv` (or `$JS_GCS_ENV`,
  `$JS_OSDF_ENV`).
- **What:**
  - js_gcs: `google-cloud-storage`, `google-crc32c` (C implementation) and `h5py`, for the
    `complete` check;
  - js_osdf: `pelicanfs`, `google-crc32c` and `h5py`.
- **How they are made:** with the `python3` that first runs the script, in a few seconds.
  Later runs use them right away.
- `./js_gcs.py setup` shows the environment (likewise `./js_osdf.py setup`), and
  `setup --reinstall` makes it again.
- `setup --remove` deletes it, and `~/.cache/js_gcs` (`js_osdf`) with it if that is then
  empty. It deletes only a venv the script made: one with its `pyvenv.cfg` and the
  script's stamp file. The next run makes it again.
- An environment is remade by itself when the requirements in the script change, or when
  the Python it was made from is gone.
- `JS_GCS_NO_ENV=1` / `JS_OSDF_NO_ENV=1` run the script in the current environment
  instead, if that one has the packages. One example is the `-gcs` production images
  ([`BuildContainerProd.md`](../BuildContainerProd.md)): they have pelicanfs,
  `google-cloud-storage`, `google-crc32c` and h5py, so both scripts run there as they are.
  The other images lack `google-crc32c`, so there the scripts make their environments.

## Tested

- **Offline:** `pytest utils/remote_transfer/test` (35 tests, in
  [`test/test_transfer.py`](test/test_transfer.py)). Every command runs against two
  stores: an in-memory GCS bucket, and `OsdfStore` on fsspec's in-memory file system in
  place of pelicanfs. The tests cover:
  - directories, single files (`--flat`) and patterns;
  - skipping, incomplete HDF5 files, `.part` files, overlaps and local name clashes;
  - the manifest: written, added to, used for skipping, a corrupted download caught, an
    unreadable manifest;
  - the key and token lookup, remote names, and that the CRC32C matches GCS's;
  - the browser login against a fake Pelican issuer: the login (pending, consent_required,
    then the token), the login file, renewal before a request and in the middle of a run, a
    refused renewal, `--web`, `logout`, and `status`; the `/fno4hic` issuer's renewal bug
    (refresh tokens from renewals answered with HTTP 500: the login's is kept and renews
    again and again; the latest used once the kept one is refused; both refused: the
    issuer's answer reported), and a running process taking up a login made meanwhile;
  - `du`: totals per directory at each depth, the files directly in the prefix, the
    manifests counted, `--what`, `--sort`, patterns and a single file;
  - `rm`: files, patterns, directories with their manifests and empty directories below,
    `--what`, `--dry-run`, the question (no terminal, no, yes), and targets that aren't
    there.
- **On `gs://test_fno` (GB10, 2026-10-01):**
  - the environment was made from `/usr/bin/python3` (3.12) outside any conda env;
  - `ls` listed the bucket;
  - `upload --dry-run` classified `AuAu_0_10_pth10-40_eta06_gridnorm` as above;
  - `ls` and `download --dry-run` with a pattern and with one object, and `upload --dry-run`
    of single files, picked the expected files.
- **On the OSDF (2026-10-01), with pelicanfs 1.4.1:**
  - on the public `/ospool/uc-shared/public/OSG-Staff`: `ls`, `ls` with a pattern, and
    downloads of a directory, a pattern and one file. The files are right, and a second
    run skipped them.
  - on `/fno4hic`, with the browser login: `login` granted `storage.read:/`,
    `storage.create:/` and `storage.modify:/`; a file uploaded to `js_osdf_test/` (with the
    manifest), downloaded identical, overwritten with `--force`; an expired token renewed
    from the refresh token; `rm` of one file, of a pattern, and `rm -r` of a nested
    directory (with its manifests and an empty directory below), `--dry-run`, and the
    refusal without a terminal. (Its origin's TLS certificate, expired on 2026-09-21, was renewed
    by then, until 2026-12-30.)
- **On `/fno4hic`, 2026-10-05 (host rhicML, 2 × RTX 3090):**
  - the renewal bug above: after a login, renewal 1 worked and renewal 2 (with the refresh
    token from renewal 1) failed with HTTP 500 "Null pointer", also with `scope` and when
    repeated; with the login's refresh token three renewals in a row worked, and one from a
    renewal failed again. With the fix: five forced renewals through the command line, the
    refresh token unchanged, then an upload and an `rm`.
  - uploads of production files (4 GB) at ~80 MB/s (4 or 8 files at once alike), a download
    at ~69 MB/s, byte-identical; a second upload skipped every file.
- **Not tested yet:** real uploads, downloads and `rm` on `gs://test_fno` (only dry runs),
  and uploads to `/fno4hic` over hours (a login now lasts its 15 days, see above).
