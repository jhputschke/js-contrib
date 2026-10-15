#!/usr/bin/env python3
"""
example/prod_AuAu_0_10_jet/hadronize.py

Hadronize what run_prod_jet.py --write-particlize stored: iSS on each leg's freeze-out
surface, X-SCAPE's ColorlessHadronization on the final partons (PLAN_particlize_h5.md).
No GPU, no MUSIC: only the stored inputs, the iSS tables and Pythia.

    conda activate js_fno
    python hadronize.py out/AuAu_0_10_jet_seed0001_particlize.h5           # all three tags
    python hadronize.py P.h5 --tags bulk_jet,bulk_bg --oversample 200
    python hadronize.py P.h5 --oversample 200 --oversample-bg auto   # reused bg: N x 200
    python hadronize.py P.h5 --oversample 200 --oversample-bg per-pthat-bin
                                        # --pthat-bins file: M x 200, M jets per window per bg
    python hadronize.py P.h5 --tags jet_frag --n-frag 50
    python hadronize.py P.h5 --use-stored-seeds         # replay a --validate-inline job
    python hadronize.py P.h5 --check        # re-make the jet_frag file's jets, compare
    python hadronize.py P.h5 --diagnose-colored         # colour-flow diagnostic only
    python hadronize.py P.h5 --skip-complete            # campaigns: only what is missing
    python hadronize.py P.h5 --keep-bits-p 12 --keep-bits-x 8   # rounded p, x: 58% of the bytes
    python hadronize.py P.h5 --eta-max 2        # only hadrons at |eta| < 2: ~60% of the bytes
    python hadronize.py P.h5 --charged --no-x   # charged hadrons, no positions
    python hadronize.py P.h5 --add-initiators           # older outputs: add initiators/ only
    python hadronize.py P.h5 --keep-primaries 20  # + undecayed hadrons of 20 oversamples

Output, next to the input (or in --out-dir), one file per tag (jetscape.hadrons_h5):

    <stem>_hadrons_bulk_jet.h5   iSS on surface/jet     bulk + wake        unit = event
    <stem>_hadrons_bulk_bg.h5    iSS on surface/bg      bulk               unit = background
    <stem>_hadrons_jet_frag.h5   Colorless on partons/  jet fragments      unit = event

and with --keep-primaries N, for SMASH later (jetscape.hadrons_h5, PRIMARY_TAGS):

    <stem>_hadrons_bulk_jet_primary.h5   bulk_jet's first N oversamples, undecayed
    <stem>_hadrons_bulk_bg_primary.h5    bulk_bg's first N oversamples, undecayed

A jet event is bulk_jet + jet_frag; its background is bulk_bg unit ``events/bg_unit`` of
the particlize file (units/event in the bulk_bg file is the first event that used it).
Oversamples of one event are samples of that event, not new events: average over them.

pTHat windows.  A run_prod_jet.py --pthat-bins file is a --reuse K*M file (K windows, M jets
per window per background) with events/pthat_bin; everything above applies unchanged, one
background unit per K*M events.  Only the background's oversampling can take the windows into
account: --oversample-bg auto gives it K*M x --oversample (right for averages over all
windows), --oversample-bg per-pthat-bin M x --oversample (right for each window on its own).

Initiators.  bulk_jet and jet_frag also get ``initiators/``: each event's shower-initiating
partons, from the particlize file's own ``initiators/`` (format version 2: the file is
self-contained).  For an older particlize file they come from the pair file's
``shower/initiators`` (its ``pair_file``, looked up next to it); without either the outputs
are written without them, with a warning.  Jet-relative analyses (HadronFileReader's
``info.initiators()``) then need only the particlize and hadron files, not the pair file with
the hydro.  --add-initiators adds the group to existing outputs without hadronizing again;
``add_initiators.py`` adds it to an old particlize file.

Seeds.  Every unit gets its own seed, derived from (--seed, the production file, tag, unit,
sample), and it is recorded in ``units/seed``: any unit can be regenerated alone.  The
production file enters through the particlize file's ``file_uuid``, so the files of a campaign
(which all get the same --seed) draw independent random numbers; before, event 0 of every
file got the same iSS and Pythia seeds, which --correlated turns into correlated events.
--legacy-seeds reproduces files made with the old scheme (``seed_scheme`` attribute: absent
or "legacy").  --use-stored-seeds takes
the seeds a --validate-inline job recorded instead (bulk_jet: iSS, jet_frag: Pythia, one
fragmentation per event), so the result must equal that job's <stem>_inline_*.h5 exactly.
--common-seeds gives each event's bulk_jet the seed of its background (bulk_bg unit), so
jet and background draw the same random numbers while their sampling stays aligned;
identical surfaces then give identical hadrons (PLAN_iSS_optim.md, Part B).  With iSS's
conventional sampling that alignment is lost at the first hadron; --correlated (implies
--common-seeds) switches iSS to random numbers addressed by the cell, so the legs keep
giving the same hadrons wherever their surfaces agree.  Sample k of an event's bulk_jet
and sample k of its background then belong together: analyse jet - background paired.

Re-hadronizing the jets.  The jets are hadronized here, from the particlize file's
``partons/``, so a jet_frag file can always be made again: the same command (or the same
--seed, --n-frag and hadronize.xml) gives the same hadrons, --seed / --n-frag other
fragmentations (a fragmentation's seed does not depend on --n-frag), a copy of
hadronize.xml with <pythia_decays> off the primary hadrons (with --tags jet_frag and
--out-dir).  --check [JET_FRAG] replays an existing jet_frag file (default: this particlize
file's, in --out-dir): it takes the settings the file records (base_seed, seed_scheme,
stored_seeds, n_samples, hadronize_xml, precision and selection), hadronizes its events
again into a temporary file and compares pid, pstat, p and x of every fragmentation; exit 1
on a difference, nothing is written.  It needs the Pythia version the file records
(``pythia_version``; files before smash_ready do not record it).

Precision.  p and x are stored as full float32 by default.  --keep-bits-p / --keep-bits-x
round them to that many mantissa bits (max relative error 2**-(bits+1)); the setting is
recorded on the datasets (jetscape.hadrons_h5.hadron_precision).  Choose it once per
campaign: --skip-complete keeps complete files whatever precision they have (with a
warning), and HadronFileReader warns about a campaign of mixed precision.

Selection.  What is stored can be cut down, in all three tags, while iSS and Pythia still
sample everything, so every sample and every average inside the selection is unchanged:

    --eta-max X   only hadrons with |eta| < X (jetscape.hadrons_h5.pseudorapidity of the
                  stored momenta); X = 2 drops ~46% of a 0-10% Au+Au bulk sample (~40% of
                  the bytes)
    --charged     only charged hadrons (|pid| in jetscape.hadrons_h5.CHARGED)
    --no-x        no positions t, x, y, z (~40% of the bytes); readers give x as empty
                  (N, 0) arrays

They are recorded as the files' ``eta_max``, ``charged_only`` and ``positions`` attributes
(jetscape.hadrons_h5.hadron_selection).  Like the precision, choose them once per campaign:
--skip-complete warns about complete files with another selection, HadronFileReader about
a campaign that mixes them.  Observables that need every hadron (the total energy balance
of jet - background, neutral hadrons, space-time) need files without them.

Decays and undecayed hadrons.  iSS decays the resonances of either hadron list: EOS 9's
UrQMD list as always, EOS 91's SMASH list with <decay_smash_list> 1 (hadronize.xml; needs
iSS from smash_ready on, core.ISS_DECAYS_SMASH_LIST), where the channels the pole masses
close get Breit-Wigner masses for their unstable daughters; eta and Sigma0 stay stable, as
in SMASH.  The files record ``hadron_list`` (SMASH / UrQMD), ``decayed`` and ``iss_decays``.
--keep-primaries N also writes the hadrons as iSS sampled them, before any decay, for the
first N oversamples of every unit: full precision, with positions and without --eta-max /
--charged / --no-x, whatever those are for the decayed files (SMASH needs every hadron, its
position and its mass).  The decayed hadrons are the same with and without it.

Settings come from hadronize.xml (the file --validate-inline also uses); the iSS paths are
made absolute and the job's own music_input -- stored in the particlize file -- is written
to iSS's working directory, so iSS reads the same EoS id and flags as it would have in the
job.  Empty surfaces (MUSIC stopped at the grid boundary) give a unit with zero samples,
which drops out of every sample average.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROD = os.path.join(os.path.dirname(HERE), "prod_AuAu_0_10")
_spec = importlib.util.spec_from_file_location("_run_prod", os.path.join(PROD, "run_prod.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)                 # also puts PyJetscape/python on sys.path

TAGS = ("bulk_jet", "bulk_bg", "jet_frag")
TAG_INDEX = {t: k for k, t in enumerate(TAGS)}


def _mantissa_bits(text):
    v = int(text)
    if not 1 <= v <= 23:
        raise argparse.ArgumentTypeError(f"must be 1..23 (float32 mantissa bits), got {v}")
    return v


def _positive_float(text):
    v = float(text)
    if not v > 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return v


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("particlize", help="a <stem>_particlize.h5 from run_prod_jet.py")
    p.add_argument("--tags", default=",".join(TAGS),
                   help=f"comma-separated subset of {', '.join(TAGS)} (default: all)")
    p.add_argument("--oversample", type=int, default=None,
                   help="iSS samples per surface (default: hadronize.xml's "
                        "number_of_repeated_sampling)")
    p.add_argument("--oversample-bg", default=None, dest="oversample_bg",
                   help="iSS samples per BACKGROUND surface: a number; 'auto' = "
                        "--oversample x the number of events using that background (the "
                        "optimal split under --reuse N); or 'per-pthat-bin' (run_prod_jet.py "
                        "--pthat-bins files) = --oversample x the events using it in one pTHat "
                        "window (the optimal split for per-window results). Both are capped at "
                        "--oversample-bg-max. Default: --oversample")
    p.add_argument("--oversample-bg-max", type=int, default=2000, dest="oversample_bg_max",
                   help="cap for --oversample-bg auto / per-pthat-bin (default 2000, ~1.7 GB "
                        "for iSS)")
    p.add_argument("--n-frag", type=int, default=10, dest="n_frag",
                   help="Colorless fragmentations per event (default 10)")
    p.add_argument("--seed", type=int, default=1, help="base seed (default 1)")
    p.add_argument("--events", default=None,
                   help="event range a:b (python slice of the file's events; default all)")
    p.add_argument("--use-stored-seeds", action="store_true", dest="use_stored_seeds",
                   help="replay the seeds a --validate-inline job recorded (bulk_jet and "
                        "jet_frag; one fragmentation per event)")
    p.add_argument("--legacy-seeds", action="store_true", dest="legacy_seeds",
                   help="derive the seeds from (--seed, tag, unit, sample) only, as before "
                        "the production file entered them: to reproduce older hadron files. "
                        "Their events share seeds across the files of a campaign")
    p.add_argument("--common-seeds", action="store_true", dest="common_seeds",
                   help="seed every event's bulk_jet with its background's bulk_bg seed, so "
                        "the two legs draw the same random numbers as long as they stay "
                        "aligned (PLAN_iSS_optim.md, Part B, first step). Not with "
                        "--use-stored-seeds or --oversample-bg")
    p.add_argument("--correlated", action="store_true",
                   help="correlated sampling in iSS (random numbers addressed by the cell), "
                        "with --common-seeds: jet and background give the same hadrons where "
                        "their surfaces agree, so jet - background is much less noisy "
                        "(PLAN_iSS_optim.md, Part B). Analyse it paired, sample by sample")
    p.add_argument("--correlated-block", default=None, dest="correlated_block",
                   metavar="DTAU,DX,DETA",
                   help="block size of --correlated in tau [fm/c], x and y [fm], eta "
                        "(default: jetscape_main.xml's, 0.5,1.0,0.5)")
    p.add_argument("--check", nargs="?", const="", default=None, metavar="JET_FRAG",
                   help="replay a jet_frag file (default: <out-dir>/<stem>_hadrons_jet_frag.h5) "
                        "with the settings and seeds it records and compare hadron by hadron; "
                        "exit 1 on a difference. Writes nothing; --seed, --n-frag, the "
                        "precision and selection options are taken from the file")
    p.add_argument("--diagnose-colored", action="store_true", dest="diagnose_colored",
                   help="only run the colour-flow diagnostic (ColoredHadronization on the "
                        "stored partons) and write <stem>_colored_diagnostic.json")
    p.add_argument("--colored-event", type=int, default=None, dest="colored_event",
                   help=argparse.SUPPRESS)   # internal: one event through Colored, in a child
    p.add_argument("--keep-bits-p", type=_mantissa_bits, default=None, dest="keep_bits_p",
                   help="round the momenta p = (E, px, py, pz) to this many float32 mantissa "
                        "bits (1-23; default: full precision). 12 bits: relative error "
                        "<= 1.2e-4. Decide per campaign (see the README)")
    p.add_argument("--keep-bits-x", type=_mantissa_bits, default=None, dest="keep_bits_x",
                   help="round the positions x = (t, x, y, z) likewise; 8 bits: <= 2e-3, "
                        "i.e. <= 0.03 fm at 15 fm (MUSIC's cells are 0.2 fm)")
    p.add_argument("--eta-max", type=_positive_float, default=None, dest="eta_max",
                   help="store only hadrons with pseudorapidity |eta| < ETA_MAX (all tags; "
                        "default: all hadrons). 2: ~54%% of the bulk hadrons, ~60%% of the "
                        "bytes. Decide per campaign (see the README)")
    p.add_argument("--charged", action="store_true",
                   help="store only charged hadrons (all tags; default: all hadrons)")
    p.add_argument("--no-x", action="store_true", dest="no_x",
                   help="store no positions t, x, y, z (all tags; ~40%% of the bytes)")
    p.add_argument("--keep-primaries", type=int, default=0, dest="keep_primaries",
                   metavar="N",
                   help="also write iSS's undecayed hadrons of the first N oversamples per "
                        "unit to <stem>_hadrons_bulk_{jet,bg}_primary.h5 (for SMASH later; "
                        "full precision, positions, no selection). Default 0: none")
    p.add_argument("--no-initiators", action="store_true", dest="no_initiators",
                   help="do not copy the pair file's shower initiators into bulk_jet and "
                        "jet_frag (initiators/)")
    p.add_argument("--add-initiators", action="store_true", dest="add_initiators",
                   help="only add initiators/ to the existing bulk_jet and jet_frag outputs "
                        "(no hadronization; --force replaces an existing group)")
    p.add_argument("--out-dir", default=None, dest="out_dir",
                   help="output directory (default: next to the input)")
    p.add_argument("--hadronize-xml", default=os.path.join(HERE, "hadronize.xml"),
                   dest="hadronize_xml")
    p.add_argument("--build", default=os.path.join(rp.XSCAPE, "build_gpu"),
                   help="X-SCAPE build tree (default: build_gpu)")
    p.add_argument("--main-xml", default=os.path.join(rp.XSCAPE, "config", "jetscape_main.xml"),
                   dest="main_xml")
    p.add_argument("--workdir", default=None,
                   help="private working directory (default: <out-dir>/work_hadronize/<stem>)")
    p.add_argument("--keep-workdir", action="store_true", dest="keep_workdir")
    p.add_argument("--force", action="store_true", help="overwrite existing outputs")
    p.add_argument("--skip-complete", action="store_true", dest="skip_complete",
                   help="for campaigns: skip tags whose output exists and is complete, redo "
                        "the missing or incomplete ones (exit 0 if nothing is left to do)")
    return p.parse_args(argv)


#: ``seed_scheme`` attribute of the hadron files; files without it are "legacy"
SEED_SCHEMES = ("production_file", "legacy")


def unit_seed(base, tag, unit, sample=0, prod=None):
    """Independent, reproducible 31-bit seed per (base, production file, tag, unit, sample).

    ``prod`` is the production file's key (:func:`production_key`); None gives the legacy
    seeds, the same for every production file."""
    key = [int(base), TAG_INDEX[tag], int(unit), int(sample)]
    if prod is not None:
        key.insert(1, int(prod))
    s = np.random.SeedSequence(key)
    return int(s.generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF) or 1


def production_key(pf):
    """The production file's seed key: its particlize file's ``file_uuid`` as an integer."""
    import uuid

    text = str(pf.attrs.get("file_uuid", ""))
    try:
        return uuid.UUID(text).int
    except ValueError:
        raise ValueError(f"the particlize file has no valid file_uuid ({text!r}) to derive "
                         "independent seeds from; pass --legacy-seeds") from None


def _set_text(parent, tag, value):
    n = parent.find(tag)
    if n is None:
        n = ET.SubElement(parent, tag)
    n.text = str(value)


#: --oversample-bg values that give every background its own count (units/n_samples)
PER_UNIT_OVERSAMPLE_BG = ("auto", "per-pthat-bin")


def background_samples(pf, oversample, spec, cap):
    """-> ({background unit: iSS samples}, [units that hit the cap]).

    ``spec`` None: ``oversample``; a number: that; 'auto': ``oversample`` x the number of
    events in the file that use the background (``events/bg_unit``); 'per-pthat-bin'
    (run_prod_jet.py --pthat-bins files): ``oversample`` x the largest number of events that
    use it in one pTHat window (``events/pthat_bin``), i.e. --jets-per-bin.  At most ``cap``.
    """
    n_bg, _ = pf.bg_units()
    if spec is None:
        return {u: oversample for u in range(n_bg)}, []
    mode = str(spec).lower()
    if mode not in PER_UNIT_OVERSAMPLE_BG:
        try:
            n = int(spec)
        except ValueError:
            n = 0
        if n < 1:
            raise ValueError(f"--oversample-bg must be >= 1, 'auto' or 'per-pthat-bin', got "
                             f"{spec!r}")
        return {u: n for u in range(n_bg)}, []
    bg_unit = np.asarray(pf.events("bg_unit"), dtype=np.int64)
    if mode == "auto":
        uses = np.bincount(bg_unit, minlength=n_bg)
    else:
        if not pf.has_events("pthat_bin"):
            raise ValueError("--oversample-bg per-pthat-bin needs a run_prod_jet.py "
                             "--pthat-bins file (events/pthat_bin); for --reuse use 'auto'")
        window = np.asarray(pf.events("pthat_bin"), dtype=np.float64)
        if np.isnan(window).any():
            raise ValueError("--oversample-bg per-pthat-bin: some events have no "
                             "events/pthat_bin")
        window = window.astype(np.int64)
        n_win = int(window.max()) + 1 if len(window) else 1
        per_window = np.bincount(bg_unit * n_win + window,
                                 minlength=n_bg * n_win).reshape(n_bg, n_win)
        uses = per_window.max(axis=1)
    out, capped = {}, []
    for u in range(n_bg):
        n = oversample * max(int(uses[u]), 1)
        if n > cap:
            capped.append(u)
            n = cap
        out[u] = n
    return out, capped


def write_job_xml(a, path, n_exec, oversample):
    """hadronize.xml with this run's iSS paths, oversample count, seed and event count."""
    tree = ET.parse(a.hadronize_xml)
    root = tree.getroot()
    _set_text(root, "nEvents", max(1, n_exec))
    root.find("Random/seed").text = str(a.seed)
    iss = root.find("SoftParticlization/iSS")
    iss_dir = os.path.join(rp.XSCAPE, "external_packages", "iSS")
    _set_text(iss, "iSS_input_file", os.path.join(iss_dir, "iSS_parameters.dat"))
    _set_text(iss, "iSS_table_path", os.path.join(iss_dir, "iSS_tables"))
    _set_text(iss, "iSS_particle_table_path", os.path.join(iss_dir, "iSS_tables"))
    _set_text(iss, "iSS_working_path", ".")
    _set_text(iss, "number_of_repeated_sampling", oversample)
    _set_text(iss, "keep_primaries", max(0, int(a.keep_primaries)))
    if a.correlated:
        _set_text(iss, "correlated_sampling", 1)
        if a.correlated_block:
            try:
                dtau, dx, deta = (float(v) for v in a.correlated_block.split(","))
            except ValueError:
                sys.exit(f"hadronize.py: --correlated-block wants DTAU,DX,DETA, got "
                         f"{a.correlated_block!r}")
            _set_text(iss, "correlated_block_dtau", dtau)
            _set_text(iss, "correlated_block_dx", dx)
            _set_text(iss, "correlated_block_deta", deta)
    tree.write(path)
    return ET.tostring(root, encoding="unicode")


#: iSS's hadron list per MUSIC EOS (iSS readindata.cpp)
HADRON_LIST = {"9": "UrQMD", "91": "SMASH"}


def iss_commit():
    """The iSS checkout the build uses (best effort; '' if git cannot tell)."""
    import subprocess
    try:
        return subprocess.run(["git", "-C", os.path.join(rp.XSCAPE, "external_packages", "iSS"),
                               "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def iss_decay_attrs(core, eos, xml_root):
    """Hadron-file attributes saying which hadron list iSS sampled and how it decayed it."""
    hadron_list = HADRON_LIST.get(eos, "other")
    iss = xml_root.find("SoftParticlization/iSS")
    decays = int((iss.findtext("Perform_resonance_decays") or "0").strip() or 0) == 1
    smash_decays = int((iss.findtext("decay_smash_list") or "0").strip() or 0) == 1
    if not decays:
        how = "none"
    elif hadron_list == "SMASH":
        how = ("iSS, Breit-Wigner masses for the unstable daughters of channels closed at "
               "the pole masses; eta, Sigma0 stable"
               if smash_decays and getattr(core, "ISS_DECAYS_SMASH_LIST", False) else "none")
    else:
        how = "iSS, pole masses"
    return {"hadron_list": hadron_list, "iss_decays": how, "decayed": how != "none",
            "iss_commit": iss_commit()}


def seed_scheme_of(path):
    """``seed_scheme`` of a hadron file ("legacy" when it predates the attribute)."""
    import h5py
    with h5py.File(path, "r") as f:
        return str(f.attrs.get("seed_scheme", "legacy"))


def _complete(path):
    """True if ``path`` is a hadron file that was closed as complete."""
    if not os.path.exists(path):
        return False
    try:
        import h5py
        with h5py.File(path, "r") as f:
            return bool(f.attrs.get("complete", False))
    except OSError:                              # truncated by a kill
        return False


def read_initiators(pf, particlize):
    """``(data, offsets)`` of the shower initiators: the particlize file's own
    ``initiators/`` (format version 2), else the pair file's next to ``particlize``; None
    (with a warning) when neither has them."""
    from jetscape.hadrons_h5 import pair_initiators

    own = pf.initiators_all()
    if own is not None:
        return own
    pair = pf.attrs.get("pair_file")
    path = os.path.join(os.path.dirname(particlize), str(pair)) if pair else None
    try:
        found = pair_initiators(path, nevents=pf.nevents)
    except ValueError as err:
        found, why = None, str(err)
    else:
        why = (f"pair file {path} not found" if path and not os.path.exists(path)
               else f"{path} has no shower/initiators" if path
               else "the particlize file names no pair_file")
    if found is None:
        print(f"hadronize.py: WARNING -- no shower initiators (not in the particlize file, "
              f"and {why}): bulk_jet and jet_frag are written without initiators/, so "
              "jet-relative analyses will need the pair file", file=sys.stderr)
    return found


def event_range(spec, n):
    if not spec:
        return list(range(n))
    parts = [int(x) if x else None for x in spec.split(":")]
    return list(range(n))[slice(*parts)]


def colour_stats(partons):
    """Colour-flow numbers of one event's final partons (FINAL_PARTON_COLUMNS rows)."""
    pid, pstat = partons[:, 1].astype(int), partons[:, 2].astype(int)
    col, acol = partons[:, 12].astype(int), partons[:, 13].astype(int)
    qg = (np.abs(pid) <= 6) | (pid == 21)

    def unpaired(sel):
        n = 0
        for sh in np.unique(partons[sel, 0]):
            m = sel & (partons[:, 0] == sh) & qg
            c = [x for x in col[m] if x]
            ac = [x for x in acol[m] if x]
            for x in list(c):
                if x in ac:
                    ac.remove(x)
                    c.remove(x)
            n += len(c) + len(ac)
        return n

    everything = np.ones(len(pid), bool)
    kept = ~np.isin(pstat, (-11, -17))
    return {
        "n_final": int(len(pid)),
        "n_qg": int(qg.sum()),
        "frac_qg_colourless": float((qg & (col == 0) & (acol == 0)).sum() / max(qg.sum(), 1)),
        "unpaired_tags_all": unpaired(everything),
        "unpaired_tags_without_absorbed": unpaired(kept),
        "n_absorbed": int((~kept).sum()),
    }


def check_settings(a, core, pf, path):
    """--check: take over what the jet_frag file ``path`` was made with (seeds, fragmentations,
    precision, selection) -> (its hadronize_xml or None, the units it holds)."""
    import h5py

    from jetscape.hadrons_h5 import hadron_precision, hadron_selection

    if not os.path.exists(path):
        sys.exit(f"hadronize.py: --check: {path} not found (pass the jet_frag file)")
    with h5py.File(path, "r") as f:
        at = dict(f.attrs)
        units = (f["units"]["unit"][:] if "units" in f and "unit" in f["units"]
                 else np.arange(len(f["hadrons"]["unit_offsets"]) - 1))
    if str(at.get("tag")) != "jet_frag":
        sys.exit(f"hadronize.py: --check: {path} is no jet_frag file (tag {at.get('tag')!r})")
    uuid, src = str(pf.attrs.get("file_uuid", "")), str(at.get("source_uuid", ""))
    if uuid and src and uuid != src:
        sys.exit(f"hadronize.py: --check: {os.path.basename(path)} was made from another "
                 f"particlize file ({at.get('source')}, file_uuid {src})")
    have, job = int(getattr(core, "PYTHIA_VERSION", 0)), int(at.get("pythia_version", 0))
    if job and have and job != have:
        sys.exit(f"hadronize.py: --check: the file was made with Pythia {job}, this build has "
                 f"{have}: the same seeds give other hadrons")
    if not job:
        print("hadronize.py: note -- --check: the file records no pythia_version (made before "
              "smash_ready); a difference may only mean another Pythia", file=sys.stderr)
    if not at.get("complete", False):
        print("hadronize.py: note -- --check: the file is incomplete; checking the events it "
              "holds", file=sys.stderr)
    prec, sel = hadron_precision(path), hadron_selection(path)
    a.seed = int(at.get("base_seed", a.seed))
    a.legacy_seeds = str(at.get("seed_scheme", "legacy")) == "legacy"
    a.use_stored_seeds = bool(at.get("stored_seeds", False))
    a.n_frag = int(at["n_samples"])
    a.keep_bits_p, a.keep_bits_x = prec["p"], prec["x"]
    a.eta_max, a.charged, a.no_x = sel["eta_max"], sel["charged_only"], not sel["positions"]
    a.common_seeds = a.correlated = False
    a.oversample_bg, a.keep_primaries = None, 0
    a.no_initiators = a.force = True
    xml = at.get("hadronize_xml")
    if not xml:
        print(f"hadronize.py: note -- --check: the file records no hadronize_xml; using "
              f"{a.hadronize_xml}", file=sys.stderr)
    return (str(xml) if xml else None), [int(u) for u in units]


def compare_jet_frag(ref_path, new_path, events):
    """--check: compare the regenerated jet_frag file with the original, event by event;
    -> the number of events that differ."""
    from jetscape.hadrons_h5 import Hadrons

    ref, new = Hadrons.from_h5(ref_path), Hadrons.from_h5(new_path)
    n_bad = 0
    for e in events:
        nr, nn = ref.n_samples(e), new.n_samples(e)
        bad = [] if nr == nn else [f"{nn} fragmentations, the file has {nr}"]
        sr = int(ref.units["seed"][ref.unit_index(e)]) if "seed" in ref.units else None
        sn = int(new.units["seed"][new.unit_index(e)])
        if sr is not None and sr != sn:
            bad.append(f"seed {sn}, the file has {sr}")
        for k in range(min(nr, nn)):
            r, n = ref.sample_event(e, k), new.sample_event(e, k)
            diff = [key for key in ("pid", "pstat", "p", "x")
                    if not np.array_equal(r[key], n[key])]
            if diff:
                bad.append(f"fragmentation {k}: {', '.join(diff)} differ ({len(n['pid'])} "
                           f"hadrons, the file {len(r['pid'])})")
        n_bad += bool(bad)
        print(f"  event {e}: " + ("same as the file" if not bad else
                                  "DIFFERENT -- " + "; ".join(bad[:3])
                                  + (f" (+{len(bad) - 3} more)" if len(bad) > 3 else "")),
              flush=True)
    return n_bad


def _terminate(signum, frame):
    # SIGTERM would kill the process without running the finally blocks, leaving
    # truncated HDF5 files; turn it into an exception so the writers close.
    raise SystemExit(128 + signum)


def main(argv=None):
    signal.signal(signal.SIGTERM, _terminate)
    a = parse_args(argv)
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    bad = set(tags) - set(TAGS)
    if bad:
        sys.exit(f"hadronize.py: unknown tag(s) {sorted(bad)}; choose from {TAGS}")
    for k in ("particlize", "hadronize_xml", "build", "main_xml"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    keep_bits = {"p": a.keep_bits_p, "x": a.keep_bits_x}

    from jetscape.hadrons_h5 import (INITIATOR_TAGS, PRIMARY_OF, HadronH5Writer,
                                     _keep_bits_map, add_initiators, hadron_precision,
                                     hadron_selection)
    from jetscape.particlize_h5 import ParticlizeFile

    pf = ParticlizeFile(a.particlize)
    stem = os.path.basename(a.particlize)
    stem = stem[:-len("_particlize.h5")] if stem.endswith("_particlize.h5") \
        else os.path.splitext(stem)[0]
    out_dir = os.path.abspath(a.out_dir or os.path.dirname(a.particlize))
    os.makedirs(out_dir, exist_ok=True)
    events = event_range(a.events, pf.nevents)
    check = check_xml = None
    if a.check is not None:
        if a.add_initiators or a.diagnose_colored or a.skip_complete:
            sys.exit("hadronize.py: --check replays a jet_frag file: not with --add-initiators, "
                     "--diagnose-colored or --skip-complete")
        from jetscape import pyjetscape_core as core
        check = os.path.abspath(a.check or os.path.join(out_dir, f"{stem}_hadrons_jet_frag.h5"))
        check_xml, held = check_settings(a, core, pf, check)
        tags, a.tags = ["jet_frag"], "jet_frag"
        keep_bits = {"p": a.keep_bits_p, "x": a.keep_bits_x}
        held = set(held)
        events = [e for e in events if e in held]
        if not events:
            sys.exit(f"hadronize.py: --check: {os.path.basename(check)} holds none of the "
                     "selected events")
        a.workdir = a.workdir or os.path.join(out_dir, "work_hadronize", f"{stem}_check")
        os.makedirs(a.workdir, exist_ok=True)
        if check_xml:
            a.hadronize_xml = os.path.join(os.path.abspath(a.workdir), "checked_hadronize.xml")
            with open(a.hadronize_xml, "w") as fh:
                fh.write(check_xml)
    want_ini = (not a.no_initiators and not a.diagnose_colored and a.colored_event is None
                and any(t in INITIATOR_TAGS for t in tags))
    ini = read_initiators(pf, a.particlize) if want_ini else None
    if a.add_initiators:
        if ini is None:
            sys.exit("hadronize.py: --add-initiators: no initiators to add (see above)")
        for t in [t for t in tags if t in INITIATOR_TAGS]:
            path = os.path.join(out_dir, f"{stem}_hadrons_{t}.h5")
            if not os.path.exists(path):
                print(f"hadronize.py: {os.path.basename(path)}: missing, skipped")
            elif add_initiators(path, *ini, force=a.force):
                print(f"hadronize.py: {os.path.basename(path)}: initiators/ added")
            else:
                print(f"hadronize.py: {os.path.basename(path)}: has initiators/ already "
                      "(--force replaces them)")
        return 0

    from jetscape import pyjetscape_core as core
    if a.diagnose_colored and a.colored_event is None:
        return diagnose_colored(a, pf, events, out_dir, stem)
    if a.colored_event is not None:
        tags = []
    for t in tags:
        if t in ("bulk_jet", "bulk_bg") and t.split("_")[1] not in pf.legs:
            sys.exit(f"hadronize.py: {a.particlize} has no {t.split('_')[1]} surface "
                     f"(legs {pf.legs}); drop {t} from --tags")
        if t == "jet_frag" and "partons" not in pf.f:
            sys.exit(f"hadronize.py: {a.particlize} has no partons/")
        if t == "jet_frag" and pf.attrs.get("jet_frag_inline"):
            # e.g. prod_OO_isr: iColoredHadronization ran in the job (it needs the beam
            # remnants); Colorless here would drop its partons (pstat 1000) and overwrite it
            sys.exit(f"hadronize.py: {a.particlize}: the jet hadrons were made in the job "
                     f"({pf.attrs['jet_frag_inline']}); drop jet_frag from --tags")
    if a.use_stored_seeds and any(v is not None for v in keep_bits.values()):
        print("hadronize.py: note -- --keep-bits with --use-stored-seeds: the in-job "
              "(--validate-inline) hadrons are full precision, so compare after rounding "
              "them the same way, or validate without --keep-bits", file=sys.stderr)
    if a.correlated:
        a.common_seeds = True
    if a.correlated_block is not None and not a.correlated:
        sys.exit("hadronize.py: --correlated-block needs --correlated")
    if a.common_seeds and (a.use_stored_seeds or a.oversample_bg is not None):
        sys.exit("hadronize.py: --common-seeds / --correlated need the jet seeds derived "
                 "from the backgrounds and the same number of samples on both legs: drop "
                 "--use-stored-seeds / --oversample-bg")
    try:
        prod = None if a.legacy_seeds else production_key(pf)
    except ValueError as err:
        sys.exit(f"hadronize.py: {err}")
    seed_scheme = "legacy" if a.legacy_seeds else "production_file"
    if a.use_stored_seeds:
        for key in ("inline_iss_seed_jet", "inline_pythia_seed"):
            if not pf.has_events(key):
                sys.exit(f"hadronize.py: --use-stored-seeds needs events/{key} (a "
                         "--validate-inline job)")
    xml_root = ET.parse(a.hadronize_xml).getroot()
    oversample = a.oversample or int(
        xml_root.find("SoftParticlization/iSS/number_of_repeated_sampling").text)
    n_frag = 1 if a.use_stored_seeds else a.n_frag
    try:
        bg_samples, bg_capped = background_samples(pf, oversample, a.oversample_bg,
                                                   a.oversample_bg_max)
    except ValueError as err:
        sys.exit(f"hadronize.py: {err}")
    if bg_capped and "bulk_bg" in [t.strip() for t in a.tags.split(",")]:
        print(f"hadronize.py: WARNING -- --oversample-bg {a.oversample_bg} capped "
              f"{len(bg_capped)} "
              f"background(s) at --oversample-bg-max {a.oversample_bg_max}", file=sys.stderr)

    if a.keep_primaries < 0:
        sys.exit("hadronize.py: --keep-primaries must be >= 0")
    outs = {t: os.path.join(out_dir, f"{stem}_hadrons_{t}.h5") for t in tags}
    if check:
        # the replay goes to the working directory, which is removed at the end
        outs = {"jet_frag": os.path.join(os.path.abspath(a.workdir),
                                         f"{stem}_hadrons_jet_frag_check.h5")}
    # --keep-primaries: the undecayed hadrons of each bulk tag, in a file of their own
    pouts = ({t: os.path.join(out_dir, f"{stem}_hadrons_{PRIMARY_OF[t]}.h5")
              for t in tags if t in PRIMARY_OF} if a.keep_primaries > 0 else {})
    if a.skip_complete:
        done = [t for t in tags if _complete(outs[t])
                and (t not in pouts or _complete(pouts[t]))]
        if done:
            print(f"hadronize.py: {os.path.basename(a.particlize)}: already complete: "
                  f"{', '.join(done)}")
        wanted = _keep_bits_map(keep_bits)
        for t in done:
            have_scheme = seed_scheme_of(outs[t])
            if have_scheme != seed_scheme:
                print(f"hadronize.py: WARNING -- {os.path.basename(outs[t])} is complete but its "
                      f"seeds are {have_scheme!r}, not {seed_scheme!r}; kept as it is "
                      "(--force redoes it)", file=sys.stderr)
            have = hadron_precision(outs[t])
            if have != wanted:
                print(f"hadronize.py: WARNING -- {os.path.basename(outs[t])} is complete but "
                      f"was written with precision {have}, not the requested {wanted} (None "
                      "= full float32); kept as it is (--force redoes it)", file=sys.stderr)
            have_sel = hadron_selection(outs[t])
            want_sel = {"eta_max": a.eta_max, "charged_only": bool(a.charged),
                        "positions": not a.no_x}
            if have_sel != want_sel:
                print(f"hadronize.py: WARNING -- {os.path.basename(outs[t])} is complete but "
                      f"was written with the selection {have_sel}, not the requested "
                      f"{want_sel} (--eta-max, --charged, --no-x); kept as it is (--force "
                      "redoes it)", file=sys.stderr)
        tags = [t for t in tags if t not in done]
        outs = {t: outs[t] for t in tags}
        pouts = {t: pouts[t] for t in tags if t in pouts}
        if not tags:
            return 0
        a.force = True                          # what is left is missing or incomplete

    bg_units = []
    if "bulk_bg" in tags:
        seen = set()
        for e in events:
            u = int(pf.events("bg_unit")[e])
            if u not in seen:
                seen.add(u)
                bg_units.append((u, e))
    n_exec = (len(events) if "bulk_jet" in tags else 0) + len(bg_units)

    for path in list(outs.values()) + list(pouts.values()):
        if os.path.exists(path) and not a.force:
            sys.exit(f"hadronize.py: {path} exists (use --force)")

    # Private working directory: iSS reads music_input from it, Pythia may write there.
    workdir = os.path.abspath(a.workdir or os.path.join(out_dir, "work_hadronize", stem))
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "music_input"), "w") as fh:
        fh.write(pf.music_input())
    eos = [ln.split()[1] for ln in pf.music_input().splitlines()
           if len(ln.split()) > 1 and ln.split()[0] == "EOS_to_use"]
    eos = eos[0] if eos else ""
    decay_attrs = iss_decay_attrs(core, eos, xml_root)
    if eos == "91" and any(t.startswith("bulk") for t in tags) and not decay_attrs["decayed"]:
        # iSS (readindata.cpp) takes the SMASH hadron list for EOS 91, and an iSS before
        # smash_ready (or <decay_smash_list> 0) leaves its decays to SMASH
        print("hadronize.py: WARNING -- MUSIC ran EOS 91 (SMASH hadron list) and this iSS "
              "does NOT decay that list (core.ISS_DECAYS_SMASH_LIST False, or "
              "<decay_smash_list> 0): rho, omega, K*, Delta ... stay in the bulk hadrons.",
              file=sys.stderr)
    job_xml = os.path.join(workdir, "hadronize_job.xml")
    xml_text = write_job_xml(a, job_xml, n_exec, oversample)
    main_xml = os.path.join(workdir, "jetscape_main.xml")
    rp._absolute_parent_paths(a.main_xml, main_xml, a.build)
    cwd = os.getcwd()
    os.chdir(workdir)
    print(f"hadronize.py: {a.particlize}\n  {len(events)} event(s), tags {', '.join(tags) or '-'}"
          f", iSS oversample {oversample}"
          + (f" (backgrounds: {sorted(set(bg_samples.values()))})"
             if "bulk_bg" in tags and set(bg_samples.values()) != {oversample} else "")
          + f", {n_frag} fragmentation(s)/event, base seed "
          f"{a.seed}{' (stored seeds)' if a.use_stored_seeds else ''}"
          + (" (legacy seeds: shared across production files)" if a.legacy_seeds else "")
          + (f", p/x rounded to {a.keep_bits_p or 23}/{a.keep_bits_x or 23} mantissa bits"
             if any(v is not None for v in keep_bits.values()) else "")
          + (f", only |eta| < {a.eta_max:g} stored" if a.eta_max is not None else "")
          + (", charged hadrons only" if a.charged else "")
          + (", no positions" if a.no_x else "")
          + (f", undecayed hadrons of {a.keep_primaries} oversample(s) kept"
             if pouts else "")
          + f", EOS {eos or '?'} ({decay_attrs['hadron_list']} list, decays: "
          f"{decay_attrs['iss_decays']})"
          + f"\n  workdir {workdir}")

    jetscape = core.JetScapePerEvent()
    jetscape.SetXMLMainFileName(main_xml)
    jetscape.SetXMLUserFileName(job_xml)
    replay = iss = None
    if "bulk_jet" in tags or "bulk_bg" in tags:
        from jetscape.surface_replay import SurfaceReplay
        replay = SurfaceReplay()
        iss = core.create_module("iSS")
        jetscape.Add(replay)
        jetscape.Add(iss)
    jetscape.Init()
    jetscape.ExecInit()
    if iss is not None:
        # hadrons as arrays, not one framework Hadron object each: same numbers, ~8x
        # less memory per oversample (nothing here needs the framework's hadron list)
        core.soft_set_compact_output(iss, True)
    colorless = None
    if "jet_frag" in tags:
        colorless = core.create_module("ColorlessHadronization")
        colorless.Init()

    if a.colored_event is not None:
        return colored_one_event(core, pf, a.colored_event, cwd)

    common = {"source": os.path.basename(a.particlize),
              "source_uuid": str(pf.attrs.get("file_uuid", "")),
              "hadronize_xml": xml_text, "base_seed": a.seed,
              "stored_seeds": bool(a.use_stored_seeds),
              "seed_scheme": seed_scheme,
              "music_input": pf.music_input()}
    bulk_attrs = {k: decay_attrs[k] for k in ("hadron_list", "iss_decays", "iss_commit")}
    # how the two iSS legs were sampled relative to each other: only on bulk_jet and bulk_bg
    # (jet_frag's Pythia seeds are its own whatever these options)
    pairing = {"common_seeds": bool(a.common_seeds), "correlated_sampling": bool(a.correlated)}
    writers = {}
    for t in tags:
        extra = {} if t == "jet_frag" else dict(pairing)
        if t == "jet_frag":
            n = n_frag
        elif t == "bulk_bg" and a.oversample_bg is not None:
            per_unit = str(a.oversample_bg).lower() in PER_UNIT_OVERSAMPLE_BG
            n = 0 if per_unit else int(a.oversample_bg)  # 0: per unit, see units/n_samples
            extra.update({"oversample_bg": str(a.oversample_bg),
                          "oversample_bg_max": int(a.oversample_bg_max),
                          "oversample_jet": int(oversample)})
        else:
            n = oversample
        if t == "jet_frag":
            extra.update(generator="ColorlessHadronization (X-SCAPE)", decayed=True,
                         pythia_version=int(getattr(core, "PYTHIA_VERSION", 0)))
        else:
            extra.update(bulk_attrs, generator="iSS (X-SCAPE iSpectraSamplerWrapper)",
                         decayed=bool(decay_attrs["decayed"]))
        writers[t] = HadronH5Writer(outs[t], tag=t, n_samples=n, keep_bits=keep_bits,
                                    eta_max=a.eta_max, charged_only=a.charged,
                                    positions=not a.no_x,
                                    initiators=ini is not None and t in INITIATOR_TAGS,
                                    attrs=dict(common, **extra))
        if t in pouts:
            # full precision, positions, no selection: what SMASH needs
            n_keep = a.keep_primaries if n == 0 else min(a.keep_primaries, n)
            writers[PRIMARY_OF[t]] = HadronH5Writer(
                pouts[t], tag=PRIMARY_OF[t], n_samples=n_keep,
                attrs=dict(common, **dict(extra, decayed=False), primaries_of=t,
                           keep_primaries=int(a.keep_primaries)))

    def run_iss(cells, seed, n_samples):
        """-> (hadrons, seed used, undecayed hadrons or None)."""
        if len(cells) == 0:
            return None, seed, None
        replay.load(cells)
        core.soft_set_number_of_samples(iss, int(n_samples))
        core.soft_set_next_random_seed(iss, int(seed))
        jetscape.ExecPerEvent()
        h = core.soft_hadrons_numpy(iss)
        prim = core.soft_primaries_numpy(iss) if pouts else None
        used = int(core.soft_last_random_seed(iss))
        jetscape.ClearPerEvent()
        if used != int(seed) or replay.n_cells != len(cells):
            # iSS (or the replay) was switched off for this framework event -- e.g. hydro
            # reuse -- and h would be the previous surface's hadrons
            raise RuntimeError(f"iSS did not sample this surface (seed {used} != {seed}, "
                               f"replayed {replay.n_cells} of {len(cells)} cells); check "
                               "<setReuseHydro> false in the hadronization XML")
        if len(h["sample_counts"]) != int(n_samples):
            raise RuntimeError(f"iSS made {len(h['sample_counts'])} samples, "
                               f"{n_samples} were asked for")
        if pouts:
            want = min(int(a.keep_primaries), int(n_samples))
            got = 0 if prim is None else len(prim["sample_counts"])
            if got != want:
                raise RuntimeError(f"iSS kept the primaries of {got} samples, {want} were "
                                   "asked for (an X-SCAPE without <keep_primaries>?)")
        return h, used, prim

    def initiators_of(e):
        return None if ini is None else ini[0][int(ini[1][e]):int(ini[1][e + 1])]

    t0 = time.time()
    finished = False
    bg_done = set()
    bg_first = dict(bg_units)
    try:
        for n, e in enumerate(events):
            msg = [f"event {e}"]
            if "bulk_jet" in tags:
                if a.use_stored_seeds:
                    seed = int(pf.events("inline_iss_seed_jet")[e])
                elif a.common_seeds:
                    seed = unit_seed(a.seed, "bulk_bg", int(pf.events("bg_unit")[e]), prod=prod)
                else:
                    seed = unit_seed(a.seed, "bulk_jet", e, prod=prod)
                cells = pf.surface_unit("jet", e)
                h, used, prim = run_iss(cells, seed, oversample)
                writers["bulk_jet"].append_unit(h if h is not None else [], initiators_of(e),
                                                unit=e, event=e, seed=used,
                                                n_cells=len(cells))
                if "bulk_jet_primary" in writers:
                    writers["bulk_jet_primary"].append_unit(
                        prim if prim is not None else [], unit=e, event=e, seed=used,
                        n_cells=len(cells))
                msg.append(f"bulk_jet {0 if h is None else len(h['pid'])} hadrons")
            if "bulk_bg" in tags:
                u = int(pf.events("bg_unit")[e])
                if u not in bg_done:
                    cells = pf.surface_unit("bg", u)
                    h, used, prim = run_iss(cells, unit_seed(a.seed, "bulk_bg", u, prod=prod),
                                             bg_samples[u])
                    writers["bulk_bg"].append_unit(h if h is not None else [], unit=u,
                                                   event=bg_first[u], seed=used,
                                                   n_cells=len(cells))
                    if "bulk_bg_primary" in writers:
                        writers["bulk_bg_primary"].append_unit(
                            prim if prim is not None else [], unit=u, event=bg_first[u],
                            seed=used, n_cells=len(cells))
                    bg_done.add(u)
                    msg.append(f"bulk_bg (unit {u}) {0 if h is None else len(h['pid'])}")
            if "jet_frag" in tags:
                partons = pf.partons(e)
                samples, seeds = [], []
                for s in range(n_frag):
                    seed = (int(pf.events("inline_pythia_seed")[e]) if a.use_stored_seeds
                            else unit_seed(a.seed, "jet_frag", e, s, prod=prod))
                    if len(partons):
                        d = core.hadronize_partons(colorless, partons, seed)
                    else:
                        d = {k: np.zeros((0, 4) if k in ("p", "x") else 0) for k in
                             ("pid", "pstat", "p", "x")}
                    samples.append(d)
                    seeds.append(seed)
                writers["jet_frag"].append_unit(samples, initiators_of(e), unit=e, event=e,
                                                seed=seeds[0], n_partons=len(partons))
                msg.append(f"jet_frag {sum(len(d['pid']) for d in samples) / n_frag:.1f}"
                           "/sample")
            print("  " + ", ".join(msg), flush=True)
        jetscape.Finish()
        finished = True
    finally:
        # complete only if every event went through: an exception, Ctrl-C or SIGTERM
        # (run_hadronize.py stopping its children) leaves readable, incomplete files that
        # --skip-complete redoes
        for w in writers.values():
            w.close(complete=finished)
        os.chdir(cwd)
    n_bad = 0
    if check:
        print(f"hadronize.py: --check against {check}")
        n_bad = compare_jet_frag(check, outs["jet_frag"], events)
    if not a.keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
        try:
            os.rmdir(os.path.dirname(workdir))  # work_hadronize/, once it is empty
        except OSError:
            pass
    print(f"hadronize.py: done in {time.time() - t0:.1f} s")
    if check:
        print(f"hadronize.py: {len(events) - n_bad} of {len(events)} event(s) reproduce "
              f"{os.path.basename(check)}" + (" -- MISMATCH" if n_bad else ""))
        return 1 if n_bad else 0
    for t, path in list(outs.items()) + [(PRIMARY_OF[t], p) for t, p in pouts.items()]:
        print(f"  {t:9s} -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return 0


def colored_one_event(core, pf, e, cwd):
    """Child process of --diagnose-colored: one event through ColoredHadronization."""
    colored = core.create_module("ColoredHadronization")
    colored.Init()
    partons = pf.partons(e)
    d = core.hadronize_partons(colored, partons) if len(partons) else {"pid": []}
    pid = np.asarray(d["pid"])
    partonic = (np.abs(pid) <= 6) | (pid == 21)
    os.chdir(cwd)
    print("COLORED_RESULT " + json.dumps({"n_out": int(len(pid)),
                                          "n_out_partons": int(partonic.sum())}), flush=True)
    return 0


def diagnose_colored(a, pf, events, out_dir, stem):
    """Colour flow of the stored partons, and what ColoredHadronization makes of it.

    ColoredHadronization can abort the process (an assertion on a Pythia output id), so each
    event runs in its own child process; a crash is recorded, not fatal.
    """
    import subprocess

    rows = []
    for e in events:
        st = colour_stats(pf.partons(e))
        st["event"] = int(e)
        cmd = [sys.executable, os.path.abspath(__file__), a.particlize, "--colored-event",
               str(e), "--build", a.build, "--main-xml", a.main_xml,
               "--hadronize-xml", a.hadronize_xml, "--out-dir", out_dir, "--force",
               "--workdir", os.path.join(out_dir, "work_hadronize", f"{stem}_colored_{e}")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        res = [ln for ln in r.stdout.splitlines() if ln.startswith("COLORED_RESULT ")]
        if r.returncode == 0 and res:
            out = json.loads(res[-1].split(" ", 1)[1])
            st.update(out, crashed=False,
                      failed=bool(out["n_out_partons"] > 0 or out["n_out"] == 0))
        else:
            tail = (r.stderr.strip().splitlines() or ["?"])[-1]
            st.update(crashed=True, failed=True, returncode=r.returncode,
                      error=tail[-300:])
        rows.append(st)
        print(f"  event {e}: {st}", flush=True)
    summary = {
        "source": a.particlize,
        "events": rows,
        "mean_frac_qg_colourless": float(np.mean([r["frac_qg_colourless"] for r in rows]))
        if rows else None,
        "failed_fraction": float(np.mean([r["failed"] for r in rows])) if rows else None,
        "crashed_fraction": float(np.mean([r["crashed"] for r in rows])) if rows else None,
    }
    path = os.path.join(out_dir, f"{stem}_colored_diagnostic.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"hadronize.py: colour diagnostic -> {path}: mean colourless q/g fraction "
          f"{summary['mean_frac_qg_colourless']}, ColoredHadronization failed in "
          f"{summary['failed_fraction']} (crashed in {summary['crashed_fraction']}) of events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
