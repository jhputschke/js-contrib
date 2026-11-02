"""
python/jetscape/hadronizer.py

Hadronizer -- the per-unit work of hadronizing a two-stage hydro event: iSS on a leg's
freeze-out cells, ColorlessHadronization on the event's final partons, their seeds, and the
hadron files (jetscape.hadrons_h5).  One code path for the two places it runs:

    example/prod_AuAu_0_10_jet/hadronize.py      afterwards, from a particlize file
    example/prod_AuAu_0_10_jet/run_prod_jet.py --hadrons-only
                                                 in the production job, from the particlize
                                                 writer's memory sink

so the job's hadrons are, by construction, what hadronize.py makes of the same cells,
partons and seeds (docs/Plans/PLAN_AuAu_hadrons_only.md).

    h = Hadronizer(core, iss=iss, colorless=colorless, base_seed=1, prod=production_key)
    h.open("bulk_jet", path, n_samples=1, attrs=common, primary_path=ppath)
    h.bulk("bulk_jet", e, cells, h.bulk_jet_seed(e, bg_unit), n_samples=1, event=e,
           initiators=ini)
    h.fragments(e, partons, [h.seed("jet_frag", e, s) for s in range(n_frag)],
                initiators=ini)
    h.extra_fragments(e, extra_partons, n_frag, initiators=extra_ini,
                      initiator_scatter=scat, scatterings=rows)   # extra hard scatterings
    h.close(complete=True)

With extra hard scatterings (PythiaGun's <extraHard>, docs/Plans/PLAN_multi_hard.md) an
event's final partons are split by their shower's scattering
(:func:`jetscape.particlize_h5.split_by_scatter`): the trigger's go to ``jet_frag`` exactly
as without extras (same partons, same seeds), the extras' to ``extra_frag``, with seeds of
their own tag: all together in one Colorless call (mode ``groups``) or one call per
scattering (``per_scattering``, each hadron's scattering kept).

iSS runs on given cells through ``core.soft_sample_cells`` (X-SCAPE's
``SoftParticlization::SampleCells``): no framework event, so it also works inside a
production process whose own event loop must not run again.  The iSS and Colorless modules
only need ``Init()`` (after the JETSCAPE instance's ``Init`` has read the XML).
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

__all__ = ["TAGS", "TAG_INDEX", "EXTRA_HADRONIZE", "SEED_SCHEMES", "PER_UNIT_OVERSAMPLE_BG", "HADRON_LIST",
           "KEEP_ALL_PRIMARIES", "unit_seed", "uuid_key", "samples_per_background",
           "iss_commit", "iss_decay_attrs", "tag_attrs", "configure_hadronize_xml",
           "collision_ecm", "set_hadronization_ecm", "Hadronizer"]

TAGS = ("bulk_jet", "bulk_bg", "jet_frag", "extra_frag")
TAG_INDEX = {t: k for k, t in enumerate(TAGS)}      # the seeds' key: append new tags only
#: how the extras' showers are hadronized (run_prod_jet.py / hadronize.py --extra-hadronize)
EXTRA_HADRONIZE = ("groups", "per_scattering")

#: ``seed_scheme`` attribute of the hadron files; files without it are "legacy"
SEED_SCHEMES = ("production_file", "legacy")

#: --oversample-bg values that give every background its own count (units/n_samples)
PER_UNIT_OVERSAMPLE_BG = ("auto", "per-pthat-bin")

#: iSS's hadron list per MUSIC EOS (iSS readindata.cpp)
HADRON_LIST = {"9": "UrQMD", "91": "SMASH"}

#: iSS <keep_primaries> for "every sample": iSS keeps min(N, samples)
KEEP_ALL_PRIMARIES = 1 << 30


def unit_seed(base, tag, unit, sample=0, prod=None):
    """Independent, reproducible 31-bit seed per (base, production file, tag, unit, sample).

    ``prod`` is the production file's key (:func:`uuid_key` of its ``file_uuid``); None gives
    the legacy seeds, the same for every production file."""
    key = [int(base), TAG_INDEX[tag], int(unit), int(sample)]
    if prod is not None:
        key.insert(1, int(prod))
    s = np.random.SeedSequence(key)
    return int(s.generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF) or 1


def uuid_key(text):
    """A ``file_uuid`` as the integer seed key (ValueError if it is not a UUID)."""
    import uuid

    return uuid.UUID(str(text)).int


def samples_per_background(oversample, spec, uses, cap):
    """iSS samples for one background -> (n, capped).

    ``spec`` None: ``oversample``; a number: that; 'auto' / 'per-pthat-bin': ``oversample``
    x ``uses``, the number of events using the background (in the file, or in one pTHat
    window), at most ``cap``."""
    if spec is None:
        return int(oversample), False
    mode = str(spec).lower()
    if mode not in PER_UNIT_OVERSAMPLE_BG:
        try:
            n = int(spec)
        except ValueError:
            n = 0
        if n < 1:
            raise ValueError(f"--oversample-bg must be >= 1, 'auto' or 'per-pthat-bin', got "
                             f"{spec!r}")
        return n, False
    n = int(oversample) * max(int(uses), 1)
    return (int(cap), True) if n > cap else (n, False)


def iss_commit(xscape):
    """The iSS checkout an X-SCAPE tree uses (best effort; '' if nothing can tell): git in
    the tree's external_packages/iSS, else the ``external_packages/iSS <sha> ...`` line of
    the tree's BUILD_INFO.txt (written by the production image and the xscape wheel)."""
    iss = os.path.join(xscape, "external_packages", "iSS")
    if os.path.isdir(iss):
        try:
            sha = subprocess.run(["git", "-C", iss, "rev-parse", "HEAD"], capture_output=True,
                                 text=True, timeout=10).stdout.strip()
            if sha:
                return sha
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        with open(os.path.join(xscape, "BUILD_INFO.txt")) as f:
            for line in f:
                words = line.split()
                if len(words) >= 2 and words[0] == "external_packages/iSS":
                    return words[1]
    except OSError:
        pass
    return ""


def iss_decay_attrs(core, eos, xml_root, xscape=None):
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
            "iss_commit": iss_commit(xscape) if xscape else ""}


def _set_text(parent, tag, value):
    import xml.etree.ElementTree as ET

    n = parent.find(tag)
    if n is None:
        n = ET.SubElement(parent, tag)
    n.text = str(value)


def collision_ecm(job_xml):
    """The collision energy sqrt(s_NN) [GeV] of a production job's XML (its text, as the
    files' ``prod_user_xml``, or a parsed root): <Hard><PythiaGun><eCM>, else <IS><MCGlauber>
    <sqrts>; None without either (or without the XML)."""
    import xml.etree.ElementTree as ET

    if job_xml is None or isinstance(job_xml, (str, bytes)):
        if not job_xml or not job_xml.strip():
            return None
        root = ET.fromstring(job_xml)
    else:
        root = job_xml
    for path in ("Hard/PythiaGun/eCM", "IS/MCGlauber/sqrts"):
        node = root.find(path)
        if node is not None and node.text and node.text.strip():
            return float(node.text)
    return None


def set_hadronization_ecm(root, ecm):
    """Colorless's <JetHadronization><eCMforHadronization> in a parsed hadronize.xml
    (``root``) set to ``ecm``, the collision's energy, in place: a beam remnant that closes a
    string carries 1/6 of it, so one hadronize.xml serves every collision system.  -> the
    value the file had (None if it had none).  ValueError without <JetHadronization>."""
    had = root.find("JetHadronization")
    if had is None:
        raise ValueError("the hadronize XML has no <JetHadronization>")
    node = had.find("eCMforHadronization")
    was = None if node is None or node.text is None else node.text.strip()
    _set_text(had, "eCMforHadronization", f"{float(ecm):g}")
    return was


def configure_hadronize_xml(root, *, xscape, seed, n_events, oversample, keep_primaries,
                            correlated=False, correlated_block=None):
    """A parsed hadronize.xml (``root``) with a run's settings, in place: the event count,
    the seed, iSS's paths (absolute, into ``xscape``'s external_packages/iSS, or into
    ``xscape`` itself for an installed tree, which keeps iSS_parameters.dat and iSS_tables in
    its data folder; its working directory ".", where the run's music_input is), the
    oversample count, <keep_primaries> and the correlated sampling (``correlated_block``
    "DTAU,DX,DETA"; ValueError if malformed)."""
    _set_text(root, "nEvents", max(1, int(n_events)))
    root.find("Random/seed").text = str(seed)
    iss = root.find("SoftParticlization/iSS")
    iss_dir = os.path.join(xscape, "external_packages", "iSS")
    if not os.path.isdir(iss_dir) and os.path.isfile(os.path.join(xscape, "iSS_parameters.dat")):
        iss_dir = xscape
    _set_text(iss, "iSS_input_file", os.path.join(iss_dir, "iSS_parameters.dat"))
    _set_text(iss, "iSS_table_path", os.path.join(iss_dir, "iSS_tables"))
    _set_text(iss, "iSS_particle_table_path", os.path.join(iss_dir, "iSS_tables"))
    _set_text(iss, "iSS_working_path", ".")
    _set_text(iss, "number_of_repeated_sampling", oversample)
    _set_text(iss, "keep_primaries", max(0, int(keep_primaries)))
    if correlated:
        _set_text(iss, "correlated_sampling", 1)
        if correlated_block:
            try:
                dtau, dx, deta = (float(v) for v in str(correlated_block).split(","))
            except ValueError:
                raise ValueError(f"--correlated-block wants DTAU,DX,DETA, got "
                                 f"{correlated_block!r}") from None
            _set_text(iss, "correlated_block_dtau", dtau)
            _set_text(iss, "correlated_block_dx", dx)
            _set_text(iss, "correlated_block_deta", deta)
    return root


def tag_attrs(tag, core, decay_attrs, *, common_seeds=False, correlated=False):
    """The tag-specific attributes of a hadron file (the generator, decayed, and for the
    bulk tags the hadron list, its decays and how the two legs were sampled relative to
    each other; for jet_frag the Pythia version)."""
    if tag in ("jet_frag", "extra_frag"):
        return {"generator": "ColorlessHadronization (X-SCAPE)", "decayed": True,
                "pythia_version": int(getattr(core, "PYTHIA_VERSION", 0))}
    out = {"common_seeds": bool(common_seeds), "correlated_sampling": bool(correlated)}
    out.update({k: decay_attrs[k] for k in ("hadron_list", "iss_decays", "iss_commit")},
               generator="iSS (X-SCAPE iSpectraSamplerWrapper)",
               decayed=bool(decay_attrs["decayed"]))
    return out


def _empty_hadrons():
    return {k: np.zeros((0, 4) if k in ("p", "x") else 0) for k in ("pid", "pstat", "p", "x")}


class Hadronizer:
    """iSS on cells, Colorless on partons, into hadron files (see the module docstring).

    Parameters
    ----------
    core : module
        ``jetscape.pyjetscape_core``.
    iss, colorless : module objects or None
        Initialised iSS and ColorlessHadronization modules (``core.create_module(...)``,
        ``.Init()``), for the bulk and the jet tags.
    base_seed : int
        The base of every unit's seed (:func:`unit_seed`).
    prod : int or None
        The production file's seed key (:func:`uuid_key` of its ``file_uuid``); None: the
        legacy seeds.
    common_seeds : bool
        ``bulk_jet`` of an event gets its background's ``bulk_bg`` seed
        (:meth:`bulk_jet_seed`; hadronize.py --common-seeds / --correlated).
    """

    def __init__(self, core, *, iss=None, colorless=None, base_seed=1, prod=None,
                 common_seeds=False):
        if iss is not None and not getattr(core, "SOFT_HAS_SAMPLE_CELLS", False):
            raise RuntimeError("Hadronizer: this PyJetscape has no soft_sample_cells "
                               "(X-SCAPE auau_hadrons_only or later)")
        self.core, self.iss, self.colorless = core, iss, colorless
        self.base_seed, self.prod = int(base_seed), prod
        self.common_seeds = bool(common_seeds)
        self.writers = {}
        self._primary_of = {}

    # ── seeds ───────────────────────────────────────────────────────────────────
    def seed(self, tag, unit, sample=0):
        return unit_seed(self.base_seed, tag, unit, sample, prod=self.prod)

    def bulk_jet_seed(self, event, bg_unit):
        """``bulk_jet``'s seed: its own, or with ``common_seeds`` its background's."""
        if self.common_seeds:
            return self.seed("bulk_bg", bg_unit)
        return self.seed("bulk_jet", event)

    # ── files ───────────────────────────────────────────────────────────────────
    def open(self, tag, path, *, n_samples, attrs, primary_tag=None, primary_path=None,
             keep_primaries=0, **writer_kw):
        """A writer for ``tag`` (and, with ``primary_path``, for its undecayed hadrons,
        ``primary_tag``: full precision, positions, no selection, the first
        ``keep_primaries`` samples; :data:`KEEP_ALL_PRIMARIES` or more: all).  ``writer_kw``
        go to HadronH5Writer (keep_bits, eta_max, charged_only, positions, initiators,
        hadronization_input, partons)."""
        from .hadrons_h5 import HadronH5Writer

        self.writers[tag] = HadronH5Writer(path, tag=tag, n_samples=n_samples, attrs=attrs,
                                           **writer_kw)
        if primary_path:
            if n_samples == 0:                    # a count per unit (units/n_samples)
                n_keep = 0 if keep_primaries >= KEEP_ALL_PRIMARIES else keep_primaries
            else:
                n_keep = min(keep_primaries, n_samples)
            pattrs = dict(attrs, decayed=False, primaries_of=tag,
                          keep_primaries=("all" if keep_primaries >= KEEP_ALL_PRIMARIES
                                          else int(keep_primaries)))
            self.writers[primary_tag] = HadronH5Writer(primary_path, tag=primary_tag,
                                                       n_samples=n_keep, attrs=pattrs)
            self._primary_of[tag] = primary_tag
        return self.writers[tag]

    def set_attrs(self, attrs):
        """Add root attributes to every open file (e.g. what is known at the end of a job)."""
        for w in self.writers.values():
            if w.f is not None:
                for k, v in attrs.items():
                    w.f.attrs[k] = v

    def close(self, complete=True):
        """Close every writer (``complete`` False leaves readable, incomplete files)."""
        for w in self.writers.values():
            w.close(complete=complete)

    # ── work ────────────────────────────────────────────────────────────────────
    def sample(self, cells, seed, n_samples, *, keep_primaries=0):
        """iSS on ``cells`` -> (hadrons, seed used, primaries or None); (None, seed, None)
        for no cells."""
        if len(cells) == 0:
            return None, seed, None
        core = self.core
        h = core.soft_sample_cells(self.iss, np.ascontiguousarray(cells, np.float32),
                                   int(seed), int(n_samples))
        prim = core.soft_primaries_numpy(self.iss) if keep_primaries else None
        used = int(core.soft_last_random_seed(self.iss))
        if used != int(seed):
            raise RuntimeError(f"iSS used seed {used}, not {seed}")
        if len(h["sample_counts"]) != int(n_samples):
            raise RuntimeError(f"iSS made {len(h['sample_counts'])} samples, "
                               f"{n_samples} were asked for")
        if keep_primaries:
            want = min(int(keep_primaries), int(n_samples))
            got = 0 if prim is None else len(prim["sample_counts"])
            if got != want:
                raise RuntimeError(f"iSS kept the primaries of {got} samples, {want} were "
                                   "asked for (<keep_primaries> in the iSS XML block?)")
        return h, used, prim

    def bulk(self, tag, unit, cells, seed, n_samples, *, event, initiators=None,
             keep_primaries=0, **scalars):
        """iSS on one surface into ``tag``'s file (and its primary file) -> (hadrons or
        None, seed used)."""
        ptag = self._primary_of.get(tag)
        h, used, prim = self.sample(cells, seed, n_samples,
                                    keep_primaries=keep_primaries if ptag else 0)
        self.writers[tag].append_unit(h if h is not None else [], initiators, unit=unit,
                                      event=event, seed=used, n_cells=len(cells), **scalars)
        if ptag:
            self.writers[ptag].append_unit(prim if prim is not None else [], unit=unit,
                                           event=event, seed=used, n_cells=len(cells),
                                           **scalars)
        return h, used

    def fragments(self, unit, partons, seeds, *, event=None, initiators=None, tag="jet_frag",
                  **scalars):
        """Colorless on ``partons`` once per seed into ``tag`` (default ``jet_frag``; and the
        partons into its ``partons/``, if the writer has one) -> the fragmentations."""
        samples = []
        for seed in seeds:
            if len(partons):
                samples.append(self.core.hadronize_partons(self.colorless, partons, int(seed)))
            else:
                samples.append(_empty_hadrons())
        w = self.writers[tag]
        kw = {"partons": partons} if getattr(w, "_par", None) is not None else {}
        w.append_unit(samples, initiators, **kw, unit=unit,
                      event=unit if event is None else event, seed=int(seeds[0]),
                      n_partons=len(partons), **scalars)
        return samples

    def extra_seeds(self, unit, n_frag, scatters=None):
        """The extras' Colorless seeds of one event: per fragmentation one (``scatters``
        None, mode groups), or per fragmentation a list, one per scattering in
        ``scatters`` (per_scattering)."""
        if scatters is None:
            return [self.seed("extra_frag", unit, s) for s in range(n_frag)]
        return [[self.seed("extra_frag", unit, s * 100000 + int(k)) for k in scatters]
                for s in range(n_frag)]

    def extra_fragments(self, unit, partons, n_frag, *, initiators, initiator_scatter,
                        scatterings=None, mode="groups", event=None, **scalars):
        """Colorless on the extras' final partons into ``extra_frag``, ``n_frag`` times:
        all together (``mode`` groups) or one call per scattering (per_scattering; every
        hadron then carries its ``scatter``).  ``partons``/``initiators`` are the extras'
        only (:func:`jetscape.particlize_h5.split_by_scatter`), ``initiator_scatter`` their
        scatterings, ``scatterings`` the event's SCATTER_COLUMNS rows -> the fragmentations."""
        if mode not in EXTRA_HADRONIZE:
            raise ValueError(f"extra_fragments: mode is one of {EXTRA_HADRONIZE}, got {mode!r}")
        partons = np.asarray(partons, dtype=np.float64).reshape(-1, 14)
        ini = np.asarray(initiators, dtype=np.float64).reshape(-1, 11)
        iscat = np.asarray(initiator_scatter, dtype=np.int64).reshape(-1)
        samples = []
        if mode == "groups":
            seeds = self.extra_seeds(unit, n_frag)
            for seed in seeds:
                samples.append(self.core.hadronize_partons(self.colorless, partons, int(seed))
                               if len(partons) else _empty_hadrons())
            first = seeds[0] if seeds else 0
        else:
            scatters = sorted(set(int(k) for k in iscat))
            shower_of = {k: ini[iscat == k, 0].astype(np.int64) for k in scatters}
            seeds = self.extra_seeds(unit, n_frag, scatters)
            for per in seeds:
                parts = []
                for k, seed in zip(scatters, per):
                    p = partons[np.isin(partons[:, 0].astype(np.int64), shower_of[k])]
                    h = (self.core.hadronize_partons(self.colorless, p, int(seed)) if len(p)
                         else _empty_hadrons())
                    h = {f: np.asarray(h[f]) for f in ("pid", "pstat", "p", "x")}
                    h["scatter"] = np.full(len(h["pid"]), k, np.int32)
                    parts.append(h)
                if not parts:
                    e = _empty_hadrons()
                    parts = [dict(e, scatter=np.zeros(0, np.int32))]
                samples.append({f: np.concatenate([np.asarray(q[f]).reshape(
                    (-1, 4) if f in ("p", "x") else (-1,)) for q in parts])
                    for f in ("pid", "pstat", "p", "x", "scatter")})
            first = seeds[0][0] if seeds and seeds[0] else 0
        w = self.writers["extra_frag"]
        kw = {"partons": partons} if getattr(w, "_par", None) is not None else {}
        w.append_unit(samples, ini if w._ini is not None else None, **kw,
                      scatterings=scatterings, initiator_scatter=iscat, unit=unit,
                      event=unit if event is None else event, seed=int(first),
                      n_partons=len(partons), **scalars)
        return samples
