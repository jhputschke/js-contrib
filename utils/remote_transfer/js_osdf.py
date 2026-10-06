#!/usr/bin/env python3
"""
utils/remote_transfer/js_osdf.py

Upload and download production directories and files to / from a Pelican namespace on the
OSDF (default osdf:///fno4hic), with pelicanfs.  The same tool as js_gcs.py, for Pelican:
on its first run the script makes its own virtual environment (``~/.cache/js_osdf/venv``,
plain ``python3 -m venv`` + pip) and from then on runs itself in it.  It needs
transfer_core.py next to it.

    ./js_osdf.py upload /data/AuAu_c1 --what root         # -> osdf:///fno4hic/AuAu_c1/
    ./js_osdf.py upload /data/AuAu_a /data/AuAu_b --what pair,h5 --prefix productions -j 8
    ./js_osdf.py upload /data/AuAu_c1/AuAu_c1_0003_hadrons.root   # one file
    ./js_osdf.py upload /data/out --what root --as AuAu_v1  # -> osdf:///fno4hic/AuAu_v1/
    ./js_osdf.py download AuAu_c1 --what root --to /scratch   # -> /scratch/AuAu_c1/
    ./js_osdf.py download 'AuAu_c1/*_0004_*' --to here        # files -> here/<file>
    ./js_osdf.py ls                                    # the namespace's top level
    ./js_osdf.py ls AuAu_c1 --what h5                  # files, sizes, kinds
    ./js_osdf.py rm -r AuAu_c1 --dry-run               # what would be removed; then without
    ./js_osdf.py rm AuAu_c1/AuAu_c1_0003_hadrons.root 'AuAu_c1/*_0004_*'   # files, patterns
    ./js_osdf.py setup --reinstall | --remove          # remake / delete the environment
    ./js_osdf.py login                                 # log in in a browser (no token file)
    ./js_osdf.py upload /data/AuAu_c1 --web            # the same, when needed, then upload
    ./js_osdf.py status                                # who, scopes, time left
    ./js_osdf.py logout                                # forget the browser login

What (``--what``, a comma list; default all) -- by file name, as for js_gcs.py:

    pair    <stem>.h5, the hydro pair file, with its <stem>.json, .xml, .log
    h5      <stem>_particlize.h5, <stem>_hadrons_{bulk_jet,bulk_bg,jet_frag}.h5,
            <stem>_hadronize.log
    root    <stem>_hadrons.root, <campaign>_campaign.root
    all     the three, plus every other file of the directory

Where files go (--prefix, --as, --flat), single files, patterns, the skip of files already
there, incomplete HDF5 and ``*.part``: as for js_gcs.py.  Pelican gives sizes but no
checksums, so every upload also records its CRC32C in a manifest in the remote directory
(``.js_transfer_crc32c.json``) and checks the size the origin reports afterwards.  The
skip check and downloads use the manifest's CRC32C; files it doesn't know (put there by
other tools) are compared by size.  Listings, sizes and downloads come from the origin
(pelicanfs direct_reads), not from a cache that may hold an older copy.

Credentials (writing needs them; a public namespace reads without):

  - a browser login: ``login`` (or ``--web`` on any command) runs the OAuth2 device flow
    of the namespace's own token issuer, which the director names: it prints a link, you
    log in in any browser, on any machine.  The token and its refresh token are kept in
    ~/.config/js_osdf/web/ and renewed by themselves, also during a long upload, until
    the refresh token expires; then ``login`` again.  ``logout`` forgets them.
  - a bearer token: --token-file, else $JS_OSDF_TOKEN_FILE, else $BEARER_TOKEN_FILE, else
    $BEARER_TOKEN, else osdf.token in the current directory, next to this script, or in
    ~/.config/js_osdf/.

--web uses the browser login even if there is a token; without --web a token comes first,
then a browser login made before.  With neither, pelicanfs looks in the WLCG default place
and can get a token through the pelican CLI (OAuth).
The namespace: --namespace, else $JS_OSDF_NAMESPACE, else /fno4hic; the federation:
--federation, else osg-htc.org (the OSDF).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transfer_core as core  # noqa: E402

ENV = core.Env("js_osdf", ("pelicanfs>=1.4", "google-crc32c>=1.5", "h5py>=3.0"),
               "JS_OSDF_ENV")
DEFAULT_NAMESPACE = "/fno4hic"
OSDF = "osg-htc.org"
TOKEN_NAME = "osdf.token"


def find_token(arg):
    """The bearer token (its text), or None: then pelicanfs looks for one itself."""
    here = Path(__file__).resolve().parent
    for c in (arg, os.environ.get("JS_OSDF_TOKEN_FILE"), os.environ.get("BEARER_TOKEN_FILE")):
        if c:                                       # given explicitly: must exist
            p = Path(c).expanduser()
            if not p.is_file():
                sys.exit(f"js_osdf: token file {c} not found")
            return p.read_text().strip()
    if os.environ.get("BEARER_TOKEN"):
        return os.environ["BEARER_TOKEN"].strip()
    for d in (Path.cwd(), here, Path("~/.config/js_osdf").expanduser()):
        if (d / TOKEN_NAME).is_file():
            return (d / TOKEN_NAME).read_text().strip()
    return None


def namespace_path(text):
    ns = "/" + text.strip().strip("/")
    if ns == "/":
        raise ValueError("the namespace can't be /")
    return ns


# ── the browser login (OAuth2 device flow) ────────────────────────────────────────
WEB_DIR = "~/.config/js_osdf/web"         # the logins, one file per federation + namespace
MARGIN = 120                              # s: a token with less left is renewed
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
# the device flow is still waiting for the browser: Pelican's issuer says consent_required
# from the login until the approval (its own client polls on, as here)
WAITING = ("authorization_pending", "consent_required")


def _http(url, form=None, json_body=None, redirect=True, timeout=30, method=None,
          headers=None):
    """(status, headers, body) of a GET, or of a POST of a form or of JSON (or method)."""
    # a User-Agent of our own: osg-htc.org refuses urllib's
    data, headers = None, {"Accept": "application/json", "User-Agent": "js_osdf",
                           **(headers or {})}
    if form is not None:
        data = urllib.parse.urlencode(form).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif json_body is not None:
        data = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kw):
            return None

    opener = (urllib.request.build_opener() if redirect
              else urllib.request.build_opener(NoRedirect))
    try:
        req = urllib.request.Request(url, data, headers, method=method)
        with opener.open(req, timeout=timeout) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:                # 3xx without redirect, 4xx, 5xx
        return e.code, e.headers, e.read()


def _json(body):
    try:
        d = json.loads(body)
    except (ValueError, TypeError):
        return {}
    return d if isinstance(d, dict) else {}


def _claims(jwt):
    """The claims of a JWT, unverified ({} if it isn't one): for its exp, sub and scope."""
    try:
        part = jwt.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    except (IndexError, ValueError, AttributeError):
        return {}


def _refresh_expiry(token):
    """When an OA4MP refresh token expires (unix time), or None: such a token is a base32
    URL whose query has ts (issued, ms) and lifetime (ms)."""
    try:
        url = base64.b32decode(token + "=" * (-len(token) % 8)).decode()
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        return (int(q["ts"][0]) + int(q["lifetime"][0])) / 1000
    except (ValueError, KeyError, IndexError, UnicodeDecodeError):
        return None


def _left(t):
    """'5 min', '3.2 h', '14.9 days' until unix time t ('expired' if past)."""
    d = t - time.time()
    if d <= 0:
        return "expired"
    return (f"{d / 60:.0f} min" if d < 5400 else f"{d / 3600:.1f} h" if d < 172800
            else f"{d / 86400:.1f} days")


def token_issuer(federation, namespace):
    """(issuer URL, the namespace's path for the scopes) from the director's
    X-Pelican-Token-Generation header for the namespace."""
    st, _, body = _http(f"https://{federation}/.well-known/pelican-configuration")
    director = _json(body).get("director_endpoint")
    if st != 200 or not director:
        raise RuntimeError(f"{federation}: no director in its pelican-configuration "
                           f"(HTTP {st})")
    st, h, _ = _http(f"{director.rstrip('/')}/api/v1.0/director/origin{namespace}/",
                     redirect=False)
    gen = h.get("X-Pelican-Token-Generation") if h else None
    if not gen:
        raise RuntimeError(f"the director names no token issuer for {namespace} (HTTP {st})")
    info = {}
    for kv in gen.split(","):
        k, _, v = kv.strip().partition("=")
        info.setdefault(k, v)                          # the first issuer, if several
    if info.get("strategy", "OAuth2") != "OAuth2" or not info.get("issuer"):
        raise RuntimeError(f"{namespace}: no OAuth2 token issuer ({gen})")
    base = "/" + info.get("base-path", "").strip("/")
    if namespace == base or namespace.startswith(base.rstrip("/") + "/"):
        path = namespace[len(base.rstrip("/")):] or "/"
    else:
        path = namespace
    return info["issuer"].rstrip("/"), path


class WebLogin:
    """A token from the namespace's issuer through the OAuth2 device flow, kept with its
    refresh token in WEB_DIR and renewed when it has less than MARGIN left.  Thread-safe:
    the transfers ask for a token before each request."""

    def __init__(self, namespace, federation=OSDF):
        self.ns, self.federation = namespace, federation
        self.file = (Path(WEB_DIR).expanduser()
                     / f"{federation}{namespace.replace('/', '_')}.json")
        self._lock = threading.Lock()
        self.last_error = None                       # why the last renewal failed
        try:
            self.state = json.loads(self.file.read_text())
        except (OSError, ValueError):
            self.state = {}

    def exists(self):
        return bool(self.state.get("refresh_token") or self.state.get("access_token"))

    def _save(self):
        self.file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = self.file.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(self.state, f, indent=1)
        os.replace(tmp, self.file)

    # -- the issuer and this script's client there
    def _issuer(self):
        s = self.state
        if "token_endpoint" not in s:
            s["issuer"], s["scope_path"] = token_issuer(self.federation, self.ns)
            st, _, body = _http(f"{s['issuer']}/.well-known/openid-configuration")
            cfg = _json(body)
            if st != 200 or "device_authorization_endpoint" not in cfg:
                raise RuntimeError(f"{s['issuer']}: no device flow (HTTP {st})")
            for k in ("token_endpoint", "device_authorization_endpoint",
                      "registration_endpoint", "revocation_endpoint"):
                s[k] = cfg.get(k)
        return s

    def _scopes(self):
        p = self.state["scope_path"]
        return (f"offline_access wlcg storage.read:{p} storage.create:{p} "
                f"storage.modify:{p}")

    def _client(self):
        s = self._issuer()
        exp = int(s.get("client_secret_expires_at") or 0)
        if not s.get("client_id") or (exp and exp < time.time() + MARGIN):
            if not s.get("registration_endpoint"):
                raise RuntimeError(f"{s['issuer']}: no client registration")
            st, _, body = _http(s["registration_endpoint"], json_body={
                "client_name": "js_osdf", "scope": self._scopes(),
                "grant_types": ["refresh_token", DEVICE_GRANT]})
            reg = _json(body)
            if st not in (200, 201) or "client_id" not in reg:
                raise RuntimeError(f"{s['issuer']}: client registration failed (HTTP {st}: "
                                   f"{body[:200]!r})")
            s.update(client_id=reg["client_id"], client_secret=reg.get("client_secret", ""),
                     client_secret_expires_at=reg.get("client_secret_expires_at", 0))
        return {"client_id": s["client_id"], "client_secret": s["client_secret"]}

    def _take(self, tok, renewal=False):
        """Keep a token response.  On a renewal the refresh token we have is kept and the
        issuer's new one only noted (refresh_token_rotated): the /fno4hic issuer (OA4MP at
        the Wayne origin, 2026-10-05) answers a renewal with a refresh token that was itself
        handed out by a renewal with HTTP 500 "Null pointer", while the login's refresh token
        renews any number of times."""
        s, claims = self.state, _claims(tok["access_token"])
        s["access_token"] = tok["access_token"]
        if tok.get("refresh_token"):
            if renewal and s.get("refresh_token"):
                s["refresh_token_rotated"] = tok["refresh_token"]
            else:
                s["refresh_token"] = tok["refresh_token"]
                s.pop("refresh_token_rotated", None)
        s["expires_at"] = int(claims.get("exp") or time.time() + int(tok.get("expires_in", 600)))
        s["scope"] = claims.get("scope") or tok.get("scope", "")
        s["sub"] = claims.get("sub", s.get("sub", ""))
        self._save()

    # -- the flows
    def login(self, out=None):
        """The device flow: prints the link (to stderr), waits for the browser login."""
        out = out or sys.stderr
        client = self._client()
        s = self.state
        st, _, body = _http(s["device_authorization_endpoint"],
                            form={**client, "scope": self._scopes()})
        dev = _json(body)
        if st != 200 or "device_code" not in dev:
            raise RuntimeError(f"{s['issuer']}: device authorization failed (HTTP {st}: "
                               f"{body[:200]!r})")
        link = dev.get("verification_uri_complete") or dev["verification_uri"]
        print(f"js_osdf: log in for {self.ns} in a browser (any machine):\n\n    {link}\n\n"
              f"  or open {dev['verification_uri']} and enter the code {dev['user_code']}"
              f"\n  waiting (up to {int(dev.get('expires_in', 900)) // 60} min; "
              f"Ctrl-C aborts) ...", file=out, flush=True)
        interval = int(dev.get("interval", 5))
        end = time.time() + int(dev.get("expires_in", 900))
        try:
            while time.time() < end:
                time.sleep(interval)
                st, _, body = _http(s["token_endpoint"], form={
                    **client, "grant_type": DEVICE_GRANT, "device_code": dev["device_code"],
                    "scope": self._scopes()})
                tok = _json(body)
                if st == 200 and "access_token" in tok:
                    self._take(tok)
                    return s["access_token"]
                err = tok.get("error", f"HTTP {st}")
                if err == "slow_down":
                    interval += 5
                elif err not in WAITING:
                    raise RuntimeError(f"login failed: {err} {tok.get('error_description', '')}")
        except KeyboardInterrupt:
            raise RuntimeError("login aborted") from None
        raise RuntimeError("login timed out: the link was not used in time")

    def refresh(self):
        """A new token from the refresh token (the kept one, else the issuer's latest, see
        _take); False if there is none or it is refused (the reason in self.last_error)."""
        s = self.state
        self.last_error = None
        if not s.get("refresh_token") or not s.get("token_endpoint"):
            self.last_error = "no refresh token"
            return False
        errors = []
        for key in ("refresh_token", "refresh_token_rotated"):
            rt = s.get(key)
            if not rt:
                continue
            st, _, body = _http(s["token_endpoint"], form={
                **self._client(), "grant_type": "refresh_token", "refresh_token": rt})
            tok = _json(body)
            if st == 200 and "access_token" in tok:
                if key == "refresh_token_rotated":     # the kept one is refused, this one not
                    s["refresh_token"] = rt
                self._take(tok, renewal=True)
                return True
            err = tok.get("error") or " ".join(
                (body[:160].decode(errors="replace") if isinstance(body, bytes)
                 else str(body)[:160]).split())
            errors.append(f"{key}: HTTP {st} {err} {tok.get('error_description', '')}".strip())
        self.last_error = "; ".join(errors)
        return False

    def reload(self):
        """Read the login file again (another process may have logged in or renewed);
        True if it changed."""
        try:
            state = json.loads(self.file.read_text())
        except (OSError, ValueError):
            return False
        if state == self.state:
            return False
        self.state = state
        return True

    def token(self, interactive=False):
        """A token with at least MARGIN left: kept, renewed, or (interactive) a new login;
        else None."""
        with self._lock:
            s = self.state
            if s.get("access_token") and s.get("expires_at", 0) > time.time() + MARGIN:
                return s["access_token"]
            if self.refresh():
                return s["access_token"]
            if self.reload():                          # e.g. a login in another terminal
                s = self.state
                if s.get("access_token") and s.get("expires_at", 0) > time.time() + MARGIN:
                    return s["access_token"]
                if self.refresh():
                    return s["access_token"]
            if self.last_error:
                print(f"js_osdf: renewing the token for {self.ns} failed: {self.last_error}",
                      file=sys.stderr, flush=True)
            if interactive:
                return self.login()
            return None

    def logout(self):
        """Revoke the refresh token (if the issuer can) and forget the login."""
        s = self.state
        if s.get("refresh_token") and s.get("revocation_endpoint") and s.get("client_id"):
            try:
                _http(s["revocation_endpoint"], form={
                    "client_id": s["client_id"], "client_secret": s.get("client_secret", ""),
                    "token": s["refresh_token"], "token_type_hint": "refresh_token"})
            except OSError:
                pass
        existed = self.file.exists()
        self.file.unlink(missing_ok=True)
        self.state = {}
        return existed

    def describe(self):
        s = self.state
        rexp = _refresh_expiry(s.get("refresh_token", ""))
        return (f"{s.get('sub') or '?'} at {s.get('issuer')}; scopes {s.get('scope') or '?'}; "
                f"token valid for {_left(s.get('expires_at', 0))}, renewed by itself"
                + ("" if s.get("refresh_token") else " -- NO refresh token: login again then")
                + (f" until {time.strftime('%Y-%m-%d %H:%M', time.localtime(rexp))}"
                   if rexp else "")
                + f" ({self.file})")

    def status(self):
        """Lines about the login, from the login file only (no network)."""
        s, when = self.state, lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t))
        exp = s.get("expires_at", 0)
        rexp = _refresh_expiry(s.get("refresh_token", ""))
        if not s.get("refresh_token"):
            refresh = "none: when the token expires, login again"
        elif rexp:
            refresh = (f"valid until {when(rexp)} ({_left(rexp)}); renewals don't extend "
                       "it: login again before then"
                       if rexp > time.time() else
                       f"expired {when(rexp)}: ./js_osdf.py login")
        else:
            refresh = "kept (its expiry is the issuer's)"
        return [
            ("user", s.get("sub") or "?"),
            ("issuer", s.get("issuer", "?")),
            ("scopes", s.get("scope") or "?"),
            ("token", (f"valid until {when(exp)} ({_left(exp)} left)" if exp > time.time()
                       else (f"expired {when(exp)}" if exp else "expired")
                       + (": renewed by the next command" if s.get("refresh_token") else ""))),
            ("refresh", refresh),
            ("file", str(self.file)),
        ]


class OsdfStore:
    """A namespace of a Pelican federation (transfer_core's store interface)."""

    native_checksums = False
    manifest = True

    def __init__(self, namespace, federation=OSDF, token=None, fs=None):
        """token: the bearer token, or a function giving the current one (WebLogin.token),
        asked before every request."""
        self.ns, self.federation = namespace, federation
        self.label = (f"osdf://{namespace}" if federation == OSDF
                      else f"pelican://{federation}{namespace}")
        self._token_fn = token if callable(token) else None
        self._headers, self._bearer = {}, None    # pelicanfs's HTTP requests use _headers
        self._origin = None                       # the namespace's URL at its origin (rm)
        self.fs = None
        self._set_token(token() if callable(token) else token)
        self.fs = fs if fs is not None else self._make_fs(federation, self._headers)

    def _set_token(self, token):
        if token and token != self._bearer:
            self._bearer = token
            self._headers["Authorization"] = f"Bearer {token}"
            if self.fs is not None:
                self.fs.token = self._headers["Authorization"]   # its WebDAV requests

    def _auth(self):
        if self._token_fn:
            self._set_token(self._token_fn())

    @staticmethod
    def _make_fs(federation, headers):
        # pelicanfs 1.4 leaves an un-awaited coroutine and unclosed sessions behind: harmless
        warnings.filterwarnings("ignore", message="coroutine .* was never awaited")
        logging.getLogger("asyncio").setLevel(logging.CRITICAL)
        from pelicanfs.core import OSDFFileSystem, PelicanFileSystem
        # the origin's view, not a cache's; headers: the same dict, so a renewed token
        # reaches the HTTP requests
        # (skip_instance_cache: not fsspec's cached instance, with another headers dict)
        kw = {"direct_reads": True, "headers": headers, "skip_instance_cache": True}
        if federation == OSDF:
            return OSDFFileSystem(**kw)
        return PelicanFileSystem(f"pelican://{federation}", **kw)

    def path(self, name):
        return f"{self.ns}/{name}" if name else self.ns

    def name_of(self, path):
        p = "/" + path.lstrip("/")
        return p[len(self.ns):].strip("/") if p.startswith(self.ns) else p.strip("/")

    def relative(self, text):
        t = text.strip()
        for scheme in ("osdf://", f"pelican://{self.federation}", "pelican://"):
            if t.startswith(scheme):
                t = t[len(scheme):]
                break
        if t.startswith("/"):                          # an absolute namespace path
            if t.rstrip("/") != self.ns and not t.startswith(self.ns + "/"):
                core.die(f"{text} is not in {self.label} (use --namespace)")
            t = t[len(self.ns):]
        return t.strip("/")

    def list(self, prefix):
        self._auth()
        try:
            found = self.fs.find(self.path(prefix.strip("/")), detail=True)
        except FileNotFoundError:
            return {}
        # (an empty directory comes without a size, and as a "file")
        return {self.name_of(p): (int(info["size"]), None) for p, info in found.items()
                if info.get("type", "file") == "file" and info.get("size") is not None}

    def stat(self, name):
        self._auth()
        p = self.path(name)
        self.fs.invalidate_cache(p)
        try:
            info = self.fs.info(p)
        except FileNotFoundError:
            return None
        # pelicanfs 1.4's info() says "file" for directories too; isdir() knows
        if info.get("type", "file") != "file" or self.fs.isdir(p):
            return None
        return int(info["size"]), None

    def top_level(self, prefix):
        self._auth()
        try:
            entries = self.fs.ls(self.path(prefix.strip("/")), detail=True)
        except FileNotFoundError:
            return [], {}
        dirs = [self.name_of(e["name"]) + "/" for e in entries if e.get("type") == "directory"]
        files = {self.name_of(e["name"]): (int(e["size"]), None) for e in entries
                 if e.get("type") != "directory"}
        return sorted(dirs), files

    def upload(self, path, name):
        self._auth()
        self.fs.put_file(str(path), self.path(name))
        self.fs.invalidate_cache(self.path(name))

    def download(self, name, path):
        self._auth()
        self.fs.get_file(self.path(name), str(path))

    def read_bytes(self, name):
        self._auth()
        try:
            return self.fs.cat_file(self.path(name))
        except FileNotFoundError:
            return None

    def delete(self, name):
        self._auth()
        self._rm(name)
        self.fs.invalidate_cache(self.path(name))

    remove_dir = delete                           # the origin removes empty ones alike

    def _rm(self, name):
        """HTTP DELETE at the origin (pelicanfs has no delete): a file, an empty directory."""
        if self._origin is None:
            from fsspec.asyn import sync
            self._origin = sync(self.fs.loop, self.fs.get_origin_url, self.ns)[0].rstrip("/")
        st, _, body = _http(f"{self._origin}/{urllib.parse.quote(name)}", method="DELETE",
                            headers=self._headers, timeout=120)
        if st in (200, 202, 204):
            return
        if st == 404:
            raise FileNotFoundError(f"{self.label}/{name}")
        if st == 409:
            raise OSError(f"{self.label}/{name}/ is not empty")
        if st in (401, 403):
            raise PermissionError(f"HTTP {st}: no right to remove {self.label}/{name} "
                                  "(./js_osdf.py login, or a token that may modify)")
        raise OSError(f"HTTP {st} removing {self.label}/{name}: {body[:200]!r}")


def add_args(p):
    p.add_argument("--token-file", dest="token_file", default=None,
                   help=f"bearer token file (default: see above; {TOKEN_NAME})")
    p.add_argument("--web", action="store_true",
                   help="use the browser login (OAuth2 device flow), logging in if needed, "
                        "instead of a token")
    p.add_argument("--namespace", type=namespace_path,
                   default=os.environ.get("JS_OSDF_NAMESPACE", DEFAULT_NAMESPACE),
                   help="namespace (default /fno4hic, or $JS_OSDF_NAMESPACE)")
    p.add_argument("--federation", default=OSDF,
                   help=f"Pelican federation (default {OSDF}, the OSDF)")


def add_commands(sub):
    sub.add_parser("login", help="log in in a browser (OAuth2 device flow); later commands "
                                 "use the login without a token")
    sub.add_parser("logout", help="forget the browser login (and revoke it at the issuer)")
    sub.add_parser("status", help="who is logged in, the scopes, the time left (no network)")


def parse_args(argv):
    return core.build_parser(__doc__, add_args, "osdf:///NAMESPACE",
                             add_commands).parse_args(argv)


def cmd_login(a):
    w = WebLogin(a.namespace, a.federation)
    try:
        w.login()
    except (OSError, RuntimeError) as e:
        core.die(f"{a.namespace}: {core.describe(e)}")
    core.say(f"logged in: {w.describe()}")
    return 0


def cmd_status(a):
    """The credentials a command would use, and the browser login's state (no network)."""
    token = find_token(a.token_file)
    if token:
        c = _claims(token)
        exp = c.get("exp")
        core.say(f"a bearer token comes first (unless --web): {c.get('sub', '?')}, scopes "
                 f"{c.get('scope', '?')}"
                 + (f", {_left(exp)} left" if exp else ""))
    w = WebLogin(a.namespace, a.federation)
    if not w.exists():
        core.say(f"no browser login for {a.namespace}: ./js_osdf.py login"
                 + ("" if token else "  (reading a public namespace needs none)"))
        return 0 if token else 1
    core.say(f"browser login for {a.namespace}:")
    for k, v in w.status():
        print(f"  {k:<8} {v}")
    rexp = _refresh_expiry(w.state.get("refresh_token", ""))
    usable = (w.state.get("expires_at", 0) > time.time()
              or (w.state.get("refresh_token") and (rexp is None or rexp > time.time())))
    return 0 if usable or token else 1


def cmd_logout(a):
    w = WebLogin(a.namespace, a.federation)
    core.say(f"logged out of {a.namespace} ({w.file} removed)" if w.logout()
             else f"no browser login for {a.namespace}")
    return 0


def credential(a):
    """The token for the store: a fixed one, WebLogin.token, or None (see above)."""
    if a.web and a.token_file:
        core.die("--web or --token-file, not both")
    token = None if a.web else find_token(a.token_file)
    if token:
        return token
    w = WebLogin(a.namespace, a.federation)
    if not a.web and not w.exists():
        return None
    try:
        if w.token(interactive=a.web) is None:
            core.say(f"the browser login for {a.namespace} has expired: ./js_osdf.py login "
                     "(going on without a token)")
            return None
    except (OSError, RuntimeError) as e:
        core.die(f"{a.namespace}: {core.describe(e)}")

    def current():
        t = w.token(interactive=a.web)
        if t is None:
            raise RuntimeError(f"the browser login for {a.namespace} has expired and could "
                               "not be renewed: ./js_osdf.py login, then run again")
        return t
    current.login = w
    return current


def main(argv=None):
    return core.main("js_osdf", core.build_parser(__doc__, add_args, "osdf:///NAMESPACE",
                                                  add_commands),
                     ENV, lambda a: OsdfStore(a.namespace, a.federation, credential(a)),
                     ("pelicanfs", "google_crc32c"), argv,
                     commands={"login": cmd_login, "logout": cmd_logout,
                               "status": cmd_status})


if __name__ == "__main__":
    core.enter_env(ENV, __file__, sys.argv[1:], "JS_OSDF_NO_ENV")
    sys.exit(main())
