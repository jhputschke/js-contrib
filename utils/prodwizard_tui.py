"""
utils/prodwizard_tui.py -- the wizard screen of prodwizard.py (run prodwizard.py without
arguments).  Needs textual (utils/analysis_env/requirements.txt).

The form on the left; on the right the plan, rebuilt on every change: the commands, the
estimates (measured or estimated, with their sources), the checks and notes.  Save writes
the launch script (prodwizard.py --save does the same); Copy puts the commands on the
clipboard (OSC 52: most terminals, also over ssh).  Nothing is started from here.

In a browser, e.g. for a workshop: textual serve "utils/prodwizard.py" (textual-dev).
"""

from __future__ import annotations

import os
from functools import lru_cache

from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Label, Select, Static

import prodwizard as pw

LEVEL_STYLE = {"ok": "green", "warn": "yellow", "error": "bold red"}
FIELDS = [   # (id, label, kind, Options attribute, shown for machine kinds)
    ("events", "events in total", "integer", "events", None),
    ("events_per_job", "events per job", "integer", "events_per_job", None),
    ("campaign", "campaign name", "text", "campaign", None),
    ("out", "output directory", "text", "out", None),
    ("gpus", "GPUs (array tasks)", "integer", "gpus", {"slurm"}),
    ("jobs_per_gpu", "jobs per GPU", "integer", "jobs_per_gpu", None),
    ("omp", "OpenMP threads per job", "integer", "omp", None),
    ("first_seed", "first seed (0: unique)", "integer", "first_seed", None),
    ("image", "image", "text", "image", {"docker2gpu", "slurm"}),
    ("upload", "upload to osdf:///fno4hic/", "text", "upload", {"docker2gpu"}),
    ("min_free", "stop below free disk", "text", "min_free", {"docker2gpu"}),
]
this_machine = lru_cache(maxsize=1)(pw.this_machine)


class WizardApp(App):
    TITLE = "prodwizard"
    SUB_TITLE = "plan a production; nothing is started from here"
    CSS = """
    #left { width: 48; border-right: solid $panel; }
    #form { height: 1fr; padding: 0 1; }
    #form Label { margin-top: 1; color: $text-muted; }
    #form Input, #form Select { width: 100%; }
    #buttons { height: auto; padding: 1 1 0 1; border-top: solid $panel; }
    #buttons Button { width: 1fr; margin-right: 1; }
    #plan-scroll { padding: 0 1; }
    .hidden { display: none; }
    """
    BINDINGS = [Binding("ctrl+s", "save", "save script"),
                Binding("ctrl+y", "copy", "copy commands"),
                Binding("ctrl+q", "quit", "quit")]

    def __init__(self, presets):
        super().__init__()
        self.P = presets
        self.plan = None
        self.opts = None

    def compose(self) -> ComposeResult:
        P = self.P
        here = this_machine()
        yield Header()
        with Horizontal():
            with Vertical(id="left"):
                with VerticalScroll(id="form"):
                    yield Label("production")
                    yield Select([(v["title"], k) for k, v in P["productions"].items()],
                                 value="auau_jet", allow_blank=False, id="production")
                    yield Label("machine" + (f" (this is the {here})" if here else ""))
                    yield Select([(v["title"], k) for k, v in P["machines"].items()],
                                 value=here or "2x3090", allow_blank=False, id="machine")
                    for fid, label, kind, _, _ in FIELDS:
                        yield Label(label, id=f"l_{fid}")
                        yield Input(id=fid, type="integer" if kind == "integer" else "text",
                                    value="2700" if fid == "events" else "")
                    yield Label("delete after a verified upload", id="l_delete")
                    yield Select([("none", "none"), ("pair (<stem>.h5)", "pair"),
                                  ("all (+ _particlize.h5)", "all")], value="none",
                                 allow_blank=False, id="delete")
                    yield Label("script file")
                    yield Input(id="script", placeholder="launch_<campaign>.sh")
                with Horizontal(id="buttons"):
                    yield Button("Save script", id="save", variant="primary")
                    yield Button("Copy", id="copy")
            with VerticalScroll(id="plan-scroll"):
                yield Static(id="plan")
        yield Footer()

    def on_mount(self):
        self.update_form()
        self.rebuild()

    # -- the form ------------------------------------------------------------------------------
    def update_form(self):
        """Show the fields of this machine kind; the preset's defaults as placeholders."""
        P = self.P
        prod = P["productions"][self.query_one("#production", Select).value]
        mkey = self.query_one("#machine", Select).value
        mach = P["machines"][mkey]
        st = mach.get("settings", {}).get(self.query_one("#production", Select).value, {})
        kind = mach["kind"]
        for fid, _, _, _, kinds in FIELDS + [("delete", "", "", "", {"docker2gpu"})]:
            show = kinds is None or kind in kinds
            for w in (self.query_one(f"#{fid}"), self.query_one(f"#l_{fid}")):
                w.set_class(not show, "hidden")
        ph = {"events_per_job": f"{prod['events_per_job']} (a multiple of "
                                f"{prod.get('events_multiple', 1)})",
              "campaign": f"{self.query_one('#production', Select).value}_<date>",
              "out": {"docker2gpu": "/data/<campaign>", "native": "out/<campaign>",
                      "slurm": "$WORK (slurm_prod_array.sh's default)"}[kind],
              "gpus": "10", "jobs_per_gpu": str(st.get("jobs_per_gpu", "")),
              "omp": str(st.get("omp", "cpus-per-task / jobs")), "first_seed": "0",
              "image": "the preset's pinned image", "upload": "no upload",
              "min_free": "100G"}
        for fid, text in ph.items():
            self.query_one(f"#{fid}", Input).placeholder = text

    def options(self):
        o = pw.Options(production=self.query_one("#production", Select).value,
                       machine=self.query_one("#machine", Select).value,
                       delete=self.query_one("#delete", Select).value)
        for fid, _, kind, attr, _ in FIELDS:
            w = self.query_one(f"#{fid}", Input)
            if w.has_class("hidden"):
                continue
            v = w.value.strip()
            if not v:
                continue
            if kind == "integer":
                try:
                    v = int(v)
                except ValueError:
                    continue
            setattr(o, attr, v)
        if not self.query_one("#upload", Input).value.strip():
            o.upload = ""
        return o

    def on_select_changed(self, ev: Select.Changed):
        if ev.select.id in ("production", "machine"):
            self.update_form()
        self.rebuild()

    def on_input_changed(self, ev: Input.Changed):
        if ev.input.id != "script":
            self.rebuild()

    # -- the plan ------------------------------------------------------------------------------
    def rebuild(self):
        try:
            self.opts = self.options()
            self.plan = pw.build(self.opts, self.P)
        except Exception as e:               # a half-typed form: say so, keep the last plan
            self.query_one("#plan", Static).update(Text(f"cannot plan this: {e}", style="red"))
            return
        p = self.plan
        parts = [Text(p.summary, style="bold")]
        if p.prereq:
            parts.append(Panel(Syntax("\n".join(p.prereq), "bash", word_wrap=True),
                               title="once, before the campaign", title_align="left"))
        parts.append(Panel(Syntax("\n".join(p.commands), "bash", word_wrap=True),
                           title="commands", title_align="left"))
        t = Table(show_header=False, box=None, padding=(0, 1))
        t.add_column(style="bold")
        t.add_column()
        t.add_column(style="dim")
        for lab, val, basis in p.estimates:
            style = "green" if basis == "measured" else ""
            t.add_row(lab, Text(val, style=style), basis)
        parts.append(Panel(t, title="estimates", title_align="left"))
        checks = Text()
        for lvl, msg in p.checks:
            checks.append(f"{lvl.upper():<6}", style=LEVEL_STYLE[lvl])
            checks.append(msg + "\n")
        parts.append(Panel(checks, title="checks", title_align="left"))
        if p.notes:
            parts.append(Panel(Text("\n".join(f"- {n}" for n in p.notes)), title="notes",
                               title_align="left"))
        parts.append(Text("sources: " + "; ".join(p.sources), style="dim"))
        self.query_one("#plan", Static).update(Group(*parts))
        camp = self.opts.campaign or f"{self.opts.production}_<date>"
        self.query_one("#script", Input).placeholder = f"launch_{camp}.sh"

    # -- actions -------------------------------------------------------------------------------
    def on_button_pressed(self, ev: Button.Pressed):
        if ev.button.id == "save":
            self.action_save()
        elif ev.button.id == "copy":
            self.action_copy()

    def action_save(self):
        if not self.plan:
            return
        path = self.query_one("#script", Input).value.strip() or \
            f"launch_{self.opts.campaign or self.opts.production}.sh"
        if os.path.exists(path):
            self.notify(f"{path} exists: give another script file", severity="error")
            return
        with open(path, "w") as f:
            f.write(pw.script(self.plan, self.opts))
        os.chmod(path, 0o755)
        if self.plan.ok:
            self.notify(f"wrote {os.path.abspath(path)}")
        else:
            self.notify(f"wrote {os.path.abspath(path)}, with ERRORS in its header",
                        severity="warning")

    def action_copy(self):
        if self.plan:
            self.copy_to_clipboard("\n".join(self.plan.commands))
            self.notify("commands copied (OSC 52; if the terminal ignores it, save the script)")


def run_wizard(presets):
    WizardApp(presets).run()
    return 0
