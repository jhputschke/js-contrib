"""
utils/jobmon_tui.py -- the terminal UI of jobmon.py: `utils/jobmon.py OUTBASE --tui`.

Needs textual (utils/analysis_env/requirements.txt).  The data are jobmon.py's: this module
only draws them.  The files are read in a worker thread, so the screen stays responsive
while a large campaign is scanned.

    Jobs       every job: state, events, s/event, ETA, log idle, warnings.  Click a column
               to sort by it (again: reverse).  The selected job's details and the end of
               its log (followed while it grows) are on the right.
    Problems   failed and short jobs, the deposition check; notes (edge hits, skips).
               Enter on one shows that job in the Jobs tab.
    Warnings   the most frequent warnings of all jobs, grouped by message

    keys: f filter (all / running / problems / done)   r refresh now   j/k or arrows: move
          1 2 3 tabs   q quit

In a browser, e.g. for a workshop: textual serve "utils/jobmon.py OUTBASE --tui"
(textual-dev).
"""

from __future__ import annotations

import os
import time

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (DataTable, Footer, Header, Log, ProgressBar, Static,
                             TabbedContent, TabPane)

import jobmon as jm

STATE_STYLE = {"done": "green", "running": "bold cyan", "FAILED": "bold red",
               "short": "yellow", "?": "dim"}
FILTERS = ("all", "running", "problems", "done")
COLUMNS = ("job", "state", "dir", "events", "s/event", "ETA", "log idle", "warnings")
LOG_TAIL = 400            # lines of the selected job's log
LOG_BYTES = 256 * 1024    # read at most this much from its end


def sort_key(col, row):
    """Sort values for a jobs-table row (row: the dict snapshot() builds)."""
    return {
        "job": row["stem"],
        "state": ("FAILED", "short", "running", "done", "?").index(row["state"]),
        "dir": row["dir"],
        "events": (row["done"] / row["total"]) if row["total"] else -1,
        "s/event": row["s_ev"] or 0.0,
        "ETA": row["eta"] or 0.0,
        "log idle": row["idle"] if row["idle"] is not None else -1.0,
        "warnings": row["warnings"],
    }[col]


def snapshot(mon):
    """Scan and copy what the screen needs into plain data (runs in the worker thread)."""
    mon.scan()
    r = mon.report()
    rows = []
    for j in r["jobs"]:
        s = j.log.s_per_event()
        eta = (j.total - j.done) * s if s and j.total and j.state == "running" else None
        rows.append(dict(
            key=os.path.join(j.outdir, j.stem), stem=j.stem, state=j.state,
            dir=os.path.basename(j.outdir), done=j.done, total=j.total, s_ev=s, eta=eta,
            idle=j.log_age if j.state == "running" else None,
            warnings=sum(j.log.warnings.values())))
    key = lambda j: os.path.join(j.outdir, j.stem) if j else None     # noqa: E731
    return dict(
        report=r, rows=rows,
        summary=jm.summary_lines(r),
        problems=[(key(j), j.stem if j else "?", j.state if j else "?", why)
                  for j, why in r["problems"]],
        notes=[(key(j), j.stem, j.state, why) for j, why in r["notes"]],
        warnings=r["warnings"].most_common(100),
        flagged={key(j) for j, _ in r["problems"] if j},
        t=time.time())


def job_detail(job):
    """The text of the detail pane for one job (read in the UI thread: cheap)."""
    lg, r = job.log, job.summary or {}
    s = lg.s_per_event()
    lines = [
        Text(job.stem, style="bold"),
        Text.assemble(("state  ", "dim"), (job.state, STATE_STYLE.get(job.state, ""))),
        Text(f"dir    {job.outdir}"),
    ]
    if job.reason:
        lines.append(Text(f"why    {job.reason}", style="red"))
    if job.pid:
        lines.append(Text(f"pid    {job.pid}"))
    ev = f"{job.done}/{job.total or '?'}"
    if lg.done:
        ev += (f", last {len(lg.times)}: {s:.1f} s/event, all: {lg.sum_s / lg.done:.1f} "
               "s/event")
    lines.append(Text(f"events {ev}"))
    if job.state == "running" and s and job.total:
        lines.append(Text(f"ETA    {jm.fmt_dt((job.total - job.done) * s)}"))
    if job.log_age is not None:
        lines.append(Text(f"log    changed {jm.fmt_dt(job.log_age)} ago"))
    for k in ("seed", "campaign", "wall_s", "droplets_total", "events_hit_edge",
              "events_skipped", "hadronize_s_total"):
        if r.get(k) not in (None, 0, {}):
            lines.append(Text(f"{k:<14} {r[k]}", style="dim"))
    if lg.warnings:
        lines.append(Text("warnings", style="bold"))
        for w, n in lg.warnings.most_common(4):
            lines.append(Text(f"{n:>5}  {w}", overflow="ellipsis", no_wrap=True))
    return Text("\n").join(lines)


def log_tail(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - LOG_BYTES))
            data = f.read()
    except OSError:
        return ["(no log)"]
    lines = jm.ANSI.sub("", data.decode("utf-8", "replace")).splitlines()
    return lines[-LOG_TAIL:]


class JobmonApp(App):
    TITLE = "jobmon"
    CSS = """
    #top { height: auto; padding: 0 1; }
    #summary { height: auto; }
    #progress { margin: 0 0 1 0; }
    #jobs-pane { height: 1fr; }
    #jobs { width: 3fr; }
    #side { width: 2fr; border-left: solid $panel; }
    #detail { height: auto; max-height: 50%; padding: 0 1; }
    #log { height: 1fr; border-top: solid $panel; }
    #problems, #notes, #warnings { height: 1fr; }
    #notes { border-top: solid $panel; }
    """
    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("r", "refresh", "refresh"),
        Binding("f", "filter", "filter"),
        Binding("1", "tab('tab-jobs')", "jobs", show=False),
        Binding("2", "tab('tab-problems')", "problems", show=False),
        Binding("3", "tab('tab-warnings')", "warnings", show=False),
        Binding("j", "cursor(1)", show=False),
        Binding("k", "cursor(-1)", show=False),
    ]

    def __init__(self, mon, interval):
        super().__init__()
        self.mon, self.interval = mon, interval
        self.snap = None
        self.filter = "all"
        self.sort_col, self.sort_rev = "job", False
        self.selected = None          # key of the job in the detail pane
        self.log_size = None          # size of its log when last shown
        self.scanning = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="top"):
            yield Static("reading the job files ...", id="summary")
            yield ProgressBar(id="progress", show_eta=False)
        with TabbedContent(initial="tab-jobs"):
            with TabPane("Jobs", id="tab-jobs"):
                with Horizontal(id="jobs-pane"):
                    yield DataTable(id="jobs", cursor_type="row", zebra_stripes=True)
                    with Vertical(id="side"):
                        yield Static("", id="detail")
                        yield Log(id="log", highlight=False, max_lines=LOG_TAIL * 2)
            with TabPane("Problems", id="tab-problems"):
                yield DataTable(id="problems", cursor_type="row")
                yield DataTable(id="notes", cursor_type="row")
            with TabPane("Warnings", id="tab-warnings"):
                yield DataTable(id="warnings", cursor_type="row")
        yield Footer()

    def on_mount(self):
        self.sub_title = " ".join(self.mon.dirs)
        self.query_one("#problems", DataTable).add_columns("job", "state", "problem")
        self.query_one("#notes", DataTable).add_columns("job", "state", "note")
        self.query_one("#warnings", DataTable).add_columns("count", "warning")
        self.action_refresh()
        self.set_interval(self.interval, self.action_refresh)
        self.set_interval(2.0, self.follow_log)

    # -- data ----------------------------------------------------------------------------------
    def action_refresh(self):
        if not self.scanning:
            self.scanning = True
            self.scan()

    @work(thread=True, exclusive=True)
    def scan(self):
        try:
            snap = snapshot(self.mon)
        finally:
            self.scanning = False
        self.call_from_thread(self.show, snap)

    def show(self, snap):
        self.snap = snap
        r = snap["report"]
        self.sub_title = (", ".join(r["campaigns"]) + "  " if r["campaigns"] else "") + \
            " ".join(self.mon.dirs)
        self.query_one("#summary", Static).update("\n".join(snap["summary"]))
        self.query_one("#progress", ProgressBar).update(total=max(r["all_ev"], 1),
                                                        progress=r["done_ev"])
        self.fill_jobs()
        self.fill_table("#problems", [(Text(s, style="bold"), st_text(st), why)
                                      for _, s, st, why in snap["problems"]],
                        [k for k, *_ in snap["problems"]],
                        "no failed, short or flagged job")
        self.fill_table("#notes", [(s, st_text(st), why) for _, s, st, why in snap["notes"]],
                        [k for k, *_ in snap["notes"]], "no grid-edge hits, no skipped events")
        self.fill_table("#warnings", [(f"{n:,}", w) for w, n in snap["warnings"]],
                        None, "no warnings")
        tabs = self.query_one(TabbedContent)
        np = len(snap["problems"])
        tabs.get_tab("tab-problems").label = f"Problems ({np})" if np else "Problems"
        self.show_detail()

    def visible_rows(self):
        rows = self.snap["rows"]
        if self.filter == "running":
            rows = [x for x in rows if x["state"] == "running"]
        elif self.filter == "problems":
            rows = [x for x in rows if x["state"] in ("FAILED", "short")
                    or x["key"] in self.snap["flagged"]]
        elif self.filter == "done":
            rows = [x for x in rows if x["state"] == "done"]
        return sorted(rows, key=lambda x: sort_key(self.sort_col, x), reverse=self.sort_rev)

    def fill_jobs(self):
        t = self.query_one("#jobs", DataTable)
        keep = self.selected
        t.clear(columns=True)
        arrow = " ▼" if self.sort_rev else " ▲"
        for c in COLUMNS:
            t.add_column(c + (arrow if c == self.sort_col else ""), key=c)
        for x in self.visible_rows():
            stem = Text(x["stem"], style="bold red" if x["key"] in self.snap["flagged"] else "")
            t.add_row(stem, st_text(x["state"]), x["dir"], f"{x['done']}/{x['total'] or '?'}",
                      f"{x['s_ev']:.1f}" if x["s_ev"] else "-",
                      jm.fmt_dt(x["eta"]) if x["eta"] else "-",
                      jm.fmt_dt(x["idle"]) if x["idle"] is not None else "-",
                      str(x["warnings"]), key=x["key"])
        if keep is not None:
            try:
                t.move_cursor(row=t.get_row_index(keep), animate=False)
            except Exception:              # the job is filtered out now
                pass
        filt = "" if self.filter == "all" else f"  [filter: {self.filter}]"
        self.query_one(TabbedContent).get_tab("tab-jobs").label = \
            f"Jobs ({t.row_count}){filt}"

    def fill_table(self, sel, rows, keys, empty):
        """Rows; keys: the job of each row (Enter jumps to it), or None."""
        t = self.query_one(sel, DataTable)
        t.clear()
        if not rows:
            t.add_row(*(["-"] * (len(t.columns) - 1) + [Text(empty, style="dim")]))
            return
        for i, row in enumerate(rows):
            t.add_row(*row, key=f"{i}|{keys[i]}" if keys and keys[i] else None)

    # -- the selected job ----------------------------------------------------------------------
    def on_data_table_row_highlighted(self, ev: DataTable.RowHighlighted):
        if ev.data_table.id != "jobs" or ev.row_key is None:
            return
        if ev.row_key.value != self.selected:
            self.selected = ev.row_key.value
            self.log_size = None
            self.show_detail()

    def on_data_table_row_selected(self, ev: DataTable.RowSelected):
        """Enter on a problem or note: show that job in the Jobs tab."""
        if ev.data_table.id not in ("problems", "notes") or ev.row_key.value is None:
            return
        key = ev.row_key.value.split("|", 1)[1]
        self.filter, self.selected, self.log_size = "all", key, None
        self.action_tab("tab-jobs")
        self.fill_jobs()
        self.show_detail()
        self.query_one("#jobs", DataTable).focus()

    def on_data_table_header_selected(self, ev: DataTable.HeaderSelected):
        if ev.data_table.id != "jobs":
            return
        col = COLUMNS[ev.column_index]
        self.sort_rev = not self.sort_rev if col == self.sort_col else False
        self.sort_col = col
        if self.snap:
            self.fill_jobs()

    def show_detail(self):
        job = self.mon.jobs.get(self.selected) if self.selected else None
        detail = self.query_one("#detail", Static)
        if job is None:
            detail.update(Text("select a job", style="dim"))
            return
        detail.update(job_detail(job))
        self.follow_log(force=True)

    def follow_log(self, force=False):
        if not self.selected:
            return
        path = self.selected + ".log"
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        if not force and size == self.log_size:
            return
        self.log_size = size
        log = self.query_one("#log", Log)
        log.clear()
        log.write_lines(log_tail(path))

    # -- actions -------------------------------------------------------------------------------
    def action_filter(self):
        self.filter = FILTERS[(FILTERS.index(self.filter) + 1) % len(FILTERS)]
        if self.snap:
            self.fill_jobs()
        self.notify(f"showing {self.filter} jobs", timeout=2)

    def action_tab(self, tab):
        self.query_one(TabbedContent).active = tab

    def action_cursor(self, step):
        t = self.query_one("#jobs", DataTable)
        t.move_cursor(row=max(0, t.cursor_row + step))


def st_text(state):
    return Text(state, style=STATE_STYLE.get(state, ""))


def run_tui(mon, interval):
    JobmonApp(mon, interval).run()
    return 0
