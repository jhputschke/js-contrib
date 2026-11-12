# AI assistants

An AI assistant can run a test event for you, write an analysis script against the readers, read a job log, or explain what is in a file. It does this well only if it knows how js-contrib is set up. Two things give it that knowledge:

- **an instruction file in your examples directory**, `AGENTS.md` (with a `CLAUDE.md` that imports it), for assistants that work in your terminal and on your files: Claude Code, OpenAI Codex, Cursor, GitHub Copilot and others;
- **this website as plain Markdown**, [`llms.txt`](https://jhputschke.github.io/js-contrib/llms.txt) and [`llms-full.txt`](https://jhputschke.github.io/js-contrib/llms-full.txt), for any assistant that can read a web page or a file: claude.ai, ChatGPT, Gemini and others.

## Coding agents: work in your examples directory

`xscape-examples` copies the examples and production scripts into a directory of yours ([Your first event](https://jhputschke.github.io/js-contrib/run/first-event/index.md)). That copy holds the two instruction files. Start the assistant there:

```bash
xscape-examples ~/xscape/examples      # once; in the container: xscape-examples /work/examples
cd ~/xscape/examples
claude                                 # Claude Code; or codex, cursor-agent, ...
```

Then ask in plain words, for example:

- "Run one O+O event and plot the charged-hadron pT spectrum."
- "Run two Au+Au jet events with seed 1, hadronize them with 200 oversamples, and check the logs."
- "What is in `out/AuAu_0_10_jet_seed0001_particlize.h5`?"
- "Write a script that reads this campaign and makes the wake in Δφ, with errors."

The instruction file tells the assistant how commands run in your installation (inside the container with `docker run` or `apptainer exec`, in the wheel's environment on a Mac, or in a source build), what a first test is, where the drivers and their options are, what the output files hold and how to read them, and the rules: test with one or two events first, start no campaign without your go, never write into the installation, do not treat oversamples as events, look options up instead of guessing.

| assistant                                                          | file it reads                                                                                               |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Claude Code                                                        | `CLAUDE.md`, which imports `AGENTS.md`                                                                      |
| OpenAI Codex, Cursor, GitHub Copilot, and most other coding agents | `AGENTS.md`                                                                                                 |
| Gemini CLI                                                         | `GEMINI.md` by default: set `AGENTS.md` as its context file in the settings, or tell it to read `AGENTS.md` |
| any other                                                          | tell it: "read `AGENTS.md` first"                                                                           |

`CLAUDE.md` is Claude Code's own file name; other assistants do not look for it. `AGENTS.md` is the name most tools share, which is why the text lives there and `CLAUDE.md` only points to it. Change `AGENTS.md` if you want to add rules of your own (your cluster's batch system, where your data are).

If your copy has no `AGENTS.md`

The files ship with the examples from 2026-11-12 on (later images and wheels). For an older installation, download them into your copy:

```bash
cd ~/xscape/examples
B=https://raw.githubusercontent.com/jhputschke/js-contrib/main/contribs/PyJetscape/example
curl -LO $B/AGENTS.md -LO $B/CLAUDE.md
```

**With the container** the assistant runs on the host and X-SCAPE in the image, so it wraps every command in `docker run` or `apptainer exec`; the instruction file shows how. Tell it your image (`jhputschke/xscape-prod:cu126`, or the `.sif` file) and which host directory is `/work`. **In a source checkout** the files are in `contribs/PyJetscape/example`, and work the same way when the assistant is started there.

## Chat assistants: give them `llms.txt`

An assistant without access to your files can still answer from the documentation. [`llms.txt`](https://jhputschke.github.io/js-contrib/llms.txt) is the index of this site for language models ([llmstxt.org](https://llmstxt.org)): a short description and one line per page, each linking to that page as Markdown.

- **With web access** (claude.ai, ChatGPT, Gemini): put the address in your question. "Using https://jhputschke.github.io/js-contrib/llms.txt, how do I run the O+O production on a SLURM cluster with Apptainer?" The assistant reads the index and then the pages it needs.
- **A single page as Markdown:** add `index.md` to its address, e.g. `https://jhputschke.github.io/js-contrib/run/oo-isr/index.md`. This is smaller and cleaner than the HTML page.
- **The whole site in one file:** [`llms-full.txt`](https://jhputschke.github.io/js-contrib/llms-full.txt), 1.2 MB, roughly 300 000 tokens. That is more than most chat models take in one conversation. Use it where the assistant searches an uploaded file instead of reading all of it (a Claude project, a custom GPT, NotebookLM), or with a model that has a context of a million tokens.
- **For commands,** also paste [`AGENTS.md`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/AGENTS.md): it is short and holds the rules and the per-installation command forms.

## What to check yourself

- **Physics numbers.** The analysis scripts are examples, not vetted results ([Analyse](https://jhputschke.github.io/js-contrib/analysis/index.md)). Ask the assistant for the cuts, the number of events and the error it used, and check a result against a second method before you rely on it.
- **Options and names.** Assistants invent plausible options, XML tags and dataset names. The truth is `--help`, the folder's `README.md`, and `h5_inspect.py` for a file.
- **What actually ran.** Ask for the exact command and the log lines that show the result. A line `No CUDA device found` means the run was on the CPU, which is slow and not the same numbers.
- **Cost.** A campaign takes hours of GPU time and many GB of disk. Decide that yourself; the instruction file tells the assistant to ask first.

How the two `llms` files are made: [Building this site](https://jhputschke.github.io/js-contrib/reference/site/#llmstxt).
