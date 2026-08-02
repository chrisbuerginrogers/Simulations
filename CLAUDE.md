# CLAUDE.md

Context for future Claude Code sessions working in this repo.

## What this repo is

MuJoCo / [mjlab](https://github.com/mujocolab/mjlab) experiments. Three projects, one
shared `uv`-managed virtual environment at the repo root (`pyproject.toml` / `uv.lock`,
`uv sync` to set up):

- **`FallingBall/`** — minimal MuJoCo example, no RL.
- **`pendulum/`** — another small physics-only example (MJCF + notebook + `viewer.py`).
- **`ur3e/`** — the actual project: a UR3e robot arm "reach" task built on mjlab, trained
  with PPO (via RSL-RL). This is where almost all the interesting work lives.

This repo is now **public** on GitHub (`chrisbuerginrogers/Simulations`) with
**GitHub Pages enabled**, serving from the `main` branch root. Anything committed here
is publicly visible and (for anything under a path Pages can reach) publicly hosted —
keep that in mind before committing secrets, personal data, or anything not meant for
a wide audience.

**Naming history:** this repo used to be called `MyCode` locally before being renamed
to `Simulations`. A few docs/scripts had stale `MyCode`/wrong-path references left over
from before the rename — if you spot another one, it's a bug, not intentional; fix it
to reference `Simulations` (or better, a relative/portable path) instead.

## The `ur3e/` task

- **`ur3e_reach_env_cfg.py`** — the actual task definition: scene, observations
  (18-dim: joint pos/vel + target position + end-effector-to-target vector), actions
  (6D relative joint-position deltas), reward terms, and `get_ur3e_reach_env_cfg()`.
- **`__init__.py`** — PPO/RSL-RL runner config (`ur3e_reach_ppo_runner_cfg()`) and task
  registration (`register_mjlab_task(...)`) for `Mjlab-UR3e-Reach` (and a
  `-ClampedPlay` variant for visually checking a velocity clamp).
- **`ur3e_gripper.xml`** — the MJCF (MuJoCo's XML model format) robot model.
- **`ur3e_walkthrough.ipynb`** — a long, deliberately beginner-friendly walkthrough of
  the whole task: `uv` primer, kernel setup, a full settings table, reward-by-reward
  breakdown, and a "Moving to the Tufts HPC" section. Runs on this repo's own
  `.venv` — no dependency on `my_mjlab_project` (see below) to use.
- **`ur3e/config_builder/`** — an interactive local web tool (see below).

**Known gotcha already fixed here:** mjlab's fixed-base mocap auto-attach step
silently discards any `<option integrator="...">` set in the MJCF XML (it gets merged
onto a fresh default spec and dropped, printing an "Attach conflict... keeping parent
value" warning). The integrator that actually takes effect is set once, in Python, via
`SimulationCfg(mujoco=MujocoCfg(integrator=...))` in `ur3e_reach_env_cfg.py` — don't
try to set it in the XML again.

**Version pin gotcha:** `mjlab==1.5.0`, `mujoco==3.10.0`, `mujoco-warp==3.10.0.1`,
`warp-lang==1.14.0`, `torch==2.12.1` are pinned in `pyproject.toml` deliberately.
Newer patch releases of `mujoco-warp`/`warp-lang` (3.10.0.3 / 1.15.0, as of writing)
have a regression that breaks multi-env `reset()` — if that starts throwing
`IndexError`, this version drift is why; pin back down.

## `my_mjlab_project` — a separate, untracked project

The actual `train`/`play` CLI (RSL-RL entry points) lives in a project called
`my_mjlab_project`, which is **not part of this git repo** — it's a separate project
that happens to live on the repo owner's machine, with its own `.venv` symlinked back
to this repo's `.venv` so both share installed package versions. Its location is
machine-specific (currently `~/Desktop/Summer26/MuJoCo/hackathon/my_mjlab_project` on
the maintainer's Mac) — **never hardcode an absolute path to it in anything meant to
be shared or downloaded** (scripts, generated files, docs aimed at other readers).
Use a clearly-marked placeholder instead (e.g. `/path/to/your/my_mjlab_project`) and
let the reader fill in their own.

## `ur3e/config_builder/` — the interactive config tool

A self-contained, dependency-free HTML+JS page (`index.html`) plus a tiny Python
stdlib server (`serve.py`, `python3 serve.py` → `http://127.0.0.1:8765`). Also hosted
live via GitHub Pages at:

**https://chrisbuerginrogers.github.io/Simulations/ur3e/config_builder/**

Layout is four tabs: **Settings** (physics/episode + PPO/training fields, two-column,
custom-rendered tooltips — see below), **Rewards** (master-detail list of all 9 reward
terms with live Python source), **Code Snippets** (three live-updating code previews),
**Download Files** (downloadable `train_custom.sh` + `custom_ur3e_task.py`).

Important: the "Download Files" tab's `custom_ur3e_task.py` **does not replace or
modify `ur3e_reach_env_cfg.py`**. It's a standalone file that imports reusable pieces
from `ur3e_reach_env_cfg.py` (config builder function, asset configs, reward
functions) and registers a *separate* task id, `Mjlab-UR3e-Reach-Custom`, alongside
the existing `Mjlab-UR3e-Reach`. `ur3e_reach_env_cfg.py` itself stays untouched on
disk either way — a user has to manually copy settings back into it if they want
their tuned config to become the permanent default.

Tooltips in this tool use a **custom-rendered tooltip** (a floating `div` positioned
with JS, appended to `<body>`), not the native HTML `title=` attribute — native
`title` tooltips are unreliable inside some embedded webviews (e.g. VS Code's Simple
Browser), which was the original bug report that led to this. If you add new
form fields, give them a `data-tip="..."` attribute (not `title=`) and they'll pick up
tooltips automatically via the existing `attachTooltips()` call.

Tooltip copy in this tool is intentionally written for **a smart reader with zero
domain background** — every acronym gets spelled out in-line (the tooltip has to
stand alone; there's no guaranteed reading order). Keep that bar if you add more.

## `template/` — the generic (robot-agnostic) version of the config tool

Same idea as `ur3e/config_builder/`, generalized to any robot instead of hardcoded to
UR3e reach. Live at **https://chrisbuerginrogers.github.io/Simulations/template/**
(also runnable locally via its own `serve.py`, port 8766 so it doesn't clash with
`ur3e/config_builder/`'s server on 8765). Five tabs: Settings (same generic
physics/PPO fields as the UR3e tool, byte-for-byte reusable), Robot (task id, entity
name, XML filename, joint names — with a "paste your MJCF, auto-detect joints/sites"
helper via regex), Rewards (two generic built-ins plus a fully custom
name/weight/Python-body reward builder), Code Snippets, Download Files.

The generated `custom_task.py` builds a complete `ManagerBasedRlEnvCfg` **from
scratch** (unlike the UR3e tool's version, which imports from an existing
`ur3e_reach_env_cfg.py`) — it has no dependency on this repo's `ur3e/` package at
all. If an end-effector site is set in the Robot tab, it also depends on
`template/commands.py`, a copy of `ur3e/commands.py`'s `ReachPositionCommandCfg`
(already entity/site-name-agnostic in its original form, just genericized in its
docstrings/defaults). Ship `commands.py`'s exact source is also embedded as a JS
string constant (`COMMANDS_PY_SOURCE`) inside `template/index.html` so the download
doesn't need a server round-trip — **keep both copies in sync by hand** if you edit
the reach-command logic.

This is a real starting-point generator, verified end-to-end (fake-DOM harness run
through Node, output checked with `ast.parse`), not just a mockup — but it's
explicitly not a fully-automatic "any robot, zero editing" tool: home pose, target
position, and reward tuning are left as clearly-marked defaults/TODOs since those are
inherently robot-specific.

## Working conventions in this repo

- Commits end with a `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`
  trailer.
- Only commit/push when explicitly asked — this repo's owner iterates a lot before
  wanting things pushed.
- When editing `.ipynb` files, use the `NotebookEdit` tool, not a raw text editor —
  cells you didn't explicitly give an `id` to get addressed by positional `cell-N`
  synthetic ids from the Read tool; cells inserted via `NotebookEdit` get real
  generated ids. Double-check which is which before editing, since misaddressing a
  cell silently overwrites the wrong one.
- Before recommending or reusing something described in old chat history/memory,
  verify it still exists in the current files — things get renamed/moved/removed.
