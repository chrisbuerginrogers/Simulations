# LEGO × mjlab — Sim-to-Real Workbench

A self-contained teaching page for running reinforcement learning on LEGO Education
hardware: train a policy in MuJoCo/mjlab, run it on the hub over Bluetooth, and
**measure how far the two worlds actually are apart**.

Live: **https://chrisbuerginrogers.github.io/Simulations/lego/**

Local: `python3 serve.py` → <http://127.0.0.1:8767> (ports: `8765` = `ur3e/config_builder`,
`8766` = `template`, `8767` = here).

Dependency-free — one `index.html`, no build step, no external requests.

## What it is for

The other two tools in this repo (`ur3e/config_builder/`, `template/`) generate a task
config. This one is aimed at a **student running the whole loop**, and its subject is
specifically the sim-to-real gap rather than the config.

The visual system encodes that: **blue is simulation, orange is hardware**, everywhere on
the page, including the chart. The palette is the `dataviz` skill's validated categorical
slots 1 and 2 and passes all six checks in both light and dark mode.

## Tabs

| Tab | What it does |
|---|---|
| **Setup** | Five ordered, collapsible steps — install mjlab, install `legoeducation` + SimpleLE, pair the hub and set the update rate, smoke-test training on a stock task, log one open-loop trajectory in both worlds. Each has a copyable command and an explicit "worked if…" check. |
| **Projects** | Six RL tasks, each with why RL beats a hand-derived controller, what sensing it needs, a **reset-burden meter**, and the *distinct* sim-to-real lesson it teaches. **Drive straight on beams** is flagged `START HERE`. |
| **Robot Model** | Three routes to an MJCF, compared: Onshape + `onshape-to-robot` (recommended), BrickLink Studio → mesh, or ruler-and-text-editor. Route A written out in full with `config.json`. |
| **Train** | Control rate (capped at the 20 Hz BLE ceiling), action-delay randomization and four domain-randomization ranges → a live `uv run train` command. Control rate drives the printed `decimation` at a 2 ms physics step and echoes the matching `set_update_rate(ms)`. |
| **Task Anatomy** | Master-detail over the eight managers that make up a task — Scene, Observations, Actions, Rewards, Terminations, Events, Simulation, PPO runner. Each shows what it decides, a **knobs table** (name / typical value / what it changes), and real Python. |
| **Deploy** | Policy export command plus a `bridge.py` built on SimpleLE that loads the exported policy, steps it at the trained rate, and logs `real_run.csv`. The `TODO` functions are the whole sim-to-real contract. |
| **Measure the Gap** | The point of the page. Overlays a simulated and a measured run, reports RMS / peak / growth-ratio / time-within-5°, and maps the **shape** of the divergence to a likely cause and fix. |

## The Measure-the-Gap tab

Ships five sample runs — `lag`, `drift`, `contact`, `flex`, `tuned` — generated
deterministically in `makeRun()`. They are **synthetic demo data**, clearly labelled as
sample runs, chosen so each one exhibits a different failure signature:

| Sample | RMS | Growth | Signature → cause |
|---|---|---|---|
| `lag` | 2.0° | 0.84× | phase-shifted, scaled down → actuator too strong in model |
| `drift` | 9.5° | 2.70× | linear one-sided growth → unmodelled asymmetry |
| `contact` | 9.9° | 3.06× | steps at footfalls → contact `solref`/`solimp` |
| `flex` | 2.3° | 1.73× | extra high-frequency ringing → structural compliance |
| `tuned` | 1.2° | 1.01× | bounded, non-growing → this is what success looks like |

Real data goes in via **Load CSV pair…** — two files with columns `t,heading_deg`, one
filename containing `sim` and one containing `real`. Chart scaling, the hover crosshair,
the stat tiles and the table stride are all derived from the loaded run's length and
duration, so a CSV of any length works.

## Hardware constraints baked into the page

- **20 Hz is a ceiling, not a preference.** Nothing is stored on the hub — every sample streams
  to the PC at the rate the code requests, and BLE tops out around 50 ms. The control-rate
  slider is capped at 20 Hz for this reason and echoes the matching `set_update_rate(ms)`.
- The bridge uses **[SimpleLE](https://github.com/chrisbuerginrogers/SimpleLE)** (`lelib.py`) on
  top of `legoeducation`, because `set_update_rate()` is the control that matters here.
  API used: `doubleMotor.connect/set_update_rate/set_speed_left/set_speed_right/run/stop`,
  `colorSensor.reflection()`.

## Caveats — read before using this in a class

- The **task ids are placeholders** (`Mjlab-Lego-DriveStraight`, etc.). No mjlab task
  registrations exist for them in this repo yet; use `template/` to generate real ones.
- **Sensing is the real design constraint.** SimpleLE's documented surface is motors, colour
  sensor and controller — there is no orientation or per-motor encoder getter. Tasks needing a
  heading (fall recovery, parking) need a gyro via `legoeducation` directly, dead reckoning, a
  taped line read with `reflection()`, or an overhead camera. The page says this in two places;
  confirm it on your hardware before designing a task around it.
- The Task Anatomy Python is **illustrative of mjlab's manager structure**, not copied from a
  task that exists in this repo. Term names (`mdp.base_heading`, `mdp.randomize_actuator_gains`)
  follow the Isaac-Lab-style convention mjlab uses but should be checked against the installed
  version.
- The `env.randomize.*` CLI overrides follow mjlab's override style but assume randomization
  fields your task config has to actually define.

Bluntly: the pedagogy, the layout and the gap analysis are the finished parts. The exact
API surface still needs one pass against real hardware and a real registered task.
