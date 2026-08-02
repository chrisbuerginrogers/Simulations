# Generic mjlab Task Config Builder

A robot-agnostic version of [`ur3e/config_builder/`](../ur3e/config_builder/) — same
idea (pick settings and reward terms, download a ready-to-register mjlab task), but
for **any robot**, not just the UR3e reach task.

## Running it

```bash
python3 serve.py
```

Opens `http://127.0.0.1:8766` in your browser automatically. No dependencies beyond
Python's standard library — the page itself is plain HTML/CSS/JS, nothing to install.

## How it works

1. **Settings tab** — physics/episode settings (timestep, decimation, num_envs, ...)
   and PPO/RSL-RL training settings. These are genuinely generic to any mjlab
   `ManagerBasedRlEnv` task, so they need no per-robot customization.
2. **Robot tab** — tell it about your robot: a task ID, an entity name, the XML
   filename, and the joint names to control. Paste your robot's MJCF XML in to
   auto-detect joint and site names instead of typing them by hand.
3. **Rewards tab** — toggle two generic built-in reward terms (`action_rate`,
   `joint_vel`), plus two "reach a target site" terms that only apply if you've set
   an end-effector site in the Robot tab. Add fully custom reward terms (your own
   Python function) as needed.
4. **Code Snippets / Download Files tabs** — same as the UR3e tool: live-updating
   code previews, individual file downloads, and a "Download All (.zip)" button that
   bundles everything plus a generated `README.md` walking through setup — including
   **where to put your robot's XML file** (next to the generated `custom_task.py`,
   named to match what you entered in the Robot tab).

## What this generates (and what it doesn't)

The generated `custom_task.py` builds a complete `ManagerBasedRlEnvCfg` from scratch:
scene/entity setup from your XML, relative joint-position actions over the joints you
listed, `joint_pos_rel`/`joint_vel_rel` observations, your chosen reward terms, a
time-out termination, and a joint-reset event — plus the full PPO/RSL-RL runner
config and task registration. If you set an end-effector site, it also wires up
`commands.py`'s generic `ReachPositionCommandCfg` (copied from the UR3e task, which
was already written to be entity/site-name-agnostic) and the matching
distance/success-bonus reward terms and observations.

It does **not** generate or include:

- Your robot's actual MJCF file or mesh assets — that's yours to provide (see the
  generated README's "Where your robot's XML goes" section).
- A tuned home pose, reachable target position, or joint velocity limits for your
  specific robot — the generated file has clearly marked defaults/TODOs for these
  (all joints default to a 0.0 home position, the reach target defaults to an
  arbitrary `(0.3, 0.0, 0.3)`) that you'll want to adjust once you can see your robot
  in the simulation.
- Domain-specific reward tuning — the built-in terms are deliberately generic
  starting points, not validated for any particular robot.

Think of this as a working starting point that saves you the mjlab boilerplate, not a
fully-automatic "any robot, zero editing" generator.
