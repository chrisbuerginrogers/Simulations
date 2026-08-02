# Simulations

MuJoCo / [mjlab](https://github.com/mujocolab/mjlab) experiments, from a simple falling-ball
sanity check up through a UR3e robot arm reach task trained with reinforcement learning.

## Projects

- **[FallingBall](FallingBall/)** — minimal MuJoCo example: a ball dropped under gravity, viewed
  both headless and in the interactive viewer. Good starting point if you're new to MuJoCo.
- **[pendulum](pendulum/)** — a simple pendulum model (MJCF + notebook), another small
  physics-only example.
- **[ur3e](ur3e/)** — a UR3e robot arm "reach" task built on [mjlab](https://github.com/mujocolab/mjlab)
  (a MuJoCo-based RL framework): a policy learns to move the arm's end-effector to randomly
  placed targets, trained with PPO via RSL-RL. Includes:
  - [`ur3e_walkthrough.ipynb`](ur3e/ur3e_walkthrough.ipynb) — a full walkthrough of the task setup,
    the reward terms, and why each setting is what it is.
  - [`config_builder/`](ur3e/config_builder/) — an interactive tool for exploring the task's
    settings and reward terms, and generating a ready-to-drop-in config file. Try it live:

    ### 👉 [UR3e Reach Config Builder](https://chrisbuerginrogers.github.io/Simulations/ur3e/config_builder/)

## Setup

Each project shares one [`uv`](https://docs.astral.sh/uv/)-managed virtual environment at the repo
root (`pyproject.toml` / `uv.lock`). To get started:

```bash
uv sync
```

See [`ur3e_walkthrough.ipynb`](ur3e/ur3e_walkthrough.ipynb) for a `uv` primer and a full breakdown
of the `ur3e` task if this is your first time in the repo.
