# ball_falling.py — Code Explanation

## Overview

This script creates a minimal MuJoCo physics simulation: a red ball dropped from 1 meter above the ground. It runs for 1 simulated second and prints the ball's final position.

---

## The XML Model (lines 5–16)

The `xml` string defines the physics scene using MuJoCo's XML format (MJCF).

### `<mujoco>` (line 6)
Root element that wraps the entire model definition.

### `<worldbody>` (line 7)
Top-level container for all physical objects in the scene.

### `<light pos="0 0 3"/>` (line 8)
Places a light source at (x=0, y=0, z=3) — 3 meters directly above the origin. Required for rendering.

### `<geom type="plane" size="5 5 0.1"/>` (line 9)
Creates a flat ground plane at the origin. `size="5 5 0.1"` sets the rendered half-extents to 5×5 meters with 0.1 m thickness. This is the surface the ball falls onto.

### `<body pos="0 0 1">` (line 10)
Defines a rigid body initially positioned 1 meter above the ground (z=1).

### `<freejoint/>` (line 11)
Gives the body 6 degrees of freedom (3 translational + 3 rotational), allowing it to move and tumble freely through space. Without this, the body would be fixed.

### `<geom type="sphere" size="0.1" rgba="1 0.3 0.3 1"/>` (line 12)
Attaches a sphere collider/visual to the body.
- `size="0.1"` — radius of 0.1 meters
- `rgba="1 0.3 0.3 1"` — reddish color (R=1, G=0.3, B=0.3, A=1 fully opaque)

---

## Simulation Setup (lines 18–19)

```python
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
```

- `MjModel` — the static model (geometry, physics parameters, constraints)
- `MjData` — the dynamic state (positions, velocities, forces) that changes each step

---

## Stepping the Simulation (lines 26–27)

```python
for _ in range(int(1.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
```

Advances the simulation by one second total. Each call to `mj_step` moves the simulation forward by one timestep (default 2 ms), so this loop runs ~500 iterations.

---

## Expected Output

```
MuJoCo version: <version>
Model has 2 bodies, 7 qpos DOFs
Ball start pos: [0. 0. 1.]
Ball pos after 1s (should have fallen): [0.    0.    0.1  ]
```

The ball falls from z=1 and comes to rest at z=0.1 (its radius), sitting on the ground plane.
