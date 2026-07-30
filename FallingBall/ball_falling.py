import mujoco
import numpy as np

# A minimal model: one free-floating ball
xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1" rgba="1 0.3 0.3 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print(f"MuJoCo version: {mujoco.__version__}")
print(f"Model has {model.nbody} bodies, {model.nq} qpos DOFs")
print(f"Ball start pos: {data.qpos[:3]}")

# Step the simulation forward 1 second
for _ in range(int(1.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)

print(f"Ball pos after 1s (should have fallen): {np.round(data.qpos[:3], 3)}")
