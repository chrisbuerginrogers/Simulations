import mujoco
import mujoco.viewer
import numpy as np

# Define a simple scene in MJCF (MuJoCo's XML format):
# a free-floating sphere with gravity enabled
xml = """
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1" rgba="0 .9 0 1" mass="1" solref="-0.8 0"/>
    </body>
  </worldbody>
</mujoco>
"""

# Load model and create simulation data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)

# Run 200 steps (~0.4 seconds of simulated time)
for i in range(200):
    mujoco.mj_step(model, data)
    if i % 50 == 0:
        z_pos = data.qpos[2]  # z-coordinate of the ball
        print(f"Step {i:3d} | time={data.time:.3f}s | ball z={z_pos:.4f}m")