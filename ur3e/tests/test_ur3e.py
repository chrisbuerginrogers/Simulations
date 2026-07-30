import mujoco
import mujoco.viewer
import time, math

m = mujoco.MjModel.from_xml_path('src\\my_mjlab_project\\tasks\\ur3e_reach\\scene.xml')
d = mujoco.MjData(m)

ee_site_id = m.site('end_effector').id

degrees = True

traj = [
    #[0, 0, 0, 0, 0, 0, 0],
    [20.721578168146426, -58.06172126407012, 59.061919810751554, -210.25601614399187, -108.17117118843998, 13.62671916779568],
]

if degrees:
    for point in traj:
        for val in point:
            point[point.index(val)] = math.radians(val)


with mujoco.viewer.launch_passive(m, d) as viewer:
    # keep the window alive/redrawing instead of sleeping blind
    for _ in range(10):
        viewer.sync()
        time.sleep(0.005)

    for point in traj:
        for i, val in enumerate(point):
            d.ctrl[i] = val
        for _ in range(500):
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(0.005)

    for _ in range(100):
        viewer.sync()
        time.sleep(0.005)

    ee_pos = d.site_xpos[ee_site_id]
    print(f"End-effector position after move: x={ee_pos[0]:.4f}, y={ee_pos[1]:.4f}, z={ee_pos[2]:.4f}")

    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(0.005)
