"""
Runs the real simulator + trained policy in closed loop toward a SPECIFIC
target point (not a random hemisphere sample), logging action, delta, and
joint_pos at every step -- for direct comparison against a real-robot
trajectory toward the same point.

Uses the UR3E_TARGET_POS env-var override already built into
ur3e_reach_env_cfg.py's _build_commands() -- setting it here BEFORE
building the env cfg switches the reach_target command to a fixed point
instead of random hemisphere sampling.
"""

import os
import sys
from datetime import datetime
from dataclasses import asdict

import torch

# Fixed target for this test run -- must be set before importing/building
# the env cfg, since _build_commands() reads this env var at construction
# time, not per-reset.
#TARGET_X, TARGET_Y, TARGET_Z = 0.428, 0.229, 0.534
TARGET_X, TARGET_Y, TARGET_Z = 0.3, 0.0, 0.3
os.environ["UR3E_TARGET_POS"] = f"{TARGET_X},{TARGET_Y},{TARGET_Z}"

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

# Same sys.path + full-dotted-path pattern used successfully elsewhere --
# adjust the path to match wherever "src" actually sits relative to this file.
sys.path.insert(0, r"C:\Users\leiah\my_mjlab_project\src")

from my_mjlab_project.tasks.ur3e_reach.ur3e_reach_env_cfg import get_ur3e_reach_env_cfg
from my_mjlab_project.tasks.ur3e_reach import ur3e_reach_ppo_runner_cfg

#CHECKPOINT_PATH = r"C:\Users\leiah\Documents\CEEO_Summer26_local\model_3999.pt"
CHECKPOINT_PATH = r"C:\Users\leiah\Documents\CEEO_Summer26_local\models\model_3100.pt"
DEVICE = "cpu"
NUM_STEPS = 500
ACTION_SCALE = 0.08
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE = os.path.abspath(f"sim_rollout_log_{TIMESTAMP}.txt")

# Real UR3e joint velocity limits (official spec): shoulder/elbow are
# Size 2/1 joints at pi rad/s, wrists are faster (this robot's own
# outlier, confirmed against the official URDF earlier) at 2*pi rad/s.
REAL_JOINT_VEL_LIMITS = [3.14159, 3.14159, 3.14159, 6.28319, 6.28319, 6.28319]

# Running right at 100% of hardware max is genuinely fast for a lab
# setting -- SAFETY_FACTOR scales the effective clamp limits down from
# there. 0.25 is a conservative starting point for initial testing;
# raise it later once you're comfortable with how the arm behaves at
# this speed. This is the one knob to adjust rather than hand-editing
# CLAMP_LIMITS directly.
SAFETY_FACTOR = 0.25
CLAMP_LIMITS = [v * SAFETY_FACTOR for v in REAL_JOINT_VEL_LIMITS]

# Intended real-robot per-step duration (matches move_joints_r's `time`
# argument in myur3e_policy_runner.py) -- this is what "implied velocity"
# is computed against, since that's the actual deadline the real
# trajectory controller has to hit.
STEP_DURATION = 0.1


def clamp_delta_to_velocity_limits(delta, step_duration, limits):
    """
    Scales the WHOLE delta vector by a single shared factor (not each
    joint independently) so no joint's implied velocity (|delta| /
    step_duration) exceeds its real hardware limit. Uniform scaling
    preserves the relative proportions across joints -- the coordinated
    "shape" of the multi-joint motion -- rather than distorting it by
    clamping only the offending joint(s) in isolation.

    Returns (clamped_delta, scale_factor). scale_factor == 1.0 means no
    clamping was needed this step.
    """
    implied_vel = [abs(d) / step_duration for d in delta]
    ratios = [v / l for v, l in zip(implied_vel, limits)]
    max_ratio = max(ratios)
    if max_ratio > 1.0:
        scale = 1.0 / max_ratio
        return [d * scale for d in delta], scale
    return list(delta), 1.0


env_cfg = get_ur3e_reach_env_cfg(play=True)
agent_cfg = ur3e_reach_ppo_runner_cfg()

env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE, render_mode=None)
wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = MjlabOnPolicyRunner(wrapped_env, asdict(agent_cfg), device=DEVICE)
runner.load(CHECKPOINT_PATH, load_cfg={"actor": True}, strict=True, map_location=DEVICE)
policy = runner.get_inference_policy(device=DEVICE)


def extract_obs(obs_container):
    """
    NOT used to feed policy() -- rsl_rl's MLPModel.get_latent() does its
    own internal group indexing (obs[obs_group] for obs_group in
    self.obs_groups), so it needs the FULL TensorDict passed in raw, not
    a pre-extracted single-group tensor. Kept only in case you want to
    inspect a specific group's raw values for debugging/logging.
    """
    if hasattr(obs_container, "get"):
        return obs_container.get("actor")
    return obs_container


reset_result = wrapped_env.reset()
obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

arm = env.scene["ur3e"]

print(f"Target: [{TARGET_X}, {TARGET_Y}, {TARGET_Z}]")
print(f"Logging {NUM_STEPS} steps to {OUTPUT_FILE}")

with open(OUTPUT_FILE, "w") as f:
    f.write(f"# Target: [{TARGET_X}, {TARGET_Y}, {TARGET_Z}]\n")
    f.write(
        "# step  |  raw_delta (6)  |  clamped_delta (6)  |  scale_factor  |  joint_pos (6, arm only)\n"
    )

    for step in range(NUM_STEPS):
        with torch.no_grad():
            action = policy(obs)  # raw TensorDict in, not extract_obs(obs)

        raw_delta = (action * ACTION_SCALE).squeeze(0).tolist()
        clamped_delta, scale_factor = clamp_delta_to_velocity_limits(
            raw_delta, STEP_DURATION, CLAMP_LIMITS
        )

        # Convert the CLAMPED delta back into an equivalent action, since
        # the environment applies target = current_qpos + ACTION_SCALE *
        # action internally -- feeding the clamped delta back through
        # that same relationship keeps the simulator's own actuator
        # dynamics in the loop, rather than just clamping the number for
        # display without it actually affecting anything.
        clamped_action = torch.tensor(
            [[d / ACTION_SCALE for d in clamped_delta]], dtype=action.dtype
        )

        step_result = wrapped_env.step(clamped_action)
        # step() return structure not independently verified beyond "obs
        # is the first element" -- if this errors, print(step_result) once
        # to see the real structure and adjust the unpacking below.
        obs = step_result[0] if isinstance(step_result, tuple) else step_result

        joint_pos = arm.data.joint_pos[0, :6]  # first 6 = arm joints, drop gripper dims
        joint_pos_list = joint_pos.tolist()

        f.write(
            f"{step:4d}  |  {raw_delta}  |  {clamped_delta}  |  {scale_factor:.4f}  |  {joint_pos_list}\n"
        )

        if step % 20 == 0:
            flag = "  <-- clamped" if scale_factor < 1.0 else ""
            print(
                f"step {step}: scale={scale_factor:.3f}{flag}  joint_pos={[round(v, 4) for v in joint_pos_list]}"
            )

print("Done.")