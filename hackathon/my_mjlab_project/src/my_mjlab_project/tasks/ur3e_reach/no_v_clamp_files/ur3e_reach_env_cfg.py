"""
UR3e reach task, restructured to match mjlab's own yam lift_cube_env_cfg.py
patterns (the shipped reference manipulation task), simplified down to pure
reaching -- no object, no camera, no domain randomization.

Action space:   6D relative joint position (RelativeJointPositionAction) --
                 the joint-delta scheme: target = current_qpos + action*scale
Observations:   joint_pos_rel, joint_vel_rel, the commanded target position,
                 and the vector from end-effector to that target.
Reward:         negative distance from end-effector site to target,
                 plus a small action-rate penalty.
Termination:    time out (truncation).

sim/decimation values (timestep=0.005, decimation=4) are copied directly
from mjlab's reference yam lift_cube_env_cfg.py -- this is ~47x less
physics compute per policy step than our earlier 0.002/75 combo, which
was tuned to match the real robot's 0.15s trajectory-goal duration rather
than training speed. Once this trains well, revisit decimation/timestep
to re-match real-robot step duration before sim2real deployment.
"""

from pathlib import Path
import os

import mujoco
import torch

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.envs.mdp.observations import joint_pos_rel, joint_vel_rel
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.envs.mdp.terminations import time_out
from mjlab.envs.mdp import rewards as mdp_rewards
from mjlab.managers import (
    ObservationTermCfg,
    ObservationGroupCfg,
    RewardTermCfg,
    TerminationTermCfg,
    EventTermCfg,
    SceneEntityCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.sensor import ContactSensor, ContactSensorCfg
from mjlab.sensor.contact_sensor import ContactMatch

from ..commands import ReachPositionCommandCfg

_UR3E_XML = Path(__file__).parent / "ur3e_gripper.xml"

_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# A reasonable "elbow up" home pose -- adjust to whatever start config you want.
_HOME_QPOS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.57,
    "elbow_joint": 0.0,
    "wrist_1_joint": -1.57,
    "wrist_2_joint": 0.0,
    "wrist_3_joint": 0.0,
}


def _get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(_UR3E_XML))
    # meshdir lives under spec.compiler, not spec directly.
    spec.compiler.meshdir = str(_UR3E_XML.parent / "assets")
    return spec


# Reuse the position actuators exactly as defined in the XML (kp/kv/forcerange
# per joint size class) -- no override needed here since we already tuned
# those to match real UR3e torque limits.
_UR3E_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(XmlActuatorCfg(target_names_expr=tuple(_ARM_JOINT_NAMES)),),
)

_UR3E_INIT = EntityCfg.InitialStateCfg(
    joint_pos=_HOME_QPOS,
    joint_vel={".*": 0.0},
)


def _get_ur3e_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_spec,
        articulation=_UR3E_ARTICULATION,
        init_state=_UR3E_INIT,
    )


# Separate SceneEntityCfgs for joints vs. the end-effector site, matching
# the pattern in mjlab's own manipulation/mdp/observations.py (they use
# `asset_cfg.site_ids` resolved once at init, rather than re-resolving a
# site name -> index lookup on every call like our first draft did).
arm_cfg = SceneEntityCfg("ur3e", joint_names=_ARM_JOINT_NAMES)
ee_cfg = SceneEntityCfg("ur3e", site_names=("end_effector",))

# ---------------------------------------------------------------------------
# Contact sensors -- self-collision and arm-vs-table detection.
#
# Adapted directly from mjlab's own tasks/velocity/config/go1/env_cfgs.py,
# which uses the same ContactSensor mechanism for a quadruped's self-collision
# and foot-ground contact. "subtree" mode on the "base" body captures the
# entire kinematic chain (base through fingers) as one primary/secondary set.
#
# Note: our XML's <contact><exclude> entries for adjacent body pairs (added
# when we enabled collision) also apply here -- excluded pairs never generate
# a contact in the first place, so these sensors only see genuine non-adjacent
# self-contact, not the expected touching at each joint.
_self_collision_sensor = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base", entity="ur3e"),
    secondary=ContactMatch(mode="subtree", pattern="base", entity="ur3e"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,  # matches decimation=4 -- catches brief mid-step contacts
)
_table_collision_sensor = ContactSensorCfg(
    name="table_collision",
    primary=ContactMatch(mode="subtree", pattern="base", entity="ur3e"),
    secondary=ContactMatch(mode="body", pattern="terrain"),  # mjlab's ground-plane body name
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
)

# ---------------------------------------------------------------------------
# Commands -- generates the reach target, handles env-origin offsetting,
# and renders a debug sphere in the viewer via _debug_vis_impl.
# ---------------------------------------------------------------------------
def _build_commands() -> dict:
    """
    Normally samples random targets from the reachable hemisphere. Set the
    UR3E_TARGET_POS environment variable (format: "x,y,z", meters, world
    frame) before running play to override this with one fixed point
    instead -- useful for testing the trained policy against a specific
    location rather than random targets every reset.
    """
    target_pos_env = os.environ.get("UR3E_TARGET_POS")
    print(f"[ur3e_reach] UR3E_TARGET_POS = {target_pos_env!r}")  # temp debug
    if target_pos_env:
        try:
            x, y, z = (float(v) for v in target_pos_env.split(","))
        except ValueError:
            raise ValueError(
                f'UR3E_TARGET_POS="{target_pos_env}" is not valid -- expected '
                'three comma-separated numbers, e.g. "0.3,0.0,0.3"'
            )
        reach_target_cfg = ReachPositionCommandCfg(
            entity_name="ur3e",
            site_name="end_effector",
            difficulty="fixed",
            fixed_target=(x, y, z),
            success_threshold=0.03,
            resampling_time_range=(1.0e6, 1.0e6),  # effectively never resample
            debug_vis=True,
        )
    else:
        reach_target_cfg = ReachPositionCommandCfg(
            entity_name="ur3e",
            site_name="end_effector",
            difficulty="hemisphere",
            hemisphere=ReachPositionCommandCfg.HemisphereRangeCfg(
                center=(0.0, 0.0, 0.09475),
                radius=0.70,
                min_z=0.05,
            ),
            success_threshold=0.03,
            resampling_time_range=(9.0, 12.0),  # widened for the hemisphere region;
                                                 # avg target-to-target distance is
                                                 # now 0.635m (was 0.254m for the old
                                                 # box), needing ~5.3s avg at the
                                                 # 12cm/s speed cap alone
            debug_vis=True,
        )
    return {"reach_target": reach_target_cfg}


# ---------------------------------------------------------------------------
# Custom observation / reward: end-effector distance to commanded target
# ---------------------------------------------------------------------------
def ee_to_target(env, asset_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Vector from end-effector site to commanded target, world frame."""
    asset = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    target_pos_w = env.command_manager.get_command(command_name)
    return target_pos_w - ee_pos_w


def reach_distance_reward(
    env, asset_cfg: SceneEntityCfg, command_name: str
) -> torch.Tensor:
    """Negative distance from end-effector to commanded target."""
    diff = ee_to_target(env, asset_cfg, command_name)
    return -torch.norm(diff, dim=-1)


def success_bonus(
    env, asset_cfg: SceneEntityCfg, command_name: str, threshold: float = 0.03
) -> torch.Tensor:
    """
    +1 per step while within `threshold` of the target, 0 otherwise.

    reach_distance_reward alone has the same gradient far away and up
    close -- nothing specifically rewards closing the last centimeter.
    This gives a real, separate incentive to actually cross the success
    threshold (and stay there), rather than just generally reducing
    distance without ever quite arriving.
    """
    diff = ee_to_target(env, asset_cfg, command_name)
    dist = torch.norm(diff, dim=-1)
    return (dist < threshold).float()


def settle_penalty(
    env,
    ee_asset_cfg: SceneEntityCfg,
    arm_asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float = 0.03,
) -> torch.Tensor:
    """
    Penalizes joint velocity, but ONLY once within `threshold` of the
    target -- while still far away, this is zero, so it doesn't discourage
    necessary motion to get there.

    Added specifically because the end_effector site sits almost exactly
    on wrist_3's own rotation axis, so spinning wrist_3 (deliberately left
    unlimited to match real hardware) barely moves the site's *position*
    at all -- reach_distance_reward doesn't notice it, and joint_vel_l2's
    weight is far too small to discourage it on its own. This term
    specifically punishes continued motion once the position task is
    already satisfied, rather than motion in general.
    """
    diff = ee_to_target(env, ee_asset_cfg, command_name)
    dist = torch.norm(diff, dim=-1)
    close = (dist < threshold).float()
    arm = env.scene[arm_asset_cfg.name]
    joint_vel = arm.data.joint_vel[:, arm_asset_cfg.joint_ids]
    vel_cost = torch.sum(torch.square(joint_vel), dim=-1)
    return close * vel_cost


def ee_speed_limit_penalty(
    env, asset_cfg: SceneEntityCfg, max_speed: float = 0.12
) -> torch.Tensor:
    """
    Penalizes end-effector linear speed only when it exceeds max_speed
    (m/s) -- a hinge, not a flat L2 term, since we want to freely allow
    any speed up to the limit and only discourage exceeding it, not
    encourage the arm to move as slowly as possible in general.
    Squared excess (not linear) so larger violations are penalized
    disproportionately more, encouraging margin under the limit rather
    than just barely staying under it.

    Returns a NON-NEGATIVE cost (matching collision_cost/joint_vel_l2/
    action_rate_l2's convention) -- pair with a NEGATIVE weight in
    RewardTermCfg. Returning a negative value here AND using a negative
    weight double-negates into a reward for violating the limit, which
    is exactly the bug that happened here originally.
    """
    asset = env.scene[asset_cfg.name]
    ee_vel_w = asset.data.site_lin_vel_w[:, asset_cfg.site_ids].squeeze(1)
    speed = torch.norm(ee_vel_w, dim=-1)
    excess = torch.clamp(speed - max_speed, min=0.0)
    return torch.square(excess)


def collision_cost(
    env, sensor_name: str, force_threshold: float = 10.0
) -> torch.Tensor:
    """
    Penalize contacts on the given ContactSensor. Adapted directly from
    mjlab's own self_collision_cost (tasks/velocity/mdp/rewards.py).

    When the sensor has force_history (history_length > 0, as configured
    above), counts substeps where any contact force exceeds
    force_threshold -- catches brief contacts that resolve mid-step, which
    a single end-of-step check could miss. Falls back to the instantaneous
    found count if no history is available.

    force_threshold=10.0 (Newtons) is mjlab's own default, tuned for a much
    heavier quadruped's legs -- this is an unverified starting point for
    our lighter UR3e, not a confirmed-correct number. Watch the logged
    metric and adjust down if real contact forces during training are
    much smaller than this and never trip the penalty.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
        return hit.sum(dim=-1).float()
    assert data.found is not None
    return data.found.sum(dim=-1).float()


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
actor_terms = {
    "joint_pos": ObservationTermCfg(
        func=joint_pos_rel, params={"asset_cfg": arm_cfg}
    ),
    "joint_vel": ObservationTermCfg(
        func=joint_vel_rel, params={"asset_cfg": arm_cfg}
    ),
    "target_pos": ObservationTermCfg(
        func=mdp.generated_commands, params={"command_name": "reach_target"}
    ),
    "ee_to_target": ObservationTermCfg(
        func=ee_to_target,
        params={"asset_cfg": ee_cfg, "command_name": "reach_target"},
    ),
}

observations = {
    "actor": ObservationGroupCfg(actor_terms),
    "critic": ObservationGroupCfg({**actor_terms}),
}

# ---------------------------------------------------------------------------
# Actions -- 6D relative joint position, matching the joint-delta scheme
# ---------------------------------------------------------------------------
actions = {
    "arm_joints": RelativeJointPositionActionCfg(
        entity_name="ur3e",
        actuator_names=_ARM_JOINT_NAMES,
        scale=0.08,  # rad, max delta per policy step -- tune alongside decimation
    ),
}

# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
def _build_rewards(enable_collision_penalty: bool) -> dict:
    rewards = {
        "reach_distance": RewardTermCfg(
            func=reach_distance_reward,
            weight=1.0,
            params={"asset_cfg": ee_cfg, "command_name": "reach_target"},
        ),
        "success_bonus": RewardTermCfg(
            func=success_bonus,
            weight=1.0,  # starting point -- tune based on whether at_goal improves
            params={"asset_cfg": ee_cfg, "command_name": "reach_target", "threshold": 0.03},
        ),
        "settle_penalty": RewardTermCfg(
            func=settle_penalty,
            weight=-0.5,  # starting point -- tune based on whether spinning stops
            params={
                "ee_asset_cfg": ee_cfg,
                "arm_asset_cfg": arm_cfg,
                "command_name": "reach_target",
                "threshold": 0.03,
            },
        ),
        "action_rate": RewardTermCfg(
            func=mdp_rewards.action_rate_l2,
            weight=-0.01,
            params={},
        ),
        "ee_speed_limit": RewardTermCfg(
            func=ee_speed_limit_penalty,
            weight=-2.0,  # conservative starting point -- see note below, tune from logs
            params={"asset_cfg": ee_cfg, "max_speed": 0.12},
        ),
        "joint_vel": RewardTermCfg(
            func=mdp_rewards.joint_vel_l2,  # built-in mjlab term
            weight=-0.001,  # small, general smoothness regularizer
            params={"asset_cfg": arm_cfg},
        ),
    }
    if enable_collision_penalty:
        rewards["self_collision"] = RewardTermCfg(
            func=collision_cost,
            weight=-1.0,  # starting point -- tune from logged metric, see note in collision_cost
            params={"sensor_name": "self_collision", "force_threshold": 10.0},
        )
        rewards["table_collision"] = RewardTermCfg(
            func=collision_cost,
            weight=-1.0,
            params={"sensor_name": "table_collision", "force_threshold": 10.0},
        )
    return rewards

# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------
terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
}

# ---------------------------------------------------------------------------
# Events -- reset arm joints around the home pose
# ---------------------------------------------------------------------------
events = {
    "reset_arm": EventTermCfg(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.3, 0.3),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": arm_cfg,
        },
    ),
}


def get_ur3e_reach_env_cfg(
    play: bool = False, enable_collision_penalty: bool = True
) -> ManagerBasedRlEnvCfg:
    """
    enable_collision_penalty=True (default): self/table collision sensors
    attached, self_collision/table_collision reward terms included.
    enable_collision_penalty=False: no sensors attached at all (not just
    zero-weighted -- avoids the sensor overhead too), rewards dict omits
    both terms entirely. Collision PHYSICS in the XML is unaffected either
    way (contype/conaffinity stay as configured there) -- this only
    controls whether collisions are measured/penalized in training, not
    whether the arm can physically interpenetrate itself/the table.
    """
    sensors = (
        (_self_collision_sensor, _table_collision_sensor)
        if enable_collision_penalty
        else ()
    )
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"ur3e": _get_ur3e_cfg()},
            sensors=sensors,
            num_envs=1 if play else 512,  # override with --num-envs on the CLI
            env_spacing=2.0,
        ),
        commands=_build_commands(),
        observations=observations,
        actions=actions,
        events=events,
        rewards=_build_rewards(enable_collision_penalty),
        terminations=terminations,
        sim=SimulationCfg(
            # Matches mjlab's own reference manipulation task
            # (tasks/manipulation/lift_cube_env_cfg.py) for training speed.
            mujoco=MujocoCfg(timestep=0.005),
        ),
        decimation=4,  # 4 * 0.005 = 0.02s per policy step (50Hz)
        episode_length_s=22.0,  # raised from 10.0 to fit ~2 resamples per episode
                                 # given the new 9-12s resampling window (was 4-6s)
    )