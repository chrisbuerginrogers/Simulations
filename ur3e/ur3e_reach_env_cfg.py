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

sim/decimation: decimation=10, timestep=0.01 -> 0.1s/step, matching
step_duration in run_policy_loop (real deployment). Earlier drafts used
decimation=4/timestep=0.005 (0.02s/step, copied from mjlab's reference
lift_cube task for training speed) and separately decimation=20/timestep=
0.005 (0.1s/step at higher substep resolution) -- this file settled on
the cheaper 0.1s/step combo. NOTE: if you tune send_ahead_fraction in
run_policy_loop away from 0.8, the EFFECTIVE re-decision cadence under
streaming is send_ahead_fraction * 0.1s, not the full 0.1s -- reconsider
decimation/scale together if that changes.
"""

from pathlib import Path
import os
from dataclasses import dataclass

import mujoco
import torch

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.envs.mdp.actions.actions import RelativeJointPositionAction
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

from .commands import ReachPositionCommandCfg

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
#
# history_length=10 matches decimation=10 (one history slot per substep),
# so no substep within a single action's window goes unchecked for a
# contact spike -- was 4, left over from the earlier decimation=4 config.
_self_collision_sensor = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base", entity="ur3e"),
    secondary=ContactMatch(mode="subtree", pattern="base", entity="ur3e"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=10,
)
_table_collision_sensor = ContactSensorCfg(
    name="table_collision",
    primary=ContactMatch(mode="subtree", pattern="base", entity="ur3e"),
    secondary=ContactMatch(mode="body", pattern="terrain"),  # mjlab's ground-plane body name
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=10,
)

# ---------------------------------------------------------------------------
# Commands -- generates the reach target, handles env-origin offsetting,
# and renders a debug sphere in the viewer via _debug_vis_impl.
# ---------------------------------------------------------------------------
def _build_commands() -> dict:
    """
    Normally samples random targets from the reachable hemisphere. Set the
    UR3E_TARGET_POS environment variable (format: "x,y,z", meters, world
    frame) before running play/a rollout script to override this with one
    fixed point instead -- used by sim_rollout_to_target.py to force a
    specific, repeatable target for direct sim/real comparison.
    """
    target_pos_env = os.environ.get("UR3E_TARGET_POS")
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

# Mirrors REAL_JOINT_VEL_LIMITS / SAFETY_FACTOR from the deployment script --
# keeping these in sync means training actually optimizes for the regime
# you'll clamp to at inference time, instead of the clamp being a surprise
# the policy never saw during training.
_JOINT_VEL_LIMITS_RAD_S = (3.14159, 3.14159, 3.14159, 6.28319, 6.28319, 6.28319)
_VEL_SAFETY_FACTOR = 0.25
_JOINT_VEL_TARGET_LIMITS = tuple(v * _VEL_SAFETY_FACTOR for v in _JOINT_VEL_LIMITS_RAD_S)


def joint_vel_limit_penalty(
    env, asset_cfg: SceneEntityCfg, limits: tuple
) -> torch.Tensor:
    """
    Hinge penalty per joint: zero cost below `limits[i]` (rad/s), quadratic
    cost above it. Unlike joint_vel_l2, this is per-joint (respecting that
    wrists have 2x the velocity budget of shoulder/lift/elbow) and doesn't
    compete with reach_distance/success_bonus below the threshold, so it
    shouldn't fight reaching speed the way a flat L2 does.
    """
    arm = env.scene[asset_cfg.name]
    joint_vel = arm.data.joint_vel[:, asset_cfg.joint_ids]  # [B, 6]
    limits_t = torch.tensor(limits, device=joint_vel.device, dtype=joint_vel.dtype)
    excess = torch.clamp(torch.abs(joint_vel) - limits_t, min=0.0)
    return torch.sum(torch.square(excess), dim=-1)


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
@dataclass(kw_only=True)
class ClampedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """
    Same as RelativeJointPositionActionCfg (target = current_pos + action *
    scale), but scales the WHOLE delta vector by one shared factor -- based
    on whichever joint implies the largest velocity relative to its own
    limit -- rather than clipping each joint independently. This preserves
    the coordinated "shape" of the multi-joint motion, matching exactly
    the clamp validated in sim_rollout_to_target.py, rather than
    RelativeJointPositionActionCfg's own built-in `clip` field (which does
    a simple independent per-joint clamp, a different and untested
    behavior).

    velocity_limits order must match actuator_names order.
    """

    step_duration: float = 0.1
    velocity_limits: tuple[float, ...] = (
        3.14159 * 0.25,
        3.14159 * 0.25,
        3.14159 * 0.25,
        6.28319 * 0.25,
        6.28319 * 0.25,
        6.28319 * 0.25,
    )

    def build(self, env: ManagerBasedRlEnvCfg) -> "ClampedRelativeJointPositionAction":
        return ClampedRelativeJointPositionAction(self, env)


class ClampedRelativeJointPositionAction(RelativeJointPositionAction):
    cfg: ClampedRelativeJointPositionActionCfg

    def apply_actions(self) -> None:
        current_pos = self._entity.data.joint_pos[:, self._target_ids]
        delta = self._processed_actions  # action * scale, before adding to current_pos

        limits = torch.tensor(
            self.cfg.velocity_limits, device=self.device, dtype=delta.dtype
        )
        implied_vel = torch.abs(delta) / self.cfg.step_duration
        ratios = implied_vel / limits
        max_ratio = ratios.max(dim=-1, keepdim=True).values
        # Only ever scale DOWN (never amplify a delta that's already within limits).
        scale_factor = torch.clamp(1.0 / max_ratio, max=1.0)
        clamped_delta = delta * scale_factor

        target = current_pos + clamped_delta
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


def _build_actions(use_velocity_clamp: bool) -> dict:
    if use_velocity_clamp:
        return {
            "arm_joints": ClampedRelativeJointPositionActionCfg(
                entity_name="ur3e",
                actuator_names=_ARM_JOINT_NAMES,
                scale=0.08,
            ),
        }
    return {
        "arm_joints": RelativeJointPositionActionCfg(
            entity_name="ur3e",
            actuator_names=_ARM_JOINT_NAMES,
            scale=0.08,  # rad, max delta per policy step. Matches decimation=10 *
                         # timestep=0.01 = 0.1s/step: 0.08/0.1 = 0.8 rad/s per unit
                         # raw action, which lands almost exactly at the 25%-safety
                         # -factor limit for shoulder/lift/elbow (0.25*pi=0.785).
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
        "joint_vel_limit": RewardTermCfg(
            func=joint_vel_limit_penalty,
            weight=-1.5,  # matching ee_speed_limit's rough order of magnitude as a
                          # starting point -- watch Episode_Reward/joint_vel_limit
                          # once training resumes and adjust if it's dominating
                          # (competing with reach_distance) or having no visible
                          # effect (policy still needs frequent deployment-side
                          # clamping despite this being enabled).
            params={"asset_cfg": arm_cfg, "limits": _JOINT_VEL_TARGET_LIMITS},
        ),
        "joint_vel": RewardTermCfg(
            func=mdp_rewards.joint_vel_l2,  # built-in mjlab term
            weight=-0.001,  # small, general smoothness regularizer -- kept
                            # alongside joint_vel_limit since they serve
                            # different purposes (general smoothness vs. hard
                            # per-joint limit avoidance), not redundant.
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
    play: bool = False,
    enable_collision_penalty: bool = True,
    use_velocity_clamp: bool = False,
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

    use_velocity_clamp=False (default): standard RelativeJointPositionAction,
    matches what checkpoints are actually trained against -- use this for
    normal training/play.
    use_velocity_clamp=True: swaps in ClampedRelativeJointPositionAction,
    reproducing the uniform-scaling velocity clamp validated in
    sim_rollout_to_target.py, so you can WATCH it live via `uv run play`.
    Intended for visual validation, not training, and not what a checkpoint
    was trained against.
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
        actions=_build_actions(use_velocity_clamp),
        events=events,
        rewards=_build_rewards(enable_collision_penalty),
        terminations=terminations,
        sim=SimulationCfg(
            # integrator matches MujocoCfg's own default ("implicitfast") -- stated
            # explicitly here because this is the one place it actually takes
            # effect (see the comment in ur3e_gripper.xml for why it's not set
            # there).
            mujoco=MujocoCfg(timestep=0.01, integrator="implicitfast"),
        ),
        decimation=10,  # 10 * 0.01 = 0.1s per policy step, matching step_duration
                        # in run_policy_loop (real deployment).
        episode_length_s=22.0,  # raised from 10.0 to fit ~2 resamples per episode
                                 # given the new 9-12s resampling window (was 4-6s)
    )