from pathlib import Path
import torch
import mujoco
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import RewardTermCfg
from mjlab.envs.mdp.actions.actions import JointVelocityActionCfg
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.envs.mdp.terminations import time_out
from mjlab.envs.mdp.observations import joint_pos_rel, joint_vel_rel
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers import SceneEntityCfg
from mjlab.scene.scene import SceneCfg, TerrainEntityCfg
from mjlab.sim.sim import SimulationCfg, MujocoCfg
from mjlab.entity.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl.config import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlModelCfg

_XML = Path(__file__).parent / "trailer.xml"


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_XML))


_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlActuatorCfg(target_names_expr=("tractor_x", "tractor_y", "tractor_yaw")),
  ),
)

_INIT = EntityCfg.InitialStateCfg(
  joint_pos={"tractor_x": 0.0, "tractor_y": 0.0, "tractor_yaw": 0.0, "hitch": 0.0},
  joint_vel={".*": 0.0},
)


def _get_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=_get_spec, articulation=_ARTICULATION, init_state=_INIT)


# Where we want the trailer parked: (x, y, z) in world coords, matching the
# parking_spot geom in the XML. z matches the height of the trailer_center
# site so the distance calc below isn't skewed by a vertical offset.
# Curriculum: start close to the target (~1.1m) so the policy can discover
# "drive there" at all. Push this back out toward the original (2.0, 2.0)
# corner once a close-range parking policy is working.
TARGET_POS = torch.tensor([0.8, 0.8, 0.08], dtype=torch.float32)
TARGET_YAW = 0.0
MAX_DIST = 1.5
SETTLE_RADIUS = 0.3

_all_joints_cfg = SceneEntityCfg("rig", joint_names=(".*",))
_rig_site_cfg = SceneEntityCfg("rig", site_names=("trailer_center",))
_tractor_yaw_cfg = SceneEntityCfg("rig", joint_names=("tractor_yaw",))
_hitch_cfg = SceneEntityCfg("rig", joint_names=("hitch",))
_tractor_xy_cfg = SceneEntityCfg("rig", joint_names=("tractor_x", "tractor_y"))

##
# Observations.
##


def trailer_to_target(env, rig_cfg) -> torch.Tensor:
  if not hasattr(env, "target_pos"):
    env.target_pos = TARGET_POS.to(env.device).expand(env.num_envs, -1)
  rig = env.scene[rig_cfg.name]
  trailer_pos = rig.data.site_pos_w[:, rig_cfg.site_ids][:, 0, :]
  return torch.nan_to_num(env.target_pos - trailer_pos, nan=0.0)


_actor_terms = {
  "joint_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": _all_joints_cfg}),
  "joint_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": _all_joints_cfg}),
  "trailer_to_target": ObservationTermCfg(func=trailer_to_target, params={"rig_cfg": _rig_site_cfg}),
}

observations = {
  "actor": ObservationGroupCfg(_actor_terms),
  "critic": ObservationGroupCfg({**_actor_terms}),
}

##
# Actions.
##

actions = {
  "drive": JointVelocityActionCfg(
    entity_name="rig",
    actuator_names=("tractor_x", "tractor_y", "tractor_yaw"),
    scale={"tractor_x": 1.0, "tractor_y": 1.0, "tractor_yaw": 1.57},
  ),
}

##
# Rewards.
##


def parking_reward(env, rig_cfg, tractor_yaw_cfg, hitch_cfg) -> torch.Tensor:
  # Multiplicative on purpose: orienting toward the spot without closing the
  # distance (or vice versa) earns ~nothing. Both must improve together,
  # otherwise a policy can farm the heading term by rotating in place.
  if not hasattr(env, "target_pos"):
    env.target_pos = TARGET_POS.to(env.device).expand(env.num_envs, -1)
  rig = env.scene[rig_cfg.name]
  trailer_pos = rig.data.site_pos_w[:, rig_cfg.site_ids][:, 0, :]
  distance = torch.norm(env.target_pos - trailer_pos, dim=-1)
  position_term = 1.0 - torch.clamp(distance / MAX_DIST, 0.0, 1.0)

  tractor_yaw = rig.data.joint_pos[:, tractor_yaw_cfg.joint_ids[0]]
  hitch = rig.data.joint_pos[:, hitch_cfg.joint_ids[0]]
  trailer_yaw = tractor_yaw + hitch
  heading_term = (1.0 + torch.cos(trailer_yaw - TARGET_YAW)) / 2.0

  return torch.nan_to_num(position_term * heading_term, nan=0.0)


def jackknife_penalty(env, hitch_cfg) -> torch.Tensor:
  rig = env.scene[hitch_cfg.name]
  hitch = rig.data.joint_pos[:, hitch_cfg.joint_ids[0]]
  return torch.nan_to_num(hitch**2, nan=0.0)


def parking_progress_reward(env, rig_cfg) -> torch.Tensor:
  # Dense, potential-based shaping: reward closing the distance *this step*,
  # rather than relying on the policy to stumble into a multi-second
  # directional drive via random exploration to earn the sparser
  # absolute-distance reward in `parking_reward`.
  if not hasattr(env, "target_pos"):
    env.target_pos = TARGET_POS.to(env.device).expand(env.num_envs, -1)
  rig = env.scene[rig_cfg.name]
  trailer_pos = rig.data.site_pos_w[:, rig_cfg.site_ids][:, 0, :]
  distance = torch.norm(env.target_pos - trailer_pos, dim=-1)

  if not hasattr(env, "prev_trailer_dist"):
    env.prev_trailer_dist = distance.clone()

  # episode_length_buf == 1 means this env was reset just before this step,
  # so prev_trailer_dist still holds the *previous episode's* final distance.
  # Comparing across that reset boundary would produce a spurious delta, so
  # treat the first post-reset step as zero progress instead.
  just_reset = env.episode_length_buf == 1
  progress = torch.where(just_reset, torch.zeros_like(distance), env.prev_trailer_dist - distance)
  env.prev_trailer_dist = distance.clone()
  return torch.nan_to_num(progress, nan=0.0)


def settle_bonus(env, rig_cfg, xy_cfg) -> torch.Tensor:
  # Reward being slow *and* close together, so stopping at the spot beats
  # driving through it. Without this, nothing distinguishes "arrived and
  # stayed" from "passed through on the way to somewhere else."
  if not hasattr(env, "target_pos"):
    env.target_pos = TARGET_POS.to(env.device).expand(env.num_envs, -1)
  rig = env.scene[rig_cfg.name]
  trailer_pos = rig.data.site_pos_w[:, rig_cfg.site_ids][:, 0, :]
  distance = torch.norm(env.target_pos - trailer_pos, dim=-1)
  close = (distance < SETTLE_RADIUS).float()
  speed = torch.norm(rig.data.joint_vel[:, xy_cfg.joint_ids], dim=-1)
  stillness = torch.exp(-speed)
  return torch.nan_to_num(close * stillness, nan=0.0)


rewards = {
  "parking": RewardTermCfg(
    func=parking_reward,
    weight=2.0,
    params={"rig_cfg": _rig_site_cfg, "tractor_yaw_cfg": _tractor_yaw_cfg, "hitch_cfg": _hitch_cfg},
  ),
  "progress": RewardTermCfg(func=parking_progress_reward, weight=20.0, params={"rig_cfg": _rig_site_cfg}),
  "settle": RewardTermCfg(func=settle_bonus, weight=3.0, params={"rig_cfg": _rig_site_cfg, "xy_cfg": _tractor_xy_cfg}),
  "jackknife": RewardTermCfg(func=jackknife_penalty, weight=-0.1, params={"hitch_cfg": _hitch_cfg}),
}

##
# Terminations.
##

terminations = {
  "time_out": TerminationTermCfg(func=time_out, time_out=True),
}

##
# Events.
##

_xy_cfg = SceneEntityCfg("rig", joint_names=("tractor_x", "tractor_y"))
_angles_cfg = SceneEntityCfg("rig", joint_names=("tractor_yaw", "hitch"))

events = {
  "reset_position": EventTermCfg(
    func=reset_joints_by_offset,
    mode="reset",
    params={"position_range": (-0.5, 0.5), "velocity_range": (-0.01, 0.01), "asset_cfg": _xy_cfg},
  ),
  "reset_angles": EventTermCfg(
    func=reset_joints_by_offset,
    mode="reset",
    params={"position_range": (-0.3, 0.3), "velocity_range": (-0.01, 0.01), "asset_cfg": _angles_cfg},
  ),
}


def trailer_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"rig": _get_cfg()},
      num_envs=1,
      env_spacing=6.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01)),
    decimation=5,
    episode_length_s=15.0,
  )


def trailer_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24,
    max_iterations=500,
    save_interval=50,
    experiment_name="trailer_park",
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(hidden_dims=(64, 64), activation="elu"),
    algorithm=RslRlPpoAlgorithmCfg(learning_rate=1e-3, num_mini_batches=4, entropy_coef=0.02),
  )


register_mjlab_task(
  task_id="Mjlab-Trailer-Park",
  env_cfg=trailer_env_cfg(),
  play_env_cfg=trailer_env_cfg(play=True),
  rl_cfg=trailer_runner_cfg(),
)
