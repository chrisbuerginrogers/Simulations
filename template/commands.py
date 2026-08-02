"""
Generic "reach a 3D target position" command term for any fixed- or
floating-base mjlab entity that has a named site to track (e.g. an
end-effector). Copied from the UR3e reach task's commands.py -- this
implementation was already entity/site-name-agnostic, just with UR3e-specific
wording in the docstrings, which has been generalized here.

Only needed if you're building a reach-style task (an entity trying to move
some site to a target position). If your task doesn't need that, delete this
file and the corresponding import in your generated custom_task.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class ReachPositionCommand(CommandTerm):
    """3D target position for some entity's named site to reach."""

    cfg: ReachPositionCommandCfg

    def __init__(self, cfg: ReachPositionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.asset: Entity = env.scene[cfg.entity_name]
        self._site_id = self.asset.site_names.index(cfg.site_name)

        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.target_pos

    def _ee_pos_w(self) -> torch.Tensor:
        return self.asset.data.site_pos_w[:, self._site_id]

    def _update_metrics(self) -> None:
        ee_pos_w = self._ee_pos_w()
        position_error = torch.norm(self.target_pos - ee_pos_w, dim=-1)
        at_goal = (position_error < self.cfg.success_threshold).float()
        self.metrics["position_error"] = position_error
        self.metrics["at_goal"] = at_goal

    def compute_success(self) -> torch.Tensor:
        return self.metrics["position_error"] < self.cfg.success_threshold

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)

        if self.cfg.difficulty == "fixed":
            target_pos = torch.tensor(
                self.cfg.fixed_target, device=self.device, dtype=torch.float32
            ).expand(n, 3)
        elif self.cfg.difficulty == "dynamic":
            r = self.cfg.target_position_range
            lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
            upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
            target_pos = sample_uniform(lower, upper, (n, 3), device=self.device)
        else:
            assert self.cfg.difficulty == "hemisphere"
            target_pos = self._sample_hemisphere(n)

        # NOTE: if your entity is fixed-base (rigidly attached to worldbody,
        # no freejoint), its site_pos_w is reported relative to its own fixed
        # base, not shifted per parallel-env tile -- do NOT add env_origins
        # here in that case. If your entity instead has a freejoint (it can
        # move/fall), you likely DO need to add env.scene.env_origins[env_ids]
        # to target_pos so targets land in each tile's own local frame.
        self.target_pos[env_ids] = target_pos

    def _sample_hemisphere(self, n: int) -> torch.Tensor:
        """
        Uniform-in-volume rejection sampling within a sphere (cfg.hemisphere
        .center / .radius), restricted to world-frame z >= cfg.hemisphere
        .min_z.

        Uniform-in-volume (not uniform-in-radius) via r ~ radius * U^(1/3),
        U~Uniform(0,1) -- sampling r uniformly instead would bias points
        toward the center, since volume grows with r^3.
        """
        h = self.cfg.hemisphere
        center = torch.tensor(h.center, device=self.device, dtype=torch.float32)
        points = torch.zeros(n, 3, device=self.device)
        remaining = torch.arange(n, device=self.device)

        while remaining.numel() > 0:
            m = remaining.numel()
            direction = torch.randn(m, 3, device=self.device)
            direction = direction / torch.norm(direction, dim=-1, keepdim=True)
            radii = h.radius * torch.rand(m, device=self.device).pow(1.0 / 3.0)
            candidates = center + direction * radii.unsqueeze(-1)

            valid = candidates[:, 2] >= h.min_z
            points[remaining[valid]] = candidates[valid]
            remaining = remaining[~valid]

        return points

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        for batch in env_indices:
            target_pos = self.target_pos[batch].cpu().numpy()
            visualizer.add_sphere(
                center=target_pos,
                radius=0.02,
                color=self.cfg.viz.target_color,
                label=f"reach_target_{batch}",
            )


@dataclass(kw_only=True)
class ReachPositionCommandCfg(CommandTermCfg):
    entity_name: str
    site_name: str = "end_effector"
    success_threshold: float = 0.03
    difficulty: Literal["fixed", "dynamic", "hemisphere"] = "fixed"
    fixed_target: tuple[float, float, float] = (0.3, 0.0, 0.3)

    @dataclass
    class TargetPositionRangeCfg:
        """Only used in dynamic mode."""

        x: tuple[float, float] = (0.2, 0.4)
        y: tuple[float, float] = (-0.2, 0.2)
        z: tuple[float, float] = (0.15, 0.4)

    target_position_range: TargetPositionRangeCfg = field(
        default_factory=TargetPositionRangeCfg
    )

    @dataclass
    class HemisphereRangeCfg:
        """Only used in hemisphere mode. Tune center/radius/min_z to match
        your own robot's reachable workspace -- these defaults are arbitrary
        placeholders, not tuned for any particular robot."""

        center: tuple[float, float, float] = (0.0, 0.0, 0.3)
        radius: float = 0.7
        min_z: float = 0.05

    hemisphere: HemisphereRangeCfg = field(default_factory=HemisphereRangeCfg)

    @dataclass
    class VizCfg:
        target_color: tuple[float, float, float, float] = (0.1, 0.9, 0.2, 0.6)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> ReachPositionCommand:
        return ReachPositionCommand(self, env)
