"""
PPO runner config + task registration for the UR3e reach task.
Network sizes are deliberately small since this is a low-dimensional
reach task (18-ish obs dims, 6 action dims) -- much smaller than a
locomotion policy needs.
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.registry import register_mjlab_task

from .ur3e_reach_env_cfg import get_ur3e_reach_env_cfg


def ur3e_reach_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(128, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(128, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.02,  # raised from 0.005 -- policy collapsed to std=0.04
                                # and got stuck in a self-colliding local optimum
                                # before finding a better strategy; this slows
                                # that collapse down. Watch Mean action std and
                                # position_error/at_goal in the next run to see
                                # if it actually helps rather than just delays.
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="ur3e_reach",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=2_000,  # simple task, shouldn't need locomotion-scale iters
    )


register_mjlab_task(
    task_id="Mjlab-UR3e-Reach",
    env_cfg=get_ur3e_reach_env_cfg(),
    play_env_cfg=get_ur3e_reach_env_cfg(play=True),
    rl_cfg=ur3e_reach_ppo_runner_cfg(),
)

# Temporary variant for visually validating the velocity clamp via the
# proven `uv run play` rendering pathway (see sim_rollout_to_target.py for
# the original headless validation this reproduces). Not for training --
# only env_cfg's use_velocity_clamp differs; same checkpoint still loads
# fine against it since the clamp only affects apply_actions(), not the
# action space's shape/dimension.
register_mjlab_task(
    task_id="Mjlab-UR3e-Reach-ClampedPlay",
    env_cfg=get_ur3e_reach_env_cfg(use_velocity_clamp=True),
    play_env_cfg=get_ur3e_reach_env_cfg(play=True, use_velocity_clamp=True),
    rl_cfg=ur3e_reach_ppo_runner_cfg(),
)