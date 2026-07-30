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
            entropy_coef=0.02,
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
        max_iterations=4_000,  # simple task, shouldn't need locomotion-scale iters
    )


register_mjlab_task(
    task_id="Mjlab-UR3e-Reach",
    env_cfg=get_ur3e_reach_env_cfg(),
    play_env_cfg=get_ur3e_reach_env_cfg(play=True),
    rl_cfg=ur3e_reach_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id="Mjlab-UR3e-Reach-NoCollision",
    env_cfg=get_ur3e_reach_env_cfg(enable_collision_penalty=False),
    play_env_cfg=get_ur3e_reach_env_cfg(play=True, enable_collision_penalty=False),
    rl_cfg=ur3e_reach_ppo_runner_cfg(),
)