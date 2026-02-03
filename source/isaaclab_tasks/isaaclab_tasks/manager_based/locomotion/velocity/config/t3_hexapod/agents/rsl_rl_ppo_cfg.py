# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for T3 Hexapod tripod gait locomotion."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class T3HexapodRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for T3 Hexapod tripod gait on rough terrain."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "t3_hexapod_tripod_rough"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 128],  # Smaller network for 6 motors
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Higher entropy for exploration
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class T3HexapodFlatPPORunnerCfg(T3HexapodRoughPPORunnerCfg):
    """PPO runner configuration for T3 Hexapod tripod gait on flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1000
        self.experiment_name = "t3_hexapod_tripod_flat"

        # Smaller network for simpler flat terrain task
        self.policy.actor_hidden_dims = [128, 128, 64]
        self.policy.critic_hidden_dims = [128, 128, 64]
