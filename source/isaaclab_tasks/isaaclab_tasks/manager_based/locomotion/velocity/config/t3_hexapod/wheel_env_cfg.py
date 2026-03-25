# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for T3 Hexapod wheel-paddle locomotion on flat terrain.

Wheel-Paddle Mode:
- All 6 Leg_joints rotate continuously in one direction (like wheels)
- Velocity control: stiffness=0, damping provides torque
- RL learns to modulate rotation speed per leg for forward motion
- Tripod pattern: Group A and Group B alternate fast/slow spinning

Tripod Groups:
- Group A: FL (Front-Left), MR (Middle-Right), RL (Rear-Left)
- Group B: FR (Front-Right), ML (Middle-Left), RR (Rear-Right)
"""

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# Import T3 specific rewards
from .mdp import rewards as t3_rewards

##
# Pre-defined configs
##
from isaaclab_assets.robots.t3_hexapod import T3_HEXAPOD_WHEEL_CFG  # isort: skip


@configclass
class T3WheelObservationsCfg:
    """Observation specifications for T3 wheel-paddle mode."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Base state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        # Velocity commands
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # Joint state - only Leg joints matter for wheel mode
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        # Previous actions
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class T3WheelRewardsCfg:
    """Reward terms for T3 wheel-paddle locomotion."""

    # -- Task rewards: velocity tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # -- Wheel-paddle specific rewards
    # Reward for spinning in the correct (forward) direction
    wheel_direction = RewTerm(
        func=t3_rewards.wheel_spin_direction,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Reward for tripod-patterned spinning (alternating fast/slow between groups)
    wheel_tripod = RewTerm(
        func=t3_rewards.wheel_tripod_pattern,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Penalty for sideways drift
    lateral_drift = RewTerm(
        func=t3_rewards.wheel_lateral_drift_penalty,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # -- Stability rewards
    stability = RewTerm(
        func=t3_rewards.stability_reward,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # -- Standard penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # Keep body flat
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)


@configclass
class T3WheelActionsCfg:
    """Action specifications for T3 wheel-paddle mode - velocity control."""

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*_Leg_joint"],  # Control all 6 Leg joints
        scale=5.0,   # Actions in [-1,1] map to [-5,5] rad/s
        offset=0.0,
    )


@configclass
class T3WheelTerminationsCfg:
    """Termination terms for T3 wheel-paddle mode."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if robot falls (base too low)
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.05, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class T3HexapodWheelEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for T3 Hexapod wheel-paddle locomotion on flat terrain."""

    # Override with wheel-specific configs
    observations: T3WheelObservationsCfg = T3WheelObservationsCfg()
    actions: T3WheelActionsCfg = T3WheelActionsCfg()
    rewards: T3WheelRewardsCfg = T3WheelRewardsCfg()
    terminations: T3WheelTerminationsCfg = T3WheelTerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Switch robot to T3 Hexapod wheel-paddle config
        self.scene.robot = T3_HEXAPOD_WHEEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # No height scan on flat terrain
        self.scene.height_scanner = None

        # No contact forces sensor
        self.scene.contact_forces = None

        # Update height scanner prim if it existed
        # (already None, so no update needed)

        # Update event randomization for T3's base_link
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names="base_link"
        )
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names="base_link"
        )
        self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names="base_link"
        )

        # No terrain curriculum on flat
        self.curriculum.terrain_levels = None

        # Simulation parameters
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 20.0


@configclass
class T3HexapodWheelEnvCfg_PLAY(T3HexapodWheelEnvCfg):
    """Configuration for playing/testing T3 Hexapod wheel-paddle mode."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable noise for play
        self.observations.policy.enable_corruption = False

        # Remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
