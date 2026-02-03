# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for T3 Hexapod tripod gait locomotion training.

Single motor per leg version - only Leg_joint is controlled.
Calf_joint is fixed at a constant angle.
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
from isaaclab_assets.robots.t3_hexapod import T3_HEXAPOD_CFG  # isort: skip


@configclass
class T3TripodObservationsCfg:
    """Observation specifications for T3 tripod gait."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group - simplified for 6 motor control."""

        # Base state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        # Velocity commands
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # Joint state - all 12 joints but only 6 are controlled
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        # Previous actions
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class T3TripodRewardsCfg:
    """Reward terms for T3 tripod gait training."""

    # -- Task rewards: velocity tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # -- Tripod gait rewards
    tripod_phase_sync = RewTerm(
        func=t3_rewards.tripod_gait_phase_sync,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    tripod_anti_phase = RewTerm(
        func=t3_rewards.tripod_gait_anti_phase,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    gait_rhythm = RewTerm(
        func=t3_rewards.tripod_gait_rhythm,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # -- Stability rewards
    stability = RewTerm(
        func=t3_rewards.stability_reward,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # -- Penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # -- Orientation penalty (keep body flat)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # -- Joint limit penalty
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)


@configclass
class T3TripodActionsCfg:
    """Action specifications for T3 tripod gait - only Leg joints."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_Leg_joint"],  # Only control Leg joints (6 motors)
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class T3TripodTerminationsCfg:
    """Termination terms for T3 tripod gait."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if robot falls over (base height too low)
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.05, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class T3HexapodRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for T3 Hexapod tripod gait training on rough terrain."""

    # Override with T3 specific configs
    observations: T3TripodObservationsCfg = T3TripodObservationsCfg()
    actions: T3TripodActionsCfg = T3TripodActionsCfg()
    rewards: T3TripodRewardsCfg = T3TripodRewardsCfg()
    terminations: T3TripodTerminationsCfg = T3TripodTerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Switch robot to T3 Hexapod (single motor version)
        self.scene.robot = T3_HEXAPOD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Disable contact forces sensor (not needed)
        self.scene.contact_forces = None

        # Update height scanner to use T3's base_link
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

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

        # Adjust simulation parameters for hexapod
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 20.0


@configclass
class T3HexapodRoughEnvCfg_PLAY(T3HexapodRoughEnvCfg):
    """Configuration for playing/testing T3 Hexapod on rough terrain."""

    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Spawn randomly
        self.scene.terrain.max_init_terrain_level = None

        # Reduce terrains
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Disable noise for play
        self.observations.policy.enable_corruption = False

        # Remove pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
