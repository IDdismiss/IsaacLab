# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for T3 Hexapod locomotion on rough terrain."""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets.robots.t3_hexapod import T3_HEXAPOD_CFG  # isort: skip


@configclass
class T3HexapodRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for T3 Hexapod on rough terrain."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Switch robot to T3 Hexapod
        self.scene.robot = T3_HEXAPOD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Update height scanner to use T3's base_link
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # ============ 暂时关闭接触传感器相关功能（调试用）============
        # 关闭接触传感器
        self.scene.contact_forces = None

        # 关闭依赖接触传感器的奖励
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None

        # 关闭依赖接触传感器的终止条件
        self.terminations.base_contact = None
        # ============================================================

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


@configclass
class T3HexapodRoughEnvCfg_PLAY(T3HexapodRoughEnvCfg):
    """Configuration for playing/testing T3 Hexapod on rough terrain."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Spawn the robot randomly in the grid (instead of terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # Reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Disable randomization for play
        self.observations.policy.enable_corruption = False

        # Remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
