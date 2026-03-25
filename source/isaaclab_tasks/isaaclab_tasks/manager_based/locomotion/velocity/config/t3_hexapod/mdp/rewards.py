# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for T3 Hexapod tripod gait training.

Tripod Gait Pattern:
- Group A: FL (Front-Left), MR (Middle-Right), RL (Rear-Left)
- Group B: FR (Front-Right), ML (Middle-Left), RR (Rear-Right)

The two groups move in anti-phase (180 degrees out of phase).
When Group A swings forward, Group B supports the body, and vice versa.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Joint indices for tripod gait groups
# These should match the order in the USD/URDF file
# FL=0, FR=1, ML=2, MR=3, RL=4, RR=5 (typical order)
TRIPOD_GROUP_A_INDICES = [0, 3, 4]  # FL, MR, RL
TRIPOD_GROUP_B_INDICES = [1, 2, 5]  # FR, ML, RR


def tripod_gait_phase_sync(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    group_a_indices: list[int] = TRIPOD_GROUP_A_INDICES,
    group_b_indices: list[int] = TRIPOD_GROUP_B_INDICES,
) -> torch.Tensor:
    """Reward for synchronizing leg phases within each tripod group.

    This encourages legs in the same group to have similar joint velocities
    (moving together in the same direction).

    Args:
        env: The environment instance.
        asset_cfg: The asset configuration for the robot.
        group_a_indices: Joint indices for tripod group A.
        group_b_indices: Joint indices for tripod group B.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]

    # Get joint velocities for Leg joints only (first 6 joints)
    joint_vel = asset.data.joint_vel[:, :6]

    # Group A synchronization: variance of velocities should be low
    group_a_vel = joint_vel[:, group_a_indices]
    group_a_mean = group_a_vel.mean(dim=1, keepdim=True)
    group_a_var = ((group_a_vel - group_a_mean) ** 2).mean(dim=1)

    # Group B synchronization: variance of velocities should be low
    group_b_vel = joint_vel[:, group_b_indices]
    group_b_mean = group_b_vel.mean(dim=1, keepdim=True)
    group_b_var = ((group_b_vel - group_b_mean) ** 2).mean(dim=1)

    # Lower variance = higher reward
    reward = torch.exp(-2.0 * (group_a_var + group_b_var))

    return reward


def tripod_gait_anti_phase(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    group_a_indices: list[int] = TRIPOD_GROUP_A_INDICES,
    group_b_indices: list[int] = TRIPOD_GROUP_B_INDICES,
) -> torch.Tensor:
    """Reward for anti-phase movement between the two tripod groups.

    This encourages the two groups to move in opposite directions,
    which is the key characteristic of tripod gait.

    Args:
        env: The environment instance.
        asset_cfg: The asset configuration for the robot.
        group_a_indices: Joint indices for tripod group A.
        group_b_indices: Joint indices for tripod group B.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]

    # Get joint velocities for Leg joints only
    joint_vel = asset.data.joint_vel[:, :6]

    # Mean velocity of each group
    group_a_mean_vel = joint_vel[:, group_a_indices].mean(dim=1)
    group_b_mean_vel = joint_vel[:, group_b_indices].mean(dim=1)

    # Anti-phase: product of mean velocities should be negative
    # When one group moves forward (positive vel), other should move backward (negative vel)
    anti_phase_score = -group_a_mean_vel * group_b_mean_vel

    # Normalize and ensure positive reward
    reward = torch.tanh(anti_phase_score * 0.5)
    reward = torch.clamp(reward, min=0.0)

    return reward


def tripod_gait_rhythm(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_frequency: float = 2.0,  # Target gait frequency in Hz
) -> torch.Tensor:
    """Reward for maintaining a rhythmic gait pattern.

    Encourages the robot to maintain a consistent stepping frequency.

    Args:
        env: The environment instance.
        asset_cfg: The asset configuration for the robot.
        target_frequency: Target gait frequency in Hz.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]

    # Get joint velocities magnitude
    joint_vel = asset.data.joint_vel[:, :6]
    vel_magnitude = torch.abs(joint_vel).mean(dim=1)

    # Reward for having non-zero velocity (active movement)
    # This encourages the robot to keep moving rather than staying still
    reward = torch.tanh(vel_magnitude * 0.5)

    return reward


def leg_swing_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for symmetric leg swing amplitudes.

    Encourages all legs to have similar swing amplitudes for balanced gait.

    Args:
        env: The environment instance.
        asset_cfg: The asset configuration for the robot.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]

    # Get joint positions for Leg joints
    joint_pos = asset.data.joint_pos[:, :6]

    # Variance of joint positions (lower = more symmetric)
    pos_var = joint_pos.var(dim=1)

    # Lower variance = higher reward
    reward = torch.exp(-pos_var)

    return reward


def forward_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_velocity: float = 0.5,  # Target forward velocity in m/s
) -> torch.Tensor:
    """Reward for moving forward at target velocity.

    Args:
        env: The environment instance.
        asset_cfg: The asset configuration for the robot.
        target_velocity: Target forward velocity in m/s.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]

    # Get base linear velocity (x direction is forward)
    base_vel = asset.data.root_lin_vel_b[:, 0]  # Forward velocity

    # Reward for being close to target velocity
    vel_error = torch.abs(base_vel - target_velocity)
    reward = torch.exp(-vel_error * 2.0)

    return reward


def stability_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maintaining body stability (minimal roll and pitch).

    Args:
        env: The environment instance.
        asset_cfg: The asset configuration for the robot.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]

    # Get angular velocity (roll and pitch should be minimal)
    ang_vel = asset.data.root_ang_vel_b[:, :2]  # Roll and pitch angular velocities
    ang_vel_magnitude = torch.norm(ang_vel, dim=1)

    # Lower angular velocity = higher reward
    reward = torch.exp(-ang_vel_magnitude * 0.5)

    return reward


# ============================================================
# 轮桨模式奖励函数 (Wheel-Paddle Mode Rewards)
# ============================================================


def wheel_forward_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_velocity: float = 0.5,
) -> torch.Tensor:
    """Reward for moving forward at target velocity in wheel-paddle mode.

    Args:
        env: The environment instance.
        asset_cfg: The asset configuration for the robot.
        target_velocity: Target forward velocity in m/s.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]
    base_vel = asset.data.root_lin_vel_b[:, 0]  # Forward velocity (x direction)
    vel_error = torch.abs(base_vel - target_velocity)
    return torch.exp(-vel_error * 2.0)


def wheel_spin_direction(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for joints spinning in the wrong direction.

    In wheel-paddle mode, all Leg_joints should spin in the same direction
    (positive velocity). This penalizes any joint with negative velocity.

    Args:
        env: The environment instance.
        asset_cfg: The asset configuration for the robot.

    Returns:
        Penalty tensor of shape (num_envs,) - negative values indicate violation.
    """
    asset = env.scene[asset_cfg.name]

    # Get joint velocities for Leg joints only (first 6)
    joint_vel = asset.data.joint_vel[:, :6]

    # Penalize negative velocities (wrong direction)
    # relu(-vel) is positive when vel is negative
    wrong_dir_penalty = torch.relu(-joint_vel).mean(dim=1)

    return -wrong_dir_penalty


def wheel_tripod_pattern(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    group_a_indices: list[int] = TRIPOD_GROUP_A_INDICES,
    group_b_indices: list[int] = TRIPOD_GROUP_B_INDICES,
) -> torch.Tensor:
    """Reward for tripod-patterned wheel spinning.

    Encourages Group A and Group B to alternate between fast and slow spinning:
    when Group A spins fast (ground contact pushing), Group B spins slow (swing),
    and vice versa. This creates a tripod gait wheel-paddle pattern.

    Args:
        env: The environment instance.
        asset_cfg: The asset configuration for the robot.
        group_a_indices: Joint indices for tripod group A.
        group_b_indices: Joint indices for tripod group B.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]

    # Get joint velocities for Leg joints
    joint_vel = asset.data.joint_vel[:, :6]

    # Mean speed of each group
    group_a_speed = joint_vel[:, group_a_indices].mean(dim=1)
    group_b_speed = joint_vel[:, group_b_indices].mean(dim=1)

    # Total speed should be high (both groups active)
    total_speed = torch.abs(group_a_speed) + torch.abs(group_b_speed)

    # Difference between groups (alternating fast/slow)
    speed_diff = torch.abs(group_a_speed - group_b_speed)

    # Reward high total speed with alternating pattern
    reward = torch.tanh(total_speed * 0.3) * torch.tanh(speed_diff * 0.5)
    reward = torch.clamp(reward, min=0.0)

    return reward


def wheel_lateral_drift_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for lateral (sideways) drift in wheel-paddle mode.

    The robot should move straight forward, not sideways.

    Args:
        env: The environment instance.
        asset_cfg: The asset configuration for the robot.

    Returns:
        Penalty tensor of shape (num_envs,) - always negative.
    """
    asset = env.scene[asset_cfg.name]

    # Lateral velocity (y direction)
    lateral_vel = asset.data.root_lin_vel_b[:, 1]
    return -torch.abs(lateral_vel)
