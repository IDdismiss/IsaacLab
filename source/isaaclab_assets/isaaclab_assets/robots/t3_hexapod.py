# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the T3 Hexapod robot.

T3 is a 6-legged robot with 12 joints (2 joints per leg: Leg + Calf).

Leg naming convention:
- FL: Front Left
- ML: Middle Left
- RL: Rear Left
- FR: Front Right
- MR: Middle Right
- RR: Rear Right

Joint naming:
- *_Leg_joint: Hip joint (connects body to upper leg) - ACTIVE
- *_Calf_joint: Knee joint (connects upper leg to lower leg) - FIXED

Tripod Gait Groups:
- Group A: FL, MR, RL (Front-Left, Middle-Right, Rear-Left)
- Group B: FR, ML, RR (Front-Right, Middle-Left, Rear-Right)
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration - Actuators
##

# 单电机版本：只控制 Leg_joint（6个电机）
T3_SINGLE_MOTOR_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*_Leg_joint"],  # 只控制 Leg 关节
    effort_limit=20.0,
    velocity_limit=10.0,
    stiffness={".*_Leg_joint": 40.0},
    damping={".*_Leg_joint": 2.0},
)
"""Configuration for T3 hexapod with single motor per leg (only Leg_joint)."""

# Calf 关节固定器：高刚度保持固定角度
T3_CALF_FIXED_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*_Calf_joint"],  # Calf 关节
    effort_limit=50.0,
    velocity_limit=0.1,  # 低速度限制
    stiffness={".*_Calf_joint": 200.0},  # 高刚度保持固定
    damping={".*_Calf_joint": 20.0},  # 高阻尼防止振动
)
"""Configuration for fixed Calf joints with high stiffness."""

# 双电机版本（原版）：控制所有关节
T3_DUAL_MOTOR_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*_Leg_joint", ".*_Calf_joint"],
    effort_limit=20.0,
    velocity_limit=10.0,
    stiffness={".*_Leg_joint": 25.0, ".*_Calf_joint": 25.0},
    damping={".*_Leg_joint": 0.5, ".*_Calf_joint": 0.5},
)
"""Configuration for T3 hexapod with dual motors per leg."""


T3_DC_MOTOR_CFG = DCMotorCfg(
    joint_names_expr=[".*_Leg_joint", ".*_Calf_joint"],
    saturation_effort=30.0,
    effort_limit=20.0,
    velocity_limit=10.0,
    stiffness={".*_Leg_joint": 25.0, ".*_Calf_joint": 25.0},
    damping={".*_Leg_joint": 0.5, ".*_Calf_joint": 0.5},
)
"""Configuration for T3 hexapod with DC motor actuator model."""


##
# Configuration - Articulation (using USD)
##

# Path to USD file - use absolute path
T3_USD_PATH = "/home/d510/IsaacLab/source/isaaclab_assets/data/Robots/T3/t3.usd"


# ============================================================
# 单电机版本 - 用于三角步态训练
# 只有 Leg_joint 被控制，Calf_joint 固定
# ============================================================
T3_HEXAPOD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=T3_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.12),  # 起始高度
        rot=(1.0, 0.0, 0.0, 0.0),  # Quaternion (w, x, y, z)
        joint_pos={
            # Leg joints - 被控制的关节（初始角度 0）
            "FL_Leg_joint": 0.0,
            "ML_Leg_joint": 0.0,
            "RL_Leg_joint": 0.0,
            "FR_Leg_joint": 0.0,
            "MR_Leg_joint": 0.0,
            "RR_Leg_joint": 0.0,
            # Calf joints - 固定角度（弯曲以便着地）
            "FL_Calf_joint": -1.2,
            "ML_Calf_joint": -1.2,
            "RL_Calf_joint": -1.2,
            "FR_Calf_joint": -1.2,
            "MR_Calf_joint": -1.2,
            "RR_Calf_joint": -1.2,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": T3_SINGLE_MOTOR_ACTUATOR_CFG,  # 控制 Leg_joint
        "calf_fixed": T3_CALF_FIXED_CFG,  # 固定 Calf_joint
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration for T3 Hexapod with single motor per leg (tripod gait ready)."""


# ============================================================
# 双电机版本 - 完整控制
# ============================================================
T3_HEXAPOD_DUAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=T3_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "FL_Leg_joint": 0.0,
            "ML_Leg_joint": 0.0,
            "RL_Leg_joint": 0.0,
            "FR_Leg_joint": 0.0,
            "MR_Leg_joint": 0.0,
            "RR_Leg_joint": 0.0,
            "FL_Calf_joint": -0.8,
            "ML_Calf_joint": -0.8,
            "RL_Calf_joint": -0.8,
            "FR_Calf_joint": -0.8,
            "MR_Calf_joint": -0.8,
            "RR_Calf_joint": -0.8,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={"legs": T3_DUAL_MOTOR_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration for T3 Hexapod with dual motors per leg (full control)."""


# Legacy alias
T3_HEXAPOD_DC_CFG = T3_HEXAPOD_DUAL_CFG


# ============================================================
# 轮桨模式 - 速度控制，所有 Leg_joint 朝同一方向持续旋转
# stiffness=0：纯速度控制，damping 提供力矩
# ============================================================
T3_WHEEL_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*_Leg_joint"],
    effort_limit=20.0,
    velocity_limit=15.0,  # 最大旋转速度 rad/s
    stiffness={".*_Leg_joint": 0.0},   # 速度控制：刚度为 0
    damping={".*_Leg_joint": 5.0},     # 阻尼提供驱动力矩
)
"""Configuration for T3 wheel-paddle actuator (velocity control, stiffness=0)."""


T3_HEXAPOD_WHEEL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=T3_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.12),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Leg joints 初始角度 0
            "FL_Leg_joint": 0.0,
            "ML_Leg_joint": 0.0,
            "RL_Leg_joint": 0.0,
            "FR_Leg_joint": 0.0,
            "MR_Leg_joint": 0.0,
            "RR_Leg_joint": 0.0,
            # Calf joints 固定弯曲
            "FL_Calf_joint": -1.2,
            "ML_Calf_joint": -1.2,
            "RL_Calf_joint": -1.2,
            "FR_Calf_joint": -1.2,
            "MR_Calf_joint": -1.2,
            "RR_Calf_joint": -1.2,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": T3_WHEEL_ACTUATOR_CFG,   # 速度控制 Leg_joint
        "calf_fixed": T3_CALF_FIXED_CFG,  # 固定 Calf_joint
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration for T3 Hexapod in wheel-paddle mode (velocity control)."""
