# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.spot import SPOT_CFG

from .nav_env_cfg import NavEnvCfg
from .waypoint import WAYPOINT_CFG


@configclass
class SpotNavRoughEnvCfg(NavEnvCfg):
    decimation = 16  # 2
    render_interval = 16
    episode_length_s = 60.0
    action_space = 3
    observation_space = 3076  # Changed from 8 to 9 to include minimum wall distance
    policy_file_path = "spot_rough_policy_custom_rslrl_final.pt"
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200, render_interval=render_interval
    )  # dt=1/250
    robot_cfg: ArticulationCfg = SPOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    dof_name = [
        "fl_hx",
        "fr_hx",
        "hl_hx",
        "hr_hx",
        "fl_hy",
        "fr_hy",
        "hl_hy",
        "hr_hy",
        "fl_kn",
        "fr_kn",
        "hl_kn",
        "hr_kn",
    ]

    waypoint_cfg = WAYPOINT_CFG
    position_tolerance = waypoint_cfg.markers["marker1"].radius
    static_friction = 1.0
    dynamic_friction = 1.0

    # Reward Coefficients
    goal_reached_bonus = 125.0
    wall_penalty_weight = -0.2  # -0.2
    linear_speed_weight = 0.0  # 0.05
    laziness_penalty_weight = -0.3  # -0.3
    # angular_speed_weight = 0.1  # 0.05
    # flip_penalty_weight = 100.0

    # Laziness
    laziness_decay = 0.3
    laziness_threshold = 8.0
    max_laziness = 10.0

    # Action Scaling
    throttle_scale = 1.0
    steering_scale = 0.5
    throttle_max = 9.0
    steering_max = 4.5


@configclass
class SpotNavRoughGridHeightEnvCfg(SpotNavRoughEnvCfg):
    grid_terrain_path = "grid_terrain.usd"