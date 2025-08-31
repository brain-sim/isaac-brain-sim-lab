# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .waypoint import WAYPOINT_CFG
from .walls import WallConfiguration

@configclass
class NavEnvCfg(DirectRLEnvCfg):
    """
    observation_space = {
        "state": 0,
        "image": (32, 32, 3),
    }
    """

    @configclass
    class NavSceneCfg(InteractiveSceneCfg):
        num_envs=4096 
        env_spacing = 40.0
        replicate_physics=True            

    # env
    decimation = 4
    episode_length_s = 30.0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)
    sim.create_stage_in_memory = True

    img_size = [3, 128, 128]
    observation_space = (
        img_size[0] * img_size[1] * img_size[2] + 4
    )  # Changed from 8 to 9 to include minimum wall distance

    # scene
    scene: NavSceneCfg = NavSceneCfg()
    env_spacing = scene.env_spacing
    waypoint_cfg = WAYPOINT_CFG
    static_friction = 1.0
    dynamic_friction = 1.0

    # Wall parameters
    room_size = 40.0
    num_goals = 10
    wall_thickness = 2.0
    wall_height = 3.0
    position_tolerance = waypoint_cfg.markers["marker1"].radius
    avoid_goal_position_tolerance = waypoint_cfg.markers["marker0"].radius
    position_margin_epsilon = 0.2  # TODO: can be removed needed to be tested

    # Initialize wall configuration (not a CFG but an interface to bsMaze) and apply to scene
    wall_config = WallConfiguration(room_size, wall_thickness, wall_height)
    wall_config.apply_to_scene_cfg(scene)

    # Terminations
    termination_on_goal_reached = True
    termination_on_vehicle_flip = True