# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .waypoint import WAYPOINT_CFG

from brain_sim_assets.props.maze import bsMazeGenerator
from brain_sim_assets import BRAIN_SIM_ASSETS_PROPS_CONFIG_DIR

# Private variables - only for use within this file
_room_size = 40.0
_wall_thickness = 2.0

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
        
        def __post_init__(self):
            wall_position = (_room_size - _wall_thickness) / 2
            offset = (-wall_position, -wall_position, 0.0)
            maze = bsMazeGenerator.create_example_maze(f"{BRAIN_SIM_ASSETS_PROPS_CONFIG_DIR}/example_maze_sq.txt", position_offset=offset)
            walls_config = maze.get_wall_collection()
            setattr(self, "wall_collection", walls_config)

    # env
    decimation = 4
    episode_length_s = 30.0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

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
    room_size = _room_size
    num_goals = 10
    wall_thickness = _wall_thickness
    wall_height = 3.0
    position_tolerance = waypoint_cfg.markers["marker1"].radius
    avoid_goal_position_tolerance = waypoint_cfg.markers["marker0"].radius

    position_margin_epsilon = 0.2  # TODO: can be removed needed to be tested

    # Terminations
    termination_on_goal_reached = True
    termination_on_vehicle_flip = True