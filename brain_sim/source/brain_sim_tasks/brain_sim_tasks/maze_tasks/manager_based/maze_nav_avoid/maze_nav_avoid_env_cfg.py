# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.spot import SPOT_CFG  

from brain_sim_assets.props.maze import bsMazeGenerator

# Import MDP configurations
from .mdp.mdp_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)

##
# Scene definition
##


@configclass
class BrainSimSceneCfg(InteractiveSceneCfg):

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(1000.0, 1000.0)),
    )

    # robot
    robot: ArticulationCfg = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # camera
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/Camera",
        update_period=1.0 / 50.0,
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=PinholeCameraCfg(
            focal_length=16.0,
            horizontal_aperture=32.0,
            vertical_aperture=32.0,
            focus_distance=1.0,
            clipping_range=(0.01, 1000.0),
            lock_camera=True,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.25, 0.0, 0.25),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    def __post_init__(self):

        maze = bsMazeGenerator.create_example_maze()
        
        start_goal = maze.get_start_goal()
        if start_goal:
            start_pos = start_goal[0]
            self.robot.init_state.pos = (start_pos[0], start_pos[1], 0.5)
        
        wall_configs = maze.get_wall_configs_dict()
        for wall_name, wall_cfg in wall_configs.items():
            setattr(self, wall_name, wall_cfg)


##
# Environment configuration
##


@configclass
class BrainSimEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BrainSimSceneCfg = BrainSimSceneCfg(num_envs=4, env_spacing=40.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1.0/200.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        self.scene.contact_forces.update_period = self.sim.dt
