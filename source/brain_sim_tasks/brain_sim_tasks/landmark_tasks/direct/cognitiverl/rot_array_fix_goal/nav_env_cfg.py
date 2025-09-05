# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass

from brain_sim_assets.props.waypoint import bsWaypointGenerator
from brain_sim_assets.props.maze_runtime import bsMazeRuntime

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
        num_envs=512  # Match your actual training env count
        env_spacing = 40.0
        lazy_sensor_update=True  # Add this for better performance
        replicate_physics=True

    # env
    decimation = 64  # Increased from 4 - fewer physics steps per RL step
    episode_length_s = 120.0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/200,  # Increased from 1/200 - ~2x faster physics!
        render_interval=8,  # Render every N simulation steps
        use_fabric=True,  # Enable USD Fabric for better performance
        device="cuda:0",  # Use GPU for physics
        physx=PhysxCfg(
            solver_type=1,  # TGS solver (more stable than PGS)
            min_position_iteration_count=1,  # Minimum solver iterations
            max_position_iteration_count=4,  # Reduced from default 16-32 for speed
            min_velocity_iteration_count=0,  # Minimum velocity iterations
            max_velocity_iteration_count=1,  # Reduced from default 4-16 for speed
            enable_ccd=False,  # Disable continuous collision detection for speed
            enable_stabilization=False,  # Disable for performance gain
            enable_enhanced_determinism=False,  # Default - don't need determinism
            bounce_threshold_velocity=0.5,  # Default value
            friction_offset_threshold=0.04,  # Default value
            friction_correlation_distance=0.025,  # Default value
            # GPU buffer sizes - reduced for memory efficiency with 512 envs
            gpu_max_rigid_contact_count=524288,  # ~1000 contacts per env
            gpu_max_rigid_patch_count=81920,  # ~160 patches per env
            gpu_found_lost_pairs_capacity=524288,
            gpu_found_lost_aggregate_pairs_capacity=1048576,
            gpu_total_aggregate_pairs_capacity=524288,
            gpu_collision_stack_size=33554432,
            gpu_heap_capacity=33554432,
            gpu_temp_buffer_capacity=8388608,
            gpu_max_num_partitions=8,
            gpu_max_soft_body_contacts=0,  # Not using soft bodies
            gpu_max_particle_contacts=0,  # Not using particles
        )
    )

    img_size = [3, 128, 128]
    observation_space = (
        img_size[0] * img_size[1] * img_size[2] + 4
    ) 

    # scene
    scene: NavSceneCfg = NavSceneCfg()
    env_spacing = scene.env_spacing
    waypoint_cfg = bsWaypointGenerator.get_waypoint_object(
        marker0_radius=1.0, marker1_radius=0.5, marker2_radius=0.0, marker3_radius=1.0
    )
    static_friction = 2.0
    dynamic_friction = 2.0

    # Wall parameters
    room_size = 40.0
    num_goals = 3
    wall_thickness = 2.0
    wall_height = 3.0
    position_tolerance = 1.0
    avoid_goal_position_tolerance = waypoint_cfg.markers["marker0"].radius
    position_margin_epsilon = 0.2  # TODO: can be removed needed to be tested

    # Initialize wall configuration (not a CFG but an interface to bsMaze) and apply to scene
    wall_config = bsMazeRuntime(room_size, maze_file="landmark.txt", maze_config="example_config.json")
    wall_config.apply_to_scene_cfg(scene)

    # Terminations
    termination_on_goal_reached = True
    termination_on_vehicle_flip = True