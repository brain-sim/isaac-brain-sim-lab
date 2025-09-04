# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.spot import SPOT_CFG

from .nav_env_cfg import NavEnvCfg
from brain_sim_assets.props.waypoint import bsWaypointGenerator


@configclass
class LandmarkEnvCfg(NavEnvCfg):
    physcis_dt = 1 / 200
    # Two-level decimation for hierarchical control
    low_level_decimation = 4  # Low-level locomotion policy runs at 50Hz
    high_level_decimation = 5  # High-level navigation policy runs at 10Hz  
    decimation = low_level_decimation * high_level_decimation  # Total decimation = 20
    render_interval = 10  # Render at same frequency as env steps (10Hz)
    episode_length_s = 120.0
    action_space = 3
    img_size = [3, 128, 128]
    observation_space = (
        img_size[0] * img_size[1] * img_size[2] + 4
    )  # Changed from 8 to 9 to include minimum wall distance
    policy_file_path = "rslrl_policy.pt"

    # Override sim to use custom render_interval
    sim: SimulationCfg = SimulationCfg(
        dt=physcis_dt,
        render_interval=render_interval,  # Uses the render_interval defined above (32)
        use_fabric=True,  # Enable USD Fabric for better performance
        device="cuda:0",  # Use GPU for physics
        physx=PhysxCfg(  # Inherit optimized PhysX settings from parent
            solver_type=1,  # TGS solver
            min_position_iteration_count=1,
            max_position_iteration_count=4,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=1,
            enable_ccd=False,
            enable_stabilization=False,
            enable_enhanced_determinism=False,
            bounce_threshold_velocity=0.5,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
            gpu_max_rigid_contact_count=524288,
            gpu_max_rigid_patch_count=150000,
            gpu_found_lost_pairs_capacity=524288,
            gpu_found_lost_aggregate_pairs_capacity=1048576,
            gpu_total_aggregate_pairs_capacity=524288,
            gpu_collision_stack_size=33554432,
            gpu_heap_capacity=33554432,
            gpu_temp_buffer_capacity=8388608,
            gpu_max_num_partitions=8,
            gpu_max_soft_body_contacts=0,
            gpu_max_particle_contacts=0,
        ),
    )
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

    waypoint_cfg = bsWaypointGenerator.get_waypoint_object(
        marker0_radius=1.0, marker1_radius=0.5, marker2_radius=0.0, marker3_radius=1.0
    )
    position_tolerance = 1.0
    avoid_goal_position_tolerance = waypoint_cfg.markers["marker0"].radius

    goal_reached_bonus = 125.0
    wall_penalty_weight = -1.0  # -0.2
    linear_speed_weight = 0.0  # 0.05
    laziness_penalty_weight = 0.0  # -0.3
    avoid_penalty_weight = -250.0  # 0.0

    fast_goal_reached_weight = 125.0
    heading_coefficient = 0.25
    heading_progress_weight = 0.0  # 0.05

    laziness_decay = 0.3
    laziness_threshold = 8.0
    max_laziness = 10.0

    throttle_scale = 1.5
    steering_scale = 1.0
    throttle_max = 4.5
    steering_max = 3.0

    termination_on_avoid_goal_collision = True
    termination_on_stuck = False
    termination_on_stuck = False
