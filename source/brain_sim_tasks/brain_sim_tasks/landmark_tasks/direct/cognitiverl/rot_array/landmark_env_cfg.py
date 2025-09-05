import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.spot import SPOT_CFG

from brain_sim_assets.props.waypoint import bsWaypointGenerator
from brain_sim_assets.props.maze_runtime import bsMazeRuntime

@configclass
class LandmarkEnvCfg(DirectRLEnvCfg):

    physics_dt = 1.0 / 200.0  # Physics timestep
    low_level_decimation = 4  # Low-level locomotion policy runs at 50Hz
    high_level_decimation = 5  # High-level navigation policy runs at 10Hz  
    decimation = low_level_decimation * high_level_decimation  # Total decimation = 20
    render_interval = 10  # Render at same frequency as env steps (10Hz)

    episode_length_s = 240.0
    action_space = 3
    img_size = [3, 128, 128]
    observation_space = (img_size[0] * img_size[1] * img_size[2] + 4)  
    policy_file_path = "rslrl_policy.pt"

    sim: SimulationCfg = SimulationCfg(
        dt=physics_dt,
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

    robot_cfg: ArticulationCfg = SPOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    dof_name = [
        "fl_hx", "fr_hx", "hl_hx", "hr_hx",
        "fl_hy", "fr_hy", "hl_hy", "hr_hy",
        "fl_kn", "fr_kn", "hl_kn", "hr_kn",
    ]

    goal_reached_bonus = 125.0
    wall_penalty_weight = -1.0  
    linear_speed_weight = 0.0  
    laziness_penalty_weight = 0.0
    avoid_penalty_weight = -250.0

    fast_goal_reached_weight = 125.0
    heading_coefficient = 0.25
    heading_progress_weight = 0.0  

    laziness_decay = 0.3
    laziness_threshold = 8.0
    max_laziness = 10.0

    throttle_scale = 1.5
    steering_scale = 1.0
    throttle_max = 4.5
    steering_max = 3.0

    @configclass
    class NavSceneCfg(InteractiveSceneCfg):
        num_envs=512  
        env_spacing = 20.0
        lazy_sensor_update=True  
        replicate_physics=True

        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(
                size=(
                    4096 * 40.0,
                    4096 * 40.0,
                ),
                color=(0.2, 0.2, 0.2),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=2.0,
                    dynamic_friction=2.0,
                    restitution=0.0,
                ),
            )
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        )

    # scene
    scene: NavSceneCfg = NavSceneCfg()
    env_spacing = scene.env_spacing
    waypoint_cfg = bsWaypointGenerator.get_waypoint_object(
        marker0_radius=0.5, marker1_radius=0.5, marker2_radius=0.0, marker3_radius=0.5
    )
    
    # Wall parameters
    room_size = 20.0
    num_goals = 3
    wall_thickness = 1.0
    wall_height = 3.0
    position_tolerance = waypoint_cfg.markers["marker1"].radius * 2.0
    avoid_goal_position_tolerance = waypoint_cfg.markers["marker0"].radius
    position_margin_epsilon = 0.2  # TODO: can be removed needed to be tested

    # Initialize wall configuration (not a CFG but an interface to bsMaze) and apply to scene
    wall_config = bsMazeRuntime(room_size, maze_file="landmark.txt", maze_config="maze_cell_1.json")
    wall_config.apply_to_scene_cfg(scene)

    # Terminations
    termination_on_goal_reached = True
    termination_on_vehicle_flip = True
    termination_on_avoid_goal_collision = False
    termination_on_stuck = False
