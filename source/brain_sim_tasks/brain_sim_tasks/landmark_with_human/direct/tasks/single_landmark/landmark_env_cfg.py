from isaaclab.utils import configclass

from brain_sim_assets.props.waypoint import bsWaypointGenerator
from brain_sim_assets.props.character import bsCharacterGenerator
from brain_sim_assets.props.maze_runtime import bsMazeRuntime

from ...landmark_env import LandmarkEnv
from ...landmark_env_cfg import LandmarkEnvCfg, NavSceneCfg
from .env_component_character import DerivedEnvComponentCharacter
from .env_component_objective import DerivedEnvComponentObjective
from .env_component_waypoint import DerivedEnvComponentWaypoint


@configclass
class DerivedLandmarkEnvCfg(LandmarkEnvCfg):

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

    room_size = 30.0
    num_groups = 10  # Total number of groups to complete the task
    num_markers_per_group = 2  # Single task: 1 target + 1 obstacle
    wall_thickness = 1.0
    wall_height = 3.0

    num_characters = 5

    # scene
    scene: NavSceneCfg = NavSceneCfg(env_spacing=room_size)
    env_spacing = scene.env_spacing
    # marker0: obstacle, marker1: target, marker2: finished goal (not used here)
    waypoint_cfg = bsWaypointGenerator.get_waypoint_object(
        marker0_radius=1.5, marker1_radius=1.0, marker2_radius=0.0
    )
    character_cfg = bsCharacterGenerator.get_character_object()

    approach_position_tolerance = (
        1.0
        if waypoint_cfg.markers["marker1"].radius == 0.0
        else waypoint_cfg.markers["marker1"].radius * 2.0
    )
    avoid_position_tolerance = waypoint_cfg.markers["marker0"].radius
    position_margin_epsilon = 0.2  # TODO: can be removed needed to be tested

    # Initialize wall configuration (not a CFG but an interface to bsMaze) and apply to scene
    wall_config = bsMazeRuntime(
        room_size, maze_file="landmark_30.txt", maze_config="maze_cell_1.json"
    )
    wall_config.apply_to_scene_cfg(scene)

class DerivedLandmarkEnv(LandmarkEnv):
    def __init__(self, cfg: DerivedLandmarkEnvCfg, **kwargs):
        super().__init__(
            cfg,
            component_character_cls=DerivedEnvComponentCharacter,
            component_objective_cls=DerivedEnvComponentObjective,
            component_waypoint_cls=DerivedEnvComponentWaypoint,
            **kwargs
        )