from typing import Dict
from isaaclab.assets import RigidObjectCfg

from brain_sim_assets.props.goal_marker import bsGoalMarkerGenerator
from brain_sim_assets.props.obstacle_marker import bsObstacleMarkerGenerator


class bsMarkersGenerator:

    def __init__(self, num_goals: int, num_obstacles: int):
        self.goal_markers = []
        self.obstacle_markers = []
        self._num_goals = num_goals
        self._num_obstacles = num_obstacles
        if num_goals and num_obstacles:
            self.create_markers()

    def create_markers(self):
        for i in range(self._num_goals):
            marker_cfg = bsGoalMarkerGenerator.get_goal_marker_object(
                prim_path=f"{{ENV_REGEX_NS}}/goal_{i}",
            )

            self.goal_markers.append(marker_cfg)

        for i in range(self._num_obstacles):
            marker_cfg = bsObstacleMarkerGenerator.get_obstacle_marker_object(
                prim_path=f"{{ENV_REGEX_NS}}/obstacle_{i}",
            )

            self.obstacle_markers.append(marker_cfg)

    def get_marker_configs_dict(self) -> Dict[str, RigidObjectCfg]:
        marker_dict = {}
        for i, marker_cfg in enumerate(self.goal_markers):
            marker_dict[f"goal_marker_{i}"] = marker_cfg
        for i, marker_cfg in enumerate(self.obstacle_markers):
            marker_dict[f"obstacle_marker_{i}"] = marker_cfg
        return marker_dict
