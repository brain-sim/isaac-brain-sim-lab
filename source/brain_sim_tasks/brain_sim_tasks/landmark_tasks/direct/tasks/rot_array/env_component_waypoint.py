from __future__ import annotations

import torch
from collections.abc import Sequence
from isaaclab.markers import VisualizationMarkers

from ...components.env_component_waypoint import EnvComponentWaypoint


class DerivedEnvComponentWaypoint(EnvComponentWaypoint):
    """Component responsible for waypoint generation and management."""

    def __init__(self, env):
        self.env = env
        self.waypoints = None

    def post_env_init(self):
        # Initialize waypoint tracking tensors
        self._target_positions = torch.zeros(
            (self.env.num_envs, self.env.cfg.num_markers, 2),
            device=self.env.device,
            dtype=torch.float32,
        )
        self._markers_pos = torch.zeros(
            (self.env.num_envs, self.env.cfg.num_markers, 3),
            device=self.env.device,
            dtype=torch.float32,
        )
        self._target_index = torch.zeros(
            (self.env.num_envs), device=self.env.device, dtype=torch.int32
        )
        self._episode_groups_passed = torch.zeros(
            (self.env.num_envs), device=self.env.device, dtype=torch.int32
        )

    def setup(self):
        self.waypoints = VisualizationMarkers(self.env.cfg.waypoint_cfg)

    def generate_random_waypoint(
        self, env_origins, num_reset, robot_xy=None, min_distance=2.5
    ):
        """Generate random valid waypoints."""
        max_attempts = 1000
        placed = torch.zeros(num_reset, dtype=torch.bool, device=self.env.device)
        waypoint_positions = torch.zeros((num_reset, 2), device=self.env.device)

        for _ in range(max_attempts):
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()

            if num_unplaced == 0:
                break

            random_positions = self.env.cfg.wall_config.get_random_valid_positions(
                num_unplaced, device=self.env.device
            )

            tx = random_positions[:, 0]
            ty = random_positions[:, 1]

            unplaced_origins = env_origins[unplaced_mask]
            candidate_positions = torch.stack([tx, ty], dim=1) + unplaced_origins

            if robot_xy is not None:
                unplaced_robot_pos = robot_xy[unplaced_mask]
                robot_distances = torch.norm(
                    candidate_positions - unplaced_robot_pos, dim=1
                )
                robot_valid = robot_distances >= min_distance
            else:
                robot_valid = torch.ones(
                    num_unplaced, dtype=torch.bool, device=self.env.device
                )

            relative_positions = candidate_positions - unplaced_origins
            bounds_valid = (
                (relative_positions[:, 0] >= -17.0)
                & (relative_positions[:, 0] <= 17.0)
                & (relative_positions[:, 1] >= -17.0)
                & (relative_positions[:, 1] <= 17.0)
            )

            valid_mask = robot_valid & bounds_valid
            valid_indices = torch.where(unplaced_mask)[0][valid_mask]
            if len(valid_indices) > 0:
                waypoint_positions[valid_indices, :] = candidate_positions[valid_mask]
                placed[valid_indices] = True

        if not placed.all():
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()
            random_positions = self.env.cfg.wall_config.get_random_valid_positions(
                num_unplaced, device=self.env.device
            )
            unplaced_origins = env_origins[unplaced_mask]
            fallback_positions = random_positions[:, :2] + unplaced_origins
            waypoint_positions[unplaced_mask, :] = fallback_positions

        return waypoint_positions

    def generate_offset_waypoint(self, env_origins, base_waypoints, offset_distance):
        """Generate waypoint at offset distance from base waypoint."""
        num_reset = base_waypoints.shape[0]
        max_attempts = 1000
        placed = torch.zeros(num_reset, dtype=torch.bool, device=self.env.device)
        waypoint_positions = torch.zeros((num_reset, 2), device=self.env.device)

        for _ in range(max_attempts):
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()

            if num_unplaced == 0:
                break

            random_angles = (
                torch.rand((num_unplaced,), device=self.env.device) * 2 * torch.pi
            )

            offset_x = offset_distance * torch.cos(random_angles)
            offset_y = offset_distance * torch.sin(random_angles)
            random_offsets = torch.stack([offset_x, offset_y], dim=1)

            unplaced_base = base_waypoints[unplaced_mask, :]
            candidate_positions = unplaced_base + random_offsets
            relative_positions = candidate_positions - env_origins[unplaced_mask]

            valid_mask = (
                (relative_positions[:, 0] >= -17.0)
                & (relative_positions[:, 0] <= 17.0)
                & (relative_positions[:, 1] >= -17.0)
                & (relative_positions[:, 1] <= 17.0)
            )

            valid_indices = torch.where(unplaced_mask)[0][valid_mask]
            if len(valid_indices) > 0:
                waypoint_positions[valid_indices, :] = candidate_positions[valid_mask]
                placed[valid_indices] = True

        if not placed.all():
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()
            random_positions = self.env.cfg.wall_config.get_random_valid_positions(
                num_unplaced, device=self.env.device
            )
            unplaced_origins = env_origins[unplaced_mask]
            fallback_positions = random_positions[:, :2] + unplaced_origins
            waypoint_positions[unplaced_mask, :] = fallback_positions

        return waypoint_positions

    def generate_perpendicular_waypoint(
        self, env_origins, waypoint1, waypoint2, perp_offset
    ):
        """Generate waypoint perpendicular to line between two waypoints."""
        num_reset = waypoint1.shape[0]
        max_attempts = 1000

        center_point = (waypoint1 + waypoint2) / 2.0

        direction_vector = waypoint2 - waypoint1
        perp_vector = torch.stack(
            [-direction_vector[:, 1], direction_vector[:, 0]], dim=1
        )

        perp_length = torch.norm(perp_vector, dim=1, keepdim=True)
        perp_length = torch.clamp(perp_length, min=1e-8)
        perp_unit_vector = perp_vector / perp_length

        candidate_positions = center_point + perp_offset * perp_unit_vector
        relative_positions = candidate_positions - env_origins

        placed = torch.zeros(num_reset, dtype=torch.bool, device=self.env.device)

        for attempt in range(max_attempts):
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()

            if num_unplaced == 0:
                break

            unplaced_candidates = candidate_positions[unplaced_mask]
            unplaced_relative = relative_positions[unplaced_mask]

            valid_mask = (
                (unplaced_relative[:, 0] >= -17.0)
                & (unplaced_relative[:, 0] <= 17.0)
                & (unplaced_relative[:, 1] >= -17.0)
                & (unplaced_relative[:, 1] <= 17.0)
            )

            valid_indices = torch.where(unplaced_mask)[0][valid_mask]
            if len(valid_indices) > 0:
                placed[valid_indices] = True

            if not placed.all():
                unplaced_mask = ~placed
                candidate_positions[unplaced_mask] = (
                    center_point[unplaced_mask]
                    - perp_offset * perp_unit_vector[unplaced_mask]
                )
                relative_positions[unplaced_mask] = (
                    candidate_positions[unplaced_mask] - env_origins[unplaced_mask]
                )

        waypoint_positions = candidate_positions.clone()

        if not placed.all():
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()
            random_positions = self.env.cfg.wall_config.get_random_valid_positions(
                num_unplaced, device=self.env.device
            )
            unplaced_origins = env_origins[unplaced_mask]
            fallback_positions = random_positions[:, :2] + unplaced_origins
            waypoint_positions[unplaced_mask, :] = fallback_positions

        return waypoint_positions

    def generate_waypoint_group(
        self,
        env_origins,
        num_reset,
        waypoint_offset,
        perpendicular_offset,
        robot_xy=None,
    ):
        """Generate a group of three related waypoints."""
        second_waypoint = self.generate_random_waypoint(
            env_origins, num_reset, robot_xy
        )
        third_waypoint = self.generate_offset_waypoint(
            env_origins, second_waypoint, waypoint_offset
        )
        first_waypoint = self.generate_perpendicular_waypoint(
            env_origins, second_waypoint, third_waypoint, perpendicular_offset
        )
        return first_waypoint, second_waypoint, third_waypoint

    def generate_waypoints(
        self, env_ids, robot_poses, waypoint_offset=None, perpendicular_offset=None
    ):
        """Generate all waypoints for reset environments."""
        num_reset = len(env_ids)
        env_origins = self.env.scene.env_origins[env_ids, :2]
        robot_xy = robot_poses[:, :2]

        if waypoint_offset is None:
            waypoint_offset = 4.0

        if perpendicular_offset is None:
            perpendicular_offset = 2.0

        waypoint_positions = torch.zeros(
            (num_reset, self.env.cfg.num_markers, 2), device=self.env.device
        )

        if self.env.cfg.num_markers >= 3:
            first_wp, second_wp, third_wp = self.generate_waypoint_group(
                env_origins, num_reset, waypoint_offset, perpendicular_offset, robot_xy
            )
            waypoint_positions[:, 0, :] = first_wp
            waypoint_positions[:, 1, :] = second_wp
            waypoint_positions[:, 2, :] = third_wp
        elif self.env.cfg.num_markers == 2:
            second_wp = self.generate_random_waypoint(env_origins, num_reset, robot_xy)
            third_wp = self.generate_offset_waypoint(
                env_origins, second_wp, waypoint_offset
            )
            waypoint_positions[:, 0, :] = second_wp
            waypoint_positions[:, 1, :] = third_wp
        elif self.env.cfg.num_markers == 1:
            first_wp = self.generate_random_waypoint(env_origins, num_reset, robot_xy)
            waypoint_positions[:, 0, :] = first_wp

        # Generate additional waypoint groups if needed
        remaining_goals = self.env.cfg.num_markers - 3
        num_complete_groups = remaining_goals // 3 if remaining_goals > 0 else 0

        for group_idx in range(num_complete_groups):
            base_idx = 3 + group_idx * 3
            first_idx = base_idx
            second_idx = base_idx + 1
            third_idx = base_idx + 2

            first_wp, second_wp, third_wp = self.generate_waypoint_group(
                env_origins, num_reset, waypoint_offset, perpendicular_offset
            )
            waypoint_positions[:, first_idx, :] = first_wp
            waypoint_positions[:, second_idx, :] = second_wp
            waypoint_positions[:, third_idx, :] = third_wp

        # Generate remaining individual waypoints
        remaining_points = remaining_goals % 3 if remaining_goals > 0 else 0
        if remaining_points > 0:
            start_idx = 3 + num_complete_groups * 3
            for point_idx in range(remaining_points):
                goal_idx = start_idx + point_idx
                random_wp = self.generate_random_waypoint(env_origins, num_reset)
                waypoint_positions[:, goal_idx, :] = random_wp

        return waypoint_positions

    def update_waypoint_visualization(self):
        """Update waypoint visualization markers."""
        # Create marker indices for visualization
        marker_indices = torch.zeros(
            (self.env.num_envs, self.env.cfg.num_markers),
            device=self.env.device,
            dtype=torch.long,
        )

        # Set current targets to 1 (green)
        marker_indices[
            torch.arange(self.env.num_envs, device=self.env.device), self._target_index
        ] = 1

        # Set the next target to 3 (cyan)
        next_target_idx = (self._target_index + 1) % self.env.cfg.num_markers
        marker_indices[
            torch.arange(self.env.num_envs, device=self.env.device), next_target_idx
        ] = 3

        # Set completed targets to 2 (invisible)
        target_mask = (self._target_index.unsqueeze(1) > 0) & (
            torch.arange(self.env.cfg.num_markers, device=self.env.device)
            < self._target_index.unsqueeze(1)
        )
        marker_indices[target_mask] = 2
        marker_indices = marker_indices.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)

    def reset(self, env_ids, robot_poses):
        """Reset waypoints for specified environments."""
        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0
        self._episode_groups_passed[env_ids] = 0

        # Generate new waypoints
        waypoint_positions = self.generate_waypoints(env_ids, robot_poses)
        self._target_positions[env_ids] = waypoint_positions

        # Reset target index and visualization
        self._target_index[env_ids] = 0
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)
