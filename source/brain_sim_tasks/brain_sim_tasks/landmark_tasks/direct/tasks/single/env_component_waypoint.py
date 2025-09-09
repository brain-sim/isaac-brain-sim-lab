from __future__ import annotations

import torch
from collections.abc import Sequence
from isaaclab.markers import VisualizationMarkers

from ...components.env_component_waypoint import EnvComponentWaypoint


class DerivedEnvComponentWaypoint(EnvComponentWaypoint):
    """Component responsible for waypoint generation and management for single task."""

    def __init__(self, env):
        self.env = env
        self.waypoints = None

    def post_env_init(self):
        # Initialize waypoint tracking tensors for single task (2 markers per group)
        self._target_positions = torch.zeros(
            (self.env.num_envs, self.env.cfg.num_markers_per_group, 2),
            device=self.env.device,
            dtype=torch.float32,
        )
        self._markers_pos = torch.zeros(
            (self.env.num_envs, self.env.cfg.num_markers_per_group, 3),
            device=self.env.device,
            dtype=torch.float32,
        )
        self._target_index = torch.zeros(
            (self.env.num_envs), device=self.env.device, dtype=torch.int32
        )
        self._target_group_index = torch.zeros(
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
            bound_limit = self.env.cfg.room_size / 3.0
            bounds_valid = (
                (relative_positions[:, 0] >= -bound_limit)
                & (relative_positions[:, 0] <= bound_limit)
                & (relative_positions[:, 1] >= -bound_limit)
                & (relative_positions[:, 1] <= bound_limit)
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

            bound_limit = self.env.cfg.room_size / 3.0
            valid_mask = (
                (relative_positions[:, 0] >= -bound_limit)
                & (relative_positions[:, 0] <= bound_limit)
                & (relative_positions[:, 1] >= -bound_limit)
                & (relative_positions[:, 1] <= bound_limit)
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

    def generate_waypoint_group(
        self,
        env_origins,
        num_reset,
        waypoint_offset,
        robot_xy=None,
    ):
        """Generate a group of two related waypoints."""
        second_waypoint = self.generate_random_waypoint(
            env_origins, num_reset, robot_xy
        )
        first_waypoint = self.generate_offset_waypoint(
            env_origins, second_waypoint, waypoint_offset
        )

    def generate_waypoints(self, env_ids, robot_poses, waypoint_offset=None):
        """Generate one group of waypoints for reset environments."""
        num_reset = len(env_ids)
        env_origins = self.env.scene.env_origins[env_ids, :2]
        robot_xy = robot_poses[:, :2]

        if waypoint_offset is None:
            waypoint_offset = 4.0

        # Generate one group of waypoints (2 markers)
        first_wp, second_wp = self.generate_waypoint_group(
            env_origins, num_reset, waypoint_offset, robot_xy
        )

        waypoint_positions = torch.zeros(
            (num_reset, self.env.cfg.num_markers_per_group, 2), device=self.env.device
        )
        waypoint_positions[:, 0, :] = first_wp  # Target waypoint
        waypoint_positions[:, 1, :] = second_wp  # Obstacle waypoint

        return waypoint_positions

    def update_waypoint_visualization(self):
        """Update waypoint visualization markers."""
        # Create marker indices for visualization
        marker_indices = torch.zeros(
            (self.env.num_envs, self.env.cfg.num_markers_per_group),
            device=self.env.device,
            dtype=torch.long,
        )

        # Set current target (always index 0 in current group) to 1 (green)
        marker_indices[:, 0] = 1

        # Set obstacle (index 1) to 0 (red)
        marker_indices[:, 1] = 0

        marker_indices = marker_indices.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)

    def reset(self, env_ids, robot_poses):
        """Reset waypoints for specified environments."""
        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0
        self._target_group_index[env_ids] = 0
        self._episode_groups_passed[env_ids] = 0

        # Generate new waypoint group
        waypoint_positions = self.generate_waypoints(env_ids, robot_poses)
        self._target_positions[env_ids] = waypoint_positions

        # Reset target index (always 0 for target in each group)
        self._target_index[env_ids] = 0
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)

    def generate_new_group(self, env_ids, robot_poses):
        """Generate a new group of waypoints for environments that completed a group."""
        if len(env_ids) == 0:
            return

        waypoint_positions = self.generate_waypoints(env_ids, robot_poses)
        self._target_positions[env_ids] = waypoint_positions
        self._target_index[env_ids] = 0
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]

        # Update visualization for all environments
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)
