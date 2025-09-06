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
        # Initialize waypoint tracking tensors
        self._target_positions = torch.zeros(
            (self.env.num_envs, self.env.cfg.num_goals, 2),
            device=self.env.device,
            dtype=torch.float32,
        )
        self._markers_pos = torch.zeros(
            (self.env.num_envs, self.env.cfg.num_goals, 3),
            device=self.env.device,
            dtype=torch.float32,
        )
        self._target_index = torch.zeros(
            (self.env.num_envs), device=self.env.device, dtype=torch.int32
        )
        self._episode_waypoints_passed = torch.zeros(
            (self.env.num_envs), device=self.env.device, dtype=torch.int32
        )

    def setup(self):
        self.waypoints = VisualizationMarkers(self.env.cfg.waypoint_cfg)

    def generate_waypoints(
        self, env_ids, robot_poses, waypoint_offset=None, fixed_first_position=None
    ):
        """Generate waypoints in groups of 2: (1→2)→(3→4)→..."""
        num_reset = len(env_ids)
        env_origins = self.env.scene.env_origins[env_ids, :2]  # (num_reset, 2)
        robot_xy = robot_poses[:, :2]  # (num_reset, 2)

        if waypoint_offset is None:
            waypoint_offset = torch.tensor([3.0, 0.5], device=self.env.device)

        if fixed_first_position is None:
            fixed_first_position = torch.tensor([5.0, 5.0], device=self.env.device)

        # Reverse offset: Current implementation has the goal generated first, landmark next
        waypoint_offset = -waypoint_offset

        waypoint_positions = torch.zeros(
            (num_reset, self.env.cfg.num_goals, 2), device=self.env.device
        )

        # Generate waypoints in groups of 2
        num_complete_groups = self.env.cfg.num_goals // 2
        remaining_points = self.env.cfg.num_goals % 2

        # Generate complete groups of 2
        for group_idx in range(num_complete_groups):
            first_idx = group_idx * 2  # First point of the group
            second_idx = group_idx * 2 + 1  # Second point of the group

            # Generate first waypoint of the group
            if group_idx == 0:
                # For the very first waypoint, use fixed position
                waypoint_positions[:, first_idx, :] = self.generate_fixed_waypoint(
                    env_origins, num_reset, fixed_first_position
                )
            else:
                # For subsequent groups, generate randomly without robot distance constraint
                waypoint_positions[:, first_idx, :] = self.generate_random_waypoint(
                    env_origins, num_reset
                )

            # Generate second waypoint with offset from first
            waypoint_positions[:, second_idx, :] = self.generate_offset_waypoint(
                env_origins,
                waypoint_positions[:, first_idx, :],
                waypoint_offset,
                num_reset,
            )

        # Handle remaining single waypoint if total is odd
        if remaining_points > 0:
            last_idx = num_complete_groups * 2
            waypoint_positions[:, last_idx, :] = self.generate_random_waypoint(
                env_origins, num_reset
            )

        return waypoint_positions

    def generate_fixed_waypoint(self, env_origins, num_reset, fixed_position):
        """Generate fixed waypoint at specified position for all environments."""
        # Create fixed position tensor for all environments
        fixed_waypoint = fixed_position.unsqueeze(0).expand(
            num_reset, -1
        )  # (num_reset, 2)

        # Add environment origins to get absolute positions
        absolute_positions = fixed_waypoint + env_origins

        # Check if positions are within bounds (-17 to +17 in both x and y)
        relative_positions = absolute_positions - env_origins
        valid_mask = (
            (relative_positions[:, 0] >= -17.0)
            & (relative_positions[:, 0] <= 17.0)
            & (relative_positions[:, 1] >= -17.0)
            & (relative_positions[:, 1] <= 17.0)
        )

        # For environments where fixed position is out of bounds, use fallback
        waypoint_positions = absolute_positions.clone()
        invalid_mask = ~valid_mask

        if invalid_mask.any():
            # Use fallback random positions for environments where fixed position is invalid
            num_invalid = invalid_mask.sum().item()
            random_positions = self.env.cfg.wall_config.get_random_valid_positions(
                num_invalid, device=self.env.device
            )
            invalid_origins = env_origins[invalid_mask]
            fallback_positions = random_positions[:, :2] + invalid_origins
            waypoint_positions[invalid_mask, :] = fallback_positions

        return waypoint_positions

    def generate_random_waypoint_with_robot_distance(
        self, env_origins, num_reset, robot_xy
    ):
        """Generate random waypoint with minimum distance from robot."""
        max_attempts = 100
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

            unplaced_robot_pos = robot_xy[unplaced_mask]
            robot_distances = torch.norm(
                candidate_positions - unplaced_robot_pos, dim=1
            )
            robot_valid = robot_distances >= 2.5

            valid_indices = torch.where(unplaced_mask)[0][robot_valid]
            if len(valid_indices) > 0:
                waypoint_positions[valid_indices, :] = candidate_positions[robot_valid]
                placed[valid_indices] = True

        # Fallback for unplaced waypoints
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

    def generate_random_waypoint(self, env_origins, num_reset):
        """Generate random waypoint without robot distance constraint."""
        random_positions = self.env.cfg.wall_config.get_random_valid_positions(
            num_reset, device=self.env.device
        )
        return random_positions[:, :2] + env_origins

    def generate_offset_waypoint(
        self, env_origins, base_waypoints, waypoint_offset, num_reset
    ):
        """Generate waypoint with offset from base waypoint."""
        max_attempts = 100
        placed = torch.zeros(num_reset, dtype=torch.bool, device=self.env.device)
        waypoint_positions = torch.zeros((num_reset, 2), device=self.env.device)

        # Apply the offset vector directly to base waypoint
        candidate_positions = base_waypoints + waypoint_offset.unsqueeze(0).expand(
            num_reset, -1
        )
        relative_positions = candidate_positions - env_origins

        for attempt in range(max_attempts):
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()

            if num_unplaced == 0:
                break

            unplaced_candidates = candidate_positions[unplaced_mask]
            unplaced_relative = relative_positions[unplaced_mask]

            # Check if positions are within bounds (-17 to +17 in both x and y)
            valid_mask = (
                (unplaced_relative[:, 0] >= -17.0)
                & (unplaced_relative[:, 0] <= 17.0)
                & (unplaced_relative[:, 1] >= -17.0)
                & (unplaced_relative[:, 1] <= 17.0)
            )

            valid_indices = torch.where(unplaced_mask)[0][valid_mask]
            if len(valid_indices) > 0:
                waypoint_positions[valid_indices, :] = unplaced_candidates[valid_mask]
                placed[valid_indices] = True

        # Fallback for unplaced waypoints
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

    def update_waypoint_visualization(self):
        """Update waypoint visualization markers."""
        # Create marker indices for visualization
        marker_indices = torch.zeros(
            (self.env.num_envs, self.env.cfg.num_goals),
            device=self.env.device,
            dtype=torch.long,
        )

        # Set current targets to 1 (green)
        marker_indices[
            torch.arange(self.env.num_envs, device=self.env.device), self._target_index
        ] = 1

        # Set completed targets to 2 (invisible)
        target_mask = (self._target_index.unsqueeze(1) > 0) & (
            torch.arange(self.env.cfg.num_goals, device=self.env.device)
            < self._target_index.unsqueeze(1)
        )
        marker_indices[target_mask] = 2
        marker_indices = marker_indices.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)

    def reset(self, env_ids, robot_poses):
        """Reset waypoints for specified environments."""
        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0
        self._episode_waypoints_passed[env_ids] = 0

        # Generate new waypoints
        waypoint_positions = self.generate_waypoints(env_ids, robot_poses)
        self._target_positions[env_ids] = waypoint_positions

        # Reset target index and visualization
        self._target_index[env_ids] = 0
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)
