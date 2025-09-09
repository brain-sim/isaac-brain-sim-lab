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

    def generate_offset_waypoint(self, env_origins, base_waypoints, offset_vector):
        """Generate waypoint with fixed vector offset from base waypoint."""
        num_reset = base_waypoints.shape[0]
        max_attempts = 1000
        placed = torch.zeros(num_reset, dtype=torch.bool, device=self.env.device)
        waypoint_positions = torch.zeros((num_reset, 2), device=self.env.device)

        # Ensure offset_vector is a tensor on the correct device
        if isinstance(offset_vector, (list, tuple)):
            offset_vector = torch.tensor(offset_vector, device=self.env.device, dtype=torch.float32)
        elif not isinstance(offset_vector, torch.Tensor):
            offset_vector = torch.tensor([offset_vector, 0.0], device=self.env.device, dtype=torch.float32)
        
        # If offset_vector is 1D, ensure it has shape [2]
        if offset_vector.dim() == 0:
            offset_vector = torch.tensor([offset_vector.item(), 0.0], device=self.env.device, dtype=torch.float32)
        elif offset_vector.shape[0] == 1:
            offset_vector = torch.cat([offset_vector, torch.zeros(1, device=self.env.device)])

        for _ in range(max_attempts):
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()

            if num_unplaced == 0:
                break

            # Apply the fixed vector offset directly to base waypoints
            unplaced_base = base_waypoints[unplaced_mask, :]
            candidate_positions = unplaced_base + offset_vector.unsqueeze(0).expand(num_unplaced, -1)
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
        fixed_second_position=None,
    ):
        """Generate a group of two related waypoints with guaranteed correct offset."""
        max_group_attempts = 1000
        placed = torch.zeros(num_reset, dtype=torch.bool, device=self.env.device)
        first_waypoints = torch.zeros((num_reset, 2), device=self.env.device)
        second_waypoints = torch.zeros((num_reset, 2), device=self.env.device)
        
        # Ensure offset_vector is a tensor on the correct device
        if isinstance(waypoint_offset, (list, tuple)):
            offset_vector = torch.tensor(waypoint_offset, device=self.env.device, dtype=torch.float32)
        elif not isinstance(waypoint_offset, torch.Tensor):
            offset_vector = torch.tensor([waypoint_offset, 0.0], device=self.env.device, dtype=torch.float32)
        else:
            offset_vector = waypoint_offset
            
        # If offset_vector is 1D, ensure it has shape [2]
        if offset_vector.dim() == 0:
            offset_vector = torch.tensor([offset_vector.item(), 0.0], device=self.env.device, dtype=torch.float32)
        elif offset_vector.shape[0] == 1:
            offset_vector = torch.cat([offset_vector, torch.zeros(1, device=self.env.device)])

        for attempt in range(max_group_attempts):
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()
            
            if num_unplaced == 0:
                break
                
            # Generate second waypoints for unplaced groups
            if fixed_second_position is not None:
                unplaced_second = self.generate_fixed_waypoint(
                    env_origins[unplaced_mask], num_unplaced, fixed_second_position
                )
            else:
                unplaced_second = self.generate_random_waypoint(
                    env_origins[unplaced_mask], num_unplaced, 
                    robot_xy[unplaced_mask] if robot_xy is not None else None
                )
            
            # Generate first waypoints with exact offset
            unplaced_first = unplaced_second + offset_vector.unsqueeze(0).expand(num_unplaced, -1)
            
            # Check if both waypoints are within bounds
            bound_limit = self.env.cfg.room_size / 3.0
            first_relative = unplaced_first - env_origins[unplaced_mask]
            second_relative = unplaced_second - env_origins[unplaced_mask]
            
            first_valid = (
                (first_relative[:, 0] >= -bound_limit)
                & (first_relative[:, 0] <= bound_limit)
                & (first_relative[:, 1] >= -bound_limit)
                & (first_relative[:, 1] <= bound_limit)
            )
            
            second_valid = (
                (second_relative[:, 0] >= -bound_limit)
                & (second_relative[:, 0] <= bound_limit)
                & (second_relative[:, 1] >= -bound_limit)
                & (second_relative[:, 1] <= bound_limit)
            )
            
            # Only accept groups where both waypoints are valid
            group_valid = first_valid & second_valid
            valid_indices = torch.where(unplaced_mask)[0][group_valid]
            
            if len(valid_indices) > 0:
                first_waypoints[valid_indices] = unplaced_first[group_valid]
                second_waypoints[valid_indices] = unplaced_second[group_valid]
                placed[valid_indices] = True
        
        # Handle any remaining unplaced groups with fallback
        if not placed.all():
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()
            
            # Generate fallback positions
            random_positions = self.env.cfg.wall_config.get_random_valid_positions(
                num_unplaced * 2, device=self.env.device  # Get positions for both waypoints
            )
            
            unplaced_origins = env_origins[unplaced_mask]
            fallback_second = random_positions[:num_unplaced, :2] + unplaced_origins
            fallback_first = random_positions[num_unplaced:, :2] + unplaced_origins
            
            first_waypoints[unplaced_mask] = fallback_first
            second_waypoints[unplaced_mask] = fallback_second
        
        return first_waypoints, second_waypoints
        """Generate a group of two related waypoints."""
        second_waypoint = self.generate_random_waypoint(
            env_origins, num_reset, robot_xy
        )
        first_waypoint = self.generate_offset_waypoint(
            env_origins, second_waypoint, waypoint_offset
        )
        return first_waypoint, second_waypoint

    def generate_waypoints(self, env_ids, robot_poses, waypoint_offset=None):
        """Generate one group of waypoints for reset environments."""
        num_reset = len(env_ids)
        env_origins = self.env.scene.env_origins[env_ids, :2]
        robot_xy = robot_poses[:, :2]

        if waypoint_offset is None:
            waypoint_offset = torch.tensor([3.0, 1.0], device=self.env.device)  # Fixed vector [x, y]

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
