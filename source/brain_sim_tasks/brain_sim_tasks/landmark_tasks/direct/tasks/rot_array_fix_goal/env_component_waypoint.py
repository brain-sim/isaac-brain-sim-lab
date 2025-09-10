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
        # Initialize waypoint tracking tensors for rot_array task (3 markers per group)
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
        """Generate a single random waypoint for each environment."""
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

            # Check robot distance if provided
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

            # Check bounds
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

    def generate_offset_waypoint(self, env_origins, base_waypoints, offset_distance):
        """Generate waypoints with random offset from base waypoints."""
        num_reset = base_waypoints.shape[0]
        max_attempts = 1000
        placed = torch.zeros(num_reset, dtype=torch.bool, device=self.env.device)
        waypoint_positions = torch.zeros((num_reset, 2), device=self.env.device)

        for _ in range(max_attempts):
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()

            if num_unplaced == 0:
                break

            # Generate random angles for offset direction
            random_angles = (
                torch.rand((num_unplaced,), device=self.env.device) * 2 * torch.pi
            )

            # Create random offset vectors
            offset_x = offset_distance * torch.cos(random_angles)
            offset_y = offset_distance * torch.sin(random_angles)
            random_offsets = torch.stack([offset_x, offset_y], dim=1)

            # Apply random offset to base waypoints
            unplaced_base = base_waypoints[unplaced_mask, :]
            candidate_positions = unplaced_base + random_offsets
            relative_positions = candidate_positions - env_origins[unplaced_mask]

            # Check bounds
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

    def generate_perpendicular_waypoint(
        self, env_origins, waypoint1, waypoint2, perp_offset
    ):
        """Generate waypoint at center of two waypoints with perpendicular offset."""
        num_reset = waypoint1.shape[0]
        max_attempts = 1000

        # Calculate center point between two waypoints
        center_point = (waypoint1 + waypoint2) / 2.0

        # Calculate direction vector and perpendicular vector
        direction_vector = waypoint2 - waypoint1
        perp_vector = torch.stack(
            [-direction_vector[:, 1], direction_vector[:, 0]], dim=1
        )

        # Normalize perpendicular vector
        perp_length = torch.norm(perp_vector, dim=1, keepdim=True)
        perp_length = torch.clamp(perp_length, min=1e-8)
        perp_unit_vector = perp_vector / perp_length

        # Try both directions for perpendicular offset
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

            # Check bounds
            bound_limit = self.env.cfg.room_size / 3.0
            valid_mask = (
                (unplaced_relative[:, 0] >= -bound_limit)
                & (unplaced_relative[:, 0] <= bound_limit)
                & (unplaced_relative[:, 1] >= -bound_limit)
                & (unplaced_relative[:, 1] <= bound_limit)
            )

            valid_indices = torch.where(unplaced_mask)[0][valid_mask]
            if len(valid_indices) > 0:
                placed[valid_indices] = True

            # For remaining unplaced, try opposite direction
            if not placed.all():
                unplaced_mask = ~placed
                candidate_positions[unplaced_mask] = (
                    center_point[unplaced_mask]
                    - perp_offset * perp_unit_vector[unplaced_mask]
                )
                relative_positions[unplaced_mask] = (
                    candidate_positions[unplaced_mask] - env_origins[unplaced_mask]
                )

        # Final waypoint positions
        waypoint_positions = candidate_positions.clone()

        # Fallback for any still unplaced
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
        fixed_second_position=None,
        fixed_third_position=None,
    ):
        """Generate a group of 3 waypoints following the pattern: 2nd→3rd→1st with guaranteed spatial relationships."""
        max_group_attempts = 100
        placed = torch.zeros(num_reset, dtype=torch.bool, device=self.env.device)
        first_waypoints = torch.zeros((num_reset, 2), device=self.env.device)
        second_waypoints = torch.zeros((num_reset, 2), device=self.env.device)
        third_waypoints = torch.zeros((num_reset, 2), device=self.env.device)

        for attempt in range(max_group_attempts):
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()
            
            if num_unplaced == 0:
                break
                
            # Generate second waypoint (fixed or random)
            if fixed_second_position is not None:
                unplaced_second = self.generate_fixed_waypoint(
                    env_origins[unplaced_mask], num_unplaced, fixed_second_position
                )
            else:
                unplaced_second = self.generate_random_waypoint(
                    env_origins[unplaced_mask], num_unplaced, 
                    robot_xy[unplaced_mask] if robot_xy is not None else None
                )

            # Generate third waypoint (fixed or with offset from second)
            if fixed_third_position is not None:
                unplaced_third = self.generate_fixed_waypoint(
                    env_origins[unplaced_mask], num_unplaced, fixed_third_position
                )
            else:
                unplaced_third = self.generate_offset_waypoint(
                    env_origins[unplaced_mask], unplaced_second, waypoint_offset
                )

            # Generate first waypoint at center with perpendicular offset
            unplaced_first = self.generate_perpendicular_waypoint(
                env_origins[unplaced_mask], unplaced_second, unplaced_third, perpendicular_offset
            )
            
            # Check if all three waypoints are within bounds
            bound_limit = self.env.cfg.room_size / 3.0
            
            first_relative = unplaced_first - env_origins[unplaced_mask]
            second_relative = unplaced_second - env_origins[unplaced_mask] 
            third_relative = unplaced_third - env_origins[unplaced_mask]
            
            first_valid = (
                (first_relative[:, 0] >= -bound_limit) &
                (first_relative[:, 0] <= bound_limit) &
                (first_relative[:, 1] >= -bound_limit) &
                (first_relative[:, 1] <= bound_limit)
            )
            
            second_valid = (
                (second_relative[:, 0] >= -bound_limit) &
                (second_relative[:, 0] <= bound_limit) &
                (second_relative[:, 1] >= -bound_limit) &
                (second_relative[:, 1] <= bound_limit)
            )
            
            third_valid = (
                (third_relative[:, 0] >= -bound_limit) &
                (third_relative[:, 0] <= bound_limit) &
                (third_relative[:, 1] >= -bound_limit) &
                (third_relative[:, 1] <= bound_limit)
            )
            
            # Only accept groups where all three waypoints are valid
            group_valid = first_valid & second_valid & third_valid
            valid_indices = torch.where(unplaced_mask)[0][group_valid]
            
            if len(valid_indices) > 0:
                first_waypoints[valid_indices] = unplaced_first[group_valid]
                second_waypoints[valid_indices] = unplaced_second[group_valid]
                third_waypoints[valid_indices] = unplaced_third[group_valid]
                placed[valid_indices] = True
        
        # Handle any remaining unplaced groups with fallback
        if not placed.all():
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()
            
            # Generate fallback positions
            random_positions = self.env.cfg.wall_config.get_random_valid_positions(
                num_unplaced * 3, device=self.env.device  # Get positions for all three waypoints
            )
            
            unplaced_origins = env_origins[unplaced_mask]
            fallback_first = random_positions[:num_unplaced, :2] + unplaced_origins
            fallback_second = random_positions[num_unplaced:2*num_unplaced, :2] + unplaced_origins
            fallback_third = random_positions[2*num_unplaced:, :2] + unplaced_origins
            
            first_waypoints[unplaced_mask] = fallback_first
            second_waypoints[unplaced_mask] = fallback_second
            third_waypoints[unplaced_mask] = fallback_third

        return first_waypoints, second_waypoints, third_waypoints

    def generate_fixed_waypoint(self, env_origins, num_reset, fixed_position):
        """Generate fixed waypoint at specified position for all environments."""
        # Create fixed position tensor for all environments
        fixed_waypoint = fixed_position.unsqueeze(0).expand(
            num_reset, -1
        )  # (num_reset, 2)

        # Add environment origins to get absolute positions
        absolute_positions = fixed_waypoint + env_origins

        # Check if positions are within bounds (-room_size/3 to +room_size/3 in both x and y)
        relative_positions = absolute_positions - env_origins
        bound_limit = self.env.cfg.room_size / 3.0
        valid_mask = (
            (relative_positions[:, 0] >= -bound_limit)
            & (relative_positions[:, 0] <= bound_limit)
            & (relative_positions[:, 1] >= -bound_limit)
            & (relative_positions[:, 1] <= bound_limit)
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

    def generate_waypoints(
        self, env_ids, robot_poses, waypoint_offset=None, perpendicular_offset=None
    ):
        """Generate one group of waypoints for reset environments."""
        num_reset = len(env_ids)
        env_origins = self.env.scene.env_origins[env_ids, :2]
        robot_xy = robot_poses[:, :2]

        if waypoint_offset is None:
            waypoint_offset = 5.0

        if perpendicular_offset is None:
            perpendicular_offset = 3.0

        # Use fixed positions for second and third waypoints
        fixed_second_position = torch.tensor([5.0, 1.0], device=self.env.device)
        fixed_third_position = torch.tensor([1.0, 2.0], device=self.env.device)

        # # Scaled
        # perpendicular_offset = 6.0
        # fixed_second_position = torch.tensor([5.0, 1.0], device=self.env.device)
        # fixed_third_position = torch.tensor([-3.0, 3.0], device=self.env.device)

        # # Translated
        # perpendicular_offset = 3.0
        # fixed_second_position = torch.tensor([-1.0, 3.0], device=self.env.device)
        # fixed_third_position = torch.tensor([-5.0, 4.0], device=self.env.device)

        # # Rotated (90 degrees clockwise)
        # perpendicular_offset = 3.0
        # fixed_second_position = torch.tensor([1.0, -5.0], device=self.env.device)
        # fixed_third_position = torch.tensor([2.0, -1.0], device=self.env.device)

        # # Rotated (45 degrees clockwise) and scaled
        # perpendicular_offset = 6.0
        # fixed_second_position = torch.tensor([1.0, -5.0], device=self.env.device)
        # fixed_third_position = torch.tensor([3.0, 3.0], device=self.env.device)

        # Generate one group of waypoints (3 markers)
        first_wp, second_wp, third_wp = self.generate_waypoint_group(
            env_origins,
            num_reset,
            waypoint_offset,
            perpendicular_offset,
            robot_xy,
            fixed_second_position,
            fixed_third_position,
        )

        waypoint_positions = torch.zeros(
            (num_reset, self.env.cfg.num_markers_per_group, 2), device=self.env.device
        )
        waypoint_positions[:, 0, :] = first_wp  # Target waypoint
        waypoint_positions[:, 1, :] = second_wp  # Obstacle waypoint
        waypoint_positions[:, 2, :] = third_wp  # Obstacle waypoint

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

        # Set obstacles (indices 1 and 2) to 3 (cyan)
        marker_indices[:, 1] = 0
        marker_indices[:, 2] = 3

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
