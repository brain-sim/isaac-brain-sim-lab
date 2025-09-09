import torch


class DerivedEnvComponentObjective:
    """Component responsible for objective management for single_fix_goal task."""

    def __init__(self, env):
        self.env = env

    def post_env_init(self):
        self.approach_position_tolerance = self.env.cfg.approach_position_tolerance
        self.avoid_position_tolerance = self.env.cfg.avoid_position_tolerance

        # Initialize tracking variables for waypoint progress
        self._previous_waypoint_reached_step = torch.zeros(
            (self.env.num_envs,), device=self.env.device, dtype=torch.int32
        )
        self._episode_obstacle_collisions = torch.zeros(
            (self.env.num_envs), device=self.env.device, dtype=torch.int32
        )
        self._obstacle_hit_this_step = torch.zeros(
            (self.env.num_envs), device=self.env.device, dtype=torch.bool
        )

        # Initialize laziness tracking
        self._accumulated_laziness = torch.zeros(
            (self.env.num_envs), device=self.env.device, dtype=torch.float32
        )

        # Book keeping for checks
        self.check_results = {"goal_reached": None, "env_has_collision": None}

    def check_goal_approached(self) -> torch.Tensor:
        return (
            self.env.env_component_observation._robot_target_distance
            < self.approach_position_tolerance
        )

    def check_obstacle_collision(self) -> torch.Tensor:
        """Check if robot collides with obstacle waypoint (index 1 in current group)."""
        robot_positions = self.env.env_component_robot.robot.data.root_pos_w[:, :2]

        # Get obstacle position (index 1 in current group)
        obstacle_positions = self.env.env_component_waypoint._target_positions[
            :, 1, :
        ]  # Index 1

        distances = torch.norm(robot_positions - obstacle_positions, dim=1)
        env_has_collision = distances < self.avoid_position_tolerance
        return env_has_collision

    def check_task_completed(self) -> torch.Tensor:
        """Check if all groups have been completed."""
        return (
            self.env.env_component_waypoint._target_group_index
            >= self.env.cfg.num_groups
        )

    def check_step_conditions(self):
        """Check all conditions for the current step - pure checking without side effects."""
        goal_reached = self.check_goal_approached()
        env_has_collision = self.check_obstacle_collision()
        task_completed = self.check_task_completed()

        self.check_results["goal_reached"] = goal_reached
        self.check_results["env_has_collision"] = env_has_collision
        self.check_results["task_completed"] = task_completed

        return self.check_results

    def update_progress_and_bookkeeping(self):
        """Update waypoint progress and collision tracking based on check results."""

        goal_reached = self.check_results["goal_reached"]

        # When goal is reached, increment group index and generate new group if not completed
        if goal_reached.any():
            goal_reached_envs = torch.where(goal_reached)[0]

            # Increment group index for environments that reached the goal
            self.env.env_component_waypoint._target_group_index[goal_reached_envs] += 1
            self.env.env_component_waypoint._episode_groups_passed[
                goal_reached_envs
            ] += 1

            # Check which environments need new groups (haven't completed all groups yet)
            incomplete_mask = (
                self.env.env_component_waypoint._target_group_index[goal_reached_envs]
                < self.env.cfg.num_groups
            )
            envs_needing_new_group = goal_reached_envs[incomplete_mask]

            if len(envs_needing_new_group) > 0:
                # For fixed goal task, generate same positions
                robot_poses = self.env.env_component_robot.robot.data.root_pos_w[
                    envs_needing_new_group
                ]
                self.env.env_component_waypoint.generate_new_group(
                    envs_needing_new_group, robot_poses
                )
            
            # Teleport robot to new start position for the new group with collision avoidance
            self._reset_robot_with_collision_check(goal_reached_envs)

        # Track collision state
        self._obstacle_hit_this_step[self.check_results["env_has_collision"]] = True
        self._episode_obstacle_collisions += self.check_results[
            "env_has_collision"
        ].int()

        # Track task completion state
        self.env.task_completed = self.check_results["task_completed"]

        # Update previous waypoint reached step tracking
        self._previous_waypoint_reached_step = torch.where(
            self.check_results["goal_reached"],
            self.env.episode_length_buf,
            self._previous_waypoint_reached_step,
        )

        # Update waypoint visualization
        self.env.env_component_waypoint.update_waypoint_visualization()

        # Update laziness tracking
        linear_speed = torch.norm(
            self.env.env_component_robot.robot.data.root_lin_vel_b[:, :2], dim=-1
        )
        current_laziness = torch.where(
            linear_speed < self.env.cfg.laziness_threshold,
            torch.ones_like(linear_speed),
            torch.zeros_like(linear_speed),
        )

        # Update accumulated laziness with decay
        self._accumulated_laziness = (
            self._accumulated_laziness * self.env.cfg.laziness_decay
            + current_laziness * (1 - self.env.cfg.laziness_decay)
        )
        self._accumulated_laziness = torch.clamp(
            self._accumulated_laziness, 0.0, self.env.cfg.max_laziness
        )

        # Reset accumulated laziness when reaching waypoint
        self._accumulated_laziness = torch.where(
            self.check_results["goal_reached"],
            torch.zeros_like(self._accumulated_laziness),
            self._accumulated_laziness,
        )

    def reset(self, env_ids):
        """Reset objective tracking for specified environments."""
        if hasattr(self, "_episode_obstacle_collisions"):
            self._episode_obstacle_collisions[env_ids] = 0
        if hasattr(self, "_previous_waypoint_reached_step"):
            self._previous_waypoint_reached_step[env_ids] = 0
        if hasattr(self, "_accumulated_laziness"):
            self._accumulated_laziness[env_ids] = 0.0

    def _reset_robot_with_collision_check(self, env_ids):
        """Reset robot positions while ensuring they don't collide with waypoints."""
        max_reset_attempts = 1000  # Maximum attempts to find valid positions
        
        for attempt in range(max_reset_attempts):

            self.env.env_component_robot.reset(env_ids)
            robot_positions = self.env.env_component_robot.robot.data.root_pos_w[env_ids, :2]
            
            waypoint_positions = self.env.env_component_waypoint._target_positions[env_ids]  # Shape: (num_envs, num_markers, 2)
            robot_pos_expanded = robot_positions.unsqueeze(1)  # Shape: (num_envs, 1, 2)
            distances = torch.norm(robot_pos_expanded - waypoint_positions, dim=2)  # Shape: (num_envs, num_markers)
            
            safety_margin = 1.0  # Additional safety distance
            min_distance_per_env = distances.min(dim=1)[0]  # Minimum distance for each environment
            collision_mask = min_distance_per_env < (self.avoid_position_tolerance + safety_margin)
            
            if not collision_mask.any():
                break
                
            if attempt == max_reset_attempts - 1:
                num_colliding = collision_mask.sum().item()
                print(f"Warning: After {max_reset_attempts} attempts, {num_colliding} robots still too close to waypoints")
                break
                
            env_ids = env_ids[collision_mask]
            if len(env_ids) == 0:
                break
