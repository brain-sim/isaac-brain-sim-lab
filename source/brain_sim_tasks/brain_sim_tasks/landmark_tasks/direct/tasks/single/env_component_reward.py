from __future__ import annotations

import torch

from ...components.env_component_reward import EnvComponentReward

class DerivedEnvComponentReward(EnvComponentReward):
    """Component responsible for reward calculation."""
    
    def __init__(self, env):
        self.env = env
    
    def post_env_init(self):

        self._accumulated_laziness = torch.zeros(
            (self.env.num_envs), device=self.env.device, dtype=torch.float32
        )
        self._previous_waypoint_reached_step = torch.zeros(
            (self.env.num_envs,), device=self.env.device, dtype=torch.int32
        )
        self._episode_avoid_collisions = torch.zeros(
            (self.env.num_envs), device=self.env.device, dtype=torch.int32
        )
        self._avoid_goal_hit_this_step = torch.zeros(
            (self.env.num_envs), device=self.env.device, dtype=torch.bool
        )

        # Load reward parameters from config
        self.position_tolerance = self.env.cfg.position_tolerance
        self.goal_reached_bonus = self.env.cfg.goal_reached_bonus
        self.laziness_penalty_weight = self.env.cfg.laziness_penalty_weight
        self.laziness_decay = self.env.cfg.laziness_decay
        self.laziness_threshold = self.env.cfg.laziness_threshold
        self.max_laziness = self.env.cfg.max_laziness
        self.wall_penalty_weight = self.env.cfg.wall_penalty_weight
        self.linear_speed_weight = self.env.cfg.linear_speed_weight
        self.avoid_penalty_weight = self.env.cfg.avoid_penalty_weight
        self.fast_goal_reached_bonus = self.env.cfg.fast_goal_reached_weight
        self.avoid_goal_position_tolerance = self.env.cfg.avoid_goal_position_tolerance
        self.heading_coefficient = self.env.cfg.heading_coefficient
        self.heading_progress_weight = self.env.cfg.heading_progress_weight

    def check_avoid_goal_collision(self) -> torch.Tensor:
        """Check if robot collides with future waypoints."""
        robot_positions = self.env.env_component_robot.robot.data.root_pos_w[:, :2]

        goal_indices = torch.arange(self.env.cfg.num_goals, device=self.env.device).unsqueeze(0)
        target_indices = self.env.env_component_waypoint._target_index.unsqueeze(1)
        future_waypoint_mask = goal_indices > target_indices

        robot_pos_expanded = robot_positions.unsqueeze(1)
        distances = torch.norm(
            robot_pos_expanded - self.env.env_component_waypoint._target_positions, dim=2
        )

        future_distances = distances * future_waypoint_mask.float()
        future_distances = torch.where(
            future_waypoint_mask,
            future_distances,
            torch.full_like(future_distances, float("inf")),
        )

        collision_mask = future_distances < self.avoid_goal_position_tolerance
        env_has_collision = collision_mask.any(dim=1)
        return env_has_collision

    def get_rewards(self) -> dict[str, torch.Tensor]:
        """Calculate all reward components."""
        # Check if goal is reached
        goal_reached = self.env.env_component_observation._position_error < self.position_tolerance
        
        # Target heading reward
        target_heading_rew = self.heading_progress_weight * torch.exp(
            -torch.abs(self.env.env_component_observation.target_heading_error) / self.heading_coefficient
        )
        target_heading_rew = torch.nan_to_num(target_heading_rew, posinf=0.0, neginf=0.0)
        
        # Goal reached reward
        goal_reached_reward = self.goal_reached_bonus * torch.nan_to_num(
            torch.where(
                goal_reached,
                1.0,
                torch.zeros_like(self.env.env_component_observation._position_error),
            ),
            posinf=0.0,
            neginf=0.0,
        )

        # Avoid goal collision penalty
        env_has_collision = self.check_avoid_goal_collision()
        avoid_penalty = torch.zeros(
            (self.env.num_envs), device=self.env.device, dtype=torch.float32
        )
        avoid_penalty[env_has_collision] = self.avoid_penalty_weight
        self._avoid_goal_hit_this_step[env_has_collision] = True
        self._episode_avoid_collisions += env_has_collision.int()

        # Update waypoint progress
        self.env.env_component_waypoint._target_index = self.env.env_component_waypoint._target_index + goal_reached
        self.env.env_component_waypoint._episode_waypoints_passed += goal_reached.int()
        
        # Check task completion
        self.env.task_completed = self.env.env_component_waypoint._target_index > (self.env.cfg.num_goals - 2)
        self.env.env_component_waypoint._target_index = self.env.env_component_waypoint._target_index % self.env.cfg.num_goals

        # Fast goal reached reward
        assert (
            self._previous_waypoint_reached_step[goal_reached]
            < self.env.episode_length_buf[goal_reached]
        ).all(), "Previous waypoint reached step is greater than episode length"
        
        k = torch.log(
            torch.tensor(self.fast_goal_reached_bonus, device=self.env.device)
        ) / (self.env.max_episode_length - 1)
        steps_taken = self.env.episode_length_buf - self._previous_waypoint_reached_step
        fast_goal_reached_reward = torch.where(
            goal_reached,
            self.fast_goal_reached_bonus * torch.exp(-k * (steps_taken - 1)),
            torch.zeros_like(self._previous_waypoint_reached_step),
        )
        fast_goal_reached_reward = torch.clamp(
            fast_goal_reached_reward, min=0.0, max=self.fast_goal_reached_bonus
        )
        self._previous_waypoint_reached_step = torch.where(
            goal_reached,
            self.env.episode_length_buf,
            self._previous_waypoint_reached_step,
        )
        
        # Laziness penalty
        linear_speed = torch.norm(
            self.env.env_component_robot.robot.data.root_lin_vel_b[:, :2], dim=-1
        )
        current_laziness = torch.where(
            linear_speed < self.laziness_threshold,
            torch.ones_like(linear_speed),
            torch.zeros_like(linear_speed),
        )

        # Update accumulated laziness with decay
        self._accumulated_laziness = (
            self._accumulated_laziness * self.laziness_decay
            + current_laziness * (1 - self.laziness_decay)
        )
        self._accumulated_laziness = torch.clamp(
            self._accumulated_laziness, 0.0, self.max_laziness
        )

        # Reset accumulated laziness when reaching waypoint
        self._accumulated_laziness = torch.where(
            goal_reached,
            torch.zeros_like(self._accumulated_laziness),
            self._accumulated_laziness,
        )
        
        laziness_penalty = torch.nan_to_num(
            -self.laziness_penalty_weight * torch.log1p(self._accumulated_laziness),
            posinf=0.0,
            neginf=0.0,
        )

        # Wall distance penalty
        min_wall_dist = self.env.env_component_robot.get_distance_to_walls()
        danger_distance = 0.5
        wall_penalty = torch.nan_to_num(
            torch.where(
                min_wall_dist > danger_distance,
                torch.zeros_like(min_wall_dist),
                self.wall_penalty_weight
                * torch.exp(1.0 - min_wall_dist / danger_distance),
            ),
            posinf=0.0,
            neginf=0.0,
        )
        
        # Linear speed reward
        linear_speed_reward = self.linear_speed_weight * torch.nan_to_num(
            linear_speed,
            posinf=0.0,
            neginf=0.0,
        )

        # Update waypoint visualization
        self.env.env_component_waypoint.update_waypoint_visualization()

        return {
            "Episode_Reward/goal_reached_reward": goal_reached_reward,
            "Episode_Reward/linear_speed_reward": linear_speed_reward,
            "Episode_Reward/laziness_penalty": laziness_penalty,
            "Episode_Reward/wall_penalty": wall_penalty,
            "Episode_Reward/fast_goal_reached_reward": fast_goal_reached_reward,
            "Episode_Reward/avoid_penalty": avoid_penalty,
            "Episode_Reward/target_heading_rew": target_heading_rew,
        }

    def reset(self, env_ids):
        """Reset reward tracking for specified environments."""
        if hasattr(self, "_episode_avoid_collisions"):
            self._episode_avoid_collisions[env_ids] = 0
        if hasattr(self, "_previous_waypoint_reached_step"):
            self._previous_waypoint_reached_step[env_ids] = 0
