from __future__ import annotations

import torch

class EnvComponentReward:
    """Component responsible for reward calculation."""

    def __init__(self, env):
        self.env = env

    def post_env_init(self):
        # Load reward parameters from config
        self.goal_reached_bonus = self.env.cfg.goal_reached_bonus
        self.laziness_penalty_weight = self.env.cfg.laziness_penalty_weight
        self.wall_penalty_weight = self.env.cfg.wall_penalty_weight
        self.linear_speed_weight = self.env.cfg.linear_speed_weight
        self.avoid_penalty_weight = self.env.cfg.avoid_penalty_weight
        self.fast_goal_reached_bonus = self.env.cfg.fast_goal_reached_weight
        self.heading_coefficient = self.env.cfg.heading_coefficient
        self.heading_progress_weight = self.env.cfg.heading_progress_weight

    def get_rewards(self) -> dict[str, torch.Tensor]:
        """Calculate all reward components based on check results."""

        goal_reached = self.env.env_component_objective.check_results["goal_reached"]
        env_has_collision = self.env.env_component_objective.check_results[
            "env_has_collision"
        ]

        # Target heading reward
        target_heading_rew = self.heading_progress_weight * torch.exp(
            -torch.abs(self.env.env_component_observation.target_heading_error)
            / self.heading_coefficient
        )
        target_heading_rew = torch.nan_to_num(
            target_heading_rew, posinf=0.0, neginf=0.0
        )

        # Goal reached reward (using passed check result)
        goal_reached_reward = self.goal_reached_bonus * torch.nan_to_num(
            torch.where(
                goal_reached,
                1.0,
                torch.zeros(self.env.num_envs, device=self.env.device),
            ),
            posinf=0.0,
            neginf=0.0,
        )

        # Avoid collision penalty (using passed check result)
        avoid_penalty = torch.zeros(
            (self.env.num_envs), device=self.env.device, dtype=torch.float32
        )
        avoid_penalty[env_has_collision] = self.avoid_penalty_weight

        # Fast goal reached reward (using data from objective component)
        assert (
            self.env.env_component_objective._previous_waypoint_reached_step[
                goal_reached
            ]
            < self.env.episode_length_buf[goal_reached]
        ).all(), "Previous waypoint reached step is greater than episode length"

        k = torch.log(
            torch.tensor(self.fast_goal_reached_bonus, device=self.env.device)
        ) / (self.env.max_episode_length - 1)
        steps_taken = (
            self.env.episode_length_buf
            - self.env.env_component_objective._previous_waypoint_reached_step
        )
        fast_goal_reached_reward = torch.where(
            goal_reached,
            self.fast_goal_reached_bonus * torch.exp(-k * (steps_taken - 1)),
            torch.zeros_like(
                self.env.env_component_objective._previous_waypoint_reached_step
            ),
        )
        fast_goal_reached_reward = torch.clamp(
            fast_goal_reached_reward, min=0.0, max=self.fast_goal_reached_bonus
        )

        # Laziness penalty (using accumulated laziness from objective)
        laziness_penalty = torch.nan_to_num(
            -self.laziness_penalty_weight
            * torch.log1p(self.env.env_component_objective._accumulated_laziness),
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
        linear_speed = torch.norm(
            self.env.env_component_robot.robot.data.root_lin_vel_b[:, :2], dim=-1
        )
        linear_speed_reward = self.linear_speed_weight * torch.nan_to_num(
            linear_speed,
            posinf=0.0,
            neginf=0.0,
        )

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
        # No state to reset - all tracking moved to objective component
        pass
