from __future__ import annotations

import os

import torch
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from termcolor import colored
from brain_sim_assets import BRAIN_SIM_ASSETS_ROBOTS_DATA_DIR

from .nav_env import NavEnv
from .linear_maze_env_cfg import LinearMazeEnvCfg
from brain_sim_assets.robots.spot import bsSpotLowLevelPolicyVanilla


class LinearMazeEnv(NavEnv):
    cfg: LinearMazeEnvCfg  
    ACTION_SCALE = 0.2  

    def __init__(
        self,
        cfg: LinearMazeEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self._goal_reached = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )
        self.task_completed = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.bool
        )
        self._target_positions = torch.zeros(
            (self.num_envs, self._num_goals, 2),
            device=self.device,
            dtype=torch.float32,
        )
        self._markers_pos = torch.zeros(
            (self.num_envs, self._num_goals, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self.env_spacing = self.cfg.env_spacing
        self._target_index = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )
        self._accumulated_laziness = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float32
        )
        self._episode_avoid_collisions = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )
        self._avoid_goal_hit_this_step = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.bool
        )

    def _setup_robot_dof_idx(self):
        self._dof_idx, _ = self.robot.find_joints(self.cfg.dof_name)

    def _setup_config(self):
        policy_file_path = os.path.join(
            BRAIN_SIM_ASSETS_ROBOTS_DATA_DIR,
            self.cfg.policy_file_path,
        )
        print(
            colored(
                f"[INFO] Loading policy from {policy_file_path}",
                "magenta",
                attrs=["bold"],
            )
        )
        self.policy = bsSpotLowLevelPolicyVanilla(policy_file_path)
        self._low_level_previous_action = torch.zeros(
            (self.num_envs, 12), device=self.device, dtype=torch.float32
        )
        self._previous_action = torch.zeros(
            (self.num_envs, self.cfg.action_space),
            device=self.device,
            dtype=torch.float32,
        )
        self._previous_waypoint_reached_step = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.int32
        )
        self.position_tolerance = self.cfg.position_tolerance
        self.goal_reached_bonus = self.cfg.goal_reached_bonus
        self.laziness_penalty_weight = self.cfg.laziness_penalty_weight
        self.laziness_decay = (
            self.cfg.laziness_decay
        )  # How much previous laziness carries over
        self.laziness_threshold = (
            self.cfg.laziness_threshold
        )  # Speed threshold for considering "lazy"
        self.max_laziness = (
            self.cfg.max_laziness
        )  # Cap on accumulated laziness to prevent extreme penalties
        self.wall_penalty_weight = self.cfg.wall_penalty_weight
        self.linear_speed_weight = self.cfg.linear_speed_weight
        self.throttle_scale = self.cfg.throttle_scale
        self._actions = torch.zeros(
            (self.num_envs, self.cfg.action_space),
            device=self.device,
            dtype=torch.float32,
        )
        self.steering_scale = self.cfg.steering_scale
        self.throttle_max = self.cfg.throttle_max
        self.steering_max = self.cfg.steering_max
        self._default_pos = self.robot.data.default_joint_pos.clone()
        self._smoothing_factor = torch.tensor([0.75, 0.3, 0.3], device=self.device)
        self.max_episode_length_buf = torch.full(
            (self.num_envs,), self.max_episode_length, device=self.device
        )

        self.avoid_penalty_weight = self.cfg.avoid_penalty_weight
        self.fast_goal_reached_bonus = self.cfg.fast_goal_reached_weight
        self.avoid_goal_position_tolerance = self.cfg.avoid_goal_position_tolerance

        self.heading_coefficient = self.cfg.heading_coefficient
        self.heading_progress_weight = self.cfg.heading_progress_weight

        self.termination_on_avoid_goal_collision = (
            self.cfg.termination_on_avoid_goal_collision
        )
        self.termination_on_goal_reached = self.cfg.termination_on_goal_reached
        self.termination_on_vehicle_flip = self.cfg.termination_on_vehicle_flip
        self.termination_on_stuck = self.cfg.termination_on_stuck

    def _setup_camera(self):
        camera_prim_path = "/World/envs/env_.*/Robot/body/Camera"
        pinhole_cfg = PinholeCameraCfg(
            focal_length=16.0,
            horizontal_aperture=32.0,
            vertical_aperture=32.0,
            focus_distance=1.0,
            clipping_range=(0.01, 1000.0),
            lock_camera=True,
        )
        camera_cfg = TiledCameraCfg(
            prim_path=camera_prim_path,
            update_period=self.step_dt,
            height=self.cfg.img_size[1],
            width=self.cfg.img_size[2],
            data_types=["rgb"],
            spawn=pinhole_cfg,
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.25, 0.0, 0.25),  # At the head, adjust as needed
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )
        self.camera = TiledCamera(camera_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = (
            self._smoothing_factor * actions
            + (1 - self._smoothing_factor) * self._previous_action
        )
        self._previous_action = actions.clone()
        self._actions = actions.clone()
        self._actions = torch.nan_to_num(self._actions, 0.0)
        self._actions[:, 0] = self._actions[:, 0] * self.throttle_scale
        self._actions[:, 1:] = self._actions[:, 1:] * self.steering_scale
        self._actions[:, 0] = torch.clamp(
            self._actions[:, 0], min=0.0, max=self.throttle_max
        )
        self._actions[:, 1:] = torch.clamp(
            self._actions[:, 1:], min=-self.steering_max, max=self.steering_max
        )

    def _apply_action(self) -> None:
        # --- Vectorized low-level Spot policy call for all environments ---
        # Gather all required robot state as torch tensors
        # TODO: Replace the following with actual command logic per environment
        default_pos = self._default_pos.clone()  # [num_envs, 12]
        # The following assumes your robot exposes these as torch tensors of shape [num_envs, ...]
        lin_vel_I = self.robot.data.root_lin_vel_w  # [num_envs, 3]
        ang_vel_I = self.robot.data.root_ang_vel_w  # [num_envs, 3]
        q_IB = self.robot.data.root_quat_w  # [num_envs, 4]
        joint_pos = self.robot.data.joint_pos  # [num_envs, 12]
        joint_vel = self.robot.data.joint_vel  # [num_envs, 12]
        # Compute actions for all environments
        actions = self.policy.get_action(
            lin_vel_I,
            ang_vel_I,
            q_IB,
            self._actions,
            self._low_level_previous_action,
            default_pos,
            joint_pos,
            joint_vel,
        )
        # Update previous action buffer
        self._low_level_previous_action = actions.detach()
        # Scale and offset actions as in Spot reference policy
        joint_positions = self._default_pos + actions * self.ACTION_SCALE
        # Apply joint position targets directly
        self.robot.set_joint_position_target(joint_positions)

    def _get_image_obs(self) -> torch.Tensor:
        image_obs = self.camera.data.output["rgb"].float().permute(0, 3, 1, 2) / 255.0
        image_obs = image_obs.reshape(self.num_envs, -1)
        return image_obs

    def _get_state_obs(self, image_obs) -> torch.Tensor:
        return torch.cat(
            (
                image_obs,
                self._actions[:, 0].unsqueeze(dim=1),
                self._actions[:, 1].unsqueeze(dim=1),
                self._actions[:, 2].unsqueeze(dim=1),
                self._get_distance_to_walls().unsqueeze(dim=1),  # Add wall distance
            ),
            dim=-1,
        )

    def _check_stuck_termination(self, max_steps: int = 300) -> torch.Tensor:
        """Early termination if robot is stuck/wandering without progress"""
        # If no goal reached in last max_steps and barely moving, terminate
        steps_since_goal = (
            self.episode_length_buf - self._previous_waypoint_reached_step
        )
        stuck_too_long = steps_since_goal > max_steps
        return stuck_too_long

    def _check_avoid_goal_collision(self) -> torch.Tensor:
        """Check if the robot has collided with the avoid goal"""
        # Check for avoid goal collisions (future waypoints) - VECTORIZED VERSION
        robot_positions = self.robot.data.root_pos_w[:, :2]  # (num_envs, 2)

        # Create mask for future waypoints (goals with index > current target index)
        goal_indices = torch.arange(self._num_goals, device=self.device).unsqueeze(
            0
        )  # (1, num_goals)
        target_indices = self._target_index.unsqueeze(1)  # (num_envs, 1)
        future_waypoint_mask = goal_indices > target_indices  # (num_envs, num_goals)

        # Calculate distances from each robot to all waypoints
        # robot_positions: (num_envs, 2) -> (num_envs, 1, 2)
        # _target_positions: (num_envs, num_goals, 2)
        robot_pos_expanded = robot_positions.unsqueeze(1)  # (num_envs, 1, 2)
        distances = torch.norm(
            robot_pos_expanded - self._target_positions, dim=2
        )  # (num_envs, num_goals)

        # Apply future waypoint mask and check for collisions
        future_distances = (
            distances * future_waypoint_mask.float()
        )  # Zero out non-future waypoints
        future_distances = torch.where(
            future_waypoint_mask,
            future_distances,
            torch.full_like(future_distances, float("inf")),  # Set non-future to inf
        )

        # Check which environments have collisions with future waypoints
        collision_mask = (
            future_distances < self.avoid_goal_position_tolerance
        )  # (num_envs, num_goals)
        env_has_collision = collision_mask.any(dim=1)  # (num_envs,)
        return env_has_collision

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        goal_reached = self._position_error < self.position_tolerance
        target_heading_rew = self.heading_progress_weight * torch.exp(
            -torch.abs(self.target_heading_error) / self.heading_coefficient
        )
        target_heading_rew = torch.nan_to_num(
            target_heading_rew, posinf=0.0, neginf=0.0
        )
        goal_reached_reward = self.goal_reached_bonus * torch.nan_to_num(
            torch.where(
                goal_reached,
                1.0,
                torch.zeros_like(self._position_error),
            ),
            posinf=0.0,
            neginf=0.0,
        )

        # Apply penalties for environments with collisions
        env_has_collision = self._check_avoid_goal_collision()
        avoid_penalty = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float32
        )
        avoid_penalty[env_has_collision] = self.avoid_penalty_weight
        self._avoid_goal_hit_this_step[env_has_collision] = True
        self._episode_avoid_collisions += env_has_collision.int()

        self._target_index = self._target_index + goal_reached
        self._episode_waypoints_passed += goal_reached.int()
        self.task_completed = self._target_index > (self._num_goals - 1)
        self._target_index = self._target_index % self._num_goals
        assert (
            self._previous_waypoint_reached_step[goal_reached]
            < self.episode_length_buf[goal_reached]
        ).all(), "Previous waypoint reached step is greater than episode length"
        # Compute k using torch.log
        k = torch.log(
            torch.tensor(self.fast_goal_reached_bonus, device=self.device)
        ) / (self.max_episode_length - 1)
        steps_taken = self.episode_length_buf - self._previous_waypoint_reached_step
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
            self.episode_length_buf,
            self._previous_waypoint_reached_step,
        )
        # Calculate current laziness based on speed
        linear_speed = torch.norm(
            self.robot.data.root_lin_vel_b[:, :2], dim=-1
        )  # XY plane velocity
        current_laziness = torch.where(
            linear_speed < self.laziness_threshold,
            torch.ones_like(linear_speed),  # Count as lazy
            torch.zeros_like(linear_speed),  # Not lazy
        )

        # Update accumulated laziness with decay
        self._accumulated_laziness = (
            self._accumulated_laziness * self.laziness_decay
            + current_laziness * (1 - self.laziness_decay)
        )
        # Clamp to prevent extreme values
        self._accumulated_laziness = torch.clamp(
            self._accumulated_laziness, 0.0, self.max_laziness
        )

        # Reset accumulated laziness when reaching waypoint
        self._accumulated_laziness = torch.where(
            goal_reached,
            torch.zeros_like(self._accumulated_laziness),
            self._accumulated_laziness,
        )
        # Calculate laziness penalty using log
        laziness_penalty = torch.nan_to_num(
            -self.laziness_penalty_weight * torch.log1p(self._accumulated_laziness),
            posinf=0.0,
            neginf=0.0,
        )  # log1p(x) = log(1 + x)

        # Add wall distance penalty
        min_wall_dist = self._get_distance_to_walls()
        danger_distance = (
            0.1
        )  # Distance at which to start penalizing
        wall_penalty = torch.nan_to_num(
            torch.where(
                min_wall_dist > danger_distance,
                torch.zeros_like(min_wall_dist),
                self.wall_penalty_weight
                * torch.exp(
                    1.0 - min_wall_dist / danger_distance
                ),  # Exponential penalty
            ),
            posinf=0.0,
            neginf=0.0,
        )
        linear_speed_reward = self.linear_speed_weight * torch.nan_to_num(
            linear_speed,
            posinf=0.0,
            neginf=0.0,
        )
        # Create a tensor of 0s (future), 1s (current), and 2s (completed)
        marker_indices = torch.zeros(
            (self.num_envs, self._num_goals),
            device=self.device,
            dtype=torch.long,
        )
        # Set current targets to 1 (green)
        marker_indices[
            torch.arange(self.num_envs, device=self.device), self._target_index
        ] = 1
        # Set completed targets to 2 (invisible)
        # Create a mask for completed targets
        target_mask = (self._target_index.unsqueeze(1) > 0) & (
            torch.arange(self._num_goals, device=self.device)
            < self._target_index.unsqueeze(1)
        )
        marker_indices[target_mask] = 2
        marker_indices = marker_indices.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)
        return {
            "Episode_Reward/goal_reached_reward": goal_reached_reward,
            "Episode_Reward/linear_speed_reward": linear_speed_reward,
            "Episode_Reward/laziness_penalty": laziness_penalty,
            "Episode_Reward/wall_penalty": wall_penalty,
            "Episode_Reward/fast_goal_reached_reward": fast_goal_reached_reward,
            "Episode_Reward/avoid_penalty": avoid_penalty,
            "Episode_Reward/target_heading_rew": target_heading_rew,
        }
