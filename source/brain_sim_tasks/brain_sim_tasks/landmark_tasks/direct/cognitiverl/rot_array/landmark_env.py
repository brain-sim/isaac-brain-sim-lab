from __future__ import annotations

import copy
import os
from collections import defaultdict
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.utils import math
from termcolor import colored
from brain_sim_assets import BRAIN_SIM_ASSETS_ROBOTS_DATA_DIR

from .landmark_env_cfg import LandmarkEnvCfg
from brain_sim_assets.robots.spot import bsSpotLowLevelPolicyVanilla


class LandmarkEnv(DirectRLEnv):
    cfg: LandmarkEnvCfg
    ACTION_SCALE = 0.2

    def __init__(
        self,
        cfg: LandmarkEnvCfg,
        render_mode: str | None = None,
        debug: bool = False,
        max_total_steps: int | None = None,
        play_mode: bool = False,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.cfg.wall_config.update_device(self.device)

        self._setup_robot_dof_idx()
        self._goal_reached = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.task_completed = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        self._target_positions = torch.zeros(
            (self.num_envs, self.cfg.num_goals, 2), device=self.device, dtype=torch.float32
        )
        self._markers_pos = torch.zeros(
            (self.num_envs, self.cfg.num_goals, 3), device=self.device, dtype=torch.float32
        )
        self._target_index = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )
        self._accumulated_laziness = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float32
        )
        self._episode_waypoints_passed = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )
        self._episode_reward_buf = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float32
        )
        self._episode_reward_infos_buf = {}
        self._metrics_infos_buf = {}
        self._terminations_infos_buf = defaultdict(
            lambda: torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        )

        self._step_rewards_buffer = []
        self._episode_step_rewards = []

        self._debug = debug
        self._setup_config()
        self.max_total_steps = max_total_steps
        self.play_mode = play_mode

        self.env_spacing = self.cfg.env_spacing
        self._episode_avoid_collisions = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )
        self._avoid_goal_hit_this_step = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.bool
        )

    def _setup_plane(self):
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                size=(
                    4096 * 40.0,
                    4096 * 40.0,
                ),
                color=(0.2, 0.2, 0.2),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.cfg.static_friction,
                    dynamic_friction=self.cfg.dynamic_friction,
                    restitution=0.0,
                ),
            ),
        )

    def _setup_scene(self):
        self._setup_plane()

        self.robot = Articulation(self.cfg.robot_cfg)
        self._setup_camera()
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        self.object_state = []

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot

        self.wall_height = self.cfg.wall_height
        self.wall_position = (self.cfg.room_size - self.cfg.wall_thickness) / 2

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

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
                self._get_distance_to_walls().unsqueeze(dim=1),
            ),
            dim=-1,
        )

    def _get_observations(self) -> dict:
        current_target_positions = self._target_positions[
            self.robot._ALL_INDICES, self._target_index
        ]
        self._position_error_vector = (
            current_target_positions - self.robot.data.root_pos_w[:, :2]
        )
        self._previous_position_error = self._position_error.clone()
        self._position_error = torch.norm(self._position_error_vector, dim=-1)

        heading = self.robot.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self.robot._ALL_INDICES, self._target_index, 1]
            - self.robot.data.root_link_pos_w[:, 1],
            self._target_positions[self.robot._ALL_INDICES, self._target_index, 0]
            - self.robot.data.root_link_pos_w[:, 0],
        )
        self.target_heading_error = torch.atan2(
            torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading)
        )
        image_obs = self._get_image_obs()
        state_obs = self._get_state_obs(image_obs)
        state_obs = torch.nan_to_num(state_obs, posinf=0.0, neginf=0.0)
        if torch.any(state_obs.isnan()):
            raise ValueError("Observations cannot be NAN")
        return {"policy": state_obs}

    def _log_episode_info(self, env_ids: torch.Tensor):
        if len(env_ids) > 0:
            log_infos = {}
            log_infos["Metrics/episode_length"] = torch.mean(
                self.episode_length_buf[env_ids].float()
            ).item()
            completion_frac = (
                self._episode_waypoints_passed[env_ids].float() / self.cfg.num_goals
            )
            log_infos["Metrics/success_rate"] = torch.mean(completion_frac).item()
            log_infos["Metrics/episode_reward"] = torch.mean(
                self._episode_reward_buf[env_ids].float()
            ).item()
            log_infos["Metrics/goals_reached"] = torch.mean(
                self._episode_waypoints_passed[env_ids].float()
            ).item()
            log_infos["Metrics/max_episode_length"] = torch.mean(
                self.max_episode_length_buf[env_ids].float()
            )
            log_infos["Metrics/max_episode_return"] = (
                self._episode_reward_buf[env_ids].float().max().item()
            )

            if hasattr(self, "_episode_avoid_collisions"):
                log_infos["Metrics/avoid_collisions_per_episode"] = torch.mean(
                    self._episode_avoid_collisions[env_ids].float()
                ).item()
                log_infos["Metrics/max_avoid_collisions_per_episode"] = (
                    self._episode_avoid_collisions[env_ids].float().max().item()
                )

            self.extras["log"].update(log_infos)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

        self._pre_physics_step(action)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.camera.update(dt=self.step_dt)
        if hasattr(self, "height_scanner"):
            self.height_scanner.update(dt=self.step_dt)
        
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.extras["log"] = {}

        self.reset_terminated[:], self.reset_time_outs[:], termination_infos = (
            self._get_dones()
        )
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        reward_dict = self._get_rewards()
        self.reward_buf = torch.stack(list(reward_dict.values()), dim=0).sum(
            dim=0
        )
        self._episode_reward_buf += self.reward_buf

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            self._log_episode_info(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self._get_observations()

        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(
                self.obs_buf["policy"]
            )
        for k, v in self.obs_buf.items():
            if k != "policy":
                del self.obs_buf[k]

        self.extras["log"].update(self._populate_step_log_dict())
        self.extras["log"].update(termination_infos)
        self.extras["log"].update(
            {
                reward_name: reward_val.mean().item()
                for reward_name, reward_val in reward_dict.items()
            }
        )

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def render(self, recompute: bool = False) -> np.ndarray | None:
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator(
                    "rgb", device="cpu"
                )
                self._rgb_annotator.attach([self._render_product])
            rgb_data = self._rgb_annotator.get_data()
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            if rgb_data.size == 0:
                return np.zeros(
                    (self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3),
                    dtype=np.uint8,
                )
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def _populate_step_log_dict(self) -> dict:
        log_dict = {}
        log_dict["Metrics/max_step_reward"] = self.reward_buf.max().item()
        log_dict["Metrics/avg_step_reward"] = self.reward_buf.mean().item()
        log_dict["Metrics/min_step_reward"] = self.reward_buf.min().item()
        log_dict["Metrics/std_step_reward"] = self.reward_buf.std().item()
        return log_dict

    def _check_flipped(self) -> torch.Tensor:
        robot_quat = self.robot.data.root_quat_w
        local_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        world_up_vector = math.quat_apply(robot_quat, local_up.repeat(self.num_envs, 1))
        world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        up_dot_product = torch.sum(world_up_vector * world_up, dim=-1)
        up_angle = torch.abs(
            torch.acos(torch.clamp(up_dot_product, -1.0, 1.0))
        )
        flipped = up_angle > (torch.pi / 3)
        return flipped

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor, dict]:
        time_outs = self.episode_length_buf > self.max_episode_length_buf
        terminated = torch.zeros_like(time_outs)
        if (
            hasattr(self, "termination_on_vehicle_flip")
            and self.termination_on_vehicle_flip
        ):
            self._vehicle_flipped = self._check_flipped()
            terminated = self._vehicle_flipped
        if (
            hasattr(self, "termination_on_goal_reached")
            and self.termination_on_goal_reached
        ):
            terminated |= self.task_completed

        termination_infos = {
            "Episode_Termination/flipped": self._vehicle_flipped.float().mean().item(),
            "Episode_Termination/time_outs": time_outs.float().mean().item(),
            "Episode_Termination/task_completed": self.task_completed.float()
            .mean()
            .item(),
        }
        if hasattr(self, "termination_on_stuck") and self.termination_on_stuck:
            stuck_termination = self._check_stuck_termination()
            time_outs |= stuck_termination
            termination_infos["Episode_Termination/stuck_termination"] = (
                stuck_termination.float().mean().item()
            )

        if (
            hasattr(self, "termination_on_avoid_goal_collision")
            and self.termination_on_avoid_goal_collision
        ):
            avoid_goal_termination = self._check_avoid_goal_collision()
            terminated |= avoid_goal_termination
            termination_infos["Episode_Termination/avoid_goal_termination"] = (
                avoid_goal_termination.float().mean().item()
            )

        termination_infos["Episode_Termination/terminated"] = (
            terminated.float().mean().item()
        )
        return terminated, time_outs, termination_infos

    def _generate_random_waypoint(self, env_origins, num_reset, robot_xy=None, min_distance=2.5):
        max_attempts = 100
        placed = torch.zeros(num_reset, dtype=torch.bool, device=self.device)
        waypoint_positions = torch.zeros((num_reset, 2), device=self.device)
        
        for _ in range(max_attempts):
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()

            if num_unplaced == 0:
                break

            random_positions = self.cfg.wall_config.get_random_valid_positions(
                num_unplaced, device=self.device
            )
            
            tx = random_positions[:, 0]
            ty = random_positions[:, 1]

            unplaced_origins = env_origins[unplaced_mask]
            candidate_positions = torch.stack([tx, ty], dim=1) + unplaced_origins

            if robot_xy is not None:
                unplaced_robot_pos = robot_xy[unplaced_mask]
                robot_distances = torch.norm(candidate_positions - unplaced_robot_pos, dim=1)
                robot_valid = robot_distances >= min_distance
            else:
                robot_valid = torch.ones(num_unplaced, dtype=torch.bool, device=self.device)

            relative_positions = candidate_positions - unplaced_origins
            bounds_valid = (
                (relative_positions[:, 0] >= -17.0) & (relative_positions[:, 0] <= 17.0) &
                (relative_positions[:, 1] >= -17.0) & (relative_positions[:, 1] <= 17.0)
            )
            
            valid_mask = robot_valid & bounds_valid
            valid_indices = torch.where(unplaced_mask)[0][valid_mask]
            if len(valid_indices) > 0:
                waypoint_positions[valid_indices, :] = candidate_positions[valid_mask]
                placed[valid_indices] = True

        if not placed.all():
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()
            random_positions = self.cfg.wall_config.get_random_valid_positions(num_unplaced, device=self.device)
            unplaced_origins = env_origins[unplaced_mask]
            fallback_positions = random_positions[:, :2] + unplaced_origins
            waypoint_positions[unplaced_mask, :] = fallback_positions

        return waypoint_positions

    def _generate_offset_waypoint(self, env_origins, base_waypoints, offset_distance):
        num_reset = base_waypoints.shape[0]
        max_attempts = 100
        placed = torch.zeros(num_reset, dtype=torch.bool, device=self.device)
        waypoint_positions = torch.zeros((num_reset, 2), device=self.device)
        
        for _ in range(max_attempts):
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()

            if num_unplaced == 0:
                break

            random_angles = torch.rand((num_unplaced,), device=self.device) * 2 * torch.pi
            
            offset_x = offset_distance * torch.cos(random_angles)
            offset_y = offset_distance * torch.sin(random_angles)
            random_offsets = torch.stack([offset_x, offset_y], dim=1)

            unplaced_base = base_waypoints[unplaced_mask, :]
            candidate_positions = unplaced_base + random_offsets
            relative_positions = candidate_positions - env_origins[unplaced_mask]

            valid_mask = (
                (relative_positions[:, 0] >= -17.0) & (relative_positions[:, 0] <= 17.0) &
                (relative_positions[:, 1] >= -17.0) & (relative_positions[:, 1] <= 17.0)
            )

            valid_indices = torch.where(unplaced_mask)[0][valid_mask]
            if len(valid_indices) > 0:
                waypoint_positions[valid_indices, :] = candidate_positions[valid_mask]
                placed[valid_indices] = True

        if not placed.all():
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()
            random_positions = self.cfg.wall_config.get_random_valid_positions(num_unplaced, device=self.device)
            unplaced_origins = env_origins[unplaced_mask]
            fallback_positions = random_positions[:, :2] + unplaced_origins
            waypoint_positions[unplaced_mask, :] = fallback_positions

        return waypoint_positions

    def _generate_perpendicular_waypoint(self, env_origins, waypoint1, waypoint2, perp_offset):
        num_reset = waypoint1.shape[0]
        max_attempts = 100
        
        center_point = (waypoint1 + waypoint2) / 2.0
        
        direction_vector = waypoint2 - waypoint1
        perp_vector = torch.stack([-direction_vector[:, 1], direction_vector[:, 0]], dim=1)
        
        perp_length = torch.norm(perp_vector, dim=1, keepdim=True)
        perp_length = torch.clamp(perp_length, min=1e-8)
        perp_unit_vector = perp_vector / perp_length
        
        candidate_positions = center_point + perp_offset * perp_unit_vector
        relative_positions = candidate_positions - env_origins
        
        placed = torch.zeros(num_reset, dtype=torch.bool, device=self.device)
        
        for attempt in range(max_attempts):
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()
            
            if num_unplaced == 0:
                break
            
            unplaced_candidates = candidate_positions[unplaced_mask]
            unplaced_relative = relative_positions[unplaced_mask]
            
            valid_mask = (
                (unplaced_relative[:, 0] >= -17.0) & (unplaced_relative[:, 0] <= 17.0) &
                (unplaced_relative[:, 1] >= -17.0) & (unplaced_relative[:, 1] <= 17.0)
            )

            valid_indices = torch.where(unplaced_mask)[0][valid_mask]
            if len(valid_indices) > 0:
                placed[valid_indices] = True
            
            if not placed.all():
                unplaced_mask = ~placed
                candidate_positions[unplaced_mask] = center_point[unplaced_mask] - perp_offset * perp_unit_vector[unplaced_mask]
                relative_positions[unplaced_mask] = candidate_positions[unplaced_mask] - env_origins[unplaced_mask]

        waypoint_positions = candidate_positions.clone()

        if not placed.all():
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()
            random_positions = self.cfg.wall_config.get_random_valid_positions(num_unplaced, device=self.device)
            unplaced_origins = env_origins[unplaced_mask]
            fallback_positions = random_positions[:, :2] + unplaced_origins
            waypoint_positions[unplaced_mask, :] = fallback_positions

        return waypoint_positions

    def _generate_waypoint_group(self, env_origins, num_reset, waypoint_offset, perpendicular_offset, robot_xy=None):
        second_waypoint = self._generate_random_waypoint(env_origins, num_reset, robot_xy)
        third_waypoint = self._generate_offset_waypoint(env_origins, second_waypoint, waypoint_offset)
        first_waypoint = self._generate_perpendicular_waypoint(env_origins, second_waypoint, third_waypoint, perpendicular_offset)
        return first_waypoint, second_waypoint, third_waypoint

    def _generate_waypoints(self, env_ids, robot_poses, waypoint_offset=None, perpendicular_offset=None):
        num_reset = len(env_ids)
        env_origins = self.scene.env_origins[env_ids, :2]
        robot_xy = robot_poses[:, :2]

        if waypoint_offset is None:
            waypoint_offset = 2.5
        
        if perpendicular_offset is None:
            perpendicular_offset = 1.25

        waypoint_positions = torch.zeros((num_reset, self.cfg.num_goals, 2), device=self.device)
        
        if self.cfg.num_goals >= 3:
            first_wp, second_wp, third_wp = self._generate_waypoint_group(
                env_origins, num_reset, waypoint_offset, perpendicular_offset, robot_xy
            )
            waypoint_positions[:, 0, :] = first_wp
            waypoint_positions[:, 1, :] = second_wp  
            waypoint_positions[:, 2, :] = third_wp
        elif self.cfg.num_goals == 2:
            second_wp = self._generate_random_waypoint(env_origins, num_reset, robot_xy)
            third_wp = self._generate_offset_waypoint(env_origins, second_wp, waypoint_offset)
            waypoint_positions[:, 0, :] = second_wp
            waypoint_positions[:, 1, :] = third_wp
        elif self.cfg.num_goals == 1:
            first_wp = self._generate_random_waypoint(env_origins, num_reset, robot_xy)
            waypoint_positions[:, 0, :] = first_wp

        remaining_goals = self.cfg.num_goals - 3
        num_complete_groups = remaining_goals // 3 if remaining_goals > 0 else 0
        
        for group_idx in range(num_complete_groups):
            base_idx = 3 + group_idx * 3
            first_idx = base_idx
            second_idx = base_idx + 1
            third_idx = base_idx + 2
            
            first_wp, second_wp, third_wp = self._generate_waypoint_group(
                env_origins, num_reset, waypoint_offset, perpendicular_offset
            )
            waypoint_positions[:, first_idx, :] = first_wp
            waypoint_positions[:, second_idx, :] = second_wp
            waypoint_positions[:, third_idx, :] = third_wp

        remaining_points = remaining_goals % 3 if remaining_goals > 0 else 0
        if remaining_points > 0:
            start_idx = 3 + num_complete_groups * 3
            for point_idx in range(remaining_points):
                goal_idx = start_idx + point_idx
                random_wp = self._generate_random_waypoint(env_origins, num_reset)
                waypoint_positions[:, goal_idx, :] = random_wp

        return waypoint_positions

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self.camera.reset(env_ids)
        if self.play_mode:
            self.max_episode_length_buf[env_ids] = self.max_episode_length
        else:
            min_episode_length = min(
                200 + self.common_step_counter, int(0.7 * self.max_episode_length)
            )
            self.max_episode_length_buf[env_ids] = torch.randint(
                min_episode_length,
                self.max_episode_length + 1,
                (len(env_ids),),
                device=self.device,
            )

        self._episode_reward_buf[env_ids] = 0.0
        self._episode_waypoints_passed[env_ids] = 0
        if hasattr(self, "_episode_avoid_collisions"):
            self._episode_avoid_collisions[env_ids] = 0
        if hasattr(self, "_previous_waypoint_reached_step"):
            self._previous_waypoint_reached_step[env_ids] = 0

        num_reset = len(env_ids)
        default_state = self.robot.data.default_root_state[env_ids]
        robot_pose = default_state[:, :7]
        robot_velocities = default_state[:, 7:]
        joint_positions = self.robot.data.default_joint_pos[env_ids]
        joint_velocities = self.robot.data.default_joint_vel[env_ids]

        robot_pose[:, :3] += self.scene.env_origins[env_ids]

        random_positions = self.cfg.wall_config.get_random_valid_positions(
            num_reset, device=self.device
        )
        robot_pose[:, :2] = random_positions[:, :2] + self.scene.env_origins[env_ids, :2]

        angles = (
            torch.pi / 6.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device)
        )
        robot_pose[:, 3] = torch.cos(angles * 0.5)
        robot_pose[:, 6] = torch.sin(angles * 0.5)

        self.robot.write_root_pose_to_sim(robot_pose, env_ids)
        self.robot.write_root_velocity_to_sim(robot_velocities, env_ids)
        self.robot.write_joint_state_to_sim(
            joint_positions, joint_velocities, None, env_ids
        )

        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0

        waypoint_positions = self._generate_waypoints(
            env_ids,
            robot_pose
        )

        self._target_positions[env_ids] = waypoint_positions

        self._target_index[env_ids] = 0
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)

        current_target_positions = self._target_positions[
            self.robot._ALL_INDICES, self._target_index
        ]
        self._position_error_vector = (
            current_target_positions[:, :2] - self.robot.data.root_pos_w[:, :2]
        )
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

        heading = self.robot.data.heading_w[:]
        target_heading_w = torch.atan2(
            self._target_positions[:, 0, 1] - self.robot.data.root_pos_w[:, 1],
            self._target_positions[:, 0, 0] - self.robot.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(
            torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading)
        )
        self._previous_heading_error = self._heading_error.clone()

    def _get_distance_to_walls(self):
        robot_positions = self.robot.data.root_pos_w[:, :2]
        env_origins = self.scene.env_origins[:, :2]
        relative_positions = robot_positions - env_origins
        return self.cfg.wall_config.get_wall_distances(relative_positions)

    def _check_stuck_termination(self, max_steps: int = 300) -> torch.Tensor:
        steps_since_goal = (
            self.episode_length_buf - self._previous_waypoint_reached_step
        )
        stuck_too_long = steps_since_goal > max_steps
        return stuck_too_long

    def _check_avoid_goal_collision(self) -> torch.Tensor:
        robot_positions = self.robot.data.root_pos_w[:, :2]

        goal_indices = torch.arange(self.cfg.num_goals, device=self.device).unsqueeze(
            0
        )
        target_indices = self._target_index.unsqueeze(1)
        future_waypoint_mask = goal_indices > target_indices

        robot_pos_expanded = robot_positions.unsqueeze(1)
        distances = torch.norm(
            robot_pos_expanded - self._target_positions, dim=2
        )

        future_distances = (
            distances * future_waypoint_mask.float()
        )
        future_distances = torch.where(
            future_waypoint_mask,
            future_distances,
            torch.full_like(future_distances, float("inf")),
        )

        collision_mask = (
            future_distances < self.avoid_goal_position_tolerance
        )
        env_has_collision = collision_mask.any(dim=1)
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
        # Complete when gets to the goal which is the third last waypoint
        self.task_completed = self._target_index > (self.cfg.num_goals - 3)
        self._target_index = self._target_index % self.cfg.num_goals
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
            0.5
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
            (self.num_envs, self.cfg.num_goals),
            device=self.device,
            dtype=torch.long,
        )
        # Set current targets to 1 (green)
        marker_indices[
            torch.arange(self.num_envs, device=self.device), self._target_index
        ] = 1
        # Set the next target to 3 (cyan)
        marker_indices[
            torch.arange(self.num_envs, device=self.device), self._target_index + 1
        ] = 3
        # Set completed targets to 2 (invisible)
        # Create a mask for completed targets
        target_mask = (self._target_index.unsqueeze(1) > 0) & (
            torch.arange(self.cfg.num_goals, device=self.device)
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
