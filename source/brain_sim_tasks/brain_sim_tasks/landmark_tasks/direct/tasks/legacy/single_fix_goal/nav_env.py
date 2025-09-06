from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math

from .nav_env_cfg import NavEnvCfg
import numpy as np

class NavEnv(DirectRLEnv):
    cfg: NavEnvCfg

    def __init__(
        self,
        cfg: NavEnvCfg,
        render_mode: str | None = None,
        debug: bool = False,
        max_total_steps: int | None = None,
        play_mode: bool = False,
        **kwargs,
    ):
        self.room_size = getattr(cfg, "room_size", 40.0)
        self._num_goals = getattr(cfg, "num_goals", 2)
        self.env_spacing = getattr(cfg, "env_spacing", 40.0)

        super().__init__(cfg, render_mode, **kwargs)

        self.cfg.wall_config.update_device(self.device)

        self._setup_robot_dof_idx()
        self._goal_reached = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )
        self.task_completed = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.bool
        )
        self._target_positions = torch.zeros(
            (self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32
        )
        self._markers_pos = torch.zeros(
            (self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32
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

    def _setup_config(self):
        raise NotImplementedError("Subclass must implement this method")

    def _setup_robot_dof_idx(self):
        raise NotImplementedError("Subclass must implement this method")

    def _setup_camera(self):
        raise NotImplementedError("Subclass must implement this method")

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

        self.wall_thickness = self.cfg.wall_thickness
        self.wall_height = self.cfg.wall_height
        self.wall_position = (self.room_size - self.wall_thickness) / 2

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        raise NotImplementedError("Subclass must implement this method")

    def _apply_action(self) -> None:
        raise NotImplementedError("Subclass must implement this method")

    def _get_image_obs(self) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")

    def _get_state_obs(self) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")

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
        """Logs episode information for the given environment IDs.
        Args:
            env_ids: A tensor of environment IDs that have been reset.
        """
        if len(env_ids) > 0:
            log_infos = {}
            log_infos["Metrics/episode_length"] = torch.mean(
                self.episode_length_buf[env_ids].float()
            ).item()
            completion_frac = (
                self._episode_waypoints_passed[env_ids].float() / self._num_goals
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
        
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        self.extras["log"] = {}

        self.reset_terminated[:], self.reset_time_outs[:], termination_infos = (
            self._get_dones()
        )
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        reward_dict = self._get_rewards()  # {reward_name: [num_envs]}
        self.reward_buf = torch.stack(list(reward_dict.values()), dim=0).sum(
            dim=0
        )  # [num_envs]
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

                # create render product
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator(
                    "rgb", device="cpu"
                )
                self._rgb_annotator.attach([self._render_product])
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data.size == 0:
                return np.zeros(
                    (self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3),
                    dtype=np.uint8,
                )
            else:
                # image_obs = self.camera.data.output["rgb"]
                # rgb_data = image_obs[0, :, :, :3].detach().cpu().numpy()
                # return cv2.resize(rgb_data, (self.cfg.viewer.resolution[0], self.cfg.viewer.resolution[1]))
                return rgb_data[:, :, :3]

        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Subclass must implement this method")

    def _populate_step_log_dict(self) -> dict:
        """
        Populate the log dictionary with RSL-style logging format matching the PPO script expectations.
        Returns a dictionary with keys formatted as Category/metric_name for proper logging.
        """
        log_dict = {}
        log_dict["Metrics/max_step_reward"] = self.reward_buf.max().item()
        log_dict["Metrics/avg_step_reward"] = self.reward_buf.mean().item()
        log_dict["Metrics/min_step_reward"] = self.reward_buf.min().item()
        log_dict["Metrics/std_step_reward"] = self.reward_buf.std().item()
        return log_dict

    def _check_flipped(self) -> torch.Tensor:
        # Get robot's orientation
        robot_quat = self.robot.data.root_quat_w  # (num_envs, 4) in (w,x,y,z)
        local_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        world_up_vector = math.quat_apply(robot_quat, local_up.repeat(self.num_envs, 1))
        world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        up_dot_product = torch.sum(world_up_vector * world_up, dim=-1)  # (num_envs,)
        up_angle = torch.abs(
            torch.acos(torch.clamp(up_dot_product, -1.0, 1.0))
        )  # radians

        # Consider flipped if angle > 60 degrees (pi/3 radians)
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
        # Add stuck termination
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
    
    def _generate_waypoints(self, env_ids, robot_poses, waypoint_offset=None, fixed_first_position=None):
        """Generate waypoints in groups of 2: (1→2)→(3→4)→..."""
        num_reset = len(env_ids)
        env_origins = self.scene.env_origins[env_ids, :2]  # (num_reset, 2)
        robot_xy = robot_poses[:, :2]  # (num_reset, 2)

        if waypoint_offset is None:
            waypoint_offset = torch.tensor([3.0, 0.5], device=self.device)

        if fixed_first_position is None:
            fixed_first_position = torch.tensor([5.0, 5.0], device=self.device)

        # Reverse offset: Current implementation has the goal generated first, landmark next
        waypoint_offset = -waypoint_offset

        waypoint_positions = torch.zeros((num_reset, self._num_goals, 2), device=self.device)
        
        # Generate waypoints in groups of 2
        num_complete_groups = self._num_goals // 2
        remaining_points = self._num_goals % 2
        
        # Generate complete groups of 2
        for group_idx in range(num_complete_groups):
            first_idx = group_idx * 2      # First point of the group
            second_idx = group_idx * 2 + 1 # Second point of the group
            
            # Generate first waypoint of the group
            if group_idx == 0:
                # For the very first waypoint, use fixed position
                waypoint_positions[:, first_idx, :] = self._generate_fixed_waypoint(
                    env_origins, num_reset, fixed_first_position
                )
            else:
                # For subsequent groups, generate randomly without robot distance constraint
                waypoint_positions[:, first_idx, :] = self._generate_random_waypoint(
                    env_origins, num_reset
                )
            
            # Generate second waypoint with offset from first
            waypoint_positions[:, second_idx, :] = self._generate_offset_waypoint(
                env_origins, waypoint_positions[:, first_idx, :], waypoint_offset, num_reset
            )
        
        # Handle remaining single waypoint if total is odd
        if remaining_points > 0:
            last_idx = num_complete_groups * 2
            waypoint_positions[:, last_idx, :] = self._generate_random_waypoint(
                env_origins, num_reset
            )

        return waypoint_positions

    def _generate_fixed_waypoint(self, env_origins, num_reset, fixed_position):
        """Generate fixed waypoint at specified position for all environments."""
        # Create fixed position tensor for all environments
        fixed_waypoint = fixed_position.unsqueeze(0).expand(num_reset, -1)  # (num_reset, 2)
        
        # Add environment origins to get absolute positions
        absolute_positions = fixed_waypoint + env_origins
        
        # Check if positions are within bounds (-17 to +17 in both x and y)
        relative_positions = absolute_positions - env_origins
        valid_mask = (
            (relative_positions[:, 0] >= -17.0) & (relative_positions[:, 0] <= 17.0) &
            (relative_positions[:, 1] >= -17.0) & (relative_positions[:, 1] <= 17.0)
        )
        
        # For environments where fixed position is out of bounds, use fallback
        waypoint_positions = absolute_positions.clone()
        invalid_mask = ~valid_mask
        
        if invalid_mask.any():
            # Use fallback random positions for environments where fixed position is invalid
            num_invalid = invalid_mask.sum().item()
            random_positions = self.cfg.wall_config.get_random_valid_positions(num_invalid, device=self.device)
            invalid_origins = env_origins[invalid_mask]
            fallback_positions = random_positions[:, :2] + invalid_origins
            waypoint_positions[invalid_mask, :] = fallback_positions
            
        return waypoint_positions

    def _generate_random_waypoint_with_robot_distance(self, env_origins, num_reset, robot_xy):
        """Generate random waypoint with minimum distance from robot."""
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

            unplaced_robot_pos = robot_xy[unplaced_mask]
            robot_distances = torch.norm(candidate_positions - unplaced_robot_pos, dim=1)
            robot_valid = robot_distances >= 2.5

            valid_indices = torch.where(unplaced_mask)[0][robot_valid]
            if len(valid_indices) > 0:
                waypoint_positions[valid_indices, :] = candidate_positions[robot_valid]
                placed[valid_indices] = True

        # Fallback for unplaced waypoints
        if not placed.all():
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()
            random_positions = self.cfg.wall_config.get_random_valid_positions(num_unplaced, device=self.device)
            unplaced_origins = env_origins[unplaced_mask]
            fallback_positions = random_positions[:, :2] + unplaced_origins
            waypoint_positions[unplaced_mask, :] = fallback_positions

        return waypoint_positions

    def _generate_random_waypoint(self, env_origins, num_reset):
        """Generate random waypoint without robot distance constraint."""
        random_positions = self.cfg.wall_config.get_random_valid_positions(num_reset, device=self.device)
        return random_positions[:, :2] + env_origins

    def _generate_offset_waypoint(self, env_origins, base_waypoints, waypoint_offset, num_reset):
        """Generate waypoint with offset from base waypoint."""
        max_attempts = 100
        placed = torch.zeros(num_reset, dtype=torch.bool, device=self.device)
        waypoint_positions = torch.zeros((num_reset, 2), device=self.device)
        
        # Apply the offset vector directly to base waypoint
        candidate_positions = base_waypoints + waypoint_offset.unsqueeze(0).expand(num_reset, -1)
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
                (unplaced_relative[:, 0] >= -17.0) & (unplaced_relative[:, 0] <= 17.0) &
                (unplaced_relative[:, 1] >= -17.0) & (unplaced_relative[:, 1] <= 17.0)
            )

            valid_indices = torch.where(unplaced_mask)[0][valid_mask]
            if len(valid_indices) > 0:
                waypoint_positions[valid_indices, :] = unplaced_candidates[valid_mask]
                placed[valid_indices] = True
        
        # Fallback for unplaced waypoints
        if not placed.all():
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()
            random_positions = self.cfg.wall_config.get_random_valid_positions(num_unplaced, device=self.device)
            unplaced_origins = env_origins[unplaced_mask]
            fallback_positions = random_positions[:, :2] + unplaced_origins
            waypoint_positions[unplaced_mask, :] = fallback_positions

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
        # Reset avoid collision counter
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
            robot_pose,
            fixed_first_position=getattr(self.cfg, 'fixed_first_position', torch.tensor([5.0, 5.0], device=self.device))
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
