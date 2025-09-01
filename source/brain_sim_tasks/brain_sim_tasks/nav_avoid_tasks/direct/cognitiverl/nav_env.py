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
        # Add room size as a class attribut
        self.room_size = getattr(cfg, "room_size", 40.0)
        self._num_goals = getattr(cfg, "num_goals", 1)
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

        # Add accumulated laziness tracker
        self._accumulated_laziness = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float32
        )

        # Add episode metrics
        self._episode_waypoints_passed = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )
        self._episode_reward_buf = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float32
        )

        # Add RSL-style logging buffers
        self._episode_reward_infos_buf = {}
        self._metrics_infos_buf = {}
        self._terminations_infos_buf = defaultdict(
            lambda: torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        )

        # Add reward tracking for step rewards
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
        # Create a large ground plane without grid
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                size=(
                    4096 * 40.0,
                    4096 * 40.0,
                ),  # Much larger ground plane (500m x 500m)
                color=(0.2, 0.2, 0.2),  # Dark gray color
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

        # Setup rest of the scene
        self.robot = Articulation(self.cfg.robot_cfg)
        self._setup_camera()
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        self.object_state = []

        # FIRST: Clone environments to initialize env_origins
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot

        # Import the necessary classes and NumPy
        import numpy as np

        # Define wall properties
        self.wall_thickness = self.cfg.wall_thickness
        self.wall_height = self.cfg.wall_height
        self.wall_position = (self.room_size - self.wall_thickness) / 2

        # Add lighting
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
            # log episode length
            log_infos = {}
            log_infos["Metrics/episode_length"] = torch.mean(
                self.episode_length_buf[env_ids].float()
            ).item()
            # calculate and log completion percentage
            completion_frac = (
                self._episode_waypoints_passed[env_ids].float() / self._num_goals
            )
            log_infos["Metrics/success_rate"] = torch.mean(completion_frac).item()
            # log episode reward
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

            # Add avoid goal collision metrics if they exist
            if hasattr(self, "_episode_avoid_collisions"):
                log_infos["Metrics/avoid_collisions_per_episode"] = torch.mean(
                    self._episode_avoid_collisions[env_ids].float()
                ).item()
                log_infos["Metrics/max_avoid_collisions_per_episode"] = (
                    self._episode_avoid_collisions[env_ids].float().max().item()
                )

            self.extras["log"].update(log_infos)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        self.camera.update(dt=self.step_dt)
        if hasattr(self, "height_scanner"):
            self.height_scanner.update(dt=self.step_dt)
        # post-step:
        # -- update env counters (used for curriculum generation)
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

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            self._log_episode_info(reset_env_ids)
            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(
                self.obs_buf["policy"]
            )
        for k, v in self.obs_buf.items():
            if k != "policy":
                del self.obs_buf[k]

        # Add RSL-style log data to extras
        self.extras["log"].update(self._populate_step_log_dict())
        self.extras["log"].update(termination_infos)
        self.extras["log"].update(
            {
                reward_name: reward_val.mean().item()
                for reward_name, reward_val in reward_dict.items()
            }
        )

        # return observations, rewards, resets and extras
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        # run a rendering step of the simulator
        # if we have rtx sensors, we do not need to render again sin
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()
        # decide the rendering mode
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            # check that if any render could have happened
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )
            # create the annotator if it does not exist
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

    def _generate_waypoints_with_spacing(self, env_ids, robot_poses, min_spacing=6.0):
        """Generate waypoints ensuring minimum spacing between them and from robot spawn - VECTORIZED."""
        num_reset = len(env_ids)
        env_origins = self.scene.env_origins[env_ids, :2]  # (num_reset, 2)
        robot_xy = robot_poses[:, :2]  # (num_reset, 2)

        # Initialize waypoint positions
        waypoint_positions = torch.zeros(
            (num_reset, self._num_goals, 2), device=self.device
        )

        # Constrain waypoints to 60% of distance from center to wall
        max_distance_from_center = 0.6 * (self.wall_position - self.wall_thickness / 2)

        # Generate waypoints sequentially (to maintain spacing constraints)
        for goal_idx in range(self._num_goals):
            max_attempts = 100
            placed = torch.zeros(num_reset, dtype=torch.bool, device=self.device)

            for attempt in range(max_attempts):
                # Generate candidates for all unplaced environments
                unplaced_mask = ~placed
                num_unplaced = unplaced_mask.sum().item()

                if num_unplaced == 0:
                    break

                # Generate random positions for unplaced environments
                tx = (
                    torch.rand(num_unplaced, device=self.device)
                    * 2
                    * max_distance_from_center
                    - max_distance_from_center
                )
                ty = (
                    torch.rand(num_unplaced, device=self.device)
                    * 2
                    * max_distance_from_center
                    - max_distance_from_center
                )

                # Convert to world coordinates
                unplaced_origins = env_origins[unplaced_mask]  # (num_unplaced, 2)
                candidate_positions = (
                    torch.stack([tx, ty], dim=1) + unplaced_origins
                )  # (num_unplaced, 2)

                # Check distance from robot (minimum 2.5m)
                unplaced_robot_pos = robot_xy[unplaced_mask]  # (num_unplaced, 2)
                robot_distances = torch.norm(
                    candidate_positions - unplaced_robot_pos, dim=1
                )  # (num_unplaced,)
                robot_valid = robot_distances >= 2.5

                # Check distance from previously placed waypoints in same environment
                waypoint_valid = torch.ones(
                    (num_unplaced,), dtype=torch.bool, device=self.device
                )
                if goal_idx > 0:
                    # Get previously placed waypoints for unplaced environments
                    prev_waypoints = waypoint_positions[
                        unplaced_mask, :goal_idx, :
                    ]  # (num_unplaced, goal_idx, 2)

                    # Calculate distances to all previous waypoints
                    candidate_expanded = candidate_positions.unsqueeze(
                        1
                    )  # (num_unplaced, 1, 2)
                    distances_to_prev = torch.norm(
                        candidate_expanded - prev_waypoints, dim=2
                    )  # (num_unplaced, goal_idx)

                    # Check if any distance is too small
                    min_distances = distances_to_prev.min(dim=1)[0]  # (num_unplaced,)
                    waypoint_valid = min_distances >= min_spacing

                # Combine all validity checks
                valid = robot_valid & waypoint_valid

                # Update positions for valid placements
                valid_indices = torch.where(unplaced_mask)[0][valid]
                if len(valid_indices) > 0:
                    waypoint_positions[valid_indices, goal_idx, :] = (
                        candidate_positions[valid]
                    )
                    placed[valid_indices] = True

            # Fallback for any remaining unplaced waypoints
            if not placed.all():
                unplaced_mask = ~placed
                num_unplaced = unplaced_mask.sum().item()

                # Use relaxed constraints - smaller area but guaranteed placement
                tx = (
                    torch.rand(num_unplaced, device=self.device)
                    * max_distance_from_center
                    - max_distance_from_center / 2
                )
                ty = (
                    torch.rand(num_unplaced, device=self.device)
                    * max_distance_from_center
                    - max_distance_from_center / 2
                )

                unplaced_origins = env_origins[unplaced_mask]
                fallback_positions = torch.stack([tx, ty], dim=1) + unplaced_origins

                unplaced_indices = torch.where(unplaced_mask)[0]
                waypoint_positions[unplaced_indices, goal_idx, :] = fallback_positions

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

        # Keep random rotation for variety
        angles = (
            torch.pi
            / 6.0
            * torch.rand((num_reset), dtype=torch.float32, device=self.device)
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

        # Generate waypoints with proper spacing
        waypoint_positions = self._generate_waypoints_with_spacing(
            env_ids,
            robot_pose,
            min_spacing=(
                self.cfg.position_tolerance + self.cfg.avoid_goal_position_tolerance
                if hasattr(self, "cfg.avoid_goal_position_tolerance")
                else self.cfg.position_tolerance
            ),
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
        """Calculate the distance from the robot to the nearest wall using precomputed distance field.
        Returns:
            torch.Tensor: Distance to nearest wall edge for each environment (num_envs,)
                         All values are positive since robots are guaranteed to spawn in open spaces.
        """
        robot_positions = self.robot.data.root_pos_w[:, :2]
        env_origins = self.scene.env_origins[:, :2]
        
        relative_positions = robot_positions - env_origins
        
        return self.cfg.wall_config.get_wall_distances(relative_positions)
