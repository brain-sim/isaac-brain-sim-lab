from __future__ import annotations

import copy
import os
from collections import defaultdict
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .landmark_env_cfg import LandmarkEnvCfg
from .env_component_robot import EnvComponentRobot
from .env_component_observation import EnvComponentObservation
from .env_component_reward import EnvComponentReward
from .env_component_termination import EnvComponentTermination
from .env_component_waypoint import EnvComponentWaypoint


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

        # Initialize environment components
        self.components = {
            'robot': EnvComponentRobot(self),
            'observation': EnvComponentObservation(self),
            'reward': EnvComponentReward(self),
            'termination': EnvComponentTermination(self),
            'waypoint': EnvComponentWaypoint(self)
        }
        
        for name, component in self.components.items():
            setattr(self, f'env_component_{name}', component)

        super().__init__(cfg, render_mode, **kwargs)

        self.cfg.wall_config.update_device(self.device)

        # Initialize tracking variables
        self._goal_reached = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.task_completed = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        self._episode_reward_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float32)
        self._episode_reward_infos_buf = {}
        self._metrics_infos_buf = {}
        self._terminations_infos_buf = defaultdict(lambda: torch.zeros((self.num_envs), device=self.device, dtype=torch.int32))

        self._step_rewards_buffer = []
        self._episode_step_rewards = []

        self._debug = debug
        self.max_total_steps = max_total_steps
        self.play_mode = play_mode
        self.env_spacing = self.cfg.env_spacing
        self.max_episode_length_buf = torch.full((self.num_envs,), self.max_episode_length, device=self.device)

        # Post-initialize all environment components
        for component in self.components.values():
            if hasattr(component, 'post_env_init'):
                component.post_env_init()

    def _setup_scene(self):
        # Setup required components
        setup_components = ['robot', 'waypoint']
        for name in setup_components:
            component = self.components[name]
            if hasattr(component, 'setup'):
                component.setup()

        # Get robot from component and setup scene
        self.robot = self.env_component_robot.robot
        self.camera = self.env_component_robot.camera
        self.waypoints = self.env_component_waypoint.waypoints
        self.object_state = []

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot

        self.wall_height = self.cfg.wall_height
        self.wall_position = (self.cfg.room_size - self.cfg.wall_thickness) / 2

    def _log_episode_info(self, env_ids: torch.Tensor):
        if len(env_ids) > 0:
            log_infos = {}
            log_infos["Metrics/episode_length"] = torch.mean(
                self.episode_length_buf[env_ids].float()
            ).item()
            completion_frac = (
                self.env_component_waypoint._episode_waypoints_passed[env_ids].float() / self.cfg.num_goals
            )
            log_infos["Metrics/success_rate"] = torch.mean(completion_frac).item()
            log_infos["Metrics/episode_reward"] = torch.mean(
                self._episode_reward_buf[env_ids].float()
            ).item()
            log_infos["Metrics/goals_reached"] = torch.mean(
                self.env_component_waypoint._episode_waypoints_passed[env_ids].float()
            ).item()
            log_infos["Metrics/max_episode_length"] = torch.mean(
                self.max_episode_length_buf[env_ids].float()
            )
            log_infos["Metrics/max_episode_return"] = (
                self._episode_reward_buf[env_ids].float().max().item()
            )

            if hasattr(self.env_component_reward, "_episode_avoid_collisions"):
                log_infos["Metrics/avoid_collisions_per_episode"] = torch.mean(
                    self.env_component_reward._episode_avoid_collisions[env_ids].float()
                ).item()
                log_infos["Metrics/max_avoid_collisions_per_episode"] = (
                    self.env_component_reward._episode_avoid_collisions[env_ids].float().max().item()
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

        self.env_component_robot.update_camera(dt=self.step_dt)
        if hasattr(self, "height_scanner"):
            self.height_scanner.update(dt=self.step_dt)
        
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.extras["log"] = {}
        
        reward_dict = self.env_component_reward.get_rewards()
        self.reward_buf = torch.stack(list(reward_dict.values()), dim=0).sum(dim=0)
        self._episode_reward_buf += self.reward_buf

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        # Get termination infos for logging
        _, _, termination_infos = self.env_component_termination.get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs

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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Pre-process actions before stepping through the physics."""
        self.env_component_robot.pre_physics_step(actions)

    def _apply_action(self) -> None:
        """Apply actions to the simulator."""
        self.env_component_robot.apply_action()

    def _get_observations(self) -> dict:
        """Compute and return the observations for the environment."""
        return self.env_component_observation.get_observations()

    def _get_rewards(self) -> torch.Tensor:
        """Compute and return the rewards for the environment."""
        reward_dict = self.env_component_reward.get_rewards()
        return reward_dict

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return the done flags for the environment."""
        terminated, time_outs, _ = self.env_component_termination.get_dones()
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        
        # Update episode length buffer
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

        # Reset episode tracking
        self._episode_reward_buf[env_ids] = 0.0
        
        # Reset robot and get pose
        robot_pose = self.env_component_robot.reset(env_ids)
        
        # Reset waypoints
        self.env_component_waypoint.reset(env_ids, robot_pose)
        
        # Reset other components
        self.env_component_observation.reset(env_ids)
        self.env_component_reward.reset(env_ids)
