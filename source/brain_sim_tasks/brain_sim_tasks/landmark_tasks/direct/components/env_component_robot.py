from __future__ import annotations

import os
import torch
from isaaclab.assets import Articulation
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from termcolor import colored
from brain_sim_assets import BRAIN_SIM_ASSETS_ROBOTS_DATA_DIR
from brain_sim_assets.robots.spot import bsSpotLowLevelPolicyVanilla


class EnvComponentRobot:
    """Component responsible for robot control, setup, and actions."""
    
    def __init__(self, env):
        self.env = env
        self.robot = None
        self.camera = None
        self.policy = None

    def setup(self):
        """Setup the robot components."""
        self._setup_robot()
        self._setup_camera()

    def post_env_init(self):
        self._setup_robot_dof_idx()
        self._setup_config()

    def _setup_robot(self):
        """Setup the robot articulation."""
        self.robot = Articulation(self.env.cfg.robot_cfg)

    def _setup_robot_dof_idx(self):
        """Setup robot DOF indices."""
        self.env._dof_idx, _ = self.robot.find_joints(self.env.cfg.dof_name)

    def _setup_camera(self):
        """Setup the robot camera."""
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
            update_period=self.env.step_dt,
            height=self.env.cfg.img_size[1],
            width=self.env.cfg.img_size[2],
            data_types=["rgb"],
            spawn=pinhole_cfg,
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.25, 0.0, 0.25),  # At the head, adjust as needed
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )
        self.camera = TiledCamera(camera_cfg)

    def _setup_config(self):
        """Setup robot policy and action parameters."""
        policy_file_path = os.path.join(
            BRAIN_SIM_ASSETS_ROBOTS_DATA_DIR,
            self.env.cfg.policy_file_path,
        )
        print(
            colored(
                f"[INFO] Loading policy from {policy_file_path}",
                "magenta",
                attrs=["bold"],
            )
        )
        self.policy = bsSpotLowLevelPolicyVanilla(policy_file_path)
        
        # Initialize action-related tensors
        self._low_level_previous_action = torch.zeros(
            (self.env.num_envs, 12), device=self.env.device, dtype=torch.float32
        )
        self._previous_action = torch.zeros(
            (self.env.num_envs, self.env.cfg.action_space),
            device=self.env.device,
            dtype=torch.float32,
        )
        self._actions = torch.zeros(
            (self.env.num_envs, self.env.cfg.action_space),
            device=self.env.device,
            dtype=torch.float32,
        )
        
        # Action parameters
        self.throttle_scale = self.env.cfg.throttle_scale
        self.steering_scale = self.env.cfg.steering_scale
        self.throttle_max = self.env.cfg.throttle_max
        self.steering_max = self.env.cfg.steering_max
        self._default_pos = self.robot.data.default_joint_pos.clone()
        self._smoothing_factor = torch.tensor([0.75, 0.3, 0.3], device=self.env.device)

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step."""
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

    def apply_action(self) -> None:
        """Apply low-level policy actions to robot."""
        # Gather all required robot state as torch tensors
        default_pos = self._default_pos.clone()  # [num_envs, 12]
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
        joint_positions = self._default_pos + actions * self.env.ACTION_SCALE
        
        # Apply joint position targets directly
        self.robot.set_joint_position_target(joint_positions)

    def reset(self, env_ids):
        """Reset robot for specified environments."""
        self.camera.reset(env_ids)
        
        # Reset robot state
        num_reset = len(env_ids)
        default_state = self.robot.data.default_root_state[env_ids]
        robot_pose = default_state[:, :7]
        robot_velocities = default_state[:, 7:]
        joint_positions = self.robot.data.default_joint_pos[env_ids]
        joint_velocities = self.robot.data.default_joint_vel[env_ids]

        robot_pose[:, :3] += self.env.scene.env_origins[env_ids]

        # Generate random positions for robot
        random_positions = self.env.cfg.wall_config.get_random_valid_positions(
            num_reset, device=self.env.device
        )
        robot_pose[:, :2] = random_positions[:, :2] + self.env.scene.env_origins[env_ids, :2]

        # Random orientation
        angles = (
            torch.pi / 6.0 * torch.rand((num_reset), dtype=torch.float32, device=self.env.device)
        )
        robot_pose[:, 3] = torch.cos(angles * 0.5)
        robot_pose[:, 6] = torch.sin(angles * 0.5)

        # Write state to simulation
        self.robot.write_root_pose_to_sim(robot_pose, env_ids)
        self.robot.write_root_velocity_to_sim(robot_velocities, env_ids)
        self.robot.write_joint_state_to_sim(
            joint_positions, joint_velocities, None, env_ids
        )
        
        return robot_pose

    def update_camera(self, dt):
        """Update camera with given timestep."""
        self.camera.update(dt=dt)

    def get_distance_to_walls(self):
        """Get distance to walls for all robots."""
        robot_positions = self.robot.data.root_pos_w[:, :2]
        env_origins = self.env.scene.env_origins[:, :2]
        relative_positions = robot_positions - env_origins
        return self.env.cfg.wall_config.get_wall_distances(relative_positions)
