import torch
from dataclasses import MISSING

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.assets import Articulation

from brain_sim_assets.robots.spot import bsSpotLowLevelPolicyRough
from brain_sim_assets import BRAIN_SIM_ASSETS_ROBOTS_DATA_DIR


class HierarchicalSpotActionTerm(ActionTerm):

    cfg: "HierarchicalSpotActionCfg"

    def __init__(self, cfg: "HierarchicalSpotActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._low_level_policy = bsSpotLowLevelPolicyRough(cfg.policy_file_path)
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._num_envs = env.num_envs
        self._device = env.device

        self._low_level_previous_action = torch.zeros(
            (self._num_envs, 12), device=self._device, dtype=torch.float32
        )
        self._default_pos = self._robot.data.default_joint_pos.clone()
        self._action_scale = cfg.action_scale

        self._raw_actions = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )
        self._processed_actions = torch.zeros(
            (self._num_envs, 12), device=self._device, dtype=torch.float32
        )

        # High-level command smoothing
        self._smoothing_factor = torch.tensor(cfg.smoothing_factor, device=self._device)
        self._previous_action = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )

        # Store scaling parameters
        self.throttle_scale = cfg.throttle_scale
        self.steering_scale = cfg.steering_scale
        self.throttle_max = cfg.throttle_max
        self.steering_max = cfg.steering_max

    @property
    def action_dim(self) -> int:
        """
        Dimension of the action term: 3D high-level commands.
        Based on Direct RL environment analysis:
        - action_space = 3
        - actions[:, 0] scaled by throttle_scale and clamped [0, throttle_max]
        - actions[:, 1:] scaled by steering_scale and clamped [-steering_max, steering_max]

        This suggests the 3D action space represents some form of:
        - Dimension 0: throttle/forward command (non-negative)
        - Dimensions 1-2: steering/lateral commands (bipolar)

        The exact semantic meaning depends on the pre-trained low-level policy.
        """
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Convert high-level commands to joint positions using low-level policy.
        Args:
            actions: High-level commands [num_envs, 3] - dimension 0: throttle, dimensions 1-2: steering
        """
        self._raw_actions = actions.clone()

        smoothed_actions = (
            self._smoothing_factor * actions
            + (1 - self._smoothing_factor) * self._previous_action
        )
        self._previous_action = smoothed_actions.clone()
        _actions = smoothed_actions.clone()
        _actions = torch.nan_to_num(_actions, 0.0)
        _actions[:, 0] = _actions[:, 0] * self.throttle_scale
        _actions[:, 1:] = _actions[:, 1:] * self.steering_scale
        _actions[:, 0] = torch.clamp(_actions[:, 0], min=0.0, max=self.throttle_max)
        _actions[:, 1:] = torch.clamp(
            _actions[:, 1:], min=-self.steering_max, max=self.steering_max
        )

        default_pos = self._default_pos.clone()  # [num_envs, 12]
        base_lin_vel = self._robot.data.root_lin_vel_b  # [num_envs, 3] - body frame
        base_ang_vel = self._robot.data.root_ang_vel_b  # [num_envs, 3] - body frame
        projected_gravity = (
            self._robot.data.projected_gravity_b
        )  # [num_envs, 3] - body frame
        joint_pos = self._robot.data.joint_pos  # [num_envs, 12]
        joint_vel = self._robot.data.joint_vel  # [num_envs, 12]

        low_level_actions = self._low_level_policy.get_action(
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            _actions,  # This is the 3D command passed to low-level policy
            self._low_level_previous_action,
            default_pos,
            joint_pos,
            joint_vel,
        )

        self._low_level_previous_action = low_level_actions.detach()
        joint_positions = self._default_pos + low_level_actions * self._action_scale

        self._processed_actions = joint_positions

        return joint_positions

    def apply_actions(self):
        """Apply the processed actions to the robot."""
        # The joint positions are already computed in process_actions
        # and stored in self._processed_actions by the base class
        self._robot.set_joint_position_target(self._processed_actions)


@configclass
class HierarchicalSpotActionCfg(ActionTermCfg):

    class_type: type[ActionTerm] = HierarchicalSpotActionTerm
    asset_name: str = "robot"
    policy_file_path: str = f"{BRAIN_SIM_ASSETS_ROBOTS_DATA_DIR}/spot_policy.pt"

    action_scale: float = 0.2

    throttle_scale: float = 1.5
    steering_scale: float = 1.0
    throttle_max: float = 4.5
    steering_max: float = 3.0

    smoothing_factor: list[float] = [0.75, 0.3, 0.3]
