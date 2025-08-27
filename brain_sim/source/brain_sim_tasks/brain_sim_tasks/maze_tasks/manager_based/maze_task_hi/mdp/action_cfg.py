import torch
from dataclasses import MISSING

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.assets import Articulation

from brain_sim_assets.robots.spot import bsSpotLowLevelPolicyVanilla
from brain_sim_assets import BRAIN_SIM_ASSETS_ROBOTS_DATA_DIR



class HierarchicalSpotActionTerm(ActionTerm):

    cfg: "HierarchicalSpotActionCfg"

    def __init__(self, cfg: "HierarchicalSpotActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        self._low_level_policy = bsSpotLowLevelPolicyVanilla(cfg.policy_file_path)
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._num_envs = env.num_envs
        self._device = env.device
        
        self._low_level_previous_action = torch.zeros(
            (self._num_envs, 12), device=self._device, dtype=torch.float32
        )
        self._default_pos = self._robot.data.default_joint_pos.clone()
        self._action_scale = cfg.action_scale
        
        # High-level command smoothing
        self._smoothing_factor = torch.tensor(cfg.smoothing_factor, device=self._device)
        self._previous_command = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )

    def action_dim(self) -> int:
        """Dimension of the action term."""
        return 12

    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term."""
        raise NotImplementedError

    def processed_actions(self) -> torch.Tensor:
        """The actions computed by the term after applying any processing."""
        raise NotImplementedError
    
    def process_actions(self, actions: torch.Tensor):
        """Convert high-level commands to joint positions using low-level policy.
        
        Args:
            actions: High-level commands [num_envs, 3] containing (vx, vy, omega)
        """
        smoothed_command = (
            self._smoothing_factor * actions + 
            (1 - self._smoothing_factor) * self._previous_command
        )
        self._previous_command = smoothed_command.clone()
        
        # Scale commands appropriately
        command = torch.zeros_like(smoothed_command)
        command[:, 0] = smoothed_command[:, 0] * self.cfg.throttle_scale  # vx
        command[:, 1] = smoothed_command[:, 1] * self.cfg.lateral_scale   # vy  
        command[:, 2] = smoothed_command[:, 2] * self.cfg.steering_scale  # omega
        
        # Clamp commands to reasonable ranges
        command[:, 0] = torch.clamp(command[:, 0], 
                                   self.cfg.min_lin_vel_x, 
                                   self.cfg.max_lin_vel_x)
        command[:, 1] = torch.clamp(command[:, 1],
                                   self.cfg.min_lin_vel_y,
                                   self.cfg.max_lin_vel_y)
        command[:, 2] = torch.clamp(command[:, 2],
                                   self.cfg.min_ang_vel_z,
                                   self.cfg.max_ang_vel_z)
        
        # Get current robot state
        base_lin_vel = self._robot.data.root_lin_vel_b
        base_ang_vel = self._robot.data.root_ang_vel_b
        projected_gravity = self._robot.data.projected_gravity_b
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        
        # Get low-level actions from policy
        low_level_actions = self._low_level_policy.get_action(
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            command,
            self._low_level_previous_action,
            self._default_pos,
            joint_pos,
            joint_vel
        )
        
        # Update previous action buffer
        self._low_level_previous_action = low_level_actions.detach()
        
        # Convert to joint position targets
        joint_positions = self._default_pos + low_level_actions * self._action_scale
        
        # Return processed joint positions
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
    
    throttle_scale: float = 1.0
    lateral_scale: float = 1.0
    steering_scale: float = 0.5
    
    max_lin_vel_x: float = 3.0
    min_lin_vel_x: float = -2.0
    max_lin_vel_y: float = 1.5  
    min_lin_vel_y: float = -1.5
    max_ang_vel_z: float = 2.0
    min_ang_vel_z: float = -2.0
    
    smoothing_factor: list[float] = [0.7, 0.7, 0.5]