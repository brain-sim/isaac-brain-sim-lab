import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

def goal_reach(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, threshold: float = 1.4) -> torch.Tensor:

    robot_asset = env.scene[robot_cfg.name]
    robot_pos = robot_asset.data.root_pos_w  
    
    goal_markers = [name for name in env.scene.keys() if name.startswith('goal_marker_')]
    if not goal_markers:
        return torch.zeros(env.num_envs, device=env.device)
    
    goal_positions = torch.stack([
        env.scene[marker_name].data.root_pos_w for marker_name in goal_markers
    ])
    
    distances = torch.norm(
        goal_positions - robot_pos.unsqueeze(0), 
        dim=-1
    )
    
    goal_reached = (distances < threshold).any(dim=0).float()
    
    # Reward for reaching goal, minor penalty for not reaching goal
    rewards = goal_reached + (1.0 - goal_reached) * (-0.001)
    
    return rewards

def obstacle_reach(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, threshold: float = 0.6) -> torch.Tensor:

    robot_asset = env.scene[robot_cfg.name]
    robot_pos = robot_asset.data.root_pos_w

    obstacle_markers = [name for name in env.scene.keys() if name.startswith('obstacle_marker_')]
    
    if not obstacle_markers:
        return torch.zeros(env.num_envs, device=env.device)
    
    obstacle_positions = torch.stack([
        env.scene[marker_name].data.root_pos_w for marker_name in obstacle_markers
    ])
    
    distances = torch.norm(
        obstacle_positions - robot_pos.unsqueeze(0), 
        dim=-1
    )
    
    obstacle_reached = (distances < threshold).any(dim=0).float()
    
    # Penalty for reaching obstacle, minor reward for not reaching obstacle
    penalties = obstacle_reached + (1.0 - obstacle_reached) * (-0.00000000) # disable
    
    return penalties

@configclass
class BaseRewardsCfg:
    
    # Task-specific rewards for navigation command tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5}
    )
    
    # Penalize excessive commands
    action_rate = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.01
    )
    
    # Penalize undesired states
    base_motion = RewTerm(
        func=spot_mdp.base_motion_penalty, 
        weight=-0.5, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # Try to keep the base close to default orientation
    base_orientation = RewTerm(
        func=spot_mdp.base_orientation_penalty, 
        weight=-1.0, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # Goal reach reward
    goal_reach = RewTerm(
        func=goal_reach, 
        weight=300.0, 
        params={"robot_cfg": SceneEntityCfg("robot"), "threshold": 1.4}
    )

    # Obstacle avoidance penalty
    obstacle_penalty = RewTerm(
        func=obstacle_reach, 
        weight=-50.0, 
        params={"robot_cfg": SceneEntityCfg("robot"), "threshold": 0.6}
    )

    # Avoid penalty

    # Wall penalty
 