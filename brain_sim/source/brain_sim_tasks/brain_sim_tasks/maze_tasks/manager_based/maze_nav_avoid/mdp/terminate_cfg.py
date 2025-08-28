import torch

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


def goal_reach(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, threshold: float = 1.4) -> torch.Tensor:

    robot_asset = env.scene[robot_cfg.name]
    robot_pos = robot_asset.data.root_pos_w  
    
    goal_markers = [name for name in env.scene.keys() if name.startswith('goal_marker_')]
    if not goal_markers:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    goal_positions = torch.stack([
        env.scene[marker_name].data.root_pos_w for marker_name in goal_markers
    ])
    
    distances = torch.norm(
        goal_positions - robot_pos.unsqueeze(0), 
        dim=-1
    )
    
    # Return boolean tensor, not float
    terminated = (distances < threshold).any(dim=0)
    
    return terminated

def obstacle_reach(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, threshold: float = 0.6) -> torch.Tensor:

    robot_asset = env.scene[robot_cfg.name]
    robot_pos = robot_asset.data.root_pos_w

    obstacle_markers = [name for name in env.scene.keys() if name.startswith('obstacle_marker_')]
    
    if not obstacle_markers:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    obstacle_positions = torch.stack([
        env.scene[marker_name].data.root_pos_w for marker_name in obstacle_markers
    ])
    
    distances = torch.norm(
        obstacle_positions - robot_pos.unsqueeze(0), 
        dim=-1
    )
    
    # Return boolean tensor, not float
    terminated = (distances < threshold).any(dim=0)
    
    return terminated

@configclass
class BaseTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.7})
    goal_reach = DoneTerm(func=goal_reach, params={"robot_cfg": SceneEntityCfg("robot"), "threshold": 1.4})
    obstacle_reach = DoneTerm(func=obstacle_reach, params={"robot_cfg": SceneEntityCfg("robot"), "threshold": 0.6})
