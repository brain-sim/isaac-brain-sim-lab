import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .action_cfg import HierarchicalSpotActionCfg
from .obs_cfg import PolicyCfg
from .reward_cfg import BaseRewardsCfg
from .terminate_cfg import BaseTerminationsCfg

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-12.0, 12.0), 
            lin_vel_y=(-12.0, 12.0), 
            ang_vel_z=(-12.0, 12.0), 
            heading=(-0.65 * math.pi, 0.65 * math.pi)
        ),
    )

@configclass
class ActionsCfg:
    hierarchical_spot = HierarchicalSpotActionCfg()

@configclass
class ObservationsCfg:
    policy = PolicyCfg()

@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

@configclass
class RewardsCfg(BaseRewardsCfg):
    pass # Already inherits from BaseRewardsCfg

@configclass
class TerminationsCfg(BaseTerminationsCfg):
    pass # Already inherits from BaseTerminationsCfg