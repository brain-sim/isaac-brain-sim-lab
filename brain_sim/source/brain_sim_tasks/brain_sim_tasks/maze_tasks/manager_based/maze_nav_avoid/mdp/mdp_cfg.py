from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .action_cfg import HierarchicalSpotActionCfg
from .obs_cfg import PolicyCfg
from .reward_cfg import BaseRewardsCfg
from .terminate_cfg import BaseTerminationsCfg

@configclass
class CommandsCfg:
    pass # No explicit commands needed

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