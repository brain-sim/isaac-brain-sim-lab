# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .action_cfg import HierarchicalSpotActionCfg
from .obs_cfg import PolicyCfg

@configclass
class CommandsCfg:
    pass

@configclass
class ActionsCfg:
    hierarchical_spot = HierarchicalSpotActionCfg()

@configclass
class ObservationsCfg:
    policy = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-1.5, 1.5),
                "y": (-1.0, 1.0),
                "z": (-0.5, 0.5),
                "roll": (-0.7, 0.7),
                "pitch": (-0.7, 0.7),
                "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

@configclass
class RewardsCfg:
    
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
    base_orientation = RewTerm(
        func=spot_mdp.base_orientation_penalty, 
        weight=-1.0, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
