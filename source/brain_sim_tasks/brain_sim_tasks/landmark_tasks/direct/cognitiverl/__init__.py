# Copyright (c) 2022-2025, The Template Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
cognitiverl driving environment. Tests and trains
"""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Train-Single-Landmark-v0",
    entry_point=f"{__name__}.single.landmark_env:LandmarkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.single.landmark_env_cfg:LandmarkEnvCfg",
    },
)

gym.register(
    id="Test-Single-Landmark-v0",
    entry_point=f"{__name__}.single.landmark_env:LandmarkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.single.landmark_env_cfg:LandmarkEnvCfg",
    },
)

gym.register(
    id="Train-Rot-Array-Landmark-v0",
    entry_point=f"{__name__}.rot_array.landmark_env:LandmarkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rot_array.landmark_env_cfg:LandmarkEnvCfg",
    },
)

gym.register(
    id="Test-Rot-Array-Landmark-v0",
    entry_point=f"{__name__}.rot_array.landmark_env:LandmarkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rot_array.landmark_env_cfg:LandmarkEnvCfg",
    },
)

gym.register(
    id="Train-Single-Landmark-Fix-Goal-v0",
    entry_point=f"{__name__}.single_fix_goal.landmark_env:LandmarkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.single_fix_goal.landmark_env_cfg:LandmarkEnvCfg",
    },
)

gym.register(
    id="Test-Single-Landmark-Fix-Goal-v0",
    entry_point=f"{__name__}.single_fix_goal.landmark_env:LandmarkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.single_fix_goal.landmark_env_cfg:LandmarkEnvCfg",
    },
)

gym.register(
    id="Train-Rot-Array-Landmark-Fix-Goal-v0",
    entry_point=f"{__name__}.rot_array_fix_goal.landmark_env:LandmarkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rot_array_fix_goal.landmark_env_cfg:LandmarkEnvCfg",
    },
)

gym.register(
    id="Test-Rot-Array-Landmark-Fix-Goal-v0",
    entry_point=f"{__name__}.rot_array_fix_goal.landmark_env:LandmarkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rot_array_fix_goal.landmark_env_cfg:LandmarkEnvCfg",
    },
)
