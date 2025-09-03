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
    id="Train-2ACDC-v0",
    entry_point=f"{__name__}.linear_maze_env:LinearMazeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.linear_maze_env_cfg:LinearMazeEnvCfg",
    },
)

gym.register(
    id="Test-2ACDC-v0",
    entry_point=f"{__name__}.linear_maze_env:LinearMazeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.linear_maze_env_cfg:LinearMazeEnvCfg",
    },
)
