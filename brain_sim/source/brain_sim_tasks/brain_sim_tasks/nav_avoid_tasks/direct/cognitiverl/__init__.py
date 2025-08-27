# Copyright (c) 2022-2025, The Template Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
cognitiverl driving environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Brain-Sim-Leatherback-Nav-v0",
    entry_point=f"{__name__}.leatherback_nav_env:LeatherbackNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leatherback_nav_env_cfg:LeatherbackNavEnvCfg",
    },
)

gym.register(
    id="Brain-Sim-Spot-Nav-v0",
    entry_point=f"{__name__}.spot_nav_env:SpotNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_nav_env_cfg:SpotNavEnvCfg",
    },
)

gym.register(
    id="Brain-Sim-Spot-Nav-Rough-v0",
    entry_point=f"{__name__}.spot_nav_rough_height_env:SpotNavRoughHeightEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_nav_rough_env_cfg:SpotNavRoughEnvCfg",
    },
)

gym.register(
    id="Brain-Sim-Spot-Nav-Rough-v1",
    entry_point=f"{__name__}.spot_nav_rough_env:SpotNavRoughEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_nav_rough_env_cfg:SpotNavRoughEnvCfg",
    },
)

gym.register(
    id="Brain-Sim-Spot-Nav-Avoid-v0",
    entry_point=f"{__name__}.spot_nav_avoid_env:SpotNavAvoidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_nav_avoid_env_cfg:SpotNavAvoidEnvCfg",
    },
)

gym.register(
    id="Brain-Sim-Spot-Nav-Grid-v0",
    entry_point=f"{__name__}.spot_nav_rough_grid_height_env:SpotNavRoughGridHeightEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_nav_rough_env_cfg:SpotNavRoughGridHeightEnvCfg",
    },
)
