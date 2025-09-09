# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing brain sim ant maze utilities."""

import os
import toml

# Conveniences to other module directories via relative paths
BRAIN_SIM_ANT_MAZE_EXT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../")
)
"""Path to the extension source directory."""

BRAIN_SIM_ANT_MAZE_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
"""Path to the extension data directory."""

BRAIN_SIM_ANT_MAZE_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
"""Path to the extension config directory."""

BRAIN_SIM_ANT_MAZE_METADATA = toml.load(
    os.path.join(BRAIN_SIM_ANT_MAZE_EXT_DIR, "config", "extension.toml")
)
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = BRAIN_SIM_ANT_MAZE_METADATA["package"]["version"]

from .ant_maze import bsAntMaze
from .ant_maze_config import bsAntMazeConfig, bsAntMazeDimensions, bsAntMazeVisuals

__all__ = ["bsAntMaze", "bsAntMazeConfig", "bsAntMazeDimensions", "bsAntMazeVisuals"]
