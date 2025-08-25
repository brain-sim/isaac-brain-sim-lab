# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing brain sim asset configurations."""

import os
import toml

# Conveniences to other module directories via relative paths
BRAIN_SIM_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

BRAIN_SIM_ASSETS_DATA_DIR = os.path.join(os.path.dirname(__file__), "props", "data")
"""Path to the extension data directory."""

BRAIN_SIM_ASSETS_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "props", "config")
"""Path to the extension config directory."""

BRAIN_SIM_ASSETS_METADATA = toml.load(os.path.join(BRAIN_SIM_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = BRAIN_SIM_ASSETS_METADATA["package"]["version"]

from .props import *