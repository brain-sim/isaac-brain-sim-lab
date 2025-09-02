# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import toml

BRAIN_SIM_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

BRAIN_SIM_ASSETS_PROPS_DATA_DIR = os.path.join(os.path.dirname(__file__), "props", "data")
BRAIN_SIM_ASSETS_PROPS_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "props", "config")
BRAIN_SIM_ASSETS_ROBOTS_DATA_DIR = os.path.join(os.path.dirname(__file__), "robots", "data")
BRAIN_SIM_ASSETS_ROBOTS_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "robots", "config")

BRAIN_SIM_ASSETS_METADATA = toml.load(os.path.join(BRAIN_SIM_ASSETS_EXT_DIR, "config", "extension.toml"))

__version__ = BRAIN_SIM_ASSETS_METADATA["package"]["version"]
