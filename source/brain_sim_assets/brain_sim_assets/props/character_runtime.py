from brain_sim_assets.props.character import bsCharacterGenerator
from brain_sim_assets import BRAIN_SIM_ASSETS_PROPS_DATA_DIR
import torch
import random
import numpy as np
import math
from typing import Dict
import omni.usd
from pxr import Usd, UsdGeom, Gf


class bsCharacterRuntime:

    def __init__(
        self,
        room_size: float = 40.0,
        character_usd_path: str = "worker.test2.usd",
        num_characters: int = 3,
        boundary_limit: float = 20.0,
        device: str | torch.device = "cuda",
    ):
        self.room_size = room_size
        self.character_usd_path = character_usd_path
        self.num_characters = num_characters
        self.boundary_limit = boundary_limit
        self.device = torch.device(device) if isinstance(device, str) else device
        self._character_generator = None
        self._character_poses = None

    def update_device(self, device: str | torch.device):
        new_device = torch.device(device) if isinstance(device, str) else device
        
        if new_device != self.device:
            self.device = new_device
            
            if self._character_poses is not None:
                self._character_poses = self._character_poses.to(self.device)

    def _get_character_generator(self):
        if self._character_generator is None:
            offset = self.get_position_offset()
            self._character_generator = bsCharacterGenerator.create_example_characters(
                f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/{self.character_usd_path}",
                position_offset=offset,
                boundary_limit=self.boundary_limit,
                num_characters=self.num_characters,
            )
        return self._character_generator

    def get_position_offset(self) -> tuple:
        return (0.0, 0.0, 0.0)

    def create_character_configuration(
        self, 
        character_usd_path: str | None = None,
        num_characters: int | None = None,
    ):
        if character_usd_path is None:
            character_usd_path = self.character_usd_path
        if num_characters is None:
            num_characters = self.num_characters
            
        offset = self.get_position_offset()
        character_gen = bsCharacterGenerator.create_example_characters(
            f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/{character_usd_path}",
            position_offset=offset,
            boundary_limit=self.boundary_limit,
            num_characters=num_characters,
        )
        
        # Store initial character poses
        if self._character_poses is None:
            self._character_poses = torch.zeros(
                (num_characters, 7), device=self.device, dtype=torch.float32
            )
            characters = character_gen.get_characters()
            for i, char_cfg in enumerate(characters):
                # Extract position from character config
                pos = char_cfg.init_state.pos
                rot = char_cfg.init_state.rot
                self._character_poses[i, :3] = torch.tensor(pos, device=self.device, dtype=torch.float32)
                self._character_poses[i, 3:] = torch.tensor(rot, device=self.device, dtype=torch.float32)

        return character_gen.get_character_collection()

    def apply_to_scene_cfg(
        self, 
        scene, 
        character_usd_path: str | None = None,
        num_characters: int | None = None,
    ):
        if character_usd_path is None:
            character_usd_path = self.character_usd_path
        if num_characters is None:
            num_characters = self.num_characters
            
        character_configs = self.create_character_configuration(character_usd_path, num_characters)
        for name, cfg in character_configs.items():
            setattr(scene, name, cfg)

    def update_character_poses(self, new_poses: torch.Tensor):
        """Update the character poses with new pose data."""
        self._character_poses = new_poses

    def get_character_distances(self, robot_positions: torch.Tensor) -> torch.Tensor:
        if self._character_poses is None or self._character_poses.numel() == 0:
            return torch.full(
                (len(robot_positions),),
                float("inf"),
                device=robot_positions.device,
                dtype=torch.float32,
            )
            
        robot_xy = robot_positions[:, :2]
        character_xy = self._character_poses[:, :2]
        
        N = robot_xy.shape[0]
        M = character_xy.shape[0]
        
        robots = robot_xy.unsqueeze(1)  # (N, 1, 2)
        characters = character_xy.unsqueeze(0)  # (1, M, 2)
        
        distances_to_characters = torch.norm(robots - characters, dim=2)  # (N, M)
        
        min_distances, _ = torch.min(distances_to_characters, dim=1)  # (N,)
        
        return min_distances

    def get_character_penalty(self, robot_positions: torch.Tensor, penalty_scale: float = 1.0) -> torch.Tensor:
        distances = self.get_character_distances(robot_positions)
        
        penalty = penalty_scale / (distances + 1e-6)
        
        return penalty