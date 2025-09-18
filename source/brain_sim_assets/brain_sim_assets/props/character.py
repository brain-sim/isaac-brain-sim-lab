from typing import List, Optional, Dict
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from brain_sim_assets import BRAIN_SIM_ASSETS_PROPS_DATA_DIR
import random


class bsCharacterGenerator:

    def __init__(
        self,
        character_usd_path: Optional[str] = None,
        position_offset: tuple = (0.0, 0.0, 0.0),
        boundary_limit: float = 20.0,
        num_characters: int = 1,
    ):
        self._characters = []
        self._character_usd_path = character_usd_path
        self._position_offset = position_offset
        self._boundary_limit = boundary_limit
        self._num_characters = num_characters

        if character_usd_path:
            self.create_characters()

    @classmethod
    def create_example_characters(
        cls,
        character_usd_path: Optional[str] = None,
        position_offset: tuple = (0.0, 0.0, 0.0),
        boundary_limit: float = 20.0,
        num_characters: int = 1,
    ):
        if character_usd_path is None:
            character_usd_path = f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/character.usd"
        
        return cls(character_usd_path, position_offset, boundary_limit, num_characters)

    def set_character_usd_path(self, character_usd_path: str):
        self._character_usd_path = character_usd_path

    def _generate_random_position(self) -> tuple:
        x = random.uniform(-self._boundary_limit, self._boundary_limit) + self._position_offset[0]
        y = random.uniform(-self._boundary_limit, self._boundary_limit) + self._position_offset[1]
        z = self._position_offset[2]
        return (x, y, z)

    def create_characters(self):
        for i in range(self._num_characters):
            position = self._generate_random_position()
            
            character_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/character_{i}",
                spawn=UsdFileCfg(
                    usd_path=self._character_usd_path,
                    rigid_props=RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        max_linear_velocity=10.0,
                        max_angular_velocity=10.0,
                        max_depenetration_velocity=1.0,
                        enable_gyroscopic_forces=False,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=position,
                    rot=(1, 0, 0, 0),
                ),
            )
            
            self._characters.append(character_cfg)

    def get_characters(self) -> List[RigidObjectCfg]:
        return self._characters

    def get_character_collection(self) -> Dict[str, RigidObjectCfg]:
        character_dict = {}
        for i, character_cfg in enumerate(self._characters):
            character_dict[f"character_{i}"] = character_cfg
        return character_dict