import os
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from brain_sim_assets import BRAIN_SIM_ASSETS_PROPS_DATA_DIR


class BrainSimBrick:
    
    @staticmethod
    def get_brick_object(prim_path: str = "{ENV_REGEX_NS}/Brick", 
                         pos: tuple = (1.0, 0.0, 0.5), 
                         rot: tuple = (1, 0, 0, 0),
                         scale: tuple = (0.3, 0.3, 0.3)) -> RigidObjectCfg:
        return RigidObjectCfg(
            prim_path=prim_path,
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
            spawn=UsdFileCfg(
                usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/brick.usd",
                scale=scale,
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                    max_angular_velocity=150,
                    max_linear_velocity=150,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
            ),
        )
    
    @classmethod
    def create_multiple_objects(cls, object_configs: list) -> dict:
        objects = {}
        
        for config in object_configs:
            obj_type = config['type']
            obj_name = config['name']
            params = config.get('params', {})
            
            if obj_type == 'brick':
                objects[obj_name] = cls.get_brick_object(**params)
            else:
                raise ValueError(f"Unknown object type: {obj_type}")
                
        return objects
    
    @staticmethod
    def get_default_rigid_props(gravity_enabled: bool = False, 
                                max_velocity: int = 200) -> RigidBodyPropertiesCfg:
        return RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
            max_angular_velocity=max_velocity,
            max_linear_velocity=max_velocity,
            max_depenetration_velocity=1.0,
            disable_gravity=not gravity_enabled,
        )
