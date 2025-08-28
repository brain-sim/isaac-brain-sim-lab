from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from brain_sim_assets import BRAIN_SIM_ASSETS_PROPS_DATA_DIR


class bsBrickGenerator:
    
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