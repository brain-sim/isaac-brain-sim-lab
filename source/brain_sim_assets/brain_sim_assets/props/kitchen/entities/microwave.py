from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg

from brain_sim_assets import BRAIN_SIM_ASSETS_PROPS_DATA_DIR


class bsMicrowave:

    @staticmethod
    def get_microwave_object(
        prim_path: str = "{ENV_REGEX_NS}/Microwave",
        pos: tuple = (0.0, 0.0, 0.0),
        rot: tuple = (1.0, 0.0, 0.0, 0.0),
        scale: tuple = (1.0, 1.0, 1.0),
    ) -> AssetBaseCfg:
        return AssetBaseCfg(
            prim_path=prim_path,
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
            spawn=UsdFileCfg(
                usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/kitchen/Microwave017/Microwave017.usd",
                scale=scale,
            ),
        )
