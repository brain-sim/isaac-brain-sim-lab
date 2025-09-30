from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg

from brain_sim_assets import BRAIN_SIM_ASSETS_PROPS_DATA_DIR


class bsKitchenPot:

    @staticmethod
    def get_pot_asset(
        prim_path: str = "{ENV_REGEX_NS}/Pot",
        pos: tuple = (0.0, 0.0, 0.0),
        rot: tuple = (1.0, 0.0, 0.0, 0.0),
        scale: tuple = (0.01, 0.01, 0.01),
    ) -> AssetBaseCfg:
        return AssetBaseCfg(
            prim_path=prim_path,
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
            spawn=UsdFileCfg(
                usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/kitchen/Pot057/Pot057.usd",
                scale=scale,
            ),
        )
