from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg

from brain_sim_assets import BRAIN_SIM_ASSETS_PROPS_DATA_DIR


_BOTTLE_PATHS = {
    "glass": "kitchen/Kitchen_Other/Kitchen_bottle002.usd",
    "metal": "kitchen/Kitchen_Other/Kitchen_Bottle.usd",
    "tall": "kitchen/Kitchen_Other/Kitchen_bottle004.usd",
}


class bsKitchenBottle:

    @staticmethod
    def get_bottle_asset(
        prim_path: str = "{ENV_REGEX_NS}/Bottle",
        pos: tuple = (0.0, 0.0, 0.0),
        rot: tuple = (1.0, 0.0, 0.0, 0.0),
        scale: tuple = (1.0, 1.0, 1.0),
        variant: str = "glass",
    ) -> AssetBaseCfg:
        default_path = next(iter(_BOTTLE_PATHS.values()))
        usd_rel_path = _BOTTLE_PATHS.get(variant, default_path)
        return AssetBaseCfg(
            prim_path=prim_path,
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
            spawn=UsdFileCfg(
                usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/{usd_rel_path}",
                scale=scale,
            ),
        )
