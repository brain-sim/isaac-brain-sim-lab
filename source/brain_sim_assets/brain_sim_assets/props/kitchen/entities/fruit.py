from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg

from brain_sim_assets import BRAIN_SIM_ASSETS_PROPS_DATA_DIR


_FRUIT_PATHS = {
    "orange1": "kitchen/Kitchen_Other/Kitchen_Orange001.usd",
    "orange2": "kitchen/Kitchen_Other/Kitchen_Orange002.usd",
    "orange_1": "kitchen/orange_1/orange_1.usd"
}


class bsKitchenFruit:

    @staticmethod
    def get_fruit_asset(
        prim_path: str = "{ENV_REGEX_NS}/Fruit",
        pos: tuple = (0.0, 0.0, 0.0),
        rot: tuple = (1.0, 0.0, 0.0, 0.0),
        scale: tuple = (1.0, 1.0, 1.0),
        variant: str = "orange_1",
    ) -> AssetBaseCfg:
        usd_rel_path = _FRUIT_PATHS.get(variant, _FRUIT_PATHS["orange_1"])
        return AssetBaseCfg(
            prim_path=prim_path,
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
            spawn=UsdFileCfg(
                usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/{usd_rel_path}",
                scale=scale,
            ),
        )
