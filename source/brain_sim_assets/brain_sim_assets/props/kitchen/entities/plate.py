from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg

from brain_sim_assets import BRAIN_SIM_ASSETS_PROPS_DATA_DIR


_PLATE_PATHS = {
    "disk1": "Kitchen_Other/Kitchen_Disk001.usd",
    "disk2": "Kitchen_Other/Kitchen_Disk002.usd",
}


class bsKitchenPlate:

    @staticmethod
    def get_plate_asset(
        prim_path: str = "{ENV_REGEX_NS}/Plate",
        pos: tuple = (0.0, 0.0, 0.0),
        rot: tuple = (1.0, 0.0, 0.0, 0.0),
        scale: tuple = (1.0, 1.0, 1.0),
        variant: str = "disk1",
    ) -> AssetBaseCfg:
        usd_rel_path = _PLATE_PATHS.get(variant, _PLATE_PATHS["disk1"])
        return AssetBaseCfg(
            prim_path=prim_path,
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
            spawn=UsdFileCfg(
                usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/kitchen/{usd_rel_path}",
                scale=scale,
            ),
        )
