from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg

from brain_sim_assets import BRAIN_SIM_ASSETS_PROPS_DATA_DIR


class bsKnifeHolder:

    @staticmethod
    def get_knife_holder_asset(
        prim_path: str = "{ENV_REGEX_NS}/KnifeHolder",
        pos: tuple = (0.0, 0.0, 0.0),
        rot: tuple = (1.0, 0.0, 0.0, 0.0),
        scale: tuple = (1.0, 1.0, 1.0),
    ) -> AssetBaseCfg:
        return AssetBaseCfg(
            prim_path=prim_path,
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
            spawn=UsdFileCfg(
                usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/kitchen/Kitchen_Other/Kitchen_KnifeHolders001.usd",
                scale=scale,
            ),
        )
