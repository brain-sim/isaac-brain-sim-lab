import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg

from brain_sim_assets import BRAIN_SIM_ASSETS_PROPS_DATA_DIR

##
# configuration
##


class bsCharacterGenerator:

    # @staticmethod
    # def get_character_object() -> VisualizationMarkersCfg:
    #     return VisualizationMarkersCfg(
    #         prim_path="/World/Visuals/Characters",
    #         markers={
    #             "worker": sim_utils.UsdFileCfg( 
    #                 usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/characters/worker.usd",
    #             ),
    #             "business_f": sim_utils.UsdFileCfg(
    #                 usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/characters/business-f-0002.usd",
    #             ),
    #             "party_m": sim_utils.UsdFileCfg(
    #                 usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/characters/party-m-0001.usd",
    #             ),
    #             "uniform_f": sim_utils.UsdFileCfg(
    #                 usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/characters/uniform_f_0001.usd",
    #             ),
    #             "uniform_m": sim_utils.UsdFileCfg(
    #                 usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/characters/uniform_m_0001.usd",
    #             ),
    #         },
    #     )

    def get_character_object() -> VisualizationMarkersCfg:
        return VisualizationMarkersCfg(
            prim_path="/World/Visuals/Characters",
            markers={
                "worker": sim_utils.UsdFileCfg( 
                    usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/characters/worker.usd",
                ),
                "business_f": sim_utils.UsdFileCfg(
                    usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/characters/worker.usd",
                ),
                "party_m": sim_utils.UsdFileCfg(
                    usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/characters/worker.usd",
                ),
                "uniform_f": sim_utils.UsdFileCfg(
                    usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/characters/worker.usd",
                ),
                "uniform_m": sim_utils.UsdFileCfg(
                    usd_path=f"{BRAIN_SIM_ASSETS_PROPS_DATA_DIR}/characters/worker.usd",
                ),
            },
        )
