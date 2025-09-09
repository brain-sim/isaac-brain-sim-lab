import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg

##
# configuration
##


class bsWaypointGenerator:

    @staticmethod
    def get_waypoint_object(
        marker0_radius: float = 0.5,
        marker1_radius: float = 0.0,
        marker2_radius: float = 0.0,
        marker3_radius: float = 0.0,
    ) -> VisualizationMarkersCfg:
        return VisualizationMarkersCfg(
            prim_path="/World/Visuals/Cones",
            markers={
                "marker0": sim_utils.SphereCfg(  # future waypoint (yellow)
                    radius=marker0_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 1.0, 0.0)
                    ),
                ),
                "marker1": sim_utils.SphereCfg(  # current goal (green)
                    radius=marker1_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0), opacity=0.9
                    ),
                ),
                "marker2": sim_utils.SphereCfg(  # finished waypoint (invisible)
                    radius=marker2_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.0, 0.0), opacity=0.0
                    ),
                ),
                # axilary marker (cyan) for orientation eg. in rot_landmark task
                "marker3": sim_utils.SphereCfg(
                    radius=marker3_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 1.0)
                    ),
                ),
            },
        )
