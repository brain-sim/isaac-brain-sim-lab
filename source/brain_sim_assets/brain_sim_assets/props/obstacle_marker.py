import os
import random
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import spawners as sim_utils
from isaaclab.sim.schemas.schemas_cfg import (
    RigidBodyPropertiesCfg,
    MassPropertiesCfg,
    CollisionPropertiesCfg,
)


class bsObstacleMarkerGenerator:

    @staticmethod
    def get_obstacle_marker_object(
        prim_path: str = "{ENV_REGEX_NS}/ObstacleMarker",
        pos: tuple | None = None,
        rot: tuple = (1, 0, 0, 0),
        radius: float = 0.2,
        height: float = 1.0,
        color: tuple = (1.0, 0.0, 0.0),
        x_range: tuple = (11.2, 27.0),
        y_range: tuple = (0.2, 27.0),
        z_height: float = 0.5,
    ) -> RigidObjectCfg:

        # Generate random position if not provided
        if pos is None:
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            pos = (x, y, z_height)

        return RigidObjectCfg(
            prim_path=prim_path,
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
            spawn=sim_utils.ConeCfg(
                radius=radius,
                height=height,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color, metallic=0.2
                ),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                    max_angular_velocity=150,
                    max_linear_velocity=150,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=100.0),
                collision_props=CollisionPropertiesCfg(),
            ),
        )
