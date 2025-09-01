import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils import configclass
from isaaclab.sensors.camera import TiledCamera

def get_image_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("camera")):
    asset: TiledCamera = env.scene[asset_cfg.name]
    image_obs = asset.data.output["rgb"].float().permute(0, 3, 1, 2) / 255.0
    image_obs = image_obs.flatten(start_dim=1)
    return image_obs

@configclass
class PolicyCfg(ObsGroup):
    """Observations for policy group."""

    # `` observation terms (order preserved)
    base_lin_vel = ObsTerm(
        func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
    )
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
    )
    image_obs = ObsTerm(
        func=get_image_obs, params={"asset_cfg": SceneEntityCfg("camera")}
    )

    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True

