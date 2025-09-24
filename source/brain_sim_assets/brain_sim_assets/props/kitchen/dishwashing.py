from typing import Dict

from isaaclab.assets import AssetBaseCfg

from brain_sim_assets.props.kitchen.dishwasher import bsDishwasher
from brain_sim_assets.props.kitchen.kitchen_background import bsKitchenBackground


class bsDishwashingEntitiesGenerator:

    @staticmethod
    def get_dishwashing_entities(
        base_pos: tuple = (0.0, -5.0, 0.0),
        include_dishwasher: bool = True,
        include_background: bool = True,
        dishwasher_name: str = "dishwasher",
        dishwasher_prim_path: str = "{ENV_REGEX_NS}/Dishwasher",
        dishwasher_pos: tuple = (2.07, -0.3, 0.4),
        dishwasher_rot: tuple = (1.0, 0.0, 0.0, 0.0),
        dishwasher_scale: tuple = (1.0, 1.0, 1.0),
        background_name: str = "kitchen_background",
        background_prim_path: str = "{ENV_REGEX_NS}/KitchenBackground",
        background_pos: tuple = (0.0, 0.0, 0.0),
        background_rot: tuple = (1.0, 0.0, 0.0, 0.0),
        background_scale: tuple = (1.0, 1.0, 1.0),
        extra_assets: Dict[str, AssetBaseCfg] | None = None,
    ) -> Dict[str, AssetBaseCfg]:
        assets: Dict[str, AssetBaseCfg] = dict(extra_assets or {})
        def _offset(pos: tuple) -> tuple:
            return tuple(a + b for a, b in zip(pos, base_pos))
        if include_dishwasher:
            assets[dishwasher_name] = bsDishwasher.get_dishwasher_object(
                prim_path=dishwasher_prim_path,
                pos=_offset(dishwasher_pos),
                rot=dishwasher_rot,
                scale=dishwasher_scale,
            )
        if include_background:
            assets[background_name] = bsKitchenBackground.get_background_asset(
                prim_path=background_prim_path,
                pos=_offset(background_pos),
                rot=background_rot,
                scale=background_scale,
            )
        return assets

    @staticmethod
    def add_asset(
        collection: Dict[str, AssetBaseCfg],
        name: str,
        asset: AssetBaseCfg,
    ) -> Dict[str, AssetBaseCfg]:
        new_collection = dict(collection)
        new_collection[name] = asset
        return new_collection
