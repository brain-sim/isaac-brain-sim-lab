from typing import Dict

from isaaclab.assets import AssetBaseCfg

from brain_sim_assets.props.kitchen.entities.dishwasher import bsDishwasher
from brain_sim_assets.props.kitchen.entities.kitchen_background import bsKitchenBackground
from brain_sim_assets.props.kitchen.entities.table import bsKitchenTable
from brain_sim_assets.props.kitchen.entities.bottle import bsKitchenBottle
from brain_sim_assets.props.kitchen.entities.plate import bsKitchenPlate
from brain_sim_assets.props.kitchen.entities.fruit import bsKitchenFruit
from brain_sim_assets.props.kitchen.entities.knife_holder import bsKnifeHolder


class bsDishwashingEntitiesGenerator:

    @staticmethod
    def get_dishwashing_entities(
        base_pos: tuple = (0.0, -1.0, 0.0),
        include_background: bool = True,
        include_dishwasher: bool = True,
        include_table: bool = True,
        include_objects: bool = True,
        dishwasher_name: str = "dishwasher",
        dishwasher_prim_path: str = "{ENV_REGEX_NS}/Dishwasher",
        dishwasher_pos: tuple = (2.075, -0.3, 0.4),
        dishwasher_rot: tuple = (1.0, 0.0, 0.0, 0.0),
        dishwasher_scale: tuple = (1.0, 1.0, 1.0),
        background_name: str = "kitchen_background",
        background_prim_path: str = "{ENV_REGEX_NS}/KitchenBackground",
        background_pos: tuple = (0.0, 0.0, 0.0),
        background_rot: tuple = (1.0, 0.0, 0.0, 0.0),
        background_scale: tuple = (1.0, 1.0, 1.0),
        table_name: str = "prep_table",
        table_prim_path: str = "{ENV_REGEX_NS}/PrepTable",
        table_pos: tuple = (1.17352, -1.58922, 0.0),
        table_rot: tuple = (1.0, 0.0, 0.0, 0.0),
        bottle_name: str = "detergent_bottle",
        bottle_pos: tuple = (0.27634, -0.06814, 0.90886),
        fruit_name: str = "rinsed_fruit",
        fruit_pos: tuple = (0.37143, -0.19129, 0.79806),
        plate_name: str = "dirty_plate",
        plate_pos: tuple = (0.51002, 0.51002, 0.76827),
        knife_holder_name: str = "knife_block",
        knife_holder_pos: tuple = (-0.04662, 0.29401, 0.90415),
        extra_assets: Dict[str, AssetBaseCfg] | None = None,
    ) -> Dict[str, AssetBaseCfg]:
        assets: Dict[str, AssetBaseCfg] = dict(extra_assets or {})

        def _offset(pos: tuple) -> tuple:
            return tuple(a + b for a, b in zip(pos, base_pos))

        if include_background:
            assets[background_name] = bsKitchenBackground.get_background_asset(
                prim_path=background_prim_path,
                pos=_offset(background_pos),
                rot=background_rot,
                scale=background_scale,
            )

        if include_dishwasher:
            assets[dishwasher_name] = bsDishwasher.get_dishwasher_object(
                prim_path=dishwasher_prim_path,
                pos=_offset(dishwasher_pos),
                rot=dishwasher_rot,
                scale=dishwasher_scale,
            )
        if include_table:
            assets[table_name] = bsKitchenTable.get_table_asset(
                prim_path=table_prim_path,
                pos=_offset(table_pos),
                rot=table_rot,
            )
        if include_objects:
            assets[plate_name] = bsKitchenPlate.get_plate_asset(
                prim_path=f"{table_prim_path}/Plate",
                pos=(plate_pos[0], plate_pos[1], plate_pos[2]),
            )
            assets[bottle_name] = bsKitchenBottle.get_bottle_asset(
                prim_path=f"{table_prim_path}/Bottle",
                pos=(bottle_pos[0], bottle_pos[1], bottle_pos[2]),
            )
            assets[fruit_name] = bsKitchenFruit.get_fruit_asset(
                prim_path=f"{table_prim_path}/Fruit",
                pos=(fruit_pos[0], fruit_pos[1], fruit_pos[2]),
            )
            assets[knife_holder_name] = bsKnifeHolder.get_knife_holder_asset(
                prim_path=f"{table_prim_path}/KnifeHolder",
                pos=(knife_holder_pos[0], knife_holder_pos[1], knife_holder_pos[2]),
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
