from typing import Dict

from isaaclab.assets import AssetBaseCfg

from brain_sim_assets.props.kitchen.entities.dishwasher import bsDishwasher
from brain_sim_assets.props.kitchen.entities.kitchen_background import bsKitchenBackground
from brain_sim_assets.props.kitchen.entities.microwave import bsMicrowave
from brain_sim_assets.props.kitchen.entities.table import bsKitchenTable
from brain_sim_assets.props.kitchen.entities.plate import bsKitchenPlate
from brain_sim_assets.props.kitchen.entities.bottle import bsKitchenBottle
from brain_sim_assets.props.kitchen.entities.fruit import bsKitchenFruit
from brain_sim_assets.props.kitchen.entities.knife_holder import bsKnifeHolder


class bsDishwashingEntitiesGenerator:

    @staticmethod
    def get_dishwashing_entities(
        base_pos: tuple = (0.0, -5.0, 0.0),
        include_dishwasher: bool = True,
        include_background: bool = True,
        include_microwave: bool = True,
        include_table: bool = True,
        include_objects: bool = True,
        dishwasher_name: str = "dishwasher",
        dishwasher_prim_path: str = "{ENV_REGEX_NS}/Dishwasher",
        dishwasher_pos: tuple = (0.5, -0.3, 0.4),
        dishwasher_rot: tuple = (1.0, 0.0, 0.0, 0.0),
        dishwasher_scale: tuple = (1.0, 1.0, 1.0),
        background_name: str = "kitchen_background",
        background_prim_path: str = "{ENV_REGEX_NS}/KitchenBackground",
        background_pos: tuple = (0.0, 0.0, 0.0),
        background_rot: tuple = (1.0, 0.0, 0.0, 0.0),
        background_scale: tuple = (1.0, 1.0, 1.0),
        microwave_name: str = "microwave",
        microwave_prim_path: str = "{ENV_REGEX_NS}/Microwave",
        microwave_pos: tuple = (3.1, -0.25, 1.9),
        microwave_rot: tuple = (1.0, 0.0, 0.0, 0.0),
        microwave_scale: tuple = (1.0, 1.0, 1.0),
        table_name: str = "service_table",
        table_prim_path: str = "{ENV_REGEX_NS}/ServiceTable",
        table_pos: tuple = (1.8, 0.2, 0.0),
        plate_name: str = "service_plate",
        plate_pos: tuple = (1.75, 0.05, 0.95),
        bottle_name: str = "service_bottle",
        bottle_pos: tuple = (1.95, 0.15, 0.95),
        fruit_name: str = "service_fruit",
        fruit_pos: tuple = (1.85, -0.2, 0.95),
        knife_holder_name: str = "service_knives",
        knife_holder_pos: tuple = (2.0, -0.05, 0.95),
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
        if include_microwave:
            assets[microwave_name] = bsMicrowave.get_microwave_object(
                prim_path=microwave_prim_path,
                pos=_offset(microwave_pos),
                rot=microwave_rot,
                scale=microwave_scale,
            )
        if include_table:
            assets[table_name] = bsKitchenTable.get_table_asset(
                prim_path=table_prim_path,
                pos=_offset(table_pos),
            )
        if include_objects:
            assets[plate_name] = bsKitchenPlate.get_plate_asset(
                prim_path=f"{table_prim_path}/Plate",
                pos=_offset(plate_pos),
            )
            assets[bottle_name] = bsKitchenBottle.get_bottle_asset(
                prim_path=f"{table_prim_path}/Bottle",
                pos=_offset(bottle_pos),
            )
            assets[fruit_name] = bsKitchenFruit.get_fruit_asset(
                prim_path=f"{table_prim_path}/Fruit",
                pos=_offset(fruit_pos),
            )
            assets[knife_holder_name] = bsKnifeHolder.get_knife_holder_asset(
                prim_path=f"{table_prim_path}/Knives",
                pos=_offset(knife_holder_pos),
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
