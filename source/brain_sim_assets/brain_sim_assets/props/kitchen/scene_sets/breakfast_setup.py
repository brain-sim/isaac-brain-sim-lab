from typing import Dict

from isaaclab.assets import AssetBaseCfg

from brain_sim_assets.props.kitchen.entities.kitchen_background import bsKitchenBackground
from brain_sim_assets.props.kitchen.entities.toaster import bsToaster
from brain_sim_assets.props.kitchen.entities.table import bsKitchenTable
from brain_sim_assets.props.kitchen.entities.plate import bsKitchenPlate
from brain_sim_assets.props.kitchen.entities.bottle import bsKitchenBottle
from brain_sim_assets.props.kitchen.entities.fruit import bsKitchenFruit
from brain_sim_assets.props.kitchen.entities.knife_holder import bsKnifeHolder


class bsBreakfastSetupEntitiesGenerator:

    @staticmethod
    def get_breakfast_setup_entities(
        base_pos: tuple = (0.0, -9.0, 0.0),
        include_background: bool = True,
        toaster_name: str = "toaster",
        toaster_prim_path: str = "{ENV_REGEX_NS}/Toaster",
        toaster_pos: tuple = (0.0, 0.0, 1.0),
        background_name: str = "kitchen_background",
        background_prim_path: str = "{ENV_REGEX_NS}/KitchenBackground",
        background_pos: tuple = (0.0, 0.0, 0.0),
        background_rot: tuple = (1.0, 0.0, 0.0, 0.0),
        background_scale: tuple = (1.0, 1.0, 1.0),
        table_name: str = "breakfast_table",
        table_prim_path: str = "{ENV_REGEX_NS}/BreakfastTable",
        table_pos: tuple = (1.17352, -1.58922, 0.0),
        plate_name: str = "toast_plate",
        plate_pos: tuple = (0.51002, 0.34525, 0.76827),
        bottle_name: str = "juice_bottle",
        bottle_pos: tuple = (-0.37634, -0.06814, 0.90886),
        fruit_name: str = "fruit_side",
        fruit_pos: tuple = (1.65, 0.3, 0.95),
        knife_holder_name: str = "butter_knives",
        knife_holder_pos: tuple = (1.7, 0.05, 0.95),
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

        assets[toaster_name] = bsToaster.get_toaster_asset(
            prim_path=toaster_prim_path,
            pos=_offset(toaster_pos),
        )
        assets[table_name] = bsKitchenTable.get_table_asset(
            prim_path=table_prim_path,
            pos=_offset(table_pos),
        )
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
