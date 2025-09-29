from typing import Dict

from isaaclab.assets import AssetBaseCfg

from brain_sim_assets.props.kitchen.entities.kitchen_background import bsKitchenBackground
from brain_sim_assets.props.kitchen.entities.shelf import bsKitchenShelf
from brain_sim_assets.props.kitchen.entities.table import bsKitchenTable
from brain_sim_assets.props.kitchen.entities.bottle import bsKitchenBottle
from brain_sim_assets.props.kitchen.entities.plate import bsKitchenPlate
from brain_sim_assets.props.kitchen.entities.fruit import bsKitchenFruit
from brain_sim_assets.props.kitchen.entities.knife_holder import bsKnifeHolder


class bsPantryLoadingEntitiesGenerator:

    @staticmethod
    def get_pantry_loading_entities(
        base_pos: tuple = (0.0, -1.0, 0.0),
        include_background: bool = True,
        shelf_name: str = "pantry_shelf",
        shelf_prim_path: str = "{ENV_REGEX_NS}/PantryShelf",
        shelf_pos: tuple = (1.55, -1.7, 0.88),
        background_name: str = "kitchen_background",
        background_prim_path: str = "{ENV_REGEX_NS}/KitchenBackground",
        background_pos: tuple = (0.0, 0.0, 0.0),
        background_rot: tuple = (1.0, 0.0, 0.0, 0.0),
        background_scale: tuple = (1.0, 1.0, 1.0),
        table_name: str = "pantry_table",
        table_prim_path: str = "{ENV_REGEX_NS}/PantryTable",
        table_pos: tuple = (1.5, -1.8, 0.0),
        bottle_name: str = "pantry_bottle",
        bottle_pos: tuple = (-0.31, -0.30, 0.90),
        fruit_name: str = "pantry_fruit",
        fruit_pos: tuple = (-0.16, -0.41, 0.75),
        plate_name: str = "pantry_plate",
        plate_pos: tuple = (-0.015, -0.29, 0.76),
        knife_holder_name: str = "pantry_knives",
        knife_holder_pos: tuple = (0.18, -0.23, 0.05),
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

        assets[shelf_name] = bsKitchenShelf.get_shelf_asset(
            prim_path=shelf_prim_path,
            pos=_offset(shelf_pos),
        )
        assets[table_name] = bsKitchenTable.get_table_asset(
            prim_path=table_prim_path,
            pos=_offset(table_pos),
        )
        assets[plate_name] = bsKitchenPlate.get_plate_asset(
            prim_path=f"{table_prim_path}/Plate",
            pos=plate_pos,
        )
        assets[bottle_name] = bsKitchenBottle.get_bottle_asset(
            prim_path=f"{table_prim_path}/Bottle",
            pos=bottle_pos,
        )
        assets[fruit_name] = bsKitchenFruit.get_fruit_asset(
            prim_path=f"{table_prim_path}/Fruit",
            pos=fruit_pos,
        )
        assets[knife_holder_name] = bsKnifeHolder.get_knife_holder_asset(
            prim_path=f"{shelf_prim_path}/Knives",
            pos=knife_holder_pos,
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
