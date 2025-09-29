from typing import Dict

from isaaclab.assets import AssetBaseCfg

from brain_sim_assets.props.kitchen.entities.kitchen_background import bsKitchenBackground
from brain_sim_assets.props.kitchen.entities.shelf import bsKitchenShelf
from brain_sim_assets.props.kitchen.entities.table import bsKitchenTable
from brain_sim_assets.props.kitchen.entities.bottle import bsKitchenBottle
from brain_sim_assets.props.kitchen.entities.plate import bsKitchenPlate
from brain_sim_assets.props.kitchen.entities.fruit import bsKitchenFruit
from brain_sim_assets.props.kitchen.entities.knife_holder import bsKnifeHolder


class bsShelfSortingEntitiesGenerator:

    @staticmethod
    def get_shelf_sorting_entities(
        base_pos: tuple = (0.0, -1.0, 0.0),
        include_background: bool = True,
        shelf_name: str = "storage_shelf",
        shelf_prim_path: str = "{ENV_REGEX_NS}/Shelf",
        shelf_pos: tuple = (2.2, -1.8, 0.9),
        background_name: str = "kitchen_background",
        background_prim_path: str = "{ENV_REGEX_NS}/KitchenBackground",
        background_pos: tuple = (0.0, 0.0, 0.0),
        background_rot: tuple = (1.0, 0.0, 0.0, 0.0),
        background_scale: tuple = (1.0, 1.0, 1.0),
        table_name: str = "sorting_table",
        table_prim_path: str = "{ENV_REGEX_NS}/SortingTable",
        table_pos: tuple = (2.0, -2.0, 0.0),
        bottle_name: str = "pantry_bottle",
        bottle_pos: tuple = (0.3, -0.1, 0.90),
        fruit_name: str = "pantry_fruit",
        fruit_pos: tuple = (0.2, -0.35, 0.76),
        plate_name: str = "stack_plate",
        plate_pos: tuple = (-0.025, -0.3, 0.77),
        knife_holder_name: str = "shelf_knives",
        knife_holder_pos: tuple = (-0.55, -0.4, 0.1),
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
