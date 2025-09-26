from typing import Dict

from isaaclab.assets import AssetBaseCfg

from brain_sim_assets.props.kitchen.entities.fridge import bsRefrigerator
from brain_sim_assets.props.kitchen.entities.kitchen_background import bsKitchenBackground
from brain_sim_assets.props.kitchen.entities.table import bsKitchenTable
from brain_sim_assets.props.kitchen.entities.bottle import bsKitchenBottle
from brain_sim_assets.props.kitchen.entities.plate import bsKitchenPlate
from brain_sim_assets.props.kitchen.entities.fruit import bsKitchenFruit
from brain_sim_assets.props.kitchen.entities.knife_holder import bsKnifeHolder


class bsFridgeStockingEntitiesGenerator:

    @staticmethod
    def get_fridge_stocking_entities(
        base_pos: tuple = (0.0, -2.0, 0.0),
        include_background: bool = True,
        fridge_name: str = "fridge",
        fridge_prim_path: str = "{ENV_REGEX_NS}/Fridge",
        fridge_pos: tuple = (4.70755, -0.42633, 0.79453),
        fridge_rot: tuple = (1.0, 0.0, 0.0, 0.0),
        background_name: str = "kitchen_background",
        background_prim_path: str = "{ENV_REGEX_NS}/KitchenBackground",
        background_pos: tuple = (0.0, 0.0, 0.0),    
        background_rot: tuple = (1.0, 0.0, 0.0, 0.0),
        background_scale: tuple = (1.0, 1.0, 1.0),
        table_name: str = "stock_table",
        table_prim_path: str = "{ENV_REGEX_NS}/StockTable",
        table_pos: tuple = (4.0, -1.5, 0.0),
        bottle_name: str = "juice_bottle",
        bottle_pos: tuple = (0.1, 2.25, 0.9),
        fruit_name: str = "fruit_bundle",
        fruit_pos: tuple = (0.3, 2.5, 0.747),
        plate_name: str = "storage_plate",
        plate_pos: tuple = (0.1, 2.3, 0.775),
        knife_holder_name: str = "prep_knives",
        knife_holder_pos: tuple = (0.1, 1.9 , 0.92),
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

        assets[fridge_name] = bsRefrigerator.get_fridge_asset(
            prim_path=fridge_prim_path,
            pos=_offset(fridge_pos),
            rot=fridge_rot,
            scale=(0.0075, 0.0075, 0.0075),
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
            scale=(1.0, 1.0, 1.0),
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
