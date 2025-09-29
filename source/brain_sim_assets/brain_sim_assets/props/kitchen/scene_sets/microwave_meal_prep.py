from typing import Dict

from isaaclab.assets import AssetBaseCfg

from brain_sim_assets.props.kitchen.entities.kitchen_background import bsKitchenBackground
from brain_sim_assets.props.kitchen.entities.microwave import bsMicrowave
from brain_sim_assets.props.kitchen.entities.table import bsKitchenTable
from brain_sim_assets.props.kitchen.entities.bottle import bsKitchenBottle
from brain_sim_assets.props.kitchen.entities.plate import bsKitchenPlate
from brain_sim_assets.props.kitchen.entities.fruit import bsKitchenFruit
from brain_sim_assets.props.kitchen.entities.knife_holder import bsKnifeHolder
from brain_sim_assets.props.kitchen.entities.fridge import bsRefrigerator


class bsMicrowaveMealPrepEntitiesGenerator:

    @staticmethod
    def get_microwave_meal_prep_entities(
        base_pos: tuple = (0.0, -1.0, 0.0),
        include_background: bool = True,
        microwave_name: str = "microwave",
        microwave_prim_path: str = "{ENV_REGEX_NS}/Microwave",
        microwave_pos: tuple = (1.8, -1.5, 0.95),
        fridge_name: str = "prep_fridge",
        fridge_prim_path: str = "{ENV_REGEX_NS}/PrepFridge",
        fridge_pos: tuple = (3.7, -1.4, 1.05),
        background_name: str = "kitchen_background",
        background_prim_path: str = "{ENV_REGEX_NS}/KitchenBackground",
        background_pos: tuple = (0.0, 0.0, 0.0),
        background_rot: tuple = (1.0, 0.0, 0.0, 0.0),
        background_scale: tuple = (1.0, 1.0, 1.0),
        table_name: str = "staging_table",
        table_prim_path: str = "{ENV_REGEX_NS}/StagingTable",
        table_pos: tuple = (2.0, -1.5, 0.0),
        bottle_name: str = "drink_bottle",
        bottle_pos: tuple = (-0.07, -0.5, 0.91),
        fruit_name: str = "side_fruit",
        fruit_pos: tuple = (0.08, -0.43, 0.76),
        plate_name: str = "meal_plate",
        plate_pos: tuple = (-0.26, -0.43, 0.77),
        knife_holder_name: str = "counter_knives",
        knife_holder_pos: tuple = (0.35, -0.28, 0.90),
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

        assets[microwave_name] = bsMicrowave.get_microwave_object(
            prim_path=microwave_prim_path,
            pos=_offset(microwave_pos),
        )
        assets[fridge_name] = bsRefrigerator.get_fridge_asset(
            prim_path=fridge_prim_path,
            pos=_offset(fridge_pos),
            scale=(0.01, 0.01, 0.01),
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
            prim_path=f"{table_prim_path}/Knives",
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
