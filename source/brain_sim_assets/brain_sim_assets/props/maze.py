import os
from typing import List, Optional, Dict, Tuple, Set
from brain_sim_ant_maze import bsAntMaze, bsAntMazeConfig
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

from brain_sim_assets import BRAIN_SIM_ASSETS_PROPS_CONFIG_DIR
from .brick import bsBrickGenerator


class bsMazeGenerator:
    
    def __init__(
            self, 
            maze_config_path: Optional[str] = None, 
            maze_txt_path: Optional[str] = None, 
            position_offset: tuple = (0.0, 0.0, 0.0),
            create_wall_on_init: bool = True
        ):
        self._walls = []
        self._maze_config = None
        self._maze_txt_path = None
        self._maze = None
        self._start_goal = None
        self._position_offset = position_offset
        
        self._wall_height = 2.0
        self._cell_size = 1.0
        
        if maze_config_path:
            self.set_maze_config(maze_config_path)
        if maze_txt_path:
            self.set_maze_txt_path(maze_txt_path)
        if maze_config_path and maze_txt_path:
            self.setup()
        if create_wall_on_init and self._maze:
            self.create_maze()

    @classmethod
    def create_example_maze(
            cls, 
            maze_txt_path_in: Optional[str] = None, 
            position_offset: tuple = (0.0, 0.0, 0.0),
            create_wall_on_init: bool = True
        ):

        if maze_txt_path_in is None:        
            maze_txt_path = f"{BRAIN_SIM_ASSETS_PROPS_CONFIG_DIR}/example_maze.txt"
        else:
            maze_txt_path = maze_txt_path_in
        
        maze_config_path = f"{BRAIN_SIM_ASSETS_PROPS_CONFIG_DIR}/example_config.json"
        return cls(maze_config_path, maze_txt_path, position_offset, create_wall_on_init)

    def setup(self):
        if self._maze_config:
            self._maze = bsAntMaze(self._maze_config)
            if self._maze_txt_path:
                self._maze.build_from_txt(self._maze_txt_path, None)
                start_pos = self._maze.get_start_position()
                goal_pos = self._maze.get_goal_position()

                world_start_x = start_pos[1] * self._cell_size + self._position_offset[0]
                world_start_y = start_pos[0] * self._cell_size + self._position_offset[1]
                world_goal_x = goal_pos[1] * self._cell_size + self._position_offset[0]
                world_goal_y = goal_pos[0] * self._cell_size + self._position_offset[1]

                self._start_goal = [
                    [world_start_x, world_start_y], 
                    [world_goal_x, world_goal_y]
                ]
        
    def set_maze_config(self, maze_config_path: str):
        self._maze_config = bsAntMazeConfig.from_json(maze_config_path)
        self._update_dimensions()

    def set_maze_txt_path(self, maze_txt_path: str):
        self._maze_txt_path = maze_txt_path
        
    def _update_dimensions(self):
        if self._maze_config:
            self._wall_height = self._maze_config.dimensions.wall_height
            self._cell_size = self._maze_config.dimensions.cell_size

    def create_maze(self, maze: bsAntMaze = None):
        if maze is None:
            maze = self._maze
        self._create_walls(maze)

    def _create_walls(self, maze: bsAntMaze):
        """Original method - kept for backwards compatibility but deprecated."""
        maze_grid = maze.get_maze()
        height, width = maze_grid.shape

        for i in range(height):
            for j in range(width):
                if maze_grid[i, j] > 0 :
                    world_x = j * self._cell_size + self._position_offset[0]
                    world_y = i * self._cell_size + self._position_offset[1]
                    world_z = self._wall_height / 2 + self._position_offset[2]
                    
                    wall_cfg = bsBrickGenerator.get_brick_object(
                        prim_path=f"{{ENV_REGEX_NS}}/wall_{len(self._walls)}",
                        pos=(world_x, world_y, world_z),
                        rot=(1, 0, 0, 0),
                        scale=(
                            self._cell_size, 
                            self._cell_size, 
                            self._wall_height
                        ),
                        type=f"{maze_grid[i, j]}"
                    )
                    
                    self._walls.append(wall_cfg)

    def get_walls(self) -> List[RigidObjectCfg]:
        return self._walls
        
    def get_start_goal(self) -> Optional[List[List[float]]]:
        return self._start_goal if self._start_goal else None

    def get_wall_collection(self) -> RigidObjectCollectionCfg:
        wall_dict = {}
        for i, wall_cfg in enumerate(self._walls):
            wall_dict[f"wall_{i}"] = wall_cfg
        return RigidObjectCollectionCfg(
            rigid_objects=wall_dict
        )
