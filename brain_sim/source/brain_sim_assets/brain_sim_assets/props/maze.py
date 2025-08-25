import os
from typing import List, Optional, Dict
from brain_sim_ant_maze import bsAntMaze, bsAntMazeConfig
from isaaclab.assets import RigidObjectCfg

from brain_sim_assets import BRAIN_SIM_ASSETS_CONFIG_DIR
from .brick import BrainSimBrick


class BrainSimMaze:
    
    def __init__(self):
        self._walls = []
        self._maze_config = None
        self._maze_txt_path = None
        self._maze = None
        self._start_goal = None
        
        self._wall_height = 2.0
        self._cell_size = 1.0

    def setup(self):
        if self._maze_config:
            self._maze = bsAntMaze(self._maze_config)
            if self._maze_txt_path:
                self._maze.build_from_txt(self._maze_txt_path, None)
        
    def set_maze_config(self, maze_config_path: str):
        self._maze_config = bsAntMazeConfig.from_json(maze_config_path)
        self._update_dimensions()

    def set_maze_txt_path(self, maze_txt_path: str):
        self._maze_txt_path = maze_txt_path
        
    def _update_dimensions(self):
        if self._maze_config:
            self._wall_height = self._maze_config.dimensions.wall_height
            self._cell_size = self._maze_config.dimensions.cell_size
        
    def create_maze(self, maze: bsAntMaze):
        start_pos = maze.get_start_position()
        goal_pos = maze.get_goal_position()
        
        world_start_x = start_pos[1] * self._cell_size
        world_start_y = start_pos[0] * self._cell_size
        world_goal_x = goal_pos[1] * self._cell_size
        world_goal_y = goal_pos[0] * self._cell_size

        self._start_goal = [
            [world_start_x, world_start_y], 
            [world_goal_x, world_goal_y]
        ]
        
        self._create_walls(maze)

    def _create_walls(self, maze: bsAntMaze):
        maze_grid = maze.get_maze()
        height, width = maze_grid.shape

        for i in range(height):
            for j in range(width):
                if maze_grid[i, j] == 1:
                    world_x = j * self._cell_size
                    world_y = i * self._cell_size
                    world_z = self._wall_height / 2
                    
                    wall_cfg = BrainSimBrick.get_brick_object(
                        prim_path=f"{{ENV_REGEX_NS}}/wall_{len(self._walls)}",
                        pos=(world_x, world_y, world_z),
                        rot=(1, 0, 0, 0),
                        scale=(
                            self._cell_size, 
                            self._cell_size, 
                            self._wall_height
                        )
                    )
                    
                    self._walls.append(wall_cfg)

    def get_walls(self) -> List[RigidObjectCfg]:
        return self._walls
        
    def get_start_goal(self) -> Optional[List[List[float]]]:
        return self._start_goal if self._start_goal else None

    def get_wall_configs_dict(self) -> Dict[str, RigidObjectCfg]:
        wall_dict = {}
        for i, wall_cfg in enumerate(self._walls):
            wall_dict[f"wall_{i}"] = wall_cfg
        return wall_dict
