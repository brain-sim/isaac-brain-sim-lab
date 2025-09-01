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

    def _find_connected_regions(self, maze_grid: np.ndarray, wall_type: int) -> List[Tuple[int, int, int, int]]:
        """Find connected rectangular regions of the same wall type.
        Returns list of (row_start, col_start, height, width) tuples."""
        
        height, width = maze_grid.shape
        visited = np.zeros_like(maze_grid, dtype=bool)
        regions = []
        
        for i in range(height):
            for j in range(width):
                if maze_grid[i, j] == wall_type and not visited[i, j]:
                    # Find the maximum rectangle starting from this point
                    rect = self._find_max_rectangle(maze_grid, visited, i, j, wall_type)
                    if rect:
                        regions.append(rect)
        
        return regions
    
    def _find_max_rectangle(self, maze_grid: np.ndarray, visited: np.ndarray, 
                           start_row: int, start_col: int, wall_type: int) -> Optional[Tuple[int, int, int, int]]:
        """Find the maximum rectangle of the same wall type starting from given position."""
        
        height, width = maze_grid.shape
        
        # First, find the maximum width from this starting point
        max_width = 0
        for j in range(start_col, width):
            if maze_grid[start_row, j] == wall_type and not visited[start_row, j]:
                max_width += 1
            else:
                break
        
        if max_width == 0:
            return None
        
        # Now find the maximum height that maintains this width
        max_height = 1
        for i in range(start_row + 1, height):
            # Check if the entire row segment matches
            valid_row = True
            for j in range(start_col, start_col + max_width):
                if maze_grid[i, j] != wall_type or visited[i, j]:
                    valid_row = False
                    break
            
            if valid_row:
                max_height += 1
            else:
                break
        
        # Mark all cells in this rectangle as visited
        for i in range(start_row, start_row + max_height):
            for j in range(start_col, start_col + max_width):
                visited[i, j] = True
        
        return (start_row, start_col, max_height, max_width)
    
    def _create_merged_walls(self, maze: bsAntMaze):
        """Create walls by merging connected regions of the same type."""
        
        maze_grid = maze.get_maze()
        height, width = maze_grid.shape
        
        # Get unique wall types (excluding 0 which is empty space)
        wall_types = np.unique(maze_grid)
        wall_types = wall_types[wall_types > 0]
        
        self._walls = []
        
        for wall_type in wall_types:
            # Find all connected regions of this wall type
            regions = self._find_connected_regions(maze_grid, int(wall_type))
            
            for region_idx, (row_start, col_start, region_height, region_width) in enumerate(regions):
                # Calculate the center position of the merged wall
                center_x = (col_start + region_width / 2.0 - 0.5) * self._cell_size
                center_y = (row_start + region_height / 2.0 - 0.5) * self._cell_size
                center_z = self._wall_height / 2
                
                # Calculate the scale for the merged wall
                scale_x = region_width * self._cell_size
                scale_y = region_height * self._cell_size
                scale_z = self._wall_height
                
                # Create a single wall object for the entire region
                wall_cfg = bsBrickGenerator.get_brick_object(
                    prim_path=f"{{ENV_REGEX_NS}}/wall_type{wall_type}_{region_idx}",
                    pos=(center_x, center_y, center_z),
                    rot=(1, 0, 0, 0),
                    scale=(scale_x, scale_y, scale_z),
                    type=f"{wall_type}"
                )
                
                self._walls.append(wall_cfg)
        
        print(f"Optimized: Created {len(self._walls)} merged walls from {np.sum(maze_grid > 0)} individual cells")
        
        # Print statistics for debugging
        for wall_type in wall_types:
            num_cells = np.sum(maze_grid == wall_type)
            num_merged = len([w for w in self._walls if f"wall_type{wall_type}" in w.prim_path])
            print(f"  Wall type {wall_type}: {num_cells} cells merged into {num_merged} objects")

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
