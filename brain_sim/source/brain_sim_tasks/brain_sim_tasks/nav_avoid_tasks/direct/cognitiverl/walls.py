from brain_sim_assets.props.maze import bsMazeGenerator
from brain_sim_assets import BRAIN_SIM_ASSETS_PROPS_CONFIG_DIR
import torch
import random


class WallConfiguration:
    
    def __init__(
            self, 
            room_size: float = 40.0, 
            wall_thickness: float = 2.0, 
            wall_height: float = 3.0, 
            maze_file: str = "example_maze_sq.txt"
        ):

        self.room_size = room_size
        self.wall_thickness = wall_thickness
        self.wall_height = wall_height
        self.maze_file = maze_file
        self._maze_generator = None
        
    def _get_maze_generator(self):
        if self._maze_generator is None:
            offset = self.get_position_offset()
            self._maze_generator = bsMazeGenerator.create_example_maze(
                f"{BRAIN_SIM_ASSETS_PROPS_CONFIG_DIR}/{self.maze_file}", 
                position_offset=offset
            )
        return self._maze_generator
        
    def get_wall_position(self) -> float:
        return (self.room_size - self.wall_thickness) / 2
    
    def get_position_offset(self) -> tuple:
        wall_position = self.get_wall_position()
        return (-wall_position, -wall_position, 0.0)
    
    def create_maze_configuration(self, maze_file: str | None = None):
        if maze_file is None:
            return self._get_maze_generator().get_wall_collection()
        else:
            offset = self.get_position_offset()
            maze = bsMazeGenerator.create_example_maze(
                f"{BRAIN_SIM_ASSETS_PROPS_CONFIG_DIR}/{maze_file}", 
                position_offset=offset
            )
            return maze.get_wall_collection()
    
    def apply_to_scene_cfg(self, scene, maze_file: str | None = None):
        if maze_file is None:
            maze_file = self.maze_file
        walls_config = self.create_maze_configuration(maze_file)
        setattr(scene, "wall_collection", walls_config)
    
    def _get_valid_positions(self):
        maze_generator = self._get_maze_generator()
        maze_grid = maze_generator._maze.get_maze()
        offset = self.get_position_offset()
        cell_size = maze_generator._cell_size
        
        valid_cells = []
        for y in range(maze_grid.shape[0]):
            for x in range(maze_grid.shape[1]):
                if maze_grid[y, x] == 0:
                    cell_x = x * cell_size + offset[0]
                    cell_y = y * cell_size + offset[1]
                    valid_cells.append((cell_x, cell_y, cell_size))
        
        return valid_cells
    
    def get_random_valid_position(self, device=None) -> torch.Tensor:
        valid_cells = self._get_valid_positions()
        if not valid_cells:
            raise ValueError("No valid positions found in maze")
        
        cell_x, cell_y, cell_size = random.choice(valid_cells)
        
        margin = 0.5
        min_pos = margin
        max_pos = cell_size - margin
        
        x = cell_x + random.uniform(min_pos, max_pos)
        y = cell_y + random.uniform(min_pos, max_pos)
        
        position = torch.tensor([x, y, 0.0], dtype=torch.float32)
        if device is not None:
            position = position.to(device)
        return position
    
    def get_random_valid_positions(self, num_positions: int, device=None) -> torch.Tensor:
        valid_cells = self._get_valid_positions()
        if not valid_cells:
            raise ValueError("No valid positions found in maze")
        
        if num_positions > len(valid_cells):
            selected_cells = random.choices(valid_cells, k=num_positions)
        else:
            selected_cells = random.sample(valid_cells, num_positions)
        
        positions = []
        margin = 0.5
        
        for cell_x, cell_y, cell_size in selected_cells:
            min_pos = margin
            max_pos = cell_size - margin
            
            x = cell_x + random.uniform(min_pos, max_pos)
            y = cell_y + random.uniform(min_pos, max_pos)
            positions.append(torch.tensor([x, y, 0.0], dtype=torch.float32))
        
        result = torch.stack(positions)
        if device is not None:
            result = result.to(device)
        return result
