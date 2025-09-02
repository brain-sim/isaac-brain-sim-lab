from brain_sim_assets.props.maze import bsMazeGenerator
from brain_sim_assets import BRAIN_SIM_ASSETS_PROPS_CONFIG_DIR
import torch
import random
import numpy as np


class WallConfiguration:
    
    def __init__(
            self, 
            room_size: float = 40.0, 
            wall_thickness: float = 2.0, 
            wall_height: float = 3.0, 
            maze_file: str = "example_maze_sq.txt",
            device: str | torch.device = "cpu"
        ):

        self.room_size = room_size
        self.wall_thickness = wall_thickness
        self.wall_height = wall_height
        self.maze_file = maze_file
        self.device = torch.device(device) if isinstance(device, str) else device
        self._maze_generator = None
        self._distance_field = None
        
    def update_device(self, device: str | torch.device):
        """Update the device and move existing tensors to the new device."""
        new_device = torch.device(device) if isinstance(device, str) else device
        
        if new_device != self.device:
            self.device = new_device
            
            if self._distance_field is not None:
                self._distance_field = self._distance_field.to(self.device)
        
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
            if self._distance_field is None:
                self._precompute_distance_field()
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
    
    def _precompute_distance_field(self, diagonal_margin = 0.125):
        """
        Exact in axis-aligned cases (robot north/south/east/west of walls)
        Not exact in diagonal cases (eg. northeast, because to corner is sqrt(2). 
            Added diagonal margin to handle this)
        """
        if self._distance_field is not None:
            return
            
        maze_generator = self._get_maze_generator()
        maze_grid = maze_generator._maze.get_maze()
        cell_size = maze_generator._cell_size
        
        from scipy.ndimage import distance_transform_edt
        
        open_mask = (maze_grid == 0).astype(np.float32)
        distance_grid = distance_transform_edt(open_mask)
        
        self._distance_field = torch.tensor(
            (distance_grid - 0.5 - diagonal_margin) * cell_size,
            dtype=torch.float32,
            device=self.device
        )
        
        self._distance_field = torch.clamp(self._distance_field, min=0.0)
    
    def get_wall_distances(self, robot_positions: torch.Tensor) -> torch.Tensor:
        
        assert self._distance_field is not None, "Distance field should be computed"
        
        offset = self.get_position_offset()
        cell_size = self._get_maze_generator()._cell_size
        
        # Convert to grid coords and SHIFT by -0.5 to account for center-based distance field
        grid_x_cont = (robot_positions[:, 0] - offset[0]) / cell_size - 0.5
        grid_y_cont = (robot_positions[:, 1] - offset[1]) / cell_size - 0.5
        
        H, W = self._distance_field.shape
        # Clamp the continuous coordinates to valid sampling range
        grid_x_cont = torch.clamp(grid_x_cont, 0.0, W - 1.0)
        grid_y_cont = torch.clamp(grid_y_cont, 0.0, H - 1.0)
        
        grid_x_floor = torch.floor(grid_x_cont).long()
        grid_y_floor = torch.floor(grid_y_cont).long()
        
        # Fractional parts for bilinear interpolation
        fx = grid_x_cont - grid_x_floor.float()
        fy = grid_y_cont - grid_y_floor.float()
        
        grid_x_ceil = torch.clamp(grid_x_floor + 1, 0, W - 1)
        grid_y_ceil = torch.clamp(grid_y_floor + 1, 0, H - 1)
        
        # Bilinear interpolation
        d00 = self._distance_field[grid_y_floor, grid_x_floor]  # top-left
        d01 = self._distance_field[grid_y_floor, grid_x_ceil]   # top-right
        d10 = self._distance_field[grid_y_ceil, grid_x_floor]   # bottom-left
        d11 = self._distance_field[grid_y_ceil, grid_x_ceil]    # bottom-right
        d0 = d00 * (1 - fx) + d01 * fx  # top edge
        d1 = d10 * (1 - fx) + d11 * fx  # bottom edge
        distances = d0 * (1 - fy) + d1 * fy  # final interpolation
        
        return distances