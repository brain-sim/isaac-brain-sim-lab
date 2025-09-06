from brain_sim_assets.props.maze import bsMazeGenerator
from brain_sim_assets import BRAIN_SIM_ASSETS_PROPS_CONFIG_DIR
import torch
import random
import numpy as np


class bsMazeRuntime:

    def __init__(
        self,
        room_size: float = 40.0,
        cell_size: float = 2.0,
        maze_file: str = "example_maze_sq_3_wall.txt",
        maze_config: str = "example_config.json",
        device: str | torch.device = "cpu",
    ):

        self.room_size = room_size
        self.cell_size = cell_size
        self.maze_file = maze_file
        self.maze_config = maze_config
        self.device = torch.device(device) if isinstance(device, str) else device
        self._maze_generator = None
        self._wall_segments = None

    def update_device(self, device: str | torch.device):
        """Update the device and move existing tensors to the new device."""
        new_device = torch.device(device) if isinstance(device, str) else device

        if new_device != self.device:
            self.device = new_device

            if self._wall_segments is not None:
                self._wall_segments = self._wall_segments.to(self.device)

    def _get_maze_generator(self):
        if self._maze_generator is None:
            offset = self.get_position_offset()
            self._maze_generator = bsMazeGenerator.create_example_maze(
                f"{BRAIN_SIM_ASSETS_PROPS_CONFIG_DIR}/{self.maze_file}",
                f"{BRAIN_SIM_ASSETS_PROPS_CONFIG_DIR}/{self.maze_config}",
                position_offset=offset,
            )
        return self._maze_generator

    def get_wall_position(self) -> float:
        return (self.room_size - self.cell_size) / 2

    def get_position_offset(self) -> tuple:
        wall_position = self.get_wall_position()
        return (-wall_position, -wall_position, 0.0)

    def create_maze_configuration(
        self, maze_file: str | None = None, maze_config: str | None = None
    ):
        if maze_file is None:
            maze_file = self.maze_file
        if maze_config is None:
            maze_config = self.maze_config
        offset = self.get_position_offset()
        maze = bsMazeGenerator.create_example_maze(
            f"{BRAIN_SIM_ASSETS_PROPS_CONFIG_DIR}/{maze_file}",
            f"{BRAIN_SIM_ASSETS_PROPS_CONFIG_DIR}/{maze_config}",
            position_offset=offset,
        )
        if self._wall_segments is None:
            self._precompute_wall_segments()
        return maze.get_wall_collection()

    def apply_to_scene_cfg(
        self, scene, maze_file: str | None = None, maze_config: str | None = None
    ):
        if maze_file is None:
            maze_file = self.maze_file
        if maze_config is None:
            maze_config = self.maze_config
        walls_config = self.create_maze_configuration(maze_file, maze_config)
        setattr(scene, "wall_collection", walls_config)

    def _get_valid_positions(self, valid_indicator: int | str = 0):
        maze_generator = self._get_maze_generator()
        maze_grid = maze_generator._maze.get_maze()
        offset = self.get_position_offset()
        self.cell_size = maze_generator._cell_size

        valid_cells = []
        for y in range(len(maze_grid)):
            for x in range(len(maze_grid[0])):
                if maze_grid[y][x] == valid_indicator:
                    cell_x = x * self.cell_size + offset[0]
                    cell_y = y * self.cell_size + offset[1]
                    valid_cells.append((cell_x, cell_y, self.cell_size))
        return valid_cells

    def get_random_valid_position(
        self, valid_indicator: int | str = 0, device=None
    ) -> torch.Tensor:
        valid_cells = self._get_valid_positions(valid_indicator)
        if not valid_cells:
            raise ValueError("No valid positions found in maze")

        cell_x, cell_y, cell_size = random.choice(valid_cells)

        margin = 0.5
        min_pos = 0.0 - cell_size / 2 + margin
        max_pos = 0.0 + cell_size / 2 - margin

        x = cell_x + random.uniform(min_pos, max_pos)
        y = cell_y + random.uniform(min_pos, max_pos)

        position = torch.tensor([x, y, 0.0], dtype=torch.float32)
        if device is not None:
            position = position.to(device)
        return position

    def get_random_valid_positions(
        self, num_positions: int, valid_indicator: int | str = 0, device=None
    ) -> torch.Tensor:
        valid_cells = self._get_valid_positions(valid_indicator)
        if not valid_cells:
            raise ValueError("No valid positions found in maze")

        if num_positions > len(valid_cells):
            selected_cells = random.choices(valid_cells, k=num_positions)
        else:
            selected_cells = random.sample(valid_cells, num_positions)

        positions = []
        margin = 0.5

        for cell_x, cell_y, cell_size in selected_cells:
            min_pos = 0.0 - cell_size / 2 + margin
            max_pos = 0.0 + cell_size / 2 - margin

            x = cell_x + random.uniform(min_pos, max_pos)
            y = cell_y + random.uniform(min_pos, max_pos)
            positions.append(torch.tensor([x, y, 0.0], dtype=torch.float32))

        result = torch.stack(positions)
        if device is not None:
            result = result.to(device)
        return result

    def is_wall(self, indicator):
        return type(indicator) is int and indicator > 0 and indicator < 999

    def _precompute_wall_segments(self):
        if self._wall_segments is not None:
            return

        maze_generator = self._get_maze_generator()
        maze_grid = maze_generator._maze.get_maze()
        self.cell_size = maze_generator._cell_size
        H, W = len(maze_grid), len(maze_grid[0])

        wall_segments = []

        # Extract wall edges that border open space
        for y in range(H):
            for x in range(W):
                if self.is_wall(maze_grid[y][x]):  # wall cell
                    # Convert to world coordinates
                    world_x = x * self.cell_size + (-self.room_size) / 2
                    world_y = y * self.cell_size + (-self.room_size) / 2

                    # Top edge (if borders open space above)
                    if y == H - 1 or not self.is_wall(maze_grid[y + 1][x]):
                        wall_segments.append(
                            [
                                world_x,
                                world_y + self.cell_size,  # start point
                                world_x + self.cell_size,
                                world_y + self.cell_size,  # end point
                            ]
                        )

                    # Bottom edge (if borders open space below)
                    if y == 0 or not self.is_wall(maze_grid[y - 1][x]):
                        wall_segments.append(
                            [
                                world_x,
                                world_y,  # start point
                                world_x + self.cell_size,
                                world_y,  # end point
                            ]
                        )

                    # Left edge (if borders open space to the left)
                    if x == 0 or not self.is_wall(maze_grid[y][x - 1]):
                        wall_segments.append(
                            [
                                world_x,
                                world_y,  # start point
                                world_x,
                                world_y + self.cell_size,  # end point
                            ]
                        )

                    # Right edge (if borders open space to the right)
                    if x == W - 1 or not self.is_wall(maze_grid[y][x + 1]):
                        wall_segments.append(
                            [
                                world_x + self.cell_size,
                                world_y,  # start point
                                world_x + self.cell_size,
                                world_y + self.cell_size,  # end point
                            ]
                        )

        if wall_segments:
            self._wall_segments = torch.tensor(
                wall_segments, dtype=torch.float32, device=self.device
            )
        else:
            self._wall_segments = torch.empty(
                (0, 4), dtype=torch.float32, device=self.device
            )

    def get_wall_distances(self, robot_positions: torch.Tensor) -> torch.Tensor:
        assert self._wall_segments is not None, "Wall segments should be computed"
        if self._wall_segments.numel() == 0:
            return torch.full(
                (len(robot_positions),),
                float("inf"),
                device=robot_positions.device,
                dtype=torch.float32,
            )
        robot_xy = robot_positions[:, :2]

        # Shape: (M, 4) where M is number of wall segments, format: [x1, y1, x2, y2]
        segments = self._wall_segments

        N = robot_xy.shape[0]
        M = segments.shape[0]

        robots = robot_xy.unsqueeze(1)  # (N, 1, 2)
        seg_start = segments[:, :2].unsqueeze(0)  # (1, M, 2) - [x1, y1]
        seg_end = segments[:, 2:].unsqueeze(0)  # (1, M, 2) - [x2, y2]

        seg_vec = seg_end - seg_start
        robot_vec = robots - seg_start

        seg_length_sq = torch.sum(seg_vec**2, dim=2, keepdim=True)  # (1, M, 1)
        seg_length_sq = torch.clamp(seg_length_sq, min=1e-8)

        # Project robot onto segment line: (N, M, 1)
        # t = dot(robot_vec, seg_vec) / seg_length_sq
        t = torch.sum(robot_vec * seg_vec, dim=2, keepdim=True) / seg_length_sq
        t = torch.clamp(t, 0.0, 1.0)

        # Closest point on segment: (N, M, 2)
        closest_point = seg_start + t * seg_vec

        # Distance from robot to closest point: (N, M)
        distances_to_segments = torch.norm(
            robots.squeeze(1).unsqueeze(1) - closest_point, dim=2
        )

        # Find minimum distance to any segment for each robot: (N,)
        min_distances, _ = torch.min(distances_to_segments, dim=1)

        return min_distances
