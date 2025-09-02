# Brain Sim Ant Maze Extension

A Python package for creating and managing ant maze environments within Isaac Lab.

## Installation

```bash
pip install -e .
```

## Basic Usage

```python
from brain_sim_ant_maze import bsAntMaze, bsAntMazeConfig

# Create maze with default configuration
maze = bsAntMaze()

# Load from text file
maze.build_from_txt("maze.txt")

# Access maze grid
print(maze.get_maze())
```

## Maze Creation

### From 2D Array
```python
custom_maze = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]
maze = bsAntMaze()
maze.create_maze(custom_maze)
maze.set_start_position((1, 1))
maze.set_goal_position((3, 3))
```

### From Text File
Create a text file with maze layout:
```
11111
1S001
11101
100G1
11111
```

Load it:
```python
maze = bsAntMaze()
maze.build_from_txt("maze.txt")
```

## Configuration

```python
config = bsAntMazeConfig(
    wall_height=0.8,
    wall_width=0.15,
    cell_size=1.2,
    ant_scale=0.7
)

maze = bsAntMaze(config)
```

## Examples

Check the `examples/` directory for usage examples:
- `basic_usage.py`: Basic maze creation and manipulation
- `maze_visualization.py`: Maze visualization with matplotlib
