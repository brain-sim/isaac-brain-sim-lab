import os
import json
from typing import List, Tuple, Optional, Dict, Any, Union

from .ant_maze_config import bsAntMazeConfig


class bsAntMaze:
    """
    Ant Maze environment builder.
    
    The environment represents a maze where:
    - 0: represents free space (where the ant can move)
    - 1: represents walls/obstacles
    - Any other integer or character: custom maze elements
    - S: represents start position
    - G: represents goal position  
    - B: represents button positions
    
    Example maze structure:
    [
        [1, 1, 1, 1],
        [1, 0, 'A', 1],
        [1, 0, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 1, 1]
    ]
    
    The maze can now hold both integers and characters for maximum flexibility.
    """
    
    def __init__(self, maze_config: Optional[bsAntMazeConfig] = None):
        """
        Initialize the Ant Maze environment builder.
        
        Args:
            maze_config: Configuration object for the maze. If None, default config is used.
        """
        self._maze_txt = None
        self._config = maze_config if maze_config else bsAntMazeConfig()
        self._maze = None
        self._start_pos = None
        self._goal_pos = None
        self._buttons = []  # List of button positions
        self._button_door_mapping = {}  # Dict mapping button positions to door positions
        self._door_states = {}  # Dict tracking door states (True = closed/wall, False = open)
        self._original_maze = None  # Store original maze without door modifications
        
        # Default maze if none is provided
        self._default_maze = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1]
        ]
        
        # Initialize with default maze
        self._maze = [row[:] for row in self._default_maze]  # Deep copy
        self._original_maze = [row[:] for row in self._default_maze]  # Deep copy
        self._start_pos = (1, 1)  # Default start position
        self._goal_pos = (3, 1)   # Default goal position

    @classmethod
    def from_config_file(cls, config_file_path: str) -> 'bsAntMaze':
        config = bsAntMazeConfig.from_json(config_file_path)
        return cls(config)
    
    def build_from_txt(self, file_path: str, buttons_config_path: Optional[str] = None) -> None:

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Maze file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            self._maze_txt = f.read()
            
        lines = self._maze_txt.strip().split('\n')
        height = len(lines)
        if height == 0:
            raise ValueError("Empty maze file")
            
        width = len(lines[0])
        
        maze = [[0 for _ in range(width)] for _ in range(height)]
        start_pos = None
        goal_pos = None
        buttons = []
        
        for i, line in enumerate(lines):
            if len(line) != width:
                raise ValueError(f"Inconsistent maze width at line {i+1}")
                
            for j, char in enumerate(line):
                if char == '1':
                    maze[i][j] = 1
                elif char == '2':
                    maze[i][j] = 2
                elif char == '3':
                    maze[i][j] = 3
                elif char == '4':
                    maze[i][j] = 4
                elif char == '5':
                    maze[i][j] = 5
                elif char == '6':
                    maze[i][j] = 6
                elif char == '7':
                    maze[i][j] = 7
                elif char == '8':
                    maze[i][j] = 8
                elif char == '9':
                    maze[i][j] = 9
                elif char == '0':
                    maze[i][j] = 0
                elif char == 'S':
                    maze[i][j] = 0
                    start_pos = (i, j)
                elif char == 'G':
                    maze[i][j] = 0
                    goal_pos = (i, j)
                elif char == 'B':
                    maze[i][j] = 0  # Buttons are on free space
                    buttons.append((i, j))
                elif char == 'N':
                    maze[i][j] = 999
                else:
                    maze[i][j] = char
        
        self._maze = maze
        self._original_maze = [row[:] for row in maze]  # Deep copy for lists
        self._buttons = buttons
        
        if start_pos:
            self._start_pos = start_pos
        if goal_pos:
            self._goal_pos = goal_pos
            
        if buttons_config_path:
            self.load_button_door_mapping(buttons_config_path)

    def load_button_door_mapping(self, config_path: str) -> None:

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Button config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        if 'button_mapping' not in config:
            raise ValueError("Invalid button config: missing 'button_mapping' key")
            
        self._button_door_mapping = {}
        self._door_states = {}
        
        for mapping in config['button_mapping']:
            if 'button' not in mapping or 'door' not in mapping:
                raise ValueError("Invalid button mapping: missing 'button' or 'door' key")
                
            button_pos = tuple(mapping['button'])
            door_pos = tuple(mapping['door'])
            
            if button_pos not in self._buttons:
                raise ValueError(f"Button position {button_pos} not found in maze")
                
            row, col = door_pos
            if not (0 <= row < len(self._maze) and 0 <= col < len(self._maze[0])):
                raise ValueError(f"Door position {door_pos} is outside maze boundaries")
                
            self._button_door_mapping[button_pos] = door_pos
            self._door_states[door_pos] = True
        
    def _update_maze_with_doors(self) -> None:
        self._maze = [row[:] for row in self._original_maze]  # Deep copy for lists
        
        for door_pos, is_closed in self._door_states.items():
            row, col = door_pos
            if is_closed:
                self._maze[row][col] = 1  # Closed door = wall
            else:
                self._maze[row][col] = 0  # Open door = free space

    def get_maze(self) -> List[List[Union[int, str]]]:

        return [row[:] for row in self._maze]  # Deep copy for lists

    def get_buttons(self) -> List[Tuple[int, int]]:

        return self._buttons.copy()
        
    def get_button_door_mapping(self) -> Dict[Tuple[int, int], Tuple[int, int]]:

        return self._button_door_mapping.copy()
        
    def get_door_states(self) -> Dict[Tuple[int, int], bool]:

        return self._door_states.copy()
        
    def is_door_open(self, door_pos: Tuple[int, int]) -> bool:

        return not self._door_states.get(door_pos, True)

    def get_start_position(self) -> Tuple[int, int]:

        return self._start_pos
    
    def get_goal_position(self) -> Tuple[int, int]:

        return self._goal_pos
    
    def set_start_position(self, pos: Tuple[int, int]) -> None:

        row, col = pos
        if not (0 <= row < len(self._maze) and 0 <= col < len(self._maze[0])):
            raise ValueError(f"Start position {pos} is outside maze boundaries")
        if self._maze[row][col] == 1:
            raise ValueError(f"Start position {pos} is inside a wall")
        
        self._start_pos = pos
    
    def set_goal_position(self, pos: Tuple[int, int]) -> None:

        row, col = pos
        if not (0 <= row < len(self._maze) and 0 <= col < len(self._maze[0])):
            raise ValueError(f"Goal position {pos} is outside maze boundaries")
        if self._maze[row][col] == 1:
            raise ValueError(f"Goal position {pos} is inside a wall")
        
        self._goal_pos = pos
    
    def create_maze(self, maze_array: List[List[Union[int, str]]]) -> None:

        # Validate input is a 2D list
        if not isinstance(maze_array, list) or not all(isinstance(row, list) for row in maze_array):
            raise ValueError("Maze must be a 2D list")
        if len(maze_array) == 0 or len(maze_array[0]) == 0:
            raise ValueError("Maze cannot be empty")
        if not all(len(row) == len(maze_array[0]) for row in maze_array):
            raise ValueError("All maze rows must have the same length")
            
        # Create deep copy of the maze
        maze = [row[:] for row in maze_array]
        self._maze = maze
        self._original_maze = [row[:] for row in maze]
        
        self._buttons = []
        self._button_door_mapping = {}
        self._door_states = {}
        
        if self._start_pos:
            row, col = self._start_pos
            if not (0 <= row < len(maze) and 0 <= col < len(maze[0])) or maze[row][col] == 1:
                for i in range(len(maze)):
                    for j in range(len(maze[0])):
                        if maze[i][j] == 0:
                            self._start_pos = (i, j)
                            break
                    if self._start_pos != (row, col):
                        break
        
        if self._goal_pos:
            row, col = self._goal_pos
            if not (0 <= row < len(maze) and 0 <= col < len(maze[0])) or maze[row][col] == 1:
                for i in range(len(maze)-1, -1, -1):
                    for j in range(len(maze[0])-1, -1, -1):
                        if maze[i][j] == 0 and (i, j) != self._start_pos:
                            self._goal_pos = (i, j)
                            break
                    if self._goal_pos != (row, col):
                        break
    
    def save_to_txt(self, file_path: str) -> None:

        try:
            with open(file_path, 'w') as f:
                # Convert maze to text representation
                for i in range(len(self._original_maze)):
                    for j in range(len(self._original_maze[0])):
                        if (i, j) == self._start_pos:
                            f.write('S')
                        elif (i, j) == self._goal_pos:
                            f.write('G')
                        elif (i, j) in self._buttons:
                            f.write('B')
                        else:
                            f.write(str(self._original_maze[i][j]))
                    f.write('\n')
        except IOError as e:
            raise IOError(f"Failed to save maze to {file_path}: {str(e)}")
    
    def get_config(self) -> bsAntMazeConfig:

        return self._config
    
    def set_config(self, maze_config: bsAntMazeConfig) -> None:

        if not isinstance(maze_config, bsAntMazeConfig):
            raise ValueError("Configuration must be an instance of bsAntMazeConfig")
        self._config = maze_config