from typing import Dict, Any, Tuple, Optional, List
import json


class bsAntMazeDimensions:
    
    def __init__(self, 
                 wall_height: float = 0.5,
                 cell_size: float = 1.0,
                 ant_scale: float = 0.8):
        
        self.wall_height = wall_height
        self.cell_size = cell_size
        self.ant_scale = ant_scale
        self.wall_width = cell_size
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'wall_height': self.wall_height,
            'cell_size': self.cell_size,
            'ant_scale': self.ant_scale
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'bsAntMazeDimensions':
        return cls(
            wall_height=data.get('wall_height', 0.5),
            cell_size=data.get('cell_size', 1.0),
            ant_scale=data.get('ant_scale', 0.8)
        )


class bsAntMazeVisuals:
    def __init__(self):
        self._colors = {
            'wall': (0.8, 0.8, 0.8),
            'floor': (0.9, 0.9, 0.9),
            'ant': (0.1, 0.3, 0.8),
            'goal': (0.0, 0.8, 0.0),
            'start': (0.8, 0.0, 0.0)
        }
    
    def set_color(self, element: str, color: Tuple[float, float, float]) -> None:
        if element not in self._colors:
            raise ValueError(f"Unknown element: {element}")
        if not (isinstance(color, tuple) and len(color) == 3):
            raise ValueError("Color must be an RGB tuple with 3 values")
        if not all(0 <= c <= 1 for c in color):
            raise ValueError("Color values must be between 0 and 1")
        self._colors[element] = color
    
    def get_color(self, element: str) -> Tuple[float, float, float]:
        if element not in self._colors:
            raise ValueError(f"Unknown element: {element}")
        return self._colors[element]
    
    def to_dict(self) -> Dict[str, Any]:
        return {'colors': self._colors}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'bsAntMazeVisuals':
        visuals = cls()
        if 'colors' in data:
            for element, color in data['colors'].items():
                visuals.set_color(element, tuple(color))
        return visuals


class bsAntMazeConfig:
    
    def __init__(self, 
                 dimensions: Optional[bsAntMazeDimensions] = None,
                 visuals: Optional[bsAntMazeVisuals] = None,
                 custom_settings: Optional[Dict[str, Any]] = None):

        self.dimensions = dimensions if dimensions else bsAntMazeDimensions()
        self.visuals = visuals if visuals else bsAntMazeVisuals()
        self.custom_settings = custom_settings or {}
    
    def to_dict(self) -> Dict[str, Any]:

        return {
            'dimensions': self.dimensions.to_dict(),
            'colors': self.visuals.to_dict()['colors'],
            **self.custom_settings
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'bsAntMazeConfig':
        dim_dict = config_dict.get('dimensions', {})
        vis_dict = {'colors': config_dict.get('colors', {})}
        
        dimensions = bsAntMazeDimensions.from_dict(dim_dict)
        visuals = bsAntMazeVisuals.from_dict(vis_dict)
        
        custom_keys = set(config_dict.keys()) - {'dimensions', 'colors'}
        custom_settings = {k: config_dict[k] for k in custom_keys}
        
        return cls(dimensions, visuals, custom_settings)
    
    def update(self, config_dict: Dict[str, Any]) -> None:

        if 'dimensions' in config_dict:
            self.dimensions = bsAntMazeDimensions.from_dict(config_dict['dimensions'])
        
        if 'colors' in config_dict:
            for element, color in config_dict['colors'].items():
                self.visuals.set_color(element, tuple(color))
        
        custom_keys = set(config_dict.keys()) - {'dimensions', 'colors'}
        for key in custom_keys:
            self.custom_settings[key] = config_dict[key]

    @classmethod
    def from_json(cls, file_path: str) -> 'bsAntMazeConfig':

        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {str(e)}")
            
    def save_to_json(self, file_path: str) -> None:

        try:
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
        except IOError as e:
            raise IOError(f"Failed to save configuration to {file_path}: {str(e)}")
