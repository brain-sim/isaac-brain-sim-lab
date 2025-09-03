from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)

import unittest
import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

# Add the specific package directories to Python path
source_dir = os.path.join(os.path.dirname(__file__), '..', 'source')
sys.path.insert(0, os.path.join(source_dir, 'brain_sim_tasks'))
sys.path.insert(0, os.path.join(source_dir, 'brain_sim_assets'))
sys.path.insert(0, os.path.join(source_dir, 'brain_sim_ant_maze'))

from brain_sim_assets.props.maze_runtime import bsMazeRuntime


class TestRaycastingDistanceCalculation():
    
    def __init__(self):
        self.config = bsMazeRuntime(room_size=20.0,
                                        maze_file="linear_maze.txt",
                                        maze_config="maze_cell_1.json")
        self.config.create_maze_configuration()
        print(self.config._wall_segments)
        print(self.config.get_position_offset())
        self.visualize_maze()
    
    def visualize_maze(self):
        wall_segments = self.config._wall_segments
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        for segment in wall_segments:
            x1, y1, x2, y2 = segment.cpu().numpy()
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=3, alpha=0.8)
        
        try:
            valid_positions = self.config.get_random_valid_positions(10)
            wall_distances = self.config.get_wall_distances(valid_positions)
            
            for i, (pos, distance) in enumerate(zip(valid_positions, wall_distances)):
                x, y = pos[0].item(), pos[1].item()
                dist_val = distance.item()
                
                ax.plot(x, y, 'ro', markersize=8, alpha=0.7)
                ax.text(x + 0.2, y + 0.2, f'{dist_val:.2f}', 
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.8),
                       ha='left', va='bottom')
        except Exception as e:
            print(f"Could not generate valid positions: {e}")
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Maze: {self.config.maze_file}')
        
        if len(wall_segments) > 0:
            wall_np = wall_segments.cpu().numpy()
            min_x = wall_np[:, [0, 2]].min() - 1
            max_x = wall_np[:, [0, 2]].max() + 1
            min_y = wall_np[:, [1, 3]].min() - 1
            max_y = wall_np[:, [1, 3]].max() + 1
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            
            ax.set_xticks(range(int(min_x), int(max_x) + 1, 1))
            ax.set_yticks(range(int(min_y), int(max_y) + 1, 1))
        
        plt.tight_layout()
        output_path = "/home/yihao/Downloads/software/isaac-maze-nav-lab/tests/visualizations/maze_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Maze visualization saved to: {output_path}")
        plt.close()


if __name__ == '__main__':
    config = TestRaycastingDistanceCalculation()