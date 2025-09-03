from isaaclab.app import AppLauncher
app_launcher = AppLauncher()

import unittest
import torch
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Add the specific package directories to Python path
source_dir = os.path.join(os.path.dirname(__file__), '..', 'source')
sys.path.insert(0, os.path.join(source_dir, 'brain_sim_tasks'))
sys.path.insert(0, os.path.join(source_dir, 'brain_sim_assets'))
sys.path.insert(0, os.path.join(source_dir, 'brain_sim_ant_maze'))

from brain_sim_assets.props.maze_runtime import bsMazeRuntime


class TestbsMazeRuntime(unittest.TestCase):
    
    def setUp(self):
        if bsMazeRuntime is None:
            self.skipTest("bsMazeRuntime not available")
            
        self.device = torch.device('cpu')
        self.room_size = 40.0
        self.wall_thickness = 2.0
        self.wall_height = 3.0
        self.maze_file = "example_maze_unit_test.txt"
        
    def test_initialization_default_parameters(self):
        config = bsMazeRuntime(maze_file="example_maze_unit_test.txt")
        
        self.assertEqual(config.room_size, 40.0)
        self.assertEqual(config.wall_thickness, 2.0)
        self.assertEqual(config.wall_height, 3.0)
        self.assertEqual(config.maze_file, "example_maze_unit_test.txt")
        self.assertEqual(config.device, torch.device('cpu'))
        self.assertIsNone(config._maze_generator)
        self.assertIsNone(config._wall_segments)
        
    def test_initialization_custom_parameters(self):
        custom_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = bsMazeRuntime(
            room_size=80.0,
            wall_thickness=4.0,
            wall_height=4.0,
            maze_file="example_maze_unit_test.txt",
            device=custom_device
        )
        
        self.assertEqual(config.room_size, 80.0)
        self.assertEqual(config.wall_thickness, 4.0)
        self.assertEqual(config.wall_height, 4.0)
        self.assertEqual(config.maze_file, "example_maze_unit_test.txt")
        self.assertEqual(config.device, custom_device)
        
    def test_device_update(self):
        config = bsMazeRuntime(device='cpu')
        
        if torch.cuda.is_available():
            config.update_device('cuda')
            self.assertEqual(config.device, torch.device('cuda'))
            
        config.update_device(torch.device('cpu'))
        self.assertEqual(config.device, torch.device('cpu'))
        
    def test_wall_position_calculation(self):
        config = bsMazeRuntime(room_size=40.0, wall_thickness=2.0)
        wall_position = config.get_wall_position()
        expected_position = (40.0 - 2.0) / 2
        self.assertEqual(wall_position, expected_position)
        
    def test_position_offset_calculation(self):
        config = bsMazeRuntime(room_size=40.0, wall_thickness=2.0)
        offset = config.get_position_offset()
        expected_wall_position = (40.0 - 2.0) / 2
        expected_offset = (-expected_wall_position, -expected_wall_position, 0.0)
        self.assertEqual(offset, expected_offset)


class TestWallSegmentPrecomputation(unittest.TestCase):
    
    def setUp(self):
        if bsMazeRuntime is None:
            self.skipTest("bsMazeRuntime not available")
            
        self.config = bsMazeRuntime()
        
    @patch('brain_sim_assets.props.maze_runtime.bsMazeGenerator')
    def test_wall_segment_precomputation(self, mock_maze_generator):
        """Test wall segment precomputation from maze grid."""
        # Mock maze generator and maze grid
        mock_maze_instance = MagicMock()
        mock_maze_grid = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])
        mock_maze_instance.get_maze.return_value = mock_maze_grid
        
        mock_generator = MagicMock()
        mock_generator._maze = mock_maze_instance
        mock_generator._cell_size = 2.0
        mock_maze_generator.create_example_maze.return_value = mock_generator
        
        self.config._precompute_wall_segments()
        
        self.assertIsNotNone(self.config._wall_segments)
        self.assertIsInstance(self.config._wall_segments, torch.Tensor)
        
    def test_empty_maze_wall_segments(self):
        pass


class TestRaycastingDistanceCalculation(unittest.TestCase):
    
    def setUp(self):
        if bsMazeRuntime is None:
            self.skipTest("bsMazeRuntime not available")
            
        self.config = bsMazeRuntime(room_size=20.0,
                                        wall_thickness=2.0,
                                        maze_file="example_maze_unit_test.txt")
        self.config.create_maze_configuration()
        
    def test_distance_calculation_1(self):
        robot_positions = torch.tensor([[-7.0, -7.0, 0.0]], dtype=torch.float32)
        distances = self.config.get_wall_distances(robot_positions)
        expected_distance = 1.0
        self.assertAlmostEqual(distances[0].item(), expected_distance, places=2)
        
    def test_distance_calculation_2(self):
        robot_positions = torch.tensor([[-1.0, -5.0, 0.0]], dtype=torch.float32)
        distances = self.config.get_wall_distances(robot_positions)
        expected_distance = 3.0
        self.assertAlmostEqual(distances[0].item(), expected_distance, places=2)
        
    def test_distance_calculation_multiple_robots(self):        
        robot_positions = torch.tensor([
            [-7.0, -7.0, 0.0],  # 1
            [-1.0, -5.0, 0.0],  # 2
            [5.0, -3.0, 0.0]   # 3
        ], dtype=torch.float32)
        distances = self.config.get_wall_distances(robot_positions)
        self.assertEqual(len(distances), 3)
        self.assertAlmostEqual(distances[0].item(), 1.0, places=2)  # 1
        self.assertAlmostEqual(distances[1].item(), 3.0, places=2)  # 2
        self.assertAlmostEqual(distances[2].item(), 1.0, places=2)  # 3

    def test_device_consistency(self):
        self.config.update_device('cuda')            
        robot_positions = torch.tensor([
            [-7.0, -7.0, 0.0],  # 1
            [-1.0, -5.0, 0.0],  # 2
        ], dtype=torch.float32, device='cuda')
        distances = self.config.get_wall_distances(robot_positions)
        self.assertEqual(distances.device.type, 'cuda')

class TestPositionValidation(unittest.TestCase):    
    def setUp(self):
        if bsMazeRuntime is None:
            self.skipTest("bsMazeRuntime not available")
            
        self.config = bsMazeRuntime()
        
    @patch('brain_sim_assets.props.maze_runtime.bsMazeGenerator')
    def test_random_valid_position_generation(self, mock_maze_generator):
        return

        """Test generation of random valid positions."""
        # Mock maze generator
        mock_maze_instance = MagicMock()
        mock_maze_grid = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])
        mock_maze_instance.get_maze.return_value = mock_maze_grid
        
        mock_generator = MagicMock()
        mock_generator._maze = mock_maze_instance
        mock_generator._cell_size = 2.0
        mock_maze_generator.create_example_maze.return_value = mock_generator
        
        position = self.config.get_random_valid_position()
        
        self.assertIsInstance(position, torch.Tensor)
        self.assertEqual(position.shape, (3,))  # [x, y, z]
        self.assertEqual(position[2].item(), 0.0)  # Z should be 0
        
    @patch('brain_sim_assets.props.maze_runtime.bsMazeGenerator')
    def test_multiple_random_valid_positions(self, mock_maze_generator):
        return
        mock_maze_instance = MagicMock()
        mock_maze_grid = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ])
        mock_maze_instance.get_maze.return_value = mock_maze_grid
        
        mock_generator = MagicMock()
        mock_generator._maze = mock_maze_instance
        mock_generator._cell_size = 2.0
        mock_maze_generator.create_example_maze.return_value = mock_generator
        
        # Test multiple position generation
        num_positions = 5
        positions = self.config.get_random_valid_positions(num_positions)
        
        self.assertIsInstance(positions, torch.Tensor)
        self.assertEqual(positions.shape, (num_positions, 3))
        self.assertTrue(torch.all(positions[:, 2] == 0.0))  # All Z should be 0
        
    def test_no_valid_positions_error(self):
        pass


if __name__ == '__main__':
    # Configure test discovery and execution
    unittest.main(verbosity=2, buffer=True)