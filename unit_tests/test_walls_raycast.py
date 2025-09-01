#!/usr/bin/env python3
"""
Unit tests for walls_raycast.py module focusing on ray casting distance calculations.

This test suite covers:
- WallConfiguration class initialization and configuration
- Wall segment precomputation and representation
- Ray casting distance calculations
- Position validation and random position generation
- Device handling (CPU/GPU)
- Edge cases and error conditions
"""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher()

import unittest
import torch
import numpy as np
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Add the source directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'source'))

from brain_sim_tasks.nav_avoid_tasks.direct.cognitiverl.walls_raycast import WallConfiguration


class TestWallConfiguration(unittest.TestCase):
    """Test cases for WallConfiguration class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if WallConfiguration is None:
            self.skipTest("WallConfiguration not available")
            
        self.device = torch.device('cpu')
        self.room_size = 40.0
        self.wall_thickness = 2.0
        self.wall_height = 3.0
        self.maze_file = "example_maze_sq.txt"
        
        # Create a simple test maze file content for mocking
        self.test_maze_content = """5 5
11111
10001
10101
10001
11111"""
        
    def test_initialization_default_parameters(self):
        """Test WallConfiguration initialization with default parameters."""
        config = WallConfiguration()
        
        self.assertEqual(config.room_size, 40.0)
        self.assertEqual(config.wall_thickness, 2.0)
        self.assertEqual(config.wall_height, 3.0)
        self.assertEqual(config.maze_file, "example_maze_sq.txt")
        self.assertEqual(config.device, torch.device('cpu'))
        self.assertIsNone(config._maze_generator)
        self.assertIsNone(config._wall_segments)
        
    def test_initialization_custom_parameters(self):
        """Test WallConfiguration initialization with custom parameters."""
        custom_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = WallConfiguration(
            room_size=50.0,
            wall_thickness=3.0,
            wall_height=4.0,
            maze_file="custom_maze.txt",
            device=custom_device
        )
        
        self.assertEqual(config.room_size, 50.0)
        self.assertEqual(config.wall_thickness, 3.0)
        self.assertEqual(config.wall_height, 4.0)
        self.assertEqual(config.maze_file, "custom_maze.txt")
        self.assertEqual(config.device, custom_device)
        
    def test_device_update(self):
        """Test device update functionality."""
        config = WallConfiguration(device='cpu')
        
        # Test string device update
        if torch.cuda.is_available():
            config.update_device('cuda')
            self.assertEqual(config.device, torch.device('cuda'))
            
        # Test torch.device object update
        config.update_device(torch.device('cpu'))
        self.assertEqual(config.device, torch.device('cpu'))
        
    def test_wall_position_calculation(self):
        """Test wall position calculation."""
        config = WallConfiguration(room_size=40.0, wall_thickness=2.0)
        wall_position = config.get_wall_position()
        expected_position = (40.0 - 2.0) / 2
        self.assertEqual(wall_position, expected_position)
        
    def test_position_offset_calculation(self):
        """Test position offset calculation."""
        config = WallConfiguration(room_size=40.0, wall_thickness=2.0)
        offset = config.get_position_offset()
        expected_wall_position = (40.0 - 2.0) / 2
        expected_offset = (-expected_wall_position, -expected_wall_position, 0.0)
        self.assertEqual(offset, expected_offset)


class TestWallSegmentPrecomputation(unittest.TestCase):
    """Test cases for wall segment precomputation."""
    
    def setUp(self):
        """Set up test fixtures."""
        if WallConfiguration is None:
            self.skipTest("WallConfiguration not available")
            
        self.config = WallConfiguration()
        
    @patch('brain_sim_tasks.nav_avoid_tasks.direct.cognitiverl.walls_raycast.bsMazeGenerator')
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
        
        # Test precomputation
        self.config._precompute_wall_segments()
        
        # Verify wall segments were created
        self.assertIsNotNone(self.config._wall_segments)
        self.assertIsInstance(self.config._wall_segments, torch.Tensor)
        
    def test_empty_maze_wall_segments(self):
        """Test wall segment handling for empty maze."""
        # This test would require mocking to simulate an empty maze
        pass


class TestRaycastingDistanceCalculation(unittest.TestCase):
    """Test cases for ray casting distance calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        if WallConfiguration is None:
            self.skipTest("WallConfiguration not available")
            
        self.config = WallConfiguration()
        
        # Create simple wall segments for testing
        self.simple_wall_segments = torch.tensor([
            [0.0, 0.0, 10.0, 0.0],  # Bottom wall
            [0.0, 10.0, 10.0, 10.0],  # Top wall
            [0.0, 0.0, 0.0, 10.0],  # Left wall
            [10.0, 0.0, 10.0, 10.0],  # Right wall
        ], dtype=torch.float32)
        
    def test_distance_calculation_center_position(self):
        """Test distance calculation for robot at center of enclosed space."""
        self.config._wall_segments = self.simple_wall_segments
        
        # Robot at center of 10x10 square
        robot_positions = torch.tensor([[5.0, 5.0, 0.0]], dtype=torch.float32)
        distances = self.config.get_wall_distances(robot_positions)
        
        # Distance should be 5.0 (half the width/height)
        expected_distance = 5.0
        self.assertAlmostEqual(distances[0].item(), expected_distance, places=2)
        
    def test_distance_calculation_near_wall(self):
        """Test distance calculation for robot near a wall."""
        self.config._wall_segments = self.simple_wall_segments
        
        # Robot near bottom wall
        robot_positions = torch.tensor([[5.0, 1.0, 0.0]], dtype=torch.float32)
        distances = self.config.get_wall_distances(robot_positions)
        
        # Distance should be 1.0
        expected_distance = 1.0
        self.assertAlmostEqual(distances[0].item(), expected_distance, places=2)
        
    def test_distance_calculation_corner_position(self):
        """Test distance calculation for robot near a corner."""
        self.config._wall_segments = self.simple_wall_segments
        
        # Robot near bottom-left corner
        robot_positions = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
        distances = self.config.get_wall_distances(robot_positions)
        
        # Distance should be 1.0 (distance to nearest wall)
        expected_distance = 1.0
        self.assertAlmostEqual(distances[0].item(), expected_distance, places=2)
        
    def test_distance_calculation_multiple_robots(self):
        """Test distance calculation for multiple robot positions."""
        self.config._wall_segments = self.simple_wall_segments
        
        robot_positions = torch.tensor([
            [5.0, 5.0, 0.0],  # Center
            [1.0, 1.0, 0.0],  # Near corner
            [9.0, 5.0, 0.0],  # Near right wall
        ], dtype=torch.float32)
        
        distances = self.config.get_wall_distances(robot_positions)
        
        self.assertEqual(len(distances), 3)
        self.assertAlmostEqual(distances[0].item(), 5.0, places=2)  # Center
        self.assertAlmostEqual(distances[1].item(), 1.0, places=2)  # Near corner
        self.assertAlmostEqual(distances[2].item(), 1.0, places=2)  # Near right wall
        
    def test_empty_wall_segments(self):
        """Test distance calculation with no wall segments."""
        self.config._wall_segments = torch.empty((0, 4), dtype=torch.float32)
        
        robot_positions = torch.tensor([[5.0, 5.0, 0.0]], dtype=torch.float32)
        distances = self.config.get_wall_distances(robot_positions)
        
        # Should return infinity when no walls present
        self.assertTrue(torch.isinf(distances[0]))
        
    def test_device_consistency(self):
        """Test that calculations maintain device consistency."""
        if torch.cuda.is_available():
            self.config.update_device('cuda')
            self.config._wall_segments = self.simple_wall_segments.to('cuda')
            
            robot_positions = torch.tensor([[5.0, 5.0, 0.0]], dtype=torch.float32, device='cuda')
            distances = self.config.get_wall_distances(robot_positions)
            
            self.assertEqual(distances.device, torch.device('cuda'))


class TestPositionValidation(unittest.TestCase):
    """Test cases for position validation and generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        if WallConfiguration is None:
            self.skipTest("WallConfiguration not available")
            
        self.config = WallConfiguration()
        
    @patch('brain_sim_tasks.nav_avoid_tasks.direct.cognitiverl.walls_raycast.bsMazeGenerator')
    def test_random_valid_position_generation(self, mock_maze_generator):
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
        
        # Test single position generation
        position = self.config.get_random_valid_position()
        
        self.assertIsInstance(position, torch.Tensor)
        self.assertEqual(position.shape, (3,))  # [x, y, z]
        self.assertEqual(position[2].item(), 0.0)  # Z should be 0
        
    @patch('brain_sim_tasks.nav_avoid_tasks.direct.cognitiverl.walls_raycast.bsMazeGenerator')
    def test_multiple_random_valid_positions(self, mock_maze_generator):
        """Test generation of multiple random valid positions."""
        # Mock maze generator
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
        """Test error handling when no valid positions exist."""
        # This would require mocking a maze with no open spaces
        pass


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test cases for edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if WallConfiguration is None:
            self.skipTest("WallConfiguration not available")
            
        self.config = WallConfiguration()
        
    def test_uninitialized_wall_segments(self):
        """Test behavior when wall segments are not precomputed."""
        robot_positions = torch.tensor([[5.0, 5.0, 0.0]], dtype=torch.float32)
        
        with self.assertRaises(AssertionError):
            self.config.get_wall_distances(robot_positions)
            
    def test_invalid_robot_positions_shape(self):
        """Test error handling for invalid robot position tensor shapes."""
        self.config._wall_segments = torch.tensor([
            [0.0, 0.0, 10.0, 0.0]
        ], dtype=torch.float32)
        
        # Test with wrong shape
        invalid_positions = torch.tensor([5.0, 5.0], dtype=torch.float32)  # Missing batch dimension
        
        with self.assertRaises((IndexError, RuntimeError)):
            self.config.get_wall_distances(invalid_positions)
            
    def test_zero_length_wall_segments(self):
        """Test handling of degenerate wall segments (zero length)."""
        # Wall segment with zero length (point)
        self.config._wall_segments = torch.tensor([
            [5.0, 5.0, 5.0, 5.0]  # Point segment
        ], dtype=torch.float32)
        
        robot_positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        distances = self.config.get_wall_distances(robot_positions)
        
        # Should still compute distance correctly
        expected_distance = np.sqrt(5.0**2 + 5.0**2)  # Distance to point (5,5)
        self.assertAlmostEqual(distances[0].item(), expected_distance, places=2)


class TestPerformanceAndAccuracy(unittest.TestCase):
    """Test cases for performance and accuracy validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        if WallConfiguration is None:
            self.skipTest("WallConfiguration not available")
            
        self.config = WallConfiguration()
        
    def test_large_batch_processing(self):
        """Test processing of large batches of robot positions."""
        self.config._wall_segments = torch.tensor([
            [0.0, 0.0, 10.0, 0.0],
            [0.0, 10.0, 10.0, 10.0],
            [0.0, 0.0, 0.0, 10.0],
            [10.0, 0.0, 10.0, 10.0],
        ], dtype=torch.float32)
        
        # Large batch of robot positions
        batch_size = 1000
        robot_positions = torch.rand(batch_size, 3) * 8.0 + 1.0  # Random positions in [1,9]
        robot_positions[:, 2] = 0.0  # Set Z to 0
        
        distances = self.config.get_wall_distances(robot_positions)
        
        self.assertEqual(len(distances), batch_size)
        self.assertTrue(torch.all(distances > 0))  # All distances should be positive
        self.assertTrue(torch.all(distances <= 5.0))  # Max distance in 10x10 square is 5.0
        
    def test_numerical_precision(self):
        """Test numerical precision of distance calculations."""
        # Create a simple case where we know the exact answer
        self.config._wall_segments = torch.tensor([
            [0.0, 0.0, 1.0, 0.0]  # Single horizontal wall segment
        ], dtype=torch.float32)
        
        # Robot directly above the wall segment
        robot_positions = torch.tensor([[0.5, 1.0, 0.0]], dtype=torch.float32)
        distances = self.config.get_wall_distances(robot_positions)
        
        # Distance should be exactly 1.0
        self.assertAlmostEqual(distances[0].item(), 1.0, places=5)


if __name__ == '__main__':
    # Configure test discovery and execution
    unittest.main(verbosity=2, buffer=True)