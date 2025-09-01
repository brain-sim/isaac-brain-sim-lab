#!/usr/bin/env python3
"""
Standalone unit tests for ray casting distance calculations in maze navigation.

This test suite provides a complete testing framework for ray casting algorithms
used in maze wall distance calculations, including a reference implementation
for validation.
"""

import unittest
import numpy as np
import torch
import math
from typing import List, Tuple, Optional


class SimpleRaycastCalculator:
    """
    Simple reference implementation of ray casting for distance calculations.
    Used as a baseline for testing more complex implementations.
    """
    
    def __init__(self, maze_grid: np.ndarray, cell_size: float = 2.0):
        """
        Initialize with a maze grid.
        
        Args:
            maze_grid: 2D numpy array where 0=open, 1=wall
            cell_size: Size of each cell in world units
        """
        self.maze_grid = maze_grid.astype(np.uint8)
        self.cell_size = float(cell_size)
        self.H, self.W = maze_grid.shape
        self.wall_segments = self._extract_wall_segments()
        
    def _extract_wall_segments(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Extract wall segments from maze grid."""
        segments = []
        
        for y in range(self.H):
            for x in range(self.W):
                if self.maze_grid[y, x] == 1:  # Wall cell
                    cell_x = x * self.cell_size
                    cell_y = y * self.cell_size
                    
                    # Top edge (if borders open space above)
                    if y == self.H - 1 or self.maze_grid[y + 1, x] == 0:
                        p1 = (cell_x, cell_y + self.cell_size)
                        p2 = (cell_x + self.cell_size, cell_y + self.cell_size)
                        segments.append((p1, p2))
                    
                    # Bottom edge (if borders open space below)
                    if y == 0 or self.maze_grid[y - 1, x] == 0:
                        p1 = (cell_x, cell_y)
                        p2 = (cell_x + self.cell_size, cell_y)
                        segments.append((p1, p2))
                    
                    # Left edge (if borders open space to the left)
                    if x == 0 or self.maze_grid[y, x - 1] == 0:
                        p1 = (cell_x, cell_y)
                        p2 = (cell_x, cell_y + self.cell_size)
                        segments.append((p1, p2))
                    
                    # Right edge (if borders open space to the right)
                    if x == self.W - 1 or self.maze_grid[y, x + 1] == 0:
                        p1 = (cell_x + self.cell_size, cell_y)
                        p2 = (cell_x + self.cell_size, cell_y + self.cell_size)
                        segments.append((p1, p2))
                        
        return segments
    
    def point_to_segment_distance(self, point: Tuple[float, float], 
                                segment: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
        """
        Calculate the minimum distance from a point to a line segment.
        
        Args:
            point: (x, y) coordinates of the point
            segment: ((x1, y1), (x2, y2)) line segment endpoints
            
        Returns:
            Minimum distance from point to segment
        """
        px, py = point
        (x1, y1), (x2, y2) = segment
        
        # Vector from segment start to end
        dx = x2 - x1
        dy = y2 - y1
        
        # If segment has zero length, return distance to point
        segment_length_sq = dx * dx + dy * dy
        if segment_length_sq == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        
        # Calculate parameter t for projection of point onto line
        # t = 0 at segment start, t = 1 at segment end
        t = ((px - x1) * dx + (py - y1) * dy) / segment_length_sq
        
        # Clamp t to [0, 1] to stay within segment
        t = max(0, min(1, t))
        
        # Find closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Return distance to closest point
        return math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
    
    def get_distance_to_walls(self, robot_x: float, robot_y: float) -> float:
        """
        Get minimum distance from robot position to any wall.
        
        Args:
            robot_x, robot_y: Robot position in world coordinates
            
        Returns:
            Minimum distance to any wall segment
        """
        if not self.wall_segments:
            return float('inf')
        
        min_distance = float('inf')
        robot_point = (robot_x, robot_y)
        
        for segment in self.wall_segments:
            distance = self.point_to_segment_distance(robot_point, segment)
            min_distance = min(min_distance, distance)
            
        return min_distance


class TorchRaycastCalculator:
    """
    Vectorized PyTorch implementation of ray casting for batch processing.
    """
    
    def __init__(self, wall_segments: torch.Tensor, device: Optional[torch.device] = None):
        """
        Initialize with precomputed wall segments.
        
        Args:
            wall_segments: Tensor of shape (N, 4) with format [x1, y1, x2, y2]
            device: PyTorch device for computations
        """
        self.device = device if device is not None else torch.device('cpu')
        self.wall_segments = wall_segments.to(self.device)
        
    def get_wall_distances(self, robot_positions: torch.Tensor) -> torch.Tensor:
        """
        Calculate distances from multiple robot positions to walls.
        
        Args:
            robot_positions: Tensor of shape (N, 2) with robot [x, y] positions
            
        Returns:
            Tensor of shape (N,) with minimum distances to walls
        """
        if self.wall_segments.numel() == 0:
            return torch.full((len(robot_positions),), float('inf'), 
                            device=robot_positions.device, dtype=torch.float32)
        
        # Ensure tensors are on the same device
        if self.wall_segments.device != robot_positions.device:
            self.wall_segments = self.wall_segments.to(robot_positions.device)
        
        N = robot_positions.shape[0]  # Number of robots
        M = self.wall_segments.shape[0]  # Number of wall segments
        
        # Expand dimensions for broadcasting
        robots = robot_positions.unsqueeze(1)  # (N, 1, 2)
        seg_start = self.wall_segments[:, :2].unsqueeze(0)  # (1, M, 2) - [x1, y1]
        seg_end = self.wall_segments[:, 2:].unsqueeze(0)    # (1, M, 2) - [x2, y2]
        
        # Calculate segment vectors and robot-to-start vectors
        seg_vec = seg_end - seg_start  # (1, M, 2)
        robot_vec = robots - seg_start  # (N, M, 2)
        
        # Calculate segment lengths squared
        seg_length_sq = torch.sum(seg_vec ** 2, dim=2, keepdim=True)  # (1, M, 1)
        seg_length_sq = torch.clamp(seg_length_sq, min=1e-8)  # Prevent division by zero
        
        # Project robots onto segment lines
        # t = dot(robot_vec, seg_vec) / seg_length_sq
        t = torch.sum(robot_vec * seg_vec, dim=2, keepdim=True) / seg_length_sq  # (N, M, 1)
        t = torch.clamp(t, 0.0, 1.0)  # Clamp to segment bounds
        
        # Find closest points on segments
        closest_points = seg_start + t * seg_vec  # (N, M, 2)
        
        # Calculate distances from robots to closest points
        distances = torch.norm(robots.squeeze(1).unsqueeze(1) - closest_points, dim=2)  # (N, M)
        
        # Find minimum distance for each robot
        min_distances, _ = torch.min(distances, dim=1)  # (N,)
        
        return min_distances


class TestSimpleRaycastCalculator(unittest.TestCase):
    """Test cases for the simple reference ray casting implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Simple 5x5 maze
        self.simple_maze = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.uint8)
        
        # Empty rectangular space
        self.empty_maze = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.uint8)
        
        self.cell_size = 2.0
        
    def test_wall_segment_extraction(self):
        """Test that wall segments are correctly extracted from maze grid."""
        calc = SimpleRaycastCalculator(self.simple_maze, self.cell_size)
        
        # Should have extracted some wall segments
        self.assertGreater(len(calc.wall_segments), 0)
        
        # Each segment should be properly formatted
        for segment in calc.wall_segments:
            self.assertIsInstance(segment, tuple)
            self.assertEqual(len(segment), 2)
            start, end = segment
            self.assertEqual(len(start), 2)
            self.assertEqual(len(end), 2)
            
    def test_point_to_segment_distance_horizontal(self):
        """Test distance calculation for horizontal line segments."""
        calc = SimpleRaycastCalculator(self.simple_maze, self.cell_size)
        
        # Horizontal segment
        segment = ((0.0, 0.0), (10.0, 0.0))
        
        # Point directly above segment
        point = (5.0, 3.0)
        distance = calc.point_to_segment_distance(point, segment)
        self.assertAlmostEqual(distance, 3.0, places=2)
        
        # Point at segment start
        point = (0.0, 0.0)
        distance = calc.point_to_segment_distance(point, segment)
        self.assertAlmostEqual(distance, 0.0, places=2)
        
        # Point beyond segment end
        point = (15.0, 0.0)
        distance = calc.point_to_segment_distance(point, segment)
        self.assertAlmostEqual(distance, 5.0, places=2)
        
    def test_point_to_segment_distance_vertical(self):
        """Test distance calculation for vertical line segments."""
        calc = SimpleRaycastCalculator(self.simple_maze, self.cell_size)
        
        # Vertical segment
        segment = ((0.0, 0.0), (0.0, 10.0))
        
        # Point to the right of segment
        point = (4.0, 5.0)
        distance = calc.point_to_segment_distance(point, segment)
        self.assertAlmostEqual(distance, 4.0, places=2)
        
    def test_distance_calculation_empty_space(self):
        """Test distance calculation in empty rectangular space."""
        calc = SimpleRaycastCalculator(self.empty_maze, self.cell_size)
        
        # Center of empty space
        # Empty space is cells (1,1) to (3,3), so center is at (4.0, 4.0) in world coords
        center_x = 2 * self.cell_size  # Center of cell 2
        center_y = 2 * self.cell_size  # Center of cell 2
        distance = calc.get_distance_to_walls(center_x, center_y)
        
        # Distance from center to edge should be cell_size (2.0)
        expected_distance = self.cell_size
        self.assertAlmostEqual(distance, expected_distance, places=1)
        
    def test_distance_calculation_near_wall(self):
        """Test distance calculation near a wall."""
        calc = SimpleRaycastCalculator(self.empty_maze, self.cell_size)
        
        # Position close to left wall
        near_wall_x = 1 * self.cell_size + 0.1  # Just inside first empty cell
        near_wall_y = 2 * self.cell_size  # Middle height
        distance = calc.get_distance_to_walls(near_wall_x, near_wall_y)
        
        # Should be very close to 0.1
        self.assertAlmostEqual(distance, 0.1, places=2)
        
    def test_distance_calculation_with_obstacles(self):
        """Test distance calculation with internal obstacles."""
        calc = SimpleRaycastCalculator(self.simple_maze, self.cell_size)
        
        # Position near internal wall at (2,2)
        obstacle_x = 2 * self.cell_size - 0.1  # Just left of internal wall
        obstacle_y = 2 * self.cell_size  # Middle height
        distance = calc.get_distance_to_walls(obstacle_x, obstacle_y)
        
        # Should be very close to internal wall
        self.assertLess(distance, 0.5)


class TestTorchRaycastCalculator(unittest.TestCase):
    """Test cases for the PyTorch vectorized ray casting implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        
        # Simple rectangular boundary
        self.simple_walls = torch.tensor([
            [0.0, 0.0, 10.0, 0.0],   # Bottom wall
            [0.0, 10.0, 10.0, 10.0], # Top wall
            [0.0, 0.0, 0.0, 10.0],   # Left wall
            [10.0, 0.0, 10.0, 10.0], # Right wall
        ], dtype=torch.float32, device=self.device)
        
    def test_single_robot_center_position(self):
        """Test distance calculation for single robot at center."""
        calc = TorchRaycastCalculator(self.simple_walls, self.device)
        
        robot_positions = torch.tensor([[5.0, 5.0]], dtype=torch.float32, device=self.device)
        distances = calc.get_wall_distances(robot_positions)
        
        # Distance from center to any wall should be 5.0
        expected_distance = 5.0
        self.assertAlmostEqual(distances[0].item(), expected_distance, places=2)
        
    def test_multiple_robots(self):
        """Test distance calculation for multiple robots."""
        calc = TorchRaycastCalculator(self.simple_walls, self.device)
        
        robot_positions = torch.tensor([
            [5.0, 5.0],  # Center
            [1.0, 1.0],  # Near corner
            [9.0, 5.0],  # Near right wall
            [5.0, 9.0],  # Near top wall
        ], dtype=torch.float32, device=self.device)
        
        distances = calc.get_wall_distances(robot_positions)
        
        expected_distances = [5.0, 1.0, 1.0, 1.0]
        
        for i, expected in enumerate(expected_distances):
            with self.subTest(i=i):
                self.assertAlmostEqual(distances[i].item(), expected, places=1)
                
    def test_empty_wall_segments(self):
        """Test handling of empty wall segments."""
        empty_walls = torch.empty((0, 4), dtype=torch.float32, device=self.device)
        calc = TorchRaycastCalculator(empty_walls, self.device)
        
        robot_positions = torch.tensor([[5.0, 5.0]], dtype=torch.float32, device=self.device)
        distances = calc.get_wall_distances(robot_positions)
        
        # Should return infinity for no walls
        self.assertTrue(torch.isinf(distances[0]))
        
    def test_device_consistency(self):
        """Test that calculations maintain device consistency."""
        calc = TorchRaycastCalculator(self.simple_walls, self.device)
        
        robot_positions = torch.tensor([[5.0, 5.0]], dtype=torch.float32, device=self.device)
        distances = calc.get_wall_distances(robot_positions)
        
        self.assertEqual(distances.device, self.device)
        
    def test_batch_processing_correctness(self):
        """Test that batch processing gives same results as individual processing."""
        calc = TorchRaycastCalculator(self.simple_walls, self.device)
        
        # Generate test positions
        robot_positions = torch.tensor([
            [2.0, 2.0],
            [3.0, 7.0],
            [8.0, 4.0],
        ], dtype=torch.float32, device=self.device)
        
        # Batch calculation
        batch_distances = calc.get_wall_distances(robot_positions)
        
        # Individual calculations
        individual_distances = []
        for i in range(len(robot_positions)):
            single_pos = robot_positions[i:i+1]
            single_dist = calc.get_wall_distances(single_pos)
            individual_distances.append(single_dist[0])
        
        individual_distances = torch.stack(individual_distances)
        
        # Results should be identical
        torch.testing.assert_close(batch_distances, individual_distances, rtol=1e-5, atol=1e-5)


class TestEdgeCasesAndRobustness(unittest.TestCase):
    """Test cases for edge cases and robustness."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        
    def test_degenerate_wall_segments(self):
        """Test handling of zero-length wall segments."""
        # Point segments (zero length)
        point_walls = torch.tensor([
            [5.0, 5.0, 5.0, 5.0],  # Point at (5,5)
        ], dtype=torch.float32, device=self.device)
        
        calc = TorchRaycastCalculator(point_walls, self.device)
        
        robot_positions = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=self.device)
        distances = calc.get_wall_distances(robot_positions)
        
        # Distance should be sqrt(5^2 + 5^2) ≈ 7.07
        expected_distance = math.sqrt(50)
        self.assertAlmostEqual(distances[0].item(), expected_distance, places=2)
        
    def test_numerical_precision(self):
        """Test numerical precision with very small distances."""
        # Very small wall segment
        tiny_walls = torch.tensor([
            [0.0, 0.0, 1e-6, 0.0],
        ], dtype=torch.float32, device=self.device)
        
        calc = TorchRaycastCalculator(tiny_walls, self.device)
        
        robot_positions = torch.tensor([[1e-7, 1e-7]], dtype=torch.float32, device=self.device)
        distances = calc.get_wall_distances(robot_positions)
        
        # Should handle small numbers without issues
        self.assertFalse(torch.isnan(distances[0]))
        self.assertFalse(torch.isinf(distances[0]))
        self.assertGreaterEqual(distances[0].item(), 0.0)
        
    def test_large_coordinates(self):
        """Test handling of large coordinate values."""
        # Large coordinate wall segments
        large_walls = torch.tensor([
            [1000.0, 1000.0, 1010.0, 1000.0],
        ], dtype=torch.float32, device=self.device)
        
        calc = TorchRaycastCalculator(large_walls, self.device)
        
        robot_positions = torch.tensor([[1005.0, 1005.0]], dtype=torch.float32, device=self.device)
        distances = calc.get_wall_distances(robot_positions)
        
        # Distance should be 5.0
        expected_distance = 5.0
        self.assertAlmostEqual(distances[0].item(), expected_distance, places=2)


class TestCrossValidation(unittest.TestCase):
    """Test cases for cross-validation between implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        
        # Test maze
        self.test_maze = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.uint8)
        
        self.cell_size = 2.0
        
    def test_implementations_consistency(self):
        """Test that different implementations give consistent results."""
        # Simple implementation
        simple_calc = SimpleRaycastCalculator(self.test_maze, self.cell_size)
        
        # Convert wall segments to torch format
        torch_segments = []
        for (x1, y1), (x2, y2) in simple_calc.wall_segments:
            torch_segments.append([x1, y1, x2, y2])
        
        if torch_segments:
            wall_tensor = torch.tensor(torch_segments, dtype=torch.float32, device=self.device)
            torch_calc = TorchRaycastCalculator(wall_tensor, self.device)
            
            # Test positions
            test_positions = [
                (3.0, 3.0),   # Center
                (2.1, 2.1),   # Near corner
                (5.9, 3.0),   # Near right edge
            ]
            
            for x, y in test_positions:
                with self.subTest(x=x, y=y):
                    # Simple implementation
                    simple_distance = simple_calc.get_distance_to_walls(x, y)
                    
                    # Torch implementation
                    robot_pos = torch.tensor([[x, y]], dtype=torch.float32, device=self.device)
                    torch_distance = torch_calc.get_wall_distances(robot_pos)[0].item()
                    
                    # Results should be very close
                    self.assertAlmostEqual(simple_distance, torch_distance, places=3)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2, buffer=True)
