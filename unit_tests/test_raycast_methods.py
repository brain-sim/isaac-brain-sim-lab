#!/usr/bin/env python3
"""
Unit tests for ray casting distance calculation methods.

This test suite focuses on testing the core ray casting algorithms
independently of the full WallConfiguration system, using the methods
from accurate_distance_methods.py as reference.
"""

import unittest
import numpy as np
import torch
import sys
import os

# Add the prototypes directory to Python path for reference implementation
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'prototypes'))

try:
    from accurate_distance_methods import AccurateDistanceCalculator
except ImportError as e:
    print(f"Warning: Could not import AccurateDistanceCalculator: {e}")
    AccurateDistanceCalculator = None


class TestRaycastingAlgorithms(unittest.TestCase):
    """Test cases for ray casting distance calculation algorithms."""
    
    def setUp(self):
        """Set up test fixtures with simple maze configurations."""
        # Simple 5x5 maze for testing
        self.simple_maze = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.uint8)
        
        # L-shaped maze
        self.l_shaped_maze = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1]
        ], dtype=np.uint8)
        
        # Empty space (no walls inside)
        self.empty_maze = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.uint8)
        
        self.cell_size = 2.0
        
    def test_wall_segment_extraction(self):
        """Test extraction of wall segments from maze grid."""
        if AccurateDistanceCalculator is None:
            self.skipTest("AccurateDistanceCalculator not available")
            
        calc = AccurateDistanceCalculator(self.simple_maze, self.cell_size)
        
        # Check that wall segments were extracted
        self.assertGreater(len(calc.wall_segments), 0)
        
        # Each wall segment should be a tuple of two points
        for segment in calc.wall_segments:
            self.assertIsInstance(segment, tuple)
            self.assertEqual(len(segment), 2)  # Start and end points
            start, end = segment
            self.assertEqual(len(start), 2)  # x, y coordinates
            self.assertEqual(len(end), 2)    # x, y coordinates
            
    def test_raycast_center_position(self):
        """Test ray casting from center of empty space."""
        if AccurateDistanceCalculator is None:
            self.skipTest("AccurateDistanceCalculator not available")
            
        calc = AccurateDistanceCalculator(self.empty_maze, self.cell_size)
        
        # Center of the 3x3 empty space (cell coordinates 2,2)
        center_x = 2 * self.cell_size + self.cell_size / 2  # 5.0
        center_y = 2 * self.cell_size + self.cell_size / 2  # 5.0
        
        distance = calc.get_distance_raycast(center_x, center_y)
        
        # Distance should be cell_size (2.0) from center to wall
        expected_distance = self.cell_size
        self.assertAlmostEqual(distance, expected_distance, places=2)
        
    def test_raycast_near_wall(self):
        """Test ray casting from position near a wall."""
        if AccurateDistanceCalculator is None:
            self.skipTest("AccurateDistanceCalculator not available")
            
        calc = AccurateDistanceCalculator(self.empty_maze, self.cell_size)
        
        # Position close to left wall
        near_wall_x = 1 * self.cell_size + 0.1  # Just inside the first empty cell
        near_wall_y = 2 * self.cell_size + self.cell_size / 2
        
        distance = calc.get_distance_raycast(near_wall_x, near_wall_y)
        
        # Distance should be approximately 0.1
        expected_distance = 0.1
        self.assertAlmostEqual(distance, expected_distance, places=2)
        
    def test_raycast_corner_position(self):
        """Test ray casting from position near a corner."""
        if AccurateDistanceCalculator is None:
            self.skipTest("AccurateDistanceCalculator not available")
            
        calc = AccurateDistanceCalculator(self.simple_maze, self.cell_size)
        
        # Position near bottom-left corner of central open space
        corner_x = 1 * self.cell_size + 0.1
        corner_y = 1 * self.cell_size + 0.1
        
        distance = calc.get_distance_raycast(corner_x, corner_y)
        
        # Distance should be approximately 0.1 (distance to nearest wall)
        expected_distance = 0.1
        self.assertAlmostEqual(distance, expected_distance, places=2)
        
    def test_raycast_obstacle_navigation(self):
        """Test ray casting around internal obstacles."""
        if AccurateDistanceCalculator is None:
            self.skipTest("AccurateDistanceCalculator not available")
            
        calc = AccurateDistanceCalculator(self.simple_maze, self.cell_size)
        
        # Position near the internal wall obstacle
        obstacle_x = 2 * self.cell_size + 0.1  # Just left of center wall
        obstacle_y = 2 * self.cell_size + self.cell_size / 2
        
        distance = calc.get_distance_raycast(obstacle_x, obstacle_y)
        
        # Should be very close to the internal wall
        self.assertLess(distance, 0.5)
        
    def test_raycast_accuracy_comparison(self):
        """Test ray casting accuracy against known distances."""
        if AccurateDistanceCalculator is None:
            self.skipTest("AccurateDistanceCalculator not available")
            
        calc = AccurateDistanceCalculator(self.empty_maze, self.cell_size)
        
        test_cases = [
            # (x, y, expected_distance)
            (3.0, 3.0, 1.0),  # 1 unit from wall
            (4.0, 4.0, 2.0),  # 2 units from wall (center)
            (2.5, 4.0, 1.5),  # 1.5 units from left wall
            (5.5, 4.0, 1.5),  # 1.5 units from right wall
        ]
        
        for x, y, expected in test_cases:
            with self.subTest(x=x, y=y):
                distance = calc.get_distance_raycast(x, y)
                self.assertAlmostEqual(distance, expected, places=1)


class TestVectorizedRaycastImplementation(unittest.TestCase):
    """Test cases for vectorized ray casting implementation."""
    
    def setUp(self):
        """Set up test fixtures for vectorized operations."""
        self.device = torch.device('cpu')
        
        # Simple wall segments for testing
        self.wall_segments = torch.tensor([
            [0.0, 0.0, 10.0, 0.0],   # Bottom wall
            [0.0, 10.0, 10.0, 10.0], # Top wall
            [0.0, 0.0, 0.0, 10.0],   # Left wall
            [10.0, 0.0, 10.0, 10.0], # Right wall
        ], dtype=torch.float32, device=self.device)
        
    def test_point_to_segment_distance(self):
        """Test point-to-line-segment distance calculation."""
        # Test distance from point to horizontal line segment
        point = torch.tensor([5.0, 5.0], dtype=torch.float32)
        segment_start = torch.tensor([0.0, 0.0], dtype=torch.float32)
        segment_end = torch.tensor([10.0, 0.0], dtype=torch.float32)
        
        # Calculate distance manually
        distance = self._point_to_segment_distance(point, segment_start, segment_end)
        
        # Expected distance is 5.0 (perpendicular distance to horizontal line)
        expected_distance = 5.0
        self.assertAlmostEqual(distance.item(), expected_distance, places=2)
        
    def test_vectorized_distance_calculation(self):
        """Test vectorized distance calculation for multiple points."""
        robot_positions = torch.tensor([
            [5.0, 5.0],  # Center
            [1.0, 1.0],  # Near corner
            [9.0, 5.0],  # Near right wall
            [5.0, 9.0],  # Near top wall
        ], dtype=torch.float32, device=self.device)
        
        distances = self._vectorized_wall_distances(robot_positions, self.wall_segments)
        
        expected_distances = [5.0, 1.0, 1.0, 1.0]
        
        for i, expected in enumerate(expected_distances):
            with self.subTest(i=i):
                self.assertAlmostEqual(distances[i].item(), expected, places=1)
                
    def test_batch_processing_performance(self):
        """Test performance of batch processing vs individual calculations."""
        import time
        
        # Generate random robot positions
        num_robots = 1000
        robot_positions = torch.rand(num_robots, 2, device=self.device) * 8.0 + 1.0
        
        # Time vectorized calculation
        start_time = time.time()
        batch_distances = self._vectorized_wall_distances(robot_positions, self.wall_segments)
        batch_time = time.time() - start_time
        
        # Time individual calculations
        start_time = time.time()
        individual_distances = []
        for i in range(num_robots):
            pos = robot_positions[i:i+1]
            dist = self._vectorized_wall_distances(pos, self.wall_segments)
            individual_distances.append(dist[0])
        individual_time = time.time() - start_time
        
        # Batch should be faster and give same results
        self.assertLess(batch_time, individual_time)
        
        individual_distances = torch.stack(individual_distances)
        torch.testing.assert_close(batch_distances, individual_distances, rtol=1e-5, atol=1e-5)
        
    def _point_to_segment_distance(self, point: torch.Tensor, seg_start: torch.Tensor, seg_end: torch.Tensor) -> torch.Tensor:
        """
        Calculate distance from point to line segment.
        Reference implementation for testing.
        """
        segment_vec = seg_end - seg_start
        point_vec = point - seg_start
        
        segment_length_sq = torch.sum(segment_vec ** 2)
        
        if segment_length_sq == 0:
            # Degenerate segment (point)
            return torch.norm(point - seg_start)
        
        # Project point onto segment line
        t = torch.clamp(torch.sum(point_vec * segment_vec) / segment_length_sq, 0.0, 1.0)
        
        # Find closest point on segment
        closest_point = seg_start + t * segment_vec
        
        # Return distance
        return torch.norm(point - closest_point)
        
    def _vectorized_wall_distances(self, robot_positions: torch.Tensor, wall_segments: torch.Tensor) -> torch.Tensor:
        """
        Vectorized implementation of wall distance calculation.
        Reference implementation based on the walls_raycast.py logic.
        """
        if wall_segments.numel() == 0:
            return torch.full((len(robot_positions),), float('inf'), 
                            device=robot_positions.device, dtype=torch.float32)
        
        N = robot_positions.shape[0]
        M = wall_segments.shape[0]
        
        # Expand dimensions for broadcasting
        robots = robot_positions.unsqueeze(1)  # (N, 1, 2)
        seg_start = wall_segments[:, :2].unsqueeze(0)  # (1, M, 2)
        seg_end = wall_segments[:, 2:].unsqueeze(0)    # (1, M, 2)
        
        # Calculate segment vectors and robot-to-start vectors
        seg_vec = seg_end - seg_start  # (1, M, 2)
        robot_vec = robots - seg_start  # (N, M, 2)
        
        # Calculate segment lengths squared
        seg_length_sq = torch.sum(seg_vec ** 2, dim=2, keepdim=True)  # (1, M, 1)
        seg_length_sq = torch.clamp(seg_length_sq, min=1e-8)
        
        # Project robots onto segment lines
        t = torch.sum(robot_vec * seg_vec, dim=2, keepdim=True) / seg_length_sq  # (N, M, 1)
        t = torch.clamp(t, 0.0, 1.0)
        
        # Find closest points on segments
        closest_points = seg_start + t * seg_vec  # (N, M, 2)
        
        # Calculate distances from robots to closest points
        distances_to_segments = torch.norm(robots.squeeze(1).unsqueeze(1) - closest_points, dim=2)  # (N, M)
        
        # Find minimum distance for each robot
        min_distances, _ = torch.min(distances_to_segments, dim=1)  # (N,)
        
        return min_distances


class TestEdgeCasesAndValidation(unittest.TestCase):
    """Test cases for edge cases and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        
    def test_empty_wall_segments(self):
        """Test handling of empty wall segments."""
        empty_segments = torch.empty((0, 4), dtype=torch.float32, device=self.device)
        robot_positions = torch.tensor([[5.0, 5.0]], dtype=torch.float32, device=self.device)
        
        # Create a simple reference implementation
        calc = TestVectorizedRaycastImplementation()
        distances = calc._vectorized_wall_distances(robot_positions, empty_segments)
        
        # Should return infinity for no walls
        self.assertTrue(torch.isinf(distances[0]))
        
    def test_degenerate_wall_segments(self):
        """Test handling of zero-length wall segments."""
        degenerate_segments = torch.tensor([
            [5.0, 5.0, 5.0, 5.0],  # Point segment
        ], dtype=torch.float32, device=self.device)
        
        robot_positions = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=self.device)
        
        calc = TestVectorizedRaycastImplementation()
        distances = calc._vectorized_wall_distances(robot_positions, degenerate_segments)
        
        # Distance should be sqrt(5^2 + 5^2) = ~7.07
        expected_distance = np.sqrt(25 + 25)
        self.assertAlmostEqual(distances[0].item(), expected_distance, places=2)
        
    def test_numerical_precision(self):
        """Test numerical precision and stability."""
        # Test with very small distances
        tiny_segments = torch.tensor([
            [0.0, 0.0, 1e-6, 0.0],  # Very small segment
        ], dtype=torch.float32, device=self.device)
        
        robot_positions = torch.tensor([[1e-7, 1e-7]], dtype=torch.float32, device=self.device)
        
        calc = TestVectorizedRaycastImplementation()
        distances = calc._vectorized_wall_distances(robot_positions, tiny_segments)
        
        # Should handle small numbers without numerical issues
        self.assertFalse(torch.isnan(distances[0]))
        self.assertFalse(torch.isinf(distances[0]))
        self.assertGreaterEqual(distances[0].item(), 0.0)


if __name__ == '__main__':
    # Configure test discovery and execution
    unittest.main(verbosity=2, buffer=True)
