#!/usr/bin/env python3

import torch
import numpy as np
import time
import sys
import os

# Add the source directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "source"))

from brain_sim_tasks.nav_avoid_tasks.direct.cognitiverl.walls import Walls


def test_vectorized_raycast():
    """Test the new vectorized ray casting implementation."""

    print("Testing Vectorized Ray Casting Implementation")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a simple maze
    maze_map = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )

    # Initialize walls
    walls = Walls(device=device)
    walls.update_walls(maze_map, cell_size=1.0, position_offset=[0.0, 0.0])

    # Test positions
    test_positions = torch.tensor(
        [
            [1.5, 1.5],  # Center of empty space
            [2.0, 2.0],  # At wall
            [1.1, 1.1],  # Near corner
            [3.9, 1.5],  # Near right wall
            [1.5, 3.9],  # Near bottom wall
        ],
        device=device,
        dtype=torch.float32,
    )

    print(f"\nTest positions: {test_positions.shape[0]} robots")

    # Test bilinear method (baseline)
    start_time = time.time()
    bilinear_distances = walls.get_wall_distances(test_positions)
    bilinear_time = time.time() - start_time

    print(f"\nBilinear method:")
    print(f"Time: {bilinear_time:.6f} seconds")
    print(f"Distances: {bilinear_distances}")

    # Test new ray casting method
    start_time = time.time()
    raycast_distances = walls.get_wall_distances_raycast(test_positions)
    raycast_time = time.time() - start_time

    print(f"\nVectorized Ray Casting method:")
    print(f"Time: {raycast_time:.6f} seconds")
    print(f"Distances: {raycast_distances}")

    # Compare results
    print(f"\nComparison:")
    print(
        f"Max difference: {torch.max(torch.abs(bilinear_distances - raycast_distances)):.6f}"
    )
    print(
        f"Mean difference: {torch.mean(torch.abs(bilinear_distances - raycast_distances)):.6f}"
    )
    print(f"Speed ratio (bilinear/raycast): {bilinear_time / raycast_time:.2f}x")

    # Performance test with many robots
    print(f"\nPerformance test with 1000 robots:")
    large_test_positions = (
        torch.rand(1000, 2, device=device) * 4 + 0.5
    )  # Random positions in maze

    # Bilinear
    start_time = time.time()
    _ = walls.get_wall_distances(large_test_positions)
    bilinear_large_time = time.time() - start_time

    # Ray casting
    start_time = time.time()
    _ = walls.get_wall_distances_raycast(large_test_positions)
    raycast_large_time = time.time() - start_time

    print(f"Bilinear (1000 robots): {bilinear_large_time:.6f} seconds")
    print(f"Ray casting (1000 robots): {raycast_large_time:.6f} seconds")
    print(f"Speed ratio: {bilinear_large_time / raycast_large_time:.2f}x")

    # Check wall segments
    print(f"\nWall segments info:")
    print(f"Number of wall segments: {walls._wall_segments.shape[0]}")
    print(f"Wall segments tensor shape: {walls._wall_segments.shape}")
    print(f"Sample wall segments:\n{walls._wall_segments[:5]}")


if __name__ == "__main__":
    test_vectorized_raycast()
