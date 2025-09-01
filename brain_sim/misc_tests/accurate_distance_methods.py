#!/usr/bin/env python3
"""
Comparison of different wall distance calculation methods for maze navigation.

Methods compared:
1. Bilinear interpolation (current)
2. Bicubic interpolation
3. Ray casting to walls
4. Signed distance function (SDF)
5. Hybrid approach
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from scipy.ndimage import distance_transform_edt
from scipy import interpolate
from typing import Tuple, List


class AccurateDistanceCalculator:
    def __init__(self, maze_grid, cell_size=2.0):
        """
        Args:
            maze_grid: 2D numpy array where 0=open, 1=wall
            cell_size: world units per grid cell
        """
        self.maze_grid = maze_grid.astype(np.uint8)
        self.cell_size = float(cell_size)
        self.H, self.W = maze_grid.shape
        
        # Precompute various distance representations
        self._precompute_distance_field()
        self._precompute_wall_segments()
        self._setup_interpolators()
    
    def _precompute_distance_field(self):
        """Standard EDT-based distance field."""
        open_mask = (self.maze_grid == 0).astype(np.float32)
        distance_grid = distance_transform_edt(open_mask)
        
        # Apply the -0.5 correction for center-to-edge distance
        self.distance_field = (distance_grid - 0.5) * self.cell_size
        self.distance_field = np.clip(self.distance_field, 0.0, None)
    
    def _precompute_wall_segments(self):
        """Extract wall segments for ray casting."""
        self.wall_segments = []
        
        # Horizontal wall segments
        for y in range(self.H):
            for x in range(self.W):
                if self.maze_grid[y, x] == 1:  # wall cell
                    # Top edge
                    if y == self.H - 1 or self.maze_grid[y + 1, x] == 0:
                        x1, y1 = x * self.cell_size, (y + 1) * self.cell_size
                        x2, y2 = (x + 1) * self.cell_size, (y + 1) * self.cell_size
                        self.wall_segments.append(((x1, y1), (x2, y2)))
                    
                    # Bottom edge
                    if y == 0 or self.maze_grid[y - 1, x] == 0:
                        x1, y1 = x * self.cell_size, y * self.cell_size
                        x2, y2 = (x + 1) * self.cell_size, y * self.cell_size
                        self.wall_segments.append(((x1, y1), (x2, y2)))
                    
                    # Left edge
                    if x == 0 or self.maze_grid[y, x - 1] == 0:
                        x1, y1 = x * self.cell_size, y * self.cell_size
                        x2, y2 = x * self.cell_size, (y + 1) * self.cell_size
                        self.wall_segments.append(((x1, y1), (x2, y2)))
                    
                    # Right edge
                    if x == self.W - 1 or self.maze_grid[y, x + 1] == 0:
                        x1, y1 = (x + 1) * self.cell_size, y * self.cell_size
                        x2, y2 = (x + 1) * self.cell_size, (y + 1) * self.cell_size
                        self.wall_segments.append(((x1, y1), (x2, y2)))
    
    def _setup_interpolators(self):
        """Setup different interpolation methods."""
        # Create coordinate grids
        x_coords = np.arange(self.W) * self.cell_size + self.cell_size * 0.5
        y_coords = np.arange(self.H) * self.cell_size + self.cell_size * 0.5
        
        # Bicubic interpolator
        self.bicubic_interp = interpolate.RectBivariateSpline(
            y_coords, x_coords, self.distance_field, kx=3, ky=3, s=0
        )
    
    # ========== Method 1: Bilinear Interpolation (baseline) ==========
    
    def get_distance_bilinear(self, robot_x: float, robot_y: float) -> float:
        """Standard bilinear interpolation with -0.5 shift."""
        gxc = robot_x / self.cell_size - 0.5
        gyc = robot_y / self.cell_size - 0.5
        
        gxc = np.clip(gxc, 0.0, self.W - 1.0)
        gyc = np.clip(gyc, 0.0, self.H - 1.0)
        
        gx0 = int(np.floor(gxc))
        gy0 = int(np.floor(gyc))
        gx1 = min(gx0 + 1, self.W - 1)
        gy1 = min(gy0 + 1, self.H - 1)
        
        fx = gxc - gx0
        fy = gyc - gy0
        
        d00 = self.distance_field[gy0, gx0]
        d01 = self.distance_field[gy0, gx1]
        d10 = self.distance_field[gy1, gx0]
        d11 = self.distance_field[gy1, gx1]
        
        d0 = d00 * (1 - fx) + d01 * fx
        d1 = d10 * (1 - fx) + d11 * fx
        return d0 * (1 - fy) + d1 * fy
    
    # ========== Method 2: Bicubic Interpolation ==========
    
    def get_distance_bicubic(self, robot_x: float, robot_y: float) -> float:
        """Bicubic interpolation for smoother gradients."""
        try:
            return float(self.bicubic_interp(robot_y, robot_x)[0, 0])
        except:
            # Fallback to bilinear if bicubic fails
            return self.get_distance_bilinear(robot_x, robot_y)
    
    # ========== Method 3: Ray Casting ==========
    
    def get_distance_raycast(self, robot_x: float, robot_y: float) -> float:
        """Ray casting to find exact distance to nearest wall segment."""
        min_distance = float('inf')
        robot_pos = np.array([robot_x, robot_y])
        
        for (x1, y1), (x2, y2) in self.wall_segments:
            # Distance from point to line segment
            segment_start = np.array([x1, y1])
            segment_end = np.array([x2, y2])
            
            # Vector from start to end of segment
            segment_vec = segment_end - segment_start
            segment_length_sq = np.dot(segment_vec, segment_vec)
            
            if segment_length_sq == 0:
                # Degenerate segment (point)
                distance = np.linalg.norm(robot_pos - segment_start)
            else:
                # Project robot position onto the line
                t = max(0, min(1, np.dot(robot_pos - segment_start, segment_vec) / segment_length_sq))
                projection = segment_start + t * segment_vec
                distance = np.linalg.norm(robot_pos - projection)
            
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    # ========== Method 4: Signed Distance Function (SDF) ==========
    
    def get_distance_sdf(self, robot_x: float, robot_y: float) -> float:
        """
        Analytical SDF for rectangular obstacles.
        More accurate near corners and edges.
        """
        min_distance = float('inf')
        robot_pos = np.array([robot_x, robot_y])
        
        # Check distance to each wall cell using box SDF
        for y in range(self.H):
            for x in range(self.W):
                if self.maze_grid[y, x] == 1:  # wall cell
                    # Box center and half-size
                    box_center = np.array([(x + 0.5) * self.cell_size, (y + 0.5) * self.cell_size])
                    half_size = np.array([self.cell_size * 0.5, self.cell_size * 0.5])
                    
                    # SDF to box
                    offset = np.abs(robot_pos - box_center) - half_size
                    distance = np.linalg.norm(np.maximum(offset, 0.0)) + min(np.max(offset), 0.0)
                    min_distance = min(min_distance, distance)
        
        return min_distance
    
    # ========== Method 5: Hybrid Approach ==========
    
    def get_distance_hybrid(self, robot_x: float, robot_y: float, corner_threshold: float = 0.3) -> float:
        """
        Hybrid approach: use ray casting near corners, interpolation elsewhere.
        """
        # Quick check: are we near a corner?
        gx = int(robot_x / self.cell_size)
        gy = int(robot_y / self.cell_size)
        
        # Check if we're near a corner (within corner_threshold of cell boundary)
        fx = (robot_x / self.cell_size) % 1.0
        fy = (robot_y / self.cell_size) % 1.0
        
        near_corner = (fx < corner_threshold or fx > 1 - corner_threshold or
                      fy < corner_threshold or fy > 1 - corner_threshold)
        
        if near_corner:
            return self.get_distance_raycast(robot_x, robot_y)
        else:
            return self.get_distance_bicubic(robot_x, robot_y)
    
    # ========== Comparison and Visualization ==========
    
    def compare_methods(self, robot_x: float, robot_y: float) -> dict:
        """Compare all methods at a given position."""
        results = {}
        
        # Time each method
        methods = [
            ('bilinear', self.get_distance_bilinear),
            ('bicubic', self.get_distance_bicubic),
            ('raycast', self.get_distance_raycast),
            ('sdf', self.get_distance_sdf),
            ('hybrid', self.get_distance_hybrid)
        ]
        
        for name, method in methods:
            start_time = time.time()
            distance = method(robot_x, robot_y)
            elapsed = time.time() - start_time
            results[name] = {'distance': distance, 'time': elapsed * 1000}  # ms
        
        return results
    
    def visualize_method_comparison(self, resolution: int = 100):
        """Create heatmaps comparing different methods."""
        x_coords = np.linspace(0.1 * self.cell_size, (self.W - 0.1) * self.cell_size, resolution)
        y_coords = np.linspace(0.1 * self.cell_size, (self.H - 0.1) * self.cell_size, resolution)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        methods = {
            'Bilinear': self.get_distance_bilinear,
            'Bicubic': self.get_distance_bicubic,
            'Ray Cast': self.get_distance_raycast,
            'SDF': self.get_distance_sdf,
            'Hybrid': self.get_distance_hybrid
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        heatmaps = {}
        
        for idx, (name, method) in enumerate(methods.items()):
            print(f"Computing {name} heatmap...")
            distances = np.full((resolution, resolution), np.nan)
            
            for i in range(resolution):
                for j in range(resolution):
                    x, y = X[i, j], Y[i, j]
                    # Check if position is in open space
                    gx = int(x / self.cell_size)
                    gy = int(y / self.cell_size)
                    if 0 <= gx < self.W and 0 <= gy < self.H and self.maze_grid[gy, gx] == 0:
                        distances[i, j] = method(x, y)
            
            heatmaps[name] = distances
            
            # Plot heatmap
            extent = [0, self.W * self.cell_size, 0, self.H * self.cell_size]
            im = axes[idx].imshow(distances, cmap='viridis', origin='lower', 
                                extent=extent, interpolation='bilinear')
            axes[idx].set_title(f'{name}')
            axes[idx].set_xlabel('X (world)')
            axes[idx].set_ylabel('Y (world)')
            
            # Add contour lines for better visualization
            contour_levels = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
            X_cont, Y_cont = np.meshgrid(x_coords, y_coords)
            contours = axes[idx].contour(X_cont, Y_cont, distances, levels=contour_levels, 
                                       colors='white', linewidths=1, alpha=0.7)
            axes[idx].clabel(contours, inline=True, fontsize=8, fmt='%.1f')
            
            plt.colorbar(im, ax=axes[idx], label='Distance')
        
        # Difference plot: Bilinear vs Ray Cast (ground truth)
        if 'Bilinear' in heatmaps and 'Ray Cast' in heatmaps:
            diff = heatmaps['Bilinear'] - heatmaps['Ray Cast']
            im_diff = axes[5].imshow(diff, cmap='RdBu', origin='lower',
                                   extent=[0, self.W * self.cell_size, 0, self.H * self.cell_size],
                                   interpolation='bilinear', vmin=-0.5, vmax=0.5)
            axes[5].set_title('Error: Bilinear - Ray Cast')
            axes[5].set_xlabel('X (world)')
            axes[5].set_ylabel('Y (world)')
            
            # Add error contours
            error_levels = [-0.3, -0.1, 0.0, 0.1, 0.3]
            X_cont, Y_cont = np.meshgrid(x_coords, y_coords)
            error_contours = axes[5].contour(X_cont, Y_cont, diff, levels=error_levels, 
                                           colors='black', linewidths=1, alpha=0.8)
            axes[5].clabel(error_contours, inline=True, fontsize=8, fmt='%.1f')
            
            plt.colorbar(im_diff, ax=axes[5], label='Error')
        
        plt.tight_layout()
        return fig, heatmaps
    
    def visualize_contour_comparison(self, resolution: int = 150):
        """Create detailed contour comparison between methods."""
        x_coords = np.linspace(0.1 * self.cell_size, (self.W - 0.1) * self.cell_size, resolution)
        y_coords = np.linspace(0.1 * self.cell_size, (self.H - 0.1) * self.cell_size, resolution)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Compute distance fields for key methods
        methods = {
            'Bilinear': self.get_distance_bilinear,
            'Bicubic': self.get_distance_bicubic,
            'Ray Cast': self.get_distance_raycast,
        }
        
        distance_fields = {}
        for name, method in methods.items():
            print(f"Computing {name} distance field for contours...")
            distances = np.full((resolution, resolution), np.nan)
            
            for i in range(resolution):
                for j in range(resolution):
                    x, y = X[i, j], Y[i, j]
                    gx = int(x / self.cell_size)
                    gy = int(y / self.cell_size)
                    if 0 <= gx < self.W and 0 <= gy < self.H and self.maze_grid[gy, gx] == 0:
                        distances[i, j] = method(x, y)
            
            distance_fields[name] = distances
        
        # Create contour comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Define consistent contour levels
        contour_levels = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
        colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta', 'brown']
        
        # Plot maze background for all subplots
        extent = [0, self.W * self.cell_size, 0, self.H * self.cell_size]
        
        # 1. Bilinear contours
        axes[0, 0].imshow(self.maze_grid, cmap='binary', origin='lower', extent=extent, alpha=0.3)
        cs1 = axes[0, 0].contour(X, Y, distance_fields['Bilinear'], levels=contour_levels, 
                               colors=colors, linewidths=2)
        axes[0, 0].clabel(cs1, inline=True, fontsize=8, fmt='%.1f')
        axes[0, 0].set_title('Bilinear Interpolation - Contours')
        axes[0, 0].set_xlabel('X (world)')
        axes[0, 0].set_ylabel('Y (world)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Bicubic contours
        axes[0, 1].imshow(self.maze_grid, cmap='binary', origin='lower', extent=extent, alpha=0.3)
        cs2 = axes[0, 1].contour(X, Y, distance_fields['Bicubic'], levels=contour_levels, 
                               colors=colors, linewidths=2)
        axes[0, 1].clabel(cs2, inline=True, fontsize=8, fmt='%.1f')
        axes[0, 1].set_title('Bicubic Interpolation - Contours')
        axes[0, 1].set_xlabel('X (world)')
        axes[0, 1].set_ylabel('Y (world)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Ray cast contours (ground truth)
        axes[1, 0].imshow(self.maze_grid, cmap='binary', origin='lower', extent=extent, alpha=0.3)
        cs3 = axes[1, 0].contour(X, Y, distance_fields['Ray Cast'], levels=contour_levels, 
                               colors=colors, linewidths=2)
        axes[1, 0].clabel(cs3, inline=True, fontsize=8, fmt='%.1f')
        axes[1, 0].set_title('Ray Casting - Contours (Ground Truth)')
        axes[1, 0].set_xlabel('X (world)')
        axes[1, 0].set_ylabel('Y (world)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Overlay comparison
        axes[1, 1].imshow(self.maze_grid, cmap='binary', origin='lower', extent=extent, alpha=0.3)
        
        # Plot different methods with different line styles
        cs_bi = axes[1, 1].contour(X, Y, distance_fields['Bilinear'], levels=[0.5, 1.0, 1.5], 
                                 colors=['red'], linewidths=2, linestyles='solid', alpha=0.8)
        cs_bc = axes[1, 1].contour(X, Y, distance_fields['Bicubic'], levels=[0.5, 1.0, 1.5], 
                                 colors=['blue'], linewidths=2, linestyles='dashed', alpha=0.8)
        cs_rc = axes[1, 1].contour(X, Y, distance_fields['Ray Cast'], levels=[0.5, 1.0, 1.5], 
                                 colors=['green'], linewidths=3, linestyles='dotted', alpha=0.8)
        
        # Add labels to key contours
        axes[1, 1].clabel(cs_bi, inline=True, fontsize=8, fmt='%.1f')
        axes[1, 1].clabel(cs_bc, inline=True, fontsize=8, fmt='%.1f')
        axes[1, 1].clabel(cs_rc, inline=True, fontsize=8, fmt='%.1f')
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Bilinear'),
            Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Bicubic'),
            Line2D([0], [0], color='green', linewidth=3, linestyle=':', label='Ray Cast (Truth)')
        ]
        axes[1, 1].legend(handles=legend_elements, loc='upper right')
        
        axes[1, 1].set_title('Method Comparison - Key Contours Overlay')
        axes[1, 1].set_xlabel('X (world)')
        axes[1, 1].set_ylabel('Y (world)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, distance_fields
    
    def analyze_gradient_smoothness(self, resolution: int = 100):
        """Analyze gradient smoothness for different methods."""
        x_coords = np.linspace(0.1 * self.cell_size, (self.W - 0.1) * self.cell_size, resolution)
        y_coords = np.linspace(0.1 * self.cell_size, (self.H - 0.1) * self.cell_size, resolution)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        methods = {
            'Bilinear': self.get_distance_bilinear,
            'Bicubic': self.get_distance_bicubic,
            'Ray Cast': self.get_distance_raycast,
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (name, method) in enumerate(methods.items()):
            print(f"Computing gradients for {name}...")
            distances = np.full((resolution, resolution), np.nan)
            
            for i in range(resolution):
                for j in range(resolution):
                    x, y = X[i, j], Y[i, j]
                    gx = int(x / self.cell_size)
                    gy = int(y / self.cell_size)
                    if 0 <= gx < self.W and 0 <= gy < self.H and self.maze_grid[gy, gx] == 0:
                        distances[i, j] = method(x, y)
            
            # Compute gradient magnitude
            grad_y, grad_x = np.gradient(distances)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Plot gradient magnitude
            extent = [0, self.W * self.cell_size, 0, self.H * self.cell_size]
            im = axes[idx].imshow(grad_magnitude, cmap='plasma', origin='lower', 
                                extent=extent, interpolation='bilinear')
            axes[idx].set_title(f'{name} - Gradient Magnitude')
            axes[idx].set_xlabel('X (world)')
            axes[idx].set_ylabel('Y (world)')
            plt.colorbar(im, ax=axes[idx], label='Gradient Magnitude')
            
            # Add gradient field arrows (subsampled)
            skip = 8  # Show every 8th arrow
            X_sub = X[::skip, ::skip]
            Y_sub = Y[::skip, ::skip]
            grad_x_sub = grad_x[::skip, ::skip]
            grad_y_sub = grad_y[::skip, ::skip]
            
            # Normalize arrows for visibility
            magnitude_sub = np.sqrt(grad_x_sub**2 + grad_y_sub**2)
            with np.errstate(divide='ignore', invalid='ignore'):
                grad_x_norm = np.where(magnitude_sub > 0, grad_x_sub / magnitude_sub, 0)
                grad_y_norm = np.where(magnitude_sub > 0, grad_y_sub / magnitude_sub, 0)
            
            axes[idx].quiver(X_sub, Y_sub, grad_x_norm, grad_y_norm, 
                           magnitude_sub, scale=20, alpha=0.7, cmap='cool')
        
        plt.tight_layout()
        return fig


def create_test_maze():
    """Create a maze with challenging corner cases."""
    return np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 1],  # Creates challenging corners
        [1, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.uint8)


def main():
    
    print("Creating distance calculation comparison...")
    maze = create_test_maze()
    
    # Test with different cell sizes to see the effect
    cell_size = 2.0
    
    print(f"\n{'='*50}")
    print(f"Testing with cell_size = {cell_size}")
    print(f"{'='*50}")
    
    calc = AccurateDistanceCalculator(maze, cell_size=cell_size)
    
    # Test at a challenging position (near corner) - scale with cell size
    test_x, test_y = 2.8 * cell_size, 2.2 * cell_size  # Near corner
    
    print(f"\nTesting at position ({test_x}, {test_y}):")
    results = calc.compare_methods(test_x, test_y)
    
    print("\nMethod Comparison:")
    print(f"{'Method':<10} {'Distance':<10} {'Time (ms)':<10}")
    print("-" * 30)
    for method, data in results.items():
        print(f"{method:<10} {data['distance']:<10.4f} {data['time']:<10.4f}")
    
    print(f"\nGenerating comparison visualization for cell_size = {cell_size}...")
    fig, heatmaps = calc.visualize_method_comparison(resolution=60)
    fig.suptitle(f"Distance Calculation Method Comparison (cell_size = {cell_size})", fontsize=16)
    plt.show()
    
    print(f"\nGenerating contour comparison for cell_size = {cell_size}...")
    fig_contour, distance_fields = calc.visualize_contour_comparison(resolution=80)
    fig_contour.suptitle(f"Distance Field Contour Analysis (cell_size = {cell_size})", fontsize=16)
    plt.show()
    
    print(f"\nAnalyzing gradient smoothness for cell_size = {cell_size}...")
    fig_gradient = calc.analyze_gradient_smoothness(resolution=60)
    fig_gradient.suptitle(f"Distance Field Gradient Analysis (cell_size = {cell_size})", fontsize=16)
    plt.show()
    
    # Performance comparison with multiple points
    print(f"\nPerformance test (100 random points) for cell_size = {cell_size}:")
    np.random.seed(42)
    test_points = []
    for _ in range(100):
        while True:
            x = np.random.uniform(0.1, (calc.W - 0.1) * calc.cell_size)
            y = np.random.uniform(0.1, (calc.H - 0.1) * calc.cell_size)
            gx, gy = int(x / calc.cell_size), int(y / calc.cell_size)
            if calc.maze_grid[gy, gx] == 0:  # open space
                test_points.append((x, y))
                break
    
    method_times = {name: [] for name in ['bilinear', 'bicubic', 'raycast', 'sdf', 'hybrid']}
    
    for x, y in test_points:
        results = calc.compare_methods(x, y)
        for method, data in results.items():
            method_times[method].append(data['time'])
    
    print(f"{'Method':<10} {'Avg Time (ms)':<15} {'Speedup':<10}")
    print("-" * 35)
    bilinear_time = np.mean(method_times['bilinear'])
    for method, times in method_times.items():
        avg_time = np.mean(times)
        speedup = bilinear_time / avg_time
        print(f"{method:<10} {avg_time:<15.4f} {speedup:<10.2f}x")


if __name__ == "__main__":
    main()
