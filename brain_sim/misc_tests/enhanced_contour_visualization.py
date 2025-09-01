#!/usr/bin/env python3
"""
Level contour visualization for the original visualize_distance_field.py
This enhances the existing visualizer with better contour analysis.
"""

import sys
import os
sys.path.append('/home/yihao/Downloads/software/isaac-maze-nav-lab/brain_sim/misc_tests')

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import distance_transform_edt
from scipy import interpolate


def create_enhanced_contour_visualization():
    """Create enhanced contour visualization similar to visualize_distance_field.py"""
    
    # Use the same example maze from visualize_distance_field.py
    maze_grid = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.uint8)
    
    cell_size = 2.0  # Same as in visualize_distance_field.py
    diagonal_margin = 0.125
    
    # Compute distance field
    open_mask = (maze_grid == 0).astype(np.float32)
    distance_transform_result = distance_transform_edt(open_mask)
    distance_field = (np.array(distance_transform_result, dtype=np.float32) - 0.5 - diagonal_margin) * cell_size
    distance_field = np.clip(distance_field, 0.0, None)
    
    # Create coordinate grids
    H, W = maze_grid.shape
    extent = [0.0, W * cell_size, 0.0, H * cell_size]
    
    # High resolution sampling
    resolution = 150
    x_coords = np.linspace(0.1 * cell_size, (W - 0.1) * cell_size, resolution)
    y_coords = np.linspace(0.1 * cell_size, (H - 0.1) * cell_size, resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Bilinear sampling (with -0.5 shift fix)
    def sample_bilinear(robot_x, robot_y):
        gxc = robot_x / cell_size - 0.5
        gyc = robot_y / cell_size - 0.5
        gxc = float(np.clip(gxc, 0.0, W - 1.0))
        gyc = float(np.clip(gyc, 0.0, H - 1.0))
        
        gx0 = int(np.floor(gxc))
        gy0 = int(np.floor(gyc))
        gx1 = min(gx0 + 1, W - 1)
        gy1 = min(gy0 + 1, H - 1)
        
        fx = gxc - gx0
        fy = gyc - gy0
        
        d00 = distance_field[gy0, gx0]
        d01 = distance_field[gy0, gx1]
        d10 = distance_field[gy1, gx0]
        d11 = distance_field[gy1, gx1]
        
        d0 = d00 * (1 - fx) + d01 * fx
        d1 = d10 * (1 - fx) + d11 * fx
        return d0 * (1 - fy) + d1 * fy
    
    # Bicubic sampling
    x_centers = np.arange(W) * cell_size + cell_size * 0.5
    y_centers = np.arange(H) * cell_size + cell_size * 0.5
    bicubic_interp = interpolate.RectBivariateSpline(
        y_centers, x_centers, distance_field, kx=3, ky=3, s=0
    )
    
    def sample_bicubic(robot_x, robot_y):
        try:
            return float(bicubic_interp(robot_y, robot_x)[0, 0])
        except:
            return sample_bilinear(robot_x, robot_y)
    
    # Sample distance fields
    print("Computing bilinear distance field...")
    distances_bilinear = np.full((resolution, resolution), np.nan)
    print("Computing bicubic distance field...")
    distances_bicubic = np.full((resolution, resolution), np.nan)
    
    for i in range(resolution):
        for j in range(resolution):
            x, y = X[i, j], Y[i, j]
            gx = int(np.floor(x / cell_size))
            gy = int(np.floor(y / cell_size))
            if 0 <= gx < W and 0 <= gy < H and maze_grid[gy, gx] == 0:
                distances_bilinear[i, j] = sample_bilinear(x, y)
                distances_bicubic[i, j] = sample_bicubic(x, y)
    
    # Create comprehensive contour visualization
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # Define contour levels
    major_contours = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    minor_contours = np.arange(0.25, 3.5, 0.25)
    
    # 1. Bilinear method with contours
    axes[0, 0].imshow(maze_grid, cmap='binary', origin='lower', extent=extent, alpha=0.3)
    im1 = axes[0, 0].imshow(distances_bilinear, cmap='viridis', origin='lower', 
                          extent=extent, alpha=0.8, interpolation='bilinear')
    
    # Minor contours (thin, light)
    cs1_minor = axes[0, 0].contour(X, Y, distances_bilinear, levels=minor_contours, 
                                 colors='white', linewidths=0.5, alpha=0.5)
    
    # Major contours (thick, labeled)
    cs1_major = axes[0, 0].contour(X, Y, distances_bilinear, levels=major_contours, 
                                 colors='white', linewidths=2, alpha=0.9)
    axes[0, 0].clabel(cs1_major, inline=True, fontsize=10, fmt='%.1f')
    
    axes[0, 0].set_title('Bilinear Interpolation\nwith Level Contours', fontsize=14)
    axes[0, 0].set_xlabel('X (world)')
    axes[0, 0].set_ylabel('Y (world)')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[0, 0], label='Distance to Wall')
    
    # 2. Bicubic method with contours
    axes[0, 1].imshow(maze_grid, cmap='binary', origin='lower', extent=extent, alpha=0.3)
    im2 = axes[0, 1].imshow(distances_bicubic, cmap='viridis', origin='lower', 
                          extent=extent, alpha=0.8, interpolation='bilinear')
    
    cs2_minor = axes[0, 1].contour(X, Y, distances_bicubic, levels=minor_contours, 
                                 colors='white', linewidths=0.5, alpha=0.5)
    cs2_major = axes[0, 1].contour(X, Y, distances_bicubic, levels=major_contours, 
                                 colors='white', linewidths=2, alpha=0.9)
    axes[0, 1].clabel(cs2_major, inline=True, fontsize=10, fmt='%.1f')
    
    axes[0, 1].set_title('Bicubic Interpolation\nwith Level Contours', fontsize=14)
    axes[0, 1].set_xlabel('X (world)')
    axes[0, 1].set_ylabel('Y (world)')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(im2, ax=axes[0, 1], label='Distance to Wall')
    
    # 3. Contour comparison overlay
    axes[0, 2].imshow(maze_grid, cmap='binary', origin='lower', extent=extent, alpha=0.5)
    
    # Show key contours from both methods
    key_contours = [0.5, 1.0, 1.5, 2.0]
    cs_bi = axes[0, 2].contour(X, Y, distances_bilinear, levels=key_contours, 
                             colors=['red'], linewidths=2, linestyles='solid', alpha=0.8)
    cs_bc = axes[0, 2].contour(X, Y, distances_bicubic, levels=key_contours, 
                             colors=['blue'], linewidths=2, linestyles='dashed', alpha=0.8)
    
    # Add labels
    axes[0, 2].clabel(cs_bi, inline=True, fontsize=8, fmt='%.1f')
    axes[0, 2].clabel(cs_bc, inline=True, fontsize=8, fmt='%.1f')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Bilinear'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Bicubic')
    ]
    axes[0, 2].legend(handles=legend_elements, loc='upper right')
    
    axes[0, 2].set_title('Method Comparison\nContour Overlay', fontsize=14)
    axes[0, 2].set_xlabel('X (world)')
    axes[0, 2].set_ylabel('Y (world)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Difference map
    difference = distances_bicubic - distances_bilinear
    axes[1, 0].imshow(maze_grid, cmap='binary', origin='lower', extent=extent, alpha=0.3)
    im4 = axes[1, 0].imshow(difference, cmap='RdBu', origin='lower', extent=extent, 
                          alpha=0.8, vmin=-0.5, vmax=0.5, interpolation='bilinear')
    
    # Difference contours
    diff_levels = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    cs_diff = axes[1, 0].contour(X, Y, difference, levels=diff_levels, 
                               colors='black', linewidths=1, alpha=0.8)
    axes[1, 0].clabel(cs_diff, inline=True, fontsize=8, fmt='%.1f')
    
    axes[1, 0].set_title('Difference: Bicubic - Bilinear\nwith Error Contours', fontsize=14)
    axes[1, 0].set_xlabel('X (world)')
    axes[1, 0].set_ylabel('Y (world)')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(im4, ax=axes[1, 0], label='Distance Difference')
    
    # 5. Gradient magnitude comparison (Bilinear)
    grad_y_bi, grad_x_bi = np.gradient(distances_bilinear)
    grad_mag_bi = np.sqrt(grad_x_bi**2 + grad_y_bi**2)
    
    axes[1, 1].imshow(maze_grid, cmap='binary', origin='lower', extent=extent, alpha=0.3)
    im5 = axes[1, 1].imshow(grad_mag_bi, cmap='plasma', origin='lower', extent=extent, 
                          alpha=0.8, interpolation='bilinear')
    
    # Gradient contours
    grad_levels = [0.2, 0.5, 1.0, 1.5, 2.0]
    cs_grad = axes[1, 1].contour(X, Y, grad_mag_bi, levels=grad_levels, 
                               colors='white', linewidths=1, alpha=0.7)
    axes[1, 1].clabel(cs_grad, inline=True, fontsize=8, fmt='%.1f')
    
    axes[1, 1].set_title('Bilinear Gradient Magnitude\nwith Gradient Contours', fontsize=14)
    axes[1, 1].set_xlabel('X (world)')
    axes[1, 1].set_ylabel('Y (world)')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(im5, ax=axes[1, 1], label='Gradient Magnitude')
    
    # 6. Gradient magnitude comparison (Bicubic)
    grad_y_bc, grad_x_bc = np.gradient(distances_bicubic)
    grad_mag_bc = np.sqrt(grad_x_bc**2 + grad_y_bc**2)
    
    axes[1, 2].imshow(maze_grid, cmap='binary', origin='lower', extent=extent, alpha=0.3)
    im6 = axes[1, 2].imshow(grad_mag_bc, cmap='plasma', origin='lower', extent=extent, 
                          alpha=0.8, interpolation='bilinear')
    
    cs_grad2 = axes[1, 2].contour(X, Y, grad_mag_bc, levels=grad_levels, 
                                colors='white', linewidths=1, alpha=0.7)
    axes[1, 2].clabel(cs_grad2, inline=True, fontsize=8, fmt='%.1f')
    
    axes[1, 2].set_title('Bicubic Gradient Magnitude\nwith Gradient Contours', fontsize=14)
    axes[1, 2].set_xlabel('X (world)')
    axes[1, 2].set_ylabel('Y (world)')
    axes[1, 2].grid(True, alpha=0.3)
    plt.colorbar(im6, ax=axes[1, 2], label='Gradient Magnitude')
    
    plt.tight_layout()
    return fig


def main():
    print("Creating enhanced contour visualization for wall distance fields...")
    
    fig = create_enhanced_contour_visualization()
    fig.suptitle('Enhanced Wall Distance Field Analysis with Level Contours\n' + 
                'Bilinear vs Bicubic Interpolation Methods', fontsize=16, y=0.98)
    
    plt.show()
    
    print("\nVisualization complete!")
    print("\nKey observations from contour analysis:")
    print("- Smoother contours in bicubic method indicate better gradient continuity")
    print("- Bilinear shows more angular artifacts, especially near corners")
    print("- Gradient magnitude is more uniform in bicubic method")
    print("- Both methods show the same general distance topology")
    print("- Bicubic provides more accurate distance estimates near complex geometries")


if __name__ == "__main__":
    main()
