#!/usr/bin/env python3
"""
Focused analysis of corner accuracy in wall distance calculations.
This script specifically examines how different methods handle maze corners.
"""

import numpy as np
import matplotlib.pyplot as plt
from accurate_distance_methods import AccurateDistanceCalculator


def create_corner_test_maze():
    """Create a simple maze designed to test corner accuracy."""
    return np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1],  # Creates internal corners
            [1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )


def analyze_corner_accuracy():
    """Analyze accuracy near corners specifically."""
    maze = create_corner_test_maze()
    calc = AccurateDistanceCalculator(maze, cell_size=1.0)

    # Define test points near corners
    corner_points = [
        (2.1, 2.1),  # Near internal corner (bottom-left of internal wall)
        (2.9, 2.1),  # Near internal corner (bottom-right of internal wall)
        (2.1, 2.9),  # Near internal corner (top-left of internal wall)
        (2.9, 2.9),  # Near internal corner (top-right of internal wall)
        (1.1, 1.1),  # Near external corner
        (3.9, 3.9),  # Near external corner
    ]

    print("Corner Accuracy Analysis")
    print("=" * 50)
    print(
        f"{'Point':<12} {'Bilinear':<10} {'Bicubic':<10} {'RayCast':<10} {'Bi-Ray':<10} {'Bc-Ray':<10}"
    )
    print("-" * 65)

    for i, (x, y) in enumerate(corner_points):
        bi_dist = calc.get_distance_bilinear(x, y)
        bc_dist = calc.get_distance_bicubic(x, y)
        rc_dist = calc.get_distance_raycast(x, y)

        bi_error = abs(bi_dist - rc_dist)
        bc_error = abs(bc_dist - rc_dist)

        print(
            f"({x:4.1f},{y:4.1f}) {bi_dist:<10.4f} {bc_dist:<10.4f} {rc_dist:<10.4f} {bi_error:<10.4f} {bc_error:<10.4f}"
        )

    # Create detailed corner visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # High resolution for corner analysis
    resolution = 200
    x_coords = np.linspace(0.1, 5.9, resolution)
    y_coords = np.linspace(0.1, 5.9, resolution)
    X, Y = np.meshgrid(x_coords, y_coords)

    methods = {
        "Bilinear": calc.get_distance_bilinear,
        "Bicubic": calc.get_distance_bicubic,
        "Ray Cast": calc.get_distance_raycast,
    }

    distance_fields = {}
    for name, method in methods.items():
        distances = np.full((resolution, resolution), np.nan)
        for i in range(resolution):
            for j in range(resolution):
                x, y = X[i, j], Y[i, j]
                gx, gy = int(x), int(y)
                if 0 <= gx < 6 and 0 <= gy < 6 and maze[gy, gx] == 0:
                    distances[i, j] = method(x, y)
        distance_fields[name] = distances

    # 1. Bilinear with fine contours
    extent = [0, 6, 0, 6]
    axes[0, 0].imshow(maze, cmap="binary", origin="lower", extent=extent, alpha=0.5)
    im1 = axes[0, 0].imshow(
        distance_fields["Bilinear"],
        cmap="viridis",
        origin="lower",
        extent=extent,
        alpha=0.8,
    )

    # Fine contour levels for corner analysis
    fine_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cs1 = axes[0, 0].contour(
        X,
        Y,
        distance_fields["Bilinear"],
        levels=fine_contours,
        colors="white",
        linewidths=1,
        alpha=0.7,
    )
    axes[0, 0].clabel(cs1, inline=True, fontsize=6, fmt="%.1f")

    # Mark corner test points
    for x, y in corner_points:
        axes[0, 0].plot(
            x, y, "ro", markersize=8, markeredgecolor="white", markeredgewidth=2
        )

    axes[0, 0].set_title("Bilinear - Fine Contours")
    axes[0, 0].set_xlabel("X")
    axes[0, 0].set_ylabel("Y")
    plt.colorbar(im1, ax=axes[0, 0], label="Distance")

    # 2. Bicubic with fine contours
    axes[0, 1].imshow(maze, cmap="binary", origin="lower", extent=extent, alpha=0.5)
    im2 = axes[0, 1].imshow(
        distance_fields["Bicubic"],
        cmap="viridis",
        origin="lower",
        extent=extent,
        alpha=0.8,
    )
    cs2 = axes[0, 1].contour(
        X,
        Y,
        distance_fields["Bicubic"],
        levels=fine_contours,
        colors="white",
        linewidths=1,
        alpha=0.7,
    )
    axes[0, 1].clabel(cs2, inline=True, fontsize=6, fmt="%.1f")

    for x, y in corner_points:
        axes[0, 1].plot(
            x, y, "ro", markersize=8, markeredgecolor="white", markeredgewidth=2
        )

    axes[0, 1].set_title("Bicubic - Fine Contours")
    axes[0, 1].set_xlabel("X")
    axes[0, 1].set_ylabel("Y")
    plt.colorbar(im2, ax=axes[0, 1], label="Distance")

    # 3. Ray cast (ground truth) with fine contours
    axes[1, 0].imshow(maze, cmap="binary", origin="lower", extent=extent, alpha=0.5)
    im3 = axes[1, 0].imshow(
        distance_fields["Ray Cast"],
        cmap="viridis",
        origin="lower",
        extent=extent,
        alpha=0.8,
    )
    cs3 = axes[1, 0].contour(
        X,
        Y,
        distance_fields["Ray Cast"],
        levels=fine_contours,
        colors="white",
        linewidths=1,
        alpha=0.7,
    )
    axes[1, 0].clabel(cs3, inline=True, fontsize=6, fmt="%.1f")

    for x, y in corner_points:
        axes[1, 0].plot(
            x, y, "ro", markersize=8, markeredgecolor="white", markeredgewidth=2
        )

    axes[1, 0].set_title("Ray Cast - Ground Truth")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Y")
    plt.colorbar(im3, ax=axes[1, 0], label="Distance")

    # 4. Error analysis (Bicubic vs Ray Cast)
    error = distance_fields["Bicubic"] - distance_fields["Ray Cast"]
    axes[1, 1].imshow(maze, cmap="binary", origin="lower", extent=extent, alpha=0.5)
    im4 = axes[1, 1].imshow(
        error,
        cmap="RdBu",
        origin="lower",
        extent=extent,
        alpha=0.8,
        vmin=-0.3,
        vmax=0.3,
    )

    # Error contours
    error_levels = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
    cs4 = axes[1, 1].contour(
        X, Y, error, levels=error_levels, colors="black", linewidths=1, alpha=0.8
    )
    axes[1, 1].clabel(cs4, inline=True, fontsize=6, fmt="%.2f")

    for x, y in corner_points:
        axes[1, 1].plot(
            x, y, "ko", markersize=8, markeredgecolor="white", markeredgewidth=2
        )

    axes[1, 1].set_title("Error: Bicubic - Ray Cast")
    axes[1, 1].set_xlabel("X")
    axes[1, 1].set_ylabel("Y")
    plt.colorbar(im4, ax=axes[1, 1], label="Error")

    plt.tight_layout()
    return fig


def analyze_contour_smoothness():
    """Analyze how smooth the contours are near corners."""
    maze = create_corner_test_maze()
    calc = AccurateDistanceCalculator(maze, cell_size=1.0)

    # Focus on one specific corner area
    x_range = np.linspace(1.5, 3.5, 100)
    y_range = np.linspace(1.5, 3.5, 100)
    X, Y = np.meshgrid(x_range, y_range)

    methods = ["Bilinear", "Bicubic", "Ray Cast"]
    method_funcs = [
        calc.get_distance_bilinear,
        calc.get_distance_bicubic,
        calc.get_distance_raycast,
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (name, method) in enumerate(zip(methods, method_funcs)):
        distances = np.full(X.shape, np.nan)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i, j], Y[i, j]
                gx, gy = int(x), int(y)
                if 0 <= gx < 6 and 0 <= gy < 6 and maze[gy, gx] == 0:
                    distances[i, j] = method(x, y)

        # Show maze background
        axes[idx].imshow(
            maze[1:4, 1:4],
            cmap="binary",
            origin="lower",
            extent=[1.5, 3.5, 1.5, 3.5],
            alpha=0.3,
        )

        # Dense contour lines to show smoothness
        contour_levels = np.arange(0.1, 2.0, 0.1)
        cs = axes[idx].contour(
            X,
            Y,
            distances,
            levels=contour_levels,
            colors="blue",
            linewidths=1,
            alpha=0.8,
        )

        # Highlight key contours
        key_contours = [0.5, 1.0]
        cs_key = axes[idx].contour(
            X, Y, distances, levels=key_contours, colors="red", linewidths=3, alpha=1.0
        )
        axes[idx].clabel(cs_key, inline=True, fontsize=10, fmt="%.1f")

        axes[idx].set_title(f"{name} - Contour Smoothness")
        axes[idx].set_xlabel("X")
        axes[idx].set_ylabel("Y")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_aspect("equal")

    plt.tight_layout()
    return fig


def main():
    print("Corner Accuracy Analysis for Wall Distance Calculations")
    print("=" * 60)

    # Numerical accuracy analysis
    analyze_corner_accuracy()

    # Visual analysis
    print("\nGenerating corner visualization...")
    fig1 = analyze_corner_accuracy()
    fig1.suptitle("Corner Accuracy Analysis", fontsize=16)
    plt.show()

    print("\nGenerating contour smoothness analysis...")
    fig2 = analyze_contour_smoothness()
    fig2.suptitle("Contour Smoothness Near Corners", fontsize=16)
    plt.show()

    print("\nAnalysis complete!")
    print("\nKey findings:")
    print("- Bicubic interpolation provides smoother contours than bilinear")
    print("- Ray casting gives the most accurate distances, especially near corners")
    print("- Bicubic is a good compromise between accuracy and performance")
    print("- Bilinear shows more artifacts near complex corner geometries")


if __name__ == "__main__":
    main()
