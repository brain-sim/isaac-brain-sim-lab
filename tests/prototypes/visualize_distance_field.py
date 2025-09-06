#!/usr/bin/env python3
"""
Wall distance visualization (no contours), correctly aligned.

Root cause of prior 0.5-cell shift:
- distance_transform_edt returns values at *cell centers*
- Bilinear sampling treated them as if at *cell corners*

Fix:
- Convert world -> grid coords and then subtract 0.5 to sample in a center-based lattice:
    gxc = x / cell_size - 0.5
    gyc = y / cell_size - 0.5
- Clamp gxc, gyc to [0, W-1], [0, H-1] before bilinear interpolation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import random
from scipy.ndimage import distance_transform_edt


class WallDistanceVisualizer:
    def __init__(self, maze_grid, cell_size=1.0, diagonal_margin=0.125):
        """
        Args:
            maze_grid: 2D numpy array where 0=open, 1=wall
            cell_size: world units per grid cell
            diagonal_margin: small correction for corner/diagonal cases
        """
        self.maze_grid = maze_grid.astype(np.uint8)
        self.cell_size = float(cell_size)
        self.diagonal_margin = float(diagonal_margin)
        self.device = torch.device("cpu")
        self._precompute_distance_field()

    # ---------- Distance field ----------

    def _precompute_distance_field(self):
        """Distance from each open cell center to nearest wall EDGE, in world units."""
        open_mask = (self.maze_grid == 0).astype(np.float32)
        # EDT returns Euclidean distance (in pixels) from nonzero pixels to the nearest zero pixel *center*
        d_center = distance_transform_edt(open_mask)

        # Convert center-to-center distance to center-to-edge:
        # subtract 0.5 cell (center to shared edge); add small diagonal margin for corner cases
        df = (d_center - 0.5 - self.diagonal_margin) * self.cell_size
        df = np.clip(df, 0.0, None)
        self._distance_field = torch.tensor(df, dtype=torch.float32, device=self.device)

    def get_wall_distance_single_robot(self, robot_x, robot_y):
        """
        Bilinear interpolation of the precomputed distance field at (robot_x, robot_y) world coords.
        Critically: sample in CENTER coordinates (shift by -0.5 cell).
        Returns:
            (distance, (gx0, gy0, gx1, gy1, fx, fy))
        """
        # Convert to grid coords and SHIFT by -0.5 to account for center-based DF
        gxc = robot_x / self.cell_size - 0.5
        gyc = robot_y / self.cell_size - 0.5

        H, W = self._distance_field.shape
        # Clamp the continuous coordinates to valid sampling range
        gxc = float(np.clip(gxc, 0.0, W - 1.0))
        gyc = float(np.clip(gyc, 0.0, H - 1.0))

        gx0 = int(np.floor(gxc))
        gy0 = int(np.floor(gyc))
        gx1 = min(gx0 + 1, W - 1)
        gy1 = min(gy0 + 1, H - 1)

        fx = gxc - gx0  # in [0,1]
        fy = gyc - gy0  # in [0,1]

        d00 = self._distance_field[gy0, gx0].item()  # lower-left (bl)
        d01 = self._distance_field[gy0, gx1].item()  # lower-right (br)
        d10 = self._distance_field[gy1, gx0].item()  # upper-left (tl)
        d11 = self._distance_field[gy1, gx1].item()  # upper-right (tr)

        d0 = d00 * (1 - fx) + d01 * fx
        d1 = d10 * (1 - fx) + d11 * fx
        dist = d0 * (1 - fy) + d1 * fy
        return dist, (gx0, gy0, gx1, gy1, fx, fy)

    def get_random_valid_position(self, margin=0.1):
        """Return a random (x,y) inside an open cell, avoiding edges by `margin` (world units)."""
        H, W = self.maze_grid.shape
        candidates = []
        for y in range(H):
            for x in range(W):
                if self.maze_grid[y, x] == 0:
                    x0 = x * self.cell_size + margin
                    y0 = y * self.cell_size + margin
                    x1 = (x + 1) * self.cell_size - margin
                    y1 = (y + 1) * self.cell_size - margin
                    if x1 > x0 and y1 > y0:
                        candidates.append(
                            (
                                x0 + random.random() * (x1 - x0),
                                y0 + random.random() * (y1 - y0),
                            )
                        )
        if not candidates:
            raise ValueError("No valid positions found.")
        return random.choice(candidates)

    # ---------- Visualization helpers ----------

    def _common_extent(self):
        """Extent used consistently for every imshow: [0, W*cell] x [0, H*cell]."""
        H, W = self.maze_grid.shape
        return [0.0, W * self.cell_size, 0.0, H * self.cell_size]

    def _plot_maze_with_robot(self, ax, robot_x, robot_y):
        H, W = self.maze_grid.shape
        extent = self._common_extent()
        ax.imshow(
            self.maze_grid,
            cmap="binary",
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )
        ax.plot(robot_x, robot_y, "ro", markersize=10, label="Robot")

        # highlight the cell the robot is in
        cx = int(np.floor(robot_x / self.cell_size))
        cy = int(np.floor(robot_y / self.cell_size))
        cx = max(0, min(cx, W - 1))
        cy = max(0, min(cy, H - 1))
        rect = patches.Rectangle(
            (cx * self.cell_size, cy * self.cell_size),
            self.cell_size,
            self.cell_size,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            alpha=0.7,
        )
        ax.add_patch(rect)

        ax.set_aspect("equal")
        ax.set_xlabel("X (world)")
        ax.set_ylabel("Y (world)")
        # Set grid with 1-unit spacing
        H, W = self.maze_grid.shape
        ax.set_xticks(np.arange(0, W * self.cell_size + 1, 1))
        ax.set_yticks(np.arange(0, H * self.cell_size + 1, 1))
        ax.grid(True, alpha=0.25)
        ax.legend()

    def _plot_robot_cell_detail(self, ax, robot_x, robot_y, gx0, gy0, gx1, gy1, fx, fy):
        # Calculate the actual cell that contains the robot (without the -0.5 shift)
        robot_cell_x = int(np.floor(robot_x / self.cell_size))
        robot_cell_y = int(np.floor(robot_y / self.cell_size))

        # cell bounds in world coords for the robot's actual cell
        left = robot_cell_x * self.cell_size
        right = (robot_cell_x + 1) * self.cell_size
        bottom = robot_cell_y * self.cell_size
        top = (robot_cell_y + 1) * self.cell_size

        rect = patches.Rectangle(
            (left, bottom),
            self.cell_size,
            self.cell_size,
            linewidth=2,
            edgecolor="black",
            facecolor="lightgray",
            alpha=0.3,
        )
        ax.add_patch(rect)

        # Corner labels (showing the interpolation grid indices used for sampling)
        # Note: These might be different from the robot cell due to the -0.5 shift in sampling
        interp_left = gx0 * self.cell_size
        interp_right = (gx0 + 1) * self.cell_size
        interp_bottom = gy0 * self.cell_size
        interp_top = (gy0 + 1) * self.cell_size

        corners = [
            (interp_left, interp_bottom, "d00 (bl)"),
            (interp_right, interp_bottom, "d01 (br)"),
            (interp_left, interp_top, "d10 (tl)"),
            (interp_right, interp_top, "d11 (tr)"),
        ]
        for x, y, txt in corners:
            ax.plot(x, y, "bs", markersize=8)
            ax.annotate(txt, (x, y), xytext=(5, 5), textcoords="offset points")

        ax.plot(robot_x, robot_y, "ro", markersize=10, label="Robot")
        ax.text(
            robot_x + 0.05 * self.cell_size,
            robot_y + 0.05 * self.cell_size,
            f"fx={fx:.2f}\nfy={fy:.2f}",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )
        ax.set_aspect("equal")
        ax.set_xlim(left - 0.2 * self.cell_size, right + 0.2 * self.cell_size)
        ax.set_ylim(bottom - 0.2 * self.cell_size, top + 0.2 * self.cell_size)
        # Set grid with 1-unit spacing for the visible area
        ax.set_xticks(
            np.arange(
                np.floor(left - 0.2 * self.cell_size),
                np.ceil(right + 0.2 * self.cell_size) + 1,
                1,
            )
        )
        ax.set_yticks(
            np.arange(
                np.floor(bottom - 0.2 * self.cell_size),
                np.ceil(top + 0.2 * self.cell_size) + 1,
                1,
            )
        )
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("X (world)")
        ax.set_ylabel("Y (world)")
        ax.legend()

    def _plot_distance_field(self, ax):
        df = self._distance_field.cpu().numpy()
        extent = self._common_extent()
        im = ax.imshow(
            df, cmap="viridis", origin="lower", extent=extent, interpolation="nearest"
        )
        ax.set_aspect("equal")
        ax.set_xlabel("X (world)")
        ax.set_ylabel("Y (world)")
        # Set grid with 1-unit spacing
        H, W = self.maze_grid.shape
        ax.set_xticks(np.arange(0, W * self.cell_size + 1, 1))
        ax.set_yticks(np.arange(0, H * self.cell_size + 1, 1))
        ax.grid(True, alpha=0.25)
        plt.colorbar(im, ax=ax, label="Distance to Wall")

    def _plot_distance_info(self, ax, final_distance, interp_info):
        gx0, gy0, gx1, gy1, fx, fy = interp_info
        d00 = self._distance_field[gy0, gx0].item()
        d01 = self._distance_field[gy0, gx1].item()
        d10 = self._distance_field[gy1, gx0].item()
        d11 = self._distance_field[gy1, gx1].item()
        d0 = d00 * (1 - fx) + d01 * fx
        d1 = d10 * (1 - fx) + d11 * fx

        txt = f"""Bilinear Interpolation (CENTER-sampled DF):
  d00 (bl): {d00:.3f}
  d01 (br): {d01:.3f}
  d10 (tl): {d10:.3f}
  d11 (tr): {d11:.3f}

  fx: {fx:.3f}, fy: {fy:.3f}
  d0 = d00*(1-fx) + d01*fx = {d0:.3f}
  d1 = d10*(1-fx) + d11*fx = {d1:.3f}

  final = d0*(1-fy) + d1*fy = {final_distance:.3f}
"""
        ax.text(
            0.05,
            0.95,
            txt,
            transform=ax.transAxes,
            fontfamily="monospace",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"),
        )
        ax.axis("off")

    # ---------- Public plots ----------

    def visualize_single_robot_analysis(self, robot_x=None, robot_y=None):
        """Create a 2×3 figure analyzing a single robot position."""
        if robot_x is None or robot_y is None:
            robot_x, robot_y = self.get_random_valid_position()

        dist, info = self.get_wall_distance_single_robot(robot_x, robot_y)
        gx0, gy0, gx1, gy1, fx, fy = info

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 12))

        # 1) Maze layout with robot
        self._plot_maze_with_robot(ax1, robot_x, robot_y)
        ax1.set_title(f"Maze Layout\nRobot at ({robot_x:.2f}, {robot_y:.2f})")

        # 2) Robot cell detail
        robot_cell_x = int(np.floor(robot_x / self.cell_size))
        robot_cell_y = int(np.floor(robot_y / self.cell_size))
        self._plot_robot_cell_detail(ax2, robot_x, robot_y, gx0, gy0, gx1, gy1, fx, fy)
        ax2.set_title(
            f"Robot Cell Detail\nRobot in Grid Cell ({robot_cell_x}, {robot_cell_y})\nInterpolation from ({gx0}, {gy0})"
        )

        # 3) Distance field (per cell center)
        self._plot_distance_field(ax3)
        ax3.set_title("Precomputed Distance Field")

        # 4) Interp info
        self._plot_distance_info(ax4, dist, info)
        ax4.set_title(f"Bilinear Interpolation\nDistance: {dist:.3f}")

        # 5) Distance heatmap with contours
        H, W = self.maze_grid.shape
        extent = self._common_extent()
        x_min, x_max, y_min, y_max = extent
        resolution = 60  # Lower resolution for faster computation

        step_x = (x_max - x_min) / resolution
        step_y = (y_max - y_min) / resolution
        xs = x_min + (np.arange(resolution) + 0.5) * step_x
        ys = y_min + (np.arange(resolution) + 0.5) * step_y
        X, Y = np.meshgrid(xs, ys)

        distances = np.full((resolution, resolution), np.nan, dtype=float)
        for i in range(resolution):
            for j in range(resolution):
                rx, ry = X[i, j], Y[i, j]
                gx = int(np.floor(rx / self.cell_size))
                gy = int(np.floor(ry / self.cell_size))
                if 0 <= gx < W and 0 <= gy < H and self.maze_grid[gy, gx] == 0:
                    distances[i, j], _ = self.get_wall_distance_single_robot(rx, ry)

        im5 = ax5.imshow(
            distances,
            cmap="viridis",
            origin="lower",
            extent=extent,
            interpolation="bilinear",
        )

        # Add contour lines
        X_cont, Y_cont = np.meshgrid(xs, ys)
        contours = ax5.contour(
            X_cont, Y_cont, distances, levels=[0.5], colors="red", linewidths=2
        )
        ax5.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

        additional_levels = [0.25, 0.75, 1.0, 1.5, 2.0]
        ax5.contour(
            X_cont,
            Y_cont,
            distances,
            levels=additional_levels,
            colors="white",
            linewidths=1,
            alpha=0.7,
        )

        # Mark robot position on heatmap
        ax5.plot(
            robot_x,
            robot_y,
            "ro",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=2,
        )

        ax5.set_aspect("equal")
        ax5.set_title("Distance Heatmap with Contours")
        ax5.set_xlabel("X (world)")
        ax5.set_ylabel("Y (world)")
        ax5.set_xticks(np.arange(0, W * self.cell_size + 1, 1))
        ax5.set_yticks(np.arange(0, H * self.cell_size + 1, 1))
        ax5.grid(True, alpha=0.25)
        plt.colorbar(im5, ax=ax5, label="Distance to Wall")

        # 6) Reference maze layout
        ax6.imshow(
            self.maze_grid,
            cmap="binary",
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )
        ax6.plot(robot_x, robot_y, "ro", markersize=8, label="Robot")
        ax6.set_aspect("equal")
        ax6.set_title("Maze Reference")
        ax6.set_xlabel("X (world)")
        ax6.set_ylabel("Y (world)")
        ax6.set_xticks(np.arange(0, W * self.cell_size + 1, 1))
        ax6.set_yticks(np.arange(0, H * self.cell_size + 1, 1))
        ax6.grid(True, alpha=0.25)
        ax6.legend()

        plt.tight_layout()
        return fig

    def create_full_heatmap(self, resolution=100, contour_level=0.5):
        """
        Heatmap of distances across the maze (open space only).
        Uses the center-aligned sampler (–0.5 shift) via get_wall_distance_single_robot.

        Args:
            resolution: Number of pixels per dimension for the heatmap
            contour_level: Distance value for the highlighted red contour line
        """
        H, W = self.maze_grid.shape
        extent = self._common_extent()
        x_min, x_max, y_min, y_max = extent

        # Sample at imshow pixel centers for the chosen extent
        step_x = (x_max - x_min) / resolution
        step_y = (y_max - y_min) / resolution
        xs = x_min + (np.arange(resolution) + 0.5) * step_x
        ys = y_min + (np.arange(resolution) + 0.5) * step_y
        X, Y = np.meshgrid(xs, ys)

        distances = np.full((resolution, resolution), np.nan, dtype=float)
        for i in range(resolution):
            for j in range(resolution):
                rx, ry = X[i, j], Y[i, j]
                gx = int(np.floor(rx / self.cell_size))
                gy = int(np.floor(ry / self.cell_size))
                if 0 <= gx < W and 0 <= gy < H and self.maze_grid[gy, gx] == 0:
                    distances[i, j], _ = self.get_wall_distance_single_robot(rx, ry)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        im1 = ax1.imshow(
            distances,
            cmap="viridis",
            origin="lower",
            extent=extent,
            interpolation="bilinear",
        )

        X_cont, Y_cont = np.meshgrid(xs, ys)
        contours = ax1.contour(
            X_cont,
            Y_cont,
            distances,
            levels=[contour_level],
            colors="red",
            linewidths=2,
        )
        ax1.clabel(contours, inline=True, fontsize=10, fmt="%.1f")

        additional_levels = [0.25, 0.75, 1.0, 1.5, 2.0]
        ax1.contour(
            X_cont,
            Y_cont,
            distances,
            levels=additional_levels,
            colors="white",
            linewidths=1,
            alpha=0.7,
        )

        ax1.set_aspect("equal")
        ax1.set_title(
            f"Robot–Wall Distance Heatmap ({resolution}×{resolution})\nRed contour: {contour_level} distance level"
        )
        ax1.set_xlabel("X (world)")
        ax1.set_ylabel("Y (world)")
        # Set grid with 1-unit spacing
        H, W = self.maze_grid.shape
        ax1.set_xticks(np.arange(0, W * self.cell_size + 1, 1))
        ax1.set_yticks(np.arange(0, H * self.cell_size + 1, 1))
        ax1.grid(True, alpha=0.25)
        plt.colorbar(im1, ax=ax1, label="Distance to Wall")

        ax2.imshow(
            self.maze_grid,
            cmap="binary",
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )
        ax2.set_aspect("equal")
        ax2.set_title("Maze Layout (Reference)")
        ax2.set_xlabel("X (world)")
        ax2.set_ylabel("Y (world)")
        # Set grid with 1-unit spacing
        ax2.set_xticks(np.arange(0, W * self.cell_size + 1, 1))
        ax2.set_yticks(np.arange(0, H * self.cell_size + 1, 1))
        ax2.grid(True, alpha=0.25)

        plt.tight_layout()
        return fig


# ---------- Example maze & main ----------


def create_example_maze():
    """Example 10×10 maze."""
    return np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )


def main():
    print("Creating wall distance visualization (aligned, no contours)...")
    maze_grid = create_example_maze()
    viz = WallDistanceVisualizer(maze_grid, cell_size=2.0, diagonal_margin=0.125)

    # Test a random position in open space
    test_x, test_y = viz.get_random_valid_position()
    d, _ = viz.get_wall_distance_single_robot(test_x, test_y)
    print(f"Test position ({test_x:.2f}, {test_y:.2f}) -> distance {d:.3f}")

    print("Generating single-robot analysis...")
    fig1 = viz.visualize_single_robot_analysis(robot_x=test_x, robot_y=test_y)
    fig1.suptitle("Single Robot Wall Distance Analysis", fontsize=16)
    plt.show()

    print("Generating full distance heatmap...")
    fig2 = viz.create_full_heatmap(resolution=80, contour_level=0.6)
    fig2.suptitle("Complete Wall Distance Heatmap", fontsize=16)
    plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
