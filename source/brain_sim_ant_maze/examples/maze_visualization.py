import numpy as np
import matplotlib.pyplot as plt
from brain_sim_ant_maze import bsAntMaze, BRAIN_SIM_ANT_MAZE_CONFIG_DIR

def visualize_maze(maze, title="Maze Visualization", save_path=None):

    maze_grid = maze.get_maze()
    start_pos = maze.get_start_position()
    goal_pos = maze.get_goal_position()
    config = maze.get_config()
    
    cell_size = config.dimensions.cell_size
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    wall_color = config.visuals.get_color('wall')
    floor_color = config.visuals.get_color('floor')
    start_color = config.visuals.get_color('start')
    goal_color = config.visuals.get_color('goal')
    
    colors = np.zeros((maze_grid.shape[0], maze_grid.shape[1], 3))
    for i in range(maze_grid.shape[0]):
        for j in range(maze_grid.shape[1]):
            if maze_grid[i, j] == 1:  # 1 represents walls
                colors[i, j] = wall_color
            else:  # 0 represents floor
                colors[i, j] = floor_color
    
    ax.imshow(colors, extent=[0, maze_grid.shape[1] * cell_size, 0, maze_grid.shape[0] * cell_size])
    
    start_x = (start_pos[1] + 0.5) * cell_size
    start_y = (maze_grid.shape[0] - start_pos[0] - 0.5) * cell_size
    goal_x = (goal_pos[1] + 0.5) * cell_size
    goal_y = (maze_grid.shape[0] - goal_pos[0] - 0.5) * cell_size
    
    ax.plot(start_x, start_y, 'o', color=start_color, markersize=10, label='Start')
    ax.plot(goal_x, goal_y, 'o', color=goal_color, markersize=10, label='Goal')
    
    ax.grid(True, which='both', color='gray', linewidth=0.5)
    ax.set_xticks(np.arange(0, maze_grid.shape[1] * cell_size + cell_size, cell_size))
    ax.set_yticks(np.arange(0, maze_grid.shape[0] * cell_size + cell_size, cell_size))
    
    ax.set_xlabel('X (units)')
    ax.set_ylabel('Y (units)')
    
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Maze visualization saved to: {save_path}")
    
    plt.show()

def main():
    maze = bsAntMaze.from_config_file(f"{BRAIN_SIM_ANT_MAZE_CONFIG_DIR}/example_config.json")
    
    maze.build_from_txt(f"{BRAIN_SIM_ANT_MAZE_CONFIG_DIR}/example_maze.txt")
    
    maze.set_start_position((1, 1))
    maze.set_goal_position((5, 5))
    
     visualize_maze(maze, "Maze with Custom Configuration", save_path="maze_visualization.png")

if __name__ == "__main__":
    main()