# Isaac Maze Navigation Lab

A reinforcement learning environment for robotic navigation tasks using NVIDIA Isaac Lab, featuring landmark-based navigation with obstacle avoidance in maze-like environments.

## Overview

This project provides a simulation environment for training and testing robotic navigation policies. The environment features multiple task variants with different complexity levels.

## Features

- **Multiple Task Variants**: Four distinct navigation tasks with varying complexity
- **Landmark-based Navigation**: Target and obstacle landmark systems
- **Collision Detection**: Real-time collision checking with configurable tolerances
- **Vector-based Offsets**: Deterministic spatial relationships between landmarks
- **Dynamic Bounds**: Room-size-relative boundary constraints
- **Visualization**: Real-time landmark and robot visualization
- **Collision-aware Reset**: Robot repositioning to avoid landmark conflicts

## Task Variants

### 1. Single (`single`)
- **Markers**: 2 landmarks per group (1 target + 1 obstacle)
- **Behavior**: Random landmark generation with vector-based spatial relationships
- **Reset**: Landmark-only reset (robot maintains position)
- **Trials**: A session of configurable trials

### 2. Single Fixed Goal (`single_fix_goal`)
- **Markers**: 2 landmarks per group (1 target + 1 obstacle)
- **Behavior**: Fixed obstacle position at [5.0, 5.0], target with vector offset
- **Reset**: Full robot reset with collision checking
- **Trials**: A session of configurable trials

### 3. Rotation Array (`rot_array`)
- **Markers**: 3 landmarks per group (1 target + 2 obstacles)
- **Behavior**: Perpendicular landmark arrangement with random generation
- **Reset**: Landmark-only reset (robot maintains position)
- **Trials**: A session of configurable trials

### 4. Rotation Array Fixed Goal (`rot_array_fix_goal`)
- **Markers**: 3 landmarks per group (1 target + 2 obstacles)
- **Behavior**: Fixed obstacle positions at [5.0, 0.0] and [-5.0, 0.0]
- **Reset**: Full robot reset with collision checking
- **Trials**: A session of configurable trials

## Architecture

```
landmark_tasks/direct/
├── components/                 # Base environment components
│   ├── env_component_objective.py    # Goal and collision detection
│   ├── env_component_observation.py  # Sensor data processing
│   ├── env_component_reward.py       # Reward function logic
│   ├── env_component_robot.py        # Robot control and actions
│   ├── env_component_termination.py  # Episode termination logic
│   └── env_component_waypoint.py     # Landmark generation base
├── tasks/                      # Task-specific implementations
│   ├── single/                 # Single waypoint task
│   ├── single_fix_goal/        # Single waypoint with fixed goal
│   ├── rot_array/              # Multi-waypoint rotation task
│   └── rot_array_fix_goal/     # Multi-waypoint with fixed goals
├── landmark_env.py             # Main environment class
└── landmark_env_cfg.py         # Environment configuration
```

## Inheritance Architecture

The project uses a component-based inheritance system where each task variant can override specific behaviors while inheriting common functionality.

### Base Components (in `components/`)
All base components provide default implementations that can be inherited by task-specific variants:

- **`EnvComponentWaypoint`**: Base landmark generation logic
- **`EnvComponentObjective`**: Base goal detection and collision checking
- **`EnvComponentObservation`**: Base sensor data processing
- **`EnvComponentReward`**: Base reward calculation logic
- **`EnvComponentRobot`**: Base robot control and actions
- **`EnvComponentTermination`**: Base episode termination conditions

### Task-Specific Derived Components (in `tasks/<task_name>/`)
Each task directory contains derived components that override specific methods:

```python
# Example: Derived landmark component
from ...components.env_component_waypoint import EnvComponentWaypoint

class DerivedEnvComponentWaypoint(EnvComponentWaypoint):
    """Task-specific landmark generation."""
    
    def generate_waypoints(self, env_ids, robot_poses, waypoint_offset=None):
        # Override with task-specific landmark generation logic
        pass
```

### Component Injection Pattern
The main environment accepts component classes as parameters, allowing flexible task configuration:

```python
from landmark_env import LandmarkEnv
from tasks.single_fix_goal.env_component_objective import DerivedEnvComponentObjective
from tasks.single_fix_goal.env_component_waypoint import DerivedEnvComponentWaypoint

# Inject task-specific components
env = LandmarkEnv(
    cfg,
    component_objective_cls=DerivedEnvComponentObjective,
    component_waypoint_cls=DerivedEnvComponentWaypoint,
    # Other components use base implementations if not specified
)
```

### Inheritance Patterns by Task

#### Single Tasks (`single`, `single_fix_goal`)
- **Landmark Component**: Override `generate_waypoint_group()` for 2-landmark logic
- **Objective Component**: Override collision detection for single obstacle
- **Other Components**: Inherit base implementations

#### Rotation Array Tasks (`rot_array`, `rot_array_fix_goal`)
- **Landmark Component**: Override `generate_waypoint_group()` for 3-landmark perpendicular logic
- **Objective Component**: Override collision detection for multiple obstacles
- **Other Components**: Inherit base implementations

#### Fixed Goal Variants (`single_fix_goal`, `rot_array_fix_goal`)
- **Landmark Component**: Override landmark generation with fixed positions
- **Objective Component**: Add collision-aware robot reset logic
- **Other Components**: Inherit base implementations

### Method Override Examples

```python
# Base landmark component
class EnvComponentWaypoint:
    def generate_waypoint_group(self, env_origins, num_reset, ...):
        # Default implementation
        pass

# Task-specific override
class DerivedEnvComponentWaypoint(EnvComponentWaypoint):
    def generate_waypoint_group(self, env_origins, num_reset, waypoint_offset, ...):
        # Task-specific implementation with different parameters
        # Can call super() to use base functionality where needed
        return self._generate_specific_pattern(...)
```

## Key Components

### Landmark Generation
- **Vector-based Offsets**: Deterministic spatial relationships using fixed [x, y] vectors
- **Group Regeneration**: Complete landmark group regeneration when spatial constraints fail
- **Bounds Checking**: Dynamic boundary validation using `room_size/3` limits
- **Collision Avoidance**: Automatic fallback to random positions when constraints cannot be met

### Robot Management
- **Collision-aware Reset**: Intelligent robot repositioning with safety margins
- **Multi-attempt Logic**: Up to 1000 attempts to find collision-free positions
- **Safety Margins**: Additional 1.0 unit clearance beyond collision tolerance

### Configuration
- **Room Size**: Configurable environment dimensions (default: 30.0 units)
- **Tolerances**: Separate approach and avoidance distance thresholds
- **Offset Vectors**: Customizable spatial relationships (default: [3.0, 1.0])
- **Safety Margins**: Configurable collision avoidance buffers


## Configuration Parameters

### Environment Settings
- `num_groups`: Number of landmark groups per episode
- `num_markers_per_group`: Landmarks per group (2 for single, 3 for rot_array)

### Tolerance Settings
- `approach_position_tolerance`: Distance threshold for goal reaching
- `avoid_position_tolerance`: Distance threshold for collision detection
- Safety margin: Additional 1.0 unit buffer for robot reset

### Landmark Configuration
- Default offset vector: [3.0, 1.0] for deterministic spatial relationships
- Fixed positions: Configurable per task variant
- Bounds: ±(room_size/3) for dynamic constraint validation

## Key Features

### Vector-based Landmark System
The environment uses deterministic vector offsets instead of random angular relationships:
```python
# Example: Target landmark offset from obstacle
offset_vector = [3.0, 1.0]  # Fixed [x, y] relationship
target_position = obstacle_position + offset_vector
```

### Group Regeneration Logic
When spatial constraints cannot be satisfied, the entire landmark group is regenerated:
```python
# Attempt to place valid landmark group
for attempt in range(max_group_attempts):
    # Generate all landmarks in group
    # Validate all spatial relationships
    # Accept only if all landmarks are valid
    if all_landmarks_valid:
        break
    # Otherwise, regenerate entire group
```

### Collision-aware Robot Reset
Intelligent robot repositioning prevents immediate collisions:
```python
# Reset robot with collision checking
for attempt in range(max_reset_attempts):
    robot_pose = reset_robot_randomly()
    if no_collision_with_landmarks(robot_pose):
        break
    # Otherwise, try new position
```
