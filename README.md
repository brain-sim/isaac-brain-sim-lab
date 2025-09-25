# Isaac Brain Sim Lab

A collection of NVIDIA Isaac Lab extensions, assets, and training utilities for brain-inspired navigation research. The repository bundles maze asset tooling, custom Gymnasium-compatible tasks, and multiple reinforcement learning pipelines for experimenting with simulated agents.

## Key Capabilities
- **Maze toolchain** (`brain_sim_ant_maze`, `brain_sim_assets`): Generate grid mazes from text or JSON specifications, convert them into Isaac-ready wall collections, and sample collision-free spawn points at runtime.
- **Isaac Lab tasks** (`brain_sim_tasks`): Landmark navigation, maze traversal, and legacy avoidance tasks registered with Gymnasium for both train and eval variants.
- **Training scripts** (`scripts/`): End-to-end runners for random/zero agents, SKRL-based trainers, and light-weight PPO/SAC/TD3 baselines in the `leanrl` suite.
- **Visualization & prototypes** (`tests/`): Raycasting validation, maze visualizations, and research prototypes for rapid iteration.

## Repository Layout
```
source/
  brain_sim_ant_maze/      # Maze authoring package (config, docs, examples)
  brain_sim_assets/        # USD asset loaders, runtime maze helper, configs
  brain_sim_tasks/         # Isaac Lab task registrations and Hydra configs
scripts/
  list_envs*.py            # Enumerate registered Gymnasium tasks
  random_agent.py          # Drive an environment with random actions
  skrl/                    # Training & evaluation using skrl runners
  leanrl/                  # Torch-based PPO/SAC/TD3 baselines and helpers
  try/try.py               # Interactive keyboard-controlled demo harness
tests/
  unit_tests/              # Example pytest suites (requires Isaac runtime)
  prototypes/, visualizations/  # Exploration scripts and plotting utilities
install_packages.sh        # Reinstall local brain_sim packages in editable mode
create_vscode_settings.sh  # Helper to point VS Code at IsaacLab sources
requirements.txt           # Full dependency lock (Isaac Sim 5.0.0.0 stack)
```

