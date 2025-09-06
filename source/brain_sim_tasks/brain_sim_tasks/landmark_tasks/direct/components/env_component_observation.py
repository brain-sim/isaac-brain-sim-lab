from __future__ import annotations

import torch


class EnvComponentObservation:
    """Component responsible for generating observations."""

    def __init__(self, env):
        self.env = env

    def post_env_init(self):

        self._position_error = torch.zeros(
            (self.env.num_envs), device=self.env.device, dtype=torch.float32
        )
        self._position_error_vector = torch.zeros(
            (self.env.num_envs, 2), device=self.env.device, dtype=torch.float32
        )
        self._previous_position_error = torch.zeros(
            (self.env.num_envs), device=self.env.device, dtype=torch.float32
        )
        self.target_heading_error = torch.zeros(
            (self.env.num_envs), device=self.env.device, dtype=torch.float32
        )

    def get_image_obs(self) -> torch.Tensor:
        """Get camera image observations."""
        image_obs = (
            self.env.env_component_robot.camera.data.output["rgb"]
            .float()
            .permute(0, 3, 1, 2)
            / 255.0
        )
        image_obs = image_obs.reshape(self.env.num_envs, -1)
        return image_obs

    def get_state_obs(self, image_obs) -> torch.Tensor:
        """Get state observations combining image and robot state."""
        return torch.cat(
            (
                image_obs,
                self.env.env_component_robot._actions[:, 0].unsqueeze(dim=1),
                self.env.env_component_robot._actions[:, 1].unsqueeze(dim=1),
                self.env.env_component_robot._actions[:, 2].unsqueeze(dim=1),
                self.env.env_component_robot.get_distance_to_walls().unsqueeze(dim=1),
            ),
            dim=-1,
        )

    def get_observations(self) -> dict:
        """Get full observations dictionary."""
        # Update position and heading errors
        current_target_positions = self.env.env_component_waypoint._target_positions[
            self.env.env_component_robot.robot._ALL_INDICES,
            self.env.env_component_waypoint._target_index,
        ]
        self._position_error_vector = (
            current_target_positions
            - self.env.env_component_robot.robot.data.root_pos_w[:, :2]
        )
        self._previous_position_error = self._position_error.clone()
        self._position_error = torch.norm(self._position_error_vector, dim=-1)

        # Calculate heading error
        heading = self.env.env_component_robot.robot.data.heading_w
        target_heading_w = torch.atan2(
            self.env.env_component_waypoint._target_positions[
                self.env.env_component_robot.robot._ALL_INDICES,
                self.env.env_component_waypoint._target_index,
                1,
            ]
            - self.env.env_component_robot.robot.data.root_link_pos_w[:, 1],
            self.env.env_component_waypoint._target_positions[
                self.env.env_component_robot.robot._ALL_INDICES,
                self.env.env_component_waypoint._target_index,
                0,
            ]
            - self.env.env_component_robot.robot.data.root_link_pos_w[:, 0],
        )
        self.target_heading_error = torch.atan2(
            torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading)
        )

        # Get image and state observations
        image_obs = self.get_image_obs()
        state_obs = self.get_state_obs(image_obs)
        state_obs = torch.nan_to_num(state_obs, posinf=0.0, neginf=0.0)

        if torch.any(state_obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        return {"policy": state_obs}

    def reset(self, env_ids):
        """Reset observations for specified environments."""
        # Reset position and heading errors
        current_target_positions = self.env.env_component_waypoint._target_positions[
            self.env.env_component_robot.robot._ALL_INDICES,
            self.env.env_component_waypoint._target_index,
        ]
        self._position_error_vector = (
            current_target_positions[:, :2]
            - self.env.env_component_robot.robot.data.root_pos_w[:, :2]
        )
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

        # Reset heading error
        heading = self.env.env_component_robot.robot.data.heading_w[:]
        target_heading_w = torch.atan2(
            self.env.env_component_waypoint._target_positions[:, 0, 1]
            - self.env.env_component_robot.robot.data.root_pos_w[:, 1],
            self.env.env_component_waypoint._target_positions[:, 0, 0]
            - self.env.env_component_robot.robot.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(
            torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading)
        )
        self._previous_heading_error = self._heading_error.clone()
