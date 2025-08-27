import io

import torch
from isaaclab.utils.math import matrix_from_quat

class bsSpotLowLevelPolicy:

    def __init__(self, policy_file_path: str):
        with open(policy_file_path, "rb") as f:
            self.policy = torch.jit.load(io.BytesIO(f.read()))

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Run the policy on the given batch of observations.
        Args:
            observation (torch.Tensor): [num_envs, obs_dim] observation input.
        Returns:
            torch.Tensor: [num_envs, action_dim] policy output (actions).
        """
        # Ensure policy is on the same device as observation
        if next(self.policy.parameters()).device != observation.device:
            self.policy = self.policy.to(observation.device)
        with torch.no_grad():
            obs_tensor = observation.float()
            action = self.policy(obs_tensor).detach()
        return action

    def compute_observation(self, *args, **kwargs) -> torch.Tensor:
        """
        Abstract method to compute the observation vector for the policy.
        This method should be implemented by subclasses to define how to
        transform robot state into the observation format expected by the policy.
        
        Returns:
            torch.Tensor: [num_envs, obs_dim] observation vector for the policy.
        """
        raise NotImplementedError(
            "Subclasses must implement compute_observation method."
        )

    def get_action(self, *args, **kwargs) -> torch.Tensor:
        """
        Generic method to compute action from robot state.
        Computes observation using compute_observation and then runs the policy.
        
        Returns:
            torch.Tensor: [num_envs, action_dim] policy output (actions).
        """
        obs = self.compute_observation(*args, **kwargs)
        actions = self(obs)
        return actions

#####################################################

"""
Derived classes below
"""

class bsSpotLowLevelPolicyVanilla(bsSpotLowLevelPolicy):

    def compute_observation(
        self,
        lin_vel_I,
        ang_vel_I,
        q_IB,
        command,
        previous_action,
        default_pos,
        joint_pos,
        joint_vel,
    ) -> torch.Tensor:
        """
        Compute the observation vector for the policy for all environments (vectorized).
        Args:
            lin_vel_I: [num_envs, 3] torch.Tensor
            ang_vel_I: [num_envs, 3] torch.Tensor
            q_IB: [num_envs, 4] torch.Tensor (quaternion)
            command: [num_envs, 3] torch.Tensor
            previous_action: [num_envs, 12] torch.Tensor
            default_pos: [num_envs, 12] torch.Tensor
            joint_pos: [num_envs, 12] torch.Tensor
            joint_vel: [num_envs, 12] torch.Tensor
        Returns:
            obs: [num_envs, 48] torch.Tensor
        """
        # Rotation matrices: [num_envs, 3, 3]
        R_IB = matrix_from_quat(q_IB)  # [num_envs, 3, 3]
        R_BI = R_IB.transpose(1, 2)  # [num_envs, 3, 3]
        lin_vel_b = torch.bmm(R_BI, lin_vel_I.unsqueeze(-1)).squeeze(
            -1
        )  # [num_envs, 3]
        ang_vel_b = torch.bmm(R_BI, ang_vel_I.unsqueeze(-1)).squeeze(
            -1
        )  # [num_envs, 3]
        gravity_vec = torch.tensor(
            [0.0, 0.0, -1], device=lin_vel_I.device, dtype=lin_vel_I.dtype
        ).reshape(1, 3, 1)
        gravity_b = torch.bmm(
            R_BI, gravity_vec.expand(lin_vel_I.shape[0], 3, 1)
        ).squeeze(-1)
        obs = torch.zeros(
            (lin_vel_I.shape[0], 48), device=lin_vel_I.device, dtype=lin_vel_I.dtype
        )
        obs[:, 0:3] = lin_vel_b
        obs[:, 3:6] = ang_vel_b
        obs[:, 6:9] = gravity_b
        obs[:, 9:12] = command
        obs[:, 12:24] = joint_pos - default_pos
        obs[:, 24:36] = joint_vel
        obs[:, 36:48] = previous_action
        return obs

class bsSpotLowLevelPolicyRough(bsSpotLowLevelPolicy):

    def compute_observation(
        self,
        base_lin_vel,
        base_ang_vel,
        projected_gravity,
        command,
        previous_action,
        default_pos,
        joint_pos,
        joint_vel,
    ) -> torch.Tensor:
        """
        Compute the observation vector for the policy for all environments (vectorized).
        Args:
            lin_vel_I: [num_envs, 3] torch.Tensor
            ang_vel_I: [num_envs, 3] torch.Tensor
            q_IB: [num_envs, 4] torch.Tensor (quaternion)
            command: [num_envs, 3] torch.Tensor
            previous_action: [num_envs, 12] torch.Tensor
            default_pos: [num_envs, 12] torch.Tensor
            joint_pos: [num_envs, 12] torch.Tensor
            joint_vel: [num_envs, 12] torch.Tensor
        Returns:
            obs: [num_envs, 48] torch.Tensor
        """
        obs = torch.zeros(
            (base_lin_vel.shape[0], 48), device=base_lin_vel.device, dtype=base_lin_vel.dtype
        )
        obs[:, 0:3] = base_lin_vel
        obs[:, 3:6] = base_ang_vel
        obs[:, 6:9] = projected_gravity
        obs[:, 9:12] = command
        obs[:, 12:24] = joint_pos - default_pos
        obs[:, 24:36] = joint_vel
        obs[:, 36:48] = previous_action
        return obs

class bsSpotLowLevelPolicyRoughHeight(bsSpotLowLevelPolicy):

    def compute_observation(
        self,
        base_lin_vel,
        base_ang_vel,
        projected_gravity,
        command,
        previous_action,
        default_pos,
        joint_pos,
        joint_vel,
        height_obs,
    ) -> torch.Tensor:
        """
        Compute the observation vector for the policy for all environments (vectorized).
        Args:
            lin_vel_I: [num_envs, 3] torch.Tensor
            ang_vel_I: [num_envs, 3] torch.Tensor
            q_IB: [num_envs, 4] torch.Tensor (quaternion)
            command: [num_envs, 3] torch.Tensor
            previous_action: [num_envs, 12] torch.Tensor
            default_pos: [num_envs, 12] torch.Tensor
            joint_pos: [num_envs, 12] torch.Tensor
            joint_vel: [num_envs, 12] torch.Tensor
        Returns:
            obs: [num_envs, 48] torch.Tensor
        """
        obs = torch.zeros(
            (base_lin_vel.shape[0], 57), device=base_lin_vel.device, dtype=base_lin_vel.dtype
        )
        obs[:, 0:3] = base_lin_vel
        obs[:, 3:6] = base_ang_vel
        obs[:, 6:9] = projected_gravity
        obs[:, 9:12] = command
        obs[:, 12:24] = joint_pos - default_pos
        obs[:, 24:36] = joint_vel
        obs[:, 36:48] = previous_action
        obs[:, 48:57] = height_obs
        return obs