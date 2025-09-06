from __future__ import annotations

import torch
from isaaclab.utils import math


class EnvComponentTermination:
    """Component responsible for termination conditions."""
    
    def __init__(self, env):
        self.env = env
        

    def post_env_init(self):

        self._vehicle_flipped = torch.zeros((self.env.num_envs), device=self.env.device, dtype=torch.bool)
        # Load termination parameters from config
        self.termination_on_avoid_goal_collision = self.env.cfg.termination_on_avoid_goal_collision
        self.termination_on_goal_reached = self.env.cfg.termination_on_goal_reached
        self.termination_on_vehicle_flip = self.env.cfg.termination_on_vehicle_flip
        self.termination_on_stuck = self.env.cfg.termination_on_stuck

    def check_flipped(self) -> torch.Tensor:
        """Check if robot is flipped over."""
        robot_quat = self.env.env_component_robot.robot.data.root_quat_w
        local_up = torch.tensor([0.0, 0.0, 1.0], device=self.env.device)
        world_up_vector = math.quat_apply(robot_quat, local_up.repeat(self.env.num_envs, 1))
        world_up = torch.tensor([0.0, 0.0, 1.0], device=self.env.device)
        up_dot_product = torch.sum(world_up_vector * world_up, dim=-1)
        up_angle = torch.abs(
            torch.acos(torch.clamp(up_dot_product, -1.0, 1.0))
        )
        flipped = up_angle > (torch.pi / 3)
        return flipped

    def check_stuck_termination(self, max_steps: int = 300) -> torch.Tensor:
        """Check if robot is stuck for too long."""
        steps_since_goal = (
            self.env.episode_length_buf - self.env.env_component_reward._previous_waypoint_reached_step
        )
        stuck_too_long = steps_since_goal > max_steps
        return stuck_too_long

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Get termination and timeout conditions."""
        time_outs = self.env.episode_length_buf > self.env.max_episode_length_buf
        terminated = torch.zeros_like(time_outs)
        
        # Check vehicle flip termination
        if hasattr(self, "termination_on_vehicle_flip") and self.termination_on_vehicle_flip:
            self._vehicle_flipped = self.check_flipped()
            terminated = self._vehicle_flipped
            
        # Check goal reached termination
        if hasattr(self, "termination_on_goal_reached") and self.termination_on_goal_reached:
            terminated |= self.env.task_completed

        # Prepare termination info
        termination_infos = {
            "Episode_Termination/flipped": self._vehicle_flipped.float().mean().item(),
            "Episode_Termination/time_outs": time_outs.float().mean().item(),
            "Episode_Termination/task_completed": self.env.task_completed.float().mean().item(),
        }
        
        # Check stuck termination
        if hasattr(self, "termination_on_stuck") and self.termination_on_stuck:
            stuck_termination = self.check_stuck_termination()
            time_outs |= stuck_termination
            termination_infos["Episode_Termination/stuck_termination"] = (
                stuck_termination.float().mean().item()
            )

        # Check avoid goal collision termination
        if hasattr(self, "termination_on_avoid_goal_collision") and self.termination_on_avoid_goal_collision:
            avoid_goal_termination = self.env.env_component_reward.check_avoid_goal_collision()
            terminated |= avoid_goal_termination
            termination_infos["Episode_Termination/avoid_goal_termination"] = (
                avoid_goal_termination.float().mean().item()
            )

        termination_infos["Episode_Termination/terminated"] = (
            terminated.float().mean().item()
        )
        
        return terminated, time_outs, termination_infos
