import torch
from isaaclab.assets import RigidObject


class DerivedEnvComponentCharacter:
    """Component responsible for character generation and management."""

    def __init__(self, env):
        self.env = env
        self.characters = {}

    def setup(self):
        """Setup character rigid objects."""
        char_cfgs = self.env.cfg.character_config.create_character_configuration()
        for character_name, character_cfg in char_cfgs.items():
            self.characters[character_name] = RigidObject(character_cfg)

    def post_env_init(self):
        """Post-initialization setup."""
        pass

    def reset(self, env_ids):
        """Reset characters for specified environments."""
        if not self.characters:
            return
            
        for character_name, character_obj in self.characters.items():
            if hasattr(character_obj, 'reset'):
                character_obj.reset(env_ids)

    # def step(self):
    #     """Update character positions and apply to simulation."""
    #     if not self.characters:
    #         return
    #     print("Updating character positions...")
    #     dt = self.env.step_dt
    #     max_velocity = 2.0  # meters per second
    #     max_displacement = max_velocity * dt
        
    #     num_characters = self.env.cfg.character_config.num_characters
    #     boundary_limit = self.env.cfg.character_config.boundary_limit
        
    #     character_poses = self.env.cfg.character_config._character_poses.clone()
    #     # Apply random offsets to all characters at once
    #     dx = (torch.rand(num_characters, device=self.env.device) - 0.5) * 2 * max_displacement
    #     dy = (torch.rand(num_characters, device=self.env.device) - 0.5) * 2 * max_displacement
        
    #     new_x = character_poses[:, 0] + dx
    #     new_y = character_poses[:, 1] + dy
        
    #     # Clamp all positions to boundary limits at once
    #     character_poses[:, 0] = torch.clamp(new_x, -boundary_limit, boundary_limit)
    #     character_poses[:, 1] = torch.clamp(new_y, -boundary_limit, boundary_limit)
        
    #     # Update with new poses
    #     self.env.cfg.character_config.update_character_poses(character_poses)
    #     num_envs = self.env.num_envs
    #     env_ids = torch.arange(num_envs, device=self.env.device)
        
    #     for i, (character_name, character_obj) in enumerate(self.characters.items()):
    #         if i < num_characters:
    #             character_pose = character_poses[i:i+1].repeat(num_envs, 1)
    #             # Vectorized addition of environment origins
    #             character_pose[:, :3] += self.env.scene.env_origins
    #             print(f"Writing pose for {character_name}: {character_pose}")
                
    #             character_obj.write_root_pose_to_sim(character_pose, env_ids)


    def step(self):
        """Update character positions and apply to simulation."""
        if not self.characters:
            return

        dt = self.env.step_dt
        max_velocity = 2.0
        max_displacement = max_velocity * dt

        num_characters = self.env.cfg.character_config.num_characters
        boundary_limit = self.env.cfg.character_config.boundary_limit

        # (N_char, ?)  Your config array. We'll treat first two columns as XY.
        character_poses = self.env.cfg.character_config._character_poses.clone()

        # Random planar increments
        dx = (torch.rand(num_characters, device=self.env.device) - 0.5) * 2 * max_displacement
        dy = (torch.rand(num_characters, device=self.env.device) - 0.5) * 2 * max_displacement

        new_x = character_poses[:, 0] + dx
        new_y = character_poses[:, 1] + dy
        character_poses[:, 0] = torch.clamp(new_x, -boundary_limit, boundary_limit)
        character_poses[:, 1] = torch.clamp(new_y, -boundary_limit, boundary_limit)

        # Persist to config
        self.env.cfg.character_config.update_character_poses(character_poses)

        # Prepare world-frame poses for each env
        num_envs = self.env.num_envs
        env_ids = torch.arange(num_envs, device=self.env.device, dtype=torch.long)
        env_origins = self.env.scene.env_origins  # (num_envs, 3)
        # ^ Per Isaac Lab, root poses you write must be in the simulation world frame (not parent Xform),
        #   so we add env origins here. :contentReference[oaicite:2]{index=2}

        # Choose a fixed z height (or pull z from your config if you have it)
        z0 = 0.0
        # Identity orientation (facing forward) in wxyz
        quat_wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.env.device).repeat(num_envs, 1)

        for i, (character_name, character_obj) in enumerate(self.characters.items()):
            if i >= num_characters:
                break

            # Build per-env XYZ in world frame: (num_envs, 3)
            pos_rel = torch.stack(
                [character_poses[i, 0].repeat(num_envs),
                character_poses[i, 1].repeat(num_envs),
                torch.full((num_envs,), z0, device=self.env.device)],
                dim=-1
            )
            pos_world = pos_rel + env_origins

            # 7D root pose (num_envs, 7) in world frame
            root_pose = torch.cat([pos_world, quat_wxyz], dim=-1)

            print(f"Writing pose for {character_name}: {root_pose}")
            # Apply to simulation and zero velocities
            character_obj.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            root_velocity = torch.zeros((num_envs, 6), device=self.env.device)
            print(f"Writing velocity for {character_name}: {root_velocity}")
            character_obj.write_root_velocity_to_sim(
                root_velocity, env_ids=env_ids
            )
            # Refresh internal buffers (matches the tutorial pattern)
            character_obj.reset(env_ids)

            # Optional debug
            # print(f"[{character_name}] pose7 world ->", root_pose[:3])