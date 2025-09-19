import torch
from isaaclab.markers import VisualizationMarkers

from ...components.env_component_character import EnvComponentCharacter

def quat_from_angle_axis(angle, axis):
    """Create quaternion from angle and axis."""
    half_angle = angle * 0.5
    sin_half = torch.sin(half_angle)
    cos_half = torch.cos(half_angle)
    
    # Ensure axis is normalized
    axis_norm = axis / torch.norm(axis, dim=-1, keepdim=True)
    
    # Create quaternion [w, x, y, z]
    quat = torch.zeros((*angle.shape, 4), device=angle.device, dtype=angle.dtype)
    quat[..., 0] = cos_half  # w
    quat[..., 1] = sin_half * axis_norm[..., 0]  # x
    quat[..., 2] = sin_half * axis_norm[..., 1]  # y  
    quat[..., 3] = sin_half * axis_norm[..., 2]  # z
    
    return quat

class DerivedEnvComponentCharacter(EnvComponentCharacter):
    """Component responsible for character generation and management."""

    def __init__(self, env):
        self.env = env
        self.characters = None

    def post_env_init(self):
        # Initialize character tracking tensors (x, y, z, qw, qx, qy, qz for full pose)
        self._characters_poses = torch.zeros(
            (self.env.num_envs, self.env.cfg.num_characters, 3 + 4), 
            device=self.env.device, 
            dtype=torch.float32
        )
        # Character type indices (0-4 for 5 different character types)
        self._characters_index = torch.zeros(
            (self.env.num_envs, self.env.cfg.num_characters), 
            device=self.env.device, 
            dtype=torch.long
        )
        
        # Movement parameters
        self.movement_speed = getattr(self.env.cfg, 'character_movement_speed', 0.1)  # units per step
        self.movement_probability = getattr(self.env.cfg, 'character_movement_probability', 0.3)  # chance to move each step
        self.rotation_speed = getattr(self.env.cfg, 'character_rotation_speed', 0.2)  # radians per step

    def setup(self):
        self.characters = VisualizationMarkers(self.env.cfg.character_cfg)

    def generate_random_character(self, env_origins, num_reset, robot_xy=None, min_distance=2.0):
        """Generate random valid character positions and orientations."""
        max_attempts = 1000
        placed = torch.zeros(num_reset, dtype=torch.bool, device=self.env.device)
        character_positions = torch.zeros((num_reset, 3), device=self.env.device)  # x, y, z
        character_orientations = torch.zeros((num_reset, 4), device=self.env.device)  # quaternion
        
        for _ in range(max_attempts):
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()

            if num_unplaced == 0:
                break

            # Get random valid positions from wall config
            random_positions = self.env.cfg.wall_config.get_random_valid_positions(
                num_unplaced, device=self.env.device
            )

            tx = random_positions[:, 0]
            ty = random_positions[:, 1]

            unplaced_origins = env_origins[unplaced_mask]
            candidate_positions = torch.stack([tx, ty], dim=1) + unplaced_origins

            # Check minimum distance from robot if provided
            if robot_xy is not None:
                unplaced_robot_pos = robot_xy[unplaced_mask]
                robot_distances = torch.norm(
                    candidate_positions - unplaced_robot_pos, dim=1
                )
                robot_valid = robot_distances >= min_distance
            else:
                robot_valid = torch.ones(
                    num_unplaced, dtype=torch.bool, device=self.env.device
                )

            # Check bounds validity
            relative_positions = candidate_positions - unplaced_origins
            bound_limit = self.env.cfg.room_size / 3.0
            bounds_valid = (
                (relative_positions[:, 0] >= -bound_limit)
                & (relative_positions[:, 0] <= bound_limit)
                & (relative_positions[:, 1] >= -bound_limit)
                & (relative_positions[:, 1] <= bound_limit)
            )

            valid_mask = robot_valid & bounds_valid
            valid_indices = torch.where(unplaced_mask)[0][valid_mask]
            
            if len(valid_indices) > 0:
                character_positions[valid_indices, :2] = candidate_positions[valid_mask]  # x, y
                character_positions[valid_indices, 2] = 0.0  # z = ground level
                
                # Generate random orientations (quaternions) for valid characters
                # Random yaw angles
                random_yaw = torch.rand(len(valid_indices), device=self.env.device) * 2 * torch.pi
                
                # Convert yaw to quaternion (w, x, y, z) format
                half_yaw = random_yaw / 2
                character_orientations[valid_indices, 0] = torch.cos(half_yaw)  # w
                character_orientations[valid_indices, 1] = 0.0  # x
                character_orientations[valid_indices, 2] = 0.0  # y
                character_orientations[valid_indices, 3] = torch.sin(half_yaw)  # z
                
                placed[valid_indices] = True

        # Fallback for any unplaced characters
        if not placed.all():
            unplaced_mask = ~placed
            num_unplaced = unplaced_mask.sum().item()
            
            random_positions = self.env.cfg.wall_config.get_random_valid_positions(
                num_unplaced, device=self.env.device
            )
            unplaced_origins = env_origins[unplaced_mask]
            fallback_positions = random_positions[:, :2] + unplaced_origins
            character_positions[unplaced_mask, :2] = fallback_positions  # x, y
            character_positions[unplaced_mask, 2] = 0.0  # z = ground level
            
            # Random orientations for fallback characters
            random_yaw = torch.rand(num_unplaced, device=self.env.device) * 2 * torch.pi
            half_yaw = random_yaw / 2
            character_orientations[unplaced_mask, 0] = torch.cos(half_yaw)  # w
            character_orientations[unplaced_mask, 1] = 0.0  # x
            character_orientations[unplaced_mask, 2] = 0.0  # y
            character_orientations[unplaced_mask, 3] = torch.sin(half_yaw)  # z

        return character_positions, character_orientations

    def generate_characters(self, env_ids, robot_poses):
        """Generate characters for reset environments."""
        num_reset = len(env_ids)
        num_characters = self.env.cfg.num_characters
        env_origins = self.env.scene.env_origins[env_ids, :2]
        robot_xy = robot_poses[:, :2]

        all_character_positions = torch.zeros(
            (num_reset, num_characters, 3), device=self.env.device
        )
        all_character_orientations = torch.zeros(
            (num_reset, num_characters, 4), device=self.env.device
        )
        all_character_indices = torch.zeros(
            (num_reset, num_characters), device=self.env.device, dtype=torch.long
        )

        # Generate each character individually at different positions
        for char_idx in range(num_characters):
            char_positions, char_orientations = self.generate_random_character(
                env_origins, num_reset, robot_xy, min_distance=2.0
            )
            
            all_character_positions[:, char_idx, :] = char_positions
            all_character_orientations[:, char_idx, :] = char_orientations
            
            # Assign random character type (0-4 for 5 different types)
            all_character_indices[:, char_idx] = torch.randint(
                0, 5, (num_reset,), device=self.env.device
            )

        return all_character_positions, all_character_orientations, all_character_indices

    def step(self, dt):
        # Create oscillating yaw rotation
        yaw = torch.linspace(0, 2 * torch.pi, self.env.num_envs * self.env.cfg.num_characters, device=self.env.device)
        yaw = yaw + (self.env.common_step_counter * 0.1)  # Rotate slowly over time
        yaw = yaw.view(self.env.num_envs, self.env.cfg.num_characters)
        
        # Create quaternions from z-axis rotation
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.env.device)
        marker_orientations = quat_from_angle_axis(yaw, z_axis)
        
        # Add oscillating translations in random directions
        # Generate random directions for each character (only once at initialization)
        if not hasattr(self, '_oscillation_directions'):
            # Random direction angles for each character
            random_angles = torch.rand(self.env.num_envs, self.env.cfg.num_characters, device=self.env.device) * 2 * torch.pi
            self._oscillation_directions = torch.stack([
                torch.cos(random_angles),
                torch.sin(random_angles)
            ], dim=-1)  # Shape: [num_envs, num_characters, 2]
        
        # Create oscillating movement
        oscillation_amplitude = 0.05  # Amplitude of oscillation in units
        oscillation_frequency = 0.1  # Frequency of oscillation
        time_factor = torch.tensor(self.env.common_step_counter * oscillation_frequency, device=self.env.device)
        
        # Calculate oscillation offset
        oscillation_offset = torch.sin(time_factor) * oscillation_amplitude
        oscillation_offset = oscillation_offset.unsqueeze(-1)  # Add dimension for x,y
        
        # Apply directional oscillation to positions
        oscillation_movement = self._oscillation_directions * oscillation_offset
        
        # Get base positions from stored poses and add oscillation
        base_positions = self._characters_poses[:, :, :3].clone()
        base_positions[:, :, :2] += oscillation_movement  # Only affect x,y, keep z=0
        
        # Update stored positions and orientations
        self._characters_poses[:, :, :3] = base_positions
        self._characters_poses[:, :, 3:] = marker_orientations
        
        # Prepare data for visualization
        marker_positions = base_positions.view(-1, 3)  # x, y, z
        marker_orientations_flat = marker_orientations.view(-1, 4)  # quaternions
        marker_indices = self._characters_index.view(-1).tolist()
        
        # Visualize
        self.characters.visualize(
            translations=marker_positions,
            orientations=marker_orientations_flat,
            marker_indices=marker_indices
        )
        
        self.update_character_visualization()

    def update_character_visualization(self):
        marker_indices = self._characters_index.view(-1).tolist()
        marker_positions = self._characters_poses[:, :, :3].view(-1, 3)  # x, y, z
        marker_orientations = self._characters_poses[:, :, 3:].view(-1, 4)  # quaternions
        
        self.characters.visualize(
            translations=marker_positions,
            orientations=marker_orientations,
            marker_indices=marker_indices
        )

    def reset(self, env_ids: torch.Tensor, robot_poses):
        self._characters_poses[env_ids, :, :] = 0.0
        self._characters_index[env_ids, :] = 0

        char_positions, char_orientations, char_indices = self.generate_characters(env_ids, robot_poses)
        
        self._characters_poses[env_ids, :, :3] = char_positions  # x, y, z (z is already 0.0)
        self._characters_poses[env_ids, :, 3:] = char_orientations  # quaternions
        
        self._characters_index[env_ids, :] = char_indices

        self.update_character_visualization()