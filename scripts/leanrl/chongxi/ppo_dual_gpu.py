# Example modifications for dual-GPU setup
# This shows the key changes needed in your ppo_continuous_action.py

import torch

# In ExperimentArgs class, add:
class ExperimentArgs:
    # ... existing args ...
    physics_device: str = "cuda:0"  # GPU for physics simulation
    nn_device: str = "cuda:1"      # GPU for neural network
    
def main(args):
    # ... existing setup ...
    
    # Set devices
    physics_device = torch.device(args.physics_device)
    nn_device = torch.device(args.nn_device)
    
    # Environment uses physics GPU
    envs = make_isaaclab_env(
        args.task,
        args.physics_device,  # Physics on GPU 0
        args.num_envs,
        # ... other args
    )()
    
    # Agent on NN GPU
    agent = CNNPPOAgent(n_obs, n_act, img_size=args.img_size)
    agent.to(nn_device)  # NN on GPU 1
    
    # Storage on NN GPU (where training happens)
    obs = torch.zeros((args.num_steps, args.num_envs, n_obs)).to(nn_device)
    actions = torch.zeros((args.num_steps, args.num_envs, n_act)).to(nn_device)
    # ... other storage on nn_device
    
    # During rollout:
    for step in range(args.num_steps):
        # Transfer observations from physics GPU to NN GPU
        next_obs_nn = next_obs.to(nn_device, non_blocking=True)
        
        # NN inference on GPU 1
        action, logprob, _, value, mu, sigma = (
            inference_agent.get_action_and_value(next_obs_nn)
        )
        
        # Transfer actions back to physics GPU
        action_physics = action.to(physics_device, non_blocking=True)
        
        # Physics simulation on GPU 0
        next_obs, reward, next_done, infos = envs.step(action_physics)
        
        # Transfer rewards/dones to NN GPU for storage
        rewards[step].copy_(reward.to(nn_device, non_blocking=True).detach().view(-1, 1))
        dones[step].copy_(next_done.to(nn_device, non_blocking=True).detach().view(-1, 1))