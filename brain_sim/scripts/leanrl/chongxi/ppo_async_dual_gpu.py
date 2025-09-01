# Advanced asynchronous dual-GPU implementation with pipelining
import torch
import torch.multiprocessing as mp
from queue import Queue
import threading

class AsyncDualGPURollout:
    """
    Asynchronous rollout with physics and NN on separate GPUs.
    Uses pipelining to overlap computation.
    """
    def __init__(self, envs, agent, args):
        self.envs = envs
        self.agent = agent
        self.args = args
        
        # Separate devices
        self.physics_device = torch.device(args.physics_device)
        self.nn_device = torch.device(args.nn_device)
        
        # Queues for async communication
        self.obs_queue = Queue(maxsize=2)  # Physics -> NN
        self.action_queue = Queue(maxsize=2)  # NN -> Physics
        
        # Pinned memory for faster GPU transfers
        self.use_pinned = True
        
    def physics_worker(self, num_steps):
        """Runs on physics GPU, handles environment stepping"""
        torch.cuda.set_device(self.physics_device)
        
        # Initial reset
        obs, _ = self.envs.reset()
        self.obs_queue.put(obs.clone())
        
        for step in range(num_steps):
            # Get action from NN thread
            action = self.action_queue.get()
            action = action.to(self.physics_device, non_blocking=True)
            
            # Step physics
            obs, reward, done, info = self.envs.step(action)
            
            # Send observation to NN thread
            if self.use_pinned:
                obs_pinned = obs.pin_memory()
                self.obs_queue.put((obs_pinned, reward, done, info))
            else:
                self.obs_queue.put((obs.clone(), reward, done, info))
    
    def nn_worker(self, storage, num_steps):
        """Runs on NN GPU, handles neural network inference"""
        torch.cuda.set_device(self.nn_device)
        
        # Get initial observation
        next_obs = self.obs_queue.get()
        next_obs = next_obs.to(self.nn_device, non_blocking=True)
        
        for step in range(num_steps):
            # NN inference
            with torch.no_grad():
                action, logprob, _, value, mu, sigma = (
                    self.agent.get_action_and_value(next_obs)
                )
            
            # Store for training
            storage['obs'][step].copy_(next_obs)
            storage['actions'][step].copy_(action.detach())
            storage['logprobs'][step].copy_(logprob.detach().view(-1, 1))
            storage['values'][step].copy_(value.detach().view(-1, 1))
            
            # Send action to physics thread
            self.action_queue.put(action.detach())
            
            # Get next observation from physics
            if step < num_steps - 1:
                next_obs, reward, done, info = self.obs_queue.get()
                next_obs = next_obs.to(self.nn_device, non_blocking=True)
                
                # Store rewards and dones
                storage['rewards'][step].copy_(reward.to(self.nn_device).detach().view(-1, 1))
                storage['dones'][step].copy_(done.to(self.nn_device).detach().view(-1, 1))
    
    def rollout(self, storage, num_steps):
        """Execute parallel rollout with physics and NN on separate GPUs"""
        # Start physics thread
        physics_thread = threading.Thread(
            target=self.physics_worker, 
            args=(num_steps,)
        )
        physics_thread.start()
        
        # Run NN worker in main thread
        self.nn_worker(storage, num_steps)
        
        # Wait for physics to complete
        physics_thread.join()
        
        return storage


# Usage in main training loop:
def main_async(args):
    # ... setup code ...
    
    # Create async rollout handler
    async_rollout = AsyncDualGPURollout(envs, agent, args)
    
    # Storage on NN GPU
    storage = {
        'obs': torch.zeros((args.num_steps, args.num_envs, n_obs)).to(args.nn_device),
        'actions': torch.zeros((args.num_steps, args.num_envs, n_act)).to(args.nn_device),
        'logprobs': torch.zeros((args.num_steps, args.num_envs, 1)).to(args.nn_device),
        'rewards': torch.zeros((args.num_steps, args.num_envs, 1)).to(args.nn_device),
        'dones': torch.zeros((args.num_steps, args.num_envs, 1)).to(args.nn_device).byte(),
        'values': torch.zeros((args.num_steps, args.num_envs, 1)).to(args.nn_device),
    }
    
    for iteration in range(args.num_iterations):
        # Asynchronous rollout with pipelining
        storage = async_rollout.rollout(storage, args.num_steps)
        
        # Training happens on NN GPU (no change needed)
        # ... PPO update code ...