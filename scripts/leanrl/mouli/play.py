# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import sys
from dataclasses import asdict

import gymnasium as gym
import imageio  # Added for video writing
import numpy as np
import torch
import tqdm
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaaclab.utils import configclass
from models import CNNPPOAgent
from termcolor import colored
from utils import load_args, print_dict  # add load_args import

### TODO : Make play callable while training and after training.
### Solution - Use ManagerBasedRL or multi threading or multiprocessing to run train and eval.

### TODO : get the name of the video file from the environment.
### Solution - Use the environment's metadata to get the name of the video file (do it elegantly).


@configclass
class EnvArgs:
    task: str = "Brain-Sim-Spot-Nav-Avoid-v0"
    """the id of the environment"""
    env_cfg_entry_point: str = "env_cfg_entry_point"
    """the entry point of the environment configuration"""
    num_envs: int = 4
    """the number of parallel environments to simulate"""
    seed: int = 1
    """seed of the environment"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    video: bool = False
    """record videos during training"""
    video_length: int = 200
    """length of the recorded video (in steps)"""
    video_interval: int = 2000
    """interval between video recordings (in steps)"""
    disable_fabric: bool = False
    """disable fabric and use USD I/O operations"""
    distributed: bool = False
    """run training with multiple GPUs or nodes"""
    headless: bool = False
    """run training in headless mode"""
    enable_cameras: bool = False
    """enable cameras to record sensor inputs."""
    # renderer: str = "PathTracing"  # "PathTracing" or "RayTracedLighting"
    # """Renderer to use."""
    # samples_per_pixel_per_frame: int = 1
    # """Number of samples per pixel per frame."""


@configclass
class ExperimentArgs:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "cuda:0"
    """device to use for training"""

    checkpoint_path: str = (
        "/home/user/cognitiverl/wandb/run-20250701_001447-0ldtuxzk/files/checkpoints/ckpt_4915200.pt"
    )
    """path to the checkpoint to load"""
    num_eval_envs: int = 32
    """number of environments to run for evaluation/play."""
    num_eval_env_steps: int = 1000
    """number of steps to run for evaluation/play."""
    agent: str = "CNNPPOAgent"


@configclass
class Args(ExperimentArgs, EnvArgs):
    pass


def launch_app(args):
    from argparse import Namespace

    app_launcher = AppLauncher(Namespace(**asdict(args)))
    return app_launcher.app


def get_args():
    return load_args(Args)


try:
    from isaaclab.app import AppLauncher

    args = get_args()
    simulation_app = launch_app(args)

except ImportError:
    raise ImportError("Isaac Lab is not installed. Please install it first.")


def make_isaaclab_env(
    task,
    device,
    num_envs,
    capture_video,
    disable_fabric,
    log_dir=None,
    video_length=200,
    *args,
    **kwargs,
):
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.torchrl import (
        IsaacLabVecEnvWrapper,
    )
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import brain_sim_tasks  # noqa: F401

    def thunk():
        cfg = parse_env_cfg(
            task, device, num_envs=num_envs, use_fabric=not disable_fabric
        )
        env = gym.make(
            task,
            cfg=cfg,
            render_mode=(
                "rgb_array" if (capture_video and log_dir is not None) else None
            ),
            play_mode=True,
        )
        if capture_video and log_dir is not None:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": video_length,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        env = IsaacLabVecEnvWrapper(env)
        return env

    return thunk


def main(args):
    print_dict(args)
    run_name = f"{args.task}__{args.exp_name}"
    run = wandb.init(
        project="play",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )
    run_dir = run.dir

    eval_envs = make_isaaclab_env(
        args.task,
        args.device,
        args.num_eval_envs,
        args.capture_video,
        args.disable_fabric,
        log_dir=run_dir,
    )()

    # Set camera eye position to (0, 0, 45) looking at origin
    print("📷 Setting camera eye position to (0, 0, 45)")
    eval_envs.unwrapped.sim.set_camera_view(eye=[0, 0, 60], target=[0.0, 0.0, 0.0])

    n_obs = int(np.prod(eval_envs.observation_space.shape[1:]))
    n_act = int(np.prod(eval_envs.action_space.shape[1:]))
    assert isinstance(
        eval_envs.action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    AGENT_LOOKUP = {
        "CNNPPOAgent": CNNPPOAgent,
    }

    agent_class = AGENT_LOOKUP[args.agent]
    agent = agent_class(n_obs, n_act, img_size=eval_envs.unwrapped.cfg.img_size)
    print(
        colored(
            "[INFO] : Loading agent from {args.checkpoint_path}",
            "green",
            attrs=["bold"],
        )
    )
    checkpoint = torch.load(
        args.checkpoint_path, map_location="cpu", weights_only=False
    )
    agent.load_state_dict(checkpoint["ema_model_state_dict"])
    device = (
        torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    )
    agent.to(device)
    agent.eval()

    @torch.no_grad()
    def run_play(play_agent, mode="play"):
        play_agent.eval()
        # environment evaluation
        eval_envs.seed(seed=args.seed)
        obs, _ = eval_envs.reset()
        step, global_step = 0, 0
        done = torch.zeros(args.num_eval_envs, dtype=torch.bool)

        # Initialize storage for observations
        obs_history = []

        # Create video directory
        video_dir = os.path.join(run_dir, "videos", "play")
        os.makedirs(video_dir, exist_ok=True)

        img_size = eval_envs.unwrapped.cfg.img_size

        pbar = tqdm.tqdm(total=args.num_eval_env_steps)
        while step < args.num_eval_env_steps and not done.all():
            # Store current observations
            obs_history.append(obs.clone().cpu().numpy())

            action = play_agent.get_action(obs)
            obs, _, done, _ = eval_envs.step(action)
            step += 1
            global_step += args.num_eval_envs
            pbar.update(1)
            pbar.set_description(f"step: {step}")

        # Save final observation
        obs_history.append(obs.clone().cpu().numpy())

        # Convert observations to videos for each environment
        obs_array = np.array(obs_history)  # Shape: (steps, n_envs, obs_dim)

        # Create videos for each environment
        for env_idx in range(args.num_eval_envs):
            # Extract observations for this environment
            env_obs = obs_array[
                :, env_idx, : img_size[0] * img_size[1] * img_size[2]
            ]  # Take RGB channels
            # Reshape to (steps, height, width, channels)
            env_frames = env_obs.reshape(
                -1, img_size[0], img_size[1], img_size[2]
            ).transpose(0, 2, 3, 1)

            # Convert from float [0,1] to uint8 [0,255] if needed
            if env_frames.max() <= 1.0:
                env_frames = (env_frames * 255).astype(np.uint8)
            else:
                env_frames = env_frames.astype(np.uint8)

            # No need to convert RGB to BGR for imageio (keep RGB format)

            # Create video using imageio
            video_path = os.path.join(video_dir, f"env_{env_idx}_fpp.mp4")

            # Save video with imageio
            imageio.mimwrite(video_path, env_frames, fps=30, quality=8)

            print(f"Saved video for environment {env_idx}: {video_path}")

        eval_envs.close()

        # Log videos to wandb if capture_video is enabled
        if args.capture_video:
            for env_idx in range(args.num_eval_envs):
                video_path = os.path.join(video_dir, f"env_{env_idx}_fpp.mp4")
                wandb.log(
                    {
                        f"{mode}/env_{env_idx}_video": wandb.Video(
                            video_path,
                            format="mp4",
                            fps=30,
                        )
                    },
                    step=global_step,
                )

    run_play(agent)
    wandb.finish()


if __name__ == "__main__":
    try:
        os.environ["WANDB_MODE"] = "dryrun"
        main(args)
    except Exception as e:
        print("Exception:", e)
    finally:
        simulation_app.close()
