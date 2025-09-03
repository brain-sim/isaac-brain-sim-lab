import argparse
import sys

import gymnasium as gym
import torch
import numpy as np
from typing import Dict, List
from isaaclab.app import AppLauncher

def launch_app():
    # add argparse arguments
    parser = argparse.ArgumentParser(
        description="This script demonstrates different legged robots."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Train-2ACDC-v0",
        help="Environment ID to run.",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="RayTracedLighting",
        choices=["RayTracedLighting", "PathTracing"],
        help="Renderer to use.",
    )
    parser.add_argument(
        "--samples_per_pixel_per_frame",
        type=int,
        default=1,
        help="Number of samples per pixel per frame.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    return simulation_app, args_cli


try:
    from isaaclab.app import AppLauncher

    simulation_app, args = launch_app()
except ImportError:
    raise ImportError("Isaac Lab is not installed. Please install it first.")

# Import carb and omni after simulation app is created
import carb
import omni.appwindow


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
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import brain_sim_tasks  # noqa: F401

    def thunk():
        cfg = parse_env_cfg(
            task, device, num_envs=num_envs, use_fabric=not disable_fabric
        )
        env = gym.make(
            task,
            cfg=cfg,
            render_mode="rgb_array"
            if (capture_video and log_dir is not None)
            else None,
            debug=False,
            play_mode=True,
        )
        return env

    return thunk


class bsKeyboard:
    def __init__(self):
        self._base_command = [0.0, 0.0, 0.0]  # Initialize base command

    def setup(self):
        # bindings for keyboard to command
        self._input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": [2.0, 0.0, 0.0],
            "UP": [2.0, 0.0, 0.0],
            "W": [2.0, 0.0, 0.0],
            # back command
            "NUMPAD_2": [-2.0, 0.0, 0.0],
            "DOWN": [-2.0, 0.0, 0.0],
            "S": [-2.0, 0.0, 0.0],
            # left command
            "NUMPAD_6": [0.0, -2.0, 0.0],
            "RIGHT": [0.0, -2.0, 0.0],
            "D": [0.0, -2.0, 0.0],
            # right command
            "NUMPAD_4": [0.0, 2.0, 0.0],
            "LEFT": [0.0, 2.0, 0.0],
            "A": [0.0, 2.0, 0.0],
            # yaw command (positive)
            "NUMPAD_7": [0.0, 0.0, 2.0],
            "N": [0.0, 0.0, 2.0],
            "Q": [0.0, 0.0, 2.0],
            # yaw command (negative)
            "NUMPAD_9": [0.0, 0.0, -2.0],
            "M": [0.0, 0.0, -2.0],
            "E": [0.0, 0.0, -2.0],
        }

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
    
    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """Subscriber callback to when kit is updated."""

        # when a key is pressed or released the command is adjusted w.r.t the key-mapping
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # on pressing, the command is incremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command = np.array(self._base_command) + np.array(self._input_keyboard_mapping[event.input.name])
                self._base_command = self._base_command.tolist()
                print(f"[DEBUG]: Key pressed: {event.input.name}, Command: {self._base_command}")

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # on release, the command is decremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command = np.array(self._base_command) - np.array(self._input_keyboard_mapping[event.input.name])
                self._base_command = self._base_command.tolist()
                print(f"[DEBUG]: Key released: {event.input.name}, Command: {self._base_command}")
        return True
    
    def get_base_command(self) -> List[float]:
        """Get the current base command."""
        return self._base_command


def main():
    """Main function."""
    print(f"[INFO]: Creating environment {args.env}...")
    env = make_isaaclab_env(
        args.env,
        "cuda:0",
        1,
        False,
        False,
    )()
    print(f"[INFO]: Environment {args.env} has been created.")
    
    # Initialize keyboard handler
    keyboard_handler = bsKeyboard()
    keyboard_handler.setup()
    
    obs, _ = env.reset()
    
    print("[INFO]: Use WASD/Arrow keys/Numpad to move, Q/E to yaw.")
    print("[INFO]: W/UP/NUMPAD_8: forward, S/DOWN/NUMPAD_2: backward")
    print("[INFO]: A/LEFT/NUMPAD_4: left, D/RIGHT/NUMPAD_6: right") 
    print("[INFO]: Q/N/NUMPAD_7: yaw left, E/M/NUMPAD_9: yaw right")
    print("[INFO]: Press ESC to exit.")

    # Main control loop
    try:
        while True:
            # Get current command from keyboard
            command = keyboard_handler.get_base_command()
            action = torch.tensor(command, dtype=torch.float32)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check if we should exit (ESC key handling would need to be added to bsKeyboard)
            if terminated or truncated:
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("[INFO]: Keyboard interrupt received. Exiting...")
    
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Exception:", e)
        raise e
    finally:
        simulation_app.close()