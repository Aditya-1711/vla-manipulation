"""
Debug script to inspect ManiSkill3 PickCube-v1 observation keys and TCP vs Cube coordinates.
"""

import gymnasium as gym
import mani_skill.envs
import numpy as np

env = gym.make("PickCube-v1", obs_mode="state_dict", control_mode="pd_ee_delta_pose")
obs, info = env.reset(seed=1000)

print("--- Observation Keys ---")
for k, v in obs.items():
    if isinstance(v, dict):
        print(f"Sub-dict '{k}':", list(v.keys()))
    else:
        print(f"Key '{k}': shape {v.shape}")

# Sample 1 step and inspect exact pose shapes and values
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print("\n--- Pose Data ---")
if "extra" in obs:
    for subk, subv in obs["extra"].items():
        print(f"extra['{subk}']: {subv}")
