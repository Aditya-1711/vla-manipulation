"""
State Machine Diagnostic Script for PickCube-v1 Scripted Policy.
Prints exact stage progression, step counts, TCP position, Object position, and Grasped state.
"""

import gymnasium as gym
import mani_skill.envs
import torch
import numpy as np
from sim_env.scripted_baseline import ScriptedPickPlacePolicy

env = gym.make("PickCube-v1", obs_mode="state_dict", control_mode="pd_ee_delta_pose")
obs, info = env.reset(seed=1000)
policy = ScriptedPickPlacePolicy(env)

print("--- Step-by-Step Diagnostic Trajectory ---")
for step in range(80):
    action = policy.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    tcp = obs["extra"]["tcp_pose"][0].cpu().numpy()[:3] if isinstance(obs["extra"]["tcp_pose"], torch.Tensor) else obs["extra"]["tcp_pose"][:3]
    obj = obs["extra"]["obj_pose"][0].cpu().numpy()[:3] if isinstance(obs["extra"]["obj_pose"], torch.Tensor) else obs["extra"]["obj_pose"][:3]
    is_grasped = info.get("is_grasped", False)
    if isinstance(is_grasped, torch.Tensor): is_grasped = is_grasped.item()
    
    print(f"Step {step+1:02d} | Stage {policy.stage} | TCP z={tcp[2]:.4f} | Obj z={obj[2]:.4f} | Dist XY={np.linalg.norm(tcp[:2]-obj[:2]):.4f} | Grasped={is_grasped}")
    
    if info.get("success", False):
        print("SUCCESS TRIGGERED!")
        break
env.close()
