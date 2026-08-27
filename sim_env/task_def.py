"""
ManiSkill3 Pick-and-Place Task Definition with Domain Randomization.
"""

import gymnasium as gym
import numpy as np

class PickPlaceTaskConfig:
    """Task configuration parameters."""
    ENV_ID = "PickCube-v1"
    ROBOT = "panda"
    IMAGE_OBS_SIZE = (224, 224)
    
    # In-distribution randomization ranges (cube pose offset in meters)
    CUBE_POS_IN_DIST_X = (-0.05, 0.05)
    CUBE_POS_IN_DIST_Y = (-0.05, 0.05)
    
    # Out-of-distribution / Generalization randomization ranges
    CUBE_POS_OOD_X = (-0.15, 0.15)
    CUBE_POS_OOD_Y = (-0.15, 0.15)

def make_env(task_config=PickPlaceTaskConfig, render_mode="rgb_array", seed=None):
    """Factory helper to instantiate ManiSkill3 task env."""
    import mani_skill.envs
    env = gym.make(
        task_config.ENV_ID,
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode=render_mode
    )
    if seed is not None:
        env.reset(seed=seed)
    return env
