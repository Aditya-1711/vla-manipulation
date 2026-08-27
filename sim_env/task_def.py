"""
ManiSkill3 Pick-and-Place Task Definition with Domain Randomization Configuration.

Defines In-Distribution and Out-of-Distribution / Generalization trial suites.
"""

import numpy as np
import gymnasium as gym

class PickPlaceTaskConfig:
    """Task configuration parameters and domain randomization boundaries."""
    ENV_ID = "PickCube-v1"
    ROBOT = "panda"
    IMAGE_OBS_SIZE = (224, 224)
    
    # In-Distribution Randomization Range (Cube starting XY offset bounds in meters)
    # Matching dataset demo collection distribution (Seeds 1000-1049)
    IN_DIST_OBJ_POS_X = (-0.05, 0.05)
    IN_DIST_OBJ_POS_Y = (-0.05, 0.05)
    
    # Out-of-Distribution / Generalization Randomization Ranges
    # Novel un-seen object starting positions (extrapolated bounds)
    OOD_OBJ_POS_X = (-0.15, 0.15)
    OOD_OBJ_POS_Y = (-0.15, 0.15)
    
    # Seed allocations for benchmark evaluation suites
    IN_DIST_SEEDS = list(range(1000, 1020))  # Seeds 1000-1019
    OOD_SEEDS = list(range(2000, 2020))      # Seeds 2000-2019

def make_env(env_id=PickPlaceTaskConfig.ENV_ID, obs_mode="state_dict", control_mode="pd_ee_delta_pose", render_mode="rgb_array", seed=None):
    """Factory helper to create ManiSkill3 environment instance."""
    import mani_skill.envs
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode
    )
    if seed is not None:
        env.reset(seed=seed)
    return env
