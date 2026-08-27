"""
ManiSkill3 Pick-and-Place Task Definition with Domain Randomization.
"""

import numpy as np

class PickPlaceTaskConfig:
    """Task configuration parameters."""
    ENV_ID = "PickCube-v1"  # ManiSkill standard pick cube task
    ROBOT = "panda"
    IMAGE_OBS_SIZE = (224, 224)
    
    # In-distribution randomization ranges
    CUBE_POS_IN_DIST_X = (-0.1, 0.1)
    CUBE_POS_IN_DIST_Y = (-0.1, 0.1)
    
    # Out-of-distribution / Generalization randomization ranges
    CUBE_POS_OOD_X = (-0.25, 0.25)
    CUBE_POS_OOD_Y = (-0.25, 0.25)
