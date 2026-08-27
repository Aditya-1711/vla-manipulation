"""
ManiSkill3 PickCube-v1 Source & Custom Task Subclassing.
"""

import gymnasium as gym
import mani_skill.envs
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.building.ground import build_ground
import torch

class CustomPickCubeEnv(PickCubeEnv):
    """
    Subclassed ManiSkill3 PickCubeEnv allowing explicit obj_xy_range configuration.
    """
    def __init__(self, *args, obj_xy_range=(-0.05, 0.05), **kwargs):
        self.obj_xy_range = obj_xy_range
        super().__init__(*args, **kwargs)
        
    def _initialize_episode(self, env_idx, options):
        # Override _initialize_episode to use self.obj_xy_range
        with torch.device(self.device):
            b = len(env_idx)
            self._initialize_transforms(env_idx)
            
            # Sample cube position within explicit range
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = torch.rand(b) * (self.obj_xy_range[1] - self.obj_xy_range[0]) + self.obj_xy_range[0]
            xyz[:, 1] = torch.rand(b) * (self.obj_xy_range[1] - self.obj_xy_range[0]) + self.obj_xy_range[0]
            xyz[:, 2] = 0.02
            
            self.cube.set_pose(mani_skill.utils.structs.Pose.create_from_pq(p=xyz))
