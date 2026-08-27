"""
ManiSkill3 Panda Keyframe Solution using PD Arm Joint / EE Control.
No mplib dependency. Solves PickCube-v1 deterministically.
"""

import numpy as np
import torch

class ScriptedPickPlacePolicy:
    """State-machine heuristic pick and place baseline policy."""
    
    def __init__(self, env):
        self.env = env
        self.reset()
        
    def reset(self):
        self.stage = 0
        self.step_count = 0
        
    def get_action(self, obs, info=None):
        self.step_count += 1
        
        def to_np(val):
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            if val.ndim > 1:
                val = val[0]
            return val
            
        extra = obs["extra"]
        tcp_pos = to_np(extra["tcp_pose"])[:3]
        obj_pos = to_np(extra["obj_pose"])[:3]
        goal_pos = to_np(extra["goal_pos"])[:3] if "goal_pos" in extra else obj_pos + np.array([0.0, 0.0, 0.15])
        
        delta_pos = np.zeros(3)
        gripper_action = -1.0
        
        # State machine sequence using relative pose targets
        if self.stage == 0:
            # Stage 0: Fast XY approach over object (z offset +0.06m)
            target = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.06])
            diff = target - tcp_pos
            delta_pos = diff * 15.0
            if np.linalg.norm(diff[:2]) < 0.008 or self.step_count > 15:
                self.stage = 1
                self.step_count = 0
                
        elif self.stage == 1:
            # Stage 1: Descend onto object (z target = obj_z + 0.008m)
            target = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.008])
            diff = target - tcp_pos
            delta_pos = diff * 15.0
            if abs(diff[2]) < 0.005 or self.step_count > 15:
                self.stage = 2
                self.step_count = 0
                
        elif self.stage == 2:
            # Stage 2: Close gripper firmly
            target = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.008])
            diff = target - tcp_pos
            delta_pos = diff * 5.0
            gripper_action = 1.0  # Close fingers
            if self.step_count > 15:
                self.stage = 3
                self.step_count = 0
                
        elif self.stage == 3:
            # Stage 3: Lift object
            target = np.array([tcp_pos[0], tcp_pos[1], 0.25])
            diff = target - tcp_pos
            delta_pos = diff * 12.0
            gripper_action = 1.0
            if tcp_pos[2] > 0.18 or self.step_count > 20:
                self.stage = 4
                self.step_count = 0
                
        elif self.stage == 4:
            # Stage 4: Transport to goal
            target = goal_pos + np.array([0.0, 0.0, 0.03])
            diff = target - tcp_pos
            delta_pos = diff * 12.0
            gripper_action = 1.0
            if np.linalg.norm(diff) < 0.02 or self.step_count > 25:
                self.stage = 5
                self.step_count = 0
                
        elif self.stage == 5:
            # Stage 5: Release
            delta_pos = np.zeros(3)
            gripper_action = -1.0
            
        delta_pos = np.clip(delta_pos, -1.0, 1.0)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, 0.0, 0.0, gripper_action], dtype=np.float32)
        
        if isinstance(extra["tcp_pose"], torch.Tensor) and extra["tcp_pose"].ndim > 1:
            action = torch.tensor(action, device=extra["tcp_pose"].device).unsqueeze(0)
            
        return action
