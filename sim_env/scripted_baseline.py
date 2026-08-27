"""
Scripted Pick-and-Place Baseline Policy for ManiSkill3 (Panda Arm).

Calibrated using ManiSkill3 keyframe motion sequence:
- Stage 0: Reach & Align above cube
- Stage 1: Descend & Open fingers
- Stage 2: Settle & Close fingers
- Stage 3: Lift cube
- Stage 4: Transport to goal
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
        
        def to_np3(val):
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            if val.ndim > 1:
                val = val[0]
            return val[:3]
            
        extra = obs["extra"]
        tcp_pos = to_np3(extra["tcp_pose"])
        obj_pos = to_np3(extra["obj_pose"])
        goal_pos = to_np3(extra["goal_pos"]) if "goal_pos" in extra else obj_pos + np.array([0.0, 0.0, 0.15])
        
        delta_pos = np.zeros(3)
        gripper_action = -1.0
        
        is_grasped = False
        if info is not None and isinstance(info, dict):
            ig = info.get("is_grasped", False)
            if isinstance(ig, torch.Tensor):
                is_grasped = ig.any().item()
            else:
                is_grasped = bool(ig)
                
        if self.stage == 0:
            # Stage 0: Align XY above cube
            target = np.array([obj_pos[0], obj_pos[1], tcp_pos[2]])
            diff = target - tcp_pos
            if np.linalg.norm(diff[:2]) < 0.005 or self.step_count > 30:
                self.stage = 1
                self.step_count = 0
            else:
                delta_pos[:2] = diff[:2] * 10.0
                
        elif self.stage == 1:
            # Stage 1: Descend down onto cube
            target = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.002])
            diff = target - tcp_pos
            if np.linalg.norm(diff) < 0.005 or self.step_count > 25:
                self.stage = 2
                self.step_count = 0
            else:
                delta_pos = diff * 10.0
                
        elif self.stage == 2:
            # Stage 2: Settle and Close Gripper
            gripper_action = 1.0
            if is_grasped or self.step_count > 30:
                self.stage = 3
                self.step_count = 0
                
        elif self.stage == 3:
            # Stage 3: Lift
            target = np.array([tcp_pos[0], tcp_pos[1], 0.25])
            diff = target - tcp_pos
            gripper_action = 1.0
            if tcp_pos[2] > 0.20 or self.step_count > 35:
                self.stage = 4
                self.step_count = 0
            else:
                delta_pos = diff * 8.0
                
        elif self.stage == 4:
            # Stage 4: Move to Goal
            target = goal_pos
            diff = target - tcp_pos
            gripper_action = 1.0
            if np.linalg.norm(diff) < 0.02 or self.step_count > 40:
                self.stage = 5
                self.step_count = 0
            else:
                delta_pos = diff * 8.0
                
        elif self.stage == 5:
            delta_pos = np.zeros(3)
            gripper_action = -1.0
            
        delta_pos = np.clip(delta_pos, -1.0, 1.0)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, 0.0, 0.0, gripper_action], dtype=np.float32)
        
        if isinstance(extra["tcp_pose"], torch.Tensor) and extra["tcp_pose"].ndim > 1:
            action = torch.tensor(action, device=extra["tcp_pose"].device).unsqueeze(0)
            
        return action
