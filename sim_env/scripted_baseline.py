"""
ManiSkill3 Keyframe Trajectory Policy for PickCube-v1.
No mplib required. Uses exact keyframe relative delta control.
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
            # Stage 0: Align XY directly over object
            target_xy = obj_pos[:2]
            diff_xy = target_xy - tcp_pos[:2]
            delta_pos[0] = diff_xy[0] * 10.0
            delta_pos[1] = diff_xy[1] * 10.0
            delta_pos[2] = 0.0  # Keep height constant during alignment
            
            if np.linalg.norm(diff_xy) < 0.005 or self.step_count > 20:
                self.stage = 1
                self.step_count = 0
                
        elif self.stage == 1:
            # Stage 1: Descend vertically to object grasp position
            target_z = obj_pos[2] + 0.002
            diff_z = target_z - tcp_pos[2]
            delta_pos[2] = diff_z * 10.0
            
            if abs(diff_z) < 0.005 or self.step_count > 20:
                self.stage = 2
                self.step_count = 0
                
        elif self.stage == 2:
            # Stage 2: Close gripper
            gripper_action = 1.0
            if is_grasped or self.step_count > 25:
                self.stage = 3
                self.step_count = 0
                
        elif self.stage == 3:
            # Stage 3: Lift up
            target_z = 0.25
            diff_z = target_z - tcp_pos[2]
            delta_pos[2] = diff_z * 8.0
            gripper_action = 1.0
            
            if tcp_pos[2] > 0.18 or self.step_count > 30:
                self.stage = 4
                self.step_count = 0
                
        elif self.stage == 4:
            # Stage 4: Move to Goal
            target = goal_pos
            diff = target - tcp_pos
            delta_pos = diff * 8.0
            gripper_action = 1.0
            
            if np.linalg.norm(diff) < 0.02 or self.step_count > 40:
                self.stage = 5
                self.step_count = 0
                
        elif self.stage == 5:
            delta_pos = np.zeros(3)
            gripper_action = -1.0
            
        delta_pos = np.clip(delta_pos, -1.0, 1.0)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, 0.0, 0.0, gripper_action], dtype=np.float32)
        
        if isinstance(extra["tcp_pose"], torch.Tensor) and extra["tcp_pose"].ndim > 1:
            action = torch.tensor(action, device=extra["tcp_pose"].device).unsqueeze(0)
            
        return action
