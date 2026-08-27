"""
Scripted Pick-and-Place Baseline Policy for ManiSkill3 (Panda Arm).

Empirically Verified Gripper Convention:
- action[6] = -1.0 -> CLOSE gripper fingers
- action[6] = +1.0 -> OPEN gripper fingers
"""

import numpy as np
import torch

class ScriptedPickPlacePolicy:
    """State-machine heuristic pick and place baseline policy."""
    
    def __init__(self, env):
        self.env = env
        self.reset()
        
    def reset(self):
        self.stage = 0  # 0: approach, 1: descend, 2: grasp, 3: lift, 4: move_to_goal, 5: release
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
        goal_pos = to_np(extra["goal_pos"])[:3] if "goal_pos" in extra else obj_pos + np.array([0.0, 0.0, 0.2])
        
        delta_pos = np.zeros(3)
        gripper_action = 1.0  # +1.0 OPEN gripper
        
        if self.stage == 0:
            # Stage 0: Move TCP above object (hover target z = obj_z + 0.10m)
            target = obj_pos + np.array([0.0, 0.0, 0.10])
            diff = target - tcp_pos
            if np.linalg.norm(diff) < 0.02 or self.step_count > 30:
                self.stage = 1
                self.step_count = 0
            else:
                delta_pos = diff * 5.0
                
        elif self.stage == 1:
            # Stage 1: Descend down to cube position (z = obj_z + 0.015m)
            target = obj_pos + np.array([0.0, 0.0, 0.015])
            diff = target - tcp_pos
            if np.linalg.norm(diff) < 0.015 or self.step_count > 25:
                self.stage = 2
                self.step_count = 0
            else:
                delta_pos = diff * 5.0
                
        elif self.stage == 2:
            # Stage 2: Close gripper to grasp
            target = obj_pos + np.array([0.0, 0.0, 0.015])
            delta_pos = (target - tcp_pos) * 2.0
            gripper_action = -1.0  # -1.0 CLOSE gripper
            if self.step_count > 15:
                self.stage = 3
                self.step_count = 0
                
        elif self.stage == 3:
            # Stage 3: Lift cube up
            target = obj_pos + np.array([0.0, 0.0, 0.25])
            diff = target - tcp_pos
            gripper_action = -1.0  # Keep gripper closed (-1.0)
            if (np.linalg.norm(diff[:2]) < 0.03 and tcp_pos[2] > 0.20) or self.step_count > 35:
                self.stage = 4
                self.step_count = 0
            else:
                delta_pos = diff * 4.0
                
        elif self.stage == 4:
            # Stage 4: Move to target goal position
            target = goal_pos
            diff = target - tcp_pos
            gripper_action = -1.0  # Keep gripper closed (-1.0)
            if np.linalg.norm(diff) < 0.03 or self.step_count > 40:
                self.stage = 5
                self.step_count = 0
            else:
                delta_pos = diff * 4.0
                
        elif self.stage == 5:
            # Stage 5: Release gripper at goal position
            delta_pos = np.zeros(3)
            gripper_action = 1.0  # +1.0 OPEN gripper
            
        delta_pos = np.clip(delta_pos, -1.0, 1.0)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, 0.0, 0.0, gripper_action], dtype=np.float32)
        
        if isinstance(extra["tcp_pose"], torch.Tensor) and extra["tcp_pose"].ndim > 1:
            action = torch.tensor(action, device=extra["tcp_pose"].device).unsqueeze(0)
            
        return action
