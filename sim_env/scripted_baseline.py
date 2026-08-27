"""
Scripted Pick-and-Place Baseline Policy for ManiSkill3 (Panda Arm).
Robustly handles batched tensor vs flat numpy array observations from GPU vector environment.
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
        
    def get_action(self, obs):
        """
        Computes delta end-effector target action [dx, dy, dz, droll, dpitch, dyaw, gripper].
        Handles tensor and numpy obs formats.
        """
        self.step_count += 1
        
        # Helper to extract numpy (3,) vector regardless of torch/batched input
        def to_np3(val):
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            if val.ndim > 1:
                val = val[0]
            return val[:3]
            
        extra = obs["extra"]
        tcp_pos = to_np3(extra["tcp_pose"])
        obj_pos = to_np3(extra["obj_pose"])
        goal_pos = to_np3(extra["goal_pos"]) if "goal_pos" in extra else obj_pos + np.array([0.0, 0.0, 0.2])
        
        delta_pos = np.zeros(3)
        gripper_action = -1.0  # -1 open, 1 close
        
        # State Machine Logic with calibrated offsets for Panda TCP
        if self.stage == 0:
            # Approach: position above cube
            target = obj_pos + np.array([0.0, 0.0, 0.08])
            diff = target - tcp_pos
            if np.linalg.norm(diff) < 0.015 or self.step_count > 25:
                self.stage = 1
                self.step_count = 0
            else:
                delta_pos = diff * 8.0
                
        elif self.stage == 1:
            # Descend: lower TCP down around cube center
            target = obj_pos + np.array([0.0, 0.0, -0.005])
            diff = target - tcp_pos
            if np.linalg.norm(diff) < 0.01 or self.step_count > 20:
                self.stage = 2
                self.step_count = 0
            else:
                delta_pos = diff * 8.0
                
        elif self.stage == 2:
            # Grasp: close fingers
            target = obj_pos + np.array([0.0, 0.0, -0.005])
            diff = target - tcp_pos
            delta_pos = diff * 4.0
            gripper_action = 1.0  # Close fingers
            if self.step_count > 15:
                self.stage = 3
                self.step_count = 0
                
        elif self.stage == 3:
            # Lift: raise cube high
            target = obj_pos + np.array([0.0, 0.0, 0.20])
            diff = target - tcp_pos
            gripper_action = 1.0
            if tcp_pos[2] > obj_pos[2] + 0.12 or self.step_count > 30:
                self.stage = 4
                self.step_count = 0
            else:
                delta_pos = diff * 6.0
                
        elif self.stage == 4:
            # Move to Goal position
            target = goal_pos
            diff = target - tcp_pos
            gripper_action = 1.0
            if np.linalg.norm(diff) < 0.02 or self.step_count > 35:
                self.stage = 5
                self.step_count = 0
            else:
                delta_pos = diff * 6.0
                
        elif self.stage == 5:
            # Release cube at goal
            delta_pos = np.zeros(3)
            gripper_action = -1.0
            
        delta_pos = np.clip(delta_pos, -1.0, 1.0)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, 0.0, 0.0, gripper_action], dtype=np.float32)
        
        # Expand dimension if environment expects batched tensor action
        if isinstance(extra["tcp_pose"], torch.Tensor) and extra["tcp_pose"].ndim > 1:
            action = torch.tensor(action, device=extra["tcp_pose"].device).unsqueeze(0)
            
        return action
