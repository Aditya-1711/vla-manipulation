"""
Scripted Pick-and-Place Baseline Policy for ManiSkill3 (Panda Arm).

Calibrated based on exact PickCube-v1 extra observation keys:
- tcp_pose z = ~0.17m (initial robot EE height above table)
- obj_pose z = ~0.02m (cube center height)
- goal_pos z = ~0.13m (target destination height)
- pd_ee_delta_pose action takes tensor shape [batch_size, 7]
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
        """
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
        gripper_action = -1.0  # -1 open, 1 close
        
        if self.stage == 0:
            # Approach: position TCP directly over cube XY (hover at z = obj_z + 0.06 = ~0.08m)
            target = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.06])
            diff = target - tcp_pos
            if np.linalg.norm(diff) < 0.015 or self.step_count > 30:
                self.stage = 1
                self.step_count = 0
            else:
                delta_pos = diff * 5.0
                
        elif self.stage == 1:
            # Descend: move fingers down around cube (target z = obj_z + 0.005 = ~0.025m)
            target = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.005])
            diff = target - tcp_pos
            if np.linalg.norm(diff) < 0.010 or self.step_count > 25:
                self.stage = 2
                self.step_count = 0
            else:
                delta_pos = diff * 5.0
                
        elif self.stage == 2:
            # Grasp: close gripper fingers around cube
            target = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.005])
            diff = target - tcp_pos
            delta_pos = diff * 2.0
            gripper_action = 1.0  # Close fingers
            if self.step_count > 15:
                self.stage = 3
                self.step_count = 0
                
        elif self.stage == 3:
            # Lift: raise cube high (target z = 0.25m)
            target = np.array([tcp_pos[0], tcp_pos[1], 0.25])
            diff = target - tcp_pos
            gripper_action = 1.0
            if tcp_pos[2] > 0.20 or self.step_count > 35:
                self.stage = 4
                self.step_count = 0
            else:
                delta_pos = diff * 5.0
                
        elif self.stage == 4:
            # Move to Goal position
            target = goal_pos + np.array([0.0, 0.0, 0.05])
            diff = target - tcp_pos
            gripper_action = 1.0
            if np.linalg.norm(diff) < 0.02 or self.step_count > 40:
                self.stage = 5
                self.step_count = 0
            else:
                delta_pos = diff * 5.0
                
        elif self.stage == 5:
            # Release cube at goal
            delta_pos = np.zeros(3)
            gripper_action = -1.0
            
        delta_pos = np.clip(delta_pos, -1.0, 1.0)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, 0.0, 0.0, gripper_action], dtype=np.float32)
        
        # Convert to tensor matching environment device/batch shape
        if isinstance(extra["tcp_pose"], torch.Tensor):
            device = extra["tcp_pose"].device
            if extra["tcp_pose"].ndim > 1:
                action = torch.tensor(action, device=device).unsqueeze(0)
            else:
                action = torch.tensor(action, device=device)
                
        return action
