"""
Scripted Pick-and-Place Baseline Policy for ManiSkill3 (Panda Arm).

Supports both `obs_mode="state_dict"` and `obs_mode="rgbd"`.
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
        self.grasped_consecutive_steps = 0
        
    def get_action(self, obs, info=None):
        self.step_count += 1
        
        def to_np(val):
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            if val.ndim > 1:
                val = val[0]
            return val
            
        # Robust pose extraction for state_dict, rgbd, and raw sensor observations
        extra = obs.get("extra", {}) if isinstance(obs, dict) else {}
        
        if "tcp_pose" in extra:
            tcp_pos = to_np(extra["tcp_pose"])[:3]
        else:
            tcp_pos = to_np(self.env.unwrapped.agent.tcp.pose.p)
            
        if "obj_pose" in extra:
            obj_pos = to_np(extra["obj_pose"])[:3]
        else:
            obj_pos = to_np(self.env.unwrapped.cube.pose.p)
            
        if "goal_pos" in extra:
            goal_pos = to_np(extra["goal_pos"])[:3]
        else:
            goal_pos = to_np(self.env.unwrapped.goal_site.pose.p)
        
        delta_pos = np.zeros(3)
        gripper_action = 1.0  # +1.0 OPEN gripper
        
        # Track consecutive is_grasped status
        is_grasped = False
        if info is not None and isinstance(info, dict):
            ig = info.get("is_grasped", False)
            if isinstance(ig, torch.Tensor):
                is_grasped = ig.any().item()
            else:
                is_grasped = bool(ig)
                
        if is_grasped:
            self.grasped_consecutive_steps += 1
        else:
            self.grasped_consecutive_steps = 0
            
        if self.stage == 0:
            target = obj_pos + np.array([0.0, 0.0, 0.08])
            diff = target - tcp_pos
            if np.linalg.norm(diff) < 0.015 or self.step_count > 30:
                self.stage = 1
                self.step_count = 0
            else:
                delta_pos = diff * 5.0
                
        elif self.stage == 1:
            target = obj_pos + np.array([0.0, 0.0, 0.010])
            diff = target - tcp_pos
            if np.linalg.norm(diff) < 0.010 or self.step_count > 25:
                self.stage = 2
                self.step_count = 0
                self.grasped_consecutive_steps = 0
            else:
                delta_pos = diff * 5.0
                
        elif self.stage == 2:
            target = obj_pos + np.array([0.0, 0.0, 0.010])
            delta_pos = (target - tcp_pos) * 1.5
            gripper_action = -1.0  # Close fingers
            
            if self.grasped_consecutive_steps >= 3 or self.step_count > 30:
                self.stage = 3
                self.step_count = 0
                
        elif self.stage == 3:
            target = np.array([tcp_pos[0], tcp_pos[1], 0.25])
            diff = target - tcp_pos
            gripper_action = -1.0  # Hold closed
            lift_gain = 1.5 if self.step_count < 10 else 4.0
            delta_pos = diff * lift_gain
            
            if tcp_pos[2] > 0.20 or self.step_count > 40:
                self.stage = 4
                self.step_count = 0
                
        elif self.stage == 4:
            target = goal_pos + np.array([0.0, 0.0, 0.015])
            diff = target - tcp_pos
            gripper_action = -1.0  # Keep gripper closed
            if np.linalg.norm(diff) < 0.02 or self.step_count > 45:
                self.stage = 5
                self.step_count = 0
            else:
                delta_pos = diff * 4.0
                
        elif self.stage == 5:
            delta_pos = np.zeros(3)
            gripper_action = -1.0 if self.step_count < 15 else 1.0  # Hold 15 steps static then open
            
        delta_pos = np.clip(delta_pos, -1.0, 1.0)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, 0.0, 0.0, gripper_action], dtype=np.float32)
        
        if "tcp_pose" in extra and isinstance(extra["tcp_pose"], torch.Tensor) and extra["tcp_pose"].ndim > 1:
            action = torch.tensor(action, device=extra["tcp_pose"].device).unsqueeze(0)
            
        return action
