"""
Scripted Pick-and-Place Baseline Policy for ManiSkill3 (Panda Arm).

Uses classical delta-pose end-effector state machine control (Grasp -> Move Up -> Move Target -> Place).
No dependency on mplib.
"""

import numpy as np

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
        obs is state_dict from ManiSkill3 PickCube-v1.
        """
        self.step_count += 1
        
        # Extract object pose and end-effector pose from observations
        if isinstance(obs, dict) and "extra" in obs and "tcp_pose" in obs["extra"]:
            tcp_pose = obs["extra"]["tcp_pose"]  # (7,) position + quaternion
            obj_pose = obs["extra"]["obj_pose"]  # (7,) position + quaternion
            goal_pos = obs["extra"]["goal_pos"] if "goal_pos" in obs["extra"] else obj_pose[:3] + np.array([0, 0, 0.2])
        else:
            # Fallback tensor extraction if batched
            tcp_pose = obs["extra"]["tcp_pose"][0].cpu().numpy()
            obj_pose = obs["extra"]["obj_pose"][0].cpu().numpy()
            goal_pos = obs["extra"]["goal_pos"][0].cpu().numpy() if "goal_pos" in obs["extra"] else obj_pose[:3] + np.array([0, 0, 0.2])
            
        tcp_pos = tcp_pose[:3]
        obj_pos = obj_pose[:3]
        
        # Delta vector calculations
        delta_pos = np.zeros(3)
        gripper_action = -1.0  # -1 open, 1 close
        
        # State Machine Steps
        if self.stage == 0:
            # Stage 0: Move TCP above object (hover target)
            target = obj_pos + np.array([0.0, 0.0, 0.10])
            diff = target - tcp_pos
            if np.linalg.norm(diff) < 0.02 or self.step_count > 30:
                self.stage = 1
                self.step_count = 0
            else:
                delta_pos = diff * 5.0
                
        elif self.stage == 1:
            # Stage 1: Descend down to cube position
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
            gripper_action = 1.0  # Close gripper
            if self.step_count > 15:
                self.stage = 3
                self.step_count = 0
                
        elif self.stage == 3:
            # Stage 3: Lift cube up
            target = obj_pos + np.array([0.0, 0.0, 0.25])
            diff = target - tcp_pos
            gripper_action = 1.0  # Keep gripper closed
            if np.linalg.norm(diff[:2]) < 0.03 and tcp_pos[2] > 0.20 or self.step_count > 35:
                self.stage = 4
                self.step_count = 0
            else:
                delta_pos = diff * 4.0
                
        elif self.stage == 4:
            # Stage 4: Move to target goal position
            target = goal_pos
            diff = target - tcp_pos
            gripper_action = 1.0  # Keep gripper closed
            if np.linalg.norm(diff) < 0.03 or self.step_count > 40:
                self.stage = 5
                self.step_count = 0
            else:
                delta_pos = diff * 4.0
                
        elif self.stage == 5:
            # Stage 5: Release gripper at goal position
            delta_pos = np.zeros(3)
            gripper_action = -1.0  # Open gripper
            
        # Clip delta action to valid EE bounds [-1, 1]
        delta_pos = np.clip(delta_pos, -1.0, 1.0)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, 0.0, 0.0, gripper_action], dtype=np.float32)
        return action
