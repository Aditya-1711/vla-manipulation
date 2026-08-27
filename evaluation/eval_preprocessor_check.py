import os
import json
import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer

os.environ["SAPIEN_RENDER_ENGINE"] = "EGL"

CHECKPOINT_PATH = "/content/drive/MyDrive/vla-manipulation/checkpoints/smolvla_pickcube_60ep/checkpoints/last/pretrained_model"
DATASET_ROOT = "/content/drive/MyDrive/vla-manipulation/data/lerobot_pick_cube"
TASK_INSTRUCTION = "Pick up the red cube and place it at the goal location."

print("=========================================================")
print("=== 1. CHECKING POLICY UN-NORMALIZATION PREPROCESSOR ===")
print("=========================================================")

policy = SmolVLAPolicy.from_pretrained(CHECKPOINT_PATH)
policy.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    policy = policy.to("cuda")

# Inspect normalization / preprocessor attributes attached to policy
for attr in ["normalize_inputs", "unnormalize_outputs", "preprocessor", "postprocessor", "normalize_targets", "stats"]:
    has_attr = hasattr(policy, attr)
    val = getattr(policy, attr, None)
    print(f"policy.{attr}: {has_attr} (type: {type(val)})")

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
tokenizer_out = tokenizer(TASK_INSTRUCTION, return_tensors="pt", padding=True)
lang_tokens = tokenizer_out["input_ids"].to(device)
lang_mask = tokenizer_out["attention_mask"].bool().to(device)

def to_np(val):
    if isinstance(val, torch.Tensor): val = val.cpu().numpy()
    if val.ndim > 1: val = val[0]
    return val

print("\n=========================================================")
print("=== 2. SINGLE EPISODE ROLLOUT DIAGNOSTIC (SEED 1000) ===")
print("=========================================================")

env = gym.make("PickCube-v1", obs_mode="rgbd", render_mode="rgb_array", control_mode="pd_ee_delta_pose")
obs, _ = env.reset(seed=1000)
policy.reset()

print("Step | TCP Position (x, y, z)       | Gripper-Inverted Action Vector (7D)")
print("-" * 75)

for step in range(15):
    frame = env.render()
    if isinstance(frame, list): frame = frame[0]
    if isinstance(frame, torch.Tensor): frame = frame.cpu().numpy()
    if frame.ndim == 4: frame = frame[0]
    
    img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    tcp_pos = to_np(env.unwrapped.agent.tcp.pose.p)
    tcp_quat = to_np(env.unwrapped.agent.tcp.pose.q)
    tcp_pose_7d = np.concatenate([tcp_pos, tcp_quat], axis=-1)
    state_tensor = torch.from_numpy(tcp_pose_7d).float().unsqueeze(0).to(device)
    
    observation = {
        "observation.images.camera1": img_tensor,
        "observation.images.camera2": img_tensor.clone(),
        "observation.images.camera3": img_tensor.clone(),
        "observation.state": state_tensor,
        "observation.language.tokens": lang_tokens,
        "observation.language.attention_mask": lang_mask
    }
    
    with torch.no_grad():
        action_pred = policy.select_action(observation)
        
    action_np = action_pred.cpu().numpy() if isinstance(action_pred, torch.Tensor) else np.array(action_pred)
    if action_np.ndim > 1: action_np = action_np[0]
    
    # Invert gripper sign (+1.0 in dataset was saved when policy intended to close)
    # Gripper channel > 0.0 -> CLOSE (-1.0), <= 0.0 -> OPEN (+1.0)
    step_action = action_np.copy()
    step_action[6] = -1.0 if action_np[6] > 0.0 else 1.0
    
    print(f"{step+1:02d}   | {tcp_pos.round(3)} | {step_action.round(4)}")
    obs, reward, terminated, truncated, info = env.step(step_action)

env.close()
