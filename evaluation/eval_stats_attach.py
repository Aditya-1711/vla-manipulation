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
print("=== 1. INSPECTING CHECKPOINT & DATASET STATS ===")
print("=========================================================")

# Check checkpoint files
ckpt_files = os.listdir(CHECKPOINT_PATH) if os.path.exists(CHECKPOINT_PATH) else []
print(f"Checkpoint directory files: {ckpt_files}")

# Check dataset stats file
dataset_stats_file = os.path.join(DATASET_ROOT, "meta/stats.json")
if not os.path.exists(dataset_stats_file):
    dataset_stats_file = os.path.join(DATASET_ROOT, "stats.json")

print(f"Dataset stats file exists ({dataset_stats_file}): {os.path.exists(dataset_stats_file)}")

if os.path.exists(dataset_stats_file):
    with open(dataset_stats_file, "r") as f:
        stats_dict = json.load(f)
    print(f"Stats keys found: {list(stats_dict.keys())}")
else:
    stats_dict = None

# Load policy
policy = SmolVLAPolicy.from_pretrained(CHECKPOINT_PATH)
policy.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    policy = policy.to("cuda")

# Attach dataset stats to policy if missing
if stats_dict is not None:
    # Convert stats dictionary to PyTorch tensors on device
    stats_tensors = {}
    for key, val in stats_dict.items():
        stats_tensors[key] = {}
        for sub_key, sub_val in val.items():
            t = torch.tensor(sub_val, dtype=torch.float32, device=device)
            stats_tensors[key][sub_key] = t
            
    # Set on policy preprocessor/postprocessor or dataset_stats attribute
    if hasattr(policy, "dataset_stats"):
        policy.dataset_stats = stats_tensors
    if hasattr(policy, "stats"):
        policy.stats = stats_tensors
        
    print(f"Successfully attached stats_tensors to policy! Keys attached: {list(stats_tensors.keys())}")

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
tokenizer_out = tokenizer(TASK_INSTRUCTION, return_tensors="pt", padding=True)
lang_tokens = tokenizer_out["input_ids"].to(device)
lang_mask = tokenizer_out["attention_mask"].bool().to(device)

def to_np(val):
    if isinstance(val, torch.Tensor): val = val.cpu().numpy()
    if val.ndim > 1: val = val[0]
    return val

print("\n=========================================================")
print("=== 2. SINGLE EPISODE ROLLOUT WITH ATTACHED STATS (SEED 1000) ===")
print("=========================================================")

env = gym.make("PickCube-v1", obs_mode="rgbd", render_mode="rgb_array", control_mode="pd_ee_delta_pose")
obs, _ = env.reset(seed=1000)
policy.reset()

print("Step | TCP Position (x, y, z)       | Denormalized Action Vector (7D)")
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
        # Select action
        action_pred = policy.select_action(observation)
        
        # If output requires manual un-normalization using action stats:
        if stats_dict is not None and "action" in stats_dict:
            action_mean = torch.tensor(stats_dict["action"]["mean"], device=device)
            action_std = torch.tensor(stats_dict["action"]["std"], device=device)
            # unnormalize: action = action * std + mean
            if action_pred.ndim == action_mean.ndim:
                action_unnorm = action_pred * action_std + action_mean
            else:
                action_unnorm = action_pred * action_std.unsqueeze(0) + action_mean.unsqueeze(0)
        else:
            action_unnorm = action_pred
        
    action_raw_np = action_pred.cpu().numpy() if isinstance(action_pred, torch.Tensor) else np.array(action_pred)
    action_unnorm_np = action_unnorm.cpu().numpy() if isinstance(action_unnorm, torch.Tensor) else np.array(action_unnorm)
    
    if action_raw_np.ndim > 1: action_raw_np = action_raw_np[0]
    if action_unnorm_np.ndim > 1: action_unnorm_np = action_unnorm_np[0]
    
    print(f"{step+1:02d}   | {tcp_pos.round(3)} | Unnorm: {action_unnorm_np.round(4)}")
    
    # Step with unnormalized action (thresholding gripper)
    step_action = action_unnorm_np.copy()
    step_action[6] = 1.0 if step_action[6] > 0.0 else -1.0
    obs, reward, terminated, truncated, info = env.step(step_action)

env.close()
