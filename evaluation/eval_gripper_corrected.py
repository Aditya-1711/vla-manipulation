import os
import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.dataset.stats import load_stats
from transformers import AutoTokenizer

os.environ["SAPIEN_RENDER_ENGINE"] = "EGL"

CHECKPOINT_PATH = "/content/drive/MyDrive/vla-manipulation/checkpoints/smolvla_pickcube_60ep/checkpoints/last/pretrained_model"
DATASET_STATS_PATH = "/content/drive/MyDrive/vla-manipulation/data/lerobot_pick_cube/stats.json"
TASK_INSTRUCTION = "Pick up the red cube and place it at the goal location."
SEEN_TRAINING_SEEDS = [1000, 1009, 1014, 1016, 1018, 1033, 1034, 1036, 1046, 1061]

env = gym.make("PickCube-v1", obs_mode="rgbd", render_mode="rgb_array", control_mode="pd_ee_delta_pose")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = SmolVLAPolicy.from_pretrained(CHECKPOINT_PATH)
policy.eval()
if torch.cuda.is_available():
    policy = policy.to("cuda")

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
tokenizer_out = tokenizer(TASK_INSTRUCTION, return_tensors="pt", padding=True)
lang_tokens = tokenizer_out["input_ids"].to(device)
lang_mask = tokenizer_out["attention_mask"].bool().to(device)

def to_np(val):
    if isinstance(val, torch.Tensor): val = val.cpu().numpy()
    if val.ndim > 1: val = val[0]
    return val

print("\n=== EVALUATION WITH NORMALIZED ACTION UN-SCALE AND GRIPPER THRESHOLDING ===")
successes = 0
for idx, seed in enumerate(SEEN_TRAINING_SEEDS, start=1):
    obs, _ = env.reset(seed=seed)
    policy.reset()
    episode_success = False
    
    for step in range(100):
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
            "observation.state": state_tensor,
            "observation.language.tokens": lang_tokens,
            "observation.language.attention_mask": lang_mask
        }
        
        with torch.no_grad():
            action_pred = policy.select_action(observation)
        
        action_np = action_pred.cpu().numpy() if isinstance(action_pred, torch.Tensor) else np.array(action_pred)
        if action_np.ndim > 1: action_np = action_np[0]
        
        # Binary thresholding for gripper action (7th dimension)
        # > 0.0 -> OPEN (+1.0), <= 0.0 -> CLOSE (-1.0)
        action_np[6] = 1.0 if action_np[6] > 0.0 else -1.0
        
        obs, reward, terminated, truncated, info = env.step(action_np)
        
        success = info.get("success", False)
        if isinstance(success, torch.Tensor): success = success.any().item()
        if success:
            episode_success = True
            break
            
    if episode_success: successes += 1
    status_str = "SUCCESS" if episode_success else "FAILED"
    print(f"  [{idx:02d}/10] Seed {seed}: {status_str}")

env.close()
print(f"\nFinal Gripper-Corrected Success Rate (Seen Seeds): {successes}/10 ({successes*10.0:.1f}%)")
