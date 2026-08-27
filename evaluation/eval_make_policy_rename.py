import os
import inspect
import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs

from lerobot.policies.factory import make_policy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from transformers import AutoTokenizer

os.environ["SAPIEN_RENDER_ENGINE"] = "EGL"

CHECKPOINT_PATH = "/content/drive/MyDrive/vla-manipulation/checkpoints/smolvla_pickcube_60ep/checkpoints/last/pretrained_model"
DATASET_ROOT = "/content/drive/MyDrive/vla-manipulation/data/lerobot_pick_cube"
TASK_INSTRUCTION = "Pick up the red cube and place it at the goal location."

print("=========================================================")
print("=== 1. CONSTRUCTING POLICY WITH RENAME_MAP & DS_META ===")
print("=========================================================")

ds_meta = LeRobotDatasetMetadata(repo_id="local/pick_cube_rebalanced", root=DATASET_ROOT)
print("LeRobotDatasetMetadata loaded successfully!")

cfg = SmolVLAConfig.from_pretrained(CHECKPOINT_PATH)
cfg.pretrained_path = CHECKPOINT_PATH

# Instantiate policy using make_policy with rename_map and ds_meta
policy = make_policy(
    cfg=cfg,
    ds_meta=ds_meta,
    rename_map={"observation.image": "observation.images.camera1"}
)
policy.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    policy = policy.to("cuda")

print("\n--- ATTACHED NORMALIZATION & PREPROCESSOR CHECK ---")
print(f"policy type: {type(policy)}")
print(f"policy._runtime_dataset_meta: {getattr(policy, '_runtime_dataset_meta', None) is not None}")
print(f"policy.preprocessor: {getattr(policy, 'preprocessor', None)}")
print(f"policy.postprocessor: {getattr(policy, 'postprocessor', None)}")
print(f"policy.dataset_stats: {hasattr(policy, 'dataset_stats')} ({type(getattr(policy, 'dataset_stats', None))})")

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
tokenizer_out = tokenizer(TASK_INSTRUCTION, return_tensors="pt", padding=True)
lang_tokens = tokenizer_out["input_ids"].to(device)
lang_mask = tokenizer_out["attention_mask"].bool().to(device)

def to_np(val):
    if isinstance(val, torch.Tensor): val = val.cpu().numpy()
    if val.ndim > 1: val = val[0]
    return val

print("\n=========================================================")
print("=== 2. SINGLE EPISODE ROLLOUT (SEED 1000) WITH NO OVERRIDES ===")
print("=========================================================")

env = gym.make("PickCube-v1", obs_mode="rgbd", render_mode="rgb_array", control_mode="pd_ee_delta_pose")
obs, _ = env.reset(seed=1000)
policy.reset()

print("Step | TCP Position (x, y, z)       | Pure Unmodified Policy Action Output (7D)")
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
    
    # PURE UNMODIFIED MODEL ACTION STEP: NO overrides, NO thresholding, NO sign flips!
    print(f"{step+1:02d}   | {tcp_pos.round(3)} | {action_np.round(4)}")
    obs, reward, terminated, truncated, info = env.step(action_np)

env.close()
