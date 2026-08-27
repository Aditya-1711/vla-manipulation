import os
import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs

from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from transformers import AutoTokenizer

os.environ["SAPIEN_RENDER_ENGINE"] = "EGL"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

CHECKPOINT_PATH = "/content/drive/MyDrive/vla-manipulation/checkpoints/smolvla_pickcube_60ep/checkpoints/last/pretrained_model"
DATASET_ROOT = "/content/drive/MyDrive/vla-manipulation/data/lerobot_pick_cube"
TASK_INSTRUCTION = "Pick up the red cube and place it at the goal location."

print("=========================================================")
print("=== 1. INITIALIZING POLICY ON CUDA PRIOR TO ENV CREATION ===")
print("=========================================================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds_meta = LeRobotDatasetMetadata(repo_id="local/pick_cube_rebalanced", root=DATASET_ROOT)
print("LeRobotDatasetMetadata loaded successfully!")

cfg = SmolVLAConfig.from_pretrained(CHECKPOINT_PATH)
cfg.pretrained_path = CHECKPOINT_PATH
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

# Load policy cleanly BEFORE ManiSkill3 EGL context initialization
policy = make_policy(cfg=cfg, ds_meta=ds_meta, rename_map={"observation.image": "observation.images.camera1"})
policy.eval()
print(f"Policy loaded on device: {policy.device}")

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=CHECKPOINT_PATH,
    dataset_stats=ds_meta.stats
)
print("Pre/Post processors initialized successfully!")

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
tokenizer_out = tokenizer(TASK_INSTRUCTION, return_tensors="pt", padding=True)
lang_tokens = tokenizer_out["input_ids"].to(device)
lang_mask = tokenizer_out["attention_mask"].bool().to(device)

def to_np(val):
    if isinstance(val, torch.Tensor): val = val.cpu().numpy()
    if val.ndim > 1: val = val[0]
    return val

print("\n=========================================================")
print("=== 2. CREATING MANISKILL3 ENV & EXECUTING ROLLOUT (SEED 1000) ===")
print("=========================================================")

env = gym.make("PickCube-v1", obs_mode="rgbd", render_mode="rgb_array", control_mode="pd_ee_delta_pose")
obs, _ = env.reset(seed=1000)
policy.reset()

print("Step | TCP Position (x, y, z)       | Postprocessed Final Policy Action (7D)")
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
    
    raw_obs = {
        "observation.images.camera1": img_tensor,
        "observation.images.camera2": img_tensor.clone(),
        "observation.images.camera3": img_tensor.clone(),
        "observation.state": state_tensor,
        "observation.language.tokens": lang_tokens,
        "observation.language.attention_mask": lang_mask
    }
    
    with torch.no_grad():
        proc_obs = preprocessor(raw_obs)
        raw_action = policy.select_action(proc_obs)
        final_action = postprocessor(raw_action)
        
    action_np = final_action.cpu().numpy() if isinstance(final_action, torch.Tensor) else np.array(final_action)
    if action_np.ndim > 1: action_np = action_np[0]
    
    print(f"{step+1:02d}   | {tcp_pos.round(3)} | {action_np.round(4)}")
    obs, reward, terminated, truncated, info = env.step(action_np)

env.close()
