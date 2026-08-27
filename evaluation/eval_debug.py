import os
import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer

os.environ["SAPIEN_RENDER_ENGINE"] = "EGL"

CHECKPOINT_PATH = "/content/drive/MyDrive/vla-manipulation/checkpoints/smolvla_pickcube_60ep/checkpoints/last/pretrained_model"
TASK_INSTRUCTION = "Pick up the red cube and place it at the goal location."

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
    if isinstance(val, torch.Tensor):
        val = val.cpu().numpy()
    if val.ndim > 1:
        val = val[0]
    return val

obs, _ = env.reset(seed=1000)
policy.reset()

print("\n--- INFERENCE DIAGNOSTIC FOR SEED 1000 ---")
for step in range(10):
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
    
    print(f"Step {step+1:02d} | TCP Pos: {tcp_pos.round(3)} | Predicted Action: {action_np.round(3)}")
    obs, reward, terminated, truncated, info = env.step(action_np)

env.close()
