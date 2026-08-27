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

CHECKPOINT_PATH = "/content/drive/MyDrive/vla-manipulation/checkpoints/smolvla_pickcube_60ep/checkpoints/last/pretrained_model"
DATASET_ROOT = "/content/drive/MyDrive/vla-manipulation/data/lerobot_pick_cube"
TASK_INSTRUCTION = "Pick up the red cube and place it at the goal location."

print("=========================================================")
print("=== FULL EVALUATION Across 35 SEEDS (4 BUCKETS) ===")
print("=========================================================")

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

ds_meta = LeRobotDatasetMetadata(repo_id="local/pick_cube_rebalanced", root=DATASET_ROOT)

cfg = SmolVLAConfig.from_pretrained(CHECKPOINT_PATH)
cfg.pretrained_path = CHECKPOINT_PATH
cfg.device = device_str

policy = make_policy(cfg=cfg, ds_meta=ds_meta, rename_map={"observation.image": "observation.images.camera1"})
policy.eval()
if torch.cuda.is_available():
    policy = policy.to(device_str)

preprocessor_overrides = {"device_processor": {"device": device_str}}
postprocessor_overrides = {"device_processor": {"device": device_str}}

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=CHECKPOINT_PATH,
    dataset_stats=ds_meta.stats,
    preprocessor_overrides=preprocessor_overrides,
    postprocessor_overrides=postprocessor_overrides
)

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
tokenizer_out = tokenizer(TASK_INSTRUCTION, return_tensors="pt", padding=True)
lang_tokens = tokenizer_out["input_ids"].to(device)
lang_mask = tokenizer_out["attention_mask"].bool().to(device)

def to_np(val):
    if isinstance(val, torch.Tensor): val = val.cpu().numpy()
    if val.ndim > 1: val = val[0]
    return val

# Define evaluation seed categories
SEED_BUCKETS = {
    "Seen Training Seeds (10)": [1000, 1009, 1014, 1016, 1018, 1033, 1034, 1036, 1046, 1061],
    "Verified In-Distribution Unseen (5)": [1062, 1063, 1064, 1065, 1066],
    "Verified Generalization Unseen (10)": [1002, 1003, 1004, 1007, 1008, 1010, 1011, 1019, 1021, 1022],
    "Fresh Held-Out Unseen (10)": [5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010]
}

env = gym.make("PickCube-v1", obs_mode="rgbd", render_mode="rgb_array", control_mode="pd_ee_delta_pose")

overall_results = {}

for bucket_name, seeds in SEED_BUCKETS.items():
    print(f"\n--- Evaluating Bucket: {bucket_name} ---")
    grasped_count = 0
    success_count = 0
    
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        policy.reset()
        
        ep_grasped = False
        ep_success = False
        
        for step in range(80):
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
                "observation.language.attention_mask": lang_mask,
                "task": TASK_INSTRUCTION
            }
            
            with torch.no_grad():
                proc_obs = preprocessor(raw_obs)
                raw_action = policy.select_action(proc_obs)
                final_action = postprocessor(raw_action)
                
            action_np = final_action.cpu().numpy() if isinstance(final_action, torch.Tensor) else np.array(final_action)
            if action_np.ndim > 1: action_np = action_np[0]
            
            obs, reward, terminated, truncated, info = env.step(action_np)
            
            if info.get("is_grasped", False):
                ep_grasped = True
            if info.get("success", False):
                ep_success = True
                break
                
        if ep_grasped: grasped_count += 1
        if ep_success: success_count += 1
        
        print(f"Seed {seed:04d}: Grasped={ep_grasped} | Success={ep_success}")
        
    grasp_rate = (grasped_count / len(seeds)) * 100.0
    success_rate = (success_count / len(seeds)) * 100.0
    overall_results[bucket_name] = {"Grasp Rate": grasp_rate, "Success Rate": success_rate}
    print(f"--> {bucket_name} Result: Grasp Rate = {grasp_rate:.1f}% | Success Rate = {success_rate:.1f}%")

env.close()

print("\n=========================================================")
print("=== FINAL EVALUATION SUMMARY METRICS ===")
print("=========================================================")
for bucket_name, res in overall_results.items():
    print(f"{bucket_name}: Grasp = {res['Grasp Rate']:.1f}% | Success = {res['Success Rate']:.1f}%")
