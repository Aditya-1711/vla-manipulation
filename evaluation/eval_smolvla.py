import os
import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer

# Ensure headless EGL rendering for Colab
os.environ["SAPIEN_RENDER_ENGINE"] = "EGL"

# Categorized Seed Sets for Honest Evaluation
SEEN_TRAINING_SEEDS = [1000, 1009, 1014, 1016, 1018, 1033, 1034, 1036, 1046, 1061]
VERIFIED_IN_DIST_UNSEEN = [1062, 1063, 1064, 1065, 1066]
VERIFIED_GEN_SEEDS = [1002, 1003, 1004, 1007, 1008, 1010, 1011, 1019, 1021, 1022]
FRESH_HELD_OUT_SEEDS = [5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010]

CHECKPOINT_PATH = "/content/drive/MyDrive/vla-manipulation/checkpoints/smolvla_pickcube_60ep/checkpoints/last/pretrained_model"
TASK_INSTRUCTION = "Pick up the red cube and place it at the goal location."
MAX_STEPS_PER_EPISODE = 100

def evaluate_seed_category(policy, tokenizer, category_name, seed_list):
    print(f"\n=======================================================")
    print(f"=== Evaluating Category: {category_name} ({len(seed_list)} Seeds) ===")
    print(f"=======================================================")
    
    env = gym.make("PickCube-v1", obs_mode="rgbd", render_mode="rgb_array", control_mode="pd_ee_delta_pose")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    successes = 0
    grasps = 0
    results = []
    
    def to_np(val):
        if isinstance(val, torch.Tensor):
            val = val.cpu().numpy()
        if val.ndim > 1:
            val = val[0]
        return val
        
    # Tokenize language task prompt using AutoTokenizer
    tokenizer_out = tokenizer(TASK_INSTRUCTION, return_tensors="pt", padding=True)
    lang_tokens = tokenizer_out["input_ids"].to(device)
    lang_mask = tokenizer_out["attention_mask"].bool().to(device)
    
    for idx, seed in enumerate(seed_list, start=1):
        obs, _ = env.reset(seed=seed)
        policy.reset()
        
        episode_grasped = False
        episode_success = False
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # Render frame
            frame = env.render()
            if isinstance(frame, list):
                frame = frame[0]
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.ndim == 4:
                frame = frame[0]
            
            img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Extract 7-dim TCP pose (position + quaternion)
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
            
            if isinstance(action_pred, torch.Tensor):
                action_np = action_pred.cpu().numpy()
            else:
                action_np = np.array(action_pred)
            if action_np.ndim > 1:
                action_np = action_np[0]
                
            # Apply binary thresholding to continuous 7th-dim gripper action:
            # > 0.0 -> OPEN (+1.0), <= 0.0 -> CLOSE (-1.0)
            action_np[6] = 1.0 if action_np[6] > 0.0 else -1.0
                
            obs, reward, terminated, truncated, info = env.step(action_np)
            
            is_grasped = info.get("is_grasped", False)
            if isinstance(is_grasped, torch.Tensor):
                is_grasped = is_grasped.any().item()
            if is_grasped:
                episode_grasped = True
                
            success = info.get("success", False)
            if isinstance(success, torch.Tensor):
                success = success.any().item()
            if success:
                episode_success = True
                break
                
            if terminated or truncated:
                break
                
        if episode_grasped:
            grasps += 1
        if episode_success:
            successes += 1
            
        status_str = "SUCCESS" if episode_success else ("GRASPED ONLY" if episode_grasped else "FAILED")
        print(f"  [{idx:02d}/{len(seed_list):02d}] Seed {seed}: {status_str} (Steps: {step+1})")
        results.append({"seed": seed, "grasped": episode_grasped, "success": episode_success})
        
    env.close()
    
    grasp_rate = (grasps / len(seed_list)) * 100.0
    success_rate = (successes / len(seed_list)) * 100.0
    
    print(f"--> {category_name} Summary: Grasp Rate = {grasp_rate:.1f}% ({grasps}/{len(seed_list)}) | Success Rate = {success_rate:.1f}% ({successes}/{len(seed_list)})\n")
    return {"category": category_name, "grasp_rate": grasp_rate, "success_rate": success_rate, "results": results}

if __name__ == "__main__":
    print(f"Loading SmolVLA Fine-Tuned Policy from: {CHECKPOINT_PATH} ...")
    policy = SmolVLAPolicy.from_pretrained(CHECKPOINT_PATH)
    policy.eval()
    if torch.cuda.is_available():
        policy = policy.to("cuda")
        print(f"Policy loaded on CUDA device: {torch.cuda.get_device_name(0)}")

    print("Loading SmolVLM Tokenizer for text prompt tokenization ...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    categories = [
        ("a. Seen Training Seeds", SEEN_TRAINING_SEEDS),
        ("b. Verified In-Dist (Unseen)", VERIFIED_IN_DIST_UNSEEN),
        ("c. Verified Generalization (Unseen)", VERIFIED_GEN_SEEDS),
        ("d. Fresh Held-Out Seeds", FRESH_HELD_OUT_SEEDS)
    ]

    summary_table = []
    for cat_name, seeds in categories:
        res = evaluate_seed_category(policy, tokenizer, cat_name, seeds)
        summary_table.append(res)
        
    print("\n=========================================================================")
    print("=== FINAL SMOLVLA POLICY EVALUATION MATRIX ===")
    print("=========================================================================")
    print(f"{'Category':<35} | {'Grasp Rate':<12} | {'Success Rate':<12}")
    print("-" * 65)
    for row in summary_table:
        print(f"{row['category']:<35} | {row['grasp_rate']:>5.1f}%       | {row['success_rate']:>5.1f}%")
    print("=========================================================================")
