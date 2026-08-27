import os
import time
import h5py
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from sim_env.scripted_baseline import ScriptedPickPlacePolicy

def collect_rebalanced_dataset_with_state(
    output_h5_path: str = "/content/drive/MyDrive/vla-manipulation/data/pick_cube_demos/trajectory_dataset.h5",
    main_seeds: list = None,
    rebalance_seeds: list = None
):
    if main_seeds is None:
        main_seeds = [s for s in range(1000, 1051) if s != 1017]  # 50 seeds
    if rebalance_seeds is None:
        rebalance_seeds = [3008, 3013, 3019, 3035, 3043, 3053, 3058, 3063, 3077, 3079] # 10 close seeds

    all_target_seeds = [("main", s) for s in main_seeds] + [("rebalance", s) for s in rebalance_seeds]
    print(f"Targeting total {len(all_target_seeds)} episodes (50 main + 10 rebalanced close-range).")

    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
    env = gym.make("PickCube-v1", obs_mode="rgbd", control_mode="pd_ee_delta_pose", render_mode="rgb_array")
    policy = ScriptedPickPlacePolicy(env)

    successful_episodes = []

    for ep_idx, (tag, seed) in enumerate(all_target_seeds, 1):
        obs, info = env.reset(seed=seed)
        policy.reset()

        ep_frames = []
        ep_actions = []
        ep_states = []
        success = False

        for step in range(160):
            action = policy.get_action(obs, info)

            # Extract visual frame
            if "sensor_data" in obs and "base_camera" in obs["sensor_data"]:
                img = obs["sensor_data"]["base_camera"]["rgb"]
                if isinstance(img, torch.Tensor): img = img.cpu().numpy()
                if img.ndim == 4: img = img[0]
                ep_frames.append(img)

            # Extract 7-dim TCP state (position 3 + quaternion 4)
            if "extra" in obs and "tcp_pose" in obs["extra"]:
                tcp_p = obs["extra"]["tcp_pose"]
                if isinstance(tcp_p, torch.Tensor): tcp_p = tcp_p.cpu().numpy()
                if tcp_p.ndim > 1: tcp_p = tcp_p[0]
                ep_states.append(tcp_p[:7].astype(np.float32))
            else:
                tcp_pos = env.unwrapped.agent.tcp.pose.p.cpu().numpy()[0]
                tcp_quat = env.unwrapped.agent.tcp.pose.q.cpu().numpy()[0]
                ep_states.append(np.concatenate([tcp_pos, tcp_quat]).astype(np.float32))

            # Action conversion
            act_np = action.cpu().numpy()[0] if isinstance(action, torch.Tensor) and action.ndim > 1 else action
            ep_actions.append(act_np.astype(np.float32))

            obs, reward, term, trunc, info = env.step(action)

            if isinstance(info, dict) and (info.get("success", False) or (isinstance(info.get("success"), torch.Tensor) and info["success"].any())):
                success = True
                break

        if success:
            successful_episodes.append({
                "seed": seed,
                "tag": tag,
                "frames": ep_frames,
                "actions": ep_actions,
                "states": ep_states
            })
            print(f"[{ep_idx:02d}/{len(all_target_seeds)}] Saved {tag.upper()} Seed {seed}: {len(ep_frames)} frames | state shape = {ep_states[0].shape}")
        else:
            print(f"[{ep_idx:02d}/{len(all_target_seeds)}] WARNING: Seed {seed} failed task criteria. Skipped.")

    env.close()

    # Save to HDF5 dataset
    print(f"\nWriting {len(successful_episodes)} successful episodes into HDF5: {output_h5_path}")
    with h5py.File(output_h5_path, "w") as h5file:
        data_grp = h5file.create_group("data")
        for i, ep in enumerate(successful_episodes):
            ep_grp = data_grp.create_group(f"episode_{i:02d}")
            ep_grp.attrs["seed"] = ep["seed"]
            ep_grp.attrs["tag"] = ep["tag"]
            ep_grp.create_dataset("observation/image", data=np.array(ep["frames"], dtype=np.uint8))
            ep_grp.create_dataset("observation/state", data=np.array(ep["states"], dtype=np.float32))
            ep_grp.create_dataset("action", data=np.array(ep["actions"], dtype=np.float32))

    print(f"HDF5 Dataset generation complete: {len(successful_episodes)} episodes saved.")
    return len(successful_episodes)

if __name__ == "__main__":
    collect_rebalanced_dataset_with_state()
