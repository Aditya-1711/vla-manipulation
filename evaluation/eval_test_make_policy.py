import os
import inspect
import torch
from lerobot.policies.factory import make_policy
from lerobot.policies.common.pipeline import make_default_pre_post_processors

CHECKPOINT_PATH = "/content/drive/MyDrive/vla-manipulation/checkpoints/smolvla_pickcube_60ep/checkpoints/last/pretrained_model"
DATASET_ROOT = "/content/drive/MyDrive/vla-manipulation/data/lerobot_pick_cube"

print("=========================================================")
print("=== INSPECTING MAKE_DEFAULT_PRE_POST_PROCESSORS SOURCE ===")
print("=========================================================")

try:
    print(inspect.getsource(make_default_pre_post_processors))
except Exception as e:
    print(f"Could not get source: {e}")

print("\n--- Testing dataset metadata loading for make_policy ---")
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    ds_meta = LeRobotDatasetMetadata(repo_id="local/pick_cube_rebalanced", root=DATASET_ROOT)
    print("LeRobotDatasetMetadata loaded successfully!")
    print(f"Features: {list(ds_meta.features.keys())}")
    print(f"Stats keys: {list(ds_meta.stats.keys()) if hasattr(ds_meta, 'stats') else 'None'}")
    
    # Instantiate policy via make_policy with dataset metadata!
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    cfg = SmolVLAConfig.from_pretrained(CHECKPOINT_PATH)
    cfg.pretrained_path = CHECKPOINT_PATH
    
    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    print("\nSUCCESS! Policy created via make_policy with dataset metadata.")
    print(f"Policy type: {type(policy)}")
    print(f"Policy preprocessor: {getattr(policy, 'preprocessor', None)}")
    print(f"Policy postprocessor: {getattr(policy, 'postprocessor', None)}")
except Exception as e:
    print(f"Error instantiating via make_policy: {e}")
