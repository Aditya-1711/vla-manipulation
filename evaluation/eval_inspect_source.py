import os
import inspect
import json
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CHECKPOINT_PATH = "/content/drive/MyDrive/vla-manipulation/checkpoints/smolvla_pickcube_60ep/checkpoints/last/pretrained_model"

print("=========================================================")
print("=== 1. CHECKPOINT DIRECTORY FILE LISTING ===")
print("=========================================================")
if os.path.exists(CHECKPOINT_PATH):
    files = os.listdir(CHECKPOINT_PATH)
    for f in sorted(files):
        size_str = f"{os.path.getsize(os.path.join(CHECKPOINT_PATH, f)) / 1024 / 1024:.2f} MB"
        print(f"  - {f:<35} ({size_str})")
else:
    print(f"Directory {CHECKPOINT_PATH} does not exist!")

print("\n=========================================================")
print("=== 2. SMOLVLA POLICY SOURCE CODE INSPECTION ===")
print("=========================================================")

print("\n--- SmolVLAPolicy.__init__ Signature ---")
print(inspect.signature(SmolVLAPolicy.__init__))

print("\n--- SmolVLAPolicy.select_action Source ---")
try:
    print(inspect.getsource(SmolVLAPolicy.select_action))
except Exception as e:
    print(f"Could not get source: {e}")

print("\n--- SmolVLAPolicy.from_pretrained Source ---")
try:
    print(inspect.getsource(SmolVLAPolicy.from_pretrained))
except Exception as e:
    print(f"Could not get source: {e}")
