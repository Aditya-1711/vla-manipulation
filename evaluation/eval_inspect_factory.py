import os
import inspect
import json
import torch

from lerobot.policies.factory import make_policy
from lerobot.processor import PolicyProcessor

CHECKPOINT_PATH = "/content/drive/MyDrive/vla-manipulation/checkpoints/smolvla_pickcube_60ep/checkpoints/last/pretrained_model"

print("=========================================================")
print("=== 1. CHECKING MAKE_POLICY & POLICYPROCESSOR LOADING ===")
print("=========================================================")

print("\n--- lerobot.policies.factory.make_policy Source ---")
try:
    print(inspect.getsource(make_policy))
except Exception as e:
    print(f"Could not get source of make_policy: {e}")

print("\n--- lerobot.processor.PolicyProcessor.from_pretrained Source ---")
try:
    print(inspect.getsource(PolicyProcessor.from_pretrained))
except Exception as e:
    print(f"Could not get source of PolicyProcessor.from_pretrained: {e}")

print("\n--- Testing make_policy(overrides=[...]) from checkpoint ---")
try:
    pol = make_policy(pretrained_policy_name_or_path=CHECKPOINT_PATH)
    print("make_policy loaded successfully!")
    print(f"Policy type: {type(pol)}")
    print(f"Policy attributes: {[a for a in dir(pol) if 'proc' in a or 'stat' in a or 'norm' in a]}")
except Exception as e:
    print(f"make_policy error: {e}")

print("\n--- Testing PolicyProcessor loading ---")
try:
    proc = PolicyProcessor.from_pretrained(CHECKPOINT_PATH)
    print("PolicyProcessor loaded successfully!")
    print(f"Processor type: {type(proc)}")
    print(f"Processor steps: {getattr(proc, 'steps', 'N/A')}")
except Exception as e:
    print(f"PolicyProcessor loading error: {e}")
