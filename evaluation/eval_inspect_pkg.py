import os
import inspect
import lerobot
import lerobot.processor
import lerobot.policies

print("=========================================================")
print("=== 1. LEROBOT PACKAGE MODULES INSPECTION ===")
print("=========================================================")

print(f"lerobot.__file__: {lerobot.__file__}")
print(f"lerobot.processor contents: {dir(lerobot.processor)}")
print(f"lerobot.policies contents: {dir(lerobot.policies)}")

print("\n--- lerobot.policies.factory.make_policy Source ---")
try:
    from lerobot.policies.factory import make_policy
    print(inspect.getsource(make_policy))
except Exception as e:
    print(f"Could not get source of make_policy: {e}")

print("\n--- Searching for processor classes in lerobot ---")
for mod in [lerobot.processor, lerobot.policies]:
    for attr in dir(mod):
        if "proc" in attr.lower() or "norm" in attr.lower() or "eval" in attr.lower() or "pre" in attr.lower():
            print(f"Found in {mod.__name__}: {attr}")
