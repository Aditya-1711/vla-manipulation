import os
import inspect
import lerobot
import lerobot.policies

print("=========================================================")
print("=== 1. INSPECTING MAKE_PRE_POST_PROCESSORS ===")
print("=========================================================")

try:
    from lerobot.policies import make_pre_post_processors
    print("Signature:", inspect.signature(make_pre_post_processors))
    print("\nSource:\n", inspect.getsource(make_pre_post_processors))
except Exception as e:
    print(f"Could not inspect make_pre_post_processors: {e}")

print("\n=========================================================")
print("=== 2. SEARCHING FOR EVAL / ROLLOUT PIPELINE IN LEROBOT ===")
print("=========================================================")

eval_files = []
for root, dirs, files in os.walk(os.path.dirname(lerobot.__file__)):
    for file in files:
        if "eval" in file.lower() or "rollout" in file.lower() or "control" in file.lower() or "predict" in file.lower():
            eval_files.append(os.path.join(root, file))

print(f"Found {len(eval_files)} relevant files in lerobot package:")
for f in eval_files[:10]:
    print(f"  - {f}")

# Inspect first available eval script source
if eval_files:
    print(f"\n--- Source snippet of {os.path.basename(eval_files[0])} ---")
    try:
        with open(eval_files[0], "r", encoding="utf-8") as file:
            content = file.read()
        lines = [line for line in content.split("\n") if "policy" in line or "process" in line or "select_action" in line]
        print("\n".join(lines[:30]))
    except Exception as e:
        print(f"Error reading file: {e}")
