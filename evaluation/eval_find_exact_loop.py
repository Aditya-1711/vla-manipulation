import os
import inspect
import lerobot.scripts.lerobot_eval as eval_mod

print("=========================================================")
print("=== FULL LEROBOT_EVAL.PY ROLLOUT LOOP SEARCH ===")
print("=========================================================")

try:
    source = inspect.getsource(eval_mod)
    lines = source.split("\n")
    print(f"Total lines in lerobot_eval.py: {len(lines)}")
    
    # Print lines containing eval_policy or rollout or select_action
    for i, line in enumerate(lines):
        if "eval_policy" in line or "rollout" in line or "select_action" in line or "predict_action" in line or "preprocessor" in line:
            print(f"Line {i+1:04d}: {line}")
except Exception as e:
    print(f"Error reading lerobot_eval.py: {e}")

print("\n=========================================================")
print("=== CHECKING LEROBOT.RL.EVAL_POLICY OR CONTROL_UTILS ===")
print("=========================================================")

try:
    import lerobot.common.control_utils as ctrl_utils
    print("Found lerobot.common.control_utils!")
    print("Source of predict_action / rollout methods:")
    src = inspect.getsource(ctrl_utils)
    for line in src.split("\n"):
        if "policy" in line or "select" in line or "predict" in line or "processor" in line:
            print(line[:120])
except Exception as e:
    print(f"Could not inspect control_utils: {e}")
