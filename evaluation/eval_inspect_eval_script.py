import os
import inspect
import lerobot.scripts.lerobot_eval as eval_mod

print("=========================================================")
print("=== LEROBOT_EVAL.PY ROLLOUT LOOP INSPECTION ===")
print("=========================================================")

try:
    source = inspect.getsource(eval_mod)
    lines = source.split("\n")
    print(f"Total lines in lerobot_eval.py: {len(lines)}")
    
    # Filter for policy rollout loop keywords
    filtered = [f"{i+1:04d}: {line}" for i, line in enumerate(lines) if any(k in line for k in ["select_action", "preprocessor", "postprocessor", "make_pre_post", "predict_action", "process"])]
    print("\n".join(filtered[:50]))
except Exception as e:
    print(f"Error inspecting lerobot_eval.py: {e}")
