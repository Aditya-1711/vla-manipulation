# Google Colab Setup Guide & Verification Instructions

Follow these exact steps in Google Colab to run the Phase 0 live verification on a Cloud GPU:

### Step 1: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Click **Upload** and select [`vla_pipeline.ipynb`](file:///d:/Manipulation/vla_pipeline.ipynb) from your local project folder (`d:\Manipulation\vla_pipeline.ipynb`).

### Step 2: Enable GPU Runtime
1. In Colab, click **Runtime** in the top menu bar -> **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (or A100 GPU).
3. Click **Save**.

### Step 3: Execute Verification Cells
Run cells 1 through 4 in sequence. Copy and paste the raw text outputs from the notebook into our chat:

1. **Cell 1 Output (CUDA Check)**:
   ```python
   import torch
   print("Torch Version:", torch.__version__)
   print("CUDA Available:", torch.cuda.is_available())
   print("GPU Device Name:", torch.cuda.get_device_name(0))
   ```

2. **Cell 2 Output (LeRobot Help CLI Output)**:
   ```bash
   !pip install -q lerobot
   !lerobot-train --help
   ```

3. **Cell 3 Output (ManiSkill3 Headless Render Verification)**:
   ```python
   import os, mani_skill.envs, gymnasium as gym
   os.environ["SAPIEN_RENDER_ENGINE"] = "EGL"
   env = gym.make("PickCube-v1", obs_mode="rgbd", render_mode="rgb_array")
   ...
   ```

4. **Cell 4 Output (Drive Persistence Setup)**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

Once you paste the live cell outputs here, Phase 0 will be officially verified with full exit criteria before we proceed to Phase 1!
